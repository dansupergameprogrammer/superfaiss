#pragma once

#include "types.h"

namespace superfaiss
{

// Query-construction helpers: build prototype (centroid) and direction queries from
// bank rows or raw vectors. Pure math, allocation-free, deterministic: accumulation is
// serial in the order the caller provides, in double precision.

// Mean of the selected rows, written as a padded query (paddedDims elements, pad lanes
// zero). Int8 rows are dequantized through their per-row scales. On Cosine banks the
// mean is renormalized to unit length; a zero-norm mean (antipodal members cancelling)
// is rejected as ZeroNormQuery, never silently renormalized. Rejects null/empty input
// and out-of-range indices as InvalidArgument. outPaddedQuery must satisfy kAlignment.
Status MakeCentroid(
	const BankView& bank,
	const int32_t* rowIndices,
	int32_t rowCount,
	float* outPaddedQuery);

// --- v2.4 integer-domain pooling (plan section 21) ---

// Fixed-point fractional width of the pooling multiplier (FAI-5, frozen at build):
// each included row's scale ratio (its decoded scale over the max included scale)
// requantizes to an integer multiplier in [0, 2^kPoolScaleFracBits], round-half-even
// in integer math. 24 keeps the multiplier inside RoundHalfEvenI32's domain and the
// accumulator bound below with ~4128x margin.
inline constexpr int32_t kPoolScaleFracBits = 24;

// Pooled-weight cap (FAI-5, frozen at build): the sum of the salience weights over
// the included rows (= the row count when unweighted) must not exceed this. The
// per-dim int64 accumulator is then bounded by
//   sum(w_r) * 127 * 2^kPoolScaleFracBits <= 2^20 * 127 * 2^24 < 2^51 << 2^63,
// overflow-free — and small enough that its double conversion is exact (< 2^53),
// which is what makes the pooled scale's arithmetic weight-cancelling (see below).
inline constexpr int64_t kMaxPooledRows = int64_t{1} << 20;

// Pools int8 bank rows into a cross-device-safe QUANTIZED query (v2.4): the int8
// image on the bank's padded grid, its per-query dequant scale (double — no float
// round-trip), and its integer self-dot — exactly the XdQuery payload QueryXd
// executes, so the executed query IS this operator's product bit for bit.
//
// The math, in order (every step deterministic across machines):
//   - Per included row, the multiplier m_r = w_r * RHE(decode(scale_r)/maxScale * 2^24):
//     the scale ratio is two single IEEE double operations on DAZ-proof decoded bits,
//     rounded half-even in integer math (the section 19 discipline); the integer
//     weight folds in by exact integer multiply, so all-equal weights scale every
//     m_r by the same integer and cancel bitwise under symmetric quantization.
//   - Per dim, acc_j = sum(v_rj * m_r) in int64 — exact and order-free (stronger than
//     pinned order); bounded per kMaxPooledRows above.
//   - The accumulator requantizes DIRECTLY in the integer domain (FAI-6, the strictly
//     stronger form): no cross-dim float norm reduction exists on this path at all —
//     symmetric per-query quantization is invariant to positive scaling, so the
//     result is definitionally identical to normalize-then-quantize. q8_j =
//     RHE(acc_j * 127 / maxAcc), exact integer arithmetic; the max-magnitude dim maps
//     to exactly +-127.
//   - outScale = (maxAcc / (sumW * 127)) * maxScale * 2^-24 — fixed-order double ops
//     over exactly-representable integers (the cap guarantees < 2^53), so all-equal
//     weights cancel in the exact rational before the one rounded division.
//
// Defined behavior: int8 banks with paddedDims <= kMaxCrossDeviceDims only
// (InvalidArgument otherwise — CrossDevice is int8-only); empty selection (rowCount
// <= 0, or every row excluded) is InvalidArgument, never a zero vector; sum(w) >
// kMaxPooledRows is InvalidArgument (reject-over-overflow); a weight <= 0 or an
// out-of-range index is InvalidArgument; an all-zero integer accumulator is
// ZeroNormQuery — the check the omitted normalization would have performed, kept
// without the float math. Rows tombstoned in a scratch snapshot are excluded via
// excludeBits (deletion is exclusion, the section 7 idiom); excluded rows contribute
// nothing and do not count toward sum(w).
//
// outQ8 must hold bank.paddedDims bytes (pad lanes are zeroed). Allocation-free:
// accumulation sweeps a fixed stack chunk across the dims range (the MakeCentroid
// idiom), recomputing per-row multipliers per sweep — pooling is query-build cost,
// not scan cost. This operator is a versioned composition operator: any change to
// its accumulation, epilogue, or quantization is a space-version change for
// consumers (see docs/DETERMINISM.md section 2d).
//
// Sub-range pooling (V3.0 Tier 2, plan section 23.5): `offset`/`length` restrict the
// accumulation to a channel's element sub-range instead of the whole [0, dims). The
// defaults (offset 0, length < 0 -> whole dims) leave the whole-vector path bit-identical
// -- every existing caller passes them and the produced centroid, scale, and self-dot are
// unchanged. When set, the centroid is `length` lanes written at outQ8[0..length) with its
// self-dot over the same range; outQ8 must still hold bank.paddedDims bytes (lanes past
// `length` are zeroed). The per-row scales, weights, and the requantization are the
// whole-vector math on the sub-range, so pooling a channel sub-range equals pooling a
// contiguous repack of that sub-range bit for bit.
Status MakeCentroidCrossDevice(
	const BankView& bank,
	const int32_t* rowIndices,
	int32_t rowCount,
	const int32_t* weights,      // optional salience weights: null = unweighted (all 1)
	const uint32_t* excludeBits, // optional exclusion words (e.g. snapshot tombstones)
	int8_t* outQ8,
	double* outScale,
	int64_t* outSqSum,
	int32_t offset = 0,          // sub-range start element (V3.0); 0 = whole vector
	int32_t length = -1);        // sub-range length; < 0 = bank.dims (whole vector)

// normalize(a - b) written as a padded query: the unit direction from b toward a, for
// axis-projection queries ("most a-like relative to b"). Inputs are padded vectors
// (paddedDims elements, pad lanes zero). a == b (zero-norm difference) is rejected as
// ZeroNormQuery. On Dot/Cosine banks the direction is a valid query as-is; on L2 banks
// score it with ScoreAs::Dot.
Status MakeDirection(
	const float* paddedA,
	const float* paddedB,
	int32_t dims,
	int32_t paddedDims,
	float* outPaddedQuery);

// Score gap between a hit and its runner-up in the scored metric's better-direction:
// non-negative whenever `better` genuinely ranks at-or-above `runnerUp`. For L2 the
// gap is runnerUp - better (smaller score is better); for Dot/Cosine it is
// better - runnerUp. Pass the metric the query was scored by (ScoringMetric()).
inline float Margin(const Hit& better, const Hit& runnerUp, Metric scoredAs)
{
	return scoredAs == Metric::L2 ? runnerUp.score - better.score
	                              : better.score - runnerUp.score;
}

} // namespace superfaiss

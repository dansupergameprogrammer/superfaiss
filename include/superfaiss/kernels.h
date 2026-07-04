#pragma once

#include "types.h"
#include "topk.h"

namespace superfaiss
{

// Scores every non-excluded row of one chunk into `inout`. Single-threaded;
// an external scheduler may call this for different chunks concurrently, each with its
// own TopK, then MergeTopK the results.
//
// `paddedQuery` must have bank.paddedDims elements, zero-filled pad lanes, and
// kAlignment-byte alignment. Callers are expected to have validated the bank and query
// (see validate.h); kernels do not re-validate.
//
// v2.1 dense row bias: when `rowBias` is non-null (bank.count floats), each row's
// composed score is score + rowBias[r] - ONE fused add after dequantized scoring,
// before top-k insertion; a non-finite bias value sets *outNonFiniteBias (callers
// return NonFiniteQuery at completion - the fused-validation law, T-055 W2). Null
// rowBias executes no add: the bit-identical unbiased path. Same trailing pair on
// every chunk kernel below (ScoreChunkPair takes one per query; the fused kernels
// apply bias once, to the fused score).
void ScoreChunk(
	const BankView& bank,
	const float* paddedQuery,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout,
	const float* rowBias = nullptr,
	bool* outNonFiniteBias = nullptr);

// Scores one chunk against TWO queries in a single row pass: row loads (and int8
// widening) are shared while each query keeps the exact single-query accumulation
// structure, so pair results are bit-identical to two ScoreChunk calls. This is the
// batch amortization primitive (plan E2).
void ScoreChunkPair(
	const BankView& bank,
	const float* paddedQueryA,
	const float* paddedQueryB,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inoutA,
	TopK& inoutB,
	const float* rowBiasA = nullptr,
	const float* rowBiasB = nullptr,
	bool* outNonFiniteBias = nullptr);

// Scores every non-excluded row of one chunk against M queries and pushes the FUSED
// score — the worst per-query score in the metric's better-direction (min for
// Dot/Cosine, max for L2) — into `inout`. The intersection primitive (plan 18.7):
// per-query scores come from the same per-row kernels as ScoreChunk, so fused scores
// are bit-identical to the corresponding single-query scores; worst-of selection is
// rounding-free. Queries are contiguous, stride bank.paddedDims.
void ScoreChunkFused(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout,
	const float* rowBias = nullptr,
	bool* outNonFiniteBias = nullptr);

// Segmented scan, dense form (V2 §10 decision). NOTE the shipped split: dot-family
// segmented scans FOLD segment weights into the query (see query.cpp) and run the
// plain ScoreChunk at V1 speed — this dense primitive is the L2 path (weights sit
// inside the square and cannot fold), and the substrate for slot-2 decomposition.
// It scores each non-excluded row of one chunk over a segment
// list — per-segment partials from the SAME per-row kernels ScoreChunk uses, combined
// as sum(weight_s * partial_s) in segment order. The degenerate one-segment list
// (0, paddedDims, 1.0) is bit-identical to ScoreChunk by construction (same kernel
// entry points; x*1.0f is bitwise identity). Weight-0 segments and omitted ranges are
// never read — masking is a bandwidth cut. Callers validate segments first
// (ValidateSegments); kernels do not re-validate.
void ScoreChunkSegmented(
	const BankView& bank,
	const float* paddedQuery,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout,
	const float* rowBias = nullptr,
	bool* outNonFiniteBias = nullptr);

// Segmented intersection: the fused worst-of law over segmented totals — each member
// query scores through the same segmented per-row path, then worst-of in the metric's
// better-direction. One segment list applies to every member query.
void ScoreChunkFusedSegmented(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout,
	const float* rowBias = nullptr,
	bool* outNonFiniteBias = nullptr);

// Per-row decomposition (V2 section 6): scores one row over the segment list and
// surfaces the post-scale post-weight per-segment contributions; the returned total
// is their ordered sum, bit-exact by construction. Per-hit cost, not per-row: call
// it on hits, not banks. Callers validate segments first.
float DecomposeRowScore(
	const BankView& bank,
	const float* paddedQuery,
	int32_t rowIndex,
	const QuerySegment* segments,
	int32_t segmentCount,
	float* outContributions);

// --- v2.2 cross-device exactness (plan section 19) ---
//
// A query quantized for Exactness::CrossDevice: int8 elements on the bank's padded
// grid plus the exact per-query dequant scale (kept in double - no float round-trip,
// so a tiny scale can never be stored as a subnormal float) and the query's integer
// self-dot (the L2 epilogue's Sum(q_i^2) term).
struct XdQuery
{
	const int8_t* q8 = nullptr;
	double scale = 0.0;
	int64_t sqSum = 0;
};

// Quantizes a validated padded float query for CrossDevice scoring. Symmetric
// per-query scale (max |q_i| / 127, the row-bake math), with every arithmetic step
// deterministic across machines: elements are decoded from their IEEE bit patterns
// (per-thread DAZ state cannot flush a subnormal input), intermediates are computed
// in double (all intermediate values are provably normal, so FTZ never fires), and
// rounding is round-half-even implemented in integer math on the bit pattern (no FP
// rounding-mode dependence). A query whose max |q_i| is zero quantizes to all zeros
// with scale 0. outQ8 must hold paddedDims bytes; pad lanes are zeroed.
void QuantizeQueryXd(
	const float* paddedQuery,
	int32_t paddedDims,
	int8_t* outQ8,
	double* outScale,
	int64_t* outSqSum);

// CrossDevice chunk scorers - the integer-accumulation twins of ScoreChunk /
// ScoreChunkSegmented / ScoreChunkFused. Integer sums are associative, so every
// SIMD width produces identical accumulators; the per-row epilogue (dequant scales,
// per-channel inverse sub-norms, segment weights, bias) is one fixed-order double
// expression ending in the subnormal floor: |score| < FLT_MIN converts to exactly
// 0.0f, otherwise one double->float conversion of a normal value. Callers validate
// the CrossDevice laws first (Int8 bank, paddedDims <= kMaxCrossDeviceDims).
void ScoreChunkXd(
	const BankView& bank,
	const XdQuery& query,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout,
	const float* rowBias = nullptr,
	bool* outNonFiniteBias = nullptr);

// Segmented CrossDevice scan: per-range integer partials with weights-at-combine in
// double (CrossDevice never folds weights into the query - folding would re-quantize
// the weighted query and change the error model per segment list). Range structure,
// gap discards, and channel inverse-sub-norm placement mirror ScoreChunkSegmented.
void ScoreChunkSegmentedXd(
	const BankView& bank,
	const XdQuery& query,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout,
	const float* rowBias = nullptr,
	bool* outNonFiniteBias = nullptr);

// Fused (intersection) CrossDevice scan: worst-of over per-query CrossDevice scores
// in the metric's better-direction; selection over bit-identical floats is
// rounding-free. Queries are pre-quantized, one XdQuery each; a null `segments`
// scores whole rows.
void ScoreChunkFusedXd(
	const BankView& bank,
	const XdQuery* queries,
	int32_t queryCount,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout,
	const float* rowBias = nullptr,
	bool* outNonFiniteBias = nullptr);

// CrossDevice per-row decomposition: per-segment contributions from the same integer
// partials + double epilogue the CrossDevice scan runs. Each reported contribution
// carries the subnormal floor (so the reported values are themselves cross-device
// bit-identical); the returned total is the scan's score bitwise (the double chain,
// converted once) - NOT the float re-sum of the reported contributions, which is a
// PerDevice-mode property only.
float DecomposeRowScoreXd(
	const BankView& bank,
	const XdQuery& query,
	int32_t rowIndex,
	const QuerySegment* segments,
	int32_t segmentCount,
	float* outContributions);

// Scores one row whole-row CrossDevice (the sparse-bias rescore path).
float ScoreRowXd(const BankView& bank, const XdQuery& query, int32_t rowIndex);

// Kernel path selected at compile time (NEON/SSE/scalar), plus a runtime AVX2+FMA
// upgrade on x86 hardware that supports it. Dispatch is per-device stable, so the
// per-device determinism promise is unaffected. Exposed for tests and diagnostics.
enum class SimdPath : uint8_t
{
	Scalar = 0,
	SSE = 1,
	NEON = 2,
	AVX2 = 3,
};
SimdPath ActiveSimdPath();

namespace detail
{
	// Scalar kernels mirror the active SIMD path's accumulation exactly, so scalar and
	// SIMD results are bit-identical on a device:
	//   - SSE/NEON mirror: four 4-lane stripes, unfused multiply-then-add.
	//   - AVX2 mirror: four 8-lane stripes accumulated with std::fma (hardware FMA and
	//     std::fma round identically).
	float DotF32Scalar(const float* row, const float* query, int32_t paddedDims);
	float L2F32Scalar(const float* row, const float* query, int32_t paddedDims);
	float DotI8Scalar(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8Scalar(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	float DotF32ScalarAvx2(const float* row, const float* query, int32_t paddedDims);
	float L2F32ScalarAvx2(const float* row, const float* query, int32_t paddedDims);
	float DotI8ScalarAvx2(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8ScalarAvx2(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	float DotF32(const float* row, const float* query, int32_t paddedDims);
	float L2F32(const float* row, const float* query, int32_t paddedDims);
	float DotI8(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	// The scalar mirror of whatever path ActiveSimdPath() reports — what the
	// bit-equality tests compare against.
	float DotF32Mirror(const float* row, const float* query, int32_t paddedDims);
	float L2F32Mirror(const float* row, const float* query, int32_t paddedDims);
	float DotI8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	// --- v2.2 cross-device internals, exposed for the proof tests ---

	// Decodes a float's IEEE-754 bit pattern to the exactly equal double via integer
	// math on the mantissa, so per-thread DAZ state cannot flush a subnormal input.
	// (A normal input takes the plain conversion, which DAZ does not touch.)
	double FloatBitsToDouble(float v);

	// Round-half-even of a finite double to int32, implemented in integer math on
	// the bit pattern: no FP rounding-mode dependence, no library rounding call.
	// |v| must be < 2^31 (the quantizer bounds it at ~127).
	int32_t RoundHalfEvenI32(double v);

	// Integer row kernels: exact int32 sums, identical for any accumulation order.
	// L2SumsI8I8 returns the cross term Sum(q_i * r_i) and the row self term
	// Sum(r_i^2) from one pass (the query self term is per-query, hoisted).
	int32_t DotI8I8(const int8_t* row, const int8_t* q8, int32_t n);
	void L2SumsI8I8(const int8_t* row, const int8_t* q8, int32_t n,
		int32_t* outCross, int32_t* outRowSq);

	// CrossDevice bias composition on a floored unbiased score:
	// XdFloor((double)unbiased + (double)bias). The sparse k+P selection composes
	// through this exact expression, which is also the dense in-scan form - that
	// identity is what keeps sparse == dense-equivalent bitwise in CrossDevice mode.
	float XdComposeBiasValue(float unbiasedScore, float bias);

	// Test-only dispatch override for the CrossDevice forced-path matrix (plan 19.4
	// W1: the promise is any machine at ANY width, so CI asserts scalar, SSE4.1, and
	// AVX2 / NEON explicitly, not just the OS-default path). Forcing a path the
	// hardware cannot run is the caller's error. Not thread-safe; the shipped
	// default (no force) is per-device-stable dispatch.
	void ForceXdSimdPath(SimdPath path);
	void ClearForcedXdSimdPath();
}

} // namespace superfaiss

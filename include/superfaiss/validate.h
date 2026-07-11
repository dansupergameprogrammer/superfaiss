#pragma once

#include "types.h"

namespace superfaiss
{

// Structural validation of a bank view: alignment, stride, enum ranges, scales
// presence, and the format geometry ceilings (count <= kMaxBankRows,
// dims <= kMaxCrossDeviceDims -> BadFormat, checked before any size arithmetic
// touches the header values). Cheap; intended to run once at bank load, not per
// query.
Status ValidateBank(const BankView& bank);

// Query validation: pointer alignment, finiteness, zero-filled pad lanes, and the
// cosine zero-norm rule. This gate covers the QUERY side only; bank content is
// covered by ValidateBankData below. Note: finite inputs whose products/sums overflow
// float range can still produce Inf/NaN *scores*; keep embedding magnitudes sane
// (normalized embeddings cannot overflow).
Status ValidateQuery(const BankView& bank, const float* paddedQuery);

// Segmented-query validation (V2): segment count in [1, kMaxSegments]; offsets and
// lengths positive, on the 16-byte element grid for the bank's quantization,
// ascending and non-overlapping, ending within paddedDims; weights finite. On Cosine
// banks, a zero-norm query sub-vector over a nonzero-weight segment is rejected as
// ZeroNormQuery (the whole-vector zero-norm law, applied per scored segment) —
// weight-0 segments are skipped before validation, matching their omission
// semantics. Call after ValidateQuery.
Status ValidateSegments(
	const BankView& bank,
	const float* paddedQuery,
	const QuerySegment* segments,
	int32_t segmentCount);

// Sparse-bias validation (v2.1): pairCount in [0, bank.count]; every index unique
// and in [0, count); every bias value finite (the finite-only law - NonFiniteQuery).
// seenBits is caller scratch, ceil(count/32) zeroed uint32 words; on return the pair
// rows' bits are set (callers reuse them to detect pair rows among scan candidates).
// Dense bias is NOT pre-validated - its non-finite check fuses into the scan
// (T-055 W2; a pre-pass would re-read count x 4 bytes and eat the int8 budget).
Status ValidateBiasPairs(
	const BankView& bank,
	const BiasPair* pairs,
	int32_t pairCount,
	uint32_t* seenBits);

// Full bank-content validation — O(count x paddedDims); intended for load time, not
// per query. Rejects what the kernels cannot tolerate: non-finite float32 lanes (a NaN
// score breaks the top-k total order), non-zero pad lanes (silently wrong L2),
// non-finite or negative int8 scales (a negative scale inverts a row's ranking),
// int8 values of -128 (outside the bake clamp; admitting them would void the
// CrossDevice overflow proof), and Cosine rows whose norm deviates from 1 beyond
// the bake's own tolerance (float rounding, plus scale/2 x sqrt(dims) quantization
// error on int8) — non-unit Cosine rows silently corrupt ranking, and zero-norm
// rows are illegal on Cosine banks by the format's own law.
// Writes the first offending row to outBadRow if non-null.
Status ValidateBankData(const BankView& bank, int32_t* outBadRow);

} // namespace superfaiss

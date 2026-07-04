#pragma once

#include "types.h"

namespace superfaiss
{

// Structural validation of a bank view: alignment, stride, enum ranges, scales
// presence. Cheap; intended to run once at bank load, not per query.
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

// Full bank-content validation — O(count x paddedDims); intended for load time, not
// per query. Rejects what the kernels cannot tolerate: non-finite float32 lanes (a NaN
// score breaks the top-k total order), non-zero pad lanes (silently wrong L2), and
// non-finite or negative int8 scales (a negative scale inverts a row's ranking).
// Writes the first offending row to outBadRow if non-null.
Status ValidateBankData(const BankView& bank, int32_t* outBadRow);

} // namespace superfaiss

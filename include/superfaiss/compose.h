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

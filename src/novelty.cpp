// SuperFAISS V3.2 — Bank Inspector I, module M2 (novelty.h).
//
// NoveltyScore: the empirical-CDF rank of a probe distance within a sorted baseline
// distribution (plan section 25.4) — limb 2's statistic. Pure arithmetic over a
// caller-provided sorted array; touches no kernel, quantization, or format. The rest of
// M2 (KthNeighborDistance, CalibrateNoveltyBaseline, the two-limb verdict) lands after
// the F-M2-1 contract (D-V32-50): the NoveltyProbeDistance dispatch entry.

#include "superfaiss/novelty.h"

#include <algorithm>

namespace superfaiss
{

Status NoveltyScore(const float* sortedBaseline, int32_t count, float distance, float* outScore)
{
	if (sortedBaseline == nullptr || count < 1 || outScore == nullptr)
	{
		return Status::InvalidArgument;
	}

	// Count of entries strictly less than `distance`: with the baseline ascending, this is
	// the position of the first entry >= distance (std::lower_bound) — ties (an entry equal
	// to distance) land at or after that position, so they are excluded from the count,
	// which is exactly "ties resolve to the lowest rank."
	const float* first = sortedBaseline;
	const float* last = sortedBaseline + count;
	const float* it = std::lower_bound(first, last, distance);
	const int32_t strictlyLess = static_cast<int32_t>(it - first);

	*outScore = static_cast<float>(strictlyLess) / static_cast<float>(count);
	return Status::Ok;
}

} // namespace superfaiss

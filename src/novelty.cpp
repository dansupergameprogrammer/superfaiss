// SuperFAISS V3.2 — Bank Inspector I, module M2 (novelty.h).
//
// NoveltyScore: the empirical-CDF rank of a probe distance within a sorted baseline
// distribution (plan section 25.4) — limb 2's statistic. Pure arithmetic over a
// caller-provided sorted array; touches no kernel, quantization, or format. GREEN.
//
// RED SCAFFOLD (Curie, 2026-07-18): NoveltyProbeDistance, KthNeighborDistance, and
// CalibrateNoveltyBaseline are UNIMPLEMENTED. Each returns Status::Ok and writes NO
// output — the proven M1/NoveltyScore red-suite pattern (the no-op stub): every cell in
// tests/test_main.cpp fails for its cell's reason against this stub — a FEAT/oracle cell
// sees Ok with a poison-initialized output where it expects the real distance/rank; a
// dim-2 rejection cell sees Ok where it expects InvalidArgument. Brunel replaces these
// bodies with the real implementation; the contract is novelty.h.

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

Status NoveltyProbeDistance(const BankView&, const float*, int32_t, int32_t, float*)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

Status KthNeighborDistance(
	const BankView&, const float*, int32_t, const uint32_t*, float*, Workspace&)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

Status CalibrateNoveltyBaseline(
	const BankView&, int32_t, int32_t, float*, int32_t*, Workspace&)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

} // namespace superfaiss

#pragma once

#include "types.h"
#include "alloc.h" // Workspace

// SuperFAISS V3.2 — Bank Inspector I, Tier 1 module M2 (plan section 25.4): the
// k-th-nearest-neighbour distance probe + baseline calibration + two-limb tri-state
// novelty verdict that backs the Inspector's Novelty view. Post-processing over exact
// query output; touches no kernel, quantization, or format.
//
// Determinism tier: PER-DEVICE (fixed sample, fixed order, integer rank over a sorted
// fixed baseline, ties pinned to the lowest rank). No CrossDevice claim.
//
// AUTHORING STATE (Curie, 2026-07-18): NoveltyScore is authored — its contract is fully
// pinned by section 25.4 and is independent of F-M2-1. KthNeighborDistance,
// CalibrateNoveltyBaseline, and the two-limb verdict entry (limb 1 = the metric's exact
// distance function == 0.0f) are pending the F-M2-1 contract decision (which public entry
// computes the exact per-path/per-quant distance), then land here.

namespace superfaiss
{

// The empirical-CDF rank of a probe distance within the baseline distribution (plan 25.4):
// the fraction of `sortedBaseline` entries STRICTLY LESS THAN `distance` (ties resolve to
// the LOWEST rank), normalized to [0, 1]. `sortedBaseline` is ascending, `count` entries.
// count < 1 or a null buffer -> InvalidArgument, no write. This is limb 2's statistic; the
// verdict compares the result against lambda (>= lambda => novel, else familiar).
Status NoveltyScore(
	const float* sortedBaseline,
	int32_t count,
	float distance,
	float* outScore);

} // namespace superfaiss

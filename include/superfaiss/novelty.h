#pragma once

#include "types.h"
#include "alloc.h" // Workspace

// SuperFAISS V3.2 — Bank Inspector I, Tier 1 module M2 (plan section 25.4): the
// k-th-nearest-neighbour distance probe + baseline calibration + two-limb tri-state
// novelty verdict that backs the Inspector's Novelty view. Post-processing over exact
// query output; touches no kernel, quantization, or format.
//
// Determinism tier: PER-DEVICE (fixed sample, fixed order, integer rank over a sorted
// fixed baseline, ties pinned to the lowest rank). No CrossDevice claim. The float32
// whole-row/channel legs of NoveltyProbeDistance accumulate in DOUBLE (the epilogue
// convention every other reduction in this library uses — analytics.cpp's XdPairScore/
// XdChannelPairScore, HitDistance, the NN reductions), ending in the same subnormal floor
// as the int8 legs; this is pinned here so both the implementation and the test author's
// independent oracle build against one arithmetic, not two that might silently diverge.
//
// CLOSED GREEN (M2, 2026-07-19; Poirot-reviewed). The two-limb tri-state verdict
// (D-V32-31/47) is composed by the CALLER from these four functions (no separate "verdict"
// entry exists; graph.h's precedent — ConnectedComponents composes BuildKnnNeighbors +
// MutualFilter + BuildDuplicateGroups — is the same shape).

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

// The limb-1 exact-distance entry (plan 25.4, D-V32-47/48/50, F-M2-1): the metric's OWN
// exact distance function between `paddedProbeQuery` and one stored row, dispatched per
// (metric, scope, quant) — the primitive both the implementation and the test author's
// independent oracle build against and cross-check by execution (the strike-14 discipline:
// one verified number, not two hopefully-agreeing ones). `channel == -1` scores the whole
// row; `channel` in [0, bank.channelCount) scores that channel's sub-range only.
//
// Dispatch:
//   - int8, whole-row: the XdPairScore arithmetic (analytics.cpp) — int8 cross/self-dot
//     accumulation, one fixed-order double epilogue, the subnormal floor (XdFloor). Cosine:
//     `1 - cross/sqrt(aSq*bSq)`; L2: the expanded `||aScale*a - bScale*b||^2` form. Requires
//     a nonzero self-dot on both operands under Cosine — always true here, since a
//     whole-row-normalized bank rejects a zero-norm row at bake and the probe is a query
//     (rejected upstream of this call on a Cosine bank).
//   - float32, whole-row: the SAME two formulas, accumulated in DOUBLE over the plain
//     floats (see the header note above), ending in the same subnormal floor.
//     **Disclosed limit (F-M2-5, D-V32-51, ratified by Dan 2026-07-19, commit 37cb3c9c27):
//     this leg does NOT carry int8's "exact zero for any scalar multiple" guarantee.** Int8 exactness rests on cross/aSq/bSq being
//     EXACT INTEGER sums (D-V32-47) — zero rounding until the one final division+sqrt, so
//     parallel int8 codes give a provably exact ratio. Float32 has no such step: even
//     constructing a scaled probe (`c * row`) is itself a rounded float32 multiplication,
//     so the probe is not bit-exactly parallel to the row before this function ever sees
//     it — confirmed executed (scale c in {0.1, 25, 1000} against a real Cosine bank:
//     nonzero, scattered residuals, not a threshold-shaped failure — genuine rounding
//     noise, not a formula defect). This entry still computes the honest distance; the
//     guarantee that a nonzero result is impossible for a TRUE duplicate holds only for
//     BYTE-IDENTICAL float32 probe/row (trivially exact there: cross == aSq == bSq). A
//     caller reading this as limb 1 gets a real, non-thresholded distance on every float32
//     input; it just cannot assume every scalar-multiple duplicate reads exactly 0 the way
//     every int8 one does.
//   - channel (either quant): the sliced form. The probe is quantized (int8 leg, via the
//     same arithmetic QuantizeQueryXd uses) or read (float32 leg) over the WHOLE
//     bank.paddedDims once, then BOTH operands restrict every sum to
//     `[bank.channels[channel].offset, +length)`; the stored row keeps its own whole-row
//     int8 scale (channel scoring never renormalizes per channel — analytics.cpp's
//     XdChannelPairScore convention). **Cosine only, BOTH sides (D-V32-43, strike 12):** a
//     zero-energy channel slice on EITHER the probe or the stored row returns
//     `Status::ZeroNormQuery`, no write — a directionless slice is a Cosine duplicate of
//     nothing, so it is never silently floored to a false distance-0 match the way the
//     shipped `XdChannelPairScore` reduction floors it for its own (different) purpose.
//     Channel L2 needs no such guard (a zero-energy slice scores a real positive distance).
//   - Dot: never dispatched — `bank.metric == Metric::Dot` is `InvalidArgument` (the
//     verdict-unavailable decision is made upstream of this call; this entry serves only
//     the verdict domain, L2 and Cosine — D-V32-22/23).
//
// Rejections (InvalidArgument, no write): `bank.count < 1`; `storedRow` outside
// `[0, bank.count)`; `channel` outside `[-1, bank.channelCount)`, or `channel != -1` on a
// bank with no channel table; a null `paddedProbeQuery`/`outDistance`; `bank.metric ==
// Metric::Dot`.
// `workspace` stages the int8 leg's quantized probe (ReserveXdQuery/XdQ8/XdScale/XdSqSum)
// -- the same warm, tracked buffer KthNeighborDistance/CalibrateNoveltyBaseline reuse via
// their own `workspace` params, never a per-call heap allocation (Finding 3,
// cf3f750-v32-core-batch-review.md: this entry used to bypass the library's own
// AllocationCount() seam via a raw std::vector). The float32 leg does not touch
// `workspace` (no quantization to stage).
Status NoveltyProbeDistance(
	const BankView& bank,
	const float* paddedProbeQuery,
	int32_t storedRow,
	int32_t channel, // -1 = whole-row
	float* outDistance,
	Workspace& workspace);

// The k-th-nearest-neighbour distance of `query` against the FULL view (plan 25.4), the
// probe of research 4.4, converted to a RankDistance BEFORE anything ranks it (D-V32-19 —
// `Hit.score` is a similarity on Cosine/Dot, higher = nearer, and a distance only on L2;
// ranking the raw score inverts the verdict). The pinned conversion, `RankDistance(metric,
// score)`: L2 -> `score` (already a squared distance); Cosine -> `1 - score` (bake-
// normalized rows, so the hit score is a similarity). Dot is EXCLUDED from the verdict
// domain (D-V32-22: `-score` is monotone but not a dissimilarity on non-normalized Dot
// rows — a verdict would track magnitude, not isolation) and returns `InvalidArgument`;
// the reject-over-degrade line the plan draws — this function never silently answers a
// cosine question a Dot bank never asked.
//
// `excludeBits` is the caller's view-space exclusion (tombstones|self for an archive
// source, self only for a fresh asset — V32-G7); null excludes nothing. This is a raw
// k-th-hit probe, not self-widening: the caller sizes `k` to what it needs surviving after
// its own exclusions (mirrors `Query`'s own contract, unlike `BuildKnnNeighbors`'
// self-widening `excludeSelf` flag). `workspace` warm for `k`.
//
// Rejections (InvalidArgument, no write): `k < 1`; `k` greater than the number of
// non-excluded rows in `bank`; a null `query`/`outDistance`; `bank.metric == Metric::Dot`.
Status KthNeighborDistance(
	const BankView& bank,
	const float* query,
	int32_t k,
	const uint32_t* excludeBits,
	float* outDistance,
	Workspace& workspace);

// The bank's own k-th-NN RankDistance distribution (plan 25.4): `bank` is the sample view
// the caller already constructed (the "one shared cap" of section 25.3 — the same
// compacted, live-rows-only, tombstone-free sample Structure runs on); this function does
// NOT re-sample or re-stride. For each row of `bank`, its own RankDistance to its k-th
// nearest OTHER row within `bank` (self excluded, V32-G7), written to
// `outSortedDistances` in ASCENDING order; `outCount` receives `bank.count`.
// `sampleLimit` is the caller's declared cap on `bank.count` (defensive: `bank.count >
// sampleLimit` is `InvalidArgument` — a view exceeding its own declared cap is a caller
// contract violation this function refuses to silently absorb), not a re-sampling
// parameter — `outSortedDistances`/`outCount` need no more than `sampleLimit` capacity.
// Dot -> `InvalidArgument` (RankDistance is undefined there, exactly as
// KthNeighborDistance). One pass per bank per parameter set; cached per section 25.3.
//
// Rejections (InvalidArgument, no write): `k < 1`; `k >= bank.count`; `sampleLimit < 1`;
// `bank.count > sampleLimit`; `bank.count < 1`; a null `bank.rows`/`outSortedDistances`/
// `outCount`; `bank.metric == Metric::Dot`. `workspace` warm for `k + 1` (self-exclusion
// widens the top-k heap by one internally, mirroring `BuildKnnNeighbors`).
Status CalibrateNoveltyBaseline(
	const BankView& bank,
	int32_t k,
	int32_t sampleLimit,
	float* outSortedDistances,
	int32_t* outCount,
	Workspace& workspace);

} // namespace superfaiss

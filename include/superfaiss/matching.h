#pragma once

#include "types.h"
#include "alloc.h" // Workspace

// SuperFAISS V3.2 — Bank Inspector I, Tier 1 module M3 (plan section 25.4, RE-MECHANIZED
// D-V32-16): mutual-NN correspondence + CSLS margins that back the Inspector's
// Correspondence view (View C). Sampled-A-VERIFIED-AGAINST-FULL-BANKS (supersedes the
// deleted two-sided sample-view design, whose independent striding made a pair
// discoverable only if both sides co-sampled -- measured 0.9% recovery / 98% spurious at
// 500k, the fourth fracture). A reported pair is KERNEL-TRUE: both the forward match (a
// sampled A row's nearest in full B) and the back-verification (that candidate's nearest
// in full A) are checked against COMPLETE banks, never a sample; the A sample bounds only
// COVERAGE (how many A rows were checked), never correctness.
//
// Determinism tier: PER-DEVICE (fixed order, deterministic top-k selection, ties ascending
// index — V32-G5's convention, inherited from the query path this rides). No CrossDevice
// claim, matching graph.h/novelty.h.
//
// This is the Inspector's disclosed HEAVY pass: O(cap x liveCountB x dims) for the forward
// pass plus O(distinct candidates x liveCountA x dims) for back-verification, linear in
// bank size and NOT sub-second at scale — the panel discloses this before running (plan
// 25.3/25.4). Chunking, progress, cancel, and the upfront wall-clock estimate
// (`Est = 2 * cap * liveCountB * dims / Rate_quant`, T4-W1) are the M4 widget's slow-task
// wrapper around this call, not part of this contract — the graph.h/novelty.h precedent:
// core modules are single synchronous passes over whatever view they are handed; chunking
// and calibrated-constant arithmetic are layered on top by the caller.
//
// AUTHORING STATE (Curie, 2026-07-19): first-round contract + red suite. See the
// test-design artifact (`Claude/Curie/superfaiss-v3.2-test-design-2026-07-18.md`) for the
// routed findings on points this round's plan text left ambiguous (F-M3-*).

namespace superfaiss
{

// One sampled A row's correspondence outcome. `sourceIndexA` is the row's NATIVE A source
// index (via `sampleSourceIndices` — the A-side remapping the sample construction carries,
// T4-W2); `sourceIndexB` is the mutual partner's NATIVE B source index, or -1 when no
// mutual partner exists (UNMATCHED — an honest outcome by design; many rows stay
// unmatched, research). `cslsMargin` is defined only when `sourceIndexB >= 0`.
// Classification (matched vs. ambiguous, by `CslsMarginThreshold`) is CALLER-composed,
// exactly as `NoveltyScore`'s raw statistic is compared against lambda by the caller
// rather than baked into the primitive — no separate "verdict" entry exists here either,
// mirroring the novelty.h precedent.
struct MatchPair
{
	int32_t sourceIndexA = -1;
	int32_t sourceIndexB = -1;
	float cslsMargin = 0.0f;
};

// Sampled-A-verified-against-full-banks mutual matching (plan 25.4, D-V32-16/20):
//   - Pass 1: for each row of `sampleViewA`, its top-`matchK` nearest rows in the FULL
//     live `fullViewB` (best-first, the bank's own metric; `excludeBitsB` honors B's own
//     tombstones in B's own source space) — the forward candidate partner is the top-1
//     entry; `r_B(i)` is the mean `Sim()` (see below) of the top-`matchK` scores.
//   - Pass 2: for each DISTINCT candidate B row surfaced by pass 1, its top-`matchK`
//     nearest rows in the FULL live `fullViewA` (`excludeBitsA` honors A's own tombstones,
//     A's own source space) — back-verification succeeds iff the candidate's top-1 in full
//     A is the sampled row's OWN native source index (`sampleSourceIndices[i]`); `r_A(j)`
//     is the mean `Sim()` of that top-`matchK`.
//   - A pair is MUTUAL iff back-verification succeeds; `sourceIndexB`/`cslsMargin` are
//     written only then, else `sourceIndexB = -1` and `cslsMargin = 0.0f`. No third pass
//     exists — both r-terms and the margin compute entirely from these two retrievals
//     (T4-S2); back-verification is pass 2's own top-1, not a separate lookup.
// CSLS margin (D-V32-20): `csls(i,j) = 2*Sim(i,j) - r_B(i) - r_A(j)`, where `Sim(metric,
// score) = -RankDistance(metric, score)` — L2's raw score is lower-is-better and would
// invert the margin exactly as an un-converted L2 score inverted the Novelty verdict
// (D-V32-18); Dot's raw score is already similarity-directioned (`Sim(Dot, score) =
// score`, identity — Dot never runs through `RankDistance`, which excludes it, but CSLS's
// own `Sim` is defined for all three metrics since the mutual test is metric-agnostic).
// `sim(i,j)` itself is pass 1's top-1 entry, `Sim()`-converted. CSLS is disclosed as a
// DIAGNOSTIC on Dot/L2 banks (not the paper's calibrated cosine-only setting) — the mutual
// test's correctness does not depend on the metric.
// `outPairs` carries `sampleViewA.count` entries, one per sample row, in sample-row order;
// `outPairCount` receives `sampleViewA.count` (every sample row is always processed — no
// partial/cancelled path exists at this synchronous, single-call level; a cancelled M4
// pass is the WIDGET discarding a call it never made, not a state this function reports).
// `workspace` warm for `matchK` (both passes retrieve top-`matchK`).
//
// Rejections (InvalidArgument, no write): `sampleViewA.count < 1`; `fullViewB.count < 1`;
// `fullViewA.count < 1`; `matchK < 1`; `matchK` greater than the number of non-excluded
// rows in `fullViewB` OR in `fullViewA`; `sampleViewA.dims`, `fullViewB.dims`, and
// `fullViewA.dims` not all equal; `sampleViewA.metric`, `fullViewB.metric`, and
// `fullViewA.metric` not all equal (the "Dims and Metric must match" law, plan 25.3
// E-D1-3 — Quantization MAY differ across views: matching runs on query scores over
// dequantized rows); a null `sampleSourceIndices`/`outPairs`/`outPairCount`.
Status MutualNearestMatches(
	const BankView& sampleViewA,
	const int32_t* sampleSourceIndices,
	const BankView& fullViewB,
	const uint32_t* excludeBitsB,
	const BankView& fullViewA,
	const uint32_t* excludeBitsA,
	int32_t matchK,
	MatchPair* outPairs,
	int32_t* outPairCount,
	Workspace& workspace);

} // namespace superfaiss

#pragma once

#include "types.h"
#include "kernels.h" // XdQuery
#include "alloc.h"   // Workspace

// SuperFAISS V2.5 — bank analytics (plan section 22). Cross-device deterministic
// reductions over int8 banks: a set-to-set distance, directed nearest-neighbour
// divergence, and within-bank dispersion, plus the shared query-vs-query pair score
// they rest on. Drift over checkpoints is CentroidDistanceCrossDevice between
// consecutive checkpoints' row sets; spread is the centroid-dispersion form.
//
// One determinism argument covers all of these (plan section 22.5): every scalar is a
// fixed-order double epilogue over cross-device-exact integer sums, ending in the
// subnormal floor. Dot/L2 reuse the shipped section 19 epilogue arithmetic; cosine adds
// one runtime sqrt (the section 22.5 item 2b limb), cross-device-exact by IEEE-754
// correctly-rounded sqrt under the project's true-sqrt / no-fast-math build regime.

namespace superfaiss
{

// Reduction kind for the divergence/spread reductions.
enum class Reduce : uint8_t
{
	Mean = 0, // serial double accumulation in ascending row index, then one divide (plan 22.5 N2)
	Max = 1,  // order-free running max
};

// Cross-device score between two XdQuery payloads (pooled centroids or lifted bank
// rows), in the metric's DISTANCE sense (larger = farther):
//   - Dot:    the dot similarity `crossDot * a.scale * b.scale` (documented similarity,
//             not a distance; a Dot consumer reads it with that convention).
//   - L2:     the squared distance `a.scale^2*a.sqSum + b.scale^2*b.sqSum
//             - 2*a.scale*b.scale*crossDot` (the shipped section 19 expansion on a pair).
//   - Cosine: `1 - crossDot / sqrt(a.sqSum * b.sqSum)` — the scales cancel; the one
//             runtime sqrt outside the shipped {+,-,*,/} epilogue (plan S1). Requires a
//             nonzero self-dot on both members (else ZeroNormQuery).
// Every result ends in the subnormal floor. Both images are `paddedDims` int8 elements
// on the padded grid; `paddedDims` in [1, kMaxCrossDeviceDims].
//
// Validates BOTH payloads (the shipped QueryXd payload law, on each member): scale finite
// and non-negative, self-dot recomputed from the image and matched (a desynced payload is
// InvalidArgument). This is the public boundary; the reductions below score trusted
// internal payloads (baked rows / operator products) without re-validation (plan W2).
Status ScoreXdPair(const XdQuery& a, const XdQuery& b, int32_t paddedDims, Metric metric,
	float* outScore);

// Set-to-set centroid distance (plan 22.4): pools each row selection with
// MakeCentroidCrossDevice and scores the pair with ScoreXdPair. Drift over checkpoints is
// this operator between consecutive checkpoints' row sets. `weights`/`excludeBits` per
// side follow MakeCentroidCrossDevice (null = unweighted / no exclusion). Both banks int8
// with equal paddedDims. `centroidScratchA`/`B` are caller scratch of paddedDims bytes
// each (zero steady-state allocation). `metric` selects the distance family. Propagates a
// pooling rejection (empty set -> InvalidArgument; zero-norm accumulator -> ZeroNormQuery).
Status CentroidDistanceCrossDevice(
	const BankView& bankA, const int32_t* rowIndicesA, int32_t rowCountA,
	const int32_t* weightsA, const uint32_t* excludeBitsA,
	const BankView& bankB, const int32_t* rowIndicesB, int32_t rowCountB,
	const int32_t* weightsB, const uint32_t* excludeBitsB,
	Metric metric, int8_t* centroidScratchA, int8_t* centroidScratchB, float* outDistance);

// Directed mean nearest-neighbour set divergence (plan 22.4): for each non-excluded row of
// `source`, its nearest-neighbour distance to `target` (QueryXdBatch, k=1), reduced by a
// fixed-order mean (ascending source-row index, excluded rows skipped; plan N2). Scores in
// the TARGET bank's own metric (Cosine -> 1-cos distance, L2 -> squared distance, Dot ->
// similarity). Both banks int8 CrossDevice with equal paddedDims. `queryScratch` holds
// source.count XdQuery, `hitScratch` source.count Hit, `countScratch` source.count int32
// (all caller scratch); `ws` warm for k=1. Directed A->B; a symmetric divergence is the
// caller's reduce of both directions. Empty source or fully-excluded target ->
// InvalidArgument.
Status MeanNNCrossDevice(
	const BankView& source, const uint32_t* sourceExcludeBits,
	const BankView& target, const uint32_t* targetExcludeBits,
	XdQuery* queryScratch, Hit* hitScratch, int32_t* countScratch, Workspace& ws,
	float* outValue);

// Directed max nearest-neighbour set divergence (plan 22.4): the same batch as
// MeanNNCrossDevice, reduced by an order-free max (the directed Hausdorff component).
Status MaxNNCrossDevice(
	const BankView& source, const uint32_t* sourceExcludeBits,
	const BankView& target, const uint32_t* targetExcludeBits,
	XdQuery* queryScratch, Hit* hitScratch, int32_t* countScratch, Workspace& ws,
	float* outValue);

// Within-bank dispersion (plan 22.4, spread = centroid-dispersion, D-V2-11): the mean (or
// max) distance of each selected row to the selection's own MakeCentroidCrossDevice
// centroid, in the bank's metric. Reduction order and floor as MeanNN/MaxNN.
// `rowIndices`/`rowCount` are the rows to include (ascending for the pinned mean order);
// `excludeBits` is passed to the pooling. `centroidScratch` is caller scratch of
// paddedDims bytes.
Status SpreadCrossDevice(
	const BankView& bank, const int32_t* rowIndices, int32_t rowCount,
	const uint32_t* excludeBits, Reduce reduce, int8_t* centroidScratch, float* outValue);

// Feature A — the probe-direction projection report (plan 22.3). OFFLINE, per-device
// float, no cross-device claim: projects each row of `bank` onto the unit probe direction
// (`paddedDirection`, paddedDims floats, caller-normalized e.g. via MakeDirection) and
// writes the per-row projection (plain dot, int8 rows dequantized through their scales)
// into `outProjections` (bank.count floats). When `groupBits` is non-null (bit set = group
// A, clear = group B, over all rows) and `outSeparation` is non-null, writes the
// Cohen's-d separation `(mean_A - mean_B) / sqrt((var_A + var_B) / 2)` — the between-group
// projected-mean gap over the pooled projected standard deviation (plan N3). Rejects a
// zero-norm or non-finite direction, an empty bank, or (for separation) an empty group as
// InvalidArgument.
Status ProjectionReport(const BankView& bank, const float* paddedDirection,
	const uint32_t* groupBits, float* outProjections, float* outSeparation);

} // namespace superfaiss

// SuperFAISS V3.2 — Bank Inspector I, module M2 (novelty.h): the k-th-nearest-neighbour
// distance probe + baseline calibration + the limb-1 exact-distance primitive backing
// the Inspector's Novelty view (plan section 25.4). Post-processing over exact query
// output; touches no kernel, quantization, or format.
//
// NoveltyProbeDistance (F-M2-1 / D-V32-50): int8 whole-row delegates to the PUBLIC
// ScoreXdPair (analytics.h), which wraps the exact strike-13/14/19 ground-truth
// arithmetic (XdPairScore -> XdL2/XdCosineDistance) — byte-identical by construction,
// not by parallel reasoning. float32 and channel legs have no public single-pair exact
// distance to delegate to, so this file reproduces the SAME fixed-order-double epilogue
// file-locally, exactly the convention analytics.cpp itself uses for kernels.cpp's
// epilogue (its own header comment states the precedent).

#include "superfaiss/novelty.h"

#include "superfaiss/analytics.h"        // ScoreXdPair
#include "superfaiss/inspector_common.h" // DequantizeRowAsQuery
#include "superfaiss/kernels.h"          // XdQuery, QuantizeQueryXd, detail::DotI8I8/FloatBitsToDouble
#include "superfaiss/query.h"            // Query, QueryBatch

#include <algorithm>
#include <cmath>
#include <vector>

namespace superfaiss
{
namespace
{

// The v2.2 cross-device epilogue's shape, reproduced file-local (the analytics.cpp
// precedent) so the float32 and channel legs use the SAME fixed-order double arithmetic
// as the int8 leg's ScoreXdPair delegate — one formula per metric, not two that might
// silently diverge. `aScaleD`/`bScaleD` are 1.0 for float32 (no quantization scale).
inline float XdFloorLocal(double score)
{
	const double lim = 1.1754943508222875e-38; // FLT_MIN, exactly
	return (score < lim && score > -lim) ? 0.0f : static_cast<float>(score);
}

inline double XdL2Local(double cross, double aSq, double bSq, double aScaleD, double bScaleD)
{
	const double a = (aScaleD * aScaleD) * aSq;
	const double b = (bScaleD * bScaleD) * bSq;
	const double c = ((aScaleD * bScaleD) * cross) * 2.0;
	return (a + b) - c;
}

inline double XdCosineDistanceLocal(double cross, double aSq, double bSq)
{
	const double denom = std::sqrt(aSq * bSq);
	return 1.0 - cross / denom;
}

// L2 -> the score itself (already a squared distance); Cosine -> 1 - score (a
// bake-normalized similarity). Dot is never converted (excluded upstream of every M2
// entry point).
inline float RankDistanceLocal(Metric metric, float score)
{
	return metric == Metric::L2 ? score : 1.0f - score;
}

} // namespace

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

Status NoveltyProbeDistance(
	const BankView& bank, const float* paddedProbeQuery, int32_t storedRow, int32_t channel, float* outDistance)
{
	if (bank.count < 1 || storedRow < 0 || storedRow >= bank.count || paddedProbeQuery == nullptr ||
		outDistance == nullptr || bank.metric == Metric::Dot)
	{
		return Status::InvalidArgument;
	}
	if (channel < -1 || channel >= bank.channelCount || (channel != -1 && bank.channels == nullptr))
	{
		return Status::InvalidArgument;
	}

	const bool wholeRow = (channel == -1);
	const int32_t offset = wholeRow ? 0 : bank.channels[channel].offset;
	const int32_t length = wholeRow ? bank.paddedDims : bank.channels[channel].length;

	double cross = 0.0, aSq = 0.0, bSq = 0.0;
	double aScaleD = 1.0, bScaleD = 1.0;

	if (bank.quant == Quantization::Int8)
	{
		// The probe's scale is a property of the WHOLE query (QuantizeQueryXd's symmetric
		// per-query scale), so it is always quantized over the full paddedDims first; a
		// channel leg then restricts every sum to [offset, offset+length) on both operands
		// (the stored row keeps its own whole-row scale too — channel scoring never
		// renormalizes, the XdChannelPairScore convention).
		std::vector<int8_t> probeQ8(static_cast<size_t>(bank.paddedDims));
		double probeScale = 0.0;
		int64_t probeSqSum = 0;
		QuantizeQueryXd(paddedProbeQuery, bank.paddedDims, probeQ8.data(), &probeScale, &probeSqSum);

		const int8_t* rowBytes = static_cast<const int8_t*>(bank.rows) + static_cast<int64_t>(storedRow) * bank.paddedDims;
		const double rowScale = detail::FloatBitsToDouble(bank.scales[storedRow]);

		if (wholeRow)
		{
			// Delegate to the PUBLIC entry: byte-identical to the strike-verified ground
			// truth by construction, not by re-deriving the same arithmetic twice.
			const XdQuery probeQuery{probeQ8.data(), probeScale, probeSqSum};
			const XdQuery rowQuery{rowBytes, rowScale, detail::DotI8I8(rowBytes, rowBytes, bank.paddedDims)};
			return ScoreXdPair(probeQuery, rowQuery, bank.paddedDims, bank.metric, outDistance);
		}

		// NOTE (F-M2-3, routed): detail::DotI8I8's SIMD paths assume `n` is a multiple of
		// 16 (kernels.cpp's own "int8 ranges are multiples of 16" comment) and have no
		// scalar remainder tail — silently wrong (not a crash) for a non-grid-aligned
		// channel length. Every PRODUCTION channel table is grid-aligned by construction
		// (validate.cpp's offset/length % grid == 0 rejection), so this is safe for any
		// validated bank; it is NOT safe against a hand-built test BankView with an
		// off-grid channel table.
		cross = static_cast<double>(detail::DotI8I8(rowBytes + offset, probeQ8.data() + offset, length));
		aSq = static_cast<double>(detail::DotI8I8(probeQ8.data() + offset, probeQ8.data() + offset, length));
		bSq = static_cast<double>(detail::DotI8I8(rowBytes + offset, rowBytes + offset, length));
		aScaleD = probeScale;
		bScaleD = rowScale;
	}
	else
	{
		const float* row = static_cast<const float*>(bank.rows) + static_cast<int64_t>(storedRow) * bank.paddedDims;
		for (int32_t d = offset; d < offset + length; ++d)
		{
			const double p = static_cast<double>(paddedProbeQuery[d]);
			const double r = static_cast<double>(row[d]);
			cross += p * r;
			aSq += p * p;
			bSq += r * r;
		}
		// aScaleD/bScaleD stay 1.0 — float32 carries no quantization scale.
	}

	// Channel Cosine only (D-V32-43, strike 12): a zero-energy slice on EITHER side is
	// directionless, never a false distance-0 match. Whole-row Cosine needs no such guard
	// — a zero-norm whole row/query is rejected upstream by construction (bake-time for a
	// stored row, query validation for the probe), so this branch is unreachable there.
	if (!wholeRow && bank.metric == Metric::Cosine && (aSq == 0.0 || bSq == 0.0))
	{
		return Status::ZeroNormQuery;
	}

	const double distance = (bank.metric == Metric::L2) ? XdL2Local(cross, aSq, bSq, aScaleD, bScaleD)
														  : XdCosineDistanceLocal(cross, aSq, bSq);
	*outDistance = XdFloorLocal(distance);
	return Status::Ok;
}

Status KthNeighborDistance(
	const BankView& bank, const float* query, int32_t k, const uint32_t* excludeBits, float* outDistance,
	Workspace& workspace)
{
	if (bank.metric == Metric::Dot || query == nullptr || outDistance == nullptr || k < 1)
	{
		return Status::InvalidArgument;
	}
	int32_t available = 0;
	for (int32_t r = 0; r < bank.count; ++r)
	{
		if (!IsExcluded(excludeBits, r))
		{
			++available;
		}
	}
	if (k > available)
	{
		return Status::InvalidArgument;
	}

	if (!workspace.Reserve(k, 1))
	{
		return Status::OutOfMemory;
	}
	std::vector<Hit> hits(static_cast<size_t>(k));
	QueryParams params;
	params.k = k;
	params.excludeBits = excludeBits;

	int32_t hitCount = 0;
	const Status s = Query(bank, query, params, workspace, hits.data(), &hitCount);
	if (s != Status::Ok)
	{
		return s;
	}
	if (hitCount < k)
	{
		return Status::InvalidArgument; // fewer scorable rows than the pre-check counted
	}

	*outDistance = RankDistanceLocal(bank.metric, hits[static_cast<size_t>(k - 1)].score);
	return Status::Ok;
}

Status CalibrateNoveltyBaseline(
	const BankView& bank, int32_t k, int32_t sampleLimit, float* outSortedDistances, int32_t* outCount,
	Workspace& workspace)
{
	if (bank.metric == Metric::Dot || k < 1 || bank.count < 1 || k >= bank.count || sampleLimit < 1 ||
		bank.count > sampleLimit || bank.rows == nullptr || outSortedDistances == nullptr || outCount == nullptr)
	{
		return Status::InvalidArgument;
	}

	// Self-excluded k-th-NN: widen retrieval by one (mirroring BuildKnnNeighbors), since
	// each row is its own nearest and must be dropped from its own baseline entry.
	const int32_t internalK = k + 1;
	const int32_t count = bank.count;

	// The batch query path (matching graph.h's BuildKnnNeighbors — Poirot casebook
	// afabc08-graph-h-m1-review.md, S1/S2/S3): the chunk loop runs OUTERMOST across all
	// `count` queries in one bank pass. Tracked, warm-reusable query scratch; QueryBatch
	// manages its own internal top-k scratch via `workspace` (never given segments here,
	// so it never touches this region).
	if (!workspace.ReserveQueryScratch(bank.paddedDims, count))
	{
		return Status::OutOfMemory;
	}
	// Pack from the base at OUR OWN stride (bank.paddedDims) — never via
	// workspace.QueryScratch(r) for r > 0, whose internal stride can be wider than
	// bank.paddedDims on a warm workspace (query.cpp's own segmented-fold precedent).
	float* queryBase = workspace.QueryScratch(0);
	for (int32_t r = 0; r < count; ++r)
	{
		DequantizeRowAsQuery(bank, r, queryBase + static_cast<int64_t>(r) * bank.paddedDims);
	}

	// Caller-owned outHits, matching every QueryBatch call site in this codebase:
	// workspace's HeapStorage slots are QueryBatch's own per-sub-batch scratch, reused and
	// overwritten across sub-batches, never a stable place for a caller's persistent output.
	std::vector<Hit> allHits(static_cast<size_t>(count) * static_cast<size_t>(internalK));
	std::vector<int32_t> hitCounts(static_cast<size_t>(count));
	QueryParams params;
	params.k = internalK;
	const Status s = QueryBatch(bank, queryBase, count, params, workspace, allHits.data(), hitCounts.data());
	if (s != Status::Ok)
	{
		return s;
	}

	for (int32_t r = 0; r < count; ++r)
	{
		const Hit* rowHits = allHits.data() + static_cast<int64_t>(r) * internalK;
		const int32_t hitCount = hitCounts[static_cast<size_t>(r)];
		int32_t seen = 0;
		bool found = false;
		float rawScore = 0.0f;
		for (int32_t j = 0; j < hitCount; ++j)
		{
			if (rowHits[j].index == r)
			{
				continue; // self-exclude
			}
			if (++seen == k)
			{
				rawScore = rowHits[j].score;
				found = true;
				break;
			}
		}
		if (!found)
		{
			return Status::InvalidArgument; // fewer than k non-self neighbors available
		}
		outSortedDistances[r] = RankDistanceLocal(bank.metric, rawScore);
	}

	std::sort(outSortedDistances, outSortedDistances + count);
	*outCount = count;
	return Status::Ok;
}

} // namespace superfaiss

// SuperFAISS V3.2 — Bank Inspector I, module M3 (matching.h): sampled-A-verified-against-
// full-banks mutual matching + CSLS margins backing the Inspector's Correspondence view
// (plan section 25.4, RE-MECHANIZED D-V32-16). Post-processing over exact query output;
// touches no kernel, quantization, or format.
//
// The heavy pass: pass 1 batches all sampleViewA.count queries against fullViewB in one
// bank scan (the chunk-outermost amortization — Poirot's M1 S3 lesson, applied here from
// the outset); pass 2 batches only the DISTINCT candidate B rows pass 1 surfaced against
// fullViewA (no third pass — both r-terms and the margin compute entirely from these two
// retrievals, T4-S2). sampleViewA/fullViewB/fullViewA may differ in quantization (E-D1-3),
// so their paddedDims can differ at equal logical dims — every cross-bank query decode
// below zero-pads to the TARGET bank's own paddedDims, never assumes the source's.
//
// KNOWN, ACCEPTED (Poirot S1, casebooks afabc08-graph-h-m1-review.md /
// 15a0668-s1s2s3-fix-verify.md): QueryBatch's outHits is an ordinary call-scoped
// std::vector here, not workspace-tracked — the same S1-class untracked allocation
// already standing against graph.h/novelty.h's identical pattern, kept for
// consistency until S1 is resolved for all three modules together. (The 2026-07-19
// heap-corruption crash that stalled this module was root-caused to a caller-array
// overrun in one test cell, not to this file — casebook
// b4e139e-m3-heap-corruption-root-cause.md.)

#include "superfaiss/matching.h"

#include "superfaiss/query.h"

#include <algorithm>
#include <vector>

namespace superfaiss
{
namespace
{

// Decodes `source` row `row` into a query sized for `targetPaddedDims` — a possibly
// DIFFERENT bank's own paddedDims (matching.h crosses views that may differ in
// quantization). Zeros the whole buffer first, then fills the real [0, dims) elements;
// correct regardless of which paddedDims is wider (the shared DequantizeRowAsQuery in
// graph.h/novelty.h assumes source and target share one bank's paddedDims, which never
// holds across a mutual-match's two distinct banks).
void DequantizeRowForTarget(const BankView& source, int32_t row, int32_t targetPaddedDims, float* out)
{
	for (int32_t d = 0; d < targetPaddedDims; ++d)
	{
		out[d] = 0.0f;
	}
	const int32_t dims = source.dims;
	if (source.quant == Quantization::Int8)
	{
		const int8_t* r = static_cast<const int8_t*>(source.rows) + static_cast<int64_t>(row) * source.paddedDims;
		const float scale = source.scales[row];
		for (int32_t d = 0; d < dims; ++d)
		{
			out[d] = static_cast<float>(r[d]) * scale;
		}
	}
	else
	{
		const float* r = static_cast<const float*>(source.rows) + static_cast<int64_t>(row) * source.paddedDims;
		for (int32_t d = 0; d < dims; ++d)
		{
			out[d] = r[d];
		}
	}
}

int32_t CountNonExcluded(const BankView& bank, const uint32_t* excludeBits)
{
	int32_t n = 0;
	for (int32_t r = 0; r < bank.count; ++r)
	{
		if (!IsExcluded(excludeBits, r))
		{
			++n;
		}
	}
	return n;
}

// Sim(metric, score) = -RankDistance(metric, score) (D-V32-20): L2's raw score is
// lower-is-better and is negated; Cosine/Dot's raw score is already similarity-directioned
// (Sim(Dot, score) = score, identity — CSLS's own Sim is defined for all three metrics,
// unlike RankDistance which excludes Dot).
inline double Sim(Metric metric, float score)
{
	return metric == Metric::L2 ? -static_cast<double>(score) : static_cast<double>(score);
}

} // namespace

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
	Workspace& workspace)
{
	if (sampleViewA.count < 1 || fullViewB.count < 1 || fullViewA.count < 1 || matchK < 1 ||
		sampleSourceIndices == nullptr || outPairs == nullptr || outPairCount == nullptr)
	{
		return Status::InvalidArgument;
	}
	if (sampleViewA.dims != fullViewB.dims || sampleViewA.dims != fullViewA.dims)
	{
		return Status::InvalidArgument;
	}
	if (sampleViewA.metric != fullViewB.metric || sampleViewA.metric != fullViewA.metric)
	{
		return Status::InvalidArgument;
	}
	if (matchK > CountNonExcluded(fullViewB, excludeBitsB) || matchK > CountNonExcluded(fullViewA, excludeBitsA))
	{
		return Status::InvalidArgument;
	}

	const int32_t sampleCount = sampleViewA.count;
	const Metric metric = sampleViewA.metric;

	// Pass 1: batch all sampleCount queries (decoded from sampleViewA, sized for
	// fullViewB's own paddedDims) against fullViewB in one bank scan.
	if (!workspace.ReserveQueryScratch(fullViewB.paddedDims, sampleCount))
	{
		return Status::OutOfMemory;
	}
	float* pass1Base = workspace.QueryScratch(0);
	for (int32_t i = 0; i < sampleCount; ++i)
	{
		DequantizeRowForTarget(sampleViewA, i, fullViewB.paddedDims, pass1Base + static_cast<int64_t>(i) * fullViewB.paddedDims);
	}

	std::vector<Hit> pass1Hits(static_cast<size_t>(sampleCount) * static_cast<size_t>(matchK));
	std::vector<int32_t> pass1Counts(static_cast<size_t>(sampleCount));
	QueryParams paramsB;
	paramsB.k = matchK;
	paramsB.excludeBits = excludeBitsB;
	Status s = QueryBatch(fullViewB, pass1Base, sampleCount, paramsB, workspace, pass1Hits.data(), pass1Counts.data());
	if (s != Status::Ok)
	{
		return s;
	}

	// Collect the DISTINCT candidate B rows pass 1 surfaced (top-1 of each sample row's
	// retrieval) — the dedup the contract's own "no third pass" text implies: multiple
	// sample rows can share the same forward candidate, and it is back-verified once.
	std::vector<int32_t> candidateOfSample(static_cast<size_t>(sampleCount), -1);
	std::vector<int32_t> distinctCandidates;
	for (int32_t i = 0; i < sampleCount; ++i)
	{
		if (pass1Counts[static_cast<size_t>(i)] > 0)
		{
			candidateOfSample[static_cast<size_t>(i)] = pass1Hits[static_cast<size_t>(i) * matchK].index;
		}
	}
	distinctCandidates = candidateOfSample;
	std::sort(distinctCandidates.begin(), distinctCandidates.end());
	distinctCandidates.erase(std::remove(distinctCandidates.begin(), distinctCandidates.end(), -1), distinctCandidates.end());
	distinctCandidates.erase(std::unique(distinctCandidates.begin(), distinctCandidates.end()), distinctCandidates.end());

	// Pass 2: batch the distinct candidates (decoded from fullViewB, sized for fullViewA's
	// own paddedDims) against fullViewA. Re-purposes the SAME tracked query scratch — pass
	// 1's buffer is no longer needed once its QueryBatch call has returned.
	const int32_t distinctCount = static_cast<int32_t>(distinctCandidates.size());
	std::vector<Hit> pass2Hits;
	std::vector<int32_t> pass2Counts;
	if (distinctCount > 0)
	{
		if (!workspace.ReserveQueryScratch(fullViewA.paddedDims, distinctCount))
		{
			return Status::OutOfMemory;
		}
		float* pass2Base = workspace.QueryScratch(0);
		for (int32_t c = 0; c < distinctCount; ++c)
		{
			DequantizeRowForTarget(fullViewB, distinctCandidates[static_cast<size_t>(c)], fullViewA.paddedDims,
				pass2Base + static_cast<int64_t>(c) * fullViewA.paddedDims);
		}
		pass2Hits.resize(static_cast<size_t>(distinctCount) * static_cast<size_t>(matchK));
		pass2Counts.resize(static_cast<size_t>(distinctCount));
		QueryParams paramsA;
		paramsA.k = matchK;
		paramsA.excludeBits = excludeBitsA;
		s = QueryBatch(fullViewA, pass2Base, distinctCount, paramsA, workspace, pass2Hits.data(), pass2Counts.data());
		if (s != Status::Ok)
		{
			return s;
		}
	}

	// Assemble: for each sample row, look up its candidate's back-verification result by
	// the candidate's position in distinctCandidates (a sorted array — binary search).
	for (int32_t i = 0; i < sampleCount; ++i)
	{
		outPairs[i].sourceIndexA = sampleSourceIndices[i];
		outPairs[i].sourceIndexB = -1;
		outPairs[i].cslsMargin = 0.0f;

		const int32_t candidateB = candidateOfSample[static_cast<size_t>(i)];
		if (candidateB < 0)
		{
			continue; // pass 1 found no scorable neighbor at all
		}
		const auto it = std::lower_bound(distinctCandidates.begin(), distinctCandidates.end(), candidateB);
		const int32_t c = static_cast<int32_t>(it - distinctCandidates.begin());
		const int32_t pass2Count = pass2Counts[static_cast<size_t>(c)];
		if (pass2Count == 0)
		{
			continue; // candidate has no scorable neighbor in fullViewA
		}
		const Hit& backTop1 = pass2Hits[static_cast<size_t>(c) * matchK];
		if (backTop1.index != sampleSourceIndices[i])
		{
			continue; // not mutual
		}

		// r_B: mean Sim of pass 1's top-matchK for this sample row.
		double rB = 0.0;
		const int32_t pass1Count = pass1Counts[static_cast<size_t>(i)];
		const Hit* pass1Row = pass1Hits.data() + static_cast<int64_t>(i) * matchK;
		for (int32_t j = 0; j < pass1Count; ++j)
		{
			rB += Sim(metric, pass1Row[j].score);
		}
		rB /= static_cast<double>(pass1Count);

		// r_A: mean Sim of pass 2's top-matchK for this candidate.
		double rA = 0.0;
		const Hit* pass2Row = pass2Hits.data() + static_cast<int64_t>(c) * matchK;
		for (int32_t j = 0; j < pass2Count; ++j)
		{
			rA += Sim(metric, pass2Row[j].score);
		}
		rA /= static_cast<double>(pass2Count);

		const double simAB = Sim(metric, pass1Row[0].score);
		const double margin = 2.0 * simAB - rB - rA;

		outPairs[i].sourceIndexB = candidateB;
		outPairs[i].cslsMargin = static_cast<float>(margin);
	}

	*outPairCount = sampleCount;
	return Status::Ok;
}

} // namespace superfaiss

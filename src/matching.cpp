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
// below calls the shared DequantizeRowAsQuery with an explicit targetPaddedDims, never
// assumes the source's own paddedDims (S4 close, Claude/Poirot/524b373-matching-m3-
// review.md — this module no longer carries a private decode duplicate).
//
// S1 CLOSED (2026-07-19, Claude/Poirot/524b373-matching-m3-review.md O1 — "ripe for the
// all-three close"): both passes' outHits/outCounts route through Workspace's
// ReserveBatchOutput (slot 0 / slot 1, since pass 1's and pass 2's results must stay alive
// simultaneously for the assembly loop below), not a per-call std::vector — the same fix
// applied to graph.h/novelty.h in this same change. (The 2026-07-19 heap-corruption crash
// that separately stalled this module was root-caused to a caller-array overrun in one
// test cell, not to this file — casebook b4e139e-m3-heap-corruption-root-cause.md.)

#include "superfaiss/matching.h"

#include "superfaiss/inspector_common.h" // DequantizeRowAsQuery
#include "superfaiss/query.h"

#include <algorithm>

namespace superfaiss
{
namespace
{

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
		DequantizeRowAsQuery(sampleViewA, i, pass1Base + static_cast<int64_t>(i) * fullViewB.paddedDims, fullViewB.paddedDims);
	}

	// Caller-owned outHits (not workspace.HeapStorage() — QueryBatch's own internal scan
	// scratch, resized mid-call). Slot 0: pass 1's result must stay alive alongside pass 2's
	// (the assembly loop below reads both), so each pass gets its own ReserveBatchOutput slot.
	if (!workspace.ReserveBatchOutput(matchK, sampleCount, 0))
	{
		return Status::OutOfMemory;
	}
	Hit* pass1Hits = workspace.BatchOutputHits(0);
	int32_t* pass1Counts = workspace.BatchOutputCounts(0);
	QueryParams paramsB;
	paramsB.k = matchK;
	paramsB.excludeBits = excludeBitsB;
	Status s = QueryBatch(fullViewB, pass1Base, sampleCount, paramsB, workspace, pass1Hits, pass1Counts);
	if (s != Status::Ok)
	{
		return s;
	}

	// Collect the DISTINCT candidate B rows pass 1 surfaced (top-1 of each sample row's
	// retrieval) — the dedup the contract's own "no third pass" text implies: multiple
	// sample rows can share the same forward candidate, and it is back-verified once.
	// Two workspace-tracked index-scratch slots (S1 close): slot 0 holds candidateOfSample
	// (one entry per sample row, read throughout the assembly loop below); slot 1 holds a
	// working copy that gets sorted/uniqued down to the distinct set in place, tracked by
	// distinctCount rather than a container resize.
	if (!workspace.ReserveIndexScratch(sampleCount, 0) || !workspace.ReserveIndexScratch(sampleCount, 1))
	{
		return Status::OutOfMemory;
	}
	int32_t* candidateOfSample = workspace.IndexScratch(0);
	for (int32_t i = 0; i < sampleCount; ++i)
	{
		candidateOfSample[i] = (pass1Counts[static_cast<size_t>(i)] > 0)
			? pass1Hits[static_cast<size_t>(i) * matchK].index
			: -1;
	}
	int32_t* distinctCandidates = workspace.IndexScratch(1);
	std::copy(candidateOfSample, candidateOfSample + sampleCount, distinctCandidates);
	std::sort(distinctCandidates, distinctCandidates + sampleCount);
	int32_t* distinctEnd = std::remove(distinctCandidates, distinctCandidates + sampleCount, -1);
	distinctEnd = std::unique(distinctCandidates, distinctEnd);

	// Pass 2: batch the distinct candidates (decoded from fullViewB, sized for fullViewA's
	// own paddedDims) against fullViewA. Re-purposes the SAME tracked query scratch — pass
	// 1's buffer is no longer needed once its QueryBatch call has returned.
	const int32_t distinctCount = static_cast<int32_t>(distinctEnd - distinctCandidates);
	// Slot 1: pass 2's own output, distinct from pass 1's slot 0 above — both must stay
	// alive together for the assembly loop below. Left null when distinctCount == 0
	// (candidateOfSample is then all -1, so the assembly loop's candidateB < 0 guard
	// always continues before either pointer would be dereferenced).
	Hit* pass2Hits = nullptr;
	int32_t* pass2Counts = nullptr;
	if (distinctCount > 0)
	{
		if (!workspace.ReserveQueryScratch(fullViewA.paddedDims, distinctCount))
		{
			return Status::OutOfMemory;
		}
		float* pass2Base = workspace.QueryScratch(0);
		for (int32_t c = 0; c < distinctCount; ++c)
		{
			DequantizeRowAsQuery(fullViewB, distinctCandidates[static_cast<size_t>(c)],
				pass2Base + static_cast<int64_t>(c) * fullViewA.paddedDims, fullViewA.paddedDims);
		}
		if (!workspace.ReserveBatchOutput(matchK, distinctCount, 1))
		{
			return Status::OutOfMemory;
		}
		pass2Hits = workspace.BatchOutputHits(1);
		pass2Counts = workspace.BatchOutputCounts(1);
		QueryParams paramsA;
		paramsA.k = matchK;
		paramsA.excludeBits = excludeBitsA;
		s = QueryBatch(fullViewA, pass2Base, distinctCount, paramsA, workspace, pass2Hits, pass2Counts);
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
		const int32_t* it = std::lower_bound(distinctCandidates, distinctCandidates + distinctCount, candidateB);
		const int32_t c = static_cast<int32_t>(it - distinctCandidates);
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
		const Hit* pass1Row = pass1Hits + static_cast<int64_t>(i) * matchK;
		for (int32_t j = 0; j < pass1Count; ++j)
		{
			rB += Sim(metric, pass1Row[j].score);
		}
		rB /= static_cast<double>(pass1Count);

		// r_A: mean Sim of pass 2's top-matchK for this candidate.
		double rA = 0.0;
		const Hit* pass2Row = pass2Hits + static_cast<int64_t>(c) * matchK;
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

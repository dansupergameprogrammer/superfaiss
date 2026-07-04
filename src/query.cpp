#include "superfaiss/query.h"

#include "superfaiss/kernels.h"
#include "superfaiss/topk.h"
#include "superfaiss/validate.h"

namespace superfaiss
{

Status Query(
	const BankView& bank,
	const float* paddedQuery,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCount)
{
	if (outHits == nullptr || outCount == nullptr || params.k < 0)
	{
		return Status::InvalidArgument;
	}
	*outCount = 0;
	if (params.k == 0 || bank.count == 0)
	{
		return Status::Ok;
	}

	// Validation applies the bank's own rules regardless of any scoring override
	// (a Cosine bank rejects zero-norm queries under ScoreAs::Dot too); scoring and
	// hit ordering then run against the effective view.
	const Status queryStatus = ValidateQuery(bank, paddedQuery);
	if (queryStatus != Status::Ok)
	{
		return queryStatus;
	}

	if (params.segments != nullptr)
	{
		const Status segStatus =
			ValidateSegments(bank, paddedQuery, params.segments, params.segmentCount);
		if (segStatus != Status::Ok)
		{
			return segStatus;
		}
	}

	BankView scoring = bank;
	scoring.metric = ScoringMetric(bank, params);

	if (!workspace.Reserve(params.k, 1))
	{
		return Status::OutOfMemory;
	}

	TopK topk;
	topk.Init(workspace.HeapStorage(0), params.k, scoring.metric);

	const int32_t chunks = ChunkCount(scoring);
	for (int32_t c = 0; c < chunks; ++c)
	{
		if (params.segments != nullptr)
		{
			ScoreChunkSegmented(scoring, paddedQuery, c, params.excludeBits,
				params.segments, params.segmentCount, topk);
		}
		else
		{
			ScoreChunk(scoring, paddedQuery, c, params.excludeBits, topk);
		}
	}

	*outCount = topk.Finalize(outHits);
	return Status::Ok;
}

// Width of one amortized sub-batch: TopK views for the sub-batch live on the stack.
// Batches wider than this are processed as consecutive sub-batches; correctness is
// unchanged (batch ≡ singles), the cache amortization simply resets per sub-batch.
static constexpr int32_t kSubBatchWidth = 64;

namespace
{
	void QuerySubBatch(
		const BankView& bank,
		const float* paddedQueries,
		int32_t queryCount,
		const QueryParams& params,
		Workspace& workspace,
		Hit* outHits,
		int32_t* outCounts)
	{
		TopK topks[kSubBatchWidth];
		for (int32_t m = 0; m < queryCount; ++m)
		{
			topks[m].Init(workspace.HeapStorage(m), params.k, bank.metric);
		}
		// Callers pass the effective (already override-resolved) view; see QueryBatch.

		// Chunk loop outermost (cache amortization); queries processed in pairs so the
		// pair kernels share each row's loads and widening.
		const int32_t chunks = ChunkCount(bank);
		for (int32_t c = 0; c < chunks; ++c)
		{
			int32_t m = 0;
			for (; m + 2 <= queryCount; m += 2)
			{
				ScoreChunkPair(
					bank,
					paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
					paddedQueries + static_cast<int64_t>(m + 1) * bank.paddedDims,
					c,
					params.excludeBits,
					topks[m],
					topks[m + 1]);
			}
			if (m < queryCount)
			{
				ScoreChunk(
					bank,
					paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
					c,
					params.excludeBits,
					topks[m]);
			}
		}

		for (int32_t m = 0; m < queryCount; ++m)
		{
			outCounts[m] = topks[m].Finalize(outHits + static_cast<int64_t>(m) * params.k);
		}
	}
}

Status QueryBatch(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCounts)
{
	if (outHits == nullptr || outCounts == nullptr || params.k < 0 || queryCount < 0)
	{
		return Status::InvalidArgument;
	}
	for (int32_t m = 0; m < queryCount; ++m)
	{
		outCounts[m] = 0;
	}
	if (params.k == 0 || bank.count == 0 || queryCount == 0)
	{
		return Status::Ok;
	}

	// Bank-rule validation, then the effective override-resolved view for scoring —
	// same contract as Query.
	for (int32_t m = 0; m < queryCount; ++m)
	{
		const Status queryStatus =
			ValidateQuery(bank, paddedQueries + static_cast<int64_t>(m) * bank.paddedDims);
		if (queryStatus != Status::Ok)
		{
			return queryStatus;
		}
	}

	if (params.segments != nullptr)
	{
		for (int32_t m = 0; m < queryCount; ++m)
		{
			const Status segStatus = ValidateSegments(bank,
				paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
				params.segments, params.segmentCount);
			if (segStatus != Status::Ok)
			{
				return segStatus;
			}
		}
	}

	BankView scoring = bank;
	scoring.metric = ScoringMetric(bank, params);

	// Segmented batches keep the chunk-outermost structure — the bank streams once
	// per batch — with the single-query segmented kernel scoring each query inside
	// the chunk (plan 18.7/W1 composition; pair-blocked variants stay V1-shaped).
	if (params.segments != nullptr)
	{
		const int32_t segWidth = queryCount < kSubBatchWidth ? queryCount : kSubBatchWidth;
		if (!workspace.Reserve(params.k, segWidth))
		{
			return Status::OutOfMemory;
		}
		for (int32_t base = 0; base < queryCount; base += kSubBatchWidth)
		{
			const int32_t width =
				(queryCount - base) < kSubBatchWidth ? (queryCount - base) : kSubBatchWidth;
			TopK topks[kSubBatchWidth];
			for (int32_t m = 0; m < width; ++m)
			{
				topks[m].Init(workspace.HeapStorage(m), params.k, scoring.metric);
			}
			const int32_t chunks = ChunkCount(scoring);
			for (int32_t c = 0; c < chunks; ++c)
			{
				for (int32_t m = 0; m < width; ++m)
				{
					ScoreChunkSegmented(scoring,
						paddedQueries + static_cast<int64_t>(base + m) * bank.paddedDims,
						c, params.excludeBits, params.segments, params.segmentCount,
						topks[m]);
				}
			}
			for (int32_t m = 0; m < width; ++m)
			{
				outCounts[base + m] = topks[m].Finalize(
					outHits + static_cast<int64_t>(base + m) * params.k);
			}
		}
		return Status::Ok;
	}

	const int32_t maxWidth = queryCount < kSubBatchWidth ? queryCount : kSubBatchWidth;
	if (!workspace.Reserve(params.k, maxWidth))
	{
		return Status::OutOfMemory;
	}

	for (int32_t base = 0; base < queryCount; base += kSubBatchWidth)
	{
		const int32_t width =
			(queryCount - base) < kSubBatchWidth ? (queryCount - base) : kSubBatchWidth;
		QuerySubBatch(
			scoring,
			paddedQueries + static_cast<int64_t>(base) * bank.paddedDims,
			width,
			params,
			workspace,
			outHits + static_cast<int64_t>(base) * params.k,
			outCounts + base);
	}
	return Status::Ok;
}

Status QueryIntersect(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCount)
{
	if (outHits == nullptr || outCount == nullptr || params.k < 0 || queryCount <= 0)
	{
		return Status::InvalidArgument;
	}
	*outCount = 0;
	if (params.k == 0 || bank.count == 0)
	{
		return Status::Ok;
	}

	// Bank-rule validation for every member query; the effective override-resolved
	// view for scoring — the same split as Query/QueryBatch.
	for (int32_t m = 0; m < queryCount; ++m)
	{
		const Status queryStatus =
			ValidateQuery(bank, paddedQueries + static_cast<int64_t>(m) * bank.paddedDims);
		if (queryStatus != Status::Ok)
		{
			return queryStatus;
		}
	}

	if (params.segments != nullptr)
	{
		for (int32_t m = 0; m < queryCount; ++m)
		{
			const Status segStatus = ValidateSegments(bank,
				paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
				params.segments, params.segmentCount);
			if (segStatus != Status::Ok)
			{
				return segStatus;
			}
		}
	}

	BankView scoring = bank;
	scoring.metric = ScoringMetric(bank, params);

	if (!workspace.Reserve(params.k, 1))
	{
		return Status::OutOfMemory;
	}

	TopK topk;
	topk.Init(workspace.HeapStorage(0), params.k, scoring.metric);

	const int32_t chunks = ChunkCount(scoring);
	for (int32_t c = 0; c < chunks; ++c)
	{
		if (params.segments != nullptr)
		{
			ScoreChunkFusedSegmented(scoring, paddedQueries, queryCount, c,
				params.excludeBits, params.segments, params.segmentCount, topk);
		}
		else
		{
			ScoreChunkFused(scoring, paddedQueries, queryCount, c, params.excludeBits,
				topk);
		}
	}

	*outCount = topk.Finalize(outHits);
	return Status::Ok;
}

} // namespace superfaiss

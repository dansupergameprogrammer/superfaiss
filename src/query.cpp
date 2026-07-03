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

	const Status queryStatus = ValidateQuery(bank, paddedQuery);
	if (queryStatus != Status::Ok)
	{
		return queryStatus;
	}

	if (!workspace.Reserve(params.k, 1))
	{
		return Status::OutOfMemory;
	}

	TopK topk;
	topk.Init(workspace.HeapStorage(0), params.k, bank.metric);

	const int32_t chunks = ChunkCount(bank);
	for (int32_t c = 0; c < chunks; ++c)
	{
		ScoreChunk(bank, paddedQuery, c, params.excludeBits, topk);
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

		// Chunk loop outermost: each chunk's rows are scored against every query while
		// cache-resident — this is the batch amortization.
		const int32_t chunks = ChunkCount(bank);
		for (int32_t c = 0; c < chunks; ++c)
		{
			for (int32_t m = 0; m < queryCount; ++m)
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

	for (int32_t m = 0; m < queryCount; ++m)
	{
		const Status queryStatus =
			ValidateQuery(bank, paddedQueries + static_cast<int64_t>(m) * bank.paddedDims);
		if (queryStatus != Status::Ok)
		{
			return queryStatus;
		}
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
			bank,
			paddedQueries + static_cast<int64_t>(base) * bank.paddedDims,
			width,
			params,
			workspace,
			outHits + static_cast<int64_t>(base) * params.k,
			outCounts + base);
	}
	return Status::Ok;
}

} // namespace superfaiss

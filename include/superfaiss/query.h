#pragma once

#include "types.h"
#include "alloc.h"

namespace superfaiss
{

// Single-threaded exact top-k over the whole bank. Validates the query (not the bank —
// validate the bank once at load with ValidateBank), scans chunks in order, and writes
// up to params.k hits best-first into outHits. Returns the hit count via outCount.
// Allocation-free once `workspace` is warm for this k.
Status Query(
	const BankView& bank,
	const float* paddedQuery,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCount);

// M queries in one bank pass: the chunk loop is outermost, so each chunk's rows are
// scored against every query while cache-resident — the memory traffic of one scan
// amortized across the batch. Queries are contiguous, stride bank.paddedDims.
// outHits holds queryCount * params.k entries (query-major); outCounts holds queryCount.
Status QueryBatch(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCounts);

} // namespace superfaiss

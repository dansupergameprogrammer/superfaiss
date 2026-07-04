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

// Intersection (set-op combinator, plan 18.7): exact top-k over the FUSED score —
// each row's worst per-query score in the metric's better-direction — in one bank
// pass. True AND semantics: every returned row scores at least the fused score
// against every query. All queries score under one metric (the bank's, or the
// params.scoreAs override); queryCount == 1 degenerates to Query() bit-identically.
// Queries are contiguous, stride bank.paddedDims; outHits holds up to params.k hits.
Status QueryIntersect(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCount);

} // namespace superfaiss

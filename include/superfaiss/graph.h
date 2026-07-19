#pragma once

#include "types.h"
#include "alloc.h" // Workspace

// SuperFAISS V3.2 — Bank Inspector I, Tier 1 module M1 (plan section 25.4): the
// mutual k-NN graph + connected components that back the Inspector's Structure view.
// Post-processing over exact query output; touches no kernel, quantization, or format.
//
// Determinism tier: PER-DEVICE. Fixed sample, fixed order, pinned ties (query tie-break
// is ascending index; union-find edge order is ascending (i, j); component ids
// canonicalize to the smallest member row index). Bit-identical on repeat calls on one
// device, given bit-identical query results. NO CrossDevice claim (the float score paths
// inherit the library's per-device tier exactly as pca.h discloses of itself).
//
// All functions validate their arguments and write no output on rejection (the dim-2
// trust-boundary contract): k < 1, k >= view count, count 0/1 where a k-th neighbor
// cannot exist, or a null buffer returns InvalidArgument with no partial write.

namespace superfaiss
{

// For each row of the (sampled) view, its top-k neighbor row indices by the bank's own
// metric, via the existing batch query path; ties break ascending index (V32-G5). When
// excludeSelf is true the row itself is dropped from its own list. outNeighbors is
// count*k int32, row-major (outNeighbors[r*k + j] is row r's j-th nearest, best-first);
// a slot with no available neighbor is -1. `workspace` warm for k (self-exclusion widens
// the top-k heap by one internally). k in [1, count-1]; k < 1, k >= count, or a null
// buffer -> InvalidArgument, no output write.
Status BuildKnnNeighbors(
	const BankView& bank,
	int32_t k,
	bool excludeSelf,
	int32_t* outNeighbors,
	Workspace& workspace);

// Mutual-edge filter (pure integer scan, order-fixed): edge (i, j) survives iff
// j in topk(i) AND i in topk(j). `neighbors` is the count*k array from BuildKnnNeighbors;
// outMutualFlags is count*k uint8 aligned to it (outMutualFlags[r*k + j] == 1 iff that
// slot's neighbor is non-negative and the edge is mutual, else 0). count < 1, k < 1, or a
// null buffer -> InvalidArgument.
Status MutualFilter(
	int32_t count,
	int32_t k,
	const int32_t* neighbors,
	uint8_t* outMutualFlags);

// Exact-duplicate grouping by CONSTRUCTION (plan 25.4, D-V32-25/28): a hash pass in
// ascending row order over the view; equality confirmed by FULL BYTE comparison before
// any grouping (hash agreement alone never groups). The identity key is the stored row
// bytes plus the per-row scale where the quantization carries one (int8); float32 is the
// bytes alone. outGroupOf is count int32: outGroupOf[r] is the SMALLEST row index whose
// stored bytes (and scale) are identical to r's, or r itself when r is unique. `scratch`
// is caller-provided, count int32 (the hash-pass workspace). A null buffer or count < 1
// -> InvalidArgument. Same-decode-but-different-bytes rows are near-duplicates, NOT
// duplicates (honest, disclosed): they do not share a group.
Status BuildDuplicateGroups(
	const BankView& bank,
	int32_t* outGroupOf,
	int32_t* scratch);

// Connected components over union-find seeded FIRST with the duplicate groups'
// construction edges (each row r unioned with outGroupOf[r]), THEN the surviving mutual
// edges processed in ascending (i, j) order. Component ids are pinned to the smallest
// member row index (a canonical relabel pass after union), so ids are deterministic and
// order-independent — never allocation-order artifacts. `duplicateGroups` is the
// BuildDuplicateGroups output (null selects no duplicate seeding). outComponentId is
// count int32; unionFindScratch is caller-provided, count int32 (the parent array).
// count < 1, k < 1, or a null neighbors/flags/out buffer -> InvalidArgument.
Status ConnectedComponents(
	int32_t count,
	int32_t k,
	const int32_t* neighbors,
	const uint8_t* mutualFlags,
	const int32_t* duplicateGroups,
	int32_t* outComponentId,
	int32_t* unionFindScratch);

} // namespace superfaiss

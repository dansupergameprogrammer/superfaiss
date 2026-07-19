// SuperFAISS V3.2 — Bank Inspector I, module M1 (graph.h): the mutual k-NN graph +
// connected components backing the Inspector's Structure view (plan section 25.4).
//
// Post-processing over exact query output: BuildKnnNeighbors rides the existing Query
// path (metric-agnostic, best-first, ties ascending index — V32-G5); everything else is
// pure integer/byte logic. Determinism tier PER-DEVICE (fixed order, pinned ties,
// component ids canonicalized to the smallest member index). No kernel, quantization, or
// format is touched.

#include "superfaiss/graph.h"

#include "superfaiss/inspector_common.h" // DequantizeRowAsQuery
#include "superfaiss/query.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace superfaiss
{
namespace
{

// FNV-1a over a row's stored bytes plus (for int8) its scale — the grouping hash. Only a
// pre-filter: full byte equality is always confirmed before two rows share a group.
uint32_t RowHash(const BankView& bank, int32_t r, size_t rowBytes)
{
	const unsigned char* p = static_cast<const unsigned char*>(bank.rows) +
		static_cast<int64_t>(r) * static_cast<int64_t>(rowBytes);
	uint32_t h = 2166136261u;
	for (size_t i = 0; i < rowBytes; ++i)
	{
		h = (h ^ p[i]) * 16777619u;
	}
	if (bank.quant == Quantization::Int8)
	{
		unsigned char sb[sizeof(float)];
		std::memcpy(sb, &bank.scales[r], sizeof(float));
		for (size_t i = 0; i < sizeof(float); ++i)
		{
			h = (h ^ sb[i]) * 16777619u;
		}
	}
	return h;
}

// Exact stored-identity of two rows: full byte equality of the stored image, plus scale
// equality for int8. Same-decode-but-different-bytes rows are near-duplicates, never a
// match (the honest, disclosed line).
bool RowsIdentical(const BankView& bank, int32_t a, int32_t b, size_t rowBytes)
{
	const char* rows = static_cast<const char*>(bank.rows);
	if (std::memcmp(rows + static_cast<int64_t>(a) * static_cast<int64_t>(rowBytes),
			rows + static_cast<int64_t>(b) * static_cast<int64_t>(rowBytes), rowBytes) != 0)
	{
		return false;
	}
	if (bank.quant == Quantization::Int8 && bank.scales[a] != bank.scales[b])
	{
		return false;
	}
	return true;
}

} // namespace

Status BuildKnnNeighbors(
	const BankView& bank, int32_t k, bool excludeSelf, int32_t* outNeighbors, Workspace& workspace)
{
	const int32_t count = bank.count;
	if (k < 1 || k >= count || outNeighbors == nullptr)
	{
		return Status::InvalidArgument;
	}

	// Widen the retrieval by one when self-excluding: the row is its own nearest (distance
	// 0 / self-dot), so k+1 hits guarantee k survivors after the self drop. k < count, so
	// internalK <= count is always a valid query k.
	const int32_t internalK = excludeSelf ? k + 1 : k;

	// The batch query path (V32-G5/§25.4): the chunk loop runs OUTERMOST across all `count`
	// queries in one bank pass, amortizing the memory traffic a per-row Query() loop would
	// re-pay `count` times. Tracked, warm-reusable scratch for the packed query buffer
	// (Workspace's own "zero allocation on warm reuse" contract); QueryBatch manages its own
	// internal top-k scratch via `workspace` and is never given segments here, so it never
	// touches this region itself.
	if (!workspace.ReserveQueryScratch(bank.paddedDims, count))
	{
		return Status::OutOfMemory;
	}
	// Index from the base at OUR OWN chosen stride (bank.paddedDims), never via
	// workspace.QueryScratch(r) for r > 0: a warm workspace's internal stride can be WIDER
	// than bank.paddedDims (grown by an earlier, larger-dims caller), so QueryScratch(r)
	// would not land at r*bank.paddedDims — the exact hazard query.cpp's own segmented-fold
	// packing works around (external bug report 2026-07-04, T25). ReserveQueryScratch's
	// total allocation is always >= count*bank.paddedDims regardless of the internal
	// stride, so packing from the base at our own stride is always safe.
	float* queryBase = workspace.QueryScratch(0);
	for (int32_t r = 0; r < count; ++r)
	{
		DequantizeRowAsQuery(bank, r, queryBase + static_cast<int64_t>(r) * bank.paddedDims);
	}

	// QueryBatch's own outHits is a caller-owned buffer, matching every other QueryBatch
	// call site in this codebase (analytics.cpp, the existing test suite): `workspace`'s
	// HeapStorage slots are QueryBatch's OWN internal per-sub-batch scratch, reused and
	// overwritten across sub-batches — not a place a caller can safely park its persistent
	// output.
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
		int32_t* outRow = outNeighbors + static_cast<int64_t>(r) * k;
		int32_t written = 0;
		for (int32_t j = 0; j < hitCount && written < k; ++j)
		{
			if (excludeSelf && rowHits[j].index == r)
			{
				continue;
			}
			outRow[written++] = rowHits[j].index;
		}
		for (; written < k; ++written)
		{
			outRow[written] = -1; // no available neighbor for this slot
		}
	}
	return Status::Ok;
}

Status MutualFilter(int32_t count, int32_t k, const int32_t* neighbors, uint8_t* outMutualFlags)
{
	if (count < 1 || k < 1 || neighbors == nullptr || outMutualFlags == nullptr)
	{
		return Status::InvalidArgument;
	}

	const auto inList = [&](int32_t row, int32_t x) -> bool {
		const int32_t* nl = neighbors + static_cast<int64_t>(row) * k;
		for (int32_t t = 0; t < k; ++t)
		{
			if (nl[t] == x)
			{
				return true;
			}
		}
		return false;
	};

	for (int32_t i = 0; i < count; ++i)
	{
		for (int32_t t = 0; t < k; ++t)
		{
			const int64_t slot = static_cast<int64_t>(i) * k + t;
			const int32_t j = neighbors[slot];
			outMutualFlags[slot] = (j >= 0 && inList(j, i)) ? uint8_t{1} : uint8_t{0};
		}
	}
	return Status::Ok;
}

Status BuildDuplicateGroups(const BankView& bank, int32_t* outGroupOf, int32_t* scratch)
{
	const int32_t count = bank.count;
	if (count < 1 || outGroupOf == nullptr || scratch == nullptr)
	{
		return Status::InvalidArgument;
	}

	const size_t rowBytes = static_cast<size_t>(bank.paddedDims) * ElementSize(bank.quant);

	// Stage the per-row hash in outGroupOf while it is free (overwritten with the final
	// group in the walk below), and an index permutation in scratch. Sorting the
	// permutation by (hash, stored bytes, index) clusters byte-identical rows contiguously
	// with the smallest index first in each run — so the run leader is the group
	// representative, and no load-factored hash table (hence no allocation) is needed.
	for (int32_t r = 0; r < count; ++r)
	{
		outGroupOf[r] = static_cast<int32_t>(RowHash(bank, r, rowBytes));
		scratch[r] = r;
	}

	const char* rows = static_cast<const char*>(bank.rows);
	std::sort(scratch, scratch + count, [&](int32_t a, int32_t b) {
		const uint32_t ha = static_cast<uint32_t>(outGroupOf[a]);
		const uint32_t hb = static_cast<uint32_t>(outGroupOf[b]);
		if (ha != hb)
		{
			return ha < hb;
		}
		const int cmp = std::memcmp(rows + static_cast<int64_t>(a) * static_cast<int64_t>(rowBytes),
			rows + static_cast<int64_t>(b) * static_cast<int64_t>(rowBytes), rowBytes);
		if (cmp != 0)
		{
			return cmp < 0;
		}
		if (bank.quant == Quantization::Int8 && bank.scales[a] != bank.scales[b])
		{
			return bank.scales[a] < bank.scales[b];
		}
		return a < b; // identical rows: smallest index leads the run (the representative)
	});

	int32_t rep = scratch[0];
	outGroupOf[scratch[0]] = rep;
	for (int32_t p = 1; p < count; ++p)
	{
		const int32_t cur = scratch[p];
		const int32_t prev = scratch[p - 1];
		if (!RowsIdentical(bank, cur, prev, rowBytes))
		{
			rep = cur; // new run: the sort put the smallest index first
		}
		outGroupOf[cur] = rep;
	}
	return Status::Ok;
}

Status ConnectedComponents(
	int32_t count,
	int32_t k,
	const int32_t* neighbors,
	const uint8_t* mutualFlags,
	const int32_t* duplicateGroups,
	int32_t* outComponentId,
	int32_t* unionFindScratch)
{
	if (count < 1 || k < 1 || neighbors == nullptr || mutualFlags == nullptr ||
		outComponentId == nullptr || unionFindScratch == nullptr)
	{
		return Status::InvalidArgument;
	}

	int32_t* parent = unionFindScratch;
	for (int32_t r = 0; r < count; ++r)
	{
		parent[r] = r;
	}

	const auto find = [&](int32_t x) -> int32_t {
		while (parent[x] != x)
		{
			parent[x] = parent[parent[x]]; // path halving
			x = parent[x];
		}
		return x;
	};
	// Union keeping the SMALLER index as root, so a component's root is its smallest member
	// — the canonical id, order-independent (dim 6).
	const auto unite = [&](int32_t a, int32_t b) {
		const int32_t ra = find(a);
		const int32_t rb = find(b);
		if (ra == rb)
		{
			return;
		}
		if (ra < rb)
		{
			parent[rb] = ra;
		}
		else
		{
			parent[ra] = rb;
		}
	};

	// Construction edges FIRST: identical content is one component by definition, seeded
	// before any discovered geometry (D-V32-28).
	if (duplicateGroups != nullptr)
	{
		for (int32_t r = 0; r < count; ++r)
		{
			if (duplicateGroups[r] != r)
			{
				unite(r, duplicateGroups[r]);
			}
		}
	}

	// Then surviving mutual edges, ascending (i, j). Both symmetric slots carry the flag;
	// union is idempotent, so processing both directions is harmless.
	for (int32_t i = 0; i < count; ++i)
	{
		for (int32_t t = 0; t < k; ++t)
		{
			const int64_t slot = static_cast<int64_t>(i) * k + t;
			if (mutualFlags[slot])
			{
				const int32_t j = neighbors[slot];
				if (j >= 0)
				{
					unite(i, j);
				}
			}
		}
	}

	for (int32_t r = 0; r < count; ++r)
	{
		outComponentId[r] = find(r); // smallest member of r's component
	}
	return Status::Ok;
}

} // namespace superfaiss

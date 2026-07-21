#pragma once

#include <cstddef>
#include <cstdint>

#include "types.h"

namespace superfaiss
{

struct XdQuery; // kernels.h

// Pluggable allocator seam. Hosts (engines) may route this to their own allocators.
// Every allocation made through the seam is counted, so a caller can assert
// zero steady-state allocation across warm queries.
struct Allocator
{
	void* (*alloc)(size_t size, size_t alignment, void* user) = nullptr;
	void (*free)(void* ptr, void* user) = nullptr;
	void* user = nullptr;
};

// Default allocator (aligned malloc/free). Used when no allocator is supplied.
Allocator DefaultAllocator();

// Total allocations made through any SuperFAISS allocator seam since process start.
// Monotonic; intended for flat-delta assertions in tests, not for accounting.
uint64_t AllocationCount();

namespace detail
{
	void* SeamAlloc(const Allocator& a, size_t size, size_t alignment);
	void SeamFree(const Allocator& a, void* ptr);
}

// Reusable query scratch. Reserve() sizes it for a given k and batch width; queries
// then run allocation-free until a larger reservation is needed.
//
// Ownership contract: a Workspace has a single owner and is NOT thread-safe. It serves
// one Query/QueryBatch call at a time; concurrent queries need one Workspace each
// (an async host should pool them, one per in-flight task).
class Workspace
{
public:
	Workspace() = default;
	explicit Workspace(const Allocator& allocator) : Allocator_(allocator) {}
	Workspace(const Workspace&) = delete;
	Workspace& operator=(const Workspace&) = delete;
	~Workspace();

	// Ensure capacity for `batchWidth` concurrent top-k heaps of size `k`.
	// Growth is counted by the allocator seam; a warm workspace never grows.
	bool Reserve(int32_t k, int32_t batchWidth);

	Hit* HeapStorage(int32_t queryIndex);
	int32_t ReservedK() const { return ReservedK_; }
	int32_t ReservedBatch() const { return ReservedBatch_; }

	// Aligned float scratch for folded segmented queries (dot/cosine weight folding):
	// `count` queries of `paddedDims` elements. Same growth accounting as Reserve().
	bool ReserveQueryScratch(int32_t paddedDims, int32_t count);
	float* QueryScratch(int32_t queryIndex);

	// Zeroed uint32 bit-scratch over `count` rows (v2.1 sparse bias: pair-uniqueness
	// validation + pair-row marking). Zeroing is O(count/8) bytes per call - noise
	// against the scan itself. Same growth accounting as Reserve().
	bool ReserveBiasBits(int32_t count);
	uint32_t* BiasBits();

	// Aligned int8 scratch for CrossDevice-quantized queries (v2.2): `count`
	// queries of `paddedDims` bytes plus per-query scale/self-dot slots. Same
	// growth accounting as Reserve().
	bool ReserveXdQuery(int32_t paddedDims, int32_t count);
	int8_t* XdQ8(int32_t queryIndex);
	double* XdScale(int32_t queryIndex);
	int64_t* XdSqSum(int32_t queryIndex);
	// `count` XdQuery slots in the same block (callers fill them from the three
	// accessors above; QueryIntersect passes the array to the fused kernel).
	XdQuery* XdSlots();

	// Number of times Reserve() actually grew the buffer. Flat across warm queries.
	uint64_t GrowthCount() const { return GrowthCount_; }

	// Caller-owned batch QUERY-OUTPUT scratch. DISTINCT
	// from HeapStorage() above: Query/QueryBatch use HeapStorage() as THEIR OWN internal
	// per-sub-batch scan scratch via their own Reserve() calls, resized and overwritten
	// mid-call — a caller cannot safely park its own persistent QueryBatch outHits/outCounts
	// there. A caller that wants that result to live in workspace-tracked memory instead of a
	// per-call std::vector reserves its own block here, sized to its own queryCount*k /
	// queryCount. Two independent SLOTS (0/1, default 0): graph.h/novelty.h use one call's
	// worth of output at a time (slot 0), but matching.h's two-pass mutual match needs
	// pass 1's and pass 2's results alive SIMULTANEOUSLY (the assembly loop reads both) —
	// slot 0 for pass 1, slot 1 for pass 2, so the second reservation cannot alias/clobber
	// the first. Same growth accounting as Reserve().
	bool ReserveBatchOutput(int32_t k, int32_t queryCount, int32_t slot = 0);
	Hit* BatchOutputHits(int32_t slot = 0);
	int32_t* BatchOutputCounts(int32_t slot = 0);

	// General int32 index scratch (V3.2 S1 close): matching.h's candidate-dedup working
	// arrays (candidateOfSample and its sorted/uniqued copy distinctCandidates) need two
	// independent, simultaneously-alive int32 buffers of caller-chosen size, the same shape
	// as BatchOutput's two slots. Same growth accounting as Reserve().
	bool ReserveIndexScratch(int32_t count, int32_t slot = 0);
	int32_t* IndexScratch(int32_t slot = 0);

private:
	Allocator Allocator_ = DefaultAllocator();
	Hit* Storage_ = nullptr;
	int32_t ReservedK_ = 0;
	int32_t ReservedBatch_ = 0;
	float* QueryScratch_ = nullptr;
	int32_t ScratchDims_ = 0;
	int32_t ScratchCount_ = 0;
	uint32_t* BiasBits_ = nullptr;
	int32_t BiasBitWords_ = 0;
	int8_t* XdQ8_ = nullptr;      // count x paddedDims int8, then count doubles, then count int64s
	int32_t XdDims_ = 0;
	int32_t XdCount_ = 0;
	Hit* OutputHits_[2] = {nullptr, nullptr};
	int32_t* OutputCounts_[2] = {nullptr, nullptr};
	int32_t OutputK_[2] = {0, 0};
	int32_t OutputCount_[2] = {0, 0};
	int32_t* IndexScratch_[2] = {nullptr, nullptr};
	int32_t IndexScratchCount_[2] = {0, 0};
	uint64_t GrowthCount_ = 0;
};

} // namespace superfaiss

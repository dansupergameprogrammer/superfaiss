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
	uint64_t GrowthCount_ = 0;
};

} // namespace superfaiss

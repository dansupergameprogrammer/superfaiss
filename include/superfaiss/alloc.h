#pragma once

#include <cstddef>
#include <cstdint>

#include "types.h"

namespace superfaiss
{

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

	// Number of times Reserve() actually grew the buffer. Flat across warm queries.
	uint64_t GrowthCount() const { return GrowthCount_; }

private:
	Allocator Allocator_ = DefaultAllocator();
	Hit* Storage_ = nullptr;
	int32_t ReservedK_ = 0;
	int32_t ReservedBatch_ = 0;
	uint64_t GrowthCount_ = 0;
};

} // namespace superfaiss

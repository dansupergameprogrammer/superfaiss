#include "superfaiss/alloc.h"

#include <atomic>
#include <cstdlib>

namespace superfaiss
{

namespace
{
	std::atomic<uint64_t> GAllocationCount{0};

	void* DefaultAlloc(size_t size, size_t alignment, void* /*user*/)
	{
#if defined(_MSC_VER)
		return _aligned_malloc(size, alignment);
#else
		// aligned_alloc requires size to be a multiple of alignment.
		const size_t rounded = ((size + alignment - 1) / alignment) * alignment;
		return std::aligned_alloc(alignment, rounded);
#endif
	}

	void DefaultFree(void* ptr, void* /*user*/)
	{
#if defined(_MSC_VER)
		_aligned_free(ptr);
#else
		std::free(ptr);
#endif
	}
}

Allocator DefaultAllocator()
{
	Allocator a;
	a.alloc = &DefaultAlloc;
	a.free = &DefaultFree;
	a.user = nullptr;
	return a;
}

uint64_t AllocationCount()
{
	return GAllocationCount.load(std::memory_order_relaxed);
}

namespace detail
{
	void* SeamAlloc(const Allocator& a, size_t size, size_t alignment)
	{
		GAllocationCount.fetch_add(1, std::memory_order_relaxed);
		return a.alloc(size, alignment, a.user);
	}

	void SeamFree(const Allocator& a, void* ptr)
	{
		if (ptr != nullptr)
		{
			a.free(ptr, a.user);
		}
	}
}

Workspace::~Workspace()
{
	detail::SeamFree(Allocator_, Storage_);
}

bool Workspace::Reserve(int32_t k, int32_t batchWidth)
{
	if (k < 0 || batchWidth < 1)
	{
		return false;
	}
	if (k <= ReservedK_ && batchWidth <= ReservedBatch_)
	{
		return true;
	}
	const int32_t newK = k > ReservedK_ ? k : ReservedK_;
	const int32_t newBatch = batchWidth > ReservedBatch_ ? batchWidth : ReservedBatch_;
	const size_t bytes = static_cast<size_t>(newK) * newBatch * sizeof(Hit);
	Hit* grown = static_cast<Hit*>(detail::SeamAlloc(Allocator_, bytes ? bytes : sizeof(Hit), alignof(Hit) > 16 ? alignof(Hit) : 16));
	if (grown == nullptr)
	{
		return false;
	}
	detail::SeamFree(Allocator_, Storage_);
	Storage_ = grown;
	ReservedK_ = newK;
	ReservedBatch_ = newBatch;
	++GrowthCount_;
	return true;
}

Hit* Workspace::HeapStorage(int32_t queryIndex)
{
	return Storage_ + static_cast<size_t>(queryIndex) * ReservedK_;
}

} // namespace superfaiss

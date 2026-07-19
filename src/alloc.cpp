#include "superfaiss/alloc.h"

#include "superfaiss/kernels.h"

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
	detail::SeamFree(Allocator_, QueryScratch_);
	detail::SeamFree(Allocator_, BiasBits_);
	detail::SeamFree(Allocator_, XdQ8_);
	detail::SeamFree(Allocator_, OutputHits_[0]);
	detail::SeamFree(Allocator_, OutputCounts_[0]);
	detail::SeamFree(Allocator_, OutputHits_[1]);
	detail::SeamFree(Allocator_, OutputCounts_[1]);
	detail::SeamFree(Allocator_, IndexScratch_[0]);
	detail::SeamFree(Allocator_, IndexScratch_[1]);
}

bool Workspace::ReserveXdQuery(int32_t paddedDims, int32_t count)
{
	if (paddedDims <= 0 || count < 1)
	{
		return false;
	}
	if (paddedDims <= XdDims_ && count <= XdCount_)
	{
		return true;
	}
	const int32_t newDims = paddedDims > XdDims_ ? paddedDims : XdDims_;
	const int32_t newCount = count > XdCount_ ? count : XdCount_;
	// One block: count x paddedDims int8 (16-aligned rows since paddedDims is a
	// multiple of 16 for int8 banks), then count doubles, then count int64s -
	// the scalar slots padded up to 16-byte alignment.
	const size_t q8Bytes =
		((static_cast<size_t>(newDims) * newCount + 15) / 16) * 16;
	const size_t bytes = q8Bytes +
		static_cast<size_t>(newCount) * (sizeof(double) + sizeof(int64_t) + sizeof(XdQuery));
	int8_t* grown = static_cast<int8_t*>(detail::SeamAlloc(Allocator_, bytes, 16));
	if (grown == nullptr)
	{
		return false;
	}
	detail::SeamFree(Allocator_, XdQ8_);
	XdQ8_ = grown;
	XdDims_ = newDims;
	XdCount_ = newCount;
	++GrowthCount_;
	return true;
}

int8_t* Workspace::XdQ8(int32_t queryIndex)
{
	return XdQ8_ + static_cast<int64_t>(queryIndex) * XdDims_;
}

double* Workspace::XdScale(int32_t queryIndex)
{
	const size_t q8Bytes =
		((static_cast<size_t>(XdDims_) * XdCount_ + 15) / 16) * 16;
	return reinterpret_cast<double*>(XdQ8_ + q8Bytes) + queryIndex;
}

int64_t* Workspace::XdSqSum(int32_t queryIndex)
{
	const size_t q8Bytes =
		((static_cast<size_t>(XdDims_) * XdCount_ + 15) / 16) * 16;
	return reinterpret_cast<int64_t*>(
		XdQ8_ + q8Bytes + static_cast<size_t>(XdCount_) * sizeof(double)) + queryIndex;
}

XdQuery* Workspace::XdSlots()
{
	const size_t q8Bytes =
		((static_cast<size_t>(XdDims_) * XdCount_ + 15) / 16) * 16;
	return reinterpret_cast<XdQuery*>(XdQ8_ + q8Bytes +
		static_cast<size_t>(XdCount_) * (sizeof(double) + sizeof(int64_t)));
}

bool Workspace::ReserveQueryScratch(int32_t paddedDims, int32_t count)
{
	if (paddedDims <= 0 || count < 1)
	{
		return false;
	}
	if (paddedDims <= ScratchDims_ && count <= ScratchCount_)
	{
		return true;
	}
	const int32_t newDims = paddedDims > ScratchDims_ ? paddedDims : ScratchDims_;
	const int32_t newCount = count > ScratchCount_ ? count : ScratchCount_;
	const size_t bytes = static_cast<size_t>(newDims) * newCount * sizeof(float);
	float* grown = static_cast<float*>(detail::SeamAlloc(Allocator_, bytes, 16));
	if (grown == nullptr)
	{
		return false;
	}
	detail::SeamFree(Allocator_, QueryScratch_);
	QueryScratch_ = grown;
	ScratchDims_ = newDims;
	ScratchCount_ = newCount;
	++GrowthCount_;
	return true;
}

bool Workspace::ReserveBiasBits(int32_t count)
{
	if (count < 0)
	{
		return false;
	}
	const int32_t words = (count + 31) / 32;
	if (words > BiasBitWords_)
	{
		uint32_t* grown = static_cast<uint32_t*>(
			detail::SeamAlloc(Allocator_, static_cast<size_t>(words) * sizeof(uint32_t), 16));
		if (grown == nullptr)
		{
			return false;
		}
		detail::SeamFree(Allocator_, BiasBits_);
		BiasBits_ = grown;
		BiasBitWords_ = words;
		++GrowthCount_;
	}
	for (int32_t w = 0; w < words; ++w)
	{
		BiasBits_[w] = 0u;
	}
	return true;
}

uint32_t* Workspace::BiasBits()
{
	return BiasBits_;
}

float* Workspace::QueryScratch(int32_t queryIndex)
{
	return QueryScratch_ + static_cast<int64_t>(queryIndex) * ScratchDims_;
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

bool Workspace::ReserveBatchOutput(int32_t k, int32_t queryCount, int32_t slot)
{
	if (k < 1 || queryCount < 1 || slot < 0 || slot > 1)
	{
		return false;
	}
	if (k <= OutputK_[slot] && queryCount <= OutputCount_[slot])
	{
		return true;
	}
	const int32_t newK = k > OutputK_[slot] ? k : OutputK_[slot];
	const int32_t newCount = queryCount > OutputCount_[slot] ? queryCount : OutputCount_[slot];

	const size_t hitBytes = static_cast<size_t>(newK) * newCount * sizeof(Hit);
	Hit* grownHits = static_cast<Hit*>(detail::SeamAlloc(
		Allocator_, hitBytes ? hitBytes : sizeof(Hit), alignof(Hit) > 16 ? alignof(Hit) : 16));
	if (grownHits == nullptr)
	{
		return false;
	}
	int32_t* grownCounts = static_cast<int32_t*>(
		detail::SeamAlloc(Allocator_, static_cast<size_t>(newCount) * sizeof(int32_t), 16));
	if (grownCounts == nullptr)
	{
		detail::SeamFree(Allocator_, grownHits);
		return false;
	}

	detail::SeamFree(Allocator_, OutputHits_[slot]);
	detail::SeamFree(Allocator_, OutputCounts_[slot]);
	OutputHits_[slot] = grownHits;
	OutputCounts_[slot] = grownCounts;
	OutputK_[slot] = newK;
	OutputCount_[slot] = newCount;
	++GrowthCount_;
	return true;
}

Hit* Workspace::BatchOutputHits(int32_t slot)
{
	return OutputHits_[slot];
}

int32_t* Workspace::BatchOutputCounts(int32_t slot)
{
	return OutputCounts_[slot];
}

bool Workspace::ReserveIndexScratch(int32_t count, int32_t slot)
{
	if (count < 1 || slot < 0 || slot > 1)
	{
		return false;
	}
	if (count <= IndexScratchCount_[slot])
	{
		return true;
	}
	int32_t* grown = static_cast<int32_t*>(
		detail::SeamAlloc(Allocator_, static_cast<size_t>(count) * sizeof(int32_t), 16));
	if (grown == nullptr)
	{
		return false;
	}
	detail::SeamFree(Allocator_, IndexScratch_[slot]);
	IndexScratch_[slot] = grown;
	IndexScratchCount_[slot] = count;
	++GrowthCount_;
	return true;
}

int32_t* Workspace::IndexScratch(int32_t slot)
{
	return IndexScratch_[slot];
}

} // namespace superfaiss

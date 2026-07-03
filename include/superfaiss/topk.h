#pragma once

#include "types.h"

namespace superfaiss
{

// Strict total order over hits: primary score (direction depends on metric),
// ties broken by ascending index. Because the order is total, the top-k *set* and its
// sorted output are unique regardless of insertion order — determinism does not depend
// on chunk scheduling.
inline bool Better(const Hit& a, const Hit& b, Metric metric)
{
	if (a.score != b.score)
	{
		return metric == Metric::L2 ? (a.score < b.score) : (a.score > b.score);
	}
	return a.index < b.index;
}

// Fixed-capacity top-k accumulator over caller-provided storage. Binary min-heap keyed
// "worst of the kept k at the root" so the eviction test is O(1).
class TopK
{
public:
	void Init(Hit* storage, int32_t k, Metric metric)
	{
		Heap_ = storage;
		K_ = k;
		Metric_ = metric;
		Size_ = 0;
	}

	int32_t Size() const { return Size_; }

	void Push(int32_t index, float score)
	{
		const Hit candidate{index, score};
		if (Size_ < K_)
		{
			Heap_[Size_++] = candidate;
			SiftUp(Size_ - 1);
		}
		else if (K_ > 0 && Better(candidate, Heap_[0], Metric_))
		{
			Heap_[0] = candidate;
			SiftDown(0);
		}
	}

	// Sorts the kept hits best-first into `out` and returns the count.
	// Destroys the heap contents.
	int32_t Finalize(Hit* out)
	{
		const int32_t n = Size_;
		for (int32_t i = n - 1; i >= 0; --i)
		{
			out[i] = Heap_[0];
			Heap_[0] = Heap_[Size_ - 1];
			--Size_;
			SiftDown(0);
		}
		return n;
	}

private:
	// Heap order: parent is worse-or-equal than children ("worst at root").
	bool HeapAbove(const Hit& a, const Hit& b) const { return Better(b, a, Metric_); }

	void SiftUp(int32_t i)
	{
		while (i > 0)
		{
			const int32_t parent = (i - 1) / 2;
			if (!HeapAbove(Heap_[i], Heap_[parent]))
			{
				break;
			}
			const Hit tmp = Heap_[i];
			Heap_[i] = Heap_[parent];
			Heap_[parent] = tmp;
			i = parent;
		}
	}

	void SiftDown(int32_t i)
	{
		for (;;)
		{
			const int32_t left = 2 * i + 1;
			const int32_t right = 2 * i + 2;
			int32_t worst = i;
			if (left < Size_ && HeapAbove(Heap_[left], Heap_[worst]))
			{
				worst = left;
			}
			if (right < Size_ && HeapAbove(Heap_[right], Heap_[worst]))
			{
				worst = right;
			}
			if (worst == i)
			{
				return;
			}
			const Hit tmp = Heap_[i];
			Heap_[i] = Heap_[worst];
			Heap_[worst] = tmp;
			i = worst;
		}
	}

	Hit* Heap_ = nullptr;
	int32_t K_ = 0;
	int32_t Size_ = 0;
	Metric Metric_ = Metric::Dot;
};

// Merge per-chunk top-k lists into a single best-first top-k. The comparator's total
// order makes the result independent of list order; lists are still conventionally
// passed in chunk order.
inline int32_t MergeTopK(
	const Hit* const* lists,
	const int32_t* listCounts,
	int32_t listCount,
	Metric metric,
	int32_t k,
	Hit* heapScratch,
	Hit* out)
{
	TopK merged;
	merged.Init(heapScratch, k, metric);
	for (int32_t l = 0; l < listCount; ++l)
	{
		for (int32_t i = 0; i < listCounts[l]; ++i)
		{
			merged.Push(lists[l][i].index, lists[l][i].score);
		}
	}
	return merged.Finalize(out);
}

} // namespace superfaiss

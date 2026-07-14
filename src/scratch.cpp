#include "superfaiss/scratch.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <new>
#include <thread>

#include "superfaiss/bake.h"
#include "superfaiss/kernels.h"   // detail::FloatBitsToDouble (DAZ-safe scale decode, Poirot #1)
#include "superfaiss/query.h"
#include "superfaiss/validate.h"

namespace superfaiss
{

namespace
{
	// Debug-build assertion of the single-writer contract (plan section 7): the
	// guard trips when two writers overlap; release builds compile it away entirely.
	struct WriterGuard
	{
		explicit WriterGuard(std::atomic<bool>& busy) : Busy_(busy)
		{
			const bool wasBusy = Busy_.exchange(true, std::memory_order_acquire);
			(void)wasBusy;
			assert(!wasBusy && "ScratchBank: concurrent writers (single-writer contract)");
		}
		~WriterGuard() { Busy_.store(false, std::memory_order_release); }
		std::atomic<bool>& Busy_;
	};

	// Deterministic std-only PRNG for the recall sweep (xorshift64*). Given the same
	// seed and bank history it draws the same self-queries on every machine, so the
	// recall number is reproducible (V2.3 plan section 20).
	struct RecallRng
	{
		uint64_t state;
		explicit RecallRng(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ull) {}
		uint64_t Next()
		{
			state ^= state >> 12;
			state ^= state << 25;
			state ^= state >> 27;
			return state * 0x2545F4914F6CDD1Dull;
		}
		// Uniform in [0, 1) from the top 53 bits (exact in double).
		double Unit() { return static_cast<double>(Next() >> 11) * (1.0 / 9007199254740992.0); }
	};

	// Fixed-capacity top-k for the recall reference scan (k <= 10, so it lives on the
	// stack — no allocation). Kept sorted best-first in the library's own total order
	// (score, then ascending index; L2 lower is better), so membership and depth are a
	// linear scan.
	struct RecallTopK
	{
		static constexpr int32_t kMax = 10;
		double sc[kMax];
		int32_t idx[kMax];
		int32_t n = 0;
		int32_t cap = 0;

		void Init(int32_t k) { cap = k; n = 0; }
		void Clear() { n = 0; }
		int32_t Count() const { return n; }

		static bool Better(double sa, int32_t ia, double sb, int32_t ib, bool l2)
		{
			if (sa != sb)
			{
				return l2 ? (sa < sb) : (sa > sb);
			}
			return ia < ib;
		}

		void Insert(int32_t index, double score, bool l2)
		{
			if (n == cap)
			{
				if (!Better(score, index, sc[n - 1], idx[n - 1], l2))
				{
					return;
				}
				--n; // drop the current worst, then place the newcomer
			}
			int32_t p = n;
			while (p > 0 && Better(score, index, sc[p - 1], idx[p - 1], l2))
			{
				sc[p] = sc[p - 1];
				idx[p] = idx[p - 1];
				--p;
			}
			sc[p] = score;
			idx[p] = index;
			++n;
		}

		bool Contains(int32_t index) const
		{
			for (int32_t i = 0; i < n; ++i)
			{
				if (idx[i] == index)
				{
					return true;
				}
			}
			return false;
		}
	};

	constexpr uint32_t kScratchMagic = 0x42535346u; // "FSSB" little-endian
	// V2.3 bumped 1 -> 2: a retention-carrying blob writes 2, a non-retention blob
	// still writes 1, and this reader accepts both. Version 0 or > 3 is a hard reject.
	constexpr uint32_t kScratchVersion = 2;
	constexpr uint32_t kScratchVersionRetain = 2; // retained floats present in the blob
	// V3.0 (plan section 23.6, Forge S1): retention is encoded AS the version integer
	// (1=plain, 2=retention), so a linear bump cannot carry channels + retention together.
	// Version 3 makes the presence-flags byte authoritative instead: a channels-carrying
	// blob writes version 3 and sets reserved[0] as a flags byte (bit 0 retention, bit 1
	// channels; bits 2-7 reserved, tolerated on read). A channel-LESS bank keeps writing
	// the legacy 1/2 (byte-identical, old-reader-compatible); channels are what trigger v3.
	constexpr uint32_t kScratchVersionChannels = 3;
	constexpr size_t kScratchFlagsByteIndex = 0;    // flags live in reserved[0] (header offset 26)
	constexpr uint8_t kScratchFlagRetention = 0x01; // reserved[0] bit 0
	constexpr uint8_t kScratchFlagChannels = 0x02;  // reserved[0] bit 1
	// Archive geometry ceiling (review M2): the largest row capacity a load will
	// entertain. With paddedDims capped at kMaxCrossDeviceDims, every ArenaBytes
	// term stays below 2^49 — far from int64 overflow.
	constexpr int32_t kMaxScratchArchiveRows = 1 << 28;

	struct ScratchHeader
	{
		uint32_t magic = kScratchMagic;
		uint32_t version = kScratchVersion;
		int32_t capacity = 0;
		int32_t count = 0;
		int32_t dims = 0;
		int32_t paddedDims = 0;
		uint8_t metric = 0;
		uint8_t quant = 0;
		uint8_t reserved[6] = {};
	};
	static_assert(sizeof(ScratchHeader) == 32, "scratch header layout is the format");
} // namespace

ScratchBank::~ScratchBank()
{
	Destroy();
}

int64_t ScratchBank::RowRegionBytes(int32_t capacity) const
{
	return static_cast<int64_t>(capacity) * PaddedDims_ * ElementSize(Quant_);
}

int64_t ScratchBank::ArenaBytes(
	int32_t capacity, int32_t dims, int32_t paddedDims, Quantization quant, bool retain,
	int32_t subNormPerRow)
{
	// One allocation: rows, scales (int8 only), tombstone words, one staging row,
	// (retention only) the retained float rows, and (channel Cosine only) the per-channel
	// inverse sub-norms. Rows come first (kAlignment-aligned block, row stride is a multiple
	// of kAlignment); every later region needs only 4-byte alignment and each region's size
	// is a multiple of 4, so the packing preserves it.
	int64_t bytes = static_cast<int64_t>(capacity) * paddedDims * ElementSize(quant);
	bytes = (bytes + 15) & ~int64_t{15};
	if (quant == Quantization::Int8)
	{
		bytes += static_cast<int64_t>(capacity) * sizeof(float);
	}
	bytes += static_cast<int64_t>(TombstoneWords(capacity)) * sizeof(uint32_t);
	bytes += static_cast<int64_t>(dims) * sizeof(float);
	if (retain)
	{
		bytes += static_cast<int64_t>(capacity) * dims * sizeof(float);
	}
	if (subNormPerRow > 0)
	{
		bytes += static_cast<int64_t>(capacity) * subNormPerRow * sizeof(float);
	}
	return bytes;
}

void ScratchBank::BindArena(uint8_t* arena, int32_t capacity)
{
	Arena_ = arena;
	Capacity_ = capacity;
	uint8_t* cursor = arena;
	Rows_ = cursor;
	int64_t rowBytes = static_cast<int64_t>(capacity) * PaddedDims_ * ElementSize(Quant_);
	rowBytes = (rowBytes + 15) & ~int64_t{15};
	cursor += rowBytes;
	if (Quant_ == Quantization::Int8)
	{
		Scales_ = reinterpret_cast<float*>(cursor);
		cursor += static_cast<int64_t>(capacity) * sizeof(float);
	}
	else
	{
		Scales_ = nullptr;
	}
	Tombstones_ = reinterpret_cast<std::atomic<uint32_t>*>(cursor);
	cursor += static_cast<int64_t>(TombstoneWords(capacity)) * sizeof(uint32_t);
	Staging_ = reinterpret_cast<float*>(cursor);
	cursor += static_cast<int64_t>(Dims_) * sizeof(float);
	if (Retain_)
	{
		Retained_ = reinterpret_cast<float*>(cursor);
		cursor += static_cast<int64_t>(capacity) * Dims_ * sizeof(float);
	}
	else
	{
		Retained_ = nullptr;
	}
	// Per-channel inverse sub-norms (V3.0): Cosine channel banks only. Written per row at
	// Append (per-row-standalone, V3-G4); sized into this same allocation (V3-G5).
	if (Metric_ == Metric::Cosine && ChannelCount_ > 0)
	{
		ChannelInvNorms_ = reinterpret_cast<float*>(cursor);
		cursor += static_cast<int64_t>(capacity) * ChannelCount_ * sizeof(float);
	}
	else
	{
		ChannelInvNorms_ = nullptr;
	}
	for (int32_t w = 0; w < TombstoneWords(capacity); ++w)
	{
		::new (static_cast<void*>(Tombstones_ + w)) std::atomic<uint32_t>(0u);
	}
}

Status ScratchBank::Create(
	int32_t capacity, int32_t dims, Metric metric, Quantization quant, bool retainFloats,
	const Allocator& allocator)
{
	if (IsCreated() || capacity <= 0 || dims <= 0)
	{
		return Status::InvalidArgument;
	}
	if (metric != Metric::Dot && metric != Metric::Cosine && metric != Metric::L2)
	{
		return Status::InvalidArgument;
	}
	if (quant != Quantization::Float32 && quant != Quantization::Int8)
	{
		return Status::InvalidArgument;
	}

	const int32_t paddedDims = PaddedDims(dims, quant);
	const int64_t bytes = ArenaBytes(capacity, dims, paddedDims, quant, retainFloats, 0);
	uint8_t* arena = static_cast<uint8_t*>(
		detail::SeamAlloc(allocator, static_cast<size_t>(bytes), kAlignment));
	if (arena == nullptr)
	{
		return Status::OutOfMemory;
	}

	Allocator_ = allocator;
	Dims_ = dims;
	PaddedDims_ = paddedDims;
	Metric_ = metric;
	Quant_ = quant;
	Retain_ = retainFloats;
	ChannelCount_ = 0; // single-space bank: no channel table
	BindArena(arena, capacity);
	PublishedCount_.store(0, std::memory_order_release);
	TombstonedCount_.store(0, std::memory_order_relaxed);
	Generation_.store(0, std::memory_order_release);
	return Status::Ok;
}

// Channel-capable Create (V3.0, plan section 23.4): the channel table becomes a
// scratch-bank property, fixed for the bank's lifetime (D-V3-2). The table is validated
// at construction with the same rules ValidateBank applies to a baked channel table
// (validation moves from import-time to construction-time); on a Cosine bank the arena
// additionally carries a capacity x channelCount per-channel inverse-sub-norm array,
// sized into the SAME single allocation (V3-G5) and filled per-row-standalone at Append
// (V3-G4). channels==null with channelCount==0 is a single-space bank, identical to the
// overloads above.
Status ScratchBank::Create(
	int32_t capacity, int32_t dims, Metric metric, Quantization quant,
	const ChannelInfo* channels, int32_t channelCount, bool retainFloats,
	const Allocator& allocator)
{
	if (IsCreated() || capacity <= 0 || dims <= 0)
	{
		return Status::InvalidArgument;
	}
	if (metric != Metric::Dot && metric != Metric::Cosine && metric != Metric::L2)
	{
		return Status::InvalidArgument;
	}
	if (quant != Quantization::Float32 && quant != Quantization::Int8)
	{
		return Status::InvalidArgument;
	}

	const int32_t paddedDims = PaddedDims(dims, quant);

	// Channel-table validation, mirroring ValidateBank's channel rules exactly (in-bounds,
	// ascending, non-overlapping, on the 16-byte element grid, count in [1, kMaxChannels]).
	// A null table with a nonzero count, or a nonzero count with no table, is malformed.
	if (channels != nullptr || channelCount != 0)
	{
		if (channels == nullptr || channelCount <= 0 || channelCount > kMaxChannels)
		{
			return Status::InvalidArgument;
		}
		const int32_t grid = kAlignment / ElementSize(quant);
		int32_t prevEnd = 0;
		for (int32_t c = 0; c < channelCount; ++c)
		{
			const ChannelInfo& channel = channels[c];
			if (channel.offset < 0 || channel.length <= 0 ||
				channel.offset % grid != 0 || channel.length % grid != 0 ||
				channel.offset < prevEnd ||
				static_cast<int64_t>(channel.offset) + channel.length > paddedDims)
			{
				return Status::InvalidArgument;
			}
			prevEnd = channel.offset + channel.length;
		}
	}

	const int32_t subNormPerRow = (metric == Metric::Cosine && channelCount > 0)
		? channelCount
		: 0;
	const int64_t bytes = ArenaBytes(capacity, dims, paddedDims, quant, retainFloats,
		subNormPerRow);
	uint8_t* arena = static_cast<uint8_t*>(
		detail::SeamAlloc(allocator, static_cast<size_t>(bytes), kAlignment));
	if (arena == nullptr)
	{
		return Status::OutOfMemory;
	}

	Allocator_ = allocator;
	Dims_ = dims;
	PaddedDims_ = paddedDims;
	Metric_ = metric;
	Quant_ = quant;
	Retain_ = retainFloats;
	ChannelCount_ = channelCount;
	for (int32_t c = 0; c < channelCount; ++c)
	{
		Channels_[c] = channels[c];
	}
	BindArena(arena, capacity);
	PublishedCount_.store(0, std::memory_order_release);
	TombstonedCount_.store(0, std::memory_order_relaxed);
	Generation_.store(0, std::memory_order_release);
	return Status::Ok;
}

void ScratchBank::Destroy()
{
	if (Arena_ != nullptr)
	{
		detail::SeamFree(Allocator_, Arena_);
	}
	Arena_ = nullptr;
	Rows_ = nullptr;
	Scales_ = nullptr;
	Tombstones_ = nullptr;
	Staging_ = nullptr;
	Retained_ = nullptr;
	ChannelInvNorms_ = nullptr;
	ChannelCount_ = 0;
	Retain_ = false;
	Capacity_ = 0;
	PublishedCount_.store(0, std::memory_order_relaxed);
	TombstonedCount_.store(0, std::memory_order_relaxed);
	Generation_.store(0, std::memory_order_relaxed);
}

bool ScratchBank::TryPinReader()
{
	if (ExclusiveWaiting_.load(std::memory_order_seq_cst))
	{
		return false;
	}
	ReaderPins_.fetch_add(1, std::memory_order_seq_cst);
	// Re-check after the increment. seq_cst makes the argument airtight: in the
	// single total order S, if the exclusive side's flag-store preceded this
	// load, we see it and back out; otherwise this load precedes the store in S,
	// so our increment does too, and the exclusive side's pin-count load (after
	// its store) observes us and keeps waiting. Acquire/release cannot order
	// this store-buffering pair (Poirot F4).
	if (ExclusiveWaiting_.load(std::memory_order_seq_cst))
	{
		ReaderPins_.fetch_sub(1, std::memory_order_seq_cst);
		return false;
	}
	return true;
}

void ScratchBank::UnpinReader()
{
	ReaderPins_.fetch_sub(1, std::memory_order_release);
}

bool ScratchBank::BeginExclusive()
{
	bool expected = false;
	if (!ExclusiveWaiting_.compare_exchange_strong(
			expected, true, std::memory_order_seq_cst))
	{
		return false;
	}
	while (ReaderPins_.load(std::memory_order_seq_cst) > 0)
	{
		std::this_thread::yield();
	}
	return true;
}

void ScratchBank::EndExclusive()
{
	ExclusiveWaiting_.store(false, std::memory_order_release);
}

Status ScratchBank::Append(const float* row, int32_t dims, int32_t* outIndex)
{
	if (!IsCreated() || row == nullptr)
	{
		return Status::InvalidArgument;
	}
	if (dims != Dims_)
	{
		return Status::DimsMismatch;
	}
	WriterGuard guard(WriterBusy_);
	return AppendValidated(row, outIndex);
}

Status ScratchBank::AppendValidated(const float* row, int32_t* outIndex)
{
	const int32_t index = PublishedCount_.load(std::memory_order_relaxed);
	if (index >= Capacity_)
	{
		// The capacity budget is exhausted; Grow() to proceed.
		return Status::OutOfMemory;
	}

	const Status valid = ValidateSourceRows(row, 1, Dims_, nullptr);
	if (valid != Status::Ok)
	{
		return valid;
	}

	// Stage the row so normalization never touches the caller's buffer, and the
	// arena slot is written exactly once, fully formed, before the count publishes.
	std::memcpy(Staging_, row, static_cast<size_t>(Dims_) * sizeof(float));
	if (Metric_ == Metric::Cosine)
	{
		const Status norm = NormalizeRows(Staging_, 1, Dims_, nullptr);
		if (norm != Status::Ok)
		{
			return norm; // ZeroNormRow: format rule, same as the importer
		}
	}

	if (Quant_ == Quantization::Int8)
	{
		int8_t* dst = static_cast<int8_t*>(Rows_) + static_cast<int64_t>(index) * PaddedDims_;
		QuantizeRowsInt8(Staging_, 1, Dims_, PaddedDims_, dst, Scales_ + index);
	}
	else
	{
		float* dst = static_cast<float*>(Rows_) + static_cast<int64_t>(index) * PaddedDims_;
		PadRowsFloat32(Staging_, 1, Dims_, PaddedDims_, dst);
	}

	// Retention (V2.3): bit-store the post-normalization row the quantizer just
	// consumed — the audit reference. Written into the retention slot before the count
	// publishes, so a snapshot never sees a row without its retained twin.
	if (Retain_)
	{
		std::memcpy(Retained_ + static_cast<int64_t>(index) * Dims_, Staging_,
			static_cast<size_t>(Dims_) * sizeof(float));
	}

	// Per-channel inverse sub-norms (V3.0, Cosine channel banks): computed from THIS row's
	// just-written QUANTIZED bytes over each channel range — per-row-standalone (V3-G4), no
	// cross-row read. A verbatim per-row recode of bake.cpp's ComputeChannelInverseNorms
	// (c outer, j inner, double accumulate, 1/sqrt or 0 on a zero-norm channel), so the
	// append-time array bit-equals that reference over the snapshot's own rows. Written into
	// the arena slot before the count publishes — a snapshot never sees a row without its
	// sub-norms.
	if (Metric_ == Metric::Cosine && ChannelCount_ > 0)
	{
		float* invNorms = ChannelInvNorms_ + static_cast<int64_t>(index) * ChannelCount_;
		for (int32_t c = 0; c < ChannelCount_; ++c)
		{
			const ChannelInfo& channel = Channels_[c];
			double norm = 0.0;
			if (Quant_ == Quantization::Int8)
			{
				const int8_t* qrow =
					static_cast<const int8_t*>(Rows_) + static_cast<int64_t>(index) * PaddedDims_;
				// DAZ-safe scale decode (Poirot #1), matching bake.cpp's reference and the
				// scoring epilogue — keeps this recode bit-equal to ComputeChannelInverseNorms.
				const double scale = detail::FloatBitsToDouble(Scales_[index]);
				for (int32_t j = channel.offset; j < channel.offset + channel.length; ++j)
				{
					const double v = scale * qrow[j];
					norm += v * v;
				}
			}
			else
			{
				const float* qrow =
					static_cast<const float*>(Rows_) + static_cast<int64_t>(index) * PaddedDims_;
				for (int32_t j = channel.offset; j < channel.offset + channel.length; ++j)
				{
					norm += static_cast<double>(qrow[j]) * qrow[j];
				}
			}
			invNorms[c] = norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 0.0f;
		}
	}

	// Row fully written; only now does it exist for readers.
	PublishedCount_.store(index + 1, std::memory_order_release);
	// A successful append is a mutation: advance the generation so any prior recall
	// report reads as stale (release-paired with Generation()'s acquire load).
	Generation_.fetch_add(1, std::memory_order_release);
	if (outIndex != nullptr)
	{
		*outIndex = index;
	}
	return Status::Ok;
}

Status ScratchBank::Remove(int32_t index)
{
	if (!IsCreated())
	{
		return Status::InvalidArgument;
	}
	WriterGuard guard(WriterBusy_);
	if (index < 0 || index >= PublishedCount_.load(std::memory_order_relaxed))
	{
		return Status::InvalidArgument;
	}
	const uint32_t bit = 1u << (index & 31);
	const uint32_t prev = Tombstones_[index >> 5].fetch_or(bit, std::memory_order_relaxed);
	if ((prev & bit) == 0)
	{
		TombstonedCount_.fetch_add(1, std::memory_order_relaxed);
		// A newly-set tombstone is a mutation; an idempotent re-Remove is not, so it
		// leaves the generation (and any prior recall report's currency) untouched.
		Generation_.fetch_add(1, std::memory_order_release);
	}
	return Status::Ok;
}

Status ScratchBank::Snapshot(BankView* outView, uint32_t* outTombstones) const
{
	if (!IsCreated() || outView == nullptr || outTombstones == nullptr)
	{
		return Status::InvalidArgument;
	}
	const int32_t count = PublishedCount_.load(std::memory_order_acquire);
	BankView view;
	view.rows = Rows_;
	view.scales = Scales_;
	view.count = count;
	view.dims = Dims_;
	view.paddedDims = PaddedDims_;
	view.quant = Quant_;
	view.metric = Metric_;
	// Channel table (V3.0): the snapshot IS a channel-carrying BankView. channelInvNorms is
	// non-null only on a Cosine channel bank (Dot/L2 channel banks score from the segments
	// alone and carry no sub-norm arena — the kernel gates on Cosine && channelInvNorms).
	if (ChannelCount_ > 0)
	{
		view.channels = Channels_;
		view.channelCount = ChannelCount_;
		view.channelInvNorms = ChannelInvNorms_;
	}
	*outView = view;
	for (int32_t w = 0; w < TombstoneWords(count); ++w)
	{
		outTombstones[w] = Tombstones_[w].load(std::memory_order_relaxed);
	}
	return Status::Ok;
}

Status ScratchBank::Grow(int32_t newCapacity)
{
	if (!IsCreated())
	{
		return Status::InvalidArgument;
	}
	WriterGuard guard(WriterBusy_);
	if (newCapacity <= Capacity_)
	{
		return Status::InvalidArgument;
	}
	const int32_t subNormPerRow = (Metric_ == Metric::Cosine && ChannelCount_ > 0)
		? ChannelCount_
		: 0;
	const int64_t bytes = ArenaBytes(newCapacity, Dims_, PaddedDims_, Quant_, Retain_,
		subNormPerRow);
	uint8_t* arena = static_cast<uint8_t*>(
		detail::SeamAlloc(Allocator_, static_cast<size_t>(bytes), kAlignment));
	if (arena == nullptr)
	{
		return Status::OutOfMemory;
	}

	const int32_t count = PublishedCount_.load(std::memory_order_relaxed);
	const void* oldRows = Rows_;
	const float* oldScales = Scales_;
	const std::atomic<uint32_t>* oldTombstones = Tombstones_;
	const float* oldRetained = Retained_;
	const float* oldChannelInvNorms = ChannelInvNorms_;
	uint8_t* oldArena = Arena_;

	BindArena(arena, newCapacity);

	// Index-preserving by construction (T-044 W4): rows, scales, tombstones, and the
	// retention arena copy straight across; nothing compacts, nothing renumbers.
	std::memcpy(Rows_, oldRows,
		static_cast<size_t>(count) * PaddedDims_ * ElementSize(Quant_));
	if (Quant_ == Quantization::Int8)
	{
		std::memcpy(Scales_, oldScales, static_cast<size_t>(count) * sizeof(float));
	}
	for (int32_t w = 0; w < TombstoneWords(count); ++w)
	{
		Tombstones_[w].store(
			oldTombstones[w].load(std::memory_order_relaxed), std::memory_order_relaxed);
	}
	if (Retain_)
	{
		std::memcpy(Retained_, oldRetained,
			static_cast<size_t>(count) * Dims_ * sizeof(float));
	}
	// Index-preserving for the sub-norm arena too (V3.0): each surviving row keeps its
	// per-channel inverse sub-norms at the same index — no recompute, bit-unchanged.
	if (subNormPerRow > 0)
	{
		std::memcpy(ChannelInvNorms_, oldChannelInvNorms,
			static_cast<size_t>(count) * ChannelCount_ * sizeof(float));
	}

	detail::SeamFree(Allocator_, oldArena);
	return Status::Ok;
}

// Mutable channel vocabulary (V3.1, plan section 24.4): atomically replace the channel
// table on a live bank. The stored rows are never touched — channels are sub-ranges over
// the same dims (section 24.3) — so a relabel swaps the table and, for a Cosine bank,
// re-derives the per-channel inverse sub-norms under the new table. EXCLUSIVE, the same
// class as Grow/Load (the host drains readers before calling). Reject-over-degrade: the
// new table is validated and the (Cosine) arena is allocated BEFORE any state changes, so
// a malformed table (InvalidArgument) or a realloc failure (OutOfMemory) leaves the bank
// exactly as before, still queryable under the old table.
Status ScratchBank::Relabel(const ChannelInfo* newChannels, int32_t newChannelCount)
{
	if (!IsCreated())
	{
		return Status::InvalidArgument;
	}
	WriterGuard guard(WriterBusy_);

	// Validate the new table with the SAME rules Create/Load apply (validation symmetry,
	// dim 2) — before a byte of state changes. A null table with nonzero count, a nonzero
	// table with zero/negative count, out-of-bounds, overlap, non-ascending, off-grid, or
	// length <= 0 each return InvalidArgument and leave the bank untouched. newChannelCount
	// == 0 with a null table is the demote-to-single-space case.
	if (newChannels != nullptr || newChannelCount != 0)
	{
		if (newChannels == nullptr || newChannelCount <= 0 || newChannelCount > kMaxChannels)
		{
			return Status::InvalidArgument;
		}
		const int32_t grid = kAlignment / ElementSize(Quant_);
		int32_t prevEnd = 0;
		for (int32_t c = 0; c < newChannelCount; ++c)
		{
			const ChannelInfo& channel = newChannels[c];
			if (channel.offset < 0 || channel.length <= 0 ||
				channel.offset % grid != 0 || channel.length % grid != 0 ||
				channel.offset < prevEnd ||
				static_cast<int64_t>(channel.offset) + channel.length > PaddedDims_)
			{
				return Status::InvalidArgument;
			}
			prevEnd = channel.offset + channel.length;
		}
	}

	const int32_t newSubNormPerRow = (Metric_ == Metric::Cosine && newChannelCount > 0)
		? newChannelCount
		: 0;
	const int32_t oldSubNormPerRow = (Metric_ == Metric::Cosine && ChannelCount_ > 0)
		? ChannelCount_
		: 0;

	// Dot/L2 banks carry no sub-norm arena (V3-G2/V31-G2), so a relabel is a validate-and-
	// swap of the by-value members with no arena touch — it cannot OOM. This runs under the
	// same exclusive drain as the Cosine path (Forge W2: Snapshot aliases Channels_/
	// ChannelCount_ into every live BankView, so even the member write is observable through
	// a held view — the host drain, not a lock-free poke, is what makes it safe). A Cosine
	// bank with no channels on either side (single-space -> single-space) has no sub-norm
	// region either way and takes the same cheap path.
	if (oldSubNormPerRow == 0 && newSubNormPerRow == 0)
	{
		ChannelCount_ = newChannelCount;
		for (int32_t c = 0; c < newChannelCount; ++c)
		{
			Channels_[c] = newChannels[c];
		}
		Generation_.fetch_add(1, std::memory_order_release);
		return Status::Ok;
	}

	// Cosine relabel with channels on either side: the sub-norm region changes size (count
	// change / promote / demote) or its values change (boundary move), so the only atomic
	// path that preserves reject-over-degrade is to build a fresh arena and adopt it on
	// success — the Grow index-preserving realloc template (V31-G7), differing only in that
	// the sub-norm region is RE-DERIVED under the new table (V31-G5), never copied.
	const int64_t bytes = ArenaBytes(Capacity_, Dims_, PaddedDims_, Quant_, Retain_,
		newSubNormPerRow);
	uint8_t* arena = static_cast<uint8_t*>(
		detail::SeamAlloc(Allocator_, static_cast<size_t>(bytes), kAlignment));
	if (arena == nullptr)
	{
		// Reject-over-degrade: the bank is unchanged and still queryable under the old table.
		return Status::OutOfMemory;
	}

	const int32_t count = PublishedCount_.load(std::memory_order_relaxed);
	const void* oldRows = Rows_;
	const float* oldScales = Scales_;
	const std::atomic<uint32_t>* oldTombstones = Tombstones_;
	const float* oldRetained = Retained_;
	uint8_t* oldArena = Arena_;

	// The allocation succeeded — the last fallible step is behind us. Commit the new table
	// so BindArena sizes the sub-norm region for the new count, then copy the immutable
	// state across index-preserving (nothing renumbers) and re-derive the sub-norms.
	ChannelCount_ = newChannelCount;
	for (int32_t c = 0; c < newChannelCount; ++c)
	{
		Channels_[c] = newChannels[c];
	}
	BindArena(arena, Capacity_);

	std::memcpy(Rows_, oldRows,
		static_cast<size_t>(count) * PaddedDims_ * ElementSize(Quant_));
	if (Quant_ == Quantization::Int8)
	{
		std::memcpy(Scales_, oldScales, static_cast<size_t>(count) * sizeof(float));
	}
	for (int32_t w = 0; w < TombstoneWords(count); ++w)
	{
		Tombstones_[w].store(
			oldTombstones[w].load(std::memory_order_relaxed), std::memory_order_relaxed);
	}
	if (Retain_)
	{
		std::memcpy(Retained_, oldRetained,
			static_cast<size_t>(count) * Dims_ * sizeof(float));
	}

	// Re-derive the per-channel inverse sub-norms under the NEW table (only when the new
	// table has channels — a demote leaves ChannelInvNorms_ null). The rows are byte-
	// identical to the old arena (just copied, never re-quantized), so this is the same
	// ComputeChannelInverseNorms a fresh Create+Append and a Load run — the parity oracle
	// (dim 6). It is total over these validated inputs (non-null rows, a valid table).
	if (newSubNormPerRow > 0)
	{
		BankView cview;
		cview.rows = Rows_;
		cview.scales = Scales_;
		cview.count = count;
		cview.dims = Dims_;
		cview.paddedDims = PaddedDims_;
		cview.quant = Quant_;
		cview.metric = Metric_;
		cview.channels = Channels_;
		cview.channelCount = ChannelCount_;
		const Status subNorm = ComputeChannelInverseNorms(cview, ChannelInvNorms_);
		if (subNorm != Status::Ok)
		{
			return subNorm;
		}
	}

	detail::SeamFree(Allocator_, oldArena);

	// A relabel is a mutation: advance the generation (only forward) so any pre-relabel
	// recall report reads stale (V31-G11).
	Generation_.fetch_add(1, std::memory_order_release);
	return Status::Ok;
}

Status ScratchBank::Freeze(void* outRows, float* outScales, int32_t* outIndexMap,
	ScratchRecallReport* outReport, Workspace* recallWs, uint64_t recallSeed) const
{
	if (!IsCreated() || outRows == nullptr)
	{
		return Status::InvalidArgument;
	}
	if (Quant_ == Quantization::Int8 && outScales == nullptr)
	{
		return Status::InvalidArgument;
	}
	WriterGuard guard(WriterBusy_);

	// Re-measure at freeze time (V2.3): the graduated bank carries a number measured
	// over the rows that survive this compaction, not a pre-mutation report. Only on a
	// retention bank with a workspace supplied; a non-retention Freeze produces none,
	// leaving *outReport untouched. Measured before compaction, but recall excludes
	// tombstoned rows either way, so the number equals the compacted bank's.
	if (outReport != nullptr && recallWs != nullptr && Retain_)
	{
		const Status measured = MeasureRecallLocked(recallSeed, *recallWs, outReport);
		if (measured != Status::Ok)
		{
			return measured;
		}
	}

	const int32_t count = PublishedCount_.load(std::memory_order_relaxed);
	const size_t rowBytes = static_cast<size_t>(PaddedDims_) * ElementSize(Quant_);
	int32_t next = 0;
	for (int32_t i = 0; i < count; ++i)
	{
		const bool dead =
			(Tombstones_[i >> 5].load(std::memory_order_relaxed) & (1u << (i & 31))) != 0;
		if (outIndexMap != nullptr)
		{
			outIndexMap[i] = dead ? -1 : next;
		}
		if (dead)
		{
			continue;
		}
		// Rows were normalized/quantized at append; compaction is a pure copy, so a
		// frozen bank scores bit-identically to the snapshot it came from (T-V2-C2).
		std::memcpy(static_cast<uint8_t*>(outRows) + static_cast<size_t>(next) * rowBytes,
			static_cast<const uint8_t*>(Rows_) + static_cast<size_t>(i) * rowBytes, rowBytes);
		if (Quant_ == Quantization::Int8)
		{
			outScales[next] = Scales_[i];
		}
		++next;
	}
	return Status::Ok;
}

// Compaction shared by the channel-aware Freeze paths (V3.0, section 23.4). Copies the
// live rows (and int8 scales) into the caller buffers exactly as the base Freeze does — a
// pure byte copy, never a re-quantize (V3-G6 / T-V2-C2), so a frozen row is bit-identical
// to the scratch row it graduated from — fills the old->new index map, then RE-DERIVES the
// per-channel inverse sub-norms over the compacted rows into outChannelInvNorms (the
// survivors renumbered, so the graduated bank's sub-norms follow them). Assumes a validated
// Cosine channel bank and that the caller holds the writer guard.
Status ScratchBank::FreezeChannelsLocked(
	void* outRows, float* outScales, int32_t* outIndexMap, float* outChannelInvNorms) const
{
	const int32_t count = PublishedCount_.load(std::memory_order_relaxed);
	const size_t rowBytes = static_cast<size_t>(PaddedDims_) * ElementSize(Quant_);
	int32_t next = 0;
	for (int32_t i = 0; i < count; ++i)
	{
		const bool dead =
			(Tombstones_[i >> 5].load(std::memory_order_relaxed) & (1u << (i & 31))) != 0;
		if (outIndexMap != nullptr)
		{
			outIndexMap[i] = dead ? -1 : next;
		}
		if (dead)
		{
			continue;
		}
		std::memcpy(static_cast<uint8_t*>(outRows) + static_cast<size_t>(next) * rowBytes,
			static_cast<const uint8_t*>(Rows_) + static_cast<size_t>(i) * rowBytes, rowBytes);
		if (Quant_ == Quantization::Int8)
		{
			outScales[next] = Scales_[i];
		}
		++next;
	}

	// Re-derive the sub-norms over the compacted survivors (bit-equal to
	// ComputeChannelInverseNorms over the frozen BankView — the graduated bank is
	// ground-truth-anchored, not a verbatim copy of the pre-compaction arena).
	BankView frozen;
	frozen.rows = outRows;
	frozen.scales = Quant_ == Quantization::Int8 ? outScales : nullptr;
	frozen.count = next;
	frozen.dims = Dims_;
	frozen.paddedDims = PaddedDims_;
	frozen.quant = Quant_;
	frozen.metric = Metric_;
	frozen.channels = Channels_;
	frozen.channelCount = ChannelCount_;
	if (next == 0)
	{
		return Status::Ok; // zero live rows: a defined empty graduation, nothing to derive
	}
	return ComputeChannelInverseNorms(frozen, outChannelInvNorms);
}

// Channel-aware Freeze (V3.0, section 23.4): base-Freeze compaction plus the re-derived
// per-channel sub-norms. Valid only on a Cosine channel bank (InvalidArgument otherwise —
// use the base Freeze). The optional outReport/recallWs re-measure the WHOLE-bank recall at
// freeze time exactly as the base Freeze (per-channel re-measure is FreezeWithRecall).
Status ScratchBank::Freeze(void* outRows, float* outScales, int32_t* outIndexMap,
	float* outChannelInvNorms, ScratchRecallReport* outReport, Workspace* recallWs,
	uint64_t recallSeed) const
{
	if (!IsCreated() || outRows == nullptr || outChannelInvNorms == nullptr)
	{
		return Status::InvalidArgument;
	}
	if (ChannelCount_ <= 0 || Metric_ != Metric::Cosine)
	{
		return Status::InvalidArgument; // a non-channel / non-Cosine bank uses the base Freeze
	}
	if (Quant_ == Quantization::Int8 && outScales == nullptr)
	{
		return Status::InvalidArgument;
	}
	WriterGuard guard(WriterBusy_);
	if (outReport != nullptr && recallWs != nullptr && Retain_)
	{
		const Status measured = MeasureRecallLocked(recallSeed, *recallWs, outReport);
		if (measured != Status::Ok)
		{
			return measured;
		}
	}
	return FreezeChannelsLocked(outRows, outScales, outIndexMap, outChannelInvNorms);
}

// Channel-aware Freeze that also re-measures per-channel recall over the compacted rows
// (V3.0, D-V3-7). Requires a retention-enabled Cosine channel bank (the float reference the
// recall sweep scans). Each report is measured at the current generation — a fresh number
// for the graduated bank, never a stale one.
Status ScratchBank::FreezeWithRecall(void* outRows, float* outScales, int32_t* outIndexMap,
	float* outChannelInvNorms, ScratchRecallReport* outRecallReports, int32_t reportCount,
	Workspace& recallWs, uint64_t recallSeed) const
{
	if (!IsCreated() || outRows == nullptr || outChannelInvNorms == nullptr ||
		outRecallReports == nullptr)
	{
		return Status::InvalidArgument;
	}
	if (ChannelCount_ <= 0 || Metric_ != Metric::Cosine || !Retain_ ||
		reportCount != ChannelCount_)
	{
		return Status::InvalidArgument;
	}
	if (Quant_ == Quantization::Int8 && outScales == nullptr)
	{
		return Status::InvalidArgument;
	}
	WriterGuard guard(WriterBusy_);
	// Measure per channel BEFORE compaction: the sweep excludes tombstones, so it already
	// scores over exactly the live (about-to-be-compacted) rows, at the current generation.
	for (int32_t c = 0; c < ChannelCount_; ++c)
	{
		const Status measured =
			MeasureRecallLockedChannel(recallSeed, recallWs, Channels_[c], &outRecallReports[c]);
		if (measured != Status::Ok)
		{
			return measured;
		}
	}
	return FreezeChannelsLocked(outRows, outScales, outIndexMap, outChannelInvNorms);
}

// Per-channel recall (V3.0, D-V3-7): one seeded recall@k report per channel over its
// sub-range. Requires a retention-enabled Cosine channel bank; InvalidArgument otherwise.
Status ScratchBank::MeasureScratchRecallPerChannel(
	Workspace& workspace, ScratchRecallReport* outReports, int32_t reportCount, uint64_t seed)
{
	if (!IsCreated() || outReports == nullptr)
	{
		return Status::InvalidArgument;
	}
	// Reject-over-degrade: no channel table (use the whole-vector routine) or no retained
	// floats (no reference to scan) is a defined InvalidArgument, never a guessed number.
	if (ChannelCount_ <= 0 || Metric_ != Metric::Cosine || !Retain_ ||
		reportCount != ChannelCount_)
	{
		return Status::InvalidArgument;
	}
	while (!TryPinReader())
	{
		std::this_thread::yield();
	}
	Status status = Status::Ok;
	for (int32_t c = 0; c < ChannelCount_; ++c)
	{
		status = MeasureRecallLockedChannel(seed, workspace, Channels_[c], &outReports[c]);
		if (status != Status::Ok)
		{
			break;
		}
	}
	UnpinReader();
	return status;
}

Status ScratchBank::Save(const ScratchArchive& archive) const
{
	if (!IsCreated() || archive.write == nullptr)
	{
		return Status::InvalidArgument;
	}
	WriterGuard guard(WriterBusy_);

	ScratchHeader header;
	// Writer version-selection (Forge S1 / Japp G-3): channels are what trigger the v3
	// presence-flags format. A channel-carrying bank writes version 3 with the flags byte
	// (retention bit + channels bit) authoritative and appends the channel table; a
	// channel-LESS bank keeps the legacy encoding — version 2 with retained floats, else
	// version 1 (byte-identical to a pre-V2.3 blob) — so old readers stay compatible.
	const bool hasChannels = ChannelCount_ > 0;
	if (hasChannels)
	{
		header.version = kScratchVersionChannels;
		header.reserved[kScratchFlagsByteIndex] = static_cast<uint8_t>(
			(Retain_ ? kScratchFlagRetention : 0u) | kScratchFlagChannels);
	}
	else
	{
		header.version = Retain_ ? kScratchVersionRetain : 1u;
	}
	header.capacity = Capacity_;
	header.count = PublishedCount_.load(std::memory_order_relaxed);
	header.dims = Dims_;
	header.paddedDims = PaddedDims_;
	header.metric = static_cast<uint8_t>(Metric_);
	header.quant = static_cast<uint8_t>(Quant_);
	if (!archive.write(archive.user, &header, sizeof(header)))
	{
		return Status::BadFormat;
	}
	// The channel table (v3 only), written immediately after the header so Load can size
	// the arena (which folds in the sub-norm region) before it reads the rows. The
	// per-channel sub-norm arena itself is NOT serialized — it is re-derived on Load from
	// {rows, channel table, scales} (Forge W3), which removes a desync surface.
	if (hasChannels)
	{
		const int32_t channelCount = ChannelCount_;
		if (!archive.write(archive.user, &channelCount, sizeof(channelCount)) ||
			!archive.write(archive.user, Channels_,
				static_cast<size_t>(channelCount) * sizeof(ChannelInfo)))
		{
			return Status::BadFormat;
		}
	}
	const size_t rowBytes =
		static_cast<size_t>(header.count) * PaddedDims_ * ElementSize(Quant_);
	if (rowBytes > 0 && !archive.write(archive.user, Rows_, rowBytes))
	{
		return Status::BadFormat;
	}
	if (Quant_ == Quantization::Int8 && header.count > 0 &&
		!archive.write(archive.user, Scales_, static_cast<size_t>(header.count) * sizeof(float)))
	{
		return Status::BadFormat;
	}
	for (int32_t w = 0; w < TombstoneWords(header.count); ++w)
	{
		const uint32_t word = Tombstones_[w].load(std::memory_order_relaxed);
		if (!archive.write(archive.user, &word, sizeof(word)))
		{
			return Status::BadFormat;
		}
	}
	if (Retain_ && header.count > 0 &&
		!archive.write(archive.user, Retained_,
			static_cast<size_t>(header.count) * Dims_ * sizeof(float)))
	{
		return Status::BadFormat;
	}
	return Status::Ok;
}

Status ScratchBank::Load(const ScratchArchive& archive, const Allocator& allocator)
{
	if (archive.read == nullptr)
	{
		return Status::InvalidArgument;
	}
	// Load's contract is stronger than single-writer (fully exclusive), so the
	// debug owner guard covers it too (Poirot F3) - a writer overlapping a Load
	// asserts in dev builds instead of racing the arena swap silently.
	WriterGuard guard(WriterBusy_);

	ScratchHeader header;
	if (!archive.read(archive.user, &header, sizeof(header)))
	{
		return Status::BadFormat;
	}
	// Accepts {1, 2, 3}; version 0 or anything newer is the standing old-reader/new-data
	// hard-reject (a future v4 archive rejects here). Legacy 1/2 encode retention as the
	// version integer; version 3 makes the flags byte authoritative.
	if (header.magic != kScratchMagic ||
		header.version < 1u || header.version > kScratchVersionChannels)
	{
		return Status::BadFormat;
	}
	// Version 3: retention and channel presence come from the reserved[0] flags byte, and
	// bits 2-7 are reserved — masked off, tolerated (never rejected, never mis-read as
	// retention or channels; Japp G-3 forward tolerance). Legacy 1/2 read retention from
	// the version integer, exactly as shipped, and carry no channel table.
	const uint8_t flags = header.reserved[kScratchFlagsByteIndex];
	const bool hasChannels = header.version == kScratchVersionChannels &&
		(flags & kScratchFlagChannels) != 0;
	const bool wantRetain = header.version == kScratchVersionChannels
		? (flags & kScratchFlagRetention) != 0
		: header.version == kScratchVersionRetain;
	// Version 3 is emitted ONLY for a channel-carrying bank (channels trigger the presence-
	// flags format; a retention-only bank stays legacy v2). A v3 blob without the channels
	// flag is therefore something the writer never produces — reject it as corrupt rather
	// than adopt it as a plain bank. This also catches a legacy blob whose version integer
	// was hand-bumped to 3 (its reserved bytes are zero, so the channels flag is clear).
	if (header.version == kScratchVersionChannels && !hasChannels)
	{
		return Status::BadFormat;
	}
	const bool metricOk = header.metric <= static_cast<uint8_t>(Metric::L2);
	const bool quantOk = header.quant <= static_cast<uint8_t>(Quantization::Int8);
	if (!metricOk || !quantOk || header.capacity <= 0 || header.dims <= 0 ||
		header.count < 0 || header.count > header.capacity ||
		header.paddedDims !=
			PaddedDims(header.dims, static_cast<Quantization>(header.quant)))
	{
		return Status::BadFormat;
	}
	// The archive is an untrusted medium (review M2, the T-062 idiom): bound the
	// geometry BEFORE any byte-size arithmetic — the arena math multiplies the
	// caller-controlled capacity and dims, and an unbounded pair is signed int64
	// overflow (UB), not merely a failed allocation. paddedDims is capped at the
	// widest bank the library proves (kMaxCrossDeviceDims); capacity at 2^28 rows
	// keeps every ArenaBytes term below 2^49. Absurd geometry is a format defect —
	// a hard BadFormat, never an allocator outcome.
	if (header.paddedDims > kMaxCrossDeviceDims ||
		header.capacity > kMaxScratchArchiveRows)
	{
		return Status::BadFormat;
	}

	// The channel table (v3), read here — before the arena is created — so the channel
	// Create sizes the sub-norm region into the single allocation. Bounded before use; the
	// channel Create re-validates the geometry (the archive is untrusted).
	ChannelInfo channels[kMaxChannels];
	int32_t channelCount = 0;
	if (hasChannels)
	{
		if (!archive.read(archive.user, &channelCount, sizeof(channelCount)))
		{
			return Status::BadFormat;
		}
		if (channelCount <= 0 || channelCount > kMaxChannels)
		{
			return Status::BadFormat;
		}
		if (!archive.read(archive.user, channels,
				static_cast<size_t>(channelCount) * sizeof(ChannelInfo)))
		{
			return Status::BadFormat;
		}
	}

	// Build the incoming state in a fresh bank so a bad archive leaves this one
	// unchanged (reject-over-degrade). A channel archive is created channel-capable so its
	// arena carries the (about-to-be-re-derived) sub-norm region.
	ScratchBank incoming;
	const Status created = hasChannels
		? incoming.Create(header.capacity, header.dims,
			  static_cast<Metric>(header.metric), static_cast<Quantization>(header.quant),
			  channels, channelCount, wantRetain, allocator)
		: incoming.Create(header.capacity, header.dims,
			  static_cast<Metric>(header.metric), static_cast<Quantization>(header.quant),
			  wantRetain, allocator);
	if (created != Status::Ok)
	{
		// A malformed channel table fails the channel Create's geometry validation; in an
		// untrusted archive that is a format defect, not an argument error.
		return (hasChannels && created == Status::InvalidArgument) ? Status::BadFormat
																   : created;
	}

	const size_t rowBytes =
		static_cast<size_t>(header.count) * header.paddedDims * ElementSize(incoming.Quant_);
	if (rowBytes > 0 && !archive.read(archive.user, incoming.Rows_, rowBytes))
	{
		return Status::BadFormat;
	}
	if (incoming.Quant_ == Quantization::Int8 && header.count > 0 &&
		!archive.read(archive.user, incoming.Scales_,
			static_cast<size_t>(header.count) * sizeof(float)))
	{
		return Status::BadFormat;
	}
	int32_t tombstoned = 0;
	for (int32_t w = 0; w < TombstoneWords(header.count); ++w)
	{
		uint32_t word = 0;
		if (!archive.read(archive.user, &word, sizeof(word)))
		{
			return Status::BadFormat;
		}
		// Bits at or above count would exclude rows that do not exist; the format
		// never writes them (Save reads live words, and Remove bounds-checks), so
		// their presence means a corrupt or hand-edited archive.
		const int32_t wordBase = w * 32;
		for (int32_t b = 0; b < 32; ++b)
		{
			if ((word & (1u << b)) == 0)
			{
				continue;
			}
			if (wordBase + b >= header.count)
			{
				return Status::BadFormat;
			}
			++tombstoned;
		}
		incoming.Tombstones_[w].store(word, std::memory_order_relaxed);
	}
	// Retained floats (version 2 only): the post-normalization audit reference, one
	// dims-wide row per published row, index-preserving.
	if (wantRetain && header.count > 0 &&
		!archive.read(archive.user, incoming.Retained_,
			static_cast<size_t>(header.count) * header.dims * sizeof(float)))
	{
		return Status::BadFormat;
	}

	// The archive is not trusted: re-validate content with the bake path's rules.
	incoming.PublishedCount_.store(header.count, std::memory_order_release);
	incoming.TombstonedCount_.store(tombstoned, std::memory_order_relaxed);
	incoming.Generation_.store(
		static_cast<uint64_t>(header.count), std::memory_order_relaxed);
	BankView view;
	view.rows = incoming.Rows_;
	view.scales = incoming.Scales_;
	view.count = header.count;
	view.dims = header.dims;
	view.paddedDims = header.paddedDims;
	view.quant = incoming.Quant_;
	view.metric = incoming.Metric_;
	const Status content = ValidateBankData(view, nullptr);
	if (content != Status::Ok)
	{
		return content;
	}

	// Re-derive the per-channel sub-norm arena on Load (Forge W3): recompute it from the
	// loaded rows + channel table rather than trust a serialized copy, so a desynced arena
	// cannot load as authoritative. The channel Create above sized the region into
	// incoming's arena; ComputeChannelInverseNorms fills it exactly as Append did.
	if (hasChannels && incoming.Metric_ == Metric::Cosine)
	{
		BankView cview = view;
		cview.channels = incoming.Channels_;
		cview.channelCount = incoming.ChannelCount_;
		const Status subNorm = ComputeChannelInverseNorms(cview, incoming.ChannelInvNorms_);
		if (subNorm != Status::Ok)
		{
			return subNorm;
		}
	}

	// Validated: adopt the incoming arena wholesale (its region pointers included —
	// re-binding would re-zero the tombstones just loaded). Exclusive by contract,
	// so the swap is not observed mid-flight.
	// A Load is the ultimate mutation (review S1): the generation advances PAST
	// every stamp this instance ever issued — never backward, never a collision.
	// Resetting it to the loaded count would re-issue an append-only bank's exact
	// stamp (generation == count there), letting a recall report taken before the
	// Load read as current against freshly-loaded rows. Captured before Destroy()
	// zeroes it.
	const uint64_t priorGeneration = Generation_.load(std::memory_order_relaxed);
	Destroy();
	Allocator_ = incoming.Allocator_;
	Arena_ = incoming.Arena_;
	Rows_ = incoming.Rows_;
	Scales_ = incoming.Scales_;
	Tombstones_ = incoming.Tombstones_;
	Staging_ = incoming.Staging_;
	Retained_ = incoming.Retained_;
	Capacity_ = incoming.Capacity_;
	Dims_ = incoming.Dims_;
	PaddedDims_ = incoming.PaddedDims_;
	Metric_ = incoming.Metric_;
	Quant_ = incoming.Quant_;
	Retain_ = incoming.Retain_;
	ChannelCount_ = incoming.ChannelCount_;
	for (int32_t c = 0; c < incoming.ChannelCount_; ++c)
	{
		Channels_[c] = incoming.Channels_[c];
	}
	ChannelInvNorms_ = incoming.ChannelInvNorms_; // points into the adopted arena
	PublishedCount_.store(header.count, std::memory_order_release);
	TombstonedCount_.store(tombstoned, std::memory_order_relaxed);
	Generation_.store(priorGeneration + 1, std::memory_order_release);
	incoming.Arena_ = nullptr; // ownership moved; incoming's dtor now no-ops
	return Status::Ok;
}

const float* ScratchBank::RetainedRow(int32_t index) const
{
	if (!Retain_ || index < 0 || index >= PublishedCount_.load(std::memory_order_acquire))
	{
		return nullptr;
	}
	return Retained_ + static_cast<int64_t>(index) * Dims_;
}

Status ScratchBank::MeasureScratchRecall(
	Workspace& workspace, ScratchRecallReport* outReport, uint64_t seed)
{
	if (!IsCreated() || outReport == nullptr)
	{
		return Status::InvalidArgument;
	}
	// Reject-over-degrade: a bank that retained no floats has no reference to scan, so
	// the audit is a defined InvalidArgument, never a guessed number.
	if (!Retain_)
	{
		return Status::InvalidArgument;
	}
	// Runs under the reader pin like any query flight: the sweep issues many queries
	// over the arena, and the pin holds off an exclusive Grow/Load for its duration.
	while (!TryPinReader())
	{
		std::this_thread::yield();
	}
	const Status status = MeasureRecallLocked(seed, workspace, outReport);
	UnpinReader();
	return status;
}

Status ScratchBank::MeasureRecallLocked(
	uint64_t seed, Workspace& ws, ScratchRecallReport* out) const
{
	const int32_t count = PublishedCount_.load(std::memory_order_acquire);
	const int32_t tombstoned = TombstonedCount_.load(std::memory_order_relaxed);
	const int32_t liveRows = count - tombstoned;

	// Fill the descriptive fields first, so even a degenerate (too-small) bank returns
	// a fully-formed report rather than a bare status.
	out->recall = 0.0f;
	out->sampleCount = liveRows < 1000 ? liveRows : 1000;
	out->liveRows = liveRows;
	out->seed = seed;
	out->generation = Generation();
	out->informative = liveRows >= kRecallInformativeRows;
	// k = min(10, liveRows-1): below kRecallMinRows this is a recall@(liveRows-1), and
	// the report's k field says so.
	const int32_t k = liveRows - 1 < 10 ? (liveRows - 1) : 10;
	out->k = k < 0 ? 0 : k;
	if (k <= 0)
	{
		// Fewer than two live rows: no self-query has a neighbour to retrieve. Defined
		// as recall 0 over an uninformative sample, not an error.
		return Status::Ok;
	}

	// The quantized snapshot the audit scores against.
	BankView view;
	view.rows = Rows_;
	view.scales = Scales_;
	view.count = count;
	view.dims = Dims_;
	view.paddedDims = PaddedDims_;
	view.quant = Quant_;
	view.metric = Metric_;

	// Workspace scratch — reserved once, reused every sample, so a warm sweep allocates
	// nothing: the padded self-query, the exclusion words (tombstones plus the self
	// row), and the Query top-k heap.
	if (!ws.ReserveQueryScratch(PaddedDims_, 1) || !ws.ReserveBiasBits(count) ||
		!ws.Reserve(k, 1))
	{
		return Status::OutOfMemory;
	}
	float* query = ws.QueryScratch(0);
	uint32_t* exclude = ws.BiasBits();
	const int32_t tombWords = TombstoneWords(count);
	for (int32_t w = 0; w < tombWords; ++w)
	{
		exclude[w] = Tombstones_[w].load(std::memory_order_relaxed);
	}

	// int8 banks score the reference cross-device (the honest, machine-independent
	// number); an f32 retention bank has no cross-device mode, so its own exact
	// per-device scan is the reference (a defined, if trivially high, number).
	QueryParams params;
	params.k = k;
	params.excludeBits = exclude;
	params.exactness = Quant_ == Quantization::Int8 ? Exactness::CrossDevice
													: Exactness::PerDevice;

	RecallRng rng(seed);
	const int32_t sampleTarget = out->sampleCount;
	int32_t selected = 0;
	int32_t seen = 0;
	int64_t totalHits = 0;
	int64_t totalPossible = 0;
	// Sequential sampling (Knuth Algorithm S): one index-order pass selects exactly
	// sampleTarget of the liveRows live rows, distinct, deterministic given the seed.
	// When sampleTarget == liveRows every live row is drawn (the exact number).
	RecallTopK refHits;
	refHits.Init(k);
	for (int32_t si = 0; si < count && selected < sampleTarget; ++si)
	{
		const bool dead =
			(Tombstones_[si >> 5].load(std::memory_order_relaxed) & (1u << (si & 31))) != 0;
		if (dead)
		{
			continue;
		}
		const int32_t remaining = liveRows - seen;
		const int32_t needed = sampleTarget - selected;
		++seen;
		if (rng.Unit() * remaining >= needed)
		{
			continue; // not drawn this pass
		}
		++selected;

		// The self-query: the retained (post-normalization) float row, padded.
		const float* self = Retained_ + static_cast<int64_t>(si) * Dims_;
		std::memcpy(query, self, static_cast<size_t>(Dims_) * sizeof(float));
		for (int32_t d = Dims_; d < PaddedDims_; ++d)
		{
			query[d] = 0.0f;
		}

		// Reference top-k: a double-precision scan of the retained float rows, the self
		// row excluded, ranked in the library's total order.
		refHits.Clear();
		for (int32_t j = 0; j < count; ++j)
		{
			if (j == si)
			{
				continue;
			}
			if ((Tombstones_[j >> 5].load(std::memory_order_relaxed) & (1u << (j & 31))) != 0)
			{
				continue;
			}
			const float* rrow = Retained_ + static_cast<int64_t>(j) * Dims_;
			double score = 0.0;
			if (Metric_ == Metric::L2)
			{
				for (int32_t d = 0; d < Dims_; ++d)
				{
					const double diff = static_cast<double>(self[d]) - rrow[d];
					score += diff * diff;
				}
			}
			else
			{
				for (int32_t d = 0; d < Dims_; ++d)
				{
					score += static_cast<double>(self[d]) * rrow[d];
				}
			}
			refHits.Insert(j, score, Metric_ == Metric::L2);
		}

		// Measured top-k: the bank's own quantized scan over the snapshot, the self row
		// excluded alongside the tombstones.
		exclude[si >> 5] |= 1u << (si & 31);
		Hit heap[RecallTopK::kMax]; // k <= 10; separate from the workspace's own heap
		int32_t got = 0;
		const Status qs = Query(view, query, params, ws, heap, &got);
		exclude[si >> 5] &= ~(1u << (si & 31));
		if (qs != Status::Ok)
		{
			return qs;
		}

		// recall@k: how many of the reference top-k the quantized scan also returned.
		for (int32_t i = 0; i < got; ++i)
		{
			if (refHits.Contains(heap[i].index))
			{
				++totalHits;
			}
		}
		totalPossible += refHits.Count();
	}

	out->recall = totalPossible > 0
		? static_cast<float>(static_cast<double>(totalHits) / static_cast<double>(totalPossible))
		: 0.0f;
	return Status::Ok;
}

Status ScratchBank::MeasureRecallLockedChannel(
	uint64_t seed, Workspace& ws, const ChannelInfo& channel, ScratchRecallReport* out) const
{
	const int32_t count = PublishedCount_.load(std::memory_order_acquire);
	const int32_t tombstoned = TombstonedCount_.load(std::memory_order_relaxed);
	const int32_t liveRows = count - tombstoned;

	// Descriptive fields identical to the whole-vector routine — only `recall` is
	// channel-specific; liveRows/k/seed/generation/informative describe the bank.
	out->recall = 0.0f;
	out->sampleCount = liveRows < 1000 ? liveRows : 1000;
	out->liveRows = liveRows;
	out->seed = seed;
	out->generation = Generation();
	out->informative = liveRows >= kRecallInformativeRows;
	const int32_t k = liveRows - 1 < 10 ? (liveRows - 1) : 10;
	out->k = k < 0 ? 0 : k;
	if (k <= 0)
	{
		return Status::Ok;
	}

	// The quantized snapshot, carrying the channel table so a single channel-aligned segment
	// scores per-channel exactly as a live channel query does.
	BankView view;
	view.rows = Rows_;
	view.scales = Scales_;
	view.count = count;
	view.dims = Dims_;
	view.paddedDims = PaddedDims_;
	view.quant = Quant_;
	view.metric = Metric_;
	view.channels = Channels_;
	view.channelCount = ChannelCount_;
	view.channelInvNorms = ChannelInvNorms_;

	if (!ws.ReserveQueryScratch(PaddedDims_, 1) || !ws.ReserveBiasBits(count) ||
		!ws.Reserve(k, 1))
	{
		return Status::OutOfMemory;
	}
	float* query = ws.QueryScratch(0);
	uint32_t* exclude = ws.BiasBits();
	const int32_t tombWords = TombstoneWords(count);
	for (int32_t w = 0; w < tombWords; ++w)
	{
		exclude[w] = Tombstones_[w].load(std::memory_order_relaxed);
	}

	const QuerySegment seg[1] = {{channel.offset, channel.length, 1.0f}};
	QueryParams params;
	params.k = k;
	params.segments = seg;
	params.segmentCount = 1;
	params.excludeBits = exclude;
	params.exactness = Quant_ == Quantization::Int8 ? Exactness::CrossDevice
													: Exactness::PerDevice;

	// The reference scan runs over the retained (dims-wide) float rows, so the channel range
	// clamps at Dims_ (pad lanes are zero on both sides and contribute nothing).
	const int32_t lo = channel.offset;
	const int32_t hi = channel.offset + channel.length < Dims_ ? channel.offset + channel.length
															   : Dims_;

	RecallRng rng(seed);
	const int32_t sampleTarget = out->sampleCount;
	int32_t selected = 0;
	int32_t seen = 0;
	int64_t totalHits = 0;
	int64_t totalPossible = 0;
	RecallTopK refHits;
	refHits.Init(k);
	for (int32_t si = 0; si < count && selected < sampleTarget; ++si)
	{
		const bool dead =
			(Tombstones_[si >> 5].load(std::memory_order_relaxed) & (1u << (si & 31))) != 0;
		if (dead)
		{
			continue;
		}
		const int32_t remaining = liveRows - seen;
		const int32_t needed = sampleTarget - selected;
		++seen;
		if (rng.Unit() * remaining >= needed)
		{
			continue;
		}
		++selected;

		const float* self = Retained_ + static_cast<int64_t>(si) * Dims_;
		std::memcpy(query, self, static_cast<size_t>(Dims_) * sizeof(float));
		for (int32_t d = Dims_; d < PaddedDims_; ++d)
		{
			query[d] = 0.0f;
		}

		// Reference top-k over the channel sub-range: per-channel cosine (dot over the
		// sub-vector norm — the D-V2-1 contract), dot, or squared-L2, matching how the
		// segmented kernel scores the same channel-aligned segment.
		refHits.Clear();
		for (int32_t j = 0; j < count; ++j)
		{
			if (j == si)
			{
				continue;
			}
			if ((Tombstones_[j >> 5].load(std::memory_order_relaxed) & (1u << (j & 31))) != 0)
			{
				continue;
			}
			const float* rrow = Retained_ + static_cast<int64_t>(j) * Dims_;
			double score = 0.0;
			if (Metric_ == Metric::L2)
			{
				for (int32_t d = lo; d < hi; ++d)
				{
					const double diff = static_cast<double>(self[d]) - rrow[d];
					score += diff * diff;
				}
			}
			else if (Metric_ == Metric::Cosine)
			{
				double dot = 0.0, subNorm = 0.0;
				for (int32_t d = lo; d < hi; ++d)
				{
					dot += static_cast<double>(self[d]) * rrow[d];
					subNorm += static_cast<double>(rrow[d]) * rrow[d];
				}
				score = subNorm > 0.0 ? dot / std::sqrt(subNorm) : 0.0;
			}
			else
			{
				for (int32_t d = lo; d < hi; ++d)
				{
					score += static_cast<double>(self[d]) * rrow[d];
				}
			}
			refHits.Insert(j, score, Metric_ == Metric::L2);
		}

		exclude[si >> 5] |= 1u << (si & 31);
		Hit heap[RecallTopK::kMax];
		int32_t got = 0;
		const Status qs = Query(view, query, params, ws, heap, &got);
		exclude[si >> 5] &= ~(1u << (si & 31));
		if (qs == Status::ZeroNormQuery)
		{
			// This is a VALID row (whole-row Cosine normalization succeeded) whose sub-vector
			// in THIS channel is exactly zero, so the per-channel self-query is a ZeroNormQuery
			// (the query-side law, C-5). Exclude it from the sample rather than abort the whole
			// audit; the reservoir budget refills from scorable rows. The API permits such rows.
			--selected;
			continue;
		}
		if (qs != Status::Ok)
		{
			return qs;
		}

		for (int32_t i = 0; i < got; ++i)
		{
			if (refHits.Contains(heap[i].index))
			{
				++totalHits;
			}
		}
		totalPossible += refHits.Count();
	}

	// Report the samples that actually scored: zero-energy-channel rows excluded above are
	// not part of this channel's sample (honest metadata; equals the pre-set target when no
	// row is degenerate in this channel).
	out->sampleCount = selected;
	out->recall = totalPossible > 0
		? static_cast<float>(static_cast<double>(totalHits) / static_cast<double>(totalPossible))
		: 0.0f;
	return Status::Ok;
}

} // namespace superfaiss

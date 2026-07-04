#include "superfaiss/scratch.h"

#include <cassert>
#include <cstring>
#include <new>

#include "superfaiss/bake.h"
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

	constexpr uint32_t kScratchMagic = 0x42535346u; // "FSSB" little-endian
	constexpr uint32_t kScratchVersion = 1;

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
	int32_t capacity, int32_t dims, int32_t paddedDims, Quantization quant)
{
	// One allocation: rows, scales (int8 only), tombstone words, one staging row.
	// Rows come first (kAlignment-aligned block, row stride is a multiple of
	// kAlignment); every later region needs only 4-byte alignment and each region's
	// size is a multiple of 4, so the packing preserves it.
	int64_t bytes = static_cast<int64_t>(capacity) * paddedDims * ElementSize(quant);
	bytes = (bytes + 15) & ~int64_t{15};
	if (quant == Quantization::Int8)
	{
		bytes += static_cast<int64_t>(capacity) * sizeof(float);
	}
	bytes += static_cast<int64_t>(TombstoneWords(capacity)) * sizeof(uint32_t);
	bytes += static_cast<int64_t>(dims) * sizeof(float);
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
	for (int32_t w = 0; w < TombstoneWords(capacity); ++w)
	{
		::new (static_cast<void*>(Tombstones_ + w)) std::atomic<uint32_t>(0u);
	}
}

Status ScratchBank::Create(
	int32_t capacity, int32_t dims, Metric metric, Quantization quant, const Allocator& allocator)
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
	const int64_t bytes = ArenaBytes(capacity, dims, paddedDims, quant);
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
	BindArena(arena, capacity);
	PublishedCount_.store(0, std::memory_order_release);
	TombstonedCount_.store(0, std::memory_order_relaxed);
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
	Capacity_ = 0;
	PublishedCount_.store(0, std::memory_order_relaxed);
	TombstonedCount_.store(0, std::memory_order_relaxed);
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

	// Row fully written; only now does it exist for readers.
	PublishedCount_.store(index + 1, std::memory_order_release);
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
	const int64_t bytes = ArenaBytes(newCapacity, Dims_, PaddedDims_, Quant_);
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
	uint8_t* oldArena = Arena_;

	BindArena(arena, newCapacity);

	// Index-preserving by construction (T-044 W4): rows, scales, and tombstones
	// copy straight across; nothing compacts, nothing renumbers.
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

	detail::SeamFree(Allocator_, oldArena);
	return Status::Ok;
}

Status ScratchBank::Freeze(void* outRows, float* outScales, int32_t* outIndexMap) const
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

Status ScratchBank::Save(const ScratchArchive& archive) const
{
	if (!IsCreated() || archive.write == nullptr)
	{
		return Status::InvalidArgument;
	}
	WriterGuard guard(WriterBusy_);

	ScratchHeader header;
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
	return Status::Ok;
}

Status ScratchBank::Load(const ScratchArchive& archive, const Allocator& allocator)
{
	if (archive.read == nullptr)
	{
		return Status::InvalidArgument;
	}

	ScratchHeader header;
	if (!archive.read(archive.user, &header, sizeof(header)))
	{
		return Status::BadFormat;
	}
	if (header.magic != kScratchMagic || header.version != kScratchVersion)
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

	// Build the incoming state in a fresh bank so a bad archive leaves this one
	// unchanged (reject-over-degrade).
	ScratchBank incoming;
	const Status created = incoming.Create(header.capacity, header.dims,
		static_cast<Metric>(header.metric), static_cast<Quantization>(header.quant), allocator);
	if (created != Status::Ok)
	{
		return created;
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

	// The archive is not trusted: re-validate content with the bake path's rules.
	incoming.PublishedCount_.store(header.count, std::memory_order_release);
	incoming.TombstonedCount_.store(tombstoned, std::memory_order_relaxed);
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

	// Validated: adopt the incoming arena wholesale (its region pointers included —
	// re-binding would re-zero the tombstones just loaded). Exclusive by contract,
	// so the swap is not observed mid-flight.
	Destroy();
	Allocator_ = incoming.Allocator_;
	Arena_ = incoming.Arena_;
	Rows_ = incoming.Rows_;
	Scales_ = incoming.Scales_;
	Tombstones_ = incoming.Tombstones_;
	Staging_ = incoming.Staging_;
	Capacity_ = incoming.Capacity_;
	Dims_ = incoming.Dims_;
	PaddedDims_ = incoming.PaddedDims_;
	Metric_ = incoming.Metric_;
	Quant_ = incoming.Quant_;
	PublishedCount_.store(header.count, std::memory_order_release);
	TombstonedCount_.store(tombstoned, std::memory_order_relaxed);
	incoming.Arena_ = nullptr; // ownership moved; incoming's dtor now no-ops
	return Status::Ok;
}

} // namespace superfaiss

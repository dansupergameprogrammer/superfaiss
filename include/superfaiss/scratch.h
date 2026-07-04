#pragma once

#include <atomic>

#include "types.h"
#include "alloc.h"

namespace superfaiss
{

// Scratch banks (V2 plan section 7): a fixed-capacity, append-friendly bank whose
// snapshot IS a BankView — every query path (V1 flat scan, V2 segmented scan)
// works on scratch content with zero new query code.
//
// Concurrency model — single writer, epoch reads:
//   - One logical writer at a time. Append/Remove/Grow/Save/Load are writer-side;
//     coordinating writers is the caller's job (a debug-build owner guard asserts it).
//   - Readers are lock-free and wait-free: Snapshot() acquires the published count
//     once; every row below it is immutable forever after (append-only arena), so
//     the view is stable without locks. Rows publish only via the count — a scan
//     never observes a partially-written row.
//   - Removal is snapshot-consistent, not mid-scan-preemptive: tombstones are
//     visible to snapshots taken after the Remove; a scan already in flight over an
//     older snapshot may still return the removed row. Deletion is exclusion — the
//     caller ORs the snapshot's tombstone words into the query's excludeBits.
//   - Grow and Load are EXCLUSIVE: no readers, no other writer. The core documents
//     the precondition; a host's pin mechanism is what enforces it in practice.
//
// Index stability is the consumer contract (T-044 W4): Grow preserves row indices —
// it copies rows and tombstones and never compacts; a saved index means the same row
// before and after. Freeze() renumbers (compaction drops tombstoned rows) and returns
// the old->new map so consumers remap stored handles as part of the freeze.

// Archive seam for scratch persistence. The caller owns the medium (file, save-game
// blob, network); the bank owns the format. Both callbacks return false on failure;
// only the direction in use need be non-null.
struct ScratchArchive
{
	bool (*write)(void* user, const void* data, size_t bytes) = nullptr;
	bool (*read)(void* user, void* data, size_t bytes) = nullptr;
	void* user = nullptr;
};

class ScratchBank
{
public:
	ScratchBank() = default;
	~ScratchBank();
	ScratchBank(const ScratchBank&) = delete;
	ScratchBank& operator=(const ScratchBank&) = delete;

	// Allocates the arena (rows + scales + tombstones + staging) in ONE allocation
	// through the seam — zero steady-state allocation after Create. Capacity is the
	// caller's memory budget made explicit. Not valid to call on a created bank.
	Status Create(
		int32_t capacity,
		int32_t dims,
		Metric metric,
		Quantization quant,
		const Allocator& allocator = DefaultAllocator());

	// Releases the arena. Exclusive (no readers in flight). Safe on an empty bank.
	void Destroy();

	bool IsCreated() const { return Arena_ != nullptr; }

	// --- Writer side ---

	// Validates like the importer (finite, dims match); Cosine banks normalize on
	// append and reject zero-norm rows (ZeroNormRow); int8 banks quantize the row
	// standalone (V2-G6, per-row symmetric scale). The row is written, THEN the
	// published count advances with a store-release. Returns OutOfMemory when the
	// capacity budget is exhausted — Grow() to proceed.
	Status Append(const float* row, int32_t dims, int32_t* outIndex);

	// Sets the row's tombstone bit atomically (T-044 W5: fetch_or — a plain bitset
	// write concurrent with snapshot reads is a C++ data race). Idempotent.
	Status Remove(int32_t index);

	// Index-preserving reallocation (T-044 W4). EXCLUSIVE: no readers, no snapshots
	// in flight — a host waits on its pin counter before calling. newCapacity must
	// exceed the current capacity.
	Status Grow(int32_t newCapacity);

	// --- Reader-pin / exclusive-drain protocol (Poirot F4) ---
	//
	// The dispatch gate hosts put in front of Grow/Load (T-044 N4): readers pin
	// for a query flight; an exclusive operation raises the drain flag (new pins
	// refused), waits the pins out, runs, releases. The critical pairs - the
	// exclusive side's flag-store + pin-count-load, and the reader side's
	// pin-increment + flag-re-load - are seq_cst BY DESIGN: with acquire/release
	// alone this is a store-buffering shape, and the interleaving "writer sees
	// pins==0, reader sees flag==false" is permitted on weakly-ordered ISAs
	// (ARM64) even though x86's lock-prefixed RMWs happen to forbid it. The
	// protocol's safety must be legible in its own orderings, not borrowed from
	// an ISA. Exclusive ops are rare (Grow/Freeze/Load); the seq_cst cost is
	// noise. The T22 litmus test drives this protocol under ThreadSanitizer in CI.

	// Pins the bank for one reader flight. Fails while an exclusive operation is
	// waiting or running. Any thread.
	bool TryPinReader();
	void UnpinReader();

	// Raises the drain flag and waits out in-flight reader pins. Fails if another
	// exclusive operation holds the flag (writer coordination is the caller's
	// job). After it returns true, run the exclusive work, then EndExclusive().
	bool BeginExclusive();
	void EndExclusive();

	// --- Reader side (lock-free) ---

	// Published row count (acquire). Rows below this are immutable.
	int32_t Count() const { return PublishedCount_.load(std::memory_order_acquire); }

	// Published rows minus tombstoned rows.
	int32_t LiveCount() const
	{
		return Count() - TombstonedCount_.load(std::memory_order_relaxed);
	}

	int32_t Capacity() const { return Capacity_; }
	int32_t Dims() const { return Dims_; }
	int32_t GetPaddedDims() const { return PaddedDims_; }
	Metric GetMetric() const { return Metric_; }
	Quantization GetQuantization() const { return Quant_; }

	// Exclusion-word count for a snapshot of `count` rows (QueryParams::excludeBits
	// sizing: ceil(count/32)).
	static int32_t TombstoneWords(int32_t count) { return (count + 31) / 32; }

	// Fills outView over rows [0, published count) and copies the tombstone words
	// (atomic loads, relaxed — visibility is ordered through the published count)
	// into outTombstones, TombstoneWords(outView->count) uint32 words. The caller
	// ORs those words into (or uses them as) the query's excludeBits: deletion is
	// exclusion. The view is valid until the next Grow/Load or Destroy. Bits at or
	// above the snapshot count may appear set in the last word (a row published and
	// removed after this snapshot's count was taken); queries never read them.
	Status Snapshot(BankView* outView, uint32_t* outTombstones) const;

	// --- Freeze: graduate accumulated memory to an immutable bank ---

	// Rows that survive compaction (published minus tombstoned) at this moment.
	int32_t FreezeLiveCount() const { return LiveCount(); }

	// Compacts live rows into caller-provided buffers, standard baked-bank layout:
	// outRows is FreezeLiveCount() x paddedDims elements (kAlignment-aligned);
	// outScales is FreezeLiveCount() floats for int8 banks (null for float32);
	// outIndexMap, if non-null, receives Count() entries — old index -> compacted
	// index, -1 for tombstoned rows. Freeze renumbers (W4): consumers remap their
	// stored handles through the map as part of the freeze. Writer-side.
	Status Freeze(void* outRows, float* outScales, int32_t* outIndexMap) const;

	// --- Persistence (archive seam) ---

	// Writes the full state (header, rows, scales, tombstones). Runs as the writer,
	// under the single-writer guard (T-044 N1).
	Status Save(const ScratchArchive& archive) const;

	// Recreates the bank from an archive. EXCLUSIVE — no readers, no writer; an
	// existing arena is replaced only after the payload validates (reject-over-
	// degrade: a bad archive leaves the bank unchanged). Content is re-validated
	// with the same rules the bake path enforces — the archive is not trusted.
	Status Load(const ScratchArchive& archive, const Allocator& allocator = DefaultAllocator());

private:
	Status AppendValidated(const float* row, int32_t* outIndex);
	int64_t RowRegionBytes(int32_t capacity) const;
	static int64_t ArenaBytes(
		int32_t capacity, int32_t dims, int32_t paddedDims, Quantization quant);
	void BindArena(uint8_t* arena, int32_t capacity);

	Allocator Allocator_ = DefaultAllocator();
	uint8_t* Arena_ = nullptr;
	void* Rows_ = nullptr;                       // capacity x paddedDims, aligned
	float* Scales_ = nullptr;                    // int8 only
	std::atomic<uint32_t>* Tombstones_ = nullptr;
	float* Staging_ = nullptr;                   // one dims-wide row (normalize scratch)
	int32_t Capacity_ = 0;
	int32_t Dims_ = 0;
	int32_t PaddedDims_ = 0;
	Metric Metric_ = Metric::Dot;
	Quantization Quant_ = Quantization::Float32;
	std::atomic<int32_t> PublishedCount_{0};
	std::atomic<int32_t> ReaderPins_{0};
	std::atomic<bool> ExclusiveWaiting_{false};
	std::atomic<int32_t> TombstonedCount_{0};
	// Debug-build owner guard: asserts the single-writer contract, costs nothing on
	// the reader side.
	mutable std::atomic<bool> WriterBusy_{false};
};

} // namespace superfaiss

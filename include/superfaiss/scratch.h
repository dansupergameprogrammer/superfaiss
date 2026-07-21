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

// Recall-audit report (V2.3 plan section 20). MeasureScratchRecall fills one from a
// seeded self-query sweep on a retention-enabled bank. The number is reproducible from
// `seed`; `generation` stamps the bank's mutation state at measurement, so a later
// append/remove renders the report stale (RecallReportStale). A recall@10 over a tiny
// personal bank is mathematically valid and statistically uninformative — the report
// says so through `informative` (false below kRecallInformativeRows live rows) and,
// below kRecallMinRows, `k` is a recall@(liveRows-1) rather than @10.
struct ScratchRecallReport
{
	float recall = 0.0f;       // hits / possible over the sweep, [0, 1]
	int32_t k = 0;             // min(10, liveRows - 1) — the actual retrieval depth
	int32_t sampleCount = 0;   // min(1000, liveRows) — self-queries drawn
	int32_t liveRows = 0;      // published minus tombstoned at measurement
	uint64_t seed = 0;         // the sampling seed, recorded for reproducibility
	uint64_t generation = 0;   // bank mutation generation at measurement (staleness)
	bool informative = false;  // false when liveRows < kRecallInformativeRows
};

class ScratchBank
{
public:
	ScratchBank() = default;
	~ScratchBank();
	ScratchBank(const ScratchBank&) = delete;
	ScratchBank& operator=(const ScratchBank&) = delete;

	// Default seed for the recall sweep (parity with the importer's pinned seed, so a
	// scratch audit and an import audit over the same rows draw the same self-queries).
	static constexpr uint64_t kDefaultRecallSeed = 0x5EEDF00DCAFEBEEFull;
	// A recall@k below this many live rows is a recall@(liveRows-1), not a true @10.
	static constexpr int32_t kRecallMinRows = 11;
	// Below this many live rows the number is marked statistically uninformative.
	static constexpr int32_t kRecallInformativeRows = 100;

	// Allocates the arena (rows + scales + tombstones + staging) in ONE allocation
	// through the seam — zero steady-state allocation after Create. Capacity is the
	// caller's memory budget made explicit. Not valid to call on a created bank.
	Status Create(
		int32_t capacity,
		int32_t dims,
		Metric metric,
		Quantization quant,
		const Allocator& allocator = DefaultAllocator())
	{
		return Create(capacity, dims, metric, quant, false, allocator);
	}

	// Retention overload (V2.3): when retainFloats is set, the arena additionally
	// retains, beside every quantized row, the post-normalization/post-validation float
	// row the quantizer consumed (the audit reference — for Cosine banks the unit-norm
	// row). Retention is a bank property (RetainsFloats), never the shipping default,
	// sized into the SAME single allocation (zero steady-state allocation preserved).
	// It is what MeasureScratchRecall needs; the honest memory cost is RetainedRowBytes
	// per row (~4.9x the quantized row at int8/256-dim, not 2x — 2x holds only on f32).
	Status Create(
		int32_t capacity,
		int32_t dims,
		Metric metric,
		Quantization quant,
		bool retainFloats,
		const Allocator& allocator = DefaultAllocator());

	// Channel-capable overload (V3.0, plan section 23.4): the channel table
	// becomes a scratch-bank property, set at Create (D-V3-2) and replaceable
	// thereafter only by the exclusive Relabel operation (V3.1).
	// Validated at construction with the same rules ValidateBank applies to a
	// baked channel table (in-bounds, ascending, non-overlapping, on the
	// 16-byte element grid, channelCount in [1, kMaxChannels]) -- validation
	// moves from import-time to construction-time. On a Cosine bank the arena
	// additionally carries a capacity x channelCount per-channel inverse-
	// sub-norm array (computed per-row-standalone at Append, V3-G4), sized
	// into the SAME single allocation (V3-G5); Dot/L2 channel banks need no
	// sub-norm arena. `channels` may be null with `channelCount` 0 for a
	// single-space bank (equivalent to the overloads above). Not valid to
	// call on a created bank.
	Status Create(
		int32_t capacity,
		int32_t dims,
		Metric metric,
		Quantization quant,
		const ChannelInfo* channels,
		int32_t channelCount,
		bool retainFloats = false,
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

	// Mutable channel vocabulary (V3.1, plan section 24.4): atomically replaces the
	// channel table on a live bank, relaxing V3.0's fixed-at-Create vocabulary (D-V3-2).
	// The stored rows are never touched — channels are sub-ranges over the same
	// dims (section 24.3), so a relabel is a vocabulary change, not a row change.
	//
	// The new table is validated with the SAME rules Create/Load apply (in-bounds
	// against PaddedDims_, ascending, non-overlapping, on the 16-byte element grid,
	// length > 0, channelCount in [1, kMaxChannels]); newChannelCount == 0 with a null
	// table is the demote-to-single-space case; a nonzero valid table on any bank is the
	// general case (promote / boundary move / count change). EXCLUSIVE, the same class as
	// Grow/Load: the host drains readers before calling, so no in-flight BankView observes
	// a torn table — a view taken before a relabel joins the Grow/Load/Destroy invalidation
	// set. Reject-over-degrade: a malformed table returns InvalidArgument and an arena
	// realloc failure returns OutOfMemory, and in both cases the table, arena, sub-norms,
	// and generation are exactly as before the call (the Load idiom).
	//
	// Dot/L2 channel banks carry no sub-norm arena, so their relabel is a validate-and-swap
	// of the by-value members with no arena touch (cannot OOM). A Cosine relabel that
	// changes the channel count (or promotes/demotes) reallocates the arena via the Grow
	// template and RE-DERIVES the per-channel inverse sub-norms under the new table (the
	// same ComputeChannelInverseNorms Load already runs), so a relabeled bank is
	// bit-identical to a fresh Create(newTable)+Append of the same rows. Advances
	// Generation() (a mutation) so a pre-relabel recall report reads stale.
	Status Relabel(const ChannelInfo* newChannels, int32_t newChannelCount);

	// --- Reader-pin / exclusive-drain protocol ---
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

	// True if the bank retains post-normalization float rows (set at Create). The
	// descriptor field DescribeBank surfaces; the gate MeasureScratchRecall requires.
	bool RetainsFloats() const { return Retain_; }

	// Monotonic mutation generation: advances on every successful Append and every
	// newly-set Remove (an idempotent re-Remove does not advance). A recall report
	// stamps this; a stamp that no longer matches means the report is stale.
	uint64_t Generation() const { return Generation_.load(std::memory_order_acquire); }

	// The retained post-normalization float row at `index` (dims elements), or null if
	// the bank does not retain or the index is out of the published range. This is the
	// row the quantizer consumed: for Cosine banks the unit-norm row, for Dot/L2 the
	// finite-validated input.
	const float* RetainedRow(int32_t index) const;

	// Per-row byte costs for the honest-budget statement (V2.3 plan section 20, B1):
	// the quantized row (padded int8/f32 lanes plus the 4-byte int8 scale) and the
	// retained float row (dims x 4, or 0 when retention is off).
	int64_t QuantizedRowBytes() const
	{
		return static_cast<int64_t>(PaddedDims_) * ElementSize(Quant_) +
			(Quant_ == Quantization::Int8 ? static_cast<int64_t>(sizeof(float)) : 0);
	}
	int64_t RetainedRowBytes() const
	{
		return Retain_ ? static_cast<int64_t>(Dims_) * sizeof(float) : 0;
	}

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
	//
	// Recall re-measurement (V2.3): when outReport and recallWs are both non-null and
	// the bank retains floats, the freeze measures recall over the compacted (live)
	// rows at freeze time and writes it to *outReport — a fresh number for the
	// graduated bank, never a stale one wearing a current face. The number INV-equals a
	// direct MeasureScratchRecall of the same rows on the same seed. A non-retention
	// Freeze leaves *outReport untouched: it produces no number to inherit.
	Status Freeze(void* outRows, float* outScales, int32_t* outIndexMap,
		ScratchRecallReport* outReport = nullptr, Workspace* recallWs = nullptr,
		uint64_t recallSeed = kDefaultRecallSeed) const;

	// Channel-aware Freeze (V3.0, plan section 23.4): as Freeze above (compacts live
	// rows into outRows/outScales, fills outIndexMap old->new), and ADDITIONALLY emits
	// the per-channel inverse sub-norms RE-DERIVED over the compacted (live) rows into
	// outChannelInvNorms (FreezeLiveCount() * GetChannelCount() floats). Compaction
	// renumbers, so the graduated channel-carrying baked payload's sub-norms follow the
	// surviving rows (exactly as recall is re-measured at freeze). Valid only on a Cosine
	// bank that carries a channel table; on a non-channel or non-Cosine bank it is
	// InvalidArgument (use the base Freeze). The channel table itself is read back via
	// GetChannels()/GetChannelCount() for building the graduated BankView.
	Status Freeze(void* outRows, float* outScales, int32_t* outIndexMap,
		float* outChannelInvNorms, ScratchRecallReport* outReport = nullptr,
		Workspace* recallWs = nullptr, uint64_t recallSeed = kDefaultRecallSeed) const;

	// The fixed channel table set at Create (V3.0), or null / 0 for a single-space bank.
	// A caller graduating a channel scratch bank attaches these to the frozen BankView.
	int32_t GetChannelCount() const { return ChannelCount_; }
	const ChannelInfo* GetChannels() const { return ChannelCount_ > 0 ? Channels_ : nullptr; }

	// Channel-aware Freeze that also re-measures per-channel recall over the compacted
	// (live) rows at freeze time (V3.0, D-V3-7). As the channel-aware Freeze above, and
	// additionally fills outRecallReports[c] (reportCount == GetChannelCount() entries)
	// with the per-channel recall@k measured over the surviving rows -- a fresh number
	// for the graduated bank, never a stale one (the §20 freeze-re-measure rule, per
	// channel). Requires a retention-enabled Cosine bank that carries a channel table and
	// a workspace; InvalidArgument otherwise (a non-retention bank has no float reference
	// to scan; use the base channel-aware Freeze for the no-recall path).
	Status FreezeWithRecall(void* outRows, float* outScales, int32_t* outIndexMap,
		float* outChannelInvNorms, ScratchRecallReport* outRecallReports, int32_t reportCount,
		Workspace& recallWs, uint64_t recallSeed = kDefaultRecallSeed) const;

	// --- Recall audit (V2.3 plan section 20) ---

	// Seeded recall@k self-query sweep on a retention-enabled bank: min(1000, liveRows)
	// live rows are drawn as self-queries (the self row excluded), each scored two ways
	// over the current snapshot — a double-precision float scan of the retained rows
	// (the reference top-k) versus the bank's own quantized Exactness::CrossDevice scan
	// — and recall is hits/possible at k = min(10, liveRows-1). Runs under the reader
	// pin like any query; the workspace supplies all scratch, so a warm call allocates
	// nothing. On a bank without retention this is a defined InvalidArgument, never a
	// guessed number. Tombstoned rows are neither sampled nor counted.
	// Quiescence precondition (review M3): the number is well-defined when no writer
	// runs concurrently. The sweep holds a reader pin, not exclusivity, so a racing
	// Append/Remove yields a SAFE but non-reproducible number (atomic value reads,
	// never UB) — determinism-given-history holds only over a quiescent bank.
	Status MeasureScratchRecall(Workspace& workspace, ScratchRecallReport* outReport,
		uint64_t seed = kDefaultRecallSeed);

	// Per-channel recall mode (V3.0, D-V3-7): on a retention-enabled Cosine bank that
	// carries a channel table, measures recall@k PER CHANNEL over each channel's
	// sub-range (the channel-scoped self-query vs the double-precision reference
	// restricted to that channel's elements), filling outReports[c] for c in
	// [0, GetChannelCount()). reportCount must equal GetChannelCount(). Runs under the
	// reader pin like any query; the workspace supplies all scratch. InvalidArgument on a
	// non-retention bank (no float reference) or a bank with no channel table (use the
	// whole-vector MeasureScratchRecall). Tombstoned rows are neither sampled nor counted,
	// exactly as the whole-vector routine.
	Status MeasureScratchRecallPerChannel(Workspace& workspace,
		ScratchRecallReport* outReports, int32_t reportCount,
		uint64_t seed = kDefaultRecallSeed);

	// A report is stale once the bank has mutated past its stamped generation. Defined
	// against the live bank: never reads a post-mutation report as silently current.
	bool RecallReportStale(const ScratchRecallReport& report) const
	{
		return report.generation != Generation();
	}

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
	// subNormPerRow is the per-channel inverse-sub-norm count stored per row (the channel
	// count on a Cosine channel bank, else 0); it sizes the sub-norm region into the same
	// single allocation (V3-G5). Dot/L2 channel banks carry no sub-norm region.
	static int64_t ArenaBytes(
		int32_t capacity, int32_t dims, int32_t paddedDims, Quantization quant, bool retain,
		int32_t subNormPerRow);
	void BindArena(uint8_t* arena, int32_t capacity);
	// The recall sweep proper — assumes the arena is stable (caller holds the reader pin
	// or the writer guard). Const: it only reads rows, retained floats, and tombstones.
	Status MeasureRecallLocked(uint64_t seed, Workspace& ws, ScratchRecallReport* out) const;
	// The per-channel recall sweep (V3.0): as MeasureRecallLocked but the self-query and
	// the reference scan are both restricted to `channel`'s element sub-range.
	Status MeasureRecallLockedChannel(
		uint64_t seed, Workspace& ws, const ChannelInfo& channel, ScratchRecallReport* out) const;
	// Compaction shared by the channel-aware Freeze paths: copies live rows/scales into the
	// caller buffers, fills outIndexMap (old->new, -1 for tombstoned), and re-derives the
	// per-channel inverse sub-norms over the compacted rows. Assumes a validated Cosine
	// channel bank and that the caller holds the writer guard.
	Status FreezeChannelsLocked(
		void* outRows, float* outScales, int32_t* outIndexMap, float* outChannelInvNorms) const;

	Allocator Allocator_ = DefaultAllocator();
	uint8_t* Arena_ = nullptr;
	void* Rows_ = nullptr;                       // capacity x paddedDims, aligned
	float* Scales_ = nullptr;                    // int8 only
	std::atomic<uint32_t>* Tombstones_ = nullptr;
	float* Staging_ = nullptr;                   // one dims-wide row (normalize scratch)
	float* Retained_ = nullptr;                  // capacity x dims, retention only
	float* ChannelInvNorms_ = nullptr;           // capacity x channelCount, Cosine+channels only
	int32_t Capacity_ = 0;
	int32_t Dims_ = 0;
	int32_t PaddedDims_ = 0;
	Metric Metric_ = Metric::Dot;
	Quantization Quant_ = Quantization::Float32;
	bool Retain_ = false;
	// Channel table (V3.0, D-V3-2; mutable since V3.1): set at Create, replaced
	// atomically only by Relabel. Empty
	// (ChannelCount_ == 0) for a single-space bank. Held by value — small (<= kMaxChannels)
	// and survives Grow/Load without an arena region of its own; the per-channel inverse
	// sub-norms (ChannelInvNorms_) live in the arena.
	int32_t ChannelCount_ = 0;
	ChannelInfo Channels_[kMaxChannels] = {};
	std::atomic<int32_t> PublishedCount_{0};
	std::atomic<int32_t> ReaderPins_{0};
	std::atomic<bool> ExclusiveWaiting_{false};
	std::atomic<int32_t> TombstonedCount_{0};
	std::atomic<uint64_t> Generation_{0};
	// Debug-build owner guard: asserts the single-writer contract, costs nothing on
	// the reader side.
	mutable std::atomic<bool> WriterBusy_{false};
};

} // namespace superfaiss

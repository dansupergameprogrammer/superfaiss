# API Reference

Everything lives in `namespace superfaiss`; include `superfaiss/superfaiss.h` (or the
individual headers named below). No exceptions are thrown; failures return `Status`.
All functions are single-threaded — threading belongs to the host (see
[INTEGRATION.md](INTEGRATION.md)).

## types.h — the vocabulary

```cpp
enum class Quantization : uint8_t { Float32, Int8 };
enum class Metric        : uint8_t { Dot, Cosine, L2 };
enum class Status        : uint8_t { Ok, InvalidArgument, DimsMismatch, NonFiniteQuery,
                                     ZeroNormQuery, NonZeroPadding, BadAlignment,
                                     BadFormat, ZeroNormRow, OutOfMemory };

struct BankView {          // non-owning; the library never allocates bank memory
    const void*  rows;     // row-major, paddedDims stride, 16-byte aligned
    const float* scales;   // Int8 only: per-row dequant scale; null for Float32
    int32_t count, dims, paddedDims;
    Quantization quant;
    Metric metric;
    const ChannelInfo* channels;      // v2.0, optional: schemaVersion-2 banks
    int32_t channelCount;
    const float* channelInvNorms;     // Cosine channel banks: count x channelCount
};

struct Hit { int32_t index; float score; };

struct ChannelInfo  { int32_t offset, length; };   // named ranges live host-side (v2.0)
struct QuerySegment { int32_t offset, length; float weight; };  // v2.0, see below

enum class ScoreAs : uint8_t { BankMetric, Dot };  // per-query metric override

struct QueryParams {
    int32_t k;
    const uint32_t* excludeBits;   // bitset, bit = skip row; null = none
    ScoreAs scoreAs;               // Dot forces dot scoring (axis projection on L2 banks)
    const QuerySegment* segments;  // v2.0: null = whole row (the V1 path, bit-identical)
    int32_t segmentCount;          // <= kMaxSegments (8)
};
```

**Segmented queries (v2.0):** a segment is a contiguous element range with a scalar
weight; scores combine additively (`total = sum(weight_s * partial_s)`). A mask is a
range simply omitted; a weight-0 segment equals omission. Ranges lie on the 16-byte
element grid, ascending, non-overlapping. On Cosine banks whose `BankView` carries a
channel table + `channelInvNorms` (baked from the QUANTIZED rows), channel-matched
segments score true per-channel cosines. Semantics feature, not a speed feature: a
segmented scan costs approximately one full scan (dot-family weights fold into the
query and run the plain V1 kernels; L2 takes a dense boundary-switched pass).

Helpers: `PaddedDims(dims, quant)`, `ElementSize(quant)`, `RowBytes(bank)`,
`BankBytes(bank)`, `ChunkRows(bank)`, `ChunkCount(bank)`, `IsExcluded(bits, index)`,
`ScoringMetric(bank, params)` (the metric hits are actually ordered by).
Constants: `kAlignment` (16), `kChunkBytes` (64 KiB), `kMaxSegments` / `kMaxChannels`
(8), `kSchemaVersion` (version.h).

## validate.h — the gates

```cpp
Status ValidateBank(const BankView&);                        // structural; cheap
Status ValidateQuery(const BankView&, const float* query);   // per query-buffer
Status ValidateSegments(const BankView&, const float* query, // v2.0: segment lists
                        const QuerySegment*, int32_t count);
Status ValidateBankData(const BankView&, int32_t* outBadRow);// full content; load-time
```

`ValidateBankData` is O(count × paddedDims): run it **once at load**, not per query. It
rejects everything the kernels cannot tolerate (non-finite lanes, non-zero padding,
bad int8 scales). `ValidateQuery` (and `ValidateSegments` when segments are present -
count/grid/order/bounds, finite weights, and the Cosine zero-norm-sub-vector rule) is
called by the query entry points automatically.

## bake.h — producing banks

```cpp
Status ValidateSourceRows(const float* rows, int32_t count, int32_t dims, int32_t* outBadRow);
Status NormalizeRows(float* rows, int32_t count, int32_t dims, int32_t* outBadRow);
void   PadRowsFloat32 (const float* rows, int32_t count, int32_t dims, int32_t paddedDims, float* out);
void   QuantizeRowsInt8(const float* rows, int32_t count, int32_t dims, int32_t paddedDims,
                        int8_t* outRows, float* outScales);
Status ComputeChannelInverseNorms(const BankView&, float* outInvNorms);  // v2.0
```

Order matters: validate source (a NaN row would otherwise quantize to a silent zero
row) → normalize (Cosine banks) → pad or quantize. Cosine banks carrying channels
then bake `channelInvNorms` from the QUANTIZED payload (the reported per-channel
cosine is the cosine of what the kernel actually dots; a zero-norm row channel
stores 0 and scores 0 - defined, never NaN). See
[FORMAT.md](FORMAT.md) for the layout rules these implement.

## query.h — the simple front door

```cpp
Status Query(const BankView&, const float* paddedQuery, const QueryParams&,
             Workspace&, Hit* outHits, int32_t* outCount);

Status QueryBatch(const BankView&, const float* paddedQueries, int32_t queryCount,
                  const QueryParams&, Workspace&, Hit* outHits, int32_t* outCounts);

Status QueryIntersect(const BankView&, const float* paddedQueries, int32_t queryCount,
                      const QueryParams&, Workspace&, Hit* outHits, int32_t* outCount);
```

Exact top-k, serial chunk loop, allocation-free once the `Workspace` is warm. `Query`
writes up to `k` hits best-first. `QueryBatch` scores all queries per chunk pass
(queries contiguous at `paddedDims` stride; results query-major, `k` slots per query)
using pair kernels that share row loads — per-query results are bit-identical to
`Query`. `QueryIntersect` (v1.1) ranks by each row's WORST score against the member
queries (true AND semantics — every returned row clears the fused score against every
query) in one bank pass; `queryCount == 1` degenerates to `Query` bit-identically.

Per-hit explanation (v2.0, `kernels.h`): `DecomposeRowScore(bank, query, rowIndex,
segments, count, outContributions)` — per-segment contributions that sum bit-exactly
to the score the same query's scan produced for that row (they are the scan's own
accumulators, not a re-computation).

## kernels.h + topk.h — the parallel toolkit

For hosts that want to fan a single query out across threads:

```cpp
void ScoreChunk(const BankView&, const float* paddedQuery, int32_t chunkIndex,
                const uint32_t* excludeBits, TopK& inout);
void ScoreChunkPair(const BankView&, const float* qA, const float* qB, int32_t chunkIndex,
                    const uint32_t* excludeBits, TopK& inoutA, TopK& inoutB);

class TopK {            // bounded top-k over caller storage
    void    Init(Hit* storage, int32_t k, Metric);
    void    Push(int32_t index, float score);
    int32_t Finalize(Hit* out);   // sorts best-first; destroys the heap
};

int32_t MergeTopK(const Hit* const* lists, const int32_t* listCounts, int32_t listCount,
                  Metric, int32_t k, Hit* heapScratch, Hit* out);

// v2.0 segmented chunk scoring (the dense path; dot-family folded queries just
// use ScoreChunk/ScoreChunkPair on the weight-folded query):
void  ScoreChunkSegmented(...);       // one query, segment list
void  ScoreChunkFusedSegmented(...);  // intersection fusion, segment list
float DecomposeRowScore(const BankView&, const float* paddedQuery, int32_t rowIndex,
                        const QuerySegment*, int32_t count, float* outContributions);
```

Pattern: one `TopK` per chunk (storage from your own pool), `ScoreChunk` chunks in
parallel, `Finalize` each, `MergeTopK` the lists — the result is bit-identical to a
serial `Query` in any list order (see [DETERMINISM.md](DETERMINISM.md)). Kernels do not
re-validate; gate inputs first. `ActiveSimdPath()` reports Scalar/SSE/AVX2/NEON.
`detail::*Mirror` functions expose the scalar mirrors for equality testing.

## compose.h — query construction (v1.1)

```cpp
Status MakeCentroid (const BankView&, const int32_t* rowIndices, int32_t rowCount,
                     float* outPaddedQuery);                 // mean of rows, padded
Status MakeDirection(const float* paddedA, const float* paddedB, int32_t dims,
                     int32_t paddedDims, float* outPaddedQuery);  // normalize(a - b)
float  Margin(const Hit& better, const Hit& runnerUp, Metric scoredAs);
```

Pure, allocation-free, deterministic (serial double-precision accumulation).
`MakeCentroid` dequantizes int8 rows through their scales; on Cosine banks the mean
renormalizes, and a zero-norm mean (antipodal members cancelling) is rejected, never
silently renormalized. `MakeDirection` builds the axis-projection query ("most a-like
relative to b"); score it with `ScoreAs::Dot` on L2 banks. `Margin` is the score gap
to the runner-up in the scored metric's better-direction — pass `ScoringMetric()`.

## pca.h — inspection projections (v1.1)

```cpp
Status ComputePrincipalComponents(const BankView&, int32_t componentCount,
                                  int32_t iterationsPerComponent, float* outMean,
                                  float* outComponents, float* scratch);
Status ProjectRowsOntoComponents (const BankView&, const float* mean,
                                  const float* components, int32_t componentCount,
                                  float* outCoords);
```

Covariance-free power iteration with Gram-Schmidt deflation: O(count x dims) per
iteration, no dims^2 matrix, no allocation, deterministic (fixed seed vector, serial
row-order accumulation). The projection-visualizer substrate; a degenerate direction
yields a zero component rather than an error.

## scratch.h — mutable banks (v2.0)

```cpp
class ScratchBank {              // single writer, lock-free readers
    Status Create(int32_t capacity, int32_t dims, Metric, Quantization,
                  const Allocator& = DefaultAllocator());   // ONE arena allocation
    Status Append(const float* row, int32_t dims, int32_t* outIndex);
    Status Remove(int32_t index);                 // atomic tombstone; idempotent
    Status Snapshot(BankView* outView, uint32_t* outTombstones) const;
    Status Grow(int32_t newCapacity);             // EXCLUSIVE; preserves indices
    int32_t FreezeLiveCount() const;
    Status Freeze(void* outRows, float* outScales, int32_t* outIndexMap) const;
    Status Save(const ScratchArchive&) const;     // writer-side
    Status Load(const ScratchArchive&, const Allocator& = ...); // EXCLUSIVE
    int32_t Count(); int32_t LiveCount(); int32_t Capacity();
    static int32_t TombstoneWords(int32_t count);
};
struct ScratchArchive { bool(*write)(void*, const void*, size_t); bool(*read)(void*, void*, size_t); void* user; };
```

A snapshot IS a `BankView` — every query entry point works on it unchanged.
Deletion is exclusion: OR (or pass) the snapshot's tombstone words as
`QueryParams::excludeBits`. Append validates like the importer (finite, dims
match; Cosine normalizes and rejects zero-norm; int8 quantizes per-row), writes
the row, then publishes the count with a store-release — readers never see a
partial row. Removal is snapshot-consistent, not mid-scan-preemptive. `Grow`
preserves indices (saved indices are the consumer contract); `Freeze` compacts
and renumbers, returning the old→new map so consumers remap stored handles.
Zero steady-state allocation: everything lives in the arena from `Create`.
Archive format: [FORMAT.md](FORMAT.md) section 3; concurrency guarantees:
[DETERMINISM.md](DETERMINISM.md) section 2b.

## alloc.h — memory policy

```cpp
struct Allocator { void*(*alloc)(size_t, size_t align, void* user); void(*free)(void*, void*); void* user; };
Allocator DefaultAllocator();
uint64_t  AllocationCount();     // process-wide, monotonic; assert flat deltas in tests

class Workspace {                // query scratch; single owner, NOT thread-safe
    explicit Workspace(const Allocator&);   // or default
    bool     Reserve(int32_t k, int32_t batchWidth);
    uint64_t GrowthCount();      // flat once warm
};
```

One `Workspace` serves one `Query`/`QueryBatch` call at a time; concurrent callers need
one each (pool them). `Reserve` failures surface as `Status::OutOfMemory`.

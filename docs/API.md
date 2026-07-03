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
};

struct Hit         { int32_t index; float score; };
struct QueryParams { int32_t k; const uint32_t* excludeBits; };  // bitset, bit=skip row
```

Helpers: `PaddedDims(dims, quant)`, `ElementSize(quant)`, `RowBytes(bank)`,
`BankBytes(bank)`, `ChunkRows(bank)`, `ChunkCount(bank)`, `IsExcluded(bits, index)`.
Constants: `kAlignment` (16), `kChunkBytes` (64 KiB), `kSchemaVersion` (version.h).

## validate.h — the gates

```cpp
Status ValidateBank(const BankView&);                        // structural; cheap
Status ValidateQuery(const BankView&, const float* query);   // per query-buffer
Status ValidateBankData(const BankView&, int32_t* outBadRow);// full content; load-time
```

`ValidateBankData` is O(count × paddedDims): run it **once at load**, not per query. It
rejects everything the kernels cannot tolerate (non-finite lanes, non-zero padding,
bad int8 scales). `ValidateQuery` is called by `Query`/`QueryBatch` automatically.

## bake.h — producing banks

```cpp
Status ValidateSourceRows(const float* rows, int32_t count, int32_t dims, int32_t* outBadRow);
Status NormalizeRows(float* rows, int32_t count, int32_t dims, int32_t* outBadRow);
void   PadRowsFloat32 (const float* rows, int32_t count, int32_t dims, int32_t paddedDims, float* out);
void   QuantizeRowsInt8(const float* rows, int32_t count, int32_t dims, int32_t paddedDims,
                        int8_t* outRows, float* outScales);
```

Order matters: validate source (a NaN row would otherwise quantize to a silent zero
row) → normalize (Cosine banks) → pad or quantize. See
[FORMAT.md](FORMAT.md) for the layout rules these implement.

## query.h — the simple front door

```cpp
Status Query(const BankView&, const float* paddedQuery, const QueryParams&,
             Workspace&, Hit* outHits, int32_t* outCount);

Status QueryBatch(const BankView&, const float* paddedQueries, int32_t queryCount,
                  const QueryParams&, Workspace&, Hit* outHits, int32_t* outCounts);
```

Exact top-k, serial chunk loop, allocation-free once the `Workspace` is warm. `Query`
writes up to `k` hits best-first. `QueryBatch` scores all queries per chunk pass
(queries contiguous at `paddedDims` stride; results query-major, `k` slots per query)
using pair kernels that share row loads — per-query results are bit-identical to
`Query`.

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
```

Pattern: one `TopK` per chunk (storage from your own pool), `ScoreChunk` chunks in
parallel, `Finalize` each, `MergeTopK` the lists — the result is bit-identical to a
serial `Query` in any list order (see [DETERMINISM.md](DETERMINISM.md)). Kernels do not
re-validate; gate inputs first. `ActiveSimdPath()` reports Scalar/SSE/AVX2/NEON.
`detail::*Mirror` functions expose the scalar mirrors for equality testing.

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

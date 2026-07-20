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
    const RowBias* bias;           // v2.1: per-row score bias; null = none
    Exactness exactness;           // v2.2: PerDevice (default) | CrossDevice
};

enum class Exactness : uint8_t { PerDevice, CrossDevice };  // v2.2

struct BiasPair { int32_t index; float bias; };
struct RowBias {                   // exactly ONE form; both empty = unbiased
    const float* dense;            // count-length view; validated FUSED into the scan
    const BiasPair* pairs;         // unique in-range indices, finite values,
    int32_t pairCount;             //   O(pairCount) validation at query build
};
```

**Per-row bias (v2.1):** the composed score (similarity + bias) ranks in-scan —
exact by construction. One fused add per biased row, after dequantized scoring;
Query/QueryIntersect read one `RowBias` (intersection applies it once, to the fused
score); QueryBatch reads queryCount entries, one per query. Bias adds in the scored
metric's own direction (a reward is NEGATIVE on L2). Non-finite bias is illegal
input (`NonFiniteQuery`) — exclusion is a mask, bias is arithmetic, orthogonal.
Null/empty is the bit-identical unbiased path; all-zeros is compare-equal, not
bitwise (IEEE `-0.0 + 0.0 == +0.0`). Costs (measured, reference workload): dense
+3.5% f32 / +1.9% int8, sparse +0.4% single and ~0% batch — sparse is the batch
form; a dense per-query view in batch streams M x count x 4 bias bytes beside the
bank, stated not hidden.

**Cross-device exactness (v2.2):** `Exactness::CrossDevice` opts a query into
bit-identical scores and hit ORDER across machines and SIMD widths — the lockstep /
rollback / networked-motion-matching contract. Int8 banks only (f32 stays
per-device; `InvalidArgument`); all metrics; composes with segments, channels, bias,
exclusion, batch, intersection, and scratch snapshots. The query is quantized to
int8 (round-half-even in integer math), scoring accumulates in integers
(associative — width-independent), and the per-row epilogue is one fixed-order
double expression ending in the subnormal-floor contract: |score| < FLT_MIN is
exactly 0.0f on every machine (the FTZ/DAZ hole, closed by specification). Query
quantization adds error beyond row quantization — measure cross-device recall per
bank before adopting. `paddedDims <= kMaxCrossDeviceDims` (131072). Costs
(measured, reference workload): the integer path is FASTER than dequant-float —
single −18.5%, batch −14.6% (100k × 256 int8). Full contract: DETERMINISM.md §2c.

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
(8), `kMaxCrossDeviceDims` (131072, v2.2), `kSchemaVersion` (version.h);
`kPoolScaleFracBits` (24) and `kMaxPooledRows` (2^20) in compose.h (v2.4);
`ScratchBank::kDefaultRecallSeed`, `kRecallMinRows` (11), and
`kRecallInformativeRows` (100) in scratch.h (v2.3).

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

// v2.4: execute a PRE-QUANTIZED CrossDevice query (the XdQuery payload) directly.
Status QueryXd     (const BankView&, const XdQuery&, const QueryParams&,
                    Workspace&, Hit* outHits, int32_t* outCount);
Status QueryXdBatch(const BankView&, const XdQuery* queries, int32_t queryCount,
                    const QueryParams&, Workspace&, Hit* outHits, int32_t* outCounts);
```

Exact top-k, serial chunk loop, allocation-free once the `Workspace` is warm. `Query`
writes up to `k` hits best-first. `QueryBatch` scores all queries per chunk pass
(queries contiguous at `paddedDims` stride; results query-major, `k` slots per query)
using pair kernels that share row loads — per-query results are bit-identical to
`Query`. `QueryIntersect` (v1.1) ranks by each row's WORST score against the member
queries (true AND semantics — every returned row clears the fused score against every
query) in one bank pass; `queryCount == 1` degenerates to `Query` bit-identically.

**Pre-quantized queries (v2.4):** `QueryXd`/`QueryXdBatch` take the `XdQuery`
payload itself — `MakeCentroidCrossDevice`'s product — so the executed query is
bit-for-bit the caller's quantized bytes: no float round-trip, no
requantization. Law set: `params.exactness` must be `CrossDevice`, int8 banks
only, segments rejected (`InvalidArgument` — segment validation is defined
against a float query the caller does not have); exclusion, `ScoreAs`, and
both bias forms compose exactly as `Query` does in CrossDevice mode. The batch
keeps the chunk-outermost structure with the single-query kernel inside, so
batch results are **bit-identical to N single calls by construction**. The
payload validates at the boundary: the scale must be finite and non-negative,
and the self-dot is recomputed from the image and must match — a desynced
payload (a hand-edited or corrupted asset) is `InvalidArgument`, never a
repaired or silently wrong ranking. A Cosine bank rejects a genuinely
zero-norm payload as `ZeroNormQuery`.

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

// v2.2 cross-device kernel entries (integer accumulation + double epilogue):
struct XdQuery {                 // the pre-quantized query payload
    const int8_t* q8;            //   int8 image on the padded grid, 16-aligned
    double scale;                //   per-query dequant scale (double; no float round-trip)
    int64_t sqSum;               //   integer self-dot Sum(q_i^2) — the L2 epilogue term
};
void  QuantizeQueryXd(const float* paddedQuery, int32_t paddedDims,
                      int8_t* outQ8, double* outScale, int64_t* outSqSum);
void  ScoreChunkXd(...);              // whole-row CrossDevice chunk scorer
void  ScoreChunkSegmentedXd(...);     // segment list, weights-at-combine in double
void  ScoreChunkFusedXd(...);         // intersection fusion
float DecomposeRowScoreXd(...);       // contributions floored; total == scan score
float ScoreRowXd(const BankView&, const XdQuery&, int32_t rowIndex);
```

Pattern: one `TopK` per chunk (storage from your own pool), `ScoreChunk` chunks in
parallel, `Finalize` each, `MergeTopK` the lists — the result is bit-identical to a
serial `Query` in any list order (see [DETERMINISM.md](DETERMINISM.md)). Kernels do not
re-validate; gate inputs first. `ActiveSimdPath()` reports Scalar/SSE/AVX2/NEON.
`detail::*Mirror` functions expose the scalar mirrors for equality testing.

## compose.h — query construction (v1.1, integer-domain pooling v2.4)

```cpp
Status MakeCentroid (const BankView&, const int32_t* rowIndices, int32_t rowCount,
                     float* outPaddedQuery);                 // mean of rows, padded
Status MakeDirection(const float* paddedA, const float* paddedB, int32_t dims,
                     int32_t paddedDims, float* outPaddedQuery);  // normalize(a - b)
float  Margin(const Hit& better, const Hit& runnerUp, Metric scoredAs);

// v2.4: pool int8 rows into a QUANTIZED cross-device query (the XdQuery payload).
Status MakeCentroidCrossDevice(const BankView&, const int32_t* rowIndices,
                               int32_t rowCount,
                               const int32_t* weights,      // null = unweighted;
                                                            //   else one positive int per row
                               const uint32_t* excludeBits, // e.g. snapshot tombstones
                               int8_t* outQ8,               // paddedDims bytes, 16-aligned
                               double* outScale, int64_t* outSqSum);
```

Pure, allocation-free, deterministic (serial double-precision accumulation).
`MakeCentroid` dequantizes int8 rows through their scales; on Cosine banks the mean
renormalizes, and a zero-norm mean (antipodal members cancelling) is rejected, never
silently renormalized. `MakeDirection` builds the axis-projection query ("most a-like
relative to b"); score it with `ScoreAs::Dot` on L2 banks. `Margin` is the score gap
to the runner-up in the scored metric's better-direction — pass `ScoringMetric()`.

**Integer-domain pooling (v2.4):** `MakeCentroid` pools in float — fine for
presentation, but float summation order is where cross-device identity breaks.
`MakeCentroidCrossDevice` pools in the integer domain instead: each row's scale
ratio requantizes to a fixed-point multiplier (`kPoolScaleFracBits` = 24
fractional bits, round-half-even in integer math), per-dim contributions
accumulate in int64 — exact and order-free — and the accumulator requantizes
directly to the int8 image (no float mean, no norm reduction: symmetric
per-query quantization is invariant to positive scaling, so the result is
definitionally identical to normalize-then-quantize). Given identical rows,
scales, indices, and weights, the payload is bit-identical on every machine.
Weights fold by exact integer multiply, so all-equal weights produce the
bit-identical unweighted payload. Laws: int8 banks only
(`paddedDims <= kMaxCrossDeviceDims`); the pooled weight sum (= row count when
unweighted) is capped at `kMaxPooledRows` (2^20) — the accumulator's proven
overflow-free bound; over it is `InvalidArgument`, never a silent wrap. An
empty (or fully-excluded) selection is `InvalidArgument`, never a zero vector;
an all-zero integer accumulator (antipodal members cancelling) is
`ZeroNormQuery` — the check the omitted normalization would have performed.
This operator is a versioned composition operator for embedding-space
consumers (DETERMINISM.md §2d).

## analytics.h — bank analytics (v2.5)

Reductions over int8 `CrossDevice` banks: a set-to-set distance, directed
nearest-neighbour divergence, within-bank dispersion, and the shared pair score they
rest on — plus an offline per-device projection report. Cross-device determinism is one
argument (integer accumulation + the fixed-order double epilogue, ascending-row-index
mean / order-free max); the full contract is [DETERMINISM.md](DETERMINISM.md) §2e.

```cpp
Status ScoreXdPair(const XdQuery& a, const XdQuery& b, int32_t paddedDims, Metric metric,
                   float* outScore);                                   // the pair score all others rest on

Status CentroidDistanceCrossDevice(                                   // set-to-set; drift = this between two checkpoints
    const BankView& bankA, const int32_t* rowIndicesA, int32_t rowCountA,
    const int32_t* weightsA, const uint32_t* excludeBitsA,
    const BankView& bankB, const int32_t* rowIndicesB, int32_t rowCountB,
    const int32_t* weightsB, const uint32_t* excludeBitsB,
    Metric metric, int8_t* centroidScratchA, int8_t* centroidScratchB, float* outDistance);

Status MeanNNCrossDevice(                                             // directed mean nearest-neighbour divergence (A->B)
    const BankView& source, const uint32_t* sourceExcludeBits,
    const BankView& target, const uint32_t* targetExcludeBits,
    XdQuery* queryScratch, Hit* hitScratch, int32_t* countScratch, Workspace& ws, float* outValue);

Status MaxNNCrossDevice(...);                                         // same batch, order-free max (directed Hausdorff component)

Status SpreadCrossDevice(                                            // within-bank dispersion vs the selection's own centroid
    const BankView& bank, const int32_t* rowIndices, int32_t rowCount,
    const uint32_t* excludeBits, Reduce reduce, int8_t* centroidScratch, float* outValue);

enum class Reduce : uint8_t { Mean = 0, Max = 1 };                    // divergence/spread reduction

Status ProjectionReport(const BankView& bank, const float* paddedDirection,           // OFFLINE, per-device float
                        const uint32_t* groupBits, float* outProjections, float* outSeparation);
```

**Result direction — read this before ordering results.** The distance-named operators
(`*Distance`, `divergence`, spread) return a **distance** for `Cosine`/`L2` but a
**similarity** for `Dot`; the ordering flips with the metric:

| Metric | The scalar these operators return | Ordering |
|---|---|---|
| `Dot` | a dot **similarity** (`a.scale · b.scale · crossDot`) | **larger = MORE similar** — NOT a distance; a `Dot` consumer reads it with that convention |
| `Cosine` | `1 − crossDot / sqrt(a.sqSum · b.sqSum)`, in `[0, 2]` | larger = farther |
| `L2` | squared Euclidean distance | larger = farther |

The nearest-neighbour divergences score in the **target** bank's metric by the same
table. `ProjectionReport` is a plain per-row dot projection (float), unaffected.

**Scratch — caller-owned, zero steady-state allocation:**

| Operator | Caller scratch |
|---|---|
| `ScoreXdPair` | none |
| `CentroidDistanceCrossDevice` | two int8 buffers, `paddedDims` bytes each, 16-aligned |
| `MeanNNCrossDevice` / `MaxNNCrossDevice` | `source.count ×` (`XdQuery` + `Hit` + `int32`) — `sizeof(XdQuery)+sizeof(Hit)+4` = 36 bytes/source-row on a 64-bit target — plus a `Workspace` reserved for `k = 1` |
| `SpreadCrossDevice` | one int8 buffer, `paddedDims` bytes, 16-aligned |
| `ProjectionReport` | `outProjections` = `bank.count` floats; `outSeparation` one float (optional, Cohen's-d over `groupBits`). **Per-device float, offline — outside the cross-device contract.** |

**Checkpoint drift** — the centroid distance between two checkpoints' row sets:

```cpp
alignas(16) int8_t scratchA[256], scratchB[256];   // paddedDims = 256
float drift = 0.f;
Status s = CentroidDistanceCrossDevice(
    bankT,  rowsT,  countT,  /*weights*/nullptr, /*exclude*/nullptr,
    bankT1, rowsT1, countT1, nullptr,            nullptr,
    Metric::Cosine, scratchA, scratchB, &drift);
// Ok -> drift is a Cosine distance in [0,2]; larger = the checkpoints' centroids moved apart.
// Bit-identical on any machine given identical bytes (DETERMINISM.md §2e).
```

**Directed set divergence** — A against B, in B's metric:

```cpp
std::vector<XdQuery> q(A.count); std::vector<Hit> h(A.count); std::vector<int32_t> c(A.count);
Workspace ws; ws.Reserve(/*k*/1, /*batchWidth*/1);   // 36 * A.count scratch bytes + a k=1 warm Workspace
float div = 0.f;
Status s = MeanNNCrossDevice(A, /*srcExclude*/nullptr, B, /*tgtExclude*/nullptr,
                             q.data(), h.data(), c.data(), ws, &div);
// Scored in TARGET B's metric (see the direction table). MaxNNCrossDevice is the
// directed Hausdorff component (order-free max).
```

Rejections are explicit: an empty or fully-excluded selection is `InvalidArgument`; a
zero-norm centroid (antipodal members cancelling) is `ZeroNormQuery`; a desynced
`XdQuery` payload (scale/self-dot mismatch) is `InvalidArgument`. Over a mutable bank,
take a `ScratchBank::Snapshot()` first and pass its tombstone words as the exclude bits.

**Channel-scoped analytics (v3.0).** Each operator gains a `channel`-parameterized form
that pools or scores over `[channel.offset, channel.offset + channel.length)` instead of
the whole row — new surface for baked banks and scratch snapshots alike:

```cpp
Status CentroidDistanceCrossDeviceChannel(/* CentroidDistanceCrossDevice args */,
    Metric, int32_t channel, int8_t* scratchA, int8_t* scratchB, float* out);
Status MeanNNCrossDeviceChannel(const BankView& source, const uint32_t* srcExclude,
    const BankView& target, const uint32_t* tgtExclude, int32_t channel,
    XdQuery*, Hit*, int32_t*, Workspace&, float* out);   // MaxNNCrossDeviceChannel: order-free max
Status SpreadCrossDeviceChannel(const BankView&, const int32_t* rows, int32_t rowCount,
    const uint32_t* exclude, Reduce, int32_t channel, int8_t* scratch, float* out);
```

The `channel` indexes the bank's channel table. For Dot/L2 it is a sub-range of the
existing integer accumulation (the overflow bound only shrinks). **For Cosine the reduction
recomputes the sub-range integer self-dot and applies one IEEE correctly-rounded `sqrt` —
`1 − crossDot / sqrt(aSq·bSq)` — it does NOT read the per-row `channelInvNorms`.** Both
paths are cross-device-reproducible; the distinction is precision and reference status: the
analytics recompute the sub-range self-dot in the int→double domain for a full-precision,
bit-exact double reference (the one the analytics REF checks against), whereas the per-row
`channelInvNorms` are the float32-precision query-path scoring convenience (and pooled
payloads carry no per-row sub-norm). Scratch sizing matches the whole-vector
operators. A channel outside `[0, channelCount)`, or a bank with no channel table, is
`InvalidArgument`; a degenerate zero-sub-norm channel member floors to a defined `0` in a
reduction (a single per-channel query on a zero-norm sub-vector is still `ZeroNormQuery`).
"This mind's identity is drifting but its appearance is stable" is a `CentroidDistanceCross
DeviceChannel` over the identity channel versus the appearance channel across checkpoints.
Determinism per channel: [DETERMINISM.md §2e](DETERMINISM.md).

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

## graph.h — mutual k-NN graph + connected components (v3.2)

```cpp
Status BuildKnnNeighbors(const BankView&, int32_t k, bool excludeSelf,
                          int32_t* outNeighbors, Workspace&);
Status MutualFilter     (int32_t count, int32_t k, const int32_t* neighbors,
                          uint8_t* outMutualFlags);
Status BuildDuplicateGroups(const BankView&, int32_t* outGroupOf, int32_t* scratch);
Status ConnectedComponents(int32_t count, int32_t k, const int32_t* neighbors,
                          const uint8_t* mutualFlags, const int32_t* duplicateGroups,
                          int32_t* outComponentId, int32_t* unionFindScratch);
```

Post-processing over exact query output — touches no kernel, quantization, or format.
`BuildKnnNeighbors` is a top-k query per row (ties break ascending index); `MutualFilter`
keeps only edges where each row is in the other's top-k (a pure integer scan, no
tolerance); `BuildDuplicateGroups` finds exact byte-identical rows by construction (a
hash pass confirmed by full byte comparison — hash agreement alone never groups, so
same-decode-different-bytes rows are near-duplicates, not duplicates, and stay ungrouped);
`ConnectedComponents` unions duplicate-group edges first, then the surviving mutual edges
in ascending `(i, j)` order, with component ids canonicalized to the smallest member row
index — deterministic, never an allocation-order artifact. PER-DEVICE deterministic, no
cross-device claim. Caller-owned `Workspace`/scratch throughout; every function rejects
`k < 1`, `k >= count`, or a null buffer with no partial write.

## novelty.h — k-th-neighbour distance + calibrated baseline (v3.2)

```cpp
Status NoveltyScore         (const float* sortedBaseline, int32_t count,
                              float distance, float* outScore);
Status NoveltyProbeDistance (const BankView&, const float* paddedProbeQuery,
                              int32_t storedRow, int32_t channel /* -1 = whole-row */,
                              float* outDistance);
Status KthNeighborDistance  (const BankView&, const float* query, int32_t k,
                              const uint32_t* excludeBits, float* outDistance, Workspace&);
Status CalibrateNoveltyBaseline(const BankView&, int32_t k, int32_t sampleLimit,
                              float* outSortedDistances, int32_t* outCount, Workspace&);
```

Two independent limbs, composed by the CALLER into a tri-state verdict (no separate
"verdict" entry exists — the same caller-composes-from-primitives shape `compose.h`'s
query construction uses). **Limb 1 (identity):** `NoveltyProbeDistance` is the metric's
own exact distance between a probe and one stored row — on Cosine int8 this is a true
bit-exact 0.0 for a real duplicate (the parallel-code int arithmetic never rounds until
one final division+sqrt); on L2's expanded form it carries a disclosed double-precision
cancellation residue even on an exact duplicate, so a caller checks it against a small
epsilon (`1e-8f`), never bare equality. `channel == -1` scores the whole row; a channel
index scores that sub-range only, and a zero-energy Cosine slice on either side rejects
(`ZeroNormQuery`) rather than silently floors to a false match. Dot banks are rejected
outright — the verdict domain is L2 and Cosine only. **Limb 2 (statistical rank):**
`KthNeighborDistance` is a raw k-th-nearest-neighbour probe against a full view (ties are
the caller's own exclusion set, not self-widened); `NoveltyScore` converts a distance
into the baseline's empirical-CDF rank (ties resolve to the lowest rank), which the
caller compares against a lambda threshold. `CalibrateNoveltyBaseline` builds that
baseline over a caller-constructed sample view (it does not sample or stride itself),
writing each row's own k-th-NN `RankDistance` in ascending order. PER-DEVICE
deterministic; Dot is rejected on every entry point (`RankDistance` is undefined there).
Caller-owned `Workspace` throughout; every function rejects malformed k/count/metric
arguments with no partial write.

## matching.h — mutual-NN correspondence + CSLS margins (v3.2)

```cpp
struct MatchPair { int32_t sourceIndexA = -1, sourceIndexB = -1; float cslsMargin = 0.0f; };

Status MutualNearestMatches(
    const BankView& sampleViewA, const int32_t* sampleSourceIndices,
    const BankView& fullViewB,   const uint32_t* excludeBitsB,
    const BankView& fullViewA,   const uint32_t* excludeBitsA,
    int32_t matchK, MatchPair* outPairs, int32_t* outPairCount, Workspace&);
```

Sampled-A-verified-against-full-banks: for each row of a caller-constructed sample of
bank A, its top-`matchK` forward candidate in the FULL live bank B, then back-verified
against the FULL live bank A — a pair is mutual iff the candidate's own top-1 in full A
is the original sampled row. Both retrievals run against complete banks, never a sample,
so the sample bounds only how many A rows are CHECKED, never correctness of a reported
pair (a design correction from an earlier two-sided-sample scheme that measured 0.9%
recovery / 98% spurious matches at 500k rows). Every checked row gets an entry —
`sourceIndexB = -1` and `cslsMargin = 0.0` mean honestly unmatched, not an error.
`cslsMargin` is the paper's cross-domain similarity local scaling
(`2*Sim(i,j) - r_B(i) - r_A(j)`), disclosed as a diagnostic on Dot/L2 banks (not the
paper's calibrated cosine-only setting) — classification against a threshold (matched vs.
ambiguous) is the caller's, exactly as `NoveltyScore`'s raw statistic is. `Dims` and
`Metric` must match across all three views; `Quantization` may differ (matching runs on
query scores over dequantized rows) — the field this library's flagship "archive vs.
baked" use case rests on: a player's saved scratch-bank archive matched against the
shipped reference bank. PER-DEVICE deterministic, no cross-device claim. This is the
library's disclosed HEAVY pass — O(sample x full-B x dims) plus back-verification, linear
in bank size, not sub-second at scale; a caller-side concern (chunking, progress, cancel),
not part of this contract.

## scratch.h — mutable banks (v2.0, recall audit v2.3)

```cpp
class ScratchBank {              // single writer, lock-free readers
    Status Create(int32_t capacity, int32_t dims, Metric, Quantization,
                  const Allocator& = DefaultAllocator());   // ONE arena allocation
    Status Create(int32_t capacity, int32_t dims, Metric, Quantization,
                  bool retainFloats, const Allocator& = ...); // v2.3 retention overload
    Status Create(int32_t capacity, int32_t dims, Metric, Quantization,   // v3.0 channel overload
                  const ChannelInfo* channels, int32_t channelCount,       //   table set at Create
                  bool retainFloats = false, const Allocator& = ...);
    Status Relabel(const ChannelInfo* newChannels, int32_t newChannelCount); // v3.1: atomically
                  // replace the channel table on a LIVE bank (exclusive drain; stored rows
                  // untouched; Cosine sub-norms re-derived; reject-over-degrade; count change,
                  // boundary move, promote, and demote all supported)
    Status Append(const float* row, int32_t dims, int32_t* outIndex);
    Status Remove(int32_t index);                 // atomic tombstone; idempotent
    Status Snapshot(BankView* outView, uint32_t* outTombstones) const;
    Status Grow(int32_t newCapacity);             // EXCLUSIVE; preserves indices
    int32_t FreezeLiveCount() const;
    Status Freeze(void* outRows, float* outScales, int32_t* outIndexMap,
                  ScratchRecallReport* outReport = nullptr,   // v2.3: fresh number
                  Workspace* recallWs = nullptr,              //   over the compacted rows
                  uint64_t recallSeed = kDefaultRecallSeed) const;
    Status Freeze(void* outRows, float* outScales, int32_t* outIndexMap,   // v3.0 channel-aware:
                  float* outChannelInvNorms, ScratchRecallReport* = nullptr,//   sub-norms re-derived
                  Workspace* = nullptr, uint64_t = kDefaultRecallSeed) const;//  over compacted rows
    Status FreezeWithRecall(void* outRows, float* outScales, int32_t* outIndexMap, // v3.0 (D-V3-7)
                  float* outChannelInvNorms, ScratchRecallReport* outRecallReports, //  per-channel
                  int32_t reportCount, Workspace&, uint64_t = kDefaultRecallSeed) const;
    int32_t GetChannelCount() const;  const ChannelInfo* GetChannels() const;       // v3.0 table read-back
    Status MeasureScratchRecallPerChannel(Workspace&, ScratchRecallReport* outReports, // v3.0 (D-V3-7)
                  int32_t reportCount, uint64_t seed = kDefaultRecallSeed);          //   recall@k per channel
    Status Save(const ScratchArchive&) const;     // writer-side
    Status Load(const ScratchArchive&, const Allocator& = ...); // EXCLUSIVE
    int32_t Count(); int32_t LiveCount(); int32_t Capacity();
    static int32_t TombstoneWords(int32_t count);

    // v2.3 recall audit:
    bool     RetainsFloats() const;               // the Create flag, read back
    const float* RetainedRow(int32_t index) const;// the post-normalization row, or null
    int64_t  QuantizedRowBytes() const;           // honest-budget getters:
    int64_t  RetainedRowBytes() const;            //   0 when retention is off
    uint64_t Generation() const;                  // monotonic mutation counter
    Status   MeasureScratchRecall(Workspace&, ScratchRecallReport* out,
                                  uint64_t seed = kDefaultRecallSeed);
    bool     RecallReportStale(const ScratchRecallReport&) const;
};
struct ScratchRecallReport {     // v2.3
    float    recall;             // hits / possible over the sweep, [0, 1]
    int32_t  k;                  // min(10, liveRows-1) — below 11 live rows this
                                 //   is a recall@(liveRows-1), and the field says so
    int32_t  sampleCount;        // min(1000, liveRows) self-queries drawn
    int32_t  liveRows;
    uint64_t seed;               // recorded so the number is reproducible
    uint64_t generation;         // the bank's mutation state at measurement
    bool     informative;        // false below liveRows = 100 (kRecallInformativeRows):
                                 //   mathematically valid, statistically uninformative
};
struct ScratchArchive { bool(*write)(void*, const void*, size_t); bool(*read)(void*, void*, size_t); void* user; };
```

A snapshot IS a `BankView` — every query entry point works on it unchanged.
Deletion is exclusion: OR (or pass) the snapshot's tombstone words as
`QueryParams::excludeBits`. Append validates like the importer (finite, dims
match; Cosine normalizes and rejects zero-norm; int8 quantizes per-row), writes
the row, then publishes the count with a store-release — readers never see a
partial row. Removal is snapshot-consistent, not mid-scan-preemptive. `Grow`
preserves indices (saved indices are the consumer contract) and copies the
retention arena in the same reallocation; `Freeze` compacts and renumbers,
returning the old→new map so consumers remap stored handles. Zero steady-state
allocation: everything lives in the arena from `Create` (retention included —
`ArenaBytes` simply grows). Archive format: [FORMAT.md](FORMAT.md) section 3;
concurrency guarantees: [DETERMINISM.md](DETERMINISM.md) section 2b.

**Recall audit (v2.3):** the import-time recall-honesty pattern, extended to the
mutable half. `Create(..., retainFloats = true)` — opt-in, never the default —
retains the post-normalization float row the quantizer consumed (on Cosine
banks, the unit-norm row) beside each quantized row; rejected appends retain
nothing. `MeasureScratchRecall` then runs a seeded self-query sweep over the
current snapshot: a double-precision scan of the retained rows (the reference
top-k) versus the bank's own quantized `Exactness::CrossDevice` scan (a
float32 retention bank, which has no cross-device mode, audits its own
per-device scan — defined, trivially high), recall@k with the sample
discipline in the report struct above. It runs under the reader
pin like any query, allocates nothing once the workspace is warm, and on a
non-retention bank returns `InvalidArgument` — a defined rejection, never a
guessed number. The number is well-defined when no writer runs concurrently; a
racing append/remove yields a safe but non-reproducible number.

**Staleness:** `Generation()` advances on every successful `Append`, every
newly-set `Remove` (idempotent re-removes do not advance), and past every prior
stamp on `Load` (a load is the ultimate mutation). A report whose stamped
generation no longer matches reads as stale through `RecallReportStale` — never
silently current. `Freeze` on a retention bank can produce a fresh report
measured over the compacted rows at freeze time (pass `outReport` + a
workspace); a non-retention freeze produces none — no stale number wearing a
current face.

**Memory cost, stated plainly:** retention adds `4 × dims` bytes per row. At
256 dims that turns an int8 row from 260 into 1284 bytes (~4.9×) and a float32
row from 1024 into 2048 (2.0×) — `QuantizedRowBytes()`/`RetainedRowBytes()`
report the exact split. A dev/audit posture, not a shipping default.

**Persistence:** a retention bank's `Save` writes archive version 2 with the
retained floats appended; a non-retention bank still writes version 1,
byte-identical to pre-v2.3. `Load` accepts versions 1 and 2 (a v1 blob loads
with retention absent — defined), hard-rejects anything else as `BadFormat`,
and bounds the archive geometry before any allocation arithmetic
(`paddedDims <= kMaxCrossDeviceDims`, capacity <= 2^28 rows — absurd headers
are a format defect, not an allocator outcome).

**Channels on a mutable bank (v3.0):** the channel table — a partition of the
row into named sub-ranges (`ChannelInfo` = offset + length, ascending,
non-overlapping, on the 16-byte element grid, up to `kMaxChannels = 8`) —
becomes a scratch-bank property set at `Create` and **fixed for the bank's
lifetime**; `Append` validates each row against it. On a Cosine channel bank,
`Append` computes the appended row's per-channel inverse sub-norms
per-row-standalone from the quantized bytes and folds a `capacity × channelCount`
sub-norm arena into the same single allocation (Dot/L2 channel banks carry none —
a per-channel Dot/L2 score is a plain segment dot / squared distance). `Snapshot`
populates the view's channel fields, so every query entry point and every
channel-scoped analytics operator serves named-channel and raw-range queries on
the snapshot with the exact code baked banks use. Channel-aware `Freeze`
graduates the bank to a `schemaVersion 2` baked asset, re-deriving the
per-channel sub-norms over the compacted (live) rows;
`MeasureScratchRecallPerChannel` and the channel-aware `FreezeWithRecall` report
recall@k per channel over each sub-range (a retention-enabled Cosine channel
bank; `InvalidArgument` otherwise). **Cost, stated plainly:** the per-channel
sub-norm is an append-time compute cost — it adds ~14% (~194 ns) to a Cosine int8
append at 4 channels / 256 dims; the query path reads the sub-norm arena but does
no sub-norm work. The arena is `channelCount × 4` bytes/row (16 B/row at 4
channels — mandatory on a Cosine channel bank, ~6% of the int8 row it sits beside,
and ~1.6% the size of the optional retention arena). Dot/L2 channel banks carry no
sub-norm arena at all. Archive format: [FORMAT.md](FORMAT.md) §3; channel-scoped
analytics: the `analytics.h` section above.

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

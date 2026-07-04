# Integrating SuperFAISS into an Engine

The library is deliberately incomplete: it has **no threads, no I/O, no JSON parser,
and no allocator opinions**. Those seams are where your engine plugs in. This guide
covers each seam, the build flags that keep the determinism promise intact, and the
performance model that should shape your integration. The reference integration is the
[SuperFAISS For Unreal Engine plugin](https://github.com/dansupergameprogrammer/superfaiss-unreal)
(SuperFAISSUnreal), which exercises every seam described here.

## 1. Build

Any C++17 compiler. Compile `src/*.cpp` into your target; add `include/` to the include
path. Non-negotiable flags (the shipped `CMakeLists.txt` applies them):

| Concern | GCC / Clang | MSVC |
|---|---|---|
| No implicit FP contraction (determinism) | `-ffp-contract=off` | `/fp:precise` |
| SSE4.1 intrinsics baseline (x86) | `-msse4.2` on `kernels.cpp` | (implied by x64) |
| AVX2 TU | nothing (`target` attributes) | `/arch:AVX2` on `kernels_avx2.cpp` |

If your build system compiles these sources under fast-math defaults (game engines
usually do), you must override the *flags* for these translation units — source-level
pragmas are NOT sufficient on clang (its backend fuses under fast-math regardless of
`float_control` or `clang fp contract` pragmas; verified at the compiler). In Unreal
Engine: `FPSemantics = FPSemanticsMode.Precise` in the module's Build.cs. Then keep a
mirror-equality test (`detail::DotF32 == detail::DotF32Mirror` over random inputs) in
your own suite as the permanent tripwire.

SIMD is selected per platform automatically (NEON on ARM, SSE4.1 on x86 with a runtime
CPUID upgrade to AVX2+FMA, scalar anywhere else). There is no platform-specific code:
if your platform has a C++17 toolchain, the library compiles.

## 2. Memory seam

- **Banks are yours.** The library scans a `BankView` — non-owning pointers. Load, own,
  and free bank memory however your asset system likes; the only rules are the layout
  rules in [FORMAT.md](FORMAT.md) (16-byte alignment, padded stride, zero pad lanes).
  Banks are immutable after load, which is what makes concurrent reads lock-free.
- **Scratch is pooled.** A `Workspace` has a single owner and serves one call at a
  time; pool one per concurrent query site. Route the `Allocator` seam to your
  engine's allocator if you track memory. `AllocationCount()` + `Workspace::GrowthCount()`
  let your tests assert zero steady-state allocation — a warm pool never allocates.
- **Scratch banks (v2.0) are the one owning type.** `ScratchBank` allocates its
  fixed-capacity arena once through the same `Allocator` seam and never again
  (append/remove/snapshot/query are allocation-free; `Grow` is an explicit,
  caller-initiated reallocation). Persistence goes through the caller-owned
  `ScratchArchive` seam — your save system provides the read/write callbacks, the
  bank owns the format.

## 3. Threading seam

The library is single-threaded by design; you own the scheduling. Three composable
levels:

1. **Blocking call on a worker** — `Query()` is safe from any thread (immutable bank,
   caller-owned workspace). Simplest correct integration.
2. **One query, many cores** — fan out with your scheduler:
   one `TopK` per chunk (storage from one pooled buffer), `ScoreChunk(chunk)` in
   parallel, `Finalize`, then `MergeTopK`. Bit-identical to the serial scan in any
   completion order — that is a designed property, not luck. Join before releasing the
   workspace (single-owner contract).
3. **Many queries, one pass** — `QueryBatch`, or parallelize chunk-outer yourself and
   score each chunk against all queries with `ScoreChunkPair` while its rows are
   cache-resident. Per-query results stay bit-identical to singles.

Scratch banks (v2.0) add one rule to this model: one logical writer, lock-free
readers. Gate queries with `TryPinReader`/`UnpinReader` and run Grow/Load inside
`BeginExclusive`/`EndExclusive` — the shipped protocol's seq_cst orderings are what
make it correct on weakly-ordered ISAs (ARM); do not substitute your own flag/counter
pair (see the pin/drain commentary in `scratch.h`).

Lifetime rule for async integrations: whatever owns the bank must outlive the scan.
If your object system can destroy assets at shutdown while workers run (Unreal's exit
purge does), **drain in-flight queries before object teardown** — count dispatches and
wait for zero in your shutdown path.

## 4. Data pipeline seam

Producers emit the two-file sidecar (trivial from Python — `tools/wvbank.py`;
`tools/glove_to_wvbank.cpp` converts GloVe text). Your importer parses the JSON header
with whatever JSON library you already have (the core deliberately has none), then
runs the bake: `ValidateSourceRows → NormalizeRows (cosine) → QuantizeRowsInt8` or
`PadRowsFloat32` (plus `ComputeChannelInverseNorms` for Cosine channel banks), and
validates the result with `ValidateBankData` at every load. Channel banks declare
`"schemaVersion": 2` with a `channels` array in the sidecar (FORMAT.md section 1);
v1-only importers hard-reject them by the format's own unknown-version rule.

Import-time practices worth copying from the
[reference plugin](https://github.com/dansupergameprogrammer/superfaiss-unreal): quantize to int8 by
default; measure recall@10 of the quantized bank against its float32 source with a
**seeded** sample and store the number on the asset; hash the sidecar pair so unchanged
re-imports are no-ops; reject malformed input with a specific diagnostic and no partial
asset.

## 5. The performance model (what to expect, and why batch exists)

A flat scan is a memory stream: cost ≈ bank bytes ÷ bandwidth, until SIMD makes it
compute-bound. Practical desktop numbers (AVX2, one core): int8 scans at roughly
10 GB/s-equivalent — a 4 MB bank (40k × 100d int8) in ~0.5 ms serial; chunk-parallel
across a task graph, ~0.13 ms wall; batched, ~0.06 ms per query. int8 is the default
for a reason: it is a 4× bandwidth cut, i.e. roughly a 4× speed and memory win, for
~1% recall loss.

Batching amortizes *memory traffic and per-call overhead*, not compute — on
compute-bound hardware the per-query gain over parallel singles is modest (~2×), but a
batch also streams the bank through cache **once** instead of M times, which matters on
a live game frame. On bandwidth-lean hardware (mobile), expect batch amortization to be
much larger. Measure on your targets; the ranking never changes, only the time.

## 6. Testing your integration

Minimum honest suite: a golden query on a real shipped bank; mirror equality (§1);
serial-vs-parallel bit-equality if you built level 2; batch-vs-singles bit-equality if
you built level 3; allocation-flatness across warm queries; and your shutdown drain
under in-flight load. The
[reference plugin](https://github.com/dansupergameprogrammer/superfaiss-unreal) ships
all of these as engine automation tests if you want the shapes.

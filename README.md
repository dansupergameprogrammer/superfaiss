# SuperFAISS

[![tests](https://github.com/dansupergameprogrammer/superfaiss/actions/workflows/tests.yml/badge.svg)](https://github.com/dansupergameprogrammer/superfaiss/actions/workflows/tests.yml)

Fast, deterministic, allocation-free k-nearest-neighbor search over baked embedding banks,
built for game runtimes. Dependency-free C++17 — the standard library and nothing else.
CI-verified on Windows x64, Linux x64 (AVX2), and macOS arm64 (NEON); four compilers.

In plainer terms: games are full of *find the best match* problems — the animation pose
that best continues this motion, the NPC memory most relevant to what just happened, the
item or sound or dialogue line closest to what the player typed, drew, or did. If you can
turn your candidates into vectors of numbers (embeddings — any encoder model produces
them, including small ones running offline in a content pipeline), SuperFAISS answers
"which of my thousands of candidates is most similar to this one?" in a fraction of a
millisecond. The answers are exact, not approximate, and the same query against the same
bank returns the same result on every run and — in v2.2's opt-in mode — on every machine.
That reliability is what lets you build actual gameplay on top of the answers.

SuperFAISS is an **independent implementation**. It is **not a fork of, derived from, or
affiliated with Meta's FAISS**; the name is nominative homage to the library that defined
the category. If you need approximate indexes over billion-scale corpora on servers, use
FAISS. If you need exact top-k over game-scale banks, in milliseconds, on a background
thread, deterministically, on every platform your game ships on — that is what this is.

## First query

```cpp
#include "superfaiss/superfaiss.h"
using namespace superfaiss;

// 1) Bake once: your float embeddings (count x dims) -> a padded, aligned bank.
const int32_t count = 40000, dims = 128;
const int32_t pd = PaddedDims(dims, Quantization::Float32);
Allocator a = DefaultAllocator();                 // or route to your engine's allocator
float* rows = static_cast<float*>(a.alloc(sizeof(float) * count * pd, kAlignment, a.user));
PadRowsFloat32(myEmbeddings, count, dims, pd, rows);

BankView bank;
bank.rows = rows;      bank.count = count;                    bank.dims = dims;
bank.paddedDims = pd;  bank.quant = Quantization::Float32;    bank.metric = Metric::Dot;

// 2) Query: the 10 most similar rows, best first. Sub-millisecond at this scale --
//    call it every frame, from any one thread.
float* query = static_cast<float*>(a.alloc(sizeof(float) * pd, kAlignment, a.user));
std::memset(query, 0, sizeof(float) * pd);        // pad lanes must be zero
/* fill query[0 .. dims) */

Workspace ws;                                     // reusable scratch; allocation-free once warm
Hit hits[10];
int32_t n = 0;
QueryParams params;
params.k = 10;
Query(bank, query, params, ws, hits, &n);         // hits[i].index, hits[i].score
```

Int8 banks (4x smaller and faster) bake with `QuantizeRowsInt8`; Cosine banks
normalize at bake with `NormalizeRows`. The `.wvbank` sidecar format and the
reference reader/writer are in [FORMAT.md](docs/FORMAT.md).

## Building

Any C++17 compiler, no dependencies. `build.bat` (MSVC), or CMake:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/superfaiss_tests
```
(MSVC multi-config: `build/Release/superfaiss_tests.exe`; or use `build.bat`.)

SIMD (SSE4.1 / AVX2+FMA / NEON, scalar fallback) is selected automatically; every
path has a scalar mirror with the identical accumulation structure, and the test
suite enforces that SIMD and mirror results are bit-identical on a device.

## What it does

- **Bank in, query in, top-k out.** Exact (non-approximate) flat scan. No index build, no
  training, no surprises.
- **float32 and int8 banks.** Symmetric per-vector int8 quantization is a 4x memory-bandwidth
  cut — flat scan is memory-bound, so it is also roughly a 4x speed win.
- **Deterministic by construction.** Identical bank + identical query produce
  bit-identical results on a device — regardless of thread count, chunk scheduling, or
  call history ([DETERMINISM.md](docs/DETERMINISM.md)).
- **Zero steady-state allocation.** Scratch comes from a caller-provided workspace; the
  allocator seam counts every allocation so you can assert zero in your own tests.
- **Your scheduler, our kernels.** The library is single-threaded by design and exposes
  chunk-level scoring; your engine's scheduler (ParallelFor, task graph, job system)
  drives the chunks. Determinism does not depend on who schedules what.
- **Batch queries.** M queries scored in one bank pass amortize the memory traffic — the
  economics that make per-tick entity queries cheap.
- **Segmented queries and named channels (v2.0).** Score a weighted combination of
  vector sub-ranges — "identity 1.0, appearance 0.2" — and decompose any hit into
  exact per-channel contributions ("appearance matched, identity didn't"). A semantics
  feature, not a speed feature: a segmented scan still costs about one full scan —
  measured, not assumed ([API.md](docs/API.md)).
- **Per-row score bias (v2.1).** A caller-provided bias added in-scan, before
  selection, so the composed ranking is exact — sparse `(index, bias)` pairs for
  motion matching's continuing-pose reward (effectively free) or a dense per-row view
  for memory salience. Rules and measured costs in [API.md](docs/API.md).
- **Cross-device bit-exactness (v2.2).** An opt-in query mode
  (`Exactness::CrossDevice`, int8 banks): bit-identical scores AND hit order across
  DIFFERENT machines — x86 and ARM, Windows/Linux/macOS, any SIMD width — the
  property lockstep and rollback multiplayer, networked motion matching, and
  server-side validation actually require. **The claim runs as a test**: CI asserts
  a pinned hash over committed fixture banks — including adversarial banks aimed at
  the mode's own weakest case — on Windows, Linux, and macOS-ARM, across every
  kernel path each runner can force. It also measures FASTER than the default int8
  scan. Scope and recall cost are stated plainly in
  [DETERMINISM.md §2c](docs/DETERMINISM.md).
- **Scratch banks (v2.0).** A fixed-capacity mutable bank for runtime-accumulated
  vectors (NPC memory, session embeddings): single writer, lock-free readers,
  tombstone removal, `Freeze` into a standard immutable bank, save/load through an
  archive seam. Every query feature above works on scratch content unchanged.

## What it deliberately is not

No approximate indexes (HNSW/IVF) in V1 — the bank format reserves a versioned index block
so they can arrive additively. No threads. No file I/O. No JSON parsing (the `.wvbank.json`
sidecar header is parsed by importers; `tools/wvbank.py` is the reference reader/writer).
No embedding generation — banks come from your pipeline.

## Format

A bank is row-major vectors, padded to a 16-byte row stride, 16-byte-aligned base pointer.
Cosine banks are pre-normalized at bake (query-time cosine is then a plain dot product; a
zero-norm row is a bake-time error, not a runtime branch). Interchange is a two-file
sidecar: `<name>.wvbank.json` (header) + `<name>.wvbank.bin` (raw float32 rows) — trivial
to emit from any pipeline in a few lines.

## Reference integration

[**SuperFAISS For Unreal Engine**](https://github.com/dansupergameprogrammer/superfaiss-unreal)
is the canonical engine integration: banks as assets with import-time validation and
recall measurement, sync/async/batch queries on the task graph, named channels with
decomposition, scratch banks, an in-editor inspector, and the full automation-test
suite. It exercises every seam [INTEGRATION.md](docs/INTEGRATION.md) describes — if
you are embedding this library anywhere, it is the worked example.

## Documentation

- [FORMAT.md](docs/FORMAT.md) — the `.wvbank` interchange format and baked memory
  layout (the format is an open standard; emit it from any pipeline)
- [API.md](docs/API.md) — full public API reference
- [DETERMINISM.md](docs/DETERMINISM.md) — the bit-exactness guarantees, their
  mechanisms, and embedder obligations
- [INTEGRATION.md](docs/INTEGRATION.md) — embedding in an engine: build flags,
  memory/threading/pipeline seams, the performance model, what to test

## Measured (desktop, AVX2, 40k vectors x 100 dims, int8, exact top-10)

| Path | Per query |
|---|---|
| Serial scan, one core | ~0.5 ms |
| Chunk-parallel (host scheduler over `ScoreChunk` + `MergeTopK`) | ~0.13 ms |
| Batched (64 queries, one pass, pair kernels) | ~0.06 ms |

Same answers on every path, bit for bit — that equivalence is test-enforced, not
aspirational.

## License

MIT. See [LICENSE](LICENSE).

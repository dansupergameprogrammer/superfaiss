# SuperFAISS

[![tests](https://github.com/dansupergameprogrammer/superfaiss/actions/workflows/tests.yml/badge.svg)](https://github.com/dansupergameprogrammer/superfaiss/actions/workflows/tests.yml)

Fast, deterministic, allocation-free k-nearest-neighbor search over baked embedding banks,
built for game runtimes. Dependency-free C++17 — the standard library and nothing else.
CI-verified on Windows x64, Linux x64 (AVX2), and macOS arm64 (NEON); four compilers.

SuperFAISS is an **independent implementation**. It is **not a fork of, derived from, or
affiliated with Meta's FAISS**; the name is nominative homage to the library that defined
the category. If you need approximate indexes over billion-scale corpora on servers, use
FAISS. If you need exact top-k over game-scale banks, in milliseconds, on a background
thread, deterministically, on every platform your game ships on — that is what this is.

## What it does

- **Bank in, query in, top-k out.** Exact (non-approximate) flat scan. No index build, no
  training, no surprises.
- **float32 and int8 banks.** Symmetric per-vector int8 quantization is a 4x memory-bandwidth
  cut — flat scan is memory-bound, so it is also roughly a 4x speed win.
- **Deterministic by construction.** The score/index comparator is a strict total order, so
  identical bank + identical query produce bit-identical results regardless of thread count,
  chunk scheduling, or call history, on a given device.
- **Zero steady-state allocation.** Scratch comes from a caller-provided workspace; the
  allocator seam counts every allocation so you can assert zero in your own tests.
- **Chunk-granular kernels.** The library is single-threaded by design and exposes
  chunk-level scoring; your engine's scheduler (ParallelFor, task graph, job system) drives
  the chunks. Determinism does not depend on who schedules what.
- **Batch queries.** M queries scored in one bank pass amortize the memory traffic — the
  economics that make per-tick entity queries cheap.
- **Segmented queries and named channels (v2.0).** Score a weighted combination of
  contiguous sub-ranges of the vector — "identity 1.0, appearance 0.2" — with exact
  per-segment decomposition of every hit ("appearance matched, identity didn't"), and
  true per-channel cosines on cosine banks. **Segments are a semantics feature, not a
  speed feature**: a segmented scan is priced at approximately one full scan (for
  dot-family scoring the weights fold into the query exactly, so it runs the plain V1
  kernels at V1 speed; masking a range does not make the scan faster on this
  row-interleaved layout — measured, not assumed). A degenerate one-segment query is
  bit-identical to the whole-row scan.
- **Per-row score bias (v2.1).** An optional caller-provided bias added to each row's
  score IN-SCAN, before top-k selection — so the composed ranking is exact (a row
  outside the similarity top-k that wins the composed score is unrecoverable after the
  scan; post-weighting the returned top-k is approximate ranking). Two forms: a dense
  count-length view (memory salience) and sparse `(index, bias)` pairs (motion
  matching's continuing-pose reward — one biased row per query, effectively free:
  measured +0.4% single / ~0% batch; the dense form measures +3.5% f32 / +1.9% int8).
  Finite values only — `-inf` is not a mask; exclusion is a mask, bias is arithmetic.
  Bias adds in the scored metric's own direction (a reward is negative on L2).
- **Scratch banks (v2.0).** A fixed-capacity mutable bank for runtime-accumulated
  vectors (NPC memory, session embeddings): single writer, lock-free wait-free readers,
  atomic tombstone removal (deletion is exclusion), index-preserving `Grow`, `Freeze`
  into a standard immutable bank (bit-identical, with an old-to-new index map), and
  save/load through a caller-owned archive seam. Determinism extends to it as
  determinism-given-history. A snapshot IS a `BankView`, so every query feature above
  works on scratch content unchanged.

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

## Building

Any C++17 compiler. `build.bat` (MSVC), or CMake:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/superfaiss_tests
```
(MSVC multi-config: `build/Release/superfaiss_tests.exe`; or use `build.bat`.)

SIMD: NEON on ARM/ARM64 and SSE4.1 on x86/x64 are selected at compile time, with a
runtime cpuid upgrade to AVX2+FMA on x86 hardware that supports it, and a scalar
fallback elsewhere. Every path has a scalar mirror with the identical accumulation
(and, for AVX2, `std::fma`) structure, so SIMD and mirror results are bit-identical on
a given device — enforced by the test suite.

## License

MIT. See [LICENSE](LICENSE).

# SuperFAISS

Fast, deterministic, allocation-free k-nearest-neighbor search over baked embedding banks,
built for game runtimes. Dependency-free C++17 — the standard library and nothing else.

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

## Building

Any C++17 compiler. `build.bat` (MSVC), or CMake:

```
cmake -B build && cmake --build build && build/tests/superfaiss_tests
```

SIMD is selected at compile time: SSE4.1 on x86/x64, NEON on ARM/ARM64, and a scalar
fallback that is bit-identical to the SIMD paths (same striped accumulation order)
everywhere else.

## License

MIT. See [LICENSE](LICENSE).

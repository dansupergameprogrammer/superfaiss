# Determinism

The library's hardest promise: **identical bank + identical query (+ identical k and
filter) ⇒ bit-identical results on a given device — regardless of thread count, chunk
scheduling, batching, or call history.** This document explains the mechanisms, exactly
what is and is not guaranteed, and what an embedder must do to keep the promise intact.

## 1. Why it holds

### The ranking is a strict total order

Hits compare by score (direction per metric), with ties broken by **ascending row
index**. Because every pair of hits is strictly ordered, the top-k *set* and its sorted
output are mathematically unique — chunk scheduling, merge order, and insertion order
cannot matter. This is why `MergeTopK` accepts per-chunk lists in *any* order and still
produces the identical result (test-pinned: merging reversed lists equals the serial
scan bit-for-bit).

The one thing that could break a total order is NaN (all comparisons false). That is
why NaN is fenced at the boundaries: query validation rejects non-finite input, and
`ValidateBankData` rejects non-finite bank content — a kernel never sees NaN.

### Kernels have one accumulation structure, mirrored exactly

Floating-point addition is not associative, so the *order* of operations is pinned:

- SSE/NEON paths: rows are walked in 16-element blocks feeding four 4-lane accumulators
  in order; remainders feed accumulators in order; lanes combine as `(l0+l1)+(l2+l3)`
  per accumulator, then `(a0+a1)+(a2+a3)`. Multiply-then-add, never fused.
- AVX2 path: same shape at 8 lanes / 32-element blocks, using FMA **explicitly** —
  mirrored by `std::fma`, which rounds identically to the hardware instruction.

Every SIMD path has a scalar **mirror** implementing the identical structure
(`DotF32Mirror` et al.); the test suite asserts SIMD ≡ mirror **bit-for-bit** across
dimensions and dtypes. The pair kernels (two queries per row pass) keep each query's
accumulation structure identical to the single kernel, so batching changes nothing —
also test-pinned.

### Dispatch is a pure function of device and shape

The AVX2 upgrade is chosen once per process by CPUID, and per call by row stride. Same
machine, same bank ⇒ same path, every time.

## 2. What is guaranteed, precisely

| Property | Guaranteed? |
|---|---|
| Repeat call ⇒ bit-identical | Yes |
| Serial scan ≡ parallel chunked scan | Yes (total order; test-pinned) |
| Batch ≡ equivalent single queries | Yes (pair kernels bit-equal singles; test-pinned) |
| Same results across thread counts / schedulers | Yes |
| Segmented queries (v2.0): same segments + weights => bit-identical; degenerate one-segment list == whole-row scan | Yes (segments process in ascending offset order, always; the segment list is part of the query, not a source of nondeterminism) |
| Decomposition (v2.0): contributions sum bit-exactly to the scan's own score | Yes (the partials ARE the scan's accumulators; no second code path exists to drift) |
| Same results across devices with the same active SIMD path | Yes in practice for int8 (integer→float conversion + pinned float ops are IEEE-determined), but **not contractually claimed** |
| Same results across different SIMD paths (e.g. AVX2 desktop vs NEON phone) | **No.** Different accumulation widths round differently. Per-device determinism only. |

If you need cross-device, replay-grade exactness (lockstep multiplayer), you need a
pure-integer scoring path — a planned extension, not present in v1.0. Note that consoles
are fixed hardware: "per device" is effectively "per SKU" there, which is the strong
version of the promise.

## 2b. Scratch banks: determinism given history

A scratch bank (v2.0) extends the guarantee to mutable state: **the same
append/remove sequence followed by the same query yields bit-identical results
per device.** Rows are normalized and quantized at append with the exact math
the importer uses (int8 quantization is per-row and standalone), so a snapshot
of a scratch bank scores bit-identically to an immutable bank baked from the
same live rows — and `Freeze()` produces exactly that bank, byte for byte.
Removal is snapshot-consistent: a tombstone is visible to every snapshot taken
after the `Remove`, and a snapshot, once taken, is immutable — querying it
twice is bit-identical regardless of concurrent writer activity.

## 3. Embedder obligations

1. **Do not let your compiler contract the float math.** Implicit FMA contraction in
   the scalar mirrors (or reassociation anywhere) breaks SIMD≡mirror equality. Build the library with `-ffp-contract=off` (GCC/Clang) or `/fp:precise` (MSVC) — the
   shipped CMake does this. **Compile flags are the only reliable mechanism: under
   clang fast-math, source-level pragmas (`float_control`, `clang fp contract(off)`,
   `STDC FP_CONTRACT`) do NOT stop backend fusion — verified at the compiler.** In
   Unreal Engine, set `FPSemantics = FPSemanticsMode.Precise` on the module that
   compiles these sources. Ship a mirror-equality test in your integration; it is the
   tripwire that catches a toolchain silently breaking this.
2. **Respect the format's content rules** (zero pad lanes, finite values, non-negative
   scales) — validate at load with `ValidateBankData`, not per query.
3. **Feed identical bytes.** Determinism is a property of bank + query bytes. If you
   re-derive queries through code that is itself nondeterministic, that is upstream of
   this library.

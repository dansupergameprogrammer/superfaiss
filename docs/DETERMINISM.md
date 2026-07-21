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
| Per-row bias (v2.1): same bias => bit-identical composed ranking; null bias == unbiased bitwise | Yes (one fused add in fixed row order; the bias is part of the query). All-zeros bias is compare-equal, NOT bitwise: IEEE -0.0 + 0.0 == +0.0 |
| Same results across devices with the same active SIMD path | Yes in practice for int8 (integer→float conversion + pinned float ops are IEEE-determined), but **not contractually claimed** in the default mode |
| Same results across different SIMD paths (e.g. AVX2 desktop vs NEON phone) | Default mode: **no** (different accumulation widths round differently; per-device only). `Exactness::CrossDevice` (v2.2, int8 banks): **yes — contractual**; see section 2c. |

Consoles are fixed hardware: "per device" is effectively "per SKU" there, which is the
strong version of the default promise.

## 2c. CrossDevice mode (v2.2): bit-exactness across machines

`QueryParams::exactness = Exactness::CrossDevice` opts a query into a stronger
contract: **identical bank bytes + identical query bytes ⇒ bit-identical scores and
bit-identical hit order on any machine at any SIMD width** — x86 and ARM, Windows,
Linux, and macOS, scalar through AVX2 and NEON. This is the property lockstep and
rollback multiplayer, networked motion matching, and server-side validation require.

Why it holds:

- **Integer accumulation.** The float kernels' one cross-device hazard is reduction
  shape (8-lane AVX2 vs 4-lane NEON vs scalar sum different rounding orders). In
  CrossDevice mode the query is quantized to int8 (symmetric per-query scale — the
  row-bake math) and scoring accumulates int8×int8 products in int32. Integer
  addition is associative: every width produces the same sums. L2 uses the expanded
  three-sum form (`qs²·Σq² + rs²·Σr² − 2·qs·rs·Σq·r`), because query and rows carry
  different scales; all three sums are exact integers.
- **Fixed-order double epilogue.** Everything after the integer sums — dequant
  scales, per-channel inverse sub-norms, segment weights (applied at combine;
  CrossDevice never folds weights into the query), bias — is one fixed-order
  double-precision expression per row. Every intermediate double is provably a
  normal value, so per-thread FTZ state can never flush one.
- **The subnormal floor, by contract.** FTZ/DAZ (flush-to-zero / denormals-are-zero)
  is per-thread hardware state this library neither controls nor observes, and
  platform defaults differ. The contract closes the hole: **any final score with
  magnitude below `FLT_MIN` is exactly `0.0f`, on every machine.** At or above
  `FLT_MIN` the single double→float conversion produces a normal float, untouched
  by FTZ. Float inputs (bank scales, inverse sub-norms, weights, biases) are decoded
  from their IEEE bit patterns, so DAZ cannot flush a subnormal input either.
- **Round-half-even in integer math.** The query quantizer's rounding is implemented
  on the bit pattern — no FP rounding-mode dependence, no library rounding call.
- **Proof in CI.** The test suite computes a hash over full hit lists (order
  included) from committed fixture banks — including adversarial tiny-scale banks
  whose scale products straddle the subnormal window — under every kernel path the
  hardware can force, and asserts it equals a golden pinned in the repo. Windows,
  Linux, and macOS-ARM runners all assert the same pin on every push. Three sibling
  batteries carry their own pins the same way: the scratch × cross-device battery
  (v2.3 — a copied-bytes baked twin of a scratch snapshot, tombstones and a `Grow`
  in its history, scoring bit-identically to the snapshot), the pooled-query
  battery (v2.4 — `MakeCentroidCrossDevice` payloads and their `QueryXd` hit lists
  over the committed fixtures), and the analytics battery (v2.5 — `ScoreXdPair`,
  set-to-set distance, nearest-neighbour divergence, and spread over the committed
  fixtures, the cosine pair's `sqrt` limb included; see §2e).

Scope and cost, stated plainly:

- **Int8 banks only.** f32 banks stay per-device (`InvalidArgument` in this mode).
- **Query quantization adds error beyond row quantization.** Measure recall in this
  mode per bank before adopting; the suite and the UE importer report it beside
  standard recall.
- **CrossDevice scores differ from default-mode scores** (different math). The
  contract is cross-device identity within the mode, not equality with the default.
- **L2 scores can round to tiny negative values** where the true distance is ~0
  (the expanded form's cancellation). They are bit-identical everywhere; the floor
  maps (−FLT_MIN, FLT_MIN) to exactly 0.0f.
- **`paddedDims` is capped at 131072** in this mode, which keeps every int32
  accumulator overflow-free (`InvalidArgument` beyond it).
- **Identity is over bank bytes.** Bake once and ship the bytes (the interchange
  posture the format already takes); the mode does not promise that two machines
  BAKING the same floats produce identical banks — runtime bake still runs in the
  platform's FP environment.
- **The default FP rounding mode (round-to-nearest-even) is assumed**, as it is for
  the entire library; FTZ/DAZ are the states this mode is explicitly immune to.

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

Channels (v3.0) ride the same guarantee: a scratch bank's channel table
and its per-channel inverse sub-norms — computed per-row-standalone at append
from the quantized bytes, re-derived identically on `Load`, at `Freeze`, and
after a `Relabel` (v3.1) — make a named-channel query on a snapshot
bit-identical to the same query on a bank baked from the same live rows under
the table currently in force.

One quiescence caveat (v2.3): `MeasureScratchRecall` sweeps many self-queries
under a reader pin, not exclusivity, so its number is well-defined — and
reproducible given history — only when no writer runs concurrently; a racing
append/remove yields a safe but non-reproducible number (atomic value reads,
never undefined behavior).

## 2d. Integer-domain pooling (v2.4): a versioned composition operator

`MakeCentroidCrossDevice` pools int8 rows into a quantized CrossDevice query with
order-free int64 accumulation and a direct integer-domain requantization (no float
norm reduction exists on the path). Given identical row bytes, scales, indices, and
weights, its product — image, scale, self-dot — is bit-identical on any machine, and
`QueryXd` executes exactly those bytes. **It is a versioned composition operator: any
change to its accumulation, epilogue, or quantization is a space-version change for
consumers** — embedding-space version machinery (stamping, mismatch rejection) lives
consumer-side; treat a change here exactly like re-baking the space.

## 2e. Bank analytics (v2.5): versioned composition operators

The v2.5 analytics score int8 CrossDevice banks with the same integer accumulation and
fixed-order double epilogue as the query path: `ScoreXdPair` (the query-vs-query pair
score the others rest on), `CentroidDistanceCrossDevice` (set-to-set distance — drift over
checkpoints is this operator between two checkpoints' row sets), `MeanNNCrossDevice` /
`MaxNNCrossDevice` (directed nearest-neighbour set divergence), and `SpreadCrossDevice`
(within-bank centroid-dispersion). Given identical bank bytes, scales, indices, and
exclusion words, each scalar is bit-identical on any machine — the mean accumulates in
fixed ascending row-index order, the max is order-free. **They are versioned composition
operators, exactly as §2d: any change to a reduction's accumulation, epilogue, or
quantization is a space-version change for consumers.**

One new arithmetic operation enters the epilogue here, and it carries a build condition.
The cosine distance is `1 − crossDot / sqrt(a.selfDot · b.selfDot)` — the one operation
outside the `{+,−,×,÷}` epilogue the other cross-device math uses. Its cross-device identity
requires a **true, IEEE-754 correctly-rounded `sqrt`: no `rsqrt` approximation, no fast-math
reciprocal-square-root substitution.** The §3.1 obligation (build with `-ffp-contract=off` /
`/fp:precise`, never fast-math) is what enforces it — fast-math is the setting that would
substitute an `rsqrt` and diverge cosine scores across machines, and `-ffp-contract=off`
alone does not stop that substitution. The core suite proves the result: the analytics
battery carries its own pinned hash (`kGoldenAnalyticsXdHash`), asserted bit-identical across
the CI matrix and across MSVC / GCC / AppleClang, the cosine pair (with its `sqrt`) included.

Channel-scoped analytics (v3.0) are the same operators over a sub-range and carry the same
contract per channel. Dot/L2 are a sub-range of the identical integer accumulation. The
Cosine limb recomputes the channel sub-range's integer self-dot and applies the same true
IEEE-754 correctly-rounded `sqrt` per channel — `1 − crossDot / sqrt(aSq·bSq)` over
`[offset, offset+length)`, NOT the per-row `channelInvNorms` — so it inherits the build
condition above (no `rsqrt`, no fast-math) once per channel. The per-channel cosine `sqrt`
is proven across the CI forced-path + cross-runner golden battery with an adversarial
tiny-channel-norm member, and a bit-exact Cosine channel-analytics REF (an independent
integer-domain recode of the epilogue) asserts the operator matches the contract.

`ProjectionReport` (v2.5) is per-device float only — an offline authoring/inspection tool
with no cross-device claim — and stands outside this contract.

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

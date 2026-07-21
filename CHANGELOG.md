# Changelog

All notable changes to SuperFAISS. Versions follow the git tags; each entry lists the
capabilities that release added over the one before it. Exactness, per-device
bit-reproducibility, and allocation-free steady state are invariants across every
release — the contract is in [docs/DETERMINISM.md](docs/DETERMINISM.md), not repeated
per entry. Reconstructed from git history 2026-07-12.

The format follows [Keep a Changelog](https://keepachangelog.com); this project versions
by feature tier (minor = new capability, patch = fix), not strict SemVer of a public ABI.

## [3.3.0] — 2026-07-21

### Added
- **`PeekScratchArchive`** — reads a serialized scratch archive's header and channel
  table out of a byte span and reports its geometry plus `archiveBytes`, exactly the
  number of bytes a `Load` consumes, without allocating or reading the payload. It runs
  the same header validation `Load` does (the two now share one validator, so they cannot
  drift). A host that appends its own trailer after the archive can therefore locate and
  validate that trailer *before* committing the load, rather than discovering a broken
  trailer with the rows already replaced.

### Changed
- **`ComputePrincipalComponents`'s `scratch` parameter is now `double*`** (was `float*`).
  The covariance-apply accumulator now matches the mean's accumulation precision — a
  float accumulator rounded once per row and degraded the power iteration with bank
  size — and the caller-owned scratch buffer changed type to match. This is
  source-breaking: every existing caller must change the parameter's declared type from
  `float*` to `double*` **and** double the size of the buffer it allocates (`dims`
  doubles, not `dims` floats).
- **`MeanNNCrossDeviceChannel`/`MaxNNCrossDeviceChannel` can now return `OutOfMemory`.**
  Both now stage a pre-lift of the target's non-excluded sub-rows into the `Workspace`
  argument (so each target's self-dot is computed once rather than once per source row,
  see the self-dot hoist below); if that reservation fails, the call now returns
  `OutOfMemory`, a status neither function could previously return.

### Fixed
- **`ScratchBank::Load` now validates the retention region.** A retained row is by
  construction the post-normalization row the quantizer consumed, so `Load` replays the
  bake on each retained row and requires it to reproduce the stored row exactly —
  allocation-free, mirroring the quantizer's own arithmetic. Previously a
  fabricated-but-finite retained array loaded clean and handed `MeasureScratchRecall` an
  invented reference to audit against, through the one API whose whole job is an honest
  number. Non-finite retained values are rejected the same way (`BadFormat`).
- **Two kernel-selection rules made the segmented scan disagree with the whole-row
  scan.** The float32 dispatchers `DotF32`/`L2F32` use AVX2+FMA only when the length they
  are handed is a multiple of 8; `ResolveRowKernels`, which feeds the segmented scan and
  `DecomposeRowScore`, wired the AVX2 kernels in directly with no such rule. Float32
  strides are multiples of 4, so at `paddedDims` 20, 36, 52, 100, 132 and similar the two
  paths computed the same row with different accumulation widths and disagreed in the
  last ulp. That falsified two published guarantees — that a degenerate one-segment query
  equals the whole-row scan, and that decomposition contributions sum bit-exactly to the
  scan's own score, "no second code path exists to drift". `ResolveRowKernels` now holds
  the dispatchers, so the rule is applied once, to the length actually in hand — which
  matters because the segmented scan calls the kernel per scan RANGE, and a bank whose
  stride is a multiple of 8 can still present a range that is not.
  **`Exactness::CrossDevice` was never affected** (int8 lengths are always multiples of
  16), so the cross-machine contract is unchanged, and no pinned golden moves: every one
  of them sits at a width or quantization where the two paths already agreed.
- **`ScratchBank::Create` and `Grow` performed signed size arithmetic before bounding
  their geometry.** `ArenaBytes` multiplies capacity by dims in signed `int64`; both
  entry points accepted any positive `int32` pair, so a large valid-typed request
  overflowed the arena computation — undefined behavior rather than a refused
  allocation. Both now apply the format's own ceilings (`kMaxBankRows` rows,
  `kMaxCrossDeviceDims` dims) before any size is computed, returning `InvalidArgument`.
  The archive loader already applied these caps; direct construction did not, and
  `Grow` bounded its request only against the current capacity.
- **Bank-inspection review follow-ups (lower-severity, from a whole-project pass).**
  `graph.cpp`: `MutualFilter`/`ConnectedComponents` bound caller-supplied neighbour and
  duplicate-group values to `[0, count)` before indexing the neighbour list and the
  union-find scratch, so a malformed hand-built input degrades to a dropped edge rather
  than an out-of-bounds access. `analytics.cpp`: the int8 off-grid channel guard from
  `novelty.cpp` now also protects the channel-analytics legs (added once in the shared
  `ChannelSubRange`), and the channel NN divergence pre-lifts each target's self-dot once
  instead of once per source row. `pca.cpp`: `ProjectRowsOntoComponents` bounds
  `componentCount` by `dims` like its sibling. Scale decode goes through
  `detail::FloatBitsToDouble` on the remaining `compose.cpp`/`pca.cpp`/`analytics.cpp`
  paths, so a subnormal scale decodes identically under any FTZ/DAZ mode. A comment
  documents that `novelty.cpp`'s `sampleLimit` is a ceiling, not a down-sampler.
- **A Cosine channel bank with zero rows was unrepresentable.** `ValidateBank` required
  the per-channel inverse sub-norm array whenever a Cosine bank carried channels, but
  those norms are one per row — a bank with no rows requires none, and no scan reads the
  array at zero rows. The rule now applies only when `count > 0`. The shape is not
  degenerate: it is what a channel-carrying scratch bank graduates into once every row
  has been removed, and rejecting it forced that graduation to drop its channels.

## [3.2.1] — 2026-07-21

### Fixed
- **Test suite built and ran only on x86 Windows.** Two defects introduced with the
  3.2.0 allocation-coverage cells, both in `tests/test_main.cpp`; the shipped library
  (`include/`, `src/`) is byte-identical to 3.2.0 and unaffected.
  - The kernel allocation cell referenced the `*ScalarAvx2` reference mirrors without
    an architecture guard. They are compiled out on ARM, so the suite failed to LINK
    on macOS arm64 rather than failing a test.
  - The header-only-math allocation cell sized its decode buffer to `dims` where
    `DequantizeRowAsQuery` writes `paddedDims` and zeroes the tail — an 8-float stack
    overrun on a 24-dim int8 bank, which aborted the suite under glibc's stack
    protector on both Linux jobs. Confirmed with AddressSanitizer, and the buffer is
    now sized from `constexpr PaddedDims` so it tracks the contract.
- Coherence checks added to this repository's CI (documented signatures match their
  declaring header, every public symbol is documented, Markdown links resolve
  case-exactly). They previously ran only in the consuming plugin repository against a
  vendored copy, so this repo could publish a doc contradicting its own headers and
  stay green.

## [3.2.0] — 2026-07-20

### Fixed
- **Stale public contracts scrubbed (docs/comments only; no code change).** `docs/API.md`,
  `docs/DETERMINISM.md`, `docs/FORMAT.md`, and `scratch.h`'s `GetChannelCount()`/`GetChannels()`
  doc comment still described the v3.0 channel table as fixed for the bank's lifetime — the
  contract v3.1's `Relabel` repealed. `docs/API.md` and `scratch.h` were missed by the v3.1.2
  scrub below; `docs/DETERMINISM.md` was never on that list. All four now state the current
  contract.

### Added
- **Bank-inspection primitives (`inspector_common.h`, `graph.h`, `novelty.h`, `matching.h`)
  — Bank Inspector I.** Four header-only Tier-1 modules answering "what does this bank
  actually look like," not just "what's nearest": a shared row->query decode helper
  (`inspector_common.h`); a mutual k-NN neighbor graph + connected components
  (`graph.h` — clustering); a two-limb novelty test, an exact-distance identity check
  plus a statistical rank against a calibrated baseline (`novelty.h` — is this row new or
  a near-duplicate); sampled-A-verified-against-full-banks mutual correspondence with
  CSLS margins (`matching.h` — which rows in bank A correspond to rows in bank B, e.g. a
  player's saved scratch archive against the shipped reference bank). The three module
  headers are pure functions over a caller-held `BankView` — no new bank state, no
  persistence surface, PER-DEVICE deterministic, no cross-device claim. `superfaiss.h`'s
  umbrella header now includes all four. See `docs/API.md` for the full contract of each.

## [3.1.2] — 2026-07-18

### Fixed
- **Stale public contracts scrubbed (docs/comments only; no code change).** `scratch.h`,
  `scratch.cpp`, and `docs/INTEGRATION.md` still described the channel table as
  "fixed for the bank's lifetime" — the v3.0 contract that v3.1's `Relabel` repealed. These three
  now state the current contract, and `Relabel` is documented in `INTEGRATION.md` (it had
  landed in the README and this changelog only). `docs/API.md` still carried the stale phrasing
  after this pass and was corrected in 3.2.0, above.
- **`version.h` corrected to the released version.** The header (and its coherence check in the
  test suite) still carried 3.0.1 through both 3.1 releases; both now read 3.1.2. Found while
  closing an external review's release-identity finding.

## [3.1.1] — 2026-07-18

### Fixed
- **AVX2 float32 dot/L2 dropped a sub-8 segment remainder.** `DotF32Avx2`/`L2F32Avx2` (and
  their scalar mirrors) accumulated only in whole 8-lane groups with no remainder; a
  float32 segment/channel stride lies on the 4-float (16-byte) grid, so a range whose
  length is congruent to 4 mod 8 had its trailing 4 elements dropped entirely — a
  length-4 channel scored exactly 0 on AVX2 while SSE/portable/NEON scored it correctly.
  **This changed numeric results for affected widths**: `ResolveRowKernels` wires the
  AVX2 kernels in directly for the segmented and per-channel-cosine scan, so any
  named-channel or segmented query over a sub-8-remainder length (e.g. a length-4
  channel) on AVX2 hardware returned a different, wrong score before this fix — a
  consumer bisecting a score change across 3.1.0 to 3.1.2 should attribute the delta at
  those widths to this fix, not to 3.1.0's `Relabel`. Whole-row scans were unaffected
  (`DotF32`/`L2F32` route non-multiple-of-8 `paddedDims` to the SSE path). The 4-element
  remainder is now added to both the intrinsics and their scalar mirrors, bit-identically.
- CI now requires the AVX2 path on the Linux job (previously accepted SSE or AVX2), so a
  runner without AVX2 hardware cannot silently skip the new sub-8 remainder coverage;
  Windows keeps the looser SSE-or-AVX2 check as the general x86 path.

## [3.1.0] — 2026-07-17

### Added
- **Runtime-mutable channel vocabulary (`Relabel`)** — the channel table set at `Create`, fixed
  for the bank's lifetime in v3.0, becomes mutable at runtime through one exclusive
  `Relabel(newChannels, newChannelCount)` on a live scratch bank. A single atomic table swap
  covers every vocabulary change: move boundaries at a fixed count, change the count, add or
  remove channels, promote a single-space bank into channels (`channelCount 0 → N`), or demote a
  channel bank back to single-space (`→ 0`). It never touches the stored rows — channels are
  sub-ranges over the same dims, so no row is re-quantized or re-normalized; on a Cosine bank it
  re-derives the `capacity × channelCount` per-channel sub-norm arena over the unchanged rows
  (the same derivation `Load` performs), while Dot/L2 banks relabel by a by-value member swap
  with no arena work. `Relabel` is atomic reject-over-degrade: the new table is validated with
  the `Create`/`Load` rules and the arena is allocated before any state changes, so a malformed
  table (`InvalidArgument`) or an allocation failure (`OutOfMemory`) leaves the bank byte-for-byte
  intact and still queryable under the old table. It runs under the same exclusive reader-drain as
  `Grow`/`Load`, joins their `BankView` invalidation set, and advances the generation. A relabeled
  bank is bit-identical to a fresh `Create(newTable)+Append` of the same rows, and its archive is
  unchanged in shape — no new on-disk format version. Measured: a count-change relabel of a
  100k × 256 Cosine bank costs ~30 ms against ~166 ms to rebuild from the rows on int8
  (~5.5× cheaper), ~2.1× on float32; promote/demote adds/frees a `capacity × channelCount × 4`
  sub-norm arena (~3.0 MB at 100k rows / 8 channels).

## [3.0.1] — 2026-07-13

### Fixed
- **`version.h` still read 2.5.0 at the v3.0 tag.** Corrected to 3.0.0, with a coherence
  test added so a future release cannot ship the same drift.
- **`MeanNNCrossDeviceChannel`/`MaxNNCrossDeviceChannel` rejected a zero-energy Cosine
  source and skipped zero-energy targets**, contradicting the documented "a degenerate
  member floors to a defined 0 in a reduction" contract already honored by
  `CentroidDistanceCrossDeviceChannel`/`SpreadCrossDeviceChannel`. Both guards are
  removed so a degenerate channel member floors to 0 through the shared pair score,
  consistent across all four channel operators.
- **`MeasureRecallLockedChannel` aborted the whole per-channel recall audit** when a
  sampled row's channel sub-vector was zero-energy — a valid row whose per-channel
  self-query is `ZeroNormQuery`. It now excludes that sample, refills the reservoir from
  the remaining rows, and reports the sample count actually scored rather than the
  pre-set target.

## [3.0.0] — 2026-07-12

### Added
- **Channel-capable scratch banks** — the channel table (a bank-side partition of the row
  into named sub-ranges, `ChannelInfo` = offset + length, up to `kMaxChannels = 8`) becomes
  a mutable-bank property set at `Create` and fixed for the bank's lifetime. A new `Create`
  overload validates the table with the importer's rules (in-bounds, ascending,
  non-overlapping, on the 16-byte element grid). On a Cosine bank, `Append` computes the
  appended row's per-channel inverse sub-norms per-row-standalone from the quantized row and
  folds a `capacity × channelCount` sub-norm arena into the single allocation (Dot/L2 channel
  banks need none). `Snapshot` carries the table, so every query and analytics entry point
  serves named-channel and raw-range queries on a scratch snapshot with the exact code baked
  banks use. Channel-aware `Freeze` graduates a channel scratch bank to a `schemaVersion 2`
  baked asset, re-deriving the per-channel sub-norms over the compacted rows. Per-channel
  recall audit (`MeasureScratchRecallPerChannel`, and the channel-aware `FreezeWithRecall`)
  reports recall@k per channel over each sub-range.
- **Channel-scoped analytics** — every v2.5 analytics operator gains a channel-scoped form
  (`CentroidDistanceCrossDeviceChannel`, `MeanNNCrossDeviceChannel`, `MaxNNCrossDeviceChannel`,
  `SpreadCrossDeviceChannel`) that pools/scores over one channel's sub-range instead of the
  whole row — new surface for baked and scratch banks alike. "This mind's identity is drifting
  but its appearance is stable" is a channel-scoped centroid distance over the identity channel
  versus the appearance channel. Dot/L2 are a sub-range of the same integer accumulation;
  Cosine recomputes the sub-range integer self-dot and applies one IEEE correctly-rounded
  `sqrt`, cross-device bit-exact per channel (it does not read the per-row `channelInvNorms`).
  A degenerate (zero-sub-norm) channel member floors to a defined `0` in a reduction, while a
  single per-channel query on a zero-norm sub-vector is still rejected (`ZeroNormQuery`).

### Changed
- The scratch-bank archive gains a **presence-flags byte** (`reserved[0]`, header offset 26:
  bit 0 retention, bit 1 channels) and writes **version 3** when the bank carries channels; a
  channel-less bank still writes version 1/2 byte-identically. The channel table is serialized
  when its flag is set; the per-channel sub-norm arena is NOT serialized — it is re-derived on
  `Load` from the quantized rows + channel table + scales and asserted bit-equal to a fresh
  derivation. Old v1/v2 readers hard-reject version 3.

## [2.5.0] — 2026-07-12

### Added
- **Bank analytics** — cross-device deterministic reductions over int8 banks: set-to-set
  centroid distance (drift over checkpoints is this operator between two checkpoints' row
  sets), directed mean/max nearest-neighbour divergence, and within-bank spread (centroid
  dispersion), all resting on a shared query-vs-query pair score (`ScoreXdPair`). The
  cosine limb adds a runtime IEEE-754 correctly-rounded `sqrt` (no `rsqrt`/fast-math),
  proven bit-identical across the CI matrix by its own pinned golden.
- **Probe-direction projection report** — per-device float: projects a bank's rows onto a
  probe direction with a Cohen's-d group-separation statistic; offline authoring/audit
  tooling, no cross-device claim.

### Changed
- `docs/DETERMINISM.md` §2e documents the analytics operators as versioned composition
  operators and states the cosine `sqrt` build condition.

## [2.4.0] — 2026-07-11

### Added
- **Integer-domain pooling** (`MakeCentroidCrossDevice` + `QueryXd`) — pools int8 rows into
  a quantized cross-device query with order-free int64 accumulation and integer-domain
  requantization; bit-identical across machines given the same rows. A versioned
  composition operator.
- **Pre-quantized cross-device batch** (`QueryXdBatch`) — many pooled cross-device queries
  in a single bank pass.

## [2.3.0] — 2026-07-11

### Added
- **Scratch-bank recall audit** (`MeasureScratchRecall`, opt-in float retention) — a
  reproducible recall@k over a live scratch bank, generation-stamped so a report taken
  before a later append/remove reads as stale rather than silently current.
- **Immutable-format geometry ceilings** — over-cap headers (`dims` > 131072 or
  `count` > 2^28) are hard-rejected on the header fields alone, before any payload-size
  arithmetic runs.

### Fixed
- Trust-boundary validation of cross-device query payloads (finite scale, matching
  self-dot); Load-generation monotonicity.

## [2.2.1] — 2026-07-04

### Fixed
- Workspace-reuse stride bug in folded batch/intersect queries (external report).

## [2.2.0] — 2026-07-04

### Added
- **Cross-device exactness** (`Exactness::CrossDevice`) — opt-in bit-identical scores and
  hit order across machines and SIMD widths (x86/ARM, any OS): the query is quantized to
  int8 and scored through integer accumulation with a fixed-order double epilogue, closed
  by a subnormal-floor contract.

## [2.1.0] — 2026-07-04

### Added
- **Per-row bias** — dense or sparse per-row score bias applied in-scan (not as a post-sort
  reweight), so a composed ranking is exact; the sparse form is effectively free at any
  scale (the motion-matching continuing-pose shape).

## [2.0.0] — 2026-07-04

### Added
- **Segmented queries and per-channel cosine** — score a weighted slice of the vector by
  named channels; true per-channel cosine on channel banks.
- **Per-row decomposition** — a channel-by-channel breakdown of why a row scored, summing
  exactly to the score the scan produced.
- **Scratch banks** — mutable append/remove/snapshot/freeze/serialize banks that grow at
  runtime and survive a save game, under a seq_cst reader pin/drain protocol (lock-free
  readers, one logical writer).

## [1.1.0] — 2026-07-03

### Changed
- First version-stamped release: `version.h` corrected (it had lagged at 0.1.0 through the
  1.0.x tags — caught at the scrub gate). Bias, API, and determinism documentation brought
  current.

## [1.0.1] — 2026-07-03

### Fixed
- Batch/singles equivalence proven across the full metric × quantization matrix.
- Corrected FP-contraction guidance: compiler flags (`-ffp-contract=off` / `/fp:precise` /
  UE `FPSemantics.Precise`) are the reliable mechanism — source-level pragmas do **not**
  defeat clang fast-math backend fusion.

## [0.1.0] — 2026-07-03

### Added
- Initial implementation: baked bank format and bake math, deterministic exact top-k,
  the query/batch API, and scalar + SSE + NEON + AVX2 kernels with per-path scalar mirrors
  (SIMD ≡ mirror, bit-equal). Multi-arch CI (Windows/Linux x64, macOS arm64), a bank-content
  validation gate, a GloVe terminal demo, and the sidecar converter. Dependency-free C++17.

> Note on early version numbers: `version.h` read `0.1.0` through the initial work and the
> `v1.0.1` tag; the `v1.1` release corrected it. The `[0.1.0]` entry above is that initial
> tagged line (first tag `v1.0.1`), listed by its actual `version.h` value at the time.

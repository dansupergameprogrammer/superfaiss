# Changelog

All notable changes to SuperFAISS. Versions follow the git tags; each entry lists the
capabilities that release added over the one before it. Exactness, per-device
bit-reproducibility, and allocation-free steady state are invariants across every
release — the contract is in [docs/DETERMINISM.md](docs/DETERMINISM.md), not repeated
per entry. Reconstructed from git history 2026-07-12.

The format follows [Keep a Changelog](https://keepachangelog.com); this project versions
by feature tier (minor = new capability, patch = fix), not strict SemVer of a public ABI.

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

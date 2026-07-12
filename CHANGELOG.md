# Changelog

All notable changes to SuperFAISS. Versions follow the git tags; each entry lists the
capabilities that release added over the one before it. Exactness, per-device
bit-reproducibility, and allocation-free steady state are invariants across every
release — the contract is in [docs/DETERMINISM.md](docs/DETERMINISM.md), not repeated
per entry. Reconstructed from git history 2026-07-12.

The format follows [Keep a Changelog](https://keepachangelog.com); this project versions
by feature tier (minor = new capability, patch = fix), not strict SemVer of a public ABI.

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

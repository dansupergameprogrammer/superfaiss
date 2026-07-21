# Contributing

## Scope

SuperFAISS is a header-and-source C++ library for deterministic, allocation-
free k-nearest-neighbor search over baked embedding banks. See `README.md`
for the library's contract and `docs/` for the API, format, integration, and
determinism references.

## Before opening a pull request

- Build and run the test suite: `cmake -B build && cmake --build build`,
  then run the resulting `superfaiss_tests` binary. It must report a clean
  pass, including the cross-device hash line and the active SIMD path.
- Match the existing code's determinism and allocation-free conventions —
  `docs/DETERMINISM.md` and `docs/INTEGRATION.md` state the guarantees a
  change must not silently break.
- Update `docs/API.md` alongside any signature change. `docs/API.md`'s
  documented signatures are checked against the headers that declare them
  by downstream consumers; a drifted signature is a shipped defect, not a
  cosmetic one.
- Keep commits scoped to one change; write commit messages that describe the
  change, not the process of making it.

## Required CI checks

Every pull request must pass the `tests` workflow (Windows, Linux, Linux
ThreadSanitizer, and macOS ARM64 jobs — see `.github/workflows/tests.yml`).

## Versioning

`include/superfaiss/version.h` and the top entry of `CHANGELOG.md` must
always agree with each other and with the git tag on a release commit.

## License

By contributing, you agree that your contribution is licensed under this
repository's license (see `LICENSE`).

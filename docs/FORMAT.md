# The Bank Format

A **bank** is an immutable set of embedding vectors prepared for scanning. This document
specifies both the interchange format (what pipelines produce) and the baked in-memory
layout (what the kernels scan). The format is versioned by `schemaVersion`; consumers
must hard-reject versions they do not know. Current versions: **1** (channel-less)
and **2** (named channels).

## 1. Interchange: the `.wvbank` sidecar pair

Two files, deliberately primitive so any pipeline can emit them in a few lines
(reference implementation: [`tools/wvbank.py`](../tools/wvbank.py)):

**This sidecar pair is the encoder contract.** SuperFAISS ships no encoders and never
will — anything that turns domain data (text, images, gameplay state) into vectors is
an encoder, and any pipeline that emits this pair *is* one. `tools/wvbank.py` (GloVe
text vectors) is the reference implementation of the contract, not a privileged path.
Everything domain-specific lives on the producer's side of this boundary; the library
and its engine integrations consume banks and query vectors, and never know or care
where the vectors came from.

### `<name>.wvbank.json` — the header

```json
{
  "schemaVersion": 1,
  "dims": 100,
  "count": 40000,
  "metric": "cosine",
  "dtype": "float32",
  "description": "optional free text",
  "ids": ["optional", "one", "string", "per", "row"],
  "channels": [{"name": "identity", "offset": 0, "dims": 64},
               {"name": "appearance", "offset": 64, "dims": 36}]
}
```

| Field | Type | Rules |
|---|---|---|
| `schemaVersion` | int | 1 (channel-less) or 2 (carries `channels`). Unknown versions are rejected, never guessed at; a v1-only reader hard-rejects 2 by this same rule. Writers emit 1 unless channels are present. |
| `dims` | int | > 0. Logical dimensions per vector. |
| `count` | int | >= 0. Number of vectors. |
| `metric` | string | `"dot"`, `"cosine"`, or `"l2"` — the metric the bank is intended for. |
| `dtype` | string | `"float32"` (the interchange payload is always float32; quantization happens at bake). |
| `ids` | string[] | Optional. Exactly `count` entries, all unique. Absent means IDs are row indices. |
| `channels` | object[] | schemaVersion 2 only (and required by it). Named contiguous element ranges `{name, offset, dims}`: non-overlapping, ascending, at most 8, names unique, boundaries on the baked layout's 16-byte element grid (section 2). Full coverage is not required — unnamed gaps are simply unaddressable by name. |

### `<name>.wvbank.bin` — the payload

Raw little-endian float32, row-major, exactly `count * dims * 4` bytes. No header, no
padding, no compression.

### Validation rules (importer obligations)

An importer must reject, with a specific diagnostic and no partial output:
payload size disagreeing with the header; any non-finite value; duplicate or
wrongly-counted ids; a zero-norm row when `metric` is `cosine`; unknown
`schemaVersion`, `metric`, or `dtype`; a malformed `channels` table (overlap,
off-grid boundary, duplicate or empty name, out-of-bounds range, more than 8
entries, or schemaVersion 2 without one).

## 2. Baked layout: what the kernels scan

Baking transforms interchange float32 into the scan-ready form. The library provides
every step (`bake.h`): `ValidateSourceRows` → `NormalizeRows` (Cosine only) →
`PadRowsFloat32` or `QuantizeRowsInt8`.

### Row storage

- Rows are stored row-major with a stride of `PaddedDims(dims, quantization)` elements:
  `dims` rounded up so the row stride is a multiple of **16 bytes** (4 floats, or
  16 int8s). Pad lanes are **zero** — this is a content rule, not a suggestion;
  `ValidateBankData` rejects violations (a non-zero pad lane silently corrupts L2).
- The base pointer of the row block must be **16-byte aligned**.

### Quantization (`Int8`)

Symmetric, per-row: `scale = maxAbs / 127`, `q[i] = round(v[i] / scale)` clamped to
[-127, 127]; dequantized value is `q[i] * scale`. A zero row stores `scale = 0` with
all-zero lanes. Scales must be finite and non-negative. Typical quality: recall@10
≈ 0.99 vs the float32 source on real embedding data — measure per bank (the reference
UE importer records a seeded recall report on every asset).

### Cosine banks are pre-normalized

`metric = cosine` banks store L2-normalized rows, so query-time scoring is a plain dot
product. Consequences: zero-norm *rows* are a bake-time error; zero-norm *queries*
are a query-validation error; there is no runtime normalization branch.

### Queries

A query is float32, padded to the bank's `paddedDims` with **zeros**, 16-byte aligned.
Non-finite values and non-zero pad lanes are validation errors — NaN never reaches a
kernel (a NaN score would break the ranking's total order).

### Scores

`dot`/`cosine`: higher is better. `l2`: the score is the **squared** L2 distance;
lower is better. Ties rank by ascending row index, always.

## 3. Scratch-bank archives (v2.0)

`ScratchBank::Save`/`Load` stream a self-contained snapshot of a scratch bank
through the caller-provided `ScratchArchive` seam (the caller owns the medium —
file, save-game blob, network; the bank owns the format):

| Field | Type | Notes |
|---|---|---|
| magic | u32 | `0x42535346` |
| version | u32 | `1` |
| capacity | i32 | arena capacity restored on load |
| count | i32 | published rows; `0 <= count <= capacity` |
| dims | i32 | logical dims |
| paddedDims | i32 | must equal `PaddedDims(dims, quant)` |
| metric, quant | u8, u8 | enum values; 6 reserved bytes follow |
| rows | payload | `count x paddedDims` elements, baked layout (section 2) |
| scales | f32[count] | int8 banks only |
| tombstones | u32[ceil(count/32)] | bit set = removed row |

Load is reject-over-degrade: a bad magic/version, inconsistent header,
truncated payload, tombstone bit at or above `count`, or content that fails
bank-data validation rejects the archive and leaves the existing bank
unchanged. Byte order is the host's; archives are save-game-grade state, not a
cross-endian interchange format (the `.wvbank` sidecar is the interchange).

## 4. Forward compatibility

Consumers embedding this format in their own asset containers should reserve a
versioned, skippable block for future index structures (approximate-search
acceleration), so adding them later is additive rather than format-breaking. The
[reference UE plugin](https://github.com/dansupergameprogrammer/superfaiss-unreal) does exactly this.

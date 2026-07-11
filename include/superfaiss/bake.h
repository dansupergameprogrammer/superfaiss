#pragma once

#include "types.h"

namespace superfaiss
{

// Bake-side math: importers call these to turn raw float32 source rows into bank
// payloads. All functions are pure and single-threaded.

// L2-normalize rows in place. Returns ZeroNormRow (and writes the offending row index
// to outBadRow if non-null) if any row has zero norm — a bake-time error for Cosine
// banks, by format rule.
Status NormalizeRows(float* rows, int32_t count, int32_t dims, int32_t* outBadRow);

// Symmetric per-row int8 quantization: scale = maxAbs/127, q = round(v/scale) clamped
// to [-127,127]. A zero row gets scale 0 and all-zero lanes (dequantizes to zero).
// Input must be finite: run ValidateSourceRows first — a NaN row would otherwise
// quantize to a silent zero row (NaN comparisons are false, so maxAbs stays 0).
// Source stride is `dims` (unpadded); destination stride is `paddedDims` with zero-filled
// pad lanes. outScales has `count` entries.
void QuantizeRowsInt8(
	const float* rows,
	int32_t count,
	int32_t dims,
	int32_t paddedDims,
	int8_t* outRows,
	float* outScales);

// Copy float32 rows from unpadded (stride dims) to padded (stride paddedDims) layout
// with zero-filled pad lanes.
void PadRowsFloat32(
	const float* rows,
	int32_t count,
	int32_t dims,
	int32_t paddedDims,
	float* outRows);

// Source-data validation for importers: every value finite. Returns the first offending
// row in outBadRow if non-null. Also enforces the format geometry ceilings
// (count <= kMaxBankRows, dims <= kMaxCrossDeviceDims -> BadFormat) ahead of every
// other check, so an importer may gate a header on it before the payload exists —
// header-derived sizes must never enter byte arithmetic unbounded.
Status ValidateSourceRows(const float* rows, int32_t count, int32_t dims, int32_t* outBadRow);

// Per-channel inverse row norms for Cosine banks carrying channels (D-V2-1): baked
// from the bank's QUANTIZED payload - the reported per-channel cosine is the cosine
// of what the kernel dots, not of the pre-quantization source. outInvNorms holds
// bank.count x bank.channelCount floats, row-major. A zero-norm row channel stores 0.
// The bank view's channels table must already be set and valid.
Status ComputeChannelInverseNorms(const BankView& bank, float* outInvNorms);

} // namespace superfaiss

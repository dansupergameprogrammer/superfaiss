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
// row in outBadRow if non-null.
Status ValidateSourceRows(const float* rows, int32_t count, int32_t dims, int32_t* outBadRow);

} // namespace superfaiss

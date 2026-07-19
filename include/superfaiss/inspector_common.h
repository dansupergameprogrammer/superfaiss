#pragma once

#include "types.h"

// SuperFAISS V3.2 — Bank Inspector I: the row->query conversion shared across every
// Tier-1 core module (plan section 25.4, temper S1 — "a named, shared step, not an
// improvisation"). `Query`/`QueryBatch` take a float query, not a stored row; an int8
// row is per-row-scaled bytes, not a valid query argument on its own. This is the
// core-side sibling of the plugin's `MakeCentroidQuery({row})` (V32-G10) — a dim-7 cell
// pins the equivalence: a row probed via this helper returns the same hits as the
// widget's single-row centroid path.
//
// Header-only (dependency-free, no new translation unit): the body is a single
// per-element loop, identical in shape to `pca.cpp`'s own row decode.

namespace superfaiss
{

// Decodes stored row `row` to a `bank.paddedDims`-length float query exactly as the
// kernels decode it: int8 is `(int8 byte) * that row's own scale`; float32 is the bytes
// as-is (pad lanes are zero in storage and copy through as zero). `outFloatQuery` must
// hold `bank.paddedDims` floats.
inline void DequantizeRowAsQuery(const BankView& bank, int32_t row, float* outFloatQuery)
{
	const int32_t pd = bank.paddedDims;
	if (bank.quant == Quantization::Int8)
	{
		const int8_t* r = static_cast<const int8_t*>(bank.rows) + static_cast<int64_t>(row) * pd;
		const float scale = bank.scales[row];
		for (int32_t d = 0; d < pd; ++d)
		{
			outFloatQuery[d] = static_cast<float>(r[d]) * scale;
		}
	}
	else
	{
		const float* r = static_cast<const float*>(bank.rows) + static_cast<int64_t>(row) * pd;
		for (int32_t d = 0; d < pd; ++d)
		{
			outFloatQuery[d] = r[d];
		}
	}
}

} // namespace superfaiss

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

// Decodes stored row `row` to a `targetPaddedDims`-length float query exactly as the
// kernels decode it: int8 is `(int8 byte) * that row's own scale`; float32 is the bytes
// as-is. `outFloatQuery` must hold `targetPaddedDims` floats.
//
// `targetPaddedDims` defaults to `bank.paddedDims` — the common case, where a module
// queries the SAME bank it decoded the row from, and every real byte plus the source's
// own zero-padding lanes copy straight through. matching.h's mutual-match crosses two
// banks that may differ in quantization (E-D1-3), so it decodes a `source` row for a
// DIFFERENT `target` bank's `paddedDims` — passed explicitly there. The tail
// `[dims, targetPaddedDims)` is always written as zero (never copied from the source's own
// storage), so the target's own pad lanes are correct regardless of `targetPaddedDims`
// (S4, plan Sec.25.4 temper S1 — "one shared helper, three callers," closing the private
// `DequantizeRowForTarget` duplicate this generalization replaces;
// Claude/Poirot/524b373-matching-m3-review.md).
inline void DequantizeRowAsQuery(
	const BankView& bank, int32_t row, float* outFloatQuery, int32_t targetPaddedDims = -1)
{
	const int32_t pd = targetPaddedDims >= 0 ? targetPaddedDims : bank.paddedDims;
	const int32_t dims = bank.dims; // pd >= dims always (paddedDims >= dims; dims validated
	                                 // equal across views at every cross-bank call site)
	if (bank.quant == Quantization::Int8)
	{
		const int8_t* r = static_cast<const int8_t*>(bank.rows) + static_cast<int64_t>(row) * bank.paddedDims;
		const float scale = bank.scales[row];
		for (int32_t d = 0; d < dims; ++d)
		{
			outFloatQuery[d] = static_cast<float>(r[d]) * scale;
		}
	}
	else
	{
		const float* r = static_cast<const float*>(bank.rows) + static_cast<int64_t>(row) * bank.paddedDims;
		for (int32_t d = 0; d < dims; ++d)
		{
			outFloatQuery[d] = r[d];
		}
	}
	// Pad lanes [dims, pd): zero, whether same-bank (bit-identical to the old direct-copy
	// form, since storage's own pad lanes are always zero) or cross-bank (the target's
	// wider tail, never assuming the source's storage past its own dims).
	for (int32_t d = dims; d < pd; ++d)
	{
		outFloatQuery[d] = 0.0f;
	}
}

} // namespace superfaiss

#pragma once

#include "types.h"

namespace superfaiss
{

// Principal components for bank inspection (dependency-free power iteration; the
// projection-visualizer substrate). Deterministic: fixed deterministic seed vector,
// fixed iteration count, serial accumulation in row order, Gram-Schmidt deflation
// against earlier components. Covariance-free: each iteration applies the implicit
// covariance operator v <- sum_r ((x_r - mean) . v)(x_r - mean) in O(count x dims),
// so no dims^2 matrix is materialized and no memory is allocated.
//
// Buffers (caller-provided):
//   outMean        dims floats — the bank mean (int8 rows dequantized).
//   outComponents  componentCount x dims floats — unit-norm, mutually orthogonal.
//   scratch        dims floats.
// A degenerate direction (zero variance after deflation) yields a zero component
// vector rather than an error; callers render fewer axes.
Status ComputePrincipalComponents(
	const BankView& bank,
	int32_t componentCount,
	int32_t iterationsPerComponent,
	float* outMean,
	float* outComponents,
	float* scratch);

// Projects every row onto the components: outCoords is count x componentCount,
// row-major — coordinate j of row r is (x_r - mean) . component_j. Rows are
// dequantized exactly as the kernels see them.
Status ProjectRowsOntoComponents(
	const BankView& bank,
	const float* mean,
	const float* components,
	int32_t componentCount,
	float* outCoords);

} // namespace superfaiss

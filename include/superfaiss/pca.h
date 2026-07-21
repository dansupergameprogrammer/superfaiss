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
//   scratch        dims doubles — the covariance-apply accumulator. Double so it
//                  matches the mean's accumulation precision; a float accumulator
//                  rounds once per row and degrades the power iteration with bank size.
// componentCount must be in [1, dims]. A degenerate direction (zero variance after
// deflation) yields a zero component vector rather than an error; callers render fewer axes.
Status ComputePrincipalComponents(
	const BankView& bank,
	int32_t componentCount,
	int32_t iterationsPerComponent,
	float* outMean,
	float* outComponents,
	double* scratch);

// Projects every row onto the components: outCoords is count x componentCount,
// row-major — coordinate j of row r is (x_r - mean) . component_j. Rows are
// dequantized exactly as the kernels see them. componentCount must be in [1, dims]:
// each of the componentCount vectors in `components` is a dims-length row, so an
// over-large count would read past the buffer (matches ComputePrincipalComponents).
Status ProjectRowsOntoComponents(
	const BankView& bank,
	const float* mean,
	const float* components,
	int32_t componentCount,
	float* outCoords);

} // namespace superfaiss

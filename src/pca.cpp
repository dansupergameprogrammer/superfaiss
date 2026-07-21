#include "superfaiss/pca.h"

#include "superfaiss/kernels.h" // detail::FloatBitsToDouble (DAZ-safe scale decode)

#include <cmath>
#include <cstdint>

namespace superfaiss
{

namespace
{
	// Dequantized element r,j — the same value the kernels score.
	inline double RowElem(const BankView& bank, int32_t r, int32_t j)
	{
		if (bank.quant == Quantization::Int8)
		{
			const int8_t* row = static_cast<const int8_t*>(bank.rows) +
				static_cast<int64_t>(r) * bank.paddedDims;
			// DAZ-safe scale decode (a subnormal scale decodes identically under FTZ/DAZ),
			// matching the reductions in analytics.cpp/compose.cpp over the same field.
			return static_cast<double>(row[j]) * detail::FloatBitsToDouble(bank.scales[r]);
		}
		const float* row = static_cast<const float*>(bank.rows) +
			static_cast<int64_t>(r) * bank.paddedDims;
		return static_cast<double>(row[j]);
	}
}

Status ComputePrincipalComponents(
	const BankView& bank,
	int32_t componentCount,
	int32_t iterationsPerComponent,
	float* outMean,
	float* outComponents,
	double* scratch)
{
	if (outMean == nullptr || outComponents == nullptr || scratch == nullptr ||
		componentCount <= 0 || componentCount > bank.dims ||
		iterationsPerComponent <= 0 || bank.count <= 0)
	{
		return Status::InvalidArgument;
	}
	const int32_t dims = bank.dims;
	const int32_t count = bank.count;

	// Mean, accumulated in double per dim (dims-sized passes keep scratch small: the
	// double accumulation rides outMean via a second pass through scratch).
	for (int32_t j = 0; j < dims; ++j)
	{
		double acc = 0.0;
		for (int32_t r = 0; r < count; ++r)
		{
			acc += RowElem(bank, r, j);
		}
		outMean[j] = static_cast<float>(acc / count);
	}

	for (int32_t c = 0; c < componentCount; ++c)
	{
		float* v = outComponents + static_cast<int64_t>(c) * dims;

		// Deterministic seed: basis axis e_(c mod dims). If that axis is degenerate
		// the iteration still mixes in every direction through the covariance apply.
		for (int32_t j = 0; j < dims; ++j)
		{
			v[j] = j == (c % dims) ? 1.0f : 0.0f;
		}

		for (int32_t it = 0; it < iterationsPerComponent; ++it)
		{
			// Deflate against earlier components (keeps the iterate orthogonal).
			for (int32_t p = 0; p < c; ++p)
			{
				const float* u = outComponents + static_cast<int64_t>(p) * dims;
				double proj = 0.0;
				for (int32_t j = 0; j < dims; ++j)
				{
					proj += static_cast<double>(v[j]) * u[j];
				}
				for (int32_t j = 0; j < dims; ++j)
				{
					v[j] = static_cast<float>(v[j] - proj * u[j]);
				}
			}

			// scratch <- Cov * v, serial in row order, accumulated in DOUBLE so the
			// operator is applied at the same precision as the mean above (a float
			// accumulator rounds once per row and loses accuracy as count grows).
			for (int32_t j = 0; j < dims; ++j)
			{
				scratch[j] = 0.0;
			}
			for (int32_t r = 0; r < count; ++r)
			{
				double dot = 0.0;
				for (int32_t j = 0; j < dims; ++j)
				{
					dot += (RowElem(bank, r, j) - outMean[j]) * v[j];
				}
				for (int32_t j = 0; j < dims; ++j)
				{
					scratch[j] += dot * (RowElem(bank, r, j) - outMean[j]);
				}
			}

			double norm = 0.0;
			for (int32_t j = 0; j < dims; ++j)
			{
				norm += scratch[j] * scratch[j];
			}
			if (norm == 0.0)
			{
				// Degenerate direction: zero component, stop iterating it.
				for (int32_t j = 0; j < dims; ++j)
				{
					v[j] = 0.0f;
				}
				break;
			}
			const double inv = 1.0 / std::sqrt(norm);
			for (int32_t j = 0; j < dims; ++j)
			{
				v[j] = static_cast<float>(scratch[j] * inv);
			}
		}

		// Final deflation + renormalize, so returned components are orthonormal even
		// when iterations were few.
		for (int32_t p = 0; p < c; ++p)
		{
			const float* u = outComponents + static_cast<int64_t>(p) * dims;
			double proj = 0.0;
			for (int32_t j = 0; j < dims; ++j)
			{
				proj += static_cast<double>(v[j]) * u[j];
			}
			for (int32_t j = 0; j < dims; ++j)
			{
				v[j] = static_cast<float>(v[j] - proj * u[j]);
			}
		}
		double norm = 0.0;
		for (int32_t j = 0; j < dims; ++j)
		{
			norm += static_cast<double>(v[j]) * v[j];
		}
		if (norm > 0.0)
		{
			const double inv = 1.0 / std::sqrt(norm);
			for (int32_t j = 0; j < dims; ++j)
			{
				v[j] = static_cast<float>(v[j] * inv);
			}
		}
	}
	return Status::Ok;
}

Status ProjectRowsOntoComponents(
	const BankView& bank,
	const float* mean,
	const float* components,
	int32_t componentCount,
	float* outCoords)
{
	if (mean == nullptr || components == nullptr || outCoords == nullptr ||
		componentCount <= 0 || componentCount > bank.dims)
	{
		// componentCount is bounded by dims exactly as ComputePrincipalComponents bounds
		// it: `components` holds componentCount rows of `dims` each, and an over-large
		// count would index past the buffer. The two functions are called in sequence and
		// must agree on the argument.
		return Status::InvalidArgument;
	}
	const int32_t dims = bank.dims;
	for (int32_t r = 0; r < bank.count; ++r)
	{
		for (int32_t c = 0; c < componentCount; ++c)
		{
			const float* v = components + static_cast<int64_t>(c) * dims;
			double dot = 0.0;
			for (int32_t j = 0; j < dims; ++j)
			{
				dot += (RowElem(bank, r, j) - mean[j]) * v[j];
			}
			outCoords[static_cast<int64_t>(r) * componentCount + c] =
				static_cast<float>(dot);
		}
	}
	return Status::Ok;
}

} // namespace superfaiss

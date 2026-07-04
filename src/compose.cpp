#include "superfaiss/compose.h"

#include <cmath>
#include <cstdint>

namespace superfaiss
{

namespace
{
	bool Aligned16(const void* p)
	{
		return (reinterpret_cast<uintptr_t>(p) & (kAlignment - 1)) == 0;
	}
}

Status MakeCentroid(
	const BankView& bank,
	const int32_t* rowIndices,
	int32_t rowCount,
	float* outPaddedQuery)
{
	if (rowIndices == nullptr || rowCount <= 0 || outPaddedQuery == nullptr)
	{
		return Status::InvalidArgument;
	}
	if (!Aligned16(outPaddedQuery))
	{
		return Status::BadAlignment;
	}
	for (int32_t i = 0; i < rowCount; ++i)
	{
		if (rowIndices[i] < 0 || rowIndices[i] >= bank.count)
		{
			return Status::InvalidArgument;
		}
	}

	// Accumulate in double, serial in the caller's index order (deterministic for a
	// given index sequence), allocation-free: a fixed stack chunk of double
	// accumulators, swept across the dims range.
	constexpr int32_t kChunk = 512;
	double acc[kChunk];

	for (int32_t base = 0; base < bank.dims; base += kChunk)
	{
		const int32_t width = (bank.dims - base) < kChunk ? (bank.dims - base) : kChunk;
		for (int32_t j = 0; j < width; ++j)
		{
			acc[j] = 0.0;
		}
		for (int32_t i = 0; i < rowCount; ++i)
		{
			const int32_t row = rowIndices[i];
			if (bank.quant == Quantization::Int8)
			{
				const int8_t* r = static_cast<const int8_t*>(bank.rows) +
					static_cast<int64_t>(row) * bank.paddedDims;
				const double scale = bank.scales[row];
				for (int32_t j = 0; j < width; ++j)
				{
					acc[j] += static_cast<double>(r[base + j]) * scale;
				}
			}
			else
			{
				const float* r = static_cast<const float*>(bank.rows) +
					static_cast<int64_t>(row) * bank.paddedDims;
				for (int32_t j = 0; j < width; ++j)
				{
					acc[j] += static_cast<double>(r[base + j]);
				}
			}
		}
		const double inv = 1.0 / static_cast<double>(rowCount);
		for (int32_t j = 0; j < width; ++j)
		{
			outPaddedQuery[base + j] = static_cast<float>(acc[j] * inv);
		}
	}
	for (int32_t j = bank.dims; j < bank.paddedDims; ++j)
	{
		outPaddedQuery[j] = 0.0f;
	}

	if (bank.metric == Metric::Cosine)
	{
		double norm = 0.0;
		for (int32_t j = 0; j < bank.dims; ++j)
		{
			norm += static_cast<double>(outPaddedQuery[j]) * outPaddedQuery[j];
		}
		if (norm == 0.0)
		{
			return Status::ZeroNormQuery;
		}
		const double invLen = 1.0 / std::sqrt(norm);
		for (int32_t j = 0; j < bank.dims; ++j)
		{
			outPaddedQuery[j] = static_cast<float>(outPaddedQuery[j] * invLen);
		}
	}
	return Status::Ok;
}

Status MakeDirection(
	const float* paddedA,
	const float* paddedB,
	int32_t dims,
	int32_t paddedDims,
	float* outPaddedQuery)
{
	if (paddedA == nullptr || paddedB == nullptr || outPaddedQuery == nullptr ||
		dims <= 0 || paddedDims < dims)
	{
		return Status::InvalidArgument;
	}
	if (!Aligned16(outPaddedQuery))
	{
		return Status::BadAlignment;
	}

	double norm = 0.0;
	for (int32_t j = 0; j < dims; ++j)
	{
		const double d = static_cast<double>(paddedA[j]) - static_cast<double>(paddedB[j]);
		norm += d * d;
	}
	if (norm == 0.0)
	{
		return Status::ZeroNormQuery;
	}
	const double invLen = 1.0 / std::sqrt(norm);
	for (int32_t j = 0; j < dims; ++j)
	{
		const double d = static_cast<double>(paddedA[j]) - static_cast<double>(paddedB[j]);
		outPaddedQuery[j] = static_cast<float>(d * invLen);
	}
	for (int32_t j = dims; j < paddedDims; ++j)
	{
		outPaddedQuery[j] = 0.0f;
	}
	return Status::Ok;
}

} // namespace superfaiss

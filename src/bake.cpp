#include "superfaiss/bake.h"

#include <cmath>
#include <cstring>

#include "superfaiss/kernels.h"   // detail::FloatBitsToDouble (DAZ-safe scale decode, Poirot #1)

namespace superfaiss
{

Status NormalizeRows(float* rows, int32_t count, int32_t dims, int32_t* outBadRow)
{
	if (rows == nullptr || count < 0 || dims <= 0)
	{
		return Status::InvalidArgument;
	}
	for (int32_t r = 0; r < count; ++r)
	{
		float* row = rows + static_cast<int64_t>(r) * dims;
		double norm = 0.0;
		for (int32_t i = 0; i < dims; ++i)
		{
			norm += static_cast<double>(row[i]) * row[i];
		}
		if (norm == 0.0)
		{
			if (outBadRow != nullptr)
			{
				*outBadRow = r;
			}
			return Status::ZeroNormRow;
		}
		const float inv = static_cast<float>(1.0 / std::sqrt(norm));
		for (int32_t i = 0; i < dims; ++i)
		{
			row[i] *= inv;
		}
	}
	return Status::Ok;
}

void QuantizeRowsInt8(
	const float* rows,
	int32_t count,
	int32_t dims,
	int32_t paddedDims,
	int8_t* outRows,
	float* outScales)
{
	for (int32_t r = 0; r < count; ++r)
	{
		const float* src = rows + static_cast<int64_t>(r) * dims;
		int8_t* dst = outRows + static_cast<int64_t>(r) * paddedDims;

		float maxAbs = 0.0f;
		for (int32_t i = 0; i < dims; ++i)
		{
			const float a = std::fabs(src[i]);
			if (a > maxAbs)
			{
				maxAbs = a;
			}
		}

		if (maxAbs == 0.0f)
		{
			outScales[r] = 0.0f;
			std::memset(dst, 0, static_cast<size_t>(paddedDims));
			continue;
		}

		const float scale = maxAbs / 127.0f;
		const float inv = 127.0f / maxAbs;
		outScales[r] = scale;
		for (int32_t i = 0; i < dims; ++i)
		{
			float q = std::nearbyint(src[i] * inv);
			if (q > 127.0f)
			{
				q = 127.0f;
			}
			if (q < -127.0f)
			{
				q = -127.0f;
			}
			dst[i] = static_cast<int8_t>(q);
		}
		if (paddedDims > dims)
		{
			std::memset(dst + dims, 0, static_cast<size_t>(paddedDims - dims));
		}
	}
}

void PadRowsFloat32(
	const float* rows,
	int32_t count,
	int32_t dims,
	int32_t paddedDims,
	float* outRows)
{
	for (int32_t r = 0; r < count; ++r)
	{
		const float* src = rows + static_cast<int64_t>(r) * dims;
		float* dst = outRows + static_cast<int64_t>(r) * paddedDims;
		std::memcpy(dst, src, static_cast<size_t>(dims) * sizeof(float));
		for (int32_t i = dims; i < paddedDims; ++i)
		{
			dst[i] = 0.0f;
		}
	}
}

Status ValidateSourceRows(const float* rows, int32_t count, int32_t dims, int32_t* outBadRow)
{
	// Format geometry ceilings (see ValidateBank): checkable from the header
	// alone, rejected before any payload-size arithmetic — an importer may gate
	// on this before the payload even exists.
	if (count > kMaxBankRows || dims > kMaxCrossDeviceDims)
	{
		return Status::BadFormat;
	}
	if (rows == nullptr || count < 0 || dims <= 0)
	{
		return Status::InvalidArgument;
	}
	for (int32_t r = 0; r < count; ++r)
	{
		const float* row = rows + static_cast<int64_t>(r) * dims;
		for (int32_t i = 0; i < dims; ++i)
		{
			if (!std::isfinite(row[i]))
			{
				if (outBadRow != nullptr)
				{
					*outBadRow = r;
				}
				return Status::NonFiniteQuery;
			}
		}
	}
	return Status::Ok;
}

Status ComputeChannelInverseNorms(const BankView& bank, float* outInvNorms)
{
	if (outInvNorms == nullptr || bank.channels == nullptr || bank.channelCount <= 0 ||
		bank.channelCount > kMaxChannels || bank.rows == nullptr)
	{
		return Status::InvalidArgument;
	}
	for (int32_t r = 0; r < bank.count; ++r)
	{
		for (int32_t c = 0; c < bank.channelCount; ++c)
		{
			const ChannelInfo& channel = bank.channels[c];
			double norm = 0.0;
			if (bank.quant == Quantization::Int8)
			{
				const int8_t* row = static_cast<const int8_t*>(bank.rows) +
					static_cast<int64_t>(r) * bank.paddedDims;
				// Decode the scale the DAZ-safe way the scoring epilogue uses
				// (compose.cpp:204), not a plain widening (Poirot #1): a plain
				// (double)scale can see a subnormal scale flushed to 0 under one
				// thread's DAZ state and preserved under another's, making the derived
				// int8 Cosine sub-norm machine-dependent against the bank's cross-device
				// promise. Identical to (double)scale for every non-subnormal scale.
				const double scale = detail::FloatBitsToDouble(bank.scales[r]);
				for (int32_t j = channel.offset; j < channel.offset + channel.length; ++j)
				{
					const double v = scale * row[j];
					norm += v * v;
				}
			}
			else
			{
				const float* row = static_cast<const float*>(bank.rows) +
					static_cast<int64_t>(r) * bank.paddedDims;
				for (int32_t j = channel.offset; j < channel.offset + channel.length; ++j)
				{
					norm += static_cast<double>(row[j]) * row[j];
				}
			}
			outInvNorms[static_cast<int64_t>(r) * bank.channelCount + c] =
				norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 0.0f;
		}
	}
	return Status::Ok;
}

} // namespace superfaiss

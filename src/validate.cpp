#include "superfaiss/validate.h"

#include <cmath>
#include <cstdint>

namespace superfaiss
{

namespace
{
	bool Aligned(const void* p)
	{
		return (reinterpret_cast<uintptr_t>(p) % kAlignment) == 0;
	}
}

Status ValidateBank(const BankView& bank)
{
	if (bank.count < 0 || bank.dims <= 0)
	{
		return Status::InvalidArgument;
	}
	if (bank.quant != Quantization::Float32 && bank.quant != Quantization::Int8)
	{
		return Status::BadFormat;
	}
	if (bank.metric != Metric::Dot && bank.metric != Metric::Cosine && bank.metric != Metric::L2)
	{
		return Status::BadFormat;
	}
	if (bank.paddedDims != PaddedDims(bank.dims, bank.quant))
	{
		return Status::BadFormat;
	}
	if (bank.count > 0)
	{
		if (bank.rows == nullptr)
		{
			return Status::InvalidArgument;
		}
		if (!Aligned(bank.rows))
		{
			return Status::BadAlignment;
		}
		const bool needsScales = bank.quant == Quantization::Int8;
		if (needsScales != (bank.scales != nullptr))
		{
			return Status::BadFormat;
		}
	}
	return Status::Ok;
}

Status ValidateQuery(const BankView& bank, const float* paddedQuery)
{
	if (paddedQuery == nullptr)
	{
		return Status::InvalidArgument;
	}
	if (!Aligned(paddedQuery))
	{
		return Status::BadAlignment;
	}

	double norm = 0.0;
	for (int32_t i = 0; i < bank.dims; ++i)
	{
		const float v = paddedQuery[i];
		if (!std::isfinite(v))
		{
			return Status::NonFiniteQuery;
		}
		norm += static_cast<double>(v) * v;
	}
	for (int32_t i = bank.dims; i < bank.paddedDims; ++i)
	{
		if (paddedQuery[i] != 0.0f)
		{
			return Status::NonZeroPadding;
		}
	}
	if (bank.metric == Metric::Cosine && norm == 0.0)
	{
		return Status::ZeroNormQuery;
	}
	return Status::Ok;
}

} // namespace superfaiss

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

Status ValidateSegments(
	const BankView& bank,
	const float* paddedQuery,
	const QuerySegment* segments,
	int32_t segmentCount)
{
	if (segments == nullptr || segmentCount <= 0 || segmentCount > kMaxSegments)
	{
		return Status::InvalidArgument;
	}
	const int32_t grid = kAlignment / ElementSize(bank.quant);
	int32_t prevEnd = 0;
	for (int32_t s = 0; s < segmentCount; ++s)
	{
		const QuerySegment& seg = segments[s];
		if (seg.offset < 0 || seg.length <= 0 ||
			seg.offset % grid != 0 || seg.length % grid != 0 ||
			seg.offset < prevEnd ||
			static_cast<int64_t>(seg.offset) + seg.length > bank.paddedDims ||
			!std::isfinite(seg.weight))
		{
			return Status::InvalidArgument;
		}
		prevEnd = seg.offset + seg.length;

		if (bank.metric == Metric::Cosine && seg.weight != 0.0f)
		{
			double norm = 0.0;
			for (int32_t j = seg.offset; j < seg.offset + seg.length; ++j)
			{
				norm += static_cast<double>(paddedQuery[j]) * paddedQuery[j];
			}
			if (norm == 0.0)
			{
				return Status::ZeroNormQuery;
			}
		}
	}
	return Status::Ok;
}

Status ValidateBankData(const BankView& bank, int32_t* outBadRow)
{
	const Status structural = ValidateBank(bank);
	if (structural != Status::Ok)
	{
		return structural;
	}

	const int32_t pd = bank.paddedDims;
	for (int32_t r = 0; r < bank.count; ++r)
	{
		bool bad = false;
		if (bank.quant == Quantization::Float32)
		{
			const float* row = static_cast<const float*>(bank.rows) + static_cast<int64_t>(r) * pd;
			for (int32_t i = 0; i < bank.dims; ++i)
			{
				if (!std::isfinite(row[i]))
				{
					bad = true;
					break;
				}
			}
			for (int32_t i = bank.dims; !bad && i < pd; ++i)
			{
				bad = row[i] != 0.0f;
			}
		}
		else
		{
			const int8_t* row = static_cast<const int8_t*>(bank.rows) + static_cast<int64_t>(r) * pd;
			for (int32_t i = bank.dims; i < pd; ++i)
			{
				if (row[i] != 0)
				{
					bad = true;
					break;
				}
			}
			const float scale = bank.scales[r];
			bad = bad || !std::isfinite(scale) || scale < 0.0f;
		}
		if (bad)
		{
			if (outBadRow != nullptr)
			{
				*outBadRow = r;
			}
			return Status::BadFormat;
		}
	}
	return Status::Ok;
}

} // namespace superfaiss

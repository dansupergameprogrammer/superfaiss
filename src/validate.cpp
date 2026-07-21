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
	// Format geometry ceilings: a header over the caps is rejected here, before
	// any size arithmetic runs on its values (the scratch-archive load applies
	// the same hard reject). Within the caps every byte-size term downstream
	// stays below 2^47 — see kMaxBankRows.
	if (bank.count > kMaxBankRows || bank.dims > kMaxCrossDeviceDims)
	{
		return Status::BadFormat;
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
	if (bank.channels != nullptr || bank.channelCount != 0)
	{
		if (bank.channels == nullptr || bank.channelCount <= 0 ||
			bank.channelCount > kMaxChannels)
		{
			return Status::BadFormat;
		}
		const int32_t grid = kAlignment / ElementSize(bank.quant);
		int32_t prevEnd = 0;
		for (int32_t c = 0; c < bank.channelCount; ++c)
		{
			const ChannelInfo& channel = bank.channels[c];
			if (channel.offset < 0 || channel.length <= 0 ||
				channel.offset % grid != 0 || channel.length % grid != 0 ||
				channel.offset < prevEnd ||
				static_cast<int64_t>(channel.offset) + channel.length > bank.paddedDims)
			{
				return Status::BadFormat;
			}
			prevEnd = channel.offset + channel.length;
		}
		// Per-channel cosine requires the baked inverse sub-norms.
		if (bank.metric == Metric::Cosine && bank.channelInvNorms == nullptr)
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

Status ValidateBiasPairs(
	const BankView& bank,
	const BiasPair* pairs,
	int32_t pairCount,
	uint32_t* seenBits)
{
	if (pairCount < 0 || (pairCount > 0 && (pairs == nullptr || seenBits == nullptr)) ||
		pairCount > bank.count)
	{
		return Status::InvalidArgument;
	}
	for (int32_t i = 0; i < pairCount; ++i)
	{
		const int32_t row = pairs[i].index;
		if (row < 0 || row >= bank.count)
		{
			return Status::InvalidArgument;
		}
		const uint32_t bit = 1u << (row & 31);
		if ((seenBits[row >> 5] & bit) != 0)
		{
			return Status::InvalidArgument; // duplicate index: ambiguous composition
		}
		seenBits[row >> 5] |= bit;
		if (!std::isfinite(pairs[i].bias))
		{
			return Status::NonFiniteQuery;
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
	// Bake-law tolerances (a loaded payload must satisfy what the
	// bake actually produces, or the "re-validated on load" claim is hollow):
	// Cosine rows are unit-norm at bake; float32 stores them exactly (up to
	// normalization rounding), int8 adds at most scale/2 per element, so the
	// dequantized norm sits within (scale/2)*sqrt(dims) of 1. A row outside its
	// bound is tampering or corruption, not quantization. The int8 bake clamp is
	// [-127, 127]: -128 never occurs, and admitting it would void the CrossDevice
	// overflow proof (an all--128 row at kMaxCrossDeviceDims reaches 2^31 in the
	// L2 self-sum) - so -128 is rejected as BadFormat.
	const bool cosine = bank.metric == Metric::Cosine;
	for (int32_t r = 0; r < bank.count; ++r)
	{
		bool bad = false;
		double normSq = 0.0;
		double normTolerance = 1e-3;
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
				if (cosine)
				{
					normSq += static_cast<double>(row[i]) * row[i];
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
			const float scale = bank.scales[r];
			for (int32_t i = 0; i < bank.dims; ++i)
			{
				if (row[i] == INT8_MIN)
				{
					bad = true; // outside the bake clamp of [-127, 127]
					break;
				}
				if (cosine)
				{
					const double v = static_cast<double>(row[i]) * scale;
					normSq += v * v;
				}
			}
			for (int32_t i = bank.dims; !bad && i < pd; ++i)
			{
				bad = row[i] != 0;
			}
			bad = bad || !std::isfinite(scale) || scale < 0.0f;
			normTolerance += 0.5 * static_cast<double>(scale) * std::sqrt(static_cast<double>(bank.dims));
		}
		if (!bad && cosine)
		{
			// Covers the zero-norm-row law too: norm 0 is maximally out of tolerance.
			const double norm = std::sqrt(normSq);
			bad = !(std::fabs(norm - 1.0) <= normTolerance);
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

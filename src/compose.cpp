#include "superfaiss/compose.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "superfaiss/kernels.h"

namespace superfaiss
{

namespace
{
	bool Aligned16(const void* p)
	{
		return (reinterpret_cast<uintptr_t>(p) & (kAlignment - 1)) == 0;
	}

	// Round-half-even of the exact rational num/den (den > 0) in pure integer math —
	// the pooling epilogue's quantization rounding (section 19 c discipline: no FP
	// rounding-mode dependence). |num| <= 127 * 2^51 stays well inside int64.
	int64_t RheDivI64(int64_t num, int64_t den)
	{
		const bool neg = num < 0;
		const uint64_t a = neg ? static_cast<uint64_t>(-num) : static_cast<uint64_t>(num);
		const uint64_t d = static_cast<uint64_t>(den);
		uint64_t q = a / d;
		const uint64_t r = a % d;
		// Round to nearest; exact halves go to even.
		if (2 * r > d || (2 * r == d && (q & 1) != 0))
		{
			++q;
		}
		return neg ? -static_cast<int64_t>(q) : static_cast<int64_t>(q);
	}

	// The per-row fixed-point scale multiplier (weight-free half): the row's decoded
	// scale over the max included scale, on the 2^kPoolScaleFracBits grid. Both
	// decodes are DAZ-proof (subnormal scales are legal bank content); the ratio and
	// the grid scaling are single IEEE double operations; the rounding is integer.
	// ratio <= 1, so the argument is <= 2^24, inside RoundHalfEvenI32's domain.
	int64_t PoolScaleMultiplier(float scale, double maxScale)
	{
		const double ratio = detail::FloatBitsToDouble(scale) / maxScale;
		return detail::RoundHalfEvenI32(
			ratio * static_cast<double>(int64_t{1} << kPoolScaleFracBits));
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
				// DAZ-safe scale decode, consistent with the other scale reads in this file
				// (a subnormal scale decodes the same regardless of the thread's FTZ/DAZ mode).
				const double scale = detail::FloatBitsToDouble(bank.scales[row]);
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

Status MakeCentroidCrossDevice(
	const BankView& bank,
	const int32_t* rowIndices,
	int32_t rowCount,
	const int32_t* weights,
	const uint32_t* excludeBits,
	int8_t* outQ8,
	double* outScale,
	int64_t* outSqSum,
	int32_t offset,
	int32_t length)
{
	if (rowIndices == nullptr || rowCount <= 0 || outQ8 == nullptr ||
		outScale == nullptr || outSqSum == nullptr)
	{
		return Status::InvalidArgument;
	}
	if (!Aligned16(outQ8))
	{
		return Status::BadAlignment;
	}
	// CrossDevice laws: int8 banks only, under the accumulator-proof dims ceiling.
	if (bank.quant != Quantization::Int8 || bank.paddedDims > kMaxCrossDeviceDims)
	{
		return Status::InvalidArgument;
	}
	// Sub-range (V3.0): default (length < 0) pools the whole [0, dims) -- byte-identical
	// to the shipped path (chOff 0, chDims bank.dims). A channel restricts the range.
	const int32_t chOff = offset;
	const int32_t chDims = length < 0 ? bank.dims : length;
	if (chOff < 0 || chDims < 0 || chOff + chDims > bank.paddedDims)
	{
		return Status::InvalidArgument;
	}

	// Pass 1 over the selection: validate indices and weights, apply exclusion, find
	// the max included scale (the fixed-point base) and the weight sum (the FAI-5
	// bound and the mean divisor). Deterministic: max over doubles is order-free.
	double maxScale = 0.0;
	int64_t sumW = 0;
	int32_t included = 0;
	for (int32_t i = 0; i < rowCount; ++i)
	{
		const int32_t row = rowIndices[i];
		if (row < 0 || row >= bank.count)
		{
			return Status::InvalidArgument;
		}
		const int64_t w = weights != nullptr ? weights[i] : 1;
		if (w <= 0)
		{
			return Status::InvalidArgument;
		}
		if (IsExcluded(excludeBits, row))
		{
			continue;
		}
		++included;
		sumW += w;
		if (sumW > kMaxPooledRows)
		{
			// Over the pinned bound the int64 accumulator proof no longer holds:
			// reject rather than overflow silently (FAI-5, reject-over-degrade).
			return Status::InvalidArgument;
		}
		const double s = detail::FloatBitsToDouble(bank.scales[row]);
		if (s > maxScale)
		{
			maxScale = s;
		}
	}
	if (included == 0)
	{
		return Status::InvalidArgument; // empty selection, never a zero vector
	}
	if (maxScale == 0.0)
	{
		// Every included row dequantizes to zero: the pooled vector is exactly zero.
		return Status::ZeroNormQuery;
	}

	// Sweep 1 (fixed stack chunk, the MakeCentroid idiom — allocation-free): find the
	// max accumulator magnitude, which both the zero-norm check and the symmetric
	// requantization need before any element can quantize. Sweep 2 re-accumulates and
	// quantizes. Twice the accumulation work, zero allocation and zero added API
	// surface; pooling is query-build cost, not scan cost.
	constexpr int32_t kChunk = 512;
	int64_t acc[kChunk];
	int64_t maxAcc = 0;
	for (int32_t base = 0; base < chDims; base += kChunk)
	{
		const int32_t width = (chDims - base) < kChunk ? (chDims - base) : kChunk;
		for (int32_t j = 0; j < width; ++j)
		{
			acc[j] = 0;
		}
		for (int32_t i = 0; i < rowCount; ++i)
		{
			const int32_t row = rowIndices[i];
			if (IsExcluded(excludeBits, row))
			{
				continue;
			}
			const int64_t w = weights != nullptr ? weights[i] : 1;
			const int64_t m = w * PoolScaleMultiplier(bank.scales[row], maxScale);
			if (m == 0)
			{
				continue; // scale ratio below the fixed-point grid: contributes nothing
			}
			const int8_t* r = static_cast<const int8_t*>(bank.rows) +
				static_cast<int64_t>(row) * bank.paddedDims;
			for (int32_t j = 0; j < width; ++j)
			{
				// Exact and order-free: |sum| <= sum(w) * 127 * 2^24 <= 2^51 (FAI-5).
				acc[j] += static_cast<int64_t>(r[chOff + base + j]) * m;
			}
		}
		for (int32_t j = 0; j < width; ++j)
		{
			const int64_t mag = acc[j] < 0 ? -acc[j] : acc[j];
			if (mag > maxAcc)
			{
				maxAcc = mag;
			}
		}
	}
	if (maxAcc == 0)
	{
		// All-zero integer accumulator (antipodal members cancelling): the check the
		// omitted normalization would have performed, kept without the float math.
		return Status::ZeroNormQuery;
	}

	// Sweep 2: requantize directly in the integer domain (FAI-6). q8_j =
	// RHE(acc_j * 127 / maxAcc) — exact rational, integer rounding; the max-magnitude
	// dim maps to exactly +-127. No float touches an element on this path.
	int64_t sqSum = 0;
	for (int32_t base = 0; base < chDims; base += kChunk)
	{
		const int32_t width = (chDims - base) < kChunk ? (chDims - base) : kChunk;
		for (int32_t j = 0; j < width; ++j)
		{
			acc[j] = 0;
		}
		for (int32_t i = 0; i < rowCount; ++i)
		{
			const int32_t row = rowIndices[i];
			if (IsExcluded(excludeBits, row))
			{
				continue;
			}
			const int64_t w = weights != nullptr ? weights[i] : 1;
			const int64_t m = w * PoolScaleMultiplier(bank.scales[row], maxScale);
			if (m == 0)
			{
				continue;
			}
			const int8_t* r = static_cast<const int8_t*>(bank.rows) +
				static_cast<int64_t>(row) * bank.paddedDims;
			for (int32_t j = 0; j < width; ++j)
			{
				acc[j] += static_cast<int64_t>(r[chOff + base + j]) * m;
			}
		}
		for (int32_t j = 0; j < width; ++j)
		{
			const int64_t q = RheDivI64(acc[j] * 127, maxAcc);
			outQ8[base + j] = static_cast<int8_t>(q);
			sqSum += q * q;
		}
	}
	if (bank.paddedDims > chDims)
	{
		std::memset(outQ8 + chDims, 0, static_cast<size_t>(bank.paddedDims - chDims));
	}

	// The dequant scale, kept in double (no float round-trip, the XdQuery discipline):
	// q8_j * scale ~= the pooled mean's element. Fixed order, and every integer is
	// exactly representable (maxAcc <= 2^51, sumW*127 < 2^27 under the cap), so the
	// division sees the exact rational maxAcc/(sumW*127) — all-equal weights scale
	// numerator and denominator by the same integer and cancel before the rounding,
	// which is what makes P6's weighted == unweighted equality bitwise.
	const double ratio = static_cast<double>(maxAcc) / static_cast<double>(sumW * 127);
	*outScale = (ratio * maxScale) *
		(1.0 / static_cast<double>(int64_t{1} << kPoolScaleFracBits));
	*outSqSum = sqSum;
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

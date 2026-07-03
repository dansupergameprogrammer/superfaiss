#include "superfaiss/kernels.h"

namespace superfaiss
{

// SIMD selection is compile-time. Day 0 ships the scalar path; SSE/NEON land next and
// must preserve the exact accumulation structure below.
SimdPath ActiveSimdPath()
{
	return SimdPath::Scalar;
}

namespace detail
{

// Accumulation structure shared by every kernel path (scalar and SIMD):
//   - the row is walked in 16-element outer blocks; each block feeds four 4-lane
//     accumulators in order (acc0 <- elements 0..3, acc1 <- 4..7, acc2 <- 8..11,
//     acc3 <- 12..15);
//   - a remainder of whole 4-element groups (paddedDims is always a multiple of 4)
//     feeds acc0, acc1, ... in order;
//   - lanes combine as (l0+l1)+(l2+l3) per accumulator, then (a0+a1)+(a2+a3).
// SIMD paths must reproduce this order exactly — that is what makes scalar and SIMD
// results bit-identical on a device.

namespace
{
	struct Acc4
	{
		float Lane[4] = {0.0f, 0.0f, 0.0f, 0.0f};

		void MulAdd(const float* a, const float* b)
		{
			Lane[0] += a[0] * b[0];
			Lane[1] += a[1] * b[1];
			Lane[2] += a[2] * b[2];
			Lane[3] += a[3] * b[3];
		}

		void DiffSq(const float* a, const float* b)
		{
			const float d0 = a[0] - b[0];
			const float d1 = a[1] - b[1];
			const float d2 = a[2] - b[2];
			const float d3 = a[3] - b[3];
			Lane[0] += d0 * d0;
			Lane[1] += d1 * d1;
			Lane[2] += d2 * d2;
			Lane[3] += d3 * d3;
		}

		void MulAddI8(const int8_t* r, const float* q)
		{
			Lane[0] += static_cast<float>(r[0]) * q[0];
			Lane[1] += static_cast<float>(r[1]) * q[1];
			Lane[2] += static_cast<float>(r[2]) * q[2];
			Lane[3] += static_cast<float>(r[3]) * q[3];
		}

		void DiffSqI8(const int8_t* r, float scale, const float* q)
		{
			const float d0 = q[0] - scale * static_cast<float>(r[0]);
			const float d1 = q[1] - scale * static_cast<float>(r[1]);
			const float d2 = q[2] - scale * static_cast<float>(r[2]);
			const float d3 = q[3] - scale * static_cast<float>(r[3]);
			Lane[0] += d0 * d0;
			Lane[1] += d1 * d1;
			Lane[2] += d2 * d2;
			Lane[3] += d3 * d3;
		}

		float Sum() const { return (Lane[0] + Lane[1]) + (Lane[2] + Lane[3]); }
	};

	inline float Combine(const Acc4& a0, const Acc4& a1, const Acc4& a2, const Acc4& a3)
	{
		return (a0.Sum() + a1.Sum()) + (a2.Sum() + a3.Sum());
	}
}

float DotF32Scalar(const float* row, const float* query, int32_t paddedDims)
{
	Acc4 acc[4];
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		acc[0].MulAdd(row + i, query + i);
		acc[1].MulAdd(row + i + 4, query + i + 4);
		acc[2].MulAdd(row + i + 8, query + i + 8);
		acc[3].MulAdd(row + i + 12, query + i + 12);
	}
	for (int32_t g = 0; i + 4 <= paddedDims; i += 4, ++g)
	{
		acc[g].MulAdd(row + i, query + i);
	}
	return Combine(acc[0], acc[1], acc[2], acc[3]);
}

float L2F32Scalar(const float* row, const float* query, int32_t paddedDims)
{
	Acc4 acc[4];
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		acc[0].DiffSq(query + i, row + i);
		acc[1].DiffSq(query + i + 4, row + i + 4);
		acc[2].DiffSq(query + i + 8, row + i + 8);
		acc[3].DiffSq(query + i + 12, row + i + 12);
	}
	for (int32_t g = 0; i + 4 <= paddedDims; i += 4, ++g)
	{
		acc[g].DiffSq(query + i, row + i);
	}
	return Combine(acc[0], acc[1], acc[2], acc[3]);
}

float DotI8Scalar(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	// scale is constant per row, so dot(scale*r, q) = scale * dot(r, q): one multiply
	// at the end instead of one per lane.
	Acc4 acc[4];
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		acc[0].MulAddI8(row + i, query + i);
		acc[1].MulAddI8(row + i + 4, query + i + 4);
		acc[2].MulAddI8(row + i + 8, query + i + 8);
		acc[3].MulAddI8(row + i + 12, query + i + 12);
	}
	for (int32_t g = 0; i + 4 <= paddedDims; i += 4, ++g)
	{
		acc[g].MulAddI8(row + i, query + i);
	}
	return scale * Combine(acc[0], acc[1], acc[2], acc[3]);
}

float L2I8Scalar(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	Acc4 acc[4];
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		acc[0].DiffSqI8(row + i, scale, query + i);
		acc[1].DiffSqI8(row + i + 4, scale, query + i + 4);
		acc[2].DiffSqI8(row + i + 8, scale, query + i + 8);
		acc[3].DiffSqI8(row + i + 12, scale, query + i + 12);
	}
	for (int32_t g = 0; i + 4 <= paddedDims; i += 4, ++g)
	{
		acc[g].DiffSqI8(row + i, scale, query + i);
	}
	return Combine(acc[0], acc[1], acc[2], acc[3]);
}

float DotF32(const float* row, const float* query, int32_t paddedDims)
{
	return DotF32Scalar(row, query, paddedDims);
}

float L2F32(const float* row, const float* query, int32_t paddedDims)
{
	return L2F32Scalar(row, query, paddedDims);
}

float DotI8(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return DotI8Scalar(row, scale, query, paddedDims);
}

float L2I8(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return L2I8Scalar(row, scale, query, paddedDims);
}

} // namespace detail

void ScoreChunk(
	const BankView& bank,
	const float* paddedQuery,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout)
{
	const int32_t chunkRows = ChunkRows(bank);
	const int32_t begin = chunkIndex * chunkRows;
	int32_t end = begin + chunkRows;
	if (end > bank.count)
	{
		end = bank.count;
	}

	const int32_t pd = bank.paddedDims;
	const bool isL2 = bank.metric == Metric::L2;

	if (bank.quant == Quantization::Float32)
	{
		const float* rows = static_cast<const float*>(bank.rows);
		for (int32_t r = begin; r < end; ++r)
		{
			if (IsExcluded(excludeBits, r))
			{
				continue;
			}
			const float* row = rows + static_cast<int64_t>(r) * pd;
			const float score = isL2
				? detail::L2F32(row, paddedQuery, pd)
				: detail::DotF32(row, paddedQuery, pd);
			inout.Push(r, score);
		}
	}
	else
	{
		const int8_t* rows = static_cast<const int8_t*>(bank.rows);
		for (int32_t r = begin; r < end; ++r)
		{
			if (IsExcluded(excludeBits, r))
			{
				continue;
			}
			const int8_t* row = rows + static_cast<int64_t>(r) * pd;
			const float scale = bank.scales[r];
			const float score = isL2
				? detail::L2I8(row, scale, paddedQuery, pd)
				: detail::DotI8(row, scale, paddedQuery, pd);
			inout.Push(r, score);
		}
	}
}

} // namespace superfaiss

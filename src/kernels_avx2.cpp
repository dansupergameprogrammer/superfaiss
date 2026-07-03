#include "superfaiss/kernels.h"

// AVX2+FMA kernel path (x86 only; selected at runtime when the CPU supports it and the
// row stride fits 8-lane blocks). Accumulation structure:
//   - 32-element outer blocks feed four 8-lane accumulators in order;
//   - remainder in whole 8-element groups feeds acc0, acc1, ... (float32; int8 remainder
//     comes in 16-element halves feeding acc0 then acc1);
//   - lanes combine as ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)) per accumulator, then
//     (a0+a1)+(a2+a3).
// FMA is used deliberately: the scalar mirrors below accumulate with std::fma, which
// rounds identically to the hardware instruction, so mirror equality is exact.

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)

#include <immintrin.h>
#include <cmath>

#if defined(__clang__) || defined(__GNUC__)
	#define SUPERFAISS_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
	#define SUPERFAISS_AVX2_TARGET
#endif

namespace superfaiss
{
namespace detail
{

namespace
{
	SUPERFAISS_AVX2_TARGET inline float SumLanes8(__m256 v)
	{
		alignas(32) float l[8];
		_mm256_store_ps(l, v);
		return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
	}

	// Widen 8 int8 values to 8 float lanes.
	SUPERFAISS_AVX2_TARGET inline __m256 WidenI8x8(const int8_t* p)
	{
		const __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
		return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
	}

	struct Acc8
	{
		float Lane[8] = {};

		void FmaMul(const float* a, const float* b)
		{
			for (int l = 0; l < 8; ++l)
			{
				Lane[l] = std::fma(a[l], b[l], Lane[l]);
			}
		}

		void FmaDiffSq(const float* q, const float* r)
		{
			for (int l = 0; l < 8; ++l)
			{
				const float d = q[l] - r[l];
				Lane[l] = std::fma(d, d, Lane[l]);
			}
		}

		void FmaMulI8(const int8_t* r, const float* q)
		{
			for (int l = 0; l < 8; ++l)
			{
				Lane[l] = std::fma(static_cast<float>(r[l]), q[l], Lane[l]);
			}
		}

		void FmaDiffSqI8(const int8_t* r, float scale, const float* q)
		{
			for (int l = 0; l < 8; ++l)
			{
				// Mirrors _mm256_fnmadd_ps(scale, f, q): single-rounded q - scale*f.
				const float d = std::fma(-scale, static_cast<float>(r[l]), q[l]);
				Lane[l] = std::fma(d, d, Lane[l]);
			}
		}

		float Sum() const
		{
			return ((Lane[0] + Lane[1]) + (Lane[2] + Lane[3])) +
				((Lane[4] + Lane[5]) + (Lane[6] + Lane[7]));
		}
	};
}

// --- AVX2 intrinsic kernels -------------------------------------------------

SUPERFAISS_AVX2_TARGET
float DotF32Avx2(const float* row, const float* query, int32_t paddedDims)
{
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(query + i), acc0);
		acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 8), _mm256_loadu_ps(query + i + 8), acc1);
		acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 16), _mm256_loadu_ps(query + i + 16), acc2);
		acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i + 24), _mm256_loadu_ps(query + i + 24), acc3);
	}
	if (i + 8 <= paddedDims)
	{
		acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(query + i), acc0);
		i += 8;
	}
	if (i + 8 <= paddedDims)
	{
		acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(query + i), acc1);
		i += 8;
	}
	if (i + 8 <= paddedDims)
	{
		acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(query + i), acc2);
	}
	return (SumLanes8(acc0) + SumLanes8(acc1)) + (SumLanes8(acc2) + SumLanes8(acc3));
}

SUPERFAISS_AVX2_TARGET
float L2F32Avx2(const float* row, const float* query, int32_t paddedDims)
{
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(query + i), _mm256_loadu_ps(row + i));
		const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(query + i + 8), _mm256_loadu_ps(row + i + 8));
		const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(query + i + 16), _mm256_loadu_ps(row + i + 16));
		const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(query + i + 24), _mm256_loadu_ps(row + i + 24));
		acc0 = _mm256_fmadd_ps(d0, d0, acc0);
		acc1 = _mm256_fmadd_ps(d1, d1, acc1);
		acc2 = _mm256_fmadd_ps(d2, d2, acc2);
		acc3 = _mm256_fmadd_ps(d3, d3, acc3);
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(query + i), _mm256_loadu_ps(row + i));
		acc0 = _mm256_fmadd_ps(d, d, acc0);
		i += 8;
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(query + i), _mm256_loadu_ps(row + i));
		acc1 = _mm256_fmadd_ps(d, d, acc1);
		i += 8;
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(query + i), _mm256_loadu_ps(row + i));
		acc2 = _mm256_fmadd_ps(d, d, acc2);
	}
	return (SumLanes8(acc0) + SumLanes8(acc1)) + (SumLanes8(acc2) + SumLanes8(acc3));
}

SUPERFAISS_AVX2_TARGET
float DotI8Avx2(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		acc0 = _mm256_fmadd_ps(WidenI8x8(row + i), _mm256_loadu_ps(query + i), acc0);
		acc1 = _mm256_fmadd_ps(WidenI8x8(row + i + 8), _mm256_loadu_ps(query + i + 8), acc1);
		acc2 = _mm256_fmadd_ps(WidenI8x8(row + i + 16), _mm256_loadu_ps(query + i + 16), acc2);
		acc3 = _mm256_fmadd_ps(WidenI8x8(row + i + 24), _mm256_loadu_ps(query + i + 24), acc3);
	}
	// Int8 stride is a multiple of 16: at most one 16-element half-block remains.
	if (i + 16 <= paddedDims)
	{
		acc0 = _mm256_fmadd_ps(WidenI8x8(row + i), _mm256_loadu_ps(query + i), acc0);
		acc1 = _mm256_fmadd_ps(WidenI8x8(row + i + 8), _mm256_loadu_ps(query + i + 8), acc1);
	}
	return scale * ((SumLanes8(acc0) + SumLanes8(acc1)) + (SumLanes8(acc2) + SumLanes8(acc3)));
}

SUPERFAISS_AVX2_TARGET
float L2I8Avx2(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	const __m256 scaleV = _mm256_set1_ps(scale);
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		const __m256 d0 = _mm256_fnmadd_ps(scaleV, WidenI8x8(row + i), _mm256_loadu_ps(query + i));
		const __m256 d1 = _mm256_fnmadd_ps(scaleV, WidenI8x8(row + i + 8), _mm256_loadu_ps(query + i + 8));
		const __m256 d2 = _mm256_fnmadd_ps(scaleV, WidenI8x8(row + i + 16), _mm256_loadu_ps(query + i + 16));
		const __m256 d3 = _mm256_fnmadd_ps(scaleV, WidenI8x8(row + i + 24), _mm256_loadu_ps(query + i + 24));
		acc0 = _mm256_fmadd_ps(d0, d0, acc0);
		acc1 = _mm256_fmadd_ps(d1, d1, acc1);
		acc2 = _mm256_fmadd_ps(d2, d2, acc2);
		acc3 = _mm256_fmadd_ps(d3, d3, acc3);
	}
	if (i + 16 <= paddedDims)
	{
		const __m256 d0 = _mm256_fnmadd_ps(scaleV, WidenI8x8(row + i), _mm256_loadu_ps(query + i));
		const __m256 d1 = _mm256_fnmadd_ps(scaleV, WidenI8x8(row + i + 8), _mm256_loadu_ps(query + i + 8));
		acc0 = _mm256_fmadd_ps(d0, d0, acc0);
		acc1 = _mm256_fmadd_ps(d1, d1, acc1);
	}
	return (SumLanes8(acc0) + SumLanes8(acc1)) + (SumLanes8(acc2) + SumLanes8(acc3));
}

// --- Pair kernels: one row pass, two queries. Each query's accumulation is the exact
// --- single-kernel structure, so results are bit-identical to the single kernels.

SUPERFAISS_AVX2_TARGET
void DotF32PairAvx2(const float* row, const float* qa, const float* qb,
	int32_t paddedDims, float* outA, float* outB)
{
	__m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
	__m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
	__m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
	__m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		const __m256 r0 = _mm256_loadu_ps(row + i);
		const __m256 r1 = _mm256_loadu_ps(row + i + 8);
		const __m256 r2 = _mm256_loadu_ps(row + i + 16);
		const __m256 r3 = _mm256_loadu_ps(row + i + 24);
		a0 = _mm256_fmadd_ps(r0, _mm256_loadu_ps(qa + i), a0);
		a1 = _mm256_fmadd_ps(r1, _mm256_loadu_ps(qa + i + 8), a1);
		a2 = _mm256_fmadd_ps(r2, _mm256_loadu_ps(qa + i + 16), a2);
		a3 = _mm256_fmadd_ps(r3, _mm256_loadu_ps(qa + i + 24), a3);
		b0 = _mm256_fmadd_ps(r0, _mm256_loadu_ps(qb + i), b0);
		b1 = _mm256_fmadd_ps(r1, _mm256_loadu_ps(qb + i + 8), b1);
		b2 = _mm256_fmadd_ps(r2, _mm256_loadu_ps(qb + i + 16), b2);
		b3 = _mm256_fmadd_ps(r3, _mm256_loadu_ps(qb + i + 24), b3);
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 r = _mm256_loadu_ps(row + i);
		a0 = _mm256_fmadd_ps(r, _mm256_loadu_ps(qa + i), a0);
		b0 = _mm256_fmadd_ps(r, _mm256_loadu_ps(qb + i), b0);
		i += 8;
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 r = _mm256_loadu_ps(row + i);
		a1 = _mm256_fmadd_ps(r, _mm256_loadu_ps(qa + i), a1);
		b1 = _mm256_fmadd_ps(r, _mm256_loadu_ps(qb + i), b1);
		i += 8;
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 r = _mm256_loadu_ps(row + i);
		a2 = _mm256_fmadd_ps(r, _mm256_loadu_ps(qa + i), a2);
		b2 = _mm256_fmadd_ps(r, _mm256_loadu_ps(qb + i), b2);
	}
	*outA = (SumLanes8(a0) + SumLanes8(a1)) + (SumLanes8(a2) + SumLanes8(a3));
	*outB = (SumLanes8(b0) + SumLanes8(b1)) + (SumLanes8(b2) + SumLanes8(b3));
}

SUPERFAISS_AVX2_TARGET
void L2F32PairAvx2(const float* row, const float* qa, const float* qb,
	int32_t paddedDims, float* outA, float* outB)
{
	__m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
	__m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
	__m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
	__m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		const __m256 r0 = _mm256_loadu_ps(row + i);
		const __m256 r1 = _mm256_loadu_ps(row + i + 8);
		const __m256 r2 = _mm256_loadu_ps(row + i + 16);
		const __m256 r3 = _mm256_loadu_ps(row + i + 24);
		const __m256 da0 = _mm256_sub_ps(_mm256_loadu_ps(qa + i), r0);
		const __m256 da1 = _mm256_sub_ps(_mm256_loadu_ps(qa + i + 8), r1);
		const __m256 da2 = _mm256_sub_ps(_mm256_loadu_ps(qa + i + 16), r2);
		const __m256 da3 = _mm256_sub_ps(_mm256_loadu_ps(qa + i + 24), r3);
		a0 = _mm256_fmadd_ps(da0, da0, a0);
		a1 = _mm256_fmadd_ps(da1, da1, a1);
		a2 = _mm256_fmadd_ps(da2, da2, a2);
		a3 = _mm256_fmadd_ps(da3, da3, a3);
		const __m256 db0 = _mm256_sub_ps(_mm256_loadu_ps(qb + i), r0);
		const __m256 db1 = _mm256_sub_ps(_mm256_loadu_ps(qb + i + 8), r1);
		const __m256 db2 = _mm256_sub_ps(_mm256_loadu_ps(qb + i + 16), r2);
		const __m256 db3 = _mm256_sub_ps(_mm256_loadu_ps(qb + i + 24), r3);
		b0 = _mm256_fmadd_ps(db0, db0, b0);
		b1 = _mm256_fmadd_ps(db1, db1, b1);
		b2 = _mm256_fmadd_ps(db2, db2, b2);
		b3 = _mm256_fmadd_ps(db3, db3, b3);
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 r = _mm256_loadu_ps(row + i);
		const __m256 da = _mm256_sub_ps(_mm256_loadu_ps(qa + i), r);
		const __m256 db = _mm256_sub_ps(_mm256_loadu_ps(qb + i), r);
		a0 = _mm256_fmadd_ps(da, da, a0);
		b0 = _mm256_fmadd_ps(db, db, b0);
		i += 8;
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 r = _mm256_loadu_ps(row + i);
		const __m256 da = _mm256_sub_ps(_mm256_loadu_ps(qa + i), r);
		const __m256 db = _mm256_sub_ps(_mm256_loadu_ps(qb + i), r);
		a1 = _mm256_fmadd_ps(da, da, a1);
		b1 = _mm256_fmadd_ps(db, db, b1);
		i += 8;
	}
	if (i + 8 <= paddedDims)
	{
		const __m256 r = _mm256_loadu_ps(row + i);
		const __m256 da = _mm256_sub_ps(_mm256_loadu_ps(qa + i), r);
		const __m256 db = _mm256_sub_ps(_mm256_loadu_ps(qb + i), r);
		a2 = _mm256_fmadd_ps(da, da, a2);
		b2 = _mm256_fmadd_ps(db, db, b2);
	}
	*outA = (SumLanes8(a0) + SumLanes8(a1)) + (SumLanes8(a2) + SumLanes8(a3));
	*outB = (SumLanes8(b0) + SumLanes8(b1)) + (SumLanes8(b2) + SumLanes8(b3));
}

SUPERFAISS_AVX2_TARGET
void DotI8PairAvx2(const int8_t* row, float rowScale, const float* qa, const float* qb,
	int32_t paddedDims, float* outA, float* outB)
{
	__m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
	__m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
	__m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
	__m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		const __m256 f0 = WidenI8x8(row + i);
		const __m256 f1 = WidenI8x8(row + i + 8);
		const __m256 f2 = WidenI8x8(row + i + 16);
		const __m256 f3 = WidenI8x8(row + i + 24);
		a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(qa + i), a0);
		a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(qa + i + 8), a1);
		a2 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(qa + i + 16), a2);
		a3 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(qa + i + 24), a3);
		b0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(qb + i), b0);
		b1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(qb + i + 8), b1);
		b2 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(qb + i + 16), b2);
		b3 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(qb + i + 24), b3);
	}
	if (i + 16 <= paddedDims)
	{
		const __m256 f0 = WidenI8x8(row + i);
		const __m256 f1 = WidenI8x8(row + i + 8);
		a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(qa + i), a0);
		a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(qa + i + 8), a1);
		b0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(qb + i), b0);
		b1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(qb + i + 8), b1);
	}
	*outA = rowScale * ((SumLanes8(a0) + SumLanes8(a1)) + (SumLanes8(a2) + SumLanes8(a3)));
	*outB = rowScale * ((SumLanes8(b0) + SumLanes8(b1)) + (SumLanes8(b2) + SumLanes8(b3)));
}

SUPERFAISS_AVX2_TARGET
void L2I8PairAvx2(const int8_t* row, float rowScale, const float* qa, const float* qb,
	int32_t paddedDims, float* outA, float* outB)
{
	const __m256 scaleV = _mm256_set1_ps(rowScale);
	__m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
	__m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
	__m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
	__m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		const __m256 f0 = WidenI8x8(row + i);
		const __m256 f1 = WidenI8x8(row + i + 8);
		const __m256 f2 = WidenI8x8(row + i + 16);
		const __m256 f3 = WidenI8x8(row + i + 24);
		const __m256 da0 = _mm256_fnmadd_ps(scaleV, f0, _mm256_loadu_ps(qa + i));
		const __m256 da1 = _mm256_fnmadd_ps(scaleV, f1, _mm256_loadu_ps(qa + i + 8));
		const __m256 da2 = _mm256_fnmadd_ps(scaleV, f2, _mm256_loadu_ps(qa + i + 16));
		const __m256 da3 = _mm256_fnmadd_ps(scaleV, f3, _mm256_loadu_ps(qa + i + 24));
		a0 = _mm256_fmadd_ps(da0, da0, a0);
		a1 = _mm256_fmadd_ps(da1, da1, a1);
		a2 = _mm256_fmadd_ps(da2, da2, a2);
		a3 = _mm256_fmadd_ps(da3, da3, a3);
		const __m256 db0 = _mm256_fnmadd_ps(scaleV, f0, _mm256_loadu_ps(qb + i));
		const __m256 db1 = _mm256_fnmadd_ps(scaleV, f1, _mm256_loadu_ps(qb + i + 8));
		const __m256 db2 = _mm256_fnmadd_ps(scaleV, f2, _mm256_loadu_ps(qb + i + 16));
		const __m256 db3 = _mm256_fnmadd_ps(scaleV, f3, _mm256_loadu_ps(qb + i + 24));
		b0 = _mm256_fmadd_ps(db0, db0, b0);
		b1 = _mm256_fmadd_ps(db1, db1, b1);
		b2 = _mm256_fmadd_ps(db2, db2, b2);
		b3 = _mm256_fmadd_ps(db3, db3, b3);
	}
	if (i + 16 <= paddedDims)
	{
		const __m256 f0 = WidenI8x8(row + i);
		const __m256 f1 = WidenI8x8(row + i + 8);
		const __m256 da0 = _mm256_fnmadd_ps(scaleV, f0, _mm256_loadu_ps(qa + i));
		const __m256 da1 = _mm256_fnmadd_ps(scaleV, f1, _mm256_loadu_ps(qa + i + 8));
		a0 = _mm256_fmadd_ps(da0, da0, a0);
		a1 = _mm256_fmadd_ps(da1, da1, a1);
		const __m256 db0 = _mm256_fnmadd_ps(scaleV, f0, _mm256_loadu_ps(qb + i));
		const __m256 db1 = _mm256_fnmadd_ps(scaleV, f1, _mm256_loadu_ps(qb + i + 8));
		b0 = _mm256_fmadd_ps(db0, db0, b0);
		b1 = _mm256_fmadd_ps(db1, db1, b1);
	}
	*outA = (SumLanes8(a0) + SumLanes8(a1)) + (SumLanes8(a2) + SumLanes8(a3));
	*outB = (SumLanes8(b0) + SumLanes8(b1)) + (SumLanes8(b2) + SumLanes8(b3));
}

// --- Scalar mirrors of the AVX2 path (std::fma; test reference, not a hot path) ------

float DotF32ScalarAvx2(const float* row, const float* query, int32_t paddedDims)
{
	Acc8 acc[4];
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		acc[0].FmaMul(row + i, query + i);
		acc[1].FmaMul(row + i + 8, query + i + 8);
		acc[2].FmaMul(row + i + 16, query + i + 16);
		acc[3].FmaMul(row + i + 24, query + i + 24);
	}
	for (int32_t g = 0; i + 8 <= paddedDims; i += 8, ++g)
	{
		acc[g].FmaMul(row + i, query + i);
	}
	return (acc[0].Sum() + acc[1].Sum()) + (acc[2].Sum() + acc[3].Sum());
}

float L2F32ScalarAvx2(const float* row, const float* query, int32_t paddedDims)
{
	Acc8 acc[4];
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		acc[0].FmaDiffSq(query + i, row + i);
		acc[1].FmaDiffSq(query + i + 8, row + i + 8);
		acc[2].FmaDiffSq(query + i + 16, row + i + 16);
		acc[3].FmaDiffSq(query + i + 24, row + i + 24);
	}
	for (int32_t g = 0; i + 8 <= paddedDims; i += 8, ++g)
	{
		acc[g].FmaDiffSq(query + i, row + i);
	}
	return (acc[0].Sum() + acc[1].Sum()) + (acc[2].Sum() + acc[3].Sum());
}

float DotI8ScalarAvx2(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	Acc8 acc[4];
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		acc[0].FmaMulI8(row + i, query + i);
		acc[1].FmaMulI8(row + i + 8, query + i + 8);
		acc[2].FmaMulI8(row + i + 16, query + i + 16);
		acc[3].FmaMulI8(row + i + 24, query + i + 24);
	}
	if (i + 16 <= paddedDims)
	{
		acc[0].FmaMulI8(row + i, query + i);
		acc[1].FmaMulI8(row + i + 8, query + i + 8);
	}
	return scale * ((acc[0].Sum() + acc[1].Sum()) + (acc[2].Sum() + acc[3].Sum()));
}

float L2I8ScalarAvx2(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	Acc8 acc[4];
	int32_t i = 0;
	for (; i + 32 <= paddedDims; i += 32)
	{
		acc[0].FmaDiffSqI8(row + i, scale, query + i);
		acc[1].FmaDiffSqI8(row + i + 8, scale, query + i + 8);
		acc[2].FmaDiffSqI8(row + i + 16, scale, query + i + 16);
		acc[3].FmaDiffSqI8(row + i + 24, scale, query + i + 24);
	}
	if (i + 16 <= paddedDims)
	{
		acc[0].FmaDiffSqI8(row + i, scale, query + i);
		acc[1].FmaDiffSqI8(row + i + 8, scale, query + i + 8);
	}
	return (acc[0].Sum() + acc[1].Sum()) + (acc[2].Sum() + acc[3].Sum());
}

} // namespace detail
} // namespace superfaiss

#endif // x86

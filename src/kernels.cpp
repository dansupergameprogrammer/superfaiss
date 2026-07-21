#include "superfaiss/kernels.h"

#include <cmath>
#include <cstring>

// Compile-time SIMD selection: NEON on ARM, SSE4.1 on x86/x64, scalar elsewhere.
// Every path implements the same accumulation structure (see the comment on the scalar
// kernels); results are bit-identical across paths on a given device.
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
	#define SUPERFAISS_SIMD_NEON 1
	#include <arm_neon.h>
#elif defined(_M_X64) || defined(__x86_64__) || defined(__SSE4_1__) || \
	(defined(_M_IX86_FP) && _M_IX86_FP >= 2)
	#define SUPERFAISS_SIMD_SSE 1
	#include <smmintrin.h>
	#if defined(_MSC_VER)
		#include <intrin.h>
	#endif
#else
	#define SUPERFAISS_SIMD_SCALAR 1
#endif

namespace superfaiss
{

#if defined(SUPERFAISS_SIMD_SSE)
namespace detail
{
	// Defined in kernels_avx2.cpp.
	float DotF32Avx2(const float* row, const float* query, int32_t paddedDims);
	float L2F32Avx2(const float* row, const float* query, int32_t paddedDims);
	float DotI8Avx2(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8Avx2(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	static bool DetectAvx2Fma()
	{
#if defined(__clang__) || defined(__GNUC__)
		return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
		int32_t info1[4] = {};
		__cpuidex(info1, 1, 0);
		const bool osxsave = (info1[2] & (1 << 27)) != 0;
		const bool avx = (info1[2] & (1 << 28)) != 0;
		const bool fma = (info1[2] & (1 << 12)) != 0;
		if (!osxsave || !avx || !fma)
		{
			return false;
		}
		// OS must preserve YMM state.
		if ((_xgetbv(0) & 0x6) != 0x6)
		{
			return false;
		}
		int32_t info7[4] = {};
		__cpuidex(info7, 7, 0);
		return (info7[1] & (1 << 5)) != 0;
#endif
	}

	// Lazy magic-static: immune to cross-TU static-init ordering.
	static bool IsAvx2()
	{
		static const bool v = DetectAvx2Fma();
		return v;
	}
}
#endif // SUPERFAISS_SIMD_SSE

SimdPath ActiveSimdPath()
{
#if defined(SUPERFAISS_SIMD_NEON)
	return SimdPath::NEON;
#elif defined(SUPERFAISS_SIMD_SSE)
	return detail::IsAvx2() ? SimdPath::AVX2 : SimdPath::SSE;
#else
	return SimdPath::Scalar;
#endif
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
// results bit-identical on a device. No FMA anywhere: multiply then add, so compilers
// must not contract (the library is built with fp-contract off / float_control precise).

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

#if defined(SUPERFAISS_SIMD_SSE)

namespace
{
	// Lane combine identical to Acc4::Sum(): (l0+l1)+(l2+l3), computed in scalar floats
	// so the ordering is exact.
	inline float SumLanes(__m128 v)
	{
		alignas(16) float lanes[4];
		_mm_store_ps(lanes, v);
		return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
	}

}

static float DotF32Sse(const float* row, const float* query, int32_t paddedDims)
{
	__m128 acc0 = _mm_setzero_ps();
	__m128 acc1 = _mm_setzero_ps();
	__m128 acc2 = _mm_setzero_ps();
	__m128 acc3 = _mm_setzero_ps();
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_load_ps(row + i), _mm_load_ps(query + i)));
		acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_load_ps(row + i + 4), _mm_load_ps(query + i + 4)));
		acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_load_ps(row + i + 8), _mm_load_ps(query + i + 8)));
		acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_load_ps(row + i + 12), _mm_load_ps(query + i + 12)));
	}
	// Tail: paddedDims % 16 is 0, 4, 8, or 12 — up to three 4-element groups feeding
	// acc0..acc2 in order. Unrolled so no accumulator is address-taken (spill trap).
	if (i + 4 <= paddedDims)
	{
		acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_load_ps(row + i), _mm_load_ps(query + i)));
		i += 4;
	}
	if (i + 4 <= paddedDims)
	{
		acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_load_ps(row + i), _mm_load_ps(query + i)));
		i += 4;
	}
	if (i + 4 <= paddedDims)
	{
		acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_load_ps(row + i), _mm_load_ps(query + i)));
	}
	return (SumLanes(acc0) + SumLanes(acc1)) + (SumLanes(acc2) + SumLanes(acc3));
}

static float L2F32Sse(const float* row, const float* query, int32_t paddedDims)
{
	__m128 acc0 = _mm_setzero_ps();
	__m128 acc1 = _mm_setzero_ps();
	__m128 acc2 = _mm_setzero_ps();
	__m128 acc3 = _mm_setzero_ps();
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		const __m128 d0 = _mm_sub_ps(_mm_load_ps(query + i), _mm_load_ps(row + i));
		const __m128 d1 = _mm_sub_ps(_mm_load_ps(query + i + 4), _mm_load_ps(row + i + 4));
		const __m128 d2 = _mm_sub_ps(_mm_load_ps(query + i + 8), _mm_load_ps(row + i + 8));
		const __m128 d3 = _mm_sub_ps(_mm_load_ps(query + i + 12), _mm_load_ps(row + i + 12));
		acc0 = _mm_add_ps(acc0, _mm_mul_ps(d0, d0));
		acc1 = _mm_add_ps(acc1, _mm_mul_ps(d1, d1));
		acc2 = _mm_add_ps(acc2, _mm_mul_ps(d2, d2));
		acc3 = _mm_add_ps(acc3, _mm_mul_ps(d3, d3));
	}
	if (i + 4 <= paddedDims)
	{
		const __m128 d = _mm_sub_ps(_mm_load_ps(query + i), _mm_load_ps(row + i));
		acc0 = _mm_add_ps(acc0, _mm_mul_ps(d, d));
		i += 4;
	}
	if (i + 4 <= paddedDims)
	{
		const __m128 d = _mm_sub_ps(_mm_load_ps(query + i), _mm_load_ps(row + i));
		acc1 = _mm_add_ps(acc1, _mm_mul_ps(d, d));
		i += 4;
	}
	if (i + 4 <= paddedDims)
	{
		const __m128 d = _mm_sub_ps(_mm_load_ps(query + i), _mm_load_ps(row + i));
		acc2 = _mm_add_ps(acc2, _mm_mul_ps(d, d));
	}
	return (SumLanes(acc0) + SumLanes(acc1)) + (SumLanes(acc2) + SumLanes(acc3));
}

static float DotI8Sse(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	__m128 acc0 = _mm_setzero_ps();
	__m128 acc1 = _mm_setzero_ps();
	__m128 acc2 = _mm_setzero_ps();
	__m128 acc3 = _mm_setzero_ps();
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		const __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(row + i));
		const __m128 f0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(v));
		const __m128 f1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(v, 4)));
		const __m128 f2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(v, 8)));
		const __m128 f3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(v, 12)));
		acc0 = _mm_add_ps(acc0, _mm_mul_ps(f0, _mm_load_ps(query + i)));
		acc1 = _mm_add_ps(acc1, _mm_mul_ps(f1, _mm_load_ps(query + i + 4)));
		acc2 = _mm_add_ps(acc2, _mm_mul_ps(f2, _mm_load_ps(query + i + 8)));
		acc3 = _mm_add_ps(acc3, _mm_mul_ps(f3, _mm_load_ps(query + i + 12)));
	}
	// Int8 paddedDims is always a multiple of 16, so there is no remainder loop.
	return scale * ((SumLanes(acc0) + SumLanes(acc1)) + (SumLanes(acc2) + SumLanes(acc3)));
}

static float L2I8Sse(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	const __m128 scaleV = _mm_set1_ps(scale);
	__m128 acc0 = _mm_setzero_ps();
	__m128 acc1 = _mm_setzero_ps();
	__m128 acc2 = _mm_setzero_ps();
	__m128 acc3 = _mm_setzero_ps();
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		const __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(row + i));
		const __m128 f0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(v));
		const __m128 f1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(v, 4)));
		const __m128 f2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(v, 8)));
		const __m128 f3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(v, 12)));
		const __m128 d0 = _mm_sub_ps(_mm_load_ps(query + i), _mm_mul_ps(scaleV, f0));
		const __m128 d1 = _mm_sub_ps(_mm_load_ps(query + i + 4), _mm_mul_ps(scaleV, f1));
		const __m128 d2 = _mm_sub_ps(_mm_load_ps(query + i + 8), _mm_mul_ps(scaleV, f2));
		const __m128 d3 = _mm_sub_ps(_mm_load_ps(query + i + 12), _mm_mul_ps(scaleV, f3));
		acc0 = _mm_add_ps(acc0, _mm_mul_ps(d0, d0));
		acc1 = _mm_add_ps(acc1, _mm_mul_ps(d1, d1));
		acc2 = _mm_add_ps(acc2, _mm_mul_ps(d2, d2));
		acc3 = _mm_add_ps(acc3, _mm_mul_ps(d3, d3));
	}
	return (SumLanes(acc0) + SumLanes(acc1)) + (SumLanes(acc2) + SumLanes(acc3));
}

// Dispatchers: AVX2+FMA when the CPU has it and the stride fits 8-lane blocks
// (float32 strides are multiples of 4, not always 8; int8 strides are always multiples
// of 16). The choice is a pure function of device and bank shape, so per-device
// determinism is preserved.

float DotF32(const float* row, const float* query, int32_t paddedDims)
{
	return (IsAvx2() && (paddedDims % 8) == 0)
		? DotF32Avx2(row, query, paddedDims)
		: DotF32Sse(row, query, paddedDims);
}

float L2F32(const float* row, const float* query, int32_t paddedDims)
{
	return (IsAvx2() && (paddedDims % 8) == 0)
		? L2F32Avx2(row, query, paddedDims)
		: L2F32Sse(row, query, paddedDims);
}

float DotI8(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return IsAvx2()
		? DotI8Avx2(row, scale, query, paddedDims)
		: DotI8Sse(row, scale, query, paddedDims);
}

float L2I8(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return IsAvx2()
		? L2I8Avx2(row, scale, query, paddedDims)
		: L2I8Sse(row, scale, query, paddedDims);
}

// Mirrors: the scalar twin of whatever the dispatcher above would pick.

float DotF32Mirror(const float* row, const float* query, int32_t paddedDims)
{
	return (IsAvx2() && (paddedDims % 8) == 0)
		? DotF32ScalarAvx2(row, query, paddedDims)
		: DotF32Scalar(row, query, paddedDims);
}

float L2F32Mirror(const float* row, const float* query, int32_t paddedDims)
{
	return (IsAvx2() && (paddedDims % 8) == 0)
		? L2F32ScalarAvx2(row, query, paddedDims)
		: L2F32Scalar(row, query, paddedDims);
}

float DotI8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return IsAvx2()
		? DotI8ScalarAvx2(row, scale, query, paddedDims)
		: DotI8Scalar(row, scale, query, paddedDims);
}

float L2I8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return IsAvx2()
		? L2I8ScalarAvx2(row, scale, query, paddedDims)
		: L2I8Scalar(row, scale, query, paddedDims);
}

#elif defined(SUPERFAISS_SIMD_NEON)

namespace
{
	inline float SumLanes(float32x4_t v)
	{
		alignas(16) float lanes[4];
		vst1q_f32(lanes, v);
		return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
	}

	// Widen 16 int8 values into four float32x4 groups, in element order.
	struct I8x16AsFloat
	{
		float32x4_t f0, f1, f2, f3;
	};

	inline I8x16AsFloat WidenI8x16(const int8_t* p)
	{
		const int8x16_t v = vld1q_s8(p);
		const int16x8_t lo = vmovl_s8(vget_low_s8(v));
		const int16x8_t hi = vmovl_s8(vget_high_s8(v));
		I8x16AsFloat r;
		r.f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
		r.f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
		r.f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
		r.f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
		return r;
	}
}

// NOTE: vmulq/vaddq are used instead of vmlaq/vfmaq deliberately — fused multiply-add
// would change rounding and break bit-equality with the scalar path.

float DotF32(const float* row, const float* query, int32_t paddedDims)
{
	float32x4_t acc0 = vdupq_n_f32(0.0f);
	float32x4_t acc1 = vdupq_n_f32(0.0f);
	float32x4_t acc2 = vdupq_n_f32(0.0f);
	float32x4_t acc3 = vdupq_n_f32(0.0f);
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(row + i), vld1q_f32(query + i)));
		acc1 = vaddq_f32(acc1, vmulq_f32(vld1q_f32(row + i + 4), vld1q_f32(query + i + 4)));
		acc2 = vaddq_f32(acc2, vmulq_f32(vld1q_f32(row + i + 8), vld1q_f32(query + i + 8)));
		acc3 = vaddq_f32(acc3, vmulq_f32(vld1q_f32(row + i + 12), vld1q_f32(query + i + 12)));
	}
	if (i + 4 <= paddedDims)
	{
		acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(row + i), vld1q_f32(query + i)));
		i += 4;
	}
	if (i + 4 <= paddedDims)
	{
		acc1 = vaddq_f32(acc1, vmulq_f32(vld1q_f32(row + i), vld1q_f32(query + i)));
		i += 4;
	}
	if (i + 4 <= paddedDims)
	{
		acc2 = vaddq_f32(acc2, vmulq_f32(vld1q_f32(row + i), vld1q_f32(query + i)));
	}
	return (SumLanes(acc0) + SumLanes(acc1)) + (SumLanes(acc2) + SumLanes(acc3));
}

float L2F32(const float* row, const float* query, int32_t paddedDims)
{
	float32x4_t acc0 = vdupq_n_f32(0.0f);
	float32x4_t acc1 = vdupq_n_f32(0.0f);
	float32x4_t acc2 = vdupq_n_f32(0.0f);
	float32x4_t acc3 = vdupq_n_f32(0.0f);
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		const float32x4_t d0 = vsubq_f32(vld1q_f32(query + i), vld1q_f32(row + i));
		const float32x4_t d1 = vsubq_f32(vld1q_f32(query + i + 4), vld1q_f32(row + i + 4));
		const float32x4_t d2 = vsubq_f32(vld1q_f32(query + i + 8), vld1q_f32(row + i + 8));
		const float32x4_t d3 = vsubq_f32(vld1q_f32(query + i + 12), vld1q_f32(row + i + 12));
		acc0 = vaddq_f32(acc0, vmulq_f32(d0, d0));
		acc1 = vaddq_f32(acc1, vmulq_f32(d1, d1));
		acc2 = vaddq_f32(acc2, vmulq_f32(d2, d2));
		acc3 = vaddq_f32(acc3, vmulq_f32(d3, d3));
	}
	if (i + 4 <= paddedDims)
	{
		const float32x4_t d = vsubq_f32(vld1q_f32(query + i), vld1q_f32(row + i));
		acc0 = vaddq_f32(acc0, vmulq_f32(d, d));
		i += 4;
	}
	if (i + 4 <= paddedDims)
	{
		const float32x4_t d = vsubq_f32(vld1q_f32(query + i), vld1q_f32(row + i));
		acc1 = vaddq_f32(acc1, vmulq_f32(d, d));
		i += 4;
	}
	if (i + 4 <= paddedDims)
	{
		const float32x4_t d = vsubq_f32(vld1q_f32(query + i), vld1q_f32(row + i));
		acc2 = vaddq_f32(acc2, vmulq_f32(d, d));
	}
	return (SumLanes(acc0) + SumLanes(acc1)) + (SumLanes(acc2) + SumLanes(acc3));
}

float DotI8(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	float32x4_t acc0 = vdupq_n_f32(0.0f);
	float32x4_t acc1 = vdupq_n_f32(0.0f);
	float32x4_t acc2 = vdupq_n_f32(0.0f);
	float32x4_t acc3 = vdupq_n_f32(0.0f);
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		const I8x16AsFloat f = WidenI8x16(row + i);
		acc0 = vaddq_f32(acc0, vmulq_f32(f.f0, vld1q_f32(query + i)));
		acc1 = vaddq_f32(acc1, vmulq_f32(f.f1, vld1q_f32(query + i + 4)));
		acc2 = vaddq_f32(acc2, vmulq_f32(f.f2, vld1q_f32(query + i + 8)));
		acc3 = vaddq_f32(acc3, vmulq_f32(f.f3, vld1q_f32(query + i + 12)));
	}
	return scale * ((SumLanes(acc0) + SumLanes(acc1)) + (SumLanes(acc2) + SumLanes(acc3)));
}

float L2I8(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	const float32x4_t scaleV = vdupq_n_f32(scale);
	float32x4_t acc0 = vdupq_n_f32(0.0f);
	float32x4_t acc1 = vdupq_n_f32(0.0f);
	float32x4_t acc2 = vdupq_n_f32(0.0f);
	float32x4_t acc3 = vdupq_n_f32(0.0f);
	int32_t i = 0;
	for (; i + 16 <= paddedDims; i += 16)
	{
		const I8x16AsFloat f = WidenI8x16(row + i);
		const float32x4_t d0 = vsubq_f32(vld1q_f32(query + i), vmulq_f32(scaleV, f.f0));
		const float32x4_t d1 = vsubq_f32(vld1q_f32(query + i + 4), vmulq_f32(scaleV, f.f1));
		const float32x4_t d2 = vsubq_f32(vld1q_f32(query + i + 8), vmulq_f32(scaleV, f.f2));
		const float32x4_t d3 = vsubq_f32(vld1q_f32(query + i + 12), vmulq_f32(scaleV, f.f3));
		acc0 = vaddq_f32(acc0, vmulq_f32(d0, d0));
		acc1 = vaddq_f32(acc1, vmulq_f32(d1, d1));
		acc2 = vaddq_f32(acc2, vmulq_f32(d2, d2));
		acc3 = vaddq_f32(acc3, vmulq_f32(d3, d3));
	}
	return (SumLanes(acc0) + SumLanes(acc1)) + (SumLanes(acc2) + SumLanes(acc3));
}

// NEON path: the plain scalar kernels are the exact mirror.

float DotF32Mirror(const float* row, const float* query, int32_t paddedDims)
{
	return DotF32Scalar(row, query, paddedDims);
}

float L2F32Mirror(const float* row, const float* query, int32_t paddedDims)
{
	return L2F32Scalar(row, query, paddedDims);
}

float DotI8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return DotI8Scalar(row, scale, query, paddedDims);
}

float L2I8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return L2I8Scalar(row, scale, query, paddedDims);
}

#else // scalar fallback

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

float DotF32Mirror(const float* row, const float* query, int32_t paddedDims)
{
	return DotF32Scalar(row, query, paddedDims);
}

float L2F32Mirror(const float* row, const float* query, int32_t paddedDims)
{
	return L2F32Scalar(row, query, paddedDims);
}

float DotI8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return DotI8Scalar(row, scale, query, paddedDims);
}

float L2I8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims)
{
	return L2I8Scalar(row, scale, query, paddedDims);
}

#endif

} // namespace detail

#if defined(SUPERFAISS_SIMD_SSE)
namespace detail
{
	// Defined in kernels_avx2.cpp.
	void DotF32PairAvx2(const float* row, const float* qa, const float* qb,
		int32_t paddedDims, float* outA, float* outB);
	void L2F32PairAvx2(const float* row, const float* qa, const float* qb,
		int32_t paddedDims, float* outA, float* outB);
	void DotI8PairAvx2(const int8_t* row, float scale, const float* qa, const float* qb,
		int32_t paddedDims, float* outA, float* outB);
	void L2I8PairAvx2(const int8_t* row, float scale, const float* qa, const float* qb,
		int32_t paddedDims, float* outA, float* outB);
}
#endif

namespace
{
	// v2.1 dense bias composition: ONE fused add after dequantized scoring, before
	// top-k insertion; a non-finite bias value raises the caller's flag (fused
	// validation, T-055 W2) - the scan completes and the caller returns
	// NonFiniteQuery. Null bias executes no add: the bit-identical unbiased path.
	inline float ComposeBias(float score, const float* rowBias, int32_t r, bool* flag)
	{
		if (rowBias == nullptr)
		{
			return score;
		}
		const float b = rowBias[r];
		// Integer exponent test (all-ones = Inf/NaN): std::isfinite lowers to a
		// classify call under /fp:precise and costs double digits of scan time;
		// this is two ALU ops.
		uint32_t bits;
		std::memcpy(&bits, &b, sizeof(bits));
		if ((bits & 0x7f800000u) == 0x7f800000u && flag != nullptr)
		{
			*flag = true;
		}
		return score + b;
	}
}

void ScoreChunkPair(
	const BankView& bank,
	const float* paddedQueryA,
	const float* paddedQueryB,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inoutA,
	TopK& inoutB,
	const float* rowBiasA,
	const float* rowBiasB,
	bool* outNonFiniteBias)
{
#if defined(SUPERFAISS_SIMD_SSE)
	// AVX2 pair kernels share the row pass; identical per-query accumulation keeps
	// results bit-equal to two single ScoreChunk calls.
	const bool pairPath = detail::IsAvx2() &&
		(bank.quant == Quantization::Int8 || (bank.paddedDims % 8) == 0);
	if (pairPath)
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
		float scoreA = 0.0f;
		float scoreB = 0.0f;

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
				if (isL2)
				{
					detail::L2F32PairAvx2(row, paddedQueryA, paddedQueryB, pd, &scoreA, &scoreB);
				}
				else
				{
					detail::DotF32PairAvx2(row, paddedQueryA, paddedQueryB, pd, &scoreA, &scoreB);
				}
				inoutA.Push(r, ComposeBias(scoreA, rowBiasA, r, outNonFiniteBias));
				inoutB.Push(r, ComposeBias(scoreB, rowBiasB, r, outNonFiniteBias));
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
				if (isL2)
				{
					detail::L2I8PairAvx2(row, scale, paddedQueryA, paddedQueryB, pd, &scoreA, &scoreB);
				}
				else
				{
					detail::DotI8PairAvx2(row, scale, paddedQueryA, paddedQueryB, pd, &scoreA, &scoreB);
				}
				inoutA.Push(r, ComposeBias(scoreA, rowBiasA, r, outNonFiniteBias));
				inoutB.Push(r, ComposeBias(scoreB, rowBiasB, r, outNonFiniteBias));
			}
		}
		return;
	}
#endif
	// Fallback: two single passes (bit-identical by definition; SSE/NEON pair kernels
	// are a recorded follow-up).
	ScoreChunk(bank, paddedQueryA, chunkIndex, excludeBits, inoutA, rowBiasA, outNonFiniteBias);
	ScoreChunk(bank, paddedQueryB, chunkIndex, excludeBits, inoutB, rowBiasB, outNonFiniteBias);
}

void ScoreChunk(
	const BankView& bank,
	const float* paddedQuery,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout,
	const float* rowBias,
	bool* outNonFiniteBias)
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
			inout.Push(r, ComposeBias(score, rowBias, r, outNonFiniteBias));
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
			inout.Push(r, ComposeBias(score, rowBias, r, outNonFiniteBias));
		}
	}
}

void ScoreChunkFused(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout,
	const float* rowBias,
	bool* outNonFiniteBias)
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
			float fused = 0.0f;
			for (int32_t m = 0; m < queryCount; ++m)
			{
				const float* q = paddedQueries + static_cast<int64_t>(m) * pd;
				const float score = isL2 ? detail::L2F32(row, q, pd)
				                         : detail::DotF32(row, q, pd);
				// Worst-of in the better-direction: max distance / min similarity.
				if (m == 0 || (isL2 ? score > fused : score < fused))
				{
					fused = score;
				}
			}
			// Bias applies once, to the fused score (plan 18.2).
			inout.Push(r, ComposeBias(fused, rowBias, r, outNonFiniteBias));
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
			float fused = 0.0f;
			for (int32_t m = 0; m < queryCount; ++m)
			{
				const float* q = paddedQueries + static_cast<int64_t>(m) * pd;
				const float score = isL2 ? detail::L2I8(row, scale, q, pd)
				                         : detail::DotI8(row, scale, q, pd);
				if (m == 0 || (isL2 ? score > fused : score < fused))
				{
					fused = score;
				}
			}
			inout.Push(r, ComposeBias(fused, rowBias, r, outNonFiniteBias));
		}
	}
}

namespace
{
	// The dense segmented scan (V2 plan section 10 decision, T-050-corrected):
	// one contiguous pass over the whole row - gaps between and after segments are
	// scored into a discarded partial so reads stay sequential and the prefetcher
	// never sees a stride - with per-segment partials produced by the SAME per-row
	// kernel bodies the V1 scan uses (range-parameterized by construction), and
	// weights applied at combine: total = sum(weight_s * partial_s). A single
	// full-row segment therefore IS the V1 computation, bit-identically. Kernel
	// dispatch is resolved once per chunk, not once per range.

	struct FRowKernels
	{
		float (*dotF32)(const float*, const float*, int32_t);
		float (*l2F32)(const float*, const float*, int32_t);
		float (*dotI8)(const int8_t*, float, const float*, int32_t);
		float (*l2I8)(const int8_t*, float, const float*, int32_t);
	};

	inline FRowKernels ResolveRowKernels()
	{
		FRowKernels k;
#if defined(SUPERFAISS_SIMD_SSE)
		if (detail::IsAvx2())
		{
			k.dotF32 = detail::DotF32Avx2;
			k.l2F32 = detail::L2F32Avx2;
			k.dotI8 = detail::DotI8Avx2;
			k.l2I8 = detail::L2I8Avx2;
			return k;
		}
#endif
		k.dotF32 = detail::DotF32;
		k.l2F32 = detail::L2F32;
		k.dotI8 = detail::DotI8;
		k.l2I8 = detail::L2I8;
		return k;
	}

	// A range is a contiguous element run: a live segment (accIndex >= 0, weight
	// applied at combine) or a gap (accIndex < 0, partial discarded). channelIndex
	// (>= 0 when the range exactly matches a bank channel) selects the per-row
	// inverse sub-norm on per-channel-cosine banks.
	struct FScanRange
	{
		int32_t offset;
		int32_t length;
		float weight;
		int32_t accIndex;     // -1 = discard
		int32_t channelIndex; // -1 = no channel match
	};

	inline int32_t MatchChannel(const BankView& bank, int32_t offset, int32_t length)
	{
		for (int32_t c = 0; c < bank.channelCount; ++c)
		{
			if (bank.channels[c].offset == offset && bank.channels[c].length == length)
			{
				return c;
			}
		}
		return -1;
	}

	// Builds the full-coverage range list: segments in order, gaps filled, trailing
	// gap to paddedDims. Returns the range count (<= 2*kMaxSegments + 1).
	inline int32_t BuildScanRanges(
		const BankView& bank,
		const QuerySegment* segments,
		int32_t segmentCount,
		FScanRange* outRanges)
	{
		int32_t n = 0;
		int32_t cursor = 0;
		for (int32_t i = 0; i < segmentCount; ++i)
		{
			const QuerySegment& seg = segments[i];
			if (seg.offset > cursor)
			{
				outRanges[n++] = {cursor, seg.offset - cursor, 0.0f, -1, -1};
			}
			// Weight-0 segments keep omission semantics: scanned for stride
			// continuity, discarded like a gap.
			outRanges[n++] = {seg.offset, seg.length, seg.weight,
				seg.weight == 0.0f ? -1 : i,
				MatchChannel(bank, seg.offset, seg.length)};
			cursor = seg.offset + seg.length;
		}
		if (cursor < bank.paddedDims)
		{
			outRanges[n++] = {cursor, bank.paddedDims - cursor, 0.0f, -1, -1};
		}
		return n;
	}

	// Scores one row over the range list. outPartials (kMaxSegments floats) receives
	// unweighted per-segment partials when non-null (the decomposition surface);
	// the return value is the weighted combine.
	// rowInvNorms: per-channel-cosine banks pass the row's inverse sub-norm slice
	// (channelCount floats); a channel-matched range's partial scales by its stored
	// inverse sub-norm BEFORE the weight (D-V2-1: true per-channel cosine; a
	// zero-norm row channel stored 0 and scores 0 - defined, never NaN).
	// outPartials receives the post-scale, post-weight per-segment contributions
	// (the decomposition surface: contributions sum bit-exactly to the total).
	inline float DenseSegmentedRowScore(
		const void* row,
		float scale,
		const float* paddedQuery,
		const FScanRange* ranges,
		int32_t rangeCount,
		const FRowKernels& kernels,
		bool isL2,
		bool isInt8,
		const float* rowInvNorms,
		float* outPartials)
	{
		float total = 0.0f;
		for (int32_t i = 0; i < rangeCount; ++i)
		{
			const FScanRange& range = ranges[i];
			float partial;
			if (isInt8)
			{
				const int8_t* r = static_cast<const int8_t*>(row) + range.offset;
				partial = isL2
					? kernels.l2I8(r, scale, paddedQuery + range.offset, range.length)
					: kernels.dotI8(r, scale, paddedQuery + range.offset, range.length);
			}
			else
			{
				const float* r = static_cast<const float*>(row) + range.offset;
				partial = isL2
					? kernels.l2F32(r, paddedQuery + range.offset, range.length)
					: kernels.dotF32(r, paddedQuery + range.offset, range.length);
			}
			if (range.accIndex >= 0)
			{
				if (rowInvNorms != nullptr && range.channelIndex >= 0)
				{
					partial *= rowInvNorms[range.channelIndex];
				}
				const float contribution = range.weight * partial;
				if (outPartials != nullptr)
				{
					outPartials[range.accIndex] = contribution;
				}
				total += contribution;
			}
			// Gap partials are computed for stride continuity and discarded; their
			// products never touch a live score (T-050 W1 discard semantics).
		}
		return total;
	}
}

void ScoreChunkSegmented(
	const BankView& bank,
	const float* paddedQuery,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout,
	const float* rowBias,
	bool* outNonFiniteBias)
{
	const int32_t chunkRows = ChunkRows(bank);
	const int32_t begin = chunkIndex * chunkRows;
	int32_t end = begin + chunkRows;
	if (end > bank.count)
	{
		end = bank.count;
	}

	const bool isL2 = bank.metric == Metric::L2;
	const bool isInt8 = bank.quant == Quantization::Int8;
	const bool perChannelCosine =
		bank.metric == Metric::Cosine && bank.channelInvNorms != nullptr;
	const FRowKernels kernels = ResolveRowKernels();
	FScanRange ranges[2 * kMaxSegments + 1];
	const int32_t rangeCount = BuildScanRanges(bank, segments, segmentCount, ranges);
	const int64_t rowBytes = RowBytes(bank);

	for (int32_t r = begin; r < end; ++r)
	{
		if (IsExcluded(excludeBits, r))
		{
			continue;
		}
		const void* row = static_cast<const uint8_t*>(bank.rows) +
			static_cast<int64_t>(r) * rowBytes;
		const float scale = isInt8 ? bank.scales[r] : 1.0f;
		const float* rowInvNorms = perChannelCosine
			? bank.channelInvNorms + static_cast<int64_t>(r) * bank.channelCount
			: nullptr;
		const float total = DenseSegmentedRowScore(row, scale, paddedQuery, ranges,
			rangeCount, kernels, isL2, isInt8, rowInvNorms, nullptr);
		inout.Push(r, ComposeBias(total, rowBias, r, outNonFiniteBias));
	}
}

void ScoreChunkFusedSegmented(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout,
	const float* rowBias,
	bool* outNonFiniteBias)
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
	const bool isInt8 = bank.quant == Quantization::Int8;
	const bool perChannelCosine =
		bank.metric == Metric::Cosine && bank.channelInvNorms != nullptr;
	const FRowKernels kernels = ResolveRowKernels();
	FScanRange ranges[2 * kMaxSegments + 1];
	const int32_t rangeCount = BuildScanRanges(bank, segments, segmentCount, ranges);
	const int64_t rowBytes = RowBytes(bank);

	for (int32_t r = begin; r < end; ++r)
	{
		if (IsExcluded(excludeBits, r))
		{
			continue;
		}
		const void* row = static_cast<const uint8_t*>(bank.rows) +
			static_cast<int64_t>(r) * rowBytes;
		const float scale = isInt8 ? bank.scales[r] : 1.0f;
		const float* rowInvNorms = perChannelCosine
			? bank.channelInvNorms + static_cast<int64_t>(r) * bank.channelCount
			: nullptr;
		float fused = 0.0f;
		for (int32_t m = 0; m < queryCount; ++m)
		{
			const float score = DenseSegmentedRowScore(row, scale,
				paddedQueries + static_cast<int64_t>(m) * pd, ranges, rangeCount,
				kernels, isL2, isInt8, rowInvNorms, nullptr);
			if (m == 0 || (isL2 ? score > fused : score < fused))
			{
				fused = score;
			}
		}
		inout.Push(r, ComposeBias(fused, rowBias, r, outNonFiniteBias));
	}
}

// Per-row decomposition (V2 section 6): the dense row scorer with its partials
// surfaced. outContributions receives segmentCount post-scale post-weight values;
// the returned total is their ordered sum (bit-exact by construction).
float DecomposeRowScore(
	const BankView& bank,
	const float* paddedQuery,
	int32_t rowIndex,
	const QuerySegment* segments,
	int32_t segmentCount,
	float* outContributions)
{
	const bool isL2 = bank.metric == Metric::L2;
	const bool isInt8 = bank.quant == Quantization::Int8;
	const bool perChannelCosine =
		bank.metric == Metric::Cosine && bank.channelInvNorms != nullptr;
	const FRowKernels kernels = ResolveRowKernels();
	FScanRange ranges[2 * kMaxSegments + 1];
	const int32_t rangeCount = BuildScanRanges(bank, segments, segmentCount, ranges);
	const void* row = static_cast<const uint8_t*>(bank.rows) +
		static_cast<int64_t>(rowIndex) * RowBytes(bank);
	const float scale = isInt8 ? bank.scales[rowIndex] : 1.0f;
	const float* rowInvNorms = perChannelCosine
		? bank.channelInvNorms + static_cast<int64_t>(rowIndex) * bank.channelCount
		: nullptr;
	for (int32_t i = 0; i < segmentCount; ++i)
	{
		outContributions[i] = 0.0f; // weight-0 segments contribute exactly 0
	}
	return DenseSegmentedRowScore(row, scale, paddedQuery, ranges, rangeCount,
		kernels, isL2, isInt8, rowInvNorms, outContributions);
}

// ---------------------------------------------------------------------------
// v2.2 cross-device exactness (plan section 19)
//
// The float kernels above are per-device exact; their cross-device hazard is
// reduction shape (8-lane AVX2, 4-lane NEON/SSE, 1-lane scalar sum in different
// orders). The CrossDevice mode removes the hazard at the root: scoring
// accumulates in integers (associative - any width, same sums), and everything
// after the integer sums is one fixed-order double-precision expression per row,
// ending in the subnormal-floor contract (|score| < FLT_MIN is exactly 0.0f).
// Doubles are used because every intermediate value in the epilogue is provably a
// normal double, so per-thread FTZ state can never flush one; float inputs (bank
// scales, channel inverse sub-norms, segment weights, biases) are decoded from
// their bit patterns so per-thread DAZ state can never flush one either.

namespace detail
{

double FloatBitsToDouble(float v)
{
	uint32_t b;
	std::memcpy(&b, &v, sizeof(b));
	if ((b & 0x7f800000u) != 0)
	{
		// Normal (or Inf/NaN): the plain conversion is exact and DAZ-immune.
		return static_cast<double>(v);
	}
	// Zero or subnormal: value = mantissa * 2^-149, exact in double (and a normal
	// double, or zero), built without touching the FP unit's denormal handling.
	const double m = static_cast<double>(static_cast<int32_t>(b & 0x7fffffu)) * 0x1p-149;
	return (b >> 31) != 0 ? -m : m;
}

int32_t RoundHalfEvenI32(double v)
{
	uint64_t bits;
	std::memcpy(&bits, &v, sizeof(bits));
	const bool neg = (bits >> 63) != 0;
	const int32_t exp = static_cast<int32_t>((bits >> 52) & 0x7ff) - 1023;
	uint64_t mant = bits & 0xfffffffffffffull;
	if (exp < -1)
	{
		return 0; // |v| < 0.5 (covers zero and subnormal doubles)
	}
	if (exp == -1)
	{
		// 0.5 <= |v| < 1: exactly 0.5 rounds to 0 (even); anything above to +/-1.
		const int32_t r = mant == 0 ? 0 : 1;
		return neg ? -r : r;
	}
	mant |= 1ull << 52; // implicit bit: value = mant * 2^(exp - 52)
	const int32_t shift = 52 - exp; // caller bounds |v| < 2^31, so shift >= 22
	const uint64_t whole = mant >> shift;
	const uint64_t rem = mant & ((1ull << shift) - 1);
	const uint64_t half = 1ull << (shift - 1);
	uint64_t rounded = whole;
	if (rem > half || (rem == half && (whole & 1) != 0))
	{
		++rounded;
	}
	const int32_t r = static_cast<int32_t>(rounded);
	return neg ? -r : r;
}

int32_t DotI8I8Scalar(const int8_t* row, const int8_t* q8, int32_t n)
{
	int32_t acc = 0;
	for (int32_t i = 0; i < n; ++i)
	{
		acc += static_cast<int32_t>(row[i]) * q8[i];
	}
	return acc;
}

void L2SumsI8I8Scalar(
	const int8_t* row, const int8_t* q8, int32_t n, int32_t* outCross, int32_t* outRowSq)
{
	int32_t cross = 0;
	int32_t rowSq = 0;
	for (int32_t i = 0; i < n; ++i)
	{
		const int32_t r = row[i];
		cross += r * static_cast<int32_t>(q8[i]);
		rowSq += r * r;
	}
	*outCross = cross;
	*outRowSq = rowSq;
}

#if defined(SUPERFAISS_SIMD_SSE)

// Defined in kernels_avx2.cpp.
int32_t DotI8I8Avx2(const int8_t* row, const int8_t* q8, int32_t n);
void L2SumsI8I8Avx2(const int8_t* row, const int8_t* q8, int32_t n,
	int32_t* outCross, int32_t* outRowSq);

namespace
{
	inline int32_t SumLanesI32Sse(__m128i v)
	{
		alignas(16) int32_t l[4];
		_mm_store_si128(reinterpret_cast<__m128i*>(l), v);
		return (l[0] + l[1]) + (l[2] + l[3]);
	}
}

// Sign-extend to int16, combine with _mm_madd_epi16 (signed x signed) - the
// _maddubs unsigned-x-signed hazard is avoided by construction.
static int32_t DotI8I8Sse(const int8_t* row, const int8_t* q8, int32_t n)
{
	__m128i acc = _mm_setzero_si128();
	for (int32_t i = 0; i + 16 <= n; i += 16) // int8 ranges are multiples of 16
	{
		const __m128i r = _mm_load_si128(reinterpret_cast<const __m128i*>(row + i));
		const __m128i q = _mm_load_si128(reinterpret_cast<const __m128i*>(q8 + i));
		const __m128i rlo = _mm_cvtepi8_epi16(r);
		const __m128i rhi = _mm_cvtepi8_epi16(_mm_srli_si128(r, 8));
		const __m128i qlo = _mm_cvtepi8_epi16(q);
		const __m128i qhi = _mm_cvtepi8_epi16(_mm_srli_si128(q, 8));
		acc = _mm_add_epi32(acc, _mm_madd_epi16(rlo, qlo));
		acc = _mm_add_epi32(acc, _mm_madd_epi16(rhi, qhi));
	}
	return SumLanesI32Sse(acc);
}

static void L2SumsI8I8Sse(
	const int8_t* row, const int8_t* q8, int32_t n, int32_t* outCross, int32_t* outRowSq)
{
	__m128i cross = _mm_setzero_si128();
	__m128i rowSq = _mm_setzero_si128();
	for (int32_t i = 0; i + 16 <= n; i += 16)
	{
		const __m128i r = _mm_load_si128(reinterpret_cast<const __m128i*>(row + i));
		const __m128i q = _mm_load_si128(reinterpret_cast<const __m128i*>(q8 + i));
		const __m128i rlo = _mm_cvtepi8_epi16(r);
		const __m128i rhi = _mm_cvtepi8_epi16(_mm_srli_si128(r, 8));
		const __m128i qlo = _mm_cvtepi8_epi16(q);
		const __m128i qhi = _mm_cvtepi8_epi16(_mm_srli_si128(q, 8));
		cross = _mm_add_epi32(cross, _mm_madd_epi16(rlo, qlo));
		cross = _mm_add_epi32(cross, _mm_madd_epi16(rhi, qhi));
		rowSq = _mm_add_epi32(rowSq, _mm_madd_epi16(rlo, rlo));
		rowSq = _mm_add_epi32(rowSq, _mm_madd_epi16(rhi, rhi));
	}
	*outCross = SumLanesI32Sse(cross);
	*outRowSq = SumLanesI32Sse(rowSq);
}

#elif defined(SUPERFAISS_SIMD_NEON)

namespace
{
	inline int32_t SumLanesI32Neon(int32x4_t v)
	{
		alignas(16) int32_t l[4];
		vst1q_s32(l, v);
		return (l[0] + l[1]) + (l[2] + l[3]);
	}
}

// vmull_s8 (int8 x int8 -> int16, exact: |product| <= 16129 < 32767) then
// pairwise-accumulate into int32 lanes.
static int32_t DotI8I8Neon(const int8_t* row, const int8_t* q8, int32_t n)
{
	int32x4_t acc = vdupq_n_s32(0);
	for (int32_t i = 0; i + 16 <= n; i += 16)
	{
		const int8x16_t r = vld1q_s8(row + i);
		const int8x16_t q = vld1q_s8(q8 + i);
		acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(r), vget_low_s8(q)));
		acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(r), vget_high_s8(q)));
	}
	return SumLanesI32Neon(acc);
}

static void L2SumsI8I8Neon(
	const int8_t* row, const int8_t* q8, int32_t n, int32_t* outCross, int32_t* outRowSq)
{
	int32x4_t cross = vdupq_n_s32(0);
	int32x4_t rowSq = vdupq_n_s32(0);
	for (int32_t i = 0; i + 16 <= n; i += 16)
	{
		const int8x16_t r = vld1q_s8(row + i);
		const int8x16_t q = vld1q_s8(q8 + i);
		cross = vpadalq_s16(cross, vmull_s8(vget_low_s8(r), vget_low_s8(q)));
		cross = vpadalq_s16(cross, vmull_s8(vget_high_s8(r), vget_high_s8(q)));
		rowSq = vpadalq_s16(rowSq, vmull_s8(vget_low_s8(r), vget_low_s8(r)));
		rowSq = vpadalq_s16(rowSq, vmull_s8(vget_high_s8(r), vget_high_s8(r)));
	}
	*outCross = SumLanesI32Neon(cross);
	*outRowSq = SumLanesI32Neon(rowSq);
}

#endif

namespace
{
	// Test-only forced dispatch for the CI forced-path matrix (19.4 W1). Not
	// thread-safe; the shipped default is the per-device-stable dispatch below.
	bool GXdPathForced = false;
	SimdPath GXdForcedPath = SimdPath::Scalar;

	inline SimdPath ActiveXdPath()
	{
#if defined(SUPERFAISS_SIMD_NEON)
		return GXdPathForced ? GXdForcedPath : SimdPath::NEON;
#elif defined(SUPERFAISS_SIMD_SSE)
		if (GXdPathForced)
		{
			return GXdForcedPath;
		}
		return IsAvx2() ? SimdPath::AVX2 : SimdPath::SSE;
#else
		return GXdPathForced ? GXdForcedPath : SimdPath::Scalar;
#endif
	}
}

void ForceXdSimdPath(SimdPath path)
{
	GXdPathForced = true;
	GXdForcedPath = path;
}

void ClearForcedXdSimdPath()
{
	GXdPathForced = false;
}

int32_t DotI8I8(const int8_t* row, const int8_t* q8, int32_t n)
{
	switch (ActiveXdPath())
	{
#if defined(SUPERFAISS_SIMD_SSE)
	case SimdPath::AVX2: return DotI8I8Avx2(row, q8, n);
	case SimdPath::SSE: return DotI8I8Sse(row, q8, n);
#elif defined(SUPERFAISS_SIMD_NEON)
	case SimdPath::NEON: return DotI8I8Neon(row, q8, n);
#endif
	default: return DotI8I8Scalar(row, q8, n);
	}
}

void L2SumsI8I8(
	const int8_t* row, const int8_t* q8, int32_t n, int32_t* outCross, int32_t* outRowSq)
{
	switch (ActiveXdPath())
	{
#if defined(SUPERFAISS_SIMD_SSE)
	case SimdPath::AVX2: L2SumsI8I8Avx2(row, q8, n, outCross, outRowSq); return;
	case SimdPath::SSE: L2SumsI8I8Sse(row, q8, n, outCross, outRowSq); return;
#elif defined(SUPERFAISS_SIMD_NEON)
	case SimdPath::NEON: L2SumsI8I8Neon(row, q8, n, outCross, outRowSq); return;
#endif
	default: L2SumsI8I8Scalar(row, q8, n, outCross, outRowSq); return;
	}
}

} // namespace detail

void QuantizeQueryXd(
	const float* paddedQuery,
	int32_t paddedDims,
	int8_t* outQ8,
	double* outScale,
	int64_t* outSqSum)
{
	// Max |q_i| over bit-decoded doubles: exact, DAZ-immune, order-free.
	double maxAbs = 0.0;
	for (int32_t i = 0; i < paddedDims; ++i)
	{
		double a = detail::FloatBitsToDouble(paddedQuery[i]);
		a = a < 0.0 ? -a : a;
		if (a > maxAbs)
		{
			maxAbs = a;
		}
	}
	if (maxAbs == 0.0)
	{
		std::memset(outQ8, 0, static_cast<size_t>(paddedDims));
		*outScale = 0.0;
		*outSqSum = 0;
		return;
	}
	// One correctly rounded divide; maxAbs in [2^-149, ~2^128] keeps the quotient
	// (and every per-element product below) a normal double - FTZ can never fire.
	const double inv = 127.0 / maxAbs;
	int64_t sq = 0;
	for (int32_t i = 0; i < paddedDims; ++i)
	{
		int32_t qi = detail::RoundHalfEvenI32(detail::FloatBitsToDouble(paddedQuery[i]) * inv);
		// |q_i| <= maxAbs makes the exact product <= 127; the clamp guards the
		// rounded edge, nothing more.
		qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
		outQ8[i] = static_cast<int8_t>(qi);
		sq += static_cast<int64_t>(qi) * qi;
	}
	*outScale = maxAbs / 127.0;
	*outSqSum = sq;
}

namespace
{
	// The subnormal-floor contract (19.2 S1): any final score with magnitude below
	// FLT_MIN is exactly 0.0f, by specification, on every machine. At or above
	// FLT_MIN the single double->float conversion produces a normal float, which
	// per-thread FTZ state cannot touch.
	inline float XdFloorToFloat(double score)
	{
		const double lim = 1.1754943508222875e-38; // FLT_MIN, exactly
		if (score < lim && score > -lim)
		{
			return 0.0f;
		}
		return static_cast<float>(score);
	}

	// Dot-family per-row epilogue: one fixed-order double expression.
	inline double XdDotScoreD(int32_t acc, double rowScaleD, double queryScaleD)
	{
		return static_cast<double>(acc) * (rowScaleD * queryScaleD);
	}

	// L2 per-row epilogue: ||qs*q8 - rs*r8||^2 in expanded form -
	// qs^2*Sum(q^2) + rs^2*Sum(r^2) - 2*qs*rs*Sum(q*r) - because query and row
	// carry different scales, so raw int8 differences are not meaningful; the three
	// sums are exact integers and the expression is fixed-order double. Note the
	// expansion can round to a tiny NEGATIVE value where the true distance is ~0;
	// the result is still bit-identical everywhere (and the floor maps
	// (-FLT_MIN, FLT_MIN) to exactly 0.0f).
	inline double XdL2ScoreD(
		int64_t cross, int64_t rowSq, int64_t querySq, double rowScaleD, double queryScaleD)
	{
		const double a = (queryScaleD * queryScaleD) * static_cast<double>(querySq);
		const double b = (rowScaleD * rowScaleD) * static_cast<double>(rowSq);
		const double c = ((rowScaleD * queryScaleD) * static_cast<double>(cross)) * 2.0;
		return (a + b) - c;
	}

	// v2.2 bias composition: defined on the FLOORED unbiased float score -
	// XdFloor((double)unbiased + (double)bias) - so the sparse k+P path (which only
	// holds the converted candidate floats) composes bit-identically to the dense
	// in-scan form. Reuses the v2.1 fused finite-only law (integer exponent test).
	inline float XdComposeBias(float unbiased, const float* rowBias, int32_t r, bool* flag)
	{
		if (rowBias == nullptr)
		{
			return unbiased;
		}
		const float b = rowBias[r];
		uint32_t bits;
		std::memcpy(&bits, &b, sizeof(bits));
		if ((bits & 0x7f800000u) == 0x7f800000u && flag != nullptr)
		{
			*flag = true;
		}
		return XdFloorToFloat(
			static_cast<double>(unbiased) + detail::FloatBitsToDouble(b));
	}

	// Whole-row CrossDevice score (unbiased): integer sums + the double epilogue.
	inline float XdWholeRowScore(
		const BankView& bank, const XdQuery& query, bool isL2, int32_t r)
	{
		const int8_t* row = static_cast<const int8_t*>(bank.rows) +
			static_cast<int64_t>(r) * bank.paddedDims;
		const double rowScaleD = detail::FloatBitsToDouble(bank.scales[r]);
		if (isL2)
		{
			int32_t cross = 0;
			int32_t rowSq = 0;
			detail::L2SumsI8I8(row, query.q8, bank.paddedDims, &cross, &rowSq);
			return XdFloorToFloat(
				XdL2ScoreD(cross, rowSq, query.sqSum, rowScaleD, query.scale));
		}
		const int32_t acc = detail::DotI8I8(row, query.q8, bank.paddedDims);
		return XdFloorToFloat(XdDotScoreD(acc, rowScaleD, query.scale));
	}

	// Segmented CrossDevice row score over a prebuilt range list. Per-range integer
	// partials feed the canonical fixed-order double combine:
	//   partial = epilogue(acc, rowScale, queryScale)   [dot or expanded L2]
	//   partial *= channelInvNorm (channel-matched cosine ranges)
	//   contribution = weight * partial
	//   total += contribution                            [range order]
	// outContributions (when non-null) receives each contribution floored to float
	// (so reported decompositions are themselves cross-device bit-identical); the
	// returned double is the unfloored total (the caller floors the final score).
	// rangeQuerySq carries the per-range Sum(q8^2) integers (L2 only; computed once
	// per scan, not per row).
	inline double XdSegmentedRowScoreD(
		const XdQuery& query,
		const FScanRange* ranges,
		int32_t rangeCount,
		const int64_t* rangeQuerySq,
		bool isL2,
		const int8_t* row,
		double rowScaleD,
		const float* rowInvNorms,
		float* outContributions)
	{
		double total = 0.0;
		for (int32_t i = 0; i < rangeCount; ++i)
		{
			const FScanRange& range = ranges[i];
			if (range.accIndex < 0)
			{
				// CrossDevice gaps are skipped, not scored-and-discarded: integer
				// sums carry no prefetch-stride obligation, and untouched bytes
				// cannot perturb a determinism proof.
				continue;
			}
			double partial;
			if (isL2)
			{
				int32_t cross = 0;
				int32_t rowSq = 0;
				detail::L2SumsI8I8(row + range.offset, query.q8 + range.offset,
					range.length, &cross, &rowSq);
				int64_t querySq;
				if (rangeQuerySq != nullptr)
				{
					querySq = rangeQuerySq[i];
				}
				else
				{
					// No precomputed table (the fused path with unbounded member
					// count): the range self-sum is exact integer either way.
					querySq = 0;
					const int8_t* q = query.q8 + range.offset;
					for (int32_t j = 0; j < range.length; ++j)
					{
						querySq += static_cast<int64_t>(q[j]) * q[j];
					}
				}
				partial = XdL2ScoreD(cross, rowSq, querySq, rowScaleD, query.scale);
			}
			else
			{
				const int32_t acc = detail::DotI8I8(row + range.offset,
					query.q8 + range.offset, range.length);
				partial = XdDotScoreD(acc, rowScaleD, query.scale);
			}
			if (rowInvNorms != nullptr && range.channelIndex >= 0)
			{
				partial *= detail::FloatBitsToDouble(rowInvNorms[range.channelIndex]);
			}
			const double contribution =
				detail::FloatBitsToDouble(ranges[i].weight) * partial;
			if (outContributions != nullptr)
			{
				outContributions[range.accIndex] = XdFloorToFloat(contribution);
			}
			total += contribution;
		}
		return total;
	}

	// Per-range query self-sums for segmented L2 (integer, once per scan).
	inline void BuildRangeQuerySq(
		const XdQuery& query, const FScanRange* ranges, int32_t rangeCount,
		int64_t* outRangeSq)
	{
		for (int32_t i = 0; i < rangeCount; ++i)
		{
			int64_t sq = 0;
			if (ranges[i].accIndex >= 0)
			{
				const int8_t* q = query.q8 + ranges[i].offset;
				for (int32_t j = 0; j < ranges[i].length; ++j)
				{
					sq += static_cast<int64_t>(q[j]) * q[j];
				}
			}
			outRangeSq[i] = sq;
		}
	}
}

void ScoreChunkXd(
	const BankView& bank,
	const XdQuery& query,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout,
	const float* rowBias,
	bool* outNonFiniteBias)
{
	const int32_t chunkRows = ChunkRows(bank);
	const int32_t begin = chunkIndex * chunkRows;
	int32_t end = begin + chunkRows;
	if (end > bank.count)
	{
		end = bank.count;
	}
	const bool isL2 = bank.metric == Metric::L2;
	for (int32_t r = begin; r < end; ++r)
	{
		if (IsExcluded(excludeBits, r))
		{
			continue;
		}
		const float score = XdWholeRowScore(bank, query, isL2, r);
		inout.Push(r, XdComposeBias(score, rowBias, r, outNonFiniteBias));
	}
}

void ScoreChunkSegmentedXd(
	const BankView& bank,
	const XdQuery& query,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout,
	const float* rowBias,
	bool* outNonFiniteBias)
{
	const int32_t chunkRows = ChunkRows(bank);
	const int32_t begin = chunkIndex * chunkRows;
	int32_t end = begin + chunkRows;
	if (end > bank.count)
	{
		end = bank.count;
	}
	const bool isL2 = bank.metric == Metric::L2;
	const bool perChannelCosine =
		bank.metric == Metric::Cosine && bank.channelInvNorms != nullptr;
	FScanRange ranges[2 * kMaxSegments + 1];
	const int32_t rangeCount = BuildScanRanges(bank, segments, segmentCount, ranges);
	int64_t rangeQuerySq[2 * kMaxSegments + 1];
	if (isL2)
	{
		BuildRangeQuerySq(query, ranges, rangeCount, rangeQuerySq);
	}

	for (int32_t r = begin; r < end; ++r)
	{
		if (IsExcluded(excludeBits, r))
		{
			continue;
		}
		const int8_t* row = static_cast<const int8_t*>(bank.rows) +
			static_cast<int64_t>(r) * bank.paddedDims;
		const double rowScaleD = detail::FloatBitsToDouble(bank.scales[r]);
		const float* rowInvNorms = perChannelCosine
			? bank.channelInvNorms + static_cast<int64_t>(r) * bank.channelCount
			: nullptr;
		const float score = XdFloorToFloat(XdSegmentedRowScoreD(query, ranges,
			rangeCount, rangeQuerySq, isL2, row, rowScaleD, rowInvNorms, nullptr));
		inout.Push(r, XdComposeBias(score, rowBias, r, outNonFiniteBias));
	}
}

void ScoreChunkFusedXd(
	const BankView& bank,
	const XdQuery* queries,
	int32_t queryCount,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout,
	const float* rowBias,
	bool* outNonFiniteBias)
{
	const int32_t chunkRows = ChunkRows(bank);
	const int32_t begin = chunkIndex * chunkRows;
	int32_t end = begin + chunkRows;
	if (end > bank.count)
	{
		end = bank.count;
	}
	const bool isL2 = bank.metric == Metric::L2;
	const bool perChannelCosine = segments != nullptr &&
		bank.metric == Metric::Cosine && bank.channelInvNorms != nullptr;
	FScanRange ranges[2 * kMaxSegments + 1];
	int32_t rangeCount = 0;
	if (segments != nullptr)
	{
		rangeCount = BuildScanRanges(bank, segments, segmentCount, ranges);
	}

	// Per-member per-range query self-sums, built ONCE per chunk call instead of
	// once per row (the sums are row-invariant integers; recomputing
	// them per row was O(rows x members x dims) of redundant work on the L2
	// segmented-intersect path). Kernels stay allocation-free: stack tables up to
	// kFusedSqTableMembers members, per-row recompute (bit-identical - the sums
	// are exact integers either way) beyond that.
	constexpr int32_t kFusedSqTableMembers = 16;
	int64_t memberQuerySq[kFusedSqTableMembers][2 * kMaxSegments + 1];
	const int32_t tabled = (segments != nullptr && isL2)
		? (queryCount < kFusedSqTableMembers ? queryCount : kFusedSqTableMembers)
		: 0;
	for (int32_t m = 0; m < tabled; ++m)
	{
		BuildRangeQuerySq(queries[m], ranges, rangeCount, memberQuerySq[m]);
	}

	for (int32_t r = begin; r < end; ++r)
	{
		if (IsExcluded(excludeBits, r))
		{
			continue;
		}
		float fused = 0.0f;
		if (segments == nullptr)
		{
			for (int32_t m = 0; m < queryCount; ++m)
			{
				const float score = XdWholeRowScore(bank, queries[m], isL2, r);
				if (m == 0 || (isL2 ? score > fused : score < fused))
				{
					fused = score;
				}
			}
		}
		else
		{
			const int8_t* row = static_cast<const int8_t*>(bank.rows) +
				static_cast<int64_t>(r) * bank.paddedDims;
			const double rowScaleD = detail::FloatBitsToDouble(bank.scales[r]);
			const float* rowInvNorms = perChannelCosine
				? bank.channelInvNorms + static_cast<int64_t>(r) * bank.channelCount
				: nullptr;
			for (int32_t m = 0; m < queryCount; ++m)
			{
				const float score = XdFloorToFloat(XdSegmentedRowScoreD(
					queries[m], ranges, rangeCount,
					m < tabled ? memberQuerySq[m] : nullptr, isL2, row,
					rowScaleD, rowInvNorms, nullptr));
				if (m == 0 || (isL2 ? score > fused : score < fused))
				{
					fused = score;
				}
			}
		}
		inout.Push(r, XdComposeBias(fused, rowBias, r, outNonFiniteBias));
	}
}

float DecomposeRowScoreXd(
	const BankView& bank,
	const XdQuery& query,
	int32_t rowIndex,
	const QuerySegment* segments,
	int32_t segmentCount,
	float* outContributions)
{
	const bool isL2 = bank.metric == Metric::L2;
	const bool perChannelCosine =
		bank.metric == Metric::Cosine && bank.channelInvNorms != nullptr;
	FScanRange ranges[2 * kMaxSegments + 1];
	const int32_t rangeCount = BuildScanRanges(bank, segments, segmentCount, ranges);
	int64_t rangeQuerySq[2 * kMaxSegments + 1];
	if (isL2)
	{
		BuildRangeQuerySq(query, ranges, rangeCount, rangeQuerySq);
	}
	const int8_t* row = static_cast<const int8_t*>(bank.rows) +
		static_cast<int64_t>(rowIndex) * bank.paddedDims;
	const double rowScaleD = detail::FloatBitsToDouble(bank.scales[rowIndex]);
	const float* rowInvNorms = perChannelCosine
		? bank.channelInvNorms + static_cast<int64_t>(rowIndex) * bank.channelCount
		: nullptr;
	for (int32_t i = 0; i < segmentCount; ++i)
	{
		outContributions[i] = 0.0f; // weight-0 segments contribute exactly 0
	}
	return XdFloorToFloat(XdSegmentedRowScoreD(query, ranges, rangeCount,
		rangeQuerySq, isL2, row, rowScaleD, rowInvNorms, outContributions));
}

float ScoreRowXd(const BankView& bank, const XdQuery& query, int32_t rowIndex)
{
	return XdWholeRowScore(bank, query, bank.metric == Metric::L2, rowIndex);
}

namespace detail
{
	float XdComposeBiasValue(float unbiasedScore, float bias)
	{
		return XdFloorToFloat(
			static_cast<double>(unbiasedScore) + FloatBitsToDouble(bias));
	}
}

} // namespace superfaiss
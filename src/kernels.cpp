#include "superfaiss/kernels.h"

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

	// Lazy magic-static: immune to cross-TU static-init ordering (Poirot O2).
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

void ScoreChunkPair(
	const BankView& bank,
	const float* paddedQueryA,
	const float* paddedQueryB,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inoutA,
	TopK& inoutB)
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
				inoutA.Push(r, scoreA);
				inoutB.Push(r, scoreB);
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
				inoutA.Push(r, scoreA);
				inoutB.Push(r, scoreB);
			}
		}
		return;
	}
#endif
	// Fallback: two single passes (bit-identical by definition; SSE/NEON pair kernels
	// are a recorded follow-up).
	ScoreChunk(bank, paddedQueryA, chunkIndex, excludeBits, inoutA);
	ScoreChunk(bank, paddedQueryB, chunkIndex, excludeBits, inoutB);
}

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

void ScoreChunkFused(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
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
			inout.Push(r, fused);
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
			inout.Push(r, fused);
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
	// applied at combine) or a gap (accIndex < 0, partial discarded).
	struct FScanRange
	{
		int32_t offset;
		int32_t length;
		float weight;
		int32_t accIndex; // -1 = discard
	};

	// Builds the full-coverage range list: segments in order, gaps filled, trailing
	// gap to paddedDims. Returns the range count (<= 2*kMaxSegments + 1).
	inline int32_t BuildScanRanges(
		const QuerySegment* segments,
		int32_t segmentCount,
		int32_t paddedDims,
		FScanRange* outRanges)
	{
		int32_t n = 0;
		int32_t cursor = 0;
		for (int32_t i = 0; i < segmentCount; ++i)
		{
			const QuerySegment& seg = segments[i];
			if (seg.offset > cursor)
			{
				outRanges[n++] = {cursor, seg.offset - cursor, 0.0f, -1};
			}
			// Weight-0 segments keep omission semantics: scanned for stride
			// continuity, discarded like a gap.
			outRanges[n++] = {seg.offset, seg.length, seg.weight,
				seg.weight == 0.0f ? -1 : i};
			cursor = seg.offset + seg.length;
		}
		if (cursor < paddedDims)
		{
			outRanges[n++] = {cursor, paddedDims - cursor, 0.0f, -1};
		}
		return n;
	}

	// Scores one row over the range list. outPartials (kMaxSegments floats) receives
	// unweighted per-segment partials when non-null (the decomposition surface);
	// the return value is the weighted combine.
	inline float DenseSegmentedRowScore(
		const void* row,
		float scale,
		const float* paddedQuery,
		const FScanRange* ranges,
		int32_t rangeCount,
		const FRowKernels& kernels,
		bool isL2,
		bool isInt8,
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
				if (outPartials != nullptr)
				{
					outPartials[range.accIndex] = partial;
				}
				total += range.weight * partial;
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
	TopK& inout)
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
	const FRowKernels kernels = ResolveRowKernels();
	FScanRange ranges[2 * kMaxSegments + 1];
	const int32_t rangeCount =
		BuildScanRanges(segments, segmentCount, bank.paddedDims, ranges);
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
		inout.Push(r, DenseSegmentedRowScore(row, scale, paddedQuery, ranges,
			rangeCount, kernels, isL2, isInt8, nullptr));
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
	const bool isInt8 = bank.quant == Quantization::Int8;
	const FRowKernels kernels = ResolveRowKernels();
	FScanRange ranges[2 * kMaxSegments + 1];
	const int32_t rangeCount = BuildScanRanges(segments, segmentCount, pd, ranges);
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
		float fused = 0.0f;
		for (int32_t m = 0; m < queryCount; ++m)
		{
			const float score = DenseSegmentedRowScore(row, scale,
				paddedQueries + static_cast<int64_t>(m) * pd, ranges, rangeCount,
				kernels, isL2, isInt8, nullptr);
			if (m == 0 || (isL2 ? score > fused : score < fused))
			{
				fused = score;
			}
		}
		inout.Push(r, fused);
	}
}

} // namespace superfaiss
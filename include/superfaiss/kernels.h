#pragma once

#include "types.h"
#include "topk.h"

namespace superfaiss
{

// Scores every non-excluded row of one chunk into `inout`. Single-threaded;
// an external scheduler may call this for different chunks concurrently, each with its
// own TopK, then MergeTopK the results.
//
// `paddedQuery` must have bank.paddedDims elements, zero-filled pad lanes, and
// kAlignment-byte alignment. Callers are expected to have validated the bank and query
// (see validate.h); kernels do not re-validate.
void ScoreChunk(
	const BankView& bank,
	const float* paddedQuery,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout);

// Scores one chunk against TWO queries in a single row pass: row loads (and int8
// widening) are shared while each query keeps the exact single-query accumulation
// structure, so pair results are bit-identical to two ScoreChunk calls. This is the
// batch amortization primitive (plan E2).
void ScoreChunkPair(
	const BankView& bank,
	const float* paddedQueryA,
	const float* paddedQueryB,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inoutA,
	TopK& inoutB);

// Scores every non-excluded row of one chunk against M queries and pushes the FUSED
// score — the worst per-query score in the metric's better-direction (min for
// Dot/Cosine, max for L2) — into `inout`. The intersection primitive (plan 18.7):
// per-query scores come from the same per-row kernels as ScoreChunk, so fused scores
// are bit-identical to the corresponding single-query scores; worst-of selection is
// rounding-free. Queries are contiguous, stride bank.paddedDims.
void ScoreChunkFused(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	TopK& inout);

// Kernel path selected at compile time (NEON/SSE/scalar), plus a runtime AVX2+FMA
// upgrade on x86 hardware that supports it. Dispatch is per-device stable, so the
// per-device determinism promise is unaffected. Exposed for tests and diagnostics.
enum class SimdPath : uint8_t
{
	Scalar = 0,
	SSE = 1,
	NEON = 2,
	AVX2 = 3,
};
SimdPath ActiveSimdPath();

namespace detail
{
	// Scalar kernels mirror the active SIMD path's accumulation exactly, so scalar and
	// SIMD results are bit-identical on a device:
	//   - SSE/NEON mirror: four 4-lane stripes, unfused multiply-then-add.
	//   - AVX2 mirror: four 8-lane stripes accumulated with std::fma (hardware FMA and
	//     std::fma round identically).
	float DotF32Scalar(const float* row, const float* query, int32_t paddedDims);
	float L2F32Scalar(const float* row, const float* query, int32_t paddedDims);
	float DotI8Scalar(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8Scalar(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	float DotF32ScalarAvx2(const float* row, const float* query, int32_t paddedDims);
	float L2F32ScalarAvx2(const float* row, const float* query, int32_t paddedDims);
	float DotI8ScalarAvx2(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8ScalarAvx2(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	float DotF32(const float* row, const float* query, int32_t paddedDims);
	float L2F32(const float* row, const float* query, int32_t paddedDims);
	float DotI8(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	// The scalar mirror of whatever path ActiveSimdPath() reports — what the
	// bit-equality tests compare against.
	float DotF32Mirror(const float* row, const float* query, int32_t paddedDims);
	float L2F32Mirror(const float* row, const float* query, int32_t paddedDims);
	float DotI8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8Mirror(const int8_t* row, float scale, const float* query, int32_t paddedDims);
}

} // namespace superfaiss

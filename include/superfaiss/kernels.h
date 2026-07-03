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

// Kernel selected at compile time. Exposed for tests and diagnostics.
enum class SimdPath : uint8_t
{
	Scalar = 0,
	SSE = 1,
	NEON = 2,
};
SimdPath ActiveSimdPath();

namespace detail
{
	// Scalar kernels mirror the SIMD paths' striped accumulation exactly (four partial
	// sums combined as (s0+s1)+(s2+s3)), so scalar and SIMD results are bit-identical.
	float DotF32Scalar(const float* row, const float* query, int32_t paddedDims);
	float L2F32Scalar(const float* row, const float* query, int32_t paddedDims);
	float DotI8Scalar(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8Scalar(const int8_t* row, float scale, const float* query, int32_t paddedDims);

	float DotF32(const float* row, const float* query, int32_t paddedDims);
	float L2F32(const float* row, const float* query, int32_t paddedDims);
	float DotI8(const int8_t* row, float scale, const float* query, int32_t paddedDims);
	float L2I8(const int8_t* row, float scale, const float* query, int32_t paddedDims);
}

} // namespace superfaiss

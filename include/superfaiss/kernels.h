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

// Segmented scan, dense form (V2 §10 decision). NOTE the shipped split: dot-family
// segmented scans FOLD segment weights into the query (see query.cpp) and run the
// plain ScoreChunk at V1 speed — this dense primitive is the L2 path (weights sit
// inside the square and cannot fold), and the substrate for slot-2 decomposition.
// It scores each non-excluded row of one chunk over a segment
// list — per-segment partials from the SAME per-row kernels ScoreChunk uses, combined
// as sum(weight_s * partial_s) in segment order. The degenerate one-segment list
// (0, paddedDims, 1.0) is bit-identical to ScoreChunk by construction (same kernel
// entry points; x*1.0f is bitwise identity). Weight-0 segments and omitted ranges are
// never read — masking is a bandwidth cut. Callers validate segments first
// (ValidateSegments); kernels do not re-validate.
void ScoreChunkSegmented(
	const BankView& bank,
	const float* paddedQuery,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
	TopK& inout);

// Segmented intersection: the fused worst-of law over segmented totals — each member
// query scores through the same segmented per-row path, then worst-of in the metric's
// better-direction. One segment list applies to every member query.
void ScoreChunkFusedSegmented(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	int32_t chunkIndex,
	const uint32_t* excludeBits,
	const QuerySegment* segments,
	int32_t segmentCount,
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

#pragma once

#include <cstdint>

namespace superfaiss
{

enum class Quantization : uint8_t
{
	Float32 = 0,
	Int8 = 1,
};

enum class Metric : uint8_t
{
	Dot = 0,
	// Cosine banks store pre-normalized rows, so query-time scoring is a plain dot product.
	// The enum survives into the runtime so validation can enforce cosine-specific rules
	// (zero-norm queries are invalid against a Cosine bank).
	Cosine = 1,
	L2 = 2,
};

enum class Status : uint8_t
{
	Ok = 0,
	InvalidArgument,
	DimsMismatch,
	NonFiniteQuery,
	ZeroNormQuery,
	NonZeroPadding,
	BadAlignment,
	BadFormat,
	ZeroNormRow,
	OutOfMemory,
};

// Required alignment of bank row data and query buffers, in bytes.
inline constexpr int32_t kAlignment = 16;

// Rows are chunked for cache locality and so an external scheduler can parallelize the
// scan at chunk granularity. ~64 KB of row data per chunk.
inline constexpr int32_t kChunkBytes = 64 * 1024;

// A non-owning view of a baked bank. The library never allocates or frees bank memory.
struct BankView
{
	const void* rows = nullptr;    // row-major, paddedDims stride, kAlignment-aligned
	const float* scales = nullptr; // Int8 only: per-row dequant scale; null for Float32
	int32_t count = 0;
	int32_t dims = 0;
	int32_t paddedDims = 0;        // stride in elements; pad lanes are zero
	Quantization quant = Quantization::Float32;
	Metric metric = Metric::Dot;
};

struct Hit
{
	int32_t index = -1;
	float score = 0.0f;
};

// Per-query scoring-metric override. BankMetric scores with the bank's own metric.
// Dot scores with the dot kernel regardless of bank family: on Dot and Cosine banks it
// is the identity (their scoring is already a dot product); on L2 banks it enables
// direction/axis-projection queries without re-baking. Query validation always applies
// the BANK's rules (a Cosine bank rejects zero-norm queries under any override).
enum class ScoreAs : uint8_t
{
	BankMetric = 0,
	Dot = 1,
};

struct QueryParams
{
	int32_t k = 0;
	// Optional exclusion bitset, ceil(count/32) uint32 words, bit set = skip that row index.
	// Null means no exclusions.
	const uint32_t* excludeBits = nullptr;
	ScoreAs scoreAs = ScoreAs::BankMetric;
};

// The metric a query with these params is actually scored (and its hits ordered) by.
inline constexpr Metric ScoringMetric(const BankView& bank, const QueryParams& params)
{
	return params.scoreAs == ScoreAs::Dot ? Metric::Dot : bank.metric;
}

// Element size of one stored value for a quantization mode.
inline constexpr int32_t ElementSize(Quantization q)
{
	return q == Quantization::Float32 ? 4 : 1;
}

// Row stride (in elements) that pads a logical dim count to a kAlignment-byte boundary.
inline constexpr int32_t PaddedDims(int32_t dims, Quantization q)
{
	const int32_t elemsPerAlign = kAlignment / ElementSize(q);
	return ((dims + elemsPerAlign - 1) / elemsPerAlign) * elemsPerAlign;
}

inline constexpr int64_t RowBytes(const BankView& bank)
{
	return static_cast<int64_t>(bank.paddedDims) * ElementSize(bank.quant);
}

inline constexpr int64_t BankBytes(const BankView& bank)
{
	return static_cast<int64_t>(bank.count) * RowBytes(bank);
}

// Number of rows per scan chunk for this bank (>= 1).
inline constexpr int32_t ChunkRows(const BankView& bank)
{
	const int64_t rowBytes = RowBytes(bank);
	if (rowBytes <= 0)
	{
		return 1;
	}
	const int64_t rows = kChunkBytes / rowBytes;
	return rows < 1 ? 1 : static_cast<int32_t>(rows);
}

inline constexpr int32_t ChunkCount(const BankView& bank)
{
	const int32_t rows = ChunkRows(bank);
	return bank.count <= 0 ? 0 : (bank.count + rows - 1) / rows;
}

inline bool IsExcluded(const uint32_t* excludeBits, int32_t index)
{
	return excludeBits != nullptr && (excludeBits[index >> 5] & (1u << (index & 31))) != 0;
}

} // namespace superfaiss

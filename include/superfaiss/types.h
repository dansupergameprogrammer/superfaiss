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

// A named channel's range (names live host-side; the core is name-free): a
// contiguous element run on the 16-byte element grid. Channels are ascending and
// non-overlapping; kMaxChannels matches kMaxSegments (channels are the bank-side
// twin of query segments).
struct ChannelInfo
{
	int32_t offset = 0;
	int32_t length = 0;
};

inline constexpr int32_t kMaxChannels = 8;

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
	// Optional channel table (schemaVersion 2 banks). For Cosine banks with channels,
	// channelInvNorms holds count x channelCount per-row inverse sub-norms baked from
	// the QUANTIZED rows (the reported per-channel cosine is the cosine of what the
	// kernel actually dots); a zero-norm row channel stores 0 and scores 0 for that
	// segment - defined, never NaN.
	const ChannelInfo* channels = nullptr;
	int32_t channelCount = 0;
	const float* channelInvNorms = nullptr;
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

// One segment of a segmented query (the V2 unifying design): a contiguous element
// range of the row with a scalar weight. Offsets and lengths lie on the 16-byte
// element grid for the bank's quantization; segments are ascending, non-overlapping.
// A mask is a range simply omitted from the list (those bytes are never read); a
// weight-0 segment is equivalent to omission. Scores combine additively:
// total = sum(weight_s * partial_s), valid for dot and squared-L2 alike.
struct QuerySegment
{
	int32_t offset = 0; // elements
	int32_t length = 0; // elements
	float weight = 1.0f;
};

// Segment-count cap per query: covers the named use cases, bounds accumulator and
// result-buffer sizing, keeps per-row loop overhead predictable. Raising is additive.
inline constexpr int32_t kMaxSegments = 8;

// One sparse bias entry (v2.1): a row index and the bias added to that row's score.
struct BiasPair
{
	int32_t index = 0;
	float bias = 0.0f;
};

// Per-query row bias (v2.1, plan section 18): an optional caller-provided score bias
// applied in-scan, after dequantized scoring, before top-k selection - the composed
// score ranks exactly or the top-k is not the true top-k. Exactly one form per query:
//   dense - count-length float view (the memory-salience shape); validated FUSED into
//           the scan (a non-finite value anywhere returns NonFiniteQuery at completion;
//           a pre-pass would re-read count x 4 bytes for nothing).
//   pairs - (index, bias) entries (the motion-matching shape: one biased row per
//           query); O(pairCount) validation at query build - indices unique and in
//           range, values finite.
// Both forms set is InvalidArgument. Both empty (all null/0) is the unbiased path -
// no add executed, bit-identical to no bias at all. All-zeros bias is compare-equal,
// ranking-identical, NOT claimed bitwise (IEEE -0.0 + 0.0 == +0.0). Bias adds in the
// scored metric's own direction: a reward is positive on Dot/Cosine, NEGATIVE on L2
// (lower is better). Non-finite bias is illegal input - exclusion is a mask, bias is
// arithmetic, orthogonal; -inf is not a mask.
struct RowBias
{
	const float* dense = nullptr; // bank.count floats
	const BiasPair* pairs = nullptr;
	int32_t pairCount = 0;
};

struct QueryParams
{
	int32_t k = 0;
	// Optional exclusion bitset, ceil(count/32) uint32 words, bit set = skip that row index.
	// Null means no exclusions.
	const uint32_t* excludeBits = nullptr;
	ScoreAs scoreAs = ScoreAs::BankMetric;
	// Optional segmented query: null means the whole row at weight 1 (the V1 path,
	// bit-identical). The degenerate one-segment list (0, paddedDims, 1.0) is also
	// bit-identical to the V1 path by construction.
	const QuerySegment* segments = nullptr;
	int32_t segmentCount = 0;
	// Optional per-row bias (v2.1): Query/QueryIntersect read ONE RowBias (for
	// QueryIntersect it applies once, to the fused score, in the fused metric's
	// direction); QueryBatch reads queryCount entries, one per query (a shared view
	// is the degenerate case - point every entry's dense at the same array). Null
	// means no bias anywhere.
	const RowBias* bias = nullptr;
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

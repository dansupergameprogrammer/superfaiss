// SuperFAISS test harness. Standard library only; no third-party test framework.
// Reference results are computed in double precision with the same total order
// (score, then ascending index); float-vs-double near-ties are handled with an
// epsilon boundary check rather than exact rank equality.

#include "superfaiss/superfaiss.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#include <algorithm>

using namespace superfaiss;

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond) \
	do \
	{ \
		++GChecks; \
		if (!(cond)) \
		{ \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do \
	{ \
		++GChecks; \
		if (!(cond)) \
		{ \
			++GFailures; \
			std::printf("FAIL %s:%d: %s — ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

// ---------------------------------------------------------------------------
// Utilities

struct Rng
{
	uint64_t state;
	explicit Rng(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ull) {}

	uint64_t Next()
	{
		state ^= state >> 12;
		state ^= state << 25;
		state ^= state >> 27;
		return state * 0x2545F4914F6CDD1Dull;
	}

	// Uniform in [-1, 1].
	float NextFloat()
	{
		return static_cast<float>(static_cast<int64_t>(Next() >> 24)) /
			static_cast<float>(1ll << 39);
	}

	int32_t NextIndex(int32_t bound)
	{
		return static_cast<int32_t>(Next() % static_cast<uint64_t>(bound));
	}
};

struct AlignedBuf
{
	Allocator alloc = DefaultAllocator();
	void* ptr = nullptr;

	explicit AlignedBuf(size_t bytes)
	{
		ptr = alloc.alloc(bytes ? bytes : 1, kAlignment, alloc.user);
		std::memset(ptr, 0, bytes ? bytes : 1);
	}
	~AlignedBuf() { alloc.free(ptr, alloc.user); }
	AlignedBuf(const AlignedBuf&) = delete;
	AlignedBuf& operator=(const AlignedBuf&) = delete;

	float* F32() { return static_cast<float*>(ptr); }
	int8_t* I8() { return static_cast<int8_t*>(ptr); }
};

// A fully baked in-memory bank plus the double-precision row values the reference
// scorer uses (for Int8 banks these are the *dequantized* values, so the reference
// scores the same numbers the kernel does).
struct TestBank
{
	std::vector<float> source;      // count x dims, unpadded, post-normalize for Cosine
	std::vector<double> refRows;    // count x dims, what the kernel effectively scores
	AlignedBuf payload;
	std::vector<float> scales;
	BankView view;

	TestBank(Rng& rng, int32_t count, int32_t dims, Quantization quant, Metric metric)
		: payload(static_cast<size_t>(count > 0 ? count : 1) * PaddedDims(dims, quant) * ElementSize(quant))
	{
		source.resize(static_cast<size_t>(count) * dims);
		for (auto& v : source)
		{
			v = rng.NextFloat();
		}
		if (metric == Metric::Cosine && count > 0)
		{
			const Status s = NormalizeRows(source.data(), count, dims, nullptr);
			CHECK(s == Status::Ok);
		}

		const int32_t pd = PaddedDims(dims, quant);
		refRows.resize(static_cast<size_t>(count) * dims);

		if (quant == Quantization::Float32)
		{
			PadRowsFloat32(source.data(), count, dims, pd, payload.F32());
			for (size_t i = 0; i < source.size(); ++i)
			{
				refRows[i] = static_cast<double>(source[i]);
			}
		}
		else
		{
			scales.resize(static_cast<size_t>(count));
			QuantizeRowsInt8(source.data(), count, dims, pd, payload.I8(), scales.data());
			for (int32_t r = 0; r < count; ++r)
			{
				for (int32_t i = 0; i < dims; ++i)
				{
					refRows[static_cast<size_t>(r) * dims + i] =
						static_cast<double>(scales[r]) *
						payload.I8()[static_cast<int64_t>(r) * pd + i];
				}
			}
		}

		view.rows = payload.ptr;
		view.scales = quant == Quantization::Int8 ? scales.data() : nullptr;
		view.count = count;
		view.dims = dims;
		view.paddedDims = pd;
		view.quant = quant;
		view.metric = metric;
	}
};

struct RefHit
{
	int32_t index;
	double score;
};

static bool RefBetter(const RefHit& a, const RefHit& b, Metric metric)
{
	if (a.score != b.score)
	{
		return metric == Metric::L2 ? (a.score < b.score) : (a.score > b.score);
	}
	return a.index < b.index;
}

// Full-precision reference scan: every non-excluded row scored in double, sorted by
// the same total order the library uses.
static std::vector<RefHit> ReferenceScan(
	const TestBank& bank, const float* query, const uint32_t* excludeBits)
{
	std::vector<RefHit> hits;
	const int32_t dims = bank.view.dims;
	for (int32_t r = 0; r < bank.view.count; ++r)
	{
		if (IsExcluded(excludeBits, r))
		{
			continue;
		}
		const double* row = bank.refRows.data() + static_cast<size_t>(r) * dims;
		double score = 0.0;
		if (bank.view.metric == Metric::L2)
		{
			for (int32_t i = 0; i < dims; ++i)
			{
				const double d = static_cast<double>(query[i]) - row[i];
				score += d * d;
			}
		}
		else
		{
			for (int32_t i = 0; i < dims; ++i)
			{
				score += static_cast<double>(query[i]) * row[i];
			}
		}
		hits.push_back({r, score});
	}
	std::sort(hits.begin(), hits.end(), [&](const RefHit& a, const RefHit& b) {
		return RefBetter(a, b, bank.view.metric);
	});
	return hits;
}

// Checks a returned top-k against the reference: output sorted by the library's total
// order; every returned row within epsilon of true top-k membership; every true top-k
// row either returned or within epsilon of the boundary.
static void CheckTopK(
	const TestBank& bank,
	const std::vector<RefHit>& ref,
	const Hit* hits,
	int32_t hitCount,
	int32_t k)
{
	const Metric metric = bank.view.metric;
	const int32_t expected = static_cast<int32_t>(ref.size()) < k
		? static_cast<int32_t>(ref.size())
		: k;
	CHECK_MSG(hitCount == expected, "hitCount=%d expected=%d", hitCount, expected);
	if (hitCount != expected)
	{
		return;
	}

	for (int32_t i = 1; i < hitCount; ++i)
	{
		CHECK(Better(hits[i - 1], hits[i], metric));
	}

	if (hitCount == 0)
	{
		return;
	}

	const double boundary = ref[static_cast<size_t>(expected) - 1].score;
	const double tol = 1e-4 * (1.0 + std::fabs(boundary));

	std::vector<double> refScoreByIndex(static_cast<size_t>(bank.view.count),
		metric == Metric::L2 ? 1e300 : -1e300);
	for (const RefHit& h : ref)
	{
		refScoreByIndex[static_cast<size_t>(h.index)] = h.score;
	}

	for (int32_t i = 0; i < hitCount; ++i)
	{
		const double rs = refScoreByIndex[static_cast<size_t>(hits[i].index)];
		const bool inTrueTopK = metric == Metric::L2 ? (rs <= boundary + tol) : (rs >= boundary - tol);
		CHECK_MSG(inTrueTopK, "index %d score %.9g vs boundary %.9g", hits[i].index, rs, boundary);
		CHECK_MSG(std::fabs(rs - static_cast<double>(hits[i].score)) <= tol,
			"score drift: got %.9g ref %.9g", static_cast<double>(hits[i].score), rs);
	}
}

static void PadQuery(const std::vector<float>& q, int32_t pd, float* out)
{
	std::memset(out, 0, static_cast<size_t>(pd) * sizeof(float));
	std::memcpy(out, q.data(), q.size() * sizeof(float));
}

// ---------------------------------------------------------------------------
// T1 — known geometry (plan A6): hand-built vectors with hand-computed neighbors.

static void TestKnownGeometry()
{
	// dims=4: rows are unit axes plus one diagonal.
	const float rows[5][4] = {
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
		{0.5f, 0.5f, 0.5f, 0.5f},
	};
	AlignedBuf payload(5 * 4 * sizeof(float));
	PadRowsFloat32(&rows[0][0], 5, 4, 4, payload.F32());

	BankView bank;
	bank.rows = payload.ptr;
	bank.count = 5;
	bank.dims = 4;
	bank.paddedDims = 4;
	bank.quant = Quantization::Float32;
	bank.metric = Metric::Dot;
	CHECK(ValidateBank(bank) == Status::Ok);

	alignas(16) float query[4] = {0.9f, 0.1f, 0.0f, 0.0f};
	Workspace ws;
	Hit hits[3];
	int32_t n = 0;
	QueryParams params;
	params.k = 3;

	CHECK(Query(bank, query, params, ws, hits, &n) == Status::Ok);
	CHECK(n == 3);
	// dot scores: row0=0.9, row4=0.5, row1=0.1, rows 2/3 = 0.
	CHECK(n == 3 && hits[0].index == 0 && hits[1].index == 4 && hits[2].index == 1);

	bank.metric = Metric::L2;
	CHECK(Query(bank, query, params, ws, hits, &n) == Status::Ok);
	// L2 distances^2: row0=0.02, row1=1.62, row4=0.82, row2=row3=0.82+... compute:
	// row2: (0.9)^2+(0.1)^2+1 = 1.82; row4: 0.16+0.16+0.25+0.25=0.82.
	CHECK(n == 3 && hits[0].index == 0 && hits[1].index == 4 && hits[2].index == 1);
}

// ---------------------------------------------------------------------------
// T2 — randomized rank match vs double reference (plan A1 seed).

static void TestRandomizedAgainstReference()
{
	const int32_t dimsSet[] = {6, 8, 64, 256};
	const int32_t countSet[] = {1, 37, 999};
	const Metric metrics[] = {Metric::Dot, Metric::Cosine, Metric::L2};
	const Quantization quants[] = {Quantization::Float32, Quantization::Int8};

	Rng rng(42);
	for (int32_t dims : dimsSet)
	{
		for (int32_t count : countSet)
		{
			for (Metric metric : metrics)
			{
				for (Quantization quant : quants)
				{
					TestBank bank(rng, count, dims, quant, metric);
					CHECK(ValidateBank(bank.view) == Status::Ok);

					std::vector<float> q(static_cast<size_t>(dims));
					for (auto& v : q)
					{
						v = rng.NextFloat();
					}

					AlignedBuf qbuf(static_cast<size_t>(bank.view.paddedDims) * sizeof(float));
					PadQuery(q, bank.view.paddedDims, qbuf.F32());

					const auto ref = ReferenceScan(bank, q.data(), nullptr);

					const int32_t k = count < 10 ? count : 10;
					std::vector<Hit> hits(static_cast<size_t>(k > 0 ? k : 1));
					int32_t n = 0;
					Workspace ws;
					QueryParams params;
					params.k = k;

					CHECK(Query(bank.view, qbuf.F32(), params, ws, hits.data(), &n) == Status::Ok);
					CheckTopK(bank, ref, hits.data(), n, k);
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// T3 — tie-break stability (plan B3): duplicate rows return ascending indices.

static void TestTieBreak()
{
	Rng rng(7);
	const int32_t dims = 8;
	std::vector<float> rows(static_cast<size_t>(6) * dims);
	// Rows 0..3 identical; rows 4..5 distinct and worse.
	for (int32_t i = 0; i < dims; ++i)
	{
		rows[i] = rng.NextFloat();
	}
	for (int32_t r = 1; r < 4; ++r)
	{
		std::memcpy(&rows[static_cast<size_t>(r) * dims], rows.data(), dims * sizeof(float));
	}
	for (int32_t r = 4; r < 6; ++r)
	{
		for (int32_t i = 0; i < dims; ++i)
		{
			rows[static_cast<size_t>(r) * dims + i] = -1.0f;
		}
	}

	AlignedBuf payload(6 * dims * sizeof(float));
	PadRowsFloat32(rows.data(), 6, dims, dims, payload.F32());

	BankView bank;
	bank.rows = payload.ptr;
	bank.count = 6;
	bank.dims = dims;
	bank.paddedDims = dims;
	bank.quant = Quantization::Float32;
	bank.metric = Metric::Dot;

	AlignedBuf qbuf(dims * sizeof(float));
	std::memcpy(qbuf.F32(), rows.data(), dims * sizeof(float));

	Workspace ws;
	Hit hits[4];
	int32_t n = 0;
	QueryParams params;
	params.k = 4;
	CHECK(Query(bank, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	CHECK(n == 4);
	CHECK(hits[0].index == 0 && hits[1].index == 1 && hits[2].index == 2 && hits[3].index == 3);
}

// ---------------------------------------------------------------------------
// T4 — edge cases (plan A2).

static void TestEdges()
{
	Rng rng(11);
	TestBank bank(rng, 5, 8, Quantization::Float32, Metric::Dot);

	AlignedBuf qbuf(bank.view.paddedDims * sizeof(float));
	std::vector<float> q(8, 0.5f);
	PadQuery(q, bank.view.paddedDims, qbuf.F32());

	Workspace ws;
	Hit hits[16];
	int32_t n = -1;
	QueryParams params;

	// k = 0.
	params.k = 0;
	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	CHECK(n == 0);

	// k > count.
	params.k = 16;
	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	CHECK(n == 5);

	// count = 0.
	TestBank empty(rng, 0, 8, Quantization::Float32, Metric::Dot);
	params.k = 3;
	CHECK(Query(empty.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	CHECK(n == 0);

	// count = 1, k = 1.
	TestBank one(rng, 1, 8, Quantization::Float32, Metric::Dot);
	params.k = 1;
	CHECK(Query(one.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	CHECK(n == 1 && hits[0].index == 0);
}

// ---------------------------------------------------------------------------
// T5 — validation rejections (plan B6 seed + bank structural checks).

static void TestValidation()
{
	Rng rng(13);
	TestBank bank(rng, 4, 8, Quantization::Float32, Metric::Cosine);

	AlignedBuf qbuf((bank.view.paddedDims + 4) * sizeof(float));
	std::vector<float> q(8, 0.25f);
	PadQuery(q, bank.view.paddedDims, qbuf.F32());

	Workspace ws;
	Hit hits[4];
	int32_t n = 0;
	QueryParams params;
	params.k = 2;

	// Healthy baseline.
	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);

	// NaN query.
	qbuf.F32()[3] = NAN;
	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::NonFiniteQuery);
	qbuf.F32()[3] = 0.25f;

	// Inf query.
	qbuf.F32()[0] = INFINITY;
	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::NonFiniteQuery);
	qbuf.F32()[0] = 0.25f;

	// Zero-norm query against a Cosine bank.
	for (int32_t i = 0; i < 8; ++i)
	{
		qbuf.F32()[i] = 0.0f;
	}
	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::ZeroNormQuery);

	// Same zero query is legal against a Dot bank.
	BankView dotView = bank.view;
	dotView.metric = Metric::Dot;
	CHECK(Query(dotView, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	PadQuery(q, bank.view.paddedDims, qbuf.F32());

	// Misaligned query pointer (offset by one float).
	CHECK(ValidateQuery(bank.view, qbuf.F32() + 1) == Status::BadAlignment);

	// Misaligned rows.
	BankView bad = bank.view;
	bad.rows = static_cast<const uint8_t*>(bank.view.rows) + 4;
	CHECK(ValidateBank(bad) == Status::BadAlignment);

	// Wrong paddedDims.
	bad = bank.view;
	bad.paddedDims = bank.view.paddedDims + 4;
	CHECK(ValidateBank(bad) == Status::BadFormat);

	// Int8 without scales.
	bad = bank.view;
	bad.quant = Quantization::Int8;
	bad.paddedDims = PaddedDims(bad.dims, bad.quant);
	bad.scales = nullptr;
	CHECK(ValidateBank(bad) == Status::BadFormat);

	// Non-zero padding lane. dims=6 gives two pad lanes in float32.
	TestBank padded(rng, 3, 6, Quantization::Float32, Metric::Dot);
	AlignedBuf q6(padded.view.paddedDims * sizeof(float));
	std::vector<float> qv(6, 0.5f);
	PadQuery(qv, padded.view.paddedDims, q6.F32());
	CHECK(ValidateQuery(padded.view, q6.F32()) == Status::Ok);
	q6.F32()[7] = 1.0f;
	CHECK(ValidateQuery(padded.view, q6.F32()) == Status::NonZeroPadding);
}

// ---------------------------------------------------------------------------
// T6 — bake math: normalization, quantization error bounds, pad-lane hygiene (A3 seed).

static void TestBake()
{
	// Zero-norm row rejected by NormalizeRows.
	std::vector<float> rows(3 * 4, 1.0f);
	for (int32_t i = 0; i < 4; ++i)
	{
		rows[4 + i] = 0.0f;
	}
	int32_t bad = -1;
	CHECK(NormalizeRows(rows.data(), 3, 4, &bad) == Status::ZeroNormRow);
	CHECK(bad == 1);

	// Quantization error bound: |v - scale*q| <= scale/2 (+ float slack).
	Rng rng(17);
	const int32_t dims = 33; // deliberately unaligned
	const int32_t count = 64;
	std::vector<float> src(static_cast<size_t>(count) * dims);
	for (auto& v : src)
	{
		v = rng.NextFloat() * 3.0f;
	}
	const int32_t pd = PaddedDims(dims, Quantization::Int8);
	std::vector<int8_t> qrows(static_cast<size_t>(count) * pd);
	std::vector<float> scales(static_cast<size_t>(count));
	QuantizeRowsInt8(src.data(), count, dims, pd, qrows.data(), scales.data());

	for (int32_t r = 0; r < count; ++r)
	{
		for (int32_t i = 0; i < dims; ++i)
		{
			const float v = src[static_cast<size_t>(r) * dims + i];
			const float dq = scales[r] * qrows[static_cast<size_t>(r) * pd + i];
			CHECK_MSG(std::fabs(v - dq) <= scales[r] * 0.5f + 1e-6f,
				"row %d lane %d: v=%g dq=%g scale=%g", r, i, v, dq, scales[r]);
		}
		for (int32_t i = dims; i < pd; ++i)
		{
			CHECK(qrows[static_cast<size_t>(r) * pd + i] == 0);
		}
	}

	// NonFinite source rejected.
	src[5] = NAN;
	CHECK(ValidateSourceRows(src.data(), count, dims, &bad) == Status::NonFiniteQuery);
	CHECK(bad == 0);
}

// ---------------------------------------------------------------------------
// T7 — exclusion filter (plan A4 seed).

static void TestExclusion()
{
	Rng rng(23);
	const int32_t count = 200;
	TestBank bank(rng, count, 16, Quantization::Float32, Metric::Dot);

	std::vector<float> q(16);
	for (auto& v : q)
	{
		v = rng.NextFloat();
	}
	AlignedBuf qbuf(bank.view.paddedDims * sizeof(float));
	PadQuery(q, bank.view.paddedDims, qbuf.F32());

	std::vector<uint32_t> excludeBits((count + 31) / 32, 0);
	for (int32_t i = 0; i < 60; ++i)
	{
		const int32_t idx = rng.NextIndex(count);
		excludeBits[idx >> 5] |= 1u << (idx & 31);
	}

	Workspace ws;
	const int32_t k = 10;
	Hit hits[10];
	int32_t n = 0;
	QueryParams params;
	params.k = k;
	params.excludeBits = excludeBits.data();

	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	for (int32_t i = 0; i < n; ++i)
	{
		CHECK(!IsExcluded(excludeBits.data(), hits[i].index));
	}
	const auto ref = ReferenceScan(bank, q.data(), excludeBits.data());
	CheckTopK(bank, ref, hits, n, k);

	// Exclude everything -> empty result.
	std::vector<uint32_t> all((count + 31) / 32, 0xFFFFFFFFu);
	params.excludeBits = all.data();
	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	CHECK(n == 0);
}

// ---------------------------------------------------------------------------
// T8 — zero steady-state allocation (plan B5 seed).

static void TestAllocationFlat()
{
	Rng rng(29);
	TestBank bank(rng, 1000, 64, Quantization::Int8, Metric::Cosine);

	std::vector<float> q(64);
	for (auto& v : q)
	{
		v = rng.NextFloat();
	}
	AlignedBuf qbuf(bank.view.paddedDims * sizeof(float));
	PadQuery(q, bank.view.paddedDims, qbuf.F32());

	Workspace ws;
	Hit hits[10];
	int32_t n = 0;
	QueryParams params;
	params.k = 10;

	// Warm-up: allowed to allocate.
	CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	for (int32_t i = 0; i < 1000; ++i)
	{
		CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
	}
	CHECK_MSG(AllocationCount() == allocsBefore, "allocations grew: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore),
		static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// ---------------------------------------------------------------------------
// T9 — batch equivalence (plan A5): QueryBatch(M) bit-identical to M singles.

static void TestBatchEquivalence()
{
	// Covers every pair-kernel variant (dot/L2 x f32/i8) plus the sub-batch boundary
	// and the odd-pair tail: batch must be bit-identical to singles in all of them.
	Rng rng(31);
	const Metric metrics[] = {Metric::Dot, Metric::Cosine, Metric::L2};
	const Quantization quants[] = {Quantization::Float32, Quantization::Int8};
	for (Metric metric : metrics)
	for (Quantization quant : quants)
	{
	TestBank bank(rng, 2000, 32, quant, metric);

	const int32_t m = 101; // crosses the sub-batch boundary (64) and the pair tail
	const int32_t pd = bank.view.paddedDims;
	AlignedBuf queries(static_cast<size_t>(m) * pd * sizeof(float));
	for (int32_t q = 0; q < m; ++q)
	{
		for (int32_t i = 0; i < bank.view.dims; ++i)
		{
			queries.F32()[static_cast<size_t>(q) * pd + i] = rng.NextFloat();
		}
	}

	const int32_t k = 7;
	QueryParams params;
	params.k = k;

	Workspace wsBatch;
	std::vector<Hit> batchHits(static_cast<size_t>(m) * k);
	std::vector<int32_t> batchCounts(static_cast<size_t>(m));
	CHECK(QueryBatch(bank.view, queries.F32(), m, params, wsBatch, batchHits.data(),
		batchCounts.data()) == Status::Ok);

	Workspace wsSingle;
	std::vector<Hit> singleHits(static_cast<size_t>(k));
	for (int32_t q = 0; q < m; ++q)
	{
		int32_t n = 0;
		CHECK(Query(bank.view, queries.F32() + static_cast<size_t>(q) * pd, params, wsSingle,
			singleHits.data(), &n) == Status::Ok);
		CHECK(n == batchCounts[static_cast<size_t>(q)]);
		for (int32_t i = 0; i < n; ++i)
		{
			const Hit& a = batchHits[static_cast<size_t>(q) * k + i];
			const Hit& b = singleHits[static_cast<size_t>(i)];
			CHECK(a.index == b.index);
			CHECK(a.score == b.score); // bit-identical, same kernels
		}
	}
	}
}

// ---------------------------------------------------------------------------
// T10 — repeat-call determinism (plan B1 seed).

static void TestRepeatDeterminism()
{
	Rng rng(37);
	TestBank bank(rng, 3000, 48, Quantization::Int8, Metric::Dot);

	std::vector<float> q(48);
	for (auto& v : q)
	{
		v = rng.NextFloat();
	}
	AlignedBuf qbuf(bank.view.paddedDims * sizeof(float));
	PadQuery(q, bank.view.paddedDims, qbuf.F32());

	Workspace ws;
	const int32_t k = 25;
	std::vector<Hit> first(static_cast<size_t>(k));
	std::vector<Hit> again(static_cast<size_t>(k));
	int32_t n1 = 0;
	int32_t n2 = 0;
	QueryParams params;
	params.k = k;

	CHECK(Query(bank.view, qbuf.F32(), params, ws, first.data(), &n1) == Status::Ok);
	for (int32_t i = 0; i < 100; ++i)
	{
		CHECK(Query(bank.view, qbuf.F32(), params, ws, again.data(), &n2) == Status::Ok);
		CHECK(n1 == n2);
		CHECK(std::memcmp(first.data(), again.data(), static_cast<size_t>(n1) * sizeof(Hit)) == 0);
	}
}

// ---------------------------------------------------------------------------
// T11 — SIMD ≡ scalar bit-equality (plan §9 mechanism, Day 1 gate). On a scalar-only
// build this is trivially true; on SSE/NEON builds it is the load-bearing check.

static void TestSimdEqualsScalar()
{
	Rng rng(41);
	const int32_t dimsSet[] = {4, 8, 20, 36, 64, 100, 256, 768};
	for (int32_t dims : dimsSet)
	{
		// float32 kernels.
		{
			const int32_t pd = PaddedDims(dims, Quantization::Float32);
			AlignedBuf row(static_cast<size_t>(pd) * sizeof(float));
			AlignedBuf query(static_cast<size_t>(pd) * sizeof(float));
			for (int32_t rep = 0; rep < 50; ++rep)
			{
				for (int32_t i = 0; i < dims; ++i)
				{
					row.F32()[i] = rng.NextFloat();
					query.F32()[i] = rng.NextFloat();
				}
				const float dotSimd = superfaiss::detail::DotF32(row.F32(), query.F32(), pd);
				const float dotScalar = superfaiss::detail::DotF32Mirror(row.F32(), query.F32(), pd);
				CHECK_MSG(dotSimd == dotScalar, "DotF32 dims=%d: simd %.9g scalar %.9g",
					dims, dotSimd, dotScalar);
				const float l2Simd = superfaiss::detail::L2F32(row.F32(), query.F32(), pd);
				const float l2Scalar = superfaiss::detail::L2F32Mirror(row.F32(), query.F32(), pd);
				CHECK_MSG(l2Simd == l2Scalar, "L2F32 dims=%d: simd %.9g scalar %.9g",
					dims, l2Simd, l2Scalar);
			}
		}
		// int8 kernels.
		{
			const int32_t pd = PaddedDims(dims, Quantization::Int8);
			AlignedBuf row(static_cast<size_t>(pd));
			AlignedBuf query(static_cast<size_t>(pd) * sizeof(float));
			for (int32_t rep = 0; rep < 50; ++rep)
			{
				for (int32_t i = 0; i < dims; ++i)
				{
					row.I8()[i] = static_cast<int8_t>(rng.NextIndex(255) - 127);
					query.F32()[i] = rng.NextFloat();
				}
				const float scale = 0.01f + 0.5f * (rng.NextFloat() + 1.0f);
				const float dotSimd = superfaiss::detail::DotI8(row.I8(), scale, query.F32(), pd);
				const float dotScalar = superfaiss::detail::DotI8Mirror(row.I8(), scale, query.F32(), pd);
				CHECK_MSG(dotSimd == dotScalar, "DotI8 dims=%d: simd %.9g scalar %.9g",
					dims, dotSimd, dotScalar);
				const float l2Simd = superfaiss::detail::L2I8(row.I8(), scale, query.F32(), pd);
				const float l2Scalar = superfaiss::detail::L2I8Mirror(row.I8(), scale, query.F32(), pd);
				CHECK_MSG(l2Simd == l2Scalar, "L2I8 dims=%d: simd %.9g scalar %.9g",
					dims, l2Simd, l2Scalar);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// T12 — MergeTopK: per-chunk scans merged externally are bit-identical to a single
// Query over the same bank (the parallel-driver contract; Poirot M1).

static void TestMergeTopK()
{
	Rng rng(43);
	// Big enough for several chunks; includes duplicate rows so merge sees ties.
	const int32_t count = 5000;
	TestBank bank(rng, count, 64, Quantization::Float32, Metric::Dot);
	// Duplicate row 7 into rows 100..104 (crosses chunk boundaries at this size).
	{
		float* rows = static_cast<float*>(const_cast<void*>(bank.view.rows));
		for (int32_t r = 100; r < 105; ++r)
		{
			std::memcpy(rows + static_cast<size_t>(r) * bank.view.paddedDims,
				rows + static_cast<size_t>(7) * bank.view.paddedDims,
				static_cast<size_t>(bank.view.paddedDims) * sizeof(float));
		}
	}

	std::vector<float> q(64);
	for (auto& v : q)
	{
		v = rng.NextFloat();
	}
	AlignedBuf qbuf(bank.view.paddedDims * sizeof(float));
	PadQuery(q, bank.view.paddedDims, qbuf.F32());

	const int32_t k = 16;
	QueryParams params;
	params.k = k;

	// Reference: the normal single-call path.
	Workspace ws;
	std::vector<Hit> whole(static_cast<size_t>(k));
	int32_t wholeCount = 0;
	CHECK(Query(bank.view, qbuf.F32(), params, ws, whole.data(), &wholeCount) == Status::Ok);

	// Chunked: one TopK per chunk, finalized, then merged — in order and in a shuffled
	// order (the total order makes list order irrelevant; prove it).
	const int32_t chunks = ChunkCount(bank.view);
	std::vector<std::vector<Hit>> lists(static_cast<size_t>(chunks));
	std::vector<Hit> heap(static_cast<size_t>(k));
	for (int32_t c = 0; c < chunks; ++c)
	{
		TopK topk;
		topk.Init(heap.data(), k, bank.view.metric);
		ScoreChunk(bank.view, qbuf.F32(), c, nullptr, topk);
		lists[static_cast<size_t>(c)].resize(static_cast<size_t>(k));
		const int32_t n = topk.Finalize(lists[static_cast<size_t>(c)].data());
		lists[static_cast<size_t>(c)].resize(static_cast<size_t>(n));
	}

	auto merge = [&](bool reversed) {
		std::vector<const Hit*> ptrs;
		std::vector<int32_t> counts;
		for (int32_t c = 0; c < chunks; ++c)
		{
			const int32_t idx = reversed ? (chunks - 1 - c) : c;
			ptrs.push_back(lists[static_cast<size_t>(idx)].data());
			counts.push_back(static_cast<int32_t>(lists[static_cast<size_t>(idx)].size()));
		}
		std::vector<Hit> scratch(static_cast<size_t>(k));
		std::vector<Hit> merged(static_cast<size_t>(k));
		const int32_t n = MergeTopK(ptrs.data(), counts.data(), chunks, bank.view.metric,
			k, scratch.data(), merged.data());
		CHECK(n == wholeCount);
		if (n == wholeCount)
		{
			CHECK(std::memcmp(merged.data(), whole.data(),
				static_cast<size_t>(n) * sizeof(Hit)) == 0);
		}
	};
	merge(false);
	merge(true);

	// Edge: merging empty lists yields empty.
	const Hit* noLists[1] = {nullptr};
	int32_t noCounts[1] = {0};
	std::vector<Hit> scratch(static_cast<size_t>(k));
	std::vector<Hit> merged(static_cast<size_t>(k));
	CHECK(MergeTopK(noLists, noCounts, 1, bank.view.metric, k, scratch.data(), merged.data()) == 0);
}

// ---------------------------------------------------------------------------
// T13 — ValidateBankData rejection matrix (Poirot S2).

static void TestValidateBankData()
{
	Rng rng(47);
	int32_t badRow = -1;

	// Healthy float32 bank passes.
	TestBank f32(rng, 20, 6, Quantization::Float32, Metric::Dot); // dims 6 -> pad lanes exist
	CHECK(ValidateBankData(f32.view, &badRow) == Status::Ok);

	// Non-finite row lane rejected.
	{
		float* rows = static_cast<float*>(const_cast<void*>(f32.view.rows));
		const float saved = rows[static_cast<size_t>(3) * f32.view.paddedDims + 2];
		rows[static_cast<size_t>(3) * f32.view.paddedDims + 2] = NAN;
		CHECK(ValidateBankData(f32.view, &badRow) == Status::BadFormat);
		CHECK(badRow == 3);
		rows[static_cast<size_t>(3) * f32.view.paddedDims + 2] = saved;
	}

	// Non-zero pad lane rejected.
	{
		float* rows = static_cast<float*>(const_cast<void*>(f32.view.rows));
		rows[static_cast<size_t>(5) * f32.view.paddedDims + 7] = 0.25f; // lane 7 is padding
		CHECK(ValidateBankData(f32.view, &badRow) == Status::BadFormat);
		CHECK(badRow == 5);
		rows[static_cast<size_t>(5) * f32.view.paddedDims + 7] = 0.0f;
	}

	// Healthy int8 bank passes; bad scales and pad bytes rejected.
	TestBank i8(rng, 20, 20, Quantization::Int8, Metric::Dot); // dims 20 -> pad bytes exist
	CHECK(ValidateBankData(i8.view, &badRow) == Status::Ok);
	{
		float* scales = const_cast<float*>(i8.view.scales);
		const float saved = scales[4];
		scales[4] = -0.5f;
		CHECK(ValidateBankData(i8.view, &badRow) == Status::BadFormat);
		CHECK(badRow == 4);
		scales[4] = NAN;
		CHECK(ValidateBankData(i8.view, &badRow) == Status::BadFormat);
		scales[4] = saved;

		int8_t* rows = static_cast<int8_t*>(const_cast<void*>(i8.view.rows));
		rows[static_cast<size_t>(9) * i8.view.paddedDims + 25] = 1; // lane 25 is padding
		CHECK(ValidateBankData(i8.view, &badRow) == Status::BadFormat);
		CHECK(badRow == 9);
		rows[static_cast<size_t>(9) * i8.view.paddedDims + 25] = 0;
	}
	CHECK(ValidateBankData(i8.view, &badRow) == Status::Ok);
}

// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// T14 — centroid helper (plan 18.1): MakeCentroid equals a double reference across
// the metric x quantization matrix; Cosine renormalizes; zero-norm mean rejected.

static void TestCentroid()
{
	Rng rng(0xCE47401Dull);
	const int32_t dims = 24;
	const int32_t indices[3] = {3, 7, 19};

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			TestBank bank(rng, 40, dims, quant, metric);
			AlignedBuf out(static_cast<size_t>(bank.view.paddedDims) * sizeof(float));
			CHECK(MakeCentroid(bank.view, indices, 3, out.F32()) == Status::Ok);

			std::vector<double> ref(static_cast<size_t>(dims), 0.0);
			for (int32_t i = 0; i < 3; ++i)
			{
				for (int32_t j = 0; j < dims; ++j)
				{
					ref[j] += bank.refRows[static_cast<size_t>(indices[i]) * dims + j];
				}
			}
			for (int32_t j = 0; j < dims; ++j)
			{
				ref[j] /= 3.0;
			}
			if (metric == Metric::Cosine)
			{
				double norm = 0.0;
				for (int32_t j = 0; j < dims; ++j)
				{
					norm += ref[j] * ref[j];
				}
				const double inv = 1.0 / std::sqrt(norm);
				for (int32_t j = 0; j < dims; ++j)
				{
					ref[j] *= inv;
				}
			}
			for (int32_t j = 0; j < dims; ++j)
			{
				CHECK_MSG(std::fabs(out.F32()[j] - ref[j]) <= 1e-5 * (1.0 + std::fabs(ref[j])),
					"centroid dim %d: got %.9g ref %.9g", j, out.F32()[j], ref[j]);
			}
			for (int32_t j = dims; j < bank.view.paddedDims; ++j)
			{
				CHECK(out.F32()[j] == 0.0f);
			}

			// The centroid is a valid query against its own bank.
			CHECK(ValidateQuery(bank.view, out.F32()) == Status::Ok);

			// Rejections: empty set, out-of-range index, misaligned output.
			CHECK(MakeCentroid(bank.view, indices, 0, out.F32()) == Status::InvalidArgument);
			const int32_t bad = bank.view.count;
			CHECK(MakeCentroid(bank.view, &bad, 1, out.F32()) == Status::InvalidArgument);
			CHECK(MakeCentroid(bank.view, indices, 3, out.F32() + 1) == Status::BadAlignment);
		}
	}

	// Zero-norm mean on a Cosine bank: antipodal members cancel; rejected, not
	// renormalized into noise.
	{
		const int32_t d = 4;
		AlignedBuf rows(2 * d * sizeof(float));
		rows.F32()[0] = 1.0f;
		rows.F32()[d] = -1.0f;
		BankView v;
		v.rows = rows.ptr;
		v.count = 2;
		v.dims = d;
		v.paddedDims = d;
		v.quant = Quantization::Float32;
		v.metric = Metric::Cosine;
		AlignedBuf out(d * sizeof(float));
		const int32_t both[2] = {0, 1};
		CHECK(MakeCentroid(v, both, 2, out.F32()) == Status::ZeroNormQuery);
	}
}

// ---------------------------------------------------------------------------
// T15 — direction helper (plan 18.1): normalize(a - b), hand-checked; a == b rejected.

static void TestDirection()
{
	const int32_t d = 4;
	alignas(16) float a[4] = {1.0f, 2.0f, 2.0f, 0.0f};
	alignas(16) float b[4] = {1.0f, 0.0f, 0.0f, 0.0f};
	alignas(16) float out[4];

	CHECK(MakeDirection(a, b, d, d, out) == Status::Ok);
	const float inv = 1.0f / std::sqrt(8.0f);
	CHECK(std::fabs(out[0] - 0.0f) <= 1e-7f);
	CHECK(std::fabs(out[1] - 2.0f * inv) <= 1e-6f);
	CHECK(std::fabs(out[2] - 2.0f * inv) <= 1e-6f);
	CHECK(std::fabs(out[3] - 0.0f) <= 1e-7f);

	double norm = 0.0;
	for (int32_t j = 0; j < d; ++j)
	{
		norm += static_cast<double>(out[j]) * out[j];
	}
	CHECK(std::fabs(norm - 1.0) <= 1e-6);

	CHECK(MakeDirection(a, a, d, d, out) == Status::ZeroNormQuery);
	CHECK(MakeDirection(a, b, d, d, nullptr) == Status::InvalidArgument);
	CHECK(MakeDirection(a, b, 0, d, out) == Status::InvalidArgument);
}

// ---------------------------------------------------------------------------
// T16 — per-query metric override (plan 18.1): ScoreAs::Dot on an L2 bank is
// bit-identical to scoring the same data through a Dot-metric view; identity on
// Dot/Cosine banks; batch equals singles under the override; bank rules hold.

static void TestScoreAsOverride()
{
	Rng rng(0x5C04EA50ull);
	const int32_t dims = 20;
	const int32_t count = 300;
	const int32_t k = 12;

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			TestBank bank(rng, count, dims, quant, metric);
			const int32_t pd = bank.view.paddedDims;
			AlignedBuf q(static_cast<size_t>(pd) * sizeof(float));
			std::vector<float> qv(static_cast<size_t>(dims));
			for (auto& x : qv)
			{
				x = rng.NextFloat();
			}
			PadQuery(qv, pd, q.F32());

			QueryParams overrideParams;
			overrideParams.k = k;
			overrideParams.scoreAs = ScoreAs::Dot;

			Workspace ws;
			std::vector<Hit> got(static_cast<size_t>(k));
			int32_t gotCount = 0;
			CHECK(Query(bank.view, q.F32(), overrideParams, ws, got.data(), &gotCount) ==
				Status::Ok);

			// Reference: the same payload viewed as a Dot bank, default params.
			BankView dotView = bank.view;
			dotView.metric = Metric::Dot;
			QueryParams plain;
			plain.k = k;
			Workspace ws2;
			std::vector<Hit> ref(static_cast<size_t>(k));
			int32_t refCount = 0;
			CHECK(Query(dotView, q.F32(), plain, ws2, ref.data(), &refCount) == Status::Ok);

			CHECK(gotCount == refCount);
			for (int32_t i = 0; i < gotCount && i < refCount; ++i)
			{
				CHECK(got[i].index == ref[i].index && got[i].score == ref[i].score);
			}

			// Identity on Dot/Cosine banks: override equals default, bit-identical.
			if (metric != Metric::L2)
			{
				Workspace ws3;
				std::vector<Hit> plainHits(static_cast<size_t>(k));
				int32_t plainCount = 0;
				CHECK(Query(bank.view, q.F32(), plain, ws3, plainHits.data(), &plainCount) ==
					Status::Ok);
				CHECK(plainCount == gotCount);
				for (int32_t i = 0; i < gotCount && i < plainCount; ++i)
				{
					CHECK(got[i].index == plainHits[i].index &&
						got[i].score == plainHits[i].score);
				}
			}

			// Batch under the override equals singles under the override, bit-identical.
			const int32_t m = 5;
			AlignedBuf batch(static_cast<size_t>(m) * pd * sizeof(float));
			for (int32_t qi = 0; qi < m; ++qi)
			{
				std::vector<float> row(static_cast<size_t>(dims));
				for (auto& x : row)
				{
					x = rng.NextFloat();
				}
				PadQuery(row, pd, batch.F32() + static_cast<int64_t>(qi) * pd);
			}
			Workspace wsB;
			std::vector<Hit> bHits(static_cast<size_t>(m) * k);
			std::vector<int32_t> bCounts(static_cast<size_t>(m));
			CHECK(QueryBatch(bank.view, batch.F32(), m, overrideParams, wsB, bHits.data(),
				bCounts.data()) == Status::Ok);
			for (int32_t qi = 0; qi < m; ++qi)
			{
				Workspace wsS;
				std::vector<Hit> sHits(static_cast<size_t>(k));
				int32_t sCount = 0;
				CHECK(Query(bank.view, batch.F32() + static_cast<int64_t>(qi) * pd,
					overrideParams, wsS, sHits.data(), &sCount) == Status::Ok);
				CHECK(sCount == bCounts[static_cast<size_t>(qi)]);
				for (int32_t i = 0; i < sCount; ++i)
				{
					const Hit& bh = bHits[static_cast<size_t>(qi) * k + i];
					CHECK(bh.index == sHits[i].index && bh.score == sHits[i].score);
				}
			}
		}
	}

	// Bank rules survive the override: a zero-norm query on a Cosine bank is rejected
	// under ScoreAs::Dot exactly as without it.
	{
		TestBank bank(rng, 8, dims, Quantization::Float32, Metric::Cosine);
		AlignedBuf zq(static_cast<size_t>(bank.view.paddedDims) * sizeof(float));
		QueryParams p;
		p.k = 3;
		p.scoreAs = ScoreAs::Dot;
		Workspace ws;
		Hit h[3];
		int32_t n = 0;
		CHECK(Query(bank.view, zq.F32(), p, ws, h, &n) == Status::ZeroNormQuery);
	}
}

// ---------------------------------------------------------------------------
// T17 — margin (plan 18.1): better-direction gap, non-negative on sorted output,
// exact against hit scores, both metric directions.

static void TestMargin()
{
	// Hand case, Dot: scores 2 then 1 give margin 1. L2: 1 then 2 give margin 1.
	CHECK(Margin(Hit{0, 2.0f}, Hit{1, 1.0f}, Metric::Dot) == 1.0f);
	CHECK(Margin(Hit{0, 1.0f}, Hit{1, 2.0f}, Metric::L2) == 1.0f);

	Rng rng(0xA46177ull);
	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			TestBank bank(rng, 120, 16, quant, metric);
			AlignedBuf q(static_cast<size_t>(bank.view.paddedDims) * sizeof(float));
			std::vector<float> qv(16);
			for (auto& x : qv)
			{
				x = rng.NextFloat();
			}
			PadQuery(qv, bank.view.paddedDims, q.F32());

			QueryParams p;
			p.k = 8;
			Workspace ws;
			Hit hits[8];
			int32_t n = 0;
			CHECK(Query(bank.view, q.F32(), p, ws, hits, &n) == Status::Ok);
			const Metric scored = ScoringMetric(bank.view, p);
			for (int32_t i = 0; i + 1 < n; ++i)
			{
				const float gap = Margin(hits[i], hits[i + 1], scored);
				CHECK_MSG(gap >= 0.0f, "negative margin %g at %d", gap, i);
				const float expect = scored == Metric::L2
					? hits[i + 1].score - hits[i].score
					: hits[i].score - hits[i + 1].score;
				CHECK(gap == expect);
			}
		}
	}
}


// ---------------------------------------------------------------------------
// T18 — intersection combinator (plan 18.7): QueryIntersect equals a full-scan
// double reference (per-row worst-of fusion, total-order sorted) across the
// metric x quantization matrix; M=1 degenerates to Query bit-identically;
// exclusion respected; externally-merged chunks bit-identical to the single pass.

static void TestIntersect()
{
	Rng rng(0x1472E5EC7ull);
	const int32_t dims = 20;
	const int32_t count = 400;
	const int32_t k = 10;
	const int32_t m = 3;

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			TestBank bank(rng, count, dims, quant, metric);
			const int32_t pd = bank.view.paddedDims;

			AlignedBuf queries(static_cast<size_t>(m) * pd * sizeof(float));
			for (int32_t qi = 0; qi < m; ++qi)
			{
				std::vector<float> qv(static_cast<size_t>(dims));
				for (auto& x : qv)
				{
					x = rng.NextFloat();
				}
				PadQuery(qv, pd, queries.F32() + static_cast<int64_t>(qi) * pd);
			}

			QueryParams params;
			params.k = k;

			Workspace ws;
			std::vector<Hit> got(static_cast<size_t>(k));
			int32_t gotCount = 0;
			CHECK(QueryIntersect(bank.view, queries.F32(), m, params, ws, got.data(),
				&gotCount) == Status::Ok);

			// Double reference: per-row worst-of over per-query double scores.
			std::vector<RefHit> ref;
			for (int32_t r = 0; r < count; ++r)
			{
				const double* row = bank.refRows.data() + static_cast<size_t>(r) * dims;
				double fused = 0.0;
				for (int32_t qi = 0; qi < m; ++qi)
				{
					const float* q = queries.F32() + static_cast<int64_t>(qi) * pd;
					double score = 0.0;
					if (metric == Metric::L2)
					{
						for (int32_t i = 0; i < dims; ++i)
						{
							const double d = static_cast<double>(q[i]) - row[i];
							score += d * d;
						}
					}
					else
					{
						for (int32_t i = 0; i < dims; ++i)
						{
							score += static_cast<double>(q[i]) * row[i];
						}
					}
					if (qi == 0 || (metric == Metric::L2 ? score > fused : score < fused))
					{
						fused = score;
					}
				}
				ref.push_back({r, fused});
			}
			std::sort(ref.begin(), ref.end(), [&](const RefHit& a, const RefHit& b) {
				return RefBetter(a, b, metric);
			});
			CheckTopK(bank, ref, got.data(), gotCount, k);

			// AND guarantee spot-check: the top fused hit scores at-or-better than its
			// fused score against EVERY member query (by construction of worst-of).
			if (gotCount > 0)
			{
				const double fusedTop = ref[0].score;
				const double* row =
					bank.refRows.data() + static_cast<size_t>(ref[0].index) * dims;
				for (int32_t qi = 0; qi < m; ++qi)
				{
					const float* q = queries.F32() + static_cast<int64_t>(qi) * pd;
					double score = 0.0;
					if (metric == Metric::L2)
					{
						for (int32_t i = 0; i < dims; ++i)
						{
							const double d = static_cast<double>(q[i]) - row[i];
							score += d * d;
						}
					}
					else
					{
						for (int32_t i = 0; i < dims; ++i)
						{
							score += static_cast<double>(q[i]) * row[i];
						}
					}
					const bool holds = metric == Metric::L2 ? score <= fusedTop + 1e-9
					                                        : score >= fusedTop - 1e-9;
					CHECK_MSG(holds, "AND guarantee: q%d score %.9g fused %.9g",
						qi, score, fusedTop);
				}
			}

			// M=1 degeneracy: bit-identical to Query.
			Workspace ws1;
			std::vector<Hit> single(static_cast<size_t>(k));
			int32_t singleCount = 0;
			CHECK(Query(bank.view, queries.F32(), params, ws1, single.data(),
				&singleCount) == Status::Ok);
			Workspace ws2;
			std::vector<Hit> degen(static_cast<size_t>(k));
			int32_t degenCount = 0;
			CHECK(QueryIntersect(bank.view, queries.F32(), 1, params, ws2, degen.data(),
				&degenCount) == Status::Ok);
			CHECK(degenCount == singleCount);
			for (int32_t i = 0; i < degenCount && i < singleCount; ++i)
			{
				CHECK(degen[i].index == single[i].index && degen[i].score == single[i].score);
			}

			// Exclusion: excluded rows never appear in fused results.
			std::vector<uint32_t> exclude(static_cast<size_t>((count + 31) / 32), 0u);
			if (gotCount > 0)
			{
				const int32_t banned = got[0].index;
				exclude[banned >> 5] |= 1u << (banned & 31);
				QueryParams px = params;
				px.excludeBits = exclude.data();
				Workspace ws3;
				std::vector<Hit> ex(static_cast<size_t>(k));
				int32_t exCount = 0;
				CHECK(QueryIntersect(bank.view, queries.F32(), m, px, ws3, ex.data(),
					&exCount) == Status::Ok);
				for (int32_t i = 0; i < exCount; ++i)
				{
					CHECK(ex[i].index != banned);
				}
			}

			// External per-chunk fusion + merge is bit-identical to the single pass
			// (the parallel path's shape; mirrors T12).
			{
				const int32_t chunks = ChunkCount(bank.view);
				std::vector<Hit> heap(static_cast<size_t>(chunks) * k);
				std::vector<Hit> sorted(static_cast<size_t>(chunks) * k);
				std::vector<const Hit*> lists(static_cast<size_t>(chunks));
				std::vector<int32_t> counts(static_cast<size_t>(chunks));
				for (int32_t c = 0; c < chunks; ++c)
				{
					TopK chunkTop;
					chunkTop.Init(heap.data() + static_cast<size_t>(c) * k, k, bank.view.metric);
					ScoreChunkFused(bank.view, queries.F32(), m, c, nullptr, chunkTop);
					lists[c] = sorted.data() + static_cast<size_t>(c) * k;
					counts[c] = chunkTop.Finalize(sorted.data() + static_cast<size_t>(c) * k);
				}
				std::vector<Hit> mergeHeap(static_cast<size_t>(k));
				std::vector<Hit> merged(static_cast<size_t>(k));
				const int32_t mergedCount = MergeTopK(lists.data(), counts.data(), chunks,
					bank.view.metric, k, mergeHeap.data(), merged.data());
				CHECK(mergedCount == gotCount);
				for (int32_t i = 0; i < mergedCount && i < gotCount; ++i)
				{
					CHECK(merged[i].index == got[i].index && merged[i].score == got[i].score);
				}
			}
		}
	}

	// Override interplay: intersect on an L2 bank under ScoreAs::Dot equals intersect
	// on the same payload viewed as a Dot bank.
	{
		TestBank bank(rng, 200, dims, Quantization::Float32, Metric::L2);
		const int32_t pd = bank.view.paddedDims;
		AlignedBuf queries(static_cast<size_t>(2) * pd * sizeof(float));
		for (int32_t qi = 0; qi < 2; ++qi)
		{
			std::vector<float> qv(static_cast<size_t>(dims));
			for (auto& x : qv)
			{
				x = rng.NextFloat();
			}
			PadQuery(qv, pd, queries.F32() + static_cast<int64_t>(qi) * pd);
		}
		QueryParams po;
		po.k = k;
		po.scoreAs = ScoreAs::Dot;
		Workspace wsA;
		std::vector<Hit> viaOverride(static_cast<size_t>(k));
		int32_t nOverride = 0;
		CHECK(QueryIntersect(bank.view, queries.F32(), 2, po, wsA, viaOverride.data(),
			&nOverride) == Status::Ok);

		BankView dotView = bank.view;
		dotView.metric = Metric::Dot;
		QueryParams plain;
		plain.k = k;
		Workspace wsB;
		std::vector<Hit> viaDotView(static_cast<size_t>(k));
		int32_t nDotView = 0;
		CHECK(QueryIntersect(dotView, queries.F32(), 2, plain, wsB, viaDotView.data(),
			&nDotView) == Status::Ok);
		CHECK(nOverride == nDotView);
		for (int32_t i = 0; i < nOverride && i < nDotView; ++i)
		{
			CHECK(viaOverride[i].index == viaDotView[i].index &&
				viaOverride[i].score == viaDotView[i].score);
		}
	}
}


// ---------------------------------------------------------------------------
// T19 — PCA (plan 18.2): power iteration recovers a known dominant direction;
// components orthonormal; bit-deterministic across runs; projection hand-checked;
// int8 path dequantizes through scales; degenerate banks yield zero components.

static void TestPca()
{
	Rng rng(0x9CA0ull);
	const int32_t dims = 12;
	const int32_t count = 300;

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		// Rows: large spread along axis 0, small noise elsewhere — PC1 must align
		// with e0 up to sign.
		std::vector<float> src(static_cast<size_t>(count) * dims);
		for (int32_t r = 0; r < count; ++r)
		{
			for (int32_t j = 0; j < dims; ++j)
			{
				src[static_cast<size_t>(r) * dims + j] =
					j == 0 ? 4.0f * rng.NextFloat() : 0.05f * rng.NextFloat();
			}
		}
		// Build a bank directly from the shaped source.
		const int32_t pd = PaddedDims(dims, quant);
		AlignedBuf payload(static_cast<size_t>(count) * pd * ElementSize(quant));
		std::vector<float> scales;
		BankView view;
		view.count = count;
		view.dims = dims;
		view.paddedDims = pd;
		view.quant = quant;
		view.metric = Metric::Dot;
		if (quant == Quantization::Float32)
		{
			PadRowsFloat32(src.data(), count, dims, pd, payload.F32());
			view.rows = payload.ptr;
			view.scales = nullptr;
		}
		else
		{
			scales.resize(static_cast<size_t>(count));
			QuantizeRowsInt8(src.data(), count, dims, pd, payload.I8(), scales.data());
			view.rows = payload.ptr;
			view.scales = scales.data();
		}

		std::vector<float> mean(static_cast<size_t>(dims));
		std::vector<float> comps(static_cast<size_t>(2) * dims);
		std::vector<float> scratch(static_cast<size_t>(dims));
		CHECK(ComputePrincipalComponents(view, 2, 48, mean.data(), comps.data(),
			scratch.data()) == Status::Ok);

		// PC1 aligns with e0 (up to sign).
		CHECK_MSG(std::fabs(comps[0]) > 0.99,
			"PC1 axis-0 loading %.6f", comps[0]);

		// Orthonormal: unit norms, near-zero mutual dot.
		double n1 = 0.0, n2 = 0.0, dot12 = 0.0;
		for (int32_t j = 0; j < dims; ++j)
		{
			n1 += static_cast<double>(comps[j]) * comps[j];
			n2 += static_cast<double>(comps[dims + j]) * comps[dims + j];
			dot12 += static_cast<double>(comps[j]) * comps[dims + j];
		}
		CHECK(std::fabs(n1 - 1.0) <= 1e-5);
		CHECK(std::fabs(n2 - 1.0) <= 1e-5);
		CHECK(std::fabs(dot12) <= 1e-4);

		// Bit-deterministic: a second run is identical.
		std::vector<float> mean2(static_cast<size_t>(dims));
		std::vector<float> comps2(static_cast<size_t>(2) * dims);
		CHECK(ComputePrincipalComponents(view, 2, 48, mean2.data(), comps2.data(),
			scratch.data()) == Status::Ok);
		for (int32_t j = 0; j < dims; ++j)
		{
			CHECK(mean[j] == mean2[j]);
			CHECK(comps[j] == comps2[j] && comps[dims + j] == comps2[dims + j]);
		}

		// Projection: coordinate of row r on PC1 equals the hand dot product.
		std::vector<float> coords(static_cast<size_t>(count) * 2);
		CHECK(ProjectRowsOntoComponents(view, mean.data(), comps.data(), 2,
			coords.data()) == Status::Ok);
		{
			const int32_t r = 7;
			double expect = 0.0;
			for (int32_t j = 0; j < dims; ++j)
			{
				double e = 0.0;
				if (quant == Quantization::Int8)
				{
					const int8_t* row = static_cast<const int8_t*>(view.rows) +
						static_cast<int64_t>(r) * pd;
					e = static_cast<double>(row[j]) * scales[r];
				}
				else
				{
					const float* row = static_cast<const float*>(view.rows) +
						static_cast<int64_t>(r) * pd;
					e = row[j];
				}
				expect += (e - mean[j]) * comps[j];
			}
			CHECK_MSG(std::fabs(coords[static_cast<size_t>(r) * 2] - expect) <=
				1e-4 * (1.0 + std::fabs(expect)),
				"projection %.9g expect %.9g", coords[static_cast<size_t>(r) * 2], expect);
		}
	}

	// Degenerate: identical rows — zero variance — components come back zero.
	{
		const int32_t d = 8;
		const int32_t n = 10;
		const int32_t pd = PaddedDims(d, Quantization::Float32);
		std::vector<float> src(static_cast<size_t>(n) * d, 0.5f);
		AlignedBuf payload(static_cast<size_t>(n) * pd * sizeof(float));
		PadRowsFloat32(src.data(), n, d, pd, payload.F32());
		BankView view;
		view.rows = payload.ptr;
		view.count = n;
		view.dims = d;
		view.paddedDims = pd;
		view.quant = Quantization::Float32;
		view.metric = Metric::Dot;
		std::vector<float> mean(static_cast<size_t>(d));
		std::vector<float> comps(static_cast<size_t>(d));
		std::vector<float> scratch(static_cast<size_t>(d));
		CHECK(ComputePrincipalComponents(view, 1, 16, mean.data(), comps.data(),
			scratch.data()) == Status::Ok);
		double norm = 0.0;
		for (int32_t j = 0; j < d; ++j)
		{
			norm += static_cast<double>(comps[j]) * comps[j];
		}
		CHECK(norm == 0.0);
	}
}


// ---------------------------------------------------------------------------
// T20 — segmented scan, slot 1 (V2 plan §4/§12):
//   A1 — degenerate one-segment list bit-identical to the V1 kernel, full
//        metric x quantization matrix, through Query, QueryBatch, and
//        QueryIntersect.
//   A2 — segmented scores equal an independent float64 reference over the same
//        segments with weights (dot / L2; cosine's per-channel norms are bank
//        data and land with schemaVersion 2 in slot 2).
//   A3 — masked ranges are never read: perturbing them leaves scores bitwise
//        unchanged.
//   B  — repeat determinism, external chunk-fusion merge bit-identity,
//        validation rejections including the query-side zero-norm segment law.

static void TestSegmentedScan()
{
	Rng rng(0x5E63E17ull);
	const int32_t dims = 32; // f32 grid 4, int8 grid 16: 32 works for both
	const int32_t count = 300;
	const int32_t k = 10;

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			TestBank bank(rng, count, dims, quant, metric);
			const int32_t pd = bank.view.paddedDims;
			AlignedBuf q(static_cast<size_t>(pd) * sizeof(float));
			std::vector<float> qv(static_cast<size_t>(dims));
			for (auto& x : qv)
			{
				x = rng.NextFloat();
			}
			PadQuery(qv, pd, q.F32());

			// --- A1: degenerate segment ≡ V1, bit-identical.
			{
				QueryParams plain;
				plain.k = k;
				Workspace ws;
				std::vector<Hit> v1(static_cast<size_t>(k));
				int32_t v1Count = 0;
				CHECK(Query(bank.view, q.F32(), plain, ws, v1.data(), &v1Count) ==
					Status::Ok);

				const QuerySegment degenerate[1] = {{0, pd, 1.0f}};
				QueryParams seg = plain;
				seg.segments = degenerate;
				seg.segmentCount = 1;
				Workspace ws2;
				std::vector<Hit> v2(static_cast<size_t>(k));
				int32_t v2Count = 0;
				CHECK(Query(bank.view, q.F32(), seg, ws2, v2.data(), &v2Count) ==
					Status::Ok);
				CHECK(v2Count == v1Count);
				for (int32_t i = 0; i < v1Count && i < v2Count; ++i)
				{
					CHECK(v2[i].index == v1[i].index && v2[i].score == v1[i].score);
				}

				// Batch degenerate ≡ plain batch, bit-identical.
				const int32_t m = 3;
				AlignedBuf batch(static_cast<size_t>(m) * pd * sizeof(float));
				for (int32_t qi = 0; qi < m; ++qi)
				{
					std::vector<float> row(static_cast<size_t>(dims));
					for (auto& x : row)
					{
						x = rng.NextFloat();
					}
					PadQuery(row, pd, batch.F32() + static_cast<int64_t>(qi) * pd);
				}
				Workspace wsB1, wsB2;
				std::vector<Hit> plainHits(static_cast<size_t>(m) * k);
				std::vector<Hit> segHits(static_cast<size_t>(m) * k);
				std::vector<int32_t> plainCounts(static_cast<size_t>(m));
				std::vector<int32_t> segCounts(static_cast<size_t>(m));
				CHECK(QueryBatch(bank.view, batch.F32(), m, plain, wsB1,
					plainHits.data(), plainCounts.data()) == Status::Ok);
				CHECK(QueryBatch(bank.view, batch.F32(), m, seg, wsB2,
					segHits.data(), segCounts.data()) == Status::Ok);
				for (int32_t qi = 0; qi < m; ++qi)
				{
					CHECK(segCounts[qi] == plainCounts[qi]);
					for (int32_t i = 0; i < segCounts[qi]; ++i)
					{
						const Hit& a = segHits[static_cast<size_t>(qi) * k + i];
						const Hit& b = plainHits[static_cast<size_t>(qi) * k + i];
						CHECK(a.index == b.index && a.score == b.score);
					}
				}

				// Intersect degenerate ≡ plain intersect, bit-identical.
				Workspace wsI1, wsI2;
				std::vector<Hit> plainF(static_cast<size_t>(k));
				std::vector<Hit> segF(static_cast<size_t>(k));
				int32_t plainFCount = 0, segFCount = 0;
				CHECK(QueryIntersect(bank.view, batch.F32(), m, plain, wsI1,
					plainF.data(), &plainFCount) == Status::Ok);
				CHECK(QueryIntersect(bank.view, batch.F32(), m, seg, wsI2,
					segF.data(), &segFCount) == Status::Ok);
				CHECK(segFCount == plainFCount);
				for (int32_t i = 0; i < plainFCount && i < segFCount; ++i)
				{
					CHECK(segF[i].index == plainF[i].index &&
						segF[i].score == plainF[i].score);
				}
			}

			// --- A2: two weighted segments vs float64 reference (dot and L2; a
			// Cosine bank's segmented semantics refine with slot 2's channel norms —
			// here its rows score as stored, which the reference mirrors).
			{
				const int32_t grid = kAlignment / ElementSize(quant);
				const int32_t segLen = (dims / 2 / grid) * grid;
				const QuerySegment segs[2] = {
					{0, segLen, 0.75f},
					{segLen, segLen, 2.0f},
				};
				QueryParams sp;
				sp.k = k;
				sp.segments = segs;
				sp.segmentCount = 2;
				Workspace ws;
				std::vector<Hit> got(static_cast<size_t>(k));
				int32_t gotCount = 0;
				CHECK(Query(bank.view, q.F32(), sp, ws, got.data(), &gotCount) ==
					Status::Ok);

				std::vector<RefHit> ref;
				for (int32_t r = 0; r < count; ++r)
				{
					const double* row = bank.refRows.data() + static_cast<size_t>(r) * dims;
					double total = 0.0;
					for (const QuerySegment& sg : segs)
					{
						double partial = 0.0;
						for (int32_t j = sg.offset; j < sg.offset + sg.length && j < dims; ++j)
						{
							if (metric == Metric::L2)
							{
								const double d = static_cast<double>(q.F32()[j]) - row[j];
								partial += d * d;
							}
							else
							{
								partial += static_cast<double>(q.F32()[j]) * row[j];
							}
						}
						total += static_cast<double>(sg.weight) * partial;
					}
					ref.push_back({r, total});
				}
				std::sort(ref.begin(), ref.end(), [&](const RefHit& a, const RefHit& b) {
					return RefBetter(a, b, metric);
				});
				CheckTopK(bank, ref, got.data(), gotCount, k);
			}

			// --- A3: masked ranges never read — perturb the omitted middle, scores
			// bitwise unchanged. (Copy the payload, poison the masked range, rescan.)
			{
				const int32_t grid = kAlignment / ElementSize(quant);
				const int32_t third = (dims / 3 / grid) * grid;
				if (third > 0 && 3 * third <= pd)
				{
					const QuerySegment maskSegs[2] = {
						{0, third, 1.0f},
						{2 * third, third, 1.0f}, // middle third omitted = masked
					};
					QueryParams mp;
					mp.k = k;
					mp.segments = maskSegs;
					mp.segmentCount = 2;
					Workspace ws;
					std::vector<Hit> before(static_cast<size_t>(k));
					int32_t beforeCount = 0;
					CHECK(Query(bank.view, q.F32(), mp, ws, before.data(),
						&beforeCount) == Status::Ok);

					const int64_t bytes = BankBytes(bank.view);
					AlignedBuf poisoned(static_cast<size_t>(bytes));
					std::memcpy(poisoned.ptr, bank.view.rows, static_cast<size_t>(bytes));
					BankView pv = bank.view;
					pv.rows = poisoned.ptr;
					for (int32_t r = 0; r < count; ++r)
					{
						if (quant == Quantization::Float32)
						{
							float* row = poisoned.F32() + static_cast<int64_t>(r) * pd;
							for (int32_t j = third; j < 2 * third; ++j)
							{
								row[j] = 12345.678f + j;
							}
						}
						else
						{
							int8_t* row = poisoned.I8() + static_cast<int64_t>(r) * pd;
							for (int32_t j = third; j < 2 * third; ++j)
							{
								row[j] = static_cast<int8_t>((j * 37) & 0x7F);
							}
						}
					}
					Workspace ws2;
					std::vector<Hit> after(static_cast<size_t>(k));
					int32_t afterCount = 0;
					CHECK(Query(pv, q.F32(), mp, ws2, after.data(), &afterCount) ==
						Status::Ok);
					CHECK(afterCount == beforeCount);
					for (int32_t i = 0; i < beforeCount && i < afterCount; ++i)
					{
						CHECK(after[i].index == before[i].index &&
							after[i].score == before[i].score);
					}
				}
			}

			// --- B: repeat determinism + external chunk merge bit-identity.
			{
				const int32_t grid = kAlignment / ElementSize(quant);
				const QuerySegment segs[1] = {{0, (dims / grid) * grid, 1.5f}};
				QueryParams sp;
				sp.k = k;
				sp.segments = segs;
				sp.segmentCount = 1;

				Workspace wsA, wsB;
				std::vector<Hit> a(static_cast<size_t>(k)), b(static_cast<size_t>(k));
				int32_t na = 0, nb = 0;
				CHECK(Query(bank.view, q.F32(), sp, wsA, a.data(), &na) == Status::Ok);
				CHECK(Query(bank.view, q.F32(), sp, wsB, b.data(), &nb) == Status::Ok);
				CHECK(na == nb);
				for (int32_t i = 0; i < na && i < nb; ++i)
				{
					CHECK(a[i].index == b[i].index && a[i].score == b[i].score);
				}

				const int32_t chunks = ChunkCount(bank.view);
				std::vector<Hit> heap(static_cast<size_t>(chunks) * k);
				std::vector<Hit> sorted(static_cast<size_t>(chunks) * k);
				std::vector<const Hit*> lists(static_cast<size_t>(chunks));
				std::vector<int32_t> counts(static_cast<size_t>(chunks));
				// The external chunk path mirrors the shipped split: dot-family
				// folds weights into the query and runs the plain chunk scan; L2
				// uses the dense segmented chunk primitive.
				AlignedBuf folded(static_cast<size_t>(pd) * sizeof(float));
				if (metric != Metric::L2)
				{
					for (int32_t j = 0; j < pd; ++j)
					{
						folded.F32()[j] = 0.0f;
					}
					for (int32_t j = segs[0].offset;
						j < segs[0].offset + segs[0].length; ++j)
					{
						folded.F32()[j] = segs[0].weight * q.F32()[j];
					}
				}
				for (int32_t c = 0; c < chunks; ++c)
				{
					TopK chunkTop;
					chunkTop.Init(heap.data() + static_cast<size_t>(c) * k, k,
						bank.view.metric);
					if (metric != Metric::L2)
					{
						ScoreChunk(bank.view, folded.F32(), c, nullptr, chunkTop);
					}
					else
					{
						ScoreChunkSegmented(bank.view, q.F32(), c, nullptr, segs, 1,
							chunkTop);
					}
					lists[c] = sorted.data() + static_cast<size_t>(c) * k;
					counts[c] = chunkTop.Finalize(
						sorted.data() + static_cast<size_t>(c) * k);
				}
				std::vector<Hit> mergeHeap(static_cast<size_t>(k));
				std::vector<Hit> merged(static_cast<size_t>(k));
				const int32_t mergedCount = MergeTopK(lists.data(), counts.data(),
					chunks, bank.view.metric, k, mergeHeap.data(), merged.data());
				CHECK(mergedCount == na);
				for (int32_t i = 0; i < mergedCount && i < na; ++i)
				{
					CHECK(merged[i].index == a[i].index && merged[i].score == a[i].score);
				}
			}
		}
	}

	// --- Validation rejections.
	{
		TestBank bank(rng, 16, dims, Quantization::Float32, Metric::Cosine);
		const int32_t pd = bank.view.paddedDims;
		AlignedBuf q(static_cast<size_t>(pd) * sizeof(float));
		std::vector<float> qv(static_cast<size_t>(dims));
		for (auto& x : qv)
		{
			x = rng.NextFloat();
		}
		PadQuery(qv, pd, q.F32());
		Workspace ws;
		Hit h[4];
		int32_t n = 0;
		QueryParams sp;
		sp.k = 4;

		// Overlap.
		const QuerySegment overlap[2] = {{0, 8, 1.0f}, {4, 8, 1.0f}};
		sp.segments = overlap;
		sp.segmentCount = 2;
		CHECK(Query(bank.view, q.F32(), sp, ws, h, &n) == Status::InvalidArgument);

		// Off-grid offset (grid is 4 for f32).
		const QuerySegment offgrid[1] = {{2, 8, 1.0f}};
		sp.segments = offgrid;
		sp.segmentCount = 1;
		CHECK(Query(bank.view, q.F32(), sp, ws, h, &n) == Status::InvalidArgument);

		// Out of range.
		const QuerySegment oob[1] = {{0, pd + 4, 1.0f}};
		sp.segments = oob;
		sp.segmentCount = 1;
		CHECK(Query(bank.view, q.F32(), sp, ws, h, &n) == Status::InvalidArgument);

		// Too many segments.
		QuerySegment many[kMaxSegments + 1];
		for (int32_t i = 0; i <= kMaxSegments; ++i)
		{
			many[i] = {i * 4, 4, 1.0f};
		}
		sp.segments = many;
		sp.segmentCount = kMaxSegments + 1;
		CHECK(Query(bank.view, q.F32(), sp, ws, h, &n) == Status::InvalidArgument);

		// Query-side zero-norm segment law on a Cosine bank: zero the query over a
		// nonzero-weight segment -> ZeroNormQuery; weight-0 on that segment -> Ok.
		AlignedBuf zq(static_cast<size_t>(pd) * sizeof(float));
		std::memcpy(zq.F32(), q.F32(), static_cast<size_t>(pd) * sizeof(float));
		for (int32_t j = 0; j < 8; ++j)
		{
			zq.F32()[j] = 0.0f;
		}
		const QuerySegment zseg[2] = {{0, 8, 1.0f}, {8, 8, 1.0f}};
		sp.segments = zseg;
		sp.segmentCount = 2;
		CHECK(Query(bank.view, zq.F32(), sp, ws, h, &n) == Status::ZeroNormQuery);
		const QuerySegment zskip[2] = {{0, 8, 0.0f}, {8, 8, 1.0f}};
		sp.segments = zskip;
		sp.segmentCount = 2;
		CHECK(Query(bank.view, zq.F32(), sp, ws, h, &n) == Status::Ok);
	}
}


// ---------------------------------------------------------------------------
// T21 — per-channel cosine (V2 plan section 5, D-V2-1; slot 2):
//   channel-matched segments on a channel-carrying Cosine bank score as TRUE
//   per-channel cosines via inverse sub-norms baked from the QUANTIZED rows;
//   zero-norm row channels score 0 (never NaN); decomposition contributions
//   sum bit-exactly to the query total; the metric override composes to raw
//   projection; bad channel tables are rejected.

static void TestPerChannelCosine()
{
	Rng rng(0xC4A22E1ull);
	const int32_t dims = 32;
	const int32_t count = 200;
	const int32_t k = 8;

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		TestBank bank(rng, count, dims, quant, Metric::Cosine);
		const int32_t pd = bank.view.paddedDims;
		const int32_t grid = kAlignment / ElementSize(quant);
		const int32_t half = (dims / 2 / grid) * grid;

		// Two channels covering the row.
		const ChannelInfo channels[2] = {{0, half}, {half, pd - half}};
		std::vector<float> invNorms(static_cast<size_t>(count) * 2);
		BankView view = bank.view;
		view.channels = channels;
		view.channelCount = 2;
		CHECK(ComputeChannelInverseNorms(view, invNorms.data()) == Status::Ok);
		view.channelInvNorms = invNorms.data();
		CHECK(ValidateBank(view) == Status::Ok);

		// Query with per-channel-renormalized sub-vectors (the D-V2-1 build rule).
		AlignedBuf q(static_cast<size_t>(pd) * sizeof(float));
		std::vector<float> qv(static_cast<size_t>(dims));
		for (auto& x : qv)
		{
			x = rng.NextFloat();
		}
		PadQuery(qv, pd, q.F32());
		for (const ChannelInfo& ch : channels)
		{
			double norm = 0.0;
			for (int32_t j = ch.offset; j < ch.offset + ch.length; ++j)
			{
				norm += static_cast<double>(q.F32()[j]) * q.F32()[j];
			}
			const double inv = norm > 0.0 ? 1.0 / std::sqrt(norm) : 0.0;
			for (int32_t j = ch.offset; j < ch.offset + ch.length; ++j)
			{
				q.F32()[j] = static_cast<float>(q.F32()[j] * inv);
			}
		}

		const QuerySegment segs[2] = {
			{0, half, 1.0f},
			{half, pd - half, 0.5f},
		};
		QueryParams sp;
		sp.k = k;
		sp.segments = segs;
		sp.segmentCount = 2;

		Workspace ws;
		std::vector<Hit> got(static_cast<size_t>(k));
		int32_t gotCount = 0;
		CHECK(Query(view, q.F32(), sp, ws, got.data(), &gotCount) == Status::Ok);

		// Double reference: true per-channel cosines of the QUANTIZED rows.
		std::vector<RefHit> ref;
		for (int32_t r = 0; r < count; ++r)
		{
			const double* row = bank.refRows.data() + static_cast<size_t>(r) * dims;
			double total = 0.0;
			for (int32_t sIdx = 0; sIdx < 2; ++sIdx)
			{
				const QuerySegment& sg = segs[sIdx];
				double dot = 0.0, rowNorm = 0.0;
				for (int32_t j = sg.offset; j < sg.offset + sg.length && j < dims; ++j)
				{
					dot += static_cast<double>(q.F32()[j]) * row[j];
					rowNorm += row[j] * row[j];
				}
				const double cosine =
					rowNorm > 0.0 ? dot / std::sqrt(rowNorm) : 0.0;
				total += static_cast<double>(sg.weight) * cosine;
			}
			ref.push_back({r, total});
		}
		std::sort(ref.begin(), ref.end(), [&](const RefHit& a, const RefHit& b) {
			return RefBetter(a, b, Metric::Cosine);
		});
		CheckTopK(bank, ref, got.data(), gotCount, k);

		// Per-channel cosines are true cosines: score of a single unit-weight
		// channel segment lies in [-1, 1] (+ float slack).
		{
			const QuerySegment one[1] = {{0, half, 1.0f}};
			QueryParams op;
			op.k = count;
			op.segments = one;
			op.segmentCount = 1;
			Workspace wsAll;
			std::vector<Hit> all(static_cast<size_t>(count));
			int32_t n = 0;
			CHECK(Query(view, q.F32(), op, wsAll, all.data(), &n) == Status::Ok);
			for (int32_t i = 0; i < n; ++i)
			{
				CHECK_MSG(all[i].score >= -1.001f && all[i].score <= 1.001f,
					"channel cosine out of range: %g", all[i].score);
			}
		}

		// Decomposition: contributions sum bit-exactly to the same total the scan
		// produced for that row (same machinery, same order).
		{
			float contributions[2] = {0.0f, 0.0f};
			const int32_t hitRow = got[0].index;
			const float total = DecomposeRowScore(view, q.F32(), hitRow, segs, 2,
				contributions);
			CHECK(total == got[0].score);
			CHECK((contributions[0] + contributions[1]) == total);
		}

		// Metric override composes: ScoreAs::Dot on the channel bank folds to raw
		// projection - a different, valid ranking (bit-identical to a channel-less
		// view scored the same way).
		{
			QueryParams po = sp;
			po.scoreAs = ScoreAs::Dot;
			Workspace wsO;
			std::vector<Hit> proj(static_cast<size_t>(k));
			int32_t nProj = 0;
			CHECK(Query(view, q.F32(), po, wsO, proj.data(), &nProj) == Status::Ok);
			BankView bare = view;
			bare.channels = nullptr;
			bare.channelCount = 0;
			bare.channelInvNorms = nullptr;
			Workspace wsB;
			std::vector<Hit> bareHits(static_cast<size_t>(k));
			int32_t nBare = 0;
			QueryParams pb = sp;
			pb.scoreAs = ScoreAs::Dot;
			CHECK(Query(bare, q.F32(), pb, wsB, bareHits.data(), &nBare) == Status::Ok);
			CHECK(nProj == nBare);
			for (int32_t i = 0; i < nProj && i < nBare; ++i)
			{
				CHECK(proj[i].index == bareHits[i].index &&
					proj[i].score == bareHits[i].score);
			}
		}
	}

	// Zero-norm row channel scores 0, never NaN: hand bank, row 1's first channel
	// all zeros.
	{
		const int32_t d = 8;
		AlignedBuf rows(2 * d * sizeof(float));
		rows.F32()[0] = 1.0f;               // row 0: energy in ch0
		rows.F32()[d + 4] = 1.0f;           // row 1: ch0 all-zero, energy in ch1
		const ChannelInfo channels[2] = {{0, 4}, {4, 4}};
		std::vector<float> invNorms(4);
		BankView v;
		v.rows = rows.ptr;
		v.count = 2;
		v.dims = d;
		v.paddedDims = d;
		v.quant = Quantization::Float32;
		v.metric = Metric::Cosine;
		v.channels = channels;
		v.channelCount = 2;
		CHECK(ComputeChannelInverseNorms(v, invNorms.data()) == Status::Ok);
		CHECK(invNorms[2] == 0.0f); // row 1, channel 0
		v.channelInvNorms = invNorms.data();

		alignas(16) float q[8] = {1.0f, 0, 0, 0, 0, 0, 0, 0};
		const QuerySegment one[1] = {{0, 4, 1.0f}};
		QueryParams p;
		p.k = 2;
		p.segments = one;
		p.segmentCount = 1;
		Workspace ws;
		Hit h[2];
		int32_t n = 0;
		CHECK(Query(v, q, p, ws, h, &n) == Status::Ok);
		for (int32_t i = 0; i < n; ++i)
		{
			CHECK(h[i].score == h[i].score); // not NaN
			if (h[i].index == 1)
			{
				CHECK(h[i].score == 0.0f); // zero-norm channel scores 0
			}
		}
	}

	// Rejections: overlapping channels; cosine channels without norms.
	{
		TestBank bank(rng, 8, dims, Quantization::Float32, Metric::Cosine);
		const ChannelInfo bad[2] = {{0, 8}, {4, 8}};
		BankView v = bank.view;
		v.channels = bad;
		v.channelCount = 2;
		std::vector<float> dummy(16, 1.0f);
		v.channelInvNorms = dummy.data();
		CHECK(ValidateBank(v) == Status::BadFormat);

		const ChannelInfo good[1] = {{0, 8}};
		v.channels = good;
		v.channelCount = 1;
		v.channelInvNorms = nullptr;
		CHECK(ValidateBank(v) == Status::BadFormat);
	}
}

// ---------------------------------------------------------------------------
// T22 — scratch banks (V2 plan section 7, T-V2-C): append/remove/snapshot/
// freeze/serialize, single-writer + lock-free-reader storm, tombstone
// semantics, zero steady-state allocation. Deterministic-given-history.

namespace
{
	// In-memory archive for the persistence seam.
	struct MemArchive
	{
		std::vector<uint8_t> bytes;
		size_t readPos = 0;

		static bool Write(void* user, const void* data, size_t n)
		{
			auto* a = static_cast<MemArchive*>(user);
			const auto* p = static_cast<const uint8_t*>(data);
			a->bytes.insert(a->bytes.end(), p, p + n);
			return true;
		}
		static bool Read(void* user, void* data, size_t n)
		{
			auto* a = static_cast<MemArchive*>(user);
			if (a->readPos + n > a->bytes.size())
			{
				return false;
			}
			std::memcpy(data, a->bytes.data() + a->readPos, n);
			a->readPos += n;
			return true;
		}
		ScratchArchive Writer() { return {&MemArchive::Write, nullptr, this}; }
		ScratchArchive Reader() { return {nullptr, &MemArchive::Read, this}; }
	};
} // namespace

static void TestScratchBanks()
{
	Rng rng(0x5C247C4Bull);
	const int32_t dims = 48;
	const int32_t count = 400;

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			// Source rows; Cosine sources keep a non-zero norm by construction of
			// NextFloat (exact all-zero rows are practically impossible; asserted).
			std::vector<float> source(static_cast<size_t>(count) * dims);
			for (auto& v : source)
			{
				v = rng.NextFloat();
			}

			ScratchBank scratch;
			CHECK(scratch.Create(count, dims, metric, quant) == Status::Ok);
			for (int32_t r = 0; r < count; ++r)
			{
				int32_t index = -1;
				CHECK(scratch.Append(source.data() + static_cast<size_t>(r) * dims,
						dims, &index) == Status::Ok);
				CHECK(index == r);
			}
			CHECK(scratch.Count() == count);
			CHECK(scratch.Append(source.data(), dims, nullptr) == Status::OutOfMemory);

			// Remove a deterministic scatter of rows.
			int32_t removed = 0;
			for (int32_t r = 0; r < count; r += 7)
			{
				CHECK(scratch.Remove(r) == Status::Ok);
				++removed;
			}
			CHECK(scratch.Remove(3) == Status::Ok); // idempotent-friendly re-remove
			CHECK(scratch.Remove(3) == Status::Ok);
			removed += 0; // 3 is 7-aligned? no: count it once below via LiveCount
			const bool threeWasCounted = (3 % 7) == 0;
			const int32_t expectLive = count - removed - (threeWasCounted ? 0 : 1);
			CHECK(scratch.LiveCount() == expectLive);
			CHECK(scratch.Remove(-1) == Status::InvalidArgument);
			CHECK(scratch.Remove(count) == Status::InvalidArgument);

			// Snapshot: deletion is exclusion.
			BankView snap;
			std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
			CHECK(scratch.Snapshot(&snap, tombs.data()) == Status::Ok);
			CHECK(snap.count == count);
			CHECK(ValidateBank(snap) == Status::Ok);
			CHECK(ValidateBankData(snap, nullptr) == Status::Ok);

			const int32_t pd = snap.paddedDims;
			std::vector<float> queryRaw(static_cast<size_t>(dims));
			for (auto& v : queryRaw)
			{
				v = rng.NextFloat();
			}
			AlignedBuf qbuf(static_cast<size_t>(pd) * sizeof(float));
			PadQuery(queryRaw, pd, qbuf.F32());

			QueryParams params;
			params.k = 10;
			params.excludeBits = tombs.data();
			Workspace ws;
			Hit hits[10];
			int32_t n = 0;
			CHECK(Query(snap, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
			CHECK(n == 10);
			for (int32_t i = 0; i < n; ++i)
			{
				CHECK(!IsExcluded(tombs.data(), hits[i].index));
			}

			// The segmented path runs on scratch snapshots too: degenerate list is
			// bit-identical to the plain scan (the compatibility proof on this type).
			{
				const QuerySegment degen[1] = {{0, pd, 1.0f}};
				QueryParams segParams = params;
				segParams.segments = degen;
				segParams.segmentCount = 1;
				Hit segHits[10];
				int32_t segN = 0;
				CHECK(Query(snap, qbuf.F32(), segParams, ws, segHits, &segN) == Status::Ok);
				CHECK(segN == n);
				for (int32_t i = 0; i < n; ++i)
				{
					CHECK(segHits[i].index == hits[i].index &&
						segHits[i].score == hits[i].score);
				}
			}

			// C2 — freeze ≡ snapshot ≡ equivalent imported bank, bit-identical.
			{
				const int32_t live = scratch.FreezeLiveCount();
				AlignedBuf frozenRows(static_cast<size_t>(live) * pd * ElementSize(quant));
				std::vector<float> frozenScales(
					quant == Quantization::Int8 ? static_cast<size_t>(live) : size_t{1});
				std::vector<int32_t> indexMap(static_cast<size_t>(count), -2);
				CHECK(scratch.Freeze(frozenRows.ptr,
						quant == Quantization::Int8 ? frozenScales.data() : nullptr,
						indexMap.data()) == Status::Ok);

				// The imported twin: bake the live source rows in live order with the
				// importer's own math; payload must match the frozen payload byte for
				// byte (per-row normalize/quantize is order-independent, V2-G6).
				std::vector<float> liveSource;
				for (int32_t r = 0; r < count; ++r)
				{
					if (IsExcluded(tombs.data(), r))
					{
						CHECK(indexMap[r] == -1);
						continue;
					}
					CHECK(indexMap[r] >= 0);
					liveSource.insert(liveSource.end(),
						source.begin() + static_cast<size_t>(r) * dims,
						source.begin() + static_cast<size_t>(r + 1) * dims);
				}
				if (metric == Metric::Cosine)
				{
					CHECK(NormalizeRows(liveSource.data(), live, dims, nullptr) == Status::Ok);
				}
				AlignedBuf importedRows(static_cast<size_t>(live) * pd * ElementSize(quant));
				std::vector<float> importedScales(static_cast<size_t>(live));
				if (quant == Quantization::Int8)
				{
					QuantizeRowsInt8(liveSource.data(), live, dims, pd,
						importedRows.I8(), importedScales.data());
					CHECK(std::memcmp(frozenScales.data(), importedScales.data(),
							static_cast<size_t>(live) * sizeof(float)) == 0);
				}
				else
				{
					PadRowsFloat32(liveSource.data(), live, dims, pd, importedRows.F32());
				}
				CHECK(std::memcmp(frozenRows.ptr, importedRows.ptr,
						static_cast<size_t>(live) * pd * ElementSize(quant)) == 0);

				// Query the frozen bank: hits are the snapshot's hits renumbered
				// through the map, scores bit-identical.
				BankView frozen;
				frozen.rows = frozenRows.ptr;
				frozen.scales = quant == Quantization::Int8 ? frozenScales.data() : nullptr;
				frozen.count = live;
				frozen.dims = dims;
				frozen.paddedDims = pd;
				frozen.quant = quant;
				frozen.metric = metric;
				CHECK(ValidateBank(frozen) == Status::Ok);
				QueryParams fParams;
				fParams.k = 10;
				Hit fHits[10];
				int32_t fN = 0;
				CHECK(Query(frozen, qbuf.F32(), fParams, ws, fHits, &fN) == Status::Ok);
				CHECK(fN == n);
				for (int32_t i = 0; i < n && i < fN; ++i)
				{
					CHECK(fHits[i].index == indexMap[hits[i].index]);
					CHECK(fHits[i].score == hits[i].score);
				}
			}

			// C3 — serialize round-trip: identical counts, identical answers.
			{
				MemArchive archive;
				CHECK(scratch.Save(archive.Writer()) == Status::Ok);

				ScratchBank loaded;
				CHECK(loaded.Load(archive.Reader()) == Status::Ok);
				CHECK(loaded.Count() == scratch.Count());
				CHECK(loaded.LiveCount() == scratch.LiveCount());
				BankView lsnap;
				std::vector<uint32_t> ltombs(ScratchBank::TombstoneWords(count), 0u);
				CHECK(loaded.Snapshot(&lsnap, ltombs.data()) == Status::Ok);
				CHECK(ltombs == tombs);
				QueryParams lParams;
				lParams.k = 10;
				lParams.excludeBits = ltombs.data();
				Hit lHits[10];
				int32_t lN = 0;
				CHECK(Query(lsnap, qbuf.F32(), lParams, ws, lHits, &lN) == Status::Ok);
				CHECK(lN == n);
				for (int32_t i = 0; i < n && i < lN; ++i)
				{
					CHECK(lHits[i].index == hits[i].index && lHits[i].score == hits[i].score);
				}

				// Corrupt magic → BadFormat, and the target bank is left unchanged.
				MemArchive bad;
				bad.bytes = archive.bytes;
				bad.bytes[0] ^= 0xFF;
				CHECK(loaded.Load(bad.Reader()) == Status::BadFormat);
				CHECK(loaded.Count() == scratch.Count());

				// Truncated payload → BadFormat.
				MemArchive trunc;
				trunc.bytes.assign(archive.bytes.begin(),
					archive.bytes.begin() + static_cast<long>(archive.bytes.size() / 2));
				CHECK(loaded.Load(trunc.Reader()) == Status::BadFormat);

				// A tombstone bit at/above count → BadFormat (corrupt archive).
				// count (400) is not a multiple of 32, so the last word has dead bits.
				{
					MemArchive tainted;
					tainted.bytes = archive.bytes;
					uint8_t* last = tainted.bytes.data() + tainted.bytes.size() - 1;
					*last |= 0x80; // top bit of the last tombstone word
					CHECK(loaded.Load(tainted.Reader()) == Status::BadFormat);
				}
			}

			// C4 — tombstone semantics: a snapshot taken BEFORE a remove still
			// returns the row (snapshot-consistent, not preemptive); one taken after
			// excludes it.
			{
				const int32_t victim = hits[0].index;
				BankView before;
				std::vector<uint32_t> beforeTombs(ScratchBank::TombstoneWords(count), 0u);
				CHECK(scratch.Snapshot(&before, beforeTombs.data()) == Status::Ok);
				CHECK(scratch.Remove(victim) == Status::Ok);

				QueryParams bParams;
				bParams.k = 10;
				bParams.excludeBits = beforeTombs.data();
				Hit bHits[10];
				int32_t bN = 0;
				CHECK(Query(before, qbuf.F32(), bParams, ws, bHits, &bN) == Status::Ok);
				CHECK(bN == n && bHits[0].index == victim); // pre-remove snapshot: still there

				BankView after;
				std::vector<uint32_t> afterTombs(ScratchBank::TombstoneWords(count), 0u);
				CHECK(scratch.Snapshot(&after, afterTombs.data()) == Status::Ok);
				QueryParams aParams;
				aParams.k = 10;
				aParams.excludeBits = afterTombs.data();
				Hit aHits[10];
				int32_t aN = 0;
				CHECK(Query(after, qbuf.F32(), aParams, ws, aHits, &aN) == Status::Ok);
				for (int32_t i = 0; i < aN; ++i)
				{
					CHECK(aHits[i].index != victim);
				}
			}

			// Grow preserves indices (T-044 W4): same hits, same scores, then room
			// for more rows.
			{
				BankView preGrow;
				std::vector<uint32_t> preTombs(ScratchBank::TombstoneWords(count), 0u);
				CHECK(scratch.Snapshot(&preGrow, preTombs.data()) == Status::Ok);
				QueryParams gParams;
				gParams.k = 10;
				gParams.excludeBits = preTombs.data();
				Hit preHits[10];
				int32_t preN = 0;
				CHECK(Query(preGrow, qbuf.F32(), gParams, ws, preHits, &preN) == Status::Ok);

				CHECK(scratch.Grow(count) == Status::InvalidArgument);
				CHECK(scratch.Grow(count + 32) == Status::Ok);
				CHECK(scratch.Capacity() == count + 32);
				CHECK(scratch.Count() == count);

				BankView postGrow;
				std::vector<uint32_t> postTombs(
					ScratchBank::TombstoneWords(count + 32), 0u);
				CHECK(scratch.Snapshot(&postGrow, postTombs.data()) == Status::Ok);
				QueryParams pParams;
				pParams.k = 10;
				pParams.excludeBits = postTombs.data();
				Hit postHits[10];
				int32_t postN = 0;
				CHECK(Query(postGrow, qbuf.F32(), pParams, ws, postHits, &postN) == Status::Ok);
				CHECK(postN == preN);
				for (int32_t i = 0; i < preN && i < postN; ++i)
				{
					CHECK(postHits[i].index == preHits[i].index &&
						postHits[i].score == preHits[i].score);
				}
				int32_t grown = -1;
				CHECK(scratch.Append(source.data(), dims, &grown) == Status::Ok);
				CHECK(grown == count);
			}
		}
	}

	// Append rejections.
	{
		ScratchBank bank;
		CHECK(bank.Create(4, 8, Metric::Cosine, Quantization::Float32) == Status::Ok);
		float row[8] = {1, 2, 3, 4, 5, 6, 7, 8};
		CHECK(bank.Append(row, 4, nullptr) == Status::DimsMismatch);
		float nan[8] = {};
		nan[3] = std::numeric_limits<float>::quiet_NaN();
		CHECK(bank.Append(nan, 8, nullptr) == Status::NonFiniteQuery);
		float zero[8] = {};
		CHECK(bank.Append(zero, 8, nullptr) == Status::ZeroNormRow);
		CHECK(bank.Count() == 0);
		CHECK(bank.Append(row, 8, nullptr) == Status::Ok);
		// The caller's buffer is never normalized in place.
		CHECK(row[0] == 1.0f && row[7] == 8.0f);
		CHECK(bank.Create(4, 8, Metric::Dot, Quantization::Float32) ==
			Status::InvalidArgument); // double-create
	}

	// C5 — zero allocation after arena creation: appends, removes, snapshots, and
	// warm queries allocate nothing; both counters stay flat.
	{
		const int32_t cap = 128;
		ScratchBank bank;
		CHECK(bank.Create(cap, dims, Metric::Dot, Quantization::Float32) == Status::Ok);
		std::vector<float> rows(static_cast<size_t>(cap) * dims);
		for (auto& v : rows)
		{
			v = rng.NextFloat();
		}
		const int32_t pd = PaddedDims(dims, Quantization::Float32);
		AlignedBuf qbuf(static_cast<size_t>(pd) * sizeof(float));
		std::vector<float> queryRaw(rows.begin(), rows.begin() + dims);
		PadQuery(queryRaw, pd, qbuf.F32());
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(cap), 0u);
		Workspace ws;
		CHECK(ws.Reserve(10, 1));

		// Warm one full cycle, then assert flat.
		int32_t idx = -1;
		CHECK(bank.Append(rows.data(), dims, &idx) == Status::Ok);
		const uint64_t allocsBefore = AllocationCount();
		const uint64_t growthBefore = ws.GrowthCount();
		BankView snap;
		Hit hits[10];
		int32_t n = 0;
		for (int32_t r = 1; r < cap; ++r)
		{
			CHECK(bank.Append(rows.data() + static_cast<size_t>(r) * dims, dims, &idx)
				== Status::Ok);
			if ((r & 3) == 0)
			{
				CHECK(bank.Remove(r) == Status::Ok);
			}
			CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
			QueryParams params;
			params.k = 10;
			params.excludeBits = tombs.data();
			CHECK(Query(snap, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
		}
		CHECK_MSG(AllocationCount() == allocsBefore,
			"scratch steady state allocated: %llu -> %llu",
			static_cast<unsigned long long>(allocsBefore),
			static_cast<unsigned long long>(AllocationCount()));
		CHECK(ws.GrowthCount() == growthBefore);
	}

	// C1 — storm: one writer appending and removing, lock-free readers
	// snapshotting and querying concurrently. Every reader result must equal its
	// serial twin computed from the same snapshot (deterministic given history);
	// CHECK is main-thread-only, so readers tally violations locally.
	{
		const int32_t cap = 2000;
		const int32_t stormDims = 32;
		ScratchBank bank;
		CHECK(bank.Create(cap, stormDims, Metric::Cosine, Quantization::Int8) == Status::Ok);

		std::vector<float> rows(static_cast<size_t>(cap) * stormDims);
		{
			Rng stormRng(0x570A11ull);
			for (auto& v : rows)
			{
				v = stormRng.NextFloat();
			}
		}
		const int32_t pd = PaddedDims(stormDims, Quantization::Int8);
		AlignedBuf qbuf(static_cast<size_t>(pd) * sizeof(float));
		std::vector<float> queryRaw(rows.begin(), rows.begin() + stormDims);
		PadQuery(queryRaw, pd, qbuf.F32());

		std::atomic<bool> done{false};
		std::atomic<int32_t> readersReady{0};
		std::atomic<int64_t> readerViolations{0};
		std::atomic<int64_t> readerQueries{0};

		auto readerFn = [&]() {
			Workspace myWs;
			myWs.Reserve(5, 1);
			std::vector<uint32_t> myTombs(ScratchBank::TombstoneWords(cap), 0u);
			readersReady.fetch_add(1, std::memory_order_release);
			Hit a[5], b[5];
			int64_t bad = 0, ran = 0;
			while (!done.load(std::memory_order_acquire))
			{
				BankView snap;
				if (bank.Snapshot(&snap, myTombs.data()) != Status::Ok)
				{
					++bad;
					continue;
				}
				if (snap.count == 0)
				{
					continue;
				}
				QueryParams params;
				params.k = 5;
				params.excludeBits = myTombs.data();
				int32_t na = 0, nb = 0;
				// The serial twin: the same snapshot queried twice must agree
				// bitwise — the snapshot is immutable, so any divergence means a
				// torn read of scratch state.
				if (Query(snap, qbuf.F32(), params, myWs, a, &na) != Status::Ok ||
					Query(snap, qbuf.F32(), params, myWs, b, &nb) != Status::Ok)
				{
					++bad;
					continue;
				}
				if (na != nb)
				{
					++bad;
				}
				else
				{
					for (int32_t i = 0; i < na; ++i)
					{
						if (a[i].index != b[i].index || a[i].score != b[i].score ||
							a[i].index >= snap.count ||
							IsExcluded(myTombs.data(), a[i].index))
						{
							++bad;
						}
					}
				}
				++ran;
			}
			readerViolations.fetch_add(bad, std::memory_order_relaxed);
			readerQueries.fetch_add(ran, std::memory_order_relaxed);
		};

		std::thread readers[3] = {
			std::thread(readerFn), std::thread(readerFn), std::thread(readerFn)};

		// Writer: append everything, removing a scatter as it goes. Wait for the
		// readers first so the storm actually overlaps (an unsynchronized writer
		// finishes 2000 appends before the OS even schedules the reader threads).
		while (readersReady.load(std::memory_order_acquire) < 3)
		{
			std::this_thread::yield();
		}
		Rng removeRng(0xDE1E7Eull);
		for (int32_t r = 0; r < cap; ++r)
		{
			int32_t idx = -1;
			if (bank.Append(rows.data() + static_cast<size_t>(r) * stormDims,
					stormDims, &idx) != Status::Ok ||
				idx != r)
			{
				readerViolations.fetch_add(1, std::memory_order_relaxed);
				break;
			}
			if ((removeRng.Next() & 7) == 0)
			{
				if (bank.Remove(removeRng.NextIndex(r + 1)) != Status::Ok)
				{
					readerViolations.fetch_add(1, std::memory_order_relaxed);
				}
			}
		}
		done.store(true, std::memory_order_release);
		for (auto& t : readers)
		{
			t.join();
		}
		CHECK_MSG(readerViolations.load() == 0, "storm violations: %lld",
			static_cast<long long>(readerViolations.load()));
		CHECK_MSG(readerQueries.load() > 0, "storm readers never ran a query");
		CHECK(bank.Count() == cap);

		// Post-storm: the final state equals the serial twin — a fresh bank built
		// from the same history answers bit-identically.
		{
			BankView snap;
			std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(cap), 0u);
			CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);

			ScratchBank serial;
			CHECK(serial.Create(cap, stormDims, Metric::Cosine, Quantization::Int8)
				== Status::Ok);
			Rng replayRng(0xDE1E7Eull);
			for (int32_t r = 0; r < cap; ++r)
			{
				CHECK(serial.Append(rows.data() + static_cast<size_t>(r) * stormDims,
						stormDims, nullptr) == Status::Ok);
				if ((replayRng.Next() & 7) == 0)
				{
					CHECK(serial.Remove(replayRng.NextIndex(r + 1)) == Status::Ok);
				}
			}
			BankView twin;
			std::vector<uint32_t> twinTombs(ScratchBank::TombstoneWords(cap), 0u);
			CHECK(serial.Snapshot(&twin, twinTombs.data()) == Status::Ok);
			CHECK(twinTombs == tombs);

			Workspace ws;
			QueryParams params;
			params.k = 20;
			params.excludeBits = tombs.data();
			Hit sa[20], sb[20];
			int32_t na = 0, nb = 0;
			CHECK(Query(snap, qbuf.F32(), params, ws, sa, &na) == Status::Ok);
			params.excludeBits = twinTombs.data();
			CHECK(Query(twin, qbuf.F32(), params, ws, sb, &nb) == Status::Ok);
			CHECK(na == nb);
			for (int32_t i = 0; i < na && i < nb; ++i)
			{
				CHECK(sa[i].index == sb[i].index && sa[i].score == sb[i].score);
			}
		}
	}
}

// F4 litmus — the pin/drain protocol's store-buffering shape, driven hard under
// ThreadSanitizer in CI (the x86 suite cannot distinguish the declared orderings
// from the ISA's incidental fences; TSan reasons about the C++ model). A shared
// NON-atomic variable is the tripwire: readers touch it only while pinned, the
// exclusive side only inside Begin/EndExclusive — if the protocol admitted the
// bad interleaving, that is a data race TSan reports even when the timing never
// bites, and the invariant counters catch actual overlap on any host.
static void TestPinDrainLitmus()
{
	ScratchBank bank;
	CHECK(bank.Create(8, 16, Metric::Dot, Quantization::Float32) == Status::Ok);
	alignas(16) float row[16] = {1.0f};
	CHECK(bank.Append(row, 16, nullptr) == Status::Ok);

	int guarded = 0; // deliberately non-atomic: the TSan tripwire
	std::atomic<bool> done{false};
	std::atomic<int64_t> violations{0};
	std::atomic<int64_t> pinsTaken{0};
	std::atomic<int64_t> exclusivesRun{0};

	auto readerFn = [&]() {
		while (!done.load(std::memory_order_acquire))
		{
			if (!bank.TryPinReader())
			{
				continue;
			}
			// Pinned: the exclusive side must not be inside its critical section.
			const int seen = guarded;
			if (seen != 0)
			{
				violations.fetch_add(1, std::memory_order_relaxed);
			}
			pinsTaken.fetch_add(1, std::memory_order_relaxed);
			bank.UnpinReader();
		}
	};
	std::thread readers[3] = {
		std::thread(readerFn), std::thread(readerFn), std::thread(readerFn)};

	for (int i = 0; i < 20000; ++i)
	{
		if (!bank.BeginExclusive())
		{
			violations.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		guarded = 1;
		guarded = 0;
		bank.EndExclusive();
		exclusivesRun.fetch_add(1, std::memory_order_relaxed);
	}
	done.store(true, std::memory_order_release);
	for (auto& t : readers)
	{
		t.join();
	}
	CHECK_MSG(violations.load() == 0, "pin/drain litmus violations: %lld",
		static_cast<long long>(violations.load()));
	CHECK(exclusivesRun.load() == 20000);
	CHECK_MSG(pinsTaken.load() > 0, "litmus readers never pinned");
}

int main()
{
	TestSimdEqualsScalar();
	TestMergeTopK();
	TestValidateBankData();
	TestKnownGeometry();
	TestRandomizedAgainstReference();
	TestTieBreak();
	TestEdges();
	TestValidation();
	TestBake();
	TestExclusion();
	TestAllocationFlat();
	TestBatchEquivalence();
	TestRepeatDeterminism();
	TestCentroid();
	TestDirection();
	TestScoreAsOverride();
	TestMargin();
	TestIntersect();
	TestPca();
	TestSegmentedScan();
	TestPerChannelCosine();
	TestScratchBanks();
	TestPinDrainLitmus();

	std::printf("superfaiss tests: %d checks, %d failures (simd path: %s)\n",
		GChecks, GFailures,
		ActiveSimdPath() == SimdPath::Scalar ? "scalar"
			: ActiveSimdPath() == SimdPath::SSE ? "sse"
			: ActiveSimdPath() == SimdPath::AVX2 ? "avx2" : "neon");
	return GFailures == 0 ? 0 : 1;
}

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS // getenv/fopen in the selection-recording emitter
#endif

// SuperFAISS test harness. Standard library only; no third-party test framework.
// Reference results are computed in double precision with the same total order
// (score, then ascending index); float-vs-double near-ties are handled with an
// epsilon boundary check rather than exact rank equality.

#include "superfaiss/superfaiss.h"

#include "xd_fixtures.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
	std::atomic<int32_t> readersReady{0};
	std::atomic<int64_t> violations{0};
	std::atomic<int64_t> pinsTaken{0};
	std::atomic<int64_t> exclusivesRun{0};

	auto readerFn = [&]() {
		readersReady.fetch_add(1, std::memory_order_release);
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

	// Start barrier + periodic yield: without them the writer can run all its
	// iterations before the OS schedules a single reader (the same failure mode
	// the storm test hit), and the liveness check below trips spuriously — seen
	// once on the macos-arm64 runner. Correctness never depended on timing;
	// liveness assertions must not either.
	while (readersReady.load(std::memory_order_acquire) < 3)
	{
		std::this_thread::yield();
	}
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
		if ((i & 63) == 0)
		{
			std::this_thread::yield(); // give readers pin windows on small runners
		}
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

// ---------------------------------------------------------------------------
// T24 — per-row bias (v2.1, plan section 18): both forms against a float64
// composed reference, null bitwise identity, all-zeros compare-equal, sparse ==
// dense-equivalent, eviction and lift exactness (the k+P construction), metric
// direction, the one-add contract, rejections, batch ≡ singles, intersection,
// segment composition, exclusion-wins, allocation flatness.

static void TestPerRowBias()
{
	Rng rng(0xB1A5ull);
	const int32_t dims = 24;
	const int32_t count = 500;
	const int32_t k = 10;

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			TestBank bank(rng, count, dims, quant, metric);
			std::vector<float> queryRaw(dims);
			for (auto& v : queryRaw)
			{
				v = rng.NextFloat();
			}
			AlignedBuf qbuf(static_cast<size_t>(bank.view.paddedDims) * sizeof(float));
			PadQuery(queryRaw, bank.view.paddedDims, qbuf.F32());

			std::vector<float> dense(count);
			for (auto& b : dense)
			{
				b = rng.NextFloat() * 0.25f;
			}

			Workspace ws;
			Hit plain[k], hits[k];
			int32_t nPlain = 0, n = 0;
			QueryParams params;
			params.k = k;
			CHECK(Query(bank.view, qbuf.F32(), params, ws, plain, &nPlain) == Status::Ok);

			// Null / empty RowBias: the bit-identical unbiased path.
			{
				RowBias empty;
				params.bias = &empty;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
				CHECK(n == nPlain);
				for (int32_t i = 0; i < n; ++i)
				{
					CHECK(hits[i].index == plain[i].index && hits[i].score == plain[i].score);
				}
				params.bias = nullptr;
			}

			// Dense against the float64 composed reference.
			{
				RowBias rb;
				rb.dense = dense.data();
				params.bias = &rb;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
				std::vector<RefHit> ref = ReferenceScan(bank, qbuf.F32(), nullptr);
				for (RefHit& h : ref)
				{
					h.score += static_cast<double>(dense[h.index]);
				}
				std::sort(ref.begin(), ref.end(), [&](const RefHit& a, const RefHit& b) {
					return RefBetter(a, b, metric);
				});
				CheckTopK(bank, ref, hits, n, k);

				// The one-add contract: composed == unbiased + bias, bitwise, for
				// every returned hit (the unbiased term from an exhaustive scan).
				std::vector<Hit> all(count);
				std::vector<int32_t> nAll(1);
				QueryParams full;
				full.k = count;
				Workspace wsFull;
				int32_t fullN = 0;
				CHECK(Query(bank.view, qbuf.F32(), full, wsFull, all.data(), &fullN)
					== Status::Ok);
				for (int32_t i = 0; i < n; ++i)
				{
					float unbiased = 0.0f;
					for (int32_t j = 0; j < fullN; ++j)
					{
						if (all[j].index == hits[i].index)
						{
							unbiased = all[j].score;
							break;
						}
					}
					CHECK(hits[i].score == unbiased + dense[hits[i].index]);
				}
				(void)nAll;
				params.bias = nullptr;
			}

			// All-zeros dense: compare-equal ranking identity (NOT claimed bitwise:
			// IEEE -0.0 + 0.0 == +0.0).
			{
				std::vector<float> zeros(count, 0.0f);
				RowBias rb;
				rb.dense = zeros.data();
				params.bias = &rb;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
				CHECK(n == nPlain);
				for (int32_t i = 0; i < n; ++i)
				{
					CHECK(hits[i].index == plain[i].index);
					CHECK(hits[i].score == plain[i].score); // value equality
				}
				params.bias = nullptr;
			}

			// Sparse: lift a bottom row in, evict the top row, and match the
			// dense-equivalent expression compare-equal.
			{
				// The unbiased worst-ranked live row (from an exhaustive scan).
				Workspace wsFull;
				std::vector<Hit> all(count);
				QueryParams full;
				full.k = count;
				int32_t fullN = 0;
				CHECK(Query(bank.view, qbuf.F32(), full, wsFull, all.data(), &fullN)
					== Status::Ok);
				const int32_t bottomRow = all[fullN - 1].index;
				const int32_t topRow = all[0].index;
				// Rewards improve in the metric's own direction: negative on L2.
				const float lift = metric == Metric::L2 ? -1000.0f : 1000.0f;
				const float evict = metric == Metric::L2 ? 1000.0f : -1000.0f;
				BiasPair pairs[2] = {{bottomRow, lift}, {topRow, evict}};
				RowBias rb;
				rb.pairs = pairs;
				rb.pairCount = 2;
				params.bias = &rb;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
				CHECK(n == k);
				CHECK(hits[0].index == bottomRow); // lifted from rank-last to rank-1
				for (int32_t i = 0; i < n; ++i)
				{
					CHECK(hits[i].index != topRow); // evicted despite rank-1 similarity
				}

				// Dense-equivalent: zeros everywhere except the pair rows.
				std::vector<float> equivalent(count, 0.0f);
				equivalent[bottomRow] = lift;
				equivalent[topRow] = evict;
				RowBias rbDense;
				rbDense.dense = equivalent.data();
				params.bias = &rbDense;
				Hit denseHits[k];
				int32_t dn = 0;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, denseHits, &dn) == Status::Ok);
				CHECK(dn == n);
				for (int32_t i = 0; i < n; ++i)
				{
					CHECK(denseHits[i].index == hits[i].index);
					CHECK(denseHits[i].score == hits[i].score); // value equality
				}

				// Exclusion wins over bias: the lifted row, excluded, never returns.
				std::vector<uint32_t> exclude((count + 31) / 32, 0u);
				exclude[bottomRow >> 5] |= 1u << (bottomRow & 31);
				params.bias = &rb;
				params.excludeBits = exclude.data();
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
				for (int32_t i = 0; i < n; ++i)
				{
					CHECK(hits[i].index != bottomRow);
				}
				params.excludeBits = nullptr;
				params.bias = nullptr;
			}

			// Rejections.
			{
				std::vector<float> bad(dense);
				bad[count / 2] = std::numeric_limits<float>::quiet_NaN();
				RowBias rb;
				rb.dense = bad.data();
				params.bias = &rb;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n)
					== Status::NonFiniteQuery);

				BiasPair nanPair[1] = {{0, std::numeric_limits<float>::infinity()}};
				RowBias rbNan;
				rbNan.pairs = nanPair;
				rbNan.pairCount = 1;
				params.bias = &rbNan;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n)
					== Status::NonFiniteQuery);

				BiasPair oor[1] = {{count, 1.0f}};
				RowBias rbOor;
				rbOor.pairs = oor;
				rbOor.pairCount = 1;
				params.bias = &rbOor;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n)
					== Status::InvalidArgument);

				BiasPair dup[2] = {{3, 1.0f}, {3, 2.0f}};
				RowBias rbDup;
				rbDup.pairs = dup;
				rbDup.pairCount = 2;
				params.bias = &rbDup;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n)
					== Status::InvalidArgument);

				BiasPair one[1] = {{0, 1.0f}};
				RowBias rbBoth;
				rbBoth.dense = dense.data();
				rbBoth.pairs = one;
				rbBoth.pairCount = 1;
				params.bias = &rbBoth;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n)
					== Status::InvalidArgument);

				RowBias rbNull;
				rbNull.pairCount = 1;
				params.bias = &rbNull;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n)
					== Status::InvalidArgument);
				params.bias = nullptr;
			}

			// Batch ≡ singles, per-query forms mixed (none / dense / sparse).
			{
				const int32_t m = 6;
				std::vector<float> queries(static_cast<size_t>(m) * bank.view.paddedDims, 0.0f);
				AlignedBuf qsBuf(queries.size() * sizeof(float));
				for (int32_t q = 0; q < m; ++q)
				{
					std::vector<float> raw(dims);
					for (auto& v : raw)
					{
						v = rng.NextFloat();
					}
					PadQuery(raw, bank.view.paddedDims,
						qsBuf.F32() + static_cast<int64_t>(q) * bank.view.paddedDims);
				}
				BiasPair pair1[1] = {{7, metric == Metric::L2 ? -50.0f : 50.0f}};
				BiasPair pair2[2] = {{11, 2.0f}, {200, metric == Metric::L2 ? -75.0f : 75.0f}};
				RowBias forms[m];
				forms[1].dense = dense.data();
				forms[2].pairs = pair1;
				forms[2].pairCount = 1;
				forms[3].dense = dense.data();
				forms[4].pairs = pair2;
				forms[4].pairCount = 2;
				QueryParams bp;
				bp.k = k;
				bp.bias = forms;
				std::vector<Hit> batchHits(static_cast<size_t>(m) * k);
				std::vector<int32_t> counts(m);
				Workspace wsB;
				CHECK(QueryBatch(bank.view, qsBuf.F32(), m, bp, wsB, batchHits.data(),
						counts.data()) == Status::Ok);
				for (int32_t q = 0; q < m; ++q)
				{
					QueryParams sp;
					sp.k = k;
					sp.bias = &forms[q];
					Hit single[k];
					int32_t sn = 0;
					Workspace wsS;
					CHECK(Query(bank.view,
							qsBuf.F32() + static_cast<int64_t>(q) * bank.view.paddedDims,
							sp, wsS, single, &sn) == Status::Ok);
					CHECK(sn == counts[q]);
					for (int32_t i = 0; i < sn; ++i)
					{
						const Hit& b = batchHits[static_cast<size_t>(q) * k + i];
						CHECK(b.index == single[i].index && b.score == single[i].score);
					}
				}
			}
		}
	}

	// Intersection: bias applies once, to the fused score. Sparse reward lifts a
	// row into the fused top-k; the composed score equals fused + bias bitwise.
	{
		TestBank bank(rng, 300, dims, Quantization::Float32, Metric::Cosine);
		const int32_t m = 2;
		AlignedBuf qsBuf(static_cast<size_t>(m) * bank.view.paddedDims * sizeof(float));
		for (int32_t q = 0; q < m; ++q)
		{
			std::vector<float> raw(dims);
			for (auto& v : raw)
			{
				v = rng.NextFloat();
			}
			PadQuery(raw, bank.view.paddedDims,
				qsBuf.F32() + static_cast<int64_t>(q) * bank.view.paddedDims);
		}
		QueryParams params;
		params.k = 8;
		Workspace ws;
		Hit plain[8], hits[8];
		int32_t nPlain = 0, n = 0;
		CHECK(QueryIntersect(bank.view, qsBuf.F32(), m, params, ws, plain, &nPlain)
			== Status::Ok);

		// Fused scan at k=count to find the fused-worst row and its fused score.
		std::vector<Hit> all(300);
		QueryParams full;
		full.k = 300;
		int32_t fullN = 0;
		Workspace wsF;
		CHECK(QueryIntersect(bank.view, qsBuf.F32(), m, full, wsF, all.data(), &fullN)
			== Status::Ok);
		const Hit worst = all[fullN - 1];

		BiasPair pair[1] = {{worst.index, 100.0f}};
		RowBias rb;
		rb.pairs = pair;
		rb.pairCount = 1;
		params.bias = &rb;
		CHECK(QueryIntersect(bank.view, qsBuf.F32(), m, params, ws, hits, &n) == Status::Ok);
		CHECK(n == 8 && hits[0].index == worst.index);
		CHECK(hits[0].score == worst.score + 100.0f); // once, to the fused score

		// Dense on intersect: composed reference over the exhaustive fused scan.
		std::vector<float> dense(300);
		Rng biasRng(0xB1A5B1A5ull);
		for (auto& b : dense)
		{
			b = biasRng.NextFloat() * 0.25f;
		}
		RowBias rbDense;
		rbDense.dense = dense.data();
		params.bias = &rbDense;
		CHECK(QueryIntersect(bank.view, qsBuf.F32(), m, params, ws, hits, &n) == Status::Ok);
		for (int32_t i = 0; i < n; ++i)
		{
			float fused = 0.0f;
			for (int32_t j = 0; j < fullN; ++j)
			{
				if (all[j].index == hits[i].index)
				{
					fused = all[j].score;
					break;
				}
			}
			CHECK(hits[i].score == fused + dense[hits[i].index]);
		}
	}

	// Segments x bias compose: on an L2 bank (the dense segmented path) a sparse
	// reward returns the pair row with score == DecomposeRowScore total + bias,
	// bitwise - the decomposition-term contract (partials + bias = total).
	{
		TestBank bank(rng, 200, 32, Quantization::Float32, Metric::L2);
		std::vector<float> raw(32);
		for (auto& v : raw)
		{
			v = rng.NextFloat();
		}
		AlignedBuf qbuf(static_cast<size_t>(bank.view.paddedDims) * sizeof(float));
		PadQuery(raw, bank.view.paddedDims, qbuf.F32());
		const QuerySegment segs[2] = {{0, 16, 1.0f}, {16, 16, 0.5f}};
		BiasPair pair[1] = {{123, -500.0f}}; // L2 reward is negative
		RowBias rb;
		rb.pairs = pair;
		rb.pairCount = 1;
		QueryParams params;
		params.k = 5;
		params.segments = segs;
		params.segmentCount = 2;
		params.bias = &rb;
		Workspace ws;
		Hit hits[5];
		int32_t n = 0;
		CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
		CHECK(n == 5 && hits[0].index == 123);
		float contributions[2] = {0.0f, 0.0f};
		const float total = DecomposeRowScore(bank.view, qbuf.F32(), 123, segs, 2,
			contributions);
		CHECK(hits[0].score == total + (-500.0f));
	}

	// Allocation flatness: warm one biased query of each form, then repeats
	// allocate nothing.
	{
		TestBank bank(rng, 400, dims, Quantization::Int8, Metric::Cosine);
		std::vector<float> raw(dims);
		for (auto& v : raw)
		{
			v = rng.NextFloat();
		}
		AlignedBuf qbuf(static_cast<size_t>(bank.view.paddedDims) * sizeof(float));
		PadQuery(raw, bank.view.paddedDims, qbuf.F32());
		std::vector<float> dense(400, 0.125f);
		BiasPair pair[1] = {{42, 0.5f}};
		Workspace ws;
		Hit hits[8];
		int32_t n = 0;
		QueryParams params;
		params.k = 8;
		RowBias rbDense;
		rbDense.dense = dense.data();
		RowBias rbSparse;
		rbSparse.pairs = pair;
		rbSparse.pairCount = 1;
		params.bias = &rbDense;
		CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
		params.bias = &rbSparse;
		CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
		const uint64_t allocs = AllocationCount();
		const uint64_t growth = ws.GrowthCount();
		for (int32_t i = 0; i < 32; ++i)
		{
			params.bias = (i & 1) ? &rbDense : &rbSparse;
			CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
		}
		CHECK(AllocationCount() == allocs);
		CHECK(ws.GrowthCount() == growth);
	}
}

// ---------------------------------------------------------------------------
// T25 — cross-device exactness (V2 plan section 19, T-V2-X).
//
// The claim under proof: Exactness::CrossDevice produces bit-identical scores
// and hit ORDER on any machine at any SIMD width. Within one process the proof
// is the forced-path sweep (scalar / SSE4.1 / AVX2 on x86, scalar / NEON on
// ARM — 19.4 W1); across machines it is the pinned golden hash over the
// COMMITTED fixture bytes in xd_fixtures.h (19.4 N2), asserted by every CI
// runner. The adversarial fixtures put epilogue products into and around the
// float subnormal window, so the arbiter rules on the FTZ/DAZ weak case (S1).

namespace xd
{
	// FNV-1a 64 over the battery's hits: per query, the count then each hit's
	// index and score bits in rank order — hit ORDER is hashed, which is what a
	// lockstep consumer consumes.
	struct BatteryHash
	{
		uint64_t h = 1469598103934665603ull;

		void Bytes(const void* p, size_t n)
		{
			const uint8_t* b = static_cast<const uint8_t*>(p);
			for (size_t i = 0; i < n; ++i)
			{
				h = (h ^ b[i]) * 1099511628211ull;
			}
		}

		void U32(uint32_t v) { Bytes(&v, 4); }

		void Hits(const Hit* hits, int32_t count)
		{
			U32(static_cast<uint32_t>(count));
			for (int32_t i = 0; i < count; ++i)
			{
				U32(static_cast<uint32_t>(hits[i].index));
				uint32_t bits;
				std::memcpy(&bits, &hits[i].score, 4);
				U32(bits);
			}
		}
	};

	// The subnormal-floor contract, asserted on every battery hit.
	inline void CheckFloor(const Hit* hits, int32_t count)
	{
		for (int32_t i = 0; i < count; ++i)
		{
			const float s = hits[i].score;
			CHECK_MSG(s == 0.0f || std::fabs(s) >= 1.1754943508222875e-38f,
				"subnormal score leaked: %.9g (index %d)", s, hits[i].index);
		}
	}

	// A fixture-backed bank view. Rows point straight at the committed bytes
	// (alignas(16) in the header); scales decode from committed float bits.
	struct FixBank
	{
		std::vector<float> scales;
		std::vector<ChannelInfo> channels;
		std::vector<float> invNorms;
		BankView view;

		FixBank(const int8_t* rows, const uint32_t* scaleBits, int32_t count,
			int32_t dims, Metric metric, int32_t channelCount = 0)
		{
			scales.resize(static_cast<size_t>(count));
			for (int32_t r = 0; r < count; ++r)
			{
				std::memcpy(&scales[static_cast<size_t>(r)], &scaleBits[r], 4);
			}
			view.rows = rows;
			view.scales = scales.data();
			view.count = count;
			view.dims = dims;
			view.paddedDims = PaddedDims(dims, Quantization::Int8);
			view.quant = Quantization::Int8;
			view.metric = metric;
			if (channelCount > 0)
			{
				const int32_t len = dims / channelCount;
				channels.resize(static_cast<size_t>(channelCount));
				for (int32_t c = 0; c < channelCount; ++c)
				{
					channels[static_cast<size_t>(c)] = {c * len, len};
				}
				view.channels = channels.data();
				view.channelCount = channelCount;
				invNorms.resize(static_cast<size_t>(count) * channelCount);
				CHECK(ComputeChannelInverseNorms(view, invNorms.data()) == Status::Ok);
				view.channelInvNorms = invNorms.data();
			}
			CHECK(ValidateBank(view) == Status::Ok);
		}
	};

	// Fixture query `q`, sliced to `pd` elements with pads zeroed, into `out`.
	inline void LoadQuery(int32_t q, int32_t pd, float* out)
	{
		std::memset(out, 0, static_cast<size_t>(pd) * sizeof(float));
		const int32_t n = pd < xdfix::kQueryDims ? pd : xdfix::kQueryDims;
		for (int32_t i = 0; i < n; ++i)
		{
			std::memcpy(&out[i], &xdfix::kQueryBits[q * xdfix::kQueryDims + i], 4);
		}
	}

	// Runs the full committed-fixture battery under the CURRENT dispatch and
	// returns its hash. Every hit is floor-checked. When `record` is non-null,
	// each hit list is appended as text (the ARM selection recording).
	inline uint64_t RunBattery(Workspace& ws, std::FILE* record)
	{
		BatteryHash hash;
		FixBank bankA(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
			xdfix::kBankADims, Metric::Dot);
		FixBank bankB(xdfix::kBankBRows, xdfix::kBankBScaleBits, xdfix::kBankBCount,
			xdfix::kBankBDims, Metric::Cosine, xdfix::kBankBChannels);
		FixBank bankC(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
			xdfix::kBankCDims, Metric::L2);
		FixBank bankD(xdfix::kBankDRows, xdfix::kBankDScaleBits, xdfix::kBankDCount,
			xdfix::kBankDDims, Metric::Dot);
		FixBank bankE(xdfix::kBankERows, xdfix::kBankEScaleBits, xdfix::kBankECount,
			xdfix::kBankEDims, Metric::L2);

		AlignedBuf queryBuf(sizeof(float) * 64 * 8);
		Hit hits[64 * 8];
		int32_t counts[8];

		QueryParams xdParams;
		xdParams.exactness = Exactness::CrossDevice;

		auto emit = [&](const char* label, const Hit* h, int32_t n)
		{
			CheckFloor(h, n);
			hash.Hits(h, n);
			if (record != nullptr)
			{
				std::fprintf(record, "%s n=%d:", label, n);
				for (int32_t i = 0; i < n; ++i)
				{
					uint32_t bits;
					std::memcpy(&bits, &h[i].score, 4);
					std::fprintf(record, " %d:%08x", h[i].index, bits);
				}
				std::fprintf(record, "\n");
			}
		};

		// --- Bank A (Dot): singles, exclusion, dense bias (subnormal values
		// included), sparse bias, batch, intersect.
		{
			std::vector<uint32_t> exclude((bankA.view.count + 31) / 32, 0u);
			for (int32_t r = 0; r < bankA.view.count; r += 3)
			{
				exclude[r >> 5] |= 1u << (r & 31);
			}
			std::vector<float> dense(static_cast<size_t>(bankA.view.count));
			for (int32_t r = 0; r < bankA.view.count; ++r)
			{
				// Mix of ordinary, tiny-normal, and SUBNORMAL bias values — the
				// DAZ-proof bias decode is part of the proof surface.
				const uint32_t bits = (r % 4 == 0) ? 0x00000007u          // subnormal
					: (r % 4 == 1) ? 0x80000123u                          // -subnormal
					: (r % 4 == 2) ? 0x3c23d70au                          // 0.01f
					: 0x00800000u;                                        // FLT_MIN
				std::memcpy(&dense[static_cast<size_t>(r)], &bits, 4);
			}
			const BiasPair pairs[2] = {{5, 0.25f}, {40, -0.5f}};
			RowBias denseBias;
			denseBias.dense = dense.data();
			RowBias pairBias;
			pairBias.pairs = pairs;
			pairBias.pairCount = 2;

			for (int32_t q = 0; q < xdfix::kQueryCount; ++q)
			{
				LoadQuery(q, bankA.view.paddedDims, queryBuf.F32());
				QueryParams p = xdParams;
				p.k = 10;
				CHECK(Query(bankA.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("A.single", hits, counts[0]);

				p.excludeBits = exclude.data();
				CHECK(Query(bankA.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				for (int32_t i = 0; i < counts[0]; ++i)
				{
					CHECK(hits[i].index % 3 != 0); // the exclusion mask held
				}
				emit("A.excluded", hits, counts[0]);

				p.excludeBits = nullptr;
				p.bias = &denseBias;
				CHECK(Query(bankA.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("A.dense-bias", hits, counts[0]);

				p.bias = &pairBias;
				CHECK(Query(bankA.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("A.sparse-bias", hits, counts[0]);
			}

			// Batch of 5 == the same 5 singles, bitwise.
			AlignedBuf batchBuf(sizeof(float) * 5 * bankA.view.paddedDims);
			for (int32_t q = 0; q < 5; ++q)
			{
				LoadQuery(q, bankA.view.paddedDims,
					batchBuf.F32() + static_cast<int64_t>(q) * bankA.view.paddedDims);
			}
			QueryParams p = xdParams;
			p.k = 8;
			CHECK(QueryBatch(bankA.view, batchBuf.F32(), 5, p, ws, hits, counts) == Status::Ok);
			for (int32_t q = 0; q < 5; ++q)
			{
				emit("A.batch", hits + static_cast<int64_t>(q) * p.k, counts[q]);
				Hit single[8];
				int32_t singleCount = 0;
				CHECK(Query(bankA.view,
					batchBuf.F32() + static_cast<int64_t>(q) * bankA.view.paddedDims,
					p, ws, single, &singleCount) == Status::Ok);
				CHECK(singleCount == counts[q]);
				for (int32_t i = 0; i < singleCount; ++i)
				{
					const Hit& b = hits[static_cast<int64_t>(q) * p.k + i];
					CHECK(b.index == single[i].index);
					uint32_t bb, sb;
					std::memcpy(&bb, &b.score, 4);
					std::memcpy(&sb, &single[i].score, 4);
					CHECK_MSG(bb == sb, "batch != single bits: %08x vs %08x", bb, sb);
				}
			}

			// Intersection over 3 members; queryCount == 1 degenerates to Query.
			int32_t n = 0;
			CHECK(QueryIntersect(bankA.view, batchBuf.F32(), 3, p, ws, hits, &n) == Status::Ok);
			emit("A.intersect", hits, n);
			Hit single[8];
			int32_t singleCount = 0;
			CHECK(QueryIntersect(bankA.view, batchBuf.F32(), 1, p, ws, hits, &n) == Status::Ok);
			CHECK(Query(bankA.view, batchBuf.F32(), p, ws, single, &singleCount) == Status::Ok);
			CHECK(n == singleCount);
			for (int32_t i = 0; i < n; ++i)
			{
				uint32_t ib, sb;
				std::memcpy(&ib, &hits[i].score, 4);
				std::memcpy(&sb, &single[i].score, 4);
				CHECK(hits[i].index == single[i].index && ib == sb);
			}
		}

		// --- Bank B (Cosine + channels): whole-row, channel-matched segments,
		// ScoreAs override, decomposition.
		{
			const QuerySegment segs[3] = {
				{bankB.channels[0].offset, bankB.channels[0].length, 1.5f},
				{bankB.channels[1].offset, bankB.channels[1].length, 0.0f}, // omitted
				{bankB.channels[3].offset, bankB.channels[3].length, -0.75f},
			};
			for (int32_t q = 0; q < xdfix::kQueryCount; ++q)
			{
				LoadQuery(q, bankB.view.paddedDims, queryBuf.F32());
				QueryParams p = xdParams;
				p.k = 10;
				CHECK(Query(bankB.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("B.single", hits, counts[0]);

				p.segments = segs;
				p.segmentCount = 3;
				CHECK(Query(bankB.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("B.segmented", hits, counts[0]);

				// Decomposition of the top hit: contributions are floored floats,
				// total is the scan score bitwise.
				if (counts[0] > 0)
				{
					AlignedBuf q8(static_cast<size_t>(bankB.view.paddedDims));
					double scale = 0.0;
					int64_t sq = 0;
					QuantizeQueryXd(queryBuf.F32(), bankB.view.paddedDims, q8.I8(), &scale, &sq);
					XdQuery xq{q8.I8(), scale, sq};
					float contributions[kMaxSegments];
					const float total = DecomposeRowScoreXd(bankB.view, xq,
						hits[0].index, segs, 3, contributions);
					uint32_t tb, sb;
					std::memcpy(&tb, &total, 4);
					std::memcpy(&sb, &hits[0].score, 4);
					CHECK_MSG(tb == sb, "decompose total != scan score: %08x vs %08x", tb, sb);
					CHECK(contributions[1] == 0.0f); // weight-0 contributes exactly 0
					for (int32_t s = 0; s < 3; ++s)
					{
						const float c = contributions[s];
						CHECK(c == 0.0f || std::fabs(c) >= 1.1754943508222875e-38f);
					}
					hash.Bytes(contributions, sizeof(float) * 3);
				}

				p.segments = nullptr;
				p.segmentCount = 0;
				p.scoreAs = ScoreAs::Dot;
				CHECK(Query(bankB.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("B.score-as-dot", hits, counts[0]);
			}
		}

		// --- Bank C (L2): whole-row and segmented (the expanded-epilogue path).
		{
			const QuerySegment segs[2] = {{0, 16, 1.0f}, {16, 16, 2.0f}};
			for (int32_t q = 0; q < xdfix::kQueryCount; ++q)
			{
				LoadQuery(q, bankC.view.paddedDims, queryBuf.F32());
				QueryParams p = xdParams;
				p.k = 10;
				CHECK(Query(bankC.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("C.single", hits, counts[0]);

				p.segments = segs;
				p.segmentCount = 2;
				CHECK(Query(bankC.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("C.segmented", hits, counts[0]);
			}
		}

		// --- Banks D and E (adversarial): full k so every row's score is pinned.
		{
			int32_t floored = 0;
			int32_t unfloored = 0;
			for (int32_t q = 0; q < xdfix::kQueryCount; ++q)
			{
				LoadQuery(q, bankD.view.paddedDims, queryBuf.F32());
				QueryParams p = xdParams;
				p.k = bankD.view.count;
				CHECK(Query(bankD.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("D.single", hits, counts[0]);
				for (int32_t i = 0; i < counts[0]; ++i)
				{
					(hits[i].score == 0.0f ? floored : unfloored) += 1;
				}
			}
			// The adversarial fixture must actually exercise the flush window:
			// both floored and unfloored scores present, or the arbiter is
			// ruling around the weak case.
			CHECK_MSG(floored > 0, "bank D produced no floored scores");
			CHECK_MSG(unfloored > 0, "bank D produced only floored scores");

			for (int32_t q = 0; q < xdfix::kQueryCount; ++q)
			{
				LoadQuery(q, bankE.view.paddedDims, queryBuf.F32());
				QueryParams p = xdParams;
				p.k = bankE.view.count;
				CHECK(Query(bankE.view, queryBuf.F32(), p, ws, hits, &counts[0]) == Status::Ok);
				emit("E.single", hits, counts[0]);
			}
		}

		return hash.h;
	}

	// Independent test-side reimplementation of the whole-row CrossDevice score,
	// coded from the 19.2/19.3 contract (not from the kernel source): plain
	// integer loops, the fixed-order double epilogue, the floor.
	inline float RefXdScore(const BankView& bank, const int8_t* q8, double qs,
		int64_t qSq, int32_t r)
	{
		const int8_t* row = static_cast<const int8_t*>(bank.rows) +
			static_cast<int64_t>(r) * bank.paddedDims;
		double rsD;
		{
			uint32_t b;
			std::memcpy(&b, &bank.scales[r], 4);
			if ((b & 0x7f800000u) != 0)
			{
				rsD = static_cast<double>(bank.scales[r]);
			}
			else
			{
				const double m =
					static_cast<double>(static_cast<int32_t>(b & 0x7fffffu)) *
					1.4012984643248171e-45; // 2^-149
				rsD = (b >> 31) != 0 ? -m : m;
			}
		}
		double d;
		if (bank.metric == Metric::L2)
		{
			int64_t cross = 0;
			int64_t rowSq = 0;
			for (int32_t i = 0; i < bank.paddedDims; ++i)
			{
				cross += static_cast<int64_t>(row[i]) * q8[i];
				rowSq += static_cast<int64_t>(row[i]) * row[i];
			}
			const double a = (qs * qs) * static_cast<double>(qSq);
			const double b = (rsD * rsD) * static_cast<double>(rowSq);
			const double c = ((rsD * qs) * static_cast<double>(cross)) * 2.0;
			d = (a + b) - c;
		}
		else
		{
			int64_t acc = 0;
			for (int32_t i = 0; i < bank.paddedDims; ++i)
			{
				acc += static_cast<int64_t>(row[i]) * q8[i];
			}
			d = static_cast<double>(acc) * (rsD * qs);
		}
		const double lim = 1.1754943508222875e-38;
		if (d < lim && d > -lim)
		{
			return 0.0f;
		}
		return static_cast<float>(d);
	}
} // namespace xd

static void TestCrossDevice()
{
	// --- Round-half-even, integer math (19.2 S1 c) ---
	CHECK(detail::RoundHalfEvenI32(0.0) == 0);
	CHECK(detail::RoundHalfEvenI32(0.5) == 0);
	CHECK(detail::RoundHalfEvenI32(-0.5) == 0);
	CHECK(detail::RoundHalfEvenI32(1.5) == 2);
	CHECK(detail::RoundHalfEvenI32(2.5) == 2);
	CHECK(detail::RoundHalfEvenI32(-1.5) == -2);
	CHECK(detail::RoundHalfEvenI32(-2.5) == -2);
	CHECK(detail::RoundHalfEvenI32(3.5) == 4);
	CHECK(detail::RoundHalfEvenI32(126.5) == 126);
	CHECK(detail::RoundHalfEvenI32(127.49999) == 127);
	CHECK(detail::RoundHalfEvenI32(0.49999999) == 0);
	CHECK(detail::RoundHalfEvenI32(-126.5) == -126);
	CHECK(detail::RoundHalfEvenI32(2.2250738585072014e-308) == 0); // min normal double
	CHECK(detail::RoundHalfEvenI32(100.25) == 100);
	CHECK(detail::RoundHalfEvenI32(100.75) == 101);

	// --- DAZ-proof float decode ---
	{
		const uint32_t cases[] = {0x00000001u, 0x007fffffu, 0x80000001u, 0x00400000u,
			0x3f800000u, 0xbf800000u, 0x00800000u, 0x7f7fffffu, 0x00000000u, 0x80000000u};
		for (uint32_t bits : cases)
		{
			float f;
			std::memcpy(&f, &bits, 4);
			const double got = detail::FloatBitsToDouble(f);
			// Reference: sign * mant * 2^-149 for exp==0, else the plain cast.
			double want;
			if ((bits & 0x7f800000u) != 0)
			{
				want = static_cast<double>(f);
			}
			else
			{
				want = static_cast<double>(static_cast<int32_t>(bits & 0x7fffffu)) *
					1.4012984643248171e-45;
				if ((bits >> 31) != 0)
				{
					want = -want;
				}
			}
			CHECK(got == want);
		}
	}

	// --- Quantizer contracts ---
	{
		AlignedBuf q(sizeof(float) * 32);
		AlignedBuf q8(32);
		double scale = 0.0;
		int64_t sq = 0;

		// All-zero query: scale 0, all-zero output.
		QuantizeQueryXd(q.F32(), 32, q8.I8(), &scale, &sq);
		CHECK(scale == 0.0 && sq == 0);

		// The max element pins to +/-127; half-way products round to even.
		q.F32()[0] = 2.0f;
		q.F32()[1] = -2.0f;
		q.F32()[2] = 1.0f;
		QuantizeQueryXd(q.F32(), 32, q8.I8(), &scale, &sq);
		CHECK(q8.I8()[0] == 127 && q8.I8()[1] == -127);
		CHECK(q8.I8()[2] == 64); // 63.5 rounds half-even UP to 64
		CHECK(scale == 2.0 / 127.0);
		CHECK(sq == 127ll * 127 + 127ll * 127 + 64ll * 64);

		// An all-subnormal query still quantizes (bit-decode, not DAZ-dependent).
		const uint32_t sub1 = 0x00000100u, sub2 = 0x80000200u;
		std::memset(q.F32(), 0, 32 * sizeof(float));
		std::memcpy(&q.F32()[0], &sub1, 4);
		std::memcpy(&q.F32()[1], &sub2, 4);
		QuantizeQueryXd(q.F32(), 32, q8.I8(), &scale, &sq);
		CHECK(scale > 0.0);
		CHECK(q8.I8()[0] == 64 && q8.I8()[1] == -127); // 0x100/0x200 = half, rounds even
	}

	Workspace ws;

	// --- Rejection matrix: f32 banks and oversized strides stay PerDevice-only ---
	{
		Rng rng(77);
		TestBank f32bank(rng, 16, 32, Quantization::Float32, Metric::Dot);
		AlignedBuf q(sizeof(float) * f32bank.view.paddedDims);
		q.F32()[0] = 1.0f;
		QueryParams p;
		p.k = 4;
		p.exactness = Exactness::CrossDevice;
		Hit hits[4];
		int32_t n = 0;
		CHECK(Query(f32bank.view, q.F32(), p, ws, hits, &n) == Status::InvalidArgument);
		CHECK(QueryBatch(f32bank.view, q.F32(), 1, p, ws, hits, &n) == Status::InvalidArgument);
		CHECK(QueryIntersect(f32bank.view, q.F32(), 1, p, ws, hits, &n) == Status::InvalidArgument);

		// paddedDims over the overflow-proof ceiling is refused.
		const int32_t bigDims = kMaxCrossDeviceDims + 16;
		AlignedBuf bigQ(sizeof(float) * bigDims);
		AlignedBuf bigRow(static_cast<size_t>(bigDims));
		std::vector<float> bigScales(1, 0.5f);
		BankView big;
		big.rows = bigRow.I8();
		big.scales = bigScales.data();
		big.count = 1;
		big.dims = bigDims;
		big.paddedDims = bigDims;
		big.quant = Quantization::Int8;
		big.metric = Metric::Dot;
		bigQ.F32()[0] = 1.0f;
		CHECK(Query(big, bigQ.F32(), p, ws, hits, &n) == Status::InvalidArgument);
	}

	// --- Whole-row scores match the independent contract reimplementation, and
	// the CrossDevice ranking tracks the full-precision reference (recall) ---
	{
		Rng rng(2026);
		for (const Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			TestBank bank(rng, 300, 40, Quantization::Int8, metric);
			AlignedBuf q(sizeof(float) * bank.view.paddedDims);
			std::vector<float> qv(40);
			for (auto& v : qv)
			{
				v = rng.NextFloat();
			}
			PadQuery(qv, bank.view.paddedDims, q.F32());

			AlignedBuf q8(static_cast<size_t>(bank.view.paddedDims));
			double scale = 0.0;
			int64_t sq = 0;
			QuantizeQueryXd(q.F32(), bank.view.paddedDims, q8.I8(), &scale, &sq);

			QueryParams p;
			p.k = bank.view.count;
			p.exactness = Exactness::CrossDevice;
			std::vector<Hit> hits(static_cast<size_t>(p.k));
			int32_t n = 0;
			CHECK(Query(bank.view, q.F32(), p, ws, hits.data(), &n) == Status::Ok);
			CHECK(n == bank.view.count);
			for (int32_t i = 0; i < n; ++i)
			{
				const float want = xd::RefXdScore(bank.view, q8.I8(), scale, sq,
					hits[i].index);
				uint32_t wb, gb;
				std::memcpy(&wb, &want, 4);
				std::memcpy(&gb, &hits[i].score, 4);
				CHECK_MSG(wb == gb, "metric %d row %d: ref %08x got %08x",
					static_cast<int>(metric), hits[i].index, wb, gb);
			}

			// Recall@10 of CrossDevice mode vs the double reference (the honesty
			// number: query quantization adds error beyond row quantization).
			const std::vector<RefHit> ref = ReferenceScan(bank, q.F32(), nullptr);
			int32_t match = 0;
			for (int32_t i = 0; i < 10; ++i)
			{
				for (int32_t j = 0; j < 10; ++j)
				{
					if (hits[static_cast<size_t>(i)].index == ref[static_cast<size_t>(j)].index)
					{
						++match;
						break;
					}
				}
			}
			// Standard-mode recall on the same query, for the side-by-side print.
			QueryParams pd;
			pd.k = 10;
			Hit stdHits[10];
			int32_t stdN = 0;
			CHECK(Query(bank.view, q.F32(), pd, ws, stdHits, &stdN) == Status::Ok);
			int32_t stdMatch = 0;
			for (int32_t i = 0; i < stdN; ++i)
			{
				for (int32_t j = 0; j < 10; ++j)
				{
					if (stdHits[i].index == ref[static_cast<size_t>(j)].index)
					{
						++stdMatch;
						break;
					}
				}
			}
			std::printf("cross-device recall@10 (metric %d): standard %d/10, cross-device %d/10\n",
				static_cast<int>(metric), stdMatch, match);
			CHECK_MSG(match >= 5, "cross-device recall collapsed: %d/10", match);
		}
	}

	// --- Scratch snapshots score CrossDevice like their imported twin ---
	{
		Rng rng(4242);
		const int32_t dims = 32;
		const int32_t capacity = 24;
		ScratchBank scratch;
		CHECK(scratch.Create(capacity, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		std::vector<float> row(static_cast<size_t>(dims));
		for (int32_t r = 0; r < capacity; ++r)
		{
			for (auto& v : row)
			{
				v = rng.NextFloat();
			}
			int32_t index = -1;
			CHECK(scratch.Append(row.data(), dims, &index) == Status::Ok);
		}
		BankView snapView;
		std::vector<uint32_t> tombstones(
			static_cast<size_t>(ScratchBank::TombstoneWords(capacity)));
		CHECK(scratch.Snapshot(&snapView, tombstones.data()) == Status::Ok);

		// Twin: a plain view over the same payload bytes.
		BankView twin = snapView;

		AlignedBuf q(sizeof(float) * snapView.paddedDims);
		std::vector<float> qv(static_cast<size_t>(dims));
		for (auto& v : qv)
		{
			v = rng.NextFloat();
		}
		PadQuery(qv, snapView.paddedDims, q.F32());
		QueryParams p;
		p.k = 8;
		p.exactness = Exactness::CrossDevice;
		Hit a[8], b[8];
		int32_t na = 0, nb = 0;
		CHECK(Query(snapView, q.F32(), p, ws, a, &na) == Status::Ok);
		CHECK(Query(twin, q.F32(), p, ws, b, &nb) == Status::Ok);
		CHECK(na == nb);
		for (int32_t i = 0; i < na; ++i)
		{
			uint32_t ab, bb;
			std::memcpy(&ab, &a[i].score, 4);
			std::memcpy(&bb, &b[i].score, 4);
			CHECK(a[i].index == b[i].index && ab == bb);
		}
	}

	// --- Allocation flatness: warm CrossDevice queries never allocate ---
	{
		Rng rng(99);
		TestBank bank(rng, 200, 48, Quantization::Int8, Metric::Dot);
		AlignedBuf q(sizeof(float) * bank.view.paddedDims);
		std::vector<float> qv(48);
		for (auto& v : qv)
		{
			v = rng.NextFloat();
		}
		PadQuery(qv, bank.view.paddedDims, q.F32());
		QueryParams p;
		p.k = 12;
		p.exactness = Exactness::CrossDevice;
		Hit hits[12];
		int32_t n = 0;
		CHECK(Query(bank.view, q.F32(), p, ws, hits, &n) == Status::Ok); // warm
		const uint64_t before = AllocationCount();
		for (int32_t i = 0; i < 16; ++i)
		{
			CHECK(Query(bank.view, q.F32(), p, ws, hits, &n) == Status::Ok);
		}
		CHECK_MSG(AllocationCount() == before, "warm CrossDevice queries allocated");
	}

	// --- The forced-path matrix + the pinned committed-fixture hash (19.4) ---
	{
		std::vector<SimdPath> paths;
		paths.push_back(SimdPath::Scalar);
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
		paths.push_back(SimdPath::SSE);
		if (ActiveSimdPath() == SimdPath::AVX2)
		{
			paths.push_back(SimdPath::AVX2);
		}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
		paths.push_back(SimdPath::NEON);
#endif

		uint64_t hashes[4] = {};
		for (size_t i = 0; i < paths.size(); ++i)
		{
			detail::ForceXdSimdPath(paths[i]);
			hashes[i] = xd::RunBattery(ws, nullptr);
			detail::ClearForcedXdSimdPath();
			if (i > 0)
			{
				CHECK_MSG(hashes[i] == hashes[0],
					"forced path %d hash %llx != path %d hash %llx",
					static_cast<int>(paths[i]),
					static_cast<unsigned long long>(hashes[i]),
					static_cast<int>(paths[0]),
					static_cast<unsigned long long>(hashes[0]));
			}
		}

		// The default dispatch matches the forced sweep.
		std::FILE* record = nullptr;
		const char* recordPath = std::getenv("SUPERFAISS_XD_SELECTION_OUT");
		if (recordPath != nullptr)
		{
			record = std::fopen(recordPath, "wb");
		}
		const uint64_t defaultHash = xd::RunBattery(ws, record);
		if (record != nullptr)
		{
			std::fprintf(record, "hash %016llx\n",
				static_cast<unsigned long long>(defaultHash));
			std::fclose(record);
		}
		CHECK(defaultHash == hashes[0]);

		std::printf("cross-device hash: %016llx (%d forced paths agree)\n",
			static_cast<unsigned long long>(defaultHash),
			static_cast<int>(paths.size()));
		if constexpr (xdfix::kGoldenXdHash != 0)
		{
			CHECK_MSG(defaultHash == xdfix::kGoldenXdHash,
				"cross-device hash %016llx != pinned golden %016llx",
				static_cast<unsigned long long>(defaultHash),
				static_cast<unsigned long long>(xdfix::kGoldenXdHash));
		}
	}
}


// ---------------------------------------------------------------------------
// T-V2.3-R — scratch-bank recall audit (V2 plan section 20). Retention posture,
// MeasureScratchRecall, generation/staleness, Freeze re-measure, persistence
// version bump, and the S5 scratch x cross-device closure at battery level.

namespace scratchv23
{
	// Calibration floor for the recall sweep (T-V2.3-R3, CAL). Pinned at the first
	// calibration run on the reference fixtures as a FLOOR (not an exact value),
	// space-version-scoped; 0 means 'not yet pinned' (the suite prints the measured
	// numbers instead of asserting). The import analogue pins >= 0.90 on random int8.
	static constexpr float kScratchRecallFloor = 0.90f;

	static bool SameBits(float a, float b)
	{
		uint32_t x, y;
		std::memcpy(&x, &a, 4);
		std::memcpy(&y, &b, 4);
		return x == y;
	}

	static void FillSeeded(ScratchBank& bank, int32_t n, int32_t dims, Rng& rng)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (int32_t r = 0; r < n; ++r)
		{
			for (auto& v : row)
			{
				v = rng.NextFloat();
			}
			int32_t idx = -1;
			CHECK(bank.Append(row.data(), dims, &idx) == Status::Ok);
			CHECK(idx == r);
		}
	}
} // namespace scratchv23

// R1, R2, R6, R7, R9: retention is an opt-in bank property; the retained reference is
// the post-normalization row; retained floats serialize with a version bump; Grow
// copies the arena index-preserving; the honest byte-cost is B1 exact.
static void TestScratchRetention()
{
	using namespace scratchv23;
	Workspace ws;

	// --- R1: retention flag is opt-in and visible ---
	std::printf("scratch retention (T-V2.3-R1): opt-in flag, descriptor-visible\n");
	{
		ScratchBank ret;
		CHECK(ret.Create(512, 256, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		CHECK(ret.RetainsFloats() == true);

		ScratchBank def;
		CHECK(def.Create(512, 256, Metric::Dot, Quantization::Int8) == Status::Ok);
		CHECK(def.RetainsFloats() == false);
		CHECK(def.RetainedRowBytes() == 0);
		// A default Create allocates no retention arena: an appended row has no twin.
		std::vector<float> row(256, 0.25f);
		int32_t idx = -1;
		CHECK(def.Append(row.data(), 256, &idx) == Status::Ok);
		CHECK(def.RetainedRow(0) == nullptr);
	}

	// --- R2: retained reference is the post-normalization row the quantizer consumed ---
	std::printf("scratch retention (T-V2.3-R2): retained == post-normalization row\n");
	{
		Rng rng(0x2E7A1234ull);
		const int32_t dims = 48;
		ScratchBank cos;
		CHECK(cos.Create(64, dims, Metric::Cosine, Quantization::Int8, true) == Status::Ok);
		ScratchBank dot;
		CHECK(dot.Create(64, dims, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		std::vector<float> src(static_cast<size_t>(dims));
		for (int32_t r = 0; r < 32; ++r)
		{
			for (auto& v : src)
			{
				v = rng.NextFloat();
			}
			int32_t idx = -1;
			CHECK(cos.Append(src.data(), dims, &idx) == Status::Ok);
			CHECK(dot.Append(src.data(), dims, &idx) == Status::Ok);

			// Cosine: the retained row is the unit-norm row (importer parity).
			std::vector<float> norm(src);
			CHECK(NormalizeRows(norm.data(), 1, dims, nullptr) == Status::Ok);
			const float* retCos = cos.RetainedRow(r);
			CHECK(retCos != nullptr);
			CHECK(std::memcmp(retCos, norm.data(),
					static_cast<size_t>(dims) * sizeof(float)) == 0);

			// Dot: the retained row is the finite-validated input, unchanged.
			const float* retDot = dot.RetainedRow(r);
			CHECK(retDot != nullptr);
			CHECK(std::memcmp(retDot, src.data(),
					static_cast<size_t>(dims) * sizeof(float)) == 0);
		}
		// A rejected append retains nothing: the count does not advance, so the slot
		// is not published.
		std::vector<float> zero(static_cast<size_t>(dims), 0.0f);
		const int32_t before = cos.Count();
		CHECK(cos.Append(zero.data(), dims, nullptr) == Status::ZeroNormRow);
		CHECK(cos.Count() == before);
		CHECK(cos.RetainedRow(before) == nullptr);
	}

	// --- R9: honest-budget byte cost equals B1 exactly ---
	std::printf("scratch retention (T-V2.3-R9): memory cost == B1\n");
	{
		ScratchBank i8;
		CHECK(i8.Create(4, 256, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		CHECK_MSG(i8.QuantizedRowBytes() == 260, "int8 quantized row %lld",
			static_cast<long long>(i8.QuantizedRowBytes()));
		CHECK_MSG(i8.RetainedRowBytes() == 1024, "int8 retained row %lld",
			static_cast<long long>(i8.RetainedRowBytes()));
		CHECK(i8.QuantizedRowBytes() + i8.RetainedRowBytes() == 1284); // 4.9x

		ScratchBank f32;
		CHECK(f32.Create(4, 256, Metric::Dot, Quantization::Float32, true) == Status::Ok);
		CHECK_MSG(f32.QuantizedRowBytes() == 1024, "f32 quantized row %lld",
			static_cast<long long>(f32.QuantizedRowBytes()));
		CHECK(f32.RetainedRowBytes() == 1024);
		CHECK(f32.QuantizedRowBytes() + f32.RetainedRowBytes() == 2048); // 2.0x
	}

	// --- R7: Grow copies the retention arena, index-preserving ---
	std::printf("scratch retention (T-V2.3-R7): Grow preserves the retention arena\n");
	{
		const int32_t dims = 48;
		Rng rng(0x6E0Full);
		ScratchBank g;
		CHECK(g.Create(32, dims, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		FillSeeded(g, 32, dims, rng);
		CHECK(g.Remove(3) == Status::Ok);
		CHECK(g.Remove(19) == Status::Ok);

		// Snapshot the retained rows before Grow.
		std::vector<float> preRetained;
		for (int32_t r = 0; r < g.Count(); ++r)
		{
			const float* rr = g.RetainedRow(r);
			CHECK(rr != nullptr);
			preRetained.insert(preRetained.end(), rr, rr + dims);
		}
		ScratchRecallReport pre;
		CHECK(g.MeasureScratchRecall(ws, &pre, 0x1234ull) == Status::Ok);

		CHECK(g.Grow(64) == Status::Ok);
		CHECK(g.Capacity() == 64);

		for (int32_t r = 0; r < g.Count(); ++r)
		{
			const float* rr = g.RetainedRow(r);
			CHECK(rr != nullptr);
			CHECK(std::memcmp(rr, preRetained.data() + static_cast<size_t>(r) * dims,
					static_cast<size_t>(dims) * sizeof(float)) == 0);
		}
		ScratchRecallReport post;
		CHECK(g.MeasureScratchRecall(ws, &post, 0x1234ull) == Status::Ok);
		CHECK(SameBits(pre.recall, post.recall)); // same seed + snapshot -> same bits
	}

	// --- R6: retained floats serialize; version bumps 1->2; hard-reject over the set ---
	std::printf("scratch retention (T-V2.3-R6): serialize + version bump + hard-reject\n");
	{
		const int32_t dims = 48;
		Rng rng(0x5A7E5ull);
		ScratchBank s;
		CHECK(s.Create(64, dims, Metric::Cosine, Quantization::Int8, true) == Status::Ok);
		FillSeeded(s, 50, dims, rng);
		CHECK(s.Remove(7) == Status::Ok);
		CHECK(s.Remove(11) == Status::Ok);
		ScratchRecallReport r0;
		CHECK(s.MeasureScratchRecall(ws, &r0, 0x99ull) == Status::Ok);

		MemArchive ar;
		CHECK(s.Save(ar.Writer()) == Status::Ok);
		// A retention blob writes version 2 (uint32 at byte offset 4, little-endian).
		CHECK_MSG(ar.bytes[4] == 2, "retention version byte %d", ar.bytes[4]);

		ScratchBank loaded;
		CHECK(loaded.Load(ar.Reader()) == Status::Ok);
		CHECK(loaded.RetainsFloats());
		CHECK(loaded.Count() == s.Count());
		CHECK(loaded.LiveCount() == s.LiveCount());
		ScratchRecallReport r1;
		CHECK(loaded.MeasureScratchRecall(ws, &r1, 0x99ull) == Status::Ok);
		CHECK(SameBits(r0.recall, r1.recall)); // round-trip bit-equal on the same seed
		CHECK(r0.k == r1.k && r0.sampleCount == r1.sampleCount && r0.liveRows == r1.liveRows);

		// A non-retention blob still writes version 1, and loads with retention absent.
		ScratchBank plain;
		Rng prng(0x5A7E5ull);
		CHECK(plain.Create(64, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		FillSeeded(plain, 20, dims, prng);
		MemArchive pa;
		CHECK(plain.Save(pa.Writer()) == Status::Ok);
		CHECK_MSG(pa.bytes[4] == 1, "non-retention version byte %d", pa.bytes[4]);
		ScratchBank fromV1;
		CHECK(fromV1.Load(pa.Reader()) == Status::Ok);
		CHECK(!fromV1.RetainsFloats());
		ScratchRecallReport rNo;
		CHECK(fromV1.MeasureScratchRecall(ws, &rNo, 0x99ull) == Status::InvalidArgument);

		// Version > 2 -> BadFormat, reject-over-degrade (the target bank unchanged).
		MemArchive bad;
		bad.bytes = ar.bytes;
		bad.bytes[4] = 3;
		CHECK(loaded.Load(bad.Reader()) == Status::BadFormat);
		CHECK(loaded.RetainsFloats());
		CHECK(loaded.Count() == s.Count());
		// Version 0 -> BadFormat.
		MemArchive zero;
		zero.bytes = ar.bytes;
		zero.bytes[4] = 0;
		CHECK(loaded.Load(zero.Reader()) == Status::BadFormat);
	}
}

// R3, R4, R5, R8: the measurement protocol (calibration), the measurement contract
// (tombstones/generation/staleness/min-size), Freeze re-measurement, and the carried
// contracts under retention.
static void TestScratchRecall()
{
	using namespace scratchv23;
	Workspace ws;
	const uint64_t seed = ScratchBank::kDefaultRecallSeed;

	// --- R3: MeasureScratchRecall protocol + calibration (CAL) ---
	std::printf("scratch recall (T-V2.3-R3): float scan vs quantized cross-device scan\n");
	{
		int32_t metricTag = 0;
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			ScratchBank b;
			CHECK(b.Create(512, 128, metric, Quantization::Int8, true) == Status::Ok);
			Rng rng(0x0C0FFEEull + static_cast<uint64_t>(metricTag) * 0x1000ull);
			FillSeeded(b, 512, 128, rng);
			ScratchRecallReport rep;
			CHECK(b.MeasureScratchRecall(ws, &rep, seed) == Status::Ok);
			std::printf(
				"  scratch recall@10 (metric %d): %.4f (n=%d, k=%d, informative=%d, seed=%016llx)\n",
				metricTag, static_cast<double>(rep.recall), rep.sampleCount, rep.k,
				rep.informative ? 1 : 0, static_cast<unsigned long long>(rep.seed));
			CHECK(rep.k == 10);
			CHECK(rep.sampleCount == 512);
			CHECK(rep.liveRows == 512);
			CHECK(rep.informative);
			CHECK(rep.seed == seed);
			CHECK(rep.recall >= 0.0f && rep.recall <= 1.0f);
			if constexpr (kScratchRecallFloor > 0.0f)
			{
				CHECK_MSG(rep.recall >= kScratchRecallFloor,
					"recall floor breached (metric %d): %.4f < %.4f", metricTag,
					static_cast<double>(rep.recall),
					static_cast<double>(kScratchRecallFloor));
			}
			++metricTag;
		}
	}

	// --- R4: tombstones excluded, generation-stamped, stale-marked, min-size stated ---
	std::printf("scratch recall (T-V2.3-R4): measurement contract\n");
	{
		const int32_t dims = 48;
		Rng rng(0x4C0Aull);
		ScratchBank b;
		CHECK(b.Create(256, dims, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		FillSeeded(b, 200, dims, rng);
		for (int32_t r = 0; r < 200; r += 9)
		{
			CHECK(b.Remove(r) == Status::Ok);
		}
		ScratchRecallReport rep;
		CHECK(b.MeasureScratchRecall(ws, &rep, seed) == Status::Ok);
		// Generation stamp + tombstones excluded from the sample space.
		CHECK(rep.generation == b.Generation());
		CHECK(!b.RecallReportStale(rep));
		CHECK(rep.liveRows == b.LiveCount());
		CHECK(rep.sampleCount == b.LiveCount()); // < 1000, so every live row is drawn
		CHECK(rep.informative); // > 100 live rows

		// An append after the report renders it stale.
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row)
		{
			v = rng.NextFloat();
		}
		CHECK(b.Append(row.data(), dims, nullptr) == Status::Ok);
		CHECK(b.RecallReportStale(rep));

		// A newly-set Remove advances the generation; an idempotent re-Remove does not.
		ScratchRecallReport rep2;
		CHECK(b.MeasureScratchRecall(ws, &rep2, seed) == Status::Ok);
		CHECK(!b.RecallReportStale(rep2));
		CHECK(b.Remove(1) == Status::Ok); // row 1 was live
		CHECK(b.RecallReportStale(rep2));
		ScratchRecallReport rep3;
		CHECK(b.MeasureScratchRecall(ws, &rep3, seed) == Status::Ok);
		const uint64_t g = b.Generation();
		CHECK(b.Remove(1) == Status::Ok); // idempotent re-Remove
		CHECK(b.Generation() == g);
		CHECK(!b.RecallReportStale(rep3));

		// Minimum-size statement: below kRecallMinRows the number is recall@(liveRows-1);
		// below kRecallInformativeRows it is marked uninformative.
		ScratchBank tiny;
		Rng trng(0x7117ull);
		CHECK(tiny.Create(16, dims, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		FillSeeded(tiny, 8, dims, trng);
		ScratchRecallReport tr;
		CHECK(tiny.MeasureScratchRecall(ws, &tr, seed) == Status::Ok);
		CHECK(tr.liveRows == 8);
		CHECK(tr.k == 7); // min(10, 8-1) — a recall@7, not @10
		CHECK(!tr.informative);

		ScratchBank mid;
		Rng mrng(0x33aaull);
		CHECK(mid.Create(128, dims, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		FillSeeded(mid, 40, dims, mrng);
		ScratchRecallReport mr;
		CHECK(mid.MeasureScratchRecall(ws, &mr, seed) == Status::Ok);
		CHECK(mr.liveRows == 40);
		CHECK(mr.k == 10);       // a true recall@10 ...
		CHECK(!mr.informative);  // ... but still statistically uninformative
		CHECK(mr.sampleCount == 40);
	}

	// --- R5: Freeze re-measures at freeze time ---
	std::printf("scratch recall (T-V2.3-R5): Freeze re-measures the compacted rows\n");
	{
		const int32_t dims = 48;
		const int32_t pd = PaddedDims(dims, Quantization::Int8);
		Rng rng(0xF2EEull);
		ScratchBank f;
		CHECK(f.Create(160, dims, Metric::Cosine, Quantization::Int8, true) == Status::Ok);
		FillSeeded(f, 120, dims, rng);
		CHECK(f.Remove(5) == Status::Ok);
		CHECK(f.Remove(60) == Status::Ok);

		ScratchRecallReport before;
		CHECK(f.MeasureScratchRecall(ws, &before, seed) == Status::Ok);

		// Mutate after the report: an append and a remove.
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row)
		{
			v = rng.NextFloat();
		}
		CHECK(f.Append(row.data(), dims, nullptr) == Status::Ok);
		CHECK(f.Remove(9) == Status::Ok);
		CHECK(f.RecallReportStale(before)); // the pre-mutation report is now stale

		// A direct measure of the current rows is the freeze-time reference.
		ScratchRecallReport direct;
		CHECK(f.MeasureScratchRecall(ws, &direct, seed) == Status::Ok);

		const int32_t live = f.FreezeLiveCount();
		AlignedBuf frozenRows(static_cast<size_t>(live) * pd * sizeof(int8_t));
		std::vector<float> frozenScales(static_cast<size_t>(live));
		std::vector<int32_t> indexMap(static_cast<size_t>(f.Count()));
		ScratchRecallReport frozen;
		CHECK(f.Freeze(frozenRows.ptr, frozenScales.data(), indexMap.data(),
				&frozen, &ws, seed) == Status::Ok);
		// The frozen number is measured at freeze time (current generation), and
		// INV-equals a direct measure of the same rows on the same seed.
		CHECK(frozen.generation == f.Generation());
		CHECK(!f.RecallReportStale(frozen));
		CHECK(SameBits(frozen.recall, direct.recall));

		// A non-retention Freeze produces no number: *outReport is left untouched.
		ScratchBank nr;
		Rng nrng(0xF2EEull);
		CHECK(nr.Create(32, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		FillSeeded(nr, 16, dims, nrng);
		ScratchRecallReport sentinel;
		sentinel.generation = 0xDEADBEEFull;
		sentinel.recall = -1.0f;
		const int32_t nrLive = nr.FreezeLiveCount();
		AlignedBuf nrRows(static_cast<size_t>(nrLive) * pd * sizeof(int8_t));
		std::vector<float> nrScales(static_cast<size_t>(nrLive));
		std::vector<int32_t> nrMap(static_cast<size_t>(nr.Count()));
		CHECK(nr.Freeze(nrRows.ptr, nrScales.data(), nrMap.data(),
				&sentinel, &ws, seed) == Status::Ok);
		CHECK(sentinel.generation == 0xDEADBEEFull && sentinel.recall == -1.0f);
	}

	// --- R8: carried contracts hold under retention ---
	std::printf("scratch recall (T-V2.3-R8): carried contracts\n");
	{
		const int32_t dims = 48;
		// R8.4 reject-over-degrade: non-retention MeasureScratchRecall -> InvalidArgument.
		ScratchBank nr;
		Rng nrng(0x8D4ull);
		CHECK(nr.Create(64, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		FillSeeded(nr, 32, dims, nrng);
		ScratchRecallReport rj;
		CHECK(nr.MeasureScratchRecall(ws, &rj, seed) == Status::InvalidArgument);

		// R8.1 zero steady-state allocation: warm one measure, then flat.
		ScratchBank b;
		Rng rng(0x8A11ull);
		CHECK(b.Create(256, dims, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		FillSeeded(b, 200, dims, rng);
		ScratchRecallReport warm;
		CHECK(b.MeasureScratchRecall(ws, &warm, seed) == Status::Ok);
		const uint64_t before = AllocationCount();
		for (int32_t i = 0; i < 8; ++i)
		{
			ScratchRecallReport r;
			CHECK(b.MeasureScratchRecall(ws, &r, seed) == Status::Ok);
		}
		CHECK_MSG(AllocationCount() == before, "warm MeasureScratchRecall allocated");

		// R8.3 determinism-given-history: identical history -> identical recall bits.
		ScratchBank b1;
		ScratchBank b2;
		Rng r1(0x0777ull);
		Rng r2(0x0777ull);
		CHECK(b1.Create(128, dims, Metric::Cosine, Quantization::Int8, true) == Status::Ok);
		CHECK(b2.Create(128, dims, Metric::Cosine, Quantization::Int8, true) == Status::Ok);
		FillSeeded(b1, 60, dims, r1);
		FillSeeded(b2, 60, dims, r2);
		CHECK(b1.Remove(3) == Status::Ok);
		CHECK(b2.Remove(3) == Status::Ok);
		ScratchRecallReport a;
		ScratchRecallReport c;
		CHECK(b1.MeasureScratchRecall(ws, &a, seed) == Status::Ok);
		CHECK(b2.MeasureScratchRecall(ws, &c, seed) == Status::Ok);
		CHECK(SameBits(a.recall, c.recall));
		ScratchRecallReport again;
		CHECK(b1.MeasureScratchRecall(ws, &again, seed) == Status::Ok);
		CHECK(SameBits(a.recall, again.recall)); // same bank twice, same bits
	}
}

// R10: S5 companion closure — a NON-shared-payload baked twin of a scratch snapshot,
// with a tombstoned row and a Grow in its history, scores bit-identically to the
// snapshot in CrossDevice mode across the forced-path sweep; the hit-list hash is
// pinned beside kGoldenXdHash.
static void TestScratchXdClosure()
{
	std::printf("scratch x cross-device closure (T-V2.3-R10)\n");
	Workspace ws;

	const int32_t dims = xdfix::kBankADims; // 48
	const int32_t pd = PaddedDims(dims, Quantization::Int8);
	const int32_t initialCap = 40;
	const int32_t finalCount = 56;

	ScratchBank scratch;
	CHECK(scratch.Create(initialCap, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
	std::vector<float> row(static_cast<size_t>(dims));
	auto appendFromFixture = [&](int32_t srcRow) {
		for (int32_t i = 0; i < dims; ++i)
		{
			row[static_cast<size_t>(i)] = static_cast<float>(
				xdfix::kBankARows[static_cast<size_t>(srcRow) * dims + i]);
		}
		int32_t idx = -1;
		CHECK(scratch.Append(row.data(), dims, &idx) == Status::Ok);
	};
	for (int32_t r = 0; r < initialCap; ++r)
	{
		appendFromFixture(r);
	}
	// A Grow in the history, then append past the old capacity.
	CHECK(scratch.Grow(64) == Status::Ok);
	for (int32_t r = initialCap; r < finalCount; ++r)
	{
		appendFromFixture(r);
	}
	// >= 1 tombstoned row present in the comparison (a low, a middle, a post-Grow row).
	CHECK(scratch.Remove(5) == Status::Ok);
	CHECK(scratch.Remove(37) == Status::Ok);
	CHECK(scratch.Remove(50) == Status::Ok);
	CHECK(scratch.LiveCount() == finalCount - 3);

	BankView snapView;
	std::vector<uint32_t> tombs(static_cast<size_t>(ScratchBank::TombstoneWords(64)), 0u);
	CHECK(scratch.Snapshot(&snapView, tombs.data()) == Status::Ok);
	const int32_t count = snapView.count;

	// The non-shared-payload twin: independent allocations, COPIED bytes (not a
	// BankView aliasing the arena — that is the gap this closes).
	AlignedBuf twinRows(static_cast<size_t>(count) * pd * sizeof(int8_t));
	std::memcpy(twinRows.ptr, snapView.rows,
		static_cast<size_t>(count) * pd * sizeof(int8_t));
	std::vector<float> twinScales(static_cast<size_t>(count));
	std::memcpy(twinScales.data(), snapView.scales,
		static_cast<size_t>(count) * sizeof(float));
	BankView twin = snapView;
	twin.rows = twinRows.ptr;
	twin.scales = twinScales.data();
	CHECK(twin.rows != snapView.rows); // genuinely independent bytes
	CHECK(ValidateBank(twin) == Status::Ok);

	auto runBattery = [&]() -> uint64_t {
		xd::BatteryHash hash;
		AlignedBuf qbuf(sizeof(float) * static_cast<size_t>(pd));
		for (int32_t qi = 0; qi < xdfix::kQueryCount; ++qi)
		{
			xd::LoadQuery(qi, pd, qbuf.F32());
			QueryParams p;
			p.k = 10;
			p.exactness = Exactness::CrossDevice;
			p.excludeBits = tombs.data();
			Hit hitsSnap[10];
			Hit hitsTwin[10];
			int32_t nSnap = 0;
			int32_t nTwin = 0;
			CHECK(Query(snapView, qbuf.F32(), p, ws, hitsSnap, &nSnap) == Status::Ok);
			CHECK(Query(twin, qbuf.F32(), p, ws, hitsTwin, &nTwin) == Status::Ok);
			CHECK(nSnap == nTwin);
			for (int32_t i = 0; i < nSnap; ++i)
			{
				uint32_t sb, tb;
				std::memcpy(&sb, &hitsSnap[i].score, 4);
				std::memcpy(&tb, &hitsTwin[i].score, 4);
				CHECK_MSG(hitsSnap[i].index == hitsTwin[i].index && sb == tb,
					"twin != snapshot: idx %d/%d bits %08x/%08x", hitsSnap[i].index,
					hitsTwin[i].index, sb, tb);
				// A tombstoned row is excluded identically in both.
				CHECK(!IsExcluded(tombs.data(), hitsSnap[i].index));
			}
			xd::CheckFloor(hitsSnap, nSnap);
			hash.Hits(hitsSnap, nSnap);
		}
		return hash.h;
	};

	// Forced-path sweep: scalar / SSE4.1 / AVX2 here; NEON/scalar on the ARM runner.
	std::vector<SimdPath> paths;
	paths.push_back(SimdPath::Scalar);
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
	paths.push_back(SimdPath::SSE);
	if (ActiveSimdPath() == SimdPath::AVX2)
	{
		paths.push_back(SimdPath::AVX2);
	}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
	paths.push_back(SimdPath::NEON);
#endif
	uint64_t hashes[4] = {};
	for (size_t i = 0; i < paths.size(); ++i)
	{
		detail::ForceXdSimdPath(paths[i]);
		hashes[i] = runBattery();
		detail::ClearForcedXdSimdPath();
		if (i > 0)
		{
			CHECK_MSG(hashes[i] == hashes[0],
				"scratch-xd forced path %d hash %llx != path %d hash %llx",
				static_cast<int>(paths[i]),
				static_cast<unsigned long long>(hashes[i]),
				static_cast<int>(paths[0]),
				static_cast<unsigned long long>(hashes[0]));
		}
	}
	const uint64_t battery = runBattery(); // default dispatch matches the forced sweep
	CHECK(battery == hashes[0]);
	std::printf("scratch cross-device hash: %016llx (%d forced paths agree)\n",
		static_cast<unsigned long long>(battery), static_cast<int>(paths.size()));
	if constexpr (xdfix::kGoldenScratchXdHash != 0)
	{
		CHECK_MSG(battery == xdfix::kGoldenScratchXdHash,
			"scratch cross-device hash %016llx != pinned golden %016llx",
			static_cast<unsigned long long>(battery),
			static_cast<unsigned long long>(xdfix::kGoldenScratchXdHash));
	}
}


// ---------------------------------------------------------------------------
// T25 — workspace reuse across shapes (external bug report, 2026-07-04): the
// query-scratch buffer grows monotonically (allocation-flat contract), so its
// internal stride can EXCEED the current bank's paddedDims after serving a
// larger reservation. The folded-batch paths packed queries at the scratch
// stride but consumed them at bank.paddedDims - wrong hits on perfectly normal
// reuse. This suite drives one Workspace through the reviewer's matrix:
// large->small dims (the repro), small->large, f32->int8, segmented->plain,
// batch->single->intersect, cross-device->per-device - every result compared
// against a FRESH workspace's answer, bit for bit. A fresh workspace cannot
// carry stale strides, so agreement proves reuse-independence.

static void TestWorkspaceReuseAcrossShapes()
{
	Rng rng(0x5E05Eull);
	const QuerySegment degenSmall[1] = {{0, 0, 1.0f}}; // patched per-bank below

	// One long-lived workspace, deliberately polluted by a LARGE segmented batch
	// first; every later shape must still answer identically to a fresh one.
	Workspace reused;

	auto checkAgainstFresh = [&](const BankView& view, const float* queries,
		int32_t queryCount, const QueryParams& params, const char* label) {
		std::vector<Hit> hitsReused(static_cast<size_t>(queryCount) * params.k);
		std::vector<Hit> hitsFresh(static_cast<size_t>(queryCount) * params.k);
		std::vector<int32_t> countsReused(queryCount), countsFresh(queryCount);
		Workspace fresh;
		if (queryCount == 1)
		{
			CHECK(Query(view, queries, params, reused, hitsReused.data(),
				countsReused.data()) == Status::Ok);
			CHECK(Query(view, queries, params, fresh, hitsFresh.data(),
				countsFresh.data()) == Status::Ok);
		}
		else
		{
			CHECK(QueryBatch(view, queries, queryCount, params, reused,
				hitsReused.data(), countsReused.data()) == Status::Ok);
			CHECK(QueryBatch(view, queries, queryCount, params, fresh,
				hitsFresh.data(), countsFresh.data()) == Status::Ok);
		}
		for (int32_t m = 0; m < queryCount; ++m)
		{
			CHECK_MSG(countsReused[m] == countsFresh[m], "%s q%d count %d != %d",
				label, m, countsReused[m], countsFresh[m]);
			for (int32_t i = 0; i < countsFresh[m] && i < countsReused[m]; ++i)
			{
				const Hit& a = hitsReused[static_cast<size_t>(m) * params.k + i];
				const Hit& b = hitsFresh[static_cast<size_t>(m) * params.k + i];
				CHECK_MSG(a.index == b.index && a.score == b.score,
					"%s q%d hit%d (%d, %.9g) != fresh (%d, %.9g)",
					label, m, i, a.index, a.score, b.index, b.score);
			}
		}
	};

	// Step 1: pollute with a big segmented dot-family batch (dims 256 -> the
	// scratch stride grows to 256-padded).
	{
		TestBank big(rng, 600, 256, Quantization::Float32, Metric::Cosine);
		const int32_t m = 8;
		AlignedBuf qs(static_cast<size_t>(m) * big.view.paddedDims * sizeof(float));
		for (int32_t q = 0; q < m; ++q)
		{
			std::vector<float> raw(256);
			for (auto& v : raw)
			{
				v = rng.NextFloat();
			}
			PadQuery(raw, big.view.paddedDims,
				qs.F32() + static_cast<int64_t>(q) * big.view.paddedDims);
		}
		const QuerySegment segs[2] = {{0, 128, 1.0f}, {128, 128, 0.5f}};
		QueryParams p;
		p.k = 5;
		p.segments = segs;
		p.segmentCount = 2;
		checkAgainstFresh(big.view, qs.F32(), m, p, "pollute-big-segbatch");
	}

	// Step 2 (the reviewer's repro): smaller bank, segmented dot-family batch on
	// the SAME workspace - the folded queries must not be read at the wrong stride.
	{
		TestBank small(rng, 200, 32, Quantization::Float32, Metric::Cosine);
		const int32_t m = 4;
		AlignedBuf qs(static_cast<size_t>(m) * small.view.paddedDims * sizeof(float));
		for (int32_t q = 0; q < m; ++q)
		{
			std::vector<float> raw(32);
			for (auto& v : raw)
			{
				v = rng.NextFloat();
			}
			PadQuery(raw, small.view.paddedDims,
				qs.F32() + static_cast<int64_t>(q) * small.view.paddedDims);
		}
		const QuerySegment segs[2] = {{0, 16, 1.0f}, {16, 16, 1.0f}};
		QueryParams p;
		p.k = 3;
		p.segments = segs;
		p.segmentCount = 2;
		checkAgainstFresh(small.view, qs.F32(), m, p, "small-after-big-segbatch");

		// Same shape through QueryIntersect (the second folded consumer).
		{
			Hit hitsReused[3], hitsFresh[3];
			int32_t nr = 0, nf = 0;
			Workspace fresh;
			CHECK(QueryIntersect(small.view, qs.F32(), m, p, reused, hitsReused, &nr)
				== Status::Ok);
			CHECK(QueryIntersect(small.view, qs.F32(), m, p, fresh, hitsFresh, &nf)
				== Status::Ok);
			CHECK_MSG(nr == nf, "intersect count %d != %d", nr, nf);
			for (int32_t i = 0; i < nf && i < nr; ++i)
			{
				CHECK_MSG(hitsReused[i].index == hitsFresh[i].index &&
					hitsReused[i].score == hitsFresh[i].score,
					"intersect hit%d (%d, %.9g) != fresh (%d, %.9g)", i,
					hitsReused[i].index, hitsReused[i].score,
					hitsFresh[i].index, hitsFresh[i].score);
			}
		}

		// Sparse bias on the segmented fold path (the rescore walks the folded
		// queries too).
		{
			BiasPair pair[1] = {{150, 100.0f}};
			RowBias rb;
			rb.pairs = pair;
			rb.pairCount = 1;
			QueryParams bp = p;
			bp.bias = &rb;
			checkAgainstFresh(small.view, qs.F32(), 1, bp, "sparse-after-big");
		}
	}

	// Step 3: the rest of the matrix on the same reused workspace.
	{
		// small -> large again (regrow), f32 -> int8, segmented -> plain,
		// batch -> single, per-device after cross-device.
		TestBank medium(rng, 300, 64, Quantization::Int8, Metric::Dot);
		const int32_t m = 5;
		AlignedBuf qs(static_cast<size_t>(m) * medium.view.paddedDims * sizeof(float));
		for (int32_t q = 0; q < m; ++q)
		{
			std::vector<float> raw(64);
			for (auto& v : raw)
			{
				v = rng.NextFloat();
			}
			PadQuery(raw, medium.view.paddedDims,
				qs.F32() + static_cast<int64_t>(q) * medium.view.paddedDims);
		}
		const QuerySegment segs[2] = {{0, 32, 1.0f}, {32, 32, 2.0f}};
		QueryParams segp;
		segp.k = 4;
		segp.segments = segs;
		segp.segmentCount = 2;
		checkAgainstFresh(medium.view, qs.F32(), m, segp, "int8-segbatch");

		QueryParams plain;
		plain.k = 4;
		checkAgainstFresh(medium.view, qs.F32(), m, plain, "plain-batch");
		checkAgainstFresh(medium.view, qs.F32(), 1, plain, "plain-single");
		checkAgainstFresh(medium.view, qs.F32(), 1, segp, "seg-single");

		QueryParams xd;
		xd.k = 4;
		xd.exactness = Exactness::CrossDevice;
		checkAgainstFresh(medium.view, qs.F32(), m, xd, "crossdevice-batch");
		checkAgainstFresh(medium.view, qs.F32(), 1, plain, "perdevice-after-xd");
	}
	(void)degenSmall;
}


// ---------------------------------------------------------------------------
// T26 — trust-boundary validation (Poirot R-2): a loaded payload must satisfy
// the bake's own laws. Tampered content that the old validation admitted -
// non-unit Cosine rows, zero-norm Cosine rows, int8 -128 lanes (outside the
// bake clamp; the CrossDevice overflow proof depends on it) - is BadFormat,
// through both ValidateBankData and the scratch-archive Load that builds on it.

static void TestTrustBoundaryValidation()
{
	Rng rng(0x7B057ull);
	const int32_t dims = 32;

	// Non-unit Cosine f32 row -> BadFormat at the offending row.
	{
		TestBank bank(rng, 20, dims, Quantization::Float32, Metric::Cosine);
		CHECK(ValidateBankData(bank.view, nullptr) == Status::Ok);
		float* rows = static_cast<float*>(const_cast<void*>(bank.view.rows));
		const int32_t victim = 7;
		for (int32_t j = 0; j < dims; ++j)
		{
			rows[victim * bank.view.paddedDims + j] *= 3.0f; // norm 3, finite
		}
		int32_t bad = -1;
		CHECK(ValidateBankData(bank.view, &bad) == Status::BadFormat);
		CHECK(bad == victim);
	}

	// Zero-norm Cosine row -> BadFormat (the append/bake law, now at load too).
	{
		TestBank bank(rng, 12, dims, Quantization::Float32, Metric::Cosine);
		float* rows = static_cast<float*>(const_cast<void*>(bank.view.rows));
		for (int32_t j = 0; j < bank.view.paddedDims; ++j)
		{
			rows[3 * bank.view.paddedDims + j] = 0.0f;
		}
		int32_t bad = -1;
		CHECK(ValidateBankData(bank.view, &bad) == Status::BadFormat);
		CHECK(bad == 3);
	}

	// Int8 -128 lane -> BadFormat (bake clamps to [-127, 127]).
	{
		TestBank bank(rng, 16, dims, Quantization::Int8, Metric::Dot);
		CHECK(ValidateBankData(bank.view, nullptr) == Status::Ok);
		int8_t* rows = static_cast<int8_t*>(const_cast<void*>(bank.view.rows));
		rows[5 * bank.view.paddedDims + 2] = INT8_MIN;
		int32_t bad = -1;
		CHECK(ValidateBankData(bank.view, &bad) == Status::BadFormat);
		CHECK(bad == 5);
	}

	// Int8 Cosine quantization noise stays WITHIN tolerance (no false rejection).
	{
		TestBank bank(rng, 200, dims, Quantization::Int8, Metric::Cosine);
		CHECK(ValidateBankData(bank.view, nullptr) == Status::Ok);
	}

	// Tampered scratch archive: denormalize a Cosine row's bytes in the blob ->
	// Load rejects (BadFormat) and the target bank is untouched.
	{
		ScratchBank bank;
		CHECK(bank.Create(8, dims, Metric::Cosine, Quantization::Float32) == Status::Ok);
		alignas(16) float row[dims];
		for (int32_t j = 0; j < dims; ++j)
		{
			row[j] = rng.NextFloat();
		}
		CHECK(bank.Append(row, dims, nullptr) == Status::Ok);
		MemArchive archive;
		CHECK(bank.Save(archive.Writer()) == Status::Ok);
		// Scale the row payload in place (header is 32 bytes; floats follow).
		float tampered[dims];
		std::memcpy(tampered, archive.bytes.data() + 32, sizeof(tampered));
		for (int32_t j = 0; j < dims; ++j)
		{
			tampered[j] *= 2.0f;
		}
		std::memcpy(archive.bytes.data() + 32, tampered, sizeof(tampered));
		ScratchBank loaded;
		CHECK(loaded.Load(archive.Reader()) == Status::BadFormat);
		CHECK(!loaded.IsCreated());
	}
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
	TestPerRowBias();
	TestWorkspaceReuseAcrossShapes();
	TestTrustBoundaryValidation();
	TestCrossDevice();
	TestScratchRetention();
	TestScratchRecall();
	TestScratchXdClosure();

	std::printf("superfaiss tests: %d checks, %d failures (simd path: %s)\n",
		GChecks, GFailures,
		ActiveSimdPath() == SimdPath::Scalar ? "scalar"
			: ActiveSimdPath() == SimdPath::SSE ? "sse"
			: ActiveSimdPath() == SimdPath::AVX2 ? "avx2" : "neon");
	return GFailures == 0 ? 0 : 1;
}

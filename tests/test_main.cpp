// SuperFAISS test harness. Standard library only; no third-party test framework.
// Reference results are computed in double precision with the same total order
// (score, then ascending index); float-vs-double near-ties are handled with an
// epsilon boundary check rather than exact rank equality.

#include "superfaiss/superfaiss.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
	Rng rng(31);
	TestBank bank(rng, 2000, 32, Quantization::Float32, Metric::L2);

	const int32_t m = 100; // crosses the sub-batch boundary (64)
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

	std::printf("superfaiss tests: %d checks, %d failures (simd path: %s)\n",
		GChecks, GFailures,
		ActiveSimdPath() == SimdPath::Scalar ? "scalar"
			: ActiveSimdPath() == SimdPath::SSE ? "sse"
			: ActiveSimdPath() == SimdPath::AVX2 ? "avx2" : "neon");
	return GFailures == 0 ? 0 : 1;
}

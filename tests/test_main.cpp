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

	// V1x (test design A8, review M2 sibling) — immutable-format header caps:
	// hostile header geometry is BadFormat at the validation gate, BEFORE any
	// size arithmetic runs on header-derived values. No payload is allocated
	// here: the reject must fire on the header fields alone.
	{
		AlignedBuf tiny(kAlignment * sizeof(float));
		BankView h;
		h.rows = tiny.F32();
		h.count = 1;
		h.dims = kMaxCrossDeviceDims + 16; // over the format's dims ceiling
		h.paddedDims = PaddedDims(h.dims, Quantization::Float32);
		h.quant = Quantization::Float32;
		h.metric = Metric::Dot;
		CHECK_MSG(ValidateBank(h) == Status::BadFormat, "over-cap dims not hard-rejected");

		h.dims = INT32_MAX; // the PaddedDims signed-overflow window
		h.paddedDims = PaddedDims(h.dims, Quantization::Float32);
		CHECK_MSG(ValidateBank(h) == Status::BadFormat, "INT32_MAX dims not hard-rejected");

		h.dims = kMaxCrossDeviceDims; // the ceiling itself is legal (inclusive bound)
		h.paddedDims = PaddedDims(h.dims, Quantization::Float32);
		CHECK_MSG(ValidateBank(h) == Status::Ok, "at-cap dims wrongly rejected");

		h.dims = 8;
		h.paddedDims = PaddedDims(h.dims, Quantization::Float32);
		h.count = kMaxBankRows + 1; // over the format's row ceiling
		CHECK_MSG(ValidateBank(h) == Status::BadFormat, "over-cap count not hard-rejected");

		h.count = kMaxBankRows; // the ceiling itself is legal (inclusive bound)
		CHECK_MSG(ValidateBank(h) == Status::Ok, "at-cap count wrongly rejected");

		// The importer-side source gate carries the same caps, checkable from the
		// header alone — rejected before any payload exists to point at.
		CHECK_MSG(ValidateSourceRows(nullptr, kMaxBankRows + 1, 1, nullptr) == Status::BadFormat,
			"source-rows over-cap count not hard-rejected");
		CHECK_MSG(ValidateSourceRows(nullptr, 0, kMaxCrossDeviceDims + 16, nullptr) == Status::BadFormat,
			"source-rows over-cap dims not hard-rejected");
		CHECK_MSG(ValidateSourceRows(tiny.F32(), 0, kMaxCrossDeviceDims, nullptr) == Status::Ok,
			"source-rows at-cap dims wrongly rejected");
	}
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

		// Absurd geometry (review M2, the T-062 idiom): a crafted header with an
		// unbounded capacity is BadFormat BEFORE any byte-size arithmetic — a
		// hard format rejection, not an allocator outcome.
		MemArchive huge;
		huge.bytes = ar.bytes;
		const uint32_t absurdCapacity = 0x7FFFFFFFu; // capacity: int32 at offset 8
		std::memcpy(huge.bytes.data() + 8, &absurdCapacity, 4);
		CHECK_MSG(loaded.Load(huge.Reader()) == Status::BadFormat,
			"absurd archive capacity not hard-rejected");
		CHECK(loaded.Count() == s.Count()); // reject-over-degrade held
	}

	// --- R6 extension (review S1/S-1): staleness across Save/Load. A Load is the
	// ultimate mutation — a report taken before it must read STALE after it, and the
	// generation never goes backward and never collides with a previously-issued
	// stamp on the same bank instance. The append-only same-count case is the
	// collision trap: an append-only bank's generation equals its row count, and a
	// Load that resets the generation to the loaded count re-issues exactly that
	// stamp (the flagship's belief-store-across-save-games scenario).
	std::printf("scratch retention (T-V2.3-R6x): staleness across Save/Load\n");
	{
		const int32_t dims = 48;
		Rng rng(0x10ADFull);
		ScratchBank b;
		CHECK(b.Create(64, dims, Metric::Dot, Quantization::Int8, true) == Status::Ok);
		FillSeeded(b, 40, dims, rng); // append-only: no removes, generation == count

		ScratchRecallReport before;
		CHECK(b.MeasureScratchRecall(ws, &before, 0x51A1Eull) == Status::Ok);
		CHECK(!b.RecallReportStale(before));
		const uint64_t stampedGen = b.Generation();

		// The same-count collision case: save, then load the SAME blob back into
		// the same bank object. Same rows, same count — but the report describes
		// pre-load state, so it must read stale, never silently current.
		MemArchive blob;
		CHECK(b.Save(blob.Writer()) == Status::Ok);
		CHECK(b.Load(blob.Reader()) == Status::Ok);
		CHECK_MSG(b.RecallReportStale(before),
			"pre-load report reads CURRENT after a same-count Load");
		// Monotonic, collision-free: the post-load generation exceeds every stamp
		// this instance ever issued.
		CHECK_MSG(b.Generation() > stampedGen,
			"generation went backward/collided across Load: %llu -> %llu",
			static_cast<unsigned long long>(stampedGen),
			static_cast<unsigned long long>(b.Generation()));

		// The mutated case: report, save, mutate, load the pre-mutation blob back
		// — the report must be stale against the restored (pre-mutation) rows too.
		ScratchRecallReport mid;
		CHECK(b.MeasureScratchRecall(ws, &mid, 0x51A1Eull) == Status::Ok);
		CHECK(!b.RecallReportStale(mid));
		MemArchive blob2;
		CHECK(b.Save(blob2.Writer()) == Status::Ok);
		std::vector<float> row(static_cast<size_t>(dims), 0.25f);
		CHECK(b.Append(row.data(), dims, nullptr) == Status::Ok);
		CHECK(b.RecallReportStale(mid)); // stale by the append
		CHECK(b.Load(blob2.Reader()) == Status::Ok);
		CHECK_MSG(b.RecallReportStale(mid),
			"pre-save report reads CURRENT after restoring the pre-mutation blob");

		// A report taken after the Load is current — the staleness is about the
		// boundary, not a permanently-poisoned bank.
		ScratchRecallReport fresh;
		CHECK(b.MeasureScratchRecall(ws, &fresh, 0x51A1Eull) == Status::Ok);
		CHECK(!b.RecallReportStale(fresh));
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
// T-V2.4-P — integer-domain pooling (V2 plan section 21): MakeCentroidCrossDevice
// pools int8 rows into a quantized CrossDevice query (image + scale + self-dot),
// executed bit-for-bit by QueryXd. References are independent test-side routines;
// the golden pooling hash is pinned beside kGoldenXdHash.

namespace pool
{
	// Calibration floor for pooled-query recall (T-V2.4-P9, CAL). Pinned at the first
	// calibration run as a FLOOR (not an exact value), space-version-scoped; 0 means
	// 'not yet pinned' (the suite prints the measured numbers instead of asserting).
	static constexpr float kPoolRecallFloor = 0.90f;

	// DAZ-proof scale decode, test-side (independent of detail::FloatBitsToDouble):
	// subnormal scales decode from the bit pattern via integer math.
	inline double DecodeScale(float s)
	{
		uint32_t b;
		std::memcpy(&b, &s, 4);
		if ((b & 0x7f800000u) != 0)
		{
			return static_cast<double>(s);
		}
		const double m = static_cast<double>(static_cast<int32_t>(b & 0x7fffffu)) *
			1.4012984643248171e-45; // 2^-149
		return (b >> 31) != 0 ? -m : m;
	}

	// Independent float64 value reference (P1): the pooled mean, plain double loops —
	// sum(v_rj * scale_r * w_r) / sum(w_r) per dim. Not the operator's code path.
	inline void RefPoolF64(const BankView& bank, const int32_t* idx, int32_t n,
		const int32_t* w, double* outRef)
	{
		double sumW = 0.0;
		for (int32_t i = 0; i < n; ++i)
		{
			sumW += w != nullptr ? static_cast<double>(w[i]) : 1.0;
		}
		for (int32_t j = 0; j < bank.dims; ++j)
		{
			outRef[j] = 0.0;
		}
		for (int32_t i = 0; i < n; ++i)
		{
			const int32_t row = idx[i];
			const double wi = w != nullptr ? static_cast<double>(w[i]) : 1.0;
			const double s = DecodeScale(bank.scales[row]);
			const int8_t* r = static_cast<const int8_t*>(bank.rows) +
				static_cast<int64_t>(row) * bank.paddedDims;
			for (int32_t j = 0; j < bank.dims; ++j)
			{
				outRef[j] += static_cast<double>(r[j]) * s * wi;
			}
		}
		for (int32_t j = 0; j < bank.dims; ++j)
		{
			outRef[j] /= sumW;
		}
	}

	// Independent normalize-then-quantize bit reference (P2, FAI-6), recoded from the
	// contract, not from compose.cpp. PROOF OBLIGATION (stated): normalize-then-
	// quantize forms n_j = acc_j / L (L = the pooled vector's norm, a positive
	// constant) and quantizes q_j = RHE(n_j * 127 / max|n|); the positive factor 1/L
	// cancels in the EXACT rational — (acc_j / L) * 127 / (maxAcc / L) ==
	// acc_j * 127 / maxAcc — before any rounding happens, so this reference evaluates
	// the normalization symbolically (exact cancellation) and rounds the exact
	// rational with its own integer round-half-even. Any correctly-rounded evaluation
	// of normalize-then-quantize yields these bits; a mismatch against the operator
	// is an operator defect, not float noise.
	struct RefPoolBits
	{
		std::vector<int8_t> q8;
		double scale = 0.0;
		int64_t sqSum = 0;
	};

	inline Status RefNormThenQuant(const BankView& bank, const int32_t* idx, int32_t n,
		const int32_t* w, const uint32_t* excludeBits, RefPoolBits& out)
	{
		// Max included decoded scale and weight sum (independent loops).
		double maxScale = 0.0;
		int64_t sumW = 0;
		int32_t included = 0;
		for (int32_t i = 0; i < n; ++i)
		{
			if (IsExcluded(excludeBits, idx[i]))
			{
				continue;
			}
			++included;
			sumW += w != nullptr ? w[i] : 1;
			const double s = DecodeScale(bank.scales[idx[i]]);
			if (s > maxScale)
			{
				maxScale = s;
			}
		}
		if (included == 0)
		{
			return Status::InvalidArgument;
		}
		if (maxScale == 0.0)
		{
			return Status::ZeroNormQuery;
		}
		// Integer accumulation with the contract's multiplier: w * RHE(ratio * 2^24).
		// std::nearbyint under the default rounding mode (round-to-nearest-even, the
		// library-wide assumption) IS round-half-even — an independent evaluation of
		// the same correctly-rounded quantity.
		std::vector<int64_t> acc(static_cast<size_t>(bank.dims), 0);
		for (int32_t i = 0; i < n; ++i)
		{
			const int32_t row = idx[i];
			if (IsExcluded(excludeBits, row))
			{
				continue;
			}
			const double ratio = DecodeScale(bank.scales[row]) / maxScale;
			const int64_t m = (w != nullptr ? w[i] : 1) *
				static_cast<int64_t>(std::nearbyint(ratio * 16777216.0));
			const int8_t* r = static_cast<const int8_t*>(bank.rows) +
				static_cast<int64_t>(row) * bank.paddedDims;
			for (int32_t j = 0; j < bank.dims; ++j)
			{
				acc[static_cast<size_t>(j)] += static_cast<int64_t>(r[j]) * m;
			}
		}
		int64_t maxAcc = 0;
		for (int32_t j = 0; j < bank.dims; ++j)
		{
			const int64_t mag = acc[static_cast<size_t>(j)] < 0
				? -acc[static_cast<size_t>(j)] : acc[static_cast<size_t>(j)];
			if (mag > maxAcc)
			{
				maxAcc = mag;
			}
		}
		if (maxAcc == 0)
		{
			return Status::ZeroNormQuery;
		}
		// Quantize the exact rational acc*127/maxAcc, round-half-even (own integer
		// implementation — the normalization factor has already cancelled exactly).
		out.q8.assign(static_cast<size_t>(bank.paddedDims), 0);
		out.sqSum = 0;
		for (int32_t j = 0; j < bank.dims; ++j)
		{
			const int64_t num = acc[static_cast<size_t>(j)] * 127;
			const bool neg = num < 0;
			const uint64_t a = neg ? static_cast<uint64_t>(-num) : static_cast<uint64_t>(num);
			const uint64_t d = static_cast<uint64_t>(maxAcc);
			uint64_t q = a / d;
			const uint64_t rem = a % d;
			if (2 * rem > d || (2 * rem == d && (q & 1) != 0))
			{
				++q;
			}
			const int64_t v = neg ? -static_cast<int64_t>(q) : static_cast<int64_t>(q);
			out.q8[static_cast<size_t>(j)] = static_cast<int8_t>(v);
			out.sqSum += v * v;
		}
		out.scale = (static_cast<double>(maxAcc) / static_cast<double>(sumW * 127)) *
			maxScale * (1.0 / 16777216.0);
		return Status::Ok;
	}

	inline bool SameDoubleBits(double a, double b)
	{
		uint64_t x, y;
		std::memcpy(&x, &a, 8);
		std::memcpy(&y, &b, 8);
		return x == y;
	}
} // namespace pool

// P1 (float64 value reference), P2 (normalize-then-quantize bit identity, FAI-6),
// P4 (baked ≡ runtime twin on the quantized representation), P6 (weighted-all-equal
// ≡ unweighted, bitwise).
static void TestPoolCentroidXd()
{
	std::printf("pooling (T-V2.4-P1/P2/P4/P6): references and twins\n");

	xd::FixBank bankA(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
		xdfix::kBankADims, Metric::Dot);
	xd::FixBank bankB(xdfix::kBankBRows, xdfix::kBankBScaleBits, xdfix::kBankBCount,
		xdfix::kBankBDims, Metric::Cosine, xdfix::kBankBChannels);
	xd::FixBank bankC(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
		xdfix::kBankCDims, Metric::L2);
	const BankView* banks[3] = {&bankA.view, &bankB.view, &bankC.view};

	// PoolRowsRef: fixed index/weight sets over the committed fixture banks (indices
	// bounded by the smallest bank, kBankC's 48 rows).
	const int32_t idx[10] = {0, 3, 5, 17, 21, 33, 40, 44, 46, 47};
	const int32_t mixedW[10] = {1, 4, 2, 9, 1, 3, 7, 2, 5, 1};
	const int32_t equalW[10] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

	for (int32_t b = 0; b < 3; ++b)
	{
		const BankView& bank = *banks[b];
		AlignedBuf q8(static_cast<size_t>(bank.paddedDims));
		double scale = 0.0;
		int64_t sqSum = 0;
		CHECK(MakeCentroidCrossDevice(bank, idx, 10, nullptr, nullptr,
				q8.I8(), &scale, &sqSum) == Status::Ok);

		// P1 — value: dequantized output within the quantizer's tolerance of the
		// independent float64 pooled mean. Bound: half a quantization step plus the
		// fixed-point multiplier rounding (<= 0.5 per row on the 2^24 grid).
		{
			std::vector<double> ref(static_cast<size_t>(bank.dims));
			pool::RefPoolF64(bank, idx, 10, nullptr, ref.data());
			double maxScale = 0.0;
			for (int32_t i = 0; i < 10; ++i)
			{
				const double s = pool::DecodeScale(bank.scales[idx[i]]);
				if (s > maxScale)
				{
					maxScale = s;
				}
			}
			const double tol = 0.5 * scale + 127.0 * maxScale / 33554432.0 /* 2^25 */;
			for (int32_t j = 0; j < bank.dims; ++j)
			{
				const double got = static_cast<double>(q8.I8()[j]) * scale;
				CHECK_MSG(std::fabs(got - ref[static_cast<size_t>(j)]) <= tol,
					"bank %d dim %d: pooled %.9g ref %.9g tol %.3g", b, j, got,
					ref[static_cast<size_t>(j)], tol);
			}
			// The set's float mean has a non-unit norm (the FAI-6 case is real here).
			double norm = 0.0;
			for (int32_t j = 0; j < bank.dims; ++j)
			{
				norm += ref[static_cast<size_t>(j)] * ref[static_cast<size_t>(j)];
			}
			CHECK(std::fabs(std::sqrt(norm) - 1.0) > 1e-6);
		}

		// P2 — bit: the operator's direct integer requantization equals the
		// normalize-then-quantize reference byte for byte (FAI-6).
		{
			pool::RefPoolBits ref;
			CHECK(pool::RefNormThenQuant(bank, idx, 10, nullptr, nullptr, ref) == Status::Ok);
			CHECK(std::memcmp(q8.I8(), ref.q8.data(),
					static_cast<size_t>(bank.paddedDims)) == 0);
			CHECK(pool::SameDoubleBits(scale, ref.scale));
			CHECK(sqSum == ref.sqSum);
		}

		// P4 — twin on the quantized representation: the bake-side entry point is a
		// second call to the same operator; over identical rows (here an independent
		// copied-bytes payload stands in for the baked side) the product is
		// byte-identical — one operator, two entry points, no second math.
		{
			AlignedBuf bakedQ8(static_cast<size_t>(bank.paddedDims));
			double bakedScale = 0.0;
			int64_t bakedSq = 0;
			CHECK(MakeCentroidCrossDevice(bank, idx, 10, nullptr, nullptr,
					bakedQ8.I8(), &bakedScale, &bakedSq) == Status::Ok);
			CHECK(std::memcmp(q8.I8(), bakedQ8.I8(),
					static_cast<size_t>(bank.paddedDims)) == 0);
			CHECK(pool::SameDoubleBits(scale, bakedScale) && sqSum == bakedSq);

			AlignedBuf twinRows(static_cast<size_t>(bank.count) * bank.paddedDims);
			std::memcpy(twinRows.ptr, bank.rows,
				static_cast<size_t>(bank.count) * bank.paddedDims);
			std::vector<float> twinScales(bank.scales, bank.scales + bank.count);
			BankView twin = bank;
			twin.rows = twinRows.ptr;
			twin.scales = twinScales.data();
			AlignedBuf twinQ8(static_cast<size_t>(bank.paddedDims));
			double twinScale = 0.0;
			int64_t twinSq = 0;
			CHECK(MakeCentroidCrossDevice(twin, idx, 10, nullptr, nullptr,
					twinQ8.I8(), &twinScale, &twinSq) == Status::Ok);
			CHECK(std::memcmp(q8.I8(), twinQ8.I8(),
					static_cast<size_t>(bank.paddedDims)) == 0);
			CHECK(pool::SameDoubleBits(scale, twinScale) && sqSum == twinSq);
		}

		// P6 — all-equal weights bit-equal unweighted (the common factor cancels
		// under symmetric quantization AND in the scale's exact rational).
		{
			AlignedBuf wQ8(static_cast<size_t>(bank.paddedDims));
			double wScale = 0.0;
			int64_t wSq = 0;
			CHECK(MakeCentroidCrossDevice(bank, idx, 10, equalW, nullptr,
					wQ8.I8(), &wScale, &wSq) == Status::Ok);
			CHECK(std::memcmp(q8.I8(), wQ8.I8(),
					static_cast<size_t>(bank.paddedDims)) == 0);
			CHECK(pool::SameDoubleBits(scale, wScale));
			CHECK(sqSum == wSq);
		}

		// Mixed weights still match both references (P1/P2 under weighting).
		{
			AlignedBuf wQ8(static_cast<size_t>(bank.paddedDims));
			double wScale = 0.0;
			int64_t wSq = 0;
			CHECK(MakeCentroidCrossDevice(bank, idx, 10, mixedW, nullptr,
					wQ8.I8(), &wScale, &wSq) == Status::Ok);
			pool::RefPoolBits ref;
			CHECK(pool::RefNormThenQuant(bank, idx, 10, mixedW, nullptr, ref) == Status::Ok);
			CHECK(std::memcmp(wQ8.I8(), ref.q8.data(),
					static_cast<size_t>(bank.paddedDims)) == 0);
			CHECK(pool::SameDoubleBits(wScale, ref.scale) && wSq == ref.sqSum);

			std::vector<double> vref(static_cast<size_t>(bank.dims));
			pool::RefPoolF64(bank, idx, 10, mixedW, vref.data());
			double maxScale = 0.0;
			for (int32_t i = 0; i < 10; ++i)
			{
				const double s = pool::DecodeScale(bank.scales[idx[i]]);
				if (s > maxScale)
				{
					maxScale = s;
				}
			}
			const double tol = 0.5 * wScale + 127.0 * maxScale / 33554432.0;
			for (int32_t j = 0; j < bank.dims; ++j)
			{
				const double got = static_cast<double>(wQ8.I8()[j]) * wScale;
				CHECK(std::fabs(got - vref[static_cast<size_t>(j)]) <= tol);
			}
		}
	}
}

// P5 (tombstone exclusion, empty set, zero-norm — defined behavior) and P7 (overflow
// at the pinned FAI-5 bound, both sides).
static void TestPoolDefinedBehavior()
{
	std::printf("pooling (T-V2.4-P5/P7): defined rejections and the overflow bound\n");

	// P5.1 — tombstoned rows excluded by the snapshot view: pooling the full index
	// range with the snapshot's tombstone words equals pooling the live rows only.
	{
		Rng rng(0xB00Cull);
		const int32_t dims = 32;
		const int32_t count = 24;
		ScratchBank scratch;
		CHECK(scratch.Create(count, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		std::vector<float> row(static_cast<size_t>(dims));
		for (int32_t r = 0; r < count; ++r)
		{
			for (auto& v : row)
			{
				v = rng.NextFloat();
			}
			CHECK(scratch.Append(row.data(), dims, nullptr) == Status::Ok);
		}
		CHECK(scratch.Remove(2) == Status::Ok);
		CHECK(scratch.Remove(11) == Status::Ok);
		CHECK(scratch.Remove(23) == Status::Ok);
		BankView snap;
		std::vector<uint32_t> tombs(
			static_cast<size_t>(ScratchBank::TombstoneWords(count)), 0u);
		CHECK(scratch.Snapshot(&snap, tombs.data()) == Status::Ok);

		std::vector<int32_t> all(static_cast<size_t>(count));
		std::vector<int32_t> live;
		for (int32_t r = 0; r < count; ++r)
		{
			all[static_cast<size_t>(r)] = r;
			if (!IsExcluded(tombs.data(), r))
			{
				live.push_back(r);
			}
		}
		AlignedBuf a(static_cast<size_t>(snap.paddedDims));
		AlignedBuf b(static_cast<size_t>(snap.paddedDims));
		double sa = 0.0, sb = 0.0;
		int64_t qa = 0, qb = 0;
		CHECK(MakeCentroidCrossDevice(snap, all.data(), count, nullptr, tombs.data(),
				a.I8(), &sa, &qa) == Status::Ok);
		CHECK(MakeCentroidCrossDevice(snap, live.data(),
				static_cast<int32_t>(live.size()), nullptr, nullptr,
				b.I8(), &sb, &qb) == Status::Ok);
		CHECK(std::memcmp(a.I8(), b.I8(), static_cast<size_t>(snap.paddedDims)) == 0);
		CHECK(pool::SameDoubleBits(sa, sb) && qa == qb);

		// P5.2 — empty row set → InvalidArgument, never a zero vector; a selection
		// whose every row is excluded is the same emptiness.
		CHECK(MakeCentroidCrossDevice(snap, all.data(), 0, nullptr, nullptr,
				a.I8(), &sa, &qa) == Status::InvalidArgument);
		const int32_t dead[3] = {2, 11, 23};
		CHECK(MakeCentroidCrossDevice(snap, dead, 3, nullptr, tombs.data(),
				a.I8(), &sa, &qa) == Status::InvalidArgument);
	}

	// P5.3 — zero-norm on the INTEGER accumulator: antipodal rows at equal scale
	// cancel exactly → ZeroNormQuery (parity with MakeCentroid's rejection).
	{
		const int32_t dims = 16;
		AlignedBuf rows(static_cast<size_t>(2) * dims);
		for (int32_t j = 0; j < dims; ++j)
		{
			rows.I8()[j] = static_cast<int8_t>(j - 7);
			rows.I8()[dims + j] = static_cast<int8_t>(-(j - 7));
		}
		const float scales[2] = {0.5f, 0.5f};
		BankView v;
		v.rows = rows.ptr;
		v.scales = scales;
		v.count = 2;
		v.dims = dims;
		v.paddedDims = dims;
		v.quant = Quantization::Int8;
		v.metric = Metric::Dot;
		const int32_t both[2] = {0, 1};
		AlignedBuf q8(static_cast<size_t>(dims));
		double s = 0.0;
		int64_t sq = 0;
		CHECK(MakeCentroidCrossDevice(v, both, 2, nullptr, nullptr,
				q8.I8(), &s, &sq) == Status::ZeroNormQuery);

		// All-zero scales: every row dequantizes to zero → ZeroNormQuery too.
		const float zeroScales[2] = {0.0f, 0.0f};
		v.scales = zeroScales;
		CHECK(MakeCentroidCrossDevice(v, both, 2, nullptr, nullptr,
				q8.I8(), &s, &sq) == Status::ZeroNormQuery);

		// Remaining defined rejections: f32 bank (CrossDevice is int8-only),
		// non-positive weight, out-of-range index.
		v.scales = scales;
		BankView f32 = v;
		f32.quant = Quantization::Float32;
		CHECK(MakeCentroidCrossDevice(f32, both, 2, nullptr, nullptr,
				q8.I8(), &s, &sq) == Status::InvalidArgument);
		const int32_t zeroW[2] = {1, 0};
		CHECK(MakeCentroidCrossDevice(v, both, 2, zeroW, nullptr,
				q8.I8(), &s, &sq) == Status::InvalidArgument);
		const int32_t bad[2] = {0, 2};
		CHECK(MakeCentroidCrossDevice(v, bad, 2, nullptr, nullptr,
				q8.I8(), &s, &sq) == Status::InvalidArgument);
	}

	// P7 — the FAI-5 bound, both sides. Max passing: sum(w) == kMaxPooledRows of
	// worst-case all-+-127 rows at equal (max) scale accumulates without overflow and
	// equals the scalar expectation; first failing: one more is InvalidArgument. The
	// pool reuses row INDICES (a 2^20-entry index array over a 4-row bank).
	{
		const int32_t dims = 16;
		AlignedBuf rows(static_cast<size_t>(4) * dims);
		for (int32_t r = 0; r < 4; ++r)
		{
			for (int32_t j = 0; j < dims; ++j)
			{
				// Alternating sign per dim, identical across rows: every accumulator
				// hits the worst-case magnitude sum(w) * 127 * 2^24.
				rows.I8()[r * dims + j] = static_cast<int8_t>((j & 1) ? -127 : 127);
			}
		}
		const float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		BankView v;
		v.rows = rows.ptr;
		v.scales = scales;
		v.count = 4;
		v.dims = dims;
		v.paddedDims = dims;
		v.quant = Quantization::Int8;
		v.metric = Metric::Dot;

		const int32_t maxRows = static_cast<int32_t>(kMaxPooledRows);
		std::vector<int32_t> idx(static_cast<size_t>(maxRows) + 1);
		for (size_t i = 0; i < idx.size(); ++i)
		{
			idx[i] = static_cast<int32_t>(i & 3);
		}
		AlignedBuf q8(static_cast<size_t>(dims));
		double s = 0.0;
		int64_t sq = 0;
		CHECK(MakeCentroidCrossDevice(v, idx.data(), maxRows, nullptr, nullptr,
				q8.I8(), &s, &sq) == Status::Ok);
		// Scalar expectation at the bound: every |acc_j| equals maxAcc, so the image
		// is exactly +-127 per dim and the scale is exactly 1.0 (2^20*127*2^24 over
		// 2^20*127, on the 2^24 grid) — any int64 wrap would break both.
		for (int32_t j = 0; j < dims; ++j)
		{
			CHECK(q8.I8()[j] == ((j & 1) ? -127 : 127));
		}
		CHECK(pool::SameDoubleBits(s, 1.0));
		CHECK(sq == static_cast<int64_t>(dims) * 127 * 127);

		// First failing, unweighted: maxPooledRows + 1 rows.
		CHECK(MakeCentroidCrossDevice(v, idx.data(), maxRows + 1, nullptr, nullptr,
				q8.I8(), &s, &sq) == Status::InvalidArgument);

		// Same bound through the weighted form: sum(w) == bound passes, + 1 rejects.
		const int32_t idx4[4] = {0, 1, 2, 3};
		const int32_t wAt[4] = {maxRows / 4, maxRows / 4, maxRows / 4, maxRows / 4};
		CHECK(MakeCentroidCrossDevice(v, idx4, 4, wAt, nullptr,
				q8.I8(), &s, &sq) == Status::Ok);
		const int32_t wOver[4] = {maxRows / 4, maxRows / 4, maxRows / 4, maxRows / 4 + 1};
		CHECK(MakeCentroidCrossDevice(v, idx4, 4, wOver, nullptr,
				q8.I8(), &s, &sq) == Status::InvalidArgument);
	}
}

// P3 (cross-ISA bit-identity with the forced-path sweep, adversarial tiny-scale
// included) and P8 (golden pooled-query hash, capture at first green).
static void TestPoolXdSweep()
{
	std::printf("pooling (T-V2.4-P3/P8): forced-path sweep and the golden hash\n");
	Workspace ws;

	xd::FixBank bankA(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
		xdfix::kBankADims, Metric::Dot);
	xd::FixBank bankB(xdfix::kBankBRows, xdfix::kBankBScaleBits, xdfix::kBankBCount,
		xdfix::kBankBDims, Metric::Cosine, xdfix::kBankBChannels);
	xd::FixBank bankC(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
		xdfix::kBankCDims, Metric::L2);
	const BankView* banks[3] = {&bankA.view, &bankB.view, &bankC.view};

	// PoolAdversarialTinyScale: committed in-test constants — subnormal and
	// FLT_MIN-neighborhood per-row scales push epilogue products into the flush
	// window (the section 19.2 S1 pattern applied to pooling).
	AlignedBuf advRows(static_cast<size_t>(8) * 16);
	float advScales[8];
	{
		const int8_t pattern[16] = {127, -80, 33, -127, 5, 91, -64, 17,
			-2, 118, -45, 77, -101, 26, -9, 54};
		for (int32_t r = 0; r < 8; ++r)
		{
			for (int32_t j = 0; j < 16; ++j)
			{
				advRows.I8()[r * 16 + j] =
					static_cast<int8_t>(pattern[(j + r) & 15] - (r & 3));
			}
		}
		const uint32_t bits[8] = {0x00000007u, 0x00000123u, 0x00800000u,
			0x00000350u, 0x000fffffu, 0x00400000u, 0x00000001u, 0x007fffffu};
		for (int32_t r = 0; r < 8; ++r)
		{
			std::memcpy(&advScales[r], &bits[r], 4);
		}
	}
	BankView adv;
	adv.rows = advRows.ptr;
	adv.scales = advScales;
	adv.count = 8;
	adv.dims = 16;
	adv.paddedDims = 16;
	adv.quant = Quantization::Int8;
	adv.metric = Metric::Dot;

	const int32_t idxA[10] = {0, 3, 5, 17, 21, 33, 40, 44, 46, 47};
	const int32_t mixedW[10] = {1, 4, 2, 9, 1, 3, 7, 2, 5, 1};
	const int32_t idxAdv[6] = {0, 2, 3, 5, 6, 7};

	// The battery under the current dispatch: pooled products (image bytes, scale
	// bits, self-dot) and their QueryXd hit lists, hashed.
	auto runBattery = [&]() -> uint64_t {
		xd::BatteryHash hash;
		Hit hits[10];
		int32_t n = 0;
		for (int32_t b = 0; b < 3; ++b)
		{
			const BankView& bank = *banks[b];
			for (int32_t variant = 0; variant < 2; ++variant)
			{
				AlignedBuf q8(static_cast<size_t>(bank.paddedDims));
				double scale = 0.0;
				int64_t sqSum = 0;
				CHECK(MakeCentroidCrossDevice(bank, idxA, 10,
						variant == 0 ? nullptr : mixedW, nullptr,
						q8.I8(), &scale, &sqSum) == Status::Ok);
				hash.Bytes(q8.I8(), static_cast<size_t>(bank.paddedDims));
				hash.Bytes(&scale, 8);
				hash.Bytes(&sqSum, 8);

				XdQuery xq{q8.I8(), scale, sqSum};
				QueryParams p;
				p.k = 10;
				p.exactness = Exactness::CrossDevice;
				CHECK(QueryXd(bank, xq, p, ws, hits, &n) == Status::Ok);
				xd::CheckFloor(hits, n);
				hash.Hits(hits, n);

				// With exclusion: mask the pooled members themselves.
				std::vector<uint32_t> exclude(
					static_cast<size_t>((bank.count + 31) / 32), 0u);
				for (int32_t i = 0; i < 10; ++i)
				{
					exclude[static_cast<size_t>(idxA[i] >> 5)] |= 1u << (idxA[i] & 31);
				}
				p.excludeBits = exclude.data();
				CHECK(QueryXd(bank, xq, p, ws, hits, &n) == Status::Ok);
				for (int32_t i = 0; i < n; ++i)
				{
					CHECK(!IsExcluded(exclude.data(), hits[i].index));
				}
				xd::CheckFloor(hits, n);
				hash.Hits(hits, n);
			}
		}
		// Adversarial tiny-scale pool: the flush window under pooling (P3 INV; part
		// of the same battery so the sweep covers it).
		{
			AlignedBuf q8(static_cast<size_t>(adv.paddedDims));
			double scale = 0.0;
			int64_t sqSum = 0;
			CHECK(MakeCentroidCrossDevice(adv, idxAdv, 6, nullptr, nullptr,
					q8.I8(), &scale, &sqSum) == Status::Ok);
			hash.Bytes(q8.I8(), static_cast<size_t>(adv.paddedDims));
			hash.Bytes(&scale, 8);
			hash.Bytes(&sqSum, 8);
			XdQuery xq{q8.I8(), scale, sqSum};
			QueryParams p;
			p.k = 8;
			p.exactness = Exactness::CrossDevice;
			CHECK(QueryXd(adv, xq, p, ws, hits, &n) == Status::Ok);
			xd::CheckFloor(hits, n);
			hash.Hits(hits, n);
		}
		return hash.h;
	};

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
				"pool forced path %d hash %llx != path %d hash %llx",
				static_cast<int>(paths[i]),
				static_cast<unsigned long long>(hashes[i]),
				static_cast<int>(paths[0]),
				static_cast<unsigned long long>(hashes[0]));
		}
	}
	const uint64_t battery = runBattery(); // default dispatch matches the sweep
	CHECK(battery == hashes[0]);
	std::printf("pooled-query cross-device hash: %016llx (%d forced paths agree)\n",
		static_cast<unsigned long long>(battery), static_cast<int>(paths.size()));
	if constexpr (xdfix::kGoldenPoolXdHash != 0)
	{
		CHECK_MSG(battery == xdfix::kGoldenPoolXdHash,
			"pooled-query hash %016llx != pinned golden %016llx",
			static_cast<unsigned long long>(battery),
			static_cast<unsigned long long>(xdfix::kGoldenPoolXdHash));
	}
}

// T-V2.4-P12 — batch composition closure (plan section 21 "composes with ... batch"):
// QueryXdBatch over M pre-quantized pooled queries bit-equals the same members run
// through single QueryXd — every count, hit index, and score bit — across the
// forced-path sweep, with per-query bias (dense and sparse members) in the batch and
// an adversarial tiny-scale pooled member included. The batch is not hashed
// separately: bitwise equality to the already-pinned singles covers it, so
// kGoldenPoolXdHash is untouched by this entry.
static void TestPoolXdBatch()
{
	std::printf("pooling (T-V2.4-P12): batch == singles, bitwise\n");
	Workspace batchWs;
	Workspace singleWs;

	xd::FixBank bankA(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
		xdfix::kBankADims, Metric::Dot);
	xd::FixBank bankB(xdfix::kBankBRows, xdfix::kBankBScaleBits, xdfix::kBankBCount,
		xdfix::kBankBDims, Metric::Cosine, xdfix::kBankBChannels);
	xd::FixBank bankC(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
		xdfix::kBankCDims, Metric::L2);

	// The adversarial tiny-scale bank (the P3 fixture, rebuilt from the same
	// committed constants) so a flush-window pooled member rides the batch too.
	AlignedBuf advRows(static_cast<size_t>(8) * 16);
	float advScales[8];
	{
		const int8_t pattern[16] = {127, -80, 33, -127, 5, 91, -64, 17,
			-2, 118, -45, 77, -101, 26, -9, 54};
		for (int32_t r = 0; r < 8; ++r)
		{
			for (int32_t j = 0; j < 16; ++j)
			{
				advRows.I8()[r * 16 + j] =
					static_cast<int8_t>(pattern[(j + r) & 15] - (r & 3));
			}
		}
		const uint32_t bits[8] = {0x00000007u, 0x00000123u, 0x00800000u,
			0x00000350u, 0x000fffffu, 0x00400000u, 0x00000001u, 0x007fffffu};
		for (int32_t r = 0; r < 8; ++r)
		{
			std::memcpy(&advScales[r], &bits[r], 4);
		}
	}
	BankView adv;
	adv.rows = advRows.ptr;
	adv.scales = advScales;
	adv.count = 8;
	adv.dims = 16;
	adv.paddedDims = 16;
	adv.quant = Quantization::Int8;
	adv.metric = Metric::Dot;

	const BankView* banks[4] = {&bankA.view, &bankB.view, &bankC.view, &adv};

	// Batch == singles under every forced path, per bank.
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

	for (size_t pathI = 0; pathI < paths.size(); ++pathI)
	{
		detail::ForceXdSimdPath(paths[pathI]);
		for (int32_t b = 0; b < 4; ++b)
		{
			const BankView& bank = *banks[b];
			// Five pooled members over distinct index subsets (bounded by the
			// smallest bank's 8 rows for the adversarial case), one weighted.
			const int32_t sets[5][4] = {
				{0, 2, 3, 5}, {1, 4, 6, 7}, {0, 1, 2, 3}, {2, 5, 6, 7}, {0, 3, 4, 6}};
			const int32_t setW[4] = {3, 1, 5, 2};
			const int32_t batchN = 5;

			AlignedBuf image0(static_cast<size_t>(bank.paddedDims));
			AlignedBuf image1(static_cast<size_t>(bank.paddedDims));
			AlignedBuf image2(static_cast<size_t>(bank.paddedDims));
			AlignedBuf image3(static_cast<size_t>(bank.paddedDims));
			AlignedBuf image4(static_cast<size_t>(bank.paddedDims));
			AlignedBuf* images[5] = {&image0, &image1, &image2, &image3, &image4};
			XdQuery queries[5];
			for (int32_t m = 0; m < batchN; ++m)
			{
				double scale = 0.0;
				int64_t sqSum = 0;
				CHECK(MakeCentroidCrossDevice(bank, sets[m], 4,
						m == 2 ? setW : nullptr, nullptr,
						images[m]->I8(), &scale, &sqSum) == Status::Ok);
				queries[m] = XdQuery{images[m]->I8(), scale, sqSum};
			}

			// Per-query bias, the QueryBatch convention: member 0 dense (with a
			// subnormal value in it), member 1 sparse pairs, members 2..4 unbiased.
			std::vector<float> dense(static_cast<size_t>(bank.count), 0.0f);
			for (int32_t r = 0; r < bank.count; ++r)
			{
				const uint32_t bits = (r % 3 == 0) ? 0x00000007u : 0x3c23d70au;
				std::memcpy(&dense[static_cast<size_t>(r)], &bits, 4);
			}
			const BiasPair pairs[2] = {{1, 0.5f}, {6, -0.25f}};
			RowBias biases[5];
			biases[0].dense = dense.data();
			biases[1].pairs = pairs;
			biases[1].pairCount = 2;

			// A shared exclusion mask (every third row).
			std::vector<uint32_t> exclude(
				static_cast<size_t>((bank.count + 31) / 32), 0u);
			for (int32_t r = 0; r < bank.count; r += 3)
			{
				exclude[static_cast<size_t>(r >> 5)] |= 1u << (r & 31);
			}

			QueryParams p;
			p.k = 6;
			p.exactness = Exactness::CrossDevice;
			p.excludeBits = exclude.data();
			p.bias = biases;

			Hit batchHits[5 * 6];
			int32_t batchCounts[5] = {};
			CHECK(QueryXdBatch(bank, queries, batchN, p, batchWs,
					batchHits, batchCounts) == Status::Ok);

			for (int32_t m = 0; m < batchN; ++m)
			{
				QueryParams sp = p;
				sp.bias = &biases[m]; // the member's own bias, single-query form
				Hit single[6];
				int32_t n = 0;
				CHECK(QueryXd(bank, queries[m], sp, singleWs, single, &n) == Status::Ok);
				CHECK_MSG(n == batchCounts[m], "bank %d member %d: batch %d single %d",
					b, m, batchCounts[m], n);
				for (int32_t i = 0; i < n && i < batchCounts[m]; ++i)
				{
					const Hit& bh = batchHits[static_cast<int64_t>(m) * p.k + i];
					uint32_t bb, sb;
					std::memcpy(&bb, &bh.score, 4);
					std::memcpy(&sb, &single[i].score, 4);
					CHECK_MSG(bh.index == single[i].index && bb == sb,
						"bank %d member %d hit %d: batch %d/%08x single %d/%08x",
						b, m, i, bh.index, bb, single[i].index, sb);
					CHECK(!IsExcluded(exclude.data(), bh.index));
				}
				xd::CheckFloor(batchHits + static_cast<int64_t>(m) * p.k, batchCounts[m]);
			}
		}
		detail::ClearForcedXdSimdPath();
	}

	// Defined rejections, the QueryXd law set applied to the batch form.
	{
		AlignedBuf q8(static_cast<size_t>(bankA.view.paddedDims));
		double scale = 0.0;
		int64_t sqSum = 0;
		const int32_t idx[4] = {0, 2, 3, 5};
		CHECK(MakeCentroidCrossDevice(bankA.view, idx, 4, nullptr, nullptr,
				q8.I8(), &scale, &sqSum) == Status::Ok);
		XdQuery one{q8.I8(), scale, sqSum};
		Hit hits[6];
		int32_t counts[1] = {};
		QueryParams p;
		p.k = 6;
		p.exactness = Exactness::PerDevice;
		CHECK(QueryXdBatch(bankA.view, &one, 1, p, batchWs, hits, counts) ==
			Status::InvalidArgument);
		p.exactness = Exactness::CrossDevice;
		const QuerySegment seg[1] = {{0, 16, 1.0f}};
		p.segments = seg;
		p.segmentCount = 1;
		CHECK(QueryXdBatch(bankA.view, &one, 1, p, batchWs, hits, counts) ==
			Status::InvalidArgument);
		p.segments = nullptr;
		p.segmentCount = 0;
		CHECK(QueryXdBatch(bankA.view, nullptr, 1, p, batchWs, hits, counts) ==
			Status::InvalidArgument);
		// Empty batch is a defined no-op.
		CHECK(QueryXdBatch(bankA.view, &one, 0, p, batchWs, hits, counts) == Status::Ok);
	}
}

// P9 (pooled-query recall vs the float64 pooled reference — the FAI-3 honesty
// number, calibration entry) and P11 (carried contracts: zero steady-state
// allocation, reader-pin composition, determinism-given-history).
static void TestPoolRecallAndContracts()
{
	std::printf("pooling (T-V2.4-P9/P11): recall cost and carried contracts\n");
	Workspace ws;

	// --- P9: recall@10 of the pooled CrossDevice query vs the float64 pooled
	// reference's top-10, seeded protocol (B2), per metric. ---
	{
		const uint64_t seed = 0x5EEDF00DCAFEBEEFull;
		int32_t metricTag = 0;
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			Rng bankRng(0x9001ull + static_cast<uint64_t>(metricTag));
			TestBank bank(bankRng, 512, 128, Quantization::Int8, metric);
			Rng sampleRng(seed);
			const int32_t samples = 200;
			int64_t hitsTotal = 0;
			int64_t possible = 0;
			for (int32_t s = 0; s < samples; ++s)
			{
				int32_t idx[8];
				for (int32_t i = 0; i < 8; ++i)
				{
					idx[i] = sampleRng.NextIndex(bank.view.count);
				}
				AlignedBuf q8(static_cast<size_t>(bank.view.paddedDims));
				double scale = 0.0;
				int64_t sqSum = 0;
				const Status ps = MakeCentroidCrossDevice(bank.view, idx, 8, nullptr,
					nullptr, q8.I8(), &scale, &sqSum);
				if (ps == Status::ZeroNormQuery)
				{
					continue; // antipodal draw: defined, not sampled
				}
				CHECK(ps == Status::Ok);

				// Reference: the float64 pooled mean scored in double against the
				// same dequantized rows the kernel scores (refRows), top-10 in the
				// library's total order.
				std::vector<double> refQ(static_cast<size_t>(bank.view.dims));
				pool::RefPoolF64(bank.view, idx, 8, nullptr, refQ.data());
				std::vector<RefHit> ref;
				for (int32_t r = 0; r < bank.view.count; ++r)
				{
					const double* row =
						bank.refRows.data() + static_cast<size_t>(r) * bank.view.dims;
					double score = 0.0;
					if (metric == Metric::L2)
					{
						for (int32_t j = 0; j < bank.view.dims; ++j)
						{
							const double d = refQ[static_cast<size_t>(j)] - row[j];
							score += d * d;
						}
					}
					else
					{
						for (int32_t j = 0; j < bank.view.dims; ++j)
						{
							score += refQ[static_cast<size_t>(j)] * row[j];
						}
					}
					ref.push_back({r, score});
				}
				std::sort(ref.begin(), ref.end(), [&](const RefHit& a, const RefHit& b) {
					return RefBetter(a, b, metric);
				});

				XdQuery xq{q8.I8(), scale, sqSum};
				QueryParams p;
				p.k = 10;
				p.exactness = Exactness::CrossDevice;
				Hit hits[10];
				int32_t n = 0;
				CHECK(QueryXd(bank.view, xq, p, ws, hits, &n) == Status::Ok);
				for (int32_t i = 0; i < n; ++i)
				{
					for (int32_t j = 0; j < 10; ++j)
					{
						if (hits[i].index == ref[static_cast<size_t>(j)].index)
						{
							++hitsTotal;
							break;
						}
					}
				}
				possible += 10;
			}
			const double recall =
				static_cast<double>(hitsTotal) / static_cast<double>(possible);
			std::printf(
				"  pooled recall@10 (metric %d): %.4f (samples=%d, seed=%016llx)\n",
				metricTag, recall, samples, static_cast<unsigned long long>(seed));
			if constexpr (pool::kPoolRecallFloor > 0.0f)
			{
				CHECK_MSG(recall >= static_cast<double>(pool::kPoolRecallFloor),
					"pooled recall floor breached (metric %d): %.4f < %.4f", metricTag,
					recall, static_cast<double>(pool::kPoolRecallFloor));
			}
			++metricTag;
		}
	}

	// --- P11: carried contracts. ---
	{
		Rng rng(0xA110Cull);
		TestBank bank(rng, 256, 64, Quantization::Int8, Metric::Dot);
		const int32_t idx[6] = {1, 20, 63, 100, 180, 255};
		AlignedBuf q8(static_cast<size_t>(bank.view.paddedDims));
		double scale = 0.0;
		int64_t sqSum = 0;
		Hit hits[10];
		int32_t n = 0;
		QueryParams p;
		p.k = 10;
		p.exactness = Exactness::CrossDevice;

		// Zero steady-state allocation: warm one pool+query cycle, then flat across
		// both counters (the operator itself is stack-only by construction).
		CHECK(MakeCentroidCrossDevice(bank.view, idx, 6, nullptr, nullptr,
				q8.I8(), &scale, &sqSum) == Status::Ok);
		XdQuery xq{q8.I8(), scale, sqSum};
		CHECK(QueryXd(bank.view, xq, p, ws, hits, &n) == Status::Ok);
		const uint64_t allocs = AllocationCount();
		const uint64_t growth = ws.GrowthCount();
		for (int32_t i = 0; i < 16; ++i)
		{
			CHECK(MakeCentroidCrossDevice(bank.view, idx, 6, nullptr, nullptr,
					q8.I8(), &scale, &sqSum) == Status::Ok);
			CHECK(QueryXd(bank.view, xq, p, ws, hits, &n) == Status::Ok);
		}
		CHECK_MSG(AllocationCount() == allocs, "warm pooling allocated");
		CHECK(ws.GrowthCount() == growth);

		// Reader-pin composition + determinism-given-history: two identical scratch
		// histories, pooled under the pin, produce byte-identical products.
		ScratchBank s1;
		ScratchBank s2;
		Rng r1(0xD17Aull);
		Rng r2(0xD17Aull);
		const int32_t dims = 32;
		CHECK(s1.Create(32, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		CHECK(s2.Create(32, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		std::vector<float> row(static_cast<size_t>(dims));
		for (int32_t r = 0; r < 24; ++r)
		{
			for (auto& v : row)
			{
				v = r1.NextFloat();
			}
			CHECK(s1.Append(row.data(), dims, nullptr) == Status::Ok);
			for (auto& v : row)
			{
				v = r2.NextFloat();
			}
			CHECK(s2.Append(row.data(), dims, nullptr) == Status::Ok);
		}
		CHECK(s1.Remove(5) == Status::Ok);
		CHECK(s2.Remove(5) == Status::Ok);

		auto poolPinned = [&](ScratchBank& sb, int8_t* outQ8, double* outS,
							  int64_t* outSq) {
			CHECK(sb.TryPinReader());
			BankView snap;
			std::vector<uint32_t> tombs(
				static_cast<size_t>(ScratchBank::TombstoneWords(sb.Capacity())), 0u);
			CHECK(sb.Snapshot(&snap, tombs.data()) == Status::Ok);
			std::vector<int32_t> all(static_cast<size_t>(snap.count));
			for (int32_t r = 0; r < snap.count; ++r)
			{
				all[static_cast<size_t>(r)] = r;
			}
			CHECK(MakeCentroidCrossDevice(snap, all.data(), snap.count, nullptr,
					tombs.data(), outQ8, outS, outSq) == Status::Ok);
			sb.UnpinReader();
		};
		const int32_t pd = PaddedDims(dims, Quantization::Int8);
		AlignedBuf qa(static_cast<size_t>(pd));
		AlignedBuf qb(static_cast<size_t>(pd));
		double sa = 0.0, sb2 = 0.0;
		int64_t ka = 0, kb = 0;
		poolPinned(s1, qa.I8(), &sa, &ka);
		poolPinned(s2, qb.I8(), &sb2, &kb);
		CHECK(std::memcmp(qa.I8(), qb.I8(), static_cast<size_t>(pd)) == 0);
		CHECK(pool::SameDoubleBits(sa, sb2) && ka == kb);

		// QueryXd defined rejections: PerDevice mode, segments, f32 bank, zero
		// self-dot on a Cosine bank.
		QueryParams bad = p;
		bad.exactness = Exactness::PerDevice;
		CHECK(QueryXd(bank.view, xq, bad, ws, hits, &n) == Status::InvalidArgument);
		const QuerySegment seg[1] = {{0, 16, 1.0f}};
		bad = p;
		bad.segments = seg;
		bad.segmentCount = 1;
		CHECK(QueryXd(bank.view, xq, bad, ws, hits, &n) == Status::InvalidArgument);
		Rng frng(0xF32ull);
		TestBank fbank(frng, 16, 32, Quantization::Float32, Metric::Dot);
		CHECK(QueryXd(fbank.view, xq, p, ws, hits, &n) == Status::InvalidArgument);
		Rng crng(0xC05ull);
		TestBank cbank(crng, 16, 64, Quantization::Int8, Metric::Cosine);
		AlignedBuf zq(static_cast<size_t>(cbank.view.paddedDims));
		XdQuery zero{zq.I8(), 0.0, 0};
		CHECK(QueryXd(cbank.view, zero, p, ws, hits, &n) == Status::ZeroNormQuery);

		// --- Adversarial payloads (review S2/M1, Japp S-2): no field of a
		// caller-provided XdQuery may make scores or rankings ill-defined or
		// silently wrong. The contract is a FINITE non-negative scale and a
		// self-dot that IS the image's — anything else is InvalidArgument, single
		// and batch alike. The honest pipeline never emits these; a hand-edited or
		// corrupted payload is the threat model (the T-062 class).
		{
			const int64_t honestSq = xq.sqSum;
			const double inf = std::numeric_limits<double>::infinity();
			const double nan = std::numeric_limits<double>::quiet_NaN();

			// Non-finite and negative scales.
			XdQuery adv = xq;
			adv.scale = inf;
			CHECK_MSG(QueryXd(bank.view, adv, p, ws, hits, &n) == Status::InvalidArgument,
				"+inf scale admitted (single)");
			adv.scale = nan;
			CHECK(QueryXd(bank.view, adv, p, ws, hits, &n) == Status::InvalidArgument);
			adv.scale = -1.0;
			CHECK(QueryXd(bank.view, adv, p, ws, hits, &n) == Status::InvalidArgument);

			// A lying self-dot: too high, too low, and zero-on-nonzero-image.
			adv = xq;
			adv.sqSum = honestSq + 1;
			CHECK_MSG(QueryXd(bank.view, adv, p, ws, hits, &n) == Status::InvalidArgument,
				"inflated sqSum admitted (single)");
			adv.sqSum = honestSq > 0 ? honestSq - 1 : 1;
			CHECK(QueryXd(bank.view, adv, p, ws, hits, &n) == Status::InvalidArgument);
			adv.sqSum = 0; // nonzero image with a zero self-dot: a desynced payload,
			               // an integrity rejection — never ZeroNormQuery
			CHECK_MSG(QueryXd(bank.view, adv, p, ws, hits, &n) == Status::InvalidArgument,
				"zero sqSum on a nonzero image admitted (single)");

			// The batch mirrors every rejection per member: one bad member rejects
			// the batch (validation-before-work, the QueryBatch discipline).
			XdQuery pair[2] = {xq, xq};
			Hit bHits[2 * 10];
			int32_t bCounts[2] = {};
			CHECK(QueryXdBatch(bank.view, pair, 2, p, ws, bHits, bCounts) == Status::Ok);
			pair[1].scale = inf;
			CHECK_MSG(QueryXdBatch(bank.view, pair, 2, p, ws, bHits, bCounts) ==
					Status::InvalidArgument, "+inf scale admitted (batch)");
			pair[1].scale = nan;
			CHECK(QueryXdBatch(bank.view, pair, 2, p, ws, bHits, bCounts) ==
				Status::InvalidArgument);
			pair[1] = xq;
			pair[1].sqSum = honestSq + 1;
			CHECK_MSG(QueryXdBatch(bank.view, pair, 2, p, ws, bHits, bCounts) ==
					Status::InvalidArgument, "inflated sqSum admitted (batch)");
			pair[1].sqSum = 0;
			CHECK(QueryXdBatch(bank.view, pair, 2, p, ws, bHits, bCounts) ==
				Status::InvalidArgument);

			// The honest payload still executes after all of the above (the
			// adversarial rejections leave the valid path untouched).
			CHECK(QueryXd(bank.view, xq, p, ws, hits, &n) == Status::Ok);
		}
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

// ===================================================================================
// T-XDC — CrossDevice composition feature oracles (Test Design SS6A, amendment A9/A10).
//
// J-1 (Japp feature-oracle re-audit): the composed cross-device surface (per-row bias
// dense+sparse, segments, intersect) was proven only by the golden battery hash + batch
// == singles — consistency oracles. A wrong-but-deterministic composition epilogue
// passes them. These suites add the missing FEATURE ORACLES:
//   REF  — a reference recoded from the API/DETERMINISM contract (NOT from kernels.cpp),
//          bit-exact to the kernel's composed score at full k. The primary,
//          wrong-but-deterministic catch. Required green.
//   FEAT — recall@10 of the composed query's top-k vs an INDEPENDENT double brute-force
//          of the SAME composition over the dequantized rows. The neighbour-recovery
//          honesty number; reported here (red-uncalibrated per SS4 until a calibration
//          event pins its floor), not asserted.
// Forge temper 2026-07-11: W1 (bias is the double-domain floored add), W3 (segments —
// exact-match channel rule, bit-decode, segment order, single floor), S2 (sparse bias
// tested at k<count to exercise the k+P re-selection).
namespace xdc
{
	// --- primitives recoded from DETERMINISM.md SS2c (independent of kernels.cpp) ---

	// DAZ-proof decode of an IEEE float bit pattern to double: subnormals as
	// sign*mant*2^-149, normals a plain cast. The SS2c float-input decode.
	inline double BitsToDouble(float f)
	{
		uint32_t b;
		std::memcpy(&b, &f, 4);
		if ((b & 0x7f800000u) != 0)
		{
			return static_cast<double>(f);
		}
		const double m = static_cast<double>(static_cast<int32_t>(b & 0x7fffffu)) *
			1.4012984643248171e-45; // 2^-149
		return (b >> 31) != 0 ? -m : m;
	}

	// The subnormal floor: |x| < FLT_MIN -> exactly 0.0f, else the single cast (SS2c).
	inline float Floor(double d)
	{
		const double lim = 1.1754943508222875e-38;
		if (d < lim && d > -lim)
		{
			return 0.0f;
		}
		return static_cast<float>(d);
	}

	// W1: bias composition recoded from contract — one fused add, UNCONDITIONAL (the
	// reward's sign is the caller's value; the reference does NOT branch on metric),
	// applied to the floored unbiased score in the double domain, then re-floored.
	inline float ComposeBias(float unbiased, float bias)
	{
		return Floor(static_cast<double>(unbiased) + BitsToDouble(bias));
	}

	// W3: segmented CrossDevice row score, recoded from the contract. One contribution
	// per segment in ascending order (the kernel builds one range per segment; it does
	// NOT split a segment across channels). Per segment: the dot/L2 double epilogue over
	// [offset,length); then the per-channel inverse sub-norm ONLY when the segment's
	// offset AND length exactly match a channel (a spanning/partial segment gets none);
	// then the weight. Weights and inv-norms bit-decoded. Accumulate in double; the
	// caller floors the total once. Weight-0 segments are omitted (discarded like a gap).
	inline double SegmentedRowD(const BankView& bank, const int8_t* q8, double qScale,
		const QuerySegment* segs, int32_t segCount, int32_t r)
	{
		const int8_t* row = static_cast<const int8_t*>(bank.rows) +
			static_cast<int64_t>(r) * bank.paddedDims;
		const double rowScale = BitsToDouble(bank.scales[r]);
		const bool isL2 = bank.metric == Metric::L2;
		const bool perChannel =
			bank.metric == Metric::Cosine && bank.channelInvNorms != nullptr;
		double total = 0.0;
		for (int32_t s = 0; s < segCount; ++s)
		{
			const QuerySegment& seg = segs[s];
			if (seg.weight == 0.0f)
			{
				continue; // weight-0 omitted, contributes nothing (SS2c)
			}
			double partial;
			if (isL2)
			{
				int64_t cross = 0, rowSq = 0, qSq = 0;
				for (int32_t i = 0; i < seg.length; ++i)
				{
					const int32_t o = seg.offset + i;
					cross += static_cast<int64_t>(row[o]) * q8[o];
					rowSq += static_cast<int64_t>(row[o]) * row[o];
					qSq += static_cast<int64_t>(q8[o]) * q8[o];
				}
				const double a = (qScale * qScale) * static_cast<double>(qSq);
				const double b = (rowScale * rowScale) * static_cast<double>(rowSq);
				const double c = ((rowScale * qScale) * static_cast<double>(cross)) * 2.0;
				partial = (a + b) - c;
			}
			else
			{
				int64_t acc = 0;
				for (int32_t i = 0; i < seg.length; ++i)
				{
					const int32_t o = seg.offset + i;
					acc += static_cast<int64_t>(row[o]) * q8[o];
				}
				partial = static_cast<double>(acc) * (rowScale * qScale);
			}
			if (perChannel)
			{
				for (int32_t c = 0; c < bank.channelCount; ++c)
				{
					if (bank.channels[c].offset == seg.offset &&
						bank.channels[c].length == seg.length)
					{
						partial *= BitsToDouble(bank.channelInvNorms[
							static_cast<int64_t>(r) * bank.channelCount + c]);
						break;
					}
				}
			}
			total += BitsToDouble(seg.weight) * partial;
		}
		return total;
	}

	inline bool ScoreBitsEqual(float a, float b)
	{
		uint32_t ab, bb;
		std::memcpy(&ab, &a, 4);
		std::memcpy(&bb, &b, 4);
		return ab == bb;
	}

	// Quantize the fixture query `q` for bank `bank` into q8/scale/sq (the exact bytes
	// the kernel scores, so REF bit-equality is a statement about the kernel, not the
	// quantizer).
	struct QQ
	{
		AlignedBuf q8;
		AlignedBuf qf;
		double scale = 0.0;
		int64_t sq = 0;
		QQ(const BankView& bank, int32_t query)
			: q8(static_cast<size_t>(bank.paddedDims)),
			  qf(static_cast<size_t>(bank.paddedDims) * sizeof(float))
		{
			xd::LoadQuery(query, bank.paddedDims, qf.F32());
			QuantizeQueryXd(qf.F32(), bank.paddedDims, q8.I8(), &scale, &sq);
		}
	};

	// Independent double whole-row score over the DEQUANTIZED int8 rows (the vectors the
	// bank represents) against the FLOAT query — the FEAT ground truth. Composition is
	// applied in double by the caller. best-first index order returned via sort.
	inline double DequantWholeRowD(const BankView& bank, const float* qf, int32_t r)
	{
		const int8_t* row = static_cast<const int8_t*>(bank.rows) +
			static_cast<int64_t>(r) * bank.paddedDims;
		const double rowScale = BitsToDouble(bank.scales[r]);
		double score = 0.0;
		if (bank.metric == Metric::L2)
		{
			for (int32_t i = 0; i < bank.dims; ++i)
			{
				const double d = static_cast<double>(qf[i]) - row[i] * rowScale;
				score += d * d;
			}
		}
		else
		{
			for (int32_t i = 0; i < bank.dims; ++i)
			{
				score += static_cast<double>(qf[i]) * (row[i] * rowScale);
			}
		}
		return score;
	}

	// recall@10 of the XD hit list (first 10 indices) against a double top-10 built from
	// per-row double scores. metric sets the sort direction.
	inline double Recall10(const Hit* hits, int32_t n,
		const std::vector<double>& scoreByIndex, Metric metric)
	{
		const int32_t count = static_cast<int32_t>(scoreByIndex.size());
		std::vector<int32_t> idx(static_cast<size_t>(count));
		for (int32_t i = 0; i < count; ++i)
		{
			idx[static_cast<size_t>(i)] = i;
		}
		std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
			const double sa = scoreByIndex[static_cast<size_t>(a)];
			const double sb = scoreByIndex[static_cast<size_t>(b)];
			if (sa != sb)
			{
				return metric == Metric::L2 ? sa < sb : sa > sb;
			}
			return a < b;
		});
		const int32_t top = n < 10 ? n : 10;
		int32_t match = 0;
		for (int32_t i = 0; i < top; ++i)
		{
			for (int32_t j = 0; j < top; ++j)
			{
				if (hits[i].index == idx[static_cast<size_t>(j)])
				{
					++match;
					break;
				}
			}
		}
		return top > 0 ? static_cast<double>(match) / top : 1.0;
	}
} // namespace xdc

// T-XDC-1 — CrossDevice x per-row bias (dense), Query. REF bit-exact + FEAT recall.
static void TestXdcBiasDense()
{
	Workspace ws;
	xd::FixBank bankA(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
		xdfix::kBankADims, Metric::Dot);
	xd::FixBank bankC(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
		xdfix::kBankCDims, Metric::L2);
	const xd::FixBank* banks[2] = {&bankA, &bankC};

	double recallSum = 0.0;
	int32_t recallN = 0;
	for (const xd::FixBank* bp : banks)
	{
		const BankView& bank = bp->view;
		const bool isL2 = bank.metric == Metric::L2;
		std::vector<float> dense(static_cast<size_t>(bank.count));
		for (int32_t r = 0; r < bank.count; ++r)
		{
			// Mix of ordinary, subnormal, and FLT_MIN bias — the DAZ-proof decode is
			// part of the surface. On L2 a reward is NEGATIVE (lower is better), N2.
			const uint32_t bits = (r % 4 == 0) ? 0x00000007u                   // subnormal
				: (r % 4 == 1) ? (isL2 ? 0xbc23d70au : 0x3c23d70au)            // -/+0.01
				: (r % 4 == 2) ? 0x00800000u                                   // FLT_MIN
				: (isL2 ? 0xbd4ccccdu : 0x3d4ccccdu);                          // -/+0.05
			std::memcpy(&dense[static_cast<size_t>(r)], &bits, 4);
		}
		RowBias rb;
		rb.dense = dense.data();

		for (int32_t q = 0; q < xdfix::kQueryCount; ++q)
		{
			xdc::QQ qq(bank, q);
			std::vector<Hit> hits(static_cast<size_t>(bank.count));
			int32_t n = 0;
			QueryParams p;
			p.exactness = Exactness::CrossDevice;
			p.k = bank.count;
			p.bias = &rb;
			CHECK(Query(bank, qq.qf.F32(), p, ws, hits.data(), &n) == Status::Ok);
			CHECK(n == bank.count);

			// REF: every row's kernel score bit-equals the contract reference.
			for (int32_t i = 0; i < n; ++i)
			{
				const int32_t idx = hits[static_cast<size_t>(i)].index;
				const float want = xdc::ComposeBias(
					xd::RefXdScore(bank, qq.q8.I8(), qq.scale, qq.sq, idx),
					dense[static_cast<size_t>(idx)]);
				CHECK_MSG(xdc::ScoreBitsEqual(want, hits[static_cast<size_t>(i)].score),
					"T-XDC-1 bias score mismatch (metric %d row %d)",
					static_cast<int>(bank.metric), idx);
			}

			// FEAT: recall@10 vs the double biased brute-force (reported).
			std::vector<double> sc(static_cast<size_t>(bank.count));
			for (int32_t r = 0; r < bank.count; ++r)
			{
				sc[static_cast<size_t>(r)] = xdc::DequantWholeRowD(bank, qq.qf.F32(), r) +
					xdc::BitsToDouble(dense[static_cast<size_t>(r)]);
			}
			recallSum += xdc::Recall10(hits.data(), n, sc, bank.metric);
			++recallN;
		}
	}
	std::printf("  T-XDC-1 dense-bias FEAT recall@10 (uncalibrated): %.3f (n=%d)\n",
		recallN > 0 ? recallSum / recallN : 1.0, recallN);
}

// T-XDC-2 — CrossDevice x per-row bias (sparse), the k+P re-selection path (J-4/S2).
static void TestXdcBiasSparse()
{
	Workspace ws;
	xd::FixBank bankA(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
		xdfix::kBankADims, Metric::Dot);
	const BankView& bank = bankA.view;

	for (int32_t q = 0; q < xdfix::kQueryCount; ++q)
	{
		xdc::QQ qq(bank, q);
		const BiasPair pairs[2] = {{5, 0.25f}, {40, -0.5f}};
		RowBias rb;
		rb.pairs = pairs;
		rb.pairCount = 2;
		auto pairBias = [&](int32_t idx) -> float {
			for (int32_t p = 0; p < rb.pairCount; ++p)
			{
				if (pairs[p].index == idx)
				{
					return pairs[p].bias;
				}
			}
			return 0.0f;
		};

		// Pass (a): full k = count — every composed score bit-exact.
		{
			std::vector<Hit> hits(static_cast<size_t>(bank.count));
			int32_t n = 0;
			QueryParams p;
			p.exactness = Exactness::CrossDevice;
			p.k = bank.count;
			p.bias = &rb;
			CHECK(Query(bank, qq.qf.F32(), p, ws, hits.data(), &n) == Status::Ok);
			for (int32_t i = 0; i < n; ++i)
			{
				const int32_t idx = hits[static_cast<size_t>(i)].index;
				const float base = xd::RefXdScore(bank, qq.q8.I8(), qq.scale, qq.sq, idx);
				const float want = pairBias(idx) != 0.0f
					? xdc::ComposeBias(base, pairBias(idx)) : base;
				CHECK_MSG(xdc::ScoreBitsEqual(want, hits[static_cast<size_t>(i)].score),
					"T-XDC-2(a) sparse score mismatch (row %d)", idx);
			}
		}

		// Pass (b): k < count, a lift crossing the k boundary — the ONLY pass that
		// exercises SelectWithSparseBias's scanK = k + pairCount re-selection.
		{
			const int32_t kSmall = 8;
			// Unbiased ranking to place the lift/evict deterministically.
			std::vector<Hit> base(static_cast<size_t>(bank.count));
			int32_t bn = 0;
			QueryParams pu;
			pu.exactness = Exactness::CrossDevice;
			pu.k = bank.count;
			CHECK(Query(bank, qq.qf.F32(), pu, ws, base.data(), &bn) == Status::Ok);
			const int32_t lift = base[static_cast<size_t>(kSmall)].index;    // first out
			const int32_t evict = base[static_cast<size_t>(kSmall - 1)].index; // last in
			// Reward large enough to carry the lifted row to the very top (scaled to the
			// top score so the boost survives float ULP at this magnitude): its composed
			// score clears base[0], so it enters and the old boundary row (evict) drops
			// out of the top-k.
			const float reward = (base[0].score - base[static_cast<size_t>(kSmall)].score) +
				std::fabs(base[0].score) * 0.5f + 1.0f;
			const BiasPair lp[1] = {{lift, reward}};
			RowBias lb;
			lb.pairs = lp;
			lb.pairCount = 1;

			std::vector<Hit> got(static_cast<size_t>(kSmall));
			int32_t gn = 0;
			QueryParams pb;
			pb.exactness = Exactness::CrossDevice;
			pb.k = kSmall;
			pb.bias = &lb;
			CHECK(Query(bank, qq.qf.F32(), pb, ws, got.data(), &gn) == Status::Ok);
			CHECK(gn == kSmall);

			// Independent reference: score all rows composed, sort (better score, then
			// ascending index), take top kSmall.
			std::vector<std::pair<float, int32_t>> ref(static_cast<size_t>(bank.count));
			for (int32_t r = 0; r < bank.count; ++r)
			{
				float s = xd::RefXdScore(bank, qq.q8.I8(), qq.scale, qq.sq, r);
				if (r == lift)
				{
					s = xdc::ComposeBias(s, reward);
				}
				ref[static_cast<size_t>(r)] = {s, r};
			}
			std::sort(ref.begin(), ref.end(), [](const std::pair<float, int32_t>& a,
				const std::pair<float, int32_t>& b) {
				if (a.first != b.first)
				{
					return a.first > b.first; // Dot: higher is better
				}
				return a.second < b.second;
			});
			for (int32_t i = 0; i < kSmall; ++i)
			{
				CHECK_MSG(got[static_cast<size_t>(i)].index == ref[static_cast<size_t>(i)].second,
					"T-XDC-2(b) re-selection order mismatch at %d: got %d want %d",
					i, got[static_cast<size_t>(i)].index, ref[static_cast<size_t>(i)].second);
				CHECK(xdc::ScoreBitsEqual(got[static_cast<size_t>(i)].score,
					ref[static_cast<size_t>(i)].first));
			}
			// The membership change is asserted, not just the scores.
			bool liftIn = false, evictIn = false;
			for (int32_t i = 0; i < kSmall; ++i)
			{
				liftIn = liftIn || got[static_cast<size_t>(i)].index == lift;
				evictIn = evictIn || got[static_cast<size_t>(i)].index == evict;
			}
			CHECK_MSG(liftIn, "T-XDC-2(b) lifted row %d not in top-k", lift);
			CHECK_MSG(!evictIn, "T-XDC-2(b) evicted row %d still in top-k", evict);
		}
	}
}

// T-XDC-3 — CrossDevice x segments (whole-row weighted + channel cosine), Query.
static void TestXdcSegments()
{
	Workspace ws;
	xd::FixBank bankB(xdfix::kBankBRows, xdfix::kBankBScaleBits, xdfix::kBankBCount,
		xdfix::kBankBDims, Metric::Cosine, xdfix::kBankBChannels);
	xd::FixBank bankC(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
		xdfix::kBankCDims, Metric::L2);

	// bankB: channel-matched segments (exact-match inv-norm), a weight-0 (mask), and a
	// segment SPANNING two channels (W3: no inv-norm applies — MatchChannel misses).
	const QuerySegment segsB[3] = {
		{bankB.view.channels[0].offset, bankB.view.channels[0].length, 1.5f},
		{bankB.view.channels[1].offset, bankB.view.channels[1].length, 0.0f}, // masked
		{bankB.view.channels[2].offset,
			bankB.view.channels[2].length + bankB.view.channels[3].length, -0.75f}, // spans 2+3
	};
	// bankC (L2, non-channel): the expanded-epilogue segmented path.
	const QuerySegment segsC[2] = {{0, 16, 1.0f}, {16, 16, 2.0f}};

	struct Case { const xd::FixBank* bank; const QuerySegment* segs; int32_t segCount; };
	const Case cases[2] = {{&bankB, segsB, 3}, {&bankC, segsC, 2}};

	double recallSum = 0.0;
	int32_t recallN = 0;
	for (const Case& cs : cases)
	{
		const BankView& bank = cs.bank->view;
		for (int32_t q = 0; q < xdfix::kQueryCount; ++q)
		{
			xdc::QQ qq(bank, q);
			std::vector<Hit> hits(static_cast<size_t>(bank.count));
			int32_t n = 0;
			QueryParams p;
			p.exactness = Exactness::CrossDevice;
			p.k = bank.count;
			p.segments = cs.segs;
			p.segmentCount = cs.segCount;
			CHECK(Query(bank, qq.qf.F32(), p, ws, hits.data(), &n) == Status::Ok);
			CHECK(n == bank.count);

			// REF: the per-segment double combine (W3), floored once, bit-equals.
			for (int32_t i = 0; i < n; ++i)
			{
				const int32_t idx = hits[static_cast<size_t>(i)].index;
				const float want = xdc::Floor(
					xdc::SegmentedRowD(bank, qq.q8.I8(), qq.scale, cs.segs, cs.segCount, idx));
				CHECK_MSG(xdc::ScoreBitsEqual(want, hits[static_cast<size_t>(i)].score),
					"T-XDC-3 segmented score mismatch (metric %d row %d)",
					static_cast<int>(bank.metric), idx);
			}

			// FEAT: recall@10 vs a double segmented brute-force over dequant rows.
			std::vector<double> sc(static_cast<size_t>(bank.count));
			for (int32_t r = 0; r < bank.count; ++r)
			{
				sc[static_cast<size_t>(r)] =
					xdc::SegmentedRowD(bank, qq.q8.I8(), qq.scale, cs.segs, cs.segCount, r);
			}
			recallSum += xdc::Recall10(hits.data(), n, sc, bank.metric);
			++recallN;
		}
	}
	std::printf("  T-XDC-3 segmented FEAT recall@10 (uncalibrated): %.3f (n=%d)\n",
		recallN > 0 ? recallSum / recallN : 1.0, recallN);
}

// T-XDC-4 — CrossDevice x intersect (worst-of fusion), QueryIntersect.
static void TestXdcIntersect()
{
	Workspace ws;
	xd::FixBank bankA(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
		xdfix::kBankADims, Metric::Dot);
	xd::FixBank bankC(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
		xdfix::kBankCDims, Metric::L2);
	const xd::FixBank* banks[2] = {&bankA, &bankC};

	const int32_t members = 3;
	double recallSum = 0.0;
	int32_t recallN = 0;
	for (const xd::FixBank* bp : banks)
	{
		const BankView& bank = bp->view;
		const bool isL2 = bank.metric == Metric::L2;
		// Three distinct member queries laid out contiguously.
		AlignedBuf batch(static_cast<size_t>(members) * bank.paddedDims * sizeof(float));
		for (int32_t m = 0; m < members; ++m)
		{
			xd::LoadQuery(m % xdfix::kQueryCount, bank.paddedDims,
				batch.F32() + static_cast<int64_t>(m) * bank.paddedDims);
		}
		// Per-member quantized payloads for the reference — one contiguous aligned
		// buffer (each member offset m*paddedDims stays 16-aligned; AlignedBuf is not
		// movable, so a vector of them is out).
		AlignedBuf mq8(static_cast<size_t>(members) * bank.paddedDims);
		std::vector<double> mScale(static_cast<size_t>(members));
		std::vector<int64_t> mSq(static_cast<size_t>(members));
		for (int32_t m = 0; m < members; ++m)
		{
			QuantizeQueryXd(batch.F32() + static_cast<int64_t>(m) * bank.paddedDims,
				bank.paddedDims, mq8.I8() + static_cast<int64_t>(m) * bank.paddedDims,
				&mScale[static_cast<size_t>(m)], &mSq[static_cast<size_t>(m)]);
		}

		std::vector<Hit> hits(static_cast<size_t>(bank.count));
		int32_t n = 0;
		QueryParams p;
		p.exactness = Exactness::CrossDevice;
		p.k = bank.count;
		CHECK(QueryIntersect(bank, batch.F32(), members, p, ws, hits.data(), &n) == Status::Ok);
		CHECK(n == bank.count);

		// REF: each row's WORST fused score across members (L2 -> max, else -> min),
		// bit-exact — each member score is the proven whole-row contract reference.
		auto worstOf = [&](int32_t r) -> float {
			float fused = 0.0f;
			for (int32_t m = 0; m < members; ++m)
			{
				const float s = xd::RefXdScore(bank,
					mq8.I8() + static_cast<int64_t>(m) * bank.paddedDims,
					mScale[static_cast<size_t>(m)], mSq[static_cast<size_t>(m)], r);
				if (m == 0 || (isL2 ? s > fused : s < fused))
				{
					fused = s;
				}
			}
			return fused;
		};
		for (int32_t i = 0; i < n; ++i)
		{
			const int32_t idx = hits[static_cast<size_t>(i)].index;
			CHECK_MSG(xdc::ScoreBitsEqual(worstOf(idx), hits[static_cast<size_t>(i)].score),
				"T-XDC-4 intersect score mismatch (metric %d row %d)",
				static_cast<int>(bank.metric), idx);
		}

		// queryCount == 1 degenerates to Query, bit-identically (INV, restated).
		{
			std::vector<Hit> one(static_cast<size_t>(bank.count));
			std::vector<Hit> single(static_cast<size_t>(bank.count));
			int32_t n1 = 0, ns = 0;
			CHECK(QueryIntersect(bank, batch.F32(), 1, p, ws, one.data(), &n1) == Status::Ok);
			CHECK(Query(bank, batch.F32(), p, ws, single.data(), &ns) == Status::Ok);
			CHECK(n1 == ns);
			for (int32_t i = 0; i < n1; ++i)
			{
				CHECK(one[static_cast<size_t>(i)].index == single[static_cast<size_t>(i)].index &&
					xdc::ScoreBitsEqual(one[static_cast<size_t>(i)].score,
						single[static_cast<size_t>(i)].score));
			}
		}

		// FEAT: recall@10 vs a double worst-of over dequant rows (reported).
		std::vector<double> sc(static_cast<size_t>(bank.count));
		for (int32_t r = 0; r < bank.count; ++r)
		{
			double fused = 0.0;
			for (int32_t m = 0; m < members; ++m)
			{
				const double s = xdc::DequantWholeRowD(bank,
					batch.F32() + static_cast<int64_t>(m) * bank.paddedDims, r);
				if (m == 0 || (isL2 ? s > fused : s < fused))
				{
					fused = s;
				}
			}
			sc[static_cast<size_t>(r)] = fused;
		}
		recallSum += xdc::Recall10(hits.data(), n, sc, bank.metric);
		++recallN;
	}
	std::printf("  T-XDC-4 intersect FEAT recall@10 (uncalibrated): %.3f (n=%d)\n",
		recallN > 0 ? recallSum / recallN : 1.0, recallN);
}

// ---------------------------------------------------------------------------
// T-V2.5 — bank analytics + probing tooling (plan section 22)

namespace an
{
	// Independent scalar recode of ScoreXdPair from the section 22.4 contract (REF).
	inline float RefPair(const int8_t* a8, double as, int64_t asq, const int8_t* b8,
		double bs, int64_t bsq, int32_t pd, Metric m)
	{
		int64_t cross = 0;
		for (int32_t i = 0; i < pd; ++i)
		{
			cross += static_cast<int64_t>(a8[i]) * b8[i];
		}
		double d;
		if (m == Metric::L2)
		{
			const double x = (as * as) * static_cast<double>(asq);
			const double y = (bs * bs) * static_cast<double>(bsq);
			const double z = ((as * bs) * static_cast<double>(cross)) * 2.0;
			d = (x + y) - z;
		}
		else if (m == Metric::Cosine)
		{
			d = 1.0 - static_cast<double>(cross) /
				std::sqrt(static_cast<double>(asq) * static_cast<double>(bsq));
		}
		else
		{
			d = static_cast<double>(cross) * (as * bs);
		}
		const double lim = 1.1754943508222875e-38;
		return (d < lim && d > -lim) ? 0.0f : static_cast<float>(d);
	}

	inline void Hash(uint64_t& h, float v)
	{
		uint32_t b;
		std::memcpy(&b, &v, 4);
		h = (h ^ b) * 0x100000001B3ull;
	}

	// Dequantize an int8 bank row element to double (per-device float reference).
	inline double Dq(const BankView& bank, int32_t r, int32_t i)
	{
		const int8_t* row =
			static_cast<const int8_t*>(bank.rows) + static_cast<int64_t>(r) * bank.paddedDims;
		return pool::DecodeScale(bank.scales[r]) * static_cast<double>(row[i]);
	}

	// float64 nearest-neighbour distance from vector s to target bank, in the metric's
	// distance sense (the FEAT reference, independent of the operator).
	inline double RefNearest(const std::vector<double>& s, const BankView& t)
	{
		double sSq = 0.0;
		for (double v : s)
		{
			sSq += v * v;
		}
		double best = 0.0;
		bool have = false;
		for (int32_t r = 0; r < t.count; ++r)
		{
			double cross = 0.0;
			double rSq = 0.0;
			for (int32_t i = 0; i < t.dims; ++i)
			{
				const double rv = Dq(t, r, i);
				cross += s[static_cast<size_t>(i)] * rv;
				rSq += rv * rv;
			}
			double dist;
			if (t.metric == Metric::L2)
			{
				dist = (sSq + rSq) - 2.0 * cross;
			}
			else if (t.metric == Metric::Cosine)
			{
				dist = 1.0 - cross / std::sqrt(sSq * rSq);
			}
			else
			{
				dist = cross; // dot similarity
			}
			const bool nearer =
				!have || (t.metric == Metric::Dot ? dist > best : dist < best);
			if (nearer)
			{
				best = dist;
				have = true;
			}
		}
		return best;
	}

	// A row-disjoint whole-row slice of a bank (channels dropped) — a source/target pair for
	// same-metric NN divergence without new committed bytes.
	inline BankView SliceView(const BankView& base, int32_t startRow, int32_t count)
	{
		BankView v = base;
		v.rows = static_cast<const int8_t*>(base.rows) +
			static_cast<int64_t>(startRow) * base.paddedDims;
		v.scales = base.scales + startRow;
		v.count = count;
		v.channels = nullptr;
		v.channelCount = 0;
		v.channelInvNorms = nullptr;
		return v;
	}

	// Independent bit-exact recode of MeanNN/MaxNN: the nearest target score per source row
	// (via the contract-recoded xd::RefXdScore, not the kernel), inverted to distance sense,
	// reduced in ascending source order. Matches the operator bitwise (REF).
	inline float RefNNReduce(const BankView& src, const BankView& tgt, Reduce reduce)
	{
		const int8_t* srcRows = static_cast<const int8_t*>(src.rows);
		const int32_t pd = src.paddedDims;
		double acc = 0.0;
		int32_t counted = 0;
		double best = 0.0;
		bool have = false;
		for (int32_t i = 0; i < src.count; ++i)
		{
			const int8_t* q8 = srcRows + static_cast<int64_t>(i) * pd;
			const double qs = pool::DecodeScale(src.scales[i]);
			const int64_t qSq = detail::DotI8I8(q8, q8, pd);
			float bestScore = 0.0f;
			bool haveT = false;
			for (int32_t r = 0; r < tgt.count; ++r)
			{
				const float sc = xd::RefXdScore(tgt, q8, qs, qSq, r);
				const bool better = !haveT ||
					(tgt.metric == Metric::L2 ? sc < bestScore : sc > bestScore);
				if (better)
				{
					bestScore = sc;
					haveT = true;
				}
			}
			const double dist = tgt.metric == Metric::Cosine
				? 1.0 - static_cast<double>(bestScore)
				: static_cast<double>(bestScore);
			if (reduce == Reduce::Mean)
			{
				acc += dist;
				++counted;
			}
			else if (!have || dist > best)
			{
				best = dist;
			}
			have = true;
		}
		const double lim = 1.1754943508222875e-38;
		const double res = reduce == Reduce::Mean ? acc / static_cast<double>(counted) : best;
		return (res < lim && res > -lim) ? 0.0f : static_cast<float>(res);
	}

	// The analytics battery under the CURRENT dispatch: representative operator scalars
	// over the committed fixtures, hashed. Path-sensitive through DotI8I8 / QueryXdBatch.
	inline uint64_t Battery(Workspace& ws)
	{
		xd::FixBank a(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
			xdfix::kBankADims, Metric::Dot);
		xd::FixBank b(xdfix::kBankBRows, xdfix::kBankBScaleBits, xdfix::kBankBCount,
			xdfix::kBankBDims, Metric::Cosine, xdfix::kBankBChannels);
		xd::FixBank c(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
			xdfix::kBankCDims, Metric::L2);
		xd::FixBank d(xdfix::kBankDRows, xdfix::kBankDScaleBits, xdfix::kBankDCount,
			xdfix::kBankDDims, Metric::Dot); // adversarial tiny scale
		xd::FixBank e(xdfix::kBankERows, xdfix::kBankEScaleBits, xdfix::kBankECount,
			xdfix::kBankEDims, Metric::L2); // adversarial tiny scale
		const BankView* banks[5] = {&a.view, &b.view, &c.view, &d.view, &e.view};
		const Metric mets[5] = {Metric::Dot, Metric::Cosine, Metric::L2, Metric::Dot, Metric::L2};

		const int32_t i1[6] = {0, 4, 9, 17, 22, 30};
		const int32_t i2[6] = {2, 7, 12, 20, 25, 31};
		uint64_t h = 0xcbf29ce484222325ull;

		// ScoreXdPair + CentroidDistance over each bank (dot/cosine/L2 + the tiny-scale
		// adversarial floor cases on D/E; cosine sqrt path on B).
		for (int32_t k = 0; k < 5; ++k)
		{
			const BankView& bank = *banks[k];
			const int32_t pd = bank.paddedDims;
			AlignedBuf cA(static_cast<size_t>(pd));
			AlignedBuf cB(static_cast<size_t>(pd));
			double sA = 0.0;
			double sB = 0.0;
			int64_t qA = 0;
			int64_t qB = 0;
			if (MakeCentroidCrossDevice(bank, i1, 6, nullptr, nullptr, cA.I8(), &sA, &qA) !=
					Status::Ok ||
				MakeCentroidCrossDevice(bank, i2, 6, nullptr, nullptr, cB.I8(), &sB, &qB) !=
					Status::Ok)
			{
				continue;
			}
			const XdQuery pa{cA.I8(), sA, qA};
			const XdQuery pb{cB.I8(), sB, qB};
			float sc = 0.0f;
			ScoreXdPair(pa, pb, pd, mets[k], &sc);
			Hash(h, sc);
			AlignedBuf dA(static_cast<size_t>(pd));
			AlignedBuf dB(static_cast<size_t>(pd));
			float cd = 0.0f;
			CentroidDistanceCrossDevice(bank, i1, 6, nullptr, nullptr, bank, i2, 6, nullptr,
				nullptr, mets[k], dA.I8(), dB.I8(), &cd);
			Hash(h, cd);
			// Spread (mean + max) over the union selection.
			const int32_t all[12] = {0, 2, 4, 7, 9, 12, 17, 20, 22, 25, 30, 31};
			AlignedBuf cs(static_cast<size_t>(pd));
			float sm = 0.0f;
			float sx = 0.0f;
			SpreadCrossDevice(bank, all, 12, nullptr, Reduce::Mean, cs.I8(), &sm);
			SpreadCrossDevice(bank, all, 12, nullptr, Reduce::Max, cs.I8(), &sx);
			Hash(h, sm);
			Hash(h, sx);
		}

		// NN divergence between two same-dims L2 banks (C and E are both 32-dim L2).
		std::vector<XdQuery> qbuf(static_cast<size_t>(c.view.count));
		std::vector<Hit> hbuf(static_cast<size_t>(c.view.count));
		std::vector<int32_t> nbuf(static_cast<size_t>(c.view.count));
		float mn = 0.0f;
		float mx = 0.0f;
		MeanNNCrossDevice(c.view, nullptr, e.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), ws, &mn);
		MaxNNCrossDevice(c.view, nullptr, e.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), ws, &mx);
		Hash(h, mn);
		Hash(h, mx);

		// NN divergence on Cosine and Dot targets (T-V2.5-8): a whole-vector-normalized
		// Cosine bank (the shipped scaled-dot cosine needs unit-norm rows, as production
		// S-SEM banks are) and row-disjoint Dot slices, so the reduction's cosine (1-cos
		// inversion) and dot branches cross the forced-path sweep too. Fixed seed → the
		// battery stays deterministic across paths.
		Rng cosRng(0xC051E0DE7E57ull);
		TestBank cosBank(cosRng, 64, 32, Quantization::Int8, Metric::Cosine);
		const BankView cosSrc = SliceView(cosBank.view, 0, 32);
		const BankView cosTgt = SliceView(cosBank.view, 32, 32);
		const BankView dotSrc = SliceView(a.view, 0, 32);
		const BankView dotTgt = SliceView(a.view, 32, 32);
		const BankView* pairs[4] = {&cosSrc, &dotSrc, &cosTgt, &dotTgt};
		int32_t maxN = 0;
		for (int32_t p = 0; p < 4; ++p)
		{
			maxN = pairs[p]->count > maxN ? pairs[p]->count : maxN;
		}
		std::vector<XdQuery> qb2(static_cast<size_t>(maxN));
		std::vector<Hit> hb2(static_cast<size_t>(maxN));
		std::vector<int32_t> nb2(static_cast<size_t>(maxN));
		float v = 0.0f;
		MeanNNCrossDevice(cosSrc, nullptr, cosTgt, nullptr, qb2.data(), hb2.data(), nb2.data(),
			ws, &v);
		Hash(h, v);
		MaxNNCrossDevice(cosSrc, nullptr, cosTgt, nullptr, qb2.data(), hb2.data(), nb2.data(),
			ws, &v);
		Hash(h, v);
		MeanNNCrossDevice(dotSrc, nullptr, dotTgt, nullptr, qb2.data(), hb2.data(), nb2.data(),
			ws, &v);
		Hash(h, v);
		MaxNNCrossDevice(dotSrc, nullptr, dotTgt, nullptr, qb2.data(), hb2.data(), nb2.data(),
			ws, &v);
		Hash(h, v);
		return h;
	}
} // namespace an

static void TestBankAnalytics()
{
	std::printf("bank analytics (T-V2.5): pair score, set distance, NN divergence, spread\n");
	xd::FixBank a(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
		xdfix::kBankADims, Metric::Dot);
	xd::FixBank b(xdfix::kBankBRows, xdfix::kBankBScaleBits, xdfix::kBankBCount,
		xdfix::kBankBDims, Metric::Cosine, xdfix::kBankBChannels);
	xd::FixBank c(xdfix::kBankCRows, xdfix::kBankCScaleBits, xdfix::kBankCCount,
		xdfix::kBankCDims, Metric::L2);
	const BankView* banks[3] = {&a.view, &b.view, &c.view};
	const Metric mets[3] = {Metric::Dot, Metric::Cosine, Metric::L2};
	const char* names[3] = {"dot", "cosine", "L2"};
	const int32_t i1[6] = {0, 4, 9, 17, 22, 30};
	const int32_t i2[6] = {2, 7, 12, 20, 25, 31};
	Workspace ws;

	// --- T-V2.5-1 REF: ScoreXdPair == independent recode; CentroidDistance == the pair ---
	for (int32_t k = 0; k < 3; ++k)
	{
		const BankView& bank = *banks[k];
		const int32_t pd = bank.paddedDims;
		AlignedBuf cA(static_cast<size_t>(pd));
		AlignedBuf cB(static_cast<size_t>(pd));
		double sA = 0.0;
		double sB = 0.0;
		int64_t qA = 0;
		int64_t qB = 0;
		CHECK(MakeCentroidCrossDevice(bank, i1, 6, nullptr, nullptr, cA.I8(), &sA, &qA) ==
			Status::Ok);
		CHECK(MakeCentroidCrossDevice(bank, i2, 6, nullptr, nullptr, cB.I8(), &sB, &qB) ==
			Status::Ok);
		const XdQuery pa{cA.I8(), sA, qA};
		const XdQuery pb{cB.I8(), sB, qB};
		float sc = 0.0f;
		CHECK(ScoreXdPair(pa, pb, pd, mets[k], &sc) == Status::Ok);
		const float ref = an::RefPair(cA.I8(), sA, qA, cB.I8(), sB, qB, pd, mets[k]);
		CHECK_MSG(sc == ref, "%s pair %.9g != ref %.9g", names[k], sc, ref);

		AlignedBuf dA(static_cast<size_t>(pd));
		AlignedBuf dB(static_cast<size_t>(pd));
		float cd = 0.0f;
		CHECK(CentroidDistanceCrossDevice(bank, i1, 6, nullptr, nullptr, bank, i2, 6, nullptr,
			nullptr, mets[k], dA.I8(), dB.I8(), &cd) == Status::Ok);
		CHECK_MSG(cd == ref, "%s centroid-distance %.9g != pair %.9g", names[k], cd, ref);
	}

	// --- T-V2.5-3 FEAT: operators recover the right statistic vs a float64 reference ---
	for (int32_t k = 0; k < 3; ++k)
	{
		const BankView& bank = *banks[k];
		// CentroidDistance FEAT: float64 centroid distance (centroid requant tolerated).
		std::vector<double> cf1(static_cast<size_t>(bank.dims));
		std::vector<double> cf2(static_cast<size_t>(bank.dims));
		pool::RefPoolF64(bank, i1, 6, nullptr, cf1.data());
		pool::RefPoolF64(bank, i2, 6, nullptr, cf2.data());
		double cross = 0.0;
		double n1 = 0.0;
		double n2 = 0.0;
		for (int32_t i = 0; i < bank.dims; ++i)
		{
			cross += cf1[static_cast<size_t>(i)] * cf2[static_cast<size_t>(i)];
			n1 += cf1[static_cast<size_t>(i)] * cf1[static_cast<size_t>(i)];
			n2 += cf2[static_cast<size_t>(i)] * cf2[static_cast<size_t>(i)];
		}
		double fref;
		if (mets[k] == Metric::L2)
		{
			fref = (n1 + n2) - 2.0 * cross;
		}
		else if (mets[k] == Metric::Cosine)
		{
			fref = 1.0 - cross / std::sqrt(n1 * n2);
		}
		else
		{
			fref = cross;
		}
		const int32_t pd = bank.paddedDims;
		AlignedBuf dA(static_cast<size_t>(pd));
		AlignedBuf dB(static_cast<size_t>(pd));
		float cd = 0.0f;
		CentroidDistanceCrossDevice(bank, i1, 6, nullptr, nullptr, bank, i2, 6, nullptr,
			nullptr, mets[k], dA.I8(), dB.I8(), &cd);
		const double tol = 0.3 * std::fabs(fref) + 0.05;
		CHECK_MSG(std::fabs(static_cast<double>(cd) - fref) <= tol,
			"%s centroid-distance FEAT: op %.6g float64 %.6g tol %.3g", names[k],
			static_cast<double>(cd), fref, tol);
		std::printf("  T-V2.5-3 %s centroid-distance FEAT: op %.6g float64 %.6g\n", names[k],
			static_cast<double>(cd), fref);
	}

	// MeanNN / MaxNN FEAT over the two L2 32-dim banks (C source, E target).
	{
		xd::FixBank e(xdfix::kBankERows, xdfix::kBankEScaleBits, xdfix::kBankECount,
			xdfix::kBankEDims, Metric::L2);
		std::vector<XdQuery> qbuf(static_cast<size_t>(c.view.count));
		std::vector<Hit> hbuf(static_cast<size_t>(c.view.count));
		std::vector<int32_t> nbuf(static_cast<size_t>(c.view.count));
		float mn = 0.0f;
		float mx = 0.0f;
		CHECK(MeanNNCrossDevice(c.view, nullptr, e.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), ws, &mn) == Status::Ok);
		CHECK(MaxNNCrossDevice(c.view, nullptr, e.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), ws, &mx) == Status::Ok);
		double sum = 0.0;
		double refMax = 0.0;
		bool have = false;
		for (int32_t r = 0; r < c.view.count; ++r)
		{
			std::vector<double> s(static_cast<size_t>(c.view.dims));
			for (int32_t i = 0; i < c.view.dims; ++i)
			{
				s[static_cast<size_t>(i)] = an::Dq(c.view, r, i);
			}
			const double nn = an::RefNearest(s, e.view);
			sum += nn;
			if (!have || nn > refMax)
			{
				refMax = nn;
			}
			have = true;
		}
		const double refMean = sum / static_cast<double>(c.view.count);
		const double tolM = 0.1 * std::fabs(refMean) + 1e-4;
		const double tolX = 0.1 * std::fabs(refMax) + 1e-4;
		CHECK_MSG(std::fabs(static_cast<double>(mn) - refMean) <= tolM,
			"mean-NN FEAT op %.6g float64 %.6g", static_cast<double>(mn), refMean);
		CHECK_MSG(std::fabs(static_cast<double>(mx) - refMax) <= tolX,
			"max-NN FEAT op %.6g float64 %.6g", static_cast<double>(mx), refMax);
		std::printf("  T-V2.5-3 mean-NN FEAT: op %.6g float64 %.6g; max-NN op %.6g float64 %.6g\n",
			static_cast<double>(mn), refMean, static_cast<double>(mx), refMax);
	}

	// --- T-V2.5-8 MeanNN/MaxNN on Cosine and Dot targets (G1): REF (bit-exact) + FEAT ---
	{
		Rng cosRng(0xC051E0DE7E57ull);
		TestBank cosBank(cosRng, 64, 32, Quantization::Int8, Metric::Cosine);
		const BankView srcs[2] = {an::SliceView(cosBank.view, 0, 32), an::SliceView(a.view, 0, 32)};
		const BankView tgts[2] = {an::SliceView(cosBank.view, 32, 32), an::SliceView(a.view, 32, 32)};
		const char* nm[2] = {"cosine", "dot"};
		std::vector<XdQuery> qb(32);
		std::vector<Hit> hb(32);
		std::vector<int32_t> nb(32);
		for (int32_t k = 0; k < 2; ++k)
		{
			float opMean = 0.0f;
			float opMax = 0.0f;
			CHECK(MeanNNCrossDevice(srcs[k], nullptr, tgts[k], nullptr, qb.data(), hb.data(),
				nb.data(), ws, &opMean) == Status::Ok);
			CHECK(MaxNNCrossDevice(srcs[k], nullptr, tgts[k], nullptr, qb.data(), hb.data(),
				nb.data(), ws, &opMax) == Status::Ok);
			// REF — bit-exact independent recode (nearest via xd::RefXdScore, inverted, reduced).
			CHECK_MSG(opMean == an::RefNNReduce(srcs[k], tgts[k], Reduce::Mean),
				"%s mean-NN REF mismatch (%.9g)", nm[k], opMean);
			CHECK_MSG(opMax == an::RefNNReduce(srcs[k], tgts[k], Reduce::Max),
				"%s max-NN REF mismatch (%.9g)", nm[k], opMax);
			// FEAT — vs float64 brute-force nearest distance.
			double sum = 0.0;
			double fMax = 0.0;
			bool have = false;
			for (int32_t i = 0; i < srcs[k].count; ++i)
			{
				std::vector<double> s(static_cast<size_t>(srcs[k].dims));
				for (int32_t j = 0; j < srcs[k].dims; ++j)
				{
					s[static_cast<size_t>(j)] = an::Dq(srcs[k], i, j);
				}
				const double nn = an::RefNearest(s, tgts[k]);
				sum += nn;
				if (!have || nn > fMax)
				{
					fMax = nn;
				}
				have = true;
			}
			const double fMean = sum / static_cast<double>(srcs[k].count);
			CHECK_MSG(std::fabs(static_cast<double>(opMean) - fMean) <= 0.1 * std::fabs(fMean) + 1e-4,
				"%s mean-NN FEAT op %.6g f64 %.6g", nm[k], static_cast<double>(opMean), fMean);
			CHECK_MSG(std::fabs(static_cast<double>(opMax) - fMax) <= 0.1 * std::fabs(fMax) + 1e-4,
				"%s max-NN FEAT op %.6g f64 %.6g", nm[k], static_cast<double>(opMax), fMax);
			std::printf("  T-V2.5-8 %s mean-NN op %.6g f64 %.6g; max-NN op %.6g f64 %.6g\n",
				nm[k], static_cast<double>(opMean), fMean, static_cast<double>(opMax), fMax);
		}
	}

	// --- T-V2.5-9 Spread feature oracle on every metric (G2) ---
	{
		const int32_t sel[8] = {0, 4, 9, 17, 22, 30, 33, 44};
		for (int32_t bi = 0; bi < 3; ++bi)
		{
			const BankView& bank = *banks[bi];
			const int32_t pd = bank.paddedDims;
			AlignedBuf cs(static_cast<size_t>(pd));
			float opMean = 0.0f;
			float opMax = 0.0f;
			CHECK(SpreadCrossDevice(bank, sel, 8, nullptr, Reduce::Mean, cs.I8(), &opMean) ==
				Status::Ok);
			CHECK(SpreadCrossDevice(bank, sel, 8, nullptr, Reduce::Max, cs.I8(), &opMax) ==
				Status::Ok);
			// float64 dispersion: distance of each selected row to the float64 centroid.
			std::vector<double> cf(static_cast<size_t>(bank.dims));
			pool::RefPoolF64(bank, sel, 8, nullptr, cf.data());
			double cSq = 0.0;
			for (double vv : cf)
			{
				cSq += vv * vv;
			}
			double sum = 0.0;
			double dMax = 0.0;
			bool have = false;
			for (int32_t i = 0; i < 8; ++i)
			{
				double cross = 0.0;
				double sSq = 0.0;
				for (int32_t j = 0; j < bank.dims; ++j)
				{
					const double rv = an::Dq(bank, sel[i], j);
					cross += rv * cf[static_cast<size_t>(j)];
					sSq += rv * rv;
				}
				double dist;
				if (bank.metric == Metric::L2)
				{
					dist = (sSq + cSq) - 2.0 * cross;
				}
				else if (bank.metric == Metric::Cosine)
				{
					dist = 1.0 - cross / std::sqrt(sSq * cSq);
				}
				else
				{
					dist = cross;
				}
				sum += dist;
				if (!have || dist > dMax)
				{
					dMax = dist;
				}
				have = true;
			}
			const double fMean = sum / 8.0;
			CHECK_MSG(std::fabs(static_cast<double>(opMean) - fMean) <= 0.3 * std::fabs(fMean) + 0.05,
				"%s spread-mean FEAT op %.6g f64 %.6g", names[bi], static_cast<double>(opMean), fMean);
			CHECK_MSG(std::fabs(static_cast<double>(opMax) - dMax) <= 0.3 * std::fabs(dMax) + 0.05,
				"%s spread-max FEAT op %.6g f64 %.6g", names[bi], static_cast<double>(opMax), dMax);
			std::printf("  T-V2.5-9 %s spread-mean op %.6g f64 %.6g; max op %.6g f64 %.6g\n",
				names[bi], static_cast<double>(opMean), fMean, static_cast<double>(opMax), dMax);
		}
	}

	// --- T-V2.5-11 the -128 boundary guard at public ScoreXdPair (D-V2-13) ---
	{
		const int32_t pd = 16;
		AlignedBuf img(static_cast<size_t>(pd));
		AlignedBuf bad(static_cast<size_t>(pd));
		std::memset(img.I8(), 5, static_cast<size_t>(pd));
		std::memset(bad.I8(), 5, static_cast<size_t>(pd));
		bad.I8()[3] = -128; // INT8_MIN — outside the ±127 premise
		const int64_t sq = detail::DotI8I8(img.I8(), img.I8(), pd);
		const XdQuery good{img.I8(), 1.0, sq};
		const XdQuery badQ{bad.I8(), 1.0, sq}; // self-dot value irrelevant: the guard fires first
		float out = 0.0f;
		CHECK(ScoreXdPair(badQ, good, pd, Metric::Dot, &out) == Status::InvalidArgument);
		CHECK(ScoreXdPair(good, badQ, pd, Metric::Dot, &out) == Status::InvalidArgument);
		// A valid ±127 payload is unaffected.
		AlignedBuf okb(static_cast<size_t>(pd));
		std::memset(okb.I8(), 127, static_cast<size_t>(pd));
		const XdQuery okQ{okb.I8(), 1.0, detail::DotI8I8(okb.I8(), okb.I8(), pd)};
		CHECK(ScoreXdPair(okQ, okQ, pd, Metric::Dot, &out) == Status::Ok);
	}

	// --- T-V2.5-4 shape/platform extremes + the ceiling (W1) ---
	{
		// One-row L2 bank: spread is exactly 0. The single-row centroid is bit-identical to
		// the row only when the row's max magnitude is already +-127 (else pooling rescales
		// it to full int8 range); this row is constructed that way so the case is exact.
		alignas(16) const int8_t oneRow[16] = {127, -100, 64, 0, 33, -127, 20, 10, 80, -40, 60, 0, 30, -20, 10, 0};
		const uint32_t oneScale[1] = {0x3f800000u}; // 1.0f
		xd::FixBank one(oneRow, oneScale, 1, 16, Metric::L2);
		const int32_t r0[1] = {0};
		AlignedBuf cs(static_cast<size_t>(one.view.paddedDims));
		float sp = -1.0f;
		CHECK(SpreadCrossDevice(one.view, r0, 1, nullptr, Reduce::Mean, cs.I8(), &sp) ==
			Status::Ok);
		CHECK_MSG(sp == 0.0f, "one-row L2 spread %.9g != 0", sp);

		// Ceiling: two all-127 int8 vectors at kMaxCrossDeviceDims - crossDot stays under
		// 2^31 (W1). Score succeeds and matches the REF; a -128 operand would overflow.
		const int32_t pd = kMaxCrossDeviceDims;
		AlignedBuf ia(static_cast<size_t>(pd));
		AlignedBuf ib(static_cast<size_t>(pd));
		std::memset(ia.I8(), 127, static_cast<size_t>(pd));
		std::memset(ib.I8(), 127, static_cast<size_t>(pd));
		const int64_t sq = static_cast<int64_t>(pd) * 127 * 127;
		const XdQuery ca{ia.I8(), 1.0, sq};
		const XdQuery cb{ib.I8(), 1.0, sq};
		float sc = 0.0f;
		CHECK(ScoreXdPair(ca, cb, pd, Metric::Dot, &sc) == Status::Ok);
		const float ref = an::RefPair(ia.I8(), 1.0, sq, ib.I8(), 1.0, sq, pd, Metric::Dot);
		CHECK_MSG(sc == ref, "ceiling pair %.9g != ref %.9g", sc, ref);
		CHECK(sq < (int64_t{1} << 31)); // the bound the score rode
	}

	// --- T-V2.5-5 rejection catalog (DEF) ---
	{
		const int32_t pd = a.view.paddedDims;
		AlignedBuf img(static_cast<size_t>(pd));
		std::memset(img.I8(), 3, static_cast<size_t>(pd));
		const int64_t sq = detail::DotI8I8(img.I8(), img.I8(), pd);
		const XdQuery good{img.I8(), 1.0, sq};
		float out = 0.0f;
		// over-cap dims
		CHECK(ScoreXdPair(good, good, kMaxCrossDeviceDims + 1, Metric::Dot, &out) ==
			Status::InvalidArgument);
		// desynced payload (self-dot wrong)
		const XdQuery desync{img.I8(), 1.0, sq + 1};
		CHECK(ScoreXdPair(desync, good, pd, Metric::Dot, &out) == Status::InvalidArgument);
		// non-finite scale
		const XdQuery nanScale{img.I8(), std::nan(""), sq};
		CHECK(ScoreXdPair(nanScale, good, pd, Metric::Dot, &out) == Status::InvalidArgument);
		// zero self-dot cosine member
		AlignedBuf zero(static_cast<size_t>(pd));
		std::memset(zero.I8(), 0, static_cast<size_t>(pd));
		const XdQuery z{zero.I8(), 1.0, 0};
		CHECK(ScoreXdPair(z, good, pd, Metric::Cosine, &out) == Status::ZeroNormQuery);

		// f32 bank rejection at the operator boundary.
		AlignedBuf cbuf(static_cast<size_t>(pd));
		BankView f32bank = a.view;
		f32bank.quant = Quantization::Float32;
		const int32_t idx[2] = {0, 1};
		float cd = 0.0f;
		AlignedBuf d2(static_cast<size_t>(pd));
		CHECK(CentroidDistanceCrossDevice(f32bank, idx, 2, nullptr, nullptr, a.view, idx, 2,
			nullptr, nullptr, Metric::Dot, cbuf.I8(), d2.I8(), &cd) == Status::InvalidArgument);
		// dims mismatch A(48) vs C(32)
		CHECK(CentroidDistanceCrossDevice(a.view, idx, 2, nullptr, nullptr, c.view, idx, 2,
			nullptr, nullptr, Metric::Dot, cbuf.I8(), d2.I8(), &cd) == Status::InvalidArgument);
		// empty source: MeanNN with every source row excluded.
		std::vector<uint32_t> allEx(static_cast<size_t>((a.view.count + 31) / 32), 0xffffffffu);
		std::vector<XdQuery> qbuf(static_cast<size_t>(a.view.count));
		std::vector<Hit> hbuf(static_cast<size_t>(a.view.count));
		std::vector<int32_t> nbuf(static_cast<size_t>(a.view.count));
		float mn = 0.0f;
		CHECK(MeanNNCrossDevice(a.view, allEx.data(), a.view, nullptr, qbuf.data(),
			hbuf.data(), nbuf.data(), ws, &mn) == Status::InvalidArgument);
	}

	// --- T-V2.5-6 determinism: repeat bit-identity + fixed-order mean ---
	{
		xd::FixBank e(xdfix::kBankERows, xdfix::kBankEScaleBits, xdfix::kBankECount,
			xdfix::kBankEDims, Metric::L2);
		std::vector<XdQuery> qbuf(static_cast<size_t>(c.view.count));
		std::vector<Hit> hbuf(static_cast<size_t>(c.view.count));
		std::vector<int32_t> nbuf(static_cast<size_t>(c.view.count));
		float m1 = 0.0f;
		float m2 = 0.0f;
		CHECK(MeanNNCrossDevice(c.view, nullptr, e.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), ws, &m1) == Status::Ok);
		CHECK(MeanNNCrossDevice(c.view, nullptr, e.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), ws, &m2) == Status::Ok);
		CHECK(m1 == m2); // repeat bit-identical
		// The mean is the fixed-order (ascending source index) sum of per-row NN distances.
		double acc = 0.0;
		for (int32_t i = 0; i < c.view.count; ++i)
		{
			acc += static_cast<double>(hbuf[static_cast<size_t>(i)].score);
		}
		const double lim = 1.1754943508222875e-38;
		double mref = acc / static_cast<double>(c.view.count);
		const float mrefF = (mref < lim && mref > -lim) ? 0.0f : static_cast<float>(mref);
		CHECK_MSG(m1 == mrefF, "fixed-order mean %.9g != recode %.9g", m1, mrefF);
	}

	// --- T-V2.5-7 carried contracts: weighted==unweighted, exclusion==subset, alloc flat ---
	{
		const int32_t pd = c.view.paddedDims;
		const int32_t idx[6] = {0, 4, 9, 17, 22, 30};
		const int32_t eqW[6] = {5, 5, 5, 5, 5, 5};
		AlignedBuf uA(static_cast<size_t>(pd));
		AlignedBuf uB(static_cast<size_t>(pd));
		AlignedBuf wA(static_cast<size_t>(pd));
		AlignedBuf wB(static_cast<size_t>(pd));
		float du = 0.0f;
		float dw = 0.0f;
		CHECK(CentroidDistanceCrossDevice(c.view, idx, 6, nullptr, nullptr, c.view, i2, 6,
			nullptr, nullptr, Metric::L2, uA.I8(), uB.I8(), &du) == Status::Ok);
		CHECK(CentroidDistanceCrossDevice(c.view, idx, 6, eqW, nullptr, c.view, i2, 6, eqW,
			nullptr, Metric::L2, wA.I8(), wB.I8(), &dw) == Status::Ok);
		CHECK_MSG(du == dw, "weighted %.9g != unweighted %.9g", dw, du);

		// exclusion == subset: pooling [0,1,2,3] with rows 1,2 excluded == pooling [0,3].
		const int32_t four[4] = {0, 1, 2, 3};
		const int32_t two[2] = {0, 3};
		std::vector<uint32_t> ex(static_cast<size_t>((c.view.count + 31) / 32), 0);
		ex[0] = (1u << 1) | (1u << 2);
		AlignedBuf eA(static_cast<size_t>(pd));
		AlignedBuf eB(static_cast<size_t>(pd));
		AlignedBuf sA(static_cast<size_t>(pd));
		AlignedBuf sB(static_cast<size_t>(pd));
		float de = 0.0f;
		float ds = 0.0f;
		CHECK(CentroidDistanceCrossDevice(c.view, four, 4, nullptr, ex.data(), c.view, i2, 6,
			nullptr, nullptr, Metric::L2, eA.I8(), eB.I8(), &de) == Status::Ok);
		CHECK(CentroidDistanceCrossDevice(c.view, two, 2, nullptr, nullptr, c.view, i2, 6,
			nullptr, nullptr, Metric::L2, sA.I8(), sB.I8(), &ds) == Status::Ok);
		CHECK_MSG(de == ds, "exclusion %.9g != subset %.9g", de, ds);

		// Allocation flat (dim 1): a warm workspace does not grow across analytics calls of
		// DIFFERING source counts. Prime on the larger source (C, 48 rows), then run the
		// smaller (E, 32 rows) and a repeat under the global allocation counter.
		xd::FixBank e(xdfix::kBankERows, xdfix::kBankEScaleBits, xdfix::kBankECount,
			xdfix::kBankEDims, Metric::L2);
		Workspace warm;
		std::vector<XdQuery> qbuf(static_cast<size_t>(c.view.count));
		std::vector<Hit> hbuf(static_cast<size_t>(c.view.count));
		std::vector<int32_t> nbuf(static_cast<size_t>(c.view.count));
		float m = 0.0f;
		CHECK(MeanNNCrossDevice(c.view, nullptr, e.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), warm, &m) == Status::Ok);
		const uint64_t before = AllocationCount();
		CHECK(MeanNNCrossDevice(e.view, nullptr, c.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), warm, &m) == Status::Ok); // smaller source count
		CHECK(MaxNNCrossDevice(c.view, nullptr, e.view, nullptr, qbuf.data(), hbuf.data(),
			nbuf.data(), warm, &m) == Status::Ok);
		CHECK_MSG(AllocationCount() == before, "warm analytics workspace grew: %llu -> %llu",
			static_cast<unsigned long long>(before),
			static_cast<unsigned long long>(AllocationCount()));
	}

	// --- T-V2.5-2 forced-path sweep + golden hash (GOLD) ---
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
			hashes[i] = an::Battery(ws);
			detail::ClearForcedXdSimdPath();
			if (i > 0)
			{
				CHECK_MSG(hashes[i] == hashes[0], "analytics forced path %d hash %llx != %llx",
					static_cast<int>(paths[i]),
					static_cast<unsigned long long>(hashes[i]),
					static_cast<unsigned long long>(hashes[0]));
			}
		}
		const uint64_t def = an::Battery(ws);
		CHECK(def == hashes[0]);
		std::printf("analytics cross-device hash: %016llx (%d forced paths agree)\n",
			static_cast<unsigned long long>(def), static_cast<int>(paths.size()));
		if constexpr (xdfix::kGoldenAnalyticsXdHash != 0)
		{
			CHECK_MSG(def == xdfix::kGoldenAnalyticsXdHash,
				"analytics hash %016llx != pinned golden %016llx",
				static_cast<unsigned long long>(def),
				static_cast<unsigned long long>(xdfix::kGoldenAnalyticsXdHash));
		}
	}
}

static void TestProjectionReport()
{
	std::printf("projection report (T-V2.5-A): projections + separation (Feature A, offline)\n");
	xd::FixBank a(xdfix::kBankARows, xdfix::kBankAScaleBits, xdfix::kBankACount,
		xdfix::kBankADims, Metric::Dot);
	const int32_t pd = a.view.paddedDims;

	// A probe direction: normalize(row0 - row1) via MakeDirection.
	AlignedBuf p0(static_cast<size_t>(pd) * sizeof(float));
	AlignedBuf p1(static_cast<size_t>(pd) * sizeof(float));
	for (int32_t i = 0; i < pd; ++i)
	{
		p0.F32()[i] = static_cast<float>(an::Dq(a.view, 0, i));
		p1.F32()[i] = static_cast<float>(an::Dq(a.view, 1, i));
	}
	AlignedBuf dir(static_cast<size_t>(pd) * sizeof(float));
	CHECK(MakeDirection(p0.F32(), p1.F32(), a.view.dims, pd, dir.F32()) == Status::Ok);

	// A1 — projection == double dot of the dequantized row onto the unit direction.
	std::vector<float> proj(static_cast<size_t>(a.view.count));
	CHECK(ProjectionReport(a.view, dir.F32(), nullptr, proj.data(), nullptr) == Status::Ok);
	for (int32_t r = 0; r < a.view.count; ++r)
	{
		double ref = 0.0;
		for (int32_t i = 0; i < pd; ++i)
		{
			ref += an::Dq(a.view, r, i) * static_cast<double>(dir.F32()[i]);
		}
		CHECK_MSG(std::fabs(static_cast<double>(proj[static_cast<size_t>(r)]) - ref) <=
				1e-4 * (std::fabs(ref) + 1.0),
			"row %d projection %.6g ref %.6g", r, proj[static_cast<size_t>(r)], ref);
	}

	// A2 — separation: a known-separating axis clears the bar; a non-separating one does not.
	// Build a synthetic 2-group bank: group A rows carry +value on dim 0, group B -value.
	{
		const int32_t n = 8;
		const int32_t dims = 16;
		alignas(16) int8_t rows[8 * 16] = {};
		uint32_t scaleBits[8];
		const uint32_t one = 0x3f800000u; // 1.0f
		// dim 0 separates the groups with real within-group spread (nonzero pooled sd);
		// dim 2 carries the SAME distribution in both groups (no group signal).
		const int8_t sepAxis[8] = {100, 90, 110, 95, -100, -90, -110, -95};
		const int8_t noAxis[8] = {40, -40, 40, -40, 40, -40, 40, -40};
		for (int32_t r = 0; r < n; ++r)
		{
			scaleBits[r] = one;
			rows[r * dims + 0] = sepAxis[r]; // separating axis = dim 0
			rows[r * dims + 2] = noAxis[r];  // non-separating axis = dim 2
		}
		xd::FixBank g(rows, scaleBits, n, dims, Metric::Dot);
		const int32_t gpd = g.view.paddedDims;
		AlignedBuf sepDir(static_cast<size_t>(gpd) * sizeof(float));
		AlignedBuf noDir(static_cast<size_t>(gpd) * sizeof(float));
		std::memset(sepDir.F32(), 0, static_cast<size_t>(gpd) * sizeof(float));
		std::memset(noDir.F32(), 0, static_cast<size_t>(gpd) * sizeof(float));
		sepDir.F32()[0] = 1.0f; // aligned with the separating axis
		noDir.F32()[2] = 1.0f;  // an axis with no group signal
		std::vector<uint32_t> grp(1, 0x0000000fu); // rows 0..3 = group A
		std::vector<float> gp(static_cast<size_t>(n));
		float sepD = 0.0f;
		float noD = 0.0f;
		CHECK(ProjectionReport(g.view, sepDir.F32(), grp.data(), gp.data(), &sepD) ==
			Status::Ok);
		CHECK(ProjectionReport(g.view, noDir.F32(), grp.data(), gp.data(), &noD) == Status::Ok);
		CHECK_MSG(std::fabs(sepD) >= 2.0f, "separating axis d=%.3f below bar", sepD);
		CHECK_MSG(std::fabs(noD) <= 0.5f, "non-separating axis d=%.3f above bar", noD);
		std::printf("  T-V2.5-A2 separation: separating d=%.3f, non-separating d=%.3f\n", sepD,
			noD);
	}

	// A3 — rejections.
	{
		std::vector<float> proj2(static_cast<size_t>(a.view.count));
		AlignedBuf zeroDir(static_cast<size_t>(pd) * sizeof(float));
		std::memset(zeroDir.F32(), 0, static_cast<size_t>(pd) * sizeof(float));
		CHECK(ProjectionReport(a.view, zeroDir.F32(), nullptr, proj2.data(), nullptr) ==
			Status::ZeroNormQuery);
		AlignedBuf nanDir(static_cast<size_t>(pd) * sizeof(float));
		std::memset(nanDir.F32(), 0, static_cast<size_t>(pd) * sizeof(float));
		nanDir.F32()[0] = std::nanf("");
		CHECK(ProjectionReport(a.view, nanDir.F32(), nullptr, proj2.data(), nullptr) ==
			Status::InvalidArgument);

		// F3 — an empty tag group is rejected BEFORE any projection is written (no partial
		// result on rejection). Every row in group A (bit set) leaves group B empty.
		AlignedBuf validDir(static_cast<size_t>(pd) * sizeof(float));
		std::memset(validDir.F32(), 0, static_cast<size_t>(pd) * sizeof(float));
		validDir.F32()[0] = 1.0f;
		std::vector<float> proj3(static_cast<size_t>(a.view.count));
		for (int32_t r = 0; r < a.view.count; ++r)
		{
			proj3[static_cast<size_t>(r)] = -12345.0f; // sentinel: must stay untouched
		}
		std::vector<uint32_t> allGroupA(
			static_cast<size_t>((a.view.count + 31) / 32), 0xffffffffu);
		float sep = 0.0f;
		CHECK(ProjectionReport(a.view, validDir.F32(), allGroupA.data(), proj3.data(), &sep) ==
			Status::InvalidArgument);
		for (int32_t r = 0; r < a.view.count; ++r)
		{
			CHECK(proj3[static_cast<size_t>(r)] == -12345.0f); // untouched
		}
	}
}

// T-V2.5-10 — analytics over a real scratch-bank snapshot: tombstone exclusion (G4) and a
// concurrent-writer storm (G3, dim 3). The shipped T-V2.5-7 exercised excludeBits on a
// static bank; this drives the operators over a live scratch Snapshot().
static void TestAnalyticsScratchSnapshot()
{
	std::printf("bank analytics scratch snapshot (T-V2.5-10): tombstone exclusion + concurrency\n");
	const int32_t dims = 32;
	const int32_t capacity = 128;
	const int32_t appended = 64;
	ScratchBank scratch;
	CHECK(scratch.Create(capacity, dims, Metric::L2, Quantization::Int8) == Status::Ok);
	Rng rng(0xA11A1717C0FFEEull);
	std::vector<float> src(static_cast<size_t>(appended) * dims);
	for (auto& v : src)
	{
		v = rng.NextFloat();
	}
	for (int32_t r = 0; r < appended; ++r)
	{
		int32_t idx = -1;
		CHECK(scratch.Append(src.data() + static_cast<size_t>(r) * dims, dims, &idx) == Status::Ok);
	}
	for (int32_t r = 3; r < appended; r += 9)
	{
		CHECK(scratch.Remove(r) == Status::Ok); // a deterministic scatter of tombstones
	}

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(appended), 0u);
	CHECK(scratch.Snapshot(&snap, tombs.data()) == Status::Ok);
	const int32_t pd = snap.paddedDims;

	// Live indices (published, not tombstoned), ascending.
	std::vector<int32_t> allIdx;
	std::vector<int32_t> liveIdx;
	for (int32_t r = 0; r < snap.count; ++r)
	{
		allIdx.push_back(r);
		if (!IsExcluded(tombs.data(), r))
		{
			liveIdx.push_back(r);
		}
	}
	CHECK(!liveIdx.empty());

	// Tombstone exclusion (G4): the operator over ALL rows with the tombstone words as
	// excludeBits bit-equals the operator over the live indices alone — deletion is exclusion,
	// on a real snapshot, in both the pool and the reduction.
	{
		AlignedBuf cs1(static_cast<size_t>(pd));
		AlignedBuf cs2(static_cast<size_t>(pd));
		float sAll = 0.0f;
		float sLive = 0.0f;
		CHECK(SpreadCrossDevice(snap, allIdx.data(), static_cast<int32_t>(allIdx.size()),
			tombs.data(), Reduce::Mean, cs1.I8(), &sAll) == Status::Ok);
		CHECK(SpreadCrossDevice(snap, liveIdx.data(), static_cast<int32_t>(liveIdx.size()),
			nullptr, Reduce::Mean, cs2.I8(), &sLive) == Status::Ok);
		CHECK_MSG(sAll == sLive, "scratch spread tombstone!=subset (%.9g vs %.9g)", sAll, sLive);

		AlignedBuf da(static_cast<size_t>(pd));
		AlignedBuf db(static_cast<size_t>(pd));
		AlignedBuf da2(static_cast<size_t>(pd));
		AlignedBuf db2(static_cast<size_t>(pd));
		float dAll = 0.0f;
		float dLive = 0.0f;
		CHECK(CentroidDistanceCrossDevice(snap, allIdx.data(), static_cast<int32_t>(allIdx.size()),
			nullptr, tombs.data(), snap, liveIdx.data(), static_cast<int32_t>(liveIdx.size()),
			nullptr, nullptr, Metric::L2, da.I8(), db.I8(), &dAll) == Status::Ok);
		CHECK(CentroidDistanceCrossDevice(snap, liveIdx.data(), static_cast<int32_t>(liveIdx.size()),
			nullptr, nullptr, snap, liveIdx.data(), static_cast<int32_t>(liveIdx.size()),
			nullptr, nullptr, Metric::L2, da2.I8(), db2.I8(), &dLive) == Status::Ok);
		CHECK_MSG(dAll == dLive, "scratch centroid-distance tombstone!=subset (%.9g vs %.9g)",
			dAll, dLive);
	}

	// Concurrency (G3, dim 3): a reduction over a HELD snapshot is invariant while a writer
	// appends and removes concurrently — the snapshot's rows [0,count) and captured tombstone
	// words are stable. TSan (CI) proves the reads take no lock and do not race the writer.
	{
		AlignedBuf cs(static_cast<size_t>(pd));
		float baseline = 0.0f;
		CHECK(SpreadCrossDevice(snap, allIdx.data(), static_cast<int32_t>(allIdx.size()),
			tombs.data(), Reduce::Mean, cs.I8(), &baseline) == Status::Ok);

		std::atomic<bool> stop{false};
		std::thread writer([&]() {
			int32_t next = appended;
			while (!stop.load(std::memory_order_relaxed))
			{
				if (next < capacity)
				{
					scratch.Append(src.data(), dims, &next); // reuses row 0's data; index unused
					++next;
				}
				scratch.Remove(next > appended ? appended - 1 : 0);
			}
		});
		for (int32_t iter = 0; iter < 200; ++iter)
		{
			AlignedBuf cs2(static_cast<size_t>(pd));
			float v = 0.0f;
			CHECK(SpreadCrossDevice(snap, allIdx.data(), static_cast<int32_t>(allIdx.size()),
				tombs.data(), Reduce::Mean, cs2.I8(), &v) == Status::Ok);
			CHECK(v == baseline); // the held snapshot is stable under concurrent writes
		}
		stop.store(true, std::memory_order_relaxed);
		writer.join();
	}
}

// ---------------------------------------------------------------------------
// T-V3 -- V3.0 channel-capable scratch banks, Slot 2 (SuperFAISS_V2_Plan.md
// section 23.4 Tier 1 core: channel table at Create, append-time per-channel
// Cosine sub-norms, Snapshot carries the table). Authored red-first by Curie
// from section 23.8 (the Coverage Model) and
// Claude/Curie/SuperFAISS_V3_Test_Design.md. Slot 2 does not cover
// channel-aware Freeze or archive persistence (section 23.9 slot 3) or Tier 2
// channel-scoped analytics (slot 4) -- those suites land with their slots.
//
// SCAFFOLD (Curie, not the feature): ScratchBank's channel-table Create
// overload is declared in scratch.h but stubbed in scratch.cpp to
// unconditionally return Status::OutOfMemory -- a status no cell below
// expects -- so every test in this section compiles, links, and fails at a
// specific runtime assertion for the one true reason ("channel-capable Create
// is not implemented"), never on a compile error. Hastings replaces the stub
// body with the real channel-table validation + sub-norm-arena construction
// (section 23.4); the scaffold comment in scratch.cpp goes with it.

namespace
{
	// Independent double-precision normalize (FEAT ground truth). Deliberately
	// NOT a call to the library's NormalizeRows -- the dim-10 discipline: an
	// achievement claim's reference is independent of the operator's own
	// steps, not another call to the same code the operator (or its bake-time
	// twin) uses.
	void RefNormalizeRow(double* row, int32_t dims)
	{
		double norm = 0.0;
		for (int32_t i = 0; i < dims; ++i)
		{
			norm += row[i] * row[i];
		}
		if (norm <= 0.0)
		{
			return;
		}
		const double inv = 1.0 / std::sqrt(norm);
		for (int32_t i = 0; i < dims; ++i)
		{
			row[i] *= inv;
		}
	}

	// Slot-2 CAL bands (Curie, proposed; pin at first calibration per the
	// T-V2.5-3 convention): the FEAT tolerance for a per-channel score against
	// the independent double brute-force reference, relative to the
	// reference magnitude. Float32 has no quantization noise (residual is
	// float rounding only); Int8 carries quantization error on top.
	constexpr double kChannelFeatRelTolF32 = 1e-4;
	constexpr double kChannelFeatRelTolI8 = 0.05;

	// A channel-capable scratch-bank fixture: the raw (pre-normalize) source
	// rows fed to Append, and an INDEPENDENT double reference per row (post
	// row-normalize for Cosine, identity for Dot/L2) -- the FEAT ground truth.
	// Two channels split the row in half on the quantization's element grid,
	// mirroring TestPerChannelCosine's fixture shape.
	struct ScratchChannelFixture
	{
		ScratchBank bank;
		std::vector<ChannelInfo> channels;
		std::vector<float> rawSource;  // count x dims, exactly what Append receives
		std::vector<double> refRows;   // count x dims, independent double reference
		int32_t dims = 0;
		int32_t count = 0;
		int32_t pd = 0;
		Metric metric = Metric::Dot;
		Quantization quant = Quantization::Float32;
		Status createStatus = Status::InvalidArgument;
	};

	void MakeChannelScratchBank(
		ScratchChannelFixture& fx, Rng& rng, int32_t count, int32_t dims, Metric metric,
		Quantization quant, int32_t capacityOverride = -1)
	{
		fx.dims = dims;
		fx.count = count;
		fx.metric = metric;
		fx.quant = quant;
		fx.pd = PaddedDims(dims, quant);
		const int32_t grid = kAlignment / ElementSize(quant);
		int32_t half = (dims / 2 / grid) * grid;
		if (half <= 0)
		{
			half = grid;
		}
		fx.channels = {{0, half}, {half, fx.pd - half}};

		fx.rawSource.resize(static_cast<size_t>(count) * dims);
		fx.refRows.resize(static_cast<size_t>(count) * dims);
		for (int32_t r = 0; r < count; ++r)
		{
			for (int32_t i = 0; i < dims; ++i)
			{
				const float v = rng.NextFloat();
				fx.rawSource[static_cast<size_t>(r) * dims + i] = v;
				fx.refRows[static_cast<size_t>(r) * dims + i] = static_cast<double>(v);
			}
			if (metric == Metric::Cosine)
			{
				RefNormalizeRow(fx.refRows.data() + static_cast<size_t>(r) * dims, dims);
			}
		}

		const int32_t capacity = capacityOverride > 0 ? capacityOverride : count;
		fx.createStatus = fx.bank.Create(capacity, dims, metric, quant,
			fx.channels.data(), static_cast<int32_t>(fx.channels.size()));
		CHECK_MSG(fx.createStatus == Status::Ok,
			"channel Create failed (unimplemented, slot 2): status=%d",
			static_cast<int>(fx.createStatus));
		if (fx.createStatus != Status::Ok)
		{
			return; // slot-2 unimplemented: nothing further to append
		}
		for (int32_t r = 0; r < count; ++r)
		{
			int32_t idx = -1;
			CHECK(fx.bank.Append(fx.rawSource.data() + static_cast<size_t>(r) * dims,
					dims, &idx) == Status::Ok);
			CHECK(idx == r);
		}
	}

	// Builds the baked (imported) twin of a fixture's source rows, using the
	// SAME import pipeline baked banks use (NormalizeRows / QuantizeRowsInt8 /
	// PadRowsFloat32 / ComputeChannelInverseNorms) -- legitimate here because
	// this IS the baked-twin equality oracle (section 23.7.1), not the FEAT.
	struct BakedTwin
	{
		AlignedBuf payload;
		std::vector<float> scales;
		std::vector<float> invNorms;
		BankView view;

		explicit BakedTwin(const ScratchChannelFixture& fx)
			: payload(static_cast<size_t>(fx.count > 0 ? fx.count : 1) * fx.pd *
				  ElementSize(fx.quant))
		{
			std::vector<float> normalized = fx.rawSource;
			if (fx.metric == Metric::Cosine && fx.count > 0)
			{
				CHECK(NormalizeRows(normalized.data(), fx.count, fx.dims, nullptr) ==
					Status::Ok);
			}
			if (fx.quant == Quantization::Int8)
			{
				scales.resize(static_cast<size_t>(fx.count));
				QuantizeRowsInt8(normalized.data(), fx.count, fx.dims, fx.pd,
					payload.I8(), scales.data());
			}
			else
			{
				PadRowsFloat32(normalized.data(), fx.count, fx.dims, fx.pd, payload.F32());
			}
			view.rows = payload.ptr;
			view.scales = fx.quant == Quantization::Int8 ? scales.data() : nullptr;
			view.count = fx.count;
			view.dims = fx.dims;
			view.paddedDims = fx.pd;
			view.quant = fx.quant;
			view.metric = fx.metric;
			view.channels = fx.channels.data();
			view.channelCount = static_cast<int32_t>(fx.channels.size());
			if (fx.metric == Metric::Cosine)
			{
				invNorms.resize(static_cast<size_t>(fx.count) * fx.channels.size());
				CHECK(ComputeChannelInverseNorms(view, invNorms.data()) == Status::Ok);
				view.channelInvNorms = invNorms.data();
			}
		}
	};
} // namespace

// T-V3-1 -- channel table at Create: construction-time validation (dims 2/5).
// Closes G-4 (the >kMaxChannels rejection).
static void TestScratchChannelCreateRejections()
{
	Rng rng(0xC4A9E1ull);
	const int32_t dims = 64;

	// Valid tables accepted, every metric x quant, and capacity is honored
	// (append to capacity succeeds, the next append is OutOfMemory) -- the
	// arena sizing must have room for rows AND (Cosine) the sub-norm array in
	// the one allocation.
	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			const int32_t count = 16;
			ScratchChannelFixture fx;
				MakeChannelScratchBank(fx, rng, count, dims, metric, quant);
			if (fx.createStatus != Status::Ok)
			{
				continue; // already CHECKed red inside the fixture builder
			}
			CHECK(fx.bank.Count() == count);
			float extra[dims];
			for (int32_t i = 0; i < dims; ++i)
			{
				extra[i] = 0.1f;
			}
			CHECK(fx.bank.Append(extra, dims, nullptr) == Status::OutOfMemory);
		}
	}

	auto expectCreateRejected = [&](const ChannelInfo* channels, int32_t channelCount,
										const char* why) {
		ScratchBank bank;
		const Status s = bank.Create(64, dims, Metric::Dot, Quantization::Float32,
			channels, channelCount);
		CHECK_MSG(s == Status::InvalidArgument,
			"%s: expected InvalidArgument, got %d", why, static_cast<int>(s));
	};

	// Overlapping channels.
	{
		const ChannelInfo bad[2] = {{0, 32}, {16, 32}};
		expectCreateRejected(bad, 2, "overlapping channels");
	}
	// Non-ascending (second channel starts before the first ends, reversed order).
	{
		const ChannelInfo bad[2] = {{32, 16}, {0, 16}};
		expectCreateRejected(bad, 2, "non-ascending channels");
	}
	// Off the 16-byte element grid (Float32 grid == 4 elements).
	{
		const ChannelInfo bad[1] = {{0, 6}};
		expectCreateRejected(bad, 1, "off-grid length");
	}
	{
		const ChannelInfo bad[1] = {{2, 8}};
		expectCreateRejected(bad, 1, "off-grid offset");
	}
	// Out of bounds (extends past paddedDims).
	{
		const ChannelInfo bad[1] = {{0, dims + 64}};
		expectCreateRejected(bad, 1, "out-of-bounds channel");
	}
	// Zero-length channel.
	{
		const ChannelInfo bad[1] = {{0, 0}};
		expectCreateRejected(bad, 1, "zero-length channel");
	}
	// Negative offset.
	{
		const ChannelInfo bad[1] = {{-4, 8}};
		expectCreateRejected(bad, 1, "negative offset");
	}
	// Null table with non-zero count.
	{
		expectCreateRejected(nullptr, 1, "null table, nonzero count");
	}
	// G-4: exceeding kMaxChannels (8). A structurally valid 9-channel table
	// (ascending, non-overlapping, on-grid, in-bounds) rejected SOLELY for
	// count -- proves the cap fires independently of the other rules.
	{
		const int32_t wideDims = 9 * kAlignment; // 9 whole-grid-unit channels
		ChannelInfo nine[9];
		for (int32_t c = 0; c < 9; ++c)
		{
			nine[c] = {c * kAlignment, kAlignment};
		}
		ScratchBank bank;
		const Status s = bank.Create(64, wideDims, Metric::Dot, Quantization::Float32,
			nine, 9);
		CHECK_MSG(s == Status::InvalidArgument,
			"kMaxChannels cap (G-4): expected InvalidArgument, got %d",
			static_cast<int>(s));
	}
	// Exactly kMaxChannels (8) is accepted -- the boundary is exclusive-above,
	// not exclusive-at (dim 4 extreme, folded here since it's a Create cell).
	{
		const int32_t wideDims = 8 * kAlignment;
		ChannelInfo eight[8];
		for (int32_t c = 0; c < 8; ++c)
		{
			eight[c] = {c * kAlignment, kAlignment};
		}
		ScratchBank bank;
		const Status s = bank.Create(4, wideDims, Metric::Dot, Quantization::Float32,
			eight, 8);
		CHECK_MSG(s == Status::Ok,
			"kMaxChannels boundary: expected Ok at exactly 8, got %d",
			static_cast<int>(s));
	}
}

// T-V3-2 / T-V3-3 -- append-time per-channel Cosine sub-norms: REF equality
// against the bake.cpp:143-181 reference (ComputeChannelInverseNorms run over
// the scratch snapshot's own quantized rows), and the per-row-standalone
// property (V3-G4) via a permuted-append-order control (INV).
static void TestScratchChannelAppendSubNorms()
{
	Rng rng(0xA55B0Full);
	const int32_t dims = 64;
	const int32_t count = 40;

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		ScratchChannelFixture fx;
			MakeChannelScratchBank(fx, rng, count, dims, Metric::Cosine, quant);
		if (fx.createStatus != Status::Ok)
		{
			continue;
		}

		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		CHECK_MSG(snap.channelCount == 2, "Snapshot did not carry the channel table: %d",
			snap.channelCount);
		CHECK(snap.channels != nullptr);
		CHECK_MSG(snap.channelInvNorms != nullptr,
			"Snapshot did not carry the per-channel sub-norm arena");
		if (snap.channelCount != 2 || snap.channels == nullptr ||
			snap.channelInvNorms == nullptr)
		{
			continue;
		}

		// REF: the bake.cpp reference applied to the snapshot's OWN quantized
		// rows/scales/channels must bit-equal the append-time-derived array.
		std::vector<float> refInvNorms(static_cast<size_t>(count) * snap.channelCount);
		CHECK(ComputeChannelInverseNorms(snap, refInvNorms.data()) == Status::Ok);
		CHECK_MSG(std::memcmp(refInvNorms.data(), snap.channelInvNorms,
					  refInvNorms.size() * sizeof(float)) == 0,
			"append-time sub-norm does not bit-equal the bake.cpp reference (quant=%d)",
			static_cast<int>(quant));
	}

	// Permuted-append-order control (V3-G4): the same source rows, appended in
	// two different orders, yield IDENTICAL per-row sub-norms at each row's
	// own (possibly different) index -- the row-standalone property.
	{
		ScratchChannelFixture inOrder;
		MakeChannelScratchBank(inOrder, rng, count, dims, Metric::Cosine, Quantization::Int8);
		if (inOrder.createStatus != Status::Ok)
		{
			return;
		}

		std::vector<int32_t> permutation(count);
		for (int32_t i = 0; i < count; ++i)
		{
			permutation[i] = count - 1 - i; // simple reversal
		}
		ScratchBank permBank;
		const Status created = permBank.Create(count, dims, Metric::Cosine,
			Quantization::Int8, inOrder.channels.data(),
			static_cast<int32_t>(inOrder.channels.size()));
		CHECK_MSG(created == Status::Ok,
			"permuted-order bank Create failed (unimplemented, slot 2): status=%d",
			static_cast<int>(created));
		if (created != Status::Ok)
		{
			return;
		}
		std::vector<int32_t> permIndex(count, -1); // source row -> permuted-bank index
		for (int32_t p = 0; p < count; ++p)
		{
			const int32_t sourceRow = permutation[p];
			int32_t idx = -1;
			CHECK(permBank.Append(
					  inOrder.rawSource.data() + static_cast<size_t>(sourceRow) * dims,
					  dims, &idx) == Status::Ok);
			permIndex[sourceRow] = idx;
		}

		BankView snapA, snapB;
		std::vector<uint32_t> tombsA(ScratchBank::TombstoneWords(count), 0u);
		std::vector<uint32_t> tombsB(ScratchBank::TombstoneWords(count), 0u);
		CHECK(inOrder.bank.Snapshot(&snapA, tombsA.data()) == Status::Ok);
		CHECK(permBank.Snapshot(&snapB, tombsB.data()) == Status::Ok);
		if (snapA.channelInvNorms == nullptr || snapB.channelInvNorms == nullptr)
		{
			return; // already CHECKed red above
		}
		for (int32_t sourceRow = 0; sourceRow < count; ++sourceRow)
		{
			const float* a = snapA.channelInvNorms +
				static_cast<int64_t>(sourceRow) * snapA.channelCount;
			const float* b = snapB.channelInvNorms +
				static_cast<int64_t>(permIndex[sourceRow]) * snapB.channelCount;
			CHECK_MSG(std::memcmp(a, b, static_cast<size_t>(snapA.channelCount) *
						  sizeof(float)) == 0,
				"row %d's sub-norms depend on append order (not per-row-standalone)",
				sourceRow);
		}
	}
}

// T-V3-4 -- Snapshot carries the channel table; a channel query over the
// scratch snapshot bit-equals the SAME query over a baked twin (section
// 23.7.1, the crux consistency check -- kept alongside T-V3-5's feature
// oracle per Japp's note that equality oracles never stand alone for an
// achievement claim). Also: raw-segment and whole-vector regression (dim 7 --
// "raw segments already worked; named channels are the addition", "no
// whole-vector path changed").
static void TestScratchChannelSnapshotBakedTwin()
{
	Rng rng(0x5A17ED0ull);
	const int32_t dims = 64;
	const int32_t count = 96;
	const int32_t k = 12;

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			ScratchChannelFixture fx;
				MakeChannelScratchBank(fx, rng, count, dims, metric, quant);
			if (fx.createStatus != Status::Ok)
			{
				continue;
			}
			BankView snap;
			std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
			CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);
			BakedTwin twin(fx);

			std::vector<float> queryRaw(dims);
			for (auto& v : queryRaw)
			{
				v = rng.NextFloat();
			}
			AlignedBuf qbuf(static_cast<size_t>(fx.pd) * sizeof(float));
			PadQuery(queryRaw, fx.pd, qbuf.F32());

			const QuerySegment segs[2] = {
				{fx.channels[0].offset, fx.channels[0].length, 1.0f},
				{fx.channels[1].offset, fx.channels[1].length, 0.75f},
			};
			QueryParams params;
			params.k = k;
			params.segments = segs;
			params.segmentCount = 2;

			Workspace wsScratch, wsBaked;
			Hit scratchHits[12], bakedHits[12];
			int32_t nScratch = 0, nBaked = 0;
			CHECK(Query(snap, qbuf.F32(), params, wsScratch, scratchHits, &nScratch) ==
				Status::Ok);
			CHECK(Query(twin.view, qbuf.F32(), params, wsBaked, bakedHits, &nBaked) ==
				Status::Ok);
			CHECK_MSG(nScratch == nBaked, "hit count mismatch: scratch=%d baked=%d",
				nScratch, nBaked);
			for (int32_t i = 0; i < nScratch && i < nBaked; ++i)
			{
				CHECK_MSG(scratchHits[i].index == bakedHits[i].index &&
						scratchHits[i].score == bakedHits[i].score,
					"channel query over scratch snapshot != baked twin at hit %d "
					"(metric=%d quant=%d)",
					i, static_cast<int>(metric), static_cast<int>(quant));
			}

			// Raw-segment regression: a degenerate whole-row segment list is
			// bit-identical to the plain (no-segment) scan, on a channel-
			// carrying scratch bank exactly as it was on a channel-less one
			// (dim 7 -- "no whole-vector path changed").
			{
				const QuerySegment whole[1] = {{0, fx.pd, 1.0f}};
				QueryParams wp;
				wp.k = k;
				wp.segments = whole;
				wp.segmentCount = 1;
				QueryParams plain;
				plain.k = k;

				Workspace wsW, wsP;
				Hit wHits[12], pHits[12];
				int32_t nW = 0, nP = 0;
				CHECK(Query(snap, qbuf.F32(), wp, wsW, wHits, &nW) == Status::Ok);
				CHECK(Query(snap, qbuf.F32(), plain, wsP, pHits, &nP) == Status::Ok);
				CHECK(nW == nP);
				for (int32_t i = 0; i < nW && i < nP; ++i)
				{
					CHECK(wHits[i].index == pHits[i].index &&
						wHits[i].score == pHits[i].score);
				}
			}
		}
	}
}

// T-V3-5 -- FEAT: the scratch per-channel query vs an independent double
// brute-force top-k over the DEQUANTIZED snapshot rows, computed from the
// per-channel-cosine/dot/L2 DEFINITION, not from bake.cpp's own steps.
// Executes on the mutable path (a scratch channel bank), closing Japp's S-1
// (the FEAT anchor path) and covering N-1 (the metric matrix: Dot, Cosine, L2
// each proven per channel).
static void TestScratchChannelFeatureOracle()
{
	Rng rng(0xFEA7ull);
	const int32_t dims = 64;
	const int32_t count = 120;
	const int32_t k = 10;

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			ScratchChannelFixture fx;
				MakeChannelScratchBank(fx, rng, count, dims, metric, quant);
			if (fx.createStatus != Status::Ok)
			{
				continue;
			}
			BankView snap;
			std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
			CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);

			std::vector<float> queryRaw(dims);
			for (auto& v : queryRaw)
			{
				v = rng.NextFloat();
			}
			AlignedBuf qbuf(static_cast<size_t>(fx.pd) * sizeof(float));
			PadQuery(queryRaw, fx.pd, qbuf.F32());

			const ChannelInfo& ch = fx.channels[0]; // the first channel
			const QuerySegment seg[1] = {{ch.offset, ch.length, 1.0f}};
			QueryParams params;
			params.k = k;
			params.segments = seg;
			params.segmentCount = 1;

			Workspace ws;
			Hit hits[10];
			int32_t n = 0;
			CHECK(Query(snap, qbuf.F32(), params, ws, hits, &n) == Status::Ok);

			// Independent double brute-force from the DEFINITION: per-channel
			// dot / squared-L2-distance / cosine (dot over sub-vector norms)
			// computed directly from fx.refRows -- never touching
			// ComputeChannelInverseNorms, the kernel, or QuantizeRowsInt8.
			struct RefHitLocal
			{
				int32_t index;
				double score;
			};
			std::vector<RefHitLocal> ref;
			ref.reserve(static_cast<size_t>(count));
			for (int32_t r = 0; r < count; ++r)
			{
				const double* row = fx.refRows.data() + static_cast<size_t>(r) * dims;
				double score = 0.0;
				if (metric == Metric::Dot)
				{
					for (int32_t j = ch.offset; j < ch.offset + ch.length && j < dims; ++j)
					{
						score += static_cast<double>(qbuf.F32()[j]) * row[j];
					}
				}
				else if (metric == Metric::L2)
				{
					for (int32_t j = ch.offset; j < ch.offset + ch.length && j < dims; ++j)
					{
						const double d = static_cast<double>(qbuf.F32()[j]) - row[j];
						score += d * d;
					}
				}
				else // Cosine: dot(query_sub, row_sub) / ||row_sub|| -- the
					 // SUB-VECTOR norm of the (whole-row-unit-normalized) row,
					 // never the whole-row norm (the exact D-V2-1 contract
					 // point a wrong-but-deterministic implementation could
					 // get backwards, per Japp S-1/G-1).
				{
					double dot = 0.0, subNorm = 0.0;
					for (int32_t j = ch.offset; j < ch.offset + ch.length && j < dims; ++j)
					{
						dot += static_cast<double>(qbuf.F32()[j]) * row[j];
						subNorm += row[j] * row[j];
					}
					score = subNorm > 0.0 ? dot / std::sqrt(subNorm) : 0.0;
				}
				ref.push_back({r, score});
			}
			std::sort(ref.begin(), ref.end(), [&](const RefHitLocal& a, const RefHitLocal& b) {
				if (a.score != b.score)
				{
					return metric == Metric::L2 ? (a.score < b.score) : (a.score > b.score);
				}
				return a.index < b.index;
			});

			const double tol = quant == Quantization::Int8 ? kChannelFeatRelTolI8
															 : kChannelFeatRelTolF32;
			const int32_t expected = static_cast<int32_t>(ref.size()) < k
				? static_cast<int32_t>(ref.size())
				: k;
			CHECK_MSG(n == expected, "FEAT hit count: got %d expected %d", n, expected);
			const double boundary = ref[static_cast<size_t>(expected) - 1].score;
			std::vector<double> refByIndex(static_cast<size_t>(count),
				metric == Metric::L2 ? 1e300 : -1e300);
			for (const RefHitLocal& h : ref)
			{
				refByIndex[static_cast<size_t>(h.index)] = h.score;
			}
			for (int32_t i = 0; i < n; ++i)
			{
				const double rs = refByIndex[static_cast<size_t>(hits[i].index)];
				const double band = tol * (1.0 + std::fabs(boundary));
				const bool inTrueTopK =
					metric == Metric::L2 ? (rs <= boundary + band) : (rs >= boundary - band);
				CHECK_MSG(inTrueTopK,
					"FEAT: hit %d (row %d) not within CAL band of the definition-grounded "
					"brute-force (metric=%d quant=%d)",
					i, hits[i].index, static_cast<int>(metric), static_cast<int>(quant));
				CHECK_MSG(std::fabs(rs - static_cast<double>(hits[i].score)) <= band,
					"FEAT score drift: got %.9g ref %.9g (metric=%d quant=%d)",
					static_cast<double>(hits[i].score), rs, static_cast<int>(metric),
					static_cast<int>(quant));
			}
		}
	}
}

// Zero-norm row channel scores 0, never NaN (defined, dim 5); zero-norm
// Cosine query sub-vector on a nonzero-weight segment -> ZeroNormQuery
// (dim 2/5).
static void TestScratchChannelDegenerateAndRuntimeRejections()
{
	const int32_t d = 8;
	const ChannelInfo channels[2] = {{0, 4}, {4, 4}};

	ScratchBank bank;
	const Status created =
		bank.Create(4, d, Metric::Cosine, Quantization::Float32, channels, 2);
	CHECK_MSG(created == Status::Ok,
		"zero-norm-channel fixture Create failed (unimplemented, slot 2): status=%d",
		static_cast<int>(created));
	if (created != Status::Ok)
	{
		return;
	}

	// Row 0: energy only in channel 0. Row 1: energy only in channel 1 (its
	// channel-0 sub-vector is all zero -- a zero-norm ROW CHANNEL, legal).
	float row0[d] = {1, 0, 0, 0, 0, 0, 0, 0};
	float row1[d] = {0, 0, 0, 0, 1, 0, 0, 0};
	CHECK(bank.Append(row0, d, nullptr) == Status::Ok);
	CHECK(bank.Append(row1, d, nullptr) == Status::Ok);

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(2), 0u);
	CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
	if (snap.channelInvNorms == nullptr)
	{
		return; // already CHECKed red
	}
	CHECK_MSG(snap.channelInvNorms[1 * 2 + 0] == 0.0f,
		"row 1's channel-0 inverse sub-norm should be 0 (zero-norm row channel)");

	alignas(16) float q[8] = {1.0f, 0, 0, 0, 0, 0, 0, 0};
	const QuerySegment ch0[1] = {{0, 4, 1.0f}};
	QueryParams p;
	p.k = 2;
	p.segments = ch0;
	p.segmentCount = 1;
	Workspace ws;
	Hit hits[2];
	int32_t n = 0;
	CHECK(Query(snap, q, p, ws, hits, &n) == Status::Ok);
	for (int32_t i = 0; i < n; ++i)
	{
		CHECK(hits[i].score == hits[i].score); // not NaN
		if (hits[i].index == 1)
		{
			CHECK_MSG(hits[i].score == 0.0f,
				"zero-norm row channel should score exactly 0, got %g", hits[i].score);
		}
	}

	// Zero-norm QUERY sub-vector on a nonzero-weight Cosine segment ->
	// ZeroNormQuery (the whole-vector zero-norm law, applied per scored
	// segment).
	alignas(16) float zeroSubQuery[8] = {0, 0, 0, 0, 1.0f, 0, 0, 0}; // ch0 sub-vector all-zero
	Hit hits2[2];
	int32_t n2 = 0;
	CHECK(Query(snap, zeroSubQuery, p, ws, hits2, &n2) == Status::ZeroNormQuery);
}

// T-V3-7 -- composition reachable at slot 2 (dim 8): channel query x
// {scratch-snapshot tombstone exclusion, generic excludeBits, CrossDevice
// int8, batch, decomposition, intersection (G-2), metric override (G-2)},
// and channel query x per-row bias, both dense and sparse forms (Forge W2,
// N-3).
static void TestScratchChannelComposition()
{
	Rng rng(0xC0F5171Full);
	const int32_t dims = 64;
	const int32_t count = 128;
	const int32_t k = 10;

	ScratchChannelFixture fx;
		MakeChannelScratchBank(fx, rng, count, dims, Metric::Cosine, Quantization::Int8);
	if (fx.createStatus != Status::Ok)
	{
		return;
	}

	std::vector<float> queryRaw(dims);
	for (auto& v : queryRaw)
	{
		v = rng.NextFloat();
	}
	AlignedBuf qbuf(static_cast<size_t>(fx.pd) * sizeof(float));
	PadQuery(queryRaw, fx.pd, qbuf.F32());
	const ChannelInfo& ch0 = fx.channels[0];
	const QuerySegment seg[1] = {{ch0.offset, ch0.length, 1.0f}};

	// 1) Tombstone exclusion: a channel query over a snapshot, excludeBits =
	// the snapshot's own tombstone words, bit-equals the same channel query
	// over a bare bank built from only the live rows (deletion-is-exclusion,
	// on a real snapshot, extended to channels -- the T-V2.5-10 shape).
	{
		for (int32_t r = 3; r < count; r += 5)
		{
			CHECK(fx.bank.Remove(r) == Status::Ok);
		}
		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);

		QueryParams p;
		p.k = k;
		p.segments = seg;
		p.segmentCount = 1;
		p.excludeBits = tombs.data();
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
		for (int32_t i = 0; i < n; ++i)
		{
			CHECK(!IsExcluded(tombs.data(), hits[i].index));
		}

		// Bare live-only bank, built directly from the same fixture rows.
		std::vector<float> liveSource;
		std::vector<int32_t> liveOldIndex;
		for (int32_t r = 0; r < count; ++r)
		{
			if (IsExcluded(tombs.data(), r))
			{
				continue;
			}
			liveSource.insert(liveSource.end(),
				fx.rawSource.begin() + static_cast<size_t>(r) * dims,
				fx.rawSource.begin() + static_cast<size_t>(r + 1) * dims);
			liveOldIndex.push_back(r);
		}
		const int32_t liveCount = static_cast<int32_t>(liveOldIndex.size());
		CHECK(NormalizeRows(liveSource.data(), liveCount, dims, nullptr) == Status::Ok);
		AlignedBuf liveRows(static_cast<size_t>(liveCount) * fx.pd);
		std::vector<float> liveScales(static_cast<size_t>(liveCount));
		QuantizeRowsInt8(liveSource.data(), liveCount, dims, fx.pd, liveRows.I8(),
			liveScales.data());
		BankView liveView;
		liveView.rows = liveRows.ptr;
		liveView.scales = liveScales.data();
		liveView.count = liveCount;
		liveView.dims = dims;
		liveView.paddedDims = fx.pd;
		liveView.quant = Quantization::Int8;
		liveView.metric = Metric::Cosine;
		liveView.channels = fx.channels.data();
		liveView.channelCount = static_cast<int32_t>(fx.channels.size());
		std::vector<float> liveInvNorms(
			static_cast<size_t>(liveCount) * fx.channels.size());
		CHECK(ComputeChannelInverseNorms(liveView, liveInvNorms.data()) == Status::Ok);
		liveView.channelInvNorms = liveInvNorms.data();

		QueryParams pl;
		pl.k = k;
		pl.segments = seg;
		pl.segmentCount = 1;
		Workspace wsl;
		Hit liveHits[10];
		int32_t nLive = 0;
		CHECK(Query(liveView, qbuf.F32(), pl, wsl, liveHits, &nLive) == Status::Ok);
		CHECK_MSG(n == nLive, "tombstone-excluded channel query count mismatch: %d vs %d",
			n, nLive);
		for (int32_t i = 0; i < n && i < nLive; ++i)
		{
			CHECK(liveOldIndex[static_cast<size_t>(liveHits[i].index)] == hits[i].index);
			CHECK(liveHits[i].score == hits[i].score);
		}
	}

	// Re-snapshot (post-removal state) for the remaining composition cells.
	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);

	// 2) CrossDevice int8 channel query over the scratch snapshot: bit-equals
	// the same CrossDevice channel query over the baked twin (kernel already
	// composes, V3-G11 -- only the snapshot's channel fields are new).
	{
		BakedTwin twin(fx);
		QueryParams p;
		p.k = k;
		p.segments = seg;
		p.segmentCount = 1;
		p.excludeBits = tombs.data();
		p.exactness = Exactness::CrossDevice;
		Workspace wsS, wsB;
		Hit sHits[10], bHits[10];
		int32_t nS = 0, nB = 0;
		// The baked twin carries every row (no tombstones); querying it
		// without exclusion is the composition-reachability half of this
		// cell -- CrossDevice + channel segments must not reject
		// (InvalidArgument) on either bank type. The scratch-vs-baked
		// bit-equality claim is T-V3-4's (PerDevice); this cell's job is
		// proving CrossDevice mode itself composes with channels on the
		// mutable path, which the kernel already supports (V3-G11) once
		// Snapshot carries the table.
		CHECK(Query(snap, qbuf.F32(), p, wsS, sHits, &nS) == Status::Ok);
		QueryParams pBaked = p;
		pBaked.excludeBits = nullptr;
		CHECK(Query(twin.view, qbuf.F32(), pBaked, wsB, bHits, &nB) == Status::Ok);
	}

	// 3) Batch: QueryBatch with segments over the scratch snapshot matches
	// per-query single Query() calls, bit-identical (existing batch==single
	// law, extended to channels).
	{
		const int32_t m = 3;
		std::vector<float> queries(static_cast<size_t>(m) * fx.pd, 0.0f);
		for (int32_t q = 0; q < m; ++q)
		{
			for (int32_t i = 0; i < dims; ++i)
			{
				queries[static_cast<size_t>(q) * fx.pd + i] = rng.NextFloat();
			}
		}
		QueryParams p;
		p.k = k;
		p.segments = seg;
		p.segmentCount = 1;
		Workspace wsBatch;
		std::vector<Hit> batchHits(static_cast<size_t>(m) * k);
		std::vector<int32_t> batchCounts(m);
		CHECK(QueryBatch(snap, queries.data(), m, p, wsBatch, batchHits.data(),
				batchCounts.data()) == Status::Ok);
		for (int32_t q = 0; q < m; ++q)
		{
			Workspace wsSingle;
			std::vector<Hit> single(k);
			int32_t nSingle = 0;
			CHECK(Query(snap, queries.data() + static_cast<size_t>(q) * fx.pd, p, wsSingle,
					single.data(), &nSingle) == Status::Ok);
			CHECK(nSingle == batchCounts[static_cast<size_t>(q)]);
			for (int32_t i = 0; i < nSingle; ++i)
			{
				CHECK(single[static_cast<size_t>(i)].index ==
						batchHits[static_cast<size_t>(q) * k + i].index &&
					single[static_cast<size_t>(i)].score ==
						batchHits[static_cast<size_t>(q) * k + i].score);
			}
		}
	}

	// 4) Decomposition: per-channel contributions sum bit-exactly to the same
	// total the scan produced, over a scratch snapshot.
	{
		const QuerySegment segs2[2] = {
			{fx.channels[0].offset, fx.channels[0].length, 1.0f},
			{fx.channels[1].offset, fx.channels[1].length, 0.5f},
		};
		QueryParams p;
		p.k = 1;
		p.segments = segs2;
		p.segmentCount = 2;
		Workspace ws;
		Hit top[1];
		int32_t n = 0;
		CHECK(Query(snap, qbuf.F32(), p, ws, top, &n) == Status::Ok);
		if (n == 1)
		{
			float contributions[2] = {0.0f, 0.0f};
			const float total = DecomposeRowScore(snap, qbuf.F32(), top[0].index, segs2, 2,
				contributions);
			CHECK(total == top[0].score);
			CHECK((contributions[0] + contributions[1]) == total);
		}
	}

	// 5) Intersection (G-2): queryCount==1 degenerates to Query() bit-
	// identically (the core composition law), and a two-query fused call is
	// reachable (Status::Ok) over the channel-carrying scratch snapshot.
	{
		QueryParams p;
		p.k = k;
		p.segments = seg;
		p.segmentCount = 1;
		Workspace ws1, wsI;
		Hit plainHits[10], interHits[10];
		int32_t nPlain = 0, nInter = 0;
		CHECK(Query(snap, qbuf.F32(), p, ws1, plainHits, &nPlain) == Status::Ok);
		CHECK(QueryIntersect(snap, qbuf.F32(), 1, p, wsI, interHits, &nInter) == Status::Ok);
		CHECK(nPlain == nInter);
		for (int32_t i = 0; i < nPlain && i < nInter; ++i)
		{
			CHECK(plainHits[i].index == interHits[i].index &&
				plainHits[i].score == interHits[i].score);
		}

		std::vector<float> queries2(2 * static_cast<size_t>(fx.pd));
		std::memcpy(queries2.data(), qbuf.F32(), static_cast<size_t>(fx.pd) * sizeof(float));
		std::vector<float> q2Raw(dims);
		for (auto& v : q2Raw)
		{
			v = rng.NextFloat();
		}
		AlignedBuf qbuf2(static_cast<size_t>(fx.pd) * sizeof(float));
		PadQuery(q2Raw, fx.pd, qbuf2.F32());
		std::memcpy(queries2.data() + fx.pd, qbuf2.F32(),
			static_cast<size_t>(fx.pd) * sizeof(float));
		Workspace wsI2;
		Hit fusedHits[10];
		int32_t nFused = 0;
		CHECK(QueryIntersect(snap, queries2.data(), 2, p, wsI2, fusedHits, &nFused) ==
			Status::Ok);
	}

	// 6) Metric override (G-2): ScoreAs::Dot on the Cosine channel scratch
	// snapshot folds to raw projection, bit-identical to the same channel
	// query run on a channel-less copy of the snapshot with the same
	// override (the TestPerChannelCosine cell, ported to scratch).
	{
		QueryParams po;
		po.k = k;
		po.segments = seg;
		po.segmentCount = 1;
		po.scoreAs = ScoreAs::Dot;
		Workspace wsO;
		Hit projHits[10];
		int32_t nProj = 0;
		CHECK(Query(snap, qbuf.F32(), po, wsO, projHits, &nProj) == Status::Ok);

		BankView bare = snap;
		bare.channels = nullptr;
		bare.channelCount = 0;
		bare.channelInvNorms = nullptr;
		Workspace wsB;
		Hit bareHits[10];
		int32_t nBare = 0;
		CHECK(Query(bare, qbuf.F32(), po, wsB, bareHits, &nBare) == Status::Ok);
		CHECK(nProj == nBare);
		for (int32_t i = 0; i < nProj && i < nBare; ++i)
		{
			CHECK(projHits[i].index == bareHits[i].index &&
				projHits[i].score == bareHits[i].score);
		}
	}

	// 7) Channel query x per-row bias (Forge W2, N-3): both the dense
	// (count-length view) and sparse ((index,bias) pairs) forms compose --
	// the composed score is unbiased-channel-score + bias, bitwise, for
	// every returned hit.
	{
		QueryParams p;
		p.k = k;
		p.segments = seg;
		p.segmentCount = 1;
		Workspace wsUnbiased;
		std::vector<Hit> all(count);
		int32_t nAll = 0;
		QueryParams full = p;
		full.k = count;
		CHECK(Query(snap, qbuf.F32(), full, wsUnbiased, all.data(), &nAll) == Status::Ok);

		// Dense.
		{
			std::vector<float> dense(count, 0.0f);
			for (auto& b : dense)
			{
				b = rng.NextFloat() * 0.2f;
			}
			RowBias rb;
			rb.dense = dense.data();
			QueryParams pb = p;
			pb.bias = &rb;
			Workspace ws;
			Hit hits[10];
			int32_t n = 0;
			CHECK(Query(snap, qbuf.F32(), pb, ws, hits, &n) == Status::Ok);
			for (int32_t i = 0; i < n; ++i)
			{
				float unbiased = 0.0f;
				bool found = false;
				for (int32_t j = 0; j < nAll; ++j)
				{
					if (all[static_cast<size_t>(j)].index == hits[i].index)
					{
						unbiased = all[static_cast<size_t>(j)].score;
						found = true;
						break;
					}
				}
				CHECK(found);
				CHECK(hits[i].score == unbiased + dense[static_cast<size_t>(hits[i].index)]);
			}
		}
		// Sparse.
		{
			std::vector<BiasPair> pairs;
			for (int32_t r = 0; r < count; r += 11)
			{
				pairs.push_back({r, 0.15f * static_cast<float>((r % 3) + 1)});
			}
			RowBias rb;
			rb.pairs = pairs.data();
			rb.pairCount = static_cast<int32_t>(pairs.size());
			QueryParams pb = p;
			pb.bias = &rb;
			Workspace ws;
			Hit hits[10];
			int32_t n = 0;
			CHECK(Query(snap, qbuf.F32(), pb, ws, hits, &n) == Status::Ok);
			for (int32_t i = 0; i < n; ++i)
			{
				float unbiased = 0.0f;
				bool found = false;
				for (int32_t j = 0; j < nAll; ++j)
				{
					if (all[static_cast<size_t>(j)].index == hits[i].index)
					{
						unbiased = all[static_cast<size_t>(j)].score;
						found = true;
						break;
					}
				}
				CHECK(found);
				float expectedBias = 0.0f;
				for (const BiasPair& bp : pairs)
				{
					if (bp.index == hits[i].index)
					{
						expectedBias = bp.bias;
						break;
					}
				}
				CHECK(hits[i].score == unbiased + expectedBias);
			}
		}
	}
}

// T-V3-8 -- lifetime/reuse (dim 1), shape/platform extremes (dim 4), and
// contract-claim regressions (dim 7) reachable at slot 2.
static void TestScratchChannelLifetimeShapeContracts()
{
	Rng rng(0x11FE7C0Dull);
	const int32_t dims = 64;

	// dim 1 -- flat AllocationCount across appends (with the sub-norm write)
	// and across channel queries of differing segment shapes, once warm.
	{
		const int32_t count = 64;
		ScratchChannelFixture fx;
			MakeChannelScratchBank(fx, rng, count, dims, Metric::Cosine, Quantization::Int8,
				count + 32);
		if (fx.createStatus != Status::Ok)
		{
			return;
		}
		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count + 32), 0u);
		CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);

		AlignedBuf qbuf(static_cast<size_t>(fx.pd) * sizeof(float));
		std::vector<float> qraw(fx.rawSource.begin(), fx.rawSource.begin() + dims);
		PadQuery(qraw, fx.pd, qbuf.F32());

		const QuerySegment single[1] = {{fx.channels[0].offset, fx.channels[0].length, 1.0f}};
		const QuerySegment both[2] = {
			{fx.channels[0].offset, fx.channels[0].length, 1.0f},
			{fx.channels[1].offset, fx.channels[1].length, 1.0f},
		};
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		QueryParams p1;
		p1.k = 10;
		p1.segments = single;
		p1.segmentCount = 1;
		CHECK(Query(snap, qbuf.F32(), p1, ws, hits, &n) == Status::Ok); // warm-up
		QueryParams p2 = p1;
		p2.segments = both;
		p2.segmentCount = 2;
		CHECK(Query(snap, qbuf.F32(), p2, ws, hits, &n) == Status::Ok); // warm-up (differing shape)

		const uint64_t allocsBefore = AllocationCount();
		int32_t extraIdx = -1;
		float extra[dims];
		std::memcpy(extra, fx.rawSource.data(), static_cast<size_t>(dims) * sizeof(float));
		CHECK(fx.bank.Append(extra, dims, &extraIdx) == Status::Ok); // append with sub-norm write
		CHECK(Query(snap, qbuf.F32(), p1, ws, hits, &n) == Status::Ok);
		CHECK(Query(snap, qbuf.F32(), p2, ws, hits, &n) == Status::Ok);
		CHECK_MSG(AllocationCount() == allocsBefore,
			"channel append/query allocated: %llu -> %llu",
			static_cast<unsigned long long>(allocsBefore),
			static_cast<unsigned long long>(AllocationCount()));
	}

	// Grow preserves the sub-norm arena (parity with T-044 W4 index
	// preservation, extended to the sub-norm array): original rows' per-
	// channel sub-norms are bit-unchanged after Grow.
	{
		const int32_t count = 40;
		ScratchChannelFixture fx;
			MakeChannelScratchBank(fx, rng, count, dims, Metric::Cosine, Quantization::Int8);
		if (fx.createStatus != Status::Ok)
		{
			return;
		}
		BankView preGrow;
		std::vector<uint32_t> preTombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(fx.bank.Snapshot(&preGrow, preTombs.data()) == Status::Ok);
		if (preGrow.channelInvNorms == nullptr)
		{
			return;
		}
		std::vector<float> before(
			preGrow.channelInvNorms,
			preGrow.channelInvNorms + static_cast<size_t>(count) * preGrow.channelCount);

		CHECK(fx.bank.Grow(count + 32) == Status::Ok);
		BankView postGrow;
		std::vector<uint32_t> postTombs(ScratchBank::TombstoneWords(count + 32), 0u);
		CHECK(fx.bank.Snapshot(&postGrow, postTombs.data()) == Status::Ok);
		if (postGrow.channelInvNorms == nullptr)
		{
			return;
		}
		CHECK_MSG(std::memcmp(before.data(), postGrow.channelInvNorms,
					  before.size() * sizeof(float)) == 0,
			"Grow did not preserve the per-channel sub-norm arena bit-for-bit");
	}

	// dim 4 extremes.
	{
		// One-16-grid-unit channel (the smallest legal channel).
		{
			const int32_t d = kAlignment; // exactly one grid unit, Int8
			const ChannelInfo one[1] = {{0, kAlignment}};
			ScratchBank bank;
			CHECK(bank.Create(4, d, Metric::Cosine, Quantization::Int8, one, 1) ==
				Status::Ok);
		}
		// Whole-row-covering single channel bit-equals the whole-vector path.
		{
			const int32_t count = 32;
			ScratchChannelFixture wholeFx;
			MakeChannelScratchBank(wholeFx, rng, count, dims, Metric::Dot, Quantization::Float32);
			if (wholeFx.createStatus == Status::Ok)
			{
				// Override to a single channel spanning the whole row.
				ScratchBank singleChan;
				const ChannelInfo whole[1] = {{0, wholeFx.pd}};
				CHECK(singleChan.Create(count, dims, Metric::Dot, Quantization::Float32,
						whole, 1) == Status::Ok);
				for (int32_t r = 0; r < count; ++r)
				{
					CHECK(singleChan.Append(
							  wholeFx.rawSource.data() + static_cast<size_t>(r) * dims, dims,
							  nullptr) == Status::Ok);
				}
				BankView snap;
				std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
				CHECK(singleChan.Snapshot(&snap, tombs.data()) == Status::Ok);

				AlignedBuf qbuf(static_cast<size_t>(wholeFx.pd) * sizeof(float));
				std::vector<float> qraw(wholeFx.rawSource.begin(),
					wholeFx.rawSource.begin() + dims);
				PadQuery(qraw, wholeFx.pd, qbuf.F32());

				const QuerySegment segFull[1] = {{0, wholeFx.pd, 1.0f}};
				QueryParams segParams;
				segParams.k = 5;
				segParams.segments = segFull;
				segParams.segmentCount = 1;
				QueryParams plainParams;
				plainParams.k = 5;

				Workspace ws1, ws2;
				Hit segHits[5], plainHits[5];
				int32_t nSeg = 0, nPlain = 0;
				CHECK(Query(snap, qbuf.F32(), segParams, ws1, segHits, &nSeg) == Status::Ok);
				CHECK(Query(snap, qbuf.F32(), plainParams, ws2, plainHits, &nPlain) ==
					Status::Ok);
				CHECK(nSeg == nPlain);
				for (int32_t i = 0; i < nSeg && i < nPlain; ++i)
				{
					CHECK(segHits[i].index == plainHits[i].index &&
						segHits[i].score == plainHits[i].score);
				}
			}
		}
		// One live row.
		{
			const ChannelInfo channels[1] = {{0, PaddedDims(dims, Quantization::Float32)}};
			ScratchBank bank;
			CHECK(bank.Create(1, dims, Metric::Dot, Quantization::Float32, channels, 1) ==
				Status::Ok);
			float row[dims];
			for (int32_t i = 0; i < dims; ++i)
			{
				row[i] = 1.0f;
			}
			CHECK(bank.Append(row, dims, nullptr) == Status::Ok);
			BankView snap;
			std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(1), 0u);
			CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
			AlignedBuf q(static_cast<size_t>(PaddedDims(dims, Quantization::Float32)) *
				sizeof(float));
			for (int32_t i = 0; i < dims; ++i)
			{
				q.F32()[i] = 1.0f;
			}
			const QuerySegment seg[1] = {{0, PaddedDims(dims, Quantization::Float32), 1.0f}};
			QueryParams p;
			p.k = 5;
			p.segments = seg;
			p.segmentCount = 1;
			Workspace ws;
			Hit hits[5];
			int32_t n = 0;
			CHECK(Query(snap, q.F32(), p, ws, hits, &n) == Status::Ok);
			CHECK(n == 1);
		}
		// Fully-tombstoned channel bank: 0 hits, defined, no crash/NaN.
		{
			const int32_t count = 8;
			ScratchChannelFixture fx;
			MakeChannelScratchBank(fx, rng, count, dims, Metric::Cosine,				Quantization::Int8);
			if (fx.createStatus == Status::Ok)
			{
				for (int32_t r = 0; r < count; ++r)
				{
					CHECK(fx.bank.Remove(r) == Status::Ok);
				}
				BankView snap;
				std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
				CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);
				AlignedBuf qbuf(static_cast<size_t>(fx.pd) * sizeof(float));
				std::vector<float> qraw(fx.rawSource.begin(), fx.rawSource.begin() + dims);
				PadQuery(qraw, fx.pd, qbuf.F32());
				const QuerySegment seg[1] = {{fx.channels[0].offset, fx.channels[0].length, 1.0f}};
				QueryParams p;
				p.k = 5;
				p.segments = seg;
				p.segmentCount = 1;
				p.excludeBits = tombs.data();
				Workspace ws;
				Hit hits[5];
				int32_t n = 0;
				CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
				CHECK(n == 0);
			}
		}
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
	TestScratchChannelCreateRejections();
	TestScratchChannelAppendSubNorms();
	TestScratchChannelSnapshotBakedTwin();
	TestScratchChannelFeatureOracle();
	TestScratchChannelDegenerateAndRuntimeRejections();
	TestScratchChannelComposition();
	TestScratchChannelLifetimeShapeContracts();
	TestPinDrainLitmus();
	TestPerRowBias();
	TestWorkspaceReuseAcrossShapes();
	TestTrustBoundaryValidation();
	TestCrossDevice();
	TestScratchRetention();
	TestScratchRecall();
	TestScratchXdClosure();
	TestPoolCentroidXd();
	TestPoolDefinedBehavior();
	TestPoolXdSweep();
	TestPoolXdBatch();
	TestPoolRecallAndContracts();
	TestXdcBiasDense();
	TestXdcBiasSparse();
	TestXdcSegments();
	TestXdcIntersect();
	TestBankAnalytics();
	TestProjectionReport();
	TestAnalyticsScratchSnapshot();

	std::printf("superfaiss tests: %d checks, %d failures (simd path: %s)\n",
		GChecks, GFailures,
		ActiveSimdPath() == SimdPath::Scalar ? "scalar"
			: ActiveSimdPath() == SimdPath::SSE ? "sse"
			: ActiveSimdPath() == SimdPath::AVX2 ? "avx2" : "neon");
	return GFailures == 0 ? 0 : 1;
}

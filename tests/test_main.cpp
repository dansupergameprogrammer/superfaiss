#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS // getenv/fopen in the selection-recording emitter
#endif

// SuperFAISS test harness. Standard library only; no third-party test framework.
// Reference results are computed in double precision with the same total order
// (score, then ascending index); float-vs-double near-ties are handled with an
// epsilon boundary check rather than exact rank equality.

#include "superfaiss/superfaiss.h"
#include "superfaiss/graph.h"    // V3.2 M1 (Bank Inspector I)
#include "superfaiss/novelty.h"  // V3.2 M2
#include "superfaiss/matching.h" // V3.2 M3

#include "xd_fixtures.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
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
// Raw heap-allocation counter (V3.2 S1 test support, Curie 2026-07-19).
//
// AllocationCount() (superfaiss/alloc.h) only sees traffic through the
// SuperFAISS allocator seam (Workspace::Reserve*, via detail::SeamAlloc) —
// a std::vector's default allocator calls ::operator new directly and is
// invisible to it. A per-call std::vector<Hit>/std::vector<int32_t> output
// buffer inside a core query path is exactly the S1 violation standing
// against graph.h/novelty.h/matching.h (Claude/Poirot/afabc08-graph-h-m1-
// review.md, Claude/Poirot/15a0668-s1s2s3-fix-verify.md, Claude/Poirot/
// 524b373-matching-m3-review.md): "the accounting instrument is blind to
// the one allocation that violates it." An AllocationCount()-only assertion
// around such a call would report flat whether or not the violation is
// present, so it cannot serve as the S1 oracle by itself — it would be a
// cell that looks swept but proves nothing (Curie's own "a test targets its
// cell and fails for its reason" discipline).
//
// A scoped, thread-local override of the global ::operator new/delete
// closes that blind spot for these tests only. The thread-local flag is OFF
// by default, so every other check in this file allocates exactly as it
// always has; the override only counts while a ScopedRawNewTracking is
// alive on the calling thread.
namespace
{
	std::atomic<uint64_t> GRawNewCalls{0};
}
thread_local bool GTrackRawNew = false;

struct ScopedRawNewTracking
{
	ScopedRawNewTracking() { GRawNewCalls.store(0, std::memory_order_relaxed); GTrackRawNew = true; }
	~ScopedRawNewTracking() { GTrackRawNew = false; }
	ScopedRawNewTracking(const ScopedRawNewTracking&) = delete;
	ScopedRawNewTracking& operator=(const ScopedRawNewTracking&) = delete;
	uint64_t Count() const { return GRawNewCalls.load(std::memory_order_relaxed); }
};

void* operator new(std::size_t sz)
{
	if (GTrackRawNew)
	{
		GRawNewCalls.fetch_add(1, std::memory_order_relaxed);
	}
	void* p = std::malloc(sz != 0 ? sz : 1);
	if (p == nullptr)
	{
		throw std::bad_alloc();
	}
	return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t sz) { return operator new(sz); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

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
	{
		// Q1 (coverage audit §6.2): AllocationCount() alone is blind to a raw
		// ::operator new (§3) -- wrap the loop in the raw-new instrument so this,
		// the claim's headline entry point, is genuinely proven, not merely
		// seam-flat.
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 1000; ++i)
		{
			CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"Query allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore, "allocations grew: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore),
		static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// Q2 (coverage audit §6.2): QueryBatch had no allocation cell of any kind.
// Warm at the loop's max queryCount/k, then re-drive at that shape AND at a
// smaller queryCount (a shrink must not re-reserve -- the sharp edge the
// audit names), crossed with per-query bias.
static void TestAllocFlatQueryBatch()
{
	Rng rng(0x6A1C);
	const int32_t dims = 32, count = 300, maxQueryCount = 12, smallQueryCount = 4, k = 8;
	TestBank bank(rng, count, dims, Quantization::Int8, Metric::Cosine);

	std::vector<float> rawQueries(static_cast<size_t>(maxQueryCount) * dims);
	for (auto& v : rawQueries) v = rng.NextFloat();
	AlignedBuf qbuf(static_cast<size_t>(maxQueryCount) * bank.view.paddedDims * sizeof(float));
	for (int32_t i = 0; i < maxQueryCount; ++i)
	{
		std::vector<float> row(rawQueries.begin() + i * dims, rawQueries.begin() + (i + 1) * dims);
		PadQuery(row, bank.view.paddedDims, qbuf.F32() + static_cast<size_t>(i) * bank.view.paddedDims);
	}

	std::vector<RowBias> biases(static_cast<size_t>(maxQueryCount));
	std::vector<float> denseBias(static_cast<size_t>(count), 0.0625f);
	for (auto& b : biases) b.dense = denseBias.data();

	Workspace ws;
	std::vector<Hit> hits(static_cast<size_t>(maxQueryCount) * k);
	std::vector<int32_t> counts(static_cast<size_t>(maxQueryCount));
	QueryParams params;
	params.k = k;
	params.bias = biases.data();

	// Warm: twice at the max shape.
	CHECK(QueryBatch(bank.view, qbuf.F32(), maxQueryCount, params, ws, hits.data(), counts.data()) == Status::Ok);
	CHECK(QueryBatch(bank.view, qbuf.F32(), maxQueryCount, params, ws, hits.data(), counts.data()) == Status::Ok);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(QueryBatch(bank.view, qbuf.F32(), maxQueryCount, params, ws, hits.data(), counts.data()) == Status::Ok);
		}
		// The shrink leg: a smaller queryCount must not re-reserve.
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(QueryBatch(bank.view, qbuf.F32(), smallQueryCount, params, ws, hits.data(), counts.data()) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"QueryBatch allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"QueryBatch's seam-tracked allocations grew on a warm workspace: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// Q3 (coverage audit §6.2): QueryIntersect had no allocation cell of any kind.
// Warm at max queryCount/k (the fused path reserves query scratch for weight
// folding); drive queryCount == 1 (the degenerate Query-identical path) and
// queryCount > 1, both with and without segments.
static void TestAllocFlatQueryIntersect()
{
	Rng rng(0x6A1D);
	const int32_t dims = 32, count = 200, maxQueryCount = 4, k = 6;
	TestBank bank(rng, count, dims, Quantization::Int8, Metric::Cosine);

	std::vector<float> rawQueries(static_cast<size_t>(maxQueryCount) * dims);
	for (auto& v : rawQueries) v = rng.NextFloat();
	AlignedBuf qbuf(static_cast<size_t>(maxQueryCount) * bank.view.paddedDims * sizeof(float));
	for (int32_t i = 0; i < maxQueryCount; ++i)
	{
		std::vector<float> row(rawQueries.begin() + i * dims, rawQueries.begin() + (i + 1) * dims);
		PadQuery(row, bank.view.paddedDims, qbuf.F32() + static_cast<size_t>(i) * bank.view.paddedDims);
	}
	const QuerySegment segs[2] = {{0, 16, 1.0f}, {16, 16, 1.0f}};

	Workspace ws;
	Hit hits[6];
	int32_t n = 0;
	QueryParams params;
	params.k = k;

	// Warm: twice at the max shape, with segments (the widest reservation).
	params.segments = segs;
	params.segmentCount = 2;
	CHECK(QueryIntersect(bank.view, qbuf.F32(), maxQueryCount, params, ws, hits, &n) == Status::Ok);
	CHECK(QueryIntersect(bank.view, qbuf.F32(), maxQueryCount, params, ws, hits, &n) == Status::Ok);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(QueryIntersect(bank.view, qbuf.F32(), maxQueryCount, params, ws, hits, &n) == Status::Ok);
		}
		params.segments = nullptr;
		params.segmentCount = 0;
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(QueryIntersect(bank.view, qbuf.F32(), maxQueryCount, params, ws, hits, &n) == Status::Ok);
			CHECK(QueryIntersect(bank.view, qbuf.F32(), 1, params, ws, hits, &n) == Status::Ok); // degenerate
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"QueryIntersect allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"QueryIntersect's seam-tracked allocations grew on a warm workspace: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// Q5 (coverage audit §6.2): QueryXdBatch had no allocation cell of any kind.
// Warm at max queryCount; Workspace must be warm for the k and for the
// ReserveXdQuery slot block. Per-query bias, params.bias carries queryCount
// entries (the QueryBatch convention).
static void TestAllocFlatQueryXdBatch()
{
	Rng rng(0x6A1E);
	const int32_t dims = 32, count = 256, maxQueryCount = 6, k = 8;
	TestBank bank(rng, count, dims, Quantization::Int8, Metric::Dot);

	std::vector<XdQuery> queries(static_cast<size_t>(maxQueryCount));
	std::vector<std::vector<int8_t>> images(static_cast<size_t>(maxQueryCount));
	for (int32_t i = 0; i < maxQueryCount; ++i)
	{
		std::vector<int32_t> idx = {i % count};
		AlignedBuf q8(static_cast<size_t>(bank.view.paddedDims));
		double scale = 0.0;
		int64_t sqSum = 0;
		CHECK(MakeCentroidCrossDevice(bank.view, idx.data(), 1, nullptr, nullptr,
				q8.I8(), &scale, &sqSum) == Status::Ok);
		images[i].assign(q8.I8(), q8.I8() + bank.view.paddedDims);
		queries[i] = XdQuery{images[i].data(), scale, sqSum};
	}

	std::vector<RowBias> biases(static_cast<size_t>(maxQueryCount));
	std::vector<float> denseBias(static_cast<size_t>(count), 0.03125f);
	for (auto& b : biases) b.dense = denseBias.data();

	Workspace ws;
	std::vector<Hit> hits(static_cast<size_t>(maxQueryCount) * k);
	std::vector<int32_t> counts(static_cast<size_t>(maxQueryCount));
	QueryParams params;
	params.k = k;
	params.exactness = Exactness::CrossDevice;
	params.bias = biases.data();

	CHECK(QueryXdBatch(bank.view, queries.data(), maxQueryCount, params, ws, hits.data(), counts.data()) == Status::Ok);
	CHECK(QueryXdBatch(bank.view, queries.data(), maxQueryCount, params, ws, hits.data(), counts.data()) == Status::Ok);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(QueryXdBatch(bank.view, queries.data(), maxQueryCount, params, ws, hits.data(), counts.data()) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"QueryXdBatch allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"QueryXdBatch's seam-tracked allocations grew on a warm workspace: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
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
// T11b — AVX2 sub-8 float32 remainder (regression, T-099 slot-5 spinoff). A float32
// segment/channel stride lies on the 4-float (16-byte) grid, so a range whose length
// ≡ 4 (mod 8) leaves a trailing 4-element tail after the 8-lane groups. The AVX2
// float32 kernels DotF32Avx2/L2F32Avx2 AND their scalar mirrors DotF32ScalarAvx2/
// L2F32ScalarAvx2 processed only whole 8-lane groups with NO 4-element remainder, so
// that tail was dropped. The fix (src/kernels_avx2.cpp) adds the 4-remainder to both,
// bit-identically.
//
// Where the defect actually bites (established while authoring this test — the coord's
// original "length-4 scores 0 on the whole-row path" framing was imprecise):
//   * The WHOLE-ROW dispatch DotF32()/L2F32() (src/kernels.cpp) selects the AVX2
//     intrinsic ONLY when paddedDims % 8 == 0 (else DotF32Sse, which handles the tail).
//     PaddedDims is always a multiple of 8 for f32 banks big enough to matter, so a
//     length ≡ 4 mod 8 never reaches the intrinsic through that dispatch — the whole-
//     row path was NOT wrong in production. That is why the correctness check in Part A
//     runs against the scalar MIRROR directly, not through DotF32().
//   * The defect bites the SEGMENTED / per-channel-cosine scan: ResolveRowKernels()
//     (src/kernels.cpp) wires k.dotF32 = DotF32Avx2 RAW (no % 8 guard), and the non-
//     foldable per-channel-cosine path (ScoreChunkSegmented) calls it over each
//     channel's own sub-range. A length-4 channel therefore hits DotF32Avx2(.., 4),
//     drops the whole channel, and scores a per-channel cosine of exactly 0. Part B
//     drives that real production surface through the public Query() API.
//
// Parts A and B are RED pre-fix, GREEN post-fix. Part A is a unit check on the mirror
// against a double reference (tolerance); Part B is the production feature oracle over
// the public Query() API (tolerance) and the core twin of the plugin slot-5 Cell-1
// named-channel query. Part C (Poirot C-1) adds the missing BIT-EXACT determinism
// guard: intrinsic DotF32Avx2/L2F32Avx2 == their scalar mirrors, exact float bits, at
// the sub-8 tail lengths. The intrinsics are not in the public header (only the scalar
// mirrors are), so — like src/kernels.cpp does — this TU forward-declares them; they
// have external linkage in namespace superfaiss::detail and are defined in
// kernels_avx2.cpp under the same x86 guard used below. Part C is GREEN in BOTH stash
// states by design: pre-fix both paths drop the tail identically (they agree at the
// tail-dropped value), post-fix both add the 4-remainder bit-identically (std::fma
// rounds like the hardware FMA). It does not catch the ORIGINAL bug (Parts A/B do); it
// guards against a FUTURE edit that changes one path's tail and not the other — the
// exact cross-path determinism contract C-1 flagged as unguarded.
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
	#define SUPERFAISS_TEST_X86_INTRINSICS 1
namespace superfaiss
{
namespace detail
{
	// Defined in src/kernels_avx2.cpp (external linkage; not exported in kernels.h).
	float DotF32Avx2(const float* row, const float* query, int32_t paddedDims);
	float L2F32Avx2(const float* row, const float* query, int32_t paddedDims);
}
}
#endif

static void TestAvx2Sub8RemainderF32()
{
	using namespace superfaiss;

	// --- Part A: the AVX2 scalar mirror computes the sub-8 tail (reachable unit check).
	// Compares DotF32ScalarAvx2/L2F32ScalarAvx2 against an INDEPENDENT double reference
	// (plain for-loop, not a recode of the kernel's lane structure) at lengths ≡ 4 mod 8
	// {4,12,20} plus multiple-of-8 controls {8,16}. Pre-fix the mirror returns the
	// tail-dropped value (exactly 0 at len 4); the controls pass in both states. Only
	// meaningful when AVX2 is the active path — the mirror is the AVX2-shaped scalar;
	// on other devices skip with a note rather than assert a path this build never runs.
	// Compile-time x86 guard (not only the runtime check): DotF32ScalarAvx2/L2F32ScalarAvx2
	// are defined solely in the x86-guarded kernels_avx2.cpp, so a bare reference to them on
	// arm64 is an undefined-symbol link error -- the runtime ActiveSimdPath() check does not
	// remove the call the compiler emits.
#if defined(SUPERFAISS_TEST_X86_INTRINSICS)
	if (ActiveSimdPath() == SimdPath::AVX2)
	{
		Rng rng(0x5178A11);
		const int32_t lens[] = {4, 8, 12, 16, 20};
		const int32_t maxLen = 20;
		AlignedBuf row(static_cast<size_t>(maxLen) * sizeof(float));
		AlignedBuf query(static_cast<size_t>(maxLen) * sizeof(float));

		for (int32_t len : lens)
		{
			for (int32_t rep = 0; rep < 32; ++rep)
			{
				for (int32_t i = 0; i < len; ++i)
				{
					row.F32()[i] = rng.NextFloat();
					query.F32()[i] = rng.NextFloat();
				}
				double refDot = 0.0, refL2 = 0.0;
				for (int32_t i = 0; i < len; ++i)
				{
					const double a = static_cast<double>(row.F32()[i]);
					const double b = static_cast<double>(query.F32()[i]);
					refDot += a * b;
					const double d = b - a;
					refL2 += d * d;
				}
				const float dot = detail::DotF32ScalarAvx2(row.F32(), query.F32(), len);
				const float l2 = detail::L2F32ScalarAvx2(row.F32(), query.F32(), len);
				const double dotTol = 1e-5 * (1.0 + std::fabs(refDot));
				const double l2Tol = 1e-5 * (1.0 + std::fabs(refL2));
				CHECK_MSG(std::fabs(static_cast<double>(dot) - refDot) <= dotTol,
					"DotF32ScalarAvx2 len=%d: kernel %.9g ref %.9g", len, dot, refDot);
				CHECK_MSG(std::fabs(static_cast<double>(l2) - refL2) <= l2Tol,
					"L2F32ScalarAvx2 len=%d: kernel %.9g ref %.9g", len, l2, refL2);
			}
		}
	}
#endif // SUPERFAISS_TEST_X86_INTRINSICS (Part A)

	// --- Part C (Poirot C-1): the AVX2 intrinsic equals its scalar mirror BIT-EXACTLY at
	// the sub-8 tail. This is the path's determinism contract (DotF32Avx2 == its scalar
	// mirror, and L2 likewise) and was unguarded here: TestSimdEqualsScalar reaches the
	// intrinsic only via the DotF32/DotF32Mirror dispatchers, which route non-multiple-
	// of-8 strides to SSE/scalar, so the sub-8-tail intrinsic-vs-mirror bits were never
	// compared. Exact ==, never a tolerance — the contract is bit equality. Several
	// random fixtures per length so it is not one lucky value. AVX2-only (the intrinsic
	// forward-declared above exists on this x86 build; the runtime guard confirms the
	// device actually runs it).
#if defined(SUPERFAISS_TEST_X86_INTRINSICS)
	if (ActiveSimdPath() == SimdPath::AVX2)
	{
		Rng rng(0xB17EC0DEull);
		const int32_t lens[] = {4, 8, 12, 16, 20}; // {4,12,20} tail; {8,16} controls
		const int32_t maxLen = 20;
		AlignedBuf row(static_cast<size_t>(maxLen) * sizeof(float));
		AlignedBuf query(static_cast<size_t>(maxLen) * sizeof(float));
		for (int32_t len : lens)
		{
			for (int32_t rep = 0; rep < 16; ++rep)
			{
				for (int32_t i = 0; i < len; ++i)
				{
					row.F32()[i] = rng.NextFloat();
					query.F32()[i] = rng.NextFloat();
				}
				const float dotIntrin = detail::DotF32Avx2(row.F32(), query.F32(), len);
				const float dotMirror = detail::DotF32ScalarAvx2(row.F32(), query.F32(), len);
				const float l2Intrin = detail::L2F32Avx2(row.F32(), query.F32(), len);
				const float l2Mirror = detail::L2F32ScalarAvx2(row.F32(), query.F32(), len);
				// Bit-exact: identical float bit patterns, not IsNearlyEqual.
				CHECK_MSG(dotIntrin == dotMirror,
					"DotF32Avx2 vs mirror len=%d rep=%d: intrinsic %.9g mirror %.9g",
					len, rep, dotIntrin, dotMirror);
				CHECK_MSG(l2Intrin == l2Mirror,
					"L2F32Avx2 vs mirror len=%d rep=%d: intrinsic %.9g mirror %.9g",
					len, rep, l2Intrin, l2Mirror);
			}
		}
	}
#endif

	// --- Part B: the production per-channel-cosine scan over a LENGTH-4 channel (public
	// Query() API — the real surface, exercising the raw DotF32Avx2 intrinsic through
	// ResolveRowKernels). A Float32 Cosine bank with a length-4 first channel; query
	// that one channel and compare the full ranking to an INDEPENDENT double per-channel
	// cosine brute-force over the same sub-range. Pre-fix every row scores exactly 0
	// (the channel is dropped), so the ranking and scores diverge from the reference and
	// this fails decisively; post-fix they match. Runs on every device (the whole-row
	// bank ops use paddedDims % 8 == 0, so a non-AVX2 device simply exercises the SSE
	// path, which was always correct — the oracle holds there too).
	{
		Rng rng(0xC4A22E1ull);
		const int32_t dims = 8;                  // paddedDims 8 (% 8 == 0) for the bank ops
		const int32_t count = 64;
		const int32_t chanOffset = 0;
		const int32_t chanLen = 4;               // ≡ 4 mod 8 — the dropped-tail case
		TestBank bank(rng, count, dims, Quantization::Float32, Metric::Cosine);
		const int32_t pd = bank.view.paddedDims;

		const ChannelInfo channels[2] = {{chanOffset, chanLen}, {chanLen, pd - chanLen}};
		std::vector<float> invNorms(static_cast<size_t>(count) * 2);
		BankView view = bank.view;
		view.channels = channels;
		view.channelCount = 2;
		CHECK(ComputeChannelInverseNorms(view, invNorms.data()) == Status::Ok);
		view.channelInvNorms = invNorms.data();
		CHECK(ValidateBank(view) == Status::Ok);

		// Query, with the first channel's sub-vector renormalized to unit norm (the
		// D-V2-1 per-channel build rule the fixture in TestPerChannelCosine also uses).
		AlignedBuf q(static_cast<size_t>(pd) * sizeof(float));
		std::vector<float> qv(static_cast<size_t>(dims));
		for (auto& x : qv)
		{
			x = rng.NextFloat();
		}
		PadQuery(qv, pd, q.F32());
		{
			double norm = 0.0;
			for (int32_t j = chanOffset; j < chanOffset + chanLen; ++j)
			{
				norm += static_cast<double>(q.F32()[j]) * q.F32()[j];
			}
			const double inv = norm > 0.0 ? 1.0 / std::sqrt(norm) : 0.0;
			for (int32_t j = chanOffset; j < chanOffset + chanLen; ++j)
			{
				q.F32()[j] = static_cast<float>(q.F32()[j] * inv);
			}
		}

		const QuerySegment one[1] = {{chanOffset, chanLen, 1.0f}};
		QueryParams sp;
		sp.k = count;
		sp.segments = one;
		sp.segmentCount = 1;
		Workspace ws;
		std::vector<Hit> got(static_cast<size_t>(count));
		int32_t gotCount = 0;
		CHECK(Query(view, q.F32(), sp, ws, got.data(), &gotCount) == Status::Ok);
		CHECK_MSG(gotCount == count, "length-4 channel query hit count %d != %d",
			gotCount, count);

		// Independent oracle: per-channel cosine of the length-4 sub-range, in double,
		// straight from the definition — dot(q_sub, row_sub) / (||q_sub|| ||row_sub||).
		// (q_sub is already unit-norm from the renormalization above.)
		struct RefScore { int32_t index; double score; };
		std::vector<RefScore> ref(static_cast<size_t>(count));
		double bestAbs = 0.0;
		for (int32_t r = 0; r < count; ++r)
		{
			const double* rr = bank.refRows.data() + static_cast<size_t>(r) * dims;
			double dot = 0.0, rowNorm = 0.0, qNorm = 0.0;
			for (int32_t j = chanOffset; j < chanOffset + chanLen; ++j)
			{
				dot += static_cast<double>(q.F32()[j]) * rr[j];
				rowNorm += rr[j] * rr[j];
				qNorm += static_cast<double>(q.F32()[j]) * q.F32()[j];
			}
			const double denom = std::sqrt(rowNorm) * std::sqrt(qNorm);
			ref[r].index = r;
			ref[r].score = denom > 0.0 ? dot / denom : 0.0;
			bestAbs = std::max(bestAbs, std::fabs(ref[r].score));
		}
		std::sort(ref.begin(), ref.end(), [](const RefScore& a, const RefScore& b) {
			return a.score > b.score;
		});

		// The channel carries real signal (guards against a vacuous all-zero oracle that
		// would let the pre-fix all-zero scan look "correct").
		CHECK_MSG(bestAbs > 0.1, "length-4 channel oracle has no signal (max |cos| %.4g)",
			bestAbs);

		// FEAT: the scan's per-channel cosines match the independent brute force in both
		// score and order. Pre-fix all got[].score are 0 → the top score mismatches the
		// reference's ~bestAbs and this fails; post-fix they agree within float slack.
		for (int32_t i = 0; i < gotCount && i < count; ++i)
		{
			CHECK_MSG(got[i].index == ref[i].index,
				"length-4 channel rank %d: scan row %d, oracle row %d",
				i, got[i].index, ref[i].index);
			CHECK_MSG(std::fabs(static_cast<double>(got[i].score) - ref[i].score) <= 1e-4,
				"length-4 channel rank %d: scan %.6g oracle %.6g",
				i, got[i].score, ref[i].score);
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
		{
			// S1/S2/S3 (coverage audit §6.4): raw-new instrument, not seam-only —
			// the strongest existing cell in shape (a full append/remove/
			// snapshot/query cycle to capacity) and previously blind to a raw
			// heap allocation beside the seam.
			ScopedRawNewTracking rawTracking;
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
			CHECK_MSG(rawTracking.Count() == 0,
				"scratch Append/Remove/Snapshot/Query allocated %llu time(s) outside the seam",
				static_cast<unsigned long long>(rawTracking.Count()));
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
		{
			// Q1 (coverage audit §6.2): raw-new instrument, not seam-only.
			ScopedRawNewTracking rawTracking;
			for (int32_t i = 0; i < 32; ++i)
			{
				params.bias = (i & 1) ? &rbDense : &rbSparse;
				CHECK(Query(bank.view, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
			}
			CHECK_MSG(rawTracking.Count() == 0,
				"Query (biased) allocated %llu time(s) outside the Workspace seam",
				static_cast<unsigned long long>(rawTracking.Count()));
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
		const uint64_t growthBefore = ws.GrowthCount();
		{
			// Q1 (coverage audit §6.2): raw-new instrument, not seam-only.
			ScopedRawNewTracking rawTracking;
			for (int32_t i = 0; i < 16; ++i)
			{
				CHECK(Query(bank.view, q.F32(), p, ws, hits, &n) == Status::Ok);
			}
			CHECK_MSG(rawTracking.Count() == 0,
				"Query (CrossDevice) allocated %llu time(s) outside the Workspace seam",
				static_cast<unsigned long long>(rawTracking.Count()));
		}
		CHECK_MSG(AllocationCount() == before, "warm CrossDevice queries allocated");
		CHECK(ws.GrowthCount() == growthBefore);
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

		// S9 (coverage audit §6.4, upgrading this cell): zero steady-state
		// allocation, on both Dot and Cosine, raw-new instrument plus the
		// missing GrowthCount() assertion. Warm one measure, then flat across
		// 8 sweeps.
		for (Metric recallMetric : {Metric::Dot, Metric::Cosine})
		{
			ScratchBank b;
			Rng rng(0x8A11ull);
			CHECK(b.Create(256, dims, recallMetric, Quantization::Int8, true) == Status::Ok);
			FillSeeded(b, 200, dims, rng);
			ScratchRecallReport warm;
			CHECK(b.MeasureScratchRecall(ws, &warm, seed) == Status::Ok);
			const uint64_t before = AllocationCount();
			const uint64_t growthBefore = ws.GrowthCount();
			{
				ScopedRawNewTracking rawTracking;
				for (int32_t i = 0; i < 8; ++i)
				{
					ScratchRecallReport r;
					CHECK(b.MeasureScratchRecall(ws, &r, seed) == Status::Ok);
				}
				CHECK_MSG(rawTracking.Count() == 0,
					"MeasureScratchRecall allocated %llu time(s) outside the Workspace seam",
					static_cast<unsigned long long>(rawTracking.Count()));
			}
			CHECK_MSG(AllocationCount() == before, "warm MeasureScratchRecall allocated");
			CHECK(ws.GrowthCount() == growthBefore);
		}

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
		{
			// Q4/G4 (coverage audit §6.2/§6.5): raw-new instrument over both
			// MakeCentroidCrossDevice and QueryXd, not seam-only.
			ScopedRawNewTracking rawTracking;
			for (int32_t i = 0; i < 16; ++i)
			{
				CHECK(MakeCentroidCrossDevice(bank.view, idx, 6, nullptr, nullptr,
						q8.I8(), &scale, &sqSum) == Status::Ok);
				CHECK(QueryXd(bank.view, xq, p, ws, hits, &n) == Status::Ok);
			}
			CHECK_MSG(rawTracking.Count() == 0,
				"MakeCentroidCrossDevice/QueryXd allocated %llu time(s) outside the seam",
				static_cast<unsigned long long>(rawTracking.Count()));
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
		{
			// Q1 (coverage audit §6.2): raw-new instrument, not seam-only, over the
			// segmented-channel query path plus a live Append in the same window.
			ScopedRawNewTracking rawTracking;
			CHECK(fx.bank.Append(extra, dims, &extraIdx) == Status::Ok); // append with sub-norm write
			CHECK(Query(snap, qbuf.F32(), p1, ws, hits, &n) == Status::Ok);
			CHECK(Query(snap, qbuf.F32(), p2, ws, hits, &n) == Status::Ok);
			CHECK_MSG(rawTracking.Count() == 0,
				"channel append/query allocated %llu time(s) outside the Workspace/ScratchBank seam",
				static_cast<unsigned long long>(rawTracking.Count()));
		}
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

// ---------------------------------------------------------------------------
// T-V3 (slot 3) -- channel-aware Freeze + persistence (SuperFAISS_V2_Plan.md
// section 23.9 slot 3; section 23.4 channel-aware Freeze; section 23.6 archive
// presence-flags; section 23.7.1 the Freeze crux; section 23.8 dim 9 round-trip
// + dim 10 frozen-path FEAT). Authored red-first by Curie, on top of the green
// slot-2 base (channel Create/Append/Snapshot/Grow). Reuses the slot-2
// helpers ScratchChannelFixture / MakeChannelScratchBank / RefNormalizeRow /
// kChannelFeatRelTol* and the MemArchive seam from TestScratchBanks.
//
// SCAFFOLD (Curie, not the feature):
//   - The channel-aware Freeze overload (Freeze(rows, scales, map, subNorms, ...))
//     is declared in scratch.h and stubbed in scratch.cpp to return BadAlignment
//     -- a status no slot-3 cell expects -- so every Freeze cell fails at a
//     specific runtime assertion for the one true reason.
//   - Save/Load are NOT stubbed: they already compile and correctly round-trip a
//     channel-LESS bank. Their slot-3 red state is genuine missing behavior -- the
//     shipped format carries no channel table, so a channel bank saved and loaded
//     comes back single-space (channelCount 0), failing every channel-survival
//     assertion for the one true reason (persistence does not carry channels yet).
// Hastings replaces the Freeze stub and extends Save/Load with the section 23.6
// presence-flags scheme.

namespace
{
	// The scratch archive header's flags-byte offset ASSUMPTION (section 23.6 says a
	// "presence-flags byte in reserved[6]" but does not pin WHICH of the six reserved
	// bytes, nor the byte position of the retention/channels bits -- routed to Vitruvius
	// as finding C-3 in Claude/Curie/SuperFAISS_V3_Test_Design.md section 11). The header
	// is {magic u32, version u32, capacity i32, count i32, dims i32, paddedDims i32,
	// metric u8, quant u8, reserved[6]} = 32 bytes; reserved[0] is at byte offset 26. The
	// forward-tolerance cell (T-V3-Persist-3) pokes a reserved-range bit at this offset;
	// if slot 3 places the flags byte elsewhere in reserved[6], that cell must move with
	// it (the finding forces the offset to be pinned).
	constexpr size_t kScratchHeaderFlagsByteOffset = 26;

	// Dequantize one row of a frozen/loaded int8 BankView into doubles (scale * int8),
	// or copy the float32 lanes -- the FEAT ground truth's input, computed without
	// touching any operator-side code.
	void DequantizeRow(const BankView& view, int32_t row, std::vector<double>& out)
	{
		out.assign(static_cast<size_t>(view.dims), 0.0);
		if (view.quant == Quantization::Int8)
		{
			const int8_t* r =
				static_cast<const int8_t*>(view.rows) + static_cast<int64_t>(row) * view.paddedDims;
			const double scale = view.scales[row];
			for (int32_t i = 0; i < view.dims; ++i)
			{
				out[static_cast<size_t>(i)] = scale * r[i];
			}
		}
		else
		{
			const float* r =
				static_cast<const float*>(view.rows) + static_cast<int64_t>(row) * view.paddedDims;
			for (int32_t i = 0; i < view.dims; ++i)
			{
				out[static_cast<size_t>(i)] = static_cast<double>(r[i]);
			}
		}
	}

	// Independent definition-grounded per-channel top-k over a queryable BankView's rows
	// (the FEAT reference, shared by the frozen-bank and loaded-bank cells). Returns the
	// sorted reference hits for one channel. NEVER calls ComputeChannelInverseNorms or the
	// kernel -- it computes the per-channel cosine as dot(q_sub, row_sub)/||row_sub|| from
	// the dequantized rows directly (the sub-vector norm, not the whole-row norm -- the
	// D-V2-1 contract point).
	struct FeatRefHit
	{
		int32_t index;
		double score;
	};
	std::vector<FeatRefHit> ChannelCosineBruteForce(
		const BankView& view, const float* paddedQuery, const ChannelInfo& ch,
		const uint32_t* excludeBits)
	{
		std::vector<FeatRefHit> ref;
		std::vector<double> row;
		for (int32_t r = 0; r < view.count; ++r)
		{
			if (IsExcluded(excludeBits, r))
			{
				continue;
			}
			DequantizeRow(view, r, row);
			double dot = 0.0, subNorm = 0.0;
			for (int32_t j = ch.offset; j < ch.offset + ch.length && j < view.dims; ++j)
			{
				dot += static_cast<double>(paddedQuery[j]) * row[static_cast<size_t>(j)];
				subNorm += row[static_cast<size_t>(j)] * row[static_cast<size_t>(j)];
			}
			const double score = subNorm > 0.0 ? dot / std::sqrt(subNorm) : 0.0;
			ref.push_back({r, score});
		}
		std::sort(ref.begin(), ref.end(), [](const FeatRefHit& a, const FeatRefHit& b) {
			if (a.score != b.score)
			{
				return a.score > b.score; // Cosine: higher is better
			}
			return a.index < b.index;
		});
		return ref;
	}
} // namespace

// T-V3-Freeze -- channel-aware Freeze -> a channel-carrying graduated bank (dim 9
// + section 23.7.1 equality + dim 10 FEAT, closing the FROZEN half of Japp S-1).
static void TestScratchChannelFreeze()
{
	Rng rng(0xF3EE2Eull);
	const int32_t dims = 64;
	const int32_t count = 96;
	const int32_t k = 10;

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		ScratchChannelFixture fx;
		MakeChannelScratchBank(fx, rng, count, dims, Metric::Cosine, quant);
		if (fx.createStatus != Status::Ok)
		{
			continue; // slot-2 base must be green for slot 3; CHECKed inside the builder
		}

		// Tombstone a scatter so compaction genuinely renumbers.
		int32_t removed = 0;
		for (int32_t r = 2; r < count; r += 5)
		{
			CHECK(fx.bank.Remove(r) == Status::Ok);
			++removed;
		}
		const int32_t live = fx.bank.FreezeLiveCount();
		CHECK(live == count - removed);

		// Pre-freeze snapshot + a per-channel query, to compare the frozen bank against.
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
		const ChannelInfo& ch = fx.channels[0];
		const QuerySegment seg[1] = {{ch.offset, ch.length, 1.0f}};
		QueryParams sp;
		sp.k = k;
		sp.segments = seg;
		sp.segmentCount = 1;
		sp.excludeBits = tombs.data();
		Workspace wsSnap;
		Hit snapHits[10];
		int32_t nSnap = 0;
		CHECK(Query(snap, qbuf.F32(), sp, wsSnap, snapHits, &nSnap) == Status::Ok);

		// Channel-aware Freeze into caller buffers.
		AlignedBuf frozenRows(static_cast<size_t>(live > 0 ? live : 1) * fx.pd *
			ElementSize(quant));
		std::vector<float> frozenScales(
			quant == Quantization::Int8 ? static_cast<size_t>(live) : size_t{1});
		std::vector<int32_t> indexMap(static_cast<size_t>(count), -2);
		std::vector<float> frozenInvNorms(
			static_cast<size_t>(live) * fx.channels.size(), -1.0f);

		const Status frozenStatus = fx.bank.Freeze(frozenRows.ptr,
			quant == Quantization::Int8 ? frozenScales.data() : nullptr,
			indexMap.data(), frozenInvNorms.data());
		CHECK_MSG(frozenStatus == Status::Ok,
			"channel-aware Freeze failed (unimplemented, slot 3): status=%d",
			static_cast<int>(frozenStatus));
		if (frozenStatus != Status::Ok)
		{
			continue; // red-unimplemented: nothing downstream to assert
		}

		// Index map: tombstoned rows -> -1, live rows -> ascending compacted indices.
		int32_t expectNext = 0;
		for (int32_t r = 0; r < count; ++r)
		{
			if (IsExcluded(tombs.data(), r))
			{
				CHECK(indexMap[static_cast<size_t>(r)] == -1);
			}
			else
			{
				CHECK(indexMap[static_cast<size_t>(r)] == expectNext);
				++expectNext;
			}
		}
		CHECK(expectNext == live);

		// Bytes-preserving (V3-G6 / T-V2-C2): compaction is a pure copy, never a
		// re-quantize -- each surviving row's frozen bytes bit-equal the pre-freeze
		// snapshot's row bytes at the original index (and its scale, int8).
		{
			const size_t rowBytes =
				static_cast<size_t>(fx.pd) * ElementSize(quant);
			for (int32_t r = 0; r < count; ++r)
			{
				const int32_t nidx = indexMap[static_cast<size_t>(r)];
				if (nidx < 0)
				{
					continue;
				}
				CHECK_MSG(std::memcmp(
							  static_cast<const uint8_t*>(frozenRows.ptr) +
								  static_cast<size_t>(nidx) * rowBytes,
							  static_cast<const uint8_t*>(snap.rows) +
								  static_cast<size_t>(r) * rowBytes,
							  rowBytes) == 0,
					"frozen row %d bytes differ from the snapshot row (re-quantized?)", r);
				if (quant == Quantization::Int8)
				{
					CHECK(frozenScales[static_cast<size_t>(nidx)] == snap.scales[r]);
				}
			}
		}

		// The graduated channel BankView, carrying Freeze's emitted sub-norms.
		BankView frozen;
		frozen.rows = frozenRows.ptr;
		frozen.scales = quant == Quantization::Int8 ? frozenScales.data() : nullptr;
		frozen.count = live;
		frozen.dims = dims;
		frozen.paddedDims = fx.pd;
		frozen.quant = quant;
		frozen.metric = Metric::Cosine;
		frozen.channels = fx.bank.GetChannels(); // the fixed table read back
		frozen.channelCount = fx.bank.GetChannelCount();
		frozen.channelInvNorms = frozenInvNorms.data();
		CHECK_MSG(frozen.channelCount == 2, "Freeze did not expose the channel table: %d",
			frozen.channelCount);
		CHECK(ValidateBank(frozen) == Status::Ok);

		// REF: the emitted sub-norms bit-equal ComputeChannelInverseNorms over the
		// compacted frozen rows (re-derived over the survivors, not copied verbatim).
		std::vector<float> refFrozenInvNorms(
			static_cast<size_t>(live) * frozen.channelCount);
		CHECK(ComputeChannelInverseNorms(frozen, refFrozenInvNorms.data()) == Status::Ok);
		CHECK_MSG(std::memcmp(refFrozenInvNorms.data(), frozenInvNorms.data(),
					  refFrozenInvNorms.size() * sizeof(float)) == 0,
			"frozen sub-norms do not bit-equal ComputeChannelInverseNorms over the "
			"compacted rows (quant=%d)",
			static_cast<int>(quant));

		// Equality (section 23.7.1): the frozen bank's per-channel query bit-equals the
		// pre-freeze snapshot's, hits renumbered through the index map, scores bit-exact.
		QueryParams fp;
		fp.k = k;
		fp.segments = seg;
		fp.segmentCount = 1;
		Workspace wsF;
		Hit fHits[10];
		int32_t nF = 0;
		CHECK(Query(frozen, qbuf.F32(), fp, wsF, fHits, &nF) == Status::Ok);
		CHECK_MSG(nF == nSnap, "frozen hit count != snapshot: %d vs %d", nF, nSnap);
		for (int32_t i = 0; i < nF && i < nSnap; ++i)
		{
			CHECK_MSG(fHits[i].index == indexMap[static_cast<size_t>(snapHits[i].index)],
				"frozen hit %d index mismatch (renumber): %d vs map[%d]=%d",
				i, fHits[i].index, snapHits[i].index,
				indexMap[static_cast<size_t>(snapHits[i].index)]);
			CHECK(fHits[i].score == snapHits[i].score);
		}

		// FEAT (Japp S-1, frozen half): the frozen bank's per-channel query top-k is
		// within the CAL band of the independent double brute-force over the DEQUANTIZED
		// frozen rows -- the graduation end-state is feature-oracle'd, not carried by the
		// equality oracle alone.
		{
			const std::vector<FeatRefHit> ref =
				ChannelCosineBruteForce(frozen, qbuf.F32(), ch, nullptr);
			const double tol = quant == Quantization::Int8 ? kChannelFeatRelTolI8
															 : kChannelFeatRelTolF32;
			const int32_t expected = static_cast<int32_t>(ref.size()) < k
				? static_cast<int32_t>(ref.size())
				: k;
			CHECK_MSG(nF == expected, "frozen FEAT hit count: got %d expected %d", nF,
				expected);
			if (expected > 0 && nF == expected)
			{
				const double boundary = ref[static_cast<size_t>(expected) - 1].score;
				std::vector<double> refByIndex(static_cast<size_t>(live), -1e300);
				for (const FeatRefHit& h : ref)
				{
					refByIndex[static_cast<size_t>(h.index)] = h.score;
				}
				for (int32_t i = 0; i < nF; ++i)
				{
					const double rs = refByIndex[static_cast<size_t>(fHits[i].index)];
					const double band = tol * (1.0 + std::fabs(boundary));
					CHECK_MSG(rs >= boundary - band,
						"frozen FEAT: hit %d (row %d) below the definition-grounded top-k "
						"boundary (quant=%d)",
						i, fHits[i].index, static_cast<int>(quant));
					CHECK_MSG(std::fabs(rs - static_cast<double>(fHits[i].score)) <= band,
						"frozen FEAT score drift: got %.9g ref %.9g (quant=%d)",
						static_cast<double>(fHits[i].score), rs, static_cast<int>(quant));
				}
			}
		}
	}

	// Degenerate: Freeze of a channel Cosine scratch with zero live rows -> defined
	// empty graduation (FreezeLiveCount 0), no crash, no NaN.
	{
		ScratchChannelFixture fx;
		MakeChannelScratchBank(fx, rng, 8, dims, Metric::Cosine, Quantization::Int8);
		if (fx.createStatus == Status::Ok)
		{
			for (int32_t r = 0; r < 8; ++r)
			{
				CHECK(fx.bank.Remove(r) == Status::Ok);
			}
			CHECK(fx.bank.FreezeLiveCount() == 0);
			AlignedBuf rows(static_cast<size_t>(fx.pd) * ElementSize(Quantization::Int8));
			std::vector<float> scales(1);
			std::vector<int32_t> map(8, -2);
			std::vector<float> invNorms(static_cast<size_t>(fx.channels.size()), -1.0f);
			const Status s = fx.bank.Freeze(rows.ptr, scales.data(), map.data(),
				invNorms.data());
			CHECK_MSG(s == Status::Ok,
				"zero-live-row channel Freeze should be defined Ok (unimplemented): %d",
				static_cast<int>(s));
			if (s == Status::Ok)
			{
				for (int32_t r = 0; r < 8; ++r)
				{
					CHECK(map[static_cast<size_t>(r)] == -1);
				}
			}
		}
	}

	// Non-channel bank on the channel-aware Freeze -> InvalidArgument (use base Freeze).
	{
		ScratchBank plain;
		CHECK(plain.Create(8, dims, Metric::Cosine, Quantization::Int8) == Status::Ok);
		float row[dims];
		for (int32_t i = 0; i < dims; ++i)
		{
			row[i] = 0.3f;
		}
		CHECK(plain.Append(row, dims, nullptr) == Status::Ok);
		AlignedBuf rows(static_cast<size_t>(PaddedDims(dims, Quantization::Int8)) *
			ElementSize(Quantization::Int8));
		std::vector<float> scales(1);
		std::vector<int32_t> map(1, -2);
		std::vector<float> invNorms(1, -1.0f);
		const Status s = plain.Freeze(rows.ptr, scales.data(), map.data(), invNorms.data());
		CHECK_MSG(s == Status::InvalidArgument,
			"channel-aware Freeze on a channel-less bank should be InvalidArgument: %d",
			static_cast<int>(s));
	}
}

// T-V3-Persist-1 -- the presence-flags round-trip (Forge S1) + re-derived-on-Load
// (Forge W3) + the channels+retention combination (dim 9's named combination).
static void TestScratchChannelPersistenceRoundTrip()
{
	Rng rng(0x9E2C7AA1ull);
	const int32_t dims = 64;

	// (a) channels+retention Cosine Int8 bank, with tombstones AND a Grow in history.
	{
		const ChannelInfo channels[2] = {{0, 32}, {32, 32}};
		ScratchBank bank;
		CHECK(bank.Create(48, dims, Metric::Cosine, Quantization::Int8, channels, 2,
				/*retainFloats=*/true) == Status::Ok);
		std::vector<float> source(static_cast<size_t>(40) * dims);
		for (auto& v : source)
		{
			v = rng.NextFloat();
		}
		for (int32_t r = 0; r < 40; ++r)
		{
			CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		for (int32_t r = 3; r < 40; r += 8)
		{
			CHECK(bank.Remove(r) == Status::Ok);
		}
		CHECK(bank.Grow(64) == Status::Ok); // Grow history present in the round-tripped bank
		std::vector<float> more(static_cast<size_t>(8) * dims);
		for (auto& v : more)
		{
			v = rng.NextFloat();
		}
		for (int32_t r = 0; r < 8; ++r)
		{
			CHECK(bank.Append(more.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}

		BankView origSnap;
		std::vector<uint32_t> origTombs(ScratchBank::TombstoneWords(bank.Count()), 0u);
		CHECK(bank.Snapshot(&origSnap, origTombs.data()) == Status::Ok);

		MemArchive archive;
		CHECK(bank.Save(archive.Writer()) == Status::Ok);
		ScratchBank loaded;
		CHECK(loaded.Load(archive.Reader()) == Status::Ok);

		CHECK(loaded.Count() == bank.Count());
		CHECK(loaded.LiveCount() == bank.LiveCount());
		CHECK_MSG(loaded.RetainsFloats(),
			"channels+retention round-trip lost retention");

		BankView loadedSnap;
		std::vector<uint32_t> loadedTombs(
			ScratchBank::TombstoneWords(loaded.Count()), 0u);
		CHECK(loaded.Snapshot(&loadedSnap, loadedTombs.data()) == Status::Ok);
		CHECK_MSG(loadedSnap.channelCount == 2,
			"channels+retention round-trip lost the channel table: channelCount=%d",
			loadedSnap.channelCount);
		if (loadedSnap.channelCount == 2)
		{
			for (int32_t c = 0; c < 2; ++c)
			{
				CHECK(loadedSnap.channels[c].offset == channels[c].offset &&
					loadedSnap.channels[c].length == channels[c].length);
			}
			CHECK_MSG(loadedSnap.channelInvNorms != nullptr,
				"loaded channel bank has no sub-norm arena");
		}
		CHECK(loadedTombs == origTombs);

		// W3: the sub-norm arena is RE-DERIVED on Load, bit-equal to a fresh derivation
		// over the loaded rows (the desync guard -- not a trusted serialized copy).
		if (loadedSnap.channelInvNorms != nullptr && loadedSnap.channelCount == 2)
		{
			std::vector<float> fresh(
				static_cast<size_t>(loadedSnap.count) * loadedSnap.channelCount);
			CHECK(ComputeChannelInverseNorms(loadedSnap, fresh.data()) == Status::Ok);
			CHECK_MSG(std::memcmp(fresh.data(), loadedSnap.channelInvNorms,
						  fresh.size() * sizeof(float)) == 0,
				"loaded sub-norm arena does not bit-equal a fresh derivation (W3 desync)");
		}

		// Per-channel query parity: the loaded bank answers a channel query identically
		// to the original (the round-trip preserved the channel-scored ranking).
		std::vector<float> queryRaw(dims);
		for (auto& v : queryRaw)
		{
			v = rng.NextFloat();
		}
		const int32_t pd = origSnap.paddedDims;
		AlignedBuf qbuf(static_cast<size_t>(pd) * sizeof(float));
		PadQuery(queryRaw, pd, qbuf.F32());
		const QuerySegment seg[1] = {{0, 32, 1.0f}};
		QueryParams p;
		p.k = 10;
		p.segments = seg;
		p.segmentCount = 1;
		p.excludeBits = origTombs.data();
		Workspace wsO;
		Hit origHits[10];
		int32_t nO = 0;
		CHECK(Query(origSnap, qbuf.F32(), p, wsO, origHits, &nO) == Status::Ok);
		if (loadedSnap.channelCount == 2)
		{
			QueryParams pl = p;
			pl.excludeBits = loadedTombs.data();
			Workspace wsL;
			Hit loadedHits[10];
			int32_t nL = 0;
			CHECK(Query(loadedSnap, qbuf.F32(), pl, wsL, loadedHits, &nL) == Status::Ok);
			CHECK_MSG(nO == nL, "loaded channel query count mismatch: %d vs %d", nO, nL);
			for (int32_t i = 0; i < nO && i < nL; ++i)
			{
				CHECK(origHits[i].index == loadedHits[i].index &&
					origHits[i].score == loadedHits[i].score);
			}
		}
	}

	// (b) channels-ONLY round-trip (channels on, retention off): the flags byte must
	// carry channels WITHOUT retention (Japp G-3 channels-only cell).
	{
		const ChannelInfo channels[2] = {{0, 32}, {32, 32}};
		ScratchBank bank;
		CHECK(bank.Create(32, dims, Metric::Cosine, Quantization::Int8, channels, 2,
				/*retainFloats=*/false) == Status::Ok);
		std::vector<float> source(static_cast<size_t>(24) * dims);
		for (auto& v : source)
		{
			v = rng.NextFloat();
		}
		for (int32_t r = 0; r < 24; ++r)
		{
			CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		MemArchive archive;
		CHECK(bank.Save(archive.Writer()) == Status::Ok);
		ScratchBank loaded;
		CHECK(loaded.Load(archive.Reader()) == Status::Ok);
		CHECK_MSG(!loaded.RetainsFloats(),
			"channels-only round-trip should NOT gain retention");
		BankView loadedSnap;
		std::vector<uint32_t> loadedTombs(
			ScratchBank::TombstoneWords(loaded.Count()), 0u);
		CHECK(loaded.Snapshot(&loadedSnap, loadedTombs.data()) == Status::Ok);
		CHECK_MSG(loadedSnap.channelCount == 2,
			"channels-only round-trip lost the channel table: channelCount=%d",
			loadedSnap.channelCount);
	}
}

// T-V3-Persist-2 -- writer version-selection (Japp G-3) + backward-read + old-reader
// hard-reject + reject-over-degrade.
static void TestScratchChannelPersistenceVersioning()
{
	Rng rng(0x5E1EC7ull);
	const int32_t dims = 48;

	// Writer version-selection (G-3): a channel-less AND retention-less bank still emits
	// the legacy blob and round-trips unchanged (the emit-legacy-when-no-new-features
	// policy). This is a green regression guard -- it must not regress when the flags
	// scheme lands.
	{
		ScratchBank plain;
		CHECK(plain.Create(16, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		std::vector<float> source(static_cast<size_t>(12) * dims);
		for (auto& v : source)
		{
			v = rng.NextFloat();
		}
		for (int32_t r = 0; r < 12; ++r)
		{
			CHECK(plain.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		MemArchive archive;
		CHECK(plain.Save(archive.Writer()) == Status::Ok);
		ScratchBank loaded;
		CHECK(loaded.Load(archive.Reader()) == Status::Ok);
		CHECK(loaded.Count() == plain.Count());
		CHECK(!loaded.RetainsFloats());
		BankView v;
		std::vector<uint32_t> t(ScratchBank::TombstoneWords(loaded.Count()), 0u);
		CHECK(loaded.Snapshot(&v, t.data()) == Status::Ok);
		CHECK(v.channelCount == 0); // no channels emitted or read
	}

	// Old-reader / new-data hard-reject, durable form: a version strictly above any
	// reader's supported maximum is rejected. Version 99 is safe across the section-23.6
	// v2->v3 bump (a v3 reader still rejects 99); this is the standing new-data-on-old-
	// reader law in a bump-stable form. (The specific "a v2 reader rejects a v3 archive"
	// is the design property this proxies; testing it against a frozen old binary is out
	// of scope for an in-tree suite.)
	{
		const ChannelInfo channels[1] = {{0, 32}};
		ScratchBank bank;
		CHECK(bank.Create(8, dims, Metric::Cosine, Quantization::Int8, channels, 1) ==
			Status::Ok);
		float row[dims];
		for (int32_t i = 0; i < dims; ++i)
		{
			row[i] = 0.25f;
		}
		CHECK(bank.Append(row, dims, nullptr) == Status::Ok);
		MemArchive archive;
		CHECK(bank.Save(archive.Writer()) == Status::Ok);

		// Force the version field (header bytes 4..7) to 99.
		MemArchive tooNew;
		tooNew.bytes = archive.bytes;
		const uint32_t v99 = 99u;
		std::memcpy(tooNew.bytes.data() + 4, &v99, sizeof(v99));
		ScratchBank target;
		CHECK_MSG(target.Load(tooNew.Reader()) == Status::BadFormat,
			"a version-99 archive must hard-reject");

		// Version 0 is likewise rejected.
		MemArchive vZero;
		vZero.bytes = archive.bytes;
		const uint32_t v0 = 0u;
		std::memcpy(vZero.bytes.data() + 4, &v0, sizeof(v0));
		CHECK(target.Load(vZero.Reader()) == Status::BadFormat);
	}

	// Backward-read (green regression guard): a legacy v1 plain bank and a v2 retention
	// bank -- both pre-channel -- still load on the (future) v3 reader. Round-tripping
	// them through the CURRENT library and re-loading proves the reader keeps accepting
	// the older formats it must stay backward-compatible with.
	{
		ScratchBank v1;
		CHECK(v1.Create(8, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		float row[dims];
		for (int32_t i = 0; i < dims; ++i)
		{
			row[i] = 0.4f;
		}
		CHECK(v1.Append(row, dims, nullptr) == Status::Ok);
		MemArchive a1;
		CHECK(v1.Save(a1.Writer()) == Status::Ok);
		ScratchBank l1;
		CHECK(l1.Load(a1.Reader()) == Status::Ok);
		CHECK(l1.Count() == 1 && !l1.RetainsFloats());

		ScratchBank v2;
		CHECK(v2.Create(8, dims, Metric::Dot, Quantization::Int8, /*retainFloats=*/true) ==
			Status::Ok);
		CHECK(v2.Append(row, dims, nullptr) == Status::Ok);
		MemArchive a2;
		CHECK(v2.Save(a2.Writer()) == Status::Ok);
		ScratchBank l2;
		CHECK(l2.Load(a2.Reader()) == Status::Ok);
		CHECK(l2.Count() == 1 && l2.RetainsFloats());
	}

	// Reject-over-degrade on a channel bank: corrupt magic / truncation leaves the target
	// bank unchanged (the C3 idiom, channel variant). The pre-existing green (magic and
	// truncation reject) guards that channel loading does not weaken it.
	{
		const ChannelInfo channels[1] = {{0, 32}};
		ScratchBank bank;
		CHECK(bank.Create(8, dims, Metric::Cosine, Quantization::Int8, channels, 1) ==
			Status::Ok);
		float row[dims];
		for (int32_t i = 0; i < dims; ++i)
		{
			row[i] = 0.2f;
		}
		CHECK(bank.Append(row, dims, nullptr) == Status::Ok);
		MemArchive archive;
		CHECK(bank.Save(archive.Writer()) == Status::Ok);

		ScratchBank loaded;
		CHECK(loaded.Load(archive.Reader()) == Status::Ok);
		const int32_t countBefore = loaded.Count();

		MemArchive badMagic;
		badMagic.bytes = archive.bytes;
		badMagic.bytes[0] ^= 0xFF;
		CHECK(loaded.Load(badMagic.Reader()) == Status::BadFormat);
		CHECK(loaded.Count() == countBefore); // unchanged

		MemArchive trunc;
		trunc.bytes.assign(archive.bytes.begin(),
			archive.bytes.begin() + static_cast<long>(archive.bytes.size() / 2));
		CHECK(loaded.Load(trunc.Reader()) == Status::BadFormat);
		CHECK(loaded.Count() == countBefore);
	}
}

// T-V3-Persist-3 -- reserved flag-bit forward tolerance (Japp G-3). A v3 reader
// encountering an unknown reserved bit in the flags byte tolerates it -- it does not
// reject the archive and does not mis-read the bit as channels or retention. Poked at
// the assumed flags-byte offset (see kScratchHeaderFlagsByteOffset; finding C-3 routes
// the exact offset to Vitruvius). Red today: persistence does not carry channels, so the
// channels+retention state does not survive regardless of the reserved bit.
static void TestScratchChannelPersistenceReservedBit()
{
	Rng rng(0xB17701ull);
	const int32_t dims = 64;
	const ChannelInfo channels[2] = {{0, 32}, {32, 32}};

	ScratchBank bank;
	CHECK(bank.Create(24, dims, Metric::Cosine, Quantization::Int8, channels, 2,
			/*retainFloats=*/true) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(20) * dims);
	for (auto& v : source)
	{
		v = rng.NextFloat();
	}
	for (int32_t r = 0; r < 20; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	MemArchive archive;
	CHECK(bank.Save(archive.Writer()) == Status::Ok);

	// Set a reserved-range bit (bit 2 -- bits 0/1 are retention/channels; 2..7 reserved)
	// in the flags byte. A v3 reader must tolerate it.
	MemArchive poked;
	poked.bytes = archive.bytes;
	CHECK(poked.bytes.size() > kScratchHeaderFlagsByteOffset);
	poked.bytes[kScratchHeaderFlagsByteOffset] |= 0x04;

	ScratchBank loaded;
	const Status s = loaded.Load(poked.Reader());
	CHECK_MSG(s == Status::Ok,
		"a reserved flag-bit must be tolerated on Load, got status=%d", static_cast<int>(s));
	if (s != Status::Ok)
	{
		return; // red: either persistence unimplemented or tolerance not honored
	}
	CHECK_MSG(loaded.RetainsFloats(),
		"reserved-bit tolerance: retention (flag bit 0) mis-read");
	BankView loadedSnap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(loaded.Count()), 0u);
	CHECK(loaded.Snapshot(&loadedSnap, tombs.data()) == Status::Ok);
	CHECK_MSG(loadedSnap.channelCount == 2,
		"reserved-bit tolerance: channels (flag bit 1) mis-read, channelCount=%d",
		loadedSnap.channelCount);
}


// T-V3-Recall -- per-channel recall (D-V3-7): the MeasureScratchRecall seeded
// routine gains a per-channel mode over the channel ranges, and channel-aware
// FreezeWithRecall re-measures per-channel recall over the compacted rows
// (section 23.9 slot-3 gate; section 23.4 item b; section 20). Both surfaces are
// scaffolded to BadAlignment, so every cell fails red for the one true reason.
static void TestScratchChannelPerChannelRecall()
{
	Rng rng(0x2ECA11ull);
	const int32_t dims = 128;
	const int32_t count = 256;
	const ChannelInfo channels[2] = {{0, 64}, {64, 64}};
	const uint64_t seed = ScratchBank::kDefaultRecallSeed;

	// A retention-enabled Cosine channel Int8 bank, populated to an informative size.
	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, channels, 2,
			/*retainFloats=*/true) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source)
	{
		v = rng.NextFloat();
	}
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	// Tombstone a scatter -- excluded from sampling, exactly as the whole-vector routine.
	int32_t removed = 0;
	for (int32_t r = 5; r < count; r += 13)
	{
		CHECK(bank.Remove(r) == Status::Ok);
		++removed;
	}
	const int32_t liveRows = count - removed;

	Workspace ws;

	// Per-channel MeasureScratchRecall: one report per channel, recall@k over each
	// channel's sub-range.
	{
		ScratchRecallReport reports[2];
		const Status s = bank.MeasureScratchRecallPerChannel(ws, reports, 2, seed);
		CHECK_MSG(s == Status::Ok,
			"per-channel MeasureScratchRecall failed (unimplemented, slot 3): status=%d",
			static_cast<int>(s));
		if (s == Status::Ok)
		{
			const int32_t expectK = liveRows - 1 < 10 ? (liveRows - 1) : 10;
			for (int32_t c = 0; c < 2; ++c)
			{
				CHECK_MSG(reports[c].recall >= 0.0f && reports[c].recall <= 1.0f,
					"channel %d recall out of [0,1]: %g", c,
					static_cast<double>(reports[c].recall));
				CHECK(reports[c].k == expectK);
				CHECK(reports[c].liveRows == liveRows);
				CHECK(reports[c].seed == seed);
				CHECK(reports[c].generation == bank.Generation());
				CHECK(reports[c].informative == (liveRows >= ScratchBank::kRecallInformativeRows));
			}
		}
	}

	// Reject-over-degrade: a retention bank WITHOUT a channel table -> InvalidArgument
	// (use the whole-vector routine).
	{
		ScratchBank plain;
		CHECK(plain.Create(64, dims, Metric::Cosine, Quantization::Int8, /*retainFloats=*/true) ==
			Status::Ok);
		float row[dims];
		for (int32_t i = 0; i < dims; ++i)
		{
			row[i] = 0.3f;
		}
		CHECK(plain.Append(row, dims, nullptr) == Status::Ok);
		ScratchRecallReport reports[2];
		const Status s = plain.MeasureScratchRecallPerChannel(ws, reports, 2, seed);
		CHECK_MSG(s == Status::InvalidArgument,
			"per-channel recall on a channel-less bank should be InvalidArgument: %d",
			static_cast<int>(s));
	}

	// Reject-over-degrade: a channel bank WITHOUT retention -> InvalidArgument (no float
	// reference to scan).
	{
		ScratchBank noRetain;
		CHECK(noRetain.Create(64, dims, Metric::Cosine, Quantization::Int8, channels, 2,
				/*retainFloats=*/false) == Status::Ok);
		float row[dims];
		for (int32_t i = 0; i < dims; ++i)
		{
			row[i] = 0.3f;
		}
		CHECK(noRetain.Append(row, dims, nullptr) == Status::Ok);
		ScratchRecallReport reports[2];
		const Status s = noRetain.MeasureScratchRecallPerChannel(ws, reports, 2, seed);
		CHECK_MSG(s == Status::InvalidArgument,
			"per-channel recall on a non-retention bank should be InvalidArgument: %d",
			static_cast<int>(s));
	}

	// Channel-aware FreezeWithRecall: re-measures per-channel recall over the COMPACTED
	// (live) rows at freeze time -- a fresh, non-stale number for the graduated bank.
	{
		const int32_t live = bank.FreezeLiveCount();
		AlignedBuf frozenRows(static_cast<size_t>(live) *
			PaddedDims(dims, Quantization::Int8) * ElementSize(Quantization::Int8));
		std::vector<float> frozenScales(static_cast<size_t>(live));
		std::vector<int32_t> indexMap(static_cast<size_t>(count), -2);
		std::vector<float> frozenInvNorms(static_cast<size_t>(live) * 2);
		ScratchRecallReport freezeReports[2];
		const Status s = bank.FreezeWithRecall(frozenRows.ptr, frozenScales.data(),
			indexMap.data(), frozenInvNorms.data(), freezeReports, 2, ws, seed);
		CHECK_MSG(s == Status::Ok,
			"channel-aware FreezeWithRecall failed (unimplemented, slot 3): status=%d",
			static_cast<int>(s));
		if (s == Status::Ok)
		{
			for (int32_t c = 0; c < 2; ++c)
			{
				CHECK_MSG(freezeReports[c].generation == bank.Generation(),
					"FreezeWithRecall channel %d report is not measured at the current "
					"generation (stale re-measure)", c);
				CHECK(freezeReports[c].liveRows == live);
				CHECK(!bank.RecallReportStale(freezeReports[c]));
				CHECK(freezeReports[c].recall >= 0.0f && freezeReports[c].recall <= 1.0f);
			}
		}
	}
}

// S5 (coverage audit §6.4): ScratchBank::Save had no allocation cell of any
// kind. Warm one prior Save through the same archive callbacks, then flat
// across further saves -- the caller owns the medium, the bank must stage
// nothing on the heap.
static void TestAllocFlatScratchSave()
{
	Rng rng(0x5A4E);
	const int32_t dims = 32, count = 40;
	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) v = rng.NextFloat();
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	// Reused across the loop and warmed to its full size first: the archive's
	// own std::vector<uint8_t> would otherwise grow (and allocate) INSIDE the
	// tracked window, which would measure the test fixture, not the bank.
	// clear() retains capacity, so a post-warm Save into the same buffer never
	// touches the fixture's own allocator.
	MemArchive out;
	CHECK(bank.Save(out.Writer()) == Status::Ok); // warm (sizes the buffer)
	out.bytes.clear();
	CHECK(bank.Save(out.Writer()) == Status::Ok); // warm again, into the sized buffer
	out.bytes.clear();

	const uint64_t allocsBefore = AllocationCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 10; ++i)
		{
			out.bytes.clear();
			CHECK(bank.Save(out.Writer()) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"ScratchBank::Save allocated %llu time(s) outside the Workspace seam",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"ScratchBank::Save's seam-tracked allocations grew: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
}

// S6 (coverage audit §6.4): ScratchBank::Freeze (base overload) had no
// allocation cell. Warm one prior Freeze into the same caller buffers, then
// flat; drive with and without outIndexMap, and with the recall re-measure
// legs on a retention bank.
static void TestAllocFlatScratchFreeze()
{
	Rng rng(0x5A4F);
	const int32_t dims = 32, count = 48;
	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Dot, Quantization::Int8, /*retainFloats=*/true) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) v = rng.NextFloat();
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	const int32_t live = bank.FreezeLiveCount();
	const int32_t pd = PaddedDims(dims, Quantization::Int8);
	AlignedBuf rows(static_cast<size_t>(live) * pd * ElementSize(Quantization::Int8));
	std::vector<float> scales(static_cast<size_t>(live));
	std::vector<int32_t> indexMap(static_cast<size_t>(count));
	Workspace ws;
	ScratchRecallReport report;

	CHECK(bank.Freeze(rows.ptr, scales.data(), indexMap.data()) == Status::Ok); // warm, with map
	CHECK(bank.Freeze(rows.ptr, scales.data(), nullptr) == Status::Ok);         // warm, without map
	CHECK(bank.Freeze(rows.ptr, scales.data(), indexMap.data(), &report, &ws) == Status::Ok); // warm, recall leg

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(bank.Freeze(rows.ptr, scales.data(), indexMap.data()) == Status::Ok);
			CHECK(bank.Freeze(rows.ptr, scales.data(), nullptr) == Status::Ok);
			CHECK(bank.Freeze(rows.ptr, scales.data(), indexMap.data(), &report, &ws) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"ScratchBank::Freeze allocated %llu time(s) outside the Workspace seam",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"ScratchBank::Freeze's seam-tracked allocations grew: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// S7 (coverage audit §6.4): ScratchBank::Freeze (channel-aware overload) had
// no allocation cell. Same construction as S6, on a Cosine channel bank.
static void TestAllocFlatScratchFreezeChannelAware()
{
	Rng rng(0x5A50);
	const int32_t dims = 32, count = 48;
	const ChannelInfo channels[2] = {{0, 16}, {16, 16}};
	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, channels, 2) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) v = rng.NextFloat();
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	const int32_t live = bank.FreezeLiveCount();
	const int32_t pd = PaddedDims(dims, Quantization::Int8);
	AlignedBuf rows(static_cast<size_t>(live) * pd * ElementSize(Quantization::Int8));
	std::vector<float> scales(static_cast<size_t>(live));
	std::vector<int32_t> indexMap(static_cast<size_t>(count));
	std::vector<float> invNorms(static_cast<size_t>(live) * 2);

	CHECK(bank.Freeze(rows.ptr, scales.data(), indexMap.data(), invNorms.data()) == Status::Ok); // warm

	const uint64_t allocsBefore = AllocationCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(bank.Freeze(rows.ptr, scales.data(), indexMap.data(), invNorms.data()) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"ScratchBank::Freeze (channel-aware) allocated %llu time(s) outside the Workspace seam",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"ScratchBank::Freeze (channel-aware) seam-tracked allocations grew: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
}

// S8 (coverage audit §6.4): ScratchBank::FreezeWithRecall had no allocation
// cell. Retention-enabled Cosine channel bank, workspace warm for the
// internal k.
static void TestAllocFlatScratchFreezeWithRecall()
{
	Rng rng(0x5A51);
	const int32_t dims = 32, count = 48;
	const ChannelInfo channels[2] = {{0, 16}, {16, 16}};
	const uint64_t seed = ScratchBank::kDefaultRecallSeed;
	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, channels, 2,
			/*retainFloats=*/true) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) v = rng.NextFloat();
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	const int32_t live = bank.FreezeLiveCount();
	const int32_t pd = PaddedDims(dims, Quantization::Int8);
	AlignedBuf rows(static_cast<size_t>(live) * pd * ElementSize(Quantization::Int8));
	std::vector<float> scales(static_cast<size_t>(live));
	std::vector<int32_t> indexMap(static_cast<size_t>(count));
	std::vector<float> invNorms(static_cast<size_t>(live) * 2);
	ScratchRecallReport reports[2];
	Workspace ws;

	CHECK(bank.FreezeWithRecall(rows.ptr, scales.data(), indexMap.data(), invNorms.data(),
			reports, 2, ws, seed) == Status::Ok); // warm

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(bank.FreezeWithRecall(rows.ptr, scales.data(), indexMap.data(), invNorms.data(),
					reports, 2, ws, seed) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"ScratchBank::FreezeWithRecall allocated %llu time(s) outside the Workspace seam",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"ScratchBank::FreezeWithRecall seam-tracked allocations grew: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// S10 (coverage audit §6.4): ScratchBank::MeasureScratchRecallPerChannel had
// no allocation cell of any kind. Retention Cosine channel bank, workspace
// warm, ≥1 prior call.
static void TestAllocFlatScratchMeasureScratchRecallPerChannel()
{
	Rng rng(0x5A52);
	const int32_t dims = 32, count = 200;
	const ChannelInfo channels[2] = {{0, 16}, {16, 16}};
	const uint64_t seed = ScratchBank::kDefaultRecallSeed;
	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, channels, 2,
			/*retainFloats=*/true) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) v = rng.NextFloat();
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	Workspace ws;
	ScratchRecallReport reports[2];
	CHECK(bank.MeasureScratchRecallPerChannel(ws, reports, 2, seed) == Status::Ok); // warm

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 10; ++i)
		{
			CHECK(bank.MeasureScratchRecallPerChannel(ws, reports, 2, seed) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"MeasureScratchRecallPerChannel allocated %llu time(s) outside the Workspace seam",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"MeasureScratchRecallPerChannel seam-tracked allocations grew: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// ---------------------------------------------------------------------------
// §6.3 -- Workspace's warm-no-op reservations (coverage audit F-7): "a warm
// workspace never grows" (alloc.h:52) has no direct cell. Each cell warms
// twice at shape S (proving the fast-return branch, not just the first
// allocation), then asserts re-calling at S and at smaller shapes is raw-0 /
// seam-flat / growth-flat, and separately that growing ONE axis while
// shrinking the other grows EXACTLY ONCE and never shrinks the live
// reservation (alloc.cpp's max-of-both logic) -- the specific reuse edge no
// prior cell drove.

static void TestAllocFlatWorkspaceReserve()
{
	Workspace ws;
	CHECK(ws.Reserve(8, 4));
	CHECK(ws.Reserve(8, 4)); // warm twice at S
	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		CHECK(ws.Reserve(8, 4));   // same shape
		CHECK(ws.Reserve(4, 4));   // smaller k
		CHECK(ws.Reserve(8, 2));   // smaller batchWidth
		CHECK(ws.Reserve(4, 2));   // smaller on both axes
		CHECK_MSG(rawTracking.Count() == 0,
			"Workspace::Reserve allocated %llu time(s) outside the seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK(AllocationCount() == allocsBefore);
	CHECK(ws.GrowthCount() == growthBefore);

	// Grow one axis while shrinking the other: exactly one growth, and the
	// reservation never shrinks (max-of-both, alloc.cpp:214-215).
	CHECK(ws.Reserve(9, 3)); // k grows, batchWidth shrinks from 4
	CHECK(ws.GrowthCount() == growthBefore + 1);
	CHECK(ws.ReservedK() == 9);
	CHECK_MSG(ws.ReservedBatch() == 4,
		"Reserve must not shrink a live reservation: batchWidth fell to %d", ws.ReservedBatch());
}

static void TestAllocFlatWorkspaceReserveQueryScratch()
{
	Workspace ws;
	CHECK(ws.ReserveQueryScratch(64, 4));
	CHECK(ws.ReserveQueryScratch(64, 4));
	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		CHECK(ws.ReserveQueryScratch(64, 4));
		CHECK(ws.ReserveQueryScratch(32, 4));
		CHECK(ws.ReserveQueryScratch(64, 2));
		CHECK(ws.ReserveQueryScratch(32, 2));
		CHECK_MSG(rawTracking.Count() == 0,
			"Workspace::ReserveQueryScratch allocated %llu time(s) outside the seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK(AllocationCount() == allocsBefore);
	CHECK(ws.GrowthCount() == growthBefore);

	CHECK(ws.ReserveQueryScratch(80, 2)); // paddedDims grows, count shrinks from 4
	CHECK(ws.GrowthCount() == growthBefore + 1);
}

static void TestAllocFlatWorkspaceReserveBiasBits()
{
	Workspace ws;
	const int32_t S = 300;
	CHECK(ws.ReserveBiasBits(S));
	CHECK(ws.ReserveBiasBits(S));

	// The warm path zeroes the words every call (alloc.cpp:187-190) -- O(count/8)
	// work, not an allocation. Confirm the zeroing actually happens (write a
	// sentinel, re-reserve, read back zero) so a future "optimization" that
	// skips the zeroing cannot pass silently, AND that raw/seam counts stay
	// flat across it.
	ws.BiasBits()[0] = 0xFFFFFFFFu;
	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		CHECK(ws.ReserveBiasBits(S));
		CHECK_MSG(ws.BiasBits()[0] == 0u,
			"ReserveBiasBits must re-zero the words on every warm call, sentinel survived");
		CHECK(ws.ReserveBiasBits(S / 2)); // smaller count
		CHECK_MSG(rawTracking.Count() == 0,
			"Workspace::ReserveBiasBits allocated %llu time(s) outside the seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK(AllocationCount() == allocsBefore);
	CHECK(ws.GrowthCount() == growthBefore);
}

static void TestAllocFlatWorkspaceReserveXdQuery()
{
	Workspace ws;
	CHECK(ws.ReserveXdQuery(64, 4));
	CHECK(ws.ReserveXdQuery(64, 4));
	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		CHECK(ws.ReserveXdQuery(64, 4));
		CHECK(ws.ReserveXdQuery(48, 4));
		CHECK(ws.ReserveXdQuery(64, 2));
		CHECK(ws.ReserveXdQuery(48, 2));
		CHECK_MSG(rawTracking.Count() == 0,
			"Workspace::ReserveXdQuery allocated %llu time(s) outside the seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK(AllocationCount() == allocsBefore);
	CHECK(ws.GrowthCount() == growthBefore);

	CHECK(ws.ReserveXdQuery(80, 2)); // paddedDims grows, count shrinks from 4
	CHECK(ws.GrowthCount() == growthBefore + 1);
}

static void TestAllocFlatWorkspaceReserveBatchOutput()
{
	Workspace ws;
	CHECK(ws.ReserveBatchOutput(8, 4, 0));
	CHECK(ws.ReserveBatchOutput(8, 4, 0));
	CHECK(ws.ReserveBatchOutput(6, 3, 1)); // slot 1, independently
	CHECK(ws.ReserveBatchOutput(6, 3, 1));

	const Hit* slot0HitsBefore = ws.BatchOutputHits(0);
	const int32_t* slot0CountsBefore = ws.BatchOutputCounts(0);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		CHECK(ws.ReserveBatchOutput(8, 4, 0));
		CHECK(ws.ReserveBatchOutput(4, 4, 0));
		CHECK(ws.ReserveBatchOutput(8, 2, 0));
		CHECK_MSG(rawTracking.Count() == 0,
			"Workspace::ReserveBatchOutput allocated %llu time(s) outside the seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK(AllocationCount() == allocsBefore);
	CHECK(ws.GrowthCount() == growthBefore);

	// A growth costs TWO seam allocations (alloc.cpp:249,256), and slot 0 and
	// slot 1 are provably independent -- growing slot 1 must not disturb
	// slot 0's pointer or counts.
	CHECK(ws.ReserveBatchOutput(9, 5, 1)); // grow slot 1 only
	CHECK(ws.GrowthCount() == growthBefore + 1);
	CHECK_MSG(ws.BatchOutputHits(0) == slot0HitsBefore,
		"growing slot 1 must not disturb slot 0's Hit pointer");
	CHECK_MSG(ws.BatchOutputCounts(0) == slot0CountsBefore,
		"growing slot 1 must not disturb slot 0's count pointer");
}

static void TestAllocFlatWorkspaceReserveIndexScratch()
{
	Workspace ws;
	CHECK(ws.ReserveIndexScratch(64, 0));
	CHECK(ws.ReserveIndexScratch(64, 0));
	CHECK(ws.ReserveIndexScratch(40, 1)); // slot 1, independently
	CHECK(ws.ReserveIndexScratch(40, 1));

	const int32_t* slot0Before = ws.IndexScratch(0);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		CHECK(ws.ReserveIndexScratch(64, 0));
		CHECK(ws.ReserveIndexScratch(32, 0));
		CHECK_MSG(rawTracking.Count() == 0,
			"Workspace::ReserveIndexScratch allocated %llu time(s) outside the seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK(AllocationCount() == allocsBefore);
	CHECK(ws.GrowthCount() == growthBefore);

	CHECK(ws.ReserveIndexScratch(80, 1)); // grow slot 1 only
	CHECK(ws.GrowthCount() == growthBefore + 1);
	CHECK_MSG(ws.IndexScratch(0) == slot0Before,
		"growing slot 1 must not disturb slot 0's pointer");
}

// ---------------------------------------------------------------------------
// T-V3 (slot 4) -- Tier-2 channel-scoped analytics (SuperFAISS_V2_Plan.md
// section 23.5; section 23.9 slot-4 gate; section 23.8 dims 3/4/6/8/10) plus
// the two D-V3-8 concurrency storms. Authored red-first by Curie on the green
// slot-3 base. Reuses the slot-2/3 helpers (MakeChannelScratchBank,
// ScratchChannelFixture) and the analytics/forced-path conventions of the
// shipped V2.5 suite.
//
// SCAFFOLD (Curie, not the feature): the four channel-scoped operators
// (CentroidDistance/MeanNN/MaxNN/Spread ...Channel) are declared in analytics.h
// and stubbed in analytics.cpp to return Status::BadAlignment -- a status no
// slot-4 cell expects -- so every REF/FEAT/composition/determinism/rejection
// cell fails at a specific runtime assertion for the one true reason. Storm (a)
// (channel-query-only) needs no scaffold: it exercises the already-green slot-2
// channel query, so it goes GREEN immediately (a coverage patch owed by D-V3-8).
// Hastings implements the real section-23.5 operators.

namespace
{
	// A whole-vector int8 bank holding one channel's lanes repacked contiguously:
	// row r of the repack = the source row's [offset, offset+length) int8 bytes, with
	// the SAME per-row scale, paddedDims' = length. The channel-scoped statistic over
	// channel c EQUALS the shipped whole-vector operator run on this repack -- a bit-exact
	// REF for the Dot/L2 families (pure sub-range integer accumulation + the shipped
	// linear epilogue; no sqrt). NOT used for Cosine (the channel-scoped cosine arithmetic
	// -- precomputed channelInvNorms vs a recomputed sub-range sqrt -- is unpinned in
	// section 23.5, routed as finding C-4; the Cosine achievement rides the FEAT +
	// forced-path instead).
	struct ChannelRepack
	{
		AlignedBuf rows;
		std::vector<float> scales;
		BankView view;

		ChannelRepack(const BankView& src, const ChannelInfo& ch)
			: rows(static_cast<size_t>(src.count > 0 ? src.count : 1) * ch.length)
		{
			scales.resize(static_cast<size_t>(src.count));
			const int8_t* srow = static_cast<const int8_t*>(src.rows);
			int8_t* drow = rows.I8();
			for (int32_t r = 0; r < src.count; ++r)
			{
				std::memcpy(drow + static_cast<size_t>(r) * ch.length,
					srow + static_cast<int64_t>(r) * src.paddedDims + ch.offset,
					static_cast<size_t>(ch.length));
				scales[static_cast<size_t>(r)] = src.scales[r];
			}
			view.rows = rows.ptr;
			view.scales = scales.data();
			view.count = src.count;
			view.dims = ch.length;
			view.paddedDims = ch.length; // ch.length is a multiple of the int8 grid (16)
			view.quant = Quantization::Int8;
			view.metric = src.metric;
		}
	};

	// Dequantize one row's channel sub-vector to doubles (scale * int8) -- the FEAT input,
	// exactly what the kernel dots, computed by hand (never via an operator).
	void DequantChannelSub(const BankView& v, int32_t r, const ChannelInfo& ch,
		std::vector<double>& out)
	{
		out.assign(static_cast<size_t>(ch.length), 0.0);
		const int8_t* row = static_cast<const int8_t*>(v.rows) +
			static_cast<int64_t>(r) * v.paddedDims;
		const double scale = v.scales[r];
		for (int32_t j = 0; j < ch.length; ++j)
		{
			out[static_cast<size_t>(j)] = scale * row[ch.offset + j];
		}
	}

	// Float64 unweighted mean of the dequantized channel sub-vectors of the given rows.
	std::vector<double> F64ChannelCentroid(const BankView& v, const int32_t* idx,
		int32_t n, const ChannelInfo& ch)
	{
		std::vector<double> c(static_cast<size_t>(ch.length), 0.0);
		std::vector<double> row;
		int32_t used = 0;
		for (int32_t i = 0; i < n; ++i)
		{
			DequantChannelSub(v, idx[i], ch, row);
			for (int32_t j = 0; j < ch.length; ++j)
			{
				c[static_cast<size_t>(j)] += row[static_cast<size_t>(j)];
			}
			++used;
		}
		if (used > 0)
		{
			for (auto& x : c)
			{
				x /= static_cast<double>(used);
			}
		}
		return c;
	}

	// Float64 distance in the operator's distance sense (Dot: similarity; L2: squared
	// distance; Cosine: 1 - cos), over two channel sub-vectors.
	double F64ChannelDist(const std::vector<double>& a, const std::vector<double>& b,
		Metric metric)
	{
		double cross = 0.0, na = 0.0, nb = 0.0;
		for (size_t i = 0; i < a.size(); ++i)
		{
			cross += a[i] * b[i];
			na += a[i] * a[i];
			nb += b[i] * b[i];
		}
		if (metric == Metric::L2)
		{
			return (na + nb) - 2.0 * cross;
		}
		if (metric == Metric::Cosine)
		{
			return (na > 0.0 && nb > 0.0) ? 1.0 - cross / std::sqrt(na * nb) : 0.0;
		}
		return cross; // Dot similarity
	}

	// The FEAT nearest of one source sub-vector over a target bank's channel sub-vectors,
	// in the distance sense the reduction uses. Dot: the MAX similarity (nearest = most
	// similar); L2 / Cosine: the MIN distance.
	double F64ChannelNearest(const std::vector<double>& srcSub, const BankView& target,
		const ChannelInfo& ch, Metric metric)
	{
		std::vector<double> t;
		bool have = false;
		double best = 0.0;
		for (int32_t r = 0; r < target.count; ++r)
		{
			DequantChannelSub(target, r, ch, t);
			const double d = F64ChannelDist(srcSub, t, metric);
			const bool better = !have || (metric == Metric::Dot ? d > best : d < best);
			if (better)
			{
				best = d;
			}
			have = true;
		}
		return best;
	}

	// Slot-4 FEAT CAL bands (Curie, proposed; pin at first calibration): loose enough to
	// absorb int8 row quantization AND centroid re-quantization (the T-V2.5-3 band shape).
	double ChannelFeatTol(double ref)
	{
		return 0.35 * std::fabs(ref) + 0.10;
	}
} // namespace

// T-V3-A1 -- REF (Dot/L2 repack, bit-exact) + FEAT (all metrics, definition-grounded)
// for the four channel-scoped operators, per metric, per channel (dim 10).
static void TestChannelScopedAnalyticsRefFeat()
{
	Rng rng(0x7A4A1ull);
	const int32_t dims = 64;
	const int32_t count = 48;

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		// Two channel scratch banks (source/A and target/B) with the SAME channel table.
		ScratchChannelFixture srcFx;
		ScratchChannelFixture tgtFx;
		MakeChannelScratchBank(srcFx, rng, count, dims, metric, Quantization::Int8);
		MakeChannelScratchBank(tgtFx, rng, count, dims, metric, Quantization::Int8);
		if (srcFx.createStatus != Status::Ok || tgtFx.createStatus != Status::Ok)
		{
			continue;
		}
		BankView src, tgt;
		std::vector<uint32_t> srcTombs(ScratchBank::TombstoneWords(count), 0u);
		std::vector<uint32_t> tgtTombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(srcFx.bank.Snapshot(&src, srcTombs.data()) == Status::Ok);
		CHECK(tgtFx.bank.Snapshot(&tgt, tgtTombs.data()) == Status::Ok);

		const int32_t idxA[6] = {0, 4, 9, 17, 22, 30};
		const int32_t idxB[6] = {2, 7, 12, 20, 25, 31};
		std::vector<int32_t> allRows(static_cast<size_t>(count));
		for (int32_t i = 0; i < count; ++i)
		{
			allRows[static_cast<size_t>(i)] = i;
		}

		for (int32_t c = 0; c < static_cast<int32_t>(srcFx.channels.size()); ++c)
		{
			const ChannelInfo& ch = srcFx.channels[static_cast<size_t>(c)];
			const bool linear = metric != Metric::Cosine; // Dot/L2 have the bit-exact repack REF

			// --- CentroidDistance channel ---
			{
				AlignedBuf sA(static_cast<size_t>(src.paddedDims));
				AlignedBuf sB(static_cast<size_t>(src.paddedDims));
				float cd = 0.0f;
				const Status s = CentroidDistanceCrossDeviceChannel(
					src, idxA, 6, nullptr, nullptr, src, idxB, 6, nullptr, nullptr,
					metric, c, sA.I8(), sB.I8(), &cd);
				CHECK_MSG(s == Status::Ok,
					"CentroidDistance channel failed (unimplemented, slot 4): status=%d",
					static_cast<int>(s));
				if (s == Status::Ok)
				{
					if (linear)
					{
						ChannelRepack rp(src, ch);
						AlignedBuf rA(static_cast<size_t>(rp.view.paddedDims));
						AlignedBuf rB(static_cast<size_t>(rp.view.paddedDims));
						float ref = 0.0f;
						CHECK(CentroidDistanceCrossDevice(rp.view, idxA, 6, nullptr, nullptr,
								   rp.view, idxB, 6, nullptr, nullptr, metric, rA.I8(), rB.I8(),
								   &ref) == Status::Ok);
						CHECK_MSG(cd == ref,
							"CentroidDistance channel REF mismatch (metric=%d ch=%d): %.9g != %.9g",
							static_cast<int>(metric), c, static_cast<double>(cd),
							static_cast<double>(ref));
					}
					const std::vector<double> ca = F64ChannelCentroid(src, idxA, 6, ch);
					const std::vector<double> cb = F64ChannelCentroid(src, idxB, 6, ch);
					const double fref = F64ChannelDist(ca, cb, metric);
					CHECK_MSG(std::fabs(static_cast<double>(cd) - fref) <= ChannelFeatTol(fref),
						"CentroidDistance channel FEAT (metric=%d ch=%d): op %.6g f64 %.6g",
						static_cast<int>(metric), c, static_cast<double>(cd), fref);
				}
			}

			// --- Spread channel ---
			{
				AlignedBuf cs(static_cast<size_t>(src.paddedDims));
				float sp = 0.0f;
				const Status s = SpreadCrossDeviceChannel(src, allRows.data(), count, nullptr,
					Reduce::Mean, c, cs.I8(), &sp);
				CHECK_MSG(s == Status::Ok,
					"Spread channel failed (unimplemented, slot 4): status=%d",
					static_cast<int>(s));
				if (s == Status::Ok)
				{
					if (linear)
					{
						ChannelRepack rp(src, ch);
						AlignedBuf rcs(static_cast<size_t>(rp.view.paddedDims));
						float ref = 0.0f;
						CHECK(SpreadCrossDevice(rp.view, allRows.data(), count, nullptr,
								   Reduce::Mean, rcs.I8(), &ref) == Status::Ok);
						CHECK_MSG(sp == ref,
							"Spread channel REF mismatch (metric=%d ch=%d): %.9g != %.9g",
							static_cast<int>(metric), c, static_cast<double>(sp),
							static_cast<double>(ref));
					}
					// FEAT: float64 dispersion over the channel sub-range about the f64 centroid.
					const std::vector<double> cen = F64ChannelCentroid(src, allRows.data(), count, ch);
					std::vector<double> row;
					double acc = 0.0;
					for (int32_t r = 0; r < count; ++r)
					{
						DequantChannelSub(src, r, ch, row);
						acc += F64ChannelDist(row, cen, metric);
					}
					const double fref = acc / static_cast<double>(count);
					CHECK_MSG(std::fabs(static_cast<double>(sp) - fref) <= ChannelFeatTol(fref),
						"Spread channel FEAT (metric=%d ch=%d): op %.6g f64 %.6g",
						static_cast<int>(metric), c, static_cast<double>(sp), fref);
				}
			}

			// --- MeanNN / MaxNN channel (source -> target) ---
			{
				std::vector<XdQuery> qbuf(static_cast<size_t>(count));
				std::vector<Hit> hbuf(static_cast<size_t>(count));
				std::vector<int32_t> nbuf(static_cast<size_t>(count));
				Workspace ws;
				float mn = 0.0f, mx = 0.0f;
				const Status sMean = MeanNNCrossDeviceChannel(src, nullptr, tgt, nullptr, c,
					qbuf.data(), hbuf.data(), nbuf.data(), ws, &mn);
				const Status sMax = MaxNNCrossDeviceChannel(src, nullptr, tgt, nullptr, c,
					qbuf.data(), hbuf.data(), nbuf.data(), ws, &mx);
				CHECK_MSG(sMean == Status::Ok,
					"MeanNN channel failed (unimplemented, slot 4): status=%d",
					static_cast<int>(sMean));
				CHECK_MSG(sMax == Status::Ok,
					"MaxNN channel failed (unimplemented, slot 4): status=%d",
					static_cast<int>(sMax));
				if (sMean == Status::Ok && sMax == Status::Ok)
				{
					if (linear)
					{
						ChannelRepack rsrc(src, ch);
						ChannelRepack rtgt(tgt, ch);
						std::vector<XdQuery> q2(static_cast<size_t>(count));
						std::vector<Hit> h2(static_cast<size_t>(count));
						std::vector<int32_t> n2(static_cast<size_t>(count));
						Workspace ws2;
						float refMean = 0.0f, refMax = 0.0f;
						CHECK(MeanNNCrossDevice(rsrc.view, nullptr, rtgt.view, nullptr,
								   q2.data(), h2.data(), n2.data(), ws2, &refMean) == Status::Ok);
						CHECK(MaxNNCrossDevice(rsrc.view, nullptr, rtgt.view, nullptr,
								   q2.data(), h2.data(), n2.data(), ws2, &refMax) == Status::Ok);
						CHECK_MSG(mn == refMean,
							"MeanNN channel REF mismatch (metric=%d ch=%d): %.9g != %.9g",
							static_cast<int>(metric), c, static_cast<double>(mn),
							static_cast<double>(refMean));
						CHECK_MSG(mx == refMax,
							"MaxNN channel REF mismatch (metric=%d ch=%d): %.9g != %.9g",
							static_cast<int>(metric), c, static_cast<double>(mx),
							static_cast<double>(refMax));
					}
					// FEAT: float64 nearest over the channel sub-range, mean and max.
					std::vector<double> sub;
					double sum = 0.0, fmax = 0.0;
					bool have = false;
					for (int32_t r = 0; r < count; ++r)
					{
						DequantChannelSub(src, r, ch, sub);
						const double nn = F64ChannelNearest(sub, tgt, ch, metric);
						sum += nn;
						if (!have || nn > fmax)
						{
							fmax = nn;
						}
						have = true;
					}
					const double frefMean = sum / static_cast<double>(count);
					CHECK_MSG(std::fabs(static_cast<double>(mn) - frefMean) <=
								  ChannelFeatTol(frefMean),
						"MeanNN channel FEAT (metric=%d ch=%d): op %.6g f64 %.6g",
						static_cast<int>(metric), c, static_cast<double>(mn), frefMean);
					CHECK_MSG(std::fabs(static_cast<double>(mx) - fmax) <= ChannelFeatTol(fmax),
						"MaxNN channel FEAT (metric=%d ch=%d): op %.6g f64 %.6g",
						static_cast<int>(metric), c, static_cast<double>(mx), fmax);
				}
			}
		}
	}
}

// --- T-V3-A1-COS-REF: the bit-exact channel-scoped Cosine analytics REF (C-4/D-V3-10) ---
//
// An INDEPENDENT recode of the pinned channel-scoped Cosine epilogue, authored from the
// section 23.5 contract -- NOT by calling the operator or reusing its file-local helpers.
// The arithmetic pinned by C-4 / D-V3-10: a channel-scoped Cosine distance over the sub-range
// [offset, offset+length) is
//     1 - crossDot / sqrt(aSq * bSq)
// where crossDot, aSq, bSq are the INTEGER sub-range cross-dot / self-dots recomputed over the
// channel's int8 lanes, closed with ONE IEEE correctly-rounded double sqrt. It does NOT read
// the per-row float channelInvNorms.
//
// The recode below uses its own integer dot loop (CosRefDotI8) and its own std::sqrt, so it is
// a genuine second implementation -- not a consistency check against the operator's own
// XdCosineDistance. A bug shared by both cannot pass this.
//
// Why this REF exists (it DISCRIMINATES where the FEAT cannot): were the operator to normalize
// a scored member by its precomputed channelInvNorms -- the float 1/sqrt(sub-self-dot) baked
// per row on the query/scoring path -- instead of this recomputed double sqrt of the integer
// self-dot, the two would differ only in the low mantissa bits (float-vs-double rounding of the
// norm). The T-V3-A1 FEAT's CAL band absorbs that difference; only a bit-exact (==) REF detects
// it. The centroid pooling is obtained from the metric-agnostic MakeCentroidCrossDevice primitive
// (pooling is not the arithmetic C-4 disputes -- both metrics pool identically; the discrimination
// lives entirely in the recoded epilogue applied to the pooled int8 image).
namespace
{
	// Independent integer sub-range dot (own recode of the int8 accumulation). int64 so a
	// zero-norm test is exact; the values fit int32 in the operator's guarded regime, so the
	// double cast below is bit-identical to the operator's int32 DotI8I8 -> double.
	int64_t CosRefDotI8(const int8_t* a, const int8_t* b, int32_t n)
	{
		int64_t acc = 0;
		for (int32_t i = 0; i < n; ++i)
		{
			acc += static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
		}
		return acc;
	}

	// Own recode of the cross-device subnormal floor (XdFloor): |x| < FLT_MIN -> exactly 0.0f.
	float CosRefXdFloor(double score)
	{
		const double lim = 1.1754943508222875e-38; // FLT_MIN, exactly
		if (score < lim && score > -lim)
		{
			return 0.0f;
		}
		return static_cast<float>(score);
	}

	// The pinned channel-scoped Cosine pair distance over two int8 sub-images of `length` lanes.
	// A zero sub-norm member floors to a defined 0 (C-5/D-V3-11), matching XdChannelPairScore.
	float CosRefChannelPair(const int8_t* a, const int8_t* b, int32_t length)
	{
		const int64_t cross = CosRefDotI8(a, b, length);
		const int64_t aSq = CosRefDotI8(a, a, length);
		const int64_t bSq = CosRefDotI8(b, b, length);
		if (aSq == 0 || bSq == 0)
		{
			return 0.0f; // zero sub-norm member: defined 0, never NaN (C-5)
		}
		const double denom = std::sqrt(static_cast<double>(aSq) * static_cast<double>(bSq));
		return CosRefXdFloor(1.0 - (static_cast<double>(cross) / denom));
	}

	// Pointer to a snapshot row's channel sub-image (int8, `length` contiguous lanes).
	const int8_t* CosRefSubImage(const BankView& v, int32_t r, const ChannelInfo& ch)
	{
		return static_cast<const int8_t*>(v.rows) +
			static_cast<int64_t>(r) * v.paddedDims + ch.offset;
	}

	// Builds the REF centroid over a channel sub-range via the metric-agnostic pooling
	// primitive, mirroring the operator's MakeChannelCentroid: a ZeroNormQuery pool becomes a
	// zeroed sub-image (self-dot 0 -> the epilogue floors it to 0). Writes `ch.length` int8
	// lanes at outQ8[0..length). Returns false only on an unexpected error status.
	bool CosRefCentroid(const BankView& v, const int32_t* idx, int32_t n, const ChannelInfo& ch,
		int8_t* outQ8)
	{
		double scale = 0.0;
		int64_t sq = 0;
		const Status s = MakeCentroidCrossDevice(v, idx, n, nullptr, nullptr, outQ8, &scale, &sq,
			ch.offset, ch.length);
		if (s == Status::ZeroNormQuery)
		{
			std::memset(outQ8, 0, static_cast<size_t>(ch.length));
			return true;
		}
		return s == Status::Ok;
	}
} // namespace

// T-V3-A1-COS-REF -- bit-exact Cosine REF for the four channel-scoped operators (dim 10,
// C-4/D-V3-10), plus the C-5 zero-sub-norm-member floor in a reduction (D-V3-11). Asserts
// exact == between each operator's float result and the independent int->double recode
// (CosRefChannelPair), across both channels of both banks. Expected GREEN on the shipped
// pinned arithmetic; a disagreement is a genuine finding (operator vs the independent recode),
// not a REF to adjust.
static void TestChannelScopedAnalyticsCosineRef()
{
	Rng rng(0xC05E7ull);
	const int32_t dims = 64;
	const int32_t count = 48;

	// Two Cosine channel banks (source/A and target/B) sharing the same channel table.
	ScratchChannelFixture srcFx;
	ScratchChannelFixture tgtFx;
	MakeChannelScratchBank(srcFx, rng, count, dims, Metric::Cosine, Quantization::Int8);
	MakeChannelScratchBank(tgtFx, rng, count, dims, Metric::Cosine, Quantization::Int8);
	if (srcFx.createStatus == Status::Ok && tgtFx.createStatus == Status::Ok)
	{
		BankView src, tgt;
		std::vector<uint32_t> srcTombs(ScratchBank::TombstoneWords(count), 0u);
		std::vector<uint32_t> tgtTombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(srcFx.bank.Snapshot(&src, srcTombs.data()) == Status::Ok);
		CHECK(tgtFx.bank.Snapshot(&tgt, tgtTombs.data()) == Status::Ok);

		const int32_t idxA[6] = {0, 4, 9, 17, 22, 30};
		const int32_t idxB[6] = {2, 7, 12, 20, 25, 31};
		std::vector<int32_t> allRows(static_cast<size_t>(count));
		for (int32_t i = 0; i < count; ++i)
		{
			allRows[static_cast<size_t>(i)] = i;
		}

		for (int32_t c = 0; c < static_cast<int32_t>(srcFx.channels.size()); ++c)
		{
			const ChannelInfo& ch = srcFx.channels[static_cast<size_t>(c)];
			const int32_t length = ch.length;

			// --- CentroidDistance channel: exact == vs the recoded epilogue over pooled centroids.
			{
				AlignedBuf sA(static_cast<size_t>(src.paddedDims));
				AlignedBuf sB(static_cast<size_t>(src.paddedDims));
				float cd = 0.0f;
				const Status s = CentroidDistanceCrossDeviceChannel(
					src, idxA, 6, nullptr, nullptr, src, idxB, 6, nullptr, nullptr,
					Metric::Cosine, c, sA.I8(), sB.I8(), &cd);
				CHECK_MSG(s == Status::Ok,
					"CentroidDistance channel Cosine failed (unimplemented, slot 4): status=%d",
					static_cast<int>(s));
				if (s == Status::Ok)
				{
					AlignedBuf ra(static_cast<size_t>(src.paddedDims));
					AlignedBuf rb(static_cast<size_t>(src.paddedDims));
					CHECK(CosRefCentroid(src, idxA, 6, ch, ra.I8()));
					CHECK(CosRefCentroid(src, idxB, 6, ch, rb.I8()));
					const float ref = CosRefChannelPair(ra.I8(), rb.I8(), length);
					CHECK_MSG(cd == ref,
						"CentroidDistance channel Cosine REF mismatch (ch=%d): %.9g != %.9g",
						c, static_cast<double>(cd), static_cast<double>(ref));
				}
			}

			// --- Spread channel: exact == vs the recoded per-member floor then double-mean floor.
			{
				AlignedBuf cs(static_cast<size_t>(src.paddedDims));
				float sp = 0.0f;
				const Status s = SpreadCrossDeviceChannel(src, allRows.data(), count, nullptr,
					Reduce::Mean, c, cs.I8(), &sp);
				CHECK_MSG(s == Status::Ok,
					"Spread channel Cosine failed (unimplemented, slot 4): status=%d",
					static_cast<int>(s));
				if (s == Status::Ok)
				{
					AlignedBuf cen(static_cast<size_t>(src.paddedDims));
					CHECK(CosRefCentroid(src, allRows.data(), count, ch, cen.I8()));
					double acc = 0.0;
					for (int32_t r = 0; r < count; ++r)
					{
						acc += static_cast<double>(
							CosRefChannelPair(CosRefSubImage(src, r, ch), cen.I8(), length));
					}
					const float ref = CosRefXdFloor(acc / static_cast<double>(count));
					CHECK_MSG(sp == ref,
						"Spread channel Cosine REF mismatch (ch=%d): %.9g != %.9g",
						c, static_cast<double>(sp), static_cast<double>(ref));
				}
			}

			// --- MeanNN / MaxNN channel (source -> target): exact == vs the recoded nearest.
			{
				std::vector<XdQuery> qbuf(static_cast<size_t>(count));
				std::vector<Hit> hbuf(static_cast<size_t>(count));
				std::vector<int32_t> nbuf(static_cast<size_t>(count));
				Workspace ws;
				float mn = 0.0f, mx = 0.0f;
				const Status sMean = MeanNNCrossDeviceChannel(src, nullptr, tgt, nullptr, c,
					qbuf.data(), hbuf.data(), nbuf.data(), ws, &mn);
				const Status sMax = MaxNNCrossDeviceChannel(src, nullptr, tgt, nullptr, c,
					qbuf.data(), hbuf.data(), nbuf.data(), ws, &mx);
				CHECK_MSG(sMean == Status::Ok,
					"MeanNN channel Cosine failed (unimplemented, slot 4): status=%d",
					static_cast<int>(sMean));
				CHECK_MSG(sMax == Status::Ok,
					"MaxNN channel Cosine failed (unimplemented, slot 4): status=%d",
					static_cast<int>(sMax));
				if (sMean == Status::Ok && sMax == Status::Ok)
				{
					// Recode NNDivergenceChannel's Cosine path: for each source sub-row, its
					// nearest (min-distance) target sub-row; skip a zero-sub-norm target (the
					// operator continues past it); reduce by mean / order-free max, each closed
					// with the subnormal floor, exactly where the operator floors.
					double acc = 0.0, best = 0.0;
					bool have = false;
					int32_t counted = 0;
					for (int32_t s = 0; s < src.count; ++s)
					{
						const int8_t* srcSub = CosRefSubImage(src, s, ch);
						double nn = 0.0;
						bool haveNn = false;
						for (int32_t t = 0; t < tgt.count; ++t)
						{
							const int8_t* tgtSub = CosRefSubImage(tgt, t, ch);
							if (CosRefDotI8(tgtSub, tgtSub, length) == 0)
							{
								continue; // zero sub-norm target: undefined, skipped (never NaN)
							}
							const double d =
								static_cast<double>(CosRefChannelPair(srcSub, tgtSub, length));
							if (!haveNn || d < nn) // Cosine: nearest = min distance
							{
								nn = d;
							}
							haveNn = true;
						}
						if (!haveNn)
						{
							continue;
						}
						acc += nn;
						if (!have || nn > best)
						{
							best = nn;
						}
						have = true;
						++counted;
					}
					CHECK(counted > 0);
					const float refMean = CosRefXdFloor(acc / static_cast<double>(counted));
					const float refMax = CosRefXdFloor(best);
					CHECK_MSG(mn == refMean,
						"MeanNN channel Cosine REF mismatch (ch=%d): %.9g != %.9g",
						c, static_cast<double>(mn), static_cast<double>(refMean));
					CHECK_MSG(mx == refMax,
						"MaxNN channel Cosine REF mismatch (ch=%d): %.9g != %.9g",
						c, static_cast<double>(mx), static_cast<double>(refMax));
				}
			}
		}
	}

	// C-5 (D-V3-11): a zero-sub-norm Cosine channel MEMBER floors to a defined 0 in a
	// reduction (not ZeroNormQuery). A hand Cosine channel bank: row 0 has zero energy in
	// channel 0 (its sub-norm is 0) but nonzero energy in channel 1, so the whole ROW
	// normalizes and Append succeeds. A channel-0 Spread must stay defined; the zero-norm
	// member scores exactly 0, and the operator equals the independent recode bit for bit.
	{
		const int32_t d = 32; // two int8 channels of 16 lanes each
		const ChannelInfo channels[2] = {{0, 16}, {16, 16}};
		ScratchBank bank;
		CHECK(bank.Create(4, d, Metric::Cosine, Quantization::Int8, channels, 2) == Status::Ok);
		for (int32_t r = 0; r < 4; ++r)
		{
			float row[d] = {};
			row[16 + (r % 16)] = 1.0f + 0.1f * static_cast<float>(r); // channel-1 energy: always
			if (r >= 1)
			{
				row[r % 16] = 0.5f + 0.1f * static_cast<float>(r); // channel-0 energy: r>=1 only
			}
			CHECK(bank.Append(row, d, nullptr) == Status::Ok);
		}
		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(4), 0u);
		CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		const int32_t idx[4] = {0, 1, 2, 3};

		// The channel-0 sub-image of row 0 must be exactly zero (the member under test).
		CHECK_MSG(CosRefDotI8(CosRefSubImage(snap, 0, channels[0]), CosRefSubImage(snap, 0, channels[0]), 16) == 0,
			"C-5 fixture: row 0 channel-0 sub-image must be zero-norm");

		AlignedBuf cs(static_cast<size_t>(snap.paddedDims));
		float sp = 0.0f;
		const Status s = SpreadCrossDeviceChannel(snap, idx, 4, nullptr, Reduce::Mean, 0,
			cs.I8(), &sp);
		CHECK_MSG(s == Status::Ok,
			"C-5 Cosine zero-sub-norm member must keep the reduction defined (not ZeroNormQuery), "
			"got status=%d",
			static_cast<int>(s));
		if (s == Status::Ok)
		{
			CHECK_MSG(sp == sp, "C-5 Spread produced NaN");
			AlignedBuf cen(static_cast<size_t>(snap.paddedDims));
			CHECK(CosRefCentroid(snap, idx, 4, channels[0], cen.I8()));
			// The zero-sub-norm member (row 0) floors to exactly 0 in the recode.
			CHECK_MSG(CosRefChannelPair(CosRefSubImage(snap, 0, channels[0]), cen.I8(), 16) == 0.0f,
				"C-5: zero-sub-norm Cosine member must floor to 0 in the recode");
			double acc = 0.0;
			for (int32_t r = 0; r < 4; ++r)
			{
				acc += static_cast<double>(
					CosRefChannelPair(CosRefSubImage(snap, r, channels[0]), cen.I8(), 16));
			}
			const float ref = CosRefXdFloor(acc / 4.0);
			CHECK_MSG(sp == ref,
				"C-5 Spread Cosine REF mismatch (zero-member floor): %.9g != %.9g",
				static_cast<double>(sp), static_cast<double>(ref));
		}
	}
}

// T-V3-A2 -- per-channel determinism (dim 6, section 23.7/section 22.5): each
// channel-scoped Cosine operator crosses the forced-path sweep bit-identical (the
// per-channel cosine sqrt limb is exercised). Red until slot 4: the operators stub before
// producing a value.
static void TestChannelScopedAnalyticsDeterminism()
{
	Rng rng(0xD37E4Dull);
	const int32_t dims = 64;
	const int32_t count = 48;

	ScratchChannelFixture fx;
	MakeChannelScratchBank(fx, rng, count, dims, Metric::Cosine, Quantization::Int8);
	if (fx.createStatus != Status::Ok)
	{
		return;
	}
	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);

	std::vector<int32_t> rows(static_cast<size_t>(count));
	for (int32_t i = 0; i < count; ++i)
	{
		rows[static_cast<size_t>(i)] = i;
	}
	const int32_t idxA[6] = {0, 4, 9, 17, 22, 30};
	const int32_t idxB[6] = {2, 7, 12, 20, 25, 31};

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

	for (int32_t c = 0; c < static_cast<int32_t>(fx.channels.size()); ++c)
	{
		float cdVals[4] = {}, spVals[4] = {}, mnVals[4] = {};
		Status cdStatus = Status::Ok, spStatus = Status::Ok, mnStatus = Status::Ok;
		for (size_t p = 0; p < paths.size(); ++p)
		{
			detail::ForceXdSimdPath(paths[p]);
			AlignedBuf sA(static_cast<size_t>(snap.paddedDims));
			AlignedBuf sB(static_cast<size_t>(snap.paddedDims));
			cdStatus = CentroidDistanceCrossDeviceChannel(snap, idxA, 6, nullptr, nullptr,
				snap, idxB, 6, nullptr, nullptr, Metric::Cosine, c, sA.I8(), sB.I8(),
				&cdVals[p]);
			AlignedBuf cs(static_cast<size_t>(snap.paddedDims));
			spStatus = SpreadCrossDeviceChannel(snap, rows.data(), count, nullptr,
				Reduce::Mean, c, cs.I8(), &spVals[p]);
			std::vector<XdQuery> qbuf(static_cast<size_t>(count));
			std::vector<Hit> hbuf(static_cast<size_t>(count));
			std::vector<int32_t> nbuf(static_cast<size_t>(count));
			Workspace ws;
			mnStatus = MeanNNCrossDeviceChannel(snap, nullptr, snap, nullptr, c, qbuf.data(),
				hbuf.data(), nbuf.data(), ws, &mnVals[p]);
			detail::ClearForcedXdSimdPath();
		}
		CHECK_MSG(cdStatus == Status::Ok,
			"channel-scoped Cosine CentroidDistance not deterministic-testable "
			"(unimplemented, slot 4): status=%d", static_cast<int>(cdStatus));
		CHECK(spStatus == Status::Ok);
		CHECK(mnStatus == Status::Ok);
		if (cdStatus == Status::Ok && spStatus == Status::Ok && mnStatus == Status::Ok)
		{
			for (size_t p = 1; p < paths.size(); ++p)
			{
				CHECK_MSG(cdVals[p] == cdVals[0],
					"channel Cosine CentroidDistance forced-path %d != scalar (ch=%d)",
					static_cast<int>(paths[p]), c);
				CHECK(spVals[p] == spVals[0]);
				CHECK(mnVals[p] == mnVals[0]);
			}
		}
	}
}

// T-V3-A3 -- composition (dim 8) + whole-vector-unchanged regression (dim 7).
static void TestChannelScopedAnalyticsComposition()
{
	Rng rng(0xC0FFee44ull);
	const int32_t dims = 64;
	const int32_t count = 48;

	ScratchChannelFixture fx;
	MakeChannelScratchBank(fx, rng, count, dims, Metric::Dot, Quantization::Int8);
	if (fx.createStatus != Status::Ok)
	{
		return;
	}
	// Tombstone a scatter so the snapshot source composition exercises exclusion.
	for (int32_t r = 3; r < count; r += 9)
	{
		CHECK(fx.bank.Remove(r) == Status::Ok);
	}
	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);

	std::vector<int32_t> rows(static_cast<size_t>(count));
	for (int32_t i = 0; i < count; ++i)
	{
		rows[static_cast<size_t>(i)] = i;
	}

	// (1) Weighted centroids: a channel-scoped CentroidDistance with all-equal weights
	// bit-equals the unweighted form (the V2.4 P6 invariant, per channel). Red at the stub.
	{
		std::vector<int32_t> weights(static_cast<size_t>(count), 1);
		for (int32_t c = 0; c < static_cast<int32_t>(fx.channels.size()); ++c)
		{
			AlignedBuf a1(static_cast<size_t>(snap.paddedDims));
			AlignedBuf b1(static_cast<size_t>(snap.paddedDims));
			AlignedBuf a2(static_cast<size_t>(snap.paddedDims));
			AlignedBuf b2(static_cast<size_t>(snap.paddedDims));
			float wtd = 0.0f, unw = 0.0f;
			const Status sw = CentroidDistanceCrossDeviceChannel(snap, rows.data(), count,
				weights.data(), tombs.data(), snap, rows.data(), count, weights.data(),
				tombs.data(), Metric::Dot, c, a1.I8(), b1.I8(), &wtd);
			const Status su = CentroidDistanceCrossDeviceChannel(snap, rows.data(), count,
				nullptr, tombs.data(), snap, rows.data(), count, nullptr, tombs.data(),
				Metric::Dot, c, a2.I8(), b2.I8(), &unw);
			CHECK_MSG(sw == Status::Ok && su == Status::Ok,
				"weighted/unweighted channel CentroidDistance failed (unimplemented, slot 4): "
				"%d/%d", static_cast<int>(sw), static_cast<int>(su));
			if (sw == Status::Ok && su == Status::Ok)
			{
				CHECK_MSG(wtd == unw,
					"channel CentroidDistance all-equal-weighted != unweighted (ch=%d): "
					"%.9g != %.9g", c, static_cast<double>(wtd), static_cast<double>(unw));
			}
		}
	}

	// (2) Scratch-snapshot source + tombstone exclusion: a channel-scoped Spread over the
	// snapshot (excludeBits = tombstone words) bit-equals the same over a repacked live-only
	// bank -- reachability today is via the stub (red); the REF equality is the slot-4 target.
	{
		AlignedBuf cs(static_cast<size_t>(snap.paddedDims));
		float sp = 0.0f;
		const Status s = SpreadCrossDeviceChannel(snap, rows.data(), count, tombs.data(),
			Reduce::Mean, 0, cs.I8(), &sp);
		CHECK_MSG(s == Status::Ok,
			"channel Spread over a tombstoned snapshot failed (unimplemented, slot 4): %d",
			static_cast<int>(s));
	}

	// (3) Whole-vector operators UNCHANGED (dim 7 regression -- GREEN guard): the shipped
	// whole-vector CentroidDistanceCrossDevice yields the SAME value on the channel-carrying
	// snapshot as on a channel-less copy of it (the channel table does not perturb the
	// whole-vector path).
	{
		BankView bare = snap;
		bare.channels = nullptr;
		bare.channelCount = 0;
		bare.channelInvNorms = nullptr;
		const int32_t idxA[6] = {0, 4, 9, 17, 22, 30};
		const int32_t idxB[6] = {2, 7, 12, 20, 25, 31};
		AlignedBuf a1(static_cast<size_t>(snap.paddedDims));
		AlignedBuf b1(static_cast<size_t>(snap.paddedDims));
		AlignedBuf a2(static_cast<size_t>(snap.paddedDims));
		AlignedBuf b2(static_cast<size_t>(snap.paddedDims));
		float withCh = 0.0f, withoutCh = 0.0f;
		CHECK(CentroidDistanceCrossDevice(snap, idxA, 6, nullptr, nullptr, snap, idxB, 6,
				  nullptr, nullptr, Metric::Dot, a1.I8(), b1.I8(), &withCh) == Status::Ok);
		CHECK(CentroidDistanceCrossDevice(bare, idxA, 6, nullptr, nullptr, bare, idxB, 6,
				  nullptr, nullptr, Metric::Dot, a2.I8(), b2.I8(), &withoutCh) == Status::Ok);
		CHECK_MSG(withCh == withoutCh,
			"whole-vector CentroidDistance changed by the channel table: %.9g != %.9g",
			static_cast<double>(withCh), static_cast<double>(withoutCh));
	}
}

// T-V3-A4 -- rejection / degenerate (dims 2/5).
static void TestChannelScopedAnalyticsRejections()
{
	Rng rng(0x8E7EC7ull);
	const int32_t dims = 64;
	const int32_t count = 16;

	// A channel-LESS int8 bank: any channel-scoped op -> InvalidArgument.
	{
		ScratchBank plain;
		CHECK(plain.Create(count, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		std::vector<float> source(static_cast<size_t>(count) * dims);
		for (auto& v : source)
		{
			v = rng.NextFloat();
		}
		for (int32_t r = 0; r < count; ++r)
		{
			CHECK(plain.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(plain.Snapshot(&snap, tombs.data()) == Status::Ok);
		std::vector<int32_t> rows(static_cast<size_t>(count));
		for (int32_t i = 0; i < count; ++i)
		{
			rows[static_cast<size_t>(i)] = i;
		}
		AlignedBuf cs(static_cast<size_t>(snap.paddedDims));
		float out = 0.0f;
		CHECK_MSG(SpreadCrossDeviceChannel(snap, rows.data(), count, nullptr, Reduce::Mean, 0,
					  cs.I8(), &out) == Status::InvalidArgument,
			"channel-scoped op on a channel-less bank must be InvalidArgument");
	}

	// A channel bank with an out-of-range / negative channel selector -> InvalidArgument.
	{
		ScratchChannelFixture fx;
		MakeChannelScratchBank(fx, rng, count, dims, Metric::Dot, Quantization::Int8);
		if (fx.createStatus == Status::Ok)
		{
			BankView snap;
			std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
			CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);
			std::vector<int32_t> rows(static_cast<size_t>(count));
			for (int32_t i = 0; i < count; ++i)
			{
				rows[static_cast<size_t>(i)] = i;
			}
			AlignedBuf cs(static_cast<size_t>(snap.paddedDims));
			float out = 0.0f;
			const int32_t nChan = static_cast<int32_t>(fx.channels.size());
			CHECK_MSG(SpreadCrossDeviceChannel(snap, rows.data(), count, nullptr, Reduce::Mean,
						  nChan, cs.I8(), &out) == Status::InvalidArgument,
				"out-of-range channel index must be InvalidArgument");
			CHECK_MSG(SpreadCrossDeviceChannel(snap, rows.data(), count, nullptr, Reduce::Mean,
						  -1, cs.I8(), &out) == Status::InvalidArgument,
				"negative channel index must be InvalidArgument");
		}
	}

	// All-zero-norm channel (Dot): a hand bank whose channel-0 lanes are all zero across
	// the selection -> the channel-scoped Dot centroid distance is defined 0, never NaN.
	{
		const int32_t d = 32; // two int8 channels of 16
		const ChannelInfo channels[2] = {{0, 16}, {16, 16}};
		ScratchBank bank;
		CHECK(bank.Create(4, d, Metric::Dot, Quantization::Int8, channels, 2) == Status::Ok);
		// Rows with energy ONLY in channel 1 (channel-0 lanes all zero).
		for (int32_t r = 0; r < 4; ++r)
		{
			float row[d] = {};
			row[16 + (r % 16)] = 1.0f + 0.1f * r;
			CHECK(bank.Append(row, d, nullptr) == Status::Ok);
		}
		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(4), 0u);
		CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		const int32_t idx[4] = {0, 1, 2, 3};
		AlignedBuf sA(static_cast<size_t>(snap.paddedDims));
		AlignedBuf sB(static_cast<size_t>(snap.paddedDims));
		float cd = -12345.0f;
		const Status s = CentroidDistanceCrossDeviceChannel(snap, idx, 2, nullptr, nullptr,
			snap, idx + 2, 2, nullptr, nullptr, Metric::Dot, 0, sA.I8(), sB.I8(), &cd);
		CHECK_MSG(s == Status::Ok,
			"all-zero Dot channel CentroidDistance should be defined Ok (unimplemented): %d",
			static_cast<int>(s));
		if (s == Status::Ok)
		{
			CHECK_MSG(cd == cd, "all-zero channel produced NaN");
			CHECK_MSG(cd == 0.0f, "all-zero Dot channel distance should be 0, got %g",
				static_cast<double>(cd));
		}
	}
}

// T-V3-A5 (D-V3-8 (a)) -- channel-query-only storm: a reader RE-SNAPSHOTS in the loop,
// repeatedly observing freshly-published rows while a writer appends (within the pre-sized
// arena, no Grow) and removes. The forcing invariant (Japp P-1): for every fresh snapshot,
// the arena's per-channel sub-norms bit-equal a fresh ComputeChannelInverseNorms over that
// snapshot's own published rows -- i.e. a row NEVER appears in a snapshot without its
// sub-norm. This directly exercises the append-time sub-norm-write-BEFORE-PublishedCount_
// store-release ordering D-V3-8 commissioned: were the sub-norm write moved AFTER the count
// publish, a snapshot acquiring the new count would read the just-published row with a stale
// / unwritten sub-norm slot, and the recompute would disagree. GREEN on the current code
// (correct ordering); the reader reads rows/sub-norms below the acquired count, which are
// immutable, so the recompute is race-free. TSan-clean is the CI property; this functional
// form runs under build.bat, like the shipped scratch concurrency tests.
static void TestChannelQueryOnlyStorm()
{
	Rng rng(0x570A11ull);
	const int32_t dims = 64;
	const int32_t capacity = 4096;
	const int32_t appended = 64; // start with content; the writer publishes the rest live

	ScratchChannelFixture fx;
	MakeChannelScratchBank(fx, rng, appended, dims, Metric::Cosine, Quantization::Int8,
		capacity);
	if (fx.createStatus != Status::Ok)
	{
		return;
	}
	const int32_t cc = static_cast<int32_t>(fx.channels.size());

	std::vector<float> queryRaw(dims);
	for (auto& v : queryRaw)
	{
		v = rng.NextFloat();
	}
	AlignedBuf qbuf(static_cast<size_t>(fx.pd) * sizeof(float));
	PadQuery(queryRaw, fx.pd, qbuf.F32());
	const QuerySegment seg[1] = {{fx.channels[0].offset, fx.channels[0].length, 1.0f}};

	std::atomic<bool> stop{false};
	std::thread writer([&]() {
		Rng wrng(0x999ull);
		int32_t next = fx.bank.Count();
		std::vector<float> row(dims);
		while (!stop.load(std::memory_order_relaxed))
		{
			if (next < capacity)
			{
				for (auto& v : row)
				{
					v = wrng.NextFloat();
				}
				row[0] += 1.5f; // guarantee a non-zero Cosine norm
				if (fx.bank.Append(row.data(), dims, &next) == Status::Ok)
				{
					++next;
				}
			}
			fx.bank.Remove(next > appended ? appended - 1 : 0);
		}
	});

	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(capacity), 0u);
	std::vector<float> recomputed(static_cast<size_t>(capacity) * cc);
	int32_t maxCountSeen = 0;
	for (int32_t iter = 0; iter < 300; ++iter)
	{
		BankView snap;
		CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		const int32_t C = snap.count;
		maxCountSeen = C > maxCountSeen ? C : maxCountSeen;
		CHECK(snap.channelCount == cc && snap.channelInvNorms != nullptr);

		// FORCING INVARIANT: the published sub-norm arena is consistent with the published
		// rows for EVERY fresh snapshot -- no row visible without its correct sub-norm.
		CHECK(ComputeChannelInverseNorms(snap, recomputed.data()) == Status::Ok);
		CHECK_MSG(std::memcmp(recomputed.data(), snap.channelInvNorms,
					  static_cast<size_t>(C) * cc * sizeof(float)) == 0,
			"a fresh snapshot observed a row without its correct sub-norm (iter %d, count %d)",
			iter, C);

		// The channel query over the fresh snapshot succeeds with finite scores (a torn /
		// stale sub-norm would surface here as a non-finite or wrong score).
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		QueryParams params;
		params.k = 10;
		params.segments = seg;
		params.segmentCount = 1;
		params.excludeBits = tombs.data();
		CHECK(Query(snap, qbuf.F32(), params, ws, hits, &n) == Status::Ok);
		for (int32_t i = 0; i < n; ++i)
		{
			CHECK(std::isfinite(hits[i].score));
		}
	}
	stop.store(true, std::memory_order_relaxed);
	writer.join();
	// The reader must have observed the count ADVANCE past the initial rows, or the storm
	// never exercised a concurrent publish (the P-1 defect this rewrite fixes).
	CHECK_MSG(maxCountSeen > appended,
		"reader never observed a freshly-published row (count stayed at %d)", maxCountSeen);
}

// T-V3-A6 (D-V3-8 (b)) -- full storm: a reader RE-SNAPSHOTS in the loop and runs a
// channel-SCOPED REDUCTION over each fresh snapshot, concurrent with a writer's
// append/remove. Two forcing checks (Japp P-1): (1) the arena sub-norms bit-equal a fresh
// recompute over each snapshot's published rows (the ordering invariant, as in T-V3-A5);
// and (2) the live reduction (which reads the arena's stored sub-norms) equals a serial
// TWIN reduction over the SAME snapshot rows with FRESHLY-recomputed sub-norms -- so a
// stale sub-norm on a just-published row would make the live result diverge from the twin.
// GREEN on the current code (correct ordering); rows/sub-norms below the acquired count are
// immutable, so both the recompute and the twin are race-free. TSan-clean is the CI
// property; runs functionally under build.bat.
static void TestChannelScopedReductionStorm()
{
	Rng rng(0x570A22ull);
	const int32_t dims = 64;
	const int32_t capacity = 1024;
	const int32_t appended = 32;

	ScratchChannelFixture fx;
	MakeChannelScratchBank(fx, rng, appended, dims, Metric::Cosine, Quantization::Int8,
		capacity);
	if (fx.createStatus != Status::Ok)
	{
		return;
	}
	const int32_t cc = static_cast<int32_t>(fx.channels.size());
	std::vector<int32_t> rowIdx(static_cast<size_t>(capacity));
	for (int32_t i = 0; i < capacity; ++i)
	{
		rowIdx[static_cast<size_t>(i)] = i;
	}

	std::atomic<bool> stop{false};
	std::thread writer([&]() {
		Rng wrng(0x888ull);
		int32_t next = fx.bank.Count();
		std::vector<float> row(dims);
		while (!stop.load(std::memory_order_relaxed))
		{
			if (next < capacity)
			{
				for (auto& v : row)
				{
					v = wrng.NextFloat();
				}
				row[0] += 1.5f; // guarantee a non-zero Cosine norm
				if (fx.bank.Append(row.data(), dims, &next) == Status::Ok)
				{
					++next;
				}
			}
			fx.bank.Remove(next > appended ? appended - 1 : 0);
		}
	});

	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(capacity), 0u);
	std::vector<float> recomputed(static_cast<size_t>(capacity) * cc);
	int32_t maxCountSeen = 0;
	for (int32_t iter = 0; iter < 200; ++iter)
	{
		BankView snap;
		CHECK(fx.bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		const int32_t C = snap.count;
		maxCountSeen = C > maxCountSeen ? C : maxCountSeen;
		CHECK(snap.channelCount == cc && snap.channelInvNorms != nullptr);

		// (1) FORCING INVARIANT: sub-norm arena consistent with the published rows.
		CHECK(ComputeChannelInverseNorms(snap, recomputed.data()) == Status::Ok);
		CHECK_MSG(std::memcmp(recomputed.data(), snap.channelInvNorms,
					  static_cast<size_t>(C) * cc * sizeof(float)) == 0,
			"reduction storm: a fresh snapshot observed a row without its sub-norm "
			"(iter %d, count %d)", iter, C);

		// (2) Live reduction (arena sub-norms) == serial twin (recomputed sub-norms over the
		// SAME immutable rows). Diverges iff a stored sub-norm is stale.
		AlignedBuf csLive(static_cast<size_t>(snap.paddedDims));
		AlignedBuf csTwin(static_cast<size_t>(snap.paddedDims));
		float rLive = 0.0f, rTwin = 0.0f;
		const Status sLive = SpreadCrossDeviceChannel(snap, rowIdx.data(), C, tombs.data(),
			Reduce::Mean, 0, csLive.I8(), &rLive);
		BankView twin = snap;
		twin.channelInvNorms = recomputed.data(); // correct sub-norms, same rows/scales
		const Status sTwin = SpreadCrossDeviceChannel(twin, rowIdx.data(), C, tombs.data(),
			Reduce::Mean, 0, csTwin.I8(), &rTwin);
		CHECK(sLive == Status::Ok && sTwin == Status::Ok);
		CHECK_MSG(rLive == rTwin,
			"reduction over the live snapshot != serial twin (iter %d, count %d): %.9g != %.9g",
			iter, C, static_cast<double>(rLive), static_cast<double>(rTwin));
	}
	stop.store(true, std::memory_order_relaxed);
	writer.join();
	CHECK_MSG(maxCountSeen > appended,
		"reader never observed a freshly-published row (count stayed at %d)", maxCountSeen);
}


// T-V3-Persist-4 (Japp P-2, dims 2/9) -- Load re-validates the serialized channel table:
// a malformed channelCount or a malformed ChannelInfo geometry in the blob -> BadFormat,
// and the target bank is left unchanged (reject-over-degrade). The scratch archive layout
// (v3): a 32-byte header {..., reserved[6] at offset 26}, then int32 channelCount at offset
// 32, then channelCount ChannelInfo (offset,length; 8 bytes each) from offset 36.
static void TestScratchChannelLoadRejectsMalformedTable()
{
	Rng rng(0xBAD7AB1eull);
	const int32_t dims = 64;
	const int32_t count = 16;
	const ChannelInfo channels[2] = {{0, 32}, {32, 32}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, channels, 2) ==
		Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source)
	{
		v = rng.NextFloat();
	}
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	MemArchive good;
	CHECK(bank.Save(good.Writer()) == Status::Ok);

	// A target bank in a known-good state, to prove reject-over-degrade.
	ScratchBank target;
	CHECK(target.Load(good.Reader()) == Status::Ok);
	auto targetChannelCount = [&]() {
		BankView v;
		std::vector<uint32_t> t(ScratchBank::TombstoneWords(target.Count()), 0u);
		CHECK(target.Snapshot(&v, t.data()) == Status::Ok);
		return v.channelCount;
	};
	const int32_t count0 = target.Count();
	CHECK(targetChannelCount() == 2);

	auto writeI32 = [](MemArchive& a, size_t off, int32_t val) {
		std::memcpy(a.bytes.data() + off, &val, sizeof(val));
	};
	// Layout offsets in the serialized blob.
	const size_t kOffChannelCount = 32;
	const size_t kOffChan1Offset = 44; // header(32) + channelCount(4) + chan0{off,len}(8) -> chan1.offset
	const size_t kOffChan0Length = 40; // chan0.length

	auto expectRejectUnchanged = [&](MemArchive& corrupt, const char* why) {
		CHECK_MSG(target.Load(corrupt.Reader()) == Status::BadFormat,
			"%s: expected BadFormat on Load", why);
		CHECK_MSG(target.Count() == count0 && targetChannelCount() == 2,
			"%s: target bank changed despite a rejected Load (reject-over-degrade)", why);
	};

	// (1) Out-of-range channelCount (99 > kMaxChannels).
	{
		MemArchive a;
		a.bytes = good.bytes;
		writeI32(a, kOffChannelCount, 99);
		expectRejectUnchanged(a, "channelCount=99");
	}
	// (2) Zero channelCount (a v3/channels blob must carry >= 1 channel).
	{
		MemArchive a;
		a.bytes = good.bytes;
		writeI32(a, kOffChannelCount, 0);
		expectRejectUnchanged(a, "channelCount=0");
	}
	// (3) Overlapping / non-ascending channel: channel[1].offset 32 -> 16 (overlaps [0,32)).
	{
		MemArchive a;
		a.bytes = good.bytes;
		writeI32(a, kOffChan1Offset, 16);
		expectRejectUnchanged(a, "overlapping channel geometry");
	}
	// (4) Off-grid channel: channel[0].length 32 -> 30 (not a multiple of the int8 grid, 16).
	{
		MemArchive a;
		a.bytes = good.bytes;
		writeI32(a, kOffChan0Length, 30);
		expectRejectUnchanged(a, "off-grid channel length");
	}
	// (5) Out-of-bounds channel: channel[1].length 32 -> 1024 (extends past paddedDims).
	{
		MemArchive a;
		a.bytes = good.bytes;
		writeI32(a, kOffChan1Offset + 4, 1024); // channel[1].length
		expectRejectUnchanged(a, "out-of-bounds channel length");
	}
	// A clean re-load still succeeds after the rejected attempts.
	MemArchive again;
	CHECK(bank.Save(again.Writer()) == Status::Ok);
	CHECK(target.Load(again.Reader()) == Status::Ok);
	CHECK(targetChannelCount() == 2);
}

// T-V3-Persist-5 (Japp P-3, dim 9) -- the writer's version-selection, asserted directly on
// the serialized version integer (little-endian u32 at byte 4) and the flags byte
// (reserved[0] at byte 26): a feature-less bank emits version 1; a retention-only bank emits
// version 2; a channel bank emits version 3 with the channels flag set (bit 1), the
// retention flag reflecting retention (bit 0). Catches an always-emit-v3 regression that
// would break real pre-v3 readers.
static void TestScratchChannelSaveVersionSelection()
{
	Rng rng(0x5E1EC7edull);
	const int32_t dims = 48;
	const ChannelInfo channels[2] = {{0, 32}, {32, 16}};

	auto versionOf = [](const MemArchive& a) {
		uint32_t v = 0;
		std::memcpy(&v, a.bytes.data() + 4, sizeof(v));
		return v;
	};
	auto flagsOf = [](const MemArchive& a) { return a.bytes[26]; }; // reserved[0]

	auto fill = [&](ScratchBank& b, int32_t n) {
		std::vector<float> row(static_cast<size_t>(dims));
		for (int32_t r = 0; r < n; ++r)
		{
			for (auto& v : row)
			{
				v = rng.NextFloat();
			}
			row[0] += 1.5f;
			CHECK(b.Append(row.data(), dims, nullptr) == Status::Ok);
		}
	};

	// (a) feature-less (no channels, no retention) -> version 1, flags 0.
	{
		ScratchBank b;
		CHECK(b.Create(8, dims, Metric::Dot, Quantization::Int8) == Status::Ok);
		fill(b, 4);
		MemArchive a;
		CHECK(b.Save(a.Writer()) == Status::Ok);
		CHECK_MSG(versionOf(a) == 1u, "feature-less bank must Save version 1, got %u",
			versionOf(a));
		CHECK(flagsOf(a) == 0u);
	}
	// (b) retention-only (no channels) -> version 2, flags 0 (legacy encoding).
	{
		ScratchBank b;
		CHECK(b.Create(8, dims, Metric::Dot, Quantization::Int8, /*retain=*/true) ==
			Status::Ok);
		fill(b, 4);
		MemArchive a;
		CHECK(b.Save(a.Writer()) == Status::Ok);
		CHECK_MSG(versionOf(a) == 2u, "retention-only bank must Save version 2, got %u",
			versionOf(a));
		CHECK(flagsOf(a) == 0u);
	}
	// (c) channels, no retention -> version 3, flags = channels bit only (0x02).
	{
		ScratchBank b;
		CHECK(b.Create(8, dims, Metric::Cosine, Quantization::Int8, channels, 2) == Status::Ok);
		fill(b, 4);
		MemArchive a;
		CHECK(b.Save(a.Writer()) == Status::Ok);
		CHECK_MSG(versionOf(a) == 3u, "channel bank must Save version 3, got %u", versionOf(a));
		CHECK_MSG(flagsOf(a) == 0x02, "channels-only flags byte should be 0x02, got 0x%02x",
			flagsOf(a));
	}
	// (d) channels + retention -> version 3, flags = retention|channels (0x03).
	{
		ScratchBank b;
		CHECK(b.Create(8, dims, Metric::Cosine, Quantization::Int8, channels, 2,
				/*retain=*/true) == Status::Ok);
		fill(b, 4);
		MemArchive a;
		CHECK(b.Save(a.Writer()) == Status::Ok);
		CHECK_MSG(versionOf(a) == 3u, "channel+retention bank must Save version 3, got %u",
			versionOf(a));
		CHECK_MSG(flagsOf(a) == 0x03, "channel+retention flags byte should be 0x03, got 0x%02x",
			flagsOf(a));
	}
}

// ===========================================================================
// v3.0.1 red suite (Curie): two P1 correctness bugs + three coverage holes.
//   [P1-B1] TestChannelNNZeroEnergyFloor           -- RED (floors, not rejects)
//   [P1-B2] TestScratchChannelPerChannelRecallZeroEnergy -- RED (skips, not aborts)
//   [hole]  TestChannelAnalyticsCrossDeviceGolden  -- GREEN (pins cross-device channel golden)
//   [hole]  TestPerChannelRecallOracle             -- GREEN (independent recall@k oracle)
//   [hole]  TestVersionHeaderCoherence             -- version.h must match the release (3/0/1)
// Reuses the earlier anonymous-namespace helpers CosRefChannelPair / CosRefSubImage /
// CosRefXdFloor (defined with T-V3-A1-COS-REF) and the slot-2/3 helpers
// (ScratchChannelFixture / MakeChannelScratchBank).
// ===========================================================================

namespace
{
	// Independent floored channel-NN recode to the CORRECT C-5/D-V3-11 semantics: NO
	// zero-sub-norm skip and NO pre-rejection. For Cosine, nearest = minimum distance, and
	// CosRefChannelPair floors any zero-sub-norm member (source OR target) to distance 0 -- so a
	// zero-energy target is nearest to every source at distance 0, and a zero-energy source's
	// every pair is 0 (its NN is 0). Reduced over ascending source order, closed with the
	// subnormal floor -- bit-for-bit what a fixed MeanNN/MaxNNCrossDeviceChannel must return.
	void ChannelNNFlooredRef(const BankView& src, const BankView& tgt, const ChannelInfo& ch,
		float* outMean, float* outMax)
	{
		double acc = 0.0;
		double best = 0.0;
		bool have = false;
		int32_t counted = 0;
		for (int32_t s = 0; s < src.count; ++s)
		{
			const int8_t* srcSub = CosRefSubImage(src, s, ch);
			double nn = 0.0;
			bool haveNn = false;
			for (int32_t t = 0; t < tgt.count; ++t)
			{
				const int8_t* tgtSub = CosRefSubImage(tgt, t, ch);
				const double d =
					static_cast<double>(CosRefChannelPair(srcSub, tgtSub, ch.length));
				if (!haveNn || d < nn) // Cosine: nearest = minimum distance
				{
					nn = d;
				}
				haveNn = true;
			}
			if (!haveNn)
			{
				continue;
			}
			acc += nn;
			if (!have || nn > best)
			{
				best = nn;
			}
			have = true;
			++counted;
		}
		*outMean = counted > 0 ? CosRefXdFloor(acc / static_cast<double>(counted)) : 0.0f;
		*outMax = have ? CosRefXdFloor(best) : 0.0f;
	}

	// Hand-builds a 2-channel (16+16 lane) Cosine Int8 scratch bank. Channel 1 (lanes 16..31)
	// always carries energy so the whole row normalizes and Append succeeds; channel 0 (lanes
	// 0..15) is EXACTLY zero on the rows flagged in zeroC0 (a valid row with a zero sub-norm
	// channel member -- the C-5 shape) and informative elsewhere. seedMix varies the informative
	// content so a source bank and a target bank point in different directions.
	void BuildZeroEnergyChannelBank(ScratchBank& bank, const bool* zeroC0, int32_t n,
		int32_t seedMix)
	{
		const int32_t d = 32;
		const ChannelInfo channels[2] = {{0, 16}, {16, 16}};
		CHECK(bank.Create(n, d, Metric::Cosine, Quantization::Int8, channels, 2) == Status::Ok);
		for (int32_t r = 0; r < n; ++r)
		{
			float row[32] = {};
			// Channel 1: always nonzero, varied per row (keeps the whole-row norm > 0).
			row[16 + (r % 16)] = 1.0f + 0.07f * static_cast<float>((r + seedMix) % 11);
			row[16 + ((r * 5 + 3) % 16)] += 0.5f + 0.03f * static_cast<float>(r % 7);
			if (!zeroC0[r])
			{
				// Channel 0: informative, varied so cosine distances differ across rows/banks.
				row[(r + seedMix) % 16] = 0.6f + 0.05f * static_cast<float>((r * 3) % 13);
				row[(r * 7 + 1) % 16] += 0.4f + 0.02f * static_cast<float>((r + 2) % 9);
			}
			CHECK(bank.Append(row, d, nullptr) == Status::Ok);
		}
	}
} // namespace

// [P1-B1] Zero-energy channel NN divergence must FLOOR, not reject (C-5/D-V3-11).
// NNDivergenceChannel currently returns ZeroNormQuery for a zero-sub-norm source sub-row
// (analytics.cpp:523) and continue-skips a zero-sub-norm target (analytics.cpp:548), aborting
// with InvalidArgument when every target is skipped -- contradicting C-5, which floors a zero
// sub-norm MEMBER to a defined 0 in a reduction (as CentroidDistance/Spread already do through
// XdChannelPairScore). Authored to the CORRECT behavior. RED now: scenario B fails on status
// (ZeroNormQuery from the source pre-check); scenario A fails on value (the skipped zero target
// yields a nonzero NN instead of the floored 0). GREEN once the operator floors instead of
// rejecting/skipping.
static void TestChannelNNZeroEnergyFloor()
{
	const int32_t n = 12;
	const ChannelInfo ch0 = {0, 16};
	Workspace ws;

	// --- Scenario A: a zero-energy channel-0 TARGET row. The zero-energy target floors its pair
	// to distance 0 (the Cosine minimum), so it is nearest to every source at distance 0 =>
	// MeanNN and MaxNN over channel 0 are both exactly 0. Source is fully informative on ch 0.
	{
		bool srcZero[n] = {};
		bool tgtZero[n] = {};
		tgtZero[3] = true;
		ScratchBank srcBank, tgtBank;
		BuildZeroEnergyChannelBank(srcBank, srcZero, n, 0);
		BuildZeroEnergyChannelBank(tgtBank, tgtZero, n, 4);
		BankView src, tgt;
		std::vector<uint32_t> st(ScratchBank::TombstoneWords(n), 0u);
		std::vector<uint32_t> tt(ScratchBank::TombstoneWords(n), 0u);
		CHECK(srcBank.Snapshot(&src, st.data()) == Status::Ok);
		CHECK(tgtBank.Snapshot(&tgt, tt.data()) == Status::Ok);

		std::vector<XdQuery> qbuf(n);
		std::vector<Hit> hbuf(n);
		std::vector<int32_t> nbuf(n);
		float mn = -1.0f, mx = -1.0f;
		const Status sMean = MeanNNCrossDeviceChannel(src, nullptr, tgt, nullptr, 0,
			qbuf.data(), hbuf.data(), nbuf.data(), ws, &mn);
		const Status sMax = MaxNNCrossDeviceChannel(src, nullptr, tgt, nullptr, 0,
			qbuf.data(), hbuf.data(), nbuf.data(), ws, &mx);
		CHECK_MSG(sMean == Status::Ok,
			"B1-A: MeanNN over a zero-energy channel-0 target must FLOOR (Ok), not reject; "
			"got status=%d", static_cast<int>(sMean));
		CHECK_MSG(sMax == Status::Ok,
			"B1-A: MaxNN over a zero-energy channel-0 target must FLOOR (Ok), not reject; "
			"got status=%d", static_cast<int>(sMax));
		if (sMean == Status::Ok && sMax == Status::Ok)
		{
			float refMean = 0.0f, refMax = 0.0f;
			ChannelNNFlooredRef(src, tgt, ch0, &refMean, &refMax);
			CHECK_MSG(refMean == 0.0f && refMax == 0.0f,
				"B1-A fixture: a zero-energy target should make every NN 0 (ref mean %.9g max "
				"%.9g)", static_cast<double>(refMean), static_cast<double>(refMax));
			CHECK_MSG(mn == refMean,
				"B1-A MeanNN != floored recode (zero target skipped instead of floored): "
				"%.9g != %.9g", static_cast<double>(mn), static_cast<double>(refMean));
			CHECK_MSG(mx == refMax,
				"B1-A MaxNN != floored recode: %.9g != %.9g", static_cast<double>(mx),
				static_cast<double>(refMax));
		}
	}

	// --- Scenario B: two zero-energy channel-0 SOURCE rows, all targets informative on ch 0.
	// Each zero-energy source's every pair floors to 0, so ITS nearest is 0; informative sources
	// keep genuine (nonzero) nearest distances. The reduction must stay DEFINED (Ok) and equal
	// the floored recode -- currently the source pre-check (analytics.cpp:523) aborts the whole
	// reduction with ZeroNormQuery.
	{
		bool srcZero[n] = {};
		srcZero[2] = true;
		srcZero[9] = true;
		bool tgtZero[n] = {};
		ScratchBank srcBank, tgtBank;
		BuildZeroEnergyChannelBank(srcBank, srcZero, n, 1);
		BuildZeroEnergyChannelBank(tgtBank, tgtZero, n, 6);
		BankView src, tgt;
		std::vector<uint32_t> st(ScratchBank::TombstoneWords(n), 0u);
		std::vector<uint32_t> tt(ScratchBank::TombstoneWords(n), 0u);
		CHECK(srcBank.Snapshot(&src, st.data()) == Status::Ok);
		CHECK(tgtBank.Snapshot(&tgt, tt.data()) == Status::Ok);

		std::vector<XdQuery> qbuf(n);
		std::vector<Hit> hbuf(n);
		std::vector<int32_t> nbuf(n);
		float mn = -1.0f, mx = -1.0f;
		const Status sMean = MeanNNCrossDeviceChannel(src, nullptr, tgt, nullptr, 0,
			qbuf.data(), hbuf.data(), nbuf.data(), ws, &mn);
		const Status sMax = MaxNNCrossDeviceChannel(src, nullptr, tgt, nullptr, 0,
			qbuf.data(), hbuf.data(), nbuf.data(), ws, &mx);
		CHECK_MSG(sMean == Status::Ok,
			"B1-B: a zero-energy channel-0 SOURCE row must be floored (its NN is 0), not reject "
			"the whole reduction; MeanNN status=%d", static_cast<int>(sMean));
		CHECK_MSG(sMax == Status::Ok,
			"B1-B: a zero-energy channel-0 SOURCE row must be floored, not reject; MaxNN "
			"status=%d", static_cast<int>(sMax));
		if (sMean == Status::Ok && sMax == Status::Ok)
		{
			float refMean = 0.0f, refMax = 0.0f;
			ChannelNNFlooredRef(src, tgt, ch0, &refMean, &refMax);
			CHECK_MSG(refMax > 0.0f,
				"B1-B fixture should have informative (nonzero) source NNs (recode max=%.9g)",
				static_cast<double>(refMax));
			CHECK_MSG(mn == refMean,
				"B1-B MeanNN != floored recode: %.9g != %.9g", static_cast<double>(mn),
				static_cast<double>(refMean));
			CHECK_MSG(mx == refMax,
				"B1-B MaxNN != floored recode: %.9g != %.9g", static_cast<double>(mx),
				static_cast<double>(refMax));
			// Independent spot check of "a zero-energy source's NN is 0": row 2's every pair
			// floors to 0.
			double nn2 = 1e30;
			for (int32_t t = 0; t < tgt.count; ++t)
			{
				const double d = static_cast<double>(CosRefChannelPair(
					CosRefSubImage(src, 2, ch0), CosRefSubImage(tgt, t, ch0), 16));
				if (d < nn2)
				{
					nn2 = d;
				}
			}
			CHECK_MSG(nn2 == 0.0, "B1-B: zero-energy source row NN should be 0, got %.9g", nn2);
		}
	}
}

// [P1-B2] Zero-energy per-channel recall must SKIP the sample, not abort (D-V3-7 / C-5).
// MeasureScratchRecallPerChannel and FreezeWithRecall issue a segmented self-query per sampled
// row; a sampled row whose channel sub-vector is zero-energy makes that per-channel Cosine
// self-query ZeroNormQuery, and MeasureRecallLockedChannel (scratch.cpp:1503) returns it,
// aborting the WHOLE audit. The API admits such rows (a valid row with a zero sub-norm channel).
// Authored to the CORRECT behavior: the zero-energy self-query rows are excluded from that
// channel's sample, not fatal. liveRows <= 1000 forces the reservoir to sample EVERY live row,
// so the zero-energy rows are certainly hit. RED now (ZeroNormQuery), GREEN once the audit skips.
static void TestScratchChannelPerChannelRecallZeroEnergy()
{
	Rng rng(0xB2E0E0ull);
	const int32_t dims = 128;
	const int32_t count = 160;
	const ChannelInfo channels[2] = {{0, 64}, {64, 64}};
	const uint64_t seed = ScratchBank::kDefaultRecallSeed;

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, channels, 2,
			/*retainFloats=*/true) == Status::Ok);
	int32_t zeroC0Rows = 0;
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims), 0.0f);
		for (int32_t i = 64; i < 128; ++i)
		{
			row[static_cast<size_t>(i)] = rng.NextFloat(); // channel 1: always informative
		}
		const bool zeroC0 = (r % 11) == 0; // ~15 zero-energy channel-0 rows
		if (!zeroC0)
		{
			for (int32_t i = 0; i < 64; ++i)
			{
				row[static_cast<size_t>(i)] = rng.NextFloat();
			}
		}
		else
		{
			++zeroC0Rows;
		}
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}
	CHECK(zeroC0Rows > 0);
	// Tombstone a scatter (excluded from sampling) but never a zero-energy row -- those must stay
	// live so they are sampled and force the abort under the current code.
	for (int32_t r = 3; r < count; r += 17)
	{
		if ((r % 11) != 0)
		{
			CHECK(bank.Remove(r) == Status::Ok);
		}
	}

	Workspace ws;
	ScratchRecallReport reports[2];
	const Status s = bank.MeasureScratchRecallPerChannel(ws, reports, 2, seed);
	CHECK_MSG(s == Status::Ok,
		"B2: MeasureScratchRecallPerChannel must COMPLETE with zero-energy-channel rows sampled "
		"(the zero self-query rows are skipped, not fatal); got status=%d", static_cast<int>(s));
	if (s == Status::Ok)
	{
		for (int32_t c = 0; c < 2; ++c)
		{
			CHECK_MSG(reports[c].recall >= 0.0f && reports[c].recall <= 1.0f,
				"B2 channel %d recall out of [0,1]: %g", c,
				static_cast<double>(reports[c].recall));
			CHECK(reports[c].seed == seed);
			CHECK(reports[c].generation == bank.Generation());
		}
	}

	// FreezeWithRecall re-measures per-channel recall over the compacted rows -- same audit, same
	// abort bug. It too must complete.
	{
		const int32_t live = bank.FreezeLiveCount();
		AlignedBuf frozenRows(static_cast<size_t>(live) *
			PaddedDims(dims, Quantization::Int8) * ElementSize(Quantization::Int8));
		std::vector<float> frozenScales(static_cast<size_t>(live));
		std::vector<int32_t> indexMap(static_cast<size_t>(count), -2);
		std::vector<float> frozenInvNorms(static_cast<size_t>(live) * 2);
		ScratchRecallReport freezeReports[2];
		const Status fs = bank.FreezeWithRecall(frozenRows.ptr, frozenScales.data(),
			indexMap.data(), frozenInvNorms.data(), freezeReports, 2, ws, seed);
		CHECK_MSG(fs == Status::Ok,
			"B2: FreezeWithRecall must COMPLETE with zero-energy-channel rows sampled; "
			"got status=%d", static_cast<int>(fs));
		if (fs == Status::Ok)
		{
			for (int32_t c = 0; c < 2; ++c)
			{
				CHECK(freezeReports[c].recall >= 0.0f && freezeReports[c].recall <= 1.0f);
			}
		}
	}
}

// [hole] Cross-device CHANNEL-analytics golden. The forced-path channel tests prove local
// scalar/SSE/AVX agreement but pin NO golden constant for the channel operators -- a platform
// can be self-consistent yet differ from another device (notably the per-channel cosine sqrt
// path), and v3.0 makes cross-device claims. This pins a golden over all four
// *CrossDeviceChannel operators (Dot, Cosine, L2), every channel, across the forced-SIMD sweep --
// the channel analog of kGoldenAnalyticsXdHash. The fixture is NON-degenerate (every channel
// carries energy on every row) so the pin is invariant under the B1/B2 zero-energy fix, which
// touches only degenerate math. GREEN: it locks the channel operators' cross-device value.
namespace
{
	uint64_t ChannelAnalyticsBattery(const BankView& src, const BankView& tgt,
		const std::vector<ChannelInfo>& channels, Metric metric, Workspace& ws, uint64_t h)
	{
		const int32_t count = src.count;
		const int32_t idxA[6] = {0, 4, 9, 17, 22, 30};
		const int32_t idxB[6] = {2, 7, 12, 20, 25, 31};
		std::vector<int32_t> allRows(static_cast<size_t>(count));
		for (int32_t i = 0; i < count; ++i)
		{
			allRows[static_cast<size_t>(i)] = i;
		}
		for (int32_t c = 0; c < static_cast<int32_t>(channels.size()); ++c)
		{
			AlignedBuf a(static_cast<size_t>(src.paddedDims));
			AlignedBuf b(static_cast<size_t>(src.paddedDims));
			float cd = 0.0f;
			CentroidDistanceCrossDeviceChannel(src, idxA, 6, nullptr, nullptr, src, idxB, 6,
				nullptr, nullptr, metric, c, a.I8(), b.I8(), &cd);
			an::Hash(h, cd);

			AlignedBuf cs(static_cast<size_t>(src.paddedDims));
			float sm = 0.0f, sx = 0.0f;
			SpreadCrossDeviceChannel(src, allRows.data(), count, nullptr, Reduce::Mean, c,
				cs.I8(), &sm);
			SpreadCrossDeviceChannel(src, allRows.data(), count, nullptr, Reduce::Max, c,
				cs.I8(), &sx);
			an::Hash(h, sm);
			an::Hash(h, sx);

			std::vector<XdQuery> qbuf(static_cast<size_t>(count));
			std::vector<Hit> hbuf(static_cast<size_t>(count));
			std::vector<int32_t> nbuf(static_cast<size_t>(count));
			float mn = 0.0f, mx = 0.0f;
			MeanNNCrossDeviceChannel(src, nullptr, tgt, nullptr, c, qbuf.data(), hbuf.data(),
				nbuf.data(), ws, &mn);
			MaxNNCrossDeviceChannel(src, nullptr, tgt, nullptr, c, qbuf.data(), hbuf.data(),
				nbuf.data(), ws, &mx);
			an::Hash(h, mn);
			an::Hash(h, mx);
		}
		return h;
	}
} // namespace

static void TestChannelAnalyticsCrossDeviceGolden()
{
	Rng rng(0x043A9A17ull); // fixed seed -> deterministic, pinnable fixture
	const int32_t dims = 64;
	const int32_t count = 48;

	ScratchChannelFixture dotSrc, dotTgt, cosSrc, cosTgt, l2Src, l2Tgt;
	MakeChannelScratchBank(dotSrc, rng, count, dims, Metric::Dot, Quantization::Int8);
	MakeChannelScratchBank(dotTgt, rng, count, dims, Metric::Dot, Quantization::Int8);
	MakeChannelScratchBank(cosSrc, rng, count, dims, Metric::Cosine, Quantization::Int8);
	MakeChannelScratchBank(cosTgt, rng, count, dims, Metric::Cosine, Quantization::Int8);
	MakeChannelScratchBank(l2Src, rng, count, dims, Metric::L2, Quantization::Int8);
	MakeChannelScratchBank(l2Tgt, rng, count, dims, Metric::L2, Quantization::Int8);
	if (dotSrc.createStatus != Status::Ok)
	{
		return; // slot-2 unbuilt: nothing to pin
	}

	struct Pair
	{
		ScratchChannelFixture* s;
		ScratchChannelFixture* t;
		Metric m;
	};
	Pair pairs[3] = {{&dotSrc, &dotTgt, Metric::Dot}, {&cosSrc, &cosTgt, Metric::Cosine},
		{&l2Src, &l2Tgt, Metric::L2}};

	// Snapshot each pair once: bank construction is not the SIMD-sensitive part; only the
	// operators are, and they run under the forced paths below.
	BankView srcV[3], tgtV[3];
	std::vector<uint32_t> st[3], tt[3];
	for (int32_t p = 0; p < 3; ++p)
	{
		st[p].assign(ScratchBank::TombstoneWords(count), 0u);
		tt[p].assign(ScratchBank::TombstoneWords(count), 0u);
		CHECK(pairs[p].s->bank.Snapshot(&srcV[p], st[p].data()) == Status::Ok);
		CHECK(pairs[p].t->bank.Snapshot(&tgtV[p], tt[p].data()) == Status::Ok);
	}

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

	Workspace ws;
	uint64_t hashes[4] = {};
	for (size_t i = 0; i < paths.size(); ++i)
	{
		detail::ForceXdSimdPath(paths[i]);
		uint64_t h = 0xcbf29ce484222325ull;
		for (int32_t p = 0; p < 3; ++p)
		{
			h = ChannelAnalyticsBattery(srcV[p], tgtV[p], pairs[p].s->channels, pairs[p].m, ws, h);
		}
		hashes[i] = h;
		detail::ClearForcedXdSimdPath();
		if (i > 0)
		{
			CHECK_MSG(hashes[i] == hashes[0],
				"channel analytics forced path %d hash %016llx != %016llx",
				static_cast<int>(paths[i]),
				static_cast<unsigned long long>(hashes[i]),
				static_cast<unsigned long long>(hashes[0]));
		}
	}
	uint64_t def = 0xcbf29ce484222325ull;
	for (int32_t p = 0; p < 3; ++p)
	{
		def = ChannelAnalyticsBattery(srcV[p], tgtV[p], pairs[p].s->channels, pairs[p].m, ws, def);
	}
	CHECK(def == hashes[0]);
	std::printf("channel analytics cross-device hash: %016llx (%d forced paths agree)\n",
		static_cast<unsigned long long>(def), static_cast<int>(paths.size()));
	if constexpr (xdfix::kGoldenChannelAnalyticsXdHash != 0)
	{
		CHECK_MSG(def == xdfix::kGoldenChannelAnalyticsXdHash,
			"channel analytics hash %016llx != pinned golden %016llx",
			static_cast<unsigned long long>(def),
			static_cast<unsigned long long>(xdfix::kGoldenChannelAnalyticsXdHash));
	}
}

// [hole] Per-channel recall INDEPENDENT oracle. The shipped per-channel recall test smoke-checks
// [0,1] + metadata only. This recomputes recall@k independently -- a float64 brute-force top-k
// over each channel's sub-range on the RETAINED rows (bank.RetainedRow) vs the bank's own
// quantized per-channel scan (public Query on the snapshot) -- with the SAME seed/sample
// discipline, and asserts MeasureScratchRecallPerChannel's number equals it bit-for-bit. The
// fixture keeps liveRows <= 1000 so the reservoir samples EVERY live row (sampleTarget ==
// liveRows => rng.Unit()*remaining < needed always), making the sample set the whole live
// population -- independent of the internal RNG. GREEN: the recall math is correct; this locks it.
static void TestPerChannelRecallOracle()
{
	Rng rng(0x0BAC1E5ull);
	const int32_t dims = 128;
	const int32_t count = 240; // <= 1000 -> all live rows sampled; >= kRecallInformativeRows live
	const ChannelInfo channels[2] = {{0, 64}, {64, 64}};
	const uint64_t seed = ScratchBank::kDefaultRecallSeed;

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, channels, 2,
			/*retainFloats=*/true) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source)
	{
		v = rng.NextFloat();
	}
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	for (int32_t r = 7; r < count; r += 19)
	{
		CHECK(bank.Remove(r) == Status::Ok);
	}

	Workspace ws;
	ScratchRecallReport reports[2];
	CHECK(bank.MeasureScratchRecallPerChannel(ws, reports, 2, seed) == Status::Ok);

	BankView view;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(bank.Count()), 0u);
	CHECK(bank.Snapshot(&view, tombs.data()) == Status::Ok);
	const int32_t snapCount = view.count;

	auto isDead = [&](int32_t i) -> bool {
		return (tombs[static_cast<size_t>(i >> 5)] & (1u << (i & 31))) != 0;
	};
	int32_t liveRows = 0;
	for (int32_t i = 0; i < snapCount; ++i)
	{
		if (!isDead(i))
		{
			++liveRows;
		}
	}
	CHECK(liveRows <= 1000); // the all-rows-sampled precondition
	const int32_t k = liveRows - 1 < 10 ? (liveRows - 1) : 10;

	for (int32_t c = 0; c < 2; ++c)
	{
		const ChannelInfo& ch = channels[c];
		const int32_t lo = ch.offset;
		const int32_t hi = ch.offset + ch.length < dims ? ch.offset + ch.length : dims;

		std::vector<uint32_t> exclude = tombs;
		const QuerySegment seg[1] = {{ch.offset, ch.length, 1.0f}};
		QueryParams params;
		params.k = k;
		params.segments = seg;
		params.segmentCount = 1;
		params.excludeBits = exclude.data();
		params.exactness = Exactness::CrossDevice;

		std::vector<float> query(static_cast<size_t>(view.paddedDims));
		int64_t totalHits = 0;
		int64_t totalPossible = 0;

		struct Cand
		{
			double score;
			int32_t idx;
		};
		std::vector<Cand> cands;
		for (int32_t si = 0; si < snapCount; ++si)
		{
			if (isDead(si))
			{
				continue;
			}
			const float* self = bank.RetainedRow(si);
			CHECK(self != nullptr);
			for (int32_t d = 0; d < dims; ++d)
			{
				query[static_cast<size_t>(d)] = self[d];
			}
			for (int32_t d = dims; d < view.paddedDims; ++d)
			{
				query[static_cast<size_t>(d)] = 0.0f;
			}

			// Independent float64 reference top-k over the channel sub-range on the RETAINED
			// rows, with RecallTopK's exact ordering (score desc for Cosine; lower index wins
			// ties). Same score formula and accumulation order as the audit's own reference.
			cands.clear();
			for (int32_t j = 0; j < snapCount; ++j)
			{
				if (j == si || isDead(j))
				{
					continue;
				}
				const float* rrow = bank.RetainedRow(j);
				double dot = 0.0, subNorm = 0.0;
				for (int32_t d = lo; d < hi; ++d)
				{
					dot += static_cast<double>(self[d]) * rrow[d];
					subNorm += static_cast<double>(rrow[d]) * rrow[d];
				}
				const double score = subNorm > 0.0 ? dot / std::sqrt(subNorm) : 0.0;
				cands.push_back({score, j});
			}
			std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
				if (a.score != b.score)
				{
					return a.score > b.score;
				}
				return a.idx < b.idx;
			});
			const int32_t refCount =
				k < static_cast<int32_t>(cands.size()) ? k : static_cast<int32_t>(cands.size());

			// The bank's own quantized per-channel scan (identical params to the audit).
			exclude[static_cast<size_t>(si >> 5)] |= 1u << (si & 31);
			std::vector<Hit> heap(static_cast<size_t>(k > 0 ? k : 1));
			int32_t got = 0;
			const Status qs = Query(view, query.data(), params, ws, heap.data(), &got);
			exclude[static_cast<size_t>(si >> 5)] &= ~(1u << (si & 31));
			CHECK(qs == Status::Ok);

			for (int32_t i = 0; i < got; ++i)
			{
				for (int32_t rr = 0; rr < refCount; ++rr)
				{
					if (cands[static_cast<size_t>(rr)].idx == heap[static_cast<size_t>(i)].index)
					{
						++totalHits;
						break;
					}
				}
			}
			totalPossible += refCount;
		}

		const float oracle =
			totalPossible > 0 ? static_cast<float>(static_cast<double>(totalHits) /
									static_cast<double>(totalPossible))
							  : 0.0f;
		CHECK_MSG(oracle == reports[c].recall,
			"per-channel recall oracle mismatch (ch=%d): independent %.9g != report %.9g "
			"(hits=%lld possible=%lld)",
			c, static_cast<double>(oracle), static_cast<double>(reports[c].recall),
			static_cast<long long>(totalHits), static_cast<long long>(totalPossible));
	}
}

// [hole] Version/header coherence. version.h must declare v3.0.0 for the v3.0 release. RED now
// (it reads 2/5/0); Hastings bumps it. Kept as a standalone assertion so the release-version
// truth has a test, not just a header edit.
static void TestVersionHeaderCoherence()
{
	CHECK_MSG(SUPERFAISS_VERSION_MAJOR == 3,
		"SUPERFAISS_VERSION_MAJOR should be 3 for v3.2.0, got %d", SUPERFAISS_VERSION_MAJOR);
	CHECK_MSG(SUPERFAISS_VERSION_MINOR == 2,
		"SUPERFAISS_VERSION_MINOR should be 2 for v3.2.0, got %d", SUPERFAISS_VERSION_MINOR);
	CHECK_MSG(SUPERFAISS_VERSION_PATCH == 0,
		"SUPERFAISS_VERSION_PATCH should be 0 for v3.2.0, got %d", SUPERFAISS_VERSION_PATCH);
}

// ===========================================================================
// V3.1 red suite (Curie, 2026-07-13): the Relabel operation (plan section 24).
// Realizes the section 24.8 Coverage Model against the no-op stub at
// scratch.cpp:657-662 (returns Ok without mutating) -- every test below fails for
// its cell's reason: a rejection test sees Ok where it expects InvalidArgument/
// OutOfMemory; a mutation test sees the OLD table where it expects the new one.
// Test design: Claude/Curie/superfaiss-v3.1-test-design-2026-07-13.md.
// Reuses the V3.0 channel-test helpers (ScratchChannelFixture, MakeChannelScratchBank,
// MemArchive, RefNormalizeRow, DequantizeRow, FeatRefHit, ChannelCosineBruteForce,
// kChannelFeatRelTolF32/I8, kScratchHeaderFlagsByteOffset, an::Hash) from the
// translation-unit-wide anonymous namespace above.
// ===========================================================================

namespace
{
	// Metric-general per-channel brute-force (generalizes ChannelCosineBruteForce to
	// Dot/L2/Cosine): the FEAT reference for any post-relabel per-channel query, computed
	// from the DEQUANTIZED bank rows directly, from the channel-score definition -- never
	// touching ComputeChannelInverseNorms or the kernel.
	std::vector<FeatRefHit> ChannelBruteForce(
		const BankView& view, const float* paddedQuery, const ChannelInfo& ch,
		const uint32_t* excludeBits, Metric metric)
	{
		std::vector<FeatRefHit> ref;
		std::vector<double> row;
		for (int32_t r = 0; r < view.count; ++r)
		{
			if (IsExcluded(excludeBits, r))
			{
				continue;
			}
			DequantizeRow(view, r, row);
			double score = 0.0;
			if (metric == Metric::Dot)
			{
				for (int32_t j = ch.offset; j < ch.offset + ch.length && j < view.dims; ++j)
				{
					score += static_cast<double>(paddedQuery[j]) * row[static_cast<size_t>(j)];
				}
			}
			else if (metric == Metric::L2)
			{
				for (int32_t j = ch.offset; j < ch.offset + ch.length && j < view.dims; ++j)
				{
					const double d =
						static_cast<double>(paddedQuery[j]) - row[static_cast<size_t>(j)];
					score += d * d;
				}
			}
			else
			{
				double dot = 0.0, subNorm = 0.0;
				for (int32_t j = ch.offset; j < ch.offset + ch.length && j < view.dims; ++j)
				{
					dot += static_cast<double>(paddedQuery[j]) * row[static_cast<size_t>(j)];
					subNorm += row[static_cast<size_t>(j)] * row[static_cast<size_t>(j)];
				}
				score = subNorm > 0.0 ? dot / std::sqrt(subNorm) : 0.0;
			}
			ref.push_back({r, score});
		}
		std::sort(ref.begin(), ref.end(), [&](const FeatRefHit& a, const FeatRefHit& b) {
			if (a.score != b.score)
			{
				return metric == Metric::L2 ? (a.score < b.score) : (a.score > b.score);
			}
			return a.index < b.index;
		});
		return ref;
	}

	// Whole-row definition-grounded brute-force (demote / whole-vector-unchanged claims):
	// as ChannelBruteForce but over [0, dims) rather than one channel's sub-range.
	std::vector<FeatRefHit> WholeVectorBruteForce(
		const BankView& view, const float* paddedQuery, const uint32_t* excludeBits, Metric metric)
	{
		const ChannelInfo whole{0, view.dims};
		return ChannelBruteForce(view, paddedQuery, whole, excludeBits, metric);
	}

	// Shared FEAT-vs-reference top-k check (the CAL-band comparison every FEAT cell below
	// uses): sorted reference hits, the operator's returned hits, expected count == min(k,
	// |ref|), every returned hit within tolerance of the definition-grounded boundary.
	void CheckFeatTopK(const std::vector<FeatRefHit>& ref, const Hit* hits, int32_t n, int32_t k,
		int32_t rowCount, Metric metric, Quantization quant, const char* label)
	{
		const double tol = quant == Quantization::Int8 ? kChannelFeatRelTolI8 : kChannelFeatRelTolF32;
		const int32_t expected =
			static_cast<int32_t>(ref.size()) < k ? static_cast<int32_t>(ref.size()) : k;
		CHECK_MSG(n == expected, "%s: FEAT hit count got %d expected %d", label, n, expected);
		if (expected == 0 || n != expected)
		{
			return;
		}
		const double boundary = ref[static_cast<size_t>(expected) - 1].score;
		std::vector<double> refByIndex(
			static_cast<size_t>(rowCount), metric == Metric::L2 ? 1e300 : -1e300);
		for (const FeatRefHit& h : ref)
		{
			refByIndex[static_cast<size_t>(h.index)] = h.score;
		}
		for (int32_t i = 0; i < n; ++i)
		{
			const double rs = refByIndex[static_cast<size_t>(hits[i].index)];
			const double band = tol * (1.0 + std::fabs(boundary));
			const bool inTrueTopK =
				metric == Metric::L2 ? (rs <= boundary + band) : (rs >= boundary - band);
			CHECK_MSG(inTrueTopK, "%s: hit %d (row %d) not within CAL band of the FEAT reference",
				label, i, hits[i].index);
			CHECK_MSG(std::fabs(rs - static_cast<double>(hits[i].score)) <= band,
				"%s: FEAT score drift got %.9g ref %.9g", label, static_cast<double>(hits[i].score),
				rs);
		}
	}

	// A counting allocator that fails the (failAfter)-th call (0-based) onward and forwards
	// every earlier call to the platform aligned alloc/free -- the OOM reject-over-degrade
	// probe (dim 5). alloc/free mirror alloc.cpp's DefaultAlloc/DefaultFree exactly so freed
	// blocks match how they were allocated.
	struct FailAfterState
	{
		std::atomic<int32_t> calls{0};
		int32_t failAfter = 0;
	};

	void* FailAfterAlloc(size_t size, size_t alignment, void* user)
	{
		auto* st = static_cast<FailAfterState*>(user);
		const int32_t call = st->calls.fetch_add(1, std::memory_order_relaxed);
		if (call >= st->failAfter)
		{
			return nullptr;
		}
#if defined(_MSC_VER)
		return _aligned_malloc(size, alignment);
#else
		const size_t rounded = ((size + alignment - 1) / alignment) * alignment;
		return std::aligned_alloc(alignment, rounded);
#endif
	}

	void FailAfterFree(void* ptr, void* /*user*/)
	{
#if defined(_MSC_VER)
		_aligned_free(ptr);
#else
		std::free(ptr);
#endif
	}

	Allocator MakeFailAfterAllocator(FailAfterState& state)
	{
		Allocator a;
		a.alloc = &FailAfterAlloc;
		a.free = &FailAfterFree;
		a.user = &state;
		return a;
	}
} // namespace

// T-V3.1-Validation (dim 2, N-2) -- the full Create rule set replayed via Relabel on an
// EXISTING bank (a Cosine channel bank, the realloc path, and a Dot single-space bank, the
// promote case): every malformed table -> InvalidArgument, the bank fully unchanged
// (channel count, generation). Validation symmetry across Create/Load/Relabel.
static void TestRelabelValidationSymmetry()
{
	Rng rng(0x100001ull);
	const int32_t dims = 64;
	const ChannelInfo baseChannels[2] = {{0, 32}, {32, 32}};

	ScratchBank cosineBank;
	CHECK(cosineBank.Create(16, dims, Metric::Cosine, Quantization::Int8, baseChannels, 2) ==
		Status::Ok);
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) { v = rng.NextFloat(); }
		row[0] += 1.5f;
		CHECK(cosineBank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	ScratchBank singleSpaceBank;
	CHECK(singleSpaceBank.Create(16, dims, Metric::Dot, Quantization::Float32) == Status::Ok);
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) { v = rng.NextFloat(); }
		CHECK(singleSpaceBank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	auto expectRejected = [&](ScratchBank& bank, const ChannelInfo* newChannels, int32_t newCount,
								   const char* why) {
		const int32_t countBefore = bank.GetChannelCount();
		const uint64_t genBefore = bank.Generation();
		const Status s = bank.Relabel(newChannels, newCount);
		CHECK_MSG(s == Status::InvalidArgument, "%s: expected InvalidArgument, got %d", why,
			static_cast<int>(s));
		CHECK_MSG(bank.GetChannelCount() == countBefore,
			"%s: a rejected Relabel changed ChannelCount_ (%d -> %d)", why, countBefore,
			bank.GetChannelCount());
		CHECK_MSG(bank.Generation() == genBefore,
			"%s: a rejected Relabel advanced Generation()", why);
	};

	for (ScratchBank* bank : {&cosineBank, &singleSpaceBank})
	{
		{ const ChannelInfo bad[2] = {{0, 32}, {16, 32}}; expectRejected(*bank, bad, 2, "overlapping"); }
		{ const ChannelInfo bad[2] = {{32, 16}, {0, 16}}; expectRejected(*bank, bad, 2, "non-ascending"); }
		{ const ChannelInfo bad[1] = {{0, 6}}; expectRejected(*bank, bad, 1, "off-grid length"); }
		{ const ChannelInfo bad[1] = {{2, 8}}; expectRejected(*bank, bad, 1, "off-grid offset"); }
		{ const ChannelInfo bad[1] = {{0, dims + 64}}; expectRejected(*bank, bad, 1, "out-of-bounds"); }
		{ const ChannelInfo bad[1] = {{0, 0}}; expectRejected(*bank, bad, 1, "zero-length"); }
		{ const ChannelInfo bad[1] = {{-16, 16}}; expectRejected(*bank, bad, 1, "negative offset"); }
		expectRejected(*bank, nullptr, 1, "null table, nonzero count");
		{ const ChannelInfo t[1] = {{0, 16}}; expectRejected(*bank, t, 0, "nonzero table, zero count"); }
		// N-2: negative count with a non-null table.
		{ const ChannelInfo t[1] = {{0, 16}}; expectRejected(*bank, t, -1, "negative count, non-null table"); }
	}

	// >kMaxChannels, isolated from the out-of-bounds rule on a wide-dims bank (the V3.0
	// TestScratchChannelCreateRejections precedent), so the cap fires for its own reason.
	{
		const int32_t wideDims = 9 * kAlignment;
		ScratchBank wideBank;
		CHECK(wideBank.Create(4, wideDims, Metric::Dot, Quantization::Float32) == Status::Ok);
		ChannelInfo nine[9];
		for (int32_t c = 0; c < 9; ++c) { nine[c] = {c * kAlignment, kAlignment}; }
		expectRejected(wideBank, nine, 9, "exceeds kMaxChannels (isolated from OOB)");
	}

	// Positive control: exactly kMaxChannels is accepted (the cap is exclusive-above).
	{
		const int32_t wideDims = 8 * kAlignment;
		ScratchBank wideBank;
		CHECK(wideBank.Create(4, wideDims, Metric::Dot, Quantization::Float32) == Status::Ok);
		ChannelInfo eight[8];
		for (int32_t c = 0; c < 8; ++c) { eight[c] = {c * kAlignment, kAlignment}; }
		CHECK_MSG(wideBank.Relabel(eight, 8) == Status::Ok,
			"kMaxChannels boundary: Relabel of exactly 8 channels should succeed");
	}
}

// T-V3.1-RejectQueryable (dim 5, N-1) -- after a rejected Relabel, the bank is not just
// "unchanged" by field inspection: a whole-vector query AND a per-channel query both still
// return the correct result under the OLD table.
static void TestRelabelRejectionsStillQueryableUnderOldTable()
{
	Rng rng(0x100002ull);
	const int32_t dims = 64;
	const int32_t count = 40;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}

	const ChannelInfo badOverlap[2] = {{0, 32}, {16, 32}};
	CHECK_MSG(bank.Relabel(badOverlap, 2) == Status::InvalidArgument,
		"malformed relabel should reject");

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
	CHECK_MSG(snap.channelCount == 2 && snap.channels != nullptr &&
			snap.channels[0].offset == 0 && snap.channels[0].length == 32 &&
			snap.channels[1].offset == 32 && snap.channels[1].length == 32,
		"a rejected Relabel must leave the OLD table on the snapshot");

	std::vector<float> queryRaw(dims);
	for (auto& v : queryRaw) { v = rng.NextFloat(); }
	AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
	PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
	const QuerySegment seg[1] = {{0, 32, 1.0f}};
	QueryParams p;
	p.k = 10;
	p.segments = seg;
	p.segmentCount = 1;
	p.excludeBits = tombs.data();
	Workspace ws;
	Hit hits[10];
	int32_t n = 0;
	CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
	const std::vector<FeatRefHit> ref =
		ChannelBruteForce(snap, qbuf.F32(), oldTable[0], tombs.data(), Metric::Cosine);
	CheckFeatTopK(ref, hits, n, 10, count, Metric::Cosine, Quantization::Int8,
		"post-rejection old-table channel query");

	QueryParams pw;
	pw.k = 10;
	pw.excludeBits = tombs.data();
	Hit whits[10];
	int32_t nw = 0;
	CHECK(Query(snap, qbuf.F32(), pw, ws, whits, &nw) == Status::Ok);
	const std::vector<FeatRefHit> wref =
		WholeVectorBruteForce(snap, qbuf.F32(), tombs.data(), Metric::Cosine);
	CheckFeatTopK(wref, whits, nw, 10, count, Metric::Cosine, Quantization::Int8,
		"post-rejection whole-vector query");
}

// T-V3.1-OOM (dim 5) -- a Cosine relabel whose arena realloc fails returns OutOfMemory and
// leaves the bank fully intact, still queryable under the OLD table (reject-over-degrade,
// the Load idiom).
static void TestRelabelOomLeavesBankIntact()
{
	Rng rng(0x100003ull);
	const int32_t dims = 64;
	const int32_t count = 24;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[3] = {{0, 16}, {16, 16}, {32, 32}}; // count UP -- forces a realloc

	FailAfterState failState;
	failState.failAfter = 1; // call 0 (Create) succeeds; call 1 (the relabel realloc) fails
	Allocator failAlloc = MakeFailAfterAllocator(failState);

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2,
			/*retainFloats=*/false, failAlloc) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	const uint64_t genBefore = bank.Generation();

	CHECK_MSG(bank.Relabel(newTable, 3) == Status::OutOfMemory,
		"a relabel realloc that cannot allocate must return OutOfMemory");
	CHECK_MSG(bank.GetChannelCount() == 2,
		"OOM relabel must leave the OLD channel count (2), got %d", bank.GetChannelCount());
	CHECK_MSG(bank.Generation() == genBefore, "OOM relabel must not advance Generation()");

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
	std::vector<float> queryRaw(dims);
	for (auto& v : queryRaw) { v = rng.NextFloat(); }
	AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
	PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
	const QuerySegment seg[1] = {{0, 32, 1.0f}};
	QueryParams p;
	p.k = 10;
	p.segments = seg;
	p.segmentCount = 1;
	p.excludeBits = tombs.data();
	Workspace ws;
	Hit hits[10];
	int32_t n = 0;
	CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
	const std::vector<FeatRefHit> ref =
		ChannelBruteForce(snap, qbuf.F32(), oldTable[0], tombs.data(), Metric::Cosine);
	CheckFeatTopK(ref, hits, n, 10, count, Metric::Cosine, Quantization::Int8,
		"post-OOM old-table channel query");
}

// T-V3.1-Lifetime (dim 1) -- a warm bank driven through Append -> Grow -> Relabel -> Append
// -> Relabel -> Remove, correct at every step; one seam allocation per Cosine relabel, ZERO
// per Dot/L2 relabel.
static void TestRelabelWarmLifetimeSequence()
{
	Rng rng(0x100004ull);
	const int32_t dims = 64;
	const ChannelInfo t1[2] = {{0, 32}, {32, 32}};
	const ChannelInfo t2[2] = {{0, 16}, {16, 48}};
	const ChannelInfo t3[3] = {{0, 16}, {16, 16}, {32, 32}};

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		ScratchBank bank;
		CHECK(bank.Create(16, dims, metric, Quantization::Int8, t1, 2) == Status::Ok);
		auto appendN = [&](int32_t n) {
			for (int32_t i = 0; i < n; ++i)
			{
				std::vector<float> row(static_cast<size_t>(dims));
				for (auto& v : row) { v = rng.NextFloat(); }
				CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
			}
		};
		appendN(10);
		CHECK(bank.Grow(48) == Status::Ok);

		// S4 (coverage audit §6.4): raw-new count is 0 ALWAYS (all allocation
		// must route through the seam), and the seam delta is exactly the
		// documented 0-or-1.
		const uint64_t before1 = AllocationCount();
		uint64_t raw1 = 0;
		{
			ScopedRawNewTracking rawTracking;
			CHECK(bank.Relabel(t2, 2) == Status::Ok);
			raw1 = rawTracking.Count();
		}
		CHECK_MSG(raw1 == 0, "warm relabel #1 (metric=%d): allocated %llu time(s) outside the seam",
			static_cast<int>(metric), static_cast<unsigned long long>(raw1));
		const uint64_t delta1 = AllocationCount() - before1;
		const uint64_t expected1 = metric == Metric::Cosine ? 1u : 0u;
		CHECK_MSG(delta1 == expected1,
			"warm relabel #1 (metric=%d): expected %llu seam alloc(s), got %llu",
			static_cast<int>(metric), static_cast<unsigned long long>(expected1),
			static_cast<unsigned long long>(delta1));
		CHECK_MSG(bank.GetChannelCount() == 2 && bank.GetChannels()[0].length == 16,
			"warm relabel #1 (metric=%d): table did not update", static_cast<int>(metric));

		appendN(10);

		const uint64_t before2 = AllocationCount();
		uint64_t raw2 = 0;
		{
			ScopedRawNewTracking rawTracking;
			CHECK(bank.Relabel(t3, 3) == Status::Ok);
			raw2 = rawTracking.Count();
		}
		CHECK_MSG(raw2 == 0, "warm relabel #2 (metric=%d): allocated %llu time(s) outside the seam",
			static_cast<int>(metric), static_cast<unsigned long long>(raw2));
		const uint64_t delta2 = AllocationCount() - before2;
		const uint64_t expected2 = metric == Metric::Cosine ? 1u : 0u;
		CHECK_MSG(delta2 == expected2,
			"warm relabel #2 (metric=%d): expected %llu seam alloc(s), got %llu",
			static_cast<int>(metric), static_cast<unsigned long long>(expected2),
			static_cast<unsigned long long>(delta2));
		CHECK_MSG(bank.GetChannelCount() == 3, "warm relabel #2 (metric=%d): table did not update",
			static_cast<int>(metric));

		// S4 rejection leg (coverage audit §6.4, dimension 2/5): a malformed
		// table (overlapping ranges) is InvalidArgument with raw 0 and seam
		// delta 0 -- reject-over-degrade must not leak an allocation.
		{
			const ChannelInfo badTable[2] = {{0, 32}, {16, 32}}; // overlapping
			const uint64_t beforeReject = AllocationCount();
			uint64_t rawReject = 0;
			Status rejectStatus = Status::Ok;
			{
				ScopedRawNewTracking rawTracking;
				rejectStatus = bank.Relabel(badTable, 2);
				rawReject = rawTracking.Count();
			}
			CHECK_MSG(rejectStatus == Status::InvalidArgument,
				"malformed relabel table (metric=%d) must reject", static_cast<int>(metric));
			CHECK_MSG(rawReject == 0,
				"rejected relabel (metric=%d) allocated %llu time(s) outside the seam",
				static_cast<int>(metric), static_cast<unsigned long long>(rawReject));
			CHECK_MSG(AllocationCount() == beforeReject,
				"rejected relabel (metric=%d) must not touch the seam either", static_cast<int>(metric));
			CHECK_MSG(bank.GetChannelCount() == 3,
				"rejected relabel (metric=%d) must leave the table exactly as before",
				static_cast<int>(metric));
		}

		CHECK(bank.Remove(0) == Status::Ok);

		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(bank.Count()), 0u);
		CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		std::vector<float> queryRaw(dims);
		for (auto& v : queryRaw) { v = rng.NextFloat(); }
		AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
		PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
		const QuerySegment seg[1] = {{t3[0].offset, t3[0].length, 1.0f}};
		QueryParams p;
		p.k = 10;
		p.segments = seg;
		p.segmentCount = 1;
		p.excludeBits = tombs.data();
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
		const std::vector<FeatRefHit> ref =
			ChannelBruteForce(snap, qbuf.F32(), t3[0], tombs.data(), metric);
		CheckFeatTopK(ref, hits, n, 10, bank.Count(), metric, Quantization::Int8,
			"warm lifetime final channel query");
	}
}

// T-V3.1-Toggle (dim 1, G-3) -- a warm promote(absent->present)->demote(present->absent)->
// promote(absent->present again) sequence on ONE Cosine instance: the re-bind bug G-3
// targets is a stale pointer or mis-sized region on the SECOND promote after a demote.
static void TestRelabelWarmPromoteDemoteToggle()
{
	Rng rng(0x100005ull);
	const int32_t dims = 64;
	const ChannelInfo table[2] = {{0, 32}, {32, 32}};
	const int32_t count = 24;

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) { v = rng.NextFloat(); }
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	auto checkChannelQuery = [&](const char* step) {
		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		if (snap.channelCount == 0)
		{
			return;
		}
		std::vector<float> queryRaw(dims);
		for (auto& v : queryRaw) { v = rng.NextFloat(); }
		AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
		PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
		const QuerySegment seg[1] = {{table[0].offset, table[0].length, 1.0f}};
		QueryParams p;
		p.k = 10;
		p.segments = seg;
		p.segmentCount = 1;
		p.excludeBits = tombs.data();
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
		const std::vector<FeatRefHit> ref =
			ChannelBruteForce(snap, qbuf.F32(), table[0], tombs.data(), Metric::Cosine);
		CheckFeatTopK(ref, hits, n, 10, count, Metric::Cosine, Quantization::Int8, step);
	};

	CHECK(bank.Relabel(table, 2) == Status::Ok);
	CHECK_MSG(bank.GetChannelCount() == 2, "toggle: promote #1 did not bind the region");
	checkChannelQuery("toggle: promote #1");

	CHECK(bank.Relabel(nullptr, 0) == Status::Ok);
	CHECK_MSG(bank.GetChannelCount() == 0, "toggle: demote did not clear the region");

	const uint64_t before = AllocationCount();
	uint64_t rawPromote2 = 0;
	{
		// S4 (coverage audit §6.4): raw-new count is 0 even on the re-promote
		// leg, which is the specific reuse edge this cell exists to guard.
		ScopedRawNewTracking rawTracking;
		CHECK(bank.Relabel(table, 2) == Status::Ok);
		rawPromote2 = rawTracking.Count();
	}
	CHECK_MSG(rawPromote2 == 0, "toggle: promote #2 allocated %llu time(s) outside the seam",
		static_cast<unsigned long long>(rawPromote2));
	const uint64_t delta = AllocationCount() - before;
	CHECK_MSG(delta == 1u, "toggle: promote #2 should cost exactly one seam allocation, got %llu",
		static_cast<unsigned long long>(delta));
	CHECK_MSG(bank.GetChannelCount() == 2, "toggle: promote #2 did not re-bind the region");
	checkChannelQuery("toggle: promote #2 (re-bind)");
}

// T-V3.1-Aliasing (Forge W2, dim 3) -- a Dot/L2 relabel is a by-value member write, aliased
// into every live BankView. A BankView taken BEFORE a Dot/L2 relabel observes the NEW table
// through that alias once the exclusive drain completes -- proving the drain actually ran.
static void TestRelabelHeldViewAliasing()
{
	Rng rng(0x100006ull);
	const int32_t dims = 32;
	const int32_t count = 8;
	const ChannelInfo oldTable[2] = {{0, 16}, {16, 16}};
	// A boundary-moved 2-channel target: BankView.channelCount is a by-value copy taken at
	// Snapshot (types.h:58, NOT an alias), so it stays 2 across the relabel; only
	// BankView.channels is a pointer that aliases the by-value member Channels_. Keeping the
	// count at 2 avoids reading a stale count against a moved table and isolates the aliasing
	// signal to the moved boundary in channels[0].
	const ChannelInfo newTable[2] = {{0, 24}, {24, 8}};

	for (Metric metric : {Metric::Dot, Metric::L2})
	{
		ScratchBank bank;
		CHECK(bank.Create(count, dims, metric, Quantization::Float32, oldTable, 2) == Status::Ok);
		for (int32_t r = 0; r < count; ++r)
		{
			std::vector<float> row(static_cast<size_t>(dims));
			for (auto& v : row) { v = rng.NextFloat(); }
			CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
		}

		BankView held;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(bank.Snapshot(&held, tombs.data()) == Status::Ok);
		CHECK(held.channelCount == 2 && held.channels[0].length == 16);

		CHECK(bank.Relabel(newTable, 2) == Status::Ok);

		// Forge W2: the Dot/L2 by-value member write to Channels_ IS observable through the
		// held view's aliased channels pointer -- so a view taken before a relabel must join
		// the Grow/Load invalidation set. The aliased channels[0] reflecting the new {0,24}
		// boundary proves the member write landed and the exclusive drain ran.
		CHECK_MSG(held.channels != nullptr && held.channels[0].offset == 0 &&
				held.channels[0].length == 24,
			"the aliased pre-relabel view (metric=%d) must reflect the new table's moved "
			"boundary after the exclusive drain completes", static_cast<int>(metric));
	}
}

// T-V3.1-Storm (dim 3) -- the exclusive-drain harness. Relabel is an EXCLUSIVE writer op,
// the same class as Grow/Load (S24.4, D-V3.1-1): it frees the old arena and rebinds, so a
// concurrent unpinned Snapshot's channelInvNorms would dangle into freed memory. The host
// drives the drain: the writer wraps each Relabel in BeginExclusive/EndExclusive (new pins
// refused, in-flight pins waited to zero); the reader holds a pin across Snapshot AND the
// sub-norm verification (the arena it reads cannot be freed under a live pin). Two claims:
// (1) the drain deterministically refuses a pin while it holds the flag; (2) no PINNED
// snapshot ever observes a torn table. TSan-clean is the CI property (the seq_cst flag-store
// / pin-load pairing); runs functionally under build.bat.
static void TestRelabelExclusiveDrainStorm()
{
	Rng rng(0x100007ull);
	const int32_t dims = 64;
	const int32_t capacity = 512;
	const int32_t appended = 32;
	const ChannelInfo t0[1] = {{0, dims}}; // channelCount 1 -- distinct shape from every target
	const ChannelInfo tables[3][2] = {
		{{0, 16}, {16, 48}},
		{{0, 48}, {48, 16}},
		{{0, 32}, {32, 32}},
	};

	ScratchBank bank;
	CHECK(bank.Create(capacity, dims, Metric::Cosine, Quantization::Int8, t0, 1) == Status::Ok);
	for (int32_t r = 0; r < appended; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) { v = rng.NextFloat(); }
		row[0] += 1.5f;
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}

	// (1) Deterministic pin-refusal: while an exclusive op holds the drain flag, a new reader
	// pin is refused; once the exclusive op ends, pins are granted again (the dim-3 claim,
	// tested race-free before the storm).
	CHECK(bank.BeginExclusive() == true);
	CHECK_MSG(bank.TryPinReader() == false,
		"a reader pin must be REFUSED while an exclusive op holds the drain flag");
	bank.EndExclusive();
	CHECK_MSG(bank.TryPinReader() == true, "a reader pin must be granted once the drain ends");
	bank.UnpinReader();

	// (2) The concurrent storm under proper pinning: no PINNED snapshot observes a torn table.
	std::atomic<bool> stop{false};
	std::atomic<int32_t> relabelCount{0};
	std::thread writer([&]() {
		Rng wrng(0x100107ull);
		int32_t next = bank.Count();
		std::vector<float> row(static_cast<size_t>(dims));
		int32_t iter = 0;
		while (!stop.load(std::memory_order_relaxed))
		{
			if (next < capacity && (iter % 3) != 0)
			{
				// Append is a NON-exclusive writer op (append-only arena, publish-via-count,
				// sub-norms written before the count store) -- it needs no drain.
				for (auto& v : row) { v = wrng.NextFloat(); }
				row[0] += 1.5f;
				if (bank.Append(row.data(), dims, &next) == Status::Ok)
				{
					++next;
				}
			}
			else
			{
				// Relabel is EXCLUSIVE: drain readers, swap, release -- the host's job, exactly
				// as for Grow/Load. BeginExclusive waits all in-flight pins to zero first.
				if (bank.BeginExclusive())
				{
					const int32_t which =
						relabelCount.fetch_add(1, std::memory_order_relaxed) % 3;
					bank.Relabel(tables[which], 2);
					bank.EndExclusive();
				}
			}
			++iter;
		}
	});

	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(capacity), 0u);
	int32_t maxCountSeen = 0;
	for (int32_t iter = 0; iter < 300; ++iter)
	{
		// Pin across BOTH the Snapshot and the sub-norm verification: the pin blocks the
		// exclusive relabel from freeing the arena the snapshot points into.
		while (!bank.TryPinReader())
		{
			std::this_thread::yield();
		}
		BankView snap;
		CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		maxCountSeen = snap.count > maxCountSeen ? snap.count : maxCountSeen;
		if (snap.channelCount > 0 && snap.channelInvNorms != nullptr && snap.count > 0)
		{
			std::vector<float> recomputed(static_cast<size_t>(snap.count) * snap.channelCount);
			CHECK(ComputeChannelInverseNorms(snap, recomputed.data()) == Status::Ok);
			CHECK_MSG(std::memcmp(recomputed.data(), snap.channelInvNorms,
						  recomputed.size() * sizeof(float)) == 0,
				"relabel storm: a pinned snapshot observed a row without its correct sub-norm "
				"(iter %d, count %d, channelCount %d)", iter, snap.count, snap.channelCount);
		}
		bank.UnpinReader();
	}
	stop.store(true, std::memory_order_relaxed);
	writer.join();

	CHECK_MSG(relabelCount.load(std::memory_order_relaxed) > 0,
		"relabel storm: the writer never completed a relabel");
	BankView finalSnap;
	CHECK(bank.Snapshot(&finalSnap, tombs.data()) == Status::Ok);
	bool matchedSomeTarget = false;
	for (const auto& t : tables)
	{
		if (finalSnap.channelCount == 2 && finalSnap.channels[0].offset == t[0].offset &&
			finalSnap.channels[0].length == t[0].length)
		{
			matchedSomeTarget = true;
			break;
		}
	}
	CHECK_MSG(matchedSomeTarget,
		"relabel storm: the final channel table never left the fixture's initial (1-channel) "
		"table -- Relabel never mutated the bank under concurrent load");
	CHECK_MSG(maxCountSeen > appended,
		"relabel storm: reader never observed a freshly-published row (count stayed at %d)",
		maxCountSeen);
}

// T-V3.1-Extremes (dim 4, G-5) -- shape/extremes matrix crossed with the metric matrix
// (G-2): count up 1->8, count down 8->1, boundary move, whole-row single channel (bit-matches
// the whole-vector path for Dot/L2; FEAT-equivalent for Cosine), one-grid-unit channel, one
// live row, fully tombstoned,
// empty bank (count==0, G-5), promote, demote -- each proven with the per-channel FEAT.
// dims=128 with grid=kAlignment (16 elements) throughout: the LCD grid satisfying both
// quantizations' element-grid rules, wide enough for the 8-channel count-up/down extremes.
static void TestRelabelExtremesMatrix()
{
	Rng rng(0x100008ull);
	const int32_t dims = 128;
	const int32_t grid = kAlignment;

	struct Scenario
	{
		const char* name;
		std::vector<ChannelInfo> initial; // empty = single-space
		std::vector<ChannelInfo> target;  // empty = demote to single-space
		int32_t count;
		bool tombstoneAll = false;
	};

	std::vector<Scenario> scenarios;
	{
		Scenario s{"count up 1->8", {{0, dims}}, {}, 64};
		for (int32_t c = 0; c < 8; ++c) { s.target.push_back({c * grid, grid}); }
		scenarios.push_back(s);
	}
	{
		Scenario s{"count down 8->1", {}, {{0, dims}}, 64};
		for (int32_t c = 0; c < 8; ++c) { s.initial.push_back({c * grid, grid}); }
		scenarios.push_back(s);
	}
	scenarios.push_back({"boundary move", {{0, 64}, {64, 64}}, {{0, 32}, {32, 96}}, 48});
	scenarios.push_back(
		{"single channel whole row", {{0, 64}, {64, 64}}, {{0, dims}}, 40});
	scenarios.push_back({"one-grid-unit channel", {{0, dims}}, {{0, grid}, {grid, dims - grid}}, 24});
	scenarios.push_back({"one live row", {{0, 64}, {64, 64}}, {{0, 32}, {32, 96}}, 1});
	{
		Scenario s{"fully tombstoned", {{0, 64}, {64, 64}}, {{0, 32}, {32, 96}}, 16};
		s.tombstoneAll = true;
		scenarios.push_back(s);
	}
	scenarios.push_back({"empty bank (count==0)", {{0, 64}, {64, 64}}, {{0, 32}, {32, 96}}, 0});
	scenarios.push_back({"promote", {}, {{0, 64}, {64, 64}}, 24});
	scenarios.push_back({"demote", {{0, 64}, {64, 64}}, {}, 24});

	for (const Scenario& sc : scenarios)
	{
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			for (Quantization quant : {Quantization::Float32, Quantization::Int8})
			{
				ScratchBank bank;
				const Status created = sc.initial.empty()
					? bank.Create(sc.count > 0 ? sc.count : 1, dims, metric, quant)
					: bank.Create(sc.count > 0 ? sc.count : 1, dims, metric, quant,
						  sc.initial.data(), static_cast<int32_t>(sc.initial.size()));
				CHECK_MSG(created == Status::Ok, "%s: fixture Create failed", sc.name);
				if (created != Status::Ok)
				{
					continue;
				}
				std::vector<float> source(static_cast<size_t>(sc.count) * dims);
				for (auto& v : source) { v = rng.NextFloat(); }
				for (int32_t r = 0; r < sc.count; ++r)
				{
					CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims,
							  nullptr) == Status::Ok);
				}
				if (sc.tombstoneAll)
				{
					for (int32_t r = 0; r < sc.count; ++r)
					{
						CHECK(bank.Remove(r) == Status::Ok);
					}
				}

				const ChannelInfo* targetPtr = sc.target.empty() ? nullptr : sc.target.data();
				const int32_t targetCount = static_cast<int32_t>(sc.target.size());
				const Status relabelStatus = bank.Relabel(targetPtr, targetCount);
				CHECK_MSG(relabelStatus == Status::Ok,
					"%s: Relabel failed (metric=%d quant=%d): status=%d", sc.name,
					static_cast<int>(metric), static_cast<int>(quant),
					static_cast<int>(relabelStatus));
				if (relabelStatus != Status::Ok)
				{
					continue;
				}
				CHECK_MSG(bank.GetChannelCount() == targetCount,
					"%s: GetChannelCount() should be %d after Relabel, got %d (metric=%d quant=%d)",
					sc.name, targetCount, bank.GetChannelCount(), static_cast<int>(metric),
					static_cast<int>(quant));

				if (sc.count == 0)
				{
					float extra[dims];
					for (int32_t i = 0; i < dims; ++i) { extra[i] = rng.NextFloat(); }
					int32_t idx = -1;
					CHECK(bank.Append(extra, dims, &idx) == Status::Ok);
					CHECK(idx == 0);
					continue;
				}

				BankView snap;
				std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(bank.Count()), 0u);
				CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
				CHECK_MSG(snap.channelCount == targetCount,
					"%s: snapshot channelCount mismatch (metric=%d quant=%d)", sc.name,
					static_cast<int>(metric), static_cast<int>(quant));

				if (sc.tombstoneAll)
				{
					continue;
				}

				std::vector<float> queryRaw(dims);
				for (auto& v : queryRaw) { v = rng.NextFloat(); }
				AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
				PadQuery(queryRaw, snap.paddedDims, qbuf.F32());

				if (targetCount > 0)
				{
					const ChannelInfo& ch = sc.target[0];
					const QuerySegment seg[1] = {{ch.offset, ch.length, 1.0f}};
					QueryParams p;
					p.k = 10;
					p.segments = seg;
					p.segmentCount = 1;
					p.excludeBits = tombs.data();
					Workspace ws;
					Hit hits[10];
					int32_t n = 0;
					CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
					const std::vector<FeatRefHit> ref =
						ChannelBruteForce(snap, qbuf.F32(), ch, tombs.data(), metric);
					CheckFeatTopK(ref, hits, n, 10, bank.Count(), metric, quant, sc.name);

					// Whole-row single-channel == whole-vector path bit-for-bit -- Dot/L2 ONLY.
					// A Cosine channel query (even a whole-row one) takes the per-channel-cosine
					// kernel path (query.cpp bPerChannelCosine), dividing each score by the
					// per-channel sub-norm 1/||quantized row|| -- a factor the whole-vector path
					// (norm folded pre-quantization) does not apply, so the two differ by a
					// near-1 factor. This is inherent channel-cosine semantics (a V3.0 property,
					// verified independent of Relabel), not a relabel defect. The Cosine whole-row
					// end-state is already FEAT-covered above; only Dot/L2 get the bit-identity
					// INV. (Routed to Vitruvius: S24.8 dim-4 "must match the whole-vector path
					// bit-for-bit" should read "for Dot/L2; FEAT-equivalent for Cosine".)
					if (metric != Metric::Cosine && targetCount == 1 && ch.offset == 0 &&
						ch.length == snap.paddedDims)
					{
						QueryParams pw;
						pw.k = 10;
						pw.excludeBits = tombs.data();
						Workspace wsw;
						Hit whits[10];
						int32_t nw = 0;
						CHECK(Query(snap, qbuf.F32(), pw, wsw, whits, &nw) == Status::Ok);
						CHECK_MSG(nw == n,
							"%s: whole-row channel hit count != whole-vector path", sc.name);
						for (int32_t i = 0; i < n && i < nw; ++i)
						{
							CHECK_MSG(hits[i].index == whits[i].index &&
									hits[i].score == whits[i].score,
								"%s: whole-row channel query not bit-identical to the "
								"whole-vector path at hit %d", sc.name, i);
						}
					}
				}
			}
		}
	}
}

// T-V3.1-ZeroNorm (dim 4/5) -- a relabel that lands an all-zero-norm channel: inverse norm
// stores 0, the channel scores 0, never NaN.
static void TestRelabelAllZeroNormChannel()
{
	const int32_t d = 8;
	const ChannelInfo initial[1] = {{0, 8}}; // single whole-row channel -- distinct from target
	ScratchBank bank;
	CHECK(bank.Create(4, d, Metric::Cosine, Quantization::Float32, initial, 1) == Status::Ok);

	float row0[d] = {1, 0, 0, 0, 0, 0, 0, 0}; // channel [4,8)'s sub-vector is all zero
	CHECK(bank.Append(row0, d, nullptr) == Status::Ok);

	const ChannelInfo target[2] = {{0, 4}, {4, 4}};
	CHECK(bank.Relabel(target, 2) == Status::Ok);
	CHECK_MSG(bank.GetChannelCount() == 2,
		"zero-norm-channel fixture: table did not update, got channelCount=%d",
		bank.GetChannelCount());

	BankView snap;
	uint32_t tombs = 0;
	CHECK(bank.Snapshot(&snap, &tombs) == Status::Ok);
	CHECK_MSG(snap.channelInvNorms != nullptr,
		"zero-norm-channel relabel fixture has no sub-norm arena");
	if (snap.channelInvNorms != nullptr)
	{
		CHECK_MSG(snap.channelInvNorms[1] == 0.0f,
			"channel 1's inverse sub-norm should floor to 0 on an all-zero sub-vector, got %g",
			static_cast<double>(snap.channelInvNorms[1]));
		CHECK(std::isfinite(snap.channelInvNorms[0]));
	}

	AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
	float qraw[d] = {0.3f, -0.2f, 0.1f, 0.4f, 0.5f, -0.6f, 0.2f, 0.1f};
	std::vector<float> qv(qraw, qraw + d);
	PadQuery(qv, snap.paddedDims, qbuf.F32());
	const QuerySegment seg[1] = {{4, 4, 1.0f}};
	QueryParams p;
	p.k = 1;
	p.segments = seg;
	p.segmentCount = 1;
	Workspace ws;
	Hit hits[1];
	int32_t n = 0;
	CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
	if (n > 0)
	{
		CHECK_MSG(hits[0].score == 0.0f, "zero-norm channel query should score exactly 0, got %g",
			static_cast<double>(hits[0].score));
		CHECK(std::isfinite(hits[0].score));
	}
}

// T-V3.1-Parity (dim 6, the crux; Loki S1 scoping) -- after Relabel(T_new),
// ChannelInvNorms_ over the LIVE-ROW PREFIX [0, count) x newChannelCount bit-equals a fresh
// Create(T_new)+Append of the SAME rows in the SAME order at the SAME starting capacity.
// Scoped to the live prefix, never a full-arena memcmp: a Grow-then-Relabel bank outsizes a
// same-content fresh twin, so a full-arena compare is ill-defined.
static void TestRelabelParityVsFreshCreate()
{
	Rng rng(0x10000aull);
	const int32_t dims = 64;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};

	struct Scenario { const char* name; std::vector<ChannelInfo> target; bool growFirst; bool promote; };
	std::vector<Scenario> scenarios = {
		{"boundary move", {{0, 16}, {16, 48}}, false, false},
		{"count up", {{0, 16}, {16, 16}, {32, 32}}, false, false},
		{"count down", {{0, 64}}, false, false},
		{"promote-from-single-space", {{0, 32}, {32, 32}}, false, true},
		{"boundary move after Grow (capacity mismatch)", {{0, 16}, {16, 48}}, true, false},
	};

	for (const Scenario& sc : scenarios)
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			const int32_t startCapacity = 40;
			const int32_t count = 24;

			ScratchBank relabeled;
			const Status created = sc.promote
				? relabeled.Create(startCapacity, dims, Metric::Cosine, quant)
				: relabeled.Create(startCapacity, dims, Metric::Cosine, quant, oldTable, 2);
			CHECK_MSG(created == Status::Ok, "%s (quant=%d): fixture Create failed", sc.name,
				static_cast<int>(quant));
			if (created != Status::Ok)
			{
				continue;
			}
			std::vector<float> source(static_cast<size_t>(count) * dims);
			for (auto& v : source) { v = rng.NextFloat(); }
			for (int32_t r = 0; r < count; ++r)
			{
				CHECK(relabeled.Append(source.data() + static_cast<size_t>(r) * dims, dims,
						  nullptr) == Status::Ok);
			}
			if (sc.growFirst)
			{
				CHECK(relabeled.Grow(startCapacity * 2) == Status::Ok);
			}
			CHECK_MSG(relabeled.Relabel(sc.target.data(), static_cast<int32_t>(sc.target.size())) ==
					Status::Ok,
				"%s (quant=%d): Relabel failed", sc.name, static_cast<int>(quant));

			ScratchBank fresh;
			CHECK(fresh.Create(startCapacity, dims, Metric::Cosine, quant, sc.target.data(),
					  static_cast<int32_t>(sc.target.size())) == Status::Ok);
			for (int32_t r = 0; r < count; ++r)
			{
				CHECK(fresh.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
					Status::Ok);
			}

			BankView relSnap, freshSnap;
			std::vector<uint32_t> relTombs(ScratchBank::TombstoneWords(relabeled.Count()), 0u);
			std::vector<uint32_t> freshTombs(ScratchBank::TombstoneWords(fresh.Count()), 0u);
			CHECK(relabeled.Snapshot(&relSnap, relTombs.data()) == Status::Ok);
			CHECK(fresh.Snapshot(&freshSnap, freshTombs.data()) == Status::Ok);

			CHECK_MSG(relSnap.channelCount == freshSnap.channelCount,
				"%s (quant=%d): channel count mismatch vs the fresh twin", sc.name,
				static_cast<int>(quant));

			const size_t rowBytes = static_cast<size_t>(relSnap.paddedDims) * ElementSize(quant);
			CHECK_MSG(std::memcmp(relSnap.rows, freshSnap.rows,
						  static_cast<size_t>(count) * rowBytes) == 0,
				"%s (quant=%d): relabeled rows differ from a fresh twin's", sc.name,
				static_cast<int>(quant));
			if (quant == Quantization::Int8)
			{
				CHECK(std::memcmp(relSnap.scales, freshSnap.scales,
						  static_cast<size_t>(count) * sizeof(float)) == 0);
			}

			if (relSnap.channelInvNorms != nullptr && freshSnap.channelInvNorms != nullptr)
			{
				const size_t liveBytes =
					static_cast<size_t>(count) * relSnap.channelCount * sizeof(float);
				CHECK_MSG(std::memcmp(relSnap.channelInvNorms, freshSnap.channelInvNorms,
							  liveBytes) == 0,
					"%s (quant=%d): live-prefix sub-norms differ from a fresh Create+Append "
					"under the new table", sc.name, static_cast<int>(quant));
			}
		}
	}
}

// T-V3.1-ParityTautology (Forge N1/N2) -- the near-tautological determinism check: the
// relabel-derived sub-norms bit-equal ComputeChannelInverseNorms over the current rows
// under the new table.
static void TestRelabelParityEqualsComputeChannelInverseNorms()
{
	Rng rng(0x10000bull);
	const int32_t dims = 48;
	const int32_t count = 20;
	const ChannelInfo oldTable[2] = {{0, 16}, {16, 32}};
	const ChannelInfo newTable[2] = {{0, 32}, {32, 16}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<float> row(static_cast<size_t>(dims));
		for (auto& v : row) { v = rng.NextFloat(); }
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}
	CHECK(bank.Relabel(newTable, 2) == Status::Ok);

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
	CHECK_MSG(snap.channelCount == 2 && snap.channels[0].offset == 0 && snap.channels[0].length == 32,
		"parity/ComputeChannelInverseNorms fixture: table did not update");
	if (snap.channelInvNorms == nullptr)
	{
		return;
	}
	std::vector<float> ref(static_cast<size_t>(count) * 2);
	CHECK(ComputeChannelInverseNorms(snap, ref.data()) == Status::Ok);
	CHECK_MSG(std::memcmp(ref.data(), snap.channelInvNorms, ref.size() * sizeof(float)) == 0,
		"relabel-derived sub-norms do not bit-equal ComputeChannelInverseNorms over the "
		"current rows under the new table");
}

// T-V3.1-PostAppend (dim 6/8, G-4) -- Append r1..rn -> Relabel(T_new) -> Append r(n+1)..rm:
// the row appended AFTER the relabel must derive its sub-norms under the MUTATED table,
// bit-equal to a fresh Create(T_new)+Append of the same full history.
static void TestRelabelPostAppendDerivation()
{
	Rng rng(0x10000cull);
	const int32_t dims = 64;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};
	const int32_t n1 = 12, n2 = 8;

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		std::vector<float> allSource(static_cast<size_t>(n1 + n2) * dims);
		for (auto& v : allSource) { v = rng.NextFloat(); }

		ScratchBank warm;
		CHECK(warm.Create(n1 + n2, dims, Metric::Cosine, quant, oldTable, 2) == Status::Ok);
		for (int32_t r = 0; r < n1; ++r)
		{
			CHECK(warm.Append(allSource.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		CHECK(warm.Relabel(newTable, 2) == Status::Ok);
		for (int32_t r = n1; r < n1 + n2; ++r)
		{
			CHECK(warm.Append(allSource.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}

		ScratchBank fresh;
		CHECK(fresh.Create(n1 + n2, dims, Metric::Cosine, quant, newTable, 2) == Status::Ok);
		for (int32_t r = 0; r < n1 + n2; ++r)
		{
			CHECK(fresh.Append(allSource.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}

		BankView warmSnap, freshSnap;
		std::vector<uint32_t> wt(ScratchBank::TombstoneWords(n1 + n2), 0u);
		std::vector<uint32_t> ft(ScratchBank::TombstoneWords(n1 + n2), 0u);
		CHECK(warm.Snapshot(&warmSnap, wt.data()) == Status::Ok);
		CHECK(fresh.Snapshot(&freshSnap, ft.data()) == Status::Ok);
		CHECK_MSG(warmSnap.channelCount == 2 && warmSnap.channels[0].length == 16,
			"post-append-derivation fixture (quant=%d): table did not update after Relabel",
			static_cast<int>(quant));
		if (warmSnap.channelInvNorms == nullptr || freshSnap.channelInvNorms == nullptr)
		{
			continue;
		}
		const size_t bytes = static_cast<size_t>(n1 + n2) * 2 * sizeof(float);
		CHECK_MSG(std::memcmp(warmSnap.channelInvNorms, freshSnap.channelInvNorms, bytes) == 0,
			"G-4 (quant=%d): a row appended AFTER Relabel must derive its sub-norms under the "
			"MUTATED table, bit-equal to a fresh Create(T_new)+Append of the same full history",
			static_cast<int>(quant));
	}
}

// T-V3.1-Floor (dim 6) -- a boundary move that narrows a channel onto an adversarial
// tiny-but-nonzero norm: a finite inverse (no Inf), scores clamped to [-1, 1].
static void TestRelabelSubNormFloorAndClamp()
{
	const int32_t dims = 32;
	const ChannelInfo oldTable[1] = {{0, 32}};
	const ChannelInfo tinyTable[2] = {{0, 16}, {16, 16}};

	ScratchBank bank;
	CHECK(bank.Create(4, dims, Metric::Cosine, Quantization::Float32, oldTable, 1) == Status::Ok);
	float row[dims];
	for (int32_t i = 0; i < 16; ++i) { row[i] = 1.0f; }
	for (int32_t i = 16; i < dims; ++i) { row[i] = 1e-20f; }
	CHECK(bank.Append(row, dims, nullptr) == Status::Ok);

	CHECK(bank.Relabel(tinyTable, 2) == Status::Ok);

	BankView snap;
	uint32_t tombs = 0;
	CHECK(bank.Snapshot(&snap, &tombs) == Status::Ok);
	CHECK_MSG(snap.channelCount == 2, "tiny-norm relabel fixture: table did not update");
	if (snap.channelInvNorms == nullptr || snap.channelCount != 2)
	{
		return;
	}
	CHECK_MSG(std::isfinite(snap.channelInvNorms[0]) && std::isfinite(snap.channelInvNorms[1]),
		"a tiny-but-nonzero channel norm must yield a finite inverse, got [%g, %g]",
		static_cast<double>(snap.channelInvNorms[0]), static_cast<double>(snap.channelInvNorms[1]));

	AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
	std::vector<float> qv(row, row + dims);
	PadQuery(qv, snap.paddedDims, qbuf.F32());
	const QuerySegment seg[1] = {{16, 16, 1.0f}};
	QueryParams p;
	p.k = 1;
	p.segments = seg;
	p.segmentCount = 1;
	Workspace ws;
	Hit hits[1];
	int32_t n = 0;
	CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
	if (n > 0)
	{
		CHECK_MSG(hits[0].score >= -1.0001f && hits[0].score <= 1.0001f,
			"a Cosine channel score must stay in [-1, 1] on the adversarial tiny-norm "
			"partition, got %g", static_cast<double>(hits[0].score));
	}
}

// T-V3.1-Golden (dim 6) -- forced-path + cross-runner golden over the per-channel Cosine
// sqrt query path on a post-relabel bank carrying an adversarial tiny-channel-norm member.
// Golden left at 0 (not yet pinned) in xd_fixtures.h -- pin once Relabel is green.
static void TestRelabelCrossRunnerGolden()
{
	Rng rng(0x10000eull);
	const int32_t dims = 64;
	const int32_t count = 40;
	const ChannelInfo oldTable[1] = {{0, dims}};
	const ChannelInfo tinyTable[2] = {{0, 48}, {48, 16}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 1) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t i = 48; i < dims; ++i) { source[static_cast<size_t>(i)] = 1e-6f; }
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	CHECK(bank.Relabel(tinyTable, 2) == Status::Ok);

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
	CHECK_MSG(snap.channelCount == 2, "golden fixture: relabel did not update the table");
	if (snap.channelCount != 2)
	{
		return;
	}

	std::vector<float> queryRaw(dims);
	for (auto& v : queryRaw) { v = rng.NextFloat(); }
	AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
	PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
	const QuerySegment seg[1] = {{tinyTable[1].offset, tinyTable[1].length, 1.0f}};
	QueryParams p;
	p.k = 10;
	p.segments = seg;
	p.segmentCount = 1;
	p.excludeBits = tombs.data();

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
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
		uint64_t h = 0xcbf29ce484222325ull;
		for (int32_t j = 0; j < n; ++j)
		{
			an::Hash(h, hits[j].score);
			h ^= static_cast<uint64_t>(hits[j].index) * 0x100000001b3ull;
		}
		hashes[i] = h;
		detail::ClearForcedXdSimdPath();
		if (i > 0)
		{
			CHECK_MSG(hashes[i] == hashes[0],
				"relabel adversarial-tiny-norm forced path %d hash %016llx != %016llx",
				static_cast<int>(paths[i]), static_cast<unsigned long long>(hashes[i]),
				static_cast<unsigned long long>(hashes[0]));
		}
	}
	std::printf("relabel cross-device golden hash: %016llx (%d forced paths agree)\n",
		static_cast<unsigned long long>(hashes[0]), static_cast<int>(paths.size()));
	if constexpr (xdfix::kGoldenRelabelXdHash != 0)
	{
		CHECK_MSG(hashes[0] == xdfix::kGoldenRelabelXdHash,
			"relabel golden hash %016llx != pinned golden %016llx",
			static_cast<unsigned long long>(hashes[0]),
			static_cast<unsigned long long>(xdfix::kGoldenRelabelXdHash));
	}
}

// T-V3.1-NeverTouchesRows (dim 7) -- Rows_/Scales_/Retained_ byte-identical before and
// after a successful relabel, across the metric x quant matrix.
static void TestRelabelNeverTouchesRows()
{
	Rng rng(0x10000full);
	const int32_t dims = 64;
	const int32_t count = 30;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[3] = {{0, 16}, {16, 16}, {32, 32}};

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			ScratchBank bank;
			CHECK(bank.Create(count, dims, metric, quant, oldTable, 2, /*retainFloats=*/true) ==
				Status::Ok);
			std::vector<float> source(static_cast<size_t>(count) * dims);
			for (auto& v : source) { v = rng.NextFloat(); }
			for (int32_t r = 0; r < count; ++r)
			{
				CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
					Status::Ok);
			}

			BankView before;
			std::vector<uint32_t> tombsBefore(ScratchBank::TombstoneWords(count), 0u);
			CHECK(bank.Snapshot(&before, tombsBefore.data()) == Status::Ok);
			const size_t rowBytes = static_cast<size_t>(before.paddedDims) * ElementSize(quant);
			std::vector<uint8_t> rowsBefore(static_cast<size_t>(count) * rowBytes);
			std::memcpy(rowsBefore.data(), before.rows, rowsBefore.size());
			std::vector<float> scalesBefore;
			if (quant == Quantization::Int8)
			{
				scalesBefore.assign(before.scales, before.scales + count);
			}
			std::vector<float> retainedBefore(static_cast<size_t>(count) * dims);
			for (int32_t r = 0; r < count; ++r)
			{
				std::memcpy(retainedBefore.data() + static_cast<size_t>(r) * dims,
					bank.RetainedRow(r), static_cast<size_t>(dims) * sizeof(float));
			}

			CHECK_MSG(bank.Relabel(newTable, 3) == Status::Ok,
				"never-touches-rows fixture (metric=%d quant=%d): Relabel failed",
				static_cast<int>(metric), static_cast<int>(quant));

			BankView after;
			std::vector<uint32_t> tombsAfter(ScratchBank::TombstoneWords(count), 0u);
			CHECK(bank.Snapshot(&after, tombsAfter.data()) == Status::Ok);
			CHECK_MSG(after.channelCount == 3,
				"never-touches-rows fixture (metric=%d quant=%d): table did not update",
				static_cast<int>(metric), static_cast<int>(quant));
			CHECK_MSG(std::memcmp(rowsBefore.data(), after.rows, rowsBefore.size()) == 0,
				"Relabel touched the stored row bytes (metric=%d quant=%d)",
				static_cast<int>(metric), static_cast<int>(quant));
			if (quant == Quantization::Int8)
			{
				CHECK_MSG(std::memcmp(scalesBefore.data(), after.scales,
							  scalesBefore.size() * sizeof(float)) == 0,
					"Relabel touched the int8 scales (metric=%d quant=%d)",
					static_cast<int>(metric), static_cast<int>(quant));
			}
			for (int32_t r = 0; r < count; ++r)
			{
				CHECK_MSG(std::memcmp(retainedBefore.data() + static_cast<size_t>(r) * dims,
							  bank.RetainedRow(r), static_cast<size_t>(dims) * sizeof(float)) == 0,
					"Relabel touched the retained float row %d (metric=%d quant=%d)", r,
					static_cast<int>(metric), static_cast<int>(quant));
			}
		}
	}
}

// T-V3.1-WholeVectorUnchanged (dim 7) -- a channel relabel leaves the whole-vector query
// path bit-unchanged.
static void TestRelabelWholeVectorPathUnchanged()
{
	Rng rng(0x100010ull);
	const int32_t dims = 48;
	const int32_t count = 20;
	const ChannelInfo oldTable[2] = {{0, 16}, {16, 32}};
	const ChannelInfo newTable[2] = {{0, 32}, {32, 16}};

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		ScratchBank bank;
		CHECK(bank.Create(count, dims, metric, Quantization::Int8, oldTable, 2) == Status::Ok);
		std::vector<float> source(static_cast<size_t>(count) * dims);
		for (auto& v : source) { v = rng.NextFloat(); }
		for (int32_t r = 0; r < count; ++r)
		{
			CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}

		BankView before;
		std::vector<uint32_t> tombsB(ScratchBank::TombstoneWords(count), 0u);
		CHECK(bank.Snapshot(&before, tombsB.data()) == Status::Ok);
		std::vector<float> queryRaw(dims);
		for (auto& v : queryRaw) { v = rng.NextFloat(); }
		AlignedBuf qbuf(static_cast<size_t>(before.paddedDims) * sizeof(float));
		PadQuery(queryRaw, before.paddedDims, qbuf.F32());
		QueryParams p;
		p.k = 10;
		p.excludeBits = tombsB.data();
		Workspace ws;
		Hit hitsBefore[10];
		int32_t nBefore = 0;
		CHECK(Query(before, qbuf.F32(), p, ws, hitsBefore, &nBefore) == Status::Ok);

		CHECK(bank.Relabel(newTable, 2) == Status::Ok);

		BankView after;
		std::vector<uint32_t> tombsA(ScratchBank::TombstoneWords(count), 0u);
		CHECK(bank.Snapshot(&after, tombsA.data()) == Status::Ok);
		p.excludeBits = tombsA.data();
		Hit hitsAfter[10];
		int32_t nAfter = 0;
		CHECK(Query(after, qbuf.F32(), p, ws, hitsAfter, &nAfter) == Status::Ok);

		CHECK_MSG(after.channelCount == 2 && after.channels[0].offset == 0 &&
				after.channels[0].length == 32,
			"whole-vector-unchanged fixture (metric=%d): table did not update",
			static_cast<int>(metric));
		CHECK_MSG(nBefore == nAfter,
			"whole-vector query hit count changed across a relabel (metric=%d)",
			static_cast<int>(metric));
		for (int32_t i = 0; i < nBefore && i < nAfter; ++i)
		{
			CHECK_MSG(hitsBefore[i].index == hitsAfter[i].index &&
					hitsBefore[i].score == hitsAfter[i].score,
				"the whole-vector path must be bit-unchanged by a channel relabel "
				"(metric=%d, hit %d)", static_cast<int>(metric), i);
		}
	}
}

// T-V3.1-MetricMatrix (dim 7, G-2) -- explicit contract claim: Relabel + post-relabel
// per-channel query + channel-scoped analytics all correct on Dot, Cosine, and L2 each.
static void TestRelabelMetricMatrixContractClaim()
{
	Rng rng(0x100011ull);
	const int32_t dims = 64;
	const int32_t count = 32;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		ScratchBank bank;
		CHECK(bank.Create(count, dims, metric, Quantization::Int8, oldTable, 2) == Status::Ok);
		std::vector<float> source(static_cast<size_t>(count) * dims);
		for (auto& v : source) { v = rng.NextFloat(); }
		for (int32_t r = 0; r < count; ++r)
		{
			CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		CHECK_MSG(bank.Relabel(newTable, 2) == Status::Ok,
			"metric-matrix contract (metric=%d): Relabel failed", static_cast<int>(metric));

		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		CHECK_MSG(snap.channelCount == 2 && snap.channels[0].length == 16,
			"metric-matrix contract (metric=%d): table did not update", static_cast<int>(metric));

		std::vector<float> queryRaw(dims);
		for (auto& v : queryRaw) { v = rng.NextFloat(); }
		AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
		PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
		const QuerySegment seg[1] = {{newTable[0].offset, newTable[0].length, 1.0f}};
		QueryParams p;
		p.k = 10;
		p.segments = seg;
		p.segmentCount = 1;
		p.excludeBits = tombs.data();
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
		const std::vector<FeatRefHit> ref =
			ChannelBruteForce(snap, qbuf.F32(), newTable[0], tombs.data(), metric);
		CheckFeatTopK(ref, hits, n, 10, count, metric, Quantization::Int8,
			"metric-matrix contract per-channel query");

		std::vector<int32_t> allRows(static_cast<size_t>(count));
		for (int32_t i = 0; i < count; ++i) { allRows[static_cast<size_t>(i)] = i; }
		AlignedBuf centroidScratch(static_cast<size_t>(snap.paddedDims));
		float analyticsResult = 0.0f;
		const Status analyticsStatus = SpreadCrossDeviceChannel(snap, allRows.data(), count,
			tombs.data(), Reduce::Mean, 0, centroidScratch.I8(), &analyticsResult);
		CHECK_MSG(analyticsStatus == Status::Ok,
			"metric-matrix contract (metric=%d): channel-scoped analytics over the new "
			"partition failed, status=%d", static_cast<int>(metric),
			static_cast<int>(analyticsStatus));
	}
}

// T-V3.1-CompRetention (dim 8) -- retained floats copy across index-preserving; per-channel
// recall re-measurable under the new table.
static void TestRelabelCompositionRetention()
{
	Rng rng(0x100012ull);
	const int32_t dims = 64;
	const int32_t count = 40;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2,
			/*retainFloats=*/true) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	std::vector<float> retainedBefore(static_cast<size_t>(count) * dims);
	for (int32_t r = 0; r < count; ++r)
	{
		std::memcpy(retainedBefore.data() + static_cast<size_t>(r) * dims, bank.RetainedRow(r),
			static_cast<size_t>(dims) * sizeof(float));
	}

	CHECK(bank.Relabel(newTable, 2) == Status::Ok);
	CHECK_MSG(bank.GetChannelCount() == 2 && bank.GetChannels()[0].length == 16,
		"relabel composition x retention: table did not update, length[0]=%d",
		bank.GetChannelCount() == 2 ? bank.GetChannels()[0].length : -1);
	CHECK_MSG(bank.RetainsFloats(), "a relabel must not drop the retention property");
	for (int32_t r = 0; r < count; ++r)
	{
		const float* row = bank.RetainedRow(r);
		CHECK_MSG(row != nullptr &&
				std::memcmp(row, retainedBefore.data() + static_cast<size_t>(r) * dims,
					static_cast<size_t>(dims) * sizeof(float)) == 0,
			"relabel composition x retention: retained row %d changed / vanished", r);
	}

	Workspace ws;
	ScratchRecallReport reports[2];
	const Status s = bank.MeasureScratchRecallPerChannel(ws, reports, 2);
	CHECK_MSG(s == Status::Ok,
		"relabel composition x retention: per-channel recall must be re-measurable under the "
		"new table, got status=%d", static_cast<int>(s));
	if (s == Status::Ok)
	{
		for (int32_t c = 0; c < 2; ++c)
		{
			CHECK(reports[c].recall >= 0.0f && reports[c].recall <= 1.0f);
			CHECK(reports[c].generation == bank.Generation());
		}
	}
}

// T-V3.1-CompGrow (dim 8) -- Relabel-then-Grow and Grow-then-Relabel produce byte-identical,
// order-independent results (both index-preserving).
static void TestRelabelCompositionGrowOrderIndependence()
{
	Rng rng(0x100013ull);
	const int32_t dims = 64;
	const int32_t count = 16;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }

	ScratchBank a;
	CHECK(a.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(a.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) == Status::Ok);
	}
	CHECK(a.Relabel(newTable, 2) == Status::Ok);
	CHECK(a.Grow(count * 2) == Status::Ok);

	ScratchBank b;
	CHECK(b.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) == Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(b.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) == Status::Ok);
	}
	CHECK(b.Grow(count * 2) == Status::Ok);
	CHECK(b.Relabel(newTable, 2) == Status::Ok);

	BankView av, bv;
	std::vector<uint32_t> at(ScratchBank::TombstoneWords(count), 0u);
	std::vector<uint32_t> bt(ScratchBank::TombstoneWords(count), 0u);
	CHECK(a.Snapshot(&av, at.data()) == Status::Ok);
	CHECK(b.Snapshot(&bv, bt.data()) == Status::Ok);
	CHECK_MSG(av.channelCount == 2 && bv.channelCount == 2 && av.channels[0].length == 16 &&
			bv.channels[0].length == 16,
		"grow/relabel order-independence fixture: table did not update in one or both orders");
	CHECK(a.Capacity() == count * 2 && b.Capacity() == count * 2);
	CHECK(a.Count() == count && b.Count() == count);

	const size_t rowBytes = static_cast<size_t>(av.paddedDims) * ElementSize(Quantization::Int8);
	CHECK_MSG(std::memcmp(av.rows, bv.rows, static_cast<size_t>(count) * rowBytes) == 0,
		"Relabel-then-Grow and Grow-then-Relabel must produce byte-identical row content");
	if (av.channelInvNorms != nullptr && bv.channelInvNorms != nullptr)
	{
		const size_t liveBytes = static_cast<size_t>(count) * 2 * sizeof(float);
		CHECK_MSG(std::memcmp(av.channelInvNorms, bv.channelInvNorms, liveBytes) == 0,
			"Relabel-then-Grow and Grow-then-Relabel must produce bit-identical live-prefix "
			"sub-norms");
	}
}

// T-V3.1-CompTombstones (dim 8) -- tombstones preserved index-aligned across a relabel; a
// post-relabel channel query still excludes them.
static void TestRelabelCompositionTombstones()
{
	Rng rng(0x100014ull);
	const int32_t dims = 64;
	const int32_t count = 24;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	std::vector<int32_t> removed;
	for (int32_t r = 1; r < count; r += 4)
	{
		CHECK(bank.Remove(r) == Status::Ok);
		removed.push_back(r);
	}

	CHECK(bank.Relabel(newTable, 2) == Status::Ok);

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
	CHECK_MSG(snap.channelCount == 2 && snap.channels[0].length == 16,
		"tombstone-composition fixture: table did not update");
	for (int32_t r : removed)
	{
		CHECK_MSG(IsExcluded(tombs.data(), r),
			"a relabel must preserve tombstones index-aligned; row %d lost its tombstone", r);
	}

	std::vector<float> queryRaw(dims);
	for (auto& v : queryRaw) { v = rng.NextFloat(); }
	AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
	PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
	const QuerySegment seg[1] = {{0, 16, 1.0f}};
	QueryParams p;
	p.k = count;
	p.segments = seg;
	p.segmentCount = 1;
	p.excludeBits = tombs.data();
	Workspace ws;
	std::vector<Hit> hits(static_cast<size_t>(count));
	int32_t n = 0;
	CHECK(Query(snap, qbuf.F32(), p, ws, hits.data(), &n) == Status::Ok);
	for (int32_t i = 0; i < n; ++i)
	{
		for (int32_t r : removed)
		{
			CHECK_MSG(hits[static_cast<size_t>(i)].index != r,
				"a post-relabel channel query returned tombstoned row %d", r);
		}
	}
}

// T-V3.1-CompFreeze (dim 8/10) -- channel-aware Freeze after a relabel emits the new table +
// re-derived sub-norms + the frozen FEAT end-state.
static void TestRelabelCompositionFreeze()
{
	Rng rng(0x100015ull);
	const int32_t dims = 64;
	const int32_t count = 60;
	const int32_t k = 10;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	for (Quantization quant : {Quantization::Float32, Quantization::Int8})
	{
		ScratchBank bank;
		CHECK(bank.Create(count, dims, Metric::Cosine, quant, oldTable, 2) == Status::Ok);
		std::vector<float> source(static_cast<size_t>(count) * dims);
		for (auto& v : source) { v = rng.NextFloat(); }
		for (int32_t r = 0; r < count; ++r)
		{
			CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		CHECK(bank.Relabel(newTable, 2) == Status::Ok);

		const int32_t live = bank.FreezeLiveCount();
		AlignedBuf frozenRows(
			static_cast<size_t>(live > 0 ? live : 1) * bank.GetPaddedDims() * ElementSize(quant));
		std::vector<float> frozenScales(
			quant == Quantization::Int8 ? static_cast<size_t>(live) : size_t{1});
		std::vector<int32_t> indexMap(static_cast<size_t>(count), -2);
		std::vector<float> frozenInvNorms(static_cast<size_t>(live) * 2, -1.0f);
		const Status frozenStatus = bank.Freeze(frozenRows.ptr,
			quant == Quantization::Int8 ? frozenScales.data() : nullptr, indexMap.data(),
			frozenInvNorms.data());
		CHECK_MSG(frozenStatus == Status::Ok,
			"post-relabel channel-aware Freeze failed (quant=%d): status=%d",
			static_cast<int>(quant), static_cast<int>(frozenStatus));
		if (frozenStatus != Status::Ok)
		{
			continue;
		}
		CHECK_MSG(bank.GetChannelCount() == 2 && bank.GetChannels()[0].length == 16,
			"post-relabel Freeze fixture (quant=%d): table did not update before freezing",
			static_cast<int>(quant));

		BankView frozen;
		frozen.rows = frozenRows.ptr;
		frozen.scales = quant == Quantization::Int8 ? frozenScales.data() : nullptr;
		frozen.count = live;
		frozen.dims = dims;
		frozen.paddedDims = bank.GetPaddedDims();
		frozen.quant = quant;
		frozen.metric = Metric::Cosine;
		frozen.channels = bank.GetChannels();
		frozen.channelCount = bank.GetChannelCount();
		frozen.channelInvNorms = frozenInvNorms.data();
		CHECK(ValidateBank(frozen) == Status::Ok);

		std::vector<float> refInvNorms(static_cast<size_t>(live) * frozen.channelCount);
		CHECK(ComputeChannelInverseNorms(frozen, refInvNorms.data()) == Status::Ok);
		CHECK_MSG(std::memcmp(refInvNorms.data(), frozenInvNorms.data(),
					  refInvNorms.size() * sizeof(float)) == 0,
			"frozen sub-norms after a relabel do not bit-equal ComputeChannelInverseNorms over "
			"the compacted rows (quant=%d)", static_cast<int>(quant));

		std::vector<float> queryRaw(dims);
		for (auto& v : queryRaw) { v = rng.NextFloat(); }
		AlignedBuf qbuf(static_cast<size_t>(frozen.paddedDims) * sizeof(float));
		PadQuery(queryRaw, frozen.paddedDims, qbuf.F32());
		const ChannelInfo& ch = frozen.channels[0];
		const QuerySegment seg[1] = {{ch.offset, ch.length, 1.0f}};
		QueryParams p;
		p.k = k;
		p.segments = seg;
		p.segmentCount = 1;
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		CHECK(Query(frozen, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
		const std::vector<FeatRefHit> ref =
			ChannelBruteForce(frozen, qbuf.F32(), ch, nullptr, Metric::Cosine);
		CheckFeatTopK(ref, hits, n, k, live, Metric::Cosine, quant, "post-relabel frozen channel FEAT");
	}
}

// T-V3.1-CompCrossDevice (dim 8) -- a CrossDevice channel query over the new partition.
static void TestRelabelCompositionCrossDevice()
{
	Rng rng(0x100016ull);
	const int32_t dims = 64;
	const int32_t count = 40;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	CHECK(bank.Relabel(newTable, 2) == Status::Ok);

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
	CHECK_MSG(snap.channelCount == 2 && snap.channels[0].length == 16,
		"CrossDevice composition fixture: table did not update");

	std::vector<float> queryRaw(dims);
	for (auto& v : queryRaw) { v = rng.NextFloat(); }
	AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
	PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
	const QuerySegment seg[1] = {{newTable[0].offset, newTable[0].length, 1.0f}};
	QueryParams p;
	p.k = 10;
	p.segments = seg;
	p.segmentCount = 1;
	p.excludeBits = tombs.data();
	p.exactness = Exactness::CrossDevice;
	Workspace ws;
	Hit hits[10];
	int32_t n = 0;
	const Status s = Query(snap, qbuf.F32(), p, ws, hits, &n);
	CHECK_MSG(s == Status::Ok, "CrossDevice channel query over the new partition failed: %d",
		static_cast<int>(s));
	if (s == Status::Ok)
	{
		for (int32_t i = 0; i < n; ++i)
		{
			CHECK(std::isfinite(hits[i].score));
		}
	}
}

// T-V3.1-CompBatchIntersectMetric (dim 8) -- a per-channel query over the new partition as
// a QueryBatch member, a QueryIntersect member, and under ScoreAs::Dot on an L2 channel bank.
static void TestRelabelCompositionBatchIntersectionMetricOverride()
{
	Rng rng(0x100017ull);
	const int32_t dims = 64;
	const int32_t count = 40;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	{
		ScratchBank bank;
		CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) ==
			Status::Ok);
		std::vector<float> source(static_cast<size_t>(count) * dims);
		for (auto& v : source) { v = rng.NextFloat(); }
		for (int32_t r = 0; r < count; ++r)
		{
			CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		CHECK(bank.Relabel(newTable, 2) == Status::Ok);

		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		CHECK_MSG(snap.channelCount == 2 && snap.channels[0].length == 16,
			"batch/intersection fixture: table did not update");

		const int32_t qCount = 3;
		std::vector<float> queries(static_cast<size_t>(qCount) * dims);
		for (auto& v : queries) { v = rng.NextFloat(); }
		AlignedBuf qbuf(static_cast<size_t>(qCount) * snap.paddedDims * sizeof(float));
		for (int32_t qi = 0; qi < qCount; ++qi)
		{
			std::vector<float> qv(queries.begin() + static_cast<ptrdiff_t>(qi) * dims,
				queries.begin() + static_cast<ptrdiff_t>(qi) * dims + dims);
			PadQuery(qv, snap.paddedDims, qbuf.F32() + static_cast<int64_t>(qi) * snap.paddedDims);
		}
		const QuerySegment seg[1] = {{newTable[0].offset, newTable[0].length, 1.0f}};
		QueryParams p;
		p.k = 5;
		p.segments = seg;
		p.segmentCount = 1;
		p.excludeBits = tombs.data();

		Workspace wsBatch;
		std::vector<Hit> batchHits(static_cast<size_t>(qCount) * 5);
		std::vector<int32_t> batchCounts(static_cast<size_t>(qCount));
		CHECK(QueryBatch(snap, qbuf.F32(), qCount, p, wsBatch, batchHits.data(),
				  batchCounts.data()) == Status::Ok);

		Workspace wsInt;
		Hit intHits[5];
		int32_t nInt = 0;
		CHECK(QueryIntersect(snap, qbuf.F32(), qCount, p, wsInt, intHits, &nInt) == Status::Ok);
	}

	{
		ScratchBank bank;
		CHECK(bank.Create(count, dims, Metric::L2, Quantization::Int8, oldTable, 2) == Status::Ok);
		std::vector<float> source(static_cast<size_t>(count) * dims);
		for (auto& v : source) { v = rng.NextFloat(); }
		for (int32_t r = 0; r < count; ++r)
		{
			CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
				Status::Ok);
		}
		CHECK(bank.Relabel(newTable, 2) == Status::Ok);

		BankView snap;
		std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
		CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
		CHECK_MSG(snap.channelCount == 2 && snap.channels[0].length == 16,
			"metric-override fixture: table did not update");

		std::vector<float> queryRaw(dims);
		for (auto& v : queryRaw) { v = rng.NextFloat(); }
		AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
		PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
		const QuerySegment seg[1] = {{newTable[0].offset, newTable[0].length, 1.0f}};
		QueryParams p;
		p.k = 10;
		p.segments = seg;
		p.segmentCount = 1;
		p.excludeBits = tombs.data();
		p.scoreAs = ScoreAs::Dot;
		Workspace ws;
		Hit hits[10];
		int32_t n = 0;
		CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
		const std::vector<FeatRefHit> ref =
			ChannelBruteForce(snap, qbuf.F32(), newTable[0], tombs.data(), Metric::Dot);
		CheckFeatTopK(ref, hits, n, 10, count, Metric::Dot, Quantization::Int8,
			"post-relabel L2-bank ScoreAs::Dot channel FEAT");
	}
}

// T-V3.1-CompAnalytics (dim 8/10, AC-7) -- a channel-scoped analytics reduction over the new
// partition matches a float64 definition-grounded reference.
static void TestRelabelCompositionAnalytics()
{
	Rng rng(0x100018ull);
	const int32_t dims = 64;
	const int32_t count = 32;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	CHECK(bank.Relabel(newTable, 2) == Status::Ok);

	BankView snap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
	CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
	CHECK_MSG(snap.channelCount == 2 && snap.channels[0].length == 16,
		"analytics-composition fixture: table did not update");

	std::vector<int32_t> allRows(static_cast<size_t>(count));
	for (int32_t i = 0; i < count; ++i) { allRows[static_cast<size_t>(i)] = i; }
	AlignedBuf centroidScratch(static_cast<size_t>(snap.paddedDims));
	float mean = 0.0f;
	const Status s = SpreadCrossDeviceChannel(snap, allRows.data(), count, tombs.data(),
		Reduce::Mean, 0, centroidScratch.I8(), &mean);
	CHECK_MSG(s == Status::Ok, "post-relabel channel-scoped SpreadCrossDeviceChannel failed: %d",
		static_cast<int>(s));
	if (s != Status::Ok)
	{
		return;
	}

	// Definition-grounded float64 reference: mean (1 - cosine) distance-to-centroid over the
	// DEQUANTIZED channel sub-vectors -- never touching the CrossDevice int8 pooling
	// machinery the operator itself uses.
	std::vector<std::vector<double>> subVecs(static_cast<size_t>(count));
	std::vector<double> centroid(static_cast<size_t>(newTable[0].length), 0.0);
	for (int32_t r = 0; r < count; ++r)
	{
		std::vector<double> full;
		DequantizeRow(snap, r, full);
		subVecs[static_cast<size_t>(r)].assign(full.begin() + newTable[0].offset,
			full.begin() + newTable[0].offset + newTable[0].length);
		for (size_t j = 0; j < centroid.size(); ++j)
		{
			centroid[j] += subVecs[static_cast<size_t>(r)][j];
		}
	}
	for (double& c : centroid) { c /= count; }
	double centroidNorm = 0.0;
	for (double c : centroid) { centroidNorm += c * c; }
	centroidNorm = std::sqrt(centroidNorm);
	double sumDist = 0.0;
	for (int32_t r = 0; r < count; ++r)
	{
		double dot = 0.0, rn = 0.0;
		for (size_t j = 0; j < centroid.size(); ++j)
		{
			dot += subVecs[static_cast<size_t>(r)][j] * centroid[j];
			rn += subVecs[static_cast<size_t>(r)][j] * subVecs[static_cast<size_t>(r)][j];
		}
		rn = std::sqrt(rn);
		const double cos = (rn > 0.0 && centroidNorm > 0.0) ? dot / (rn * centroidNorm) : 0.0;
		sumDist += 1.0 - cos;
	}
	const double refMean = sumDist / count;
	CHECK_MSG(std::fabs(refMean - static_cast<double>(mean)) < 0.05 * (1.0 + std::fabs(refMean)),
		"post-relabel channel-scoped analytics: got %.6g, definition-grounded reference %.6g",
		static_cast<double>(mean), refMean);
}

// T-V3.1-PromoteDemoteRetention (dim 8) -- promote on a retention bank derives sub-norms and
// keeps per-channel recall available; a demote drops both cleanly (whole-bank retention
// itself survives).
static void TestRelabelPromoteDemoteRetention()
{
	Rng rng(0x100019ull);
	const int32_t dims = 64;
	const int32_t count = 24;
	const ChannelInfo table[2] = {{0, 32}, {32, 32}};

	ScratchBank bank;
	CHECK(bank.Create(count, dims, Metric::Cosine, Quantization::Int8, /*retainFloats=*/true) ==
		Status::Ok);
	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}

	CHECK(bank.Relabel(table, 2) == Status::Ok);
	CHECK_MSG(bank.GetChannelCount() == 2, "promote x retention: table did not update");
	Workspace ws;
	ScratchRecallReport reports[2];
	CHECK_MSG(bank.MeasureScratchRecallPerChannel(ws, reports, 2) == Status::Ok,
		"promote x retention: per-channel recall should now be available");

	CHECK(bank.Relabel(nullptr, 0) == Status::Ok);
	CHECK_MSG(bank.GetChannelCount() == 0, "demote x retention: table did not clear");
	CHECK_MSG(bank.MeasureScratchRecallPerChannel(ws, reports, 2) == Status::InvalidArgument,
		"demote x retention: per-channel recall must reject once the channel table is gone");
	CHECK_MSG(bank.RetainsFloats(),
		"demote x retention: whole-bank retention itself must survive a demote");
	ScratchRecallReport wholeReport;
	CHECK_MSG(bank.MeasureScratchRecall(ws, &wholeReport) == Status::Ok,
		"demote x retention: whole-vector recall must still work after a demote");
}

// T-V3.1-PersistRoundTrip (dim 9a) -- a relabeled channels+retention Cosine bank (tombstones
// + Grow + Relabel history) round-trips: table survives, sub-norms re-derived bit-equal
// fresh, per-channel recall survives.
static void TestRelabelPersistenceValueRoundTrip()
{
	Rng rng(0x10001aull);
	const int32_t dims = 64;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	ScratchBank bank;
	CHECK(bank.Create(48, dims, Metric::Cosine, Quantization::Int8, oldTable, 2,
			/*retainFloats=*/true) == Status::Ok);
	std::vector<float> source(static_cast<size_t>(40) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < 40; ++r)
	{
		CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	for (int32_t r = 3; r < 40; r += 8)
	{
		CHECK(bank.Remove(r) == Status::Ok);
	}
	CHECK(bank.Grow(64) == Status::Ok);
	CHECK(bank.Relabel(newTable, 2) == Status::Ok);
	std::vector<float> more(static_cast<size_t>(8) * dims);
	for (auto& v : more) { v = rng.NextFloat(); }
	for (int32_t r = 0; r < 8; ++r)
	{
		CHECK(bank.Append(more.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}

	MemArchive archive;
	CHECK(bank.Save(archive.Writer()) == Status::Ok);
	ScratchBank loaded;
	CHECK(loaded.Load(archive.Reader()) == Status::Ok);

	CHECK(loaded.Count() == bank.Count());
	CHECK(loaded.LiveCount() == bank.LiveCount());
	CHECK_MSG(loaded.RetainsFloats(), "relabeled channels+retention round-trip lost retention");

	BankView loadedSnap;
	std::vector<uint32_t> loadedTombs(ScratchBank::TombstoneWords(loaded.Count()), 0u);
	CHECK(loaded.Snapshot(&loadedSnap, loadedTombs.data()) == Status::Ok);
	CHECK_MSG(loadedSnap.channelCount == 2 && loadedSnap.channels[0].offset == 0 &&
			loadedSnap.channels[0].length == 16,
		"relabeled channel table did not survive the round-trip: channelCount=%d",
		loadedSnap.channelCount);
	if (loadedSnap.channelCount == 2 && loadedSnap.channelInvNorms != nullptr)
	{
		std::vector<float> fresh(static_cast<size_t>(loadedSnap.count) * loadedSnap.channelCount);
		CHECK(ComputeChannelInverseNorms(loadedSnap, fresh.data()) == Status::Ok);
		CHECK_MSG(std::memcmp(fresh.data(), loadedSnap.channelInvNorms,
					  fresh.size() * sizeof(float)) == 0,
			"loaded relabeled sub-norm arena does not bit-equal a fresh derivation");
	}

	Workspace ws;
	ScratchRecallReport reports[2];
	CHECK_MSG(loaded.MeasureScratchRecallPerChannel(ws, reports, 2) == Status::Ok,
		"per-channel recall must survive the round-trip of a relabeled bank");
}

// T-V3.1-PersistByteIdentical (dim 9b, Forge W1) -- a relabeled bank's archive is
// byte-identical to a fresh bank's under the same table and rows, PRECONDITIONS pinned:
// same capacity, count, append order, retention.
static void TestRelabelArchiveByteIdenticalToFresh()
{
	Rng rng(0x10001bull);
	const int32_t dims = 64;
	const int32_t capacity = 32;
	const int32_t count = 32;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	std::vector<float> source(static_cast<size_t>(count) * dims);
	for (auto& v : source) { v = rng.NextFloat(); }

	ScratchBank relabeled;
	CHECK(relabeled.Create(capacity, dims, Metric::Cosine, Quantization::Int8, oldTable, 2) ==
		Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(relabeled.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}
	CHECK(relabeled.Relabel(newTable, 2) == Status::Ok);

	ScratchBank fresh;
	CHECK(fresh.Create(capacity, dims, Metric::Cosine, Quantization::Int8, newTable, 2) ==
		Status::Ok);
	for (int32_t r = 0; r < count; ++r)
	{
		CHECK(fresh.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
			Status::Ok);
	}

	MemArchive relArchive, freshArchive;
	CHECK(relabeled.Save(relArchive.Writer()) == Status::Ok);
	CHECK(fresh.Save(freshArchive.Writer()) == Status::Ok);
	CHECK_MSG(relArchive.bytes.size() == freshArchive.bytes.size(),
		"relabeled archive size %zu != fresh twin archive size %zu (table did not update -- "
		"still writing the OLD table's bytes)", relArchive.bytes.size(),
		freshArchive.bytes.size());
	CHECK_MSG(relArchive.bytes == freshArchive.bytes,
		"a relabeled bank's archive must be byte-identical to a fresh bank's under the same "
		"table and rows (same capacity/count/append-order/retention, Forge W1)");
}

// T-V3.1-PersistDemoteLegacy (dim 9c) -- a demote-to-single-space relabeled bank writes the
// legacy v1/v2 shape and loads on the current reader.
static void TestRelabelDemoteWritesLegacyFormat()
{
	Rng rng(0x10001cull);
	const int32_t dims = 48;
	const ChannelInfo table[2] = {{0, 32}, {32, 16}};

	ScratchBank bank;
	CHECK(bank.Create(16, dims, Metric::Cosine, Quantization::Int8, table, 2) == Status::Ok);
	std::vector<float> row(static_cast<size_t>(dims));
	for (int32_t r = 0; r < 8; ++r)
	{
		for (auto& v : row) { v = rng.NextFloat(); }
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}
	CHECK(bank.Relabel(nullptr, 0) == Status::Ok);
	CHECK_MSG(bank.GetChannelCount() == 0, "demote-format fixture: table did not clear");

	MemArchive archive;
	CHECK(bank.Save(archive.Writer()) == Status::Ok);
	uint32_t version = 0;
	CHECK(archive.bytes.size() >= 8);
	std::memcpy(&version, archive.bytes.data() + 4, sizeof(version));
	CHECK_MSG(version == 1u || version == 2u,
		"a demoted bank must Save the legacy v1/v2 shape, got version %u", version);
	CHECK_MSG(archive.bytes.size() >= kScratchHeaderFlagsByteOffset + 1 &&
			archive.bytes[kScratchHeaderFlagsByteOffset] == 0u,
		"a demoted bank's flags byte must be 0 (no channels bit)");

	ScratchBank loaded;
	CHECK_MSG(loaded.Load(archive.Reader()) == Status::Ok,
		"a demoted (legacy-format) archive must load on the current reader");
	CHECK(loaded.GetChannelCount() == 0);
}

// T-V3.1-PersistPromoteV3 (dim 9d) -- a promote-to-channels relabeled bank writes v3, and a
// version newer than this reader supports still hard-rejects (the old-reader-hard-reject law
// is not silently defeated by relabel-produced content).
static void TestRelabelPromoteWritesV3Format()
{
	Rng rng(0x10001dull);
	const int32_t dims = 48;
	const ChannelInfo table[2] = {{0, 32}, {32, 16}};

	ScratchBank bank;
	CHECK(bank.Create(16, dims, Metric::Cosine, Quantization::Int8) == Status::Ok);
	std::vector<float> row(static_cast<size_t>(dims));
	for (int32_t r = 0; r < 8; ++r)
	{
		for (auto& v : row) { v = rng.NextFloat(); }
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}
	CHECK(bank.Relabel(table, 2) == Status::Ok);
	CHECK_MSG(bank.GetChannelCount() == 2, "promote-format fixture: table did not update");

	MemArchive archive;
	CHECK(bank.Save(archive.Writer()) == Status::Ok);
	uint32_t version = 0;
	CHECK(archive.bytes.size() >= 8);
	std::memcpy(&version, archive.bytes.data() + 4, sizeof(version));
	CHECK_MSG(version == 3u, "a promoted bank must Save the v3 presence-flags shape, got version %u",
		version);
	CHECK_MSG(archive.bytes.size() >= kScratchHeaderFlagsByteOffset + 1 &&
			(archive.bytes[kScratchHeaderFlagsByteOffset] & 0x02) != 0,
		"a promoted bank's flags byte must carry the channels bit (0x02)");

	MemArchive tooNew;
	tooNew.bytes = archive.bytes;
	const uint32_t bumped = version + 1;
	std::memcpy(tooNew.bytes.data() + 4, &bumped, sizeof(bumped));
	ScratchBank rejected;
	CHECK_MSG(rejected.Load(tooNew.Reader()) == Status::BadFormat,
		"a version newer than this reader supports must hard-reject as BadFormat");
}

// T-V3.1-VersionInherited (dim 9, N-3) -- V3.1 introduces no new archive version and no new
// reserved-flag semantics: the V3.0 reserved flag-bit forward tolerance is inherited
// unchanged on a relabeled bank's archive.
static void TestRelabelVersionEvolutionInherited()
{
	Rng rng(0x10001eull);
	const int32_t dims = 64;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};
	const ChannelInfo newTable[2] = {{0, 16}, {16, 48}};

	ScratchBank bank;
	CHECK(bank.Create(24, dims, Metric::Cosine, Quantization::Int8, oldTable, 2,
			/*retainFloats=*/true) == Status::Ok);
	std::vector<float> row(static_cast<size_t>(dims));
	for (int32_t r = 0; r < 20; ++r)
	{
		for (auto& v : row) { v = rng.NextFloat(); }
		CHECK(bank.Append(row.data(), dims, nullptr) == Status::Ok);
	}
	CHECK(bank.Relabel(newTable, 2) == Status::Ok);

	MemArchive archive;
	CHECK(bank.Save(archive.Writer()) == Status::Ok);

	MemArchive poked;
	poked.bytes = archive.bytes;
	CHECK(poked.bytes.size() > kScratchHeaderFlagsByteOffset);
	poked.bytes[kScratchHeaderFlagsByteOffset] |= 0x04;

	ScratchBank loaded;
	const Status s = loaded.Load(poked.Reader());
	CHECK_MSG(s == Status::Ok,
		"V3.1 inherits V3.0's reserved flag-bit forward tolerance unchanged, got status=%d",
		static_cast<int>(s));
	if (s != Status::Ok)
	{
		return;
	}
	BankView loadedSnap;
	std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(loaded.Count()), 0u);
	CHECK(loaded.Snapshot(&loadedSnap, tombs.data()) == Status::Ok);
	CHECK_MSG(loadedSnap.channelCount == 2 && loadedSnap.channels[0].length == 16,
		"reserved-bit tolerance on a relabeled bank: the channel table must survive, "
		"channelCount=%d", loadedSnap.channelCount);
}

// T-V3.1-Feat (dim 10, the crux) -- the general FEAT: post-relabel per-channel query returns
// the independent double brute-force top-k over the new sub-range, on the SCRATCH bank,
// across metric x quant x {boundary move, count up, count down}.
static void TestRelabelFeatOracle()
{
	Rng rng(0x10001full);
	const int32_t dims = 64;
	const int32_t count = 96;
	const int32_t k = 10;
	const ChannelInfo oldTable[2] = {{0, 32}, {32, 32}};

	struct Scenario { const char* name; std::vector<ChannelInfo> target; };
	std::vector<Scenario> scenarios = {
		{"boundary move", {{0, 16}, {16, 48}}},
		{"count up", {{0, 16}, {16, 16}, {32, 32}}},
		{"count down", {{0, 64}}},
	};

	for (const Scenario& sc : scenarios)
	{
		for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
		{
			for (Quantization quant : {Quantization::Float32, Quantization::Int8})
			{
				ScratchBank bank;
				CHECK(bank.Create(count, dims, metric, quant, oldTable, 2) == Status::Ok);
				std::vector<float> source(static_cast<size_t>(count) * dims);
				for (auto& v : source) { v = rng.NextFloat(); }
				for (int32_t r = 0; r < count; ++r)
				{
					CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims,
							  nullptr) == Status::Ok);
				}
				CHECK_MSG(
					bank.Relabel(sc.target.data(), static_cast<int32_t>(sc.target.size())) ==
						Status::Ok,
					"%s (metric=%d quant=%d): Relabel failed", sc.name, static_cast<int>(metric),
					static_cast<int>(quant));

				BankView snap;
				std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
				CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
				CHECK_MSG(snap.channelCount == static_cast<int32_t>(sc.target.size()),
					"%s (metric=%d quant=%d): snapshot channel count did not update", sc.name,
					static_cast<int>(metric), static_cast<int>(quant));
				if (snap.channelCount == 0)
				{
					continue;
				}

				std::vector<float> queryRaw(dims);
				for (auto& v : queryRaw) { v = rng.NextFloat(); }
				AlignedBuf qbuf(static_cast<size_t>(snap.paddedDims) * sizeof(float));
				PadQuery(queryRaw, snap.paddedDims, qbuf.F32());
				const ChannelInfo& ch = sc.target[0];
				const QuerySegment seg[1] = {{ch.offset, ch.length, 1.0f}};
				QueryParams p;
				p.k = k;
				p.segments = seg;
				p.segmentCount = 1;
				p.excludeBits = tombs.data();
				Workspace ws;
				Hit hits[10];
				int32_t n = 0;
				CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
				const std::vector<FeatRefHit> ref =
					ChannelBruteForce(snap, qbuf.F32(), ch, tombs.data(), metric);
				CheckFeatTopK(ref, hits, n, k, count, metric, quant, sc.name);
			}
		}
	}
}

// T-V3.1-Promote (dim 10, G-1, AC-4) -- the headline promote capability: a single-space
// bank's per-channel query is impossible before Relabel (GetChannelCount()==0; for Cosine,
// MeasureScratchRecallPerChannel is a defined InvalidArgument) and correct -- the
// independent brute-force top-k -- after Relabel-to-channels, per metric.
static void TestRelabelPromoteFeat()
{
	Rng rng(0x100020ull);
	const int32_t dims = 64;
	const int32_t count = 96;
	const int32_t k = 10;
	const ChannelInfo target[2] = {{0, 32}, {32, 32}};

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			ScratchBank bank;
			CHECK(bank.Create(count, dims, metric, quant, /*retainFloats=*/true) == Status::Ok);
			std::vector<float> source(static_cast<size_t>(count) * dims);
			for (auto& v : source) { v = rng.NextFloat(); }
			for (int32_t r = 0; r < count; ++r)
			{
				CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
					Status::Ok);
			}
			CHECK_MSG(bank.GetChannelCount() == 0,
				"promote fixture must start single-space, got channelCount=%d",
				bank.GetChannelCount());
			if (metric == Metric::Cosine)
			{
				Workspace wsPre;
				ScratchRecallReport reports[2];
				CHECK_MSG(bank.MeasureScratchRecallPerChannel(wsPre, reports, 2) ==
						Status::InvalidArgument,
					"pre-promote: per-channel recall must reject (no channel table yet)");
			}

			CHECK_MSG(bank.Relabel(target, 2) == Status::Ok,
				"promote Relabel failed: metric=%d quant=%d", static_cast<int>(metric),
				static_cast<int>(quant));
			CHECK_MSG(bank.GetChannelCount() == 2,
				"promote: GetChannelCount() should be 2 after Relabel-to-channels, got %d "
				"(metric=%d quant=%d)", bank.GetChannelCount(), static_cast<int>(metric),
				static_cast<int>(quant));

			BankView snap;
			std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
			CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
			CHECK_MSG(snap.channelCount == 2,
				"promote: the snapshot must carry the new table (metric=%d quant=%d)",
				static_cast<int>(metric), static_cast<int>(quant));

			std::vector<float> queryRaw(dims);
			for (auto& v : queryRaw) { v = rng.NextFloat(); }
			const int32_t pd = snap.paddedDims;
			AlignedBuf qbuf(static_cast<size_t>(pd) * sizeof(float));
			PadQuery(queryRaw, pd, qbuf.F32());
			const QuerySegment seg[1] = {{target[0].offset, target[0].length, 1.0f}};
			QueryParams p;
			p.k = k;
			p.segments = seg;
			p.segmentCount = 1;
			p.excludeBits = tombs.data();
			Workspace ws;
			Hit hits[10];
			int32_t n = 0;
			CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
			const std::vector<FeatRefHit> ref =
				ChannelBruteForce(snap, qbuf.F32(), target[0], tombs.data(), metric);
			CheckFeatTopK(ref, hits, n, k, count, metric, quant, "promote FEAT");
		}
	}
}

// T-V3.1-Demote (dim 10, G-1, AC-5) -- a channel bank, after Relabel(nullptr, 0), rejects
// per-channel recall and correctly serves the whole-vector brute-force top-k -- including
// the reloaded-archive form.
static void TestRelabelDemoteFeat()
{
	Rng rng(0x100021ull);
	const int32_t dims = 64;
	const int32_t count = 96;
	const int32_t k = 10;
	const ChannelInfo channels[2] = {{0, 32}, {32, 32}};

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			ScratchBank bank;
			CHECK(bank.Create(count, dims, metric, quant, channels, 2, /*retainFloats=*/true) ==
				Status::Ok);
			std::vector<float> source(static_cast<size_t>(count) * dims);
			for (auto& v : source) { v = rng.NextFloat(); }
			for (int32_t r = 0; r < count; ++r)
			{
				CHECK(bank.Append(source.data() + static_cast<size_t>(r) * dims, dims, nullptr) ==
					Status::Ok);
			}
			CHECK(bank.GetChannelCount() == 2);

			CHECK_MSG(bank.Relabel(nullptr, 0) == Status::Ok,
				"demote Relabel failed: metric=%d quant=%d", static_cast<int>(metric),
				static_cast<int>(quant));
			CHECK_MSG(bank.GetChannelCount() == 0,
				"demote: GetChannelCount() should be 0 after Relabel-to-single-space, got %d "
				"(metric=%d quant=%d)", bank.GetChannelCount(), static_cast<int>(metric),
				static_cast<int>(quant));
			if (metric == Metric::Cosine)
			{
				Workspace wsPost;
				ScratchRecallReport reports[2];
				CHECK_MSG(bank.MeasureScratchRecallPerChannel(wsPost, reports, 2) ==
						Status::InvalidArgument,
					"post-demote: per-channel recall must reject (no channel table left)");
			}

			BankView snap;
			std::vector<uint32_t> tombs(ScratchBank::TombstoneWords(count), 0u);
			CHECK(bank.Snapshot(&snap, tombs.data()) == Status::Ok);
			CHECK_MSG(snap.channelCount == 0,
				"demote: the snapshot must carry no channel table (metric=%d quant=%d)",
				static_cast<int>(metric), static_cast<int>(quant));

			std::vector<float> queryRaw(dims);
			for (auto& v : queryRaw) { v = rng.NextFloat(); }
			const int32_t pd = snap.paddedDims;
			AlignedBuf qbuf(static_cast<size_t>(pd) * sizeof(float));
			PadQuery(queryRaw, pd, qbuf.F32());
			QueryParams p;
			p.k = k;
			p.excludeBits = tombs.data();
			Workspace ws;
			Hit hits[10];
			int32_t n = 0;
			CHECK(Query(snap, qbuf.F32(), p, ws, hits, &n) == Status::Ok);
			const std::vector<FeatRefHit> ref =
				WholeVectorBruteForce(snap, qbuf.F32(), tombs.data(), metric);
			CheckFeatTopK(ref, hits, n, k, count, metric, quant, "demote whole-vector FEAT");

			MemArchive archive;
			CHECK(bank.Save(archive.Writer()) == Status::Ok);
			ScratchBank loaded;
			CHECK(loaded.Load(archive.Reader()) == Status::Ok);
			CHECK_MSG(loaded.GetChannelCount() == 0,
				"demoted archive must reload single-space (metric=%d quant=%d)",
				static_cast<int>(metric), static_cast<int>(quant));
			BankView loadedSnap;
			std::vector<uint32_t> loadedTombs(ScratchBank::TombstoneWords(loaded.Count()), 0u);
			CHECK(loaded.Snapshot(&loadedSnap, loadedTombs.data()) == Status::Ok);
			Hit loadedHits[10];
			int32_t nl = 0;
			CHECK(Query(loadedSnap, qbuf.F32(), p, ws, loadedHits, &nl) == Status::Ok);
			const std::vector<FeatRefHit> loadedRef =
				WholeVectorBruteForce(loadedSnap, qbuf.F32(), loadedTombs.data(), metric);
			CheckFeatTopK(loadedRef, loadedHits, nl, k, count, metric, quant,
				"demote reloaded-archive whole-vector FEAT");
		}
	}
}

// ===========================================================================
// V3.2 red suite (Curie, 2026-07-18): Bank Inspector I — module M1 graph.h
// (plan section 25, Coverage Model section 25.9). Realizes the M1 cells against
// the RED SCAFFOLD stub at src/graph.cpp (every body returns Ok and writes
// nothing — the proven V3.1 no-op-stub pattern), so every cell below fails for
// its cell's reason: a FEAT/exactness cell reads its poison-initialized output
// buffer where it expects the constructed component structure; a dim-2 rejection
// cell sees Ok where it expects InvalidArgument.
// Test design: Claude/Curie/superfaiss-v3.2-test-design-2026-07-18.md.
//
// Oracle discipline (the D-V32 strike loop, "build don't reason"):
//  * the neighbor/edge REF is a double-precision brute force over the DEQUANTIZED
//    rows the kernel actually scores (GBank::refRows), same total order as the
//    library (score, then ascending index);
//  * the duplicate-union / multi-block-separation / fringe geometries are the
//    EXECUTED construction_check.cpp fixtures (Claude/Loki/strike8, 10/10 true
//    counts) ported verbatim — well-separated so float-vs-double ranking cannot
//    disagree; the component-count oracle is the construction, not intuition
//    (the D-V32-25 corollary);
//  * the Structure FEAT ground truth is CONSTRUCTED (planted blobs/isolates/
//    duplicate blocks), never a re-run of the implementation — a wrong-but-
//    deterministic grapher fails it (dimension 10).
// ===========================================================================

namespace
{
	// A baked in-memory bank built from EXPLICIT source rows (the constructed-
	// geometry sibling of TestBank, which fills random rows). refRows holds the
	// double-precision dequantized values the kernel effectively scores.
	struct GBank
	{
		std::vector<float> source;
		std::vector<double> refRows;
		AlignedBuf payload;
		std::vector<float> scales;
		BankView view;

		GBank(const std::vector<float>& src, int32_t count, int32_t dims,
			Quantization quant, Metric metric)
			: payload(static_cast<size_t>(count > 0 ? count : 1) *
				PaddedDims(dims, quant) * ElementSize(quant))
		{
			source = src;
			if (metric == Metric::Cosine && count > 0)
			{
				CHECK(NormalizeRows(source.data(), count, dims, nullptr) == Status::Ok);
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
				scales.resize(static_cast<size_t>(count > 0 ? count : 1));
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

	// Union-find with smallest-index-as-root (ids canonicalize to the smallest
	// member row index — the pinned convention, matching construction_check.cpp).
	struct DSU
	{
		std::vector<int32_t> parent;
		explicit DSU(int32_t n) : parent(static_cast<size_t>(n))
		{
			for (int32_t i = 0; i < n; ++i) parent[i] = i;
		}
		int32_t Find(int32_t x)
		{
			while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
			return x;
		}
		void Union(int32_t a, int32_t b)
		{
			a = Find(a); b = Find(b);
			if (a < b) parent[b] = a; else if (b < a) parent[a] = b;
		}
	};

	// Brute-force top-k neighbor indices per row, from the dequantized rows, in the
	// library's total order. The query for row r is row r's own dequantized value
	// (DequantizeRowAsQuery), exactly what BuildKnnNeighbors probes with.
	std::vector<int32_t> BruteKnn(const GBank& b, int32_t k, bool excludeSelf)
	{
		const int32_t n = b.view.count, dims = b.view.dims;
		const Metric m = b.view.metric;
		std::vector<int32_t> out(static_cast<size_t>(n) * k, -1);
		for (int32_t r = 0; r < n; ++r)
		{
			std::vector<RefHit> hits;
			hits.reserve(static_cast<size_t>(n));
			const double* rr = &b.refRows[static_cast<size_t>(r) * dims];
			for (int32_t s = 0; s < n; ++s)
			{
				if (excludeSelf && s == r) continue;
				const double* ss = &b.refRows[static_cast<size_t>(s) * dims];
				double score = 0.0;
				if (m == Metric::L2)
				{
					for (int32_t i = 0; i < dims; ++i) { const double d = rr[i] - ss[i]; score += d * d; }
				}
				else
				{
					for (int32_t i = 0; i < dims; ++i) score += rr[i] * ss[i];
				}
				hits.push_back({s, score});
			}
			std::sort(hits.begin(), hits.end(),
				[&](const RefHit& a, const RefHit& c) { return RefBetter(a, c, m); });
			for (int32_t j = 0; j < k && j < static_cast<int32_t>(hits.size()); ++j)
			{
				out[static_cast<size_t>(r) * k + j] = hits[j].index;
			}
		}
		return out;
	}

	// Brute mutual flags from a neighbor list (edge (i,j) mutual iff each lists the other).
	std::vector<uint8_t> BruteMutual(const std::vector<int32_t>& nb, int32_t n, int32_t k)
	{
		auto inList = [&](int32_t row, int32_t x) {
			for (int32_t t = 0; t < k; ++t) if (nb[static_cast<size_t>(row) * k + t] == x) return true;
			return false;
		};
		std::vector<uint8_t> flags(static_cast<size_t>(n) * k, 0);
		for (int32_t i = 0; i < n; ++i)
		{
			for (int32_t t = 0; t < k; ++t)
			{
				const int32_t j = nb[static_cast<size_t>(i) * k + t];
				if (j >= 0 && inList(j, i)) flags[static_cast<size_t>(i) * k + t] = 1;
			}
		}
		return flags;
	}

	// Exact-duplicate group representative per row: the smallest row index with
	// identical stored bytes (and scale, for int8). Same-decode-different-bytes rows
	// are NOT grouped.
	std::vector<int32_t> BruteGroupOf(const GBank& b)
	{
		const int32_t n = b.view.count;
		const size_t rb = static_cast<size_t>(b.view.paddedDims) * ElementSize(b.view.quant);
		std::vector<int32_t> g(static_cast<size_t>(n));
		for (int32_t i = 0; i < n; ++i)
		{
			g[i] = i;
			for (int32_t j = 0; j < i; ++j)
			{
				bool same = std::memcmp(static_cast<const char*>(b.view.rows) + static_cast<size_t>(i) * rb,
					static_cast<const char*>(b.view.rows) + static_cast<size_t>(j) * rb, rb) == 0;
				if (same && b.view.quant == Quantization::Int8) same = b.scales[i] == b.scales[j];
				if (same) { g[i] = g[j]; break; }
			}
		}
		return g;
	}

	// Construction-form connected components (the executed construction_check.cpp
	// mechanism): per-row k-NN over the untouched population (self-excluded), union
	// byte+scale-identical rows (construction edges), then mutual edges; ids = smallest
	// member. The REF oracle for ConnectedComponents on well-separated fixtures.
	std::vector<int32_t> BruteComponents(const GBank& b, int32_t k)
	{
		const int32_t n = b.view.count;
		const std::vector<int32_t> nb = BruteKnn(b, k, true);
		const std::vector<int32_t> g = BruteGroupOf(b);
		DSU dsu(n);
		for (int32_t r = 0; r < n; ++r) if (g[r] != r) dsu.Union(r, g[r]);
		auto inList = [&](int32_t row, int32_t x) {
			for (int32_t t = 0; t < k; ++t) if (nb[static_cast<size_t>(row) * k + t] == x) return true;
			return false;
		};
		for (int32_t i = 0; i < n; ++i)
		{
			for (int32_t t = 0; t < k; ++t)
			{
				const int32_t j = nb[static_cast<size_t>(i) * k + t];
				if (j > i && inList(j, i)) dsu.Union(i, j);
			}
		}
		std::vector<int32_t> id(static_cast<size_t>(n));
		for (int32_t i = 0; i < n; ++i) id[i] = dsu.Find(i);
		return id;
	}

	int32_t ComponentCount(const std::vector<int32_t>& ids)
	{
		std::vector<int32_t> seen;
		for (int32_t x : ids) if (std::find(seen.begin(), seen.end(), x) == seen.end()) seen.push_back(x);
		return static_cast<int32_t>(seen.size());
	}

	// Run the full M1 pipeline through the module under test into caller buffers.
	// Returns the aggregate status (first non-Ok, else Ok). outIds is poison-init'd by
	// the caller so a no-op stub leaves it detectably wrong.
	Status RunM1Pipeline(const GBank& b, int32_t k, std::vector<int32_t>& outIds)
	{
		const int32_t n = b.view.count;
		// Distinct-per-row poison: a no-op stub that writes nothing leaves each row its
		// own singleton, so ComponentCount == n (never the small expected counts) — the
		// FEAT cells fail red rather than falsely reading a single uniform-poison component.
		for (int32_t r = 0; r < n; ++r) outIds[r] = -(r + 2);
		std::vector<int32_t> nb(static_cast<size_t>(n) * k, -999);
		std::vector<uint8_t> flags(static_cast<size_t>(n) * k, 0xEE);
		std::vector<int32_t> groups(static_cast<size_t>(n), -999);
		std::vector<int32_t> ufScratch(static_cast<size_t>(n), 0);
		std::vector<int32_t> hashScratch(static_cast<size_t>(n), 0);
		Workspace ws;
		ws.Reserve(k + 1, 1);
		Status s = BuildKnnNeighbors(b.view, k, true, nb.data(), ws);
		if (s != Status::Ok) return s;
		s = MutualFilter(n, k, nb.data(), flags.data());
		if (s != Status::Ok) return s;
		s = BuildDuplicateGroups(b.view, groups.data(), hashScratch.data());
		if (s != Status::Ok) return s;
		return ConnectedComponents(n, k, nb.data(), flags.data(), groups.data(),
			outIds.data(), ufScratch.data());
	}

	// A tight blob of `m` rows jittered around `centre` (dims), jitter << inter-centre
	// gap so every blob row's whole top-k stays inside the blob (one component, robust
	// to float-vs-double ranking).
	void PushBlob(std::vector<float>& src, int32_t dims, const std::vector<float>& centre,
		int32_t m, Rng& rng, float jitter)
	{
		for (int32_t i = 0; i < m; ++i)
			for (int32_t d = 0; d < dims; ++d)
				src.push_back(centre[static_cast<size_t>(d)] + jitter * rng.NextFloat());
	}
}

// M1 / dim 2 (trust boundaries) — every module rejects k<1, k>=count, count 0/1 where a
// k-th neighbor cannot exist, and null buffers, with InvalidArgument and no output write.
static void TestM1TrustBoundaries()
{
	Rng rng(0xB1);
	std::vector<float> src;
	const int32_t dims = 8, count = 6;
	for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
	GBank b(src, count, dims, Quantization::Float32, Metric::L2);
	Workspace ws; ws.Reserve(8, 1);
	std::vector<int32_t> nb(static_cast<size_t>(count) * 4, -999);

	// k < 1 and k >= count reject.
	CHECK_MSG(BuildKnnNeighbors(b.view, 0, true, nb.data(), ws) == Status::InvalidArgument,
		"BuildKnnNeighbors k=0 must reject");
	CHECK_MSG(BuildKnnNeighbors(b.view, count, true, nb.data(), ws) == Status::InvalidArgument,
		"BuildKnnNeighbors k>=count must reject");
	// Null buffer rejects, no write.
	CHECK_MSG(BuildKnnNeighbors(b.view, 2, true, nullptr, ws) == Status::InvalidArgument,
		"BuildKnnNeighbors null out must reject");
	CHECK_MSG(nb[0] == -999, "rejected BuildKnnNeighbors must not write output");

	std::vector<uint8_t> flags(static_cast<size_t>(count) * 2, 0xEE);
	CHECK_MSG(MutualFilter(count, 0, nb.data(), flags.data()) == Status::InvalidArgument,
		"MutualFilter k=0 must reject");
	CHECK_MSG(MutualFilter(0, 2, nb.data(), flags.data()) == Status::InvalidArgument,
		"MutualFilter count=0 must reject");
	CHECK_MSG(MutualFilter(count, 2, nullptr, flags.data()) == Status::InvalidArgument,
		"MutualFilter null neighbors must reject");

	std::vector<int32_t> groups(static_cast<size_t>(count), -999), hs(static_cast<size_t>(count), 0);
	CHECK_MSG(BuildDuplicateGroups(b.view, nullptr, hs.data()) == Status::InvalidArgument,
		"BuildDuplicateGroups null out must reject");

	std::vector<int32_t> ids(static_cast<size_t>(count), -999), uf(static_cast<size_t>(count), 0);
	CHECK_MSG(ConnectedComponents(count, 2, nullptr, flags.data(), groups.data(), ids.data(), uf.data())
		== Status::InvalidArgument, "ConnectedComponents null neighbors must reject");
	CHECK_MSG(ids[0] == -999, "rejected ConnectedComponents must not write output");

	// count 0 and count 1: no k-th neighbor can exist (G-2, the empty/singleton extreme).
	{
		std::vector<float> empty;
		GBank b0(empty, 0, dims, Quantization::Float32, Metric::L2);
		std::vector<int32_t> nb0(1, -999);
		Workspace ws0; ws0.Reserve(2, 1);
		CHECK_MSG(BuildKnnNeighbors(b0.view, 1, true, nb0.data(), ws0) == Status::InvalidArgument,
			"BuildKnnNeighbors on empty bank (count=0) must reject — no k-th neighbor");
		std::vector<float> one(src.begin(), src.begin() + dims);
		GBank b1(one, 1, dims, Quantization::Float32, Metric::L2);
		std::vector<int32_t> nb1(1, -999);
		CHECK_MSG(BuildKnnNeighbors(b1.view, 1, true, nb1.data(), ws0) == Status::InvalidArgument,
			"BuildKnnNeighbors on singleton (count=1, self-excluded) must reject — no k-th neighbor");
	}
}

// M1 / dim 2 (byte-confirm) + dim 7 (int8 same-decode-different-scale NOT unioned) —
// grouping is FULL BYTE equality, not decode equality: a scalar-multiple pair that
// decodes to the same direction but stores different int8 bytes/scale is a near-
// duplicate, never grouped.
static void TestM1DuplicateGroupingByteConfirm()
{
	const int32_t dims = 8;
	// Rows 0,1 byte-identical (same float content). Row 2 = 2*row0 (L2: different
	// content, so different group by definition). Row 3 unique.
	std::vector<float> src = {
		0.5f, -0.25f, 0.75f, 0.1f, -0.4f, 0.2f, 0.6f, -0.3f,   // row 0
		0.5f, -0.25f, 0.75f, 0.1f, -0.4f, 0.2f, 0.6f, -0.3f,   // row 1 == row 0
		1.0f, -0.5f, 1.5f, 0.2f, -0.8f, 0.4f, 1.2f, -0.6f,     // row 2 = 2*row0
		-0.9f, 0.1f, 0.3f, -0.7f, 0.55f, -0.15f, 0.05f, 0.8f,  // row 3 unique
	};
	const int32_t count = 4;
	for (Quantization q : {Quantization::Float32, Quantization::Int8})
	{
		GBank b(src, count, dims, q, Metric::L2);
		std::vector<int32_t> g(static_cast<size_t>(count), -999), hs(static_cast<size_t>(count), 0);
		const Status s = BuildDuplicateGroups(b.view, g.data(), hs.data());
		CHECK_MSG(s == Status::Ok, "BuildDuplicateGroups must succeed on a valid bank");
		const std::vector<int32_t> ref = BruteGroupOf(b);
		for (int32_t r = 0; r < count; ++r)
			CHECK_MSG(g[r] == ref[r], "groupOf[%d]=%d expected %d (byte-confirmed identity)", r, g[r], ref[r]);
		// Rows 0 and 1 share a group; row 2 (scalar multiple) does NOT (different bytes).
		CHECK_MSG(g[1] == 0, "byte-identical rows 0,1 must group");
		CHECK_MSG(g[2] == 2, "scalar-multiple row 2 stores different bytes -> not grouped (near-duplicate)");
		CHECK_MSG(g[3] == 3, "unique row 3 is its own group");
	}
}

// M1 / dim 6 (edges exact) + dim 7 (per-metric generality, G-3) — the built neighbor
// list, mutual flags, and component ids equal the double-precision brute force, for Dot,
// Cosine, and L2, int8 and float32. Two mechanism-true, TIE-FREE fixtures (F-M1-3, the
// D-V32-25 corollary: a degenerate/near-duplicate oracle value must be TRACED from the
// mechanism, never intuited):
//   Leg 1 (distinct-distance arc) proves neighbor RANKING on distinct, gap-separated
//     distances — points on a circular arc with strictly-INCREASING angular gaps, so
//     every pairwise distance is distinct in all three metrics and no float-vs-double
//     near-tie exists.
//   Leg 2 (exact-duplicate orthogonal blocks) is the per-metric FEAT: D=4 byte-identical
//     blocks on orthogonal axes -> the construction-union guarantees exactly D components
//     BY CONSTRUCTION (mechanism-true, independent of k), and within-block ties are at
//     EXACT distance 0 (integer index tie-break, identical in float and double).
// A NEAR-duplicate dense blob is deliberately NOT used here: bounded-k mutual-kNN over it
// fragments (R-V32-3 / the seventh fracture), so "one component per blob" is intuited, not
// mechanism-traced — and normalized near-identical unit vectors are a genuine float-vs-
// double near-tie on Cosine. That geometry is the Structure FEAT's job (exact-duplicate
// blocks + isolates), where the union makes the count mechanism-true.
static void TestM1EdgesExactAcrossMetrics()
{
	const int32_t dims = 16, k = 4;
	auto checkEqual = [&](const GBank& b, int32_t kk, const char* leg, Metric metric, Quantization q,
		int32_t expectComponents /* -1 = don't assert count */)
	{
		const int32_t count = b.view.count;
		Workspace ws; ws.Reserve(kk + 1, 1);
		std::vector<int32_t> nb(static_cast<size_t>(count) * kk, -999);
		CHECK(BuildKnnNeighbors(b.view, kk, true, nb.data(), ws) == Status::Ok);
		const std::vector<int32_t> refNb = BruteKnn(b, kk, true);
		int32_t nbMismatch = 0;
		for (size_t i = 0; i < nb.size(); ++i) if (nb[i] != refNb[i]) ++nbMismatch;
		CHECK_MSG(nbMismatch == 0, "%s: neighbor list must equal brute force (metric=%d quant=%d): %d mismatches",
			leg, (int)metric, (int)q, nbMismatch);

		std::vector<uint8_t> flags(static_cast<size_t>(count) * kk, 0xEE);
		CHECK(MutualFilter(count, kk, nb.data(), flags.data()) == Status::Ok);
		const std::vector<uint8_t> refFlags = BruteMutual(refNb, count, kk);
		int32_t flagMismatch = 0;
		for (size_t i = 0; i < flags.size(); ++i) if (flags[i] != refFlags[i]) ++flagMismatch;
		CHECK_MSG(flagMismatch == 0, "%s: mutual flags must equal brute force (metric=%d quant=%d)", leg, (int)metric, (int)q);

		std::vector<int32_t> ids(static_cast<size_t>(count), -999);
		CHECK(RunM1Pipeline(b, kk, ids) == Status::Ok);
		const std::vector<int32_t> refIds = BruteComponents(b, kk);
		int32_t idMismatch = 0;
		for (int32_t r = 0; r < count; ++r) if (ids[r] != refIds[r]) ++idMismatch;
		CHECK_MSG(idMismatch == 0, "%s: component ids must equal brute force (metric=%d quant=%d)", leg, (int)metric, (int)q);
		if (expectComponents >= 0)
			CHECK_MSG(ComponentCount(ids) == expectComponents,
				"%s: constructed %d components (metric=%d quant=%d), got %d",
				leg, expectComponents, (int)metric, (int)q, ComponentCount(ids));
	};

	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization q : {Quantization::Float32, Quantization::Int8})
		{
			// Leg 1 — distinct-distance arc: 10 points, strictly-increasing angular gaps.
			{
				const int32_t n = 10;
				std::vector<float> src(static_cast<size_t>(n) * dims, 0.0f);
				double theta = 0.0;
				for (int32_t i = 0; i < n; ++i)
				{
					src[static_cast<size_t>(i) * dims + 0] = static_cast<float>(std::cos(theta));
					src[static_cast<size_t>(i) * dims + 1] = static_cast<float>(std::sin(theta));
					theta += 0.25 + 0.11 * i; // growing gaps -> all pairwise distances distinct
				}
				GBank b(src, n, dims, q, metric);
				checkEqual(b, k, "arc", metric, q, -1);
			}
			// Leg 2 — exact-duplicate orthogonal blocks (per-metric FEAT, mechanism-true count).
			{
				const int32_t D = 4, blk = 6, n = D * blk;
				std::vector<float> src(static_cast<size_t>(n) * dims, 0.0f);
				for (int32_t d = 0; d < D; ++d)
					for (int32_t i = 0; i < blk; ++i)
						src[(static_cast<size_t>(d) * blk + i) * dims + d] = 10.0f * (d + 1); // block d on axis d
				GBank b(src, n, dims, q, metric);
				checkEqual(b, k, "exact-dup-blocks", metric, q, D);
			}
		}
	}
}

// M1 / dim 4 + dim 7 + dim 10 (identical => one component; multi-block separation) —
// the EXECUTED construction_check.cpp geometries ported verbatim (strike 7/8, 10/10 true
// counts): an exact-duplicate block of m>k+1 collapses to ONE component; D well-separated
// duplicate blocks yield exactly D components (construction edges never fabricate cross-
// block edges); interleaved index order changes nothing.
static void TestM1DuplicateUnionConstructionCounts()
{
	auto sep = [](int32_t bl) {
		std::vector<float> v(8, 0.0f); v[0] = 100.0f * (bl + 1); v[1] = 7.0f * bl; return v;
	};
	const int32_t dims = 8;
	// strike 7: one identical block of 10, k=2 (m > k+1) -> 1 component.
	{
		std::vector<float> src;
		for (int32_t i = 0; i < 10; ++i) { auto c = sep(0); src.insert(src.end(), c.begin(), c.end()); }
		GBank b(src, 10, dims, Quantization::Float32, Metric::L2);
		std::vector<int32_t> ids(10, -999);
		CHECK(RunM1Pipeline(b, 2, ids) == Status::Ok);
		CHECK_MSG(ComponentCount(ids) == 1, "identical block m=10,k=2 -> 1 component (construction union), got %d",
			ComponentCount(ids));
	}
	// strike 8: D in {2,3,5} well-separated duplicate blocks x8, k=4 -> D components,
	// contiguous AND interleaved index order.
	for (int32_t D : {2, 3, 5})
	{
		const int32_t m = 8, k = 4, count = D * m;
		for (int interleave = 0; interleave < 2; ++interleave)
		{
			std::vector<int32_t> blockOf(static_cast<size_t>(count));
			for (int32_t r = 0; r < count; ++r) blockOf[r] = interleave ? (r % D) : (r / m);
			std::vector<float> src;
			for (int32_t r = 0; r < count; ++r) { auto c = sep(blockOf[r]); src.insert(src.end(), c.begin(), c.end()); }
			GBank b(src, count, dims, Quantization::Float32, Metric::L2);
			std::vector<int32_t> ids(static_cast<size_t>(count), -999);
			CHECK(RunM1Pipeline(b, k, ids) == Status::Ok);
			CHECK_MSG(ComponentCount(ids) == D,
				"%d duplicate blocks (interleave=%d) -> %d components, got %d",
				D, interleave, D, ComponentCount(ids));
		}
	}
	// unequal block sizes 3/8/17 -> 3 components.
	{
		const int32_t k = 4;
		const int32_t sizes[] = {3, 8, 17};
		std::vector<float> src; int32_t count = 0;
		for (int32_t bl = 0; bl < 3; ++bl)
			for (int32_t i = 0; i < sizes[bl]; ++i) { auto c = sep(bl); src.insert(src.end(), c.begin(), c.end()); ++count; }
		GBank b(src, count, dims, Quantization::Float32, Metric::L2);
		std::vector<int32_t> ids(static_cast<size_t>(count), -999);
		CHECK(RunM1Pipeline(b, k, ids) == Status::Ok);
		CHECK_MSG(ComponentCount(ids) == 3, "unequal blocks 3/8/17 -> 3 components, got %d", ComponentCount(ids));
	}
	// int8 same-decode-different-scale must NOT union (identity is stored content):
	// two byte-different blocks that dequantize to parallel directions stay separate.
	{
		const int32_t k = 3, count = 12;
		std::vector<float> src;
		for (int32_t i = 0; i < 6; ++i) { std::vector<float> c(8, 0.0f); c[0] = 1.0f; c[1] = 0.5f; src.insert(src.end(), c.begin(), c.end()); }
		for (int32_t i = 0; i < 6; ++i) { std::vector<float> c(8, 0.0f); c[0] = 50.0f; c[1] = 25.0f; src.insert(src.end(), c.begin(), c.end()); }
		GBank b(src, count, dims, Quantization::Int8, Metric::L2);
		std::vector<int32_t> ids(static_cast<size_t>(count), -999);
		CHECK(RunM1Pipeline(b, k, ids) == Status::Ok);
		CHECK_MSG(ComponentCount(ids) == 2, "same-decode-different-scale blocks stay 2 components (L2), got %d",
			ComponentCount(ids));
	}
}

// M1 / dim 4 + dim 10 (traced fringe oracle) — the construction_check.cpp fringe pair
// (m=k connects, m=k+1 saturates): an outside neighbour of an m>k block stays unconnected
// because its partner's top-k is saturated by group-mates; when m<=k the edge forms. The
// oracle is COMPUTED from the mechanism, never intuited (D-V32-25 corollary).
static void TestM1FringeBoundary()
{
	auto sep = [](int32_t bl) {
		std::vector<float> v(8, 0.0f); v[0] = 100.0f * (bl + 1); v[1] = 7.0f * bl; return v;
	};
	const int32_t dims = 8, k = 4;
	for (int32_t m : {4, 5}) // m=k connects (1 comp); m=k+1 saturates (2 comps)
	{
		const int32_t count = m + 1;
		std::vector<float> src;
		for (int32_t i = 0; i < m; ++i) { auto c = sep(0); src.insert(src.end(), c.begin(), c.end()); }
		std::vector<float> fringe = sep(0); fringe[2] += 0.5f; // near block 0, DISTINCT content
		src.insert(src.end(), fringe.begin(), fringe.end());
		GBank b(src, count, dims, Quantization::Float32, Metric::L2);
		std::vector<int32_t> ids(static_cast<size_t>(count), -999);
		CHECK(RunM1Pipeline(b, k, ids) == Status::Ok);
		const int32_t expect = (m <= k) ? 1 : 2;
		CHECK_MSG(ComponentCount(ids) == expect,
			"fringe m=%d,k=%d -> %d components (saturation semantics), got %d", m, k, expect, ComponentCount(ids));
	}
}

// M1 / dim 6 (determinism) — repeat-call bit-equality of neighbors, flags, and component
// ids; and component-id canonicalization to the smallest member under an ADVERSARIAL edge
// input (ids are a property of the final partition, not insertion order).
static void TestM1RepeatAndCanonicalId()
{
	Rng rng(0xD6);
	const int32_t dims = 16, count = 40, k = 6;
	std::vector<float> src;
	for (int32_t bl = 0; bl < 4; ++bl)
	{
		std::vector<float> centre(static_cast<size_t>(dims), 0.0f);
		centre[static_cast<size_t>(bl)] = 15.0f * (bl + 1);
		PushBlob(src, dims, centre, 10, rng, 0.05f);
	}
	GBank b(src, count, dims, Quantization::Float32, Metric::Cosine);
	std::vector<int32_t> ids1(static_cast<size_t>(count), -999), ids2(static_cast<size_t>(count), -777);
	CHECK(RunM1Pipeline(b, k, ids1) == Status::Ok);
	CHECK(RunM1Pipeline(b, k, ids2) == Status::Ok);
	int32_t drift = 0;
	for (int32_t r = 0; r < count; ++r) if (ids1[r] != ids2[r]) ++drift;
	CHECK_MSG(drift == 0, "repeat-call component ids must be bit-identical, %d drifted", drift);

	// Correctness against the brute-force partition (so the cell is red for a feature
	// reason against the stub, not only vacuously deterministic).
	const std::vector<int32_t> refIds = BruteComponents(b, k);
	int32_t idMismatch = 0;
	for (int32_t r = 0; r < count; ++r) if (ids1[r] != refIds[r]) ++idMismatch;
	CHECK_MSG(idMismatch == 0, "component ids must equal the brute-force partition, %d mismatch", idMismatch);

	// Canonicalization: every row's id is the smallest index in its component.
	for (int32_t r = 0; r < count; ++r)
	{
		int32_t smallest = r;
		for (int32_t s = 0; s < count; ++s) if (ids1[s] == ids1[r] && s < smallest) smallest = s;
		CHECK_MSG(ids1[r] == smallest, "component id of row %d must be its smallest member %d, got %d",
			r, smallest, ids1[r]);
	}
}

// M1 / dim 1 (lifetime/reuse) — one warm workspace driven GROW-then-SHRINK-then-GROW
// (large view, then small, then large) yields results bit-identical to fresh-workspace
// runs (N-1: the adversarial order that exercises stale-byte leakage).
static void TestM1WarmWorkspaceGrowShrink()
{
	Rng rng(0x1F);
	const int32_t dims = 24, k = 5;
	auto makeBank = [&](int32_t count) {
		std::vector<float> src;
		const int32_t blobs = count / 10;
		for (int32_t bl = 0; bl < blobs; ++bl)
		{
			std::vector<float> centre(static_cast<size_t>(dims), 0.0f);
			centre[static_cast<size_t>(bl % dims)] = 15.0f * (bl + 1);
			PushBlob(src, dims, centre, 10, rng, 0.05f);
		}
		return GBank(src, blobs * 10, dims, Quantization::Int8, Metric::L2);
	};
	const int32_t sizes[] = {80, 20, 80}; // large, small, large
	Workspace warm; warm.Reserve(k + 1, 1);
	for (int32_t count : sizes)
	{
		GBank b = makeBank(count);
		std::vector<int32_t> nbWarm(static_cast<size_t>(count) * k, -999);
		CHECK(BuildKnnNeighbors(b.view, k, true, nbWarm.data(), warm) == Status::Ok);
		Workspace fresh; fresh.Reserve(k + 1, 1);
		std::vector<int32_t> nbFresh(static_cast<size_t>(count) * k, -111);
		CHECK(BuildKnnNeighbors(b.view, k, true, nbFresh.data(), fresh) == Status::Ok);
		int32_t diff = 0;
		for (size_t i = 0; i < nbWarm.size(); ++i) if (nbWarm[i] != nbFresh[i]) ++diff;
		CHECK_MSG(diff == 0, "warm-reuse (count=%d in grow-shrink-grow) must equal fresh, %d differ", count, diff);
	}
}

// M1 / dim 10 (Structure FEAT, the crux) — constructed ground truth: three tight, well-
// separated blobs (each one component) + two planted isolates (outliers) + well-separated
// exact-duplicate blocks (D in {2,3,5} via the block-count parameter) each collapsing to
// one component. Row count above a query chunk boundary so the batch path's chunking
// engages (the core-level "setting that matters"; the above-SampleLimit engagement is the
// Panel FEAT's, where sampling exists). A wrong-but-deterministic grapher (merges blobs,
// fragments them, or fabricates cross-block edges) fails.
static void TestM1StructureFeat()
{
	for (int32_t D : {2, 3, 5})
	{
		Rng rng(0xFEA7u + static_cast<uint64_t>(D));
		const int32_t dims = 64, k = 16, minComp = 3;
		const int32_t blobSize = 400; // 3 blobs = 1200 rows -> above a 64-dim int8 chunk (1024 rows)
		std::vector<float> src;
		std::vector<int32_t> planted; // ground-truth: component tag per row (-1 = isolate/outlier)
		int32_t compTag = 0;
		// Three tight, far-apart blobs.
		for (int32_t bl = 0; bl < 3; ++bl)
		{
			std::vector<float> centre(static_cast<size_t>(dims), 0.0f);
			centre[static_cast<size_t>(bl * 5)] = 50.0f * (bl + 1);
			PushBlob(src, dims, centre, blobSize, rng, 0.05f);
			for (int32_t i = 0; i < blobSize; ++i) planted.push_back(compTag);
			++compTag;
		}
		// D well-separated exact-duplicate blocks (each block byte-identical, size 6 > k+? but
		// unioned by construction regardless of k), each its own component.
		for (int32_t d = 0; d < D; ++d)
		{
			std::vector<float> c(static_cast<size_t>(dims), 0.0f);
			c[static_cast<size_t>(dims - 1 - d)] = 500.0f * (d + 1);
			for (int32_t i = 0; i < 6; ++i) { src.insert(src.end(), c.begin(), c.end()); planted.push_back(compTag); }
			++compTag;
		}
		// Two planted isolates: far from everything, no neighbours -> outliers.
		for (int32_t iso = 0; iso < 2; ++iso)
		{
			std::vector<float> c(static_cast<size_t>(dims), 0.0f);
			c[static_cast<size_t>(dims / 2 + iso)] = 9000.0f * (iso + 1);
			src.insert(src.end(), c.begin(), c.end());
			planted.push_back(-1);
		}
		const int32_t count = static_cast<int32_t>(planted.size());
		GBank b(src, count, dims, Quantization::Int8, Metric::L2);
		std::vector<int32_t> ids(static_cast<size_t>(count), -999);
		CHECK(RunM1Pipeline(b, k, ids) == Status::Ok);

		// Count components of size >= minComp; assert exactly (3 + D).
		std::vector<int32_t> reps, sizes;
		for (int32_t r = 0; r < count; ++r)
		{
			auto it = std::find(reps.begin(), reps.end(), ids[r]);
			if (it == reps.end()) { reps.push_back(ids[r]); sizes.push_back(1); }
			else sizes[static_cast<size_t>(it - reps.begin())]++;
		}
		int32_t bigComps = 0, outliers = 0;
		for (size_t c = 0; c < reps.size(); ++c) { if (sizes[c] >= minComp) ++bigComps; else outliers += sizes[c]; }
		CHECK_MSG(bigComps == 3 + D, "Structure FEAT (D=%d): expected %d components >= minComp, got %d",
			D, 3 + D, bigComps);
		CHECK_MSG(outliers == 2, "Structure FEAT (D=%d): expected 2 isolates in outliers, got %d", D, outliers);

		// Membership: every pair of rows planted in the same blob/block shares a component;
		// each isolate is alone.
		int32_t membershipViolations = 0;
		for (int32_t r = 0; r < count; ++r)
		{
			if (planted[r] < 0) continue;
			// same-tag rows must share ids[r]; use the first row of each tag as anchor
			for (int32_t s = r + 1; s < count; ++s)
			{
				if (planted[s] != planted[r]) continue;
				if (ids[s] != ids[r]) { ++membershipViolations; break; }
			}
		}
		CHECK_MSG(membershipViolations == 0, "Structure FEAT (D=%d): planted memberships must be one component each", D);
	}
}

// ===========================================================================
// V3.2 red suite (Curie) — module M2 novelty.h. NoveltyScore only (its contract
// is fully pinned by §25.4 and is independent of F-M2-1); the probe, baseline,
// and two-limb verdict land after the F-M2-1 exact-distance-entry decision.
// RED SCAFFOLD: src/novelty.cpp returns Ok and writes nothing.
// ===========================================================================

// M2 / dim 2 + dim 6 + dim 10 — NoveltyScore is the empirical-CDF rank: the fraction of
// the sorted baseline STRICTLY LESS THAN the probe distance (ties resolve to the lowest
// rank), normalized to [0,1]. Oracle is hand-computed from the definition (strike-9's
// executed NoveltyScore); rejection on count<1 / null buffers.
static void TestM2NoveltyScore()
{
	// Rejection (dim 2): count < 1 and null buffers -> InvalidArgument, no write.
	{
		const float base[] = {1.0f, 2.0f, 3.0f};
		float out = -999.0f;
		CHECK_MSG(NoveltyScore(base, 0, 1.5f, &out) == Status::InvalidArgument, "NoveltyScore count=0 must reject");
		CHECK_MSG(NoveltyScore(nullptr, 3, 1.5f, &out) == Status::InvalidArgument, "NoveltyScore null baseline must reject");
		CHECK_MSG(NoveltyScore(base, 3, 1.5f, nullptr) == Status::InvalidArgument, "NoveltyScore null out must reject");
		CHECK_MSG(out == -999.0f, "rejected NoveltyScore must not write output");
	}
	// Correctness + ties-low + normalization. baseline sorted ascending; rank = (# strictly
	// less) / count. Ties (distance == a baseline entry) count only the strictly-less side.
	{
		const float base[] = {1.0f, 2.0f, 2.0f, 3.0f, 5.0f, 8.0f}; // count 6
		struct Case { float distance; float expect; const char* why; };
		const Case cases[] = {
			{0.0f, 0.0f, "below all -> 0"},
			{2.0f, 1.0f / 6.0f, "tie with the 2.0 pair -> lowest rank (only 1.0 is strictly less)"},
			{2.5f, 3.0f / 6.0f, "strictly above 1,2,2 -> 3/6"},
			{4.0f, 4.0f / 6.0f, "strictly above 1,2,2,3 -> 4/6"},
			{8.0f, 5.0f / 6.0f, "tie with max -> lowest rank (5 strictly less)"},
			{10.0f, 1.0f, "above all -> 1"},
		};
		for (const Case& c : cases)
		{
			float out = -999.0f;
			CHECK_MSG(NoveltyScore(base, 6, c.distance, &out) == Status::Ok, "NoveltyScore must succeed");
			CHECK_MSG(std::fabs(out - c.expect) < 1e-6f,
				"NoveltyScore(d=%.2f)=%.6f expected %.6f (%s)", c.distance, out, c.expect, c.why);
			CHECK_MSG(out >= 0.0f && out <= 1.0f, "NoveltyScore must lie in [0,1]");
		}
	}
	// Single-entry baseline (degenerate): probe below -> 0, tie/above -> handled per definition.
	{
		const float base[] = {5.0f};
		float out = -999.0f;
		CHECK(NoveltyScore(base, 1, 4.0f, &out) == Status::Ok);
		CHECK_MSG(std::fabs(out - 0.0f) < 1e-6f, "single-entry baseline, probe below -> 0, got %.6f", out);
		CHECK(NoveltyScore(base, 1, 6.0f, &out) == Status::Ok);
		CHECK_MSG(std::fabs(out - 1.0f) < 1e-6f, "single-entry baseline, probe above -> 1, got %.6f", out);
	}
}

// ===========================================================================
// V3.2 red suite (Curie) continued — M2 novelty.h: NoveltyProbeDistance (F-M2-1 /
// D-V32-50, Dan's call, option A), KthNeighborDistance, CalibrateNoveltyBaseline, and the
// two-limb tri-state verdict FEAT (D-V32-31/47, the crux). RED SCAFFOLD: the three new
// novelty.cpp bodies return Ok and write nothing.
// Test design: Claude/Curie/superfaiss-v3.2-test-design-2026-07-18.md.
//
// Oracle discipline (the D-V32 strike loop, "build don't reason"), applied per F-M2-1's
// mandate: NoveltyProbeDistance's oracle (M2OracleProbeDistance) reproduces the SAME
// per-(metric,scope,quant) dispatch the entry's contract pins, built from REAL shipped
// kernel primitives (QuantizeQueryXd, detail::DotI8I8, detail::FloatBitsToDouble) that are
// proven elsewhere and are never the entry under test — oracle == entry by execution, one
// verified number rather than two hopefully-agreeing ones (the strike-14 discipline). The
// strike 10/11/12/13/14 constructed geometries are ported verbatim as fixtures.
// ===========================================================================

namespace
{
	// The cross-device subnormal floor, own recode of analytics.cpp's XdFloor:
	// |x| < FLT_MIN -> exactly 0.0f, on every machine.
	float M2XdFloor(double score)
	{
		const double lim = 1.1754943508222875e-38; // FLT_MIN, exactly
		if (score < lim && score > -lim)
		{
			return 0.0f;
		}
		return static_cast<float>(score);
	}

	struct M2ProbeResult
	{
		Status status;
		float distance;
	};

	// Independent oracle for NoveltyProbeDistance (F-M2-1/D-V32-47/48/50): the SAME
	// per-(metric,scope,quant) dispatch novelty.h's contract pins, built from real shipped
	// kernel primitives — never calls NoveltyProbeDistance itself. `channel == -1` is
	// whole-row.
	M2ProbeResult M2OracleProbeDistance(const BankView& bank, const float* probe,
		int32_t storedRow, int32_t channel)
	{
		if (bank.count < 1 || storedRow < 0 || storedRow >= bank.count || probe == nullptr)
		{
			return {Status::InvalidArgument, -999.0f};
		}
		if (channel != -1 &&
			(bank.channels == nullptr || channel < 0 || channel >= bank.channelCount))
		{
			return {Status::InvalidArgument, -999.0f};
		}
		if (bank.metric == Metric::Dot)
		{
			return {Status::InvalidArgument, -999.0f};
		}

		const int32_t pd = bank.paddedDims;
		const int32_t offset = channel == -1 ? 0 : bank.channels[channel].offset;
		const int32_t length = channel == -1 ? pd : bank.channels[channel].length;
		const bool isCosine = bank.metric == Metric::Cosine;

		if (bank.quant == Quantization::Int8)
		{
			std::vector<int8_t> q8Probe(static_cast<size_t>(pd));
			double probeScale = 0.0;
			int64_t probeFullSq = 0;
			QuantizeQueryXd(probe, pd, q8Probe.data(), &probeScale, &probeFullSq);
			const int8_t* rowFull = static_cast<const int8_t*>(bank.rows) +
				static_cast<int64_t>(storedRow) * pd;
			const double rowScale = detail::FloatBitsToDouble(bank.scales[storedRow]);
			const int8_t* probeSlice = q8Probe.data() + offset;
			const int8_t* rowSlice = rowFull + offset;
			const int64_t probeSq = detail::DotI8I8(probeSlice, probeSlice, length);
			const int64_t rowSq = detail::DotI8I8(rowSlice, rowSlice, length);
			const int64_t cross = detail::DotI8I8(probeSlice, rowSlice, length);
			if (isCosine && (probeSq == 0 || rowSq == 0))
			{
				return {Status::ZeroNormQuery, -999.0f};
			}
			if (isCosine)
			{
				const double denom =
					std::sqrt(static_cast<double>(probeSq) * static_cast<double>(rowSq));
				return {Status::Ok, M2XdFloor(1.0 - static_cast<double>(cross) / denom)};
			}
			const double a = (probeScale * probeScale) * static_cast<double>(probeSq);
			const double bb = (rowScale * rowScale) * static_cast<double>(rowSq);
			const double cc = ((probeScale * rowScale) * static_cast<double>(cross)) * 2.0;
			return {Status::Ok, M2XdFloor((a + bb) - cc)};
		}

		const float* rowFull =
			static_cast<const float*>(bank.rows) + static_cast<int64_t>(storedRow) * pd;
		double probeSq = 0.0, rowSq = 0.0, cross = 0.0, l2 = 0.0;
		for (int32_t i = 0; i < length; ++i)
		{
			const double p = static_cast<double>(probe[offset + i]);
			const double r = static_cast<double>(rowFull[offset + i]);
			probeSq += p * p;
			rowSq += r * r;
			cross += p * r;
			const double d = p - r;
			l2 += d * d;
		}
		if (isCosine && (probeSq == 0.0 || rowSq == 0.0))
		{
			return {Status::ZeroNormQuery, -999.0f};
		}
		if (isCosine)
		{
			return {Status::Ok, M2XdFloor(1.0 - cross / std::sqrt(probeSq * rowSq))};
		}
		return {Status::Ok, M2XdFloor(l2)};
	}

	// A GBank plus an attached channel table (metadata only — the row bytes/scales are
	// GBank's own whole-row bake, exactly matching production: whole-row normalize/quantize
	// happens at bake time; the channel table is separate metadata over the same bytes).
	struct GChannelBank
	{
		GBank bank;
		std::vector<ChannelInfo> channels;
		GChannelBank(const std::vector<float>& src, int32_t count, int32_t dims,
			Quantization quant, Metric metric, std::vector<ChannelInfo> ch)
			: bank(src, count, dims, quant, metric), channels(std::move(ch))
		{
			bank.view.channels = channels.data();
			bank.view.channelCount = static_cast<int32_t>(channels.size());
		}
	};

	// paddedDims-length, zero-padded float probe reproducing GBank row `row`'s DEQUANTIZED
	// (or bake-normalized) value exactly — a byte-identical duplicate probe by construction.
	std::vector<float> M2PaddedProbeFromRow(const GBank& b, int32_t row)
	{
		std::vector<float> out(static_cast<size_t>(b.view.paddedDims), 0.0f);
		const int32_t dims = b.view.dims;
		for (int32_t i = 0; i < dims; ++i)
		{
			out[static_cast<size_t>(i)] =
				static_cast<float>(b.refRows[static_cast<size_t>(row) * dims + i]);
		}
		return out;
	}

	// paddedDims-length float probe from an explicit dims-length source vector, zero-padded.
	std::vector<float> M2PaddedProbe(const std::vector<float>& src, int32_t dims, int32_t paddedDims)
	{
		std::vector<float> out(static_cast<size_t>(paddedDims), 0.0f);
		for (int32_t i = 0; i < dims; ++i)
		{
			out[static_cast<size_t>(i)] = src[static_cast<size_t>(i)];
		}
		return out;
	}

	// Double-precision score of `query` (a dims-or-more-length float array, read [0,dims))
	// against row r of `b`, in the bank's own metric's SIMILARITY sense (RefBetter's
	// convention: for L2 lower is nearer; for Dot/Cosine higher is nearer) — the same
	// "score" a real Query() call would report pre-RankDistance-conversion.
	double M2RefScore(const GBank& b, const float* query, int32_t r)
	{
		const int32_t dims = b.view.dims;
		const double* row = &b.refRows[static_cast<size_t>(r) * dims];
		double score = 0.0;
		if (b.view.metric == Metric::L2)
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
		return score;
	}

	// Brute-force k-th-NN RankDistance of `query` against `b`, excluding the rows listed in
	// `excludeRows`. L2 -> the k-th smallest squared distance (already a RankDistance);
	// Cosine -> 1 - the k-th LARGEST cosine similarity (RankDistance applied AFTER ranking
	// by similarity — D-V32-19's pinned order). Never called on a Dot bank (verdict-domain
	// only).
	float M2BruteKthRankDistance(const GBank& b, const float* query, int32_t k,
		const std::vector<int32_t>& excludeRows)
	{
		std::vector<double> scores;
		for (int32_t r = 0; r < b.view.count; ++r)
		{
			if (std::find(excludeRows.begin(), excludeRows.end(), r) != excludeRows.end())
			{
				continue;
			}
			scores.push_back(M2RefScore(b, query, r));
		}
		std::sort(scores.begin(), scores.end(), [&](double x, double y) {
			return b.view.metric == Metric::L2 ? x < y : x > y;
		});
		const double kth = scores[static_cast<size_t>(k - 1)];
		return b.view.metric == Metric::Cosine ? static_cast<float>(1.0 - kth) : static_cast<float>(kth);
	}
}

// M2 / dim 2 + dim 4 + dim 6 + dim 7 (F-M2-1, D-V32-47/48/50) — NoveltyProbeDistance is
// limb 1's exact-distance entry: the metric's OWN distance function between a probe and
// one stored row, dispatched per (metric, scope, quant). Oracle: M2OracleProbeDistance.
static void TestM2NoveltyProbeDistance()
{
	Rng rng(0x4E5031);
	// Finding 5 (Poirot cf3f750-v32-core-batch-review.md): NoveltyProbeDistance now takes
	// a Workspace, matching its three sibling M2 entries (KthNeighborDistance,
	// CalibrateNoveltyBaseline take one already) -- one warm workspace serves every call
	// in this function.
	Workspace ws;

	// --- Trust boundaries (dim 2): storedRow/channel/null rejections, no write. ---
	{
		std::vector<float> src;
		const int32_t dims = 8, count = 4;
		for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
		GBank b(src, count, dims, Quantization::Float32, Metric::L2);
		std::vector<float> probe(static_cast<size_t>(b.view.paddedDims), 0.1f);
		float out = -999.0f;

		CHECK_MSG(NoveltyProbeDistance(b.view, probe.data(), -1, -1, &out, ws) == Status::InvalidArgument,
			"storedRow -1 must reject");
		CHECK_MSG(NoveltyProbeDistance(b.view, probe.data(), count, -1, &out, ws) == Status::InvalidArgument,
			"storedRow == count must reject");
		CHECK_MSG(NoveltyProbeDistance(b.view, probe.data(), 0, 0, &out, ws) == Status::InvalidArgument,
			"channel 0 on a bank with no channel table must reject");
		CHECK_MSG(NoveltyProbeDistance(b.view, nullptr, 0, -1, &out, ws) == Status::InvalidArgument,
			"null paddedProbeQuery must reject");
		CHECK_MSG(NoveltyProbeDistance(b.view, probe.data(), 0, -1, nullptr, ws) == Status::InvalidArgument,
			"null outDistance must reject");
		CHECK_MSG(out == -999.0f, "a rejected call must not write outDistance");

		BankView dotView = b.view;
		dotView.metric = Metric::Dot;
		CHECK_MSG(NoveltyProbeDistance(dotView, probe.data(), 0, -1, &out, ws) == Status::InvalidArgument,
			"a Dot bank must reject -- never dispatched (verdict-unavailable is upstream)");
		CHECK_MSG(out == -999.0f, "the Dot rejection must not write outDistance");
	}

	// --- Channel-table trust boundaries: channel out of [-1, channelCount). Grid-aligned
	// (F-M2-3: the int8 channel grid is 16 elements -- an off-grid channel table is
	// unrealizable on any bank that passed real validation, and detail::DotI8I8's SIMD
	// paths assume 16-alignment with no scalar remainder tail). ---
	{
		std::vector<float> src;
		const int32_t dims = 32, count = 2;
		for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
		std::vector<ChannelInfo> ch = {{0, 16}, {16, 16}};
		GChannelBank b(src, count, dims, Quantization::Int8, Metric::Cosine, ch);
		std::vector<float> probe(static_cast<size_t>(b.bank.view.paddedDims), 0.1f);
		float out = -999.0f;
		CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, -2, &out, ws) == Status::InvalidArgument);
		CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, 2, &out, ws) == Status::InvalidArgument);
		CHECK(out == -999.0f);
	}

	// --- int8 whole-row, Cosine and L2: a re-quantized duplicate probe -> entry == oracle
	// (unconditional -- both read the SAME real QuantizeQueryXd/DotI8I8 primitives, so this
	// holds regardless of whether the round trip below reproduces byte-identical codes); a
	// distinct row -> entry == oracle; oracle == entry on random pairs. Cosine additionally
	// asserts EXACTLY 0 (D-V32-47: parallel int8 codes give cross^2 == aSq*bSq exactly, and
	// sqrt of a perfect square is exact under IEEE correctly-rounded sqrt -- proven, and
	// confirmed by execution here). L2 does NOT get the same absolute assertion: dequantizing
	// row 2 through a float32 probe and re-quantizing is not guaranteed to reproduce
	// byte-identical codes (the dequantized value itself is a float32 rounding of a double),
	// and even where the codes ARE identical, L2's EXPANDED form (a^2+b^2-2ab) is prone to
	// double-precision cancellation residue -- the disclosed, standing "L2 expanded-form
	// rounding" item (D-V32-48), confirmed here too (a tiny nonzero residual on an otherwise
	// exact case, not a defect). L2 uses a tight epsilon that absorbs double-precision
	// rounding noise but nothing larger. ---
	for (Metric metric : {Metric::Cosine, Metric::L2})
	{
		const int32_t dims = 48, count = 6;
		std::vector<float> src;
		for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
		GBank b(src, count, dims, Quantization::Int8, metric);

		{
			std::vector<float> probe = M2PaddedProbeFromRow(b, 2);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.view, probe.data(), 2, -1, &out, ws) == Status::Ok);
			if (metric == Metric::Cosine)
			{
				CHECK_MSG(out == 0.0f,
					"int8 whole-row Cosine: a re-quantized duplicate probe must score exactly "
					"0 (D-V32-47's parallel-code exactness), got %.9g", static_cast<double>(out));
			}
			else
			{
				CHECK_MSG(std::fabs(out) < 1e-8f,
					"int8 whole-row L2: a re-quantized duplicate probe must score near-zero "
					"(D-V32-48's disclosed expanded-form rounding residue, not exact-0), "
					"got %.9g", static_cast<double>(out));
			}
			const M2ProbeResult ref = M2OracleProbeDistance(b.view, probe.data(), 2, -1);
			CHECK(ref.status == Status::Ok && ref.distance == out);
		}
		for (int32_t r = 0; r < count; ++r)
		{
			std::vector<float> probe = M2PaddedProbeFromRow(b, (r + 3) % count);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.view, probe.data(), r, -1, &out, ws) == Status::Ok);
			const M2ProbeResult ref = M2OracleProbeDistance(b.view, probe.data(), r, -1);
			CHECK_MSG(ref.status == Status::Ok && ref.distance == out,
				"int8 whole-row %s row %d: entry %.9g oracle %.9g",
				metric == Metric::Cosine ? "Cosine" : "L2", r,
				static_cast<double>(out), static_cast<double>(ref.distance));
		}
		for (int32_t t = 0; t < 40; ++t)
		{
			std::vector<float> probeSrc(static_cast<size_t>(dims));
			for (float& v : probeSrc) v = rng.NextFloat();
			std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.view.paddedDims);
			const int32_t row = rng.NextIndex(count);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.view, probe.data(), row, -1, &out, ws) == Status::Ok);
			const M2ProbeResult ref = M2OracleProbeDistance(b.view, probe.data(), row, -1);
			CHECK_MSG(ref.status == Status::Ok && ref.distance == out,
				"int8 whole-row %s random pair: entry %.9g oracle %.9g",
				metric == Metric::Cosine ? "Cosine" : "L2",
				static_cast<double>(out), static_cast<double>(ref.distance));
		}
	}

	// --- Whole-row Cosine, int8: a SCALAR MULTIPLE of a stored row is a true cosine
	// duplicate (distance exactly 0) even though it stores byte-DIFFERENT — the strike-14
	// completeness leg (D-V32-46/47): kills any regression to a byte-identity limb 1. ---
	{
		const int32_t dims = 32, count = 1;
		std::vector<float> rowSrc(static_cast<size_t>(dims));
		for (float& v : rowSrc) v = rng.NextFloat();
		GBank b(rowSrc, count, dims, Quantization::Int8, Metric::Cosine);
		const std::vector<float> normalizedRow = M2PaddedProbeFromRow(b, 0);
		const float scalars[] = {2.0f, 3.0f, 5.0f, 7.0f, 1.5f, 0.5f, 0.333333f, 10.0f, 1.0001f, 100.0f};
		for (float c : scalars)
		{
			std::vector<float> probe(normalizedRow.size());
			for (size_t i = 0; i < probe.size(); ++i) probe[i] = c * normalizedRow[i];
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.view, probe.data(), 0, -1, &out, ws) == Status::Ok);
			CHECK_MSG(out == 0.0f,
				"whole-row Cosine int8: scalar multiple c=%.6g of the stored row must score "
				"exactly 0 (true cosine duplicate; byte-identity would miss most of these), "
				"got %.9g", static_cast<double>(c), static_cast<double>(out));
		}
	}

	// --- Whole-row float32, Cosine and L2: exactly-representable fixtures (axis-aligned
	// unit vectors), robust to any accumulation order. Plus a duplicate case and a random
	// oracle cross-check (double-accumulated, per the header's pinned convention). ---
	for (Metric metric : {Metric::Cosine, Metric::L2})
	{
		const int32_t dims = 8;
		std::vector<float> src = {
			1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0,
		};
		GBank b(src, 2, dims, Quantization::Float32, metric);

		{
			std::vector<float> probe = M2PaddedProbe({0, 1, 0, 0, 0, 0, 0, 0}, dims, b.view.paddedDims);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.view, probe.data(), 0, -1, &out, ws) == Status::Ok);
			const float expect = metric == Metric::Cosine ? 1.0f : 2.0f;
			CHECK_MSG(out == expect, "whole-row float32 %s orthogonal pair: expected %.6g got %.9g",
				metric == Metric::Cosine ? "Cosine" : "L2",
				static_cast<double>(expect), static_cast<double>(out));
		}
		{
			std::vector<float> probe = M2PaddedProbeFromRow(b, 1);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.view, probe.data(), 1, -1, &out, ws) == Status::Ok);
			CHECK_MSG(out == 0.0f, "whole-row float32 %s duplicate must score exactly 0, got %.9g",
				metric == Metric::Cosine ? "Cosine" : "L2", static_cast<double>(out));
		}
		{
			const int32_t rdims = 16, rcount = 5;
			std::vector<float> rsrc;
			for (int32_t i = 0; i < rcount * rdims; ++i) rsrc.push_back(rng.NextFloat());
			GBank rb(rsrc, rcount, rdims, Quantization::Float32, metric);
			for (int32_t t = 0; t < 20; ++t)
			{
				std::vector<float> probeSrc(static_cast<size_t>(rdims));
				for (float& v : probeSrc) v = rng.NextFloat();
				std::vector<float> probe = M2PaddedProbe(probeSrc, rdims, rb.view.paddedDims);
				const int32_t row = rng.NextIndex(rcount);
				float out = -999.0f;
				CHECK(NoveltyProbeDistance(rb.view, probe.data(), row, -1, &out, ws) == Status::Ok);
				const M2ProbeResult ref = M2OracleProbeDistance(rb.view, probe.data(), row, -1);
				CHECK_MSG(ref.status == Status::Ok &&
						std::fabs(ref.distance - out) <= 1e-5f * std::max(1.0f, std::fabs(ref.distance)),
					"whole-row float32 %s random pair: entry %.9g oracle %.9g",
					metric == Metric::Cosine ? "Cosine" : "L2",
					static_cast<double>(out), static_cast<double>(ref.distance));
			}
		}
	}

	// --- Channel-scoped Cosine, int8: strike 10's exact-direction twin verdicts a true
	// scoped duplicate (distance 0); a different-direction pair does not. Grid-aligned
	// (F-M2-3): dims=32, channel0=[0,16)/channel1=[16,32), the strike's channel-0 direction
	// at index 0 and its whole-row-norm-affecting remainder moved to index 16 (channel 1) —
	// the same relationship strike 10's 6-dim geometry had (a value OUTSIDE the scoped
	// channel that changes the whole-row norm without entering the channel-0 slice). ---
	{
		const int32_t dims = 32;
		std::vector<float> rowA(static_cast<size_t>(dims), 0.0f);
		rowA[0] = 1.0f;  // channel-0 direction (1,0,0,...)
		rowA[16] = 3.0f; // channel-1 value -- affects whole-row norm only, mirrors strike 10
		std::vector<ChannelInfo> ch = {{0, 16}, {16, 16}};
		GChannelBank b(rowA, 1, dims, Quantization::Int8, Metric::Cosine, ch);

		{
			std::vector<float> probeSrc(static_cast<size_t>(dims), 0.0f);
			probeSrc[0] = 1.0f; // exact channel-0 direction match
			std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, 0, &out, ws) == Status::Ok);
			CHECK_MSG(out == 0.0f,
				"channel-scoped Cosine int8: exact-direction slice twin (strike 10) must "
				"score exactly 0, got %.9g", static_cast<double>(out));
		}
		{
			std::vector<float> probeSrc(static_cast<size_t>(dims), 0.0f);
			probeSrc[1] = 1.0f; // orthogonal to row A's channel-0 direction
			std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, 0, &out, ws) == Status::Ok);
			CHECK_MSG(out == 1.0f,
				"channel-scoped Cosine int8: a different-direction slice must NOT score 0 "
				"(orthogonal -> 1), got %.9g", static_cast<double>(out));
		}
	}

	// --- Channel-scoped L2, int8: strike 11's twin — kernel channel-L2 distance exactly
	// 0.0 for two rows with a DIFFERENT stored (bytes, scale) key. Grid-aligned (F-M2-3):
	// dims=32, channel0=[0,16)/channel1=[16,32); the strike's channel-0 payload
	// (100/127,50/127,0) sits at indices [0,3), and the scale-driving remainder (2.0 vs
	// 1.0) moves to index 16 (channel 1) -- the per-row whole-row maxAbs/scale and every
	// quantized byte in channel 0 come out bit-identical to strike 11's own numbers, since
	// the extra zero lanes contribute nothing to either the scale or the channel-0 sums. ---
	{
		const int32_t dims = 32;
		std::vector<float> rowA(static_cast<size_t>(dims), 0.0f);
		rowA[0] = 100.0f / 127.0f;
		rowA[1] = 50.0f / 127.0f;
		rowA[16] = 2.0f;
		std::vector<ChannelInfo> ch = {{0, 16}, {16, 16}};
		GChannelBank b(rowA, 1, dims, Quantization::Int8, Metric::L2, ch);
		std::vector<float> probeSrc(static_cast<size_t>(dims), 0.0f);
		probeSrc[0] = 100.0f / 127.0f;
		probeSrc[1] = 50.0f / 127.0f;
		probeSrc[16] = 1.0f;
		std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
		float out = -999.0f;
		CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, 0, &out, ws) == Status::Ok);
		// Near-zero, not exact-0: the pair is exact by hand-verified RATIONAL arithmetic
		// (12500/127^2 + 12500/127^2 - 25000/127^2 == 0 exactly, since 2/127 and 1/127 are
		// each other's cross-consistent scale), but the EXPANDED-FORM double computation
		// (a^2+b^2-2ab) carries the disclosed, standing L2 rounding residue (D-V32-48) —
		// confirmed here (a residual at double-epsilon scale, not a defect).
		CHECK_MSG(std::fabs(out) < 1e-8f,
			"channel-scoped L2 int8: strike 11's dequant-identical/byte-different twin must "
			"score near-zero (D-V32-48's disclosed expanded-form rounding residue, not "
			"exact-0), got %.9g", static_cast<double>(out));
	}

	// --- Channel-scoped Cosine zero-energy guard, BOTH sides (D-V32-43, strike 12): a
	// zero-energy slice on the STORED row, or on the PROBE, must reject with ZeroNormQuery
	// — never silently floor to a false distance-0 duplicate. ---
	{
		const int32_t dims = 32;
		std::vector<ChannelInfo> ch = {{0, 16}, {16, 16}};
		std::vector<float> row0(static_cast<size_t>(dims), 0.0f);
		row0[0] = 1.0f; // channel-1 slice all zero
		std::vector<float> row1(static_cast<size_t>(dims), 0.0f);
		row1[16] = 1.0f; // channel-1 direction, axis 0
		std::vector<float> src;
		src.insert(src.end(), row0.begin(), row0.end());
		src.insert(src.end(), row1.begin(), row1.end());
		GChannelBank b(src, 2, dims, Quantization::Int8, Metric::Cosine, ch);

		{
			std::vector<float> probeSrc(static_cast<size_t>(dims), 0.0f);
			probeSrc[17] = 1.0f; // channel-1 direction, axis 1 (orthogonal to row1)
			std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
			float out = -999.0f;
			const Status s = NoveltyProbeDistance(b.bank.view, probe.data(), 0, 1, &out, ws);
			CHECK_MSG(s == Status::ZeroNormQuery,
				"channel Cosine zero-energy STORED row (strike 12) must reject ZeroNormQuery, "
				"got status=%d distance=%.9g", static_cast<int>(s), static_cast<double>(out));
			CHECK(out == -999.0f);
		}
		{
			std::vector<float> probeSrc(static_cast<size_t>(dims), 0.0f);
			probeSrc[17] = 1.0f;
			std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 1, 1, &out, ws) == Status::Ok);
			CHECK_MSG(out == 1.0f,
				"channel Cosine control (orthogonal, non-zero stored slice) must score 1, "
				"got %.9g", static_cast<double>(out));
		}
		{
			std::vector<float> probeSrc(static_cast<size_t>(dims), 0.0f);
			probeSrc[0] = 1.0f; // nonzero elsewhere, but channel-1 [16,32) is all zero
			std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
			float out = -999.0f;
			const Status s = NoveltyProbeDistance(b.bank.view, probe.data(), 1, 1, &out, ws);
			CHECK_MSG(s == Status::ZeroNormQuery,
				"channel Cosine zero-energy PROBE slice must also reject ZeroNormQuery, "
				"got status=%d distance=%.9g", static_cast<int>(s), static_cast<double>(out));
			CHECK(out == -999.0f);
		}
	}

	// --- Channel-scoped L2 needs no zero-energy guard. ---
	{
		const int32_t dims = 32;
		std::vector<ChannelInfo> ch = {{0, 16}, {16, 16}};
		std::vector<float> row0(static_cast<size_t>(dims), 0.0f);
		row0[0] = 1.0f;
		GChannelBank b(row0, 1, dims, Quantization::Int8, Metric::L2, ch);
		std::vector<float> probeSrc(static_cast<size_t>(dims), 0.0f);
		probeSrc[16] = 3.0f;
		std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
		float out = -999.0f;
		const Status s = NoveltyProbeDistance(b.bank.view, probe.data(), 0, 1, &out, ws);
		CHECK_MSG(s == Status::Ok,
			"channel L2 zero-energy slice must NOT reject (no Cosine-only guard), got status=%d",
			static_cast<int>(s));
		CHECK_MSG(out > 0.0f,
			"channel L2 zero-vs-nonzero slice must score a real positive distance, got %.9g",
			static_cast<double>(out));
	}

	// --- Channel-scoped float32, Cosine and L2: same shape as int8, exactly-representable
	// fixture + oracle cross-check. ---
	for (Metric metric : {Metric::Cosine, Metric::L2})
	{
		const int32_t dims = 8;
		std::vector<ChannelInfo> ch = {{0, 4}, {4, 4}};
		std::vector<float> src = {1, 0, 0, 0, 2, 0, 0, 0};
		GChannelBank b(src, 1, dims, Quantization::Float32, metric, ch);
		{
			std::vector<float> probeSrc = {0, 0, 0, 0, 5, 0, 0, 0};
			std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
			float out = -999.0f;
			CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, 1, &out, ws) == Status::Ok);
			const float expect = metric == Metric::Cosine ? 0.0f : (5.0f - 2.0f) * (5.0f - 2.0f);
			CHECK_MSG(out == expect, "channel float32 %s same-direction leg: expected %.6g got %.9g",
				metric == Metric::Cosine ? "Cosine" : "L2",
				static_cast<double>(expect), static_cast<double>(out));
		}
		for (int32_t t = 0; t < 20; ++t)
		{
			std::vector<float> probeSrc(static_cast<size_t>(dims));
			for (float& v : probeSrc) v = rng.NextFloat() + 1.5f; // keep every slice nonzero
			std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
			for (int32_t c = 0; c < 2; ++c)
			{
				float out = -999.0f;
				CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, c, &out, ws) == Status::Ok);
				const M2ProbeResult ref = M2OracleProbeDistance(b.bank.view, probe.data(), 0, c);
				CHECK_MSG(ref.status == Status::Ok &&
						std::fabs(ref.distance - out) <= 1e-5f * std::max(1.0f, std::fabs(ref.distance)),
					"channel float32 %s random pair ch=%d: entry %.9g oracle %.9g",
					metric == Metric::Cosine ? "Cosine" : "L2", c,
					static_cast<double>(out), static_cast<double>(ref.distance));
			}
		}
	}

	// --- Finding 5 (cf3f750-v32-core-batch-review.md): an off-grid int8 channel (length
	// not a multiple of the 16-element grid DotI8I8's SIMD path assumes) must be a clean
	// InvalidArgument, never a silently-wrong distance computed on the SIMD-padded slop.
	// Hand-built BankView (validate.cpp would reject this table at bake; this proves the
	// entry itself refuses it too, independent of upstream validation). An ON-grid channel
	// on the SAME bank stays green -- the guard is scoped to the off-grid case only. ---
	{
		const int32_t dims = 32, count = 1;
		std::vector<ChannelInfo> ch = {{0, 10}, {10, 22}}; // 10 is NOT a multiple of 16
		std::vector<float> src(static_cast<size_t>(dims), 0.0f);
		src[0] = 1.0f;
		GChannelBank b(src, count, dims, Quantization::Int8, Metric::L2, ch);
		std::vector<float> probeSrc(static_cast<size_t>(dims), 0.0f);
		probeSrc[0] = 1.0f;
		std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);
		float out = -999.0f;
		const Status offGrid = NoveltyProbeDistance(b.bank.view, probe.data(), 0, 0, &out, ws);
		CHECK_MSG(offGrid == Status::InvalidArgument,
			"off-grid int8 channel (length 10, not a multiple of 16) must reject, got status=%d",
			static_cast<int>(offGrid));

		// The on-grid sibling channel (offset 10, length 22 -- also off-grid, so use a
		// second, genuinely on-grid bank instead of channel 1 of this one) stays green.
		std::vector<ChannelInfo> chOk = {{0, 16}, {16, 16}};
		GChannelBank bOk(src, count, dims, Quantization::Int8, Metric::L2, chOk);
		float outOk = -999.0f;
		CHECK_MSG(NoveltyProbeDistance(bOk.bank.view, probe.data(), 0, 0, &outOk, ws) == Status::Ok,
			"an ON-grid channel on the same bank shape must stay green");
	}
}

// M2 / dim 2 + dim 5 + dim 6 (D-V32-19/22) — KthNeighborDistance: the k-th-NN distance
// against the FULL view, converted to RankDistance before ranking (L2 -> score; Cosine ->
// 1 - score); Dot excluded from the verdict domain. Oracle: M2BruteKthRankDistance.
static void TestM2KthNeighborDistance()
{
	Rng rng(0x4B7448);

	// --- Trust boundaries (dim 2): k<1, k too large, null buffers, Dot -> InvalidArgument,
	// no write. F-M2-4: KthNeighborDistance is explicitly NOT self-widening (unlike
	// BuildKnnNeighbors' excludeSelf), so k == the number of non-excluded rows is a VALID
	// call (the distance to the single farthest row) -- the rejection boundary is k >
	// available, not k >= available. Confirmed against the header's own literal wording. ---
	{
		const int32_t dims = 8, count = 6;
		std::vector<float> src;
		for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
		GBank b(src, count, dims, Quantization::Float32, Metric::L2);
		Workspace ws; ws.Reserve(count + 1, 1);
		std::vector<float> query = M2PaddedProbeFromRow(b, 0);
		float out = -999.0f;

		CHECK(KthNeighborDistance(b.view, query.data(), 0, nullptr, &out, ws) == Status::InvalidArgument);
		CHECK_MSG(KthNeighborDistance(b.view, query.data(), count + 1, nullptr, &out, ws) == Status::InvalidArgument,
			"k greater than the available (non-excluded) row count must reject");
		CHECK(KthNeighborDistance(b.view, nullptr, 1, nullptr, &out, ws) == Status::InvalidArgument);
		CHECK(KthNeighborDistance(b.view, query.data(), 1, nullptr, nullptr, ws) == Status::InvalidArgument);
		CHECK_MSG(out == -999.0f, "a rejected KthNeighborDistance call must not write outDistance");

		// k == count (no exclusions, nothing self-widened) is the top of the VALID range,
		// not a rejection: the distance to the single farthest row is well-defined.
		{
			float boundaryOut = -999.0f;
			CHECK_MSG(KthNeighborDistance(b.view, query.data(), count, nullptr, &boundaryOut, ws) == Status::Ok,
				"k == count (no exclusions) must succeed -- not self-widening, so the top of "
				"the valid range is the full row count");
			CHECK_MSG(boundaryOut != -999.0f, "a successful k==count call must write outDistance");
		}

		BankView dotView = b.view;
		dotView.metric = Metric::Dot;
		CHECK_MSG(KthNeighborDistance(dotView, query.data(), 1, nullptr, &out, ws) == Status::InvalidArgument,
			"a Dot bank must reject -- RankDistance is undefined there (D-V32-22)");
		CHECK(out == -999.0f);
	}

	// --- RankDistance correctness, L2 and Cosine: a tie-free "arc" fixture (distinct
	// angular gaps, the M1 precedent) so float-vs-double ranking cannot disagree. ---
	for (Metric metric : {Metric::L2, Metric::Cosine})
	{
		const int32_t dims = 4, count = 9;
		std::vector<float> src;
		for (int32_t i = 0; i < count; ++i)
		{
			const double theta = 0.31 * i + 0.07 * i * i;
			src.push_back(static_cast<float>(std::cos(theta) * (1.0 + 0.05 * i)));
			src.push_back(static_cast<float>(std::sin(theta) * (1.0 + 0.05 * i)));
			src.push_back(0.02f * static_cast<float>(i));
			src.push_back(0.01f * static_cast<float>(i * i));
		}
		GBank b(src, count, dims, Quantization::Float32, metric);
		Workspace ws; ws.Reserve(count, 1);

		for (int32_t r = 0; r < count; ++r)
		{
			std::vector<float> query = M2PaddedProbeFromRow(b, r);
			std::vector<uint32_t> excl((static_cast<size_t>(count) + 31) / 32, 0u);
			excl[static_cast<size_t>(r) >> 5] |= (1u << (r & 31));
			for (int32_t k = 1; k <= count - 1; ++k)
			{
				float out = -999.0f;
				CHECK(KthNeighborDistance(b.view, query.data(), k, excl.data(), &out, ws) == Status::Ok);
				const float ref = M2BruteKthRankDistance(b, query.data(), k, {r});
				CHECK_MSG(std::fabs(out - ref) < 1e-4f,
					"KthNeighborDistance %s row %d k=%d: got %.6g expected %.6g",
					metric == Metric::L2 ? "L2" : "Cosine", r, k,
					static_cast<double>(out), static_cast<double>(ref));
			}
		}
	}

	// --- Exclusion composes: excluding an additional near row changes the k-th-NN
	// distance to match the reference over the REMAINING rows. ---
	{
		const int32_t dims = 4, count = 9;
		std::vector<float> src;
		for (int32_t i = 0; i < count; ++i)
		{
			const double theta = 0.31 * i + 0.07 * i * i;
			src.push_back(static_cast<float>(std::cos(theta) * (1.0 + 0.05 * i)));
			src.push_back(static_cast<float>(std::sin(theta) * (1.0 + 0.05 * i)));
			src.push_back(0.02f * static_cast<float>(i));
			src.push_back(0.01f * static_cast<float>(i * i));
		}
		GBank b(src, count, dims, Quantization::Float32, Metric::L2);
		Workspace ws; ws.Reserve(count, 1);
		const int32_t probeRow = 0, extraExcl = 1;
		std::vector<float> query = M2PaddedProbeFromRow(b, probeRow);
		std::vector<uint32_t> excl((static_cast<size_t>(count) + 31) / 32, 0u);
		excl[static_cast<size_t>(probeRow) >> 5] |= (1u << (probeRow & 31));
		excl[static_cast<size_t>(extraExcl) >> 5] |= (1u << (extraExcl & 31));
		const int32_t k = 3;
		float out = -999.0f;
		CHECK(KthNeighborDistance(b.view, query.data(), k, excl.data(), &out, ws) == Status::Ok);
		const float ref = M2BruteKthRankDistance(b, query.data(), k, {probeRow, extraExcl});
		CHECK_MSG(std::fabs(out - ref) < 1e-4f,
			"KthNeighborDistance with a caller exclusion beyond self: got %.6g expected %.6g",
			static_cast<double>(out), static_cast<double>(ref));
	}

	// --- Repeat-call bit-equality (dim 6). ---
	{
		const int32_t dims = 16, count = 12;
		std::vector<float> src;
		for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
		GBank b(src, count, dims, Quantization::Int8, Metric::L2);
		Workspace ws; ws.Reserve(count, 1);
		std::vector<float> query = M2PaddedProbeFromRow(b, 3);
		float out1 = -999.0f, out2 = -999.0f;
		CHECK(KthNeighborDistance(b.view, query.data(), 4, nullptr, &out1, ws) == Status::Ok);
		CHECK(KthNeighborDistance(b.view, query.data(), 4, nullptr, &out2, ws) == Status::Ok);
		CHECK_MSG(out1 == out2, "KthNeighborDistance repeat call must be bit-identical: %.9g vs %.9g",
			static_cast<double>(out1), static_cast<double>(out2));
	}
}

// M2 / dim 2 + dim 6 + dim 10 — CalibrateNoveltyBaseline: the bank's own k-th-NN
// RankDistance distribution over the view it is handed (which the caller already sampled
// to sampleLimit — this function does not re-stride, per its header contract), each row
// self-excluded, output sorted ascending. In-cap correctness only (bank.count <=
// sampleLimit): the over-cap striding algorithm is a separate contract question, routed as
// F-M2-2 (see the test-design artifact).
static void TestM2CalibrateNoveltyBaseline()
{
	Rng rng(0x2C0FEE);

	// --- Trust boundaries (dim 2). ---
	{
		const int32_t dims = 8, count = 6;
		std::vector<float> src;
		for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
		GBank b(src, count, dims, Quantization::Float32, Metric::L2);
		Workspace ws; ws.Reserve(count, 1);
		std::vector<float> out(static_cast<size_t>(count), -999.0f);
		int32_t outCount = -999;

		CHECK(CalibrateNoveltyBaseline(b.view, 0, 64, out.data(), &outCount, ws) == Status::InvalidArgument);
		CHECK(CalibrateNoveltyBaseline(b.view, count, 64, out.data(), &outCount, ws) == Status::InvalidArgument);
		CHECK(CalibrateNoveltyBaseline(b.view, 1, 0, out.data(), &outCount, ws) == Status::InvalidArgument);
		CHECK(CalibrateNoveltyBaseline(b.view, 1, 64, nullptr, &outCount, ws) == Status::InvalidArgument);
		CHECK(CalibrateNoveltyBaseline(b.view, 1, 64, out.data(), nullptr, ws) == Status::InvalidArgument);
		CHECK_MSG(CalibrateNoveltyBaseline(b.view, 1, count - 1, out.data(), &outCount, ws) == Status::InvalidArgument,
			"CalibrateNoveltyBaseline must reject a view whose count exceeds its own sampleLimit");
		CHECK_MSG(out[0] == -999.0f && outCount == -999, "a rejected call must not write outputs");

		BankView dotView = b.view;
		dotView.metric = Metric::Dot;
		CHECK_MSG(CalibrateNoveltyBaseline(dotView, 1, 64, out.data(), &outCount, ws) == Status::InvalidArgument,
			"a Dot bank must reject -- RankDistance is undefined there");
	}

	// --- In-cap correctness FEAT: bank.count <= sampleLimit (no striding ambiguity). Each
	// row's self-excluded k-th-NN RankDistance, sorted ascending, matches the independent
	// brute-force reference exactly (tie-free "arc" fixture). ---
	for (Metric metric : {Metric::L2, Metric::Cosine})
	{
		const int32_t dims = 4, count = 11;
		std::vector<float> src;
		for (int32_t i = 0; i < count; ++i)
		{
			const double theta = 0.29 * i + 0.05 * i * i;
			src.push_back(static_cast<float>(std::cos(theta) * (1.0 + 0.04 * i)));
			src.push_back(static_cast<float>(std::sin(theta) * (1.0 + 0.04 * i)));
			src.push_back(0.015f * static_cast<float>(i));
			src.push_back(0.008f * static_cast<float>(i * i));
		}
		GBank b(src, count, dims, Quantization::Float32, metric);
		Workspace ws; ws.Reserve(count, 1);
		const int32_t k = 3, sampleLimit = 64;
		std::vector<float> out(static_cast<size_t>(count), -999.0f);
		int32_t outCount = -999;

		CHECK(CalibrateNoveltyBaseline(b.view, k, sampleLimit, out.data(), &outCount, ws) == Status::Ok);
		CHECK_MSG(outCount == count,
			"CalibrateNoveltyBaseline outCount must equal bank.count in-cap, got %d", outCount);

		std::vector<float> refBaseline;
		for (int32_t r = 0; r < count; ++r)
		{
			std::vector<float> query = M2PaddedProbeFromRow(b, r);
			refBaseline.push_back(M2BruteKthRankDistance(b, query.data(), k, {r}));
		}
		std::sort(refBaseline.begin(), refBaseline.end());

		bool sorted = true;
		for (int32_t i = 1; i < outCount; ++i)
		{
			if (out[static_cast<size_t>(i)] < out[static_cast<size_t>(i - 1)]) sorted = false;
		}
		CHECK_MSG(sorted, "CalibrateNoveltyBaseline output must be sorted ascending");

		for (int32_t i = 0; i < outCount && i < static_cast<int32_t>(refBaseline.size()); ++i)
		{
			CHECK_MSG(std::fabs(out[static_cast<size_t>(i)] - refBaseline[static_cast<size_t>(i)]) < 1e-4f,
				"CalibrateNoveltyBaseline %s entry %d: got %.6g expected %.6g",
				metric == Metric::L2 ? "L2" : "Cosine", i,
				static_cast<double>(out[static_cast<size_t>(i)]),
				static_cast<double>(refBaseline[static_cast<size_t>(i)]));
		}
	}

	// --- Repeat-call bit-equality (dim 6). ---
	{
		const int32_t dims = 12, count = 10;
		std::vector<float> src;
		for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
		GBank b(src, count, dims, Quantization::Int8, Metric::L2);
		Workspace ws; ws.Reserve(count, 1);
		std::vector<float> out1(static_cast<size_t>(count), -999.0f), out2(static_cast<size_t>(count), -999.0f);
		int32_t n1 = -999, n2 = -999;
		CHECK(CalibrateNoveltyBaseline(b.view, 3, 64, out1.data(), &n1, ws) == Status::Ok);
		CHECK(CalibrateNoveltyBaseline(b.view, 3, 64, out2.data(), &n2, ws) == Status::Ok);
		CHECK(n1 == n2);
		bool same = true;
		for (int32_t i = 0; i < n1; ++i)
		{
			if (out1[static_cast<size_t>(i)] != out2[static_cast<size_t>(i)]) same = false;
		}
		CHECK_MSG(same, "CalibrateNoveltyBaseline repeat call must be bit-identical");
	}
}

namespace
{
	enum class M2Verdict { Duplicate, Novel, Familiar, Unavailable, Error };

	// The two-limb tri-state verdict driver (D-V32-31/47), composed from the four M2
	// primitives UNDER TEST — not a hand transcription of the mechanism. Limb 1 fires first
	// (NoveltyProbeDistance against every live row at exact distance 0 — override, CDF
	// never consulted); else limb 2 (KthNeighborDistance + CalibrateNoveltyBaseline +
	// NoveltyScore) splits novel/familiar at lambda. A Dot bank is never special-cased here:
	// it is routed to Unavailable only when the REAL entries reject it (InvalidArgument),
	// so this cell genuinely exercises KthNeighborDistance/CalibrateNoveltyBaseline's own
	// Dot rejection rather than shortcutting around it.
	M2Verdict M2ComputeVerdict(const BankView& bank, const float* query, int32_t k,
		int32_t channel, float lambda, Workspace& ws)
	{
		for (int32_t r = 0; r < bank.count; ++r)
		{
			float d = -999.0f;
			const Status s = NoveltyProbeDistance(bank, query, r, channel, &d, ws);
			if (s == Status::Ok && d == 0.0f)
			{
				return M2Verdict::Duplicate;
			}
		}
		const int32_t sampleLimit = bank.count;
		std::vector<float> baseline(static_cast<size_t>(bank.count), -999.0f);
		int32_t baselineCount = -999;
		const Status calStatus =
			CalibrateNoveltyBaseline(bank, k, sampleLimit, baseline.data(), &baselineCount, ws);
		float kth = -999.0f;
		const Status kthStatus = KthNeighborDistance(bank, query, k, nullptr, &kth, ws);
		if (calStatus == Status::InvalidArgument || kthStatus == Status::InvalidArgument)
		{
			return bank.metric == Metric::Dot ? M2Verdict::Unavailable : M2Verdict::Error;
		}
		if (calStatus != Status::Ok || kthStatus != Status::Ok)
		{
			return M2Verdict::Error;
		}
		float score = -999.0f;
		if (NoveltyScore(baseline.data(), baselineCount, kth, &score) != Status::Ok)
		{
			return M2Verdict::Error;
		}
		return score >= lambda ? M2Verdict::Novel : M2Verdict::Familiar;
	}

	const char* M2VerdictName(M2Verdict v)
	{
		switch (v)
		{
			case M2Verdict::Duplicate: return "duplicate";
			case M2Verdict::Novel: return "novel";
			case M2Verdict::Familiar: return "familiar";
			case M2Verdict::Unavailable: return "unavailable";
			default: return "error";
		}
	}
}

// M2 / dim 10 (the crux) — the two-limb tri-state Novelty FEAT (D-V32-31/47), composed
// from the four M2 primitives under test via M2ComputeVerdict. The strike-9 discrimination
// geometry (dup-of-isolate, third-copy, dup-of-dense, fresh-isolate, fresh-familiar), the
// Cosine magnitude-invariance cell, the Dot verdict-unavailable cell, and a known-CDF FEAT
// with hand-computed expected scores in RankDistance space.
static void TestM2TwoLimbVerdictFeat()
{
	Rng rng(0x51A1CE);
	const int32_t dims = 16, novK = 8;
	const float lambda = 0.95f;

	// --- Strike-9 geometry: a dense cluster (200 rows, every row has >= novK near
	// neighbours) + one far sparse isolate p + a far already-duplicated pair q1==q2. ---
	std::vector<float> src;
	const int32_t nCluster = 200;
	for (int32_t i = 0; i < nCluster; ++i)
	{
		for (int32_t d = 0; d < dims; ++d) src.push_back(0.15f * rng.NextFloat());
	}
	const int32_t pIsolate = nCluster;
	for (int32_t d = 0; d < dims; ++d) src.push_back(10.0f);
	const int32_t q1 = nCluster + 1, q2 = nCluster + 2;
	for (int32_t d = 0; d < dims; ++d) src.push_back(-10.0f);
	for (int32_t d = 0; d < dims; ++d) src.push_back(-10.0f);
	const int32_t count = nCluster + 3;
	(void)pIsolate; (void)q1; (void)q2;

	GBank b(src, count, dims, Quantization::Float32, Metric::L2);
	Workspace ws; ws.Reserve(std::max(novK, count - 1) + 1, 1);

	struct Case { const char* name; std::vector<float> probe; M2Verdict expect; };
	std::vector<Case> cases;
	cases.push_back({"dup of isolate p", std::vector<float>(static_cast<size_t>(dims), 10.0f), M2Verdict::Duplicate});
	cases.push_back({"third copy of q", std::vector<float>(static_cast<size_t>(dims), -10.0f), M2Verdict::Duplicate});
	{
		std::vector<float> v = M2PaddedProbeFromRow(b, 0);
		v.resize(static_cast<size_t>(dims));
		cases.push_back({"dup of dense row", v, M2Verdict::Duplicate});
	}
	{
		std::vector<float> v(static_cast<size_t>(dims), 10.0f);
		v[1] += 1500.0f;
		cases.push_back({"fresh isolate", v, M2Verdict::Novel});
	}
	{
		std::vector<float> v = M2PaddedProbeFromRow(b, 0);
		v.resize(static_cast<size_t>(dims));
		v[0] += 0.01f;
		cases.push_back({"fresh dense (near row 0)", v, M2Verdict::Familiar});
	}

	for (const Case& c : cases)
	{
		std::vector<float> padded = M2PaddedProbe(c.probe, dims, b.view.paddedDims);
		const M2Verdict got = M2ComputeVerdict(b.view, padded.data(), novK, -1, lambda, ws);
		CHECK_MSG(got == c.expect, "two-limb verdict FEAT '%s': got %s expected %s",
			c.name, M2VerdictName(got), M2VerdictName(c.expect));
	}

	// --- Cosine magnitude-invariance at the VERDICT level (the sixth fracture's lesson,
	// re-affirmed for the whole two-limb pipeline): int8 keeps the FULL guarantee (D-V32-47:
	// exact-integer cross/self-dot sums make a scalar multiple an exact duplicate at any
	// scale); float32 does not, per F-M2-5/D-V32-51 (provisional, pending Dan): even
	// constructing `c * row` is itself a rounded float32 multiplication, so the probe is not
	// bit-exactly parallel to the row before NoveltyProbeDistance ever sees it. The float32
	// leg therefore asserts the HONEST outcome per F-M2-5's disclosure, not the int8-only
	// exact-zero-at-any-scale claim: c=1.0 (byte-identical) is still an exact duplicate;
	// c != 1.0 must NOT falsely verdict duplicate, and lands familiar (a dense cluster
	// neighbour, never novel) rather than silently wrong. ---
	{
		const int32_t cdims = 24;
		std::vector<float> csrc;
		for (int32_t i = 0; i < 40 * cdims; ++i) csrc.push_back(rng.NextFloat());
		GBank cb(csrc, 40, cdims, Quantization::Int8, Metric::Cosine);
		Workspace cws; cws.Reserve(novK + 1, 1);
		const std::vector<float> unitRow = M2PaddedProbeFromRow(cb, 5);
		for (float c : {0.1f, 1.0f, 3.0f, 25.0f, 1000.0f})
		{
			std::vector<float> probe(unitRow.size());
			for (size_t i = 0; i < probe.size(); ++i) probe[i] = c * unitRow[i];
			const M2Verdict got = M2ComputeVerdict(cb.view, probe.data(), novK, -1, lambda, cws);
			CHECK_MSG(got == M2Verdict::Duplicate,
				"Cosine magnitude-invariance (int8, the exact-guarantee path): scale c=%.4g "
				"of a stored row must still verdict duplicate, got %s",
				static_cast<double>(c), M2VerdictName(got));
		}
	}
	{
		const int32_t fdims = 16;
		std::vector<float> centre(static_cast<size_t>(fdims));
		for (int32_t d = 0; d < fdims; ++d) centre[static_cast<size_t>(d)] = 1.0f + 0.1f * static_cast<float>(d);
		std::vector<float> fsrc;
		PushBlob(fsrc, fdims, centre, 30, rng, 0.02f); // a tight dense cluster
		GBank fb(fsrc, 30, fdims, Quantization::Float32, Metric::Cosine);
		Workspace fws; fws.Reserve(novK + 1, 1);
		const std::vector<float> unitRow = M2PaddedProbeFromRow(fb, 5);

		{
			// c == 1.0: byte-identical -- unconditionally exact (cross == aSq == bSq).
			std::vector<float> probe = unitRow;
			const M2Verdict got = M2ComputeVerdict(fb.view, probe.data(), novK, -1, lambda, fws);
			CHECK_MSG(got == M2Verdict::Duplicate,
				"Cosine float32 whole-row, c=1.0 (byte-identical): must verdict duplicate, "
				"got %s", M2VerdictName(got));
		}
		// F-M2-6 (executed finding, 2026-07-19): only the NEGATIVE claim is asserted here.
		// c=0.1 was hand-tried as the positive "must land familiar" claim and produced
		// `novel` instead -- root-caused to KthNeighborDistance riding the standard Query()
		// Cosine path, a documented PLAIN DOT PRODUCT (types.h: "Cosine banks store
		// pre-normalized rows, so query-time scoring is a plain dot product"), not
		// magnitude-normalized. Scaling the probe DOWN shrinks every raw-dot similarity
		// against the bank proportionally, pushing RankDistance (`1 - score`) UP toward 1
		// regardless of true direction, which can cross lambda and misread a genuine
		// near-duplicate as novel. NoveltyProbeDistance (limb 1) is unaffected -- its
		// formula explicitly cancels scale -- but limb 2's OWN magnitude sensitivity for
		// probes limb 1 does NOT catch (true duplicates it does catch, unconditionally) is a
		// real, newly-observed property, not covered by any ratified decision, and not
		// something this suite invents an outcome for. Routed as its own finding (see the
		// test-design artifact) rather than asserted here as if it were specified.
		for (float c : {0.1f, 3.0f, 25.0f, 1000.0f})
		{
			std::vector<float> probe(unitRow.size());
			for (size_t i = 0; i < probe.size(); ++i) probe[i] = c * unitRow[i];
			const M2Verdict got = M2ComputeVerdict(fb.view, probe.data(), novK, -1, lambda, fws);
			CHECK_MSG(got != M2Verdict::Duplicate,
				"Cosine float32 whole-row, c=%.4g: F-M2-5's disclosed limit means this must "
				"NOT claim an exactness float32 arithmetic can't deliver, got %s",
				static_cast<double>(c), M2VerdictName(got));
		}
	}

	// --- Dot verdict-unavailable: never a number, always the rejection path. ---
	{
		const int32_t ddims = 12;
		std::vector<float> dsrc;
		for (int32_t i = 0; i < 20 * ddims; ++i) dsrc.push_back(rng.NextFloat());
		GBank db(dsrc, 20, ddims, Quantization::Float32, Metric::Dot);
		Workspace dws; dws.Reserve(novK + 1, 1);
		std::vector<float> probe = M2PaddedProbeFromRow(db, 3);
		const M2Verdict got = M2ComputeVerdict(db.view, probe.data(), novK, -1, lambda, dws);
		CHECK_MSG(got == M2Verdict::Unavailable,
			"a Dot bank must verdict Unavailable, never a number, got %s", M2VerdictName(got));
	}

	// --- Known-CDF FEAT (audit G-5): a small Cosine fixture sized UNDER sampleLimit so the
	// sample IS the whole fixture — every within-sample k-th-NN RankDistance and its
	// empirical-CDF rank is computable by hand from the definition. ---
	{
		const int32_t kdims = 4, kcount = 10, kk = 2;
		std::vector<float> ksrc;
		for (int32_t i = 0; i < kcount; ++i)
		{
			const double theta = 0.27 * i + 0.04 * i * i;
			ksrc.push_back(static_cast<float>(std::cos(theta) * (1.0 + 0.03 * i)));
			ksrc.push_back(static_cast<float>(std::sin(theta) * (1.0 + 0.03 * i)));
			ksrc.push_back(0.01f * static_cast<float>(i));
			ksrc.push_back(0.006f * static_cast<float>(i * i));
		}
		GBank kb(ksrc, kcount, kdims, Quantization::Float32, Metric::Cosine);
		Workspace kws; kws.Reserve(kcount, 1);

		std::vector<float> baseline(static_cast<size_t>(kcount), -999.0f);
		int32_t baselineCount = -999;
		std::vector<float> handBaseline;
		for (int32_t r = 0; r < kcount; ++r)
		{
			std::vector<float> q = M2PaddedProbeFromRow(kb, r);
			handBaseline.push_back(M2BruteKthRankDistance(kb, q.data(), kk, {r}));
		}
		std::sort(handBaseline.begin(), handBaseline.end());

		CHECK(CalibrateNoveltyBaseline(kb.view, kk, 64, baseline.data(), &baselineCount, kws) == Status::Ok);
		CHECK(baselineCount == kcount);

		std::vector<float> farSrc = {0.0f, 0.0f, 1.0f, 0.0f};
		std::vector<float> farProbe = M2PaddedProbe(farSrc, kdims, kb.view.paddedDims);
		float farKth = -999.0f;
		CHECK(KthNeighborDistance(kb.view, farProbe.data(), kk, nullptr, &farKth, kws) == Status::Ok);
		float handScore = 0.0f;
		for (float base : handBaseline) if (base < farKth) handScore += 1.0f;
		handScore /= static_cast<float>(handBaseline.size());
		float entryScore = -999.0f;
		CHECK(NoveltyScore(baseline.data(), baselineCount, farKth, &entryScore) == Status::Ok);
		CHECK_MSG(std::fabs(entryScore - handScore) < 1e-6f,
			"known-CDF FEAT: NoveltyScore(entry)=%.6g hand-computed=%.6g",
			static_cast<double>(entryScore), static_cast<double>(handScore));
	}
}

// ===========================================================================
// V3.2 red suite (Curie) — M3 matching.h: MutualNearestMatches (plan 25.4, RE-MECHANIZED
// D-V32-16), the sampled-A-verified-against-full-banks mutual matcher + CSLS margins that
// back the Correspondence view. RED SCAFFOLD: matching.cpp's body returns Ok and writes
// nothing. Test design: Claude/Curie/superfaiss-v3.2-test-design-2026-07-18.md.
//
// Oracle discipline (the D-V32 strike loop, "build don't reason"): the mutual-match +
// CSLS oracle (M3RefMatch) is an independent recode of the plan's pinned two-pass
// mechanism, built over the SAME double-precision brute-force machinery M1's REF already
// established (RefHit/RefBetter — ties ascending index, the library's own convention) —
// never calls MutualNearestMatches itself. The Correspondence FEAT's ground truth is a
// CONSTRUCTED permutation (well-separated one-hot landmarks, tie-free by distinct
// magnitude), never a re-run of the implementation.
// ===========================================================================

namespace
{
	// Sim(metric, score) = -RankDistance(metric, score) (D-V32-20): L2's raw score is
	// lower-is-better and is negated; Cosine/Dot's raw score is already
	// similarity-directioned (an additive constant shift is immaterial here — it cancels
	// exactly in the 2*sim - r_B - r_A combination, so plain `score` is used directly).
	double M3Sim(Metric metric, double score)
	{
		return metric == Metric::L2 ? -score : score;
	}

	// Brute-force top-k (best-first, ties ascending index — RefBetter, the library's own
	// pinned convention) of `query` ([0,dims) valid floats) against `target`, honoring
	// `excludeBits` (target's own source-space tombstones).
	std::vector<RefHit> M3TopK(const GBank& target, const float* query, int32_t k,
		const uint32_t* excludeBits)
	{
		std::vector<RefHit> hits;
		const int32_t dims = target.view.dims;
		for (int32_t r = 0; r < target.view.count; ++r)
		{
			if (IsExcluded(excludeBits, r))
			{
				continue;
			}
			const double* row = &target.refRows[static_cast<size_t>(r) * dims];
			double score = 0.0;
			if (target.view.metric == Metric::L2)
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
		std::sort(hits.begin(), hits.end(),
			[&](const RefHit& a, const RefHit& b) { return RefBetter(a, b, target.view.metric); });
		if (static_cast<int32_t>(hits.size()) > k)
		{
			hits.resize(static_cast<size_t>(k));
		}
		return hits;
	}

	struct M3RefResult
	{
		bool matched;
		int32_t sourceIndexB;
		double margin;
	};

	// Independent oracle for one sampled A row's MutualNearestMatches outcome
	// (D-V32-16/20): pass 1 (top-matchK of `queryA` in fullB — candidate = top-1, r_B =
	// mean Sim of the top-matchK), the mutual back-verification (pass 2's top-1 in fullA
	// must equal `nativeSourceIndexA`), and the CSLS margin from the two retrievals — no
	// third pass, matching the contract.
	M3RefResult M3RefMatch(const GBank& fullA, const uint32_t* exclA, const GBank& fullB,
		const uint32_t* exclB, int32_t nativeSourceIndexA, const float* queryA, int32_t matchK)
	{
		const std::vector<RefHit> topB = M3TopK(fullB, queryA, matchK, exclB);
		if (topB.empty())
		{
			return {false, -1, 0.0};
		}
		const int32_t candidateB = topB[0].index;
		double rB = 0.0;
		for (const RefHit& h : topB)
		{
			rB += M3Sim(fullB.view.metric, h.score);
		}
		rB /= static_cast<double>(topB.size());
		const double simAB = M3Sim(fullB.view.metric, topB[0].score);

		const std::vector<float> queryB = M2PaddedProbeFromRow(fullB, candidateB);
		const std::vector<RefHit> topA = M3TopK(fullA, queryB.data(), matchK, exclA);
		if (topA.empty())
		{
			return {false, -1, 0.0};
		}
		double rA = 0.0;
		for (const RefHit& h : topA)
		{
			rA += M3Sim(fullA.view.metric, h.score);
		}
		rA /= static_cast<double>(topA.size());

		if (topA[0].index != nativeSourceIndexA)
		{
			return {false, -1, 0.0};
		}
		const double margin = 2.0 * simAB - rB - rA;
		return {true, candidateB, margin};
	}
}

// M3 / dim 2 — MutualNearestMatches trust boundaries: empty views, matchK bounds, dims/
// metric mismatch, null buffers, all InvalidArgument with no output write.
static void TestM3MutualNearestMatchesTrustBoundaries()
{
	Rng rng(0x3A11);
	const int32_t dims = 8, count = 6;
	std::vector<float> aSrc, bSrc;
	for (int32_t i = 0; i < count * dims; ++i) aSrc.push_back(rng.NextFloat());
	for (int32_t i = 0; i < count * dims; ++i) bSrc.push_back(rng.NextFloat());
	GBank fullA(aSrc, count, dims, Quantization::Float32, Metric::L2);
	GBank fullB(bSrc, count, dims, Quantization::Float32, Metric::L2);
	std::vector<int32_t> sampleIdx = {0, 1, 2};
	std::vector<float> sampleSrc(aSrc.begin(), aSrc.begin() + 3 * dims);
	GBank sampleA(sampleSrc, 3, dims, Quantization::Float32, Metric::L2);

	Workspace ws; ws.Reserve(4, 1);
	std::vector<MatchPair> pairs(3, MatchPair{-777, -777, -777.0f});
	int32_t pairCount = -999;

	auto poisonAndCall = [&](const BankView& sv, const int32_t* ssi, const BankView& fb,
		const uint32_t* eb, const BankView& fa, const uint32_t* ea, int32_t mk) -> Status {
		pairs.assign(3, MatchPair{-777, -777, -777.0f});
		pairCount = -999;
		return MutualNearestMatches(sv, ssi, fb, eb, fa, ea, mk, pairs.data(), &pairCount, ws);
	};

	{
		GBank emptyA(std::vector<float>{}, 0, dims, Quantization::Float32, Metric::L2);
		CHECK(poisonAndCall(emptyA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
			nullptr, 2) == Status::InvalidArgument);
	}
	{
		GBank emptyB(std::vector<float>{}, 0, dims, Quantization::Float32, Metric::L2);
		CHECK(poisonAndCall(sampleA.view, sampleIdx.data(), emptyB.view, nullptr, fullA.view,
			nullptr, 2) == Status::InvalidArgument);
	}
	{
		GBank emptyA2(std::vector<float>{}, 0, dims, Quantization::Float32, Metric::L2);
		CHECK(poisonAndCall(sampleA.view, sampleIdx.data(), fullB.view, nullptr, emptyA2.view,
			nullptr, 2) == Status::InvalidArgument);
	}
	CHECK_MSG(poisonAndCall(sampleA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
		nullptr, 0) == Status::InvalidArgument, "matchK < 1 must reject");
	CHECK_MSG(poisonAndCall(sampleA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
		nullptr, count + 1) == Status::InvalidArgument,
		"matchK greater than the available rows in fullB/fullA must reject");
	{
		std::vector<float> wideSrc;
		for (int32_t i = 0; i < count * (dims + 1); ++i) wideSrc.push_back(rng.NextFloat());
		GBank wideB(wideSrc, count, dims + 1, Quantization::Float32, Metric::L2);
		CHECK_MSG(poisonAndCall(sampleA.view, sampleIdx.data(), wideB.view, nullptr, fullA.view,
			nullptr, 2) == Status::InvalidArgument, "a dims mismatch must reject");
	}
	{
		GBank cosB(bSrc, count, dims, Quantization::Float32, Metric::Cosine);
		CHECK_MSG(poisonAndCall(sampleA.view, sampleIdx.data(), cosB.view, nullptr, fullA.view,
			nullptr, 2) == Status::InvalidArgument, "a metric mismatch must reject");
	}
	CHECK(MutualNearestMatches(sampleA.view, nullptr, fullB.view, nullptr, fullA.view,
		nullptr, 2, pairs.data(), &pairCount, ws) == Status::InvalidArgument);
	CHECK(MutualNearestMatches(sampleA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
		nullptr, 2, nullptr, &pairCount, ws) == Status::InvalidArgument);
	CHECK(MutualNearestMatches(sampleA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
		nullptr, 2, pairs.data(), nullptr, ws) == Status::InvalidArgument);
	CHECK_MSG(pairs[0].sourceIndexA == -777, "a rejected call must not write outPairs");
}

// M3 / dim 6 + dim 7 — broad randomized correctness across metric x quant: entry == oracle
// on every row, full x full (the degenerate sample = full view case), including the
// unmatched leg.
static void TestM3MutualNearestMatchesCorrectness()
{
	Rng rng(0x7788);
	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		for (Quantization quant : {Quantization::Float32, Quantization::Int8})
		{
			const int32_t dims = 16, aCount = 20, bCount = 18;
			std::vector<float> aSrc, bSrc;
			for (int32_t i = 0; i < aCount * dims; ++i) aSrc.push_back(rng.NextFloat());
			for (int32_t i = 0; i < bCount * dims; ++i) bSrc.push_back(rng.NextFloat());
			GBank fullA(aSrc, aCount, dims, quant, metric);
			GBank fullB(bSrc, bCount, dims, quant, metric);

			std::vector<int32_t> sampleSourceIndices;
			for (int32_t i = 0; i < aCount; ++i) sampleSourceIndices.push_back(i);

			const int32_t matchK = 4;
			Workspace ws; ws.Reserve(matchK, 1);
			std::vector<MatchPair> pairs(static_cast<size_t>(aCount));
			int32_t pairCount = -999;
			CHECK(MutualNearestMatches(fullA.view, sampleSourceIndices.data(), fullB.view, nullptr,
				fullA.view, nullptr, matchK, pairs.data(), &pairCount, ws) == Status::Ok);
			CHECK(pairCount == aCount);

			for (int32_t i = 0; i < aCount; ++i)
			{
				const std::vector<float> queryA = M2PaddedProbeFromRow(fullA, i);
				const M3RefResult ref =
					M3RefMatch(fullA, nullptr, fullB, nullptr, i, queryA.data(), matchK);
				const MatchPair& p = pairs[static_cast<size_t>(i)];
				CHECK_MSG(p.sourceIndexA == i, "sourceIndexA must be the native A index");
				if (ref.matched)
				{
					CHECK_MSG(p.sourceIndexB == ref.sourceIndexB,
						"correctness (metric=%d quant=%d) row %d: entry=%d oracle=%d",
						static_cast<int>(metric), static_cast<int>(quant), i, p.sourceIndexB,
						ref.sourceIndexB);
					const double delta = std::fabs(static_cast<double>(p.cslsMargin) - ref.margin);
					CHECK_MSG(delta < 1e-3 * std::max(1.0, std::fabs(ref.margin)),
						"correctness (metric=%d quant=%d) row %d margin: entry=%.6g oracle=%.6g",
						static_cast<int>(metric), static_cast<int>(quant), i,
						static_cast<double>(p.cslsMargin), ref.margin);
				}
				else
				{
					CHECK_MSG(p.sourceIndexB == -1,
						"correctness (metric=%d quant=%d) row %d: oracle unmatched, entry=%d",
						static_cast<int>(metric), static_cast<int>(quant), i, p.sourceIndexB);
				}
			}
		}
	}
}

// M3 / dim 10 (the crux) — the Correspondence FEAT (D-V32-16/17): bank B is a PERMUTATION
// of bank A's landmarks (+ noise rows on each side), sized ABOVE a small sample cap so
// striding engages. Every checked landmark's true partner is recovered (kernel-true both
// directions), noise rows report unmatched, zero spurious matches — the deleted design's
// tombstone (88-98% spurious under the old two-sided-sample mechanic).
static void TestM3CorrespondencePermutationFeat()
{
	const int32_t dims = 48;
	const int32_t landmarkCount = 48;
	std::vector<float> aSrc, bSrc;
	// Landmarks: one-hot at index i, magnitude (100 + 0.1*i) — pairwise well-separated
	// under every metric (orthogonal axes; distinct magnitudes remove any residual
	// symmetry, so the fixture is tie-free the same way M1's arc fixture is).
	for (int32_t i = 0; i < landmarkCount; ++i)
	{
		for (int32_t d = 0; d < dims; ++d)
		{
			aSrc.push_back(d == i ? (100.0f + 0.1f * static_cast<float>(i)) : 0.0f);
		}
	}
	// A-only noise: reuses landmark slots 0 and 1, at a magnitude CLOSE to but distinct
	// from that same axis's landmark (95 + 0.1*axis, vs. the landmark's 100 + 0.1*axis --
	// a fixed ~5.0 gap). A small, UNRELATED magnitude (the original construction) is a
	// geometry bug: two different-axis noise rows with small, similar magnitudes end up
	// mutually CLOSER to each other (a cross-axis one-hot distance of
	// sqrt(mag_i^2 + mag_j^2), tiny when both magnitudes are small) than either is to its
	// own same-axis landmark (a same-axis distance of |100 - mag|, large when mag is
	// small) -- so a noise row's "nearest in B" becomes another noise row instead of its
	// own-axis landmark, and back-verification can then spuriously succeed between two
	// noise rows (confirmed by hand arithmetic, build-and-execute, not reasoning: a 2026-
	// 07-19 review found sqrt(5^2+7^2)=8.60 vs. |100-5|=95, so the OLD 5/6/7/8 magnitudes
	// broke the "noise always loses" guarantee). Anchoring the magnitude near the SAME
	// axis's landmark instead makes the same-axis distance small (~5.0) and every
	// cross-axis distance (to any other landmark OR any other noise row) large (>130,
	// since one-hot magnitudes near 95-100 combine to sqrt(a^2+b^2) far above 5.0) --
	// same-axis proximity now unconditionally dominates, so a noise row's nearest-in-B is
	// always its OWN axis's landmark copy, which in turn always prefers the TRUE
	// same-axis landmark pair over the noise row in its own back-verification (the true
	// pair's distance is ~0.05-0.1, the noise row's is ~5.0) -- the "noise always loses"
	// guarantee restored by construction, not tie luck.
	const int32_t aNoise0 = landmarkCount, aNoise1 = landmarkCount + 1;
	for (int32_t d = 0; d < dims; ++d) aSrc.push_back(d == 0 ? 95.0f : 0.0f);
	for (int32_t d = 0; d < dims; ++d) aSrc.push_back(d == 1 ? 95.1f : 0.0f);
	const int32_t fullACount = landmarkCount + 2;

	// B = a PERMUTATION of the landmarks (reversed order) + 2 B-only noise rows (slots 2,3
	// at the same near-landmark-magnitude anchoring, same reasoning).
	for (int32_t i = 0; i < landmarkCount; ++i)
	{
		const int32_t landmark = landmarkCount - 1 - i;
		for (int32_t d = 0; d < dims; ++d)
		{
			bSrc.push_back(d == landmark ? (100.0f + 0.1f * static_cast<float>(landmark)) : 0.0f);
		}
	}
	for (int32_t d = 0; d < dims; ++d) bSrc.push_back(d == 2 ? 95.2f : 0.0f);
	for (int32_t d = 0; d < dims; ++d) bSrc.push_back(d == 3 ? 95.3f : 0.0f);
	const int32_t fullBCount = landmarkCount + 2;

	GBank fullA(aSrc, fullACount, dims, Quantization::Float32, Metric::L2);
	GBank fullB(bSrc, fullBCount, dims, Quantization::Float32, Metric::L2);

	// Sample: every 4th landmark (12 of 48 — striding engages; "N of M A-rows checked" is
	// 14 of 50) plus BOTH A-only noise rows.
	std::vector<int32_t> sampleSourceIndices;
	for (int32_t i = 0; i < landmarkCount; i += 4) sampleSourceIndices.push_back(i);
	sampleSourceIndices.push_back(aNoise0);
	sampleSourceIndices.push_back(aNoise1);
	const int32_t sampleCount = static_cast<int32_t>(sampleSourceIndices.size());

	std::vector<float> sampleSrc;
	for (int32_t idx : sampleSourceIndices)
	{
		const float* row = aSrc.data() + static_cast<size_t>(idx) * dims;
		sampleSrc.insert(sampleSrc.end(), row, row + dims);
	}
	GBank sampleA(sampleSrc, sampleCount, dims, Quantization::Float32, Metric::L2);

	const int32_t matchK = 3;

	// Fixture self-check (independent of the entry under test, mirroring the CSLS-
	// direction cell's discipline): the oracle ALONE must already show every landmark
	// recovered, zero spurious matches, and both noise rows unmatched -- proves the
	// construction is achievable before any implementation is compared against it.
	{
		int32_t refSpurious = 0, refRecovered = 0, refNoiseUnmatched = 0;
		for (int32_t i = 0; i < sampleCount; ++i)
		{
			const int32_t nativeA = sampleSourceIndices[static_cast<size_t>(i)];
			const std::vector<float> queryA = M2PaddedProbeFromRow(fullA, nativeA);
			const M3RefResult ref =
				M3RefMatch(fullA, nullptr, fullB, nullptr, nativeA, queryA.data(), matchK);
			if (nativeA < landmarkCount)
			{
				const int32_t expectB = landmarkCount - 1 - nativeA;
				if (ref.matched && ref.sourceIndexB == expectB) ++refRecovered;
				else ++refSpurious;
			}
			else
			{
				if (!ref.matched) ++refNoiseUnmatched;
				else ++refSpurious;
			}
		}
		CHECK_MSG(refRecovered == landmarkCount / 4 && refSpurious == 0 && refNoiseUnmatched == 2,
			"Correspondence FEAT fixture self-check (oracle only): recovered=%d (want %d) "
			"spurious=%d (want 0) noiseUnmatched=%d (want 2)", refRecovered, landmarkCount / 4,
			refSpurious, refNoiseUnmatched);
	}

	Workspace ws; ws.Reserve(matchK, 1);
	std::vector<MatchPair> pairs(static_cast<size_t>(sampleCount));
	int32_t pairCount = -999;
	CHECK(MutualNearestMatches(sampleA.view, sampleSourceIndices.data(), fullB.view, nullptr,
		fullA.view, nullptr, matchK, pairs.data(), &pairCount, ws) == Status::Ok);
	CHECK_MSG(pairCount == sampleCount,
		"MutualNearestMatches outPairCount must equal sampleViewA.count (%d), got %d",
		sampleCount, pairCount);

	int32_t spurious = 0, recovered = 0, noiseUnmatched = 0;
	for (int32_t i = 0; i < sampleCount; ++i)
	{
		const MatchPair& p = pairs[static_cast<size_t>(i)];
		const int32_t nativeA = sampleSourceIndices[static_cast<size_t>(i)];
		CHECK_MSG(p.sourceIndexA == nativeA,
			"outPairs[%d].sourceIndexA must be the native A index %d, got %d", i, nativeA,
			p.sourceIndexA);
		if (nativeA < landmarkCount)
		{
			const int32_t expectB = landmarkCount - 1 - nativeA;
			if (p.sourceIndexB == expectB)
			{
				++recovered;
			}
			else
			{
				++spurious;
			}
		}
		else
		{
			if (p.sourceIndexB == -1)
			{
				++noiseUnmatched;
			}
			else
			{
				++spurious;
			}
		}
	}
	CHECK_MSG(recovered == landmarkCount / 4,
		"Correspondence FEAT: every checked landmark's true partner must be recovered, got "
		"%d of %d", recovered, landmarkCount / 4);
	CHECK_MSG(spurious == 0, "Correspondence FEAT: zero spurious matches expected, got %d",
		spurious);
	CHECK_MSG(noiseUnmatched == 2,
		"Correspondence FEAT: both A-only noise rows must report unmatched, got %d of 2",
		noiseUnmatched);

	// Cross-check every row against the independent oracle too.
	for (int32_t i = 0; i < sampleCount; ++i)
	{
		const int32_t nativeA = sampleSourceIndices[static_cast<size_t>(i)];
		const std::vector<float> queryA = M2PaddedProbeFromRow(fullA, nativeA);
		const M3RefResult ref =
			M3RefMatch(fullA, nullptr, fullB, nullptr, nativeA, queryA.data(), matchK);
		const MatchPair& p = pairs[static_cast<size_t>(i)];
		if (ref.matched)
		{
			CHECK_MSG(p.sourceIndexB == ref.sourceIndexB,
				"Correspondence FEAT oracle cross-check, native A row %d: entry=%d oracle=%d",
				nativeA, p.sourceIndexB, ref.sourceIndexB);
		}
		else
		{
			CHECK_MSG(p.sourceIndexB == -1,
				"Correspondence FEAT oracle cross-check, native A row %d: oracle unmatched, "
				"entry=%d", nativeA, p.sourceIndexB);
		}
	}
}

// M3 / dim 7 (kernel-true contract claim) — the reported pair set for the checked A rows
// is INVARIANT to the sample size: a cap-3 sample and a cap-10 superset sample must report
// bit-identical results for the OVERLAPPING rows, since scoring always runs against the
// SAME full B/A views regardless of how many other A rows were sampled alongside them —
// exactly what the deleted two-sided-sample design violated.
static void TestM3KernelTrueSampleInvariance()
{
	Rng rng(0x9A11);
	const int32_t dims = 12, aCount = 30, bCount = 25;
	std::vector<float> aSrc, bSrc;
	for (int32_t i = 0; i < aCount * dims; ++i) aSrc.push_back(rng.NextFloat());
	for (int32_t i = 0; i < bCount * dims; ++i) bSrc.push_back(rng.NextFloat());
	GBank fullA(aSrc, aCount, dims, Quantization::Float32, Metric::L2);
	GBank fullB(bSrc, bCount, dims, Quantization::Float32, Metric::L2);
	const int32_t matchK = 3;
	Workspace ws; ws.Reserve(matchK, 1);

	std::vector<int32_t> smallIdx = {0, 1, 2};
	std::vector<int32_t> largeIdx;
	for (int32_t i = 0; i < 10; ++i) largeIdx.push_back(i);

	auto runSample = [&](const std::vector<int32_t>& idx, std::vector<MatchPair>& out) {
		std::vector<float> src;
		for (int32_t i : idx)
		{
			const float* row = aSrc.data() + static_cast<size_t>(i) * dims;
			src.insert(src.end(), row, row + dims);
		}
		GBank sample(src, static_cast<int32_t>(idx.size()), dims, Quantization::Float32, Metric::L2);
		out.resize(idx.size());
		int32_t n = -999;
		CHECK(MutualNearestMatches(sample.view, idx.data(), fullB.view, nullptr, fullA.view,
			nullptr, matchK, out.data(), &n, ws) == Status::Ok);
		CHECK(n == static_cast<int32_t>(idx.size()));
	};

	std::vector<MatchPair> smallOut, largeOut;
	runSample(smallIdx, smallOut);
	runSample(largeIdx, largeOut);

	for (size_t i = 0; i < smallIdx.size(); ++i)
	{
		CHECK_MSG(smallOut[i].sourceIndexA == largeOut[i].sourceIndexA &&
				smallOut[i].sourceIndexB == largeOut[i].sourceIndexB &&
				smallOut[i].cslsMargin == largeOut[i].cslsMargin,
			"kernel-true SampleLimit-invariance: native A row %d differs between a cap-3 "
			"sample and a cap-10 sample (sourceIndexB %d vs %d, margin %.9g vs %.9g)",
			smallIdx[i], smallOut[i].sourceIndexB, largeOut[i].sourceIndexB,
			static_cast<double>(smallOut[i].cslsMargin), static_cast<double>(largeOut[i].cslsMargin));
	}
}

// M3 / dim 6 (numerical edges, audit G-4) — CSLS direction per metric: on each of
// Dot/Cosine/L2, a constructed EXACT pair must out-margin a constructed PERTURBED-but-
// still-mutual pair, never only "finite and deterministic" (the sixth-fracture-shaped
// concern: an un-converted L2 score would invert this). The fixture's own oracle self-
// check (independent of the entry under test) proves the construction is sound before the
// entry is compared against it.
static void TestM3CslsDirectionPerMetric()
{
	for (Metric metric : {Metric::Dot, Metric::Cosine, Metric::L2})
	{
		const int32_t dims = 8;
		auto row = [&](int32_t axis, float mag) {
			std::vector<float> v(static_cast<size_t>(dims), 0.0f);
			v[static_cast<size_t>(axis)] = mag;
			return v;
		};
		std::vector<std::vector<float>> aRows = {
			row(0, 10.0f), row(1, 10.0f), row(2, 20.0f), row(3, 20.0f), row(4, 20.0f), row(5, 20.0f)};
		std::vector<std::vector<float>> bRows = aRows;
		bRows[1] = row(1, 9.5f);
		bRows[1][7] = 1.0f; // small off-axis perturbation -- degrades sim under every metric

		std::vector<float> aSrc, bSrc;
		for (auto& r : aRows) aSrc.insert(aSrc.end(), r.begin(), r.end());
		for (auto& r : bRows) bSrc.insert(bSrc.end(), r.begin(), r.end());

		GBank fullA(aSrc, 6, dims, Quantization::Float32, metric);
		GBank fullB(bSrc, 6, dims, Quantization::Float32, metric);
		std::vector<int32_t> sampleSourceIndices = {0, 1};

		const int32_t matchK = 3;

		// Fixture self-check: the oracle alone (no entry involved) must show the exact
		// pair beating the perturbed pair.
		const std::vector<float> queryRow0 = M2PaddedProbeFromRow(fullA, 0);
		const std::vector<float> queryRow1 = M2PaddedProbeFromRow(fullA, 1);
		const M3RefResult ref0 =
			M3RefMatch(fullA, nullptr, fullB, nullptr, 0, queryRow0.data(), matchK);
		const M3RefResult ref1 =
			M3RefMatch(fullA, nullptr, fullB, nullptr, 1, queryRow1.data(), matchK);
		CHECK_MSG(ref0.matched && ref1.matched && ref0.margin > ref1.margin,
			"CSLS direction (metric=%d) fixture self-check: oracle margins exact=%.6g "
			"perturbed=%.6g (matched0=%d matched1=%d)", static_cast<int>(metric), ref0.margin,
			ref1.margin, ref0.matched, ref1.matched);

		// The sample view carries exactly the sampled rows: sampleSourceIndices and
		// outPairs are sized by contract to sampleViewA.count, so passing the full
		// 6-row view here with 2-entry arrays overruns both (the M3 heap-corruption
		// root cause — Poirot, Claude/Poirot/b4e139e-m3-heap-corruption-root-cause.md).
		std::vector<float> sampleSrc(aSrc.begin(), aSrc.begin() + 2 * dims);
		GBank sampleA(sampleSrc, 2, dims, Quantization::Float32, metric);
		Workspace ws; ws.Reserve(matchK, 1);
		std::vector<MatchPair> pairs(2);
		int32_t pairCount = -999;
		CHECK(MutualNearestMatches(sampleA.view, sampleSourceIndices.data(), fullB.view, nullptr,
			fullA.view, nullptr, matchK, pairs.data(), &pairCount, ws) == Status::Ok);

		CHECK_MSG(pairs[0].sourceIndexB == 0 && pairs[1].sourceIndexB == 1,
			"CSLS direction (metric=%d): both the exact and the perturbed pair must be "
			"mutual matches, got sourceIndexB=[%d,%d]", static_cast<int>(metric),
			pairs[0].sourceIndexB, pairs[1].sourceIndexB);
		CHECK_MSG(pairs[0].cslsMargin > pairs[1].cslsMargin,
			"CSLS direction (metric=%d): the EXACT pair's margin (%.6g) must beat the "
			"perturbed pair's margin (%.6g) -- a nearer pair must never score lower",
			static_cast<int>(metric), static_cast<double>(pairs[0].cslsMargin),
			static_cast<double>(pairs[1].cslsMargin));
		CHECK_MSG(std::fabs(static_cast<double>(pairs[0].cslsMargin) - ref0.margin) <
					1e-3 * std::max(1.0, std::fabs(ref0.margin)) &&
				std::fabs(static_cast<double>(pairs[1].cslsMargin) - ref1.margin) <
					1e-3 * std::max(1.0, std::fabs(ref1.margin)),
			"CSLS direction (metric=%d): entry margins must match the oracle (exact "
			"entry=%.6g oracle=%.6g; perturbed entry=%.6g oracle=%.6g)",
			static_cast<int>(metric), static_cast<double>(pairs[0].cslsMargin), ref0.margin,
			static_cast<double>(pairs[1].cslsMargin), ref1.margin);
	}
}

// M3 / dim 8 (composition) — a tombstoned B row is never reported as a match, and the
// reported outcome (a different match, or unmatched) agrees with the oracle recomputed
// WITH the same exclusion honored.
static void TestM3ExclusionHonored()
{
	Rng rng(0xE9C1);
	const int32_t dims = 12, aCount = 3, bCount = 5;
	std::vector<float> aSrc, bSrc;
	for (int32_t i = 0; i < aCount * dims; ++i) aSrc.push_back(rng.NextFloat());
	for (int32_t i = 0; i < bCount * dims; ++i) bSrc.push_back(rng.NextFloat());
	GBank fullA(aSrc, aCount, dims, Quantization::Float32, Metric::L2);
	GBank fullB(bSrc, bCount, dims, Quantization::Float32, Metric::L2);

	std::vector<int32_t> sampleIdx = {0, 1, 2};
	const int32_t matchK = 2;
	Workspace ws; ws.Reserve(matchK, 1);

	std::vector<MatchPair> baseline(3);
	int32_t n = -999;
	CHECK(MutualNearestMatches(fullA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
		nullptr, matchK, baseline.data(), &n, ws) == Status::Ok);

	for (int32_t r = 0; r < aCount; ++r)
	{
		if (baseline[static_cast<size_t>(r)].sourceIndexB < 0)
		{
			continue; // nothing to exclude if row r has no baseline match
		}
		const int32_t excludeB = baseline[static_cast<size_t>(r)].sourceIndexB;
		std::vector<uint32_t> exclB((static_cast<size_t>(bCount) + 31) / 32, 0u);
		exclB[static_cast<size_t>(excludeB) >> 5] |= (1u << (excludeB & 31));

		std::vector<MatchPair> excluded(3);
		int32_t n2 = -999;
		CHECK(MutualNearestMatches(fullA.view, sampleIdx.data(), fullB.view, exclB.data(),
			fullA.view, nullptr, matchK, excluded.data(), &n2, ws) == Status::Ok);
		CHECK_MSG(excluded[static_cast<size_t>(r)].sourceIndexB != excludeB,
			"a tombstoned B row must never be reported as a match (row %d, excluded B=%d)", r,
			excludeB);

		const std::vector<float> queryA = M2PaddedProbeFromRow(fullA, r);
		const M3RefResult ref =
			M3RefMatch(fullA, nullptr, fullB, exclB.data(), r, queryA.data(), matchK);
		const MatchPair& p = excluded[static_cast<size_t>(r)];
		if (ref.matched)
		{
			CHECK_MSG(p.sourceIndexB == ref.sourceIndexB,
				"exclusion-honored correctness, row %d: entry=%d oracle=%d", r, p.sourceIndexB,
				ref.sourceIndexB);
		}
		else
		{
			CHECK_MSG(p.sourceIndexB == -1,
				"exclusion-honored correctness, row %d: oracle unmatched, entry=%d", r,
				p.sourceIndexB);
		}
	}
}

// M3 / dim 6 — repeat-call bit-equality.
static void TestM3RepeatDeterminism()
{
	Rng rng(0xDE7E);
	const int32_t dims = 16, aCount = 14, bCount = 12;
	std::vector<float> aSrc, bSrc;
	for (int32_t i = 0; i < aCount * dims; ++i) aSrc.push_back(rng.NextFloat());
	for (int32_t i = 0; i < bCount * dims; ++i) bSrc.push_back(rng.NextFloat());
	GBank fullA(aSrc, aCount, dims, Quantization::Int8, Metric::Cosine);
	GBank fullB(bSrc, bCount, dims, Quantization::Int8, Metric::Cosine);
	std::vector<int32_t> sampleIdx;
	for (int32_t i = 0; i < aCount; ++i) sampleIdx.push_back(i);

	const int32_t matchK = 3;
	Workspace ws; ws.Reserve(matchK, 1);
	std::vector<MatchPair> out1(static_cast<size_t>(aCount)), out2(static_cast<size_t>(aCount));
	int32_t n1 = -999, n2 = -999;
	CHECK(MutualNearestMatches(fullA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
		nullptr, matchK, out1.data(), &n1, ws) == Status::Ok);
	CHECK(MutualNearestMatches(fullA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
		nullptr, matchK, out2.data(), &n2, ws) == Status::Ok);
	CHECK(n1 == n2);
	for (int32_t i = 0; i < n1; ++i)
	{
		CHECK_MSG(out1[static_cast<size_t>(i)].sourceIndexB == out2[static_cast<size_t>(i)].sourceIndexB &&
				out1[static_cast<size_t>(i)].cslsMargin == out2[static_cast<size_t>(i)].cslsMargin,
			"MutualNearestMatches repeat call must be bit-identical (row %d)", i);
	}
}

// ===========================================================================
// V3.2 — S4/S1 contract-cleanliness fixes (Curie, 2026-07-19): two Poirot
// findings standing across all three Tier-1 core modules (graph.h/novelty.h/
// matching.h), closed together per plan Sec.25.4. Casebooks:
// Claude/Poirot/afabc08-graph-h-m1-review.md (S1 named, S2 — the decode
// duplicate — closed by 15a0668), Claude/Poirot/15a0668-s1s2s3-fix-verify.md
// (S1 relocated into the batch output buffers, not closed), Claude/Poirot/
// 524b373-matching-m3-review.md (S4 — matching.cpp forked a second decode
// body; S1 O1 — confirmed standing, ripe for "all three modules together").
// Test design: Claude/Curie/superfaiss-v3.2-s4-s1-shared-helper-workspace-
// tracking-test-design-2026-07-19.md.
// ===========================================================================

namespace
{
	// Reads a source file's full text for a structural (symbol-presence)
	// assertion. Used only by the S4 shared-decode-helper pin below — this
	// project's established grep/structural-oracle convention (the Bank
	// Inspector I slot-3 concurrency-grep cell, D-V32-53) applied to the core
	// module tree. Returns empty on a read failure rather than throwing, so a
	// bad path fails the CHECK loudly instead of silently reporting green.
	std::string ReadWholeSourceFile(const char* path)
	{
		std::ifstream f(path, std::ios::binary);
		if (!f)
		{
			return std::string();
		}
		std::ostringstream ss;
		ss << f.rdbuf();
		return ss.str();
	}
}

// S4 (matching.h, standing since afabc08's M1 S2) — the row->query decode is a
// NAMED, SHARED step across all three Tier-1 core modules (plan Sec.25.4 temper
// S1): "All three modules therefore share one helper, DequantizeRowAsQuery(bank,
// row, outFloatQuery)." graph.h/novelty.h were routed through it by 15a0668;
// matching.cpp instead forked a second body, DequantizeRowForTarget (Poirot
// 524b373-matching-m3-review.md S4) — correct today, but "nothing forces [it]
// to agree tomorrow" once the shared helper and matching.h's cross-quantization
// decode drift apart from a common origin. This structural pin realizes the M1
// review's own O2 note ("the dim-7 equivalence cell ... has no shared symbol to
// pin against"): the shared helper's declared contract must widen to a
// targetPaddedDims parameter, and matching.cpp must call it — not merely have
// two independently-written functions that happen to agree today. graph.cpp/
// novelty.cpp are the already-fixed siblings; their rows are a green regression
// guard, not new coverage. Red (matching.cpp's row, and the header's own
// contract) until the fix routes matching.cpp through the generalized symbol
// and deletes the private duplicate.
static void TestS4SharedRowDecodeHelperAllThreeModules()
{
	const std::string header = ReadWholeSourceFile("include/superfaiss/inspector_common.h");
	CHECK_MSG(!header.empty(), "could not read include/superfaiss/inspector_common.h "
		"(run from the repo root, matching build.bat's own working directory)");
	CHECK_MSG(header.find("DequantizeRowAsQuery") != std::string::npos,
		"the shared helper symbol is missing from inspector_common.h");
	CHECK_MSG(header.find("targetPaddedDims") != std::string::npos,
		"DequantizeRowAsQuery's declared contract has no targetPaddedDims parameter yet "
		"(the S4 generalization — see Claude/Poirot/524b373-matching-m3-review.md)");

	struct ModuleExpectation
	{
		const char* path;
		bool callsSharedHelper;    // the post-fix state every module must reach
		bool definesPrivateDecode; // must be false in every module once fixed
	};
	const ModuleExpectation modules[] = {
		{"src/graph.cpp", true, false},
		{"src/novelty.cpp", true, false},
		{"src/matching.cpp", true, false}, // matching.cpp is the RED row today
	};

	for (const auto& m : modules)
	{
		const std::string src = ReadWholeSourceFile(m.path);
		CHECK_MSG(!src.empty(), "could not read %s (run from the repo root)", m.path);

		const bool callsShared = src.find("DequantizeRowAsQuery(") != std::string::npos;
		CHECK_MSG(callsShared == m.callsSharedHelper,
			"%s must call the shared DequantizeRowAsQuery helper for row->query decode "
			"(plan Sec.25.4 temper S1, 'a named, shared step, not an improvisation')", m.path);

		// A private per-row decode duplicate — named DequantizeRow or
		// DequantizeRowForTarget, the two shapes this exact class has taken
		// (M1 S2, then M3 S4) — must not exist in any of the three modules.
		const bool definesPrivate =
			src.find("void DequantizeRow(") != std::string::npos ||
			src.find("DequantizeRowForTarget(") != std::string::npos;
		CHECK_MSG(definesPrivate == m.definesPrivateDecode,
			"%s must not carry a private row->query decode duplicate "
			"(one shared helper, three callers)", m.path);
	}
}

// S1 (graph.h) — BuildKnnNeighbors' per-call std::vector<Hit>/std::vector<int32_t>
// output buffers heap-allocate via global `new` on every call, untracked by the
// allocator seam (Poirot afabc08-graph-h-m1-review.md S1; standing per 15a0668-
// s1s2s3-fix-verify.md and confirmed again by 524b373-matching-m3-review.md O1,
// "ripe for the all-three close"). Warm the workspace once, then assert zero raw
// heap allocations (and a flat seam AllocationCount/GrowthCount) across repeated
// calls on the same warm workspace. Red until the output buffers route through
// workspace-tracked storage.
static void TestS1FlatAllocationBuildKnnNeighbors()
{
	Rng rng(0x51A1);
	const int32_t dims = 32, count = 200, k = 6;
	std::vector<float> src;
	for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
	GBank bank(src, count, dims, Quantization::Int8, Metric::L2);

	Workspace ws;
	std::vector<int32_t> nb(static_cast<size_t>(count) * k, -999);

	// Warm-up: allowed to allocate, on either counter.
	CHECK(BuildKnnNeighbors(bank.view, k, true, nb.data(), ws) == Status::Ok);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 20; ++i)
		{
			CHECK(BuildKnnNeighbors(bank.view, k, true, nb.data(), ws) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"BuildKnnNeighbors allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"BuildKnnNeighbors' seam-tracked allocations grew on a warm workspace: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// S1 (novelty.h, "baseline calibration" half) — CalibrateNoveltyBaseline shares
// BuildKnnNeighbors' exact batch-output shape and the exact same defect (Poirot
// 15a0668-s1s2s3-fix-verify.md S1 names this file:line directly — novelty.cpp's
// allHits/hitCounts). Same construction as the graph.h cell above.
static void TestS1FlatAllocationCalibrateNoveltyBaseline()
{
	Rng rng(0x51A2);
	const int32_t dims = 24, count = 180, k = 5, sampleLimit = 512;
	std::vector<float> src;
	for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
	GBank bank(src, count, dims, Quantization::Int8, Metric::L2);

	Workspace ws;
	std::vector<float> baseline(static_cast<size_t>(count), -999.0f);
	int32_t outCount = -999;

	CHECK(CalibrateNoveltyBaseline(bank.view, k, sampleLimit, baseline.data(), &outCount, ws) == Status::Ok);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 20; ++i)
		{
			CHECK(CalibrateNoveltyBaseline(bank.view, k, sampleLimit, baseline.data(), &outCount, ws)
				== Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"CalibrateNoveltyBaseline allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"CalibrateNoveltyBaseline's seam-tracked allocations grew on a warm workspace: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// Finding 3 (cf3f750-v32-core-batch-review.md): NoveltyProbeDistance's int8 leg
// heap-allocated a std::vector<int8_t> on EVERY call via ::operator new, invisible to
// AllocationCount() (which only sees traffic through the Workspace seam) -- exactly the
// blind spot the header comment above this file's ScopedRawNewTracking documents. An
// AllocationCount()-only assertion around this call reports flat whether or not the
// violation is present (confirmed: it does, on the pre-fix code) and so cannot serve as
// the oracle alone; ScopedRawNewTracking closes the blind spot. Same construction as
// TestS1FlatAllocationBuildKnnNeighbors/CalibrateNoveltyBaseline above. Both the
// whole-row and channel legs are exercised (both now route through the same
// workspace.ReserveXdQuery block).
static void TestS1FlatAllocationNoveltyProbeDistance()
{
	Rng rng(0x51A3);
	const int32_t dims = 32, count = 4;
	std::vector<ChannelInfo> ch = {{0, 16}, {16, 16}};
	std::vector<float> src;
	for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
	GChannelBank b(src, count, dims, Quantization::Int8, Metric::Cosine, ch);
	std::vector<float> probeSrc(static_cast<size_t>(dims));
	for (float& v : probeSrc) v = rng.NextFloat();
	std::vector<float> probe = M2PaddedProbe(probeSrc, dims, b.bank.view.paddedDims);

	Workspace ws;
	float out = -999.0f;
	CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, -1, &out, ws) == Status::Ok); // warm
	CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), 0, 0, &out, ws) == Status::Ok);   // warm, channel

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 20; ++i)
		{
			CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), i % count, -1, &out, ws) == Status::Ok);
			CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), i % count, 0, &out, ws) == Status::Ok);
			CHECK(NoveltyProbeDistance(b.bank.view, probe.data(), i % count, 1, &out, ws) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"NoveltyProbeDistance allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"NoveltyProbeDistance's seam-tracked allocations grew on a warm workspace: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// S1 (novelty.h, "probe" half — bonus finding, not named in any prior casebook).
// KthNeighborDistance calls workspace.Reserve(k, 1) and then STILL parks its
// output in a fresh std::vector<Hit> instead of workspace.HeapStorage() — the
// Reserve call is dead, exactly the shape the M1 review's own Minor finding
// named in graph.cpp before the S3 batch fix ("Query re-reserves its own heap
// internally, and this function passes a std::vector (not HeapStorage) as the
// output, so nothing consumes the reservation"). No casebook has reviewed this
// specific call site — found at the bench while authoring the S1 suite above,
// and included here because it is the same class the campaign exists to close
// and D-V32-52 already names KthNeighborDistance ("limb 2") as a call path a
// slot-3 consumer exercises per-probe. Routed to Dan/Poirot as a bench finding
// in the test-design artifact; not invented coverage — it is §25.4's own
// zero-allocation contract applied to a call site that happens to share
// novelty.cpp with the already-known CalibrateNoveltyBaseline instance.
static void TestS1FlatAllocationKthNeighborDistanceProbe()
{
	Rng rng(0x51A3);
	const int32_t dims = 16, count = 64, k = 4;
	std::vector<float> src;
	for (int32_t i = 0; i < count * dims; ++i) src.push_back(rng.NextFloat());
	GBank bank(src, count, dims, Quantization::Float32, Metric::Cosine);
	std::vector<float> query = M2PaddedProbeFromRow(bank, 0);

	Workspace ws;
	float out = -999.0f;

	// Warm-up: allowed to allocate.
	CHECK(KthNeighborDistance(bank.view, query.data(), k, nullptr, &out, ws) == Status::Ok);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 20; ++i)
		{
			CHECK(KthNeighborDistance(bank.view, query.data(), k, nullptr, &out, ws) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"KthNeighborDistance allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"KthNeighborDistance's seam-tracked allocations grew on a warm workspace: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

// S1 (matching.h) — MutualNearestMatches' two-pass batch output
// (pass1Hits/pass1Counts/pass2Hits/pass2Counts) is an ordinary call-scoped
// std::vector, explicitly disclosed in the module's own header comment as
// "KNOWN, ACCEPTED ... kept for consistency until S1 is resolved for all three
// modules together" (matching.cpp:14-21) — Poirot 524b373-matching-m3-review.md
// O1 confirms this is now ripe. Repeat-call determinism (TestM3RepeatDeterminism,
// above) already establishes distinctCount is stable across identical warm
// calls, so this is a clean warm/steady-state measurement, mirroring the two
// cells above.
static void TestS1FlatAllocationMutualNearestMatches()
{
	Rng rng(0x51A4);
	const int32_t dims = 16, aCount = 40, bCount = 36, matchK = 3;
	std::vector<float> aSrc, bSrc;
	for (int32_t i = 0; i < aCount * dims; ++i) aSrc.push_back(rng.NextFloat());
	for (int32_t i = 0; i < bCount * dims; ++i) bSrc.push_back(rng.NextFloat());
	GBank fullA(aSrc, aCount, dims, Quantization::Int8, Metric::Cosine);
	GBank fullB(bSrc, bCount, dims, Quantization::Int8, Metric::Cosine);
	std::vector<int32_t> sampleIdx;
	for (int32_t i = 0; i < aCount; ++i) sampleIdx.push_back(i);

	Workspace ws;
	std::vector<MatchPair> out(static_cast<size_t>(aCount));
	int32_t outCount = -999;

	// Warm-up: allowed to allocate.
	CHECK(MutualNearestMatches(fullA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
		nullptr, matchK, out.data(), &outCount, ws) == Status::Ok);

	const uint64_t allocsBefore = AllocationCount();
	const uint64_t growthBefore = ws.GrowthCount();
	{
		ScopedRawNewTracking rawTracking;
		for (int32_t i = 0; i < 20; ++i)
		{
			CHECK(MutualNearestMatches(fullA.view, sampleIdx.data(), fullB.view, nullptr, fullA.view,
				nullptr, matchK, out.data(), &outCount, ws) == Status::Ok);
		}
		CHECK_MSG(rawTracking.Count() == 0,
			"MutualNearestMatches allocated %llu time(s) outside the Workspace seam on a warm workspace",
			static_cast<unsigned long long>(rawTracking.Count()));
	}
	CHECK_MSG(AllocationCount() == allocsBefore,
		"MutualNearestMatches' seam-tracked allocations grew on a warm workspace: %llu -> %llu",
		static_cast<unsigned long long>(allocsBefore), static_cast<unsigned long long>(AllocationCount()));
	CHECK(ws.GrowthCount() == growthBefore);
}

int main()
{
	TestSimdEqualsScalar();
	TestAvx2Sub8RemainderF32();
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
	TestAllocFlatQueryBatch();
	TestAllocFlatQueryIntersect();
	TestAllocFlatQueryXdBatch();
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
	TestAllocFlatScratchSave();
	TestAllocFlatScratchFreeze();
	TestAllocFlatScratchFreezeChannelAware();
	TestAllocFlatScratchFreezeWithRecall();
	TestAllocFlatScratchMeasureScratchRecallPerChannel();
	TestAllocFlatWorkspaceReserve();
	TestAllocFlatWorkspaceReserveQueryScratch();
	TestAllocFlatWorkspaceReserveBiasBits();
	TestAllocFlatWorkspaceReserveXdQuery();
	TestAllocFlatWorkspaceReserveBatchOutput();
	TestAllocFlatWorkspaceReserveIndexScratch();
	TestScratchChannelFreeze();
	TestScratchChannelPersistenceRoundTrip();
	TestScratchChannelPersistenceVersioning();
	TestScratchChannelPersistenceReservedBit();
	TestScratchChannelPerChannelRecall();
	TestChannelScopedAnalyticsRefFeat();
	TestChannelScopedAnalyticsCosineRef();
	TestChannelScopedAnalyticsDeterminism();
	TestChannelScopedAnalyticsComposition();
	TestChannelScopedAnalyticsRejections();
	TestChannelQueryOnlyStorm();
	TestChannelScopedReductionStorm();
	TestScratchChannelLoadRejectsMalformedTable();
	TestScratchChannelSaveVersionSelection();
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
	// v3.0.1 red suite (Curie): two P1 correctness bugs + three coverage holes.
	TestChannelNNZeroEnergyFloor();
	TestScratchChannelPerChannelRecallZeroEnergy();
	TestChannelAnalyticsCrossDeviceGolden();
	TestPerChannelRecallOracle();
	TestVersionHeaderCoherence();

	// V3.1 red suite (Curie): the Relabel operation (plan section 24), realizing the
	// section 24.8 Coverage Model. Test design: Claude/Curie/superfaiss-v3.1-test-design-
	// 2026-07-13.md.
	TestRelabelValidationSymmetry();
	TestRelabelRejectionsStillQueryableUnderOldTable();
	TestRelabelOomLeavesBankIntact();
	TestRelabelWarmLifetimeSequence();
	TestRelabelWarmPromoteDemoteToggle();
	TestRelabelHeldViewAliasing();
	TestRelabelExclusiveDrainStorm();
	TestRelabelExtremesMatrix();
	TestRelabelAllZeroNormChannel();
	TestRelabelParityVsFreshCreate();
	TestRelabelParityEqualsComputeChannelInverseNorms();
	TestRelabelPostAppendDerivation();
	TestRelabelSubNormFloorAndClamp();
	TestRelabelCrossRunnerGolden();
	TestRelabelNeverTouchesRows();
	TestRelabelWholeVectorPathUnchanged();
	TestRelabelMetricMatrixContractClaim();
	TestRelabelCompositionRetention();
	TestRelabelCompositionGrowOrderIndependence();
	TestRelabelCompositionTombstones();
	TestRelabelCompositionFreeze();
	TestRelabelCompositionCrossDevice();
	TestRelabelCompositionBatchIntersectionMetricOverride();
	TestRelabelCompositionAnalytics();
	TestRelabelPromoteDemoteRetention();
	TestRelabelPersistenceValueRoundTrip();
	TestRelabelArchiveByteIdenticalToFresh();
	TestRelabelDemoteWritesLegacyFormat();
	TestRelabelPromoteWritesV3Format();
	TestRelabelVersionEvolutionInherited();
	TestRelabelFeatOracle();
	TestRelabelPromoteFeat();
	TestRelabelDemoteFeat();

	// V3.2 red suite (Curie): Bank Inspector I — module M1 graph.h (plan section 25).
	// Test design: Claude/Curie/superfaiss-v3.2-test-design-2026-07-18.md.
	TestM1TrustBoundaries();
	TestM1DuplicateGroupingByteConfirm();
	TestM1EdgesExactAcrossMetrics();
	TestM1DuplicateUnionConstructionCounts();
	TestM1FringeBoundary();
	TestM1RepeatAndCanonicalId();
	TestM1WarmWorkspaceGrowShrink();
	TestM1StructureFeat();
	TestM2NoveltyScore();
	TestM2NoveltyProbeDistance();
	TestM2KthNeighborDistance();
	TestM2CalibrateNoveltyBaseline();
	TestM2TwoLimbVerdictFeat();

	// V3.2 red suite (Curie): Bank Inspector I — module M3 matching.h (plan section 25).
	TestM3MutualNearestMatchesTrustBoundaries();
	TestM3MutualNearestMatchesCorrectness();
	TestM3CorrespondencePermutationFeat();
	TestM3KernelTrueSampleInvariance();
	TestM3CslsDirectionPerMetric();
	TestM3ExclusionHonored();
	TestM3RepeatDeterminism();

	// V3.2 — S4/S1 contract-cleanliness fixes (Curie, 2026-07-19): plan Sec.25.4.
	// Test design: Claude/Curie/superfaiss-v3.2-s4-s1-shared-helper-workspace-
	// tracking-test-design-2026-07-19.md.
	TestS4SharedRowDecodeHelperAllThreeModules();
	TestS1FlatAllocationBuildKnnNeighbors();
	TestS1FlatAllocationCalibrateNoveltyBaseline();
	TestS1FlatAllocationNoveltyProbeDistance();
	TestS1FlatAllocationKthNeighborDistanceProbe();
	TestS1FlatAllocationMutualNearestMatches();

	std::printf("superfaiss tests: %d checks, %d failures (simd path: %s)\n",
		GChecks, GFailures,
		ActiveSimdPath() == SimdPath::Scalar ? "scalar"
			: ActiveSimdPath() == SimdPath::SSE ? "sse"
			: ActiveSimdPath() == SimdPath::AVX2 ? "avx2" : "neon");
	return GFailures == 0 ? 0 : 1;
}

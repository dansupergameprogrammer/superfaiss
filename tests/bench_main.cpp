// SuperFAISS single-thread benchmark. Produces the calibration numbers the plugin's
// performance ceilings are pinned from (plan §13 / S-V2); the multi-threaded numbers
// come from the engine-side harness, since threading is the host's job.

#include "superfaiss/superfaiss.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace superfaiss;

namespace
{
	struct Rng
	{
		uint64_t state = 0x243F6A8885A308D3ull;
		uint64_t Next()
		{
			state ^= state >> 12;
			state ^= state << 25;
			state ^= state >> 27;
			return state * 0x2545F4914F6CDD1Dull;
		}
		float NextFloat()
		{
			return static_cast<float>(static_cast<int64_t>(Next() >> 24)) /
				static_cast<float>(1ll << 39);
		}
	};

	double Now()
	{
		return std::chrono::duration<double>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	struct Bench
	{
		Allocator alloc = DefaultAllocator();

		void Run(int32_t count, int32_t dims, Quantization quant, int32_t batch)
		{
			Rng rng;
			std::vector<float> src(static_cast<size_t>(count) * dims);
			for (auto& v : src)
			{
				v = rng.NextFloat();
			}

			const int32_t pd = PaddedDims(dims, quant);
			const size_t payloadBytes = static_cast<size_t>(count) * pd * ElementSize(quant);
			void* payload = alloc.alloc(payloadBytes, kAlignment, alloc.user);
			std::vector<float> scales;

			BankView bank;
			bank.rows = payload;
			bank.count = count;
			bank.dims = dims;
			bank.paddedDims = pd;
			bank.quant = quant;
			bank.metric = Metric::Dot;

			if (quant == Quantization::Float32)
			{
				PadRowsFloat32(src.data(), count, dims, pd, static_cast<float*>(payload));
			}
			else
			{
				scales.resize(static_cast<size_t>(count));
				QuantizeRowsInt8(src.data(), count, dims, pd, static_cast<int8_t*>(payload), scales.data());
				bank.scales = scales.data();
			}

			const int32_t k = 10;
			QueryParams params;
			params.k = k;
			Workspace ws;

			// Query buffers (batch-many, contiguous).
			void* qmem = alloc.alloc(static_cast<size_t>(batch) * pd * sizeof(float), kAlignment, alloc.user);
			float* queries = static_cast<float*>(qmem);
			for (int32_t m = 0; m < batch; ++m)
			{
				float* q = queries + static_cast<size_t>(m) * pd;
				for (int32_t i = 0; i < pd; ++i)
				{
					q[i] = i < dims ? rng.NextFloat() : 0.0f;
				}
			}

			std::vector<Hit> hits(static_cast<size_t>(batch) * k);
			std::vector<int32_t> counts(static_cast<size_t>(batch));

			// Warm-up.
			int32_t n = 0;
			Query(bank, queries, params, ws, hits.data(), &n);
			QueryBatch(bank, queries, batch, params, ws, hits.data(), counts.data());

			// Single query, best of 5 runs of `reps`.
			const int32_t reps = count >= 100000 ? 20 : 100;
			double bestSingle = 1e300;
			for (int32_t run = 0; run < 5; ++run)
			{
				const double t0 = Now();
				for (int32_t r = 0; r < reps; ++r)
				{
					Query(bank, queries, params, ws, hits.data(), &n);
				}
				const double perQuery = (Now() - t0) / reps;
				if (perQuery < bestSingle)
				{
					bestSingle = perQuery;
				}
			}

			// Batch, best of 5.
			double bestBatch = 1e300;
			for (int32_t run = 0; run < 5; ++run)
			{
				const double t0 = Now();
				for (int32_t r = 0; r < 5; ++r)
				{
					QueryBatch(bank, queries, batch, params, ws, hits.data(), counts.data());
				}
				const double perBatch = (Now() - t0) / 5;
				if (perBatch < bestBatch)
				{
					bestBatch = perBatch;
				}
			}

			const double amortization = (bestSingle * batch) / bestBatch;
			std::printf(
				"%8d x %4d %-7s | single %8.3f ms | batch%-3d %9.3f ms | amortization %5.2fx | %6.1f MB\n",
				count, dims, quant == Quantization::Float32 ? "float32" : "int8",
				bestSingle * 1e3, batch, bestBatch * 1e3, amortization,
				static_cast<double>(payloadBytes) / (1024.0 * 1024.0));

			alloc.free(qmem, alloc.user);
			alloc.free(payload, alloc.user);
		}
	};
}


namespace
{
	// V2 slot-1 calibration (plan section 10): mask bandwidth win, weight overhead,
	// segment break-even, V1 non-regression baseline, segmented-batch never-worse.
	struct SegBench
	{
		Allocator alloc = DefaultAllocator();

		static double Median(std::vector<double>& v)
		{
			std::sort(v.begin(), v.end());
			return v[v.size() / 2];
		}

		void Run(int32_t count, int32_t dims, Quantization quant)
		{
			Rng rng;
			std::vector<float> src(static_cast<size_t>(count) * dims);
			for (auto& v : src)
			{
				v = rng.NextFloat();
			}
			const int32_t pd = PaddedDims(dims, quant);
			const int64_t payloadBytes =
				static_cast<int64_t>(count) * pd * ElementSize(quant);
			void* payload = alloc.alloc(static_cast<size_t>(payloadBytes), kAlignment,
				alloc.user);
			std::vector<float> scales(static_cast<size_t>(count));
			BankView view;
			view.rows = payload;
			view.count = count;
			view.dims = dims;
			view.paddedDims = pd;
			view.quant = quant;
			view.metric = Metric::Dot;
			if (quant == Quantization::Float32)
			{
				PadRowsFloat32(src.data(), count, dims, pd, static_cast<float*>(payload));
				view.scales = nullptr;
			}
			else
			{
				QuantizeRowsInt8(src.data(), count, dims, pd,
					static_cast<int8_t*>(payload), scales.data());
				view.scales = scales.data();
			}

			void* qmem = alloc.alloc(static_cast<size_t>(pd) * sizeof(float), kAlignment,
				alloc.user);
			float* q = static_cast<float*>(qmem);
			std::memset(q, 0, static_cast<size_t>(pd) * sizeof(float));
			for (int32_t j = 0; j < dims; ++j)
			{
				q[j] = rng.NextFloat();
			}

			Workspace ws;
			std::vector<Hit> hits(16);
			int32_t n = 0;
			QueryParams plain;
			plain.k = 10;

			auto timeQuery = [&](const QueryParams& params) -> double
			{
				std::vector<double> samples;
				for (int32_t rep = 0; rep < 9; ++rep)
				{
					const double t0 = Now();
					Query(view, q, params, ws, hits.data(), &n);
					samples.push_back(Now() - t0);
				}
				return Median(samples) * 1000.0;
			};

			const double v1Ms = timeQuery(plain);

			const QuerySegment degen[1] = {{0, pd, 1.0f}};
			QueryParams degenParams = plain;
			degenParams.segments = degen;
			degenParams.segmentCount = 1;
			const double degenMs = timeQuery(degenParams);

			const int32_t grid = kAlignment / ElementSize(quant);
			const int32_t quarter = (pd / 4 / grid) * grid;
			const QuerySegment mask[1] = {{0, quarter, 1.0f}};
			QueryParams maskParams = plain;
			maskParams.segments = mask;
			maskParams.segmentCount = 1;
			const double maskMs = timeQuery(maskParams);

			QuerySegment eight[8];
			const int32_t seg8 = (pd / 8 / grid) * grid;
			for (int32_t i = 0; i < 8; ++i)
			{
				eight[i].offset = i * seg8;
				eight[i].length = seg8;
				eight[i].weight = 1.0f + 0.1f * static_cast<float>(i);
			}
			QueryParams eightParams = plain;
			eightParams.segments = eight;
			eightParams.segmentCount = 8;
			const double eightMs = timeQuery(eightParams);

			const int32_t m = 64;
			void* bmem = alloc.alloc(static_cast<size_t>(m) * pd * sizeof(float),
				kAlignment, alloc.user);
			float* batch = static_cast<float*>(bmem);
			std::memset(batch, 0, static_cast<size_t>(m) * pd * sizeof(float));
			for (int32_t qi = 0; qi < m; ++qi)
			{
				for (int32_t j = 0; j < dims; ++j)
				{
					batch[static_cast<int64_t>(qi) * pd + j] = rng.NextFloat();
				}
			}
			std::vector<Hit> bhits(static_cast<size_t>(m) * 10);
			std::vector<int32_t> bcounts(static_cast<size_t>(m));
			std::vector<double> bsamples;
			for (int32_t rep = 0; rep < 5; ++rep)
			{
				const double t0 = Now();
				QueryBatch(view, batch, m, eightParams, ws, bhits.data(), bcounts.data());
				bsamples.push_back(Now() - t0);
			}
			const double batchPerQueryMs = Median(bsamples) * 1000.0 / m;

			std::printf(
				"seg %7d x %3d %s: v1 %.3f ms | degen %.3f (%+.1f%%) | mask1/4 %.3f "
				"(%.2fx) | 8seg %.3f (%+.1f%%) | segbatch/q %.3f (%.2fx single)\n",
				count, dims, quant == Quantization::Int8 ? "int8" : "f32 ",
				v1Ms, degenMs, (degenMs / v1Ms - 1.0) * 100.0,
				maskMs, v1Ms / maskMs,
				eightMs, (eightMs / v1Ms - 1.0) * 100.0,
				batchPerQueryMs, batchPerQueryMs / eightMs);

			alloc.free(bmem, alloc.user);
			alloc.free(qmem, alloc.user);
			alloc.free(payload, alloc.user);
		}

		void BreakEven(int32_t count, int32_t dims)
		{
			Rng rng;
			std::vector<float> src(static_cast<size_t>(count) * dims);
			for (auto& v : src)
			{
				v = rng.NextFloat();
			}
			const int32_t pd = PaddedDims(dims, Quantization::Float32);
			void* payload = alloc.alloc(
				static_cast<size_t>(count) * pd * sizeof(float), kAlignment, alloc.user);
			PadRowsFloat32(src.data(), count, dims, pd, static_cast<float*>(payload));
			BankView view;
			view.rows = payload;
			view.count = count;
			view.dims = dims;
			view.paddedDims = pd;
			view.quant = Quantization::Float32;
			view.metric = Metric::Dot;

			void* qmem = alloc.alloc(static_cast<size_t>(pd) * sizeof(float), kAlignment,
				alloc.user);
			float* q = static_cast<float*>(qmem);
			std::memset(q, 0, static_cast<size_t>(pd) * sizeof(float));
			for (int32_t j = 0; j < dims; ++j)
			{
				q[j] = rng.NextFloat();
			}
			Workspace ws;
			std::vector<Hit> hits(16);
			int32_t n = 0;

			std::printf("break-even %d x %d f32 (seglen:ms)", count, dims);
			for (int32_t segLen = 32; segLen <= pd && pd % segLen == 0; segLen *= 2)
			{
				const int32_t segCount = pd / segLen;
				if (segCount > kMaxSegments)
				{
					continue;
				}
				std::vector<QuerySegment> segs(static_cast<size_t>(segCount));
				for (int32_t i = 0; i < segCount; ++i)
				{
					segs[static_cast<size_t>(i)].offset = i * segLen;
					segs[static_cast<size_t>(i)].length = segLen;
					segs[static_cast<size_t>(i)].weight = 1.0f;
				}
				QueryParams sp;
				sp.k = 10;
				sp.segments = segs.data();
				sp.segmentCount = segCount;
				std::vector<double> samples;
				for (int32_t rep = 0; rep < 9; ++rep)
				{
					const double t0 = Now();
					Query(view, q, sp, ws, hits.data(), &n);
					samples.push_back(Now() - t0);
				}
				std::printf(" %d:%.3f", segLen, Median(samples) * 1000.0);
			}
			std::printf("\n");
			alloc.free(qmem, alloc.user);
			alloc.free(payload, alloc.user);
		}
	};
}

namespace
{
	// V2.1 calibration (plan section 18.3): dense and sparse bias vs the unbiased
	// scan, single and batch. Quiet-machine medians, same harness discipline as
	// SegBench.
	struct BiasBench
	{
		Allocator alloc = DefaultAllocator();

		static double Median(std::vector<double>& v)
		{
			std::sort(v.begin(), v.end());
			return v[v.size() / 2];
		}

		double MedianQuery(const BankView& view, const float* q, const QueryParams& p,
			Workspace& ws, Hit* hits)
		{
			int32_t n = 0;
			std::vector<double> samples;
			Query(view, q, p, ws, hits, &n); // warm
			for (int32_t rep = 0; rep < 9; ++rep)
			{
				const double t0 = Now();
				for (int32_t r = 0; r < 3; ++r)
				{
					Query(view, q, p, ws, hits, &n);
				}
				samples.push_back((Now() - t0) / 3);
			}
			return Median(samples);
		}

		void Run(int32_t count, int32_t dims, Quantization quant)
		{
			Rng rng;
			std::vector<float> src(static_cast<size_t>(count) * dims);
			for (auto& v : src)
			{
				v = rng.NextFloat();
			}
			const int32_t pd = PaddedDims(dims, quant);
			const int64_t payloadBytes =
				static_cast<int64_t>(count) * pd * ElementSize(quant);
			void* payload =
				alloc.alloc(static_cast<size_t>(payloadBytes), kAlignment, alloc.user);
			std::vector<float> scales(static_cast<size_t>(count));
			BankView view;
			view.rows = payload;
			view.count = count;
			view.dims = dims;
			view.paddedDims = pd;
			view.quant = quant;
			view.metric = Metric::Dot;
			if (quant == Quantization::Float32)
			{
				PadRowsFloat32(src.data(), count, dims, pd, static_cast<float*>(payload));
			}
			else
			{
				QuantizeRowsInt8(src.data(), count, dims, pd,
					static_cast<int8_t*>(payload), scales.data());
				view.scales = scales.data();
			}

			void* qmem = alloc.alloc(static_cast<size_t>(pd) * sizeof(float),
				kAlignment, alloc.user);
			float* q = static_cast<float*>(qmem);
			for (int32_t j = 0; j < pd; ++j)
			{
				q[j] = j < dims ? rng.NextFloat() : 0.0f;
			}

			std::vector<float> dense(static_cast<size_t>(count));
			for (auto& b : dense)
			{
				b = rng.NextFloat() * 0.25f;
			}
			BiasPair pair[1] = {{count / 2, 10.0f}};

			Workspace ws;
			std::vector<Hit> hits(16);
			QueryParams p;
			p.k = 10;
			const double v1 = MedianQuery(view, q, p, ws, hits.data());
			RowBias rbDense;
			rbDense.dense = dense.data();
			p.bias = &rbDense;
			const double dbias = MedianQuery(view, q, p, ws, hits.data());
			RowBias rbSparse;
			rbSparse.pairs = pair;
			rbSparse.pairCount = 1;
			p.bias = &rbSparse;
			const double sbias = MedianQuery(view, q, p, ws, hits.data());

			// Batch: 64 queries, per-query bias (dense-shared / one sparse pair each).
			const int32_t batch = 64;
			void* qsmem = alloc.alloc(static_cast<size_t>(batch) * pd * sizeof(float),
				kAlignment, alloc.user);
			float* qs = static_cast<float*>(qsmem);
			for (int32_t m = 0; m < batch; ++m)
			{
				for (int32_t j = 0; j < pd; ++j)
				{
					qs[static_cast<int64_t>(m) * pd + j] = j < dims ? rng.NextFloat() : 0.0f;
				}
			}
			std::vector<Hit> bh(static_cast<size_t>(batch) * 10);
			std::vector<int32_t> counts(batch);
			QueryParams bp;
			bp.k = 10;
			auto medianBatch = [&](const QueryParams& qp) {
				std::vector<double> samples;
				QueryBatch(view, qs, batch, qp, ws, bh.data(), counts.data());
				for (int32_t rep = 0; rep < 7; ++rep)
				{
					const double t0 = Now();
					QueryBatch(view, qs, batch, qp, ws, bh.data(), counts.data());
					samples.push_back(Now() - t0);
				}
				return Median(samples);
			};
			const double vb = medianBatch(bp);
			std::vector<RowBias> denseForms(batch);
			for (auto& f : denseForms)
			{
				f.dense = dense.data();
			}
			bp.bias = denseForms.data();
			const double db = medianBatch(bp);
			std::vector<BiasPair> pairsPerQ(batch);
			std::vector<RowBias> sparseForms(batch);
			for (int32_t m = 0; m < batch; ++m)
			{
				pairsPerQ[m] = {m * (count / batch), 10.0f};
				sparseForms[m].pairs = &pairsPerQ[m];
				sparseForms[m].pairCount = 1;
			}
			bp.bias = sparseForms.data();
			const double sb = medianBatch(bp);

			std::printf(
				"BiasBench %6d x %3d %-7s | v1 %7.3f ms | dense %7.3f (%+5.1f%%) | sparse %7.3f (%+5.1f%%) | batch v1 %8.3f dense %8.3f (%+5.1f%%) sparse %8.3f (%+5.1f%%)\n",
				count, dims, quant == Quantization::Float32 ? "float32" : "int8",
				v1 * 1e3, dbias * 1e3, (dbias / v1 - 1.0) * 100.0,
				sbias * 1e3, (sbias / v1 - 1.0) * 100.0,
				vb * 1e3, db * 1e3, (db / vb - 1.0) * 100.0,
				sb * 1e3, (sb / vb - 1.0) * 100.0);

			alloc.free(qsmem, alloc.user);
			alloc.free(qmem, alloc.user);
			alloc.free(payload, alloc.user);
		}
	};
}

int main()
{
	std::printf("superfaiss bench (single thread, simd path: %s)\n",
		ActiveSimdPath() == SimdPath::Scalar ? "scalar"
			: ActiveSimdPath() == SimdPath::SSE ? "sse"
			: ActiveSimdPath() == SimdPath::AVX2 ? "avx2" : "neon");

	Bench bench;
	// Typical game bank.
	bench.Run(10000, 128, Quantization::Int8, 64);
	// Reference workload (plan §13).
	bench.Run(100000, 256, Quantization::Int8, 64);
	bench.Run(100000, 256, Quantization::Float32, 64);
	// Scaling guard points (E3).
	bench.Run(200000, 256, Quantization::Int8, 64);

	// V2 slot-1 calibration (segmented scan; plan section 10 predictions).
	SegBench segBench;
	segBench.Run(100000, 256, Quantization::Float32);
	segBench.Run(100000, 256, Quantization::Int8);
	segBench.BreakEven(100000, 256);

	// V2.1 calibration (per-row bias; plan section 18.3 predictions).
	BiasBench biasBench;
	biasBench.Run(100000, 256, Quantization::Float32);
	biasBench.Run(100000, 256, Quantization::Int8);
	return 0;
}

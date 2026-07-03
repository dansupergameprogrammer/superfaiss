// SuperFAISS single-thread benchmark. Produces the calibration numbers the plugin's
// performance ceilings are pinned from (plan §13 / S-V2); the multi-threaded numbers
// come from the engine-side harness, since threading is the host's job.

#include "superfaiss/superfaiss.h"

#include <chrono>
#include <cstdio>
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
	return 0;
}

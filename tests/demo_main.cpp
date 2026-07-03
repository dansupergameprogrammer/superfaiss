// SuperFAISS terminal demo: semantic nearest-neighbor search over GloVe word vectors.
// Loads a GloVe .txt (word v1 v2 ... vD per line), bakes a Cosine int8 bank in-process
// with the library's own bake functions, then answers queries.
//
//   superfaiss_demo <glove.txt> <dims> <maxWords> [words...]
//
// With trailing words: answers each and exits. Without: interactive stdin loop.

#include "superfaiss/superfaiss.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using namespace superfaiss;

namespace
{
	double NowMs()
	{
		return std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	struct Demo
	{
		Allocator alloc = DefaultAllocator();
		std::vector<std::string> words;
		std::unordered_map<std::string, int32_t> index;
		std::vector<float> normalized; // count x dims, L2-normalized
		void* payload = nullptr;
		std::vector<float> scales;
		BankView bank;
		Workspace ws;

		~Demo() { alloc.free(payload, alloc.user); }

		bool Load(const char* path, int32_t dims, int32_t maxWords)
		{
			const double t0 = NowMs();
			FILE* f = std::fopen(path, "rb");
			if (f == nullptr)
			{
				std::printf("cannot open %s\n", path);
				return false;
			}

			std::vector<float> rows;
			rows.reserve(static_cast<size_t>(maxWords) * dims);
			char line[8192];
			while (static_cast<int32_t>(words.size()) < maxWords && std::fgets(line, sizeof(line), f))
			{
				char* p = std::strchr(line, ' ');
				if (p == nullptr)
				{
					continue;
				}
				*p = '\0';
				bool ok = true;
				const size_t base = rows.size();
				rows.resize(base + dims);
				char* cursor = p + 1;
				for (int32_t d = 0; d < dims; ++d)
				{
					char* end = nullptr;
					rows[base + d] = std::strtof(cursor, &end);
					if (end == cursor)
					{
						ok = false;
						break;
					}
					cursor = end;
				}
				if (!ok)
				{
					rows.resize(base);
					continue;
				}
				index.emplace(line, static_cast<int32_t>(words.size()));
				words.emplace_back(line);
			}
			std::fclose(f);

			const int32_t count = static_cast<int32_t>(words.size());
			if (count == 0)
			{
				std::printf("no vectors parsed\n");
				return false;
			}

			int32_t badRow = -1;
			if (ValidateSourceRows(rows.data(), count, dims, &badRow) != Status::Ok ||
				NormalizeRows(rows.data(), count, dims, &badRow) != Status::Ok)
			{
				std::printf("bad source row %d\n", badRow);
				return false;
			}
			normalized = rows;

			const int32_t pd = PaddedDims(dims, Quantization::Int8);
			const size_t bytes = static_cast<size_t>(count) * pd;
			payload = alloc.alloc(bytes, kAlignment, alloc.user);
			scales.resize(static_cast<size_t>(count));
			QuantizeRowsInt8(rows.data(), count, dims, pd, static_cast<int8_t*>(payload), scales.data());

			bank.rows = payload;
			bank.scales = scales.data();
			bank.count = count;
			bank.dims = dims;
			bank.paddedDims = pd;
			bank.quant = Quantization::Int8;
			bank.metric = Metric::Cosine;
			if (ValidateBank(bank) != Status::Ok)
			{
				std::printf("bank failed validation\n");
				return false;
			}

			std::printf(
				"bank: %d words x %d dims, int8 cosine, %.1f MB (float32 source was %.1f MB), "
				"baked in %.0f ms, simd path: %s\n\n",
				count, dims, bytes / (1024.0 * 1024.0),
				(static_cast<size_t>(count) * dims * 4) / (1024.0 * 1024.0),
				NowMs() - t0,
				ActiveSimdPath() == SimdPath::Scalar ? "scalar"
					: ActiveSimdPath() == SimdPath::SSE ? "sse"
					: ActiveSimdPath() == SimdPath::AVX2 ? "avx2" : "neon");
			return true;
		}

		void Ask(const std::string& word)
		{
			const auto found = index.find(word);
			if (found == index.end())
			{
				std::printf("  '%s' is not in the bank\n\n", word.c_str());
				return;
			}
			const int32_t self = found->second;

			// The query is the word's own normalized vector, padded and aligned; the
			// word itself is excluded via the filter — "nearest neighbors other than me".
			const int32_t pd = bank.paddedDims;
			std::vector<float> queryStorage(static_cast<size_t>(pd) + 4, 0.0f);
			float* query = queryStorage.data();
			while ((reinterpret_cast<uintptr_t>(query) % kAlignment) != 0)
			{
				++query;
			}
			std::memcpy(query, normalized.data() + static_cast<size_t>(self) * bank.dims,
				static_cast<size_t>(bank.dims) * sizeof(float));

			std::vector<uint32_t> exclude((bank.count + 31) / 32, 0);
			exclude[self >> 5] |= 1u << (self & 31);

			Hit hits[10];
			int32_t hitCount = 0;
			QueryParams params;
			params.k = 10;
			params.excludeBits = exclude.data();

			const double t0 = NowMs();
			const Status s = Query(bank, query, params, ws, hits, &hitCount);
			const double elapsed = NowMs() - t0;

			if (s != Status::Ok)
			{
				std::printf("  query failed (%d)\n\n", static_cast<int>(s));
				return;
			}
			std::printf("  %s -> (%.3f ms over %d words)\n", word.c_str(), elapsed, bank.count);
			for (int32_t i = 0; i < hitCount; ++i)
			{
				std::printf("    %2d. %-20s %.4f\n", i + 1,
					words[static_cast<size_t>(hits[i].index)].c_str(), hits[i].score);
			}
			std::printf("\n");
		}
	};
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		std::printf("usage: superfaiss_demo <glove.txt> <dims> <maxWords> [words...]\n");
		return 1;
	}

	Demo demo;
	if (!demo.Load(argv[1], std::atoi(argv[2]), std::atoi(argv[3])))
	{
		return 1;
	}

	if (argc > 4)
	{
		for (int i = 4; i < argc; ++i)
		{
			demo.Ask(argv[i]);
		}
		return 0;
	}

	std::printf("type a word (blank line to quit):\n");
	char line[256];
	while (std::fgets(line, sizeof(line), stdin))
	{
		std::string word(line);
		while (!word.empty() && (word.back() == '\n' || word.back() == '\r' || word.back() == ' '))
		{
			word.pop_back();
		}
		if (word.empty())
		{
			break;
		}
		demo.Ask(word);
	}
	return 0;
}

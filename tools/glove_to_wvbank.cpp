// Converts GloVe-format text ("word v1 v2 ... vD" per line) into the .wvbank sidecar
// pair. Standard library only.
//
//   glove_to_wvbank <glove.txt> <dims> <maxWords> <outName>
//
// Writes <outName>.wvbank.json and <outName>.wvbank.bin.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
	if (argc != 5)
	{
		std::printf("usage: glove_to_wvbank <glove.txt> <dims> <maxWords> <outName>\n");
		return 1;
	}
	const char* srcPath = argv[1];
	const int dims = std::atoi(argv[2]);
	const int maxWords = std::atoi(argv[3]);
	const std::string outName = argv[4];

	FILE* src = std::fopen(srcPath, "rb");
	if (src == nullptr)
	{
		std::printf("cannot open %s\n", srcPath);
		return 1;
	}

	FILE* bin = std::fopen((outName + ".wvbank.bin").c_str(), "wb");
	if (bin == nullptr)
	{
		std::printf("cannot write %s.wvbank.bin\n", outName.c_str());
		std::fclose(src);
		return 1;
	}

	std::vector<std::string> words;
	std::vector<float> row(static_cast<size_t>(dims));
	char line[8192];
	while (static_cast<int>(words.size()) < maxWords && std::fgets(line, sizeof(line), src))
	{
		char* p = std::strchr(line, ' ');
		if (p == nullptr)
		{
			continue;
		}
		*p = '\0';
		char* cursor = p + 1;
		bool ok = true;
		for (int d = 0; d < dims; ++d)
		{
			char* end = nullptr;
			row[static_cast<size_t>(d)] = std::strtof(cursor, &end);
			if (end == cursor)
			{
				ok = false;
				break;
			}
			cursor = end;
		}
		// Skip malformed lines and words that are not clean JSON string material.
		if (!ok || std::strchr(line, '"') != nullptr || std::strchr(line, '\\') != nullptr)
		{
			continue;
		}
		std::fwrite(row.data(), sizeof(float), static_cast<size_t>(dims), bin);
		words.emplace_back(line);
	}
	std::fclose(src);
	std::fclose(bin);

	FILE* json = std::fopen((outName + ".wvbank.json").c_str(), "wb");
	if (json == nullptr)
	{
		std::printf("cannot write %s.wvbank.json\n", outName.c_str());
		return 1;
	}
	std::fprintf(json,
		"{\n \"schemaVersion\": 1,\n \"dims\": %d,\n \"count\": %d,\n"
		" \"metric\": \"cosine\",\n \"dtype\": \"float32\",\n"
		" \"description\": \"GloVe 6B %dd subset (PDDL); WizardSimilarity demo bank\",\n"
		" \"ids\": [",
		dims, static_cast<int>(words.size()), dims);
	for (size_t i = 0; i < words.size(); ++i)
	{
		std::fprintf(json, "%s\"%s\"", i ? ", " : "", words[i].c_str());
	}
	std::fprintf(json, "]\n}\n");
	std::fclose(json);

	std::printf("%s: %d words x %d dims\n", outName.c_str(), static_cast<int>(words.size()), dims);
	return 0;
}

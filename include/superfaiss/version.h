#pragma once

#include <cstdint>

#define SUPERFAISS_VERSION_MAJOR 2
#define SUPERFAISS_VERSION_MINOR 2
#define SUPERFAISS_VERSION_PATCH 0

namespace superfaiss
{
	// Highest interchange/bank schema version this library understands: 1 =
	// channel-less, 2 = named channels. Readers accept [1, kSchemaVersion] and
	// hard-reject anything newer, never guessing; writers emit 1 unless channels
	// are present (V2 plan section 8).
	inline constexpr int32_t kSchemaVersion = 2;
}

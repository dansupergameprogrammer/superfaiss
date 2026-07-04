#pragma once

#include <cstdint>

#define SUPERFAISS_VERSION_MAJOR 1
#define SUPERFAISS_VERSION_MINOR 1
#define SUPERFAISS_VERSION_PATCH 0

namespace superfaiss
{
	// Version of the baked bank memory/interchange layout. A consumer must hard-reject a
	// payload whose schema version it does not know.
	inline constexpr int32_t kSchemaVersion = 1;
}

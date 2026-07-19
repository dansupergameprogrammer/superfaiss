// SuperFAISS V3.2 — Bank Inspector I, module M1 (graph.h).
//
// RED SCAFFOLD (Curie, 2026-07-18): these bodies are UNIMPLEMENTED. Each returns
// Status::Ok and writes NO output — the proven V3.1 red-suite pattern (the no-op stub):
// every M1 cell in tests/test_main.cpp fails for its cell's reason against this stub —
// a FEAT/positive cell sees Ok with poison-initialized output where it expects the
// constructed component structure; a dim-2 rejection cell sees Ok where it expects
// InvalidArgument. Brunel replaces these bodies with the real implementation (slot 1);
// the contract is graph.h.

#include "superfaiss/graph.h"

namespace superfaiss
{

Status BuildKnnNeighbors(const BankView&, int32_t, bool, int32_t*, Workspace&)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

Status MutualFilter(int32_t, int32_t, const int32_t*, uint8_t*)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

Status BuildDuplicateGroups(const BankView&, int32_t*, int32_t*)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

Status ConnectedComponents(
	int32_t, int32_t, const int32_t*, const uint8_t*, const int32_t*, int32_t*, int32_t*)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

} // namespace superfaiss

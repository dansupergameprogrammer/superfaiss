// SuperFAISS V3.2 — Bank Inspector I, module M3 (matching.h).
//
// RED SCAFFOLD (Curie, 2026-07-19): MutualNearestMatches is UNIMPLEMENTED. Returns
// Status::Ok and writes NO output — the proven M1/M2 red-suite pattern (the no-op stub):
// every cell in tests/test_main.cpp fails for its cell's reason against this stub — a
// FEAT/oracle cell reads its poison-initialized output where it expects the real match
// pairs; a dim-2 rejection cell sees Ok where it expects InvalidArgument. Brunel replaces
// this body with the real implementation; the contract is matching.h.

#include "superfaiss/matching.h"

namespace superfaiss
{

Status MutualNearestMatches(
	const BankView&, const int32_t*, const BankView&, const uint32_t*,
	const BankView&, const uint32_t*, int32_t, MatchPair*, int32_t*, Workspace&)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

} // namespace superfaiss

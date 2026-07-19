// SuperFAISS V3.2 — Bank Inspector I, module M2 (novelty.h).
//
// RED SCAFFOLD (Curie, 2026-07-18): NoveltyScore is UNIMPLEMENTED — returns Status::Ok and
// writes nothing (the V3.1 no-op-stub pattern), so its cells in tests/test_main.cpp fail
// for their reason: a correctness cell reads its poison-initialized output where it expects
// the empirical-CDF rank; a dim-2 rejection cell sees Ok where it expects InvalidArgument.
// Brunel replaces this body with the implementation. The rest of M2 (probe, baseline,
// verdict) lands after the F-M2-1 contract decision.

#include "superfaiss/novelty.h"

namespace superfaiss
{

Status NoveltyScore(const float*, int32_t, float, float*)
{
	return Status::Ok; // RED SCAFFOLD — writes nothing.
}

} // namespace superfaiss

#pragma once

// SuperFAISS — fast, deterministic, allocation-free k-NN over baked embedding banks.
// MIT licensed. Independent implementation; not affiliated with Meta's FAISS.

#include "version.h"
#include "types.h"
#include "alloc.h"
#include "topk.h"
#include "kernels.h"
#include "validate.h"
#include "bake.h"
#include "query.h"
#include "compose.h"
#include "analytics.h"
#include "pca.h"
#include "scratch.h"

// V3.2 Bank Inspector I, Tier 1 (plan section 25.4).
#include "inspector_common.h"
#include "graph.h"
#include "novelty.h"
#include "matching.h"

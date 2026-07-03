#pragma once

#include "types.h"

namespace superfaiss
{

// Structural validation of a bank view: alignment, stride, enum ranges, scales
// presence. Cheap; intended to run once at bank load, not per query.
Status ValidateBank(const BankView& bank);

// Query validation: pointer alignment, finiteness, zero-filled pad lanes, and the
// cosine zero-norm rule. NaN/Inf never propagates past this gate.
Status ValidateQuery(const BankView& bank, const float* paddedQuery);

} // namespace superfaiss

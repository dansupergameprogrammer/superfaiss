#include "superfaiss/analytics.h"

#include "superfaiss/compose.h" // MakeCentroidCrossDevice
#include "superfaiss/query.h"   // QueryXdBatch

#include <cmath>
#include <cstdint>

namespace superfaiss
{

namespace
{

// The v2.2 cross-device epilogue, transcribed from src/kernels.cpp (XdFloorToFloat /
// XdDotScoreD / XdL2ScoreD, ~lines 1485-1515) so the pair score is the SAME fixed-order
// double arithmetic as the shipped scan. Kept file-local per the shipped
// file-local-epilogue convention; bit-equivalence is pinned by the REF test (T-V2.5-1) and
// the cross-path GOLD battery (T-V2.5-2). The cosine limb (XdCosineDistance) is new (plan
// S1, section 22.5 item 2b) - the one runtime sqrt, absent from the shipped epilogue.

// |score| < FLT_MIN -> exactly 0.0f, on every machine (the subnormal-floor contract).
inline float XdFloor(double score)
{
	const double lim = 1.1754943508222875e-38; // FLT_MIN, exactly
	if (score < lim && score > -lim)
	{
		return 0.0f;
	}
	return static_cast<float>(score);
}

// Dot-family pair epilogue: one fixed-order double expression.
inline double XdDot(int32_t acc, double aScaleD, double bScaleD)
{
	return static_cast<double>(acc) * (aScaleD * bScaleD);
}

// L2 pair epilogue: ||aScale*a8 - bScale*b8||^2 in the shipped expanded form.
inline double XdL2(int64_t cross, int64_t aSq, int64_t bSq, double aScaleD, double bScaleD)
{
	const double a = (aScaleD * aScaleD) * static_cast<double>(aSq);
	const double b = (bScaleD * bScaleD) * static_cast<double>(bSq);
	const double c = ((aScaleD * bScaleD) * static_cast<double>(cross)) * 2.0;
	return (a + b) - c;
}

// Cosine distance limb (plan S1): the per-query scales cancel, so
// cos = crossDot / sqrt(aSq * bSq); distance = 1 - cos. The sqrt is the one operation
// outside the shipped {+,-,*,/} epilogue - cross-device-exact by IEEE-754 correctly-rounded
// sqrt under the project's true-sqrt / no-fast-math build regime (/fp:precise,
// -ffp-contract=off). Callers guarantee aSq, bSq > 0 (ZeroNormQuery upstream).
inline double XdCosineDistance(int32_t cross, int64_t aSq, int64_t bSq)
{
	const double denom = std::sqrt(static_cast<double>(aSq) * static_cast<double>(bSq));
	return 1.0 - (static_cast<double>(cross) / denom);
}

inline bool IsInt8CrossDevice(const BankView& bank)
{
	return bank.quant == Quantization::Int8 && bank.paddedDims > 0 &&
		bank.paddedDims <= kMaxCrossDeviceDims;
}

// True if any image element is INT8_MIN (-128). The ±127 premise the int32 cross-dot
// bound rests on (plan W1) admits only [-127, 127]; a -128 element at paddedDims near the
// ceiling overflows the int32 accumulator (D-V2-13). Checked at the public boundary before
// any accumulation - the self-dot recompute below would itself overflow on such a payload.
inline bool HasMinInt8(const int8_t* image, int32_t n)
{
	for (int32_t i = 0; i < n; ++i)
	{
		if (image[i] == -128)
		{
			return true;
		}
	}
	return false;
}

// Distance-sense pair score over trusted (unvalidated) payloads. `metric` picks the family.
inline float XdPairScore(const XdQuery& a, const XdQuery& b, int32_t paddedDims, Metric metric)
{
	const int32_t cross = detail::DotI8I8(a.q8, b.q8, paddedDims);
	if (metric == Metric::L2)
	{
		return XdFloor(XdL2(cross, a.sqSum, b.sqSum, a.scale, b.scale));
	}
	if (metric == Metric::Cosine)
	{
		return XdFloor(XdCosineDistance(cross, a.sqSum, b.sqSum));
	}
	return XdFloor(XdDot(cross, a.scale, b.scale));
}

// Per-element hit score -> distance sense in `metric` (see MeanNN/MaxNN doc).
inline double HitDistance(float score, Metric metric)
{
	if (metric == Metric::Cosine)
	{
		return 1.0 - static_cast<double>(score);
	}
	// L2: squared distance already; Dot: similarity (documented).
	return static_cast<double>(score);
}

} // namespace

Status ScoreXdPair(const XdQuery& a, const XdQuery& b, int32_t paddedDims, Metric metric,
	float* outScore)
{
	if (outScore == nullptr || a.q8 == nullptr || b.q8 == nullptr || paddedDims <= 0 ||
		paddedDims > kMaxCrossDeviceDims)
	{
		return Status::InvalidArgument;
	}
	// Payload law, both members (the shipped QueryXd boundary check): scale finite and
	// non-negative, self-dot recomputed from the image and matched.
	if (!std::isfinite(a.scale) || a.scale < 0.0 || !std::isfinite(b.scale) || b.scale < 0.0)
	{
		return Status::InvalidArgument;
	}
	// -128 guard (D-V2-13): enforce the +-127 premise before any int32 accumulation.
	if (HasMinInt8(a.q8, paddedDims) || HasMinInt8(b.q8, paddedDims))
	{
		return Status::InvalidArgument;
	}
	if (detail::DotI8I8(a.q8, a.q8, paddedDims) != a.sqSum ||
		detail::DotI8I8(b.q8, b.q8, paddedDims) != b.sqSum)
	{
		return Status::InvalidArgument;
	}
	if (metric == Metric::Cosine && (a.sqSum == 0 || b.sqSum == 0))
	{
		return Status::ZeroNormQuery;
	}
	*outScore = XdPairScore(a, b, paddedDims, metric);
	return Status::Ok;
}

Status CentroidDistanceCrossDevice(
	const BankView& bankA, const int32_t* rowIndicesA, int32_t rowCountA,
	const int32_t* weightsA, const uint32_t* excludeBitsA,
	const BankView& bankB, const int32_t* rowIndicesB, int32_t rowCountB,
	const int32_t* weightsB, const uint32_t* excludeBitsB,
	Metric metric, int8_t* centroidScratchA, int8_t* centroidScratchB, float* outDistance)
{
	if (outDistance == nullptr || centroidScratchA == nullptr || centroidScratchB == nullptr ||
		!IsInt8CrossDevice(bankA) || !IsInt8CrossDevice(bankB) ||
		bankA.paddedDims != bankB.paddedDims)
	{
		return Status::InvalidArgument;
	}
	double scaleA = 0.0;
	int64_t sqA = 0;
	Status s = MakeCentroidCrossDevice(bankA, rowIndicesA, rowCountA, weightsA, excludeBitsA,
		centroidScratchA, &scaleA, &sqA);
	if (s != Status::Ok)
	{
		return s;
	}
	double scaleB = 0.0;
	int64_t sqB = 0;
	s = MakeCentroidCrossDevice(bankB, rowIndicesB, rowCountB, weightsB, excludeBitsB,
		centroidScratchB, &scaleB, &sqB);
	if (s != Status::Ok)
	{
		return s;
	}
	if (metric == Metric::Cosine && (sqA == 0 || sqB == 0))
	{
		return Status::ZeroNormQuery;
	}
	const XdQuery a{centroidScratchA, scaleA, sqA};
	const XdQuery b{centroidScratchB, scaleB, sqB};
	*outDistance = XdPairScore(a, b, bankA.paddedDims, metric);
	return Status::Ok;
}

namespace
{

// Shared NN reduction: lift each non-excluded source row to a trusted XdQuery, batch-score
// k=1 against the target, reduce the per-source-row nearest distances by mean or max in
// ascending source-row order (plan N2). Scores in the target bank's own metric.
Status NNDivergence(
	const BankView& source, const uint32_t* sourceExcludeBits,
	const BankView& target, const uint32_t* targetExcludeBits,
	Reduce reduce, XdQuery* queryScratch, Hit* hitScratch, int32_t* countScratch,
	Workspace& ws, float* outValue)
{
	if (outValue == nullptr || queryScratch == nullptr || hitScratch == nullptr ||
		countScratch == nullptr || !IsInt8CrossDevice(source) || !IsInt8CrossDevice(target) ||
		source.paddedDims != target.paddedDims)
	{
		return Status::InvalidArgument;
	}
	// The target must have at least one non-excluded row, or no nearest neighbour exists.
	bool targetHasRow = false;
	for (int32_t r = 0; r < target.count; ++r)
	{
		if (!IsExcluded(targetExcludeBits, r))
		{
			targetHasRow = true;
			break;
		}
	}
	if (!targetHasRow)
	{
		return Status::InvalidArgument;
	}

	const int8_t* srcRows = static_cast<const int8_t*>(source.rows);
	const int32_t pd = source.paddedDims;
	int32_t m = 0; // non-excluded source rows, lifted in ascending index order
	for (int32_t r = 0; r < source.count; ++r)
	{
		if (IsExcluded(sourceExcludeBits, r))
		{
			continue;
		}
		const int8_t* img = srcRows + static_cast<int64_t>(r) * pd;
		XdQuery& q = queryScratch[m];
		q.q8 = img;
		q.scale = detail::FloatBitsToDouble(source.scales[r]);
		q.sqSum = detail::DotI8I8(img, img, pd); // trusted-internal lift (plan W2)
		++m;
	}
	if (m == 0)
	{
		return Status::InvalidArgument; // empty source
	}

	if (!ws.Reserve(1, m))
	{
		return Status::OutOfMemory;
	}
	QueryParams params;
	params.k = 1;
	params.excludeBits = targetExcludeBits;
	params.exactness = Exactness::CrossDevice;
	const Status s = QueryXdBatch(target, queryScratch, m, params, ws, hitScratch, countScratch);
	if (s != Status::Ok)
	{
		return s;
	}

	// F4 count guard: k=1 + a verified non-empty target gives exactly one hit per query, so
	// under contract every countScratch[i] == 1 and `counted` == m (green path unchanged). The
	// guard hardens against a future k/exclusion change that could leave a query with no hit -
	// a default score=0 must not enter the reduction silently.
	const Metric metric = target.metric;
	if (reduce == Reduce::Mean)
	{
		double acc = 0.0;
		int32_t counted = 0;
		for (int32_t i = 0; i < m; ++i)
		{
			if (countScratch[i] < 1)
			{
				continue;
			}
			acc += HitDistance(hitScratch[i].score, metric); // fixed order (plan N2)
			++counted;
		}
		if (counted == 0)
		{
			return Status::InvalidArgument;
		}
		*outValue = XdFloor(acc / static_cast<double>(counted));
	}
	else
	{
		double best = 0.0;
		bool have = false;
		for (int32_t i = 0; i < m; ++i)
		{
			if (countScratch[i] < 1)
			{
				continue;
			}
			const double d = HitDistance(hitScratch[i].score, metric);
			if (!have || d > best)
			{
				best = d; // order-free max
			}
			have = true;
		}
		if (!have)
		{
			return Status::InvalidArgument;
		}
		*outValue = XdFloor(best);
	}
	return Status::Ok;
}

} // namespace

Status MeanNNCrossDevice(
	const BankView& source, const uint32_t* sourceExcludeBits,
	const BankView& target, const uint32_t* targetExcludeBits,
	XdQuery* queryScratch, Hit* hitScratch, int32_t* countScratch, Workspace& ws,
	float* outValue)
{
	return NNDivergence(source, sourceExcludeBits, target, targetExcludeBits, Reduce::Mean,
		queryScratch, hitScratch, countScratch, ws, outValue);
}

Status MaxNNCrossDevice(
	const BankView& source, const uint32_t* sourceExcludeBits,
	const BankView& target, const uint32_t* targetExcludeBits,
	XdQuery* queryScratch, Hit* hitScratch, int32_t* countScratch, Workspace& ws,
	float* outValue)
{
	return NNDivergence(source, sourceExcludeBits, target, targetExcludeBits, Reduce::Max,
		queryScratch, hitScratch, countScratch, ws, outValue);
}

Status SpreadCrossDevice(
	const BankView& bank, const int32_t* rowIndices, int32_t rowCount,
	const uint32_t* excludeBits, Reduce reduce, int8_t* centroidScratch, float* outValue)
{
	if (outValue == nullptr || centroidScratch == nullptr || rowIndices == nullptr ||
		rowCount <= 0 || !IsInt8CrossDevice(bank))
	{
		return Status::InvalidArgument;
	}
	double cScale = 0.0;
	int64_t cSq = 0;
	const Status s = MakeCentroidCrossDevice(bank, rowIndices, rowCount, nullptr, excludeBits,
		centroidScratch, &cScale, &cSq);
	if (s != Status::Ok)
	{
		return s;
	}
	const Metric metric = bank.metric;
	if (metric == Metric::Cosine && cSq == 0)
	{
		return Status::ZeroNormQuery;
	}
	const XdQuery centroid{centroidScratch, cScale, cSq};

	const int8_t* rows = static_cast<const int8_t*>(bank.rows);
	const int32_t pd = bank.paddedDims;
	bool haveFirst = false;
	double acc = 0.0;
	double best = 0.0;
	int32_t counted = 0;
	// Reduce over the selection in the given (ascending) order (plan N2). Excluded rows
	// are dropped by the pooling; a tombstoned row is skipped here too so the dispersion
	// is measured over the same rows the centroid pooled.
	for (int32_t i = 0; i < rowCount; ++i)
	{
		const int32_t r = rowIndices[i];
		if (r < 0 || r >= bank.count || IsExcluded(excludeBits, r))
		{
			continue;
		}
		const int8_t* img = rows + static_cast<int64_t>(r) * pd;
		const XdQuery rowQ{img, detail::FloatBitsToDouble(bank.scales[r]),
			detail::DotI8I8(img, img, pd)};
		const double d = static_cast<double>(XdPairScore(rowQ, centroid, pd, metric));
		if (reduce == Reduce::Mean)
		{
			acc += d;
		}
		else if (!haveFirst || d > best)
		{
			best = d;
		}
		haveFirst = true;
		++counted;
	}
	if (counted == 0)
	{
		return Status::InvalidArgument;
	}
	*outValue = reduce == Reduce::Mean
		? XdFloor(acc / static_cast<double>(counted))
		: XdFloor(best);
	return Status::Ok;
}

// --- V3.0 Tier 2: channel-scoped analytics (plan section 23.5) ---

namespace
{

// Resolves `channel` against the bank's channel table to its element sub-range. False when
// the bank carries no channel table or the selector is out of [0, channelCount) -- the
// InvalidArgument the section 23.8 rejection cells require.
inline bool ChannelSubRange(const BankView& bank, int32_t channel, int32_t* outOffset,
	int32_t* outLength)
{
	if (bank.channels == nullptr || bank.channelCount <= 0 || channel < 0 ||
		channel >= bank.channelCount)
	{
		return false;
	}
	*outOffset = bank.channels[channel].offset;
	*outLength = bank.channels[channel].length;
	return true;
}

// Pair score over a `length`-lane channel sub-range: the whole-vector cross-device epilogue
// (bit-identical to the shipped scan for Dot/L2), except a zero sub-norm member under Cosine
// scores a DEFINED 0 distance instead of the public boundary's ZeroNormQuery -- the plan's
// "all-zero channel scores 0, never NaN". Linear metrics never hit the guard, so the Dot/L2
// repack REF stays bit-exact.
inline float XdChannelPairScore(const XdQuery& a, const XdQuery& b, int32_t length,
	Metric metric)
{
	if (metric == Metric::Cosine && (a.sqSum == 0 || b.sqSum == 0))
	{
		return 0.0f;
	}
	return XdPairScore(a, b, length, metric);
}

// Pools a channel sub-range via MakeCentroidCrossDevice's sub-range path. A genuinely zero
// sub-vector (the pooler's ZeroNormQuery: an all-zero-norm channel, or antipodal cancel) is
// a defined zero centroid here (image zeroed, scale 0, self-dot 0) -- the whole-vector
// rejection is relaxed because Dot/L2 of a zero sub-vector is well defined and Cosine floors
// to 0 (XdChannelPairScore). Empty selections and bad arguments still propagate.
Status MakeChannelCentroid(const BankView& bank, const int32_t* rowIndices, int32_t rowCount,
	const int32_t* weights, const uint32_t* excludeBits, int32_t offset, int32_t length,
	int8_t* outQ8, double* outScale, int64_t* outSqSum)
{
	const Status s = MakeCentroidCrossDevice(bank, rowIndices, rowCount, weights, excludeBits,
		outQ8, outScale, outSqSum, offset, length);
	if (s == Status::ZeroNormQuery)
	{
		for (int32_t j = 0; j < length; ++j)
		{
			outQ8[j] = 0;
		}
		*outScale = 0.0;
		*outSqSum = 0;
		return Status::Ok;
	}
	return s;
}

// Directed nearest-neighbour channel divergence: for each non-excluded source sub-row, its
// nearest target sub-row distance (over the same channel sub-range), reduced by mean
// (ascending source order) or max (order-free). Mirrors NNDivergence's whole-vector
// semantics, but scores the sub-range in place (source/target rows keep their paddedDims
// stride, scored from `offset` over `length`) with the transcribed epilogue -- bit-equal to
// the whole-vector operator on a contiguous repack for Dot/L2. The nearest is selected in
// the target metric's better-direction (Dot: max similarity; L2/Cosine: min distance),
// matching the QueryXdBatch k=1 selection the repack REF runs.
Status NNDivergenceChannel(
	const BankView& source, const uint32_t* sourceExcludeBits,
	const BankView& target, const uint32_t* targetExcludeBits, int32_t channel, Reduce reduce,
	XdQuery* queryScratch, float* outValue)
{
	// Unlike the whole-vector NNDivergence (which routes through QueryXdBatch), the channel
	// path scores each nearest in place, so it needs only the source-lift buffer
	// (queryScratch) — the Hit/count/Workspace scratch the public signature carries for
	// parity with the whole-vector twin is unused here and not required non-null (Poirot #2).
	if (outValue == nullptr || queryScratch == nullptr ||
		!IsInt8CrossDevice(source) || !IsInt8CrossDevice(target) ||
		source.paddedDims != target.paddedDims)
	{
		return Status::InvalidArgument;
	}
	int32_t sOff = 0, sLen = 0, tOff = 0, tLen = 0;
	if (!ChannelSubRange(source, channel, &sOff, &sLen) ||
		!ChannelSubRange(target, channel, &tOff, &tLen) || sLen != tLen)
	{
		return Status::InvalidArgument;
	}
	const int32_t length = sLen;
	const Metric metric = target.metric;

	// The target must have at least one non-excluded row, or no nearest neighbour exists.
	bool targetHasRow = false;
	for (int32_t r = 0; r < target.count; ++r)
	{
		if (!IsExcluded(targetExcludeBits, r))
		{
			targetHasRow = true;
			break;
		}
	}
	if (!targetHasRow)
	{
		return Status::InvalidArgument;
	}

	const int8_t* srcRows = static_cast<const int8_t*>(source.rows);
	const int8_t* tgtRows = static_cast<const int8_t*>(target.rows);
	const int32_t spd = source.paddedDims;
	const int32_t tpd = target.paddedDims;

	// Lift non-excluded source sub-rows in ascending index order (trusted-internal, the
	// self-dot over the sub-range).
	int32_t m = 0;
	for (int32_t r = 0; r < source.count; ++r)
	{
		if (IsExcluded(sourceExcludeBits, r))
		{
			continue;
		}
		const int8_t* img = srcRows + static_cast<int64_t>(r) * spd + sOff;
		XdQuery& q = queryScratch[m];
		q.q8 = img;
		q.scale = detail::FloatBitsToDouble(source.scales[r]);
		q.sqSum = detail::DotI8I8(img, img, length);
		++m;
	}
	if (m == 0)
	{
		return Status::InvalidArgument; // empty source
	}
	// Cosine: a zero sub-norm source cannot be scored -- the bank's ZeroNormQuery law,
	// mirrored from QueryXdBatch's per-member rejection on a Cosine bank.
	if (metric == Metric::Cosine)
	{
		for (int32_t i = 0; i < m; ++i)
		{
			if (queryScratch[i].sqSum == 0)
			{
				return Status::ZeroNormQuery;
			}
		}
	}

	double acc = 0.0;
	double best = 0.0;
	bool have = false;
	int32_t counted = 0;
	for (int32_t i = 0; i < m; ++i)
	{
		const XdQuery& q = queryScratch[i];
		double nn = 0.0;
		bool haveNn = false;
		for (int32_t r = 0; r < target.count; ++r)
		{
			if (IsExcluded(targetExcludeBits, r))
			{
				continue;
			}
			const int8_t* timg = tgtRows + static_cast<int64_t>(r) * tpd + tOff;
			const XdQuery tq{timg, detail::FloatBitsToDouble(target.scales[r]),
				detail::DotI8I8(timg, timg, length)};
			if (metric == Metric::Cosine && tq.sqSum == 0)
			{
				continue; // zero sub-norm target: distance undefined, skip (never NaN)
			}
			const double d = static_cast<double>(XdChannelPairScore(q, tq, length, metric));
			const bool better = !haveNn || (metric == Metric::Dot ? d > nn : d < nn);
			if (better)
			{
				nn = d;
			}
			haveNn = true;
		}
		if (!haveNn)
		{
			continue; // no scorable target
		}
		if (reduce == Reduce::Mean)
		{
			acc += nn;
		}
		else if (!have || nn > best)
		{
			best = nn;
		}
		have = true;
		++counted;
	}
	if (counted == 0)
	{
		return Status::InvalidArgument;
	}
	*outValue = reduce == Reduce::Mean ? XdFloor(acc / static_cast<double>(counted))
	                                   : XdFloor(best);
	return Status::Ok;
}

} // namespace

Status CentroidDistanceCrossDeviceChannel(
	const BankView& bankA, const int32_t* rowIndicesA, int32_t rowCountA,
	const int32_t* weightsA, const uint32_t* excludeBitsA,
	const BankView& bankB, const int32_t* rowIndicesB, int32_t rowCountB,
	const int32_t* weightsB, const uint32_t* excludeBitsB,
	Metric metric, int32_t channel, int8_t* centroidScratchA, int8_t* centroidScratchB,
	float* outDistance)
{
	if (outDistance == nullptr || centroidScratchA == nullptr || centroidScratchB == nullptr ||
		!IsInt8CrossDevice(bankA) || !IsInt8CrossDevice(bankB) ||
		bankA.paddedDims != bankB.paddedDims)
	{
		return Status::InvalidArgument;
	}
	int32_t offA = 0, lenA = 0, offB = 0, lenB = 0;
	if (!ChannelSubRange(bankA, channel, &offA, &lenA) ||
		!ChannelSubRange(bankB, channel, &offB, &lenB) || lenA != lenB)
	{
		return Status::InvalidArgument;
	}
	const int32_t length = lenA;
	double scaleA = 0.0;
	int64_t sqA = 0;
	Status s = MakeChannelCentroid(bankA, rowIndicesA, rowCountA, weightsA, excludeBitsA,
		offA, length, centroidScratchA, &scaleA, &sqA);
	if (s != Status::Ok)
	{
		return s;
	}
	double scaleB = 0.0;
	int64_t sqB = 0;
	s = MakeChannelCentroid(bankB, rowIndicesB, rowCountB, weightsB, excludeBitsB,
		offB, length, centroidScratchB, &scaleB, &sqB);
	if (s != Status::Ok)
	{
		return s;
	}
	const XdQuery a{centroidScratchA, scaleA, sqA};
	const XdQuery b{centroidScratchB, scaleB, sqB};
	*outDistance = XdChannelPairScore(a, b, length, metric);
	return Status::Ok;
}

Status MeanNNCrossDeviceChannel(
	const BankView& source, const uint32_t* sourceExcludeBits,
	const BankView& target, const uint32_t* targetExcludeBits, int32_t channel,
	XdQuery* queryScratch, Hit*, int32_t*, Workspace&,
	float* outValue)
{
	return NNDivergenceChannel(source, sourceExcludeBits, target, targetExcludeBits, channel,
		Reduce::Mean, queryScratch, outValue);
}

Status MaxNNCrossDeviceChannel(
	const BankView& source, const uint32_t* sourceExcludeBits,
	const BankView& target, const uint32_t* targetExcludeBits, int32_t channel,
	XdQuery* queryScratch, Hit*, int32_t*, Workspace&,
	float* outValue)
{
	return NNDivergenceChannel(source, sourceExcludeBits, target, targetExcludeBits, channel,
		Reduce::Max, queryScratch, outValue);
}

Status SpreadCrossDeviceChannel(
	const BankView& bank, const int32_t* rowIndices, int32_t rowCount,
	const uint32_t* excludeBits, Reduce reduce, int32_t channel, int8_t* centroidScratch,
	float* outValue)
{
	if (outValue == nullptr || centroidScratch == nullptr || rowIndices == nullptr ||
		rowCount <= 0 || !IsInt8CrossDevice(bank))
	{
		return Status::InvalidArgument;
	}
	int32_t offset = 0, length = 0;
	if (!ChannelSubRange(bank, channel, &offset, &length))
	{
		return Status::InvalidArgument;
	}
	double cScale = 0.0;
	int64_t cSq = 0;
	const Status s = MakeChannelCentroid(bank, rowIndices, rowCount, nullptr, excludeBits,
		offset, length, centroidScratch, &cScale, &cSq);
	if (s != Status::Ok)
	{
		return s;
	}
	const Metric metric = bank.metric;
	const XdQuery centroid{centroidScratch, cScale, cSq};

	const int8_t* rows = static_cast<const int8_t*>(bank.rows);
	const int32_t pd = bank.paddedDims;
	bool haveFirst = false;
	double acc = 0.0;
	double best = 0.0;
	int32_t counted = 0;
	// Reduce over the selection in ascending order about the channel centroid, dropping the
	// same excluded/out-of-range rows the pooling dropped (matching SpreadCrossDevice).
	for (int32_t i = 0; i < rowCount; ++i)
	{
		const int32_t r = rowIndices[i];
		if (r < 0 || r >= bank.count || IsExcluded(excludeBits, r))
		{
			continue;
		}
		const int8_t* img = rows + static_cast<int64_t>(r) * pd + offset;
		const XdQuery rowQ{img, detail::FloatBitsToDouble(bank.scales[r]),
			detail::DotI8I8(img, img, length)};
		const double d = static_cast<double>(XdChannelPairScore(rowQ, centroid, length, metric));
		if (reduce == Reduce::Mean)
		{
			acc += d;
		}
		else if (!haveFirst || d > best)
		{
			best = d;
		}
		haveFirst = true;
		++counted;
	}
	if (counted == 0)
	{
		return Status::InvalidArgument;
	}
	*outValue = reduce == Reduce::Mean ? XdFloor(acc / static_cast<double>(counted))
	                                   : XdFloor(best);
	return Status::Ok;
}

Status ProjectionReport(const BankView& bank, const float* paddedDirection,
	const uint32_t* groupBits, float* outProjections, float* outSeparation)
{
	if (paddedDirection == nullptr || outProjections == nullptr || bank.count <= 0 ||
		bank.paddedDims <= 0)
	{
		return Status::InvalidArgument;
	}
	// Per-device float, offline (plan 22.3): no cross-device claim, so double accumulation
	// is used for accuracy and the result stored as float.
	double dirNormSq = 0.0;
	for (int32_t i = 0; i < bank.paddedDims; ++i)
	{
		const float d = paddedDirection[i];
		if (!std::isfinite(d))
		{
			return Status::InvalidArgument;
		}
		dirNormSq += static_cast<double>(d) * static_cast<double>(d);
	}
	if (dirNormSq == 0.0)
	{
		return Status::ZeroNormQuery;
	}
	// F3: reject an empty tag group BEFORE writing any projection, so a rejected call leaves
	// outProjections untouched (the no-partial-result-on-rejection convention).
	if (groupBits != nullptr && outSeparation != nullptr)
	{
		int32_t nA = 0;
		int32_t nB = 0;
		for (int32_t r = 0; r < bank.count; ++r)
		{
			if ((groupBits[static_cast<uint32_t>(r) >> 5] &
					(1u << (static_cast<uint32_t>(r) & 31u))) != 0u)
			{
				++nA;
			}
			else
			{
				++nB;
			}
		}
		if (nA == 0 || nB == 0)
		{
			return Status::InvalidArgument;
		}
	}

	const int32_t pd = bank.paddedDims;
	const bool isInt8 = bank.quant == Quantization::Int8;
	const int8_t* i8 = isInt8 ? static_cast<const int8_t*>(bank.rows) : nullptr;
	const float* f32 = isInt8 ? nullptr : static_cast<const float*>(bank.rows);
	for (int32_t r = 0; r < bank.count; ++r)
	{
		double acc = 0.0;
		if (isInt8)
		{
			const int8_t* row = i8 + static_cast<int64_t>(r) * pd;
			for (int32_t i = 0; i < pd; ++i)
			{
				acc += static_cast<double>(row[i]) * static_cast<double>(paddedDirection[i]);
			}
			acc *= static_cast<double>(bank.scales[r]);
		}
		else
		{
			const float* row = f32 + static_cast<int64_t>(r) * pd;
			for (int32_t i = 0; i < pd; ++i)
			{
				acc += static_cast<double>(row[i]) * static_cast<double>(paddedDirection[i]);
			}
		}
		outProjections[r] = static_cast<float>(acc);
	}

	if (groupBits != nullptr && outSeparation != nullptr)
	{
		// Cohen's-d separation (plan N3): the between-group projected-mean gap over the
		// pooled projected standard deviation.
		double sumA = 0.0;
		double sumB = 0.0;
		int32_t nA = 0;
		int32_t nB = 0;
		for (int32_t r = 0; r < bank.count; ++r)
		{
			const bool inA =
				(groupBits[static_cast<uint32_t>(r) >> 5] &
					(1u << (static_cast<uint32_t>(r) & 31u))) != 0u;
			if (inA)
			{
				sumA += static_cast<double>(outProjections[r]);
				++nA;
			}
			else
			{
				sumB += static_cast<double>(outProjections[r]);
				++nB;
			}
		}
		if (nA == 0 || nB == 0)
		{
			return Status::InvalidArgument;
		}
		const double meanA = sumA / static_cast<double>(nA);
		const double meanB = sumB / static_cast<double>(nB);
		double varA = 0.0;
		double varB = 0.0;
		for (int32_t r = 0; r < bank.count; ++r)
		{
			const bool inA =
				(groupBits[static_cast<uint32_t>(r) >> 5] &
					(1u << (static_cast<uint32_t>(r) & 31u))) != 0u;
			const double p = static_cast<double>(outProjections[r]);
			if (inA)
			{
				varA += (p - meanA) * (p - meanA);
			}
			else
			{
				varB += (p - meanB) * (p - meanB);
			}
		}
		varA /= static_cast<double>(nA);
		varB /= static_cast<double>(nB);
		const double pooled = std::sqrt((varA + varB) / 2.0);
		*outSeparation = pooled > 0.0
			? static_cast<float>((meanA - meanB) / pooled)
			: 0.0f;
	}
	return Status::Ok;
}

} // namespace superfaiss

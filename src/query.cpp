#include "superfaiss/query.h"

#include "superfaiss/kernels.h"
#include "superfaiss/topk.h"
#include "superfaiss/validate.h"

namespace superfaiss
{

namespace
{
	// Weight folding (T-050 W1 bench clause, taken and recorded in plan section 10):
	// for dot-family scoring, sum_s w_s * sum_{j in s} r_j q_j == sum_j r_j (w(j) q_j)
	// exactly, so segmented dot/cosine scans fold the segment weights into a query
	// copy once (gaps and weight-0 segments fold to 0, preserving omission
	// semantics) and run the plain V1 kernels at V1 speed. L2 cannot fold (the
	// weight sits inside the square) and keeps the dense range path.
	void FoldSegmentsIntoQuery(
		const float* paddedQuery,
		int32_t paddedDims,
		const QuerySegment* segments,
		int32_t segmentCount,
		float* outFolded)
	{
		for (int32_t j = 0; j < paddedDims; ++j)
		{
			outFolded[j] = 0.0f;
		}
		for (int32_t s = 0; s < segmentCount; ++s)
		{
			const QuerySegment& seg = segments[s];
			for (int32_t j = seg.offset; j < seg.offset + seg.length; ++j)
			{
				outFolded[j] = seg.weight * paddedQuery[j];
			}
		}
	}

	// --- v2.1 per-row bias (plan section 18) ---

	// One query's bias, form-resolved. Exactly one form; empty = unbiased.
	struct FBiasForm
	{
		const float* dense = nullptr;
		const BiasPair* pairs = nullptr;
		int32_t pairCount = 0;
	};

	// Form law: dense XOR pairs (both set is ambiguous composition); a pair count
	// without a pair pointer is malformed. Range/uniqueness/finiteness of sparse
	// entries is ValidateBiasPairs (needs scratch bits; runs at the call sites).
	Status ResolveBiasForm(const RowBias* bias, FBiasForm& out)
	{
		out = FBiasForm{};
		if (bias == nullptr)
		{
			return Status::Ok;
		}
		const bool hasDense = bias->dense != nullptr;
		const bool hasPairs = bias->pairs != nullptr || bias->pairCount != 0;
		if ((hasDense && hasPairs) || bias->pairCount < 0 ||
			(bias->pairCount > 0 && bias->pairs == nullptr))
		{
			return Status::InvalidArgument;
		}
		out.dense = bias->dense;
		out.pairs = bias->pairs;
		out.pairCount = bias->pairs != nullptr ? bias->pairCount : 0;
		return Status::Ok;
	}

	// Scores one row exactly as the surrounding scan scored it - the same kernel
	// entry points on the same effective query - so a sparse-bias pair row outside
	// the candidate set composes against a bit-identical similarity term.
	float RescoreRow(
		const BankView& scoring,
		const float* effectiveQuery,
		const float* originalQuery,
		const QueryParams& params,
		bool bFoldable,
		const XdQuery* xd,
		int32_t r)
	{
		if (xd != nullptr)
		{
			// CrossDevice rescore: the same integer kernels + double epilogue the
			// scan ran, so the composed similarity term is bit-identical.
			if (params.segments != nullptr)
			{
				float contributions[kMaxSegments];
				return DecomposeRowScoreXd(scoring, *xd, r, params.segments,
					params.segmentCount, contributions);
			}
			return ScoreRowXd(scoring, *xd, r);
		}
		if (params.segments != nullptr && !bFoldable)
		{
			float contributions[kMaxSegments];
			return DecomposeRowScore(scoring, originalQuery, r, params.segments,
				params.segmentCount, contributions);
		}
		const int32_t pd = scoring.paddedDims;
		const bool isL2 = scoring.metric == Metric::L2;
		if (scoring.quant == Quantization::Float32)
		{
			const float* row =
				static_cast<const float*>(scoring.rows) + static_cast<int64_t>(r) * pd;
			return isL2 ? detail::L2F32(row, effectiveQuery, pd)
			            : detail::DotF32(row, effectiveQuery, pd);
		}
		const int8_t* row =
			static_cast<const int8_t*>(scoring.rows) + static_cast<int64_t>(r) * pd;
		const float scale = scoring.scales[r];
		return isL2 ? detail::L2I8(row, scale, effectiveQuery, pd)
		            : detail::DotI8(row, scale, effectiveQuery, pd);
	}

	// Sparse-bias selection (the k+P construction): the scan ran UNBIASED at
	// capacity k+P, so its candidate list is guaranteed to contain the true
	// composed top-k's non-pair rows (at most P slots are pair rows), and every
	// pair row is evaluated explicitly - rewards can lift a rank-anything row in,
	// penalties can evict, and the result is exact either way. Candidates that are
	// pair rows are skipped here (their composed entry comes from the pair pass);
	// caller-excluded pair rows stay excluded - exclusion is a mask, bias is
	// arithmetic, orthogonal, and the mask wins.
	int32_t SelectWithSparseBias(
		const BankView& scoring,
		const Hit* candidates,
		int32_t candidateCount,
		const FBiasForm& bias,
		const uint32_t* pairBits,
		const float* effectiveQuery,
		const float* originalQuery,
		const QueryParams& params,
		bool bFoldable,
		const XdQuery* xd,
		Hit* selectionStorage,
		Hit* outHits)
	{
		TopK selection;
		selection.Init(selectionStorage, params.k, scoring.metric);
		for (int32_t i = 0; i < candidateCount; ++i)
		{
			if (!IsExcluded(pairBits, candidates[i].index))
			{
				selection.Push(candidates[i].index, candidates[i].score);
			}
		}
		for (int32_t p = 0; p < bias.pairCount; ++p)
		{
			const int32_t row = bias.pairs[p].index;
			if (IsExcluded(params.excludeBits, row))
			{
				continue;
			}
			// The candidate list already carries the row's unbiased score if the
			// row ranked; reuse it (bit-identical by the one-path argument), else
			// rescore through the same kernels. Linear probe: P is small at every
			// named scale (motion matching: one pair) - a caller with huge P wants
			// the dense form.
			float s = 0.0f;
			bool found = false;
			for (int32_t i = 0; i < candidateCount; ++i)
			{
				if (candidates[i].index == row)
				{
					s = candidates[i].score;
					found = true;
					break;
				}
			}
			if (!found)
			{
				s = RescoreRow(scoring, effectiveQuery, originalQuery, params,
					bFoldable, xd, row);
			}
			// CrossDevice composes through the floored-double form - the same
			// expression the dense in-scan path uses, so sparse == dense bitwise.
			selection.Push(row, xd != nullptr
				? detail::XdComposeBiasValue(s, bias.pairs[p].bias)
				: s + bias.pairs[p].bias);
		}
		return selection.Finalize(outHits);
	}
}

Status Query(
	const BankView& bank,
	const float* paddedQuery,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCount)
{
	if (outHits == nullptr || outCount == nullptr || params.k < 0)
	{
		return Status::InvalidArgument;
	}
	*outCount = 0;
	if (params.k == 0 || bank.count == 0)
	{
		return Status::Ok;
	}

	// Validation applies the bank's own rules regardless of any scoring override
	// (a Cosine bank rejects zero-norm queries under ScoreAs::Dot too); scoring and
	// hit ordering then run against the effective view.
	const Status queryStatus = ValidateQuery(bank, paddedQuery);
	if (queryStatus != Status::Ok)
	{
		return queryStatus;
	}

	if (params.segments != nullptr)
	{
		const Status segStatus =
			ValidateSegments(bank, paddedQuery, params.segments, params.segmentCount);
		if (segStatus != Status::Ok)
		{
			return segStatus;
		}
	}

	FBiasForm bias;
	const Status biasStatus = ResolveBiasForm(params.bias, bias);
	if (biasStatus != Status::Ok)
	{
		return biasStatus;
	}
	if (bias.pairCount > 0)
	{
		if (static_cast<int64_t>(params.k) + bias.pairCount > INT32_MAX ||
			!workspace.ReserveBiasBits(bank.count))
		{
			return Status::OutOfMemory;
		}
		const Status pairStatus =
			ValidateBiasPairs(bank, bias.pairs, bias.pairCount, workspace.BiasBits());
		if (pairStatus != Status::Ok)
		{
			return pairStatus;
		}
	}

	// CrossDevice laws (v2.2): Int8 banks only, and the paddedDims ceiling that
	// keeps every integer accumulator overflow-free.
	const bool bXd = params.exactness == Exactness::CrossDevice;
	if (bXd &&
		(bank.quant != Quantization::Int8 || bank.paddedDims > kMaxCrossDeviceDims))
	{
		return Status::InvalidArgument;
	}

	BankView scoring = bank;
	scoring.metric = ScoringMetric(bank, params);

	// Sparse bias scans at capacity k+P (see SelectWithSparseBias); three slots:
	// scan heap, finalized candidates, final-selection heap.
	const int32_t scanK = params.k + bias.pairCount;
	if (!workspace.Reserve(scanK, bias.pairCount > 0 ? 3 : 1))
	{
		return Status::OutOfMemory;
	}

	// Per-channel-cosine banks score through the dense path (the inverse sub-norm
	// is a row-side factor and cannot fold into the query); the metric override
	// composes to raw-dot projection and folds (section 5). CrossDevice never
	// folds: weights apply at combine, in double, on integer partials.
	const bool bPerChannelCosine = bank.metric == Metric::Cosine &&
		bank.channelInvNorms != nullptr && params.scoreAs == ScoreAs::BankMetric;
	const bool bFoldable = !bXd && params.segments != nullptr &&
		scoring.metric != Metric::L2 && !bPerChannelCosine;
	const float* effectiveQuery = paddedQuery;
	if (bFoldable)
	{
		if (!workspace.ReserveQueryScratch(bank.paddedDims, 1))
		{
			return Status::OutOfMemory;
		}
		FoldSegmentsIntoQuery(paddedQuery, bank.paddedDims, params.segments,
			params.segmentCount, workspace.QueryScratch(0));
		effectiveQuery = workspace.QueryScratch(0);
	}

	XdQuery xdQuery;
	if (bXd)
	{
		if (!workspace.ReserveXdQuery(bank.paddedDims, 1))
		{
			return Status::OutOfMemory;
		}
		QuantizeQueryXd(paddedQuery, bank.paddedDims, workspace.XdQ8(0),
			workspace.XdScale(0), workspace.XdSqSum(0));
		xdQuery.q8 = workspace.XdQ8(0);
		xdQuery.scale = *workspace.XdScale(0);
		xdQuery.sqSum = *workspace.XdSqSum(0);
	}

	TopK topk;
	topk.Init(workspace.HeapStorage(0), scanK, scoring.metric);

	bool nonFiniteBias = false;
	const int32_t chunks = ChunkCount(scoring);
	for (int32_t c = 0; c < chunks; ++c)
	{
		if (bXd)
		{
			if (params.segments != nullptr)
			{
				ScoreChunkSegmentedXd(scoring, xdQuery, c, params.excludeBits,
					params.segments, params.segmentCount, topk, bias.dense,
					&nonFiniteBias);
			}
			else
			{
				ScoreChunkXd(scoring, xdQuery, c, params.excludeBits, topk,
					bias.dense, &nonFiniteBias);
			}
		}
		else if (params.segments != nullptr && !bFoldable)
		{
			ScoreChunkSegmented(scoring, paddedQuery, c, params.excludeBits,
				params.segments, params.segmentCount, topk, bias.dense, &nonFiniteBias);
		}
		else
		{
			ScoreChunk(scoring, effectiveQuery, c, params.excludeBits, topk,
				bias.dense, &nonFiniteBias);
		}
	}
	if (nonFiniteBias)
	{
		return Status::NonFiniteQuery; // the finite-only bias law (fused check)
	}

	if (bias.pairCount > 0)
	{
		const int32_t candidateCount = topk.Finalize(workspace.HeapStorage(1));
		*outCount = SelectWithSparseBias(scoring, workspace.HeapStorage(1),
			candidateCount, bias, workspace.BiasBits(), effectiveQuery, paddedQuery,
			params, bFoldable, bXd ? &xdQuery : nullptr, workspace.HeapStorage(2),
			outHits);
		return Status::Ok;
	}

	*outCount = topk.Finalize(outHits);
	return Status::Ok;
}

// Width of one amortized sub-batch: TopK views for the sub-batch live on the stack.
// Batches wider than this are processed as consecutive sub-batches; correctness is
// unchanged (batch ≡ singles), the cache amortization simply resets per sub-batch.
static constexpr int32_t kSubBatchWidth = 64;

namespace
{
	// One sub-batch over the chunk-outer loop. `biases` (nullable, width entries,
	// form-resolved) rides the pair kernels: dense views pass straight into
	// ScoreChunkPair/ScoreChunk (per-query bias params - the row pass stays shared),
	// sparse queries scan unbiased at capacity k + P_m into workspace-side storage
	// and compose afterwards in QueryBatch. The bank still streams once (W1).
	void QuerySubBatch(
		const BankView& bank,
		const float* paddedQueries,
		int32_t queryCount,
		const QueryParams& params,
		const FBiasForm* biases,
		Workspace& workspace,
		Hit* outHits,
		int32_t* outCounts,
		bool* outNonFiniteBias,
		int32_t* outCandidateCounts)
	{
		TopK topks[kSubBatchWidth];
		for (int32_t m = 0; m < queryCount; ++m)
		{
			const int32_t pairs = biases != nullptr ? biases[m].pairCount : 0;
			topks[m].Init(workspace.HeapStorage(m), params.k + pairs, bank.metric);
		}
		// Callers pass the effective (already override-resolved) view; see QueryBatch.

		// Chunk loop outermost (cache amortization); queries processed in pairs so the
		// pair kernels share each row's loads and widening.
		const int32_t chunks = ChunkCount(bank);
		for (int32_t c = 0; c < chunks; ++c)
		{
			int32_t m = 0;
			for (; m + 2 <= queryCount; m += 2)
			{
				ScoreChunkPair(
					bank,
					paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
					paddedQueries + static_cast<int64_t>(m + 1) * bank.paddedDims,
					c,
					params.excludeBits,
					topks[m],
					topks[m + 1],
					biases != nullptr ? biases[m].dense : nullptr,
					biases != nullptr ? biases[m + 1].dense : nullptr,
					outNonFiniteBias);
			}
			if (m < queryCount)
			{
				ScoreChunk(
					bank,
					paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
					c,
					params.excludeBits,
					topks[m],
					biases != nullptr ? biases[m].dense : nullptr,
					outNonFiniteBias);
			}
		}

		for (int32_t m = 0; m < queryCount; ++m)
		{
			const bool bSparse = biases != nullptr && biases[m].pairCount > 0;
			if (bSparse)
			{
				// Finalize into the second storage bank; QueryBatch composes.
				outCandidateCounts[m] =
					topks[m].Finalize(workspace.HeapStorage(queryCount + m));
				outCounts[m] = 0;
			}
			else
			{
				outCounts[m] = topks[m].Finalize(outHits + static_cast<int64_t>(m) * params.k);
			}
		}
	}
}

Status QueryBatch(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCounts)
{
	if (outHits == nullptr || outCounts == nullptr || params.k < 0 || queryCount < 0)
	{
		return Status::InvalidArgument;
	}
	for (int32_t m = 0; m < queryCount; ++m)
	{
		outCounts[m] = 0;
	}
	if (params.k == 0 || bank.count == 0 || queryCount == 0)
	{
		return Status::Ok;
	}

	// Bank-rule validation, then the effective override-resolved view for scoring —
	// same contract as Query.
	for (int32_t m = 0; m < queryCount; ++m)
	{
		const Status queryStatus =
			ValidateQuery(bank, paddedQueries + static_cast<int64_t>(m) * bank.paddedDims);
		if (queryStatus != Status::Ok)
		{
			return queryStatus;
		}
	}

	if (params.segments != nullptr)
	{
		for (int32_t m = 0; m < queryCount; ++m)
		{
			const Status segStatus = ValidateSegments(bank,
				paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
				params.segments, params.segmentCount);
			if (segStatus != Status::Ok)
			{
				return segStatus;
			}
		}
	}

	// Bias forms per query (v2.1): params.bias carries queryCount entries. Sparse
	// entries pre-validate here - before any scan work - reusing one bits buffer
	// serially; the bits are rebuilt per query at composition time.
	FBiasForm biasStack[kSubBatchWidth];
	FBiasForm* biasForms = nullptr; // per sub-batch slice, resolved below
	int32_t maxPairs = 0;
	if (params.bias != nullptr)
	{
		for (int32_t m = 0; m < queryCount; ++m)
		{
			FBiasForm form;
			const Status biasStatus = ResolveBiasForm(&params.bias[m], form);
			if (biasStatus != Status::Ok)
			{
				return biasStatus;
			}
			if (form.pairCount > 0)
			{
				if (!workspace.ReserveBiasBits(bank.count))
				{
					return Status::OutOfMemory;
				}
				const Status pairStatus = ValidateBiasPairs(bank, form.pairs,
					form.pairCount, workspace.BiasBits());
				if (pairStatus != Status::Ok)
				{
					return pairStatus;
				}
				maxPairs = form.pairCount > maxPairs ? form.pairCount : maxPairs;
			}
		}
		if (static_cast<int64_t>(params.k) + maxPairs > INT32_MAX)
		{
			return Status::OutOfMemory;
		}
	}

	const bool bXd = params.exactness == Exactness::CrossDevice;
	if (bXd &&
		(bank.quant != Quantization::Int8 || bank.paddedDims > kMaxCrossDeviceDims))
	{
		return Status::InvalidArgument;
	}

	BankView scoring = bank;
	scoring.metric = ScoringMetric(bank, params);

	const bool bBatchPerChannelCosine = bank.metric == Metric::Cosine &&
		bank.channelInvNorms != nullptr && params.scoreAs == ScoreAs::BankMetric;
	const bool bFoldable = !bXd && params.segments != nullptr &&
		scoring.metric != Metric::L2 && !bBatchPerChannelCosine;

	// Dot-family segmented batch: fold each query once, then the batch IS the
	// plain V1 batch (pair kernels included) over the folded queries.
	const float* effectiveQueries = paddedQueries;
	if (bFoldable)
	{
		if (!workspace.ReserveQueryScratch(bank.paddedDims, queryCount))
		{
			return Status::OutOfMemory;
		}
		// Pack the folded queries at THIS bank's stride, not the scratch buffer's
		// internal one: the scratch grows monotonically (allocation-flat contract),
		// so after serving a larger reservation QueryScratch(m) sits at a WIDER
		// stride than bank.paddedDims - and every consumer below walks
		// effectiveQueries by bank.paddedDims. Packing from the base keeps producer
		// and consumers on one stride regardless of the workspace's history.
		// (External bug report 2026-07-04: wrong hits on perfectly normal reuse;
		// T25 pins this. paddedDims is a whole number of kAlignment blocks, so
		// every packed query stays aligned.)
		{
			float* foldedBase = workspace.QueryScratch(0);
			for (int32_t m = 0; m < queryCount; ++m)
			{
				FoldSegmentsIntoQuery(
					paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
					bank.paddedDims, params.segments, params.segmentCount,
					foldedBase + static_cast<int64_t>(m) * bank.paddedDims);
			}
			effectiveQueries = foldedBase;
		}
	}

	const int32_t maxWidth = queryCount < kSubBatchWidth ? queryCount : kSubBatchWidth;
	// Sparse-bias queries finalize into a second storage bank (slots width..2*width)
	// before composition; every slot is sized for the widest k+P.
	if (!workspace.Reserve(params.k + maxPairs, maxPairs > 0 ? 2 * maxWidth + 1 : maxWidth))
	{
		return Status::OutOfMemory;
	}
	if (bXd && !workspace.ReserveXdQuery(bank.paddedDims, maxWidth))
	{
		return Status::OutOfMemory;
	}

	const bool bNonFoldSegmented = params.segments != nullptr && !bFoldable;
	bool nonFiniteBias = false;
	int32_t candidateCounts[kSubBatchWidth];

	for (int32_t base = 0; base < queryCount; base += kSubBatchWidth)
	{
		const int32_t width =
			(queryCount - base) < kSubBatchWidth ? (queryCount - base) : kSubBatchWidth;
		if (params.bias != nullptr)
		{
			for (int32_t m = 0; m < width; ++m)
			{
				FBiasForm form;
				(void)ResolveBiasForm(&params.bias[base + m], form); // validated above
				biasStack[m] = form;
			}
			biasForms = biasStack;
		}

		XdQuery* xdSlots = nullptr;
		if (bXd)
		{
			// CrossDevice quantization per sub-batch query; the scan below keeps
			// the chunk-outermost structure with the single-query CrossDevice
			// kernel inside (batch == singles bitwise by construction; the pair
			// row-sharing kernels are a PerDevice throughput device, not used).
			xdSlots = workspace.XdSlots();
			for (int32_t m = 0; m < width; ++m)
			{
				QuantizeQueryXd(
					paddedQueries + static_cast<int64_t>(base + m) * bank.paddedDims,
					bank.paddedDims, workspace.XdQ8(m), workspace.XdScale(m),
					workspace.XdSqSum(m));
				xdSlots[m].q8 = workspace.XdQ8(m);
				xdSlots[m].scale = *workspace.XdScale(m);
				xdSlots[m].sqSum = *workspace.XdSqSum(m);
			}
		}

		if (bNonFoldSegmented || bXd)
		{
			// Segmented (and CrossDevice) batches keep the chunk-outermost
			// structure — the bank streams once per batch — with the single-query
			// kernel scoring each query inside the chunk (plan 18.7/W1).
			TopK topks[kSubBatchWidth];
			for (int32_t m = 0; m < width; ++m)
			{
				const int32_t pairs = biasForms != nullptr ? biasForms[m].pairCount : 0;
				topks[m].Init(workspace.HeapStorage(m), params.k + pairs, scoring.metric);
			}
			const int32_t chunks = ChunkCount(scoring);
			for (int32_t c = 0; c < chunks; ++c)
			{
				for (int32_t m = 0; m < width; ++m)
				{
					const float* denseBias =
						biasForms != nullptr ? biasForms[m].dense : nullptr;
					if (bXd)
					{
						if (params.segments != nullptr)
						{
							ScoreChunkSegmentedXd(scoring, xdSlots[m], c,
								params.excludeBits, params.segments,
								params.segmentCount, topks[m], denseBias,
								&nonFiniteBias);
						}
						else
						{
							ScoreChunkXd(scoring, xdSlots[m], c, params.excludeBits,
								topks[m], denseBias, &nonFiniteBias);
						}
					}
					else
					{
						ScoreChunkSegmented(scoring,
							paddedQueries + static_cast<int64_t>(base + m) * bank.paddedDims,
							c, params.excludeBits, params.segments, params.segmentCount,
							topks[m], denseBias, &nonFiniteBias);
					}
				}
			}
			for (int32_t m = 0; m < width; ++m)
			{
				const bool bSparse = biasForms != nullptr && biasForms[m].pairCount > 0;
				if (bSparse)
				{
					candidateCounts[m] =
						topks[m].Finalize(workspace.HeapStorage(width + m));
				}
				else
				{
					outCounts[base + m] = topks[m].Finalize(
						outHits + static_cast<int64_t>(base + m) * params.k);
				}
			}
		}
		else
		{
			QuerySubBatch(
				scoring,
				effectiveQueries + static_cast<int64_t>(base) * bank.paddedDims,
				width,
				params,
				biasForms,
				workspace,
				outHits + static_cast<int64_t>(base) * params.k,
				outCounts + base,
				&nonFiniteBias,
				candidateCounts);
		}
		if (nonFiniteBias)
		{
			return Status::NonFiniteQuery;
		}

		// Sparse composition per query (bits rebuilt per query; the pairs were
		// validated before any scanning).
		if (biasForms != nullptr)
		{
			for (int32_t m = 0; m < width; ++m)
			{
				if (biasForms[m].pairCount == 0)
				{
					continue;
				}
				if (!workspace.ReserveBiasBits(bank.count))
				{
					return Status::OutOfMemory;
				}
				uint32_t* bits = workspace.BiasBits();
				for (int32_t p = 0; p < biasForms[m].pairCount; ++p)
				{
					const int32_t row = biasForms[m].pairs[p].index;
					bits[row >> 5] |= 1u << (row & 31);
				}
				const float* effective =
					effectiveQueries + static_cast<int64_t>(base + m) * bank.paddedDims;
				const float* original =
					paddedQueries + static_cast<int64_t>(base + m) * bank.paddedDims;
				outCounts[base + m] = SelectWithSparseBias(scoring,
					workspace.HeapStorage(width + m), candidateCounts[m], biasForms[m],
					bits, effective, original, params, bFoldable,
					bXd ? &xdSlots[m] : nullptr,
					workspace.HeapStorage(2 * maxWidth), // shared selection slot
					outHits + static_cast<int64_t>(base + m) * params.k);
			}
		}
	}
	return Status::Ok;
}

Status QueryIntersect(
	const BankView& bank,
	const float* paddedQueries,
	int32_t queryCount,
	const QueryParams& params,
	Workspace& workspace,
	Hit* outHits,
	int32_t* outCount)
{
	if (outHits == nullptr || outCount == nullptr || params.k < 0 || queryCount <= 0)
	{
		return Status::InvalidArgument;
	}
	*outCount = 0;
	if (params.k == 0 || bank.count == 0)
	{
		return Status::Ok;
	}

	// Bank-rule validation for every member query; the effective override-resolved
	// view for scoring — the same split as Query/QueryBatch.
	for (int32_t m = 0; m < queryCount; ++m)
	{
		const Status queryStatus =
			ValidateQuery(bank, paddedQueries + static_cast<int64_t>(m) * bank.paddedDims);
		if (queryStatus != Status::Ok)
		{
			return queryStatus;
		}
	}

	if (params.segments != nullptr)
	{
		for (int32_t m = 0; m < queryCount; ++m)
		{
			const Status segStatus = ValidateSegments(bank,
				paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
				params.segments, params.segmentCount);
			if (segStatus != Status::Ok)
			{
				return segStatus;
			}
		}
	}

	// Intersection bias (v2.1): ONE RowBias, applied once to the fused score in the
	// fused metric's direction.
	FBiasForm bias;
	const Status biasStatus = ResolveBiasForm(params.bias, bias);
	if (biasStatus != Status::Ok)
	{
		return biasStatus;
	}
	if (bias.pairCount > 0)
	{
		if (static_cast<int64_t>(params.k) + bias.pairCount > INT32_MAX ||
			!workspace.ReserveBiasBits(bank.count))
		{
			return Status::OutOfMemory;
		}
		const Status pairStatus =
			ValidateBiasPairs(bank, bias.pairs, bias.pairCount, workspace.BiasBits());
		if (pairStatus != Status::Ok)
		{
			return pairStatus;
		}
	}

	const bool bXd = params.exactness == Exactness::CrossDevice;
	if (bXd &&
		(bank.quant != Quantization::Int8 || bank.paddedDims > kMaxCrossDeviceDims))
	{
		return Status::InvalidArgument;
	}

	BankView scoring = bank;
	scoring.metric = ScoringMetric(bank, params);

	const int32_t scanK = params.k + bias.pairCount;
	if (!workspace.Reserve(scanK, bias.pairCount > 0 ? 3 : 1))
	{
		return Status::OutOfMemory;
	}

	TopK topk;
	topk.Init(workspace.HeapStorage(0), scanK, scoring.metric);

	const bool bIntersectPerChannelCosine = bank.metric == Metric::Cosine &&
		bank.channelInvNorms != nullptr && params.scoreAs == ScoreAs::BankMetric;
	const bool bFoldable = !bXd && params.segments != nullptr &&
		scoring.metric != Metric::L2 && !bIntersectPerChannelCosine;
	const float* effectiveQueries = paddedQueries;
	if (bFoldable)
	{
		if (!workspace.ReserveQueryScratch(bank.paddedDims, queryCount))
		{
			return Status::OutOfMemory;
		}
		// Pack the folded queries at THIS bank's stride, not the scratch buffer's
		// internal one: the scratch grows monotonically (allocation-flat contract),
		// so after serving a larger reservation QueryScratch(m) sits at a WIDER
		// stride than bank.paddedDims - and every consumer below walks
		// effectiveQueries by bank.paddedDims. Packing from the base keeps producer
		// and consumers on one stride regardless of the workspace's history.
		// (External bug report 2026-07-04: wrong hits on perfectly normal reuse;
		// T25 pins this. paddedDims is a whole number of kAlignment blocks, so
		// every packed query stays aligned.)
		{
			float* foldedBase = workspace.QueryScratch(0);
			for (int32_t m = 0; m < queryCount; ++m)
			{
				FoldSegmentsIntoQuery(
					paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
					bank.paddedDims, params.segments, params.segmentCount,
					foldedBase + static_cast<int64_t>(m) * bank.paddedDims);
			}
			effectiveQueries = foldedBase;
		}
	}

	XdQuery* xdSlots = nullptr;
	if (bXd)
	{
		if (!workspace.ReserveXdQuery(bank.paddedDims, queryCount))
		{
			return Status::OutOfMemory;
		}
		xdSlots = workspace.XdSlots();
		for (int32_t m = 0; m < queryCount; ++m)
		{
			QuantizeQueryXd(
				paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
				bank.paddedDims, workspace.XdQ8(m), workspace.XdScale(m),
				workspace.XdSqSum(m));
			xdSlots[m].q8 = workspace.XdQ8(m);
			xdSlots[m].scale = *workspace.XdScale(m);
			xdSlots[m].sqSum = *workspace.XdSqSum(m);
		}
	}

	bool nonFiniteBias = false;
	const int32_t chunks = ChunkCount(scoring);
	for (int32_t c = 0; c < chunks; ++c)
	{
		if (bXd)
		{
			ScoreChunkFusedXd(scoring, xdSlots, queryCount, c, params.excludeBits,
				params.segments, params.segmentCount, topk, bias.dense,
				&nonFiniteBias);
		}
		else if (params.segments != nullptr && !bFoldable)
		{
			ScoreChunkFusedSegmented(scoring, paddedQueries, queryCount, c,
				params.excludeBits, params.segments, params.segmentCount, topk,
				bias.dense, &nonFiniteBias);
		}
		else
		{
			ScoreChunkFused(scoring, effectiveQueries, queryCount, c,
				params.excludeBits, topk, bias.dense, &nonFiniteBias);
		}
	}
	if (nonFiniteBias)
	{
		return Status::NonFiniteQuery;
	}

	if (bias.pairCount > 0)
	{
		const int32_t candidateCount = topk.Finalize(workspace.HeapStorage(1));

		// Fused rescore for pair rows outside the candidate set: worst-of over the
		// member queries' per-row scores, the same selection the fused kernels run.
		TopK selection;
		selection.Init(workspace.HeapStorage(2), params.k, scoring.metric);
		const Hit* candidates = workspace.HeapStorage(1);
		const uint32_t* pairBits = workspace.BiasBits();
		const bool isL2 = scoring.metric == Metric::L2;
		for (int32_t i = 0; i < candidateCount; ++i)
		{
			if (!IsExcluded(pairBits, candidates[i].index))
			{
				selection.Push(candidates[i].index, candidates[i].score);
			}
		}
		for (int32_t p = 0; p < bias.pairCount; ++p)
		{
			const int32_t row = bias.pairs[p].index;
			if (IsExcluded(params.excludeBits, row))
			{
				continue;
			}
			float fused = 0.0f;
			bool found = false;
			for (int32_t i = 0; i < candidateCount; ++i)
			{
				if (candidates[i].index == row)
				{
					fused = candidates[i].score;
					found = true;
					break;
				}
			}
			if (!found)
			{
				for (int32_t m = 0; m < queryCount; ++m)
				{
					const float score = RescoreRow(scoring,
						effectiveQueries + static_cast<int64_t>(m) * bank.paddedDims,
						paddedQueries + static_cast<int64_t>(m) * bank.paddedDims,
						params, bFoldable, bXd ? &xdSlots[m] : nullptr, row);
					if (m == 0 || (isL2 ? score > fused : score < fused))
					{
						fused = score;
					}
				}
			}
			selection.Push(row, bXd
				? detail::XdComposeBiasValue(fused, bias.pairs[p].bias)
				: fused + bias.pairs[p].bias);
		}
		*outCount = selection.Finalize(outHits);
		return Status::Ok;
	}

	*outCount = topk.Finalize(outHits);
	return Status::Ok;
}

} // namespace superfaiss

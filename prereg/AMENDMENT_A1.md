# READSWAP — Amendment A1 (locked before any confirmatory output exists)

Locked 2026-09-24, after an independent framework review.

**State at locking.** Only the pilot exists: 40 DeepMath problems, OctoThinker-3B, seed 0. The confirmatory phase-1 driver (generation and grading) is waiting in the GPU queue and no confirmatory response has been generated.

PREREG_v1 stays in force. This amendment fixes the analytic choices that PREREG_v1 leaves open, and it changes the exclusion rule for R3.

## 1. Scores, probabilities and weighted voting
- **Score.** For every readout, the score of a response is s = β_v · (log π(z | x, y) − c), exactly as in the released scorer. Here z is the readout's token, c the readout's reference term, and β_v the checkpoint's coefficient. For R4, c = 0.
  - LaSeR regresses this score onto the 0/1 verifiable reward and thresholds it at 0.5, so the score is used on its own scale.
- **Probability** (for ECE and Brier) is clip(s, 0, 1) for every readout. No normalization is fitted to any condition.
  - ECE uses 15 equal-width bins on [0, 1].
- **Weighted majority voting (RM@8)**, the rule used in L4:
  - answers are grouped into equivalence classes with the grader: two answers belong together when `math_verify` judges them equal;
  - each class receives the sum of the scores s of its responses, with no clipping and no transformation;
  - the class with the largest sum is selected;
  - ties go to the class with more responses, then to the class whose first response comes earliest;
  - responses without an extracted answer form no class.
- **Why linear weights.** This linear rule is the one that makes L4 meaningful. Under any shift-invariant weighting, such as exp(s) or a softmax, subtracting a common constant multiplies every weight by the same factor. The constant then cannot change the selection, and L4 would be 0 by construction.
- **Secondary.** RM@8 with exp(s) weights is reported for every readout, together with this analytic fact.
- **Unweighted majority voting** (Maj@8) uses the same equivalence classes.

## 2. AUROC: stratified primary, pooled secondary
- **Primary AUROC** is the response-weighted average, over the five benchmarks, of each benchmark's AUROC computed over all responses of that benchmark (three seeds pooled). This removes comparisons between responses to problems of different benchmarks.
- **Secondary:** the AUROC pooled over all responses, and per-benchmark AUROCs.
- L1, L2 and L5 are computed on the primary AUROC.

## 3. R3 when a response has no final `\boxed{}`
- **R3** is defined only for responses with a parsable final `\boxed{...}`.
- **Exclusion rule.** The PREREG_v1 rule ("problems with any missing condition are removed") applies to R0, R1, R2, R4 and R5, which are defined for every response. It does not apply to R3.
- **R3-based quantities.**
  - R3's AUROC and L5 are computed on the box-compliant responses only.
  - Missingness is reported by checkpoint × benchmark × correctness.
  - A sensitivity analysis sets R3 := R2 for responses without a box.

## 4. Equivalence bounds (for claims of "no difference")
A difference is called negligible only when the 90% bootstrap interval (two one-sided tests at α = 0.05) lies inside these bounds:

| Contrast | Bound |
|---|---|
| L1, L2 (AUROC difference) | ±0.02 |
| L4 (RM@8 accuracy difference) | ±2 percentage points |
| L5 (difference in the short–long AUROC gap) | ±0.02 |
| L3 (partial correlation) | ±0.05 |

A difference that is neither significant (the PREREG_v1 rule) nor inside its bound is reported as inconclusive.

## 5. Interpretation, fixed in advance
- **L3** is reported as the conditional association between score and response length, given correctness. It is not a causal length effect.
- **R2 and R3** change the prefix endpoint: R2 drops the EOS token; R3 drops the text after the final answer. Their contrasts are reported as effects of these specified prefix changes, not of position alone.
- **R1 tokens** are described as "special tokens that were not the LaSeR regression target"; each is reported separately next to their mean.
- **Uncertainty.** Inference is conditional on the generated solutions. Per-seed estimates are reported.
- A contrast that holds in at least two of the three checkpoints is described as "replicated in k of 3 checkpoints". The checkpoints are listed by base-model lineage.

## 6. Response length
Length is the number of generated tokens. Characters are reported only as a secondary check.

## Unchanged
Checkpoints, datasets, generation settings, seeds, the six readouts and their token lists, the cluster bootstrap over problems (10,000 resamples), Holm across L1–L5 within each checkpoint, and the PREREG_v1 decision rule.

# AdapR: Adaptive Criterion Auto-Rubric

This document describes the AdapR method implemented in `src/aeollm2_adapr/`. AdapR achieved a train-set Spearman of **0.799** (+0.193 over the official baseline of 0.606).

## Overview

AdapR departs from rubric-based scoring (ODT/OSODT) in a fundamental way: rather than scoring against the official human-authored rubric, it generates question-specific scoring criteria automatically from the research question text alone, then applies an adaptive scoring strategy based on the question type.

The method has three core components:

1. **AutoR** — question-specific criterion generation with per-criterion scoring anchors
2. **Question type classification** — classifies each question as Type A or Type B
3. **Adaptive IF scoring** — uses an item-by-item checklist for Type A, generated criteria for Type B

Each document is scored with **four independent LLM calls**, one per dimension.

## Step 0: Preprocessing (once per research question)

### AutoR criterion generation

For each research question, AdapR calls the LLM to generate 3–5 scoring sub-criteria for each of the three content dimensions: Comprehensiveness, Insight, and Instruction Following. Each sub-criterion contains:

- **Name and weight** — weights within a dimension sum to 1.0
- **Explanation** — 1–2 sentences describing what the criterion measures and how to judge it
- **Four scoring anchors** — one sentence each for the 9–10, 7–8, 5–6, and 0–4 score bands, describing the specific document characteristics at that level

The anchor descriptions are question-specific and criterion-specific. For example, a criterion about coverage of nine social strata might anchor the 9–10 band as "all nine strata covered with independent income data per stratum," rather than a generic "excellent coverage." This narrows the scoring ambiguity that arises when a single dimension-level description is applied across heterogeneous sub-criteria.

The generation prompt is written in Chinese, consistent with the judging prompts. See `src/aeollm2_adapr/criterion_gen.py` for the exact prompt text. Generated criteria are cached to disk and reused across documents for the same question.

Readability uses a fixed three-criterion rubric (logical structure, expression, bias check) that is question-agnostic and therefore excluded from AutoR generation.

### Question type classification

AdapR classifies each research question into one of two types:

| Type | Characteristics | IF scoring strategy |
|------|----------------|---------------------|
| **A** (checklist) | Contains multiple distinct, verifiable sub-tasks with clear boundaries (e.g., "survey X, then predict Y, then recommend Z") | Item-by-item requirement checklist |
| **B** (criteria) | Open-ended survey or synthesis; depth and quality are the primary differentiators; no explicit step structure | AutoR-generated IF sub-criteria |

Classification is performed by a single LLM call with a binary output. Results are cached. On the ten training-set questions, Q1, Q2, Q4, Q5, Q6, and Q16 were classified as Type A; Q3, Q7, Q17, and Q18 as Type B.

### IF requirement checklist (Type A only)

For Type A questions, the research question is decomposed into 3–6 concrete, independently verifiable requirements, each with a weight. This checklist replaces the AutoR IF criteria during scoring for that question.

## Steps 1–4: Scoring (four LLM calls per document)

| Call | Dimension | Type A | Type B |
|------|-----------|--------|--------|
| 1 | Comprehensiveness | AutoR criteria + per-criterion anchors | same |
| 2 | Insight | AutoR criteria + per-criterion anchors | same |
| 3 | Instruction Following | Item-by-item IF requirement checklist | AutoR criteria + per-criterion anchors |
| 4 | Readability | Fixed rubric (structure / expression / bias) | same |

For calls 1–2 and the Type B path of call 3, the prompt provides the full document text, the research question, and the sub-criteria with their anchors. The model returns one numeric score (0–10) per sub-criterion. The final dimension score is the weighted sum of sub-criterion scores.

For the Type A IF checklist (call 3), each requirement is listed with its weight and a four-level coverage description (9–10 / 7–8 / 5–6 / 0–4). The model scores each requirement independently, and the dimension score is the weighted sum.

For readability (call 4), the prompt omits the research question entirely and uses the fixed three-criterion rubric.

All scoring prompts use `temperature=0` and disable thinking mode.

## Scoring anchors: motivation

ODT and OSODT use dimension-level anchors from `prompts/prompt_scales.json` — one generic four-level scale per dimension applied across all sub-criteria. AdapR moves these anchors to the sub-criterion level, generated jointly with the criteria. The effect is most pronounced for Comprehensiveness: per-criterion anchors gave a Spearman improvement of approximately +0.23 over the comparable ODT configuration on the training set, while the other dimensions were roughly unchanged. The likely mechanism is that coverage judgments benefit more from concrete, topic-specific reference points than do insight or readability judgments, which depend more on holistic evaluation.

## Adaptive IF strategy: motivation

Applying a checklist to all questions (without A/B classification) gives a train Spearman of 0.782. Applying AutoR criteria to all questions gives 0.773. The adaptive strategy reaches 0.799. The degradation from a uniform checklist comes from Type B questions: checklist scoring enumerates sub-tasks that are not pre-specified in the question, so the decomposition is either incomplete or poorly aligned with what the question actually demands, introducing noise. Uniform criteria are too unstructured for Type A questions, where explicit sub-tasks exist and can be checked objectively.

## Reproducibility notes

The generation prompt and scoring prompts are in Chinese. English translations are not substituted because doing so changes the model input and may shift scores; see [`docs/prompt.md`](prompt.md) for the parallel note on ODT/OSODT prompts.

All LLM outputs — generated criteria, question type labels, IF checklists, and per-dimension scores — are cached in the directory specified by `--cache-dir`. Re-running with the same cache directory will replay cached results without API calls, except for documents not yet scored. Changing the model or any prompt requires a new cache directory to avoid stale hits.

## Training-set results

| Method | Spearman | Kendall | Accuracy |
|--------|----------|---------|----------|
| Baseline (glm-4.7) | 0.606 | 0.463 | 0.731 |
| Method 7 (4 independent calls, dim-level anchors) | 0.773 | 0.612 | 0.804 |
| **AdapR** | **0.799** | **0.646** | **0.821** |

Per-dimension training-set Spearman (10-topic average):

| Dimension | AdapR | Method 7 |
|-----------|-------|----------|
| Comprehensiveness | 0.535 | 0.308 |
| Insight | 0.667 | 0.711 |
| Instruction Following | 0.513 | 0.546 |
| Readability | 0.556 | 0.586 |

The large Comprehensiveness gain (+0.23) dominates the overall improvement. Insight, IF, and Readability are slightly lower than Method 7, likely due to noise at this effect size and indirect score-distribution shifts from the adaptive IF strategy.

## Directions explored but not adopted

| Direction | Spearman | Outcome |
|-----------|----------|---------|
| C-dimension checklist | 0.773 | Comprehensiveness needs depth judgment, not binary coverage |
| C-dimension two-step (evidence extraction then scoring) | 0.774 | Extraction summary loses document detail; hurts discrimination on uniform-quality topics |
| Pairwise post-processing reranking | 0.799 | No change; only 4 minor adjustments; wrong-ranked pairs were non-adjacent |
| Multiple scoring runs (n=3, temperature=0.3) | +0.005 | Marginal gain at 3× API cost |

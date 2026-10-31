# Judging prompts: English overview

This document explains the ODT/OSODT judging prompt structure in English. **It is explanatory documentation, not an alternative executable prompt.** The actual model inputs use the Chinese instructions in [`src/aeollm2_osodt/prompting.py`](../src/aeollm2_osodt/prompting.py) and Chinese scoring anchors in [`prompts/prompt_scales.json`](../prompts/prompt_scales.json). Those source files remain unchanged to preserve the experimental protocol.

## Message structure

For each target dimension, the implementation constructs a system message and a user message:

1. **System message:** require strict JSON output, including the specified beginning and ending markers for `paper-v1`. The judge is asked to examine the evidence internally but not to output reasoning or explanations.
2. **Role and scope:** act as an expert reviewer of the target dimension only. Avoid mixing other dimensions into the assessment unless the criterion explicitly requires it.
3. **Dimension-specific strategy:** evaluate coverage for comprehensiveness; analytical depth for insight; explicit task requirements for instruction following; and clarity, organization, and reading experience for readability.
4. **Scoring anchors:** apply the dimension-specific 0–10 scoring guidance loaded from `prompts/prompt_scales.json`.
5. **Original task:** insert the task/question from the official rubric dataset.
6. **Report:** insert the full report text, delimited by explicit document-begin and document-end markers.
7. **Rubric criteria:** list the target dimension's criteria and explanations in prompt order, together with their weights.
8. **Output instruction:** return one numeric score per criterion, in the presented order, as a JSON array under the target dimension's key. Do not include analysis, criterion names, Markdown, or other text.

The rubric criteria and original task are dynamic inputs obtained from official task data. The explanation above does not translate or replace those data.

## Original versus reversed criterion order

For a target dimension with criteria `[c1, c2, ..., cm]`:

- **Original ODT:** present `[c1, c2, ..., cm]`.
- **Reversed ODT:** present `[cm, ..., c2, c1]`.
- **OSODT:** run both orders, align the returned scores by criterion identity, average the two scores for each criterion, then apply the official rubric weights.

The original/reversed prompts differ **only in criterion presentation order**; role, dimension-specific instructions, scoring anchors, original question, and report text remain the same. The implementation also supports an `e9-schema-v1` response-format variant; `paper-v1` is the paper configuration.

## Reproducibility note

Replacing Chinese instructions or scoring anchors with English translations changes the model input and may change its scores, even when the meaning appears equivalent. For an experimental replication, use the code's original Chinese messages. The English descriptions here are intended to help readers understand the protocol, including when preparing an English figure or presentation.

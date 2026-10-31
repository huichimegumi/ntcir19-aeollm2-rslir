# SRAD: final readability method

Sequential Reference and Attention Decay (SRAD) supplies **readability only** in the
formal SRAD+OSODT system. OSODT supplies comprehensiveness, insight, and instruction
following. This package contains the final SRAD workflow; earlier readability
methods and parameter-search scripts are not included.

## Implementation and historical protocol

- `src/aeollm2_srad/evaluator.py`: section scoring, reference chains, aggregation,
  retry handling and resumable run artifacts.
- `src/aeollm2_srad/sections.py`: structural section splitting.
- `src/aeollm2_srad/prompting.py`: original Chinese prompt templates.
- `src/aeollm2_srad/prompt_scales.json`: the historical readability anchors, packaged
  with the module so installed commands can find them outside the source tree.
- `src/aeollm2_srad/cli.py`: configuration, input discovery and input fingerprints.
- `src/aeollm2_srad/submission.py`: validation and final-system merging.

The scoring code was extracted from the formal-run script
`one-dimension-with-sequential-global-reference.py` (SHA-256
`a3b5b1982a9d6dbc7d70847ab612cc832a86e7c1696ff3c4fd0a9a3906cffc27`).
The extraction preserves prompt text, section splitting, score parsing, rounding,
reference-order generation and context fallback. Earlier evaluation runners and
workbook-based model-identity lookup have been removed. Official `ReportN/Doc_###.docx`
paths now identify documents directly. The OSODT and AdapR implementations are unchanged.

SRAD retains the historical paragraph-based DOCX reader. It does not switch to the
OSODT reader's paragraph-and-table extraction, which would change the model input.
References can be excluded from local section units by the original splitter;
the full-document prompt still receives the original extracted document text.

For each document, score structural sections with full-document context. Weight
section positions linearly from 1 to beta=0.4, normalize the weighted sum, and
average three repetitions. For each question, evaluate reports in three orders.
The first report is judged independently; subsequent reports receive the previous
report's full text and **global** score. Average each document's three global
scores, then combine `0.3 * section + 0.7 * global`. No forced ordering or fixed
score gap is used. Historical four-decimal score rounding is preserved.

Order generation sorts numeric document IDs, shuffles with
`random.Random(seed * 100000 + question_id)`, and swaps the first document when
needed to obtain different starting documents across orders where possible.
The seeds are 42, 43 and 44. Use the complete document population for each question
when reproducing its reference chains; a subset changes those chains.

The original failure behavior is retained within scoring: missing section scores
are retried up to twice with the section-only retry prompt; at most 10% may be
omitted, with the remaining position weights renormalized. Global context-length
errors (including a provider `finish_reason=length`) trigger an independent global
judgment instead of truncating the paired prompt. These events are recorded in
run artifacts. A failed question now exits with an error rather than silently
continuing to produce a partial final submission.

## Configuration and inputs

Use [the final configuration](../configs/srad-deepseek-v4-flash-paper.yaml):
DeepSeek-v4-Flash, temperature 0, thinking off, section maximum output 4096 tokens,
global maximum output 2048 tokens, beta 0.4, alpha 0.3, three section repetitions,
and three reference orders. Global calls within each order are sequential.
The historical OpenAI SDK version was 2.44.0; the shared dependency excludes SDK
major version 3. No extra Excel or progress-bar dependency is needed for SRAD.

Follow [data.md](data.md) to obtain reports and rubrics. The inference command does
not read human labels or the original mapping workbook. Keep original question IDs
and blind document IDs. Use the same report population and split for both methods.

```bash
export DEEPSEEK_API_KEY='your-key'
aeollm2-srad \
  --config configs/srad-deepseek-v4-flash-paper.yaml \
  --report-root data/official/aeollm-2-train/reports \
  --rubric-dir data/official/aeollm-2-train/rubric_dataset \
  --output-dir outputs/deepseek-v4-flash-srad
```

For another available split, point both methods at that split's reports and
rubrics. The command above runs the public training population; it does not
regenerate the organizer's held-out formal metrics.

## Outputs and resuming

The output directory contains:

- `predictions/srad.tsv`: headed seven-column TSV; the three non-readability score
  columns are empty. This is an intermediate file, not the final submission.
- `diagnostics/section.tsv`, `diagnostics/global.tsv`: component scores produced
  during the same final-method run, not separate legacy methods.
- `section_repeat_scores.jsonl`, `global_chain_scores.jsonl`,
  `document_summary.jsonl`, `chain_orders.json`: scores, reference identities,
  fallback events and the actual report orders.
- `manifest.json`, `experiment_params.json`, `run.log`: input/code hashes,
  configuration and run status. The actual API key is not serialized.

Rerun the identical command to reuse completed work. Changed input files,
configuration or implementation require a new output directory. Do not edit or
mix state files between runs. Runtime artifacts remain under ignored `outputs/`.

## Form the final SRAD+OSODT submission

First obtain OSODT predictions using the existing README command. Its runner
currently generates all four dimensions; only its first three are retained here.

```bash
aeollm2-merge-final \
  --osodt outputs/deepseek-v4-flash-osodt/predictions/osodt.tsv \
  --srad outputs/deepseek-v4-flash-srad/predictions/srad.tsv \
  --output outputs/final/SRAD-OSODT.tsv
```

Alternatively use `python scripts/merge_final_submission.py` with the same flags.
The merger aligns on `(questionId, answerId)`, requires identical document sets,
rejects duplicates and invalid scores, and writes the existing headerless
seven-column submission format. It preserves OSODT's first three scores and
replaces only readability. It refuses to overwrite an input or an existing output.

The existing evaluator accepts the merged submission:

```bash
aeollm2-osodt evaluate \
  --predictions outputs/final/SRAD-OSODT.tsv \
  --labels data/official/aeollm-2-train/train_deepresearch.csv \
  --rubric-dir data/official/aeollm-2-train/rubric_dataset \
  --output-dir outputs/final/evaluation
```

This release enables new runs of the combined method. Hosted-model changes and
inference variability mean those runs need not exactly reproduce historical
scores. The committed formal metrics remain organizer-returned snapshots.

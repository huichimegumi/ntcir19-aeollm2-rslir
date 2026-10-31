# RSLIR at the NTCIR-19 AEOLLM-2 Task

**Code and Results for the NTCIR-19 AEOLLM-2 Participant Paper**

[中文说明](README_zh-CN.md)

This repository accompanies the RSLIR participant paper, *RSLIR at the NTCIR-19 AEOLLM-2 Task*. It provides implementations of One-Dimension-at-a-Time (ODT) and Order-Symmetrized One-Dimension-at-a-Time (OSODT) evaluation, experimental configurations, evaluation utilities, and rounded result snapshots for Tables 4–7 and 10.

Our formal submission used OSODT for comprehensiveness, insight, and instruction following, and Sequential Reference and Attention Decay (SRAD) for readability. **This repository implements ODT and OSODT, but not SRAD or AdapR.** Consequently, its code does not reproduce the complete SRAD+OSODT formal submission; Table 10 contains organizer-returned metrics rather than metrics regenerated from the archived submission.

ODT judges each of the four quality dimensions independently. OSODT judges each dimension with original and reversed rubric-criterion order, aligns scores by criterion identity, averages the aligned pairs, then applies rubric weights. The DOCX reader includes paragraphs and tables in document order.

The actual judging instructions and scoring anchors are written in Chinese, as used in the experiments. See [docs/prompt.md](docs/prompt.md) for an English overview; do not substitute translated instructions when attempting to reproduce the historical protocol. Task reports, rubrics, human labels, raw model responses, and historical per-document predictions are obtained or generated separately.

## Install

Python 3.10 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Optional tokenizer preflight for the Qwen example: `python -m pip install -e '.[tokenizer]'`. Download official inputs using [docs/data.md](docs/data.md) before inference. All commands below run from this repository root.

## Run the paper method with DeepSeek-v4-Flash

The paper configuration is [configs/deepseek-v4-flash-paper.yaml](configs/deepseek-v4-flash-paper.yaml): temperature 0, 2,048 maximum output tokens, thinking disabled, and four concurrent requests. Set a valid service key in the environment:

```bash
export DEEPSEEK_API_KEY='your-key'
aeollm2-osodt run \
  --method osodt \
  --config configs/deepseek-v4-flash-paper.yaml \
  --report-root data/official/aeollm-2-train/reports \
  --rubric-dir data/official/aeollm-2-train/rubric_dataset \
  --output-dir outputs/deepseek-v4-flash-osodt
```

The model ID is a hosted provider name, so a later run may use a different provider-side model revision. Outputs are a new run, not a replay of the paper's historical inference.

For an original-order ODT run, change `--method osodt` to `--method odt --order original` and choose another output directory. Use `--order reverse` for reverse-order ODT.

## Run with local vLLM

Install vLLM separately in a compatible GPU environment, then launch an OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --api-key local-example-key
```

In another shell with this repository installed:

```bash
export VLLM_API_KEY=local-example-key
aeollm2-osodt run \
  --method osodt \
  --config configs/qwen3-8b-vllm.yaml \
  --report-root data/official/aeollm-2-train/reports \
  --rubric-dir data/official/aeollm-2-train/rubric_dataset \
  --output-dir outputs/qwen3-8b-osodt
```

For another OpenAI-compatible provider, copy `configs/openai-compatible.example.yaml` to an ignored `configs/<name>.local.yaml` and edit its model, URL, key environment variable, and request fields.

## Evaluate and export

Public training labels are used only during evaluation, after inference:

```bash
aeollm2-osodt evaluate \
  --predictions outputs/deepseek-v4-flash-osodt/predictions/osodt.tsv \
  --labels data/official/aeollm-2-train/train_deepresearch.csv \
  --rubric-dir data/official/aeollm-2-train/rubric_dataset \
  --output-dir outputs/deepseek-v4-flash-osodt/evaluation
```

To make a headerless seven-column task submission file from a prediction TSV:

```bash
aeollm2-osodt export-submission \
  --predictions outputs/deepseek-v4-flash-osodt/predictions/osodt.tsv \
  --output outputs/deepseek-v4-flash-osodt/submission.tsv
```

Check the organizer's current submission instructions before submitting. The `run` command writes `manifest.json`, raw request responses, criterion-level original/reverse/symmetric scores, and final predictions under `outputs/`.

## Paper results and local checks

[results/paper/README.md](results/paper/README.md) maps each CSV to Tables 4–7 and 10 and explains its source and limits. These are rounded result snapshots. The formal Table 10 values came from organizer evaluation against hidden reserved-set labels.

```bash
python scripts/check_paper_materials.py
pytest -q
```

The first command checks CSV structure and arithmetic. It does not regenerate the historical judge outputs or the paper metrics from model inference. The tests use synthetic inputs and run locally; this repository has no GitHub Actions workflow.

## Citation and license

See [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE). Official task data are available separately from [THUIR/AEOLLM](https://huggingface.co/datasets/THUIR/AEOLLM).

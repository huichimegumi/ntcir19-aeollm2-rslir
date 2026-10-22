# Official AEOLLM-2 inputs

Obtain task data from [THUIR/AEOLLM](https://huggingface.co/datasets/THUIR/AEOLLM). The dataset card links the [official DOCX report folder](https://drive.google.com/drive/folders/1ujR5Gj889Y8RbK2eBmA-fikBQ1qcjXDe?usp=sharing). Data remain under ignored `data/`; model outputs remain under ignored `outputs/`.

## Download a fixed dataset revision

Install the [Hugging Face `hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then run from the repository root:

```bash
hf download THUIR/AEOLLM \
  --type dataset \
  --revision ea97403042b2ed991586601a9e0716e3a2d02191 \
  --include 'aeollm-2-train/train_deepresearch.csv' \
  --include 'aeollm-2-train/rubric_dataset/*.json' \
  --local-dir data/official
```

The commit above was the published dataset revision inspected for this repository. The train CSV provides 160 labelled reports across 10 questions; it does **not** contain the DOCX report files. The CLI downloads only the selected paths and preserves their directory structure. A later dataset revision can be used, but record its commit ID with your run.

## Obtain and arrange the DOCX reports

Download the AEOLLM-2 train DOCX reports through the folder linked on the dataset card. Place each file under the matching `ReportN` directory using its official question ID and `Doc_###` answer ID. Do not rename IDs. Example:

```text
data/official/aeollm-2-train/
├── train_deepresearch.csv
├── rubric_dataset/
│   ├── criterion1.json
│   ├── criterion2.json
│   └── ...
└── reports/
    ├── Report1/
    │   ├── Doc_001.docx
    │   └── ...
    ├── Report2/
    │   └── ...
    └── ...
```

If your download already contains `ReportN` folders, copy their train split into `data/official/aeollm-2-train/reports/`. The reader discovers immediate `ReportN/*.docx` files. Check the count and names before a full train run:

```bash
find data/official/aeollm-2-train/reports -type f -name 'Doc_*.docx' | wc -l
```

For the complete labelled train split, the expected count is 160. Match `ReportN` to `questionId=N`, and each DOCX stem to `answerId` in the CSV. A small subset is also valid for an inference smoke test; evaluating against the full train CSV requires identical document IDs on both sides.

## CLI path map

| Argument | Input | Example |
| --- | --- | --- |
| `--report-root` | Parent of `ReportN/Doc_*.docx` folders | `data/official/aeollm-2-train/reports` |
| `--rubric-dir` | Directory of official `criterionN.json` files | `data/official/aeollm-2-train/rubric_dataset` |
| `--labels` | Training labels, used only by `evaluate` | `data/official/aeollm-2-train/train_deepresearch.csv` |
| `--config` | Model service and generation settings | `configs/deepseek-v4-flash-paper.yaml` |
| `--output-dir` | New run and evaluation artifacts | `outputs/deepseek-v4-flash-osodt` |

`run` needs reports and rubrics, and never reads labels. `evaluate` needs prediction TSV, labels, and rubrics. The report and rubric files must belong to the same split and question IDs. The held-out formal Table 9 labels are organizer-side and are not distributed here.

# Paper result snapshots

These CSV files record rounded values from the RSLIR NTCIR-19 AEOLLM-2 participant paper. They are table-level materials, not raw judge responses or per-document predictions.

| File | Paper table | Source and interpretation |
| --- | --- | --- |
| `table4_dimension_isolation.csv` | Table 4 | Training-set Joint versus original-order ODT results for four local judges; source: controlled E9-A1 experiment metrics. |
| `table5_order_sensitivity.csv` | Table 5 | Same-order repeat and criterion-order variation for six local judges; source: E9 full-data noise summary. |
| `table6_order_symmetrization_accuracy.csv` | Table 6 | Fixed-order means and two OSODT pairings for six local judges; source: E9-A2 and E9-A3 order-control metric summaries. |
| `table7_deepseek_osodt.csv` | Table 7 | Original, reverse, and OSODT aggregate metrics for DeepSeek-v4-Flash and DeepSeek-v4-Pro; source: historical table-aware DeepSeek experiment aggregates. |
| `formal/official_metrics.csv` | Table 10 | Baseline, RSLIR SRAD+OSODT, and difference rows; source: organizer-returned reserved-set evaluation as reported in the paper. |

Tables 4–7 concern the labelled train population of 10 questions, 160 documents, and 1,200 within-question pairs. Table 10 uses organizer-held reserved-set labels. The table values are rounded to paper display precision. The original per-document predictions, criterion-level outputs, and training labels are not bundled in this repository, so these snapshots cannot by themselves regenerate the paper's historical metrics. New runs with official public data can be evaluated with `aeollm2-osodt evaluate`.

For Table 6, `fixed_original_mean` and `fixed_reverse_mean` correspond to averaging two original-order or two reverse-order runs. `osodt_s1` pairs the first original and reverse runs; `osodt_s2` pairs the second pair. Table 7's OSODT row averages aligned criterion scores before rubric-weighted dimension aggregation.

Check the committed materials locally:

```bash
python scripts/check_paper_materials.py
```

This checks expected table shape and derived delta arithmetic. It is a snapshot integrity check, not historical inference replay or independent validation of the measured values.

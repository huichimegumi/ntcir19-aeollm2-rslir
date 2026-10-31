# Table 9 formal result

`official_metrics.csv` transcribes the paper's organizer-returned reserved-set metrics for the official DeepSeek-v3.2 baseline, the RSLIR DeepSeek-v4-Flash **SRAD+OSODT** submission, and their differences. OSODT was used for comprehensiveness, insight, and instruction following; SRAD was used for readability. The RSLIR weighted overall Spearman, Kendall, and Accuracy are 0.7071, 0.5592, and 0.7796.

The exact organizer-submitted prediction file and hidden gold labels are not included. These metrics cannot be locally recomputed from this repository. The code implements ODT/OSODT but not the SRAD readability component of the formal submission. With authorized reports and rubrics, the commands in the root README produce a **new ODT/OSODT run**, not a replay of the full formal SRAD+OSODT system.

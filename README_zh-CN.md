# RSLIR at the NTCIR-19 AEOLLM-2 Task

**NTCIR-19 AEOLLM-2 Participant Paper 的代码与结果**

[English](README.md)

本仓库对应 RSLIR 团队的论文 *RSLIR at the NTCIR-19 AEOLLM-2 Task*，提供 One-Dimension-at-a-Time (ODT)、Order-Symmetrized One-Dimension-at-a-Time (OSODT) 的实现、实验配置、评估工具，以及论文 Tables 4–7、10 的四舍五入数值快照。

正式提交在全面性、洞察深度和指令遵循三个维度使用 OSODT，在可读性维度使用 Sequential Reference and Attention Decay (SRAD)。**本仓库实现 ODT 和 OSODT，但不包含 SRAD 或 AdapR 的实现。** 因此，单凭当前代码无法完整复现正式提交的 SRAD+OSODT 系统；Table 10 是组织方返回的正式评测指标，而非本仓库重新计算的结果。

ODT 对四个评价维度分别调用模型。OSODT 对每个维度分别使用原顺序、逆顺序的 criterion；按 criterion 身份对齐两次分数并取平均，然后按官方 rubric 权重汇总。DOCX 读取器按文档顺序读取段落和表格。

实际实验使用的评审指令与评分锚点为中文。英文结构说明见 [docs/prompt.md](docs/prompt.md)；复现实验时不应直接用英文翻译替换原始 Prompt。官方报告、rubric、人工标签、原始模型响应和逐文档预测均未随仓库打包。

## 安装与数据

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Qwen 示例若启用 tokenizer 上下文预检，还需 `python -m pip install -e '.[tokenizer]'`。官方数据的固定版本下载命令、DOCX 来源和目录结构见 [docs/data.md](docs/data.md)。下面命令均在仓库根目录执行。

## 运行论文方法

论文使用 DeepSeek-v4-Flash、temperature 0、最多 2048 输出 token、关闭 thinking、4 路并发；配置文件为 `configs/deepseek-v4-flash-paper.yaml`：

```bash
export DEEPSEEK_API_KEY='your-key'
aeollm2-osodt run \
  --method osodt \
  --config configs/deepseek-v4-flash-paper.yaml \
  --report-root data/official/aeollm-2-train/reports \
  --rubric-dir data/official/aeollm-2-train/rubric_dataset \
  --output-dir outputs/deepseek-v4-flash-osodt
```

若运行 ODT，将 `--method osodt` 改为 `--method odt --order original`，并使用新的输出目录。逆序 ODT 使用 `--order reverse`。托管模型名称不对应固定的服务端修订版本，因此新运行结果是新的推理，不保证与论文历史结果逐字节一致。

本地 vLLM 示例：

```bash
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b \
  --max-model-len 32768 --dtype bfloat16 --api-key local-example-key
```

在另一终端运行：

```bash
export VLLM_API_KEY=local-example-key
aeollm2-osodt run \
  --method osodt \
  --config configs/qwen3-8b-vllm.yaml \
  --report-root data/official/aeollm-2-train/reports \
  --rubric-dir data/official/aeollm-2-train/rubric_dataset \
  --output-dir outputs/qwen3-8b-osodt
```

其他 OpenAI 兼容服务可复制 `configs/openai-compatible.example.yaml` 为受忽略的 `configs/<name>.local.yaml`，修改模型、URL、密钥环境变量和请求参数。

## 评估与导出

训练集人工标签只在推理结束后由 `evaluate` 读取：

```bash
aeollm2-osodt evaluate \
  --predictions outputs/deepseek-v4-flash-osodt/predictions/osodt.tsv \
  --labels data/official/aeollm-2-train/train_deepresearch.csv \
  --rubric-dir data/official/aeollm-2-train/rubric_dataset \
  --output-dir outputs/deepseek-v4-flash-osodt/evaluation

aeollm2-osodt export-submission \
  --predictions outputs/deepseek-v4-flash-osodt/predictions/osodt.tsv \
  --output outputs/deepseek-v4-flash-osodt/submission.tsv
```

提交前仍须核对组织方当期要求。`run` 的输出包含运行 manifest、原始响应、criterion 分数和最终预测，保存在受忽略的 `outputs/`。

## 论文数值与本地检查

[results/paper/README.md](results/paper/README.md) 列出各表来源和复现边界。Table 10 是组织方用隐藏的保留集标签评估后返回的指标快照。

```bash
python scripts/check_paper_materials.py
pytest -q
```

第一条命令只检查论文 CSV 的结构与算术关系，不会重新执行历史推理或重算论文历史指标。测试使用合成输入，在本地执行；本仓库没有 GitHub Actions。

引用和许可见 [CITATION.cff](CITATION.cff)、[LICENSE](LICENSE)。

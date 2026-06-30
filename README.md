# AskMed

AskMed 当前包含两个主要流程：

1. 从 MedDG 对话合成医学事实抽取器训练数据。
2. 使用 Qwen3-4B-Instruct-2507 或 Qwen3-0.6B 对抽取器进行 LoRA 微调、验证和测试。

项目中的路径都以 `AskMed/` 为根目录解析，可以整体复制到 Linux 服务器。

## 微调环境

训练依赖 Conda 环境中的 LLaMA-Factory CLI：

```bash
llamafactory-cli version
```

目标环境版本为 `llamafactory 0.9.5.dev0`。`environment.yml` 记录 AskMed 直接使用的补充依赖；PyTorch、CUDA、Transformers、PEFT、TRL 和 LLaMA-Factory 应由服务器现有训练环境提供。

默认硬件假设：

- Linux
- 单张约 24GB NVIDIA GPU
- CUDA 和 BF16 可用

## 数据

LLaMA-Factory 数据注册文件：

```text
data/dataset_info.json
```

当前数据：

```text
data/synthetic_extractor/MedDG_extractor_15k_train_alpaca.jsonl
data/synthetic_extractor/MedDG_extractor_15k_valid_alpaca.jsonl
data/synthetic_extractor/MedDG_extractor_15k_test_alpaca.jsonl
```

对应样本数：

```text
train: 13508
valid:   673
test:    776
```

## 统一微调入口

先激活服务器上的训练环境：

```bash
conda activate AskMed
```

检查环境与数据：

```bash
bash scripts/finetuning/run_extractor.sh check
```

通过 ModelScope 下载模型：

```bash
bash scripts/finetuning/run_extractor.sh download
```

训练并在验证集上评估：

```bash
bash scripts/finetuning/run_extractor.sh train
```

在测试集上生成预测并计算结构化指标：

```bash
bash scripts/finetuning/run_extractor.sh test
```

执行完整流程：

```bash
bash scripts/finetuning/run_extractor.sh all
```

## Qwen3-0.6B 低显存流程

当 GPU 只剩约 10GB 可用显存时，可以先用 Qwen3-0.6B 跑通相同的训练、验证和测试流程。它使用独立的模型和输出目录，不会覆盖 4B 结果。

环境与数据检查：

```bash
bash scripts/finetuning/run_extractor_0_6b.sh check
```

通过 ModelScope 下载到 `models/Qwen3-0.6B/`：

```bash
bash scripts/finetuning/run_extractor_0_6b.sh download
```

建议先进行 200 条、20 step 的 smoke test：

```bash
bash scripts/finetuning/run_extractor_0_6b.sh smoke
```

完整训练与验证：

```bash
bash scripts/finetuning/run_extractor_0_6b.sh train
```

测试与结构化评估：

```bash
bash scripts/finetuning/run_extractor_0_6b.sh test
```

完整流程：

```bash
bash scripts/finetuning/run_extractor_0_6b.sh all
```

0.6B 默认配置：

```text
template: qwen3_nothink
LoRA rank: 16
cutoff_len: 4096
effective batch size: 16
epochs: 3
train_on_prompt: false
packing: false
```

输出目录：

```text
outputs/extractor_qwen3_0_6b/smoke/
outputs/extractor_qwen3_0_6b/train/
outputs/extractor_qwen3_0_6b/test/
```

默认 GPU 是 `0`，可以覆盖：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/finetuning/run_extractor.sh train
```

从 checkpoint 恢复：

```bash
RESUME_FROM_CHECKPOINT=outputs/extractor_qwen3_4b/train/checkpoint-1000 \
  bash scripts/finetuning/run_extractor.sh train
```

## 模型与输出

ModelScope 模型目录：

```text
models/Qwen3-4B-Instruct-2507/
```

训练输出：

```text
outputs/extractor_qwen3_4b/train/
outputs/extractor_qwen3_4b/train.log
```

测试输出：

```text
outputs/extractor_qwen3_4b/test/generated_predictions.jsonl
outputs/extractor_qwen3_4b/test/structured_metrics.json
outputs/extractor_qwen3_4b/test.log
outputs/extractor_qwen3_4b/metrics.log
```

训练和测试进度使用 LLaMA-Factory/Hugging Face 原生 tqdm。训练每 10 step 输出 loss、学习率、epoch 和累计 token 数。

## 训练配置

训练配置：

```text
configs/finetuning/extractor_qwen3_4b_lora.yaml
```

关键参数：

```text
model: Qwen3-4B-Instruct-2507
template: qwen3_nothink
LoRA rank: 16
cutoff_len: 8192
effective batch size: 16
epochs: 3
train_on_prompt: false
packing: false
```

`train_on_prompt: false` 表示模型读取完整输入，但 loss 只计算目标 JSON。`packing: false` 表示不同患者样本不会拼接到同一个训练序列中。

如果 24GB 显存出现 OOM：

1. 服务器已安装 FlashAttention 2 时，将 `flash_attn` 改为 `fa2`。
2. 否则将 `cutoff_len` 降为 `6144`。
3. 最后再降为 `4096`。

## 测试指标

`evaluate_predictions.py` 计算：

- JSON 可解析率
- facts schema 合法率
- 样本级规范化 exact match
- fact 级 micro precision、recall、F1
- 空 facts 样本准确率
- evidence 可追溯率
- 按 type 和 status 的统计

事实列表比较忽略列表顺序，但不会忽略字段内容差异。

## 数据合成

数据合成脚本位于：

```text
scripts/extractor_synthesis/
```

详细用法见：

```text
scripts/extractor_synthesis/README.md
```

### API 调用示例

DeepSeek / OpenAI-compatible：

```bash
python scripts/extractor_synthesis/run_extractor_pipeline.py \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "你的key" \
  --model deepseek-v4-flash \
  --run-name MedDG_extractor_test \
  --max-user-turns 20
```

Anthropic 官方 Messages API：

```bash
python scripts/extractor_synthesis/run_extractor_pipeline.py \
  --api-format anthropic \
  --base-url https://api.anthropic.com \
  --api-key "你的key" \
  --model claude-3-5-sonnet-20241022 \
  --run-name MedDG_extractor_claude_test \
  --max-user-turns 20
```

API key 只通过 `--api-key` 参数传入；主流程日志会自动脱敏。

## Terminology normalization

AskMed supports an optional lightweight terminology layer. Put legally obtained source files under:

```text
data/terminology/sources/
```

Build a local SQLite index:

```bash
python scripts/terminology/import_terminology.py \
  --icd-file data/terminology/sources/icd10_cn.csv \
  --loinc-table data/terminology/sources/LoincTable/Loinc.csv \
  --loinc-zh data/terminology/sources/AccessoryFiles/LinguisticVariants/zhCN5LinguisticVariant.csv \
  --output data/terminology/terminology.sqlite
```

Create terminology-derived extractor data without overwriting the original 15k files:

```bash
python scripts/terminology/normalize_extractor_dataset.py \
  --input data/synthetic_extractor/MedDG_extractor_15k_validated.jsonl \
  --terminology-db data/terminology/terminology.sqlite \
  --output-prefix data/synthetic_extractor/MedDG_extractor_15k_terminology
```

During live synthesis, pass the same database if you want rolling patient state to use terminology-aware merging:

```bash
python scripts/extractor_synthesis/run_extractor_pipeline.py \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "your-key" \
  --model deepseek-v4-flash \
  --run-name terminology_test \
  --max-user-turns 20 \
  --terminology-db data/terminology/terminology.sqlite
```

Training JSON keeps the original fact shape and only changes `normalized_name`; `standard_code` and `terminology` remain `null`. Runtime state can keep `standard_code` and `terminology` internally, while prompt-facing state remains natural-language only.

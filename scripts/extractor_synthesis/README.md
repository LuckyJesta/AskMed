# AskMed 医学事实抽取器数据合成脚本

本目录包含从 MedDG 对话构造医学事实抽取器 SFT 数据的脚本。主流程按对话顺序滚动合成：每个患者轮次生成 `facts` 后，立即更新问诊状态，并把状态用于后续轮次。

## 推荐入口

```bash
python scripts/extractor_synthesis/run_extractor_pipeline.py \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "你的key" \
  --model deepseek-v4-flash \
  --work-dir data/synthetic_extractor \
  --run-name train_1k \
  --max-user-turns 1000
```

主入口会按顺序执行：

```text
1. synthesize_extractor_dialogues.py
2. validate_extractor_data.py
3. convert_to_alpaca.py 或 split_extractor_dataset.py
```

所有默认路径都基于 `AskMed/` 项目根目录解析，项目整体迁移到服务器后仍可运行。

## API 参数

合成阶段支持两种教师模型 API 格式：

```text
--api-format openai
--api-format anthropic
```

通用参数：

```text
--api-key KEY
--base-url URL
--model MODEL_NAME
--max-output-tokens 4096
```

API key 必须通过 `--api-key` 显式传入。脚本不会读取 `DEEPSEEK_API_KEY`、`ANTHROPIC_API_KEY` 或其他环境变量。主入口日志会将 key 脱敏为 `***REDACTED***`。

DeepSeek / OpenAI-compatible 示例：

```bash
python scripts/extractor_synthesis/run_extractor_pipeline.py \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "你的key" \
  --model deepseek-v4-flash \
  --run-name MedDG_extractor_test \
  --max-user-turns 20
```

Anthropic 官方 Messages API 示例：

```bash
python scripts/extractor_synthesis/run_extractor_pipeline.py \
  --api-format anthropic \
  --base-url https://api.anthropic.com \
  --api-key "你的key" \
  --model claude-3-5-sonnet-20241022 \
  --run-name MedDG_extractor_claude_test \
  --max-user-turns 20
```

直接运行合成脚本：

```bash
python scripts/extractor_synthesis/synthesize_extractor_dialogues.py \
  --api-format anthropic \
  --base-url https://api.anthropic.com \
  --api-key "你的key" \
  --model claude-3-5-sonnet-20241022 \
  --output data/synthetic_extractor/debug_synthesized.jsonl \
  --failed data/synthetic_extractor/debug_synthesis_failed.jsonl \
  --final-states data/synthetic_extractor/debug_final_states.jsonl \
  --checkpoint data/synthetic_extractor/debug_checkpoints.jsonl \
  --max-user-turns 20
```

## 数据集分割

默认不分割数据集，最终生成一个 `{run_name}_alpaca.jsonl`。如果传入 `--split-ratios TRAIN VALID TEST`，第 3 步会按完整对话切分并生成 train/valid/test 文件：

```bash
python scripts/extractor_synthesis/run_extractor_pipeline.py \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "你的key" \
  --model deepseek-v4-flash \
  --work-dir data/synthetic_extractor \
  --run-name train_main \
  --max-user-turns 10000 \
  --split-ratios 0.9 0.05 0.05 \
  --split-seed 42
```

## 输出文件

以 `--run-name train_1k` 为例，输出文件为：

```text
data/synthetic_extractor/train_1k_synthesized.jsonl
data/synthetic_extractor/train_1k_synthesis_failed.jsonl
data/synthetic_extractor/train_1k_final_states.jsonl
data/synthetic_extractor/train_1k_checkpoints.jsonl
data/synthetic_extractor/train_1k_validated.jsonl
data/synthetic_extractor/train_1k_validation_failed.jsonl
data/synthetic_extractor/train_1k_report.json
data/synthetic_extractor/train_1k_alpaca.jsonl
data/synthetic_extractor/train_1k_pipeline.log
```

启用分割时额外输出：

```text
data/synthetic_extractor/train_1k_train_validated.jsonl
data/synthetic_extractor/train_1k_valid_validated.jsonl
data/synthetic_extractor/train_1k_test_validated.jsonl
data/synthetic_extractor/train_1k_train_alpaca.jsonl
data/synthetic_extractor/train_1k_valid_alpaca.jsonl
data/synthetic_extractor/train_1k_test_alpaca.jsonl
data/synthetic_extractor/train_1k_split_report.json
```

## 断点续跑

继续同一任务时保持 `--source`、`--work-dir`、`--run-name` 不变即可。脚本会读取同名 checkpoint，从上次停止位置继续，并恢复该对话已经形成的 `patient_state`。

`--max-user-turns` 表示本次运行最多新处理多少个患者轮次，不是完整多轮对话数，也不是累计目标数。若设置值超过 source 剩余患者轮次数，脚本会跑完整个 source 后自然结束。

checkpoint 记录示例：

```json
{
  "dialogue_id": "train/dialog1",
  "completed_turn_id": 18,
  "patient_state": {},
  "finished": false
}
```

如果在一条多轮对话中间中断，续跑时会跳过 `completed_turn_id` 及之前的患者轮次，并用 checkpoint 中的 `patient_state` 作为下一轮输入。

## 说明

- `meddg_weak_labels` 只用于合成阶段参考，不进入最终 Alpaca 输入。
- `patient_state_before_turn` 会进入合成输入和最终 Alpaca 输入。
- `patient_state_after_turn` 用于后续问诊器数据构造和状态分析。
- 属性类事实会优先合并到目标症状/疾病，例如“两三天了”会更新到“腹痛”的病程属性。
- 第一版不做标准医学编码，`standard_code` 和 `terminology` 固定为 `null`。
- `old/` 中保留早期脚本，仅作备份，不再作为主流程使用。

# Optional terminology database

If `data/terminology/terminology.sqlite` has been built with `scripts/terminology/import_terminology.py`, pass it to the synthesis pipeline:

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

The runtime patient state uses terminology-aware merging. The synthesized training output still keeps `standard_code` and `terminology` as `null`, so the extractor model is not trained to predict medical codes.

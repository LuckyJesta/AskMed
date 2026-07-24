# Extractor synthesis

主入口按完整对话执行：

```text
教师模型逐轮生成 JSON
→ FactPipeline 原始校验
→ 可选术语标准化
→ 状态归并
→ 对话级 validation
→ Alpaca 或对话级 split
```

## Basic run

```bash
python -m scripts.extractor.synthesis.run_extractor_pipeline \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "your-key" \
  --model deepseek-v4-flash \
  --source data/MedDG_clean.jsonl \
  --work-dir data/synthetic_extractor \
  --run-name extractor_v3 \
  --max-user-turns 1000 \
  --workers 3
```

`--workers` controls complete-dialogue concurrency and defaults to `3`. Dialogue turns remain sequential. The selected source prefix is fixed before scheduling, so checkpoint resume does not add another `--max-user-turns` batch. Completed dialogues are consolidated by source order before validation.

During synthesis, complete dialogue results are atomically staged under `_dialogue_work/`. The directory is removed after a successful rebuild and retained after interruption for checkpoint recovery.

支持 `openai` 和 `anthropic` API 格式。API key 只通过参数传入，日志中的命令会自动脱敏。

## Standardization switch

```bash
python -m scripts.extractor.synthesis.run_extractor_pipeline \
  ... \
  --terminology-db data/terminology/terminology.sqlite
```

- 不传 `--terminology-db`：开放事实按名称和 target 归并。
- 传入数据库：每轮 fact 立即标准化，状态优先按 `terminology + standard_code` 归并。
- 未匹配术语是正常结果，不会写入失败账本。
- 主 `parsed_output` 保存运行时编码；Alpaca 自动清空编码。

## Split

默认只生成一个 Alpaca 文件。显式传比例时按 `dialogue_id` 切分：

```bash
python -m scripts.extractor.synthesis.run_extractor_pipeline \
  ... \
  --split-ratios 0.9 0.05 0.05 \
  --split-seed 42
```

## Checkpoint and repair

继续任务时保持 `source/work-dir/run-name` 不变。`--max-dialogues` 表示 source 前缀内允许处理的对话总数；`--max-user-turns` 采用完整对话优先，最终轮次数可略高于上限。

任意一轮 API、解析、原始校验、标准化、状态或投影失败，整条对话进入：

```text
{run_name}_failed_dialogues.jsonl
```

修复命令：

```bash
python -m scripts.extractor.synthesis.run_extractor_pipeline \
  ... \
  --run-name extractor_v3 \
  --repair-failed-dialogues
```

修复阶段使用与正常合成相同的稳定 source 前缀。传入 `--max-dialogues` 或
`--max-user-turns` 时，只重试该范围内的失败对话，进度条也只统计这些对话；
范围外的账本记录保持不变且不会消耗 API 请求。

严格校验前，一维标量 attribute 列表会按原顺序去重，并使用中文分号 `；`
合并为字符串；包含嵌套对象或嵌套列表的值仍判定为非法。

教师输出发生 `raw_validation_error` 时，当前失败轮次会收到具体校验错误和
上一份输出，并在 `--max-retries` 范围内重新生成。只有反馈重试仍失败时，
整条对话才会保留在失败账本中。

修复会重新合成完整对话并 rebuild；再次失败只更新同一账本记录。

## Conservative cleaning

Keep the source run unchanged and create a cleaned derived run with replayed
dialogue state:

```bash
python -m scripts.extractor.synthesis.clean_extractor_dataset \
  --input data/synthetic_extractor/MedDG_extractor_prompt_v3_30k/MedDG_extractor_prompt_v3_30k_validated.jsonl \
  --output-prefix data/synthetic_extractor/MedDG_extractor_prompt_v3_1_30k/MedDG_extractor_prompt_v3_1_30k \
  --terminology-db data/terminology/terminology.sqlite
```

The cleaner removes Unicode format characters, normalizes only conservative
attribute aliases, merges examination `value` into `result`, removes
high-confidence short hedged facts, and writes remaining hedged expressions to
`*_cleaning_candidates.jsonl` for review. It rebuilds checkpoints and final
states instead of editing Alpaca text in place. Run validation and Alpaca
conversion on the derived synthesized file afterward.

## Outputs

```text
data/synthetic_extractor/{run_name}/
├─ {run_name}_synthesized.jsonl
├─ {run_name}_validated.jsonl
├─ {run_name}_alpaca.jsonl
├─ {run_name}_failed_dialogues.jsonl   # 仅非空时存在
├─ {run_name}_final_states.jsonl
├─ {run_name}_checkpoints.jsonl
├─ {run_name}_report.json
└─ {run_name}_pipeline.log
```

最终 Alpaca input 不包含 MedDG 弱标签、`dialogue_id`、`turn_id` 或内部编码信息。

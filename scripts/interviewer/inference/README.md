# AskMed 双模型调用

分别启动提取器和问诊器 API：

```bash
API_PORT=8000 llamafactory-cli api configs/extractor/extractor_qwen3_4b_api.yaml
API_PORT=8001 llamafactory-cli api configs/interviewer/interviewer_qwen3_4b_api.yaml
```

运行共享会话编排器：

```bash
python -m scripts.interviewer.inference.run_preconsultation \
  --extractor-base-url http://127.0.0.1:8000/v1 \
  --extractor-model Qwen3-4B-Instruct-2507 \
  --interviewer-base-url http://127.0.0.1:8001/v1 \
  --interviewer-model Qwen3-4B-Instruct-2507 \
  --terminology-db data/terminology/terminology.sqlite \
  --input requests.jsonl \
  --output responses.jsonl
```

输入：

```json
{"session_id":"demo-1","patient_utterance":"肚脐周围疼了三天"}
```

提取器只更新 `medical_state`；问诊器读取其精简投影，只更新 `dialogue_control.asked_targets` 和结束状态。

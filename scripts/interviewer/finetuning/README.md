# AskMed 问诊器微调

问诊器使用 `Qwen/Qwen3-4B-Instruct-2507` 和独立 LoRA：

```bash
bash scripts/interviewer/finetuning/run_interviewer.sh check
bash scripts/interviewer/finetuning/run_interviewer.sh download
CUDA_VISIBLE_DEVICES=0 bash scripts/interviewer/finetuning/run_interviewer.sh train
CUDA_VISIBLE_DEVICES=0 bash scripts/interviewer/finetuning/run_interviewer.sh test
```

测试输出除 LLaMA-Factory 文本指标外，还生成 `structured_metrics.json`，包括动作准确率、目标 F1、结束判断、重复询问、安全性和下一状态目标增益。

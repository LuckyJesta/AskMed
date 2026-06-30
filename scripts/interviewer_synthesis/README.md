# AskMed 问诊器数据合成脚本

这个目录预留给后续“问诊器”训练数据合成流程。

计划中的输入来源：

- 提取器合成输出中的 `patient_state_after_turn`
- 原始对话中的下一轮医生问话
- 已问过的问题目标
- 少量最近原始上下文

计划中的训练目标：

- `next_question_target`
- `next_question`
- `reason`
- `should_end`

当前目录暂不包含可执行脚本。

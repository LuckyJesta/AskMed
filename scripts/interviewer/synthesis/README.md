# AskMed 问诊器数据合成

问诊器数据以 extractor v3.1 的完整对话、逐轮 `patient_state_after_turn` 和 MedDG 原对话为基础。

## 流程

```text
MedDG 原对话 + v3.1 逐轮 facts + v3.1 split
→ 合并连续同角色消息并保守拆分二次问诊
→ 在每个子会话内从空状态重放 facts
→ 教师只解析原医生块中的安全单目标问题
→ 程序使用下一患者状态验证问题是否被实际回答
→ 对话级校验、checkpoint 和失败账本
→ 继承原 split 转换为 Alpaca
```

v2.2 教师请求中不包含下一患者回复、下一患者状态或状态差量候选。教师输出的问题必须带有可追溯到原医生回复的连续 `source_text`；程序随后通过精确名称、保守词面关联、泛化类型或明确短回答选择第一个可证明被回答的问题，不再使用无条件同类型唯一增量。属性问题必须绑定当前状态中的唯一真实实体，无法解析时以 `TARGET_NOT_RESOLVED` 跳过。已有 attribute 的新取值不再作为新问题。

attribute 使用英文规范键；教师返回的病程、部位、次数等中文别名会在本地转换为 `duration/body_part/frequency` 等规范键。明确的“感谢/告别 + 医生闭合回复”之后，只有显式新话题或带有独立新主诉和新发病程表达时才拆成 `#sessionN` 子会话。药物补充、检查更正和原问题续述不拆分，只记录模糊边界审计警告。子会话继承父对话 split，checkpoint 和失败账本仍以父对话为原子单位。

教师候选、原医生回复、下一患者回复和下一状态只保留在审计数据中，不进入最终 Alpaca input。默认运行前缀为 `MedDG_interviewer_v2_2_from_v3_1_30k`，并生成 synthesis、validation、conversion 和 pipeline 四份统计报告。

完整合成：

```bash
python -m scripts.interviewer.synthesis.run_interviewer_pipeline \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "your-key" \
  --model deepseek-chat \
  --workers 3
```

先测试 10 条完整对话：

```bash
python -m scripts.interviewer.synthesis.run_interviewer_pipeline \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "your-key" \
  --model deepseek-chat \
  --max-dialogues 10
```

整对话修复失败账本：

```bash
python -m scripts.interviewer.synthesis.run_interviewer_pipeline \
  --api-format openai \
  --base-url https://api.deepseek.com \
  --api-key "your-key" \
  --model deepseek-chat \
  --repair-failed-dialogues
```

API key 仅通过命令行传入，入口打印命令时会隐藏真实值。

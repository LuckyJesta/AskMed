# Interviewer pipeline

Shared interviewer contracts:

- `project_interviewer_state()` removes terminology codes, aliases, evidence and internal turn IDs.
- `validate_decision()` enforces the `ask/end` schema, one-question output, safety and target deduplication.
- `build_interviewer_input()` is shared by Alpaca conversion and runtime inference.

# 3.2 frozen protocol

- baseline: `7a18146` on `feat/domain-task-suite`
- model: `deepseek-flash`, `zero_shot`, `reasoning_effort=low`, `max_tokens=32768`, `temperature=0`, `n=1`
- hardware: `ascend910b4`, backend `ascendc`, build_backend `cmake`
- tasks: cann/nn/rmsnorm, cann/nn/swiglu, cann/nn/softmax, cann/nn/gather, cann/nn/apply_rotary_pos_emb
- run ids: `s32_<task>` (unique; history never overwritten)
- one candidate per task, no automatic retry, generated code never edited
- failures (truncation, compile, correctness) are recorded as results

#!/usr/bin/env bash
# Run the frozen 3.2 generations once per task, in order, into unique run ids.
set -u
cd "$(dirname "$0")/../.." || exit 1
set -a; . /Users/anangxu/文件/论文/.akb_env; set +a
PY=/Users/anangxu/文件/论文/local_pass10_20260909/.venv/bin/python
LOG=experiments/phase3_2
for task in rmsnorm swiglu softmax gather apply_rotary_pos_emb; do
  run="s32_${task}"
  if [ -d "runs/${run}" ] && find "runs/${run}" -name custom_op.asc | grep -q .; then
    echo "skip ${run}: already has a candidate" >> "$LOG/progress.log"; continue
  fi
  echo "start ${task} $(date -u +%H:%M:%S)" >> "$LOG/progress.log"
  "$PY" scripts/generate.py --task "cann/nn/${task}" --model deepseek-flash \
    --hardware ascend910b4 --run-name "$run" --n-samples 1 --prompt-mode zero_shot \
    --max-tokens 32768 --reasoning-effort low > "$LOG/gen_${task}.log" 2>&1
  echo "done ${task} exit=$? $(date -u +%H:%M:%S)" >> "$LOG/progress.log"
done
echo "all done $(date -u +%H:%M:%S)" >> "$LOG/progress.log"

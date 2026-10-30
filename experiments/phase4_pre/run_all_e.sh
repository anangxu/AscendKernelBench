#!/usr/bin/env bash
# Round 5: case prompts now include the structural skeleton.
set -u
cd "$(dirname "$0")/../.." || exit 1
set -a; . /Users/anangxu/文件/论文/.akb_env; set +a
PY=/Users/anangxu/文件/论文/local_pass10_20260909/.venv/bin/python
LOG=experiments/phase4_pre
for pair in "swiglu:cann/nn/swiglu" "softmax:cann/nn/softmax" "gather:cann/nn/gather" \
            "scatter:cann/auto/scatter" "linear_cross_entropy:cann/auto/linear_cross_entropy"; do
  name="${pair%%:*}"; task="${pair##*:}"
  run="s4e_${name}"
  find "runs/${run}" -name custom_op.asc 2>/dev/null | grep -q . && { echo "skip ${name}" >> "$LOG/progress_e.log"; continue; }
  echo "start ${name} $(date -u +%H:%M:%S)" >> "$LOG/progress_e.log"
  "$PY" scripts/generate.py --task "$task" --model deepseek-flash --hardware ascend910b4 \
    --run-name "$run" --n-samples 1 --prompt-mode zero_shot \
    --max-tokens 65536 --reasoning-effort low > "$LOG/gene_${name}.log" 2>&1
  echo "done ${name} exit=$? $(date -u +%H:%M:%S)" >> "$LOG/progress_e.log"
done
echo "all done $(date -u +%H:%M:%S)" >> "$LOG/progress_e.log"

#!/usr/bin/env bash
# Budgeted alternating contrast on the small-write-back manifestation.
#
# Runs _small_writeback_contrast.py for the original and fixed arms in
# alternating fresh processes, one round each, up to BUDGET rounds. The budget
# is fixed up front: waiting indefinitely for a rare failure is not a design.
#
# Usage: BUDGET=30 PY=/path/to/python bash _small_writeback_runner.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 2

CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
# shellcheck disable=SC1090
source "$CANN_SET_ENV" >/dev/null 2>&1

export PYASC_CACHE_DIR="${PYASC_CACHE_DIR:-$HERE/_small_cache}"
mkdir -p "$PYASC_CACHE_DIR"
export PYTHONDONTWRITEBYTECODE=1

PY="${PY:-python}"
BUDGET="${BUDGET:-30}"

for round in $(seq 1 "$BUDGET"); do
  for arm in original fixed; do
    timeout 600 "$PY" _small_writeback_contrast.py "$arm" "$round" 2>&1 \
      | grep -E "round=|Error|Traceback" \
      | sed "s/^/run=$(printf '%02d' "$round") /"
  done
done

echo "SMALL_AB_DONE budget=$BUDGET cache_dir=$PYASC_CACHE_DIR"

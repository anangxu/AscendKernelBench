#!/usr/bin/env bash
# Bounded A/B on the invocation path, one variable at a time.
#
# A = probe_s1_tque_copy_roundtrip.py (body run through _probe_common.run())
# B = _ab_entry_b.py                  (same kernel, launched directly)
#
# Everything else is pinned: same kernel module, input construction, shape,
# dtype, device, stream, synchronisation, sentinel, working directory and
# PYASC_CACHE_DIR. Each run is a fresh process, and the two entries alternate so
# they see the same device and cache state.
#
# Records per run: exit code, device error, sentinel-uncovered count, wrong
# element count, max absolute error. After a device fault every later run is
# tagged post_fault so contaminated runs stay out of the normal contrast.
#
# Usage: ROUNDS=20 PY=/path/to/python bash _ab_runner.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 2

CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
# shellcheck disable=SC1090
source "$CANN_SET_ENV" >/dev/null 2>&1

export PYASC_CACHE_DIR="${PYASC_CACHE_DIR:-$HERE/_ab_cache}"
mkdir -p "$PYASC_CACHE_DIR"
export PYTHONDONTWRITEBYTECODE=1

PY="${PY:-python}"
ROUNDS="${ROUNDS:-20}"

faults=0
post_fault=0

for round in $(seq 1 "$ROUNDS"); do
  for entry in A B; do
    if [ "$entry" = "A" ]; then
      file="probe_s1_tque_copy_roundtrip.py"
    else
      file="_ab_entry_b.py"
    fi

    output="$(timeout 600 "$PY" "$file" 2>&1)"
    code=$?
    metrics="$(printf '%s\n' "$output" | grep -o 'METRICS label=[AB].*' | head -1)"
    [ -n "$metrics" ] || metrics="METRICS label=$entry missing"
    if printf '%s' "$output" | grep -qE '507035|507033|EZ9999|vector core'; then
      device="fault"
      faults=$((faults + 1))
    else
      device="none"
    fi

    phase="clean"
    [ "$post_fault" -eq 1 ] && phase="post_fault"
    printf 'run=%02d entry=%s phase=%s exit=%d device=%s %s\n' \
      "$round" "$entry" "$phase" "$code" "$device" "$metrics"

    if [ "$device" = "fault" ]; then
      post_fault=1
      echo "  NOTE: device fault; later runs are tagged post_fault"
    fi
  done
done

echo "AB_DONE rounds=$ROUNDS faults=$faults cache_dir=$PYASC_CACHE_DIR"

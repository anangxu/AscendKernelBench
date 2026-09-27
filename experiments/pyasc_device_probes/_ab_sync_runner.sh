#!/usr/bin/env bash
# Bounded one-variable contrast: TQue synchronisation vs an explicit wait.
#
# plain and event run from _ab_sync_variant.py, which differs only in the
# MTE2_MTE3 flag pair between the inbound copy and the write-back. Each run is a
# fresh process, the two variants alternate, and the cache directory, working
# directory, input, shape, dtype, device, stream and sentinel are pinned.
#
# Usage: ROUNDS=60 PY=/path/to/python bash _ab_sync_runner.sh
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
ROUNDS="${ROUNDS:-60}"

faults=0
post_fault=0

for round in $(seq 1 "$ROUNDS"); do
  for variant in plain event; do
    output="$(timeout 600 "$PY" _ab_sync_variant.py "$variant" 2>&1)"
    code=$?
    metrics="$(printf '%s\n' "$output" | grep -o "METRICS label=$variant .*" | head -1)"
    [ -n "$metrics" ] || metrics="METRICS label=$variant missing"
    if printf '%s' "$output" | grep -qE '507035|507033|EZ9999|vector core'; then
      device="fault"
      faults=$((faults + 1))
    else
      device="none"
    fi

    phase="clean"
    [ "$post_fault" -eq 1 ] && phase="post_fault"
    printf 'run=%02d variant=%s phase=%s exit=%d device=%s %s\n' \
      "$round" "$variant" "$phase" "$code" "$device" "$metrics"

    if [ "$device" = "fault" ]; then
      post_fault=1
      echo "  NOTE: device fault; later runs are tagged post_fault"
    fi
  done
done

echo "SYNC_AB_DONE rounds=$ROUNDS faults=$faults cache_dir=$PYASC_CACHE_DIR"

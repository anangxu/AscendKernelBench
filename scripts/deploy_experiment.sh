#!/usr/bin/env bash
# Deploy the checkout to a device and run one experiment under a unique,
# evidence-preserving directory. History is never deleted.
#
# Usage: deploy_experiment.sh <host> <port> <key> <archive.tgz> <run-id> [cmd...]
set -euo pipefail

HOST=${1:?host}; PORT=${2:?port}; KEY=${3:?key}; ARCHIVE=${4:?archive}
RUN_ID=${5:?run-id}
shift 5 || true

BASE=/cache/akb
SRC="$BASE/src"                       # deploy target, replaced on purpose
EXP="$BASE/exp/$RUN_ID"               # per-run evidence, never deleted
REMOTE_KEY="$BASE/.key"

mkdir -p "$BASE" 2>/dev/null || true
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$PORT" -i "$KEY" "$ARCHIVE" "$HOST:$BASE/incoming.tgz"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PORT" -i "$KEY" "$HOST" "
set -euo pipefail
mkdir -p '$SRC' '$EXP'
# Deploy only into the source tree; the experiment directory is separate.
tar xzf '$BASE/incoming.tgz' -C '$SRC'
rm -f '$BASE/incoming.tgz'
# Every run keeps its own configure/build logs and copies of the shipped files.
echo 'run-id=$RUN_ID' > '$EXP/run.txt'
date -u +%Y-%m-%dT%H:%M:%SZ >> '$EXP/run.txt'
cd '$SRC'
set +e
$*
CODE=\$?
set -e
echo \$CODE > '$EXP/exit_code.txt'
# Preserve generated build evidence before anything can overwrite it.
find . -path '*/build/build.log' -o -path '*/build/configure.log' -o -name 'custom_op_launch_decls.h' \
  | while read -r f; do mkdir -p \"'$EXP'/evidence\$(dirname \"\$f\")\"; cp \"\$f\" \"'$EXP'/evidence\$(dirname \"\$f\")/\"; done
exit \$CODE
"

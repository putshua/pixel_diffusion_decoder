#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PID_PYTHON:-python}
RUN=${PID_RUN:?Set PID_RUN to a prepared run directory}
MODE=${1:-train}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$ROOT"
if [[ "$MODE" == audit ]]; then
 "$PYTHON" -m pid_video.preflight --run "$RUN" --audit-only
 "$PYTHON" -m pid_video.audit_vae --run "$RUN" 2>&1 | tee -a "$RUN/vae_audit.log"
elif [[ "$MODE" == train || "$MODE" == preflight ]]; then
 "$PYTHON" -m pid_video.preflight --run "$RUN"
 NPROC=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["world_size"])' "$RUN/config.json")
 END=$("$PYTHON" -c 'import json,sys,pathlib;c=json.load(open(sys.argv[1]));print((json.load(open(pathlib.Path(c["fork_from"])/"complete.json"))["step"] if c.get("fork_from") else 0)+2)' "$RUN/config.json")
 if [[ "$MODE" == train && ! -f "$RUN/latest.json" ]]; then
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$NPROC" --max-restarts=0 -m -- pid_video.train --run "$RUN" --stop-after "$END" 2>&1 | tee -a "$RUN/preflight.log"
 fi
 args=(--run "$RUN")
 [[ ! -f "$RUN/latest.json" ]] || args+=(--resume)
 [[ "$MODE" != preflight ]] || args+=(--stop-after "$END")
 "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$NPROC" --max-restarts=0 -m -- pid_video.train "${args[@]}" 2>&1 | tee -a "$RUN/train.log"
else
 echo 'Usage: scripts/run_pai.sh train|preflight|audit' >&2; exit 2
fi
# A custom local/cluster holder remains an explicitly configured external facility.
if [[ -n "${PID_HOLDER_GUARD:-}" ]]; then
 [[ -f "$PID_HOLDER_GUARD" ]] || { echo "Missing holder guard: $PID_HOLDER_GUARD" >&2; exit 3; }
 exec "$PYTHON" "$PID_HOLDER_GUARD" --ensure
fi

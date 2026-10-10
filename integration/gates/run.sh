#!/bin/sh
set -eu
export PATH=/Users/pedro/Developer/projects/.mainboard/envs/default/.pixi/envs/default/bin:$PATH
gate_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export MB_GATE_WORKTREE=$(CDPATH= cd -- "$gate_dir/../.." && pwd)
export MB_GATE_PROJECTS=/Users/pedro/Developer/projects
export MB_GATE_ROOT=$(mktemp -d /private/tmp/mb-gate.XXXXXXXX)
export MB_GATE_REPORT=${MB_GATE_REPORT:-/private/tmp/claude-501/-Users-pedro-Developer-projects/b52c03f0-d920-5f82-9aac-285c87707d12/scratchpad/codex/mb/M00}
case "$MB_GATE_REPORT" in
    /private/tmp/claude-501/-Users-pedro-Developer-projects/b52c03f0-d920-5f82-9aac-285c87707d12/scratchpad/codex/mb/*) ;;
    *) echo "M00: reports must be under scratchpad/codex/mb" >&2; exit 64 ;;
esac
mkdir -p "$MB_GATE_REPORT" "$MB_GATE_ROOT/cache" "$MB_GATE_ROOT/tmp"
export PYTHONPATH="$MB_GATE_WORKTREE/src:$gate_dir"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MB_GATE_BOOTSTRAP=1
export XDG_CACHE_HOME="$MB_GATE_ROOT/cache" PIXI_CACHE_DIR="$MB_GATE_ROOT/cache/pixi"
export UV_CACHE_DIR="$MB_GATE_ROOT/cache/uv" TMPDIR="$MB_GATE_ROOT/tmp"
cd "$MB_GATE_PROJECTS"
case "${1:-suite}" in
    suite) gate_limit=900 ;;
    target|inspect) gate_limit=120 ;;
    ruff|pyrefly|ty|mypy|mcmr) gate_limit=300 ;;
    *) echo "M00: unknown gate mode" >&2; exit 64 ;;
esac
exec mainboard proc timeout "$gate_limit" mainboard run --as-is env MB_GATE_BOOTSTRAP=0 \
    PYTHONPATH="$PYTHONPATH" python "$gate_dir/runner.py" "$@"

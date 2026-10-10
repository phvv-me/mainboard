#!/bin/sh
# usage: run.sh MODE [--report DIR] [--budget SECONDS] [NODEID...]
# MODE is suite, target (NODEIDs), inspect, ruff, pyrefly, ty, mypy or mcmr. Results go to DIR or
# else a new temporary folder; SECONDS bounds the whole run and defaults to the mode's limit.
set -eu
projects=${MB_GATE_PROJECTS:-/Users/pedro/Developer/projects}
bin=$projects/.mainboard/envs/default/.pixi/envs/default/bin
gate_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mode=${1:-suite}
[ $# -eq 0 ] || shift
report= budget=
while [ $# -gt 1 ]; do
    case "$1" in
        --report) report=$2 ;;
        --budget) budget=$2 ;;
        *) break ;;
    esac
    shift 2
done
case "$mode" in
    suite) limit=900 ;;
    target|inspect) limit=120 ;;
    ruff|pyrefly|ty|mypy|mcmr) limit=300 ;;
    *) echo "unknown gate mode: $mode" >&2; exit 64 ;;
esac
root=$(mktemp -d /tmp/mb-gate.XXXXXXXX)
report=${report:-$(mktemp -d /tmp/mb-gate-report.XXXXXXXX)}
mkdir -p "$report" "$root/tmp" "$root/cache"
export MB_GATE_PROJECTS=$projects MB_GATE_ROOT=$root MB_GATE_BUDGET=${budget:-$limit}
export MB_GATE_REPORT=$(CDPATH= cd -- "$report" && pwd -P)
export MB_GATE_WORKTREE=$(CDPATH= cd -- "$gate_dir/../.." && pwd)
echo "gate $mode: budget ${MB_GATE_BUDGET}s, report $MB_GATE_REPORT" >&2
export PATH=$bin:$PATH TMPDIR=$root/tmp XDG_CACHE_HOME=$root/cache
cd "$root"
status=0
# The environment's own interpreter and timeout wrapper: no compile, lock or install touches the
# live workspace, and the wrapper is not the code under test.
"$bin/mainboard" proc timeout "$MB_GATE_BUDGET" env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$MB_GATE_WORKTREE/src:$gate_dir/worker" \
    "$bin/python" "$gate_dir/runner.py" "$mode" "$@" || status=$?
rm -rf "$root" || true
exit "$status"

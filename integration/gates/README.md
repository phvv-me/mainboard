# M00 execution gate

Only test infrastructure changes here. Run from any directory; the launcher changes to
`/Users/pedro/Developer/projects` before `mainboard run` and finds the worktree from its own path.

```sh
export PATH=/Users/pedro/Developer/projects/.mainboard/envs/default/.pixi/envs/default/bin:$PATH
sh /Users/pedro/Developer/mb-M00/integration/gates/run.sh inspect
sh /Users/pedro/Developer/mb-M00/integration/gates/run.sh suite
sh /Users/pedro/Developer/mb-M00/integration/gates/run.sh target integration/gates/test_harness.py
sh /Users/pedro/Developer/mb-M00/integration/gates/run.sh ruff
sh /Users/pedro/Developer/mb-M00/integration/gates/run.sh pyrefly
sh /Users/pedro/Developer/mb-M00/integration/gates/run.sh ty
sh /Users/pedro/Developer/mb-M00/integration/gates/run.sh mypy
sh /Users/pedro/Developer/mb-M00/integration/gates/run.sh mcmr
```

Later packages use the same commands from their integrated worktree's `integration/gates/run.sh`;
replace the targeted nodeid with their applicable golden/fault/recovery witnesses. `suite` includes
every cumulative witness. Report files default to scratchpad `codex/mb/M00/`; set `MB_GATE_REPORT`
to another directory under scratchpad `codex/mb/` to keep each package's results separate.

The wrapper uses `mainboard proc timeout <limit> mainboard run --as-is ... python`: 900 seconds
for the entire suite, 120 for inspection or a target selection, and 300 per checker. Inner worker
budgets are 850, 90 and 280 seconds, reserving time for launch, preflight and cleanup. A worker timeout
terminates its process group, waits three seconds, then kills and reaps it. Inspection has a
100-second process alarm. CLI fixture calls are capped at 120 seconds. The launcher prints and
asserts `mainboard.__file__` on every invocation. All pytest commands name this worktree's
`pyproject.toml`, disable `randomly`, cacheprovider and plugin autoload, and use temporary basetemp.

## Isolation boundary

The launcher creates a fresh `/private/tmp/mb-gate.*` root. Worker environment variables are rebuilt
from a small allowlist: HOME, configuration, cache, plotting, solver and pytest files go into that
root. No remote-workspace, lake endpoint, provider credential or SSH agent variable is inherited.
Existing tests create their workspaces and `.mb` lakes below basetemp. No global lake override is
used, because several scenarios intentionally distinguish two independent lakes. Installed DuckDB
extensions are copied into the temporary cache after preflight; no extension download is allowed.

macOS workers use Seatbelt with filesystem writes denied except the temporary root, report output
and `/dev/null`. Network operations are denied except Unix sockets below the temporary root.
Linux workers require bubblewrap, a fresh filesystem root with read-only runtime/source mounts,
private network and PID namespaces, private `/tmp` and `/run`, and a minimal `/dev`. TCP services
in the Linux namespace cannot reach the real fleet. There is no native-worker fallback when OS
isolation is unavailable. A `python -S` canary proves native filesystem and external-network denial
without loading the audit hook before any tests run. Direct pytest execution without this proof
is refused by `conftest.py`.

The Python audit hook adds diagnostics for forbidden writes (including symlink escapes and
dir-fd paths), transport executables and DNS/connections. It is **not** an OS sandbox. In particular,
the `inspect` lane runs only collection, negative Python-boundary witnesses and one reviewed CLI
plan scenario in process; it denies all subprocesses. An inspection success is not a suite pass.
The static checker lanes use fixed read-only arguments, disabled writeback/repair and temporary
caches. They are trusted read-only tools, not native test scenarios, and do not claim OS isolation.

The outer, trusted `mainboard run --as-is` bootstrap preserves the existing live sync lock using
`O_RDONLY` plus `flock`; it injects `--no-install` into that launcher's frozen Pixi invocation.
Stale compiler writes are refused, not bypassed. These patches exist only in `sitecustomize.py`
on the launcher PYTHONPATH. Startup fails closed if the guard cannot initialize. Production files
are unchanged. The native Pixi launcher is not itself covered by the later worker sandbox;
the fixed frozen/no-install invocation is the narrowly scoped bootstrap dependency.

## Baseline and obligations

`baseline.json` preserves all 392 original nodeids at c97cb81 and their historical classifications:
379 passed, 5 skipped, 5 excluded and 3 restricted-environment outcomes. These are the earlier audit's
results, not new OS-isolated passes. Inspection checks that every original nodeid is still collected
and lists additions separately. No xfails or known source-negative expectations are introduced.

| Obligation | Scope and owner |
|---|---|
| M00-OS | Restricted macOS denies nested Seatbelt. Full suite, PostgreSQL lifecycle, native denial and console-process CLI flow must run on an OS-isolated runner. Exit 78 is blocked, never green. |
| M00-SOLVER | Four original install/solve/upgrade cases remain explicit offline-fixture skips, owned by M14/M15/M16. Faking a successful solver would not establish second-stage lock/install correctness. |
| M00-CATALOG | The original offer-list case now uses one recorded gpuhunt row; the fixture asserts exact query filters and one call. Its end-to-end execution awaits the OS lane. |
| M00-FLEET | Two original real-fleet cases are always skipped, even when an opt-in was inherited. M17 supplies fake-host capability witnesses. |
| M00-CONSUMERS | One original discovery case needs a monorepo consumer checkout; M33 owns the isolated consumer lane. |
| M00-PROCESS | Two original setsid skips and the timeout failure require a real isolated Linux process-group proof, owned by M06. |
| M00-QUACK | Two original served-lake cleanup restrictions require isolated Linux TCP and process cleanup, owned by M22. No connection to an existing localhost server is allowed on macOS. |
| M00-POSTGRES | Missing workspace PostgreSQL binaries produce an explicit fixture skip naming conda-forge postgresql; never fall back to system binaries or install a dependency. |
| M00-LEGACY | `legacy_runtime` extracts the actual immutable c97cb81 Git archive into temporary state, hashes it, creates a separate no-pip interpreter prefix, replaces PYTHONPATH and proves the legacy import. No new JobStore emulates old code. Its execution awaits the OS lane. |

The PostgreSQL fixture uses binaries beside the environment's Python. Each invocation creates a
fresh data directory and mode-0700 Unix socket directory, disables TCP, starts with bounded pg_ctl
waits, and stops in `finally`. Its witness creates a schema/table, inserts and counts a row, and
verifies the temporary data directory and disabled TCP. PostgreSQL 18.6 is installed in this
workspace, but the restricted macOS run did not start a server.

## Checker contract

The launcher refuses tool-version drift against the recorded versions: pytest 9.1.1, Ruff 0.17.0,
pyrefly 1.4.0.dev3, ty 0.0.85, mypy 2.4.0 and mcmr 0.0.4. Do not install or upgrade tools during gates.
Ruff covers `src` and `integration`; the three type checkers retain the existing `src` scope and
explicit worktree config; pyrefly includes warnings. Deterministic MCMR covers the worktree with
contextual/external lanes, writeback and repair disabled. Logs record exact argv and elapsed time.

Historical debt is Ruff 0; pyrefly 4 errors and 49 warnings; ty 25 diagnostics; mypy 9 errors;
MCMR 1908 findings and 1587 failures. Nonzero checker exits are preserved, not converted to success.
Compare diagnostics, not only counts: no new diagnostic outside the package's recorded ownership
is acceptable. Fix baseline errors in changed source files; do not add ignores. M00's final report
records its remeasurement and any test-harness-owned MCMR delta. A new skip must have an explicit
obligation and owner; missing cases, unexpected skips, failures and incomplete timeout runs remain
unaccepted. M33 must close the cumulative six flow families and nine incident witnesses.

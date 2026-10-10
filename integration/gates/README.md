# M00 execution gate

Only test infrastructure lives here. Run it from any worktree of this repository:

```sh
sh integration/gates/run.sh MODE [--report DIR] [--budget SECONDS] [NODEID...]
```

`MODE` is `suite`, `target` (the given pytest node ids), `inspect`, `ruff`, `pyrefly`, `ty`, `mypy`
or `mcmr`. `--report` names the folder the logs, `MODE.json` and `MODE.xml` land in; without it a
new `/tmp/mb-gate-report.*` folder is made and printed. `--budget` is the hard limit of the
whole run in seconds: 900 for `suite`, 120 for `inspect` and `target`, 300 for a checker. The worker
gets the budget less 30 seconds reserved for launch, preflight and cleanup, and `MODE.json` records
`budget_seconds`, `wall_seconds` and each step's `limit_seconds` and `seconds`.

```sh
sh integration/gates/run.sh suite --report /tmp/m07-gate
sh integration/gates/run.sh target --budget 90 integration/test_dead_host.py
sh integration/gates/run.sh mcmr
```

Later packages use the same commands from their integrated worktree. A package adds its golden,
fault and recovery witnesses to the existing fixtures and passes them as `target` node ids; `suite`
includes every cumulative witness. The launcher runs the environment's own interpreter
(`$MB_GATE_PROJECTS/.mainboard/envs/default`, default `/Users/pedro/Developer/projects`) beneath
`mainboard proc timeout`, so no compile, sync lock or install touches the live workspace, and it
prints and asserts `mainboard.__file__` on every invocation. Every pytest command names this
worktree's `pyproject.toml`, disables `randomly`, the cache provider and plugin autoload, and uses a
temporary basetemp. Each test starts in its own empty directory, never in the checkout.

## Isolation boundary

The launcher creates a fresh `/tmp/mb-gate.*` root and removes it at the end. Worker
environment variables are rebuilt from a small allowlist: HOME, configuration, cache, plotting,
solver and pytest files go into that root. No remote-workspace, lake endpoint, provider credential
or SSH agent variable is inherited. Existing tests create their workspaces and `.mb` lakes below
basetemp. No global lake override is used, because several scenarios intentionally distinguish two
independent lakes. Installed DuckDB extensions are copied into the temporary cache after preflight;
no extension download is allowed.

macOS workers use Seatbelt with filesystem writes denied except the temporary root, the report
folder, `/dev/null` and pseudo-terminals (a pty carries no state; the real-terminal test needs
one). Network operations are denied except Unix sockets below the temporary root. Linux workers
require bubblewrap, a fresh filesystem root with read-only runtime/source mounts, private network
and PID namespaces, private `/tmp` and `/run`, and a minimal `/dev`. TCP services in the Linux
namespace cannot reach the real fleet. There is no native-worker fallback when OS isolation is
unavailable. A `python -S` canary proves native filesystem and external-network denial without
loading the audit hook before any tests run. Direct pytest execution without this proof is refused
by `conftest.py`. Real `ssh` runs inside the sandbox like any other program and the kernel refuses
its connection (`test_a_real_ssh_cannot_connect`), so a test of a dead host sees the sentence it
would on a dead host.

`worker/guard.py` is a Python audit hook that adds diagnostics for forbidden writes (including
symlink escapes and dir-fd paths) and for DNS and connections. It is **not** an OS sandbox.
`worker/sitecustomize.py` installs it in every interpreter the gate starts and exits if it cannot,
and installs the offline gpuhunt double when a test asks for one. The tests of refusal in
`test_harness.py` ask the guard for its decision, or aim at a path that does not exist, so a missing
guard never turns them into the forbidden write; outside a gate worker they skip with a named
reason. The `inspect` lane runs only collection, those witnesses and one reviewed CLI plan scenario
in process, and denies all subprocesses. An inspection success is not a suite pass. The static
checker lanes use fixed read-only arguments, disabled writeback/repair and temporary caches. They
are trusted read-only tools, not native test scenarios, and do not claim OS isolation.

A subprocess `mb` runs this worktree's source: `conftest.py` puts `src` first on `PYTHONPATH` for
every test and `test_a_subprocess_mb_runs_the_worktrees_source` proves the console script resolves
there. A worker timeout ends its whole process group: terminate, three seconds of grace, then kill.

## Baseline and obligations

`baseline.json` preserves all 392 original nodeids at c97cb81 and their historical classifications:
379 passed, 5 skipped, 5 excluded and 3 restricted-environment outcomes. These are the earlier
audit's results, not new OS-isolated passes. Inspection checks that every original nodeid is still
collected and lists additions separately. No xfails or known source-negative expectations are
introduced.

| Obligation | Scope and owner |
|---|---|
| M00-OS | Where the host forbids a nested sandbox (a restricted macOS agent), `suite` and `target` exit 78 and the OS-isolated lane must run elsewhere. Exit 78 is blocked, never green; only `inspect` runs. |
| M00-SOLVER | Four original install/solve/upgrade cases remain explicit offline-fixture skips, owned by M14/M15/M16. Faking a successful solver would not establish second-stage lock/install correctness. |
| M00-FLEET | Two original real-fleet cases are always skipped, even when an opt-in was inherited. M17 supplies fake-host capability witnesses. |
| M00-CONSUMERS | One original discovery case needs a monorepo consumer checkout; M33 owns the isolated consumer lane. |
| M00-PROCESS | Two original setsid skips and the timeout case require a real isolated Linux process-group proof, owned by M06. |
| M00-QUACK | Two original served-lake cases require isolated Linux TCP and process cleanup, owned by M22. No connection to an existing localhost server is allowed on macOS. |
| M00-POSTGRES | Missing workspace PostgreSQL binaries produce an explicit fixture skip naming conda-forge postgresql; never fall back to system binaries or install a dependency. |

The offer-list case runs against one recorded gpuhunt row and asserts the exact query filters and
one call. `legacy_runtime` extracts the actual immutable c97cb81 Git archive into temporary state,
hashes it, creates a separate no-pip interpreter prefix, replaces `PYTHONPATH` and proves the legacy
import; no new JobStore emulates old code. Both run in the suite lane.

The PostgreSQL fixture uses binaries beside the environment's Python. Each invocation creates a
fresh data directory and mode-0700 Unix socket directory, disables TCP, starts with bounded pg_ctl
waits, and stops in a finalizer. Its witness creates a schema/table, inserts and counts a row, and
verifies the temporary data directory and disabled TCP.

## Checker contract

The launcher refuses tool-version drift against the recorded versions: pytest 9.1.1, Ruff 0.17.0,
pyrefly 1.4.0.dev3, ty 0.0.85, mypy 2.4.0 and mcmr 0.0.4. Do not install or upgrade tools during
gates. Ruff covers `src` and `integration`; the three type checkers retain the existing `src` scope
and explicit worktree config; pyrefly includes warnings. Deterministic MCMR covers the worktree with
contextual/external lanes, writeback and repair disabled. Logs record exact argv and elapsed time.

Historical debt is Ruff 0; pyrefly 4 errors and 49 warnings; ty 25 diagnostics; mypy 9 errors;
MCMR 1908 findings and 1587 failures. Nonzero checker exits are preserved, not converted to success.
Compare diagnostics, not only counts: no new diagnostic outside the package's recorded ownership
is acceptable. Fix baseline errors in changed source files; do not add ignores. The harness itself
adds no MCMR finding beyond the baseline conftest's `MB` constant. A new skip must have an explicit
obligation and owner; missing cases, unexpected skips, failures and incomplete timeout runs remain
unaccepted. M33 must close the cumulative six flow families and nine incident witnesses.

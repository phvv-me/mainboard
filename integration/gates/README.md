# M00 execution gate

Only test infrastructure lives here. Run it from any worktree of this repository:

```sh
sh integration/gates/run.sh MODE [--report DIR] [--budget SECONDS] [NODEID...]
```

`MODE` is `suite`, `target` (the given pytest node ids), `inspect`, `seed`, `ruff`, `pyrefly`, `ty`,
`mypy` or `mcmr`. `--report` names the folder the logs, `MODE.json` and `MODE.xml` land in; without
it a new `/tmp/mb-gate-report.*` folder is made and printed. `--budget` is the hard limit of the
whole run in seconds: 900 for `suite`, 120 for `inspect` and `target`, 300 for `seed` and a
checker. The worker gets the budget less 30 seconds reserved for launch, preflight and cleanup, and
`MODE.json` records `budget_seconds`, `wall_seconds` and each step's `limit_seconds` and `seconds`.

```sh
sh integration/gates/run.sh suite --report /tmp/m07-gate
sh integration/gates/run.sh target --budget 90 integration/test_dead_host.py
sh integration/gates/run.sh mcmr
```

On macOS, the native Swift cases need Xcode's SwiftParser host libraries; Command Line Tools
alone are insufficient. Select Xcode for the invocation rather than changing global
`xcode-select` preferences. From the monorepo root:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer mainboard run --as-is \
  sh packages/mainboard/integration/gates/run.sh suite --report /tmp/m07-gate --budget 900
```

The launcher preserves `DEVELOPER_DIR` in its otherwise isolated worker environment. Check
available disk space before a full lane: temporary databases, copied extensions and Swift
compiler caches need writable capacity, independently of the report folder's size.

Later packages use the same commands from their integrated worktree. A package adds its golden,
fault and recovery witnesses to the existing fixtures and passes them as `target` node ids; `suite`
includes every cumulative witness. The launcher runs an environment's own interpreter beneath
`mainboard proc timeout`, so no compile, sync lock or install touches the live workspace: the
environment `uv run` leaves in `VIRTUAL_ENV` when there is one, else the workspace's
(`$MB_GATE_PROJECTS/.mainboard/envs/default`, default `/Users/pedro/Developer/projects`). It prints
and asserts `mainboard.__file__` on every invocation. Every pytest command names this
worktree's `pyproject.toml`, disables `randomly`, the cache provider and plugin autoload, and uses a
temporary basetemp. The worker and each test start in an empty directory, never in the checkout.

## In CI

`[tool.mainboard.ci]` runs the gate as two steps, `uv run sh integration/gates/run.sh seed` and
`uv run sh integration/gates/run.sh suite --report build/gate-report --budget 1500`, so
`mainboard ci`, `mainboard ci --matrix` and the GitHub workflows gate the installation this
checkout's lock builds. A bare `pytest` is refused by `conftest.py`, with no fallback.

- `seed` is the one lane that goes online. The sandbox has no network and downloads nothing, so
  this installs the DuckDB extensions `Lake` loads into the user cache that every sandbox copies
  from; what is already there is left alone. A `suite` without them stops with `M00-EXTENSIONS`.
- macOS needs nothing: Seatbelt ships with it.
- Linux needs bubblewrap and, where AppArmor restricts unprivileged user namespaces (Ubuntu
  24.04's default), `kernel.apparmor_restrict_unprivileged_userns=0`; without that, bubblewrap
  stops at `loopback: Failed RTM_NEWADDR`. It also needs zsh, which an Ubuntu runner lacks and
  the three `test_managed_python` login-shell cases skip without, an unexpected skip.
  `linux-runner.sh` installs both and lifts the restriction, with sudo and only where a probe
  shows bubblewrap cannot run, and fails unless bubblewrap then makes a network namespace. It
  changes the whole machine, so only a throwaway CI runner runs it, from the workflow, never from
  a step of `[tool.mainboard.ci]`. Where nothing isolates, `suite` exits 78 (`M00-OS`).
- The checkout needs its history (`fetch-depth: 0`): the legacy-runtime witness extracts the
  c97cb81 archive with `git archive`.
- A lane holds only its own tool to the recorded versions: pytest for `suite`, `target` and
  `inspect`, the checker for a checker lane, nothing for `seed`. The lock supplies pytest and
  pyrefly; other checker versions must satisfy their own guards before a lane runs.
- The worker starts in the gate's empty temporary root, not in the checkout. Test modules are
  imported there, so a constant computed at import (`Project().out_dir()` in
  `test_managed_python`) agrees with the same call inside a test, whether or not the checkout
  sits beneath a workspace manifest, as this one does in the monorepo.

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

A lane refuses drift of its own tool against the recorded versions: pytest 9.1.1, Ruff 0.17.0,
pyrefly 1.4.0, ty 0.0.85, mypy 2.4.0 and mcmr 0.0.4. The pyrefly stable-release version rebaseline
is recorded in `baseline.json`; its old diagnostic counts remain historical until remeasured.
Do not install or upgrade tools during gates. Ruff covers `src` and `integration`; the three type
checkers retain the existing `src` scope
and explicit worktree config; pyrefly includes warnings. Deterministic MCMR covers the worktree with
contextual/external lanes, writeback and repair disabled. Pyrefly and ty query the gate's actual
interpreter, so a stale project-local `.venv` cannot replace the locked workspace's dependency
metadata. Logs record exact argv and elapsed time.

Historical debt is Ruff 0; pyrefly 4 errors and 49 warnings; ty 25 diagnostics; mypy 9 errors;
MCMR 1908 findings and 1587 failures. Nonzero checker exits are preserved, not converted to success.
Compare diagnostics, not only counts: no new diagnostic outside the package's recorded ownership
is acceptable. Fix baseline errors in changed source files; do not add ignores. The harness itself
adds no MCMR finding beyond the baseline conftest's `MB` constant. A new skip must have an explicit
obligation and owner; missing cases, unexpected skips, failures and incomplete timeout runs remain
unaccepted. M33 must close the cumulative six flow families and nine incident witnesses.

## October 21 validation checkpoint

The full isolated run recorded in `/private/tmp/usagi-mainboard-final-suite` finished with
**1060 passed, 1 failed, 11 skipped** and three warnings in 446 seconds. Its only failure was
`test_the_legacy_runtime_imports_only_its_own_archive`: a workspace editable `.pth` file added the
current source tree to the legacy interpreter's path after `PYTHONPATH`. The legacy helper now
removes that resolved current-source path after interpreter site initialization and before the
requested program; the immutable archive and dependency site-packages remain unchanged. This is
a focused isolation repair, not a replacement legacy implementation.

The subsequent bounded `target` run in `/private/tmp/usagi-mainboard-checkpoint-target` passed
both that legacy-isolation witness and `test_a_subprocess_mb_runs_the_worktrees_source`
(**2 passed**, no skips; native OS preflight exit 0). No full-suite rerun followed this repair.
The earlier failing suite must therefore remain a failing receipt, not be relabeled green.

The fixed pyrefly invocation explicitly uses the gate's managed Python interpreter rather than
discovering a potentially stale project `.venv`. The read-only checker run in
`/private/tmp/usagi-mainboard-checkpoint-typing` used Python **3.14.8**, pyrefly **1.4.0**, explicit
package configuration and warning severity. It preserved exit **1**: **4 errors and 45 warnings**.
All four errors concern TOML `Item` operations in unchanged `src/mainboard/deps/editing.py`:
assignment at line 70, membership at line 94, assignment at line 97 and indexing at line 98.
Warnings span existing package source and include missing stubs; the log also reports the
deliberate `PYTHONPATH` environment. These are not accepted or suppressed, and the historical
diagnostic counts in `baseline.json` were not replaced by this measurement.

Root-managed Ruff **0.17.1**, with this package's explicit configuration, checked and verified
formatting of `integration/legacy.py` and `integration/gates/runner.py` successfully after import
ordering was fixed. This focused check was not the version-guarded Ruff lane, whose baseline
remains 0.17.0. The focused pytest lane used its pinned **9.1.1**. No tool installation, dependency
change or unrelated source repair was performed for this checkpoint.

### Requirement typing and full-suite follow-up

The subsequent requirement-editor fix narrows each nested TOML table before traversal and keeps
the checked requirement value in one local before updating its version. It removes all four
`deps/editing.py` checker errors without a cast, ignore or weaker checker setting. Scalars,
arrays and inline-table parents now return a consistent `MissionError` without mutation; six
parameter cases cover both lookup and creation modes. The existing comment alignment and
table-extra-field assertions remain.

`/private/tmp/usagi-mainboard-manifest-typing-target` reports **11 passed, 4 expected solver skips**.
The new complete isolated run, `/private/tmp/usagi-mainboard-corrected-suite`, exits 0 with
**1067 passed, 11 skipped, 3 warnings** in 496.10 seconds. It includes the previously failing
legacy-isolation case and all six new malformed-parent cases. The skips still require Linux TCP,
offline solver fixtures, setsid or an explicitly configured real remote workspace; none is new.
The three warnings retain the existing invalid escape and Pathspec deprecations.

Managed pyrefly `/private/tmp/usagi-mainboard-manifest-typing` reports **0 errors, 45 warnings**,
preserving exit **1** because warnings are enabled. Remaining warnings are not suppressed or
accepted as a clean full typing gate. Explicit-config Ruff and formatting pass for both changed
files. Historical receipts and baseline counts above remain unchanged.

### Warning cleanup and final validation attempts

The next change removes 40 source warnings across 24 production files: redundant conversions
of already typed values and exception subclasses already caught by their parent. SQL cell
formatting retains explicit integer/Boolean handling before float formatting; removing its
old conversion without that distinction would change runtime behavior. Twelve literal-output
cases now cover null/text, Booleans, integers beyond exact Double precision, floats and nested
values. No new casts, ignores, checker overrides or dependency declarations were added to this
package.

`/private/tmp/usagi-mainboard-typing-cleanup-target` passes **117 tests**, with no failures,
skips or warnings, using pytest **9.1.1** and native OS isolation. It covers query rendering,
CUPTI synthetic records, dead hosts, cloud catalog imports and managed Python execution in
82.54 seconds. The intermediate checker receipt `/private/tmp/usagi-mainboard-typing-cleanup`
retains **0 errors, 5 missing-stub warnings**, exit **1**.

After the root workspace declared and synchronized `types-pyyaml` **6.0.12.20260906** and
`types-seaborn` **0.13.2.20261006** (with `pandas-stubs` **3.0.5.260914**), the managed pyrefly
**1.4.0** lane `/private/tmp/usagi-mainboard-final-typing-clean` passes with **0 diagnostics**,
exit **0**. It retains explicit package configuration, Python **3.14.8**, `--min-severity warn`
and the two existing configured suppressions. The deliberate `PYTHONPATH` notice is not a
source diagnostic. Explicit-config Ruff **0.17.1** check and format check pass for all 25
changed Python files; this does not replace the separately version-guarded Ruff lane.

The first final full-suite attempt, `/private/tmp/usagi-mainboard-final-typed-suite`, exits
**1** with **965 passed, 114 failed, 11 skipped, 3 warnings** in 363.191 worker seconds. Every
one of its 114 JUnit failures identifies missing SwiftParser under the globally selected
Command Line Tools path. The invocation omitted `DEVELOPER_DIR`; no source fallback or global
toolchain preference was changed. This remains a failed receipt.

The explicit-Xcode retry, `/private/tmp/usagi-mainboard-final-typed-xcode-suite`, also exits
**1** and is **incomplete**: the disk filled, pytest teardown raised `OSError(28)`, and the gate
could not save JUnit or its final JSON receipt. Only partial progress and the disk-full
traceback remain. Per-test outcomes cannot be reconstructed or attributed from that output.
The launcher removed its temporary root; no parent build or unrelated cache was removed.
At that checkpoint, a new full-suite pass was required after restoring capacity. Neither
failed attempt supersedes the earlier 1067-pass receipt or establishes acceptance for this
source change.

After capacity was restored, `/private/tmp/usagi-mainboard-typed-capacity-rerun` passes the
complete isolated suite with **1079 passed, 11 skipped, 3 warnings**, exit **0**. The invocation
explicitly selects Xcode with `DEVELOPER_DIR`, uses pytest **9.1.1** and retains the 900-second
bound. Native isolation preflight exits **0**; pytest reports 468.80 seconds, the worker
469.999 seconds and the complete gate 470.46 seconds. Sampled free space stayed between
18 and 19 GiB. The JUnit and JSON receipts are both complete.

The 11 unchanged skips comprise three isolated Linux TCP cases, four offline solver fixtures,
two `setsid` cases and two explicitly unconfigured real-fleet cases. The three existing
warnings are an invalid escape sequence and two Pathspec deprecations. They remain visible;
this Mac result does not establish Linux, real-fleet or CUDA hardware acceptance.

The 25-file source snapshot remained unchanged throughout all three full-suite attempts. Its
per-file hashes are in `/private/tmp/usagi-mainboard-typing-source.json`, whose SHA-256 is
`47854dd8b96851a4f667e8c3d84ebff6a5e7b70ae93b995f0116bd9b04d3298c`.
Historical receipts and `baseline.json` remain unchanged; no installation ran during a gate.

The CUDA conversion cleanup follows the exact locked `cupti-python` **13.2.0** package's
published contract: `cupti/cupti.pyi` declares `ActivityKind(enum.IntEnum)` at line 873,
`ActivityKernel11.kind: int` at 830, `ActivityMemcpy6.copy_kind: int` at 1049, its `kind: int`
at 1058, and `get_timestamp() -> int` at 101. The lock's [published Linux aarch64 wheel](https://files.pythonhosted.org/packages/07/4a/95ffcb57f8131030e7f175b611df2997fc9755c541229996f1adda1adff5/cupti_python-13.2.0-cp314-cp314-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl)
was read without installation or execution and verified against SHA-256
`cd04be232ad46d603dbd4c654b7347ddb5c06c38bc8160345ad6d54c6c08a729`.
IntEnum members retain integer equality and hashing. The audit is recorded in
`/private/tmp/usagi-cupti-contract-audit.json`; it checks published metadata, not real CUDA
execution. This Mac has no installed CUDA/CUPTI binding, and the synthetic record tests do not
establish hardware compatibility.

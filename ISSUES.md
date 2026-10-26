# Open issues

Found while running real work, kept here to fix later with care rather than in the middle of a
campaign. Each entry says where it showed up, what it costs, and what is already done. Newest
campaign first; move an entry to the bottom section once fixed and verified.

## Found drawing the MLSys kernel figure in the plot language, 2026-10-25

The two RTX 4090 panels of the paper's kernel figure (a Range and a Dot layer per series, Dodge,
log axes, one SQL file per panel) render from a `[figures.*]` specification and two SQL files, but
only after the workarounds below. The paper's generators draw the figure in plain matplotlib
instead, so every item here is open.

1. **`--config` replaces the project style where it should extend it.** A config that adds a
   few markers made `gpt2` and `qwen3` vanish ("style has no explicit colors for ..."), so the
   palette of `research/cuda-tokenization/plots.toml` had to be copied into the overlay.
2. **Legends duplicate and can collapse the layout.** A mapped `marker` or `fill` draws a
   panel key and a figure-level key, `fill` lists the values `True`/`False`, and at column width
   (3.25 in) the figure legend made constrained layout fail ("axes sizes collapsed to zero").
   `legend = false` on every panel is the only way out, which leaves no key.
3. **No reference line.** A dashed line at 1 (a ratio, a shipped baseline) needs a fake series
   with its own marker. A `Rule`/`HLine` mark with a legend entry would do.
4. **Panels are subfigures, so nothing is shared and nothing is lettered.** Stacked panels
   cannot share an x axis (tick labels repeat, or are blanked by hand with `xticklabels = []`),
   and there is no panel-letter option.
5. **`Range` has no caps**, which the paper's other plots use for error bars.
6. **Display names of categories live in the shared `labels` table**, so a one-figure
   rename (`doc4k` to `1 doc`) is a global style change.
7. **Numbers the text quotes cannot come from the same query.** The paper prints values read
   off these plots (`\MergeBlockCrawl`) and guards claims (pooled values equal a sealed table).
   A figure spec has no `macros` or `guards` table, so the generators keep figure and macros
   apart and they can drift. A `[figures.NAME.macros]` of scalar SQL, digests written beside the
   figure's own `.metadata.json`, and SQL `error()` guards would close that.

## Found cleaning mainboard with `mb rules fix`, 2026-10-23

1. **A `safe` fix is proven only by the type checkers, which a semantic change slips past.**
   PY-PYDA0004 rewrote `Scope.model_validate({_PYTHON: {...}})` in `manifest/members.py` as
   `Scope(_PYTHON=...)`: the key was a variable holding `"python"`, the keyword is a field named
   `_PYTHON` the model ignores. Type diagnostics stayed at 94, the run wrote it, and the next
   `mainboard run` composed a manifest without the members nothing else required (aizk,
   rlsalchemy, research-compression, cutoken), relocked without them and uninstalled them, which
   the gate then failed on. Fixed: the rule takes string keys only (`is_string_key` in
   `call_mapping_entries`), its example keeps a name-keyed call; the lock was restored and the
   environment reinstalled frozen. Still open: a written fix's proof should include the package's
   tests (or at least an import and manifest-compose smoke), not only the diagnostics count.
2. **ALL-FUNC0003's fix moves a helper into its class without rewrapping it.** The added indent
   pushed 20 docstring and string lines past 99 columns, rewrapped by hand. The fix should rewrap
   what it indents (or `mb rules fix` should run the formatter on the files it wrote).
3. **Three rule false positives, fixed in their rules.** PY-DOCU0001 flagged a cyclopts command's
   `Args:` heading, which cyclopts needs for parameter help (exempted for `*.command`/`*.default`),
   and read any prose line ending in a colon as a label (a label is now at most three words).
   PY-PYDA0004's string-key fix is above.
5. **PY-MODU0002's fix deleted empty initializers, and the rule catalog stopped loading.**
   `load_catalog` walked `mainboard.analysis.rules` with `pkgutil.walk_packages`, which does not
   descend into namespace portions, so after the fix only 16 of 244 rules registered and `mb
   rules check` reported 355 findings as if the code were nearly clean. The gate passed, and the
   fix's proof had failed its baseline, so nothing judged it. Fixed: `core.modules.submodules`
   lists module files on disk (catalog and the CLI profiler), and `test_catalog_modules` fails
   when a rule file is not imported.
4. **`mb rules` has no inline waiver.** A finding the code is right to keep (a remote verb called
   by name, `census`) can only be excluded per file in `[tool.mcmr.rules."ID"] exclude`, which
   also hides every later finding of that rule in the file.
6. **The route table hid the verbs from PY-DOCU0001, and dropped `mb ci`.** `cli.build` now
   registers every verb from one `_ROUTES` table of unbound methods, so no verb is passed to
   `.command(` by name and the rule flagged every `Args:` heading. The rule now also counts a
   method named through its class (`JobVerbs.show`) in a module that calls `.command(`. The first
   table also left out `ci`'s default (`ci.default`, not `.command`), and `test_every_command`
   did not notice: `mb ci` answered with no command until `_DEFAULTS` put it back. That test
   should compare the registered verbs against the previous set.
7. **`cli.py` imports `mainboard.verbs` from the root package (ALL-ARCH0003).** The fix is moving
   the module into `mainboard.verbs`, which changes the console scripts (`mainboard.cli:main` in
   `pyproject.toml` and the generated `mb`/`mainboard` scripts) and twelve test imports. It needs
   the editable tool reinstalled in a quiet window, so it waits for the end of this pass.
8. **Fact materialization, not rule evaluation, is what makes `mb rules check` slower than mcmr.**
   On one 733-file copy of this package, mcmr took 6.5 s and `mb rules check` 17.4 s. The 17 s
   split into 1.8 s parsing trees, 11.5 s creating 445 derived relations one `CREATE TABLE`
   after another (`CommentFact/groups` alone 1.8 s), and 1.9 s running 231 rules. Remedies,
   fastest first: a per-file fact cache keyed by content hash, materializing independent
   relations concurrently along their dependency graph, building only what the selected rules
   read, and fixing the outlier. The minutes a cleanup loop costs come from the fix proof,
   which runs the whole gate (7.5 min) where only the affected tests are owed. Done: relations
   are now adopted concurrently, one dependency wave at a time (adoption 11.5 s -> 7.0 s, same
   findings). Open: the fact cache, building only what the selected rules read, the outlier,
   and a proof that runs only the affected tests.
9. **`mb rules fix` cannot apply PY-NAMI0001 across files.** A rename leaves another module's
   import of the old name in place, ty reports it unresolved, and the whole component is
   withheld; 15 renames were withheld this way. The fixer now says which rules grew in each
   withheld file, and the fix now leaves alone a name another module imports or reaches as an
   attribute. Open: renaming the importers too.
10. **PY-EXCE0002 and PY-INTE0001 reported a whole file at `1:0`.** The port kept mcmr's value
    but dropped the location, so a finding named `tryblockfact:<path>` with no line. Both now
    report each site: EXCE0002 each clause with its kind and count, INTE0001 each `isinstance`
    call with the subject, the concrete types and the capability; INTE0001 also reads a tuple of
    types now (`plots/panel.py:101` was missed). EXCE0002 bounded every clause at 1, so a
    handler that logs and re-raises failed; the `try` clause keeps 1 and the handler clauses 3.
11. **ALL-CONT0004 and ALL-FUNC0009 measured the same nesting.** Both fired at the fourth level,
    reporting 39 of FUNC0009's 40 functions twice; FUNC0009 also called four levels "depth 3"
    and skipped `with`, which Pylint's PLR1702 counts. FUNC0009 is gone and PLR1702 maps to
    CONT0004.
12. **ALL-PARA0001 repeated one signature on every override.** `Image.request` and its two
    overrides were three findings. An override of a project base method is now exempt, so the
    finding stands once at the base.
13. **PY-CONS0001 with PY-CONS0002 leaves a shared constant no compliant form.** CONS0002 flags
    `from .schema import ALIAS` but not `schema.ALIAS`, which CONS0001 still flags as public.
    Spelling the access module-qualified only moved the finding. Owner's rule (2026-10-24): a
    shared constant lives in a `constants.py`, or a module of a `constants/` package, and only
    those are imported. CONS0002 now reads attribute access as well, and both rules exempt those
    modules. The 162 shared constants moved:
    - Pure SQL modules went into `rules/{core,families,python}/constants/`.
    - Mixed modules gave up their shared constants, with any private builders.
    - The dispatch states became the `vocabulary.Word` enum.
    - The schema's `tables`/`views`/`dialect` and the scheduler registry are objects, so they
      are lowercase.
    - The agent offers `closure_path`/`wrapper_path`.

    The waivers that hid 24 more were removed.
14. **Alphabetical member order and mypy's attribute inference disagree.** With `__enter__`
    before `__init__`, mypy takes an attribute's type from its first assignment in `__enter__`
    and reports the declaration in `__init__` as a redefinition (`profile/spans.py`,
    `profiler.py`, `providers/nvidia/tracer.py`). Fixed by declaring the attributes in the
    class body.
15. **The type checkers disagree on a function's `__name__`.** ty allows it only on
    `FunctionType` (its FAQ suggests a Protocol), and mypy refuses a `def` passed where a
    `FunctionType` is expected. `cli._route` and `lab.board_surface.experiment` take a
    `Callable` and narrow it with `isinstance(..., FunctionType)`, which all three accept.
16. **mypy had never checked the integration tests.** `mb lint` hands it those files, and the
    first one with a relative import stopped mypy before it checked any other. With
    `explicit_package_bases` and `mypy_path` set it checks them all: 93 errors in 30 files,
    mostly `SimpleNamespace` doubles passed where a real type is declared. Open.
17. **37 Protocols describe foreign libraries** (cuda-python, NVML, CUPTI, nvtx, roctx,
    os_signpost, the Arrow stream interface), which no ABC can stand for since their objects
    never subclass ours. Open, the owner's call: adapters at each boundary, or the libraries'
    own types, which a platform without them cannot check. PY-INTE0003 now refuses a Protocol
    in a `.py` file. Proposed: a stub (`stubs/<library>.pyi`) per library, the standard way to
    type a module a platform may lack, then top-level guarded imports instead of casts. Needs
    the 4090 to verify the NVIDIA paths.
18. **PY-INTE0001's five remaining findings are all JSON and TOML boundaries**
    (`monitor.py:192`, `project_configuration.py:65`, `panel.py:101`, `render/query.py:45,47`),
    which mcmr's own text exempts. Fixed in the code: the JSON reads check `Mapping`, the
    rotation `numbers.Real` (`numbers.Rational` excludes `float`), and the normalizer of
    DuckDB's closed value union dispatches with `match`, since `Sequence` would also take
    `str`.
19. **PY-PYDA0004 rewrote `model_validate` of computed values into a constructor call** that
    mypy and pyrefly refuse, while ty, the fix proof's only checker, accepted it. The rule now
    fires only when every value is a literal or a name. Open: the proof should run all three
    checkers.
20. **PY-NAMI0001 read only five prefixes.** It flagged `fits`, `failed`, `locked` and
    `_recoverable`. It now takes auxiliaries (`is_`, `has_`, `can_`, `needs_`, ...) and a
    first word ending like a third-person verb, a participle or `-able`, minus nouns such as
    `status`. Four true findings remained and were renamed (`is_under`, `is_unsafe`,
    `is_execution_timeout`, `_is_hex_sha256`).

## Found hardening `mb query` (streaming, result hash, read-only lake), 2026-10-13

1. **A served lake is writable from `mb query`.** A local lake is protected by DuckLake's own
   `READ_ONLY` attach (`Lake.attach`'s default), which refuses every lake write including a
   data-modifying CTE. Over Quack (`MB_LAKE=quack:`) the pinned 2.0 dev build does not honor a client
   `READ_ONLY` attach, and `quack_query_by_name('lake', 'DELETE …')` committed a delete on the served
   lake during review. The server is where this belongs: serve a second, read-only endpoint (a
   `quack_serve` from a read-only attach) for queries, beside the writable one mb's writers append
   through. Meanwhile DuckLake time travel can recover such a delete only until `lake compact` expires
   snapshots older than 30 days (`expire_older_than`; files go 7 days after that, `delete_older_than`). Fixed 2026-10-25: read access attaches a READ_ONLY endpoint.
2. **`integration/gates/baseline.json` names a renamed test.** It still lists
   `integration/evidence/test_core.py::test_check_records_the_checksums_of_chunks_kept_before_them`, so the
   gate's inspect lane fails its baseline accounting (the suite lane is unaffected). Present on `main`
   before this work. Refresh the nodeid when the baseline is next re-measured.

## Found renting a vast RTX 4090 for the cutok version sweep, 2026-10-11 23:30

1. **`host hold vast --env cutok` fails on every fresh machine: `ModuleNotFoundError: sqlalchemy`.**
   Two tries (a Threadripper PRO 3975WX / RTX 4090 landed first, then another), both at the step
   "installing cutok in <alias>'s mirror from the shipped lock", which ran `mainboard install cutok
   --profile <alias>` on the host and died with `ModuleNotFoundError: sqlalchemy`; the hold then
   released the rental correctly (credit 83.1451 to 83.1274, about $0.02 for the two). The tool is
   installed from the mirror's `packages/mainboard`, whose dependencies now include
   `sqlalchemy>=2.1.3` and `duckdb-sqlalchemy` (the lake layer), so a machine that never had the
   tool before is the first to meet them; gold, pedro-cvlab and crimson were set up earlier and
   nothing reinstalled their tool. Most likely the same fault as item 1 of "Found setting up
   pedro-home-wsl" below (a machine with no CPython gets 3.15, for which a dependency has no wheel);
   `hold` gives no way to pre-install the tool onto 3.14 as that workaround did. Cost: no rented
   machine can be onboarded, so the card-versus-host
   arm of a diagnosis (a 4090 on a non-Intel host) cannot run. Not investigated further: the
   provider's sshd log is all `job logs` returns for the hold. Check first whether the host's
   `uv tool install --editable` resolves the workspace fork (`packages/sqlalchemy`) or PyPI's
   sqlalchemy, and whether `host setup` of a new SSH host hits the same.

## Found fixing the fleet's host issues, 2026-10-11

Owner rule (17:55): hosts keep their daemons alive without sudo. No `loginctl enable-linger`, no
root unit files, no `/etc` edits.

### Mainboard

1. **Nothing restarts pueued after a reboot.** `Onboarding.verify_queue` (`dispatch/onboard.py`)
   answers `pueue status` in the activated shell and only when it does not, installs pueue with
   `pixi global install pueue` (when `command -v pueued` finds none) and starts `pueued -d`. Every
   step is user-level; the only `sudo` in the source is the OS package upkeep in `upkeep.py`. But
   `setup` and `sync` run only when someone runs them: pedro-cvlab rebooted at 15:25 on 10-11 and
   its queue stayed dead until an agent started `pueued` by hand at 15:37. crimson, purple and
   gold hold no boot entry either (Linger=no, no crontab). Sudo-free mechanisms,
   chosen by what each host allows (`loginctl show-user <user> -p Linger` and `systemctl
   is-active cron` are readable without sudo; cron is active on all four hosts):
   - Lingering already on (pedro-cvlab, a pre-existing setting): a `systemd --user` unit
     `~/.config/systemd/user/pueued.service`, enabled, running the pixi-global pueued by full path
     in the foreground. `ExecCondition=/bin/sh -c '! pueue status ...'` skips it when a daemon
     already answers, so it never starts a second one. `XDG_RUNTIME_DIR` is the user manager's own,
     so the socket sits where every client looks.
   - Lingering off (crimson, purple, gold): a user crontab `@reboot` line that is `verify_queue`'s
     own recipe, `bash -lc 'cd ~/.mb-jobs && . .mainboard/envs/default/activate.sh && { pueue
     status || pueued -d; }'`, idempotent by the same `||`.
   - Pitfall for the cron route: pueue puts its socket and pid under `$XDG_RUNTIME_DIR` (pueue's
     changelog, #243) and falls back to the pueue directory when it is unset (shown on macOS
     2026-10-11: socket and pid landed in `pueue_directory`). Cron and a boot without lingering
     have no `/run/user/<uid>`, while an ssh session's clients look there, so the boot daemon would
     be invisible and the next `sync` would start a second daemon on the same state file. The host's
     `~/.config/pueue/pueue.yml` must set `shared.runtime_directory` to one absolute path so the
     daemon and every client agree.
   Code to write: `host setup` probes `Linger`, installs the matching mechanism (and the
   `runtime_directory` key for cron), and `doctor` reports a host with neither.
2. **`verify_queue` never replaces a healthy daemon that runs from a deleted path.** It starts a
   daemon only when `pueue status` fails, so a daemon whose binary or directory was wiped lives on
   through every `sync`: crimson's pid 65612 (up since 09-24) runs `.mainboard.aside-r5/.../pueued
   (deleted)`, and gold's pid 49321 (up since 10-07) has both cwd and exe `(deleted)` (the
   10-07 wipe), which the next env rebuild would also leave behind. `doctor` should warn on a
   `/proc/<pid>/exe` or `cwd` ending ` (deleted)`; `sync` could shut such a daemon down and start a
   fresh one when its queue holds nothing running or waiting.
3. **`ssh <host> pueue status` cannot work on a plain ssh PATH, by design.** Dispatch finds pueue
   through the activation (`wrapping.USER_BINS` plus the environment's `bin`). The login PATH of
   the bash hosts adds the dotfiles' pixi-global dirs (crimson: `~/.pixi/x86_64/bin` has `pueue`),
   the non-interactive one adds nothing, and `~/.config/mainboard/path.sh` is written for the center
   only. Supported spellings: `ssh <host> 'bash -lc "pueue status"'`, the full path
   (`~/.mb-jobs/.mainboard/envs/default/.pixi/envs/default/bin/pueue`), or `mainboard shell --on
   <host>`. Putting the environment on every non-interactive PATH would shadow the system tools.
4. **A job host keeps a staging lake, on purpose.** `Lake.at(root).ready()` in
   `dispatch/provenance.py` (`archive`) creates `.mainboard/lake` and `lake.sqlite` the first time
   a registered trial runs on a host: pedro-cvlab 10-07, crimson 10-08, gold 10-11 08:40. It is not
   an artifact of a scratch run, and later real jobs write into it (gold's `events` at 11:23).
   Nothing collects it; the evidence travels through the tree. Say so in `host setup`'s output
   or `doctor`, so nobody deletes it.
5. **An A/B scratch tree links into the live mirror.** Each of gold's `/tmp/ab-r2/{new,old,prime,
   ps,final,split}` held `.mainboard` and `data` symlinks to `~/.mb-jobs`. `rm -rf` does not follow
   them, `find -L`, `rsync -L` and `du -L` do. Whatever builds such trees should copy the compiled
   artifact, not link the mirror's `.mainboard`.
6. **Owed once the queue drains: `mainboard host sync pedro-cvlab`.** The 4090's mirror and
   environment were left alone because the registered v26 program (jobs 677-709) was running.
7. **crimson and gold done (2026-10-11 18:25, the lead, on the owner's "fix all open issues").**
   Both queues were empty. On each host the steps were:
   - `pueue shutdown`;
   - `runtime_directory: ~/.local/share/pueue` in pueue.yml, with the old file kept as
     `pueue.yml.before-runtime-directory`;
   - a cron-like start under `env -i`, from the live environment binary;
   - a second start, which was a no-op;
   - the user crontab `@reboot` line.
   The gate-lock guard was dropped, because a 5 s near-zero-CPU restart cannot disturb the gate
   chain. Results:
   - crimson: pid 65612 (deleted aside-r5 binary) became 821436, with 1,935 done tasks kept, and
     `job submit --on crimson -- true` ran as 1936, ok.
   - gold: pid 49321 (deleted binary) became 1825573, with 360 done tasks kept, and the trivial
     job ran as 361, Success.
   A reboot test is still owed on both. purple got the same fix at 18:30: pid 1004140 (pixi-global pueued) became 3449891 (environment binary), with 19 done tasks kept, and a crontab entry.

## Found in the cutok v26 r2 re-acquisition, 2026-10-11

### Mainboard

1. **`job submit --job` ignores `--walltime`.** A single-job submit to crimson wrote 00:30:00
   into the job script although `--walltime 00:07:00` was passed; batch specs set walltimes per
   job and are unaffected.
   FIXED on fix/mb-bugs (474bd73): a `--job` batch takes the resource flags (`--walltime`, `--queue`, `--gpus`, ...) as
   every job's defaults, and `--batch spec.toml` refuses them by name since the spec sets them per job.
2. **crimson's pueued runs from a deleted path, and `pueue` is not on a plain ssh PATH.** The
   daemon's working directory is the removed `~/projects` mirror; dispatch still works, but a
   restart from that path would fail and `ssh crimson pueue status` does not resolve the binary.
   Diagnosed 2026-10-11: the daemon's cwd is `/home/crimson`; its binary is the deleted
   `.mainboard.aside-r5/.../pueued`; the queue is empty (1935 done, nothing running or waiting).
   The PATH half is by design (see the first section, items 2 and 3). Restart pending a gate-free
   window (first section, item 7).
3. **A job referenced by a module-name string ships without that module.** The serving profile
   launched `python -m experiments.serving_gh200.pressure`, which no static import reached, so the
   closure lacked it and the job failed on the host (fixed in the node, 1989cbb79, by importing
   the module). The closure walker cannot see `-m` strings; M12's declarations (`imports=`) or a
   refusal of unresolvable `-m` targets at submit would catch it before any queue wait.
4. **`job list --json` can print nothing at all while settlement is busy.** A watcher reading it
   took the empty output as every job ended; consumers must treat an empty listing as unknown.
5. **Settling passes outlive their 15-minute deadline.** The agent settling the GH200 r2 batch
   saw passes run past the bound that 13m's fix sets, so a held lock again makes every listing
   answer from the cache. Check whether the bound covers the periodic launchd pass as well as an
   interactive one, and whether a pass blocked in a native DuckDB scan ever reaches the check.
   Second witness (RTX 4090 settlement, 10-11): an agent's `job list` was killed at its ssh
   timeout and left an orphan pass holding the settlement lock for 15 minutes, so parent death
   did not end it either.
   Third witness (10-12 08:25): two orphaned `ssh pedro-cvlab sh -c 'p=".../bin/python" ...'`
   remote commands (parent launchd) had been alive for 7 and 8 hours, children of passes that
   ended. A pass's remote commands need their own bound (`ConnectTimeout` covers only the
   connect) and must die with the pass.
6. **An interrupted evidence copy leaves a finished job blocked until some later pass.** Job
   3525659 (miyabi-g, exit 0) reads `blocked: settlement pending; remote evidence retained:
   [Errno 32] Broken pipe`. Its rows reached the node's report, so nothing is lost, but the job
   stays blocked until a full pass copies its evidence again, and that must happen before miyabi
   stops on 10-28. A broken pipe mid-copy should be retried within the same pass, with a bound.
7. **`mainboard ci` leaves the workspace environment behind PATH entries already ahead of it.** On
   the center, the login PATH starts with `~/.pixi/arm64/bin`. So `cargo` in mcmr's
   `extension-clippy` step resolved to the pixi-global trampoline, which sets
   `CONDA_PREFIX=~/.pixi/arm64/envs/rust`. PyO3's build script then looked for
   `$CONDA_PREFIX/bin/python` and failed (10-11). With the environment's bin prepended, the same
   step runs. `ci` should run its steps in the activated workspace environment, as `run` does, and
   not in whatever PATH the caller has.
8. **A member's manifest edit reinstalls the live environment under running processes.** At about
   21:40 on 10-11, fast-forwarding `packages/patos` main changed its `pyproject.toml` (ty floor
   0.0.59 to 0.0.86). The next `mainboard run` re-locked (the `mb.lock` refresh of f273f1495) and
   reinstalled the default environment in place, and every concurrent process saw `mb` and
   `patos` vanish:
   - an agent's `job show` failed with `importlib.metadata.PackageNotFoundError`;
   - a gate suite had 130 failures.
   An implicit re-lock and reinstall should never replace an environment that running processes
   use. Two fixes are possible: build the new environment beside the old one and swap it
   atomically (a generation directory and a symlink, as M14's recoverable generations do for
   edits), or refuse with a message naming the processes using it. Until then, change a workspace
   member's manifest only in a quiet window.
9. **The center test suite leaks System V shared memory until PostgreSQL cannot start.** The
   killed-server tests leave 1 to 3 orphaned 56-byte segments per run. macOS allows 32
   (`kern.sysv.shmmni`), so after about ten runs `initdb` fails with `shmget: No space left on
   device`. Codex's "sandbox denies shm" was this exhaustion (33 segments from 14:43 to 14:53 on
   10-11).
   - Workaround: a sweep of segments owned by the user with no attachments and a dead creator
     (scratchpad `commit/sweep-shm.sh`).
   - Fix: the fixture that kills a server should remove its segments (`ipcrm` for the
     postmaster's key) or stop the server with a fast shutdown, so the gate never depends on a
     global, sudo-only limit.
   FIXED on refactor/mb-integration (ci36, merged 8910fb3): a killed test postmaster marks its
   own segment for deletion first. The isolated suite left 0 segments.
10. **Nothing stops a registered acquisition on a degraded host link.** From 07:50 UTC on 10-11,
    pedro-cvlab's RTX 4090 and its root port sat at PCIe gen 1 (2.5 GT/s) under load. Pinned
    32 MB copies fell from 18.4 to about 1.9 GB/s, and 19 registered timing jobs ran to "ok" on it.
    Admission checks only utilization and other processes; the facts are probed at setup.
    - Proposed: before timing, the acquisition fixture runs one pinned copy of the cell's size.
      It reads the trained link (sysfs `current_link_speed` or NVML during the copy) and refuses
      below a fraction of what the slot and card both support, recording the reading in the
      receipt.
    - The refactor's PCIe facts (refactor/mb-pcie) already compute the ceiling and the 80%
      finding, so the gate reuses them.
11. **`mainboard run` mis-parses `--` when several test files are given.** The E-stages
    settlement agent (10-12) passed two test files and pytest flags after `--`. Part of the line
    reached pytest as paths, or the reverse. Reproduce with two node test files and `-- -q -x`,
    then compare against how `run` splits a single target's arguments (memory note
    mainboard-local-pytest).
12. **DuckDB's `shape` raises on an ordered, joined relation.** Calling `.shape` on a relation
    built by a join followed by `.order(...)` raised in a settlement helper (10-12). `count('*')`
    on the same relation works. If mainboard's `Results` or `Relations` helpers call `.shape`,
    they should count instead, or the case should get a regression test against the pinned
    DuckDB.

13. **The isolated gate's fixed 900 s budget fails a passing suite on a loaded machine.** On
    10-12, main plus the host-Python fix ran 534 tests in 827 s as plain pytest, all passing. The
    isolated gate killed the same suite at 51% after its 870 s suite limit, because a lake ingest
    and several agents shared the Mac. The suite grows with every package: integration's 1,108
    tests take 9 to 11 minutes when quiet. Scale the budget from the collected test count and the
    last passing duration, or make it a declared, reviewed constant raised with the suite. A
    budget miss should say "budget" in the report, not look like a hang.

14. **`lake evict` over a large evidence store runs for an hour with no output and no deletion,
    competing with settlement.** On 10-12 the lead evicted `current_engine_refresh/evidence`:
    332,328 files, 21 GB, every one already indexed, as `lake ingest` confirmed. After 75 minutes
    at about 300% CPU it had printed only its start line and freed nothing. Meanwhile a settling
    pass spent more than 12 minutes reading one small batch's receipts from the same lake disk,
    which held up a GPU experiment. The lead killed it by PID: nothing was set aside, nothing
    lost.
    - evict should report progress (files checked, removed, bytes);
    - it should remove in batches as it verifies, so an interruption keeps what it freed;
    - it should yield to a settling pass (the settlement lock) or run at low I/O priority.
    The ingest of the same paths took 1,294 s to index nothing new, so a fast path for "already
    indexed by path with the same size and mtime" would help both verbs.

15. **A fresh host cannot build the workspace's editable `packages/sqlalchemy` fork.** The host
    Python fix (fd32f38) got a vast hold past the interpreter step, then `pixi install` of `cutok`
    failed building the fork: setuptools `attr: sqlalchemy.__version__` raised
    `ModuleNotFoundError: sqlalchemy` (10-12, $0.02, released). Existing hosts carry an old
    build and never rebuild it.
    - Locally, `uv build --wheel packages/sqlalchemy` refuses the fork's `pyproject.toml`:
      `duplicate normalized extra name mssql-pymssql` (line 46).
    - Either problem stops every first-time host and every rental.
    - Open questions: whether the fork's extras and the `find where = ["lib"]` with an `attr:`
      version work under the newest setuptools and uv; whether the mirror ships `lib/` to a fresh
      host; and whether the default and cutok environments need the fork editable at all, or a
      built wheel from the fork's branch.

16. **`mainboard run` resolves a passthrough command's relative paths at the monorepo root, not in
    the caller's directory.** On 10-12 the lead ran `mainboard run ruff check --select I --fix .`
    from `packages/mainboard`, meaning that package. It ran at the root and sorted imports in
    4,500 files across 60 repositories, among them vendored paper sources, two forks and
    research/llm. They were restored file by file (an isort-only check against HEAD), except 7
    vendored googletest files whose tree already differed from its commit. A passthrough
    command should run in the caller's working directory, as every shell tool does. Or `run`
    should refuse a mutating tool given `.` outside the root.

## Found setting up pedro-home-wsl, 2026-10-11

### Mainboard

1. **A fresh host's `host setup` fails once CPython 3.15 exists.** `Bootstrap.python` runs
   `uv python install --no-bin '>=3.14'` (`dispatch/onboard.py`, `Dialect.requires_python` from the
   package's `Requires-Python`), and with nothing installed uv 0.13.0 downloads the newest match,
   3.15.0. The next step, `uv tool install --managed-python --python '>=3.14' --editable
   packages/mainboard`, then fails: `cuda-bindings>=13.0` has no cp315 wheel (v13.4.3 ships
   cp310 through cp314). Every host set up for the first time from now on hits it; hosts that
   already hold a 3.14 keep working, since uv reuses an installed interpreter. Worked around on
   pedro-home-wsl by installing the tool once onto `--python 3.14`, deleting the 3.15 folders
   under `~/.local/share/uv/python`, and rerunning `host setup`. The fix is an upper bound where
   the interpreter is requested (or on `Requires-Python`) until cuda-bindings ships cp315.
2. **A failed tool install shows only uv's first line.** The refusal printed
   `error: No solution found when resolving dependencies` and dropped the `cause:` and `hint:`
   lines that name the real reason (here the missing cp315 wheel); the reproduction took a manual
   `uv tool install` over ssh.
3. **A host's memory is read once, so a raised WSL cap keeps printing the old size.** After the
   owner set `memory=` to 8 GB, `host list pedro-home-wsl --facts` re-probed and read 8,321,191,936
   bytes, but the line every `job submit` prints still read `4 GB RAM`, from `sysmem_gb` in the
   capabilities recorded at `host setup` (`host sync` probes nothing). The submit line and the
   queue-policy check can rest on a size the host no longer has; a facts probe should refresh the
   recorded capabilities, or the submit line should read the facts it just printed.
4. **`host sync HOST` provisions the profile's `default` environment only.** On pedro-home-wsl the
   cutok jobs enter `env = "cutok"`, and a plain `host sync` printed `installing default ... from
   the shipped lock` and left the cutok environment on the lock of the setup three hours earlier;
   `host sync HOST --env cutok` is the call that moves it. A submit without `--env` builds and
   runs `default`, so a diagnostic taken that way measures a different environment from the arm's
   jobs without saying so. A job spec that names an environment should make `host sync` (or the
   submit's own check) say which environment the host's lock is behind on.
5. **`job show --wait` prints one `settlement busy` line every five seconds while another session
   settles.** With a settling pass held for ten minutes by another process, a 240 s wait produced
   about fifty identical lines (the same holder, phase and deadline) and the cached `running` row
   until the holder finished. One line per holder change would carry the same information.

## Found building `host hold` for PBS, 2026-10-09

### Mainboard

1. **`mb host sync miyabi-g` cannot run.** Its first step, `uv python install --no-bin '>=3.14'`,
   fails with `No download found for request: cpython->=3.14-linux-aarch64-gnu`: uv has no
   aarch64 CPython 3.14 build. The sync-only path should skip the managed-Python install when
   the host's environment already runs a satisfying interpreter. Worked around by calling
   `Dispatcher.mirror(plan, root)` directly, which is all a code-only update needs.
2. **`Pbs.cancel` ignores `qdel`'s exit status** (`dispatch/schedulers/pbs.py`), so a refused
   cancel reads as done. `Line` checks its own `qdel`; `Pbs.cancel` should raise unless the job
   had already finished. FIXED on `refactor/mb-M02`, not yet switched live (M02, B16): `Pbs.cancel`
   and `Line._stop` share one checked `qdel`, which raises on a refusal and on a lost
   connection. The same package fixes B04 (`host release` tombstoned the running job and killed it:
   a release now fences only the entries no allocation claimed), B13 (`Held.cancel` ignored a
   failed fence write) and B18 (pueue Locked and Stashed tasks read as failed). A cancel its
   scheduler refused now raises before `Verdicts` records `cancelled`. `Slurm.cancel` and
   `Local.cancel` still acknowledge unchecked, and a host that does not answer still settles as
   `cancelled` with a note that nothing was stopped there.
3. **`snapshots.queued_here()` only asks the local pueue daemon.** A pinned tree that a queued PBS
   job, or a held job waiting in the spool inbox, will run from is invisible to pruning once the
   age thresholds pass. Include `claims/` and `inbox/` entries of the spool (their `cwd`) and the
   scripts of queued PBS jobs. FIXED on `refactor/mb-M03b`, not yet switched live (M03, B05):
   each pin writes a retention record before launch, and pruning keeps whatever a record, pueue
   task, spool entry, staged script or registry run names, and everything when one cannot be
   read. Records protect the existing `dispatch/sources` and `prefixes/<environment>` roots in
   place, so no host needs a sync for a new address and no environment is rebuilt; trees pinned
   before the switch hold no record and age out under the same inventory. Retention assumes one
   release prunes a host at a time, which the drain-and-switch cutover guarantees: until a host
   changes release, its old `provide` prunes without reading the records. Nothing retires a pin
   yet, so pinned trees and prefixes grow until the submission authority lands. Also
   fixed: `Prefixes.prune` listed the environment names under `prefixes/` instead of the digests
   under one environment, so a third environment on a host lost its whole directory.
4. **`mb shell --on miyabi-g` knows nothing of a held line.** PBS refuses the second interactive
   job with its raw quota text. `Board.interact` could read the spool's `line.json` and say who
   holds the slot and until when.
5. **`login_run` had no deadline**, so a responsive ssh over a hung Lustre read waited forever
   (the review's point 11). Fixed: `login_ask` bounds every probe at 90 s and raises
   `HostUnreachable`.
6. **The workspace ruff (newer than the package's pinned one) flags `Iterator` on context
   managers** in `dispatch/agent/program.py`, `render/human.py` and `trials/flags.py`; the
   package's own `ruff` passes. Align the two versions.
7. Held jobs get no walltime from a queue, so the runner now enforces it under PBS too
   (`jobs/spec.py`); a PBS queue that kills first never lets the runner's deadline fire, but the
   `JobSpec` docstring still describes the old split.
8. **`qsub -I -- <command>` needs an absolute executable.** `pbs_mom` execs the command itself
   with a bare PATH (`/bin:/usr/bin:/usr/local/bin`), so `-- bash script` ends "exec of bash
   failed" while `-- /bin/bash script` runs on the node with a pty as stdin (measured 2026-10-09,
   job 3517082). A tmux pane started by a long-lived server also has no `/opt/pbs/bin` on PATH
   (`zsh: command not found: qsub`), so the keeper runs under `bash -l`. `Pbs.interactive`'s
   docstring still says PBS takes no command; it does, absolutely spelled.
9. **A held submit costs 70-80 s of dispatch** (mirror of a few paths, two login-shell
   activations in `_verify`, the pin) where the old spool took a second. Measure which step
   dominates; a hot line wants a cheaper path for a repeated command.
10. **Miyabi's `qstat` wrapper caches.** A job shows `RUNNING` with a frozen ELAPSE for minutes
    after it ended (3517146 read `00:04:59` after the keeper logged its end; 3516531 stayed
    `01:58:42` for four minutes). `Held` asks PBS only about allocations whose claims have no
    exit artifact, so it is safe, but any wait on a PBS state lags by that cache.
11. **PBS refuses a new `qsub -I` for about 30 s after the previous interactive job ends**
    (`exit 39`, the `njobs_int-g` count). A line's renewal gap is therefore the keeper's retry
    interval (10 s) plus that; measure the real gap before promising continuity.

## Found during the cutok GH200 transfer, 2026-10-07

### Fixed locally, not pushed (mainboard `ef2809d`, root `d3dda40be`)

1. **Every trial's receipt failed to serialize.** `Relations.rows` staged rows as NDJSON and let
   `read_json` infer types, so strings holding an ISO instant, a date, a clock time or a UUID
   became TIMESTAMPTZ/DATE/TIME/UUID in the receipt Parquet. They read back as Python objects
   (instants in the reader's zone, JST on the center) and `ledger.wire` raised at pytest session
   close. Fix: text-valued keys are pinned to VARCHAR. Follow-ups:
   - `state.relations.records` is annotated `JsonValue` but returns raw DuckDB values (datetime,
     UUID, Decimal). Decide: convert there, or change the annotation and audit every caller.
   - Parts written between 840a4ab and the fix hold TIMESTAMPTZ; unioned with new VARCHAR parts
     they read as `2026-10-07 19:00:00+09` text. Find and rewrite or retire them.
   - Other typed `read_json` uses (`results.py` jobs/events, `lake.py` cells) infer the same way.
2. **`mb job show --wait` crashed** with `AttributeError: dropped`: `schema.latest` assumed every
   log table has a `dropped` column; `pulse` does not. Fix: filter drops only where present.
3. **DuckDB settings had no home** (owner request). Done: `state.database.connect` is the one
   factory; `[duckdb]` in `mainboard.toml`, `MB_DUCKDB_<NAME>` from `.env` or the environment,
   UTC default. Follow-ups: `Lake.attach` still forces `TimeZone`/progress bar by itself (by
   design for lake reads, but document it in `mb help`); check `mb query`'s own connection uses
   `connect`; consider exposing `mb doctor` output of the effective settings.
4. Release: mainboard's `publish.yml` runs on every push to `main`. Pushing `ef2809d` needs the
   owner's go-ahead and probably a 0.5.2 bump; the root commit `d3dda40be` points at it, so the
   root cannot be pushed first.

### Mainboard

5. **`__pycache__/*.pyc` under `packages/mainboard` enter job source closures**, so the seal digest
   changes whenever the center runs `mb` between dispatches (`4328e3…` vs `fd9931…` differed only
   by .pyc files). `GitignoreFilter` does not drop them for that submodule. Seals should hash
   source only.
6. Editing any closure file while a dispatch pins sources is refused with `source blob mismatch`
   (correct), but the message could name the file's change time and suggest re-running.
7. **`mb upgrade` raised cutok's Python to `>=3.15.0rc3`**, a pre-release, and still left the lock
   unchanged. Upgrade should skip pre-releases unless asked.
8. **`mb upgrade` has no `--dry-run`**, which `pixi upgrade` has; "mb verbs are pixi's" says the
   flags should match.
9. **`mb lint <dir>` rewrites files the change never touched**, including frozen analysis sources
   (`lifecycle_figures.py`, reverted by hand). Directory lint should check, not fix, untouched
   files, or refuse to rewrite sealed sources. 2026-10-08: it also rewrote
   `cutoken/src/cutoken/types.py` (`from numba import cuda as cuda` to the plain form) although
   cutoken's own `pyproject.toml` exempts that file from PLC0414, because the alias is the
   explicit re-export ten kernel modules import. Lint must honor each package's nearest ruff
   config (hierarchical discovery), not force the root's.
   2026-10-09, pyrefly: `[lint.tools.pyrefly]` runs `pyrefly check .` in each owner, and an
   explicit path bypasses that owner's `project-excludes`: cutoken reports ~500 errors from its
   device modules where its configured run (`pyrefly check`, no path) reports 1. Dropping the `.`
   everywhere is wrong too: an owner without a pyrefly config (`scripts`) then climbs to an
   ancestor config and checks 899 errors' worth of other code, where `.` checks its 0. Fix in the
   runner: no path where the owner's own pyproject has `[tool.pyrefly]`, `.` otherwise (ty
   applies its excludes either way: 682 both forms in research/cuda-tokenization).
   FIXED locally (unreleased): a tool's `path-when-unconfigured` is appended to its command in an
   owner whose own `pyproject.toml` has no `[tool.<name>]` table; the root `mainboard.toml` sets
   it to `.` for pyrefly and drops the `.` from the command. `mb lint --check --only pyrefly` on
   a cutoken device file reports nothing, on a `scripts/` file with a type error reports it as
   before.
10. **Shipping `needs` to miyabi ran at ~2.4 MB/s** through the agent's ssh pipe: 2.1 GB of corpora
    took about 15 minutes per first dispatch. Compression, rsync, or a staged copy would help.
10a. **Collecting from miyabi times out.** A `job list` settle pass on 2026-10-07 22:20 gave up on
    3500519, 3500521 and 3500602 with "ssh collect to 'miyabi-g' timed out after 600s": the
    same slow link as issue 10, now on the way back. Large evidence needs a resumable transfer
    or a longer, size-aware deadline. Root of it: every cutok job's `fetch` is the whole
    `datasets/experiments/current_engine_refresh` folder, 9.4 GB in 78,000 files on miyabi by
    2026-10-08, so one collect pass must sync all of it and a failed pass leaves the host
    "quiet" for the rest of the pass. Settlement then stays pending (cold-native 3500492 ran
    12 of 12 cases but none reached the center). Worked around with a manual `rsync --partial`
    of that folder. Fetch should name a job's own run directories, not the experiment's.
    Fixed locally (unreleased): collection counts paths the lake indexes as known, so an evicted tree no longer
    pulls the whole folder back; the remote side still hashes every known file on the login
    node, which is what makes a pass slow. FIXED locally (2026-10-09, unreleased): `pack.py`
    keeps a remote digest cache (path, size, mtime) under the workspace's `run/`, seeded once
    from a compute node (170,011 files, 20 GB), and runs on the host's workspace-environment
    Python, never the login node's OS `python3`.
    Measured 2026-10-09 on a GH200 node over 182,903 files: the scan cost 2 min of per-file
    metadata (lstat 49 s, resolve 35 s, stat 35 s; now one `lstat` per file through `scandir`,
    containment checked once, `known` sent zlib-compressed: 133 s, 16 MB), and a pass moved
    9.6 GB: 71,701 files under `evidence/artifacts/<run>/<trial>/objects/<sha256>`, each run
    holding its own copy of objects the lake already kept (cutok's artifacts referenced 59.7 GB
    of which 4.1 GB was unique). The 10-09 backlog was moved once as an archive (rsync) and merged.
    FIXED locally (2026-10-09, unreleased, owner-approved):
    - Artifacts by reference: a trial's bytes go once into its node's content store,
      `<node>/evidence/objects/<sha256[:2]>/<sha256>` beside its receipts, shared by every run on
      the host; the receipt references it (digest, size, media type, and `source`, e.g.
      `hf://<repo>@<revision>/<file>`, read from a hub cache path or passed to `log.artifact`).
      Per-run `objects/` of older runs read as before.
    - A host sends no file whose digest the center's lake keeps: `pack` names it in
      `mainboard-held.json` and `Evidence.adopt` indexes its path from the blob. An unchanged
      live event log is not sent again (its `collected-<start>-<end>` name is known).
    - Members are zstd level 3: 1.5 GB of cutok evidence went 3.9x smaller for 7.5 s of one
      GH200 core (level 1: 3.5x, 3.7 s; level 6: 4.3x, 16.6 s).
    - `mb lake dedup` hard-links identical settled files to one copy (verified digests, runs
      quiet for an hour, nothing deleted). 10-09: miyabi 24.4 to 2.4 GiB of cutok evidence
      (104,328 names linked, 225 s on a GH200 node), pedro-cvlab 3.78 to 0.62 GiB, crimson and
      gold 81 MB to 80 MB.
    - Collection keeps each transfer in the lake and writes nothing to the tree; it stages in
      `<out>/tmp/collect-<pid>-*`, and a dead pass's staging is cleared when the next one starts
      (31 root-level `.mb-collect-*` dirs, 7.2 GB, had piled up). A trial session on the center
      moves its run into the lake (kept, then evicted); a dispatch mirror's staging lake keeps
      nothing.
    - A settle pass collects each host's fetch path once, not once per run: the 10-09 pass
      before the fix collected the same folder from miyabi 24 times in 2,798 s; the next pass,
      with it, took 147 s with one collection (76 s alone, of the 600 s deadline; the earlier
      pass had already moved everything, so nothing new crossed).
    Still open: a job's fetch still names the whole experiment folder, whose remote scan
    (~50 s on miyabi's login node) is most of a collection now.
11. **Settlement backlog**: `job list` keeps erroring on old runs ("settlement pending; remote
    evidence retained: result transfer failed" for pedro-cvlab 348, 363, 400, 446, 479 and
    crimson 1862) and on today's 594 and 3500391 ("Expecting value: line 1 column 1"), whose
    receipts were never written (issue 1). Settle or retire them explicitly.
12. DuckDB prints a nanobind leak report at interpreter exit in every job log ("leaked function
    fetchdf ..."): relations or connections alive at exit. Close them, or it hides real errors.
13a. **Reading a cohort's artifacts back is slow on the center.** Re-rendering the admitted v18
    available3 roster report spent about 24 s per process in `Artifact.read`, so 126 processes
    took over 50 minutes before the audit started. Measure where the time goes (lake lookup per
    object, decompression, hashing) and batch it.
13b. **Materializing a whole experiment's datasets** wrote 9.6 GB without reaching the one file a
    renderer needed, on a center disk 93% full. Renderers should read through the lake, or say
    exactly which paths to materialize. Since 0.5.4 `mb lake evict` gives the disk back after a
    materialize, but renderers that read through `Path` still need the tree.
13c. **The center's disk is nearly full** (2026-10-08 03:20: 12 GiB free of 460, 98%). The lake
    keeps 48 GB under `.mainboard/lake` on the internal disk, and tree copies under
    `research/cuda-tokenization/datasets` hold 29 GB, much of it duplicates of lake objects.
    There is no `mb lake evict` to drop tree copies the lake verifiably holds, and no way to see
    which tree files are cached copies. Add both, and consider the lake on `/Volumes/PORTABLE`.
    Done 2026-10-08: the lake moved to PORTABLE (0.5.3, 52 GiB free) and `mb lake evict`
    (0.5.4) dropped 151,166 cutok tree copies (87 GiB free). Still missing: a listing of which
    tree files are cached copies.
13d. **`[workspace] lake` (2026-10-08) is not yet known to center migration.**
    `center/state.py` ships `lake.sqlite` and the data as part of `.mainboard`; with the lake
    elsewhere, `host setup --center` should carry it from `Lake.home` (or leave an external
    volume where it is). `doctor` should also report the lake's home and refuse clearly when the
    volume is unmounted.
13f2. **An unresponsive lake volume hangs every verb, without a timeout** (2026-10-08 ~10:00).
    With `[workspace] lake` on `/Volumes/PORTABLE` and that drive mounted but not answering I/O
    (even `ls` blocks), `mb host sync <host>`, `mb lint` and three concurrent syncs all sat at
    0% CPU for over ten minutes after loading the DuckLake and SQLite extensions. Lint and sync
    should not need the lake at all unless they restore files; where they do, opening the lake
    should time out and name the volume, and `doctor` should report it.
    Cause found: macOS privacy (TCC) denies Full Disk Access to processes Ghostty is
    responsible for, so opening a file on the external volume waits forever while `stat`
    still answers. FIXED locally (unreleased): the manifest consults held rentals only for an
    alias neither declared nor `local`, so sync, lock, lint and run never open the lake.
    2026-10-09: the same wait hit every pytest trial started inside tmux (the agent sessions):
    the tmux server is its own responsible process and holds no Full Disk Access, so attaching
    the catalog never returned. FIXED locally (unreleased): `Lake.attach` reads the lake's home
    once per process first and raises after 10 s, naming the missing access. Granting tmux the
    access would restart the terminal, so agent sessions run lake verbs through `ssh localhost`,
    whose sessions hold Full Disk Access (Remote Login allows it); the message says so.
    2026-10-10: the launchd settler (`job list --every 20m`) met the same refusal on every pass,
    since a launchd agent holds no Full Disk Access either. FIXED locally (unreleased): the agent
    runs its pass over `ssh -o BatchMode=yes localhost`, and installing refuses while that login
    fails.
13f3. **Every job host inherits the center's absolute `[workspace] lake`** (2026-10-08).
    Since the lake moved to `/Volumes/PORTABLE/mainboard-lake` (9ffcc472d) the shipped manifest
    names that path on every host, and a pytest trial there dies in `pytest_configure`
    (`trials/provenance.source` -> `dispatch/provenance.archive` -> `Lake.ready` -> `create`)
    with `Unable to open database "/Volumes/PORTABLE/mainboard-lake/lake.sqlite"` on Linux.
    The key names the center's lake only: sync should ship the manifest without it (a host keeps
    its staging lake in its generated directory, as before), or `lake_home` should apply it on
    the center alone. Registered trials on hosts are blocked until then.
    FIXED locally (unreleased): `lake` applies only on the center, the checkout holding
    `.git`; a dispatch mirror ships without one and keeps its staging lake.
13f5. **`mb lake ingest <relative path>` fails at its end** (2026-10-09). `mainboard lake ingest
    research/cuda-tokenization/datasets` (run from the workspace root through `ssh localhost`)
    raised `ValueError: 'research/cuda-tokenization/datasets' is not in the subpath of
    '/Users/pedro/Developer/projects'` from a `relative_to` on the unresolved argument, after the
    ingest windows had committed (the 611 files were indexed with intact blobs; a following
    `evict` verified and dropped them). Resolve CLI paths against the workspace before any
    `relative_to`, and record the import run.
    FIXED locally (unreleased): `ingest` resolves its paths before it files them, so the import
    run is recorded under the workspace-relative source; `evict`, `materialize` and `dedup` share
    `state.evidence.within`, which resolves a path (relative to where `mb` runs) against the
    workspace and refuses one outside it with a `MissionError` (`dedup` raised a bare
    `ValueError`). `test_every_lake_verb_takes_paths_relative_to_where_it_runs` covers all four.
13f4. **Stale generated inputs on a host break every dispatch's environment pin** (2026-10-08).
    The compiled artifact ships as named files, so a file the center stopped generating stays in
    the host's `.mainboard/envs/default` forever. `GeneratedFiles.inputs` digests every file
    there, so miyabi-g's `package-lock.json` (2026-09-29, from before the pnpm second stage)
    and `activation-windows.json` (2026-10-01) give `df134abbb018105e` against the center's
    `e5b7ed78`, and `job submit` refuses: "could not build default ... describes environment".
    `host sync` and `host setup` both pass, since neither removes them. Fix: prune generated
    inputs the center no longer ships when the artifact lands; until then, delete them by hand.
13e. **Palettes are only validated by hand.** The 2026-10-08 Hugging Face colour change came from
    running the dataviz skill's `validate_palette.js` outside mainboard. `[plots.*]` colours and
    palettes should be checked (lightness band, chroma, colour-blind and normal-vision
    separation, contrast) when a style loads and in `mb lint`.
13f. **Hashing could move to DuckDB with a fast hash** (owner idea, 2026-10-08). xxh3 as a change
    detector for evict, collection and the KeptDigests cache, with SHA-256 kept as the identity
    every receipt, seal and lake object is addressed by. Measure first: evict hashed 151,166
    files in 36 s, while materializing one report folder took over 600 s, so hashing is not the
    current bottleneck. The remote `pack.py` must stay standard-library only. Source: DuckDB's
    `hashfuncs` community extension (Query.Farm) has `xxh3_64`, `xxh3_128` and `xxh3_128_hex`,
    the last matching Python `xxhash.xxh3_128().hexdigest()`; it needs `INSTALL ... FROM
    community`, while `Relations` disables extension autoinstall.
13g. **`mb lake materialize` reads one object per query** (2026-10-09, v25 settlement): about 10
    files/s, so 69,000 files projected past 2 h, and eight parallel runs gained nothing (load
    average above 90). A windowed `Blobs.read` restored the same files in about 2 min; reuse
    `replicate`'s windowed read. Since `Artifact.read` already falls back to the lake by digest,
    an audit needs no materialized tree at all, only pinned path readers do.
13h. **A bare `mb query` over the whole `trials` or `artifacts` view costs 3-9 CPU minutes**
    (2026-10-09). Filter by seal or project, or index the views by seal.
13i. **An audit that needs a pinned file absent from the tree fails after all its work**
    (2026-10-09): the v25 roster audit reached `bulk-baselines-20261006-v17/support-snapshot.json`
    last and wrote `admitted: false`. Resolve every pinned input (from the lake) before the audit
    reads anything.
13j. **A test whose fetch is a folder beside its evidence store never settles** (2026-10-09, cutok
    smoke 621, 631, 632 on pedro-cvlab). Every pass failed with "no lake around
    .../stage-probe-v26 holds 862d65c8...". The digest is one of the receipt's artifacts in
    `heterogeneous_pipeline/evidence/objects/`, and which one is named varies by pass, since
    `Artifacts.verify` walks a set. The declared fetch was
    `heterogeneous_pipeline/stage-probe-v26`, and a sealed snapshot links only the results path
    back to the mirror. So the trial wrote its
    store (receipts, the `run` record and every object the receipt pins) inside
    `.mainboard/dispatch/sources/<key>/` on the host, which nothing collects. Collection was
    fine, and nothing was held or rerun: the lake indexes all 16 fetched files, and the missing
    objects sit only in the snapshots (`find ~/.mb-jobs -name '862d65c8*'`). FIXED locally:
    `Target.results` widens an empty fetch, or one inside a test's `datasets/experiments/<x>`
    home, to the home, and `Board.results` applies it to an explicit `--fetch` or batch `fetch`
    too. 621, 631 and 632 keep their recorded fetch, so they cannot settle verified unless their
    snapshot stores reach the lake. Do not redispatch an identical closure before then: its
    snapshot key is reused, and `__results` replaces the real `heterogeneous_pipeline` folder
    there with the link.
    Still open: a run that rewrites its results folder conflicts with what the lake already
    holds. 622 rewrote `stream-hypotheses-v26` after 621's first pull, so every pull of
    `heterogeneous_pipeline` is refused as conflicting until that is resolved. Write each run's
    output under a run-scoped folder, or collect before the next run of that test starts.
13k. **`mb upgrade` (2026-10-10)**, five defects fixed locally, one open:
    - It crashed on `[workspace] members` (`NonExistentKey: llm-head`), since the manifest model
      lists members the text never declares. FIXED: `ManifestText.versioned` answers False for
      an undeclared name.
    - A member the text names by version (`mainboard = "*"`, `patos = "*"` in every lean
      environment, `sqlalchemy` in `[python.deps]`) was raised to PyPI's release
      (`patos >=0.0.14, <0.0.15`), which the local member need not satisfy. FIXED: `upgrade`
      skips every member's distribution.
    - Open: `Index.pin` caps a date version at its own next day (`pdfminer.six >=20260107,
      <20260108`), which is an exact pin in effect. A date-like first component should get a
      floor alone, as an unreadable date stamp already does.
    - Open: `Index.latest` takes what one source lists, never what every platform can install.
      `Conda` asks `pixi search` for this machine's platform alone (qpdf 12.4.2 had no linux-64
      build yet), and `Python` reads the PEP 691 `versions` list, yanked releases included
      (pytest-timeout 2.5.0, cypari2 2.2.6, syrupy 2026.4.6.124150327040). Both made the raised
      manifest unsolvable until the floor was lowered by hand.
    - Open: a PyPI floor above what conda-forge builds is unsolvable when a conda package
      already brings it (`tokenizers >=0.23.3` beside conda's transformers, built to 0.23.1).
      Even when conda-forge does build it, `mb lock` keeps the conda package at its locked
      version and refuses the raised PyPI floor (`filelock >=4.1.0` against a locked conda
      4.0.12, 2026-10-10); `mb update --env default` moves it. A raise of a conda-provided PyPI
      requirement should re-solve that conda package.
    - Open: `--env X` alone narrows to `[envs.X.deps]`, since `searched` defaults the ecosystem
      to conda, and never reaches `[envs.X.python.deps]` or `[envs.X.on.*]`.
13l. **`mb lock` then `mb install` never resolves the second stage** (2026-10-10). `lock`
    returns before the second stage, and `install` resolves it only when its own call moved the
    pixi lock, so the regenerated `package.json` met `pnpm install --frozen-lockfile` and
    failed (`ERR_PNPM_OUTDATED_LOCKFILE`). `mb update <env>` (which always counts as moved) got
    through. `lock` should solve the second-stage locks too (`pnpm install --lockfile-only`), or
    `install` should resolve a stage whose inputs changed since its lock.
    - It raised `python` to `>=3.15.0rc3, <4`. FIXED: `_newest` takes the newest final release,
      and a pre-release only when nothing final is listed.
    - A raised entry kept its comment at its old character offset, so a longer pin broke the
      column. FIXED: `ManifestText.put` shrinks or widens the space before the comment by what
      the value grew, one space at least.
    - The manifest was written before the solve, so a failed solve left every `mb run`
      re-locking against a manifest that cannot solve until it was restored by hand. FIXED:
      `Dependencies.settled` puts the manifest and `mb.lock` back when the reload, solve or
      install fails (or is interrupted), as pixi does, for `add`, `remove` and `upgrade` alike.
      Since a failed upgrade now leaves nothing to fix by hand, `upgrade --frozen` (as `add` and
      `remove` have) writes the raised manifest alone, so the hand edits a raise needs join one
      later `mb lock`.
13m. **A held settlement lock makes `job list` answer from the cache without saying so**
    (2026-10-10). A `job list` started through `ssh localhost` outlived its killed ssh and held
    `run/settlement.lock` for 29 minutes, busy in a DuckDB scan of the lake. Every listing
    meanwhile printed "done: settling" with stale states (`sweep.running` null), so six ended
    GH200 jobs read as `finished` with verdict `running`, and `job show --wait` heartbeated a
    finished job until its timeout. The listing should name the holder (pid, start time) and
    say its states are cached. A pass should also stop when its parent session dies.
    PARTLY FIXED locally (unreleased, M34 bridge, 2026-10-10): the holder of the lock records
    pid, host, start, phase, heartbeat and deadline beside it; `job list` (and `--json`, and
    `--batch`) and `wait` name it and list cached states without asking a host; `sweep.running`
    is never null; a pass ends its process when its parent dies or after 15 minutes; the last
    pass that ran to its end is stamped apart from the service's own last run. Still open until
    M24 replaces the file lock with an advisory lock: a pass stuck inside a native call that
    holds the GIL, the processes a pass starts, and `job show`, which still reads cached states
    without saying so.
13n. **A settling pass costs minutes on a lake of small files** (2026-10-10). With 34,192
    Parquet files (49 GB) one pass took 164 s, 447 s of it system time spent opening files
    (`CachingFileSystem::OpenFile`, `ScanReadAheadJob`). Each settled run replays its receipt
    stream (`bus.replay()`), so the cost grows with runs times files. The pass should read each
    stream once and never scan blob files. `mb lake compact` (6.6 s) only took the count to
    32,054, so the files are blob data, which compaction does not merge.
    PARTLY FIXED locally (unreleased, M34 bridge, 2026-10-10): a pass replays each stream once
    and keeps its latest state and evidence lines in memory as it publishes, so 100 runs sharing
    a stream cost one replay instead of 300. Still open until M23 replaces the replay with
    durable cursors: that replay (`Journal.replay`, a scan of the events table) is unchanged, and
    what opens the blob files is not diagnosed.
    Second witness, 2026-10-12. Two passes in a row spent more than 13 minutes at about 450%
    CPU and 2.5 GB resident "reading the receipts of" one kernel-tier batch (44 MB, 4,485 files).
    A native sample showed DuckDB `FileOpener`/`OpenFile`/`sqlite_scanner` frames dominating.
    Each pass died at its 15-minute deadline, so the batch never settled and a GPU experiment
    stalled for about an hour. After `mb lake compact` (under a minute), the same batch settled
    and `job list --batch` answered in 3.2 s. So for now compaction IS the remedy for the events
    and data tables. The settler should compact on its own when the small-file count or a pass's
    duration crosses a bound, and a pass that hits its deadline twice on the same stream should
    say so.
13o. **Archive members are unreadable once eviction removes their archive parts** (2026-10-10).
    53,444 early-September cutok artifacts (396 MB, 1,415 runs, 15 nodes) have no standalone blob:
    they live inside the 1,631 parts of `datasets/experiments/artifacts.parquet/<xx>/part-N.parquet`
    (archived 2026-09-15, b3b8e554a; ingested as ordinary file blobs on 09-30). `ParquetArtifacts.read`
    opens `artifacts.parquet/<digest[:2]>/` only on disk, and `Artifact.read`'s fallback
    `EvidenceTree.recall(sha)` finds only standalone blobs, so after `mb lake evict` (10-09)
    dropped the tree copies every such read raised "no lake around ... holds <sha>". Nothing was
    lost: the lake, the replica and git (3fe1c3eb6^) hold the parts byte-identical, and
    `mb lake materialize` of the archive folder restored the reads. Fix: let the archive reader
    reach its parts through the lake (`EvidenceTree(parent).files(...)`), test it with the parts
    only in the lake, and make eviction refuse an archive container whose members readers address
    by path. Read-only audit and the 53,444-row list: scratchpad lakeaudit/ of 2026-10-10.
13. The miyabi agent bootstraps with the login node's OS `python3` (3.9) (`python3 -c ...
    mainboard-agent`). It works, but contradicts the rule that nothing runs on the OS interpreter;
    document it as the one exception or ship the agent's own interpreter.
    FIXED locally (unreleased, 2026-10-09): `Dialect.python` runs the env's Python, else uv's CPython.
14. A path wheel on one platform and an index wheel on another is refused ("one source per
    package"), so TokTier's released x86_64 wheel had to be pinned by path too. Fine, but the
    message could suggest that pattern.

### Hosts

15. **Dangling `~/.local/bin/pueue` → `~/mainboard-managed/...`** on pedro-cvlab and purple after the
    October 7 wipe; plumbum resolved it first and the 4090 submit failed with exit 127. Removed by
    hand on both (verified gone 2026-10-11, see "Fixed and verified"). Still open in code:
    `host setup` should detect and repair dangling tool links.
16. (fixed on the host, see "Fixed and verified": pedro-cvlab's daemon now runs from the live
    pixi-global binary and a boot unit exists. gold's and crimson's are the same defect, first
    section, item 2.)
17. `macmini` is a declared host that does not resolve from the Mac mini itself, so `doctor` always
    warns about the fleet. Mark the center's own entry as such, or drop it.
18. pedro-home still answers ssh and sits in `~/.ssh/config`, but left mainboard's hosts with the
    Windows drop. Decide whether to keep it as a plain ssh box or decommission it.
19. Miyabi announces a scheduled stop at 2026-10-28 09:00 JST, and short-g estimated start times of
    10–36 hours for 24 jobs requested with generous walltimes. Request walltimes near the expected
    runtime so jobs backfill.

### Cutok

19a. **TokTier's artifact cache is per host, and GH200 compute nodes are offline.** TokTier routes
    Qwen3, DeepSeek, gpt-oss and Llama 3 natively and needs `~/.cache/toktier/artifacts`;
    miyabi had only a partial cache, so GH200 TokTier input-size allocations 3500491/3500493
    failed at `qwen3-100` with `ArtifactNotFound ... fetching is disabled`. The `72-0-gpt2` gate
    could not catch it, since TokTier routes GPT-2 through HF. The 4090's exact cache (22
    entries) was copied to miyabi and checked file by file. Host setup for an environment
    should stage such caches, and a gate should cover one natively routed model.
20. Analysis renderers were hard-coded to the RTX 4090 (host, UUID, 32 workers, titles). Being made
    machine-aware for v22 (`machines.py`); `bulk_figures.py` (v15) still is.
21. Contract tests error on hosts without the center's datasets (the support pin lives under
    `datasets/`), and `test_scaling_figures.py` cannot import in the cutok environment (no
    seaborn). They only pass on the center.
21a. Three `test_scaling_figures.py` contract tests fail for the input study on the committed
    renderer too (`test_full_grid_and_derived_outputs[input]` expects 3 plots and gets 9, plus
    `test_route_plot_uses_process_observations[input]` and
    `test_late_route_appearance_has_zero_share_in_prior_calls_and_processes[input]`). The
    admitted v20 report draws nine figures (three per document length), so the test fixture is
    probably stale. Check and fix the tests, not the renderer.
22. Pre-existing type errors in `current_engine_refresh`: `os.sched_getaffinity` on macOS,
    incompatible mixin overrides in the TokTier classes, `Log.metrics` given a tuple.
23. TokTier publishes only an x86_64 wheel; the GH200 uses a source build of the v0.2.9 tag
    (`research/cuda-tokenization/wheels/README.md`). Ask upstream for aarch64 wheels.
24. **A pinned upstream input was edited by an end-of-file fixer.** Commit `e72cacd1a`
    (2026-09-25, "retire the lint hooks") gave GPT-2's `encoder.json` under
    `current_engine_refresh/inputs/tiktoken/` a trailing newline. RTX 4090 job 557 (Oct 6) and
    GH200 allocation 3500494 (Oct 7) refused it before measuring; Oct 6 only restored the live
    file, so the committed blob kept the newline and this morning's CRLF rewrite from the index
    brought it back. Fixed in `b5e18d458` (exact bytes, `-text`, lint exclusion
    `**/experiments/*/inputs/`). Still to do: find which tool added the newline, and hash-check
    every other pinned input and resource in the workspace against its registration.

## Fixed and verified

### 2026-10-11, hosts (no sudo anywhere; killed no process, touched no other user's)

- **pedro-cvlab, pueued without boot persistence.** Added `~/.config/systemd/user/pueued.service`
  (pixi-global `pueued` by full path, `ExecCondition` skipping it while a daemon answers), then
  `systemctl --user daemon-reload` and `enable`, not started. Lingering was already on
  (`loginctl show-user pedro -p Linger` read `Linger=yes` before any change). Evidence:
  `systemd-analyze --user verify` exit 0; `is-enabled` reads `enabled`, `is-active` `inactive`;
  the condition exits 1 while the daemon answers (the unit would skip) and 0 with
  `XDG_RUNTIME_DIR=/nonexistent` (it would start); the running daemon, pid 12643 up since 15:37,
  kept its pid and its running task through every step. A reboot will test the one thing not
  shown: that the user manager starts the unit at boot and the queue it restores resumes.
- **pedro-cvlab, items 15 and 16.** `~/.local/bin` holds no `pueue` link, and `command -v pueue`
  resolves `~/.pixi/x86_64/bin/pueue`. The running daemon's exe is
  `~/.pixi/x86_64/envs/pueue/bin/pueued`, not deleted, cwd `/home/pedro`. No switch is owed. Its
  `/tmp/ab-r2` does not exist.
- **gold, A/B leftovers.** `/tmp/ab-r2` (164 MB, six trees) deleted after unlinking the twelve
  `.mainboard` and `data` symlinks into the live mirror (`find -type l -delete`, then none left,
  then `rm -rf`); `~/.mb-jobs/.mainboard` and `data` listed intact afterwards. Nothing referenced
  it: `grep -rl ab-r2` over `envs/`, `pins/`, `prefixes/`, `run/` and `dispatch/` found no file, and
  `envs/default/state.toml` already read `compiled_at = /home/pedro/.mb-jobs` (`cutok` reads the
  center path, as it does on pedro-cvlab: the shipped artifact). `ssh localhost "cd
  ~/Developer/projects && mainboard proc timeout 2400 mainboard host sync gold"` took 24.5 s
  (70 of 14774 paths, `installing default ... from the shipped lock`), re-shipped `envs/default/
  pixi.toml` and `state.toml` at 17:55:21, and every `doctor` finding passed. `cd ~/.mb-jobs &&
  ~/.local/bin/mainboard run python -c "print(6*7)"` printed 42, exit 0, in the mirror's environment.
- **gold's lake is legitimate and was kept** (first section, item 4): a code path creates it
  (`provenance.archive`), pedro-cvlab and crimson hold the same, and jobs of 09:48 and 11:23 wrote
  into it after the A/B run.

## 2026-10-12 — RESOLVED on refactor/mb-envfix: per-env overrides + update with editable members

- **Per-env PyPI dependency-overrides** now reach a `no-default` environment. `pypi_options`
  (`engines/compile/pixi/manifest.py`) reads the raw `[python]` table instead of a validated
  toolchain, so an env declaring only a `dependency-overrides` (no `deps`/`dev`) still carries it;
  `members._alone` carries the env's own pypi-options through composition so the member path-pins no
  longer overwrite them. A `[envs.<name>.python.dependency-overrides]` now lands in
  `[feature.<name>.pypi-options]`. Test: `integration/environments/test_env_overrides.py`.
- **`mb update`/`mb upgrade` with editable members** now work. `Pixi.update`
  (`engines/compile/backend/pixi.py`) drops pixi's `--no-install` when `_has_editable_paths()` is
  true, so pixi builds the env to read the editable sources' metadata instead of failing
  "installation of conda environment is required to solve PyPI source dependencies".

## 2026-10-12 — follow-up: collapse below 4 environments (not yet done)

The workspace can reach 4 envs today (default, cutok, toktier, tl) by deleting the dead `[envs.repro]`
and pruning the orphan `gpu`/`cutile` solves. Going lower is blocked by: numpy `<2.5` in default
(numba-cuda 0.30.4 + pytensor via pymc + cutoken's own pin) vs `>=2.5.3` in toktier/tl; and vLLM 0.31
pinning default to torch 2.13 + transformers <5.18. To merge further, bump numba-cuda (numpy>=2.5) +
pymc/pytensor + lift cutoken's numpy pin together, and move to a vLLM accepting torch 2.14 +
transformers>=5.19. Each tested per project; cutok frozen until its campaign ends.

## 2026-10-12 — RE engine (`probe/reverse/`) follow-ups

The no-REA GPU reverse-engineering core shipped (0.5.6). Two deliberate `TODO(lead)`s remain:
- **`probe device --on HOST` dispatch is not wired** — the verb raises for a non-local host. Wire it
  through the job path (artifact staging + evidence collection) so device analysis runs as a settled
  mainboard job on the GPU hosts.
- **The toolkit-flag contract (`device.STANDARD_FLAGS`) and the parsers are validated against synthetic
  fixtures only.** Exercise them against real cuobjdump/nvdisasm `--help`/`--version` and real cuBLAS
  13.x output on a GPU host, and adjust the contract/parsers if the live output differs.

The REA host-code/breadth half (`rea.py`, callgraph/trace/compare-host, the `mainboard.toml`
`rea-agents` pin) is deferred to its own phase, pending the owner's npm install allow.

## 2026-10-12 — killed test runs leak SysV shared memory until PostgreSQL tests cannot start

Integration runs killed mid-suite (two Codex lanes stopped by their time limits on 10-12) left 33
SysV shared-memory segments: owner pedro, keys 0x07c0xxxx, NATTCH 0, creators dead, no postgres
running. macOS caps the table at `kern.sysv.shmmni = 32`, so every later PostgreSQL test failed at
initdb with `shmget ... No space left on device` (51 to 52 setup errors in the plain and isolated
suites alike, mistaken at first for a host-capacity limit). ci36 freed the segments of test
postmasters that exit; a SIGKILLed suite never reaches that cleanup. Removed by hand at 20:05 (each
segment checked: no attachment, creator gone). To fix: the gate's harness (or `mb doctor`) reports
orphaned segments of the current user at start and reaps those whose creator is dead and NATTCH is 0
before a suite, and a lane's time limit sends SIGTERM with a grace period before SIGKILL so the
test postmasters' teardown runs.

## 2026-10-15 — analysis engine: a 3 GB DuckDB budget runs out of memory on large projects

The mb-analysis worktree's `Budget` (facts.py: 3 GB memory, 8 threads, 2 GB spill cap, all by SET;
measured in scratchpad/mcmr-mb/DUCKDB-RESOURCES.md after the 14:58 kernel panic) holds
research/reproducibility near 3 GB. On packages/sqlalchemy/lib (256 files) a kernel-only
`run_lint.py check --rule PY-COLL0001` fails with DuckDB OutOfMemory at 3 GB (review4,
scratchpad/mcmr-mb/progress/review4.md). Failing at the cap is the intended behavior, but the engine
should finish: find the relation that peaks (likely a multi-join derivation over a large python CTE,
as ComprehensionFact's set_loop_candidates on reproducibility), and either make it cheaper or let
the budget scale threads down for large trees rather than raise memory.

## 2026-10-15 — release provider (seed of the separate dependency checker): four defects

The mb-analysis worktree's `analysis/providers/releases.py` ports mcmr's DependencyRefresher (PyPI
simple index, release JSON, GitHub archived flag; 24 h cache in ~/.cache/mainboard/releases). The
owner decided online dependency auditing becomes its own mb checker, apart from `check`. Review14
(scratchpad/mcmr-mb/progress/review14.md) confirmed it matches mcmr and found, to fix when that
checker is built:
1. A GitHub 403 (unauthenticated limit) or 5xx is not served from an expired cache entry: HTTPError
   escapes the stale fallback, the dependency becomes "unknown" and raises ALL-DEPE0004.
2. `_store` sits outside the try, so a read-only cache directory crashes the run.
3. A malformed `pyproject.toml` (TOMLDecodeError) or a list-valued `mainboard.toml` dependency
   (AttributeError in `_constraint`) crashes the run; mcmr crashes there too.
4. One unparsable version in `uv.lock` breaks `locked` for every package.
Also: a single timeout marks the whole run offline, so a dependency with no cached answer empties
the family.

## 2026-10-17 — pedro-cvlab: every cuDNN convolution fails (system cuDNN 9.27 shadows the pip 9.20)

On pedro-cvlab (torch 2.13.0+cu132, lock's `nvidia-cudnn-cu13` 9.20.0.48) any `conv2d` raises
`CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed ... CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`
(bf16, fp32 and under `cudnn.deterministic`; jobs 1026, 1027 of the exploratory routes study). The pip
cuDNN ships no `libcudnn_engines_tensor_ir.so.9`, so the loader falls through to the system copy
`/usr/lib/x86_64-linux-gnu/libcudnn_engines_tensor_ir.so.9.27.0` (apt `cudnn9-cuda-12` 9.27.0.42,
installed 2026-09-29), and the version check between the 9.20 graph library and that 9.27 sub-library
fails. Every CNN node (classifier sandbox and flip density CNNs) cannot run there until this is fixed.
Workaround used (inside one probe, no system change): compile an empty library with that soname and
`ctypes.CDLL(..., RTLD_GLOBAL)` it before the first convolution; conv then works (job 1028) and the
default route reproduced the September CNN decisions on every image. Proper fixes: remove the apt
cuDNN from the host (no sudo here, owner), or have the host setup put the stub on the environment's
library path, or move the lock to a cuDNN that ships the sub-library.

### Resolved 2026-10-17 (monorepo e308d3ed4): cuDNN fault on pedro-cvlab

`nvidia-cudnn-cu13 = { version = ">=9.27.0.42,<9.28", env-markers = "sys_platform == 'linux'" }` in the root
`[python.dependency-overrides]` (the wheel then ships all ten libraries; 9.28.0.27.dev… is a sdist-only prerelease
that fails to build, hence the upper bound). Moving the lock alone was not enough: cuDNN 9.27 dlopens its sublibraries by
their full versioned name (`libcudnn_graph.so.9.27.0`), the wheel ships only `.so.9`, so the loader fell to the host's apt
CUDA 12 copy of the same version and aborted on `libcublasLt.so.12`. `scripts/activate.sh` now links
`libcudnn*.so.9` to `.so.9.<minor>.<patch>` inside the wheel's directory, which sits first on `LD_LIBRARY_PATH`. Proof job
1042 (fp32, bf16, deterministic conv and CUDA graph replay, all cuDNN maps from the environment). Any host with an apt
cuDNN of the same version needs the links, which the activation script makes. The old comment's "aborts inside any
traced graph" was this same system-library mix, not the 9.24 wheel.

### Library doctor, 2026-10-17 (merged from feat/lib-doctor into fix/mb-bugs)

What was missing was the check that would have named this before a kernel failed:

- `doctor` gains a `libraries` row per installed environment, and `host list <host> --audit` carries the
  same rows (the host's own `doctor` code answers it, so a host shows them once it has the release).
  `mainboard.libs` reads the CUDA libraries the wheels (RECORD) and conda packages (`conda-meta`)
  installed, parses each ELF's `DT_NEEDED`, `RPATH` and `RUNPATH` itself (no `readelf`), reads the bare
  sub-library names cuDNN carries (`libcudnn_graph.so`, version appended when opened), and replays the
  loader order (RPATH without RUNPATH, `LD_LIBRARY_PATH` as `Runtime.entered` builds it, RUNPATH,
  `ldconfig -p`, default directories). It reports a family soname that loads from outside the
  environment, a linked one found nowhere, a family in two releases (host copy included, the break; two
  packages of the environment, a warning), and wheels for several CUDA majors.
- `doctor --fix` repairs only what leaks in from the host: `Raise` (the package that owns the family goes
  through `add <package> --lang python|conda`, a manifest edit, re-locked and installed) first, then
  `Relink` (a symlink inside the environment's lib directory when the file is there under another
  spelling). Neither touches a system package. A transitive pin torch holds makes `add` refuse; the row
  says so and points at `[python.dependency-overrides]`. A family split between packages of the
  environment itself (a pip wheel next to a conda build: pip `nvidia-cuda-runtime` 13.2.75 against conda
  `cuda-cudart` 13.4.92, NCCL 2.29.7 against 2.32.3.1 on pedro-cvlab) is report only: which one to keep is
  the owner's call.
- Run on pedro-cvlab with this code (shipped to `/tmp`, read-only against the environments): the
  `default`, `cutok` and `toktier` environments no longer have the cuDNN clash. The original clash was
  reproduced from the 9.24 wheel minus `libcudnn_engines_tensor_ir` against the apt 9.27:
  `outside`/`split fail`, fix `add nvidia-cudnn-cu13 --lang python`.
- Open: the remedy for a library whose directory is not on the loader path at all (neither `nvidia/*/lib`
  nor `lib*/lib64` nor the prefix's `lib/`) is none; it needs a persistent activation setting. The
  `Relink` links are lost on a reinstall, which `activate.sh` or `doctor --fix` makes again.
  `mainboard add` for pip/conda splits asks the solver once per package; batch them behind one lock if
  `--fix` is run on many.

## 2026-10-17 — trial jobs exit 1 after a passing test (nanobind DuckDB leak), and two smaller traps

1. Every pytest job of `classifier_ess` (and `pytest` on the center, e.g. the settlement) prints `nanobind: leaked 2 instances
   _duckdb.DuckDBPyConnection` at interpreter exit and exits 1 after `1 known` and a complete receipt. `mb job list` then shows
   `failed`, `settled` empty. The connections come from `mainboard.trials` (`Relations()` behind `Log.table` and the artifact
   store). They should be closed before exit, or the exit status taken from the trial's receipt.
2. `@job(resources=...)` and `fetch=` must be literal in the decorator: a module constant is refused with `vanished ... must be
   literal` before anything runs.
3. `mainboard lint <folder>` also rewrites sealed `.json` files (it reformatted `pins.json`, whose bytes are an artifact's
   digest). Lint python files by name when a folder holds pinned data.
4. `mb job list` settles but leaves a trial's receipts on the host (`result transfer failed` for older jobs); the receipts of a new node
   came in with `mainboard job collect --on <host> <node dir>` and `mainboard lake materialize <node dir>`, both through `ssh localhost`.

### Resolved 2026-10-17 on fix/mb-bugs

1. **Not the nanobind report that failed these jobs, and the leak is fixed anyway.** nanobind only prints the leak report
   at exit (reproduced: a passing trial exits 0 beside it). The exit status 1 of the classifier_ess jobs came from the trial
   session's own refusal, printed just above it: `classifier_ess did not release what it held: 41943040 bytes resident
   against the 0 it opened with` (`trials/stage.py`; a 40 MiB cuBLASLt workspace the node keeps, in research code), and the
   job's `cause` showed the report's last line instead. The leak: the evidence keeper (`state/evidence.py _kept`, cached for
   the life of the process) holds a lake `Session`; its finalizer closed the attach but the session kept the closed
   connection, still alive when the extension modules were torn down. With torch imported and a lake present, a tiny trial
   reproduced it (1 instance); the session's connection now lives in an `_Attachment` its finalizer clears (5740204,
   `test_a_sessions_exit_time_detach_leaves_it_holding_no_connection`). The cause line of a failed job should now name the
   real refusal. Still open in the node: release or account for the 40 MiB workspace.
   The center's pytest settlement ran without a GPU, so its exit 1 has no stage residue to blame; it was not reproduced.
2. (unchanged, by design: literal `fetch=`/`resources=`.)
3. **Lint rewrote a sealed file.** A node whose `node.md` carries a real `registration_sha256` seals every file in or under its
   folder; no lint step (text hygiene, writing tools, the analysis engine's fixes) writes them any more, and a `fix` command
   must take `{files}` so none can rewrite a whole owner (0bade83, `test_no_step_rewrites_a_file_a_registration_sealed`).
   Sources a seal names from outside the node's folder (shared `src/` modules) are not discoverable from the manifest and stay
   a `[lint] exclude` entry.
4. **Not a failed fetch.** The pass did fetch (`fetched .../classifier_ess from pedro-cvlab` in the settler log); collection
   writes the lake and no tree files, so the folder looked empty until `lake materialize`. The fetch line now says so
   (415c14a). Old `result transfer failed` rows (pedro-cvlab 621, 631, 632, 1029) are runs whose fetch folder was never
   created on the host: `ssh collect` raises FileNotFoundError every pass. Retire them with `job cancel`, or make a missing
   remote folder of a failed job settle as "nothing to fetch" (not done).

## 2026-10-17 — resolved on fix/mb-bugs: consumers warning, divergence table, stale-compile rows

- `lint` on one patos file warned that the Python grammar misreads two `packages/mcmr` files: the consumer guard reads every
  owned repository that imports the project (mcmr, frozen, still does) and reported grammar gaps in them. A consumer read
  now parses silently (`Ducks(report_gaps=False)`, 9d202a5); the files still count as consumers, so a rename cannot break
  frozen code unseen.
- The PY-CUDA divergence table gave `warp.inclusive_sum` and `warp.reserve` result level 1 (the warp's); each lane gets its
  own value (level 2). Fixed in `CudaCollectiveCalls` (fc1450b); regression suite unchanged (0 unaccepted, 0 stale), and a new
  PY-CUDA0019 example fails on the old table.
- pedro-cvlab's doctor said `compiled before the current manifest` after a sync. Real staleness of the generated
  `pixi.toml`, not of the environment: the host syncs the manifest and compiles at its next run (it did at 13:45:11 JST), and
  the digest covers tasks, so any edit to a task or to activation raised the row with the fix `install`. The row now fires
  only when the edit moves the resolution or the second-stage toolchains (`Compiler.resolution_moved`, b996d8b). The `hosts`
  row (`diverged from the current manifest`) still compares the full compile digest and flags task-only edits; open.

## 2026-10-17 — found by the dependency upgrade: lean environments and the Node lock

1. **Lean environments skip the workspace activation scripts.** `[workspace] scripts` is documented as sourced after the dotenv
   loader on every entry, but pixi's `no-default` drops the workspace `[activation]` table, and `declared_feature` re-adds only
   the unset script (`pixi/manifest.py`, for a clearing manifest). So `cutok`, `toktier`, `repro` and `tl` never source
   `scripts/activate.sh`: no `.env` loading, no wheel library paths, no nvJitLink link, and (what showed it) no cuDNN versioned
   links, so job 1082 on `repro` failed until the links became a runtime step. Not fixed: the script also exports
   `PYTHONPATH=<repo>:<research>` and loads every `.env`, which lean environments are lean to avoid, so sourcing it there changes
   what they run. Decide first which of the two is true: scripts apply to default-feature environments only (then fix the
   documentation of `scripts`), or every environment gets them (then split the script into what every environment needs, the
   library paths, and what only the default environment wants, the `PYTHONPATH`). Runtime steps (`Runtime`, `activation.py`) already
   reach every environment and are the place for what must hold everywhere.
2. **Node lock regeneration only checks, it never resolves.** `Node.sync(resolve=False)` runs `pnpm install --frozen-lockfile`
   (`npm ci`), so a manifest edit that moves a Node dependency leaves `pnpm-lock.yaml` stale and the install refuses; the upgrade
   regenerated it by hand with `pnpm install --lockfile-only`. `mainboard lock` / `update` should resolve the Node lock the way
   they resolve the conda and Python ones.

Done with the upgrade: the cuDNN versioned links are the runtime step `CudnnSublibraries` (553e99a), and the duplicate block in the
root `scripts/activate.sh` (and its `unset` names) is removed, uncommitted in the monorepo.

## 2026-10-20: settlement and collection

Fixed, uncommitted: (1) a collection whose staged file had other bytes than the center's copy at the same path (a second
registration writing the same file names) refused the whole transfer, blocking every other file and the job; `Collector._divert`
now keeps the center's copy at its path, keeps the divergent bytes under `<scope>/.conflicts/<sha8>/`, ingests the rest and warns.
(2) A native test run that claimed success with a fetch path and left no receipt (a development job outside the trials harness)
stayed `settlement pending` for the cache's lifetime and was re-captured every pass (102 jobs made a `job list` take five minutes);
it now settles with evidence `unverified` (`Unverifiable`). A full `job list` went from 5 minutes to 25 seconds.
(3) `job digest` and `job events` (`digest.py`, `events.py`); `Monitor.once` appends every moved job to `.mainboard/events/jobs.jsonl`.

(4) A fetch path the job never wrote (`zz_cvpr_routes`, `heterogeneous_pipeline/*-v26`) made `pack.py` raise
`FileNotFoundError`, so those runs stayed pending; it is an empty selection now, and a run whose receipts name artifacts the host no
longer holds settles unverified. The steady-state `job list` over 2,385 runs takes 2 seconds.

Tested 2026-10-23: the empty selection (`evidence/test_core.py::test_a_results_path_never_written_packs_nothing`) and the
events stream (`test_events.py`); `Collector._divert` by `evidence/test_core.py::test_collection_keeps_transfers_in_the_lake_never_the_tree`.
`Unverifiable` by `evidence/test_core.py::test_a_native_run_claiming_results_without_a_receipt_is_unverified`.

## 2026-10-22: DuckDB 2.0.0 stable

Done, uncommitted: `duckdb>=2.0.0` replaces the `2.0.0.dev2609250715` pin. `sitting_duck` is the community repository's signed
build, loaded in-process from the tool's extension folder through `state.database.load` (shared with the lake), so the unsigned
self-built extension, `allow_unsigned_extensions` and the DuckDB 1.5.6 command-line fallback (`DuckCli`) are gone; the gate seeds
`sitting_duck` with the lake's extensions. 2.0 removed `fetchone`/`fetchmany`/`execute` from relations (`results.py` hashes through
`iter_rows`), and a blocking aggregate now times out inside `execute`, so `Query.started` renders it as a timed-out partial. The
center's lake was migrated from DuckLake spec 1.1-dev1 to 1.1 (`mb lake upgrade`; catalog backup at
`/Volumes/PORTABLE/mainboard-lake-backup-2026-10-22/`), and an attach of an older spec now names `mb lake upgrade`.

Fleet, the same day, uncommitted:
- A host's tool is installed with `--constraints` from the shipped lock's PyPI releases (`Lockfile.pins`), so the tool and the
  jobs run one DuckDB; before, `uv tool install` re-solved on every host and would have taken a newer DuckDB than the lock.
  The reinstall digest (`Bootstrap.requirements`) covers those releases, so a sync reinstalls the tool when the lock moves one.
- Setup and sync end with `Bootstrap.lake`: a host's staging lake is brought to its new DuckDB's spec (`mb lake upgrade`)
  once every reader there runs it.
- gold, crimson, pedro-cvlab and purple synced: tool and environment on duckdb 2.0.0, lakes at 1.1, stale extension folders
  removed. Job 423 on gold built the new lock's environment and parsed with the community `sitting_duck`.
- The center's uv tool (`~/.local/bin/mainboard`, shadowed by the environment's on PATH) still had the dev build; reinstalled.
- Settlement outlived its deadline on every pass since the morning: `Artifacts.verify` recalled each lake-only artifact
  alone, a scan of every `blobs` file's digests per object (0.14 s each). `EvidenceTree.recalled` reads them in 1 GB windows:
  1,991 objects in 3.6 s instead of 284 s. The settler completes again (19 s at rest).

Open:
1. **miyabi-g still runs the dev build**: its cutok batches (`gh200-refresh-20261022-d011da66`,
   `cutok-current-serving-gh200-20261022-87f6cbd0`) owe outcomes, so `host sync miyabi-g` refuses until they settle. Then
   `mainboard host sync miyabi-g` (tool, environment and lake in one step). Quack needs both ends on the same release.
2. **Resolved 2026-10-23** (the workspace takes stable releases only). **`prerelease-mode = "allow"` admits DuckDB's nightlies.** The workspace caps `duckdb` at `<2.0.1` so `2.0.1.devN` stays out;
   `mb upgrade` would write `>=2.0.0, <3` and let them back in. The same policy holds about 20 packages at pre-releases (grpcio rc,
   sentry 3.0 alpha, opentelemetry betas, tornado rc). Decide whether the workspace moves to `if-necessary-or-explicit`.
3. **Quack still drops bound parameters** at 2.0.0 (duckdb-quack#323), so `inlined` stays; #282 (remote `query()` runs twice) is open.
4. **`hashfuncs` has no 2.0.0 build**, so blob checksums stay MD5.
5. **Resolved 2026-10-23** (`Apple.select_xcode`). **Swift analysis fails under the Command Line Tools**: `xcode-select -p` is `/Library/Developer/CommandLineTools`, which has no
   SwiftParser host libraries, so `mb lint` and 114 gate tests fail unless `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`
   (the gate passes it through). `sudo xcode-select -s /Applications/Xcode.app` fixes the machine.
6. **Resolved 2026-10-23**: `comment_parses` is SQL (`tree_clean`, `python_source` over `parse_ast_list`); on the
   workspace's 51,775 punctuated comment bodies it disagrees with `ast.parse` on 8 (3 of them `noqa` directives no probe
   is asked about). No rule read `parses_as_source`, since ALL-COMM0002 had gone to Ruff's ERA001, which reads Python
   only; ALL-COMM0002 is back for Rust, TypeScript, JavaScript, C, C++ and CUDA. **The comment probe was the analysis engine's last Python UDF over Arrow** (`CommentProbe._answer`). `sitting_duck`'s
   `parse_ast_list` parses strings in SQL, so `comment_parses` could become a macro, once its missing-token count and Python's
   `ast.parse` verdict are matched.

## 2026-10-23: stable releases and a whole-workspace upgrade

The workspace now takes stable releases: `prerelease-mode = "allow"` is gone (uv's default takes a pre-release only when a
requirement names one), so the ~20 pre-releases it held (grpcio rc, cuda_tile rc, curl_cffi beta, bottleneck rc, ...)
moved to finals, and duckdb's nightly ceiling is lifted. `mb upgrade` then ran over everything; fixed on the way, uncommitted:
1. `ManifestText.table` refused a table split across the file (tomlkit's `OutOfOrderTableProxy`, `[on.linux.*]`).
2. `versioned` raised a binary pinned by `url` and `sha256` beside its `version` (swiftlint), and an explicit `==` pin; both
   are left as written now (`scikit-learn ==1.9.0` keeps the Ledoit-Wolf estimator byte-stable).
3. A Python requirement for a package the lock takes from conda is raised to the channels' newest, by the channel's own
   spelling (`filelock`, `huggingface_hub`), since pixi holds the PyPI requirement to the conda build.
4. `upgrade` re-locked minimally, so a conda pin under a raised PyPI requirement could not move; it frees what it raises now,
   everything when none is named, as `pixi upgrade` does.
5. `Python.latest` took yanked releases (pytest-timeout 2.5.0); a version every file of which is yanked is skipped.
6. `pin` capped a lone date stamp at the next day (`pdfminer.six >=20260107, <20260108`); a date gets a floor only.
7. The Node lock was re-resolved only when the pixi lock moved in the same call (the 2026-10-17 item 2): an `install` under
   the update policy now solves a stale second-stage lock.
8. `mb lake upgrade` reaches every project in the workspace keeping its own lake; `research/reproducibility`'s was still at
   1.1-dev1 and broke `mb query`. (`research/cuda-tokenization/.mb` is an orphan: that project has no manifest.)
9. `mb ci` drops a `CONDA_PREFIX` naming a pixi-global tool's environment and names the step's `python` for PyO3; a
   pixi-global `uv`/`cargo` launcher pointed ty and PyO3 at a toolchain with no Python.
10. `Rust.pin` wrote a range, which a host's frozen `cargo install` refuses (every sync stopped on `[rust] has no frozen
    installation contract`); a crate is raised to one exact release, as Go's is.
11. `Nodejs.latest` took npm's `latest` tag hours after publishing, which a host's frozen pnpm refuses under its default
    one-day `minimumReleaseAge`; it takes the newest release at least a day old.

Fleet: gold, crimson, pedro-cvlab and purple synced onto the upgraded lock; job 424 on gold ran duckdb 2.0.0, pandas 3.1,
vLLM 0.32 on torch 2.13+cu132 with CUDA. purple, a shared server, ran out of disk during its first sync (309 MB left on
`/`, mostly the other user's 3.3 TB): uv's and pip's caches were cleared (27.5 GiB) and the sync finished with 30 GB free.
Its rattler cache still holds 65 GB of ours. A sync reports the setup's cached disk figure (67 GB) instead of measuring.

Held, each by a stated reason (`mb upgrade --exclude ...`):
- python `<3.15`: no cp315 wheels for macOS torch 2.14, cupy-cuda13x, sentencepiece, xformers, conda-forge libsass.
- numpy `<2.5`: numba-cuda 0.30.4 (release and main) overloads `np.row_stack` at import.
- torch, torchvision, triton, transformers, fastapi, lark, setuptools, nvtx, nvidia-cutlass-dsl: vLLM 0.32 pins them.
- fsspec (datasets 5.1 caps it), pillow (gradio 6.30), mpmath (sympy 1.14), z3-solver (tilelang 0.1.15),
  huggingface-hub (the diffusers dev wheel), cupti-python (13.4 needs cupti 13.4, torch cu132 pins 13.2), sphinx
  (zzzeeksphinx), gxx (nvcc 13.4 takes GCC below 16), python-dateutil `<3` (botocore, invisible to the conda solve).

Then open, since addressed below: a persistent hold list in the manifest; `rls` alone cannot import
`CreatePolicy` (it needs the SQLAlchemy fork the workspace supplies), before and after its lock upgrade; patos's lock was
left alone while its copier update is in progress.

Later the same day, after vLLM left `default` for its own `serving` environment (the vLLM holds above lifted:
torch 2.14.1+cu132, triton 3.8.0, cutlass-dsl 4.8.1) and five libraries moved from `[deps]` to `[python.deps]`:
12. A requirement moved from conda to PyPI stayed on its conda build: pixi takes a conda package as satisfying the
    PyPI requirement of its name, and a minimal re-lock keeps it. Fixed: a compile records each requirement whose
    ecosystem changed since the last solve (`SyncState.moved`), and the next solve frees exactly those.
13. A member's environment named like the root's was refused; the two now join (`Env.merged`, the root's settings
    winning), so llm-head contributes its serving extra and task to `serving`.
14. `selected_manifest` cleared `[system]` for a `no-default` environment, so `serving` solved for glibc 2.28 and
    refused pynvvideocodec's 2.34 wheel; lean environments keep the workspace's floors.
15. `mb lint`'s `mb` step fails on the center with `Swift analysis requires SwiftParser host libraries at
    /Library/Developer/CommandLineTools/...`: `xcode-select` points at the Command Line Tools. It passes with
    `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. Fixed: mainboard chooses the newest
    Xcode.app itself when `xcode-select` names the Command Line Tools, for its own process and plumbum's
    copy of the environment (which `mb run` launches through and had snapshotted at import).
16. Hosts gold, crimson, pedro-cvlab and purple synced to this lock (job 425 on gold: torch 2.14.1+cu132 with CUDA).

Fixed afterwards: `[holds]` (and `[envs.<name>.holds]`) record what `upgrade` leaves as written and why, so the 18
`--exclude` flags are gone; a full `mb upgrade` with none completes, raising 14 requirements (the vLLM-era floors of torch,
torchvision, triton, cutlass-dsl, z3, lark, fastapi took ordinary bounds) and holding 8 plus `serving`'s torch. Its first run
proved the scoping: raising `serving`'s `torch = "*"` broke vLLM 0.32's exact pin, and the failed solve put the manifest
and lock back. A sync measures the free disk again (`_measured`, one `df -Pk`) instead of reporting the setup's figure.

Open: `rls` alone still cannot import `CreatePolicy`. A lone checkout must take the fork's `postgresql-row-security`
branch (`[tool.uv.sources]`), but uv refuses that branch's `pyproject.toml`: SQLAlchemy's legacy pre-PEP-685 extras
(`mssql_pymssql`, ...) normalize onto the current ones. Removing them from the fork needs a commit and push to
`Pedrexus/sqlalchemy`, the owner's call; the published `rlsalchemy` needs the fork upstreamed regardless.

Then miyabi-g, whose sync had waited on the cutok batches (all settled `ok`), failed three times, each a real fault:
1. Its tool ran on CPython 3.14.0rc2, which `uv python install 3.14` keeps satisfying; the pydantic the lock moved to cannot
   import there (`_eval_type() got an unexpected keyword argument 'prefer_fwd_module'`). The tool's interpreter is now the
   request `Lockfile.python_request` (`>=3.14.0,<3.15`: a final release of the solved minor, or the candidate a lock itself
   solved), and the reinstall digest carries it. Not the solved patch: conda-forge publishes 3.14.9 before uv's builds have it.
2. Its uv (0.8.11, copied in rather than placed by Astral's installer) predates 3.14.0 and refuses `uv self update`; the
   bootstrap updates uv each time and, when it refuses, runs the installer over it.
3. The environment install held 12-14 GB on the login node and the guard stopped it. Not pixi's 50 concurrent downloads, as
   first guessed (bounding them changed nothing; reverted): sampling the user's RSS showed `cargo` and a dozen `cc1plus`.
   `mb upgrade` had raised `[on.linux.python.deps] z3-solver` from `>=4.13` to `>=5.1`, and PyPI's z3 5.x has no aarch64
   wheel, so `default` locked its sdist and every aarch64 host compiled z3. z3 is conda-forge's on Linux now, as `repro`'s
   already was; `serving` keeps vLLM's 4.15.4 wheels.
Open: the lock holds five more PyPI sdists (pytest-clarity, peewee, baycomp, watchdog, nvtx 0.2.16). A lock check that names
an sdist a host would compile needs to know pure Python from extensions, which the lock does not say.
Fleet at the end of 2026-10-23: gold, crimson, pedro-cvlab, purple and miyabi-g all synced to the z3 lock. miyabi-g's tool runs on
CPython 3.14.9 with uv 0.13.5, and its install peaked at 2.9 GB on the login node. A sync's disk row is measured now: purple shows
29 GB, and miyabi-g shows the whole Lustre filesystem's free space (6.5 PB), which says nothing about the group's quota.
Later on 2026-10-23:
- Tool interpreter: the bootstrap runs `uv python install --upgrade` on the request, so a host takes each new 3.14 patch
  (the tool's venv follows uv's minor link). miyabi-g's 3.14.0rc2 was uninstalled. Job 3680459 on a GH200 ran Python
  3.14.9, torch 2.14.1+cu132 with CUDA, DuckDB 2.0.0.
- Comment probe: exact agreement with `ast.parse` on all 51,780 punctuated Python comment bodies of the workspace, and with
  the old file-based tree-sitter route on all 1,544 C-family ones. The last differences were tree-sitter accepting a
  `list_pattern` target and a `constrained_type` outside a type parameter, rejecting a bracket that continues an expression
  (`layer_past[0] [b, h]`), and having no slices in annotations (`python_annotated` reads the annotation as an expression).
- `rules fix` crashed (`parse_ast_list does not exist`): a macro binds its functions when created, and a fact base built
  without parsing never loaded sitting_duck. `Budget.connect` loads it for every base.
- On mainboard's own code, ALL-REAC0001 called 505 registered rules dead, and its review fix would have deleted them; a class
  an ancestor's `__init_subclass__` registers is reached through that hook now, and a name a string literal carries is
  reached as the fix already assumed (539 -> 4 findings). PY-MODE0006 exempts hook-registered classes the same way (273 -> 0).

## 2026-10-24: the package files itself into folders

No folder of mainboard holds more than five modules directly now (ALL-FILE0003), at most six levels deep
(ALL-FILE0002, PY-IMPO0005), so `mb rules check --rule ALL-FILE0002 --rule ALL-FILE0003 --rule ALL-FILE0005
--rule PY-IMPO0005` reports nothing on src or integration.
- ALL-FILE0005 (review fix) moves modules sharing a prefix into a folder of it and respells every import of them. The
  module the prefix names becomes the folder's `core.py`, and each member keeps its own leading word, so
  `test_apple.py` becomes `apple/test_core.py` and `reverse_fixtures.py` beside the tests becomes `reverse/fixtures.py`.
  It moved 34 groups.
- PY-IMPO0008 (safe fix) exports from a package's `__init__.py` the names modules outside it import, so PY-IMPO0006
  can route them to the shorter path.
- The rest had no shared prefix and was grouped by hand, by subsystem: 320 modules in src and integration. Modules
  research imports stayed where they were, since research is sealed. `mainboard.log` and `mainboard.profile.result`
  became packages that re-export what is imported from them.
- The gate baseline is re-based. `case_rebaselines` in `integration/gates/baseline.json` records every old and new path.

Found on the way, fixed:
1. Respelling an import dropped an explicit `NAME as NAME`, and F401 then deleted the re-export.
   - It lost `dispatch.shared.CLOSURE_VAR`/`DIGEST_VAR` and `trials.artifacts.NO_TABLE`, which research imports; they
     are back, declared by `__all__`, since PY-RUFF0002 (PLC0414) flags a self-alias outside an initializer.
   - FILE0005's example now covers it.
2. The Swift syntax provider was registered only by a re-export in `rules/families/swift.py`. That re-export was lost
   the same way, so `mb rules` ran no Swift family outside the tests, which import the provider themselves.
   `load_catalog` imports every provider module now, and the empty module is gone.
3. FILE0005 dropped a whole group's fix when one importer sat under the rule's excluded paths (the `pixi_*` group,
   whose importers include two integration tests).
4. The inspect lane failed inside the monorepo. The baseline holds `test_consumers.py::...[NOTSET]`, the id an empty
   parametrization gets outside it. The test is one case now, listing every missing name, so its id is the same
   everywhere.
5. `ProjectConfig` and PY-IMPO0007 built pathspec's deprecated `gitwildmatch` patterns; they take `GitIgnoreSpec`, as
   the rest of the package does.

Open:
- `mb rules fix` writes import blocks unsorted and unformatted, and `mb lint` puts them right afterwards. A fix could
  run ruff on the files it touches before its proof.
- Grouping modules without a shared prefix needs someone to name the folders, so no rule fixes it. ALL-FILE0003 stays
  report-only.

## 2026-10-25: smaller dispatch API, and cuts three Haiku reviews found

`Dispatcher` takes one `Order` (plan, shipment, resources, name, node, watch, root, prefix, artifact,
sampler, attestation, containerize) instead of up to fifteen arguments; `Board.rented`, `Landing` and
`allocating` take it too. `dispatch/dispatcher/` is a package: `order.py` (`Order`, `Handle`, `Verdict`),
`shipping.py` (`Shipper`: mirror, stage, pin) and `core.py` (`Dispatcher(Shipper)`). Polling moved onto the
fleet's `Job` (`poll`, `state`, `wait`, `pull`, `Job.states`), which already wrapped it; `fleet/board/` is a
package with `Job`/`ProviderJob` in `runs.py`.

Removed because no caller used them (checked across src, integration and research):
- `run`'s `verify`, `fetch` (it was `shipment.fetch`) and `gpu_in_select`, `Scheduler.submit`'s `args`, and
  `_prepare_script`, which staged a script `run` had already staged.
- `smallest_fit`, `Facts.fits`/`vram_gb` and the memory fields only they read (the host probe stops asking
  for them; old host records still load, read with `extra="ignore"`).
- `ReconcileRow`, `verdict_line`, `log_excerpt`, `SshTransport.batch_mode`, `isolate_pythonpath`, the
  `Link` ABC with one implementation, `profile.bottleneck.wait_for_idle`, `profile.tracer.Vendor` (the
  probe's enum serves), `Readiness` (the workstation checks answer `Section` rows).
- 36 `verified = True` lines: a family is verified by default, and the two that are not say so.

Shared instead of repeated: `Credentials.key` reads every provider key; the rental key checks run through
`ToolRunner`; a `_Membership` base replaces 11 eight-line `Function` classes; the collector uses
`provenance.blob_of`; `Scheduler.interactive` defaults to the workspace session.

How the review ran: three Haiku 5.5 agents read a signature skeleton of the package (outside `analysis/`),
one of `analysis/`, and every file of `dispatch/`. Their high-confidence dead-code claims all held. Two
claims did not: `census.py` and `migration/remote.py` keep their own `subprocess.run` because they are
stdlib-only scripts sent to a remote Python, and removing `verified` outright would have started the
unverified TypeScript `ModuleSurfaceFact` rules.

Open:
- `load_catalog` prints `skipped <module>: <error>` and keeps going when a rule module fails to import, so
  a broken rule file silently drops its rules from `mb rules check`. It should fail.
- Candidates not done (small or needing a decision): the mcmr parity harness (`compare`, about 60 lines)
  once the port is judged complete; the provider JSON clients and "no key" `Standing` rows; the
  wait-for-address loops across backends; process-tree kills in five places.

### Later the same day: four large modules split by responsibility

- `fleet/board.py` (1,308 lines, 43 public methods) is `fleet/board/` in four layers below `Board`:
  `Workspace` (manifest, plans, shared caches), `Environments` (install, provide, pack, shell),
  `Tracking` (receipts, samples, facts) and `Dispatching` (rent, run, job). `Board` keeps only the
  ten verbs that hand `self` to a subsystem, so its `ALL-CLAS0003` waiver is gone. `install` is
  this machine only; onboarding a host is `setup(onboard=Onboard.MINIMAL | DOTFILES | SYNC)`,
  replacing the `sync_only` and `dotfiles` flags. `Job`/`ProviderJob` live in `fleet/runs.py`.
- `dispatch/agent/program.py` (1,156 lines, shipped to hosts) is the package
  `dispatch/agent/program/` (`constants`, `exceptions`, `walking`, `snapshot`, `transfer`). The
  host still receives one framed source: `runner._bundled` frames a loader that registers each
  module under one package, so their relative imports resolve there as here. Checked by running
  the real bootstrap in a fresh interpreter. `Entry` is a `NamedTuple`; the ignore-literal parser
  is one regular expression, equal to the old loop on 600,000 generated patterns.
- `dispatch/backends/vast.py` (763 lines, 23 public methods) is `dispatch/backends/vast/`: `api`
  (`VastApi`), `market` (`VastMarket`, searches by an `Ask`) and `core` (`VastBackend`, its nine
  capability methods). Equal to the old module on 240 market cases (picks, refusal messages and
  the request bodies sent) and on the catalog.
- `analysis/commands/swift_inventory.py` is `analysis/commands/swift/` (`constants`, `models`,
  `inventory`, `comparison`). Equal to the old module on `packages/dataclass` and
  `apps/usagiunagi` (inventory JSON and comparison) and on ten tampered-inventory refusals; no
  test covers these commands, which is worth adding.
- `analysis/engine/fixing.py`: `Fixpoint.run` (complexity 29, a 42-statement `try`) is
  `_fixed`/`_pass`/`_judged`/`_closing` under `contextlib.closing` mirrors; proven by running the
  fixer itself over the package.

Waivers that named moved files were re-pointed or dropped once nothing they covered still fired.
A moved file silently lost its waiver: the entry still named a path that no longer existed, and
nothing reported it (four did after these splits). Fixed: ALL-CONF0002 reports any rule exclude
matching no file of the project, at the line quoting it (`manifest_rule_excludes`, matched by the
same gitignore spec `RuleConfig.accepts` uses).

### Later still: rendering, settlement, consumers, evidence

- `render/query.py`: `Query.render` (complexity 46) prints through one of two sinks, `_Streamed`
  (every row as it arrives) or `_Windowed` (a capped prefix and suffix); how a scan ended is an
  `Ending` enum behind the unchanged `Stamp` fields. Byte-identical to the old renderer on 135
  cases (9 queries x 3 modes x 5 caps, stdout, stderr and stamp).
- `waiting/verdicts.py`: `_stop` settles a run with no machine to contact through a verdict-keyed
  table, then preserves, releases and records a live one; `wait` passes a small `_Watched` state
  through `__pass`/`__looked`. `conclude`, `handled`, `harvest`, `swept` are private (no caller
  outside), so `Verdicts` has seven public verbs. `_parsed` equal to the old on 500 random inputs.
- `analysis/project/consumers.py`: `_surface` is one `singledispatch` on statement type; equal to
  the old on all 14,112 parseable Python files of the monorepo. Found on the way: the consumer
  index named a different consumer file for a module from run to run (its queries had no
  `ORDER BY`), so a refusal's message changed between identical runs. Fixed by ordering both.
- `state/evidence.py` (948 lines) is `state/evidence/` (`models`, `keeping`, `core`, `tree`,
  `replica`): `Evidence(Keeping)`, the write side below the read side, each under the class-size
  limit. `Replica`, an abstract base with one implementation, is gone; `DirectoryReplica` stays.
  `READERS` and `WINDOW_BYTES` moved to `state/constants.py`. Tests that patched module globals
  patch the submodule that reads them.
- `Profiler` keeps its public surface: research calls it, and its method names are too common to
  prove unused by search.
- `analysis/engine/facts.py`: narrowing a family per language (`_narrow`, complexity 20, and its
  seven-argument `_clause`) is a `_Narrowing` model whose `views` yields each relation once its
  parent is narrowed. Every finding `mb rules check` reports outside `facts.py` itself was the same
  before and after.
- `probe/reverse/device/parsers.py`: the cuobjdump/nvdisasm line parsers are three reader classes
  (`_Entries`, `_Listings`, `_Resources`), each holding its parse state instead of threading it
  through branchy functions. Equal to the old parsers on 120,000 fuzzed listings.
- `dispatch/backends/cloud/`: `CloudBackend` had 19 public methods. A provider's four calls
  (`create`, `machine`, `machines`, `terminate`) and its keyed JSON request are now a `CloudApi`
  (`LambdaApi`, `RunPodApi`) the backend composes, as vast's `VastApi` is; the backend keeps the
  nine verbs its contracts name. `Machine` is a `Rented`, so `rentals` hands machines back as they
  are. gpuhunt is one adapter module (`offers.py`), the one module allowed its private
  `OFFLINE_PROVIDERS` (a `PY-IMPO0007` fence), and asks gpuhunt with typed keywords instead of a
  `dict[str, Any]`. `CapacityGone` moved to `backends/exceptions.py`; the two providers' capacity
  refusals share `CloudApi._launched`. The module function `keyed(kind)` was one caller's helper
  and is now `RentalVerbs._keyed`. API answers are read through a pydantic `TypeAdapter` of the
  shape each call expects rather than `Any`.
- Found on the way: `host offers` would crash on a gpuhunt row with no GPU name (a CPU offer when
  no card is asked), since `capability(None)` calls `card(None)`; such a row is now listed as `cpu`.
  The key hint for a missing cloud key now says where a key comes from instead of repeating the
  cloud's name, which the variable names already say.
- PY-LOGG0001 flagged `logging.getLogger("gpuhunt").setLevel(...)`, which tells a library's logger
  to be quiet and logs nothing; configuring calls (`setLevel`, `addHandler`, `removeHandler`,
  `addFilter`) chained on `getLogger` are no longer findings. A second example pins this.
- `dispatch/backends/hpcai.py` is `dispatch/backends/hpcai/` (`api`, `core`), the shape vast and
  the clouds have: `HpcAiApi` holds the calls (`request`, `instances`, `instance`, `types`,
  `balance`), and listed rows are models, `Instance` (a `Rented` carrying its ssh address and why
  its start is stuck) and `InstanceType` (with `card` and `fits`), instead of dicts read by key in
  four places. `HpcAiBackend` keeps nine public verbs, down from fifteen; `chosen` answers an
  `InstanceType`. `rent` ends a rental it cannot open through `except BaseException` rather than a
  flag read in `finally`.
- Found on the way: the bare `CloudBackend` stayed abstract only through its four abstract
  primitives; once those moved to `CloudApi`, `ProviderBackend.implementations()` listed it and
  `doctor` failed constructing it. Its `client` is now an abstract property, so the base stays out
  of the registry's concrete set.
- `upkeep/doctor.py`: `Doctor` answers four public verbs (`sections`, `repair`, `library_rows`,
  `hosts`); every other section is private, being the report's own. Its `survey`, `probe` and
  `settler` constructor seams had no caller and are gone. The `environment` row's findings are
  `_drift`, one (finding, holds) list in the order the row prints them. Equal to the old rows on
  the workspace and under all 128 combinations of forced drift (pixi version, no committed lock,
  a moved sync state, damaged wheels, a stale compile, not installed, no pixi).
- The doctor's `layout` section is gone: it found environments under the root every pixi prefix
  left on 2026-09-04, and no machine has had one since the 2026-10-07 re-setup of every node.
- `analysis/audit/releases.py` (the parked seed of the dependency checker, 268 statements) is
  `analysis/audit/releases/` (`models`, `manifests`, `web`, `exceptions`, `core`). `Web._download`
  raises `Unreachable` itself, so `get` reads a fresh copy, the network's, or a stale copy without
  telling a server's refusal from a silent network; the three tables are one DDL; an unparsable
  index version is skipped through `_version` instead of `except: continue`. Equal to the old
  evidence for patos and mainboard, live into a scratch cache and served stale with the network
  cut. The package's docstring-only `__init__.py` folded into `family.py`. The four defects the
  2026-10-15 review listed still wait for that checker.
- `dispatch/backends/modal.py`: the warnings filter sits before the `try` that imports modal.
- `engines/compile/pixi/manifest.py` (593 lines): how a generated Pixi file spells workspace
  locations (`rerooted`, `anchored`, `normalized`, `self_installed`, `local_sources`, and the
  rerooting of dependency `path` sources) is `pixi/paths.py`, and `GENERATED_DIR` sits in
  `compile/constants.py`. `PixiManifest` keeps `from_manifest` and `to_toml` public, the two any
  caller reads; its table builders are private, and `m` is `manifest`. `_source_specs` walks a
  list's items the way it walks a table's values. Equal to the old compiler on every environment
  shard of the workspace at three generated depths, and the respelling on 50,000 fuzzed texts.
- PY-INTE0001 would suggest `Sequence` for `isinstance(value, list)` on a TOML value, which a
  `str` also satisfies, so the recursion would walk strings forever; the rule already leaves TOML
  boundaries to its `exclude`, which now names `pixi/paths.py` beside `pixi/manifest.py`.
- `analysis/catalog.py` is `analysis/catalog/` (`rules`, `relations`), its names re-exported, so
  no rule module's import changed. A relation installs itself (`Relation.install`, its stored
  rows in `_stored`), which leaves `install` a loop over the relations the rules need instead of
  a seven-statement `try` at complexity 17. `Rule.repair` answers `list[Edit] | None`, the shape
  every override already returned, so the `repairs` wrapper is gone; how a repair is worded
  (`describe`, `_KINDS`) moved beside `placed`, its one reader; `rows` and `edits` bind their
  parameters through one `_bound`. Every rule example passes.
- `dispatch/state/cache.py`: `Cache` had 26 public methods over two logs. Host onboarding is
  `cache.hosts` (`Hosts`: `recorded`, `all`, `save`, `synced`, `drop`), provider creation intents
  are `cache.creations` (`Creations`: `reserve`, `bind`, `prepared`, `reopen`, `named`), both on
  the one session through a shared `Log` base in `state/registry.py`, with `RunRecord` (still
  importable from `cache`, as research reads it). The five run listings are one
  `listed(only=..., project=, limit=, since=)` and the two counts one `count(name=, target=)`,
  leaving `Cache` nine verbs. `Cache.private()` and its finalizer are gone: the two tests that
  used it build a cache on their own `tmp_path`. One-line wrappers over the session
  (`_append`, `_rows`) are gone, and so is their `Any`.
- `jobs/closure.py`: `Closure.of` (24 statements, complexity 18) reads as its phases,
  `__reached` (the import walk and a pytest target's harness), `__shipped` (what each reached
  package ships, or defers), `__needed` (needs and pins, checked and restored); the static walk
  (`Module`, `Walker`) is `jobs/walker.py`, still importable from `closure`, where research reads
  `Walker`. `Walker.imported` is one `match` per syntax node (`_named`) and `reach` one
  comprehension per module. Old and new closures equal on real pytest targets of the workspace.
- `center/standalone.py`: `uv` is looked up apart from running it, so a missing tool and a silent
  install are refused by their own `try`; a manifest that is not there reads as empty instead of
  `except: continue`; `root` is a property over the composition's instead of a copy; the
  `PYTHONPATH` sources, the root mentions and the shape rows are comprehensions. Equal to the old
  member checks on the workspace (21 rows), the clone-and-install step left out of the comparison.
- `engines/compile/compiler.py`: `Compiler(stage, vendor, *, environment)` reads the root, the
  projected manifest, its directory and pixi from the second stage it was already handed (they
  were passed twice, and the manifest projected twice), down from seven arguments; `install` was
  `pixi.install` under another name, and the provisioner calls pixi. The dotenv loader is the
  compile's own static method. Equal digests, resolution digests, `resolution_moved` and `stale`
  on all six workspace environments.
- A hazard met on the way: moving a function into a class by indenting it also indents a
  multi-line string inside it, here the generated `dotenv.sh`, whose bytes enter every
  environment's address. Caught by comparing the loaders byte for byte; the loader is now
  concatenated lines, indentation-proof. The FUNC0003 autofix already refuses a helper with such
  a string.
- `analysis/rules/python/imports/hygiene.py` (541 lines, 13 classes): the package-surface rules
  (`ImportsFacades`, PY-IMPO0006, PY-IMPO0008) are `imports/surface.py`. PY-IMPO0008's `repair`
  (13 statements, complexity 11) makes its three edits through `_joined` (an existing import),
  `_appended` (a new one, placed) and `_relisted` (`__all__`), with one `_span` for a node's
  place. The three rules' examples pass.
- REGRESSIONS FOUND AND FIXED, both from this round's autofixes and neither seen by the suite:
  `dispatch/collection/pack.py` is sent to a host as its own text and run by `python -`, but a
  CONS0001/0002 fix had moved `HELD` into a new `collection/constants.py` and given the script
  `from .constants import HELD`, so every collection would have died on the host with an
  ImportError; and PY-LOGG0002 had rewritten the migration agent's census
  (`probe/census.py`, shipped by the carrier into an interpreter without mainboard) to print
  through `from ..log import logger` (and, on a later lint, `from mb import logger`, which even
  broke `mb` locally with a circular import). Both are standard-library-only again, with waivers
  saying why (PY-CONS0001/0002 for `pack.py`, PY-LOGG0002 for `census.py`).
- The guard: `integration/test_standalone_programs.py` runs the collection script and the
  migration agent the way a host meets them, in `python -I -S` with nothing of mainboard
  importable (the agent through the real loader, via a new public `Carrier.preamble`). Both
  broken variants fail it, so the fixer's proof now withholds any fix that hands those programs an
  import. Worth a rule some day: a module read as text and shipped (`with_name("x.py")
  .read_text()`, the carrier's `_modules`) must import nothing first-party.
- `link`, the duplicate linker, is `collection/linking.py`: it runs on the center, so it imports
  the walker, digests and exclusions from `pack` (now public `walked`, `excluded`, `sha256`);
  `pack.py` keeps only what the host runs. `link` reads as its phases in `_Linking`
  (`grouped`: settled same-size inodes hashed and grouped by digest, damaged ones apart;
  `linked`: one digest's names moved onto its copy). Equal to the committed `link` on 120
  generated trees (duplicates, digest-named and damaged files, fresh runs, dry runs).
- ALL-FILE0005 BUG, found by its own fix: with three `test_` modules among six in
  `integration/`, `test_` was not a majority prefix, so `test` became a group and the fix moved
  `test_clouds.py`, `test_consumers.py` and `test_standalone_programs.py` to
  `integration/test/clouds.py` and the like, which pytest no longer collected; the proof passed
  because a suite that runs fewer tests still exits 0. Moved back; `test_` now always counts as a
  shared prefix, so the next word groups and every moved module keeps it.
- OPEN, the gap that let it through: the fixer's proof judges by exit status, and the gate's
  `suite` lane never checks what it collected (baseline accounting runs only in `inspect`,
  against 392 historical cases). A fix that drops tests from collection passes. Proposal: the
  `suite` lane records its collected count in its report, and the proof refuses a fix set whose
  run collects fewer tests than the run on the project as it stands (one `--collect-only`, a few
  seconds). Needs a design choice on where the comparison lives (the generic Fixpoint or the
  project's gate), so left for the owner.
- `_Relocation(..., package=)` is keyword-only (FBT001).
- `analysis/project/consumers.py` (537 lines, 19 members): what a module offers its importers
  (`surface`, `positional`, the statement bindings as a `_Bindings` single-dispatch class) is
  `project/surface.py`, and where the consumers live (`Workspace`, `module_key`) is
  `project/workspace.py`; `consumers.py` keeps the index and its build. `_build` hands the timed
  search and read to `_indexed`, `_read` leaves its `with` blocks before indexing the rows,
  `_narrowed` words both reasons from one list, and `add_import` takes `name` by keyword. Equal
  surfaces and positions on 2,373 files of every house package, and equal consumer indexes for
  mainboard and patos.
- ALL-CLAS0001 asked for an order Python refuses: implementations registered with
  `@x.register` on a single-dispatch `x` were ranked by their own decorators, so the rule wanted
  them above `x`, where the decorator cannot yet name it (its fix already declined such a class,
  leaving the finding unfixable). They now take `x`'s key, as a property's setter takes the
  property's, and stay beside it.
- `verbs/help.py`: the README scan (`_docs`, complexity 21) is a `_Readme` reader holding its
  heading, fence and paragraph; `show` matches (`_matches`) and prints (`_listed`) apart, and
  `_definitions` filters the public declarations up front. Equal README pieces (378), Python
  declarations (3,888) and `show` output on eight queries. Found on the way: plain `mb help`
  lists a lazily loaded group (`rules`) without its description, since the group is not loaded
  yet; a walk of the commands first would show it, at the cost of loading every group. Left as
  is (OPEN, cosmetic).
- PY-CLAS0009 counted `self.heading = path.name` in `__init__` as a copy of a component's
  attribute, though another method reassigns `heading`; an attribute assigned again outside
  `__init__` is state that starts from the component's value and no longer counts.
- `_PASS_THROUGH` (a SQL fragment) is `_FORWARDED`: ruff's S105 read it as a password.
- `agents/workspace.py`: `Agents` keeps the six verbs its callers use (`at`, `sections`, `sync`,
  `update`, `drift`, `rendered`); the section builders are private. The harnesses are built once
  in `__init__` (a cached property computed the same for every instance); `sync` hands its
  writes to `_written`, and `_server` and `_stale` are comprehensions. Equal `rendered`, `drift`
  and `sections` on the workspace; the agents tests pass.
- `plots/table/core.py`: `_publish` (complexity 17) stages each format through `_staged` (the
  figure, its provenance in the file's metadata field and in a sidecar) and links each through
  `_linked`; `_draw` takes the `Chart` instead of its four fields, the hue colors are `_colors`,
  and `_outputs` finds an existing target in one comprehension. Byte-identical PNGs over every
  kind, hue and named-color style, and identical files with provenance across PNG, SVG, PDF and
  EPS.
- `engines/compile/ecosystems/binary.py`: the curl version check is `_curl_checked`, the ZIP
  member's checks `_member` (still read inside the one `try`, so its refusals are worded as
  before: `MissionError` is a `RuntimeError` the `except` rewraps), `_satisfied` parses its
  receipt alone in its `try`, and `sync` prunes (`_pruned`) and installs (`_installed`) apart.
  The binary-tool tests pass.
- `analysis/providers/swift.py`: finding the toolchain, the helper's identity, its provenance
  record and its compile are a `_Build` model (`found`, `digest`, `recorded`, `compiled`), so
  `_helper` (21 statements, complexity 14) reads as five steps; `_validate` is typed rows,
  covered files and placed rows, and `run` hands the helper's call to `_analyzed`. Equal tables
  on two real Swift projects (dataclass, and usagiunagi's ~120,000 rows); the Swift tests pass.
- `center/migration/remote.py` (shipped to the destination, standard library only): `_write`
  compares and backs up inside `suppress(FileNotFoundError)` instead of `except: pass`, and
  `_checkout` hands its preconditions to `_unfit` (an existing repository: another remote, or
  commits the target lacks) and `_cloned` (a fresh clone into an empty root). Equal to the old
  on throwaway repositories (fresh clone on a branch and detached, a non-empty root, an ancestor
  head, a divergent head, another repository, an unreachable remote) and in every `_write` case;
  the standalone-programs test still loads it into a bare interpreter. No test covered these
  verbs before; the module's member count (21) waits for a regrouping.
- `probe/reverse/device/core.py`: `_evidence` narrows through `_narrowed` (one `match` per result
  type) and assesses each tool run through `_assessed`; cuobjdump runs one method per operation
  (`_sass`, `_resource`, `_listed`, `_inventory`) from a table, and nvdisasm's extraction path is
  `_extracted`, `_disassembled` per image and `_joined`. Equal evidence over 1,575 cases: every
  operation and artifact kind, every architecture and function selector, and each tool run made
  truncated, failing or noisy.
- `probe/reverse/device/parsers.py`, second round: each reader's `read` dispatches to small steps
  (`_flagged` and `_unrecognized` for listings, `_headed` and `_counted` for resources), the
  repeated question is one `_WHICH_TARGET`, and `_entries(text)` is `_Entries.of`. Equal to the
  first round's parsers on 160,000 generated outputs across the four parse functions.
- `render/query.py` (299 statements) is three places: `render/stamps.py` (the printed cell,
  `canonical` and `normalized`, and the multiset stamp `ResultHash`/`Stamp`/`Ending`), and a
  `render/query/` package with `core` (`Query`, reading chunks through `_fetched` and `_fed`,
  the footer through `_footed`) and `sinks` (`Window`, its byte check `_overfull` and suffix
  reservation `_reserved` apart, and the streamed and windowed sinks). `render/tabular.py` folded
  into `values.py`, and its `decode`, read by nothing, is gone. Byte-identical stdout, stderr and
  stamp over 162 renders (9 statements, 3 modes, 6 cap settings, a zero-column statement among
  them).
- `center/migration/migrate.py`: `Migration` takes the board, the destination, `root` and
  `watch`, the only arguments its one caller passes; its transport, home and GitHub readers were
  injection seams nothing used, and are now its own (the two GitHub one-liners inlined where
  they are read). Every step is private, `run` the one verb; `run` (27 statements) reads as
  preflight, `_landed` (the destination reached and judged, a `_Landing`) and `_moved` (sign in,
  carry, clone, install, verify); `extras` is a comprehension over `_installed`. Equal step
  sequences and reports with every step stubbed and recorded, on four paths (unpublished work, a
  blocking finding, a failed clone, a full move). OPEN: the module is still past 200
  statements.
- `dispatch/schedulers/pbs/held.py`: the spool's tagged answer is parsed by `_parsed`, one
  comprehension per tag where a six-way `if` chain stood (ALL-BRAN0001), a claimed job is judged
  by `_claimed`, and `_publish`'s retry reads as refuse, else return once queued or confirmed.
  Equal facts and job states over 3,000 generated spool answers (every tag, repeats, junk rows);
  the held-line tests cover the publish path.
- `fleet/fitness.py`: `Fitness` answers `judge`, the one verb its callers use; its twelve checks
  are private. `engines/compile/ecosystems/stage.py`: `SecondStage`'s scope helpers (`merged`,
  `scopes`, `toolchains`, `overlays`, `shared_scopes`) are private, read by nothing outside.
- ALL-FUNC0002's inliner (`functions_inlined()`) fixed none of mainboard's 15 single-use trivial
  helpers. Four gaps, each now closed: (1) a call naming arguments by keyword was refused
  outright; keywords now map to their parameters by name, when their values are atomic, so
  evaluation order cannot change; (2) `def f(a, *, b)` read as variadic (`, *`), so every helper
  with keyword-only parameters was refused; only `*name` and `**name` count now; (3) a
  zero-parameter helper had no reading to judge and so never counted as sound; (4) any read of a
  parameter inside a comprehension counted as shadowed; now only a comprehension whose own `for`
  targets bind the parameter's name, or a name the argument reads, shadows it (a nested
  function still always does). The fix then inlined 12 of the 15, proven by the suite; the
  other three it rightly refuses (a non-atomic argument read twice, a dict argument with `**`
  read as starred) and were inlined by hand, binding the argument once where needed.
- ALL-ERRO0001 (a handler answering a failure with nothing): the eleven `except ...: continue`
  and `pass` guards are `with suppress(...)` now, each scoping exactly what the `continue` used
  to skip (`return` or `break` moved inside the block where the old code fell through after the
  `try`). `Lockfile.pins` reads a wheel's or an sdist's name by its suffix inside one
  suppression instead of a nested `try`. `Portability.found` and the consumers test grew too
  deep under the new block, so each reads its lines through a private helper (`_code`,
  `_imported`) and filters in a comprehension. `FileBudget` restores through `_release`, and
  `ProcessTree._members` and `Lease._holder` suppress instead of catching (PY-EXCE0002). The
  full suite passed (1123 passed, 11 skipped), proving these and the three hand inlines above.
- `dispatch/hosts/onboard.py` (763 lines, 338 statements) is a package now: `constants` (`TOOL`,
  `SOURCE`), `bootstrap` (the install routes and `Bootstrap`), `record` (`HostSetup`,
  `read_facts`, the facts and GPU commands) and `core` (`Onboarding`), re-exported from its
  `__init__`. `Onboarding.run` reads as undisturbed, then `_onboard` or `_sync`, which share
  `_prepared` (uv's CPython, the mirror) and `_provisioned` (pixi, environment, lake, queue,
  dotfiles); a sync now also announces its pixi and queue checks. `installers()` (seven
  parameters, `source` never passed) is `Bootstrap.routes(vendored=)`, reading the bootstrap's
  own request, floor, extras and constraints. BUG FIXED: the `present` route compared the kept
  tool's interpreter minor (`3.14`) with the whole interpreter request (`>=3.14.0,<3.15`) since
  2026-10-23, so it never held and an index-consuming host reinstalled a tool that already
  satisfied it on every setup; `Existing` now takes the minor from the request, guarded by
  `test_a_satisfying_tool_on_the_solved_minor_is_kept`. OPEN: `Onboarding.__init__` keeps eight
  parameters (PLR0913).
- `profile/providers/nvidia/tracer.py` (529 lines, 351 statements) is five modules now:
  `bindings` (the optional nvtx and CUPTI handles, the CUDA Runtime loaded once through
  `functools.cache` instead of two `global` flags, `synchronize`, and the binding Protocols that
  were `protocols.py`), `constants` (the Activity-to-CUPTI kind table and the two typed kinds),
  `delivery` (`Delivery`, the process's one buffer subscription, whose class attributes replace
  four module globals and a `global _registered`, with `RawGeneric`), `collector`
  (`CuptiCollector`) and `tracer` (`CuptiCallbackSession`, `NvtxTracer`). The buffer callback
  marks its collector tainted before converting and restores it after, which a raising
  conversion leaves set, instead of a `try`/`finally` around the loop; the window check is
  `_checked`, and `_records` takes `since` and `until` by keyword. `CuptiCollector` lost the
  `max_records` knob no caller passed. The probe of supported kinds suppresses
  `NotImplementedError`, so its ALL-ERRO0001 waiver is gone; the other waivers follow the
  moved code.
- `settling/durable.py` (535 lines, 255 statements, no tests) is a package: `base` (`locally`,
  `Every`, `Settling`, `Settler`, `Unsupported`, `complaint`), `systemd`, `launchd`, and `core`
  (the settler registry, `settler`, `schedule`). What both settlers repeated is the base's now:
  the not-installed answer (`_absent`), the pass's command line with the tool's path or the
  refusal naming the runner (`_pass`), and the root stamp (`stamp`); the tool's name is a
  `Settler.tool` class attribute instead of a constant three modules would share. The refusal
  for a missing tool now says "install the tool first" for both (systemd said "the snapshot").
  Equal answers, unit files, plists and command logs to the old module over 96 generated cases
  (systemd active or not, lingering or not, refused or not, three periods; launchd loaded or
  not, bootstrap and ssh refused or not, four periods, a log or none; eight period spellings).
- `manifest/members.py` (557 lines, 253 statements) is a package: `models` (`Package`,
  `Member`), `rebasing` (the path and name rebasing a member goes through: `rebased`,
  `sources_rebased`, `moved`, `namespaced`, `anchored`) and `core` (`Composition`), re-exported
  from its `__init__`. `composed` merges its eight composed tables in one `body |= {...}` instead
  of eight assignments (ALL-FUNC0001, 14 statements to 7). Equal members and composed manifests
  to the old module for the workspace and its five member manifests; the composition tests
  pass. On the way, the move turned a local `moved` into a shadow of the imported function and
  `mb` failed on its first composition until the local was renamed. Ruff's F823 names exactly
  that, but `mb lint` composes the manifest before running any tool, so a broken composition
  cannot be linted by the linter it breaks; OPEN: lint could fall back to the root manifest
  alone when composing fails.
- FIXED, the proof's blindness to dropped tests (OPEN above, from ALL-FILE0005): the fixer now
  hands its proof `MB_FIX_ROOT`, the project as it stands, beside `MB_FIX_MIRROR`, and the gate's
  `suite` lane, when it is set, first counts what pytest collects from both trees
  (`Gate._counted`, a collection-only run in inspect mode, seconds) and refuses a mirror that
  collects fewer, before running anything (`Gate._shrunk`; the counts land in the report as
  `collected`). The comparison lives in the project's gate, since only it knows what a test is;
  the generic `Fixpoint` only names the root. Shown on a copy with `test_clouds.py` moved to
  `integration/moved/clouds.py`: "the fix collects 1117 tests where the project collects 1135",
  exit 1 in seconds, where the same tree passed the suite before (1104 passed). A copy outside
  the monorepo collects the same tests as the workspace's own tree otherwise, so the check does
  not refuse a mirror for where it sits.
- `analysis/engine/fixing.py` (538 lines, 326 statements) is a package: `components` (`Applied`,
  `Components`), `tallies` (the finding counts a fix is judged by: `by_file`, `by_rule`,
  `holding`, `places`, `split`, `typed_growth`, and the three judgments that were `Fixpoint`
  statics, `grown`, `progressed`, `cleared`), `proof` (`Proof`: the project's proof command,
  `passes` and the bisecting `culprit`, and what its last run `said`), `planning` (`Planner`:
  one pass's taken fixes, the files a pass would break joining the shared `held`, the refused
  removals of names other repositories import) and `core` (`Fixpoint`, 218 statements down to
  under 200). The analysis tests and the CLI command tests pass (63). Two more shadowing slips on
  the way (`holding = holding(files)`, `gone := gone(...)`), caught by pyrefly and ty this time
  since the fixer is not on `mb lint`'s own path.
- `analysis/engine/facts.py` (611 lines, 278 statements) is a package: `files` (`listed`,
  `project_files` with its walk fallback as `_walked` (PY-EXCE0002), `lint_excludes`,
  `vendored`, `marked`), `sql` (`Setting`, `SqlNames`, `snake`, `Query`), `budget` (`Budget`)
  and `core` (`Facts`, `_Narrowing`), every public name re-exported, so its forty importers are
  unchanged. The findings over mainboard before and after differ only in moved paths and the
  size findings the splits removed, and `rules check research/reproducibility` runs.
- `analysis/rules/families/calls/core.py` (775 lines): the CallFact family's name resolution SQL
  (symbols, bindings, types, links, reach, resolved) is `calls/resolution.py` and its
  expression-tree SQL (spine, roots, edges, tree, keys, bools, expressions, ancestry, mapping
  entries) `calls/expressions.py`, each a function answering its relations by name, so no SQL
  text becomes a shared constant; `core.py` (332 lines) composes them in the old order. The
  family's 33 relations are equal, in order, to the old module's.
- PLR0912 and EXCE0002 in `manifest/render/interpolate.py`: a template's evaluation reads as a
  call (`_called`) or a dotted lookup (`_looked_up`) instead of one ten-branch method; a `{% %}`
  statement is refused before the `try` and `_exec` returns the run's stripped output in one
  statement. Equal renders and refusals to the old module over 25 templates (names, nested
  tables, calls, unknown names, functions named without a call, statements, `exec` failures).
- ALL-PARA0001's fix (`parameters_design.py`): `_argument_starts` finds the call's own bracket
  through `_opening` and moves its depth by booleans (PLR0912, 8 branches); equal starts to the
  old one over 20,000 generated calls. The repair is `repair` itself now, its two halves
  `_declared` (the empty edit at the `def` and the `*, `) and `_named` (one call's `name=` edits),
  both on a module `_point`, the empty span an insertion takes; it still stops at the first
  call that cannot take the names. Every rule's examples pass (`mb rules test`), PARA0001's fix
  among them. `PythonBase.module_files` is public, the one src private access (SLF001) gone.
- PLR0912 batch: `Line._status` reads the spool's tags into a dict and names its state through
  the statics `_beat` (the serving allocation's fresh beat) and `_state` (equal statuses to the
  old method over 5,000 generated spool answers); `Line.open` builds its spec in `_spec`, and
  `_start` writes its three files in one loop and takes the account from the spec it was handed
  (ALL-FUNC0001). `Syntax.broken` is `_unparsed` (Python and TOML) plus `_grown` (tree-sitter's
  error counts). `resolve` (`dispatch/transport/targets.py`) fills a profile's gaps from a
  table of (needed, value) per field, its root kept by `_kept_root`; equal profiles over 6,912
  combinations of profile, probe, recorded root and home.
- PLR0912 in `git/operations`: each push, pull and check judgment is a function answering one
  question. `push`: `_detached`, `_delivered` (which first sees a commit already waiting on the
  fallback branch), `_diverted`. `pull`: `_following`, `_reattached` and `_levelled` each answer
  a held step or the note of what they did. `check`: `_distance` and `_line`, and `_pointer`
  per submodule. `commit`: `_holder` resolves and refuses one named path, `_given` commits named
  paths, `_merging` merges after a done sweep, and `Intake` is `operations/intake.py`. Equal
  steps, details (commit ids aside), remote refs and checkouts to the old code over real git
  worlds: push in 8 situations (level, ahead, behind, both detached cases, protected, protected
  and already waiting, an untracked branch), pull in 10 (a clean and a conflicting divergence,
  fast-forward, both detached cases, following a parent's pointer on a branch and detached, no
  upstream, an unfetched remote), check in 9 with a submodule (moved, unpushed, not checked
  out). Nothing tested push before; the workspace tests pass.
- `Scheduler.logs` reads the job's log file by default (`read_log`), so the local, Slurm and PBS
  schedulers no longer repeat it (ALL-DUPL0001); pueue keeps its own. `build_qsub_flags` is a
  table of (needed, flags) and `qstat -f` parses block by block (`Pbs._attributes`), the
  module's helpers now `Pbs` statics (ALL-FUNC0003); equal flags and records to the old module
  over 64 resource combinations and 5,000 generated `qstat -f` answers.
- UNVERIFIED (2026-10-25, the shell's permission check was down): `profile/profiler.py`
  (`_armed`, `_capturable`, `_measured`, `_windowed`), `verbs/delimiter.py` (`_command_at`,
  `_skipped`) and `trials/store/dataset.py` (`Dataset._tagged`) were edited by hand; lint, an
  old-against-new comparison and the suite are owed before anything else lands.

## Planned: `.plot.sql`, one figure per file (owner, 2026-10-25)

Make `mb query` and the plot extensions easier, more powerful, reproducible and consistent.

- A `.plot.sql` file is one figure: plain SQL whose leading `--` comments carry the whole
  specification, so the query, the marks and the numbers the text quotes live in one diffable
  file. `mb plot fig.plot.sql` renders it; `--watch` previews live, `--set key=value` tries a
  limit, `--panel a` draws one panel, `--check` fails when any recorded input changed.
- `-- figure:` gives the output paths, size and style name; the style extends the project's
  `plots.toml` (itself over the root `mainboard.toml` styles), never replaces it.
- `-- with <name>` blocks are shared relations, defined once and visible to every panel, so a
  rule such as "three repetitions with controls within 10%" is written once.
- `-- panel a` blocks are each a `SELECT` returning tidy columns, then `--` lines naming the
  layers (`Range`, `Dot`, `Line`, `Rule`) and the columns mapped to x, y, color, marker, fill and
  the interval bounds; category order, per-figure display labels, axis scale, limits, ticks, and
  one legend entry list that also covers reference lines and says what a hollow fill means.
  Panel letters and shared axes are automatic.
- `-- guard:` lines are selects that must return no rows, or the render fails.
- `-- macro Name digits=2 bound=lower:` lines are scalar selects written to the LaTeX numbers
  file, so figure and text read the same relations.
- Every render writes a sidecar: the SHA-256 of each input file and of the `.plot.sql` itself,
  the lake snapshot, row counts and the resolved style; `--check` compares against it.
- `-- draw: module:function` hands one panel to Python for what the marks cannot express.
- Acceptance: re-render the two merge-kernel panels (Fig 6, cutok MLSys) and one ICLR figure
  from `.plot.sql` files, with identical macros and visually identical output.
- Build on what exists: `mb paper plot`, `plots/table` (staged/linked/sidecar), the `Query`
  render layer and the DuckDB `Results` relations; DuckDB SQL only, no pandas.

## Planned: a structural edit language for fixes (2026-10-25)

Most refactors done by hand this session were the same few structural operations, spelled as raw
text edits or not automated at all. Proposal: rules emit typed operations over the symbol facts,
and per-language adapters render them, so one grammar fixes Python, Swift, Rust and TypeScript
alike and the codebases come out uniform.

- Operations: `Move(symbol, to=module)` (references and imports respelled, re-export left when
  the old path is a consumer surface), `Rename(symbol, to)` (refused when the new name is bound
  in any scope a reference sits in: the `moved = moved(...)` and `holding = holding(...)`
  slips), `Privatize(symbol)`, `Split(module, groups)` into a package with a re-exporting
  facade, `Hoist(method, to=base)` for identical overrides, `Extract(span, name)`,
  `Inline(helper)`, `Reorder(class)`, `CreateFile(path, header)`, `AddImport`/`DropImport`.
- Language adapters own what differs: import syntax, visibility (`_name`, `private`, `pub`,
  `export`), file layout (`pkg/__init__.py`, `mod.rs`, Swift targets), and where a module's
  shared constants live.
- New autofixes that become cheap with it: ALL-ERRO0001 (`try/except: pass|continue` as the
  last statement to `with suppress`), ALL-FUNC0003 (module helper into its one class as a
  static), ALL-DUPL0001 (hoist an override repeated in every subclass), ALL-REAC0002 and
  PY-CONS0001/0002 (privatize, or move to `constants.py`), ALL-MODU0001/0006 and ALL-FILE0003
  (split a module or folder along its call graph, review applicability), PY-EXCE0002 (move a
  `try`'s trailing statements to `else`, unsafe applicability).
- Safety the engine adds once, for every operation: the scope-collision check above, the
  proof's collected-test count (done), and a generated old-against-new harness over the moved
  callables where the facts know their inputs.

## Structural operations in the fix language (2026-10-25, started)

- Verified the three hand edits from when the shell was down: `Delimiter._command_at` (equal to
  the original scan over 150,000 generated argvs, its branches flattened to three),
  `Dataset._tagged` and the profiler's `_armed`/`_capturable`/`_measured`/`_windowed`; the suite
  passed (1124 passed, 11 skipped).
- ALL-FUNC0003 flagged five helpers whose one call sits in the class body itself
  (`relations = _relations()`), which runs while the class is being built, when no static
  method of it can be called; its fix rightly refused them. The rule now asks for a call inside a
  method (`sole_reference_owner_definition_start_line IS NOT NULL`): 5 findings to 0.
- Two new file edits. `Create(path, text)` writes a new file, refused when the path is taken.
  `Append(path, text)` adds to the end of a file, creating it when missing; appends to one file
  stack in the order taken (`TextEdit.stacks`), so many fixes may add definitions to one shared
  module in a single pass. A fix row states them with `operation` `create` or `append`. `Plan`
  reads a missing file as empty, `Mirror.apply` makes the parents, and `Source`/`TextEdit` moved
  to `engine/sources.py` (apply.py was past 200 statements).
- Structural operations (`analysis/operations/`): a fix row naming any other `operation` is
  handed to a registered `Operation` (patos `Registry`, keyed by kebab name), which renders the
  edits from the row's columns or refuses, a refusal withholding the row's whole group. The SQL
  gathers what moves and its context; Python, per language, writes it. Two so far:
  - `move-constants` (PY-CONS0002, now `review`): one row per defining module. Every literal
    constant other modules use goes, with the comments above it, to the `constants.py` beside its
    module (inside a package); every `from ... import NAME` is respelled (a mixed statement split
    in two), every `module.NAME` reads `constants.NAME` and the old import is respelled once
    nothing else reads the module through it; the old module imports back what it still reads
    (identifier nodes and `__all__` entries, not comments). Refused: non-literal values, a name
    the destination holds otherwise, an importer binding `constants` already, a module someone
    star-imports or that sits beside a `constants/` package, a constant some use of which the
    facts did not count.
  - `raise-constants` (PY-CONS0003, now `review`): each late constant moves, with its comments,
    to just below the last import or the last statement binding a name its value reads.
- Tried on a scratch git copy of research/reproducibility (1681 tracked sources): one pass moved
  the shared constants of 47 modules and raised 102 late ones; findings 2264 -> 1563 over 240
  files, type diagnostics 1362 -> 1362, ruff's pyflakes findings identical before and after, and
  of the 240 changed modules none that imported before fails now (each imported in its own
  process; six test modules that died on `triton` now import their constants without it). 129
  groups stay deferred every pass: files sealed by registrations, which no fix may touch.
- The first version moved one constant per row and judged uses by regex: a docstring mention
  re-imported a constant nobody read, and constants reached through one `import` could never
  drop it, each seeing the others' uses. Grouping by defining module and counting `ast` reads
  fixed both.
- BUG (mine, from the fixer split): `mb rules gate --fix`, which `mb lint` runs over changed
  files, read `Fixpoint.refusals`, which had moved to the planner; `mb lint` died with an
  AttributeError once a safe fix applied. `Fixpoint.refusals` is a property again. OPEN: no test
  runs `rules gate --fix` with a fix that applies, which is why the suite passed.
- Two more operations, and the operations split into a package (`operations/base.py`,
  `operations/python/` with `source`, `imports`, `moves`, `classes` and `constants`, and
  `operations/toml.py`):
  - `pull-up` (ALL-DUPL0001): a base declaring the repeated method only as an `@abstractmethod`
    placeholder takes the first copy in its place, the stub's docstring kept when the copy has
    none, and the copies go with the blank lines above them. A base defining the method for real
    is still left to an intermediate class. The engine's type check is the guard: on mainboard
    the pull-up of `Edit.file` reads `self.path`, which `Edit` does not declare, and ty and
    pyrefly refused it.
  - `drop-exclude` (ALL-CONF0002, `safe`): a rule exclude matching no file leaves its array, with
    its line when it stands alone there, cut from a one-line array otherwise. The first TOML
    operation.
- BUG: `manifest_rule_excludes` placed each stale exclude on the first line quoting its pattern
  anywhere, so the 74 stale `datasets/**` of cuda-tokenization's 74 rule tables all read line
  11, and a fix of each collided with every other: one per pass, 74 passes. It is now the line
  quoting the pattern inside the rule's own table (`ProjectManifest._place`); one pass.
- BUG: `Mirror` linked the Ruff configuration a project `extend`s even when no file stands there
  (a project copied out of its workspace), and the fixer died with FileNotFoundError before its
  first pass. An extension naming no file is none (`Mirror._extension`).
- `raise-constants` now lays out what it moves as a formatter would: the blank lines above the
  constant leave with it (`Module.cleared`, shared with `pull-up` and `move-constants`), and it
  arrives a blank line below an import, two below a function or a class.
- Tried on a scratch copy of research/cuda-tokenization: one pass dropped all 74 stale excludes
  and moved `COMPACT_READBACK_BYTES` into `cutoken/tokenizer/constants.py`, respelling its
  importer; findings 973 -> 898, type diagnostics 1638 -> 1638. On mainboard the full fixer kept
  one `raise-constants` move (`rules/core/constants/calls.py`) with the suite proof passing, and
  withheld the `Edit.file` pull-up on its type error. Every rule example passes; the suite passed
  (1124 passed, 11 skipped).
- ALL-DUPL0005's fix takes any plain string of word, path and flag characters, not only
  lowercase identifiers (`staged.txt`, `/usr/bin/xcrun`, `--version`, `runtimeWarnings`), named
  by its words in capitals (`_USR_BIN_XCRUN`, `_RUNTIME_WARNINGS`); a capitalized word or a
  dotted name ending in one stays out, being most often a string annotation. On mainboard the
  fixer kept four (test_launch, evidence/test_core, lake/test_query_stream, swift/test_analysis)
  with the suite proof passing.
- PY-CONS0001 at scale (scratch copy of research/reproducibility): its existing rename fix took
  596 of 2994 findings in one pass; with PY-CONS0002 and PY-CONS0003 alongside, 5093 -> 3433.
  Of the 1330 PY-CONS0002 left, 1202 sit in registration-sealed files, 104 are no lone literal
  (`HERE = Path(__file__).parent`, which a move would change), 17 are imported from a module
  outside the project and 7 are unexplained.

## 2026-10-25: `mb refactor` tools, constants by depth, long literals, model naming

- `mainboard rules` is now `mainboard refactor`: the rules (`check`, `fix`, `gate`, `catalog`,
  `explain`, `test`) and the tools beside them. New tools:
  - `refactor sql QUERY [ROOT]`: rows of any fact view, relation or table macro, those only a fix
    reads included (`imported_names()`, `long_literals()`).
  - `refactor proof [ROOT]`: the project's own proof over a copy of it as it stands. Its first
    run caught a real failure the `rules` -> `refactor` rename left (test_analysis_explain
    expected "rules catalog"), fixed.
  - `refactor trial [ROOT] --rule ... [--unsafe] [--imports] [--patch FILE]`: the fixer on a
    git copy, ROOT untouched; reports the fixer's lines, the files changed, the pyflakes and
    syntax findings added or removed, and with `--imports` every changed module that imported
    before and no longer does (each in its own process, the "before" from a worktree of the
    copy). Exit 1 on an added finding or a broken import. `--patch` writes a `git apply` patch.
  - `refactor move PATH::NAME... --to MODULE [--write]`: moves top-level definitions (functions,
    classes, assignments, with decorators and the comments above) into another module, created
    when missing; the imports they read come along respelled from there, every importer is
    respelled (`from ... import` and `module.NAME`), the old module imports back what it still
    reads. Refusals say why (a read of a name of the module that stays, a type-checking-only
    project import, an aliased `import`, a cycle). Without `--write` it prints a diff.
  - `refactor check --new` and `refactor gate --new`: only what each file gained since it was
    last checked (`Seen`, in the project's state folder). After the first check a file never
    recorded is new and all its findings are.
- The mirror copies files instead of hard-linking them: an in-place edit made while a proof ran
  reached the proof through the shared inode (`Mirror._copied`; 0.3 s for mainboard).
- PY-CONS0002's `move-constants` picks the depth: a constant goes to the `constants.py` of the
  lowest package holding its module and every user of it in its own top package (beside the
  module when nothing else shares it), and every import is respelled from the importer's place,
  relatively when it was relative (`Place.spelled`). A user in another top package (a test
  suite) keeps its full name. On a scratch copy of research/reproducibility: 184 files changed,
  2040 -> 1415 findings, no pyflakes finding added, no type diagnostic added, no import broken.
- `move-constants` is now `move-definitions` restricted to literals (`MoveConstants.literal`):
  one operation moves any top-level definition, the constants' move being its special case.
  `Module.block` reads functions and classes with their decorators; `Module.binds` is scoped (a
  method's locals no longer count as module names, which also made `raise-constants` anchor a
  constant lower than it could stand).
- The importer queries are relations of their own (`imported_names()`, `reached_names()`), read
  by PY-CONS0002's fix and by `refactor move`.
- PY-CONS0004 (review): a long literal inside a function: code-shaped text from 40 characters (SQL,
  a pattern, a shell line, one token such as a flag list) or a number from 1000 or from four
  significant digits; prose messages and file modes stay. Its `name-constants` operation lifts a
  module's literals to module constants after the last import, every occurrence of a literal
  reading one constant. The name comes from where the literal stands (the keyword, parameter
  or variable it is the whole value of) or else from Claude Haiku 5.5 through `claude --print`
  under a JSON schema (`operations/model.py`, ~6 s and under $0.001 a module), which only ever
  chooses the identifier; answers are cached by question in
  `~/.cache/mainboard/analysis/names.json` with the names already given to each literal, which
  the next question lists so a value keeps one name across the project.
- PY-CONS0005 (review): one private constant defined the same way in several modules of one top
  package; its `share-constants` operation defines it once, public, in the `constants.py` of the
  lowest package holding them all, and each module imports it.
- On mainboard, PY-CONS0004 alone: 62 files, 185 -> 69 findings, no pyflakes finding or type
  diagnostic added, every changed module still imports. Before the vocabulary, 1024 came out as
  `_BYTES_PER_KIB`, `_BYTES_PER_KB`, `_BYTES_PER_KIBIBYTE` and `_BINARY_UNIT_BASE`.
- A Stop hook (`.agents/hooks/refactor_new.py`, in `.agents/settings.json`): when a turn ends,
  each `[tool.mcmr]` project holding a Python file modified since the hook last ran checks
  `--new`, and new findings hold the turn with the report; a held turn stops the second time.
- `mb agents sync` rendered `.codex/hooks.json` for the first time, and its commands read
  `$CLAUDE_PROJECT_DIR`, which Codex does not set; both hooks now fall back to the git root
  (`${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}`), checked with the variable unset.
- OPEN: PY-CONS0004 classifies a prose message with `${NAME}` in it as shell
  (`_PLAIN_REFERENCE_REQUIRED_MESSAGE`).
- OPEN: `move` leaves import order and unused imports to `mainboard lint`.
- `refactor preview ROOT --rule ... [--unsafe] [--only FILE]`: one pass of fixes as a diff, before
  any proof (what the fixer would try, a fix it would withhold included). It found the next
  three defects in minutes.
- BUG (found by the fixer's own guards on the combined constants trial, mainboard copy): an
  implicitly concatenated string spanning lines, lifted to module level, kept its continuation
  indent (`unexpected indent`); `name-constants` now parenthesizes a multi-line value of more
  than one string token (`NameConstants._spelled`).
- BUG: `share-constants` dropped a `Final` annotation's import (`unknown-name` in the
  `constants.py`; the type checkers narrow `destination != SSH_FAMILY` through `Final`, so it must
  move). The arrival now goes through `Relocation.carried`, the same import carrying as
  `move-definitions`.
- BUG: `_SECONDS_PER_HOUR = 3600` and `= 3600.0` formed two groups claiming one public name; the
  float won and a function returning `int` broke. A public name two values of a package claim is
  now left alone.
- BUG: a module reading `_SPACE` only inside `'{_SPACE}'` templates `.format` fills got an unused
  import and lost the definition its template needs; a module spelling the private name more
  often than its syntax tree reads it is refused.
- PY-CONS0005 needs a name of two words or more and no `VERSION`: `_BATCH`, `_JSON`, `_VERSION = 1`
  in two modules are as often two things (two schema versions merged would bump together).
- `refactor trial` settles the imports a fix leaves unused or unsorted as `mainboard lint` would
  (`ruff --fix --select F401,I001` on the changed files) before counting, and always checks
  imports (no flag).
- After those fixes, PY-CONS0005 on the mainboard copy carrying PY-CONS0004's constants: 24 fixes
  kept, none withheld, 151 -> 85 findings, 63 files, no type diagnostic, pyflakes finding or
  broken import added. Depth as intended: `dispatch/backends/constants.py` (vast and runpod),
  `probe/facts/constants.py`, `profile/constants.py`, cross-package names in
  `mainboard/constants.py`.
- Model naming quality: the vocabulary converges (1024 -> mostly `BYTES_PER_KIBIBYTE`), but some
  names are wrong (`BYTES_PER_GIBIBYTE = 1024`); PY-CONS0004 stays `review` for that reason.
- PY-DOCU0001 treats a function listed bare in a tuple or list of a module registering cyclopts
  commands (`_COMMANDS = (sql, move)`) as a command, so its `Args:` heading stands.
- ALL-CLAS0001 stays `review`: reordering can swap two `@field_validator`s of one field, which
  pydantic runs in definition order.

## 2026-10-25 (evening): DuckDB 2.0 / DuckLake 1.1 adoption (Haiku surveys, Sonnet reviews)

Five Haiku 5.5 surveys, each reviewed by a Sonnet 5.5 agent; reports in the session scratchpad
`duckdb2/{A..E}-*.md` and `*-review.md`.
- BUG (fixed): on DuckDB 2.0 `get_table_names(qualified=True)` answers `runs AS r` for an aliased
  table, so `mb query "SELECT count(*) FROM runs r"` failed (Catalog Error) while `FROM runs`
  worked. `state/storage/parsed.py` `names(sql)` reads tables and table macros off DuckDB's own
  `json_serialize_sql` tree; `Views._needed` and the analysis `SqlNames._parse` both use it. It
  matched sqlglot on 1145/1145 catalog SQL texts (cold 1.0-1.2 s vs 1.44 s). sqlglot is no longer
  imported by mainboard; drop `"sqlglot>=30.0"` from pyproject in a quiet window (a member
  manifest edit reinstalls the live env), then `uv lock` + `mb ci packages/mainboard`.
- `Seen` and `Namer` load tolerantly (`json.JSONDecodeError`) and write atomically (staged file +
  `os.replace`): a reader racing a `write_text` writer failed on 31% of reads in the review's test.
  A DuckDB file for `Seen` was rejected: a second process fails on the file lock.
- Rule settings: `Rule.constants` (ClassVar) replaces six `parameters()` overrides; three counts
  moved into SQL (`len(list_distinct($x::VARCHAR[]))`, `$discovery = 'explicit'`); net -37 lines,
  bindings verified identical by a reviewer.
- One role-tagged recursive closure in `symbols/reach.py` (was two); `UNION` instead of
  `UNION ALL` + `DISTINCT` in the CUDA class ancestors.
- A rule's failing query now carries `in the query of rule <id>` (DuckDB 2.0 errors carry a
  caret snippet but no rule name).
- SECURITY (implementing): the lake served over Quack is writable by any holder of the one token,
  over the `ssh -R` tunnels too; a client's READ_ONLY attach is not enforced (a client dropped
  tables and expired snapshots, erasing time travel, in the review's scratch test). Fix: a second
  endpoint on port+1 whose server opens the DuckLake READ_ONLY (enforced), its own token, plus
  `allowed_directories`/`enable_external_access=false`/`lock_configuration`; read access attaches
  it (`MB_LAKE_READ`, `MB_LAKE_READ_TOKEN`). The writer token keeps full power (narrowing it needs
  SQL parsing, ruled out). Deployment (restart `mb lake serve`, second `-R`, host env) is the
  owner's.
- DuckLake 1.1: the center lake is already 1.1 (since 2026-10-22). OPEN (owner): the
  research/cuda-tokenization project lake is `1.1-dev1`, which DuckDB 2.0.0 refuses even READ_ONLY;
  migration is one-way (backup steps in `B-review.md`). OPEN (owner): the `gpu` env in mb.lock pins
  `duckdb 2.0.0.dev2609250715`; check hosts and re-lock.
- Maintenance: `upgrade()` gets the maintenance lock; `_set_apart` becomes `ALTER TABLE SET`
  inside the creating transaction (implementing).
- Not adopted, with reasons: triggers and RETURNING (unsupported on DuckLake tables), VARIANT for
  trial columns (no gain over `->>`, frozen papers read the trees), nested schemas (rejected in
  `search_path`), `SET VARIABLE` (per connection), `USING KEY` for min-depth closures (keeps the
  latest row), data-modifying CTEs on the lake (no RETURNING).
- BUG (fixed): `Seen` recorded a check of a few rules as if they were all, so the next full check
  reported every other rule's findings as new (808 at once). It now records only the rules a
  check ran.
- BUG (fixed): the Stop hook checked every Python file modified since it last ran, so concurrent
  sessions received each other's findings (the MLSys session was handed `state/storage/parsed.py`
  and fixed it), and one session's check consumed another's new findings. The hook now gates
  only the files its own transcript's Edit/Write calls name, through `refactor gate --new` under a
  per-session record (`MB_REFACTOR_SESSION` -> `.mainboard/refactor/sessions/<id>.json`, started
  from the project's `seen.json`). Verified: a never-seen probe file edited in a synthetic
  transcript is reported; this session's own files with nothing new pass.
- Lake serving after review: a host with only the read pair reads the served lake (it silently
  read a local lake before); token files are 0600; the served connections get a real
  `temp_directory` (a spilling server-side query failed before, pre-existing); the served-lake
  fixture provides both pairs. Gate on this Mac: lake/test_core.py 4 passed, 3 skipped (the served
  tests run only on the Linux TCP lane), so the served path is proven by scratch scripts only.
  OPEN: run the Linux lane; the owner deploys (restart `mb lake serve`, second `ssh -R` for
  port+1, `MB_LAKE_READ`/`MB_LAKE_READ_TOKEN` on hosts that read). Quack PR #329 (merged
  2026-10-22) will reject client READ_ONLY attaches; ours sends only TOKEN, unaffected.
- Upstream: `get_table_names` alias form is by design (asserted in duckdb's tests); Quack ignoring
  a client READ_ONLY attach is duckdb-quack #327 (fixed by #329). Drafted for duckdb-web, awaiting
  the owner: the Quack security page's read-only authorization examples refuse every client.

## 2026-10-25 (night): every lake on DuckLake 1.1, every dependency on its latest release

- Lakes: the center lake (/Volumes/PORTABLE/mainboard-lake) and research/reproducibility's were
  already 1.1. research/cuda-tokenization/.mb/lake.sqlite (1.1-dev1, unreadable by DuckDB 2.0.0,
  last written 2026-10-11) is now 1.1: catalog backed up to
  /Volumes/PORTABLE/lake-backups/cuda-tokenization-pre11-2026-10-25.sqlite (no data files), then
  `Lake(...).upgrade()`; 17 tables and 3 snapshots read back. Left as they were on purpose: the
  2026-10-22 backups and the old workspace copy under /Volumes/PORTABLE/projects (1.1-dev1).
  Hosts' staging lakes migrate at their next `mb host sync` (CVLAB hosts are down in an outage).
- BUG: `mb lake upgrade` never reached cuda-tokenization's lake: it walks `nested_projects`, and
  that project lost its own manifest on 2026-10-07, so its `.mb` lake is an orphan nothing reads.
  Its docstring named it as the example; fixed to name reproducibility. The orphan lake was
  deleted on the owner's word (catalog backup kept in /Volumes/PORTABLE/lake-backups/).
- `mainboard upgrade`: every requirement raised (harfbuzz 15, starlette 1.10, kornia 0.9,
  importlib_metadata 10, fastapi 0.162, ...); `sqlglot` dropped from mainboard's pyproject. The
  proof suite passes on the new lock; `mb lock --check` is clean.
- BUG (fixed): mb.lock kept `[environments.gpu]`, an environment the manifest no longer declares,
  pinning `duckdb==2.0.0.dev2609250715` through a stale editable-mainboard record; nothing pruned
  sections of retired environments. `Lockfile.put` keeps only declared environments, and
  `mb lock` drops undeclared sections (`Lockfile.retain`). The section is gone.

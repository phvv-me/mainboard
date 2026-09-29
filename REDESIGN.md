# mainboard state redesign: objectives, phases, status

Living tracker, updated as each step lands. Last update: 2026-09-29.

## Objectives (from the owner)

1. `.mb/` (legacy `.mainboard/`) is fully git-ignored and fully managed by the CLI; nothing in it
   is hand-maintained, and everything durable in it is reachable through DuckDB/DuckLake.
2. Only `mb.toml` (legacy `mainboard.toml`) and `mb.lock` are committed; mb writes no git objects
   of its own (it reads git, and `mb git` manages repos).
3. `mb` is an alias of `mainboard` defined once (`[project.scripts]`); every name answers to both,
   hosts running the old release keep working (read both names indefinitely).
4. No duplicated caches of other libraries (HF, wandb); no large committed files; zstd level 3
   where compression applies.
5. Minimal abstractions and dependencies: drop wandb, polars, seaborn/pandas, loguru, jinja2;
   keep matplotlib (paper figures, `[plots.paper]` styling), numpy, modal (optional).
6. DuckLake is the state store (speed is secondary to the heavy jobs; import speed later).

## Decisions

- DuckDB `2.0.0.dev2609250715` pinned; move to `2.0.0` stable when released (2026-10-21).
- DuckLake with a SQLite catalog at the latest spec (`1.1-dev1`); catalog `.mb/lake.sqlite`,
  data `.mb/lake/`; append-only tables, "current" state as views.
- Attach rules: `CREATE_IF_NOT_EXISTS false`, `OVERRIDE_DATA_PATH true`, WAL, busy timeout; `at`
  is reserved in DuckDB 2.0 (columns use `ts`); maintenance (`CHECKPOINT`) under a file lock.
- Source provenance: commit SHA + zstd patch in the lake (no zip archives, no git refs).
- Activation lives in `mb.toml`, computed in Python (no generated `.sh`/`.bat`).

## Phases

| # | Phase | Status |
|---|---|---|
| 1 | Names and aliases (`mb`, `mb.toml`, `.mb/`, `MB_*`, markers read both) | done, on main |
| 2 | Committed `mb.lock` (all envs, byte-exact, adoption of legacy cache locks) | done, on main |
| 3 | Lake foundation (`state/lake.py`, schema, `mb center migrate-state` importer) | done, on main |
| 4 | Move every writer into the lake; drop wandb; structlog | done, on main; `D:\projects` migrated |
| 5 | Activation from `mb.toml`; drop generated scripts and jinja2 | done, on main |
| 6 | Source in the lake, pins in the HF cache, query over the lake | done, on main |
| 7 | Remove old paths, polars -> DuckDB, seaborn/pandas out, regression tests, center setup | pending |
| 8 | API review, remote connectivity, miyabi-g multiplexing on Windows | pending |

## Step checklist (worked strictly in order, one at a time)

Legend: [x] done and committed, [~] in progress, [ ] pending.

Phase 4 (branch `mb-writers`)
- [x] 4.1 Run/host registry (`dispatch/db.sqlite`) -> `runs_log`/`host_facts` + views (`7de5835`)
- [x] 4.2 Shared lake `Session` (lazy attach, reattach on locked/stale) (`7de5835`)
- [x] 4.3 Batch event journal (`events.ndjson`) -> `events` via `Journal` (`f762aae`)
- [x] 4.4 (`48aaff1`) Drop the wandb mirror: receipts are the record, `[tracking]` is just `interval`; one
      session serves threads by taking turns (a shared DuckDB connection deadlocks otherwise)
- [x] 4.5 (`8bd88ee`) structlog replaces loguru and stdlib loggers; `from mb import logger`; `mb` import alias;
      trial-bound events routed to the trial spool; remove loguru
- [x] 4.6 Harvested receipts (`receipts.ndjson`) -> `receipts`
- [x] 4.7 Captured job logs (`<handle>.log`) -> `log_lines`; the durable pass logs to the systemd
      journal instead of `monitor.log`
- [x] 4.8 (`165e52c`) Costs ledger -> `costs`
- [x] 4.9 GPU quotes catalog -> `quotes` (each save one stamped roster, the newest is the catalog)
- [x] 4.10 (`3e27892`) Pulse -> `pulse`, a row per growth; one shared session per lake per process
- [x] 4.11 (`cb5b097`) Holds -> `holds_log`
- [x] 4.12 (`b3884b9`) Studies -> `studies`
- [x] 4.13 (`44b84b0`) Digests (mirror, collection) -> `digests`, rows only for moved stamps; the remote agent keeps its file
- [x] 4.14 (`2bf1f46`) Job specs / closures -> `job_specs` / `closures` (staged files are only what the mirror ships)
- [x] 4.15 Full suite green; `mb-writers` fast-forwarded into main
- [x] 4.16 Imported `D:\projects` (every source matched, logs rebuilt byte for byte; a 32 MB lake
      against 3.2 GB of files) and reinstalled `mb` from source; the old files stay until the
      owner deletes them

Phase 5
- [x] 5.1 No activation in the state directory: each environment's activation lives in its own
      directory (`<state>/envs/<env>/activate.sh`, as prefixes already did), built from `mb.toml`
      by plain Python; `eval "$(mb activate)"` enters it. Kept on disk because second-stage
      binaries need the manifest, which a job's entry should not load.
- [x] 5.2 jinja2 removed: the activation is plain Python and manifests use a literal-only
      `{{ name }}` / `{{ fn('arg') }}` evaluator

Phase 6
- [x] 6.1 (`4a26d72`) Source kept in the lake instead of zips: each file's bytes once in `blobs`
      (by SHA-256) and the listing in `closures`; `SourceTree.restore` rebuilds a tree. Changed
      from "git SHA + patch": the workspace spans 50 submodules and untracked files, and an
      unpushed commit can vanish; content addressing gives the same dedup exactly. D:\projects'
      198 zips (812 MB) imported as 1,612 blobs (186 MiB raw, +61 MB lake).
- [x] 6.2 (`6d35575`) Pins staged as links into the HF cache; D:\projects' 63 pins (1.6 GB, 57 of
      them held nowhere else) moved into `~/.cache/huggingface/hub` with verified links left.
- [x] 6.3 (`5e34f6e`) No separate report verb: `mb query` reads `lake.<table>` beside results.

Phase 7
- [~] 7.1 polars: gone from every state path (lake staging is DuckDB `unnest`, `95d097a`) and no
      longer imported by any command that does not read results (`863723e`). Kept in `trials`,
      whose public API (`log.table`, `log.read_table`) hands experiments polars frames.
- [ ] 7.2 seaborn/pandas: DECISION. `seaborn.objects` drives 216 figure layers (mostly the ICLR
      2027 cutok paper); replacing it re-implements its grammar and re-verifies every figure.
      Both live only in the optional `[plot]` extra, so no host installs them. Recommend keep.
- [x] 7.3 (`9feb360`) The file receipts transport removed; the center migration carries the lake
      (catalog as a SQLite backup, immutable Parquet as is). The old files in D:\projects stay
      until the owner deletes them.
- [ ] 7.4 Center verify: blocked by the stale default lock (every env gate syncs the env first).
- [~] 7.5 CLI import 894 -> 765 ms; the rest is plumbum/filelock/cyclopts at import time.

Phase 8 (added 2026-09-28)
- [~] 8.1 API review: `setup --sync-only` dropped for `sync` (`1a561ee`); verbs otherwise orthogonal
- [x] 8.2 All 6 key-less hosts answer `mb compute` in 5 s; `mb shell --on` fixed on Windows
      (`997878f`, exec unquoted args). Setup of hosts blocked by the stale default lock (below).
- [x] 8.3 No Windows ssh can multiplex; `mb unlock <host>` keeps one agent on
      `~/.ssh/mb-agent.sock` every command adopts. miyabi-g needs the owner to run it once
      (key passphrase).

Owner decisions pending: (1) BLOCKING remote setup/sync: the default env lock is stale (uncommitted
mainboard.toml edits add remaster-lab/m2/glinet deps; mainboard's own deps changed) and the
re-solve fails on Windows (zzzeeksphinx -> libsass sdist needs MSVC): restrict/drop zzzeeksphinx
or solve on Linux. (2) pushing `main`/branches. (3) deleting the migrated legacy files.
Tests: per the owner (2026-09-28) tests are not maintained for now; one known failure in
tests/dispatch/test_provenance.py (archive test edits the job file).

## Workspace state (D:\projects)

- `mb.lock` committed (`c4683155a`), seeded from gold's recovered default-env lock (2026-09-25).
- Re-solve of `default` fails on Windows: `libsass` sdist needs `distutils.msvc9compiler` to lock
  linux-aarch64. Owner decision pending (restrict `zzzeeksphinx`, or solve on Linux).
- `cutile` env has no surviving lock; needs `mb install cutile --resolve` when first used.

## Known costs (accepted)

- Registry op in the lake: ~40 ms in a held session, ~130 ms with a fresh attach; lake create
  0.3 s. Reusing one DuckDB instance per process would halve attach cost (later).
- `mb --version` 1.8 s warm (eager imports of dispatcher/estimate/lake/polars); lazy CLI imports
  planned for later.

## Round 2 (2026-09-29): simplify, then test

- CLI shaped like pixi/uv (`f1a692c`): `install` (never solves), `lock`, `add`, `remove`,
  `upgrade`, `run`, `shell`, `shell-hook`; `mb host setup|sync|list|facts|gpus|unlock|hold|
  release`; `mb job submit|list|logs|wait|cancel|verdict|monitor|collect|batch|lanes`;
  `mb self update`. Nothing runs on start any more (no self-update, no agent lookup).
- polars out of mb's dependencies (only in the `plot` extra); DuckDB does query/export.
- Bulk data reaches DuckDB as staged NDJSON: this DuckDB build binds list parameters at ~2 ms an
  element (20k rows = 40 s), which hung `mb query`.
- `integration/` is the default suite (`22f0e7e`): real `mb` processes, no tracebacks allowed;
  `MB_REMOTE_WORKSPACE=D:/projects pytest integration/test_remote.py` reaches the fleet.
  Fleet today: gold, crimson, pedro-cvlab answer; macmini, pedro-home, purple need
  `mb host setup` (blocked by the default lock decision); miyabi-g needs `mb host unlock`.
- Command groups completed the pixi/uv way: `mb list` / `mb tree` (pixi's, frozen, no
  re-implementation), `mb completion bash|zsh|fish`, `mb self version`, and a `mb lake` group
  (`check`, `compact`, `upgrade`, `import` = the old `center migrate-state`, `serve`), the lake's
  maintenance having had no verb at all.
- Quack studied and measured (DuckDB 2.0.0.dev2609250715, quack 974927a394):
  - `mb lake serve` opens DuckDB on the lake's catalog and serves it on localhost:9494;
    `MB_LAKE=quack:localhost:9494` + `MB_LAKE_TOKEN` makes every attach that lake, same `lake`
    alias, so all SQL and appends (NDJSON staged on the client) run unchanged.
  - Measured: 8 writers x 50 commits, 0 failures, ~150 ms/commit, the same as 8 writers on the
    SQLite catalog directly (DuckLake commits serialize; Quack adds reach, not speed). From gold
    through `ssh -R` to this PC: attach 156 ms, 200-row append 105 ms, one-row insert 75 ms.
  - Limits found: a Quack client resolves only the server's default database (so DuckDB opens
    on the lake, not attaches it); bound parameters are dropped (inlined at `Lake.execute`);
    `query()` runs writes twice (duckdb-quack#282, never used); a 1.5.5 client cannot read a 2.0
    server, so every host needs the same DuckDB (hosts still run 1.5.5 until `mb host setup`).
  - DuckLake-with-a-Quack-catalog (ducklake#1151) exists but is experimental (rollback, commit
    atomicity FIXMEs); not used.
- Next (after hosts run this release): open the tunnel from `mb job submit`/`monitor` so a job's
  receipts and log lines land in the center's lake live instead of being collected over ssh.

## Round 3 (2026-09-29): three hosts on this release, Quack across the fleet

- Default lock re-solved on Windows (owner decision 1 settled): `libsass` from conda-forge on
  Unix (PyPI has no linux-aarch64 wheel and its sdist cannot build under a Windows solve).
- Node: pnpm is the default manager (conda-forge builds it for all four platforms; bun has no
  win-64 build there); `[nodejs] builds` writes pnpm's `allowBuilds`; `[on.<platform>.nodejs]`
  is refused, since a per-platform package.json broke `npm ci` on every Linux host. qmd dropped.
- Fixed on the way: 128-bit Windows file ids overflowed `digests.inode` (folded to 64 bits);
  `self update` recorded its baseline on first check, so an install made before a source edit
  read as current (now recorded after a successful reinstall; the dead startup refresh removed);
  hints naming `install --resolve` now name `mb lock`.
- gold, crimson, pedro-cvlab set up (DuckDB 2.0.0.dev2609250715 everywhere). Through `ssh -R`,
  each read the center's real lake (659 runs, matching) and all three wrote a served scratch lake
  at once: 20 appends each at 62-120 ms, 60/60 rows, parameterized reads included.
  `MB_REMOTE_WORKSPACE=D:/projects pytest integration/test_remote.py`: those three pass;
  macmini, pedro-home, purple still need `mb host setup`.

## Gotchas met so far, and the one standard each should follow

Status: [x] standardized, [~] worked around (one place, not yet the rule), [ ] open.

Packages and locks
- [x] conda-forge lacks some tools on win-64 (zsh, tmux, jq, btop, aria2, bun). Standard: one
      source, pixi from conda-forge everywhere; a gap is filled by a repackaged build
      (dotfiles `recipes/`), a cross-platform alternative (jaq, bottom), or dropped.
- [~] A PyPI sdist the center cannot build blocks every platform's solve (libsass). Standard:
      compiled dependencies come from conda-forge; `mb lock` could say which sdist broke.
- [x] A per-platform manifest cannot share one lock (npm). Standard: identical package.json
      everywhere, `[on.<platform>.nodejs]` refused.
- [x] pnpm stops on an undecided build script. Standard: `[nodejs] builds` decides each.
- [x] `pixi global sync` exposes nothing without `exposed`. Standard: the manifest lists them.
- [x] conda orders `3.7c` below `3.7`. Standard: no lower bound on a letter-suffixed version.

Windows
- [x] `os.exec*` drops argument quoting. Standard: `core.shell.become`.
- [~] Windows PowerShell 5.1 mangles quotes in native arguments and writes UTF-16 on `>`.
      Standard (proposed): pwsh 7 in the toolbox (conda-forge `powershell` 7.6, all platforms),
      hints printed for pwsh; every multi-command example a script, never `&&`.
- [x] CRLF checkouts broke bash, tmux and MSYS zsh. Standard: `* text=auto eol=lf` in every owned
      repository (D:\projects and dotfiles have it); `mb center git check` could enforce it.
- [~] 260-character paths: cmd.exe cannot start in a longer cwd (rattler-build in a deep temp
      dir). Standard: build and cache directories stay short (`~/.cache/...`).
- [x] 128-bit file ids overflow a 64-bit column. Standard: fold at the one place stamps are made.
- [ ] Two MSYS runtimes in one process tree (Git Bash launching pixi's MSYS tools) fail to fork.
      Standard (proposed): MSYS tools only from zsh; agents and scripts call Windows-native ones.
- [x] MSYS takes home from nsswitch, not HOME; its /etc/profile resets PATH and fails on /dev/shm
      in a pixi prefix. Standard: `db_home: windows`, zsh started non-login, ~/.zshenv sets PATH.
- [x] No ssh can multiplex on Windows (Win32-OpenSSH, Git's and MSYS2's alike, measured).
      Standard: key-less hosts through one agent (`mb host unlock`); Quack over one tunnel.
- [ ] Machine PATH wins over user PATH, and `center verify` wrote user-PATH entries for an old
      center (C:\Users\vazva\Documents\projects\.mainboard\...) that are still there.
- [ ] Localized OS error text (Portuguese here). Standard: match error codes, never messages.

DuckDB and the lake
- [x] List parameters bind at ~2 ms per element. Standard: bulk data staged as NDJSON.
- [x] Quack drops bound parameters; resolves only the server's default database; `query()` runs a
      write twice; both ends need one DuckDB release. Standard: `Lake.execute` inlines, serving
      opens DuckDB on the catalog, writes go through tables, `mb self version` names DuckDB.
- [x] `at` is reserved in DuckDB 2.0. Standard: timestamps are `ts`.

Shell and startup
- [x] zsh's `(#q)` qualifier is a plain string without extendedglob (compinit rebuilt every time).
- [x] Per-shell init commands cost a process each (900 ms on Windows). Standard: cached init.
- [ ] Two owners of the shell's startup files: `center/exposure.py` appends marked lines to
      ~/.zshenv, ~/.profile and ~/.bashrc, which chezmoi now owns and rewrites on every apply.
- [ ] gold's /etc/bash.bashrc execs zsh unconditionally, so bash cannot be kept there.

## What else to unify and clean up in mainboard (proposed, by payoff)

1. One owner for shell startup: dotfiles source `~/.config/mb/path.sh` when present; mainboard
   writes only that file (and the Windows user PATH, replacing stale entries), never rc files.
   Removes the fight above and most of `center/exposure.py`'s rc editing.
2. Jobs write the center's lake live over Quack: `mb job submit`/`monitor` open the tunnel, the
   remote agent appends receipts and log lines, and the ssh collection path (`collect`, harvest,
   `KeptDigests` mirror of the agent's JSON) shrinks to a fallback for offline hosts.
3. Finish the rename: the center still uses `mainboard.toml`, `.mainboard/` and
   `~/.mainboard-jobs`; move them to `mb.toml`, `.mb/`, `~/.mb-jobs` once every host runs this
   release (both are read anyway), so new files and docs speak one name.
4. Retire `tests/` (327 files, unmaintained, one known failure) now that `integration/` runs the
   real CLI; port the few cases worth keeping (lake check, importer, delimiter) first.
5. Split the research layer from the tool: `profile/` (3.9k lines), `trials/` (3.5k, the only
   polars user), `experiments/`, `lab/`, `manuscript/`, `plots/` are libraries experiments
   import, not CLI plumbing; one optional `mb[research]` extra (or a member package) keeps the
   core (env, hosts, jobs, lake: ~25k lines) small and fast to import.
6. Platform branches: 68 in 30 files (workstation 12, census 7, exposure 5, pixi 4, engine 4).
   Route each through `core.host` so the difference is stated once per concern.
7. Upstream what was built here: m2-zsh, m2-tmux, m2-libevent to conda-forge's MSYS2 recipes,
   and a win-64 bun; then the dotfiles' local channel disappears.
8. Owner decisions still open: delete the migrated legacy files in D:\projects\.mainboard;
   push; set up macmini, pedro-home, purple; `mb host unlock miyabi-g`.

## Round 4 (2026-09-29): one experience everywhere, and a deadline week

Worked in this order, each checked off as it lands:
- [x] 4.1 qmd gone wherever it was used (skills, vault conventions, atpx roadmap, chefe docs and
      tests, the QMD_ variables); history (changelogs, archives, aizk's prior-art page) kept
- [x] 4.2 (dotfiles `9f0e6d8`) Toolbox: herdr (Apache-2.0, conda-forge on all four platforms, native Windows,
      persistent ssh per machine) as the one multiplexer; pwsh 7 everywhere; the Windows tmux
      and libevent recipes retired; the two toolbox scripts become one Python script
- [x] 4.3 (`3399b2e`, dotfiles `15c53a3`) (1) One owner for shell startup: mb writes only its PATH file, the dotfiles source it
- [x] 4.4 (`dfcf169`) Every host gets the dotfiles (same commands, lvim) from `mb host setup`; `mb shell
      --on <host> --keep` stays tmux (it must wrap a queued allocation); herdr, saved per host
      by setup, is the everyday multi-machine window. Needs the dotfiles pushed and
      `[workspace] dotfiles = "Pedrexus/dotfiles"`
- [x] 4.5 (`624464b`) `mb host audit` (what could and should be updated, read-only) and `mb host upgrade`
      (apt update/full-upgrade/autoremove, brew, winget, pixi, uv, chezmoi, mb)
- [x] 4.6 (4) `tests/` retired: 329 files, 55,227 lines, three dev dependencies
- [x] 4.7 (`3e58a11`) (6) The 26 checks of the running OS read `core.host` WINDOWS/MACOS/LINUX;
      a host's probed system stays data; the three standalone agents keep their own
- [x] 4.8 (`21cfef6`) (5) `import mainboard.cli` loads no research module (was 13); startup
      still pays for plumbum (136 ms), structlog (111 ms), importlib.metadata (72 ms)
- [ ] 4.9 (3) DECISION, recommend defer: the center's `.mainboard/` is 20 GB (16 GB of
      environments to rebuild at a new path) and every host needs a fresh setup, while mb must
      read both names forever anyway (3,585 dataset files record old paths): no code shrinks
- [ ] 4.10 (2) DECISION: live writes need the lake reachable from nodes. The center is a
      Windows desktop behind NAT, so a tunnel only it can open gives no more reach than the ssh
      collection has today. Two ways: serve the lake from an always-on Linux host (gold), or a
      self-hosted mesh VPN (Headscale or NetBird, userspace mode on HPC nodes) so hosts attach
      `quack:center:9494` directly
- [ ] 4.11 (7) READY, needs the owner: zsh upstream is one line, `"zsh"` in `to_process` of
      conda-forge/m2-binary-packages-feedstock `recipe/msys2-pkgs.py` (its deps are already
      there); then the dotfiles' local channel goes. bun: its feedstock builds from source and
      has no win-64; npm's `bun` package ships official binaries for every platform, so
      `[nodejs]` can carry it today
- [x] 4.12 The dotfiles' two toolbox scripts are one Python script; mainboard keeps none by
      hand (what it generates is shell by necessity: activation and scheduler job scripts); the
      workspace's 1,665 lines of project glue are shorter as shell, converted only when touched
- [~] 4.13 (`09cd392`, `e0332a2`) The deadline stress case: runs record their project (the
      submit's directory under research/ or packages/, or MB_PROJECT), `mb job list --project`;
      dispatch from the Windows center to a Linux host fixed twice (snapshot named for its
      environment; the center's drive root read on Linux). See the plan below.

### A deadline week with several projects (4.13)

What a week with cutok and reproducibility both near a deadline needs, and where mb stands:
- Tell runs apart: project (new), label (`--name`), study node (`--node`), commit, dirty flag and
  source digest are on every run; `mb job list --project`, `mb query` over `lake.runs`.
- Many jobs at once: `mb job batch` (a spec file, priced and watched as one flow), per-host
  pueue queues, PBS on miyabi-g; `host hold`/`release` for rented capacity.
- Environments that adapt per OS and architecture: one `mb.lock` solved for linux-64,
  linux-aarch64, osx-arm64 and win-64; per-platform tables compile to pixi targets. The traps
  found: a PyPI sdist the center cannot build (take it from conda-forge) and per-platform Node
  manifests (now refused).
- Next, in order of payoff: a `project` column in the `runs` view (today it is inside the
  record JSON); results per project in `mb query` (a `lake.results` view keyed by run);
  `mb job list --since` for "what did I launch today"; a per-project default host and queue.

## Tools that could do part of mb's job (researched 2026-09-29)

| mb does | open-source tool | verdict |
|---|---|---|
| upgrade everything on a machine (`host upgrade`) | topgrade 17.12 (Rust; `uvx topgrade` runs on Windows from PyPI) | strong candidate: its dry run here covered winget, VS Code extensions, pixi global, npm globals, gh extensions, uv and chezmoi. Before it takes over `host upgrade` (keeping the audit), a dotfiles `topgrade.toml` must disable pixi's self-update (the fleet pins pixi) and chezmoi's pull on the center (a working checkout) |
| mirror the workspace to hosts | Mutagen (continuous sync over ssh, Windows native, ignore rules) | candidate for the mirror only; pinned snapshots and provenance stay mb's |
| ship environments to hosts | pixi-pack (pack a solved env, unpack offline, self-extracting) | candidate for HPC nodes without network or with slow installs |
| dispatch to ssh hosts, Slurm, clouds | SkyPilot (SSH node pools, Slurm, k8s, clouds), dstack (SSH fleets, Slurm) | not a replacement: neither drives PBS (miyabi-g) nor pueue hosts, and both bring a server; worth borrowing their SSH-pool model |
| Slurm from Python | submitit | not needed: no Slurm host in the fleet today |
| reach nodes behind NAT (Quack, 4.10) | Headscale (self-hosted Tailscale control), NetBird (fully open) | the missing piece for a live lake |
| multi-machine terminal | herdr | adopted (4.2) |
| dotfiles, toolbox | chezmoi, pixi global | adopted |
| experiment metrics and dashboards | trackio (HF, local-first SQLite and Parquet), Aim, MLflow | the lake already holds receipts; trackio could be a dashboard over them, not a second store |

## Capacity: what breaks first as jobs pile up (measured 2026-09-29)

Lake (scratch lake, synthetic runs and log lines; this Windows center):

| runs | append one run | live() before / after SQL filter | count via view |
|---|---|---|---|
| 1,000 | 23 ms | 0.03 s | 0.02 s |
| 10,000 | 25 ms | 0.08 s | 0.01 s |
| 100,000 | 25 ms | 0.70 s (linear) / filtered in the lake now (`38b2a26`) | 0.03 s |

- 5 M log lines: a job's tail 0.03-0.25 s, count 0.01 s. Appends stay ~25 ms at any size.
- Fixed: the catalog grew with every dispatch because source blobs inlined into SQLite (102 MB of
  104 at 660 runs); now 1.7 MB after `mb lake compact`, blobs go to Parquet (`c2c1430`).
- Commits serialize (~150 ms each under 8 writers): a 1,000-job batch spends a few minutes in
  registry commits, fine. Nothing compacts on its own: `lake check` names a catalog past 64 MB.

Hosts (the real limit):
- Every dispatch of an edited tree pins a new snapshot (~3,700 directories plus hardlinks to
  ~14,600 files, over a minute to build) and nothing prunes them: ~15 MB of directories each, so
  a week of 1,000 edit-dispatch cycles is ~15 GB per host. gold has 170 GB free (96% used).
- Each re-solved lock adds a ~20 GB environment prefix per host until `Prefixes.prune` finds it
  unreferenced, which waits for the snapshots that name it.
- PROPOSED (needs the owner): prune a host's snapshots at pin time, keeping every snapshot a run
  the center still tracks names and anything younger than 7 days (a queued PBS job may wait that
  long), so the environments behind them become prunable too.

IDs: batch ids are 32-bit (50% chance of a collision around 77k batches), held-run handles
40-bit, run labels 32-bit display only, environment digests 64-bit, sources SHA-256. Fine at this
scale; widening the batch id would rename existing batches, so it is left.

## Bug sweep (2026-09-29): what was found and fixed
- Windows center to Linux host: snapshot named for its source only (`e0332a2`); the center's
  drive root misread on Linux (`e0332a2`); the pin's silence read as an unreachable host
  (`951cf4a`); a query reading views an older release left (`951cf4a`).
- Windows reports NUL as a terminal: `job submit` with stdin redirected died on EOFError
  (`09cd392`).
- Nine text-mode subprocess calls decoded with cp1252 (`38b2a26`).
- All 71 commands and 4 group paths answer --help (integration/test_every_command.py);
  exercised for real on the fleet today: job submit/list/logs/wait/monitor, host setup/audit/
  gpus/facts, lake check/compact/serve/query, list/tree/run/doctor/check, center git status.
- Open: six runs on `blackwell` (a rental no longer declared) stay live until
  `mb job cancel` settles them; `doctor` fails only on the workspace's own `math` gate.

## Round 5 (2026-09-29): leaner surface, minimal hosts, experiments that survive failure

Owner's asks, worked in order, each checked off as it lands:
- [x] 5.1 The six stale `blackwell` runs deleted from the lake (runs, host record, 88 log lines,
      30 events); the vast batches that ran a job named `blackwell` keep their history
- [x] 5.2 The research `math` gate and `math-doctor` task removed from the workspace manifest
- [x] 5.3 mainboard, dotfiles and the workspace's manifest, lock and pointers pushed (the
      owner's unrelated work in liereadout, remaster-lab, reverse-lab, glinet left uncommitted)
- [x] 5.4 Output for agents by default: a header line then tab-separated rows, columns empty in
      every row and empty record fields left out, compact JSON, plain help (no boxes, no
      colour, no `--no-<flag>` twins), plain tracebacks, stderr logs as short lines (dispatched
      jobs keep JSON); `--human` for rich tables. Found on the way: no verb's parameter
      descriptions ever reached `--help` (the docstrings lacked `Args:`); 47 fixed
- [x] 5.5 `host list [HOSTS] [--facts] [--gpus] [--audit] [--plan]`: one verb, each flag one
      more table keyed by host; `facts`, `gpus`, `audit` and `check` gone (the far side answers
      `host list local --facts --json`, so hosts need this release)
- [x] 5.6 `job` is six verbs: `submit` (one job, `--batch spec`, a lane with `--split`/`--per-job`
      over `--on a,b`, `--estimate`, `--wait`), `list` (settles what ended, then lists;
      `--batch`, `--watch`, `--every`), `show` (the receipts' verdict, `--wait`), `logs`,
      `cancel`, `collect`. Gone: `wait`, `verdict`, `monitor` (kept hidden for installed
      timers), `batch prepare|estimate|run|watch|wait`, `lanes run`. Checked live: job 318 on
      gold, submitted with `--wait`, settled ok
- [x] 5.7 No `center` group: `center verify` is `doctor --center`, `center members` is
      `doctor --members`, `center migrate` is `host setup <dest> --center`, `center git` is
      `mb git`. `plot` and `center paper` are one `paper` group (`paper build`, `paper plot`); the
      cutok ICLR artifact vendors mainboard and runs `mainboard plot`, kept hidden until it is
      frozen. DECISION later: move `plots/` and `manuscript/` (1.4k lines, 2 external imports)
      into their own package once the artifact freezes
- [x] 5.8 `mb shell` starts your own shell (zsh, bash, fish, pwsh, cmd) with the environment
      entered: pixi's own `shell` refused zsh on Windows. `shell-hook` gone (nothing used it).
      `mb completion powershell` added (a static table, no Python per keystroke), so bash, zsh,
      fish and PowerShell complete on every system
- [x] 5.9 `proc` works on any process by pid; added `proc list [pattern]` (pid, parent, user,
      cpu, memory, age, command; everyone's when a pattern is given), `proc kill --match`, and
      `--on HOST` for both
- [x] 5.10 `host setup --minimal` (tool, pixi, environment; no dotfiles); `dotfiles = false` per
      host (purple, a shared server); `host hold` rentals always minimal; `[workspace] dotfiles =
      "Pedrexus/dotfiles"` declared
- [x] 5.11 (dotfiles `44a85b8`) herdr is the one multiplexer: tmux, tpm and `.tmux.conf` left the
      dotfiles. herdr keeps a server per machine, but none of the fleet has it and a `--minimal`
      host never will, while `/usr/bin/tmux` is on every Linux host; so `mb shell --on <host>
      --keep` still wraps a queued allocation in the host's own tmux, installed and configured
      by nobody
- [~] 5.12 Experiment environments: `[envs.gpu]` (`no-default`, Linux only: Python, CUDA torch,
      cutoken) is 5.7 GB on crimson against `default`'s 18 GB; `host setup crimson --minimal
      --env gpu` took 3m05s (warm caches). Fixed on the way: a `no-default` environment lost the
      member pins, so cutoken's `mainboard` came from PyPI (0.4.8, months old) and shadowed the
      tool on PATH; members are now pinned (without `default`'s extras) in every such env
- [ ] 5.13 Resume a failed job from its last checkpoint, recorded in the lake
- [ ] 5.14 Wipe mainboard from every host and set it up again, measuring size and time
- [ ] 5.15 Rented GPUs that stay set up (Vast, HPC-AI, AWS Blackwell): images and volumes
- [ ] 5.16 Monorepo layout proposal
- [ ] 5.17 Try the researched tools (topgrade, Mutagen, pixi-pack) on the real fleet

### Ship a built environment instead of installing one (asked 2026-09-29)

Hosts never solve today: they install the lock the center solved. What costs time on a fresh
node is downloading and linking a few GB of packages, and what costs space is everything the one
`default` environment carries. Options, measured where possible:

| way | what moves | fresh node needs | trade-offs |
|---|---|---|---|
| install the lock (today) | nothing from the center; the node pulls from conda-forge/PyPI CDNs | pixi, network | datacenter CDN bandwidth is usually the fastest source; every node repeats the download |
| pixi-pack `--create-executable` | one self-extracting file per platform and lock (`gpu`: 3.96 GB, packed here in 1m45s) | nothing (no pixi, no network) | editable workspace code is not packed (mb ships it in the pinned tree anyway); from this PC the upload is the bottleneck, so the file belongs in object storage near the providers, keyed by the environment digest mb already computes |
| OCI image (Docker) from the lock | an image in a registry | docker; Vast/RunPod/AWS start from an image directly | the only form a rented GPU can boot already set up; layers cache on providers; build needs Linux (a host or CI) |
| Apptainer/Singularity SIF | one file | apptainer (miyabi has singularity) | the HPC form of the image; one file, runs without root |
| single-binary Python (PyInstaller, Nuitka, PyApp) | one executable | nothing | fine for the mb tool itself; not for CUDA stacks (torch + NVIDIA wheels are 3+ GB of shared libraries a binary cannot shrink, and freezing them is fragile) |

Recommendation: keep the lock as the one truth, and derive two artifacts from it, content
addressed by the environment digest: a pixi-pack executable (HPC and ssh hosts, offline nodes)
and an OCI image (rented GPUs, 5.15). The lean per-experiment environment is what makes either
small; packing `default` would still move 18 GB.

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
- [x] 5.13 (`ed8c6b6`) Every job exports `MB_CHECKPOINT` (one directory per run name under the
      mirror, outliving pinned trees) and `MB_ATTEMPT`; `job submit --resume <handle|name>` runs
      it again as the next attempt with today's code; a name resolves to its newest attempt.
      Checked on gold: 319 failed at step 1, 320 resumed at step 2, settled ok. Open: a
      checkpoint on a rented disk dies with the rental (5.15)
- [x] 5.14 Every mainboard directory on gold, pedro-cvlab and crimson moved aside (`*.aside-r5`,
      nothing deleted: they hold pre-lake registries and batch evidence from when those machines
      were centers) and each set up again `--minimal --env gpu`. Before: 73-103 GB per host
      (`~/.mainboard-jobs` 25-41 GB plus an old checkout's `~/projects/.mainboard` 48-62 GB,
      whose pueue daemons the fleet was still using). After: 6.6 GB (environment 5.6, source
      mirror 1.0, tool 0.2), in 1m40s-2m01s from nothing (84 s of it the mirror uploaded from
      this PC), 46 s again. A fresh minimal host runs CUDA jobs (crimson, RTX 3090).
      Fixed on the way: a lean environment has no pueue, so setup now installs it with pixi
      global where the host has none; mb's PATH now includes the dotfiles' per-architecture pixi
      home (`~/.pixi/<arch>/bin`); dangling `~/.local/bin/pueue` links an old release left;
      OpenSSH's post-quantum warning was reported as the reason a remote command failed (and
      `LogLevel=ERROR` now rides every connection); exit 127 reads as "command not found".
      Dispatch latency, profiled: 96 s before a job started, 25 s now. The center asked git
      about ~70 repositories sixteen times per submit (listings cached per command, the
      fingerprint's roots listed in one pass, repositories listed in parallel), and one
      repository's `info/exclude` held 7,558 single paths from the LFS removal, which cost git
      itself 20 s; rewritten as 255 directories and 515 paths with byte-identical `git
      ls-files` results (the original kept beside it). The owner deletes the `*.aside-r5`
      directories once their old registries are imported or declared unneeded
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

Measured on crimson (RTX 3090, a university network), the lean `gpu` environment (5.7 GB):

| route | time to a working `import torch` on CUDA | size moved |
|---|---|---|
| install the lock, cold (bare Debian container, no caches, pixi fetched first) | 5 s pixi + 43 s install | from the CDNs |
| pixi-pack executable pushed from this PC, then unpacked | 4m55s copy (about 14 MB/s upload) + 37 s unpack | 3.96 GB |
| minimal OCI image (slim Debian + the unpacked pack) | built in 5m46s on the host; starts in 4 s | 3.0 GB zstd, 6.07 GB unpacked |

So on a well-connected machine the lock itself is as fast as any artifact; what made it fast is
the lean environment (18 GB to 5.7 GB). An artifact pays off where the node is slow, offline
or billed while it installs, and on providers that boot from an image and cache its layers.

Recommendation: keep the lock as the one truth, and derive two artifacts from it, content
addressed by the environment digest: a pixi-pack executable (HPC and ssh hosts, offline nodes)
and an OCI image (rented GPUs, 5.15). The lean per-experiment environment is what makes either
small; packing `default` would still move 18 GB.

### Guards instead of fixes (asked 2026-09-29: "look for general ways to fix them")

The gotchas keep coming in five families. Each now has one guard that fails early, so the next
member of the family is caught by a test or a stage line rather than found on a host:

| family | met as | guard |
|---|---|---|
| docs drift from the CLI | no parameter description ever reached `--help`; skills naming verbs removed rounds ago | `test_every_command.py`: every parameter of every command is described, and every `mb ...` a README or skill spells in code resolves to a real command |
| one package, two sources | a lean env took `mainboard` from PyPI and shadowed the tool | `mb lock` refuses a lock where a package the workspace builds from source comes from an index in any environment (`Lockfile.mixed_sources`) |
| cost that only shows at scale | a 7,662-way regex per path per depth; linear scans of every run; catalog inlining | every multi-stage verb prints each stage with its elapsed seconds and a total, so the slow stage is named in every log |
| text that differs by platform | CRLF from Windows writes, cp1252 decoding, heredoc backslashes | `test_source_rules.py`: every text write names its newline (26 fixed); `PLW1514` already requires an encoding |
| an action that "succeeded" | a pipe hid pytest's exit; setup kept an old tool | verify the outcome, not the exit: setup reads the host back through its new activation (it caught the old tool today) |

### Rented GPUs that stay set up (5.15, proposal)

Today a rental is set up from nothing each time (pixi, the tool, the environment) and its disk,
checkpoints included, dies with it. In order of payoff:
1. Boot from the environment's image: `mb pack <env> --on <linux host> --image --push
   ghcr.io/<owner>/mb-<env>` once per lock; `host hold` passes that image to Vast (and any
   provider that boots from an image) and skips provisioning, so a rental ships only code. The
   provider caches the layers, so a second rental of the same lock starts in seconds. Needs the
   owner's registry login on the building host (`gh auth token | docker login ghcr.io -u
   <user> --password-stdin`); a private package needs the provider's registry credentials too.
2. Checkpoints off the node: a rental's `MB_CHECKPOINT` synced to object storage (R2 or S3,
   rclone) every few minutes and restored by `--resume` on whatever machine takes the next
   attempt, so a preempted or released rental loses minutes, not the run.
3. One provisioning layer for clouds: SkyPilot (Apache-2.0) launches the same image on AWS,
   GCP, Lambda, RunPod, Vast and Kubernetes with autostop and spot recovery, which is exactly
   the part of mb that is provider-specific (`backends/vast.py` and `hpcai.py`, 1,076 lines).
   mb keeps pueue, PBS and ssh hosts, the lake, the receipts and `--resume`. Blackwell to try:
   RTX 5090 and RTX PRO 6000 (sm_120) on Vast and RunPod by the hour; B200 on Lambda, RunPod and
   AWS (`p6-b200`, 8 GPUs, capacity blocks).
Decision needed: adopt (3), or keep mb's own provider backends and do (1) and (2) inside them.

### Monorepo layout (5.16, proposal)

Today's root mixes projects, tool configs and personal folders: `apps/`, `career/`,
`finances/`, `health/`, `packages/`, `personal/`, `remaster-lab/` (a project at the root),
`research/` (24 entries: projects, a Lean toolchain's `lakefile.toml`, `common/`, `stubs/`,
`scripts/`), `scripts/`, `templates/`, `tmp/`, `vault/`, `writing/`, plus `atpx.toml`,
`pgrls.toml` and `maskfile.md` beside the manifest. Proposed, one kind of thing per top folder:

| folder | holds | from |
|---|---|---|
| `packages/` | libraries and tools anyone can install (mainboard, patos, rls, aizk, atpx, mcmr, the sqlalchemy fork, a new `research-common` from `research/common` and `research/stubs`) | as today |
| `research/<project>/` | one member per project, each with its own `pyproject.toml`, `mb.toml` (its tasks and its lean experiment environment, `[envs.<project>]`), papers and experiments | the projects under `research/`, `remaster-lab/`, the root's per-project tasks |
| `research/math/` | the Lean project with its `lakefile.toml`, `lean-toolchain`, `lake-manifest.json` | `research/`'s root |
| `apps/` | end-user apps | as today |
| `life/` | career, finances, health, writing, vault | five root folders |
| `personal/` | dotfiles and profile repos | as today |

The root keeps `mb.toml`, `mb.lock`, `AGENTS.md`, `README.md`, `.gitignore`, `.gitattributes`;
tool configs move beside what they configure (`atpx.toml` to `research/math`, `pgrls.toml` to
`packages/aizk`, `maskfile.md` to the project it serves). The root manifest then only composes
members, and each project's lean environment lives with the project. Cost: paths in datasets,
receipts and host mirrors (3,585 dataset files record old paths), every host re-mirrored.
Recommendation: move projects one at a time, each with its own environment, after the cutok
deadline; `life/` and the root configs any time.

## Round 6 (2026-09-29): minimal by default, a cleaner manifest and layout

- [x] 6.1 The stale `mainboard-monitor` timer on pedro-cvlab disabled and its units removed
- [x] 6.2 Source snapshots pruned on the host whenever a job builds its environment: kept are
      the newest three, any used in the last seven days (re-pinning counts as use) and any a
      queued or running pueue task names; the environment prefixes only those name go next
- [x] 6.3 `host setup` is minimal by default (tool, pixi, pueue, the environment); `--dotfiles`
      adds the shell, editor and toolbox, and every later sync keeps what setup chose. The
      per-host `dotfiles` switch is gone
- [x] 6.4 The `*.aside-r5` directories deleted on gold, pedro-cvlab and crimson (owner's go). Freed
      19, 18 and 52 GB: the rest of their bytes were hardlinks into pixi's package cache, which
      still holds 52-56 GB per host, mostly packages of old `default` environments
- [x] 6.5 `MB_PROFILE=<file> mb <verb>` runs any verb under mb's own profiler with every module
      instrumented (`job list`: 0.7 s, 0.39 s of it reading three failed runs' transcripts for a
      cause). The report names regions by their whole nesting path, which reads poorly for a
      deep call tree; a flat per-function table would serve this use better
- [x] 6.6 Records read back from the lake ignore fields a release added or removed (renaming one
      field made crimson read as never set up); pydantic 2.13 applies it to nested models too

### Where a minimal host's bytes are (6.7)

| what | size | verdict |
|---|---|---|
| the `gpu` environment | 5.6 GB | the floor for CUDA torch: `libtorch_cuda` links cuBLAS, cuDNN, cuFFT, cuRAND, cuSPARSE, NCCL, NVRTC and cuFile; only cusparselt and nvshmem (0.3 GB) are optional. A CPU-only experiment environment would be ~1 GB |
| the source mirror | 1.0 GB | shrinkable: a minimal host needs only its environment's members and the job's code |
| the tool | 0.2 GB | fine |
| pixi's package cache | 52-56 GB | every package ever installed; the environment is hardlinked from it, so only stale packages are waste. `pixi clean cache` on a host frees them |
| old checkouts `~/projects` | 123 GB gold, 52 GB crimson | from when those machines were centers; candidates for deletion (owner) |

### SkyPilot, compared (6.8)

| | mb | SkyPilot |
|---|---|---|
| machines | ssh hosts (pueue), PBS, Slurm, Vast, HPC-AI, Modal | 20+ clouds and marketplaces (AWS, GCP, Azure, Lambda, RunPod, Vast, Nebius...), Kubernetes, Slurm, ssh node pools |
| picks the machine | the host you name; `host hold` rents on Vast | an optimizer over price and availability, failover across regions and clouds |
| survives preemption | `--resume` from `MB_CHECKPOINT` (a rental's disk dies with it) | managed jobs relaunch preempted spot work; checkpoints to a bucket mount |
| environment | lock solved once for four platforms, installed or packed per digest | a `setup:` script or an image per task |
| record | the lake: runs, receipts, verdicts, logs, costs, queryable | its own state database; logs per cluster |
| runs on the center | Windows, macOS, Linux | Linux and macOS only: the client imports `resource` and fails on Windows (tried); Python <= 3.13 while mb is on 3.14 |

Fit: SkyPilot is best at the part mb does worst (getting a machine anywhere, cheaply, and
getting it back after preemption) and does none of what mb is for (the lock, the lake, the
receipts, PBS and pueue hosts). The clean seam is acquisition: SkyPilot launches the machine from
the environment's image and writes an ssh alias for it; from there it is an ssh host to mb. Cost
of that seam: a Linux launcher (gold, or WSL here), SkyPilot's own state beside the lake, a
second Python. Recommendation: keep mb's Vast backend for now (it works and is measured), move it
to boot from the environment's image (6.9), and try SkyPilot from gold for AWS or Lambda
Blackwell when that is wanted; if it earns its place, retire `backends/vast.py` and `hpcai.py`.

### Cleanup and deletion candidates (6.10, marked for the owner; nothing here is deleted yet)

| candidate | size | why it can go | how |
|---|---|---|---|
| center `.mainboard/batches`, `source-archives`, `recovery`, `audits`, `collection.digests.json`, `catalog.ndjson`, `pulse.json`, `costs/`, `activate*.sh` | ~4.1 GB | imported into the lake and verified byte for byte (4.16); the generated scripts were retired in Phase 5 | after `mb lake check` passes, delete them |
| ~~hosts' `~/projects` (old center checkouts)~~ | done 6.14 | | |
| hosts' pixi package caches | 52-56 GB each | mostly packages of retired environments | `pixi clean cache --yes` per host (live environments are hardlinks and survive) |
| `packages/chefe`, `packages/lote` | 6 MB | retired, absorbed by mainboard; already excluded from members | remove the submodules |
| `packages/cuda-python-meta`, `packages/sqlalchemy-cockroachdb` | <1 MB | excluded from members: nothing requires them | remove, or keep as archives |
| `research/llm` | 284 MB | its nested `transformer_engine/3rdparty/googletest` submodule is broken, and every `git status` in the workspace prints two errors for it | fix or remove the submodule |
| `research/compression/references` (24 submodules) | 1.3 GB | vendored reference repos read once; every git listing walks them | keep the few still cited, drop the rest |
| `maskfile.md` | 16 KB | old HPC singularity recipes; mb runs jobs now | delete |
| `tmp/` | 4 MB | untracked scratch | delete |
| manifest deps no code, task or doc mentions: gallery-dl, instaloader, yt-dlp, youtube-transcript-api, imageio-ffmpeg, libcst, linkify-it-py, pytorch-ignite | - | maybe tools run by hand; the owner's call | remove from `[python.deps]` |
| `remaster-lab/` (untracked, 136 GB of data) | - | a project at the root; belongs under `research/` once committed | move when it is committed |

### Layout and manifest, done (6.11)

- `life/` holds career, finances, health, writing and the vault (the vault and career-ops
  submodules re-registered; Obsidian opens the vault at `life/vault` now); tasks, lint and git
  rules, nine skills and the finances dashboard follow; `life` joins `research` on PYTHONPATH.
- `mainboard.toml` is one manifest in seven sections (workspace, dependencies, environments,
  hosts, tasks and gates, quality, research output), every table moved whole with its comments
  and the parsed document proven equal; the chefe-era header rewritten; 8 dead tasks removed.
- One lean environment per project: `cutok` (torch, cutoken, cupy, polars, gigatoken, pytest)
  and `repro` (torch, transformers, the numerics stack at reproducibility's pins), both Linux.
  `doctor` no longer reports an environment declared for other platforms as never installed.
- Kept apart on purpose: `research/reproducibility`, `research/llm-head`, `research/bale` and
  `packages/mcmr` keep their own manifests for people who clone them alone. Found: making
  reproducibility a member folded its conda `optuna` beside the root's PyPI one, which dragged
  conda's sqlalchemy against the workspace fork. Composition should let the root win across
  ecosystems, not only per table; until then reproducibility stays a path dependency.
- mainboard asks `cyclopts>=4.23` again: aizk caps it below 5, and mb passes on both.

### Experiments on several machines, in the lean environments (6.12)

On crimson (x86-64, RTX 3090) and gold (aarch64, GB10), each set up minimal:

| | cutok (`[envs.cutok]`) | reproducibility (`[envs.repro]`) |
|---|---|---|
| collection, first try | 9 tests, 36 errors (no numba, cuda.core) | refused: nothing provisioned; then no mainboard |
| collection now | 1,036 tests, 0 errors, both hosts | 2,942 tests, 28 errors, both hosts: 24 experiments refuse collection without their registered policy plugin (their tasks pass it), 1 imports compression's package, 3 inside test files |
| a real run | `pbt_parity`: 89 tests start, each refuses the card ("registered for GH200 or RTX 4090"), the protocol working | `registration_protocol`: 5 pass, 3 need a git checkout, which a mirror host has none of |

What running them found and fixed in mb, each a class of bug:
- `mainboard.dispatch.shared.git` (removed 2026-09-25) and `schedulers.JobState` were still imported
  by ten experiments: restored, and `test_consumers.py` now imports every name the workspace takes
  from mainboard (96).
- `mb.lock` reached a pinned tree only as a link back to the mirror, which an experiment's source
  seal refuses: the lock now ships wherever the manifest does (`Sync.shipped`).
- A trial opening a lake in its pinned tree was refused for `dispatch/digests.json`, the host
  agent's live memory, mistaken for pre-lake state.
- A 25 MB source file broke the seal's append (DuckDB's 16 MB JSON object default).
- A lean environment installed workspace packages it reached transitively as frozen copies, so
  mb's fixes never reached the job: an environment names the workspace packages it imports.
- z3-solver has no aarch64 wheel: taken from conda-forge (the standing rule for compiled deps).
- A `| tail` in a job command hides the exit; `bash -o pipefail` in every such check.

### Vast.ai with the new systems (6.13)

`mb host hold vast --gpu-name "RTX 4090" --for 1h --max-usd 2 --env cutok`, from nothing to a
parked, set-up machine in 3m42s ($0.56/h, NL):

| stage | s |
|---|---|
| rent until ssh answers | 49 |
| probe | 18 |
| mirror (8,007 files; `[envs.cutok] sources` now cuts the roots from 16 to 8) | 34 |
| install mainboard (uv) | 22 |
| pixi | 11 |
| the `cutok` environment, cold from the CDNs | 42 |
| pueue, read back, park | 46 |

cutok's `pbt_parity`, registered for the RTX 4090, ran there end to end (22.5 min, refused on
the 3090 and GB10): 88 of 89 failed because I ran it as a plain `pytest` command, which ships no
pins, and each trial loads its tokenizer offline (`local_files_only`) from a pinned revision. The
job spelling (`mb job submit --on <rental> --env cutok .../test_pbt.py::test_pbt_parity`) stages
those pins; that run is the next one. The whole trial cost $0.60 (credit 84.83 to 84.24), and the
rental was released and its alias removed. `job submit --wait` called the run stalled after 20
minutes of a CPU-only phase with its output held by `| tail`, a false positive of the stall rule
(no output and an idle card) that a CPU-bound job will always meet.
Found on the way: `UserKnownHostsFile=nul` wrote every rental's host key into a file called
`nul` wherever mb ran (MSYS2's ssh reads `nul` as a name); fixed per ssh flavor.

Image hub (asked 2026-09-29): the environment's image lives in a registry near the providers
(GHCR or Docker Hub; both CDN-backed), built by `mb pack <env> --on <linux host> --image --push
<repo>` and named by the environment's digest. From the stages above an image saves the tool,
pixi, environment and pueue steps (~2 min) but adds a pull on a host that has not cached it
(3.0 GB at ~100 MB/s, under a minute); Vast caches the layers per host, so repeated rentals and
batches gain most. Needs the owner's registry login once on the building host; wiring `host hold`
to boot that image (the Vast backend already boots a plan's own image) is the next step.

### SkyPilot on Windows, and its alternatives (6.14, measured)

How SkyPilot is built (0.13.0, 577 files, 228k lines): a Python package holding both a client
and an API server. The server (FastAPI on uvloop, SQLite or Postgres state, alembic migrations)
does everything: provisioning through each cloud's SDK, ssh and rsync to clusters, the managed
jobs controller, a dashboard. The client sends requests over HTTP and streams logs; with no server
configured it starts one locally in the background. The split exists for teams (one endpoint,
shared clusters, requests that outlive a laptop), and it is also what makes Windows reachable.

On Windows, tried: the client fails on `import resource`; with three stub modules (`resource`,
`fcntl`, `termios`) and UTF-8 output, `sky --help`, `sky api info` and `sky gpus list --infra
vast` all worked against an API server started on gold and reached through `ssh -L`. The server
itself does not start on Windows (uvloop has no Windows build; it forks, uses ssh
ControlMaster, rsync). So a Windows fork is small for the client (guard the imports in about
seven files, force UTF-8) and large for the server; the working shape is a Windows client with
the server on gold, or in WSL. An upstream PR for the client guards is the cheaper route than a
fork. Requirements: Python 3.9 to 3.13, a Linux or macOS machine for the server, each cloud's
credentials on the server.

Alternatives: dstack (MPL-2.0, 0.22.1) is the closest: server and CLI, backends for AWS, GCP,
Azure, Lambda, RunPod, Vast, Nebius, Kubernetes and ssh fleets. Its CLI and its server both run
natively on Windows (tried; the server needs `sqlalchemy<2.1` beside its sqlalchemy-utils). Its
unit is a container: every run is an image, so it assumes the image route mb measured as
unnecessary for its own use. Others: Ray's cluster launcher (AWS, GCP, Azure, Kubernetes; Linux
client), Modal (hosted, already an mb backend), RunPod and Vast's own CLIs (single provider).

Is the image route needed now: no. The lean lock installs cold in under a minute on a good
network and a Vast rental is ready in 3m42s without an image; an image saves ~2 min per rental
and costs a registry, a build per lock and a pull on every uncached host. It pays when renting
many machines at once, for offline nodes or HPC (SIF). `mb pack` keeps the capability; no hub
is set up until a batch needs it.

### Old checkouts deleted (6.15)

`~/projects` removed on gold, crimson and pedro-cvlab (old center checkouts, chefe-era state):
freed 37, 35 and 88 GB. Kept, since nothing else holds them:
- pedro-cvlab's 28 unpushed commits (branches `wt/fix-segmentation`, `wt/fix-tokenization`,
  `wt/fix-transforms`, `wt/mcmr-a`, `wt/mcmr-merged`, 2026-09-04) as a git bundle at
  `.mainboard/recovery/pedro-cvlab-unpushed.bundle` on the center (`git fetch <bundle>`);
- in `~/mb-salvage/` per host: gold's `results/dsv4_flash_e8p_full` (83 GB, a quantized
  DeepSeek V4 Flash checkpoint nothing else holds), `data/` and `outputs/`; pedro-cvlab's
  `data/cutoken` (70 GB of corpora), `japanese/` (3.5 GB), 13 ignored result dirs (11 MB tar);
  crimson's `data/` (2.3 GB).
gold's pueue daemon ran from that old checkout's environment; `host sync` now checks the queue
and starts it, as setup did.

### Clouds inside mb, no server (6.16)

What SkyPilot and dstack keep, and what mb takes from each:
- SkyPilot: a provisioner per cloud of seven functions (run, wait, query, stop, terminate
  instances; cluster info; ports), 600-800 lines each, called from its API server. Worth taking:
  the thin lifecycle and how each cloud is started (RunPod's start command installing sshd, Lambda
  keys registered by name). Not worth taking: the server, its database, the controller.
- dstack: a `Compute` base (get offers, create instance, terminate, update provisioning data) plus
  capability mixins (`ComputeWithCreateInstanceSupport`, volumes, multinode, gateways), also
  behind a server; its offers come from gpuhunt, a standalone library. mb already had the same
  shape (`ProviderBackend` plus `Account`, `Inventory`, `Market`, `Rentable`, `LogSource`,
  `Delivery`), so the missing pieces were a catalog across clouds and a template for new ones.

Built:
- `dispatch/backends/cloud.py`: `CloudBackend`, the template. A cloud names its gpuhunt catalog
  and key variables and implements `create`, `machine`, `machines`, `terminate`; renting (cheapest
  on-demand offer under the budget, a priced lease before each create, the next offer on a
  capacity refusal, waiting for an address and for ssh, ending a machine that never answers),
  reaching, cancelling and listing are shared. One-shot `job submit` to a cloud is refused in
  favour of `host hold`, which sets a machine up once.
- RunPod (`runpod.py`, REST v1: a plain Ubuntu pod whose start command installs sshd and
  authorizes the key) and Lambda (`lambdacloud.py`: the key registered under a name derived from
  it, a VM reached as `ubuntu`), about 100 lines each, tested against recorded API answers
  (`test_clouds.py`); a live rent waits for the owner's key.
- `mb host offers <card>`: every cloud's offers through gpuhunt, cheapest first, marked with the
  provider `host hold` rents them through and whether its key is here. gpuhunt joins mb's
  dependencies (it brings only `requests`).

Which clouds, measured 2026-09-29 (single card, $/h, cheapest):

| cloud | 4090 | 5090 | H100 | B200 | API | in mb |
|---|---|---|---|---|---|---|
| Vast | 0.31 | 0.41 | 0.99 | 7.75 | REST, containers | yes (per job and held) |
| RunPod | 0.34 | 0.99 | - | 6.79 | REST, containers | built, needs a key |
| Lambda | - | - | 2.49-3.29 | 6.99 | REST, VMs | built, needs a key |
| Verda (DataCrunch) | - | - | 1.78 | 3.43 spot / 6.85 | REST, VMs | next: cheapest B200 |
| Nebius | - | - | 2.15 | 3.95 spot | gRPC SDK | later |
| AWS, GCP | - | - | 1.09-1.26 spot | - | SDKs, quotas, IAM | later, for Blackwell capacity blocks |

Next on this base, in order: Verda (the cheapest B200); stopping instead of ending where a cloud
keeps the disk (RunPod pods), and persistent volumes (RunPod network volumes, Lambda filesystems)
so a machine stays set up across rentals; spot machines resumed by `--resume` on preemption.

### Spot, per-job clouds, HPC-AI, and renting by architecture (6.17)

Asked: spot and batch jobs on clouds, HPC-AI working, and "look for arch and then get the
cheapest with that arch" (an RTX PRO 4500 shares the RTX 5090's instruction set, not the B200's).

- Architecture is the compute capability, not the generation (`dispatch/arch.py`). Blackwell is
  two instruction sets: datacenter B200/GB200 are sm_100 and B300/GB300 sm_103 (tcgen05, tensor
  memory); RTX 50xx and RTX PRO 4000/4500/6000 are sm_120, DGX Spark's GB10 sm_121. An `sm_100a`
  cubin loads on sm_100 alone, so `--arch sm_120` matches exactly, `sm_90+` asks for at least,
  and families (`ampere`, `ada`, `hopper`, `blackwell`, `blackwell-dc`, `blackwell-rtx`) span.
  Cards come from gpuhunt's table with its gaps corrected (it lists B300 as sm_100).
- `--arch` on `host offers`, `host hold` and `job submit` (and `arch` in a host's defaults or a
  batch job): the cheapest card of that capability when no card is named. Offers print an `sm`
  column. Vast is asked through its own `compute_cap` field; the other clouds filter gpuhunt's
  rows. Measured: the cheapest sm_120 is an RTX 5060 Ti at $0.10/h spot on Vast, an RTX PRO 4500
  $0.30/h; the cheapest sm_100 a B200 at $3.43/h spot on Verda.
- Spot is part of the request (`Resources.spot`, `--spot`, `spot = true` in defaults), which
  every backend now reads: backends were always built with no arguments, so Vast's and HPC-AI's
  existing spot switches could never turn on. A machine taken back reads `vanished`, and
  `mb job submit --resume <name> --spot` continues from `MB_CHECKPOINT`. Held requests and
  creation intents record both fields; batch ids digest them only when set, so no id moved.
- Per-job cloud machines: RunPod and Lambda take `job submit` and batch jobs like Vast does. A
  cloud VM has no entrypoint mb can rely on, so once ssh answers the rent starts the landing's
  waiter detached over ssh, writing `/tmp/mainboard.log` and `/tmp/mainboard.exit`; `state` and
  `logs` read them, and the settle path ends the machine. Holds park the same waiter.
- HPC-AI picks its instance type from its own catalog by the request (card, architecture,
  spot, count, the spend cap over walltime plus landing) unless the host pins one; its offers
  join `host offers` with out-of-stock types marked, and it is a `Market`, so estimates price it.
  Its GPU types are whole 8-card nodes (8x 4090 $4, 8x 5090 $5.20, 8x B200 spot $11.92,
  8x H200 $15.92/h) and every one was out of stock on 2026-09-29; only CPU types were in stock.
  Live, 2026-09-29: a landing on the CPU type ($0.24, the one-hour floor) sat in
  `DownloadImage` / `BackOff`: the pinned `image-id` no longer pulls, and the API lists no
  public images (`/image/list` answers the account's own, none). mb ended it at the address
  deadline and nothing was left billing. Now a start stuck in back-off for three minutes is
  refused naming the image, and the ssh-key hint is kept for ssh failures only. HPC-AI needs a
  current image id from the console (owner) before it can run anything.
- Live on Vast, 2026-09-29: `mb job submit --on vast --env cutok --spot --arch sm_120` rented
  an RTX 5060 Ti (sm_120) as a spot bid, landed cutok and ran torch on it, ok, in about four
  minutes for cents. Getting there found four faults, each now fixed and tested:
  - a bid of exactly `min_bid` is refused as `no_such_ask` (six offers in a row read as
    "taken"); a bid 5% above the floor rents the same machine, and the lease is priced at it;
  - Vast rate-limits creates after a few quick refusals (HTTP 429); the create now waits and
    asks again;
  - `[envs] sources` narrowed the mirror below the lock's inputs: the digest reads every local
    project's `pyproject.toml`, the host lacked six of them and refused the lock. A narrowed
    plan now ships each one;
  - that made one file listed twice (under a source directory and by name), and the snapshot
    refused to link it twice; `Sync.scope` drops a root lying under another.
  Before the fix the arch search also lost to the host's default card (Vast's RTX 4090): an
  asked arch now replaces a default card and an asked card a default arch.
- Reliability per provider (`mb host list --reliability [--days N]`, and a `delivered` column in
  `host offers`), read from the run registry, which already records every dispatch however it
  ended. A run counts as `ran` when its command reached an exit (its own failure included),
  `unstarted` when the landing never started it, `lost` when the machine vanished or it ended
  with no exit code; `delivered` = ran / judged. Over 90 days: owned hosts and Miyabi 100%, a
  held Vast machine 2/2, per-job Vast 10% (2 of 20: 15 unstarted, 3 lost), hpc-ai 0/1. Most of
  the Vast losses were mb's own landing faults of that period, which the number is meant to
  expose: it measures what a dispatch there costs, whoever is at fault.
- Smaller: `host list hpcai` now lists the `hpc-ai` row (rows are named by kind); Modal's
  Python 3.14 warning is filtered process-wide, since `catch_warnings` is not thread-safe and
  leaked during parallel probes; gpuhunt's "offline provider" notice is silenced; the spend cap
  a cloud rent searches under is the hourly one (`hourly_cap`, shared with Vast), not the total.

## Round 7 (2026-09-30): what a deadline day on the Windows center broke

The full account, with the designs left open, is `FIXES-2026-09-30.md`. Uncommitted.

- [x] A wrapped command keeps its own flags and `--` (`proc timeout 900 mb job submit --on
      gold -- python x.py` died on `--on`); `proc timeout` knows its seconds come first
- [x] A command bound for a host refuses an argument Git Bash rewrote under its own folder
      (`/home/crimson/y` arrived as `C:/Program Files/Git/home/crimson/y`), naming
      `MSYS_NO_PATHCONV=1`
- [x] A dead host costs one knock per command and per half minute of a wait, never a
      traceback: `job list` 34 s to 6 s with gold down; its runs say so, a dropped host names
      the commands that settle its runs, and `job cancel` settles a run the host never heard
- [x] `git status` (a `broken` column) and `git check` (a `submodule` warning) name the
      checkout a bare `git status` aborts on
- [x] `host offers` survives one catalog failing; `doctor` names no fix for a dropped alias;
      setup says it installs the mirror's environment, not the job's pinned copy
- [ ] Setup builds the pinned environment; the landing ships the closure plus what installs;
      a Task Scheduler settler for the Windows center (each designed in the fixes file)

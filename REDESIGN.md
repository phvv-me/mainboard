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

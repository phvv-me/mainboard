# mainboard state redesign: objectives, phases, status

Living tracker, updated as each step lands. Last update: 2026-09-28.

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
| 4 | Move every writer into the lake; drop wandb; structlog | in progress (branch `mb-writers`) |
| 5 | Activation from `mb.toml`; drop generated scripts and jinja2 | pending |
| 6 | Git provenance instead of source zips; HF pins into the HF cache; native report | pending |
| 7 | Remove old paths, polars -> DuckDB, seaborn/pandas out, regression tests, center setup | pending |

## Step checklist (worked strictly in order, one at a time)

Legend: [x] done and committed, [~] in progress, [ ] pending.

Phase 4 (branch `mb-writers`)
- [x] 4.1 Run/host registry (`dispatch/db.sqlite`) -> `runs_log`/`host_facts` + views (`7de5835`)
- [x] 4.2 Shared lake `Session` (lazy attach, reattach on locked/stale) (`7de5835`)
- [x] 4.3 Batch event journal (`events.ndjson`) -> `events` via `Journal` (`f762aae`)
- [x] 4.4 Drop the wandb mirror: receipts are the record, `[tracking]` is just `interval`; one
      session serves threads by taking turns (a shared DuckDB connection deadlocks otherwise)
- [~] 4.5 structlog replaces loguru and stdlib loggers; `from mb import logger`; `mb` import alias;
      trial-bound events routed to the trial spool; remove loguru
- [ ] 4.6 Harvested receipts (`receipts.ndjson`) -> `receipts`
- [ ] 4.7 Captured job logs (`<handle>.log`, `monitor.log`) -> `log_lines`
- [ ] 4.8 Costs ledger -> lake table
- [ ] 4.9 GPU quotes catalog -> lake table
- [ ] 4.10 Pulse -> lake table
- [ ] 4.11 Holds -> lake table
- [ ] 4.12 Studies -> lake table
- [ ] 4.13 Digests (mirror, collection) -> `digests`
- [ ] 4.14 Job specs / closures -> `job_specs` / `closures` (transient files only for shipping)
- [ ] 4.15 Full suite green, merge `mb-writers` into main
- [ ] 4.16 Run `mb center migrate-state` on `D:\projects` and verify

Phase 5
- [ ] 5.1 Activation declared in `mb.toml`, computed in Python
- [ ] 5.2 Remove generated `.sh`/`.bat` activation scripts and jinja2

Phase 6
- [ ] 6.1 Source provenance: git SHA + zstd patch in the lake instead of source zips
- [ ] 6.2 HF pins resolved into the HF cache (no mb-side copy)
- [ ] 6.3 Native `mb report` over the lake (replaces the wandb dashboard)

Phase 7
- [ ] 7.1 polars -> DuckDB
- [ ] 7.2 seaborn/pandas out (grouped plots in matplotlib, hex palettes)
- [ ] 7.3 Remove old file-state paths; regression tests for legacy reads
- [ ] 7.4 Center setup: `mb install` / `center verify`, PATH
- [ ] 7.5 Lazy CLI imports; one DuckDB instance per process

Owner decisions pending (not blocking): libsass re-solve approach; pushing `main`/branches.

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

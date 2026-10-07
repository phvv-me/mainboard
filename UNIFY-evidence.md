# mainboard + AIZK review findings (working notes, 2026-10-06)

Legend: [BUG] defect, [LIB] replace with a library, [PATOS] use patos, [DRY] duplication,
[TRIM] lines to cut, [UNIFY] matters for the AIZK merge, [WIN] Windows-only weight.

## Survey
- mainboard src 58,951 lines / 378 files; integration tests 2,801. AIZK src 35k, tests 40k.
- SQL literals: ~160 across 29 files; hot spots state/lake.py 25, results.py 23, state/blobs.py 19,
  dispatch/state/cache.py 18, state/importer.py 16, state/schema.py 10, observe/store.py 6 (sqlite3),
  center/state.py (sqlite3 backup), monitor.py (sqlite3 errors).
- Process layers: 114 raw subprocess sites + 70 plumbum sites (two abstractions for one job).
- AIZK: SQLAlchemy/SQLModel native (178 imports), raw SQL mostly in migrations/doctor/queue;
  logs with loguru (45 imports) while mainboard logs with structlog [UNIFY].

## Root / core
- [BUG] `__init__.__version__ = "0.1.0"` while pyproject says 0.4.8; read importlib.metadata.
- [DRY] `__init__.py` keeps the export list three times (TYPE_CHECKING imports, `_HOMES`, `__all__`);
  `__all__ = list(_HOMES)` at least; `lazy_loader.attach_stub` would make it one .pyi.

## core/
- [UNIFY][TRIM] `core/project.py` dual vocabulary (mb|mainboard names, .mb|.mainboard, MB_|MAINBOARD_ vars,
  legacy marker stem): owner chose "read both names indefinitely" (2026-09-28); finishing the rename and
  deleting the legacy half is an owner decision, cost spread across Variable/markers/out_dir everywhere.
- shell.py ok (t-string sh/script good); mixes plumbum FG with subprocess.

## state/ (the DuckLake)
- [LIB] state/schema.py hand-builds DDL from (column, type) tuples and views from f-strings: SQLAlchemy
  `MetaData`/`Table` (or patos.sql/SQLModel classes) + duckdb_engine dialect; views as `select()` objects.
- [LIB] lake.py `_quoted/_literal/inlined` hand-escape SQL literals for Quack (which drops bound params):
  SQLAlchemy `compile(compile_kwargs={"literal_binds": True})` does this per type, safely.
- [PATOS] lake.py `_SESSIONS` WeakValueDictionary + `_SHARING` RLock + `Session` = patos.Shared (keyed,
  single-flight, refcounted, closable). Or a SQLAlchemy Engine (pool) + `event.listen("connect")` doing
  the ATTACH, which also replaces `_attached()`.
- [LIB] lake.py `cache_home()` = platformdirs.user_cache_dir(appname).
- [TRIM] lake.py mixes 8 concerns in 934 lines: layout, extensions, attach, insert-by-NDJSON staging,
  schema evolve, maintenance/WAL check, Quack serving, sessions. Split: Catalog (attach/engine),
  Schema (metadata+evolve), Writer (staged insert), Maintenance (checkpoint/check), Server (quack).
- insert-by-NDJSON is a measured perf workaround (2 ms/param binding); keep the idea but generate the
  typed SELECT from SQLAlchemy column types instead of `_decoded()` string building.
- tenacity already used (good) for locked-catalog retries.
- [UNIFY][TRIM] state/blobs.py (279) is a content-addressed object store built INSIDE DuckLake (8 MB BLOB
  chunks, base64 NDJSON staging, MD5 side table); its comments list the fights (16 MB JSON limit, OOM on
  BLOB slices, 102 MB catalog, no xxh3). evidence.py already has the right design as `DirectoryReplica`
  (`<root>/<2hex>/<sha256>`, verify, atomic rename). AIZK runs obstore (+SeaweedFS). Proposal: bytes in an
  object store (obstore LocalStore on the center, S3/SeaweedFS once merged), lake keeps only the index
  (`evidence_log`). Deletes blobs.py + checksums table + chunk audits; replication = store-to-store copy.
- [TRIM] evidence.py `Replica` ABC has one implementation.
- [BUG] state/blobs.py logs through stdlib `logging.getLogger` with %-format, not `mainboard.log.logger`.
- [TRIM] state/importer.py (623) is the one-time pre-lake import, done on D: 2026-09-28; with `_LEGACY`
  refusal in lake.py, `mb lake import`, the `strays` table. Legacy files still on disk (~4 GB:
  .mainboard/batches 3.2G, source-archives 812M, collection.digests.json, recovery, activate*.sh).
  Delete code + files once the owner confirms no other workspace still holds pre-lake state.

## results.py / observe/
- [UNIFY] Two result systems: lake tables (runs_log, receipts, events) and project evidence files
  (receipt Parquet parts, event NDJSON) re-read per query through EvidenceTree into ad-hoc DuckDB views
  (runs, metrics, emitted_artifacts, receipt_contexts, receipt_artifacts, artifacts, jobs). One data layer:
  trials append receipts/events to lake tables; the views become lake views defined once.
- [TRIM] results._events parses every frame in Python, json.dumps it, stages NDJSON, reads it back.
- [BUG] results.Results.table: the docstring sits after the local imports, so it is a dead string.
- [TRIM] observe/{store,channels,agentmain}.py + manifest/schema/observe.py (~400 lines) have no callers;
  the `observe` host setting is declared, never read. observe/store.py comment names a deleted
  `dispatch/state/storage.py`.
- [LIB] observe/store.py says it hand-writes sqlite3 "since patos.sql needs a `sql` extra mainboard does
  not declare": the ORM gap is a dependency decision. Declaring `patos[sql]` (SQLAlchemy+SQLModel) costs
  a few MB on hosts whose envs are GBs.

## dispatch/state (run registry)
- [UNIFY][LIB] cache.py keeps an OLTP registry (reserve, bind, compare-and-set transitions, terminal guard)
  on the append-only DuckLake: record-as-JSON + `runs` "last row per key" view + `dropped` tombstones + a
  global registry FileLock around every read-then-append to fake atomicity. Right tool: ORM tables in a
  transactional DB (SQLite on the center, AIZK's Postgres once merged): `Run`, `Host`, `Hold` as SQLModel
  (patos.sql.Model) rows, real UPDATEs inside a transaction, no registry lock, no tombstones. History
  (receipts, events, log_lines, costs, quotes, pulse) stays in DuckLake. DuckDB still reaches both
  (ATTACH sqlite/postgres), which keeps the "everything through DuckDB" objective.
- [PATOS] vocabulary.py builds `tracker()` = patos.Lifecycle(VERDICTS) but nothing calls it; cache.py
  re-implements transition checks (`_transition`, `_change` terminal guard). Put the Lifecycle on the
  Run model (`run.advance(verdict)`).
- [STYLE] verdicts are bare module string constants (QUEUED="queued" ...): a `Verdict(StrEnum)` with auto().
- RunRecord carries 24 fields incl. provider-intent fields (creation, request, lease); becomes the Run table.

## git/ (mb git)
- [BUG][OWNER-ASK] commit.py stages EVERY change in every owned repo (`_stage` = `git add -A` over all
  changes minus withheld) and commits all repos with one message. With several agent sessions sharing
  one working tree this sweeps another session's WIP into a commit with the wrong message (live case:
  packages/mainboard has 24 dirty files from a cutok session). Redesign: `mb git commit -m MSG [PATHS...]`
  commits what is staged plus the named paths (workspace-relative, routed to their owning repo); the
  bottom-up submodule pointer of a child that committed is staged automatically (the real value of the
  verb); `--all` keeps today's sweep for the owner's deliberate "commit everything". Nothing staged and
  no paths: refuse with the hint, never sweep.
- [WIN] repo.unlinked()/faithful() + commit relink step exist only because a Windows checkout writes
  symlinks as files; delete after the center leaves Windows.
- process.py: subprocess git with timeout/env/credential; fine (pygit2/dulwich would not cut lines for a
  submodule walker). Unify with the one process layer (plumbum or subprocess, not both).
- [STRUCT] cli.py (2,579 lines) defines every verb group inline in one function; cyclopts sub-apps per
  package (git/cli.py, state/cli.py ...) with cli.py only assembling; many docstrings carry empty `Args:`.

## dispatch/ (12.6k)
- Usage (lake, 2026-10-06): runs by kind ssh 804, pbs 281, vast 36, hpc-ai 1; NEVER modal, runpod, lambda,
  slurm. [TRIM][OWNER-ASK] backends/modal.py 231, runpod 107, lambdacloud 86, schedulers/slurm.py 225
  (+ `modal` extra, their tests/docs): 650+ lines with zero use. hpcai: 1 run.
- [BUG?] lake.costs has 25 rows, all $0 crimson/miyabi-g; 36 vast rentals left no cost row.
- [TRIM] lake tables `studies` and `strays` hold 0 rows.
- [STRUCT] dispatcher.py `Dispatcher` is a 928-line god class (run, submit, allocating, mirror, fetch,
  verify, stage, probe). Three meanings of "Verdict": dispatcher.Verdict (model), core.section.Verdict
  (PASS/WARN/FAIL), verdicts.py module.
- [LIB][RISK] agent/program.py (1,106, stdlib-only, py3.9) + mirror.py + sync.py + snapshots.py + landing.py
  re-implement rsync (survey digests, tar stream, prune) and hard-linked pinned snapshots. rsync
  `-a --delete --files-from=<git-aware list> --link-dest=<prev snap>` does transfer+prune+snapshot;
  present on every Linux host and macOS; Windows host needs msys2 rsync or keeps tar. Biggest single
  LOC cut available, also the riskiest: spike after the deadlines.
- [WIN] keys.py (538): Git/MSYS ssh selection, proxy-mode login relay daemon, keystore, askpass; on the
  Mac center = ssh config ControlMaster/ControlPersist + UseKeychain/AddKeysToAgent. Keep ~`shared()`.
- [WIN] shells.py Windows dialect (PowerShell EncodedCommand staging): stays only while pedro-home is a
  native Windows job host (vs WSL).

## board.py / verdicts.py / monitor.py
- [STRUCT] board.py `Board` (1,400 lines, ~60 methods) is a service locator for every subsystem
  (dispatcher, manifest, resolver, git, doctor, monitor, verdicts, paper, scaffold, fleet, deps, batch,
  submit, shell, provide...). `Job` and `ProviderJob` repeat one 7-method protocol with no shared type.
  `Board.once()` hand-rolls memoization (functools.cached_property / patos.DerivedCache).
- [UNIFY] verdicts.py (934) reconciles three sources per answer (receipt files in two shapes, lake
  events, run registry floor): the cost of two result systems; one data layer shrinks it.

## engines/
- [STRATEGY] engines/compile (~5k) + manifest (~1.9k) translate mb.toml into per-env pixi manifests and
  drive pixi. Question for the plan / Rust evaluation: mb.toml as a pixi manifest plus `[tool.mb]`
  extensions, compiling less.
- [WIN] engines/compile/backend/windows_task.py (303) runs pixi tasks without a shell on Windows.

## Cross-package duplication
- [DRY] host facts modeled 5 times: dispatch.targets.Facts, probe.system.System, probe.snapshot.HostFacts,
  probe.census.Census, probe.machine.Machine (Singleton) (+ backends.cloud.Machine for rentals).
- [DRY] `CudaRuntime` Protocol defined twice (probe/providers/nvidia/protocols.py, profile/providers/
  nvidia/protocols.py); two `Event` (diagnostics, batch.receipts); two `Snapshot` (agent program,
  staleness); two `Study` (experiments.study, profile.study).
- [DRY] experiment declaration/recording spread over experiments/ (827), trials/ (3,626), lab/ (541);
  jobs over batch/ (1,277), jobs/ (1,418), dispatch/jobs/ (189). Candidates for one package each.

## Spike 2026-10-06: SQLAlchemy on the pinned DuckDB
- SQLAlchemy 2.1.3 + duckdb-engine run on duckdb 2.0.0.dev2609250715; an engine `connect` event loads
  ducklake/sqlite from mb's extension dir and ATTACHes the real lake READ_ONLY; Core `select()` against
  `lake.runs` works (`MetaData(schema="lake")`).
- `compile(literal_binds=True)` inlines correctly for Quack ('it''s' escaped; timestamps as text literals).
- Bulk INSERT through SQLAlchemy is ~4.6-5 ms/row (executemany 5,000 rows 23 s; multi-VALUES 2,000 rows
  10 s): the binding cost lake.py's NDJSON staging exists for. Keep the staged writer for appends;
  generate its typed SELECT from the SQLAlchemy Table. SQLAlchemy for DDL, views and queries only.

## Reliability review (critic, 2026-10-06) - decisions to fold into UNIFY.md
- W1 observe: GO only after Holdings.read validates with extra="ignore" (held.py:52) or holds are
  rewritten; HostProfile.observe is dumped into every stored hold and Declared forbids extras.
- W1 modal/runpod/lambda: runs on held rentals are kind=ssh; check `SELECT DISTINCT held->>'provider' FROM
  lake.holds_log` + costs/quotes first; remove with mainboard.toml:929-943, cli.py:1190-1195 rentable map,
  integration/test_clouds.py:13, pyproject extra. Slurm: also change targets.py:32 probe.
- W1 importer: NO-GO, research/reproducibility/.mainboard/dispatch/db.sqlite is a pre-lake registry with
  no lake beside it (the _LEGACY refusal protects it). studies: code live (experiments/study.py, reporting,
  fleet); keep. Windows code: only after the Mac ran unlock + full dispatch/collect on every host kind.
- W2: GO with: plan-before-touch (route each path to the deepest initialized owned repo; refuse foreign,
  symlink-crossing, typo paths; ls-files spelling); submodule root = pointer bump in parent; dir with
  child gitlinks gets :(exclude)child; never-commit/oversized/undeclared nested/ignored named paths are
  REFUSED with reason; renames need both sides; commit via `git commit --only` over named paths (others'
  staged content untouched); pointers when child committed this run or child touched and HEAD != gitlink;
  a held child does not hold the parent's unrelated paths; retry on index.lock; skip untouched repos (no
  attach/relink/merge); merge upstream only in committed repos and only if no dirty path is touched.
- W3.2 SQLite registry: CAS = conditional UPDATE + rowcount; partial unique index for reservations;
  surrogate PK + UNIQUE(target, handle, submitted_at) (bind changes handle); real columns; settlement
  lock STAYS (held minutes across ssh); WAL + busy_timeout + synchronous=FULL + fullfsync; never create on
  read; add state.sqlite to center/state.py _SNAPSHOTS. Lifecycle NOT yet: VERDICTS forbids transitions the
  code performs (SUBMITTING->PREPARED reopen, QUEUED->FAILED allocation.py:61, QUEUED->TIMEOUT
  monitor.py:394, QUEUED->OK) [BUG: the declared table is wrong]. A SQLite registry ends remote registry
  access over Quack (README "gold$ mb job list ... read and written live") - owner decision.
  Migration: quiesce, copy raw JSON losslessly, read back per key, bump schema 7 and rename the logs to
  fence old writers; tested reverse script; keep logs 30 days.
- W3.3: literal_binds cannot render JSON/binary (sqlalchemy#10832); keep `inlined` until TypeDecorators
  pass a matrix. platformdirs(appauthor=False, opinion=False) or the Windows ext dir moves. Pin
  duckdb-engine after DuckDB 2.0.0 (Oct 21).
- W3.4: GO but the export must include SOURCE blobs (closures, provenance.py:150-215), not only evidence;
  stream all distinct sha256, hash+fsync+rename, resumable, read back evidence AND closures, write order
  object-then-index, read fallback to lake during transition, delete blob rows last then expire/cleanup.
  Quack clients lose recall from a LocalStore.
- W3.5: hosts keep writing files; the CENTER ingests them (anti-join natural keys; new table names, since
  `events`/`receipts`/`runs` exist with other shapes); backfill + EXCEPT ALL both ways.
- W6: NO-GO for snapshots (kernel locks, atomic verified pins, inode/ctime digest memory, case-fold
  pruning); `--delete --files-from` does not delete unlisted files; macOS openrsync --link-dest issues;
  saving ~300-400 lines only. Low-priority spike for mirror transfer at most.

## Found during the move (2026-10-07)
- [BUG] The Node second stage's lock (`pnpm-lock.yaml`) lives only in the git-ignored
  `.mainboard/envs/<env>/`, never in the committed mb.lock, so a fresh center fails `mb install`
  ("has no pnpm lock; run `mb lock default` locally"). The lock that pins an env must travel with
  mb.lock (or the center move must carry it).
- [BUG] center/state.py excludes `*.sqlite-wal` as machine-local, so a raw copy of a live catalog loses
  committed transactions; snapshot through the SQLite backup API.

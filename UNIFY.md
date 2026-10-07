# Unify: one trimmed mainboard that AIZK can join

Version 2 of 2026-10-06. Version 1 was reviewed by two critics, one on simplicity and real line
counts, one on correctness and migration risk. Both read the code. This version keeps what
survived them and records what changed.

Sources: the evidence is in `UNIFY-evidence.md` (the critics' findings are in its last section), and
`SERVE.md` holds the AIZK audit and the `mb serve` phases. The figures come from mainboard at 58,951
source lines (378 files, 2,801 test lines) and from the lake's usage records.

## What the critics changed

- **SQLAlchemy does not shrink the lake.**
  - `state/schema.py` tuples are already as dense as `Column()` lines.
  - The views rely on `QUALIFY`, `EXCLUDE` and `rowid`.
  - `results.py` is DuckDB-only SQL (`json_merge_patch`, `UNION ALL BY NAME`, `read_ndjson_objects`).
  - `literal_binds` cannot render JSON or binary (sqlalchemy#10832).
  - Bulk inserts through SQLAlchemy measured about 5 ms a row on the pinned DuckDB.

  So W3.3 is cut, except `platformdirs` (with `appauthor=False, opinion=False`). SQLAlchemy enters
  where it removes code: the registry, once it lives in Postgres.
- **The registry moves once, straight to Postgres** (SERVE Phase 2), not lake → SQLite → Postgres.
  patos.sql's JSON and enum types are Postgres types already. Until then the lake keeps the registry.
- **`patos.Lifecycle` does not go on runs yet.** The declared `VERDICTS` table forbids transitions
  the code performs:
  - SUBMITTING→PREPARED (`reopen`);
  - QUEUED→FAILED (`allocation.py:61`);
  - QUEUED→TIMEOUT (`monitor.py:394`);
  - QUEUED→OK.

  Fixing the table to match reality is a bug fix of its own.
- **Moving bytes to an object store must include source trees.** `dispatch/provenance.py` keeps every
  dispatched tree in `Blobs` too. Use `DirectoryReplica` as the local store (it already is one), and
  add obstore only when S3 arrives.
- **rsync cannot replace pinned snapshots.** Those need kernel locks, atomic verified pins, digest
  memory by inode and ctime, and case-folding-safe pruning, and `--delete` does nothing with
  `--files-from` unless rsync recurses. Spike the mirror transfer only, after the deadlines.
- **One result system means the center ingests the files hosts write.** Hosts never write to the
  catalog. Use new table names: `events`, `receipts` and `runs` already exist with other shapes.
- **W5's structural moves are cut.** Splitting cli.py, slimming `Board`, splitting `Dispatcher`,
  merging fact models and merging packages move code without removing it. `Board.once` cannot
  become a `cached_property`: it is shared across `on()` copies, emptied by key and guarded by a
  lock. The host-fact classes compose rather than duplicate.
- **Wrong counts corrected.**
  - Process launches: 57, not 114.
  - The "stdlib logging users" and "httpx" bullets were worth nothing.
  - The provider removal is about 720 lines, because `CloudBackend` has only RunPod and Lambda under it.

## Order

### 1. Remove what nothing uses (no behavior change)

| Remove | Lines | Condition |
|---|---|---|
| observe store, channels, agentmain, the `observe` host schema | ~400 | read holds with `extra="ignore"` first (`manifest/held.py:52`): every stored hold dumps the field |
| `mainboard.testing` (seal, windows, plugin) | 568 | registered by no conftest or `pytest_plugins` in the workspace |
| `trials/distribution.py` | 97 | "for an executor not yet built", only re-exported |
| `vocabulary.tracker` | 3 | no caller |
| free fixes: `__version__` from metadata, `__all__ = list(_HOMES)`, the dead docstring in `results.table` | ~30 | none |

### 2. `mb git commit` stops sweeping (owner request)

- `mb git commit -m MSG PATH...` commits exactly the named paths, with `git commit --only` per repository, bottom-up.
- `mb git commit -m MSG` commits each repository's index.
- `--all` keeps today's sweep.
- Nothing staged and no paths: refuse.

Details:
- **Routing:** each path goes to the deepest owned repository holding it, and a directory takes the owned repositories under it whole.
- **Refusals:**
  - a foreign, outside or nonexistent path is refused before anything is staged;
  - a named `never-commit`, oversized, link-as-file or undeclared nested path holds its repository with the reason.
- **Pointers:** a parent stages only the pointers of submodules that committed in this run. Pointers under a named directory are excluded otherwise.
- **Failures:** a held child no longer holds its parent's unrelated paths in the selective mode.
- **Merging:** only `--all` merges upstream.

### 3. macOS center gaps (the move needs them)

- A launchd settler beside the systemd one (`durable.py`), so `mb job list --every 20m` works on the Mac.
- Center snapshots must copy SQLite through the backup API; a raw copy loses the WAL.

### 4. Owner decisions that unlock the large cuts

| Decision | Lines |
|---|---|
| Drop RunPod, Lambda and the orphaned `CloudBackend`. Modal: the extra and backend; the workspace's own `modal` pin is AIZK's. Check `holds_log` providers first. | ~950 |
| Drop HPC-AI (one run ever) | 513 |
| Drop Slurm (zero runs; also change the `sbatch` probe in `targets.py:32`) | 225 |
| Drop the study/fleet feature (`experiments/{study,fleet,reporting}`, the `studies` table, its monitor hooks; never one row) | ~450 |
| Drop the center-move machinery (`center/migrate`, `state`, `remote`, `carrier`), now that a manual copy moved the center | ~1,500 |
| `lint/` becomes pre-commit/prek hooks (check-yaml, check-toml, whitespace, line endings) | ~500 |
| `log.py` becomes loguru directly, keeping the JSON line format job-log readers parse. Aligns with AIZK's 46 loguru imports. | ~100 |
| After the Mac has run an unlock plus a full dispatch and collect on every host kind: the Windows center code (keys.py relay, daemon and keystore, MSYS guard, git relink, Windows task runner) | ~900 |
| Import or archive reproducibility's pre-lake registry, then delete the importer, `_LEGACY` and `strays` | ~680 |

### 5. Bytes leave the lake (after the deadlines)

Evidence and source-tree objects move to the content-addressed directory store (`<root>/<2 hex>/<sha256>`, written with fsync and an atomic rename):
1. Run `mb lake check` and get a clean result.
2. Export every distinct digest, resumably, hashing on write.
3. Read back every digest in `evidence` and in `closures`.
4. From then on, write the object first and the index row second.
5. During the transition, read from the store and fall back to the lake.
6. Delete the blob rows last, then expire snapshots and clean up the old files.

This deletes `state/blobs.py` and the `checksums` table: about −300 lines. Quack clients lose recall from a local store; that ends with Postgres anyway.

### 6. AIZK joins (SERVE phases)

1. Retrieval fixes (SERVE Phase 0).
2. Slimming (Phase 1).
3. Drop CockroachDB (24 files).
4. The DuckLake catalog and the run registry move into AIZK's Postgres (Phase 2). The registry becomes SQLModel tables with conditional-UPDATE compare-and-set and a partial unique index for reservations. The settlement lock stays. Quack serving (~175 lines) is deleted, since remote machines attach `ducklake:postgres` directly.
5. Then the one result system: the center ingests the files hosts write.

### 7. Spikes before the rewrite question

- **pixi native features:** can they replace the per-environment manifest compiler (engines/compile plus manifest, about 7,000 lines)? This is the largest block nobody has examined.
- **rsync for the mirror transfer only:** recursive, an include list naming every file and its parent directories, `--exclude='*'`, `--delete`, with protect rules for host-written files. Pin rsync 3.x through pixi, because macOS ships openrsync.
- **Rust:** evaluated by a separate agent at the end, against the trimmed Python.

## Guardrails

- Work happens on the Mac center. Each step leaves the integration suite and `mb ci` green.
- Nothing is committed or pushed without the owner's word.
- What crosses the wire (the agent program, closures, markers, receipts) keeps reading the old shape until every host is set up again.
- A migration is a one-shot script with a read-back comparison, a tested reverse, and the old data kept 30 days.

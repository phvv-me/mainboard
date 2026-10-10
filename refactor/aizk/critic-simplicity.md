# Critic: simplicity and real size of the AIZK-into-mainboard plan

Reviewed 2026-10-10 (measurements dated the same day). Inputs: `unify.md`, `unify.json`, `state.md`, `options.md`,
design v3 (`design.md`, `owner-decisions.md`), `packages/mainboard/SERVE.md`, against the code of `packages/aizk`
(HEAD `6ebeab1`) and `packages/mainboard` (0.5.4). Nothing was changed or committed. On crimson I ran only reads: `ssh crimson`
with `docker exec aizk-worker-1 cat /proc/1/status` and the container's cgroup `memory.stat`. On the center I ran `df`,
`vm_stat`, `sysctl`, `ps`, `du`, `docker info` and public API lookups.

## Real sizes (measured)

| thing | measured |
|---|---|
| `packages/aizk/src/aizk` | 35,043 lines, 235 files (plus `src/eval` 3,978; tests 40,142 lines in 138 files; deploy 2,904 lines, compose 1,182, Caddyfile 208; web TS 6,132) |
| `packages/mainboard/src/mainboard` | 57,174 lines, 371 files; `cli.py` 2,749 (one build function of closures), `durable.py` 520, `center/` 2,468, `state/lake.py` 1,021 |
| the plan | 22 packages (11 aizk, 7 mainboard, 4 both or live), 131 touched-path entries (85 aizk, 29 mainboard, 17 both) and 50 new files (19 tests), 95 change bullets, 108 acceptance items, 6 live events (L-1, L0, L0.5, L1, L2, L3), 18 owner decisions |
| code that the slimming makes dead, not deleted by any package | Docling client and models 469 + `OneShotDoclingConverter` + `DoclingConversionJob` (353) + the `heavy-converter` extra; ClamAV client 132; GLiNER gate client 205 + `services/gliner` 77; web-search lane `web/` 1,114 + `integrations/web` 655 (zero `web_search` or `web_fetch` events ever); `ops/hardware.py` 173; 34 gliner/docling/clamav/vllm mentions in `settings.py` |
| CockroachDB, "about 8,400 lines" | executable CockroachDB code is about 1,325 lines (4 migration modules 788, `background/portable.py` 230, `commands/aws_*.py` 227, `store/ddl/compiler.py` 80) plus branches in 27 files; AWS `infra/` 970; the rest of the 8,400 is `hackathon/` at 5,592 lines, of which 4,418 are Markdown and 1,056 are two Python files |
| center now | 24 GiB RAM, 2.5 free, 10.5 inactive, swap 3.4 of 5.1 GiB used; non-reclaimable about 9.2 GiB (anon 5.8 + wired 2.7 + compressor 0.7); data volume 410 of 460 GiB used, **21 GiB free, 96%** (the plan says 28 GiB, 94%) |
| crimson worker, read-only | `VmRSS` 5,636,864 kB, `RssAnon` 5,597,060 kB, `RssFile` 39,804 kB, **`VmHWM` 8,540,820 kB**; the cgroup's 7.6 GB of file cache is separate. It is a real 5.6 GiB heap with an 8.5 GiB peak, not page cache |

## Findings

### 1. The three facts that decide the plan are measured last, after 14 packages

`unify.md` section 9 lists the worker's 5.6 GiB, the Japanese lexical lane and the find latency on a 2 GB `shared_buffers`
Mac as unsettled, yet U16 (the first real measurement) needs U03 to U13 first. If any one fails, most of U03 to U15 is wasted.
Evidence: the RAM budget (6.2 to 7.7 GiB) holds only if the worker drops from 8.5 GiB peak to 2 GiB; the center already swaps
3.4 GiB with the owner's apps open, and carries about 9.2 GiB it cannot reclaim, so the budget is a floor and not a margin.
Change: insert a read-only wave W-1 of three days before U00, with written go or no-go numbers:
(a) peak RSS of each of the 17 schedules, run one at a time against a restored copy under `/usr/bin/time -l` (go: every job
at or below 1.5 GiB);
(b) restore the live dump on the center with pgvector HNSW and time the real `find` lanes (go: p50 at or below 5 s);
(c) the lexical A/B on the 60-case fixture plus the first golden slice;
(d) OpenRouter embedding throughput and whether `dimensions=1024` is honored;
(e) native Logto and cloudflared (finding 7).
U16 then shrinks to rehearsing the cutover, not discovering feasibility.

### 2. Do not ship intermediate builds to crimson (drop L0, L0.5, the two release trains and the dual backends)

Evidence: crimson runs a clone at `be29fcb`, independent of repository HEAD; five containers cannot restart (F1), so every
live event on crimson (L0 recreate, L0.5 swap) risks the one stack that works. L0 exists only to deliver the keep journal
and maintenance mode (U10) plus U09, U08, U02; the "two release trains" and the weak point "the keep-journal format must not
change after L0" follow from it. L0 also forces `index_backend` and `lexical_backend` settings, the `vendor_pg` marker, double
test matrices ("the same swap tests pass with vchordrq and with hnsw") and the U18 deletion pass, which contradicts the house
rule that correct behavior is never an opt-in flag and that backwards compatibility is ignored.
Change: build once, for the center. AIZK HEAD becomes pgvector-plus-FTS only now; the vendor-to-pgvector conversion lives in
a one-off migration tool (finding 9), not in the product. Crimson stays untouched until the freeze (a dump of the running
database needs no restart). Remove L0, L0.5 and L-1's role as a gate for them; keep L-1 as a read-only baseline. Move U02
(SERVE phase 0 ranking fixes) to its own track so ranking, backend, embedder and lexical changes are never confounded in one
golden-set comparison.

### 3. U10 (maintenance mode, keep journal, replay, typed retryable errors) rebuilds "stop the front door"

Evidence: load is 0.96 keeps an hour (689 in 30 days), six shares in the ledger's life, 168 reports in two months, and the freeze
budget is 15 minutes expected, 25 maximum, with the owner present. "Never acknowledge a keep the surviving side does not hold"
is satisfied by `docker stop` of `aizk-server-1`, `api` and `worker` (no process, nothing acknowledged), with zero AIZK code.
Needing `find` to survive the freeze is the only reason U10 patches the read path (access-count writes skipped, usage lines
journaled to a file), and a one-time announced 15 to 25 minute read outage costs nothing against that. `document` is already
a journal: `Timestamped`, unique `(source_uri, scopes)`, `content_hash`.
Change: delete U10 (7 files, one migration, the `keep_journal` table, `journal replay`, the 200-run kill property tests).
Cutover step T0 becomes "stop the three containers". Rollback after the flip, within 24 hours, is forward-only, or one documented
`\copy (select ... where created_at > T)` of `document`, `artifact` and `blob` rows back to the old database. Record the accepted
risk (at most about 24 keeps) as an owner decision.

### 4. U07 spool plus U15 lake streams rebuild a pull that DuckDB already does

Evidence: `usage_event` is immutable (`mutable = False`), has a unique `capture_key` and `created_at`, and holds 5,572 rows
(2.5 MB). The lake on a PostgreSQL catalog must load DuckDB's `postgres` extension anyway (M36 to M38), so
`ATTACH 'dbname=aizk ...' AS aizk (TYPE postgres, READ_ONLY)` and `INSERT INTO memory_usage SELECT ... WHERE created_at > :cursor`
with M23's cursor is about 20 lines in the settler. The spool introduces a crash window (commit, then append) that U07's acceptance
must close with a property test over 200 random kill points, a size-capped drop counter, segment files, a versioned JSONL contract and
a quarantine rule for unknown versions (U15's 5 files). The pull has no such window, and AIZK still never imports DuckDB.
`memory_events` has no named consumer ("incident review"); the five OpenTelemetry packages are removed either way.
Change: delete `spool.py`, `AIZK_SPOOL_DIR`, U15 and the `memory_events` stream. Keep the loguru JSON sink with rotation; read it with
`read_json` when an incident needs it. Add one policy on `usage_event` for a role `aizk_usage_reader` (RLS is forced, so the
reader sees zero rows without it). Keep the `memory_usage` mirror as a 20-line pull, owned by mainboard (inside U13's doctor or
M23), not a package. What remains of U07 is "remove OpenTelemetry and the observability profile", which belongs with U05.

### 5. U12's PITR is the highest-risk and least-needed package, and its backup set has holes

Evidence: `archive_command` writes to the internal disk (96% full). An archive that fails or fills stalls WAL recycling and takes
down all three databases, including `mainboard` control, so the settler stops too. The "hard cap" can only be a doctor row, because
PostgreSQL itself cannot cap it. A same-disk WAL archive is not a backup, and the offsite copy cannot be made from PostgreSQL's
`archive_command`: `durable.py` states that "an agent holds no Full Disk Access, so a lake on an external volume never answers it",
and the postmaster runs under the same launchd context. The plan never mentions this for U12 or U13. The backup set also omits
the 659 MB of originals in `aizk_objects`, the FastMCP state files and the env secrets, which `migration.scope` itself lists.
AIZK already has `backup.py` (192 lines of `pg_dump -Fc` and `pg_restore`) and `aizk admin database backup|restore`, and M38 already
dumps `mainboard`. The plan's own status quo is a 24-hour RPO.
Change: replace U12 with one daily job run from the settler context (`ssh localhost`, which has Full Disk Access): `pg_dump -Fc`
of `aizk`, `logto`, `mainboard` (M38's mechanism for the last), `pg_dumpall --globals-only`, an rsync of the object directory and env
files, then PORTABLE and crimson by checksum, and a weekly restore into the throwaway cluster. Drop `pitr.py`, WAL archiving, WAL
summaries and incremental chaining. If the owner wants a minutes-level RPO, add it as an explicit decision (D19) with
`pg_receivewal` to PORTABLE through the settler context, not as the default.

### 6. U17, `route.py` and the 300-random-kill chaos acceptance are a library for a one-time event

Evidence: one cutover, owner present, 2.7 GB, 1 keep an hour. Design S10 says "three migration layers are unnecessary; use one
center orchestrator"; `center/migrate.py` (477 lines) already has the converging-steps pattern. U17 adds `memory/migrate.py`,
`memory/route.py` (a Cloudflare API client "with a manual-confirm fallback", i.e. the dashboard click), edits `center/migration.py`
and `cli.py`, and asks for kills "at every state boundary and 300 random points". After L3 it is dead code that U18 does not delete.
Change: U17 becomes a lead runbook of about eight idempotent steps (stop the three containers, `pg_dump | pg_restore`, one verify
script that is also the doctor probe, flip the hostnames in the dashboard, start) rehearsed twice in U16. No package, no
`route.py`, no chaos tests.

### 7. Docker Desktop is avoidable, and with it two of the plan's own unverified risks

Evidence: the plan keeps Logto and cloudflared as containers because "no conda-forge build" (cloudflared is indeed 404 there),
so it pays for Docker Desktop: a VM configured at 8.3 GB (`docker info`), `Docker.raw` already 3.7 GB allocated, a GUI app with
`AutoStart: false` (after a reboot of the headless Mac the tunnel and Logto stay down until a login), the inferred 1.0 to 1.5 GiB,
and the unverified `host.docker.internal` routes in section 9. But cloudflared publishes `cloudflared-darwin-arm64.tgz`
(release 2026.10.0, a single static binary), and Logto documents a Node deployment (`npm init @logto`, "requires PostgreSQL").
`oauth2-proxy` exists only to give Grafana, the tracing UI and the operator pages one sign-in; every `/api/admin/*` route already calls
`require_admin`, and Grafana and Tempo are being deleted. ClamAV is already dropped by D7.
Change: run cloudflared as a checksum-pinned release binary and Logto from its release tarball under the env's Node (spike in W-1;
if Node 26 is rejected, keep that one container), decide now that oauth2-proxy and the `console` hostname gate are dropped, and drop
Docker from the plan. Record the one real cost: the compose isolation ("an exploit in the MCP server therefore cannot obtain the
owner connection") is lost because all services run as the same macOS user that runs the agents; keep the owner password readable only
by the worker's plist and accept the remainder knowingly.

### 8. The worker's 2 GiB target has no mechanism; the Docling pattern already in the code is the mechanism

Evidence: U09 says "profile with tracemalloc or memray and remove the top retained structures". The measured heap is 5.6 GiB with an
8.5 GiB peak. The code materializes embeddings as Python floats (`raptor.to_floats`, `list[float]`, about 32 KB per 1,024-dimension
vector against 2 KB as `halfvec`), builds `networkx` graphs over every live fact for Louvain in pure Python (`graph/communities.py`),
and `rewrite_embeddings` calls `.all()` on a whole table. CPython rarely returns such memory to the OS, so a profile-and-trim pass will
not hold a 2 GiB ceiling. The converter already solves exactly this for Docling: "convert in one subprocess ... then release all
model memory".
Change: in U09 run `communities`, `raptor`, `insight`, `dedup`, `decay` and `profile_refresh` as short-lived subprocesses
(`aizk admin graph <job>`, the CLI commands already exist) launched by the queue, so the resident worker is the dispatcher at about
0.3 to 0.5 GiB and each job's peak is bounded in time. Add a per-job peak-RSS row to the doctor. Keep the fan-out and
`pgqueuer_log` fixes, which are sound.

### 9. U03's `restore-plan` (`pg_restore -L`) cannot skip a column; prove it or convert on the donor

Evidence: `chunk` stores a `bm25vector` column in its `CREATE TABLE`, and `COPY chunk (..., bm25) FROM stdin` names it. `pg_restore -L`
selects whole TOC entries (TABLE, TABLE DATA, INDEX, TRIGGER), so no list can drop one column; without the vendor type the table
creation and its data load both fail. U03's acceptance claims the "generated -L list ... skips the bm25 column" restores without errors.
I did not run this; U00's 5,000-row fixture dump will settle it in an hour.
Change: add the check to W-1. If it fails, convert on the donor at the freeze with the extensions present: with the server stopped,
`CREATE DATABASE aizk_flat TEMPLATE aizk`, `ALTER TABLE chunk DROP COLUMN bm25`, `DROP EXTENSION vchord, vchord_bm25, pg_tokenizer CASCADE`,
then a plain `pg_dump -Fc` of `aizk_flat` restores on the center with stock `pg_restore`. This removes `restore-plan` and keeps
`convert` for the HNSW and FTS builds.

### 10. U08's shadow columns and one-transaction swap are built twice and may not be needed

Evidence: U08 adds shadow columns in five tables, a resumable batch writer, a catch-up pass, a swap transaction tested with 50
concurrent finds, and a retention period for the old columns (disk). U16 then re-embeds the staging copy the same way, and L0.5 does
it again live. AIZK already has `graph/reembed.py` (92 lines), described as "the one-command model migration". The whole corpus is
about 23 M tokens, about $0.25.
Change: measure throughput in W-1 and pick the smaller of two paths: (a) if a full re-embed takes at most about 15 minutes, run the
existing `reembed` on the restored copy inside the freeze (reads are down anyway under finding 3) and delete the shadow machinery;
(b) if not, re-embed the staging copy once, promote it, and copy at SQL level the `document`, `artifact` and `blob` rows with
`updated_at` after the staging snapshot (a few dozen at one keep an hour; the worker rebuilds chunks and graph from them).
Preserve document ids in the copy, since agents quote them. Keep U08's startup dimension probe (it is tiny), the rerank
fall-through and the deletion of the GLiNER, vLLM and Docling services. Drop the shadow columns and the swap test.

### 11. U13 builds a second service supervisor beside M36's and `durable.py`'s

Evidence: design v3's "Service managers" paragraph already specifies a same-user LaunchAgent or systemd user unit with `KeepAlive`
and a readiness probe for PostgreSQL, and `durable.py` already registers `LaunchdAgent` and `SystemdUser` strategies that write
the plist with `plistlib` and call `launchctl`. U13 adds `memory/units.py`, `services.py` and `doctor.py` and acceptance for
dependency order, reverse stop order within a bounded time, restart throttling and no side-effect starts for 6 to 9 services.
launchd retries (`KeepAlive` plus `ThrottleInterval`) make start ordering unnecessary because AIZK's own connect loop already waits
for PostgreSQL. Five packages (U11, U12, U13, U14, U17) all edit `cli.py`, a 2,749-line file.
Also, AGENTS.md says to run `mainboard upgrade` before every experiment; the services "from the workspace environment" would change
`sqlalchemy==2.1.0b3`, `fastmcp` and `asyncpg` under live processes on every campaign. M36 retains a binary generation for
PostgreSQL but nothing does for the Python env.
Change: extend the one renderer in `durable.py` with `keep_alive` and use it for PostgreSQL (M36), AIZK's three processes, Caddy,
the frontend, Logto and cloudflared. No `units.py`, no ordering machinery. Point the units at a frozen env generation updated only by an
explicit `mb memory up --upgrade`. Put all `mb memory` verbs in their own `App` module mounted like `host`, `job` and `lake`, so `cli.py`
gains one line. If ordering is ever needed, use `process-compose` (conda-forge, osx-arm64) instead of writing it.

### 12. Delete the code of everything the plan removes, and take U05 first

Evidence: see the sizes table; about 2,600 source lines (plus their tests under `tests/web`, `tests/integrations/docling`,
`tests/integrations/clamav`, `tests/serving/test_entity_gate.py`) become dead, and the plan deletes only compose entries, the Cockroach
profile and, in U18, the vendor branches. U08 even edits `web/sanitizer.py`, for a lane with zero uses. `settings.py:375` already says
"VectorChord is the low-memory default. HNSW and tsvector are the portable fallback", and `Embedded.__table_args__` takes
`settings.vector_index_backend` and `vector_opclass`, so U03 is mostly a setting, one migration and `hnsw.iterative_scan`. U03, U04 and
U05 all rewrite `vector.py`, `settings.py` and `0001_init.py` and are marked conflicting, yet U05 is ordered after them.
Change: U05 goes first and absorbs the dead-service deletions: Docling, ClamAV, the gate, the `heavy-converter` extra, `ops/hardware.py`,
the 34 settings, the OTLP ingress (`AIZK_OTLP_TOKEN` is a required Caddy variable) and, unless the owner wants web search kept as a
product feature, the web lane. Restate D2's benefit as about 2.3k executable lines, not 8,400.

### 13. Put pgvector in `mainboard.toml` now and delete the extension-directory machinery

Evidence: U00 supplies pgvector "through `extension_control_path` and `dynamic_library_path` from an unpacked conda-forge package until
the workspace pins pgvector", and U11 owns an "extension directory" and tier 2 (source-built VectorChord, bm25, tokenizer). conda-forge
`pgvector` 0.8.6 exists for osx-arm64, linux-64 and linux-aarch64 and installs into the PostgreSQL prefix, so once it is a dependency,
`CREATE EXTENSION vector` works with no path settings.
Change: the lead adds one line (`pgvector`) to `mainboard.toml` first; U00 and U03 use the environment's own extension; U11 loses
"extension directory" and `extension_control_path`; tier 2 becomes a note, built only if the golden gate fails.

### 14. The center's disk floor is not met today and the plan stacks copies on it

Evidence: 21 GiB free now (was 28), against the plan's own prerequisite "at least 15 GiB free after the move". Steady state is
about 8 GB (U16's target), plus a `aizk_stage` copy (2.8 GB) during rehearsal, shadow columns (about 0.5 GB), `max_wal_size` 4 GB, a
350 to 450 MB dump, the restore staging and the base-backup chain on the same disk. That puts the peak at roughly 12 to 15 GB, so a
free floor under 15 GiB is already breached before the cutover.
Change: set the floor as a number before L1: at least 30 GiB free before staging (candidates the owner decides on: `~/Library/Caches`
18.1 GiB, `~/.cache` 11.0 GiB), `max_wal_size` 2 GB, drop `aizk_stage` after U16, and no WAL archive or base backups on the internal
disk (finding 5).

### 15. U14 depends on an `aizk[client]` extra that no package creates

Evidence: `packages/aizk/pyproject.toml` has one extra, `heavy-converter`. `fastmcp`, `httpx` and `keyring` are base dependencies
beside `sqlalchemy==2.1.0b3`, `asyncpg` and `pgqueuer`; U14 lists only mainboard files. Mainboard ships to every host and Miyabi's
14 GB login node. AIZK's own CLI already has `find`, `keep`, `share`, `status` and `auth` (`cli.py`, `commands/client.py` 389 lines); only `report` is
missing. `aizk` is already installed in the mainboard environment, and cyclopts 5.2 supports lazy import-path commands.
Change: U14 becomes `app.command("aizk.cli:memory_app", name="memory")` in a `memory` App (about five lines), imported only when
the verb runs; add `report` to the AIZK CLI. No extra, no `memory/client.py`.

### 16. Cut U19, U20, U21 and most of the owner decisions

Evidence: U19 (a plugin hook in the public MCP server so it can expose job and host tools) and U20 (DEV LOG writes) are labeled
"deferred", yet U21 depends on them. Agents on the center already have a shell with `mb`, and a DEV LOG note is a `keep`. U21 is the
close-out audit that M33 already is. Of 18 decisions, D3 (prune crimson's build cache) is a standalone cleanup, and D1, D2, D13, D14,
D15, D16, D17 and D18 are defaults nobody will contest.
Change: move U19 and U20 to the excluded list with triggers, fold U21 into M33's ledger, and ask the owner for five decisions:
D4 placement, D6 hosted models and privacy, D10 timing, D11 Ethernet plus the disk floor, and the new RPO decision (finding 5).

### 17. Do not tie AIZK's move to the catalog migration

Evidence: U16 needs M37 and U17 needs M32 and M37; L2 waits for "the center's PostgreSQL ... control and catalog authority for 7 days".
AIZK's `aizk` and `logto` databases are independent of the `mainboard` database; they need M36 (service) and the dump mechanism. Twelve of the
131 touched paths do not exist yet (`center/postgres.py`, `control/coordinator.py`, `state/ingestion.py`, `memory/*`), so U11 to U20
edit files that M-packages must first create, and a slip in M37 delays AIZK for no technical reason.
Change: require only M36 (and M38's dump) for U16 and U17; soak the cluster for 7 days under staged AIZK load instead of waiting
for M37 to flip control authority.

## What I checked and found sound

- Three databases, not schemas: confirmed (`public`, 42 policies and `pgrls.toml` are hard-coded; extensions are per database; M38's snapshot stays short).
- AIZK stays a separate package and repository (D1), and mainboard never imports it except the client mount.
- Placement on the center as one unit (D4): the latency measurements are consistent, and splitting adds 0.24 s (+85%) to every keep.
- pgvector HNSW with `hnsw.iterative_scan = relaxed_order` is already in U03, which covers the post-filter-under-RLS problem; the HNSW
  numbers (75 s, 260 MB per 100,000 vectors) come from a real throwaway cluster.
- No external pooler is needed (18 connections in use).
- Native OpenRouter reranking via the existing `rerank_request_format = "openrouter"`, with fall-through to the fused order, is the right size.
- The U09 queue-churn fix (one dispatch task per tick, `pgqueuer_log` retention) is correct and small.

## Proposed shape after the cuts

| | before | after |
|---|---|---|
| packages | 22 | about 13: U00 fixture, U01 golden set, U03 HNSW, U04 lexical, U05 Cockroach and dead-code deletion (first), U06 local object store, U08 hosted lanes (clients, probe, fall-through, one re-embed path), U09 worker memory and queue, U11 layout plus daily dumps, U13 services with the usage pull and doctor, U14 `mb memory` mount, U16 W-1 spikes and rehearsal, U18 decommission |
| removed | | U02 (own track), U07 spool (OpenTelemetry removal folds into U05), U10, U12 PITR, U15, U17 (runbook), U19, U20, U21 |
| live events on crimson | L-1, L0, L0.5, L1, L2, L3 | L-1 baseline (read-only), the L2 freeze, L3 decommission; L1 stages the center only |
| runtime pieces on the center | PostgreSQL, 3 AIZK processes, Caddy, frontend, Logto, cloudflared, oauth2-proxy, Docker VM, spool, journal | PostgreSQL, 3 AIZK processes, Caddy, frontend, Logto, cloudflared: all native, no Docker |
| AIZK code paths | vendor and pgvector, bm25 and FTS, Cockroach, hosted and local models | pgvector and FTS, hosted models |

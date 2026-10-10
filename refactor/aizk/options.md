# One DuckLake and one PostgreSQL: the options, with a recommendation each

Step 2 of the AIZK-into-mainboard plan, 2026-10-10. It builds on `state.md` (same directory) and on mainboard
design v3 (one center PostgreSQL holding the DuckLake catalog and a `control` schema, packages M36 to M38).
What is measured is marked as such; what is estimated says so. Raw measurements are in `evidence/`.

## Recommendations at a glance

| question | recommendation | what would change it |
|---|---|---|
| 1a. databases or schemas | One cluster, three databases: `mainboard` (schemas `control`, `ducklake_<LakeId>`), `aizk` (AIZK's `public`, unchanged), `logto`. Roles per database. | A need for cross-database SQL joins (none today). |
| 1b. extensions | Run AIZK on extensions conda-forge carries on every platform: pgvector 0.8.6 HNSW plus a native lexical lane (tsvector over application-side tokens, pg_trgm). Gate it on the golden set. Fallback: build VectorChord, vchord_bm25 and pg_tokenizer from source into a directory outside the environment. Last resort: keep AIZK's database on Linux. | The golden set shows a recall or MRR loss beyond the declared tolerance. |
| 1c. backups, pooling, upgrades | Native tools only: `pg_basebackup` (incremental), WAL archiving, per-database `pg_dump`; copy offsite to crimson's empty 1.7 TB SSD; no external pooler; minor upgrades by restart on a retained binary generation, major by dump and restore. | A pooler becomes necessary above about 100 connections. |
| 2. placement | The center Mac mini hosts the server and AIZK's whole application tier. crimson stops serving AIZK: it keeps the old stack stopped for rollback, then becomes the offsite backup target and returns GPU 0 to the lease pool. | Wired Ethernet cannot be had, or the 24 GB budget does not hold after the worker is slimmed (section 4). |
| 3. DuckLake against PostgreSQL | PostgreSQL keeps everything transactional (queue, graph, vectors, RLS, usage serving, watermarks). DuckLake gets append-only mirrors and new streams: usage ledger copy, AIZK structured logs and spans, evaluation receipts. Source blobs go to a content-addressed directory, not into DuckLake. | A need to query usage across scopes at volume (not today: 5,572 rows). |
| 4. slimming | Drop Docling, GLiNER, SeaweedFS, the five observability containers and (default) ClamAV as services; move embeddings and reranking to OpenRouter behind the golden set. Result: about 6 to 8 GiB resident on the center, no GPU. | The golden set rejects hosted models, or privacy rules forbid sending note text out. |
| 5. packaging | Keep AIZK a separate package and repository. Mainboard owns the shared server, service rendering and a thin `mb memory` front that mounts AIZK's existing client. Revisit a merge only on measured triggers. | Paired commits across the two repositories dominate the log for a quarter. |

## 1. One PostgreSQL server for the catalog, the control schema and AIZK

### 1a. Databases or schemas

Design v3 already fixes the shape of mainboard's side: database `mainboard`, schema `control`, one
`ducklake_<LakeId>` schema per registered lake, `pg_dump` of "the entire mainboard database" in M38. DuckLake's
documentation names PostgreSQL as the catalog for "a multi-user lakehouse with potentially remote clients" and
requires PostgreSQL 12 or newer; its `METADATA_SCHEMA` parameter places the catalog in a schema of an existing
database. Measured on the center with the workspace environment's PostgreSQL 18.6 and DuckDB 2.0.0.dev: a
DuckLake attached with `ducklake:postgres:dbname=mainboard ...` and `METADATA_SCHEMA 'ducklake_main'` created
30 catalog tables in that schema beside `control` and `memory` schemas, committed single-row inserts in 2.3 ms
(inlined into the catalog) and wrote bulk inserts as Parquet.

AIZK cannot simply become more schemas of `mainboard`:

- AIZK hard-codes `public`: Alembic's `env.py`, 42 RLS policies, `pgrls.toml` (`schemas = ["public"]`), the
  `search_path` (`"$user", public, bm25_catalog, tokenizer_catalog`), 272 extension functions installed into
  `public`. A move into a named schema would touch every migration and policy for no gain.
- M38's consistent dump holds an exported snapshot open for the whole `pg_dump`; AIZK's 2.7 GB, whose largest tables
  took 35 s each to `COPY` on crimson, would stretch that window from seconds to minutes. Snapshots cannot span
  databases anyway, so separate databases give M38 a short snapshot for the control and catalog schemas and give AIZK
  its own dump with its own cadence.
- Extensions are per database. Keeping `vector` (and, if the fallback is ever needed, `vchord`) out of `mainboard`
  means the catalog database depends on contrib only, and DuckDB's PostgreSQL scanner never sees `halfvec` types.
- Logto expects its own database and migrations; it already has one (`logto`, 25 MB).

Recommendation: one cluster, three databases, with roles that cannot cross over.

| database | owner role | runtime roles | contents |
|---|---|---|---|
| `mainboard` | `mainboard` | `mainboard_rw` (control), `ducklake_writer`, `ducklake_reader` | `control`, `ducklake_<LakeId>` |
| `aizk` | `aizk_admin` (no superuser) | `aizk_app` (NOBYPASSRLS, forced RLS) | AIZK's `public`; pgvector, pg_trgm, pgcrypto |
| `logto` | `logto` | `logto_tenant_*` | Logto's tables |

Today `aizk_admin` and `aizk` are superusers; on the shared server they must not be (state finding F7), because a
superuser in `aizk` can read `mainboard`. `pg_trgm` and `pgcrypto` are trusted extensions (their control files say
`trusted = true`), but pgvector's control file is not, so the bootstrap role that owns the cluster creates `vector` in
`aizk` once and `aizk_admin` stays unprivileged. Role grants follow DuckLake's access-control guide: the writer
needs INSERT/UPDATE/DELETE/SELECT on the catalog schema, the reader SELECT.

### 1b. Resource isolation and pooling

PostgreSQL has no per-database memory or CPU control, and macOS has no cgroups, so isolation is by role settings:
`statement_timeout`, `lock_timeout`, `idle_in_transaction_session_timeout` and `CONNECTION LIMIT` per role. Proposed
starting point: `max_connections` 120; `aizk_app` 40, `aizk_admin` 10, `mainboard_rw` 20, `ducklake_writer` 20, `logto*`
10. AIZK's live statement timeout is 10 s with `lock_timeout` 1.5 s; the control role gets 5 s; DuckLake writers 30 s.
Live connection count today is 18 of 100, so no external pooler is needed: none is on conda-forge anyway (pgbouncer
and pgpool2 return 404), the asyncpg and psycopg pools suffice, and ownership connections must be unpooled
(`NullPool`) by M19's own rule. Settings for the Mac (24 GB, shared with agents and apps): `shared_buffers` 2 GB,
`effective_cache_size` 8 GB, `maintenance_work_mem` 1 GB, `max_wal_size` 4 GB, `wal_compression` zstd,
`default_toast_compression` lz4 as AIZK has them today; a restore must be run with `PGOPTIONS` raising
`maintenance_work_mem` (measured: the same restore took 192.8 s at the 64 MB default and 117.7 s tuned, because the
HNSW build spills).

### 1c. Extensions: exactly what exists today

Read from the Anaconda API and GitHub on 2026-10-10:

| item | version | osx-arm64 | linux-64 | linux-aarch64 |
|---|---|---|---|---|
| conda-forge `postgresql` (with contrib) | 18.6 (build 3, 2026-10-01) | yes | yes | yes |
| conda-forge `libpq` | 18.6 | yes | yes | yes |
| conda-forge `pgvector` | 0.8.6 (2026-08-20), depends on libpq 18.6 | yes | yes | yes |
| conda-forge `psycopg` / `psycopg-c` | 3.3.4 / 3.3.6 | yes | yes | yes |
| conda-forge `postgis` | 3.6.4 | yes | yes | yes |
| conda-forge `vectorchord`, `vchord`, `vchord_bm25`, `pg_tokenizer`, `pg_cron`, `pgbouncer`, `pgbackrest`, `wal-g`, `pgaudit`, `timescaledb`, `pgroonga` | none (HTTP 404 on each) | no | no | no |
| VectorChord upstream | 1.1.1 (2026-02-28, the version live on crimson) | no macOS asset | `.deb` and `.zip` for PostgreSQL 14 to 18 | same |
| vchord_bm25 upstream | 0.3.0 (2025-12-15) | no macOS asset | linux assets only | |
| pg_tokenizer.rs upstream | 0.1.1 (2025-09-29) | no macOS asset | linux assets only | |

The workspace environment already carries `postgresql` 18.6 and `libpq` 18.6 for osx-arm64 (with `pg_trgm`,
`pgcrypto`, `pg_stat_statements` and every client and backup tool); it does not carry pgvector. VectorChord's own
installation page lists Docker, Debian and Ubuntu packages and "Source" (cargo-pgrx 0.17.0); it is dual-licensed
AGPLv3 or ELv2. PostgreSQL 18's `extension_control_path` and `dynamic_library_path` let an extension live outside
the environment prefix. Measured: the conda-forge pgvector 0.8.6 osx-arm64 package, unpacked to a scratch
directory, loaded into the environment's server with those two settings (`create extension vector` returned 0.8.6).
A source-built VectorChord would sit in the same kind of directory, but nobody has built it on macOS here.

What AIZK needs from them: the dense lane (vchordrq, 763 MB of indexes on `halfvec(1024)`), the lexical lane (a
460 MB `bm25` index over `llmlingua2` tokens, which is what makes Japanese work), and the tokenizer. Three tiers:

1. Extension-free (recommended first). pgvector HNSW for vectors: measured on the center, 100,000 random
   `halfvec(1024)` rows build in 75 s (260 MB, so about 650 MB for AIZK's 251,821 vectors) and answer top-50 in 2.5
   to 7 ms. The live `dense_fact_content` statement averages 2.6 s per call on crimson, so index speed is not what
   makes find slow. For the lexical lane AIZK's CockroachDB path already contains a `to_tsvector`/`ts_rank` branch, but
   PostgreSQL's built-in text search does not segment Japanese (measured: `to_tsvector('simple', ...)` turns a
   Japanese sentence into one or two tokens); `pg_trgm` works (`word_similarity('基準価額', '投資信託の基準価額を確認する')` =
   0.4 against 0 for an unrelated sentence). The extension-free lexical lane therefore tokenizes in the application
   (the same `llmlingua2` tokenizer through `tokenizers`) into the existing `lexical` column, indexes it with a `simple`
   tsvector GIN index and uses `pg_trgm` for substring and title matches. It is not BM25 (no IDF), so it is gated
   on measured recall and MRR (SERVE phase 0's golden set), not on opinion.
2. Source-built extensions on macOS. A mainboard-managed Rust toolchain builds the three pgrx extensions once per
   PostgreSQL minor into `<center-state>/services/postgresql/ext/<pg>/`; M36's environment-owned service gains
   `extension_control_path`. Cost: a permanent build burden and an unvalidated macOS build. Use only if tier 1 fails
   its gate.
3. Keep the `aizk` database on a Linux PostgreSQL (crimson). Rejected as the default: it breaks "one server" and puts
   every AIZK statement across a 12 ms jump-host path (section 2).

Side benefit of tier 1: M00's temporary-PostgreSQL fixture, built from environment binaries, can then run AIZK's whole
test suite (`aizk_test`) on macOS and Linux. Today AIZK's migrations require the three vendor extensions, so its
suite cannot run on any PostgreSQL the workspace environment can start.

### 1d. Backups, point-in-time recovery, upgrades

Today AIZK has one daily `pg_dump --format=custom` on the same host, no WAL archiving (`archive_mode` off), and a
backup directory the crimson login cannot read. For the shared server:

- Base and WAL: `pg_basebackup` weekly and incrementally in between (PostgreSQL 17 and later; `pg_combinebackup` and
  `pg_walsummary` are in the environment), `archive_command` copying WAL segments to
  `<center-state>/backups/pg/wal` on the internal disk, with a hard cap and a doctor row so a stalled copy cannot fill the
  disk (28 GiB free, 94% used). Recovery to a time uses `recovery_target_time`.
- Logical: `pg_dump --format=custom` per database (`aizk`, `logto`, `mainboard`) and `pg_dumpall --globals-only`; the
  `aizk` dump keeps AIZK's existing portable format and is what crosses architectures.
- DuckLake's backup guide: a PostgreSQL catalog can be backed up by SQL dump or by PITR, and compaction and cleanup
  belong before manual backups. M38 owns the lake-consistent case for `mainboard`; the new work is covering the other two
  databases and shipping everything offsite.
- Offsite: crimson has two empty 1.8 TB SSDs (`/mnt/ssd1`, `/mnt/ssd2`). Through the jump host the path carries 10.8 to
  14 MB/s (measured), so a 1.5 GB weekly base copy takes about 2 minutes and a day of WAL seconds. `/Volumes/PORTABLE`
  is the second copy.
- Upgrades: minor releases by restart on a retained binary generation (M36); a major (PostgreSQL 19) by dump and restore,
  which for AIZK's 2.7 GB is under 15 minutes (measured restore throughput above); `pg_upgrade --link` is available.

## 2. Where it runs

Measured from the center (Mac mini, home LAN, Wi-Fi) to crimson (lab, through the `cvlab` jump host):

| quantity | value |
|---|---|
| application-level round trip (100 echoes over ssh) | p50 11.5 to 12.0 ms, p90 14.6 to 16.4 ms, p99 94 to 101 ms |
| bulk through ssh | 14 MB/s down from crimson, 10.8 MB/s up |
| home LAN hop (Mac to router, Wi-Fi) | 30 pings, 0% loss, 2.4 ms min, 10.7 ms mean, 74 ms max, stddev 21 ms |
| statements in one AIZK find | about 8 to 10 (measured by a `pg_stat_statements` difference); keep about 15 to 25 (inferred) |
| server-side find / keep | p50 4,954 ms / 280 ms |

| placement | find | keep | job hosts and the lake | verdict |
|---|---|---|---|---|
| A. Everything on the center: shared PostgreSQL, AIZK server, worker, Logto, cloudflared, no local models | no change in database time; about the same hairpin through Cloudflare as today | faster than today | hosts keep writing files; the center ingests; lake data path stays on the external volume | recommended |
| B. PostgreSQL on the center, AIZK application on crimson | +8 round trips: +0.1 s (+2%) | +20 round trips: +0.24 s (+85%) | unchanged | rejected: the worker also issues 1.3 statements a second (5.27 M dequeue calls in 48 days), and an outage of the Wi-Fi hop or the jump host would take AIZK down while both machines are up |
| C. Shared PostgreSQL on crimson | control and DuckLake catalog statements cross the jump host (settler passes, `list`, `show`, DuckLake commits) | unchanged | M19's advisory-lock ownership assumes a prompt session death; over an ssh jump and Wi-Fi it is not prompt | rejected; also contradicts the owner decision that the server runs on the center |
| D. Two servers (AIZK's stays on crimson) | unchanged | unchanged | two backup chains, two upgrade paths | not unification |

Job hosts are unaffected by the placement: design v3 has hosts write files and the center ingest them, and agents on any
host reach memory through the public MCP route, which is the same Cloudflare path in A as today. A moves the tunnel
connector from crimson to the center; the hostnames stay.

Conditions for A, and what they cost:

- The center is already the single point everything depends on; A adds AIZK to it. Its disk is 94% full (28 GiB free),
  so PostgreSQL's footprint (about 6 GB with indexes and WAL after the move) is the binding constraint, not RAM. The
  PostgreSQL data directory stays on the internal disk (design v3), never on the removable `/Volumes/PORTABLE`.
- The Mac is on Wi-Fi with 74 ms spikes to its own router and a history of 50 to 80% packet loss (2026-10-06 notes).
  Ethernet (`en0` exists, unused) should be used before the move; a cloudflared outage then means agents cannot reach
  memory, so `mb memory keep` must spool and replay when unreachable (package U12).
- GPU 0 of crimson is freed only if the model lanes leave (section 4); otherwise AIZK would keep two vLLM lanes on a
  card that design v3's universal lease says every user must enroll in.

The Cloudflare route flips atomically by giving the center its own tunnel and switching the hostnames' routes
(`aizk`, `auth`, `admin`, `console`); rollback is switching them back. Two connectors on one tunnel would not work
because the tunnel's remotely managed ingress names container hostnames (`caddy`, `logto`) that exist only on crimson.

## 3. What of AIZK's data belongs where

| data | size today | transactional need | verdict |
|---|---|---|---|
| chunks, documents, entities, facts, claims, profiles, communities, ontology, `live_fact` | 2.4 GB incl. indexes | RLS on 16 tables, vectors, bitemporal updates, the single find statement | PostgreSQL, unchanged |
| queue (`pgqueuer`), schedules, watermarks | 47 MB + 62 MB | row locks, `LISTEN/NOTIFY`, transactional enqueue with the write | PostgreSQL |
| `usage_event` | 5,572 rows, 2.5 MB | serving reads by `StatusReport` and quotas under RLS; enqueue is transactional with the operation | serve from PostgreSQL; append a copy to DuckLake (`memory_usage`) for cross-system analysis (cost per find against OpenRouter spend and GPU lease hours) |
| `pgqueuer_log` | 777,695 rows, 427 MB, +60 MB a day | none: it is a run log that nothing aggregates (`pgqueuer_statistics` is empty) | do not migrate it; stop the bloat at the source (one fan-out task per minute instead of one per scope, prune or aggregate the log) |
| `operator_snapshot`, `monthly_quota_counter`, `coordination_lock` | tiny | RLS / transactions | PostgreSQL |
| structured logs, spans, metrics (Tempo 553 MB, VictoriaMetrics 249 MB, Loki about 0, Grafana 118 MB) | 1.0 GB | none | DuckLake streams `memory_events`; AIZK writes JSONL, the center ingests like receipts; the five containers go |
| evaluation results (golden set, ablations) | new | none | DuckLake receipts via `mb` jobs (SERVE phase 0 already says so) |
| source blobs (artifact originals) | 659 MB stored, 847 MB logical | content-addressed, immutable, referenced by `blob` rows | a content-addressed directory (obstore `LocalStore`), later mainboard's own object store (UNIFY.md section 5); not DuckLake tables |
| Logto tenant data | 25 MB | its own | its own database on the same cluster |

DuckLake has no row-level security, so nothing that depends on RLS may be served from it. The mirror direction is
one-way: AIZK writes files, the settler appends them (idempotent on event id), nothing in AIZK imports DuckDB. That
keeps the public AIZK package free of a DuckDB 2.0 development dependency.

## 4. Slimming and the RAM budget

Measured RAM on crimson today is 25.4 GiB (SERVE.md said about 32). Proposed disposition:

| component | today | proposal | resident freed on the center |
|---|---|---|---|
| Docling service | 0.83 GiB | delete; HEAD already routes conversion through LiteParse, Pandoc, text, image lanes with Docling as a one-shot CLI | 0.8 |
| GLiNER (gate and mention detector) | 3.77 GiB, 2.6 GB VRAM | extraction gate off (extraction costs pennies); the mention detector only serves the web-egress sanitizer, and the ledger shows 0 `web_search`/`web_fetch` events ever, so load it lazily on CPU only if web search is enabled | 3.8 |
| ClamAV | 1.04 GiB | not on conda-forge; either run it on demand, keep one container, or drop scanning for a single-owner deployment (owner decision) | 1.0 |
| SeaweedFS | 0.37 GiB | obstore `LocalStore` on a directory; `download_url` has no caller | 0.4 |
| observability (5 containers) | 1.83 GiB | drop; usage ledger, `operator_snapshot` and JSONL cover it | 1.8 |
| vLLM embedder (Qwen3-VL-Embedding-2B) | 3.67 GiB + 7.6 GB VRAM | OpenRouter `qwen/qwen3-embedding-8b`, $0.01 per million tokens, truncated to 1,024 dimensions; a full re-embed of about 23 M tokens costs about $0.25 | 3.7 |
| vLLM reranker (Qwen3-Reranker-4B) | 2.69 GiB + 10 GB VRAM | OpenRouter `qwen/qwen3-reranker-8b`, $0.20 per million tokens (released 2026-08-13); AIZK already has `rerank_request_format = "openrouter"`; about 26k tokens a find is about $0.005, so roughly $4 a month at today's load | 2.7 |
| worker | 5.62 GiB (one Python process) | investigate; target 2 GiB | 3.6 if reached |
| PostgreSQL | 4.44 GiB | 2 GB `shared_buffers` | about 1.4 |

Estimated resident set on the center after slimming (native processes, no GPU):

| item | GiB |
|---|---|
| PostgreSQL (2 GB shared buffers, about 25 backends) | 3.0 |
| MCP server 0.43, API 0.29, worker 2.0 (target) | 2.7 |
| Logto (container 0.33; native Node unverified) | 0.3 |
| Caddy 0.06, cloudflared 0.05, frontend 0.09 | 0.2 |
| Docker Desktop VM if any container remains (inferred) | 1.0 to 1.5 |
| total | 6.2 to 7.7 |

The target SERVE.md set was "under 10 GB". If the worker stays at 5.6 GiB the total is 9.8 to 11.3 GiB. The center's
free plus reclaimable memory measured now is about 13 GB with the owner's apps open (5.7 GB free, 7.3 GB inactive),
so the budget holds with about 5 GB of headroom only if the worker is slimmed; the swing factor is the worker, not the
models. Local models would add: MLX embedder (Qwen3-Embedding 0.6B or 4B 4-bit) 1.2 to 2.5 GB, and a reranker.
Estimate from FLOPs (not measured): 50 documents of about 450 tokens through a 4B reranker is about 200 TFLOP, tens of
seconds on an M4 Pro against about 1 to 2 s on the 3090, so a local reranker on the Mac is only plausible at 0.6B and
depth 20. If the owner insists on local models, the incoming Mac Studio (512 GB, ownership unsettled) or crimson's GPU 0
behind an authenticated tunnel are the places, and the lanes are already URL-configured, so that is a configuration
change, not a code change.

Gates before any model change: the golden set (recall@k, MRR) on a copy of the corpus; parity of OpenRouter Qwen3
embeddings truncated to 1,024 dimensions (check that the provider honors `dimensions`); find p50 not worse than the
current 5 s; privacy decision (note text already goes to DeepSeek for extraction under zero-retention terms).

## 5. How AIZK becomes mb memory

| option | for | against |
|---|---|---|
| A. Separate package on the shared server, mainboard supervises and fronts it (recommended) | AIZK is a public Apache-2.0 product (PyPI, plugin marketplace, docs, 40k test lines) and "no longer depends on mainboard"; mainboard ships light to every host; the two packages already share `patos`, `rls`, `sqlalchemy` as workspace members; only a contract is needed | two repositories to keep in step |
| B. `mainboard[serve]` extra containing AIZK | one install | pulls fastmcp, FastAPI, sqlmodel, pgqueuer into the package that ships to Miyabi's 14 GB-capped login node and every job host; makes the public product depend on a private tool |
| C. Merged repository | one history | AIZK's public identity (README, plugins, releases) breaks; mainboard's "release at the end" cadence conflicts with AIZK's; no technical coupling needs it |

The contract in A:

- Mainboard owns the PostgreSQL service (M36), the `aizk` and `logto` databases and roles (new U-package), backups
  (M38 extended), the launchd units for AIZK's three processes (`aizk admin server mcp|api|worker`), Caddy and the
  container-only pieces, and `mb doctor` rows replacing `[gates.aizk]`.
- AIZK owns its schema and Alembic history inside `aizk`, its MCP and HTTP surfaces, and writes append-only JSONL into a
  spool directory. It never imports mainboard or DuckDB.
- The user surface is `mb memory find|keep|status|share|report|login`, mounted from AIZK's existing client (`aizk.client`
  already implements OAuth login, upload, keep and find over MCP, 462 lines), through a `mainboard[memory]` extra that
  depends on an `aizk[client]` extra (fastmcp, httpx, keyring only). Do not write a second client.
- Service control lives under `mb memory up|down|status|logs|migrate`. The name `mb serve` is unusable: `mainboard host
  serve` is the node-side job claimer (`serve.py`) and the lake already has a `serve` verb.
- Login: Logto 1.41 supports the OAuth device flow (added in 1.38); `mb memory login --device` for headless hosts,
  the existing browser flow on the center, and a machine-to-machine client with the `control` scope for the settler and
  gates.

Triggers to revisit a merge: more than half of AIZK commits for a quarter also change mainboard, or mainboard code needs
to import AIZK models rather than call its client.

## 6. Evidence still missing, to be produced by the first packages

- The golden set and baseline metrics of the current stack (does not exist yet; SERVE phase 0, item 1).
- Why the worker holds 5.6 GiB, and the per-command statement counts for keep.
- Whether OpenRouter's embedding endpoint honors `dimensions=1024` and its rerank latency for 50 documents.
- Whether Logto 1.41 runs natively on conda-forge Node 26 (the repository's `nodejs` pin) or must stay a container.
- Whether `~/.aizk/backups` holds restorable dumps (unreadable by the crimson login).
- The wall time of a full AIZK restore on the center with the extension-free indexes (extrapolated here at 6 to 8
  minutes tuned from a 100k-row measurement).

## Sources

DuckLake: choosing a catalog database, access control, backup and recovery, data inlining, transactions
(`ducklake.select/docs/stable/duckdb/...`). VectorChord installation page and GitHub releases
(`docs.vectorchord.ai`, `github.com/tensorchord/VectorChord`, `supervc-stack/VectorChord-bm25`, `.../pg_tokenizer.rs`).
Anaconda API for conda-forge `postgresql`, `pgvector`, `libpq`, `psycopg` and the 404s. Logto device flow
(`docs.logto.io/quick-starts/device-flow`) and the March 2026 changelog (v1.38.0). OpenRouter model pages for
`qwen/qwen3-embedding-8b` and `qwen/qwen3-reranker-8b`. PostgreSQL 18 behavior verified directly on the center.

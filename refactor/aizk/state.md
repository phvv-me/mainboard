# AIZK as it is, 2026-10-10

Step 1 of the AIZK-into-mainboard plan. Everything here was read from the repository at
packages/aizk HEAD `6ebeab1` (2026-10-10, a dependency bump on top of `be29fcb` of 2026-09-25), from
the live stack on crimson with read-only commands (`docker ps`, `docker stats --no-stream`,
`docker system df`, `nvidia-smi`, `df`, read-only `psql` inside `aizk-db-1`), and from measurements
taken on the center. The earlier Codex runs died when their credits ran out; their partial files in
`evidence/` were reused where they were sound (table counts, index sizes, extension list, conda-forge
and VectorChord release data, docker stats) and re-measured where they were empty (ssh round trips).
Numbers that are inferred rather than read are marked "inferred".

## 1. What AIZK is

A self-hosted memory engine for one owner and a few collaborators. Agents reach it over MCP at
`https://aizk.phvv.me/mcp` (Claude Code, Codex and opencode all use it; the `aizk` skill in
`.agents/skills/aizk` is the usage contract). It stores notes and files as scoped documents, chunks
them, embeds them, extracts a bitemporal knowledge graph with an LLM, and answers `find` with one SQL
statement over several lanes followed by a cross-encoder rerank. Identity is Logto only (no user table).
Row-level security in PostgreSQL separates private memory from organization memory.

Load (usage ledger, 5,572 rows from 2026-07-19 to now): lifetime 2,849 `keep` of text (`remember_text`), 2,445
`find` (`recall`), 257 file keeps (last 2026-09-10), 15 artifact reads, 6 `share`. The last 30 days hold 1,427
events: 730 finds, 689 text keeps, 6 artifact reads, no file keep (SERVE.md's 836/967/73 counted differently).
Typical days see 10 to 60 calls (roughly 1 keep an hour), the peak was 275 finds on 2026-10-05. Server-side
latency over those 30 days:

| operation | n | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| find | 730 | 4,954 ms | 6,036 ms | 7,981 ms | 11,150 ms |
| keep (text) | 689 | 280 ms | 352 ms | 476 ms | 904 ms |
| artifact read | 6 | 20 ms | 35 ms | 35 ms | 35 ms |

Anatomy of one `find` (measured as the difference of `pg_stat_statements` before and after a single call,
background ticks removed): about 8 to 10 SQL statements, all in one transaction: `BEGIN`, two `set_config`
calls that bind the caller's scopes, the lane union (`dense_fact_content ... MATERIALIZED`, mean 2.6 s over
1,338 calls), one `UPDATE fact_claim` for access counts, `COMMIT`, a `usage_event` insert, a queue
enqueue and a `pg_notify`. The database is therefore the dominant cost of a find (about half of 5 s), the
embedder and reranker the rest. Round-trip latency to the database matters little for find (8 statements)
and noticeably for keep (inferred 15 to 25 statements).

## 2. Code at HEAD

Python 3.14, one package `aizk` (35,043 lines, 235 files) plus `eval` (3,978 lines, benchmarks and
GroupMemBench), `services/gliner` (77 lines), a SvelteKit app `src/web` (6,132 TypeScript lines),
a docs site (148 content files), and `infra/aws` plus `hackathon/` (the CockroachDB and AWS profile). Tests:
138 files, 40,142 lines, 1,188 test functions; default run excludes `integration` and `benchmark` markers and
needs a real PostgreSQL (`aizk_test`, default port 5433) with the vchord, vchord_bm25 and pg_tokenizer
extensions because migration `0001_init` creates them.

| responsibility | path | lines |
|---|---|---|
| storage layer: SQLModel tables, `live_fact` view, RLS DDL, identity (`User`), Alembic (16 revisions, 4 CockroachDB revisions, `scoped_cspann` 443), engine and `backend.py` (DatabaseBackend), quota, locking | `store/` | 8,422 |
| knowledge graph: build, writer, grounding, communities, RAPTOR, profiles, insight, promote, decay, dedupe, repair, reembed, session tier | `graph/` | 3,396 |
| integrations: Logto client and policy (1,325), web search providers (527+), artifact converter (299), Docling client, ClamAV client | `integrations/` | 3,089 |
| artifacts: intake service, repository, description, boilerplate cleaner, uploads, formats | `artifacts/` | 2,976 |
| operations: `doctor` (1,125), probes, provisioning, hardware, reports, snapshots | `ops/` | 2,493 |
| background: PgQueuer wrapper (455), CockroachDB portable worker (230), maintenance/conversion/projection jobs, schedule | `background/` | 1,819 |
| retrieval: find orchestrator and program, six lane kinds, rerank, packing, models | `retrieval/` | 1,697 |
| browser API (FastAPI, about 25 routes) | `api/` | 1,291 |
| CLI (`aizk admin ...`, client commands, AWS commands) | `commands/` | 1,156 |
| extraction: ingest, extractor, declaration | `extract/` | 1,144 |
| model clients: embed, rerank, gate (GLiNER), LLM extract, chunker | `serving/` | 1,092 |
| web search with egress sanitizer | `web/` | 1,114 |
| MCP server (5 tools + 1 resource), middleware, runtime | `mcp/` | 853 |
| top level: memory 398, status 359, admin 319, usage 302, storage 244, backup 192, auth 123 | `*.py` | ~2,100 |
| settings (1,154), client library (462), ontology (397) | `config/ client/ ontology/` | 2,000 |

Dependency notes that matter for unification: `sqlalchemy==2.1.0b3` (the workspace overrides it to the local
RLS fork `packages/sqlalchemy`), `rlsalchemy` (house package `packages/rls`), `patos[sql]` (local),
`asyncpg` 0.32 and async SQLAlchemy, `pgqueuer` 1.x, `fastmcp` 4.1, `obstore`, `liteparse`, `pypandoc-binary`,
`sqlalchemy-cockroachdb`, `opentelemetry-*` (5 packages), `boto3`/`mangum` (Lambda profile). AIZK is already
a workspace member of the monorepo (`mainboard.toml` `members`), installed editable in the mainboard
environment, and its repository is public (`github.com/phvv-me/aizk`, Apache-2.0, Claude and Codex plugin
marketplace entries, PyPI).

## 3. PostgreSQL

Server: PostgreSQL 18.3 (Debian, `tensorchord/vchord-suite:pg18-latest`) in `aizk-db-1`, up since
2026-09-24, data checksums on, `shared_buffers` 16 GB, `effective_cache_size` 128 GB, `work_mem` 16 MB,
`maintenance_work_mem` 2 GB, `wal_level` replica, `archive_mode` off (no point-in-time recovery), WAL
directory 1.5 GB, `max_connections` 100 with 18 in use (13 `aizk_app`, 5 `aizk_admin`), lz4 TOAST, zstd WAL
compression. Databases: `aizk` 2,756 MB, `aizk_bench` 79 MB, `logto` 25 MB, `aizk_extract_bench` 12 MB.
The docker volume `aizk_db-data` is 4.6 GB.

Extensions in `aizk`: vchord 1.1.1, vchord_bm25 0.3.0, pg_tokenizer 0.1.1, vector (pgvector) 0.8.2, pg_trgm 1.6,
pgcrypto 1.4, pg_stat_statements 1.12, plpgsql. Preloaded: `vchord,vchord_bm25,vector,pg_tokenizer,pg_stat_statements`.
The `public` schema holds 272 functions (VectorChord's), `tokenizer_catalog` 24, `bm25_catalog` 11.

Roles: `aizk` (superuser bootstrap), `aizk_admin` (superuser, owns the schema, bypasses RLS, used by the worker,
migrations and backups), `aizk_app` (NOSUPERUSER NOBYPASSRLS, every request path), and `logto*` roles for the
separate `logto` database.

Row-level security: 42 policies on 16 tables, all forced, all `TO aizk_app`, driven by transaction-local
settings `app.scopes` (a JSON object with `read`, `write`, `public` UUID arrays) and `app.operator`; no server
default, so an unbound session sees nothing. `live_fact` is a security-barrier view. The audit config is
`pgrls.toml` at the monorepo root (task `rls-report`).

Tables (rows and total bytes including indexes, from the live database):

| table | rows | total | notes |
|---|---|---|---|
| chunk | 33,268 | 788 MB | halfvec(1024) + `bm25vector`; `ix_chunk_bm25` 460 MB, vchordrq 100 MB |
| entity_content | 77,215 | 482 MB | halfvec(1024); vchordrq 228 MB |
| fact_content | 70,727 | 433 MB | halfvec(1024); vchordrq 205 MB |
| profile | 64,493 | 450 MB | halfvec(1024); vchordrq 198 MB |
| community | 7,004 | 77 MB | halfvec(1024); vchordrq 32 MB |
| entity_claim / fact_claim | 77,892 / 70,899 | 28 / 53 MB | bitemporal claims; partial unique index `uq_fact_claim_live` |
| watermark | 77,868 | 62 MB | per-scope processing counters |
| document / artifact / artifact_content / blob | 3,556 / 478 / 490 / 480 | 2 / 0.5 / 24 / 0.4 MB | blob rows point into the object store; 847 MB logical |
| usage_event | 5,566 | 2.5 MB | immutable ledger, RLS insert and read policies |
| pgqueuer / pgqueuer_log / pgqueuer_schedules | 79 / 777,695 / 17 | 47 / 427 / 0.6 MB | see section 5 |
| queue_task / queue_event / queue_schedule | 0 | 40 kB each | CockroachDB portable queue, unused on PostgreSQL |
| session_item, upload_capability, operator_snapshot, monthly_quota_counter | 0 / 0 / 3 / 0 | <0.5 MB | |

Public schema: tables 2,746 MB, indexes 1,687 MB. Vectors: 251,821 with an embedding (chunk 33,265, entity 77,203,
fact 69,856, profile 64,493, community 7,004), all `halfvec(1024)` (Matryoshka truncation of Qwen3-VL-Embedding-2B's
2048). Corpus text is about 90 M characters (chunks 56.4 M, profiles 26 M, facts 6.5 M, entity names 1.6 M),
roughly 23 M tokens, so a full re-embed through OpenRouter's `qwen/qwen3-embedding-8b` ($0.01/M) costs about
$0.25.

The lexical lane is `vchord_bm25` over `llmlingua2` tokens (multilingual, so Japanese works); the CockroachDB
code path already has a `to_tsvector('english', ...)`/`ts_rank` lexical lane, and `index_backend` (default
`vchordrq`) is a free setting in migration `0001_init`, so the vector index backend is already swappable.

## 4. Object storage, auth, surfaces, pipelines

**Object storage.** SeaweedFS 4.29 (`weed mini`) in `aizk-objects-1`, 366 MiB resident, volume 659 MB.
`ByteStore` (`storage.py`, 244 lines) wraps obstore's `S3Store`: zstd-compressed immutable blobs addressed by
content hash, `Blob` rows in PostgreSQL, a retirement table for obsolete layouts. `ByteStore.download_url`
(presigned URLs) has no caller in the tree. Uploads from agents do not use presigned S3: they go through an
`upload_capability` row and the API's upload route.

**Auth.** Self-hosted Logto 1.41 in `aizk-logto-1` at `https://auth.phvv.me`, database `logto` on the same
PostgreSQL. Bearer tokens are verified by `auth.py` (FastMCP `RemoteAuthProvider`, JWT verifier); organizations and
roles are read from Logto with a short fail-closed cache; `User.authorized` derives UUID5 identities. The operator
console is a second Caddy site gated by `oauth2-proxy` and the `aizk-admin` role. Logto 1.38 and later support the
OAuth device authorization grant, so `mb login` can use the device flow.
(`scripts/aizk-gate.py` still carries a stale Logto Cloud issuer `3ado57.logto.app`; the live issuer is `auth.phvv.me`.)

**MCP surface** (`mcp/server.py`): `status`, `find`, `keep`, `report`, `share`, plus the resource
`aizk://artifacts/{id}/contents/{content_id}`; modern-protocol only, per-caller rate limit. Agents call `find` and
`keep` almost exclusively (the ledger operations are `recall`, `remember_text`, `remember_file`, `share`, `artifact_read`,
`web_search`, `web_fetch`; the last two never occurred, so the public-web branch is off in this deployment);
`report` writes operator reports into the `aizk:reports` scope (168 between 2026-08-07 and 2026-10-06); `share`
copies or moves documents into an organization.

**HTTP surface** (`api/app.py`, FastAPI, about 25 routes): `/healthz`, `/api/me`, `/api/status`, `/api/overview`,
`/api/usage`, `/api/processing`, sources, findings, subjects, themes, graph, organizations, uploads, and the
admin set (`/api/admin/health`, `doctor`, `usage`, `links`). It serves the SvelteKit frontend; Caddy routes
`/mcp` to `server:8080`, uploads to `api:8010`, `/app` to the frontend and the rest to the docs container.

**Write pipeline.** `keep(text)` goes to `extract.ingest.ingest_text` (document, chunks, embeddings, one
transaction), then enqueues graph work; files go through `ArtifactIntake`: ClamAV scan, then conversion by a
`ConversionRouter` (LiteParse for simple PDFs, Pandoc, text, image lane, Docling as a one-shot CLI for the
rest). In the live stack Docling still runs as a service. Graph build: gate (GLiNER, threshold 0.7, optional) then LLM
extraction (OpenRouter `deepseek/deepseek-v4-flash-0731`) then grounding then consolidation into entities and bitemporal facts.

**Read pipeline.** `Memory.find` builds one cached statement from a `Plan`: dense chunk lane (vchordrq), lexical lane
(bm25), exact-title lane, fact lane (dense seeds, one-hop neighbours, personalized PageRank), entity catalog lane,
working memory, profiles, communities, RAPTOR overviews; unions them, reranks with Qwen3-Reranker-4B
(`rerank_depth` 50, document cap 1,408 tokens), packs to a token budget, renders Markdown with a privacy receipt.
The optional public-web branch rewrites the question and runs it through a sanitizer that uses a GLiNER mention detector.

**Background.** 17 PgQueuer schedules: per-minute `artifact_dispatch`, `chunk_dispatch`, `chunk_recovery`,
`profile_projection`; 5-minute `doctor_snapshot`, `health_snapshot`; 15-minute `session_promote`; 30-minute
`usage_snapshot`; nightly `cleanup` 01:00, `backup` 02:00 (one `pg_dump --format=custom`, 14-day retention,
same host), `decay` 03:00, `dedup` 03:30, `artifact_integrity` 06:00; weekly `communities`, `profile_refresh`,
`raptor`, `insight`. `raptor` last ran 2026-08-02 although it is weekly (finding F5).

**Tests and gates.** `mb run test-aizk` (four isolated test databases), `bench-aizk` and `bench-aizk-gate`
(statistical retrieval baseline on the frozen live corpus), `aizk-eval`, `rls-report`, the web build/check tasks, and
the doctor gate `[gates.aizk]` (`scripts/aizk-gate.py`: one real keep over the public MCP route, falling back to one
keep inside the server container over ssh).

## 5. The live deployment on crimson

crimson: Ubuntu 24.04, 48 cores, 251 GB RAM, 2 x RTX 3090, root disk 915 GB (590 GB used, 68%), two empty
1.8 TB SSDs at `/mnt/ssd1` and `/mnt/ssd2`. Reached from the center through the jump host `cvlab`
(`ProxyCommand ssh cvlab connect`), address 10.0.0.204. Project `aizk`, 21 containers up for 2 weeks,
controlled by `~/.aizk/compose` from `~/aizk` (a clone at `be29fcb`).

| container | RAM now | notes |
|---|---|---|
| db (PostgreSQL 18 + vchord) | 4.44 GiB | `aizk` plus `logto` databases |
| worker (`aizk admin server worker`) | 5.62 GiB | one Python process; largest after the database; cause unmeasured |
| vllm-emb (Qwen3-VL-Embedding-2B) | 3.67 GiB | GPU 0, 7.56 GB VRAM (inferred) |
| vllm-rerank (Qwen3-Reranker-4B) | 2.69 GiB | GPU 0, 10.0 GB VRAM (inferred) |
| gliner (gliner2-large-v1) | 3.77 GiB | GPU 0, 2.62 GB VRAM |
| clamav | 1.04 GiB | |
| docling (service) | 0.83 GiB | HEAD already routes conversion without it |
| victoriametrics / tempo / alloy / grafana / loki | 0.75 / 0.36 / 0.32 / 0.30 / 0.10 GiB | observability, 1.83 GiB together |
| server (MCP) / api / frontend / logto | 0.43 / 0.29 / 0.09 / 0.33 GiB | |
| objects (SeaweedFS) / caddy / cloudflared / oauth2-proxy / docs | 0.37 / 0.06 / 0.05 / 0.01 / 0.02 GiB | |
| total | 25.4 GiB | SERVE.md said about 32 GB; its 10.6 GB reranker figure was a transient |

GPU 0 holds 20.2 GB of 24.6 GB at 0% utilization; GPU 1 is free for registered work (a 300 MiB research
process was on it at measurement time). Docker volumes total 6.6 GB (db 4.64, object 0.66, tempo 0.55, metrics 0.25,
grafana 0.12, backups 0.20, clamav 0.18). Docker images 113 GB and build cache 289 GB (223 GB reclaimable).
Root disk was 90% full on 2026-10-06 and is 68% now.

Findings in the live state (to be fixed or consciously carried):

- F1. Five containers (db, tempo, loki, alloy, grafana) bind-mount files from the deleted `/home/crimson/projects/...`
  path, and the whole stack was built on 2026-09-17. They cannot restart cleanly until the stack is recreated from `~/aizk`
  (pending since 2026-10-01). Any plan that restarts the database or deploys a new release starts here.
- F2. There is no verified off-host backup and no point-in-time recovery. The daily dump lands in `~/.aizk/backups`
  on the same disk; that directory is root-owned and unreadable by the crimson login (`Permission denied`), so size, age
  and restorability could not be checked read-only. The named volume `aizk_backups` (197 MB) has no containers attached.
- F3. `pgqueuer_log` is 777,695 rows and 427 MB and grows by about 110k rows (60 MB) a day: the one-per-minute
  fan-out creates about 47,500 chunk and 47,500 artifact dispatch tasks a day, and `pgqueuer_statistics` has 0 rows, so
  nothing aggregates or prunes the log (`pgqueuer_log_not_aggregated` alone is 179 MB). The dequeue statement ran 5.27 M
  times in 48 days. This is the largest "append-only history" in the database and it is waste, not history.
- F4. All 79 rows left in `pgqueuer` are `failed` (56 graph builds, 10 insights, 10 profile projections, 3 communities).
- F5. The weekly `raptor` schedule last ran 2026-08-02.
- F6. `aizk_zitadel-machinekey` and `aizk_oauth` volumes remain from a rejected identity provider.
- F7. The database role `aizk_admin` is a superuser and `aizk` has BYPASSRLS; only `aizk_app` is constrained.

## 6. SERVE.md (2026-10-06): what has landed

Nothing from phases 0 to 4 has landed in AIZK. The only AIZK commit since the plan is `6ebeab1` (dependency bump
2026-10-10); the live stack still runs the 2026-09-17 build. The audit settings are unchanged
(`find_recency_half_life_days` 30, `find_recency_weight` 0.1, `Document.named_in_query` unchanged).

| phase | status |
|---|---|
| 0 retrieval fixes (golden set, title authority, document-ID read path, reports scope exclusion, supersession, overview cap, source prior) | not started; its deploy also needs F1 fixed |
| 1 slimming (GLiNER gate, embeddings, reranker, CockroachDB, observability, <10 GB) | not started; conversion routing without Docling exists in HEAD (pre-plan) |
| 2 one data layer (catalog into AIZK's PostgreSQL) | superseded by mainboard design v3: the catalog lives in a `mainboard` database of the center PostgreSQL, next to AIZK's databases; M36 to M38 not started (no `center/postgres.py`, no `control/` in the tree) |
| 3 one login and API (`mb login`, mainboard tools in MCP, structured DEV LOG) | not started |
| 4 app | not started |

Mainboard since 2026-10-06 (git log): Windows dropped, polars removed, DuckDB the one table engine, receipts keep text,
the lake moved to `/Volumes/PORTABLE/mainboard-lake`, evidence by reference, `mb git commit` selective, launchd settler
over `ssh localhost`; UNIFY.md steps 1 to 3 are partly done. Two facts changed since SERVE.md: OpenRouter now lists
`qwen/qwen3-reranker-8b` ($0.20/M tokens, 41K context, released 2026-08-13) and a 0.6B reranker beside
`qwen/qwen3-embedding-8b` ($0.01/M), so "OpenRouter lists no reranker" no longer holds; and AIZK's `rerank_request_format`
already has an `openrouter` value. The name `mb serve` is taken twice already: `mainboard host serve` is the node-side
job claimer (`serve.py`) and `mainboard lake ... serve` serves the lake.

## 7. Where mainboard and AIZK already touch

- The monorepo lists `packages/aizk` (and `packages/rls`, `packages/sqlalchemy`, `packages/patos`) as workspace
  members; AIZK's tests, benchmarks, web build and AWS tasks are `mainboard.toml` tasks (`test-aizk`, `bench-aizk`,
  `aizk-eval`, `rls-report`, `aizk-web-*`, `infra-*`, `ccloud*`).
- `[gates.aizk]` in `mainboard.toml`: `mainboard doctor` runs `scripts/aizk-gate.py` (MCP keep with a client-credentials
  token, ssh fallback into `aizk-server-1`).
- `[hosts.crimson]`: `CUDA_VISIBLE_DEVICES` is exported to GPU 1 only, with the comment "GPU 0 hosts AIZK", and
  `status-note = "GPU 1 only; AIZK owns GPU 0 and runs from ~/aizk."`
- Agents: `.mcp.json`, `.agents/mcp.json`, Codex and Claude configs point at `https://aizk.phvv.me/mcp`; the `aizk` and
  `zettelkasten` skills and AGENTS.md require `find` before answering from memory and a DEV LOG note per piece of work
  (`DEV LOG: ...` titles with `#project:` and `#area:` tags).
- `git/repo.py` recognises the `phvv-me/aizk` remote; `pgrls` (Linux-only pin in `[on.linux.python.dev]`) audits the
  database; GPU 0 of crimson is reserved for AIZK in the host registry.
- SERVE.md and UNIFY.md/UNIFY-evidence.md in `packages/mainboard`; the refactor's design v3 (`/Users/pedro/Developer/mb-refactor/refactor`)
  that M36 to M38 extend.

## 8. Measurements taken on the center for the next steps

- Round trip center to crimson over ssh (through the cvlab jump), 100 echo round trips on one connection:
  min 10.0 ms, p50 11.5 to 12.0 ms, p90 14.6 to 16.4 ms, p99 94 to 101 ms (a few slow packets), mean 15 to 17 ms. A fresh ssh
  command costs 0.51 to 0.65 s (handshake through the jump); a multiplexed command costs 50 to 60 ms.
  Bulk through ssh: crimson to center 200 MB in 14.2 s (14 MB/s), center to crimson in 18.5 s (10.8 MB/s).
- Center: Mac mini M4 Pro, 24 GB, internal data volume 460 GiB with 28 GiB free (94% used), `/Volumes/PORTABLE` 3.6 TB with
  3.5 TB free. The lake is `/Volumes/PORTABLE/mainboard-lake` (`lake.sqlite` 11.4 MB). The mainboard environment already carries
  conda-forge `postgresql` 18.6 and `libpq` 18.6 (osx-arm64 build `hf773c0a_3`) with contrib (`pg_trgm`, `pgcrypto`,
  `pg_stat_statements`); `initdb`, `pg_basebackup`, `pg_combinebackup`, `pg_dump` and the rest are in the environment's `bin`.
- A throwaway cluster from those binaries started and served; a DuckLake with a PostgreSQL catalog in schema
  `ducklake_main` of a database that also holds `control` and `memory` schemas worked with DuckDB 2.0.0.dev
  (single-row commit p50 2.3 ms); pgvector 0.8.6 from the conda-forge osx-arm64 package loaded through PostgreSQL 18's
  `extension_control_path` and `dynamic_library_path` from a directory outside the environment; 100,000 random
  `halfvec(1024)` rows built an HNSW index in 75 s (260 MB) and answered top-50 in 2.5 to 7 ms. Details in `options.md`.

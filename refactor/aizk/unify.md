# Unifying AIZK into mb memory: one PostgreSQL, one DuckLake, one login

Design of 2026-10-10, step 3 of the plan, extending mainboard design v3 (packages M00 to M38, revision
`c97cb81`). The evidence is in `state.md` and `options.md` beside this file; the machine-readable version is
`unify.json` (keys overview, layout, packages, order, migration, open_questions, excluded, critique). The two critic
stages and the revision of the original `plan.sh` did not run (the earlier Codex runs died when their credits ran
out); section 9 lists what a critic would still have to check.

## 1. The goal and the recommendation

AIZK keeps its public identity (its own repository, the five MCP tools, Logto login), but it runs on the same
system as mainboard. The center's single PostgreSQL cluster holds the DuckLake catalog, mainboard's control schema
and AIZK's transactional data, in separate databases. AIZK's append-only history and logs land in the same DuckLake the
evidence lives in. One CLI front (`mb memory`), one doctor, one backup chain. crimson stops serving AIZK, and its GPU 0
goes back to the lease pool.

- Layout: one cluster, three databases (`mainboard`, `aizk`, `logto`), roles that cannot cross databases, extensions
  limited to what conda-forge carries on macOS and Linux (contrib plus pgvector 0.8.6).
- Placement: the center Mac mini runs the cluster and AIZK's whole application tier as launchd services from the
  workspace environment; containers only for Logto and cloudflared, which have no conda-forge build.
- Slimming: no Docling, GLiNER, SeaweedFS, vLLM lanes or observability containers; embeddings and reranking on
  OpenRouter behind a golden-set gate; expected resident set 6.2 to 7.7 GiB against 25.4 GiB on crimson today.
- Packaging: AIZK stays a separate package and repository; mainboard supervises it and fronts it. Nothing is merged.
- Packages: U00 to U21, 22 in all, 3 of them live (U16, U17, U18), ordered beside the M waves.

## 2. What the plan stands on

- Measured round trip from the center to crimson over ssh through the `cvlab` jump: p50 11.5 to 12.0 ms, p99 about
  100 ms, 10.8 to 14 MB/s. One `find` is 8 to 10 SQL statements and takes 4.95 s at p50 (the database is about half);
  one `keep` takes 0.28 s. Splitting the database from the application tier would add 0.1 s to a find and 0.24 s (+85%)
  to a keep, and the worker alone issues 1.3 statements a second.
- The workspace environment already holds PostgreSQL 18.6. A throwaway cluster from it served a DuckLake catalog in a
  schema beside `control`, committed single rows in 2.3 ms, loaded conda-forge pgvector 0.8.6 through PostgreSQL 18's
  `extension_control_path` from outside the environment, built an HNSW index on 100,000 `halfvec(1024)` rows in 75 s
  (260 MB), and restored a 346 MiB dump in 117.7 s tuned (192.8 s at server defaults).
- VectorChord 1.1.1, vchord_bm25 and pg_tokenizer (what AIZK runs today) have no macOS binary and no conda-forge
  package. PostgreSQL's built-in text search does not segment Japanese (measured), so the lexical lane needs
  application-side tokens.
- AIZK is small: 2.76 GB database (251,821 vectors), 659 MB of blobs, 5,572 usage events, about 1 keep an hour. The
  center's internal disk is 94% full (28 GiB free), which binds before RAM does. Its Wi-Fi hop shows 74 ms spikes.
- Nothing from SERVE's phases has landed. The live stack was built on 2026-09-17 and cannot restart cleanly: five
  containers bind-mount files from the deleted `~/projects` path. Backups sit on the database's own disk, unreadable
  by the crimson login; there is no WAL archiving. `pgqueuer_log` grows 60 MB a day with nothing reading it.
- OpenRouter now lists `qwen/qwen3-reranker-8b` ($0.20 per million tokens) beside `qwen3-embedding-8b` ($0.01), so
  SERVE's "no reranker" no longer holds; a full re-embed costs about $0.25, hosted reranking about $4 a month.
- `mb serve` is taken (`mainboard host serve` claims jobs on nodes; the lake has its own serve verb), so service control
  is `mb memory up|down|status|logs|doctor|migrate`.

## 3. The shared server

One PostgreSQL 18.x cluster per center, from the workspace environment (M36), data directory on the internal disk, a
short socket directory (the spike's scratchpad path exceeded macOS's 103-byte limit), launchd agent apart from the
settler. Tuning renders from one RAM number: `shared_buffers` 2 GB, `effective_cache_size` 8 GB, `maintenance_work_mem`
1 GB, `max_wal_size` 4 GB, `max_connections` 120, WAL archiving and WAL summaries on.

| database | schemas | roles | extensions |
|---|---|---|---|
| `mainboard` | `control`, `ducklake_<LakeId>` (M19, M36, M37, M38) | `mainboard`, `mainboard_rw`, `ducklake_writer`, `ducklake_reader` | contrib only |
| `aizk` | AIZK's `public`, unchanged | `aizk_admin` (not a superuser), `aizk_app` (NOBYPASSRLS, forced RLS on 16 tables, 42 policies) | vector, pg_trgm, pgcrypto, pg_stat_statements |
| `logto` | Logto's | `logto`, `logto_tenant_*` | none |

Why separate databases rather than schemas: AIZK hard-codes `public` in Alembic, in 42 policies, in `pgrls.toml` and in
its `search_path`; M38's consistent snapshot should stay short for the control and catalog schemas while AIZK's 2.7 GB
dumps on its own cadence; extensions are per database, so the catalog database never depends on pgvector; DuckDB's
PostgreSQL scanner never sees `halfvec`. Per-role `statement_timeout`, `lock_timeout` and connection limits give the
isolation PostgreSQL offers on a Mac. No external pooler (none is on conda-forge, 18 connections are in use).

Extensions, by tier. Tier 1 (recommended): pgvector HNSW for vectors and an application-tokenized text-search lane plus
`pg_trgm`, gated by the golden set (U03, U04, U16). Tier 2 if the gate fails: build the three pgrx extensions from source
into an `extension_control_path` directory owned by M36. Tier 3: keep the `aizk` database on Linux (rejected as default).

Backups: native `pg_basebackup` (incremental), WAL archiving, per-database `pg_dump`, globals; copies to
`/Volumes/PORTABLE` and to crimson's empty 1.8 TB SSD (U12). AIZK's own nightly dump job is retired.

## 4. What moves where, and the services after slimming

| data | goes to |
|---|---|
| documents, chunks, entities, facts, claims, profiles, communities, ontology, queue, watermarks, RLS, vectors, usage serving rows | PostgreSQL `aizk`, unchanged |
| usage ledger copy | DuckLake `memory_usage` (append-only mirror; served from PostgreSQL because DuckLake has no RLS) |
| structured logs and events | DuckLake `memory_events` (JSONL spool, settler ingest) |
| golden-set and ablation results | DuckLake receipts via `mb` jobs |
| artifact originals (480 objects, 659 MB) | a content-addressed directory on the internal disk (obstore LocalStore); later mainboard's own store |
| `pgqueuer_log` (777,695 rows) | not migrated; fixed at the source (U09) |
| Logto tenant data | `logto` database, moved as a dump so the issuer and signing keys survive |

| service | runs | RAM (GiB) |
|---|---|---|
| PostgreSQL | native launchd (M36) | 3.0 |
| aizk-mcp, aizk-api, aizk-worker | native launchd from the workspace environment | 0.43, 0.29, 2.0 (5.62 today; target set by U09) |
| Caddy, frontend | native (conda-forge caddy 2.11.7, Node) | 0.15 |
| Logto, cloudflared | pinned containers | 0.4, plus the Docker VM (about 1.25, inferred) |
| total | | 6.2 without Docker, 7.7 with; 9.8 to 11.3 if the worker stays at 5.6 |

Removed: Docling, GLiNER (gate off by default; the mention detector loads lazily on CPU only if web search is enabled, and
the ledger shows no web operation ever), SeaweedFS, vllm-emb, vllm-rerank, Tempo, Loki, VictoriaMetrics, Alloy, Grafana, the
docs container (Caddy serves static files) and, by default, ClamAV (not on conda-forge; D7).

## 5. Surfaces, auth, observability

- MCP: the five tools and the artifact resource at `https://aizk.phvv.me/mcp`, unchanged. A plugin hook (U19, deferred)
  lets mainboard add read-only tools (jobs, hosts, costs, lake query) behind a Logto scope without AIZK importing mainboard.
- CLI: `mb memory find|keep|status|share|report|login|logout`, mounted from AIZK's existing client behind a
  `mainboard[memory]` extra (U14); `mb memory up|down|status|logs|doctor|migrate` supervises the services (U13, U17). The
  `[gates.aizk]` script is replaced by a native doctor probe (keep and find as a synthetic user).
- Auth: Logto only, same issuer `https://auth.phvv.me`, same database and signing keys, so agents keep their sessions.
  Device flow (Logto 1.38 and later) for headless hosts; a machine-to-machine client with the control scope for the
  settler, the doctor and DEV LOG writes.
- Observability: no tracing stack. Logs and the usage spool are JSONL; the settler appends them to the lake (U07, U15);
  `mb query` and the operator console read them.

## 6. The packages, beside the M waves

Each fits one Codex run in a designated worktree, with acceptance against a throwaway PostgreSQL started from the
workspace environment and fake services. `mainboard.toml` changes are rollout inputs. Detail (files, changes, acceptance,
risk) is in `unify.json`.

| id | package | repo | needs from M | live |
|---|---|---|---|---|
| U00 | AIZK suite on a throwaway extension-free PostgreSQL | aizk | M00, M36 | no |
| U01 | Golden set, scorer and benchmark job from the operator reports | aizk | M00 | no |
| U02 | Retrieval fixes from SERVE phase 0 | aizk | U00, U01 | no |
| U03 | pgvector HNSW backend and conversion of an existing database | aizk | U00, U01 | no |
| U04 | Lexical lane without vendor extensions | aizk | U03 | no |
| U05 | Drop CockroachDB (about 8,400 lines) | aizk | U00 | no |
| U06 | Object bytes on a local content-addressed store | aizk | U00 | no |
| U07 | Telemetry to files and a usage spool | aizk | U00 | no |
| U08 | Hosted embedding and reranking, optional gate, resumable re-embed | aizk | U00, U01 | no |
| U09 | Worker memory (5.6 GiB) and queue churn (110k log rows a day) | aizk | U00 | no |
| U10 | Maintenance mode and the keep journal | aizk | U00 | no |
| U11 | The shared server's layout: databases, roles, extension directory, doctor rows | mainboard | M36, U03, U04 | no |
| U12 | Backups and PITR for every database, offsite | mainboard | M38, U11 | no |
| U13 | AIZK's services under mainboard (`mb memory up|down|...`) | mainboard | M36, M24, U11 | no |
| U14 | `mb memory` client front over AIZK's client | mainboard | M00 | no |
| U15 | Lake streams `memory_usage` and `memory_events` | mainboard | M22, M23, M24, U07 | no |
| U16 | Staging rehearsal with the real AIZK data | both | U03 to U13, M37 | reads live |
| U17 | Migration orchestrator and the cutover | mainboard | U10, U12, U13, U16, M32, M37 | yes |
| U18 | Decommission crimson, enroll GPU 0 in the lease, delete the losing backends | both | U17, M30 | yes |
| U19 | MCP plugin hook and mainboard tools (deferred) | both | U14, U17, M21, M24 | no |
| U20 | Structured DEV LOG writes and current-state answers (deferred) | mainboard | U14, U15, U17, M26 | no |
| U21 | Close-out audit | both | U18 to U20, M33 | no |

Order. U-wave 0 (U00 to U10, U14) is AIZK worktree work that runs now beside M waves 0 to 3 and needs only M00 and the
pgvector pin. Its merge order is U00, U01, U02, U10, U09, U08, then the lead cuts the L0 build for crimson, then U05, U06,
U07, U03, U04, U14. U-wave 1 (U11, U13) follows M36 and M24 as implemented; U-wave 2 (U12, U15) follows M38 and M22 to M24;
U-wave 3 (U16, U17) needs M37 and M32 live and the center's PostgreSQL soaked 7 days; U-wave 4 (U18) needs the lease (M30);
U-wave 5 closes after M33.

Live events (each a lead action with drain discipline and the owner present):

| event | what | outage |
|---|---|---|
| L-1 | golden-set baseline against the live stack (about 200 read-only finds) | none |
| L0 | recreate the crimson stack from `~/aizk` and deploy the L0 build (U02, U10, U09, U08) | at most 5 minutes |
| L1 | stage the center: empty `aizk` and `logto` databases, WAL archiving, units installed stopped | none for agents |
| L0.5 | hosted embeddings on crimson: shadow re-embed, one swap transaction, lanes stopped, GPU 0 idle | seconds |
| L2 | the cutover (section 7) | write freeze at most 25 minutes, read outage at most 2 |
| L3 | soak, then archive and decommission | none |

Calendar: L-1 and L0 may precede the MLSys deadline (2026-10-30) if approved; L2 not before 2026-11-17 (after CVPR),
not inside a mainboard wave switch, and only after the center's PostgreSQL has been control and catalog authority for 7
days without an incident. L0.5 follows U16's go. The sequence "embedding change on crimson first, data move second"
keeps one change per step and moves a stable vector space, since the center has no GPU for the old embedder.

## 7. The live migration

Principle: move a frozen snapshot, never acknowledge a keep the surviving side does not hold, keep the old side intact until
the new side has been verified and has soaked.

Prerequisites: L0 and L0.5 done and soaked a week; L1 done with a verified base backup and offsite copy; U16 passed with a
measured freeze budget; wired Ethernet on the center and at least 15 GiB free on its data volume; a tunnel for the center
tested on a temporary hostname; objects pre-copied with hash verification (they are immutable); a rollback rehearsal on the
staging copy; the owner present.

| time | step |
|---|---|
| T0 | old side to draining: keep, share and report return a typed retryable error (never acknowledged); find continues without access-count writes; usage lines go to a journal file; background jobs pause; wait for in-flight transactions (at most 30 s) |
| T0 + 1 min | `pg_dump -Fc -j 4` of `aizk` (consistent snapshot), stream to the center (350 to 450 MB at 11 to 14 MB/s) |
| T0 + 4 min | restore with `maintenance_work_mem` 1 GB and 4 workers, using the generated restore plan that skips the vendor extensions; build HNSW and the text-search index; copy the object delta |
| T0 + 12 min | verify: alembic head, per-table counts and id digests, RLS probe, doctor keep and find on the local route, five golden queries, queue recovery pass |
| T0 + 15 min | stop Logto on crimson, dump and restore `logto` (25 MB), start Logto on the center, flip the four hostnames (`aizk`, `auth`, `admin`, `console`) to the center's tunnel |
| T0 + 18 min | new side serving with its keep journal on; replay the usage lines journaled during the freeze (idempotent on capture key); thaw |
| T0 + 20 min | abort line: anything not through the flip by now aborts the move and thaws the old side |

Written budgets: write freeze at most 25 minutes (15 expected), read outage at most 2 minutes (the flip window; find
is served by the old side until then), Logto sign-in and refresh gap at most 2 minutes (issued access tokens stay valid),
L0 stack recreation at most 5 minutes, rollback write freeze at most 10 minutes.

No lost keep. Before T0 every acknowledged keep is in the snapshot. Between T0 and the flip no keep is acknowledged: the old
side refuses with a retryable error, so the agent knows to retry and nothing acknowledged can be missing. After the flip
every acknowledged keep is committed on the new side and written to its keep journal in the same transaction. At about one
keep an hour, the expected number of keeps that see the retryable error is at most one.

Rollback. Before the flip: thaw the old side; nothing was acknowledged during the freeze, so nothing is lost. After the flip,
within 24 hours: drain the new side, flip the hostnames back, replay the new side's keep journal onto the old side
(idempotent), thaw the old side; the old tunnel, images, volumes and database are retained for that purpose (24 hours
running, 14 days stopped). After that, recovery is forward, or from the archived crimson volumes plus the journal; U18's
acceptance rehearses it.

If U16 measures a freeze above 25 minutes, the variant is logical replication of the `aizk` database over an ssh tunnel,
which shrinks the freeze to the final catch-up and costs one more database restart on crimson inside L0.

## 8. Decisions for the owner, each with a recommendation

| id | decision | recommendation |
|---|---|---|
| D1 | Keep AIZK a separate product, or merge repositories (SERVE 2) | Separate. Mainboard supervises and fronts it. Revisit if, for a quarter, more than half of AIZK's commits also change mainboard. |
| D2 | Drop CockroachDB (SERVE 3) | Yes, in U05, after tagging the repository. About 8,400 lines, four dependencies and a second queue go; the profile stays reachable from the tag. |
| D3 | Prune crimson's Docker build cache (SERVE 4) | Yes, the cache only: `docker builder prune` of entries older than a week is safe now (223 GB of 289 GB reclaimable; the root disk is 68% full, down from 90%). Images stay until the soak ends; volumes follow the archive rehearsal. |
| D4 | Where AIZK runs | The center, as one unit. Splitting costs 8 to 20 round trips of 12 ms per call and ties AIZK to the Wi-Fi hop and the jump host. |
| D5 | Retrieval backend on the shared server | Extension-free first, gated by the golden set; source-built VectorChord next; Linux database last. |
| D6 | Hosted or local models | Hosted, behind the golden gate, switched on crimson first (L0.5). About $4 a month plus $0.25 once; frees 6.4 GiB and GPU 0. Local only if sending note text to a provider is ruled out (extraction already sends it to DeepSeek under zero-retention terms). |
| D7 | ClamAV | Drop for this single-owner deployment (257 file keeps in 84 days, all from your own disk); one container if scanning is required. |
| D8 | Observability stack | Drop (1.83 GiB, 2 GB disk). Keep VictoriaMetrics and Grafana only if you want the dashboard. |
| D9 | Logto on the center | Pinned container first; try native Node later; not Logto Cloud (the issuer would change and every agent would sign in again). |
| D10 | Timing | L0 allowed before 2026-10-30 if you want better retrieval now; L0.5 after U16's go; L2 not before 2026-11-17 and after a 7-day soak of the center's PostgreSQL. |
| D11 | Wired Ethernet for the center | Yes, before L2 (your action; the port exists and is unused). |
| D12 | Cloudflare route | A new tunnel for the center and an atomic flip of four hostnames; the old tunnel's ingress names crimson-only container hostnames. Needs dashboard or API access. |
| D13 | One lake or two | One lake, new table names. |
| D14 | Usage ledger into DuckLake (SERVE phase 2) | No: serve from PostgreSQL (RLS, transactional enqueue), mirror into DuckLake. |
| D15 | Command name | `mb memory` (client verbs and service control); `mb serve` is taken. |
| D16 | Offsite backups | crimson's empty SSD plus `/Volumes/PORTABLE`. |
| D17 | Outage budgets | Accept 25 min write freeze (15 expected), 2 min read, 5 min for L0, 10 min rollback. |
| D18 | Drop the extraction gate | Yes, default off, no service. |

## 9. What is not settled, and what a critic would still check

- The extension-free lexical lane is the biggest quality risk (Japanese). Its gate needs the golden set, which does not
  exist yet; U01 builds it from the 168 operator reports and L-1 captures the baseline.
- The worker's 5.62 GiB is unexplained; the RAM budget holds only if U09 reaches about 2 GiB.
- The freeze budget extrapolates a 100,000-vector measurement to 251,821 vectors in five tables; U16 replaces it with a
  measurement, and the replication variant exists if it overruns.
- Native Logto, a Logto container reaching the host database through `host.docker.internal`, and cloudflared reaching Caddy on
  the host are unverified. So is the Cloudflare route flip and whether OpenRouter honors `dimensions=1024`.
- The doctor probe and DEV LOG writes need a Logto machine-to-machine client with the control scope; the gate script says that
  grant was still pending.
- Two release trains (the L0 build on crimson, the center build) exist for weeks; the keep-journal format must not change after L0.
- The internal disk is 94% full; PostgreSQL's growth, WAL and backups must stay under a floor enforced by a doctor row.
- Keep statement counts (15 to 25) are inferred; the find count (8 to 10) was measured.
- I changed nothing in the repositories, on crimson or in the lake. On crimson I ran only `docker ps/stats/system df/top/inspect`,
  `nvidia-smi`, `df` and read-only `psql`; against AIZK I made four read-only `find` calls (which append usage events and bump
  access counts, two of them to measure the statements of one find). A throwaway PostgreSQL ran in the scratchpad and was
  stopped and removed afterwards.

## 10. Files

All under `/private/tmp/claude-501/-Users-pedro-Developer-projects/b52c03f0-d920-5f82-9aac-285c87707d12/scratchpad/codex/aizk/`:
`state.md`, `state.json`, `options.md`, `options.json`, `unify.md`, `unify.json`, and `evidence/` (the earlier runs' sound
files plus `ssh-roundtrips.txt`, `center-measurements-2026-10-10.txt`, `restore-timing.txt`).

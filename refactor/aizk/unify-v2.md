# Unifying AIZK into mb memory, version 2

Revision of `unify.md` (2026-10-10) after its two critics, written 2026-10-11: `critic-simplicity.md` (17 findings, cited
CS1 to CS17) and `critic-migration.md` (19 findings, CM1 to CM19). The machine-readable version is `unify-v2.json` (keys
overview, layout, packages, order, migration, open_questions, excluded, critique). The inputs are unchanged: `state.md`,
`options.md`, mainboard design v3 (`design.md`, `owner-decisions.md`).

I changed no source, committed nothing and touched nothing on crimson. Locally I ran read-only checks to confirm what the critics
claimed: `df`, `fdesetup status`, `pmset`, the settler's plist, `git log` and file reads in `packages/aizk` and
`packages/mainboard`. What only crimson can show (the running image, `/backups`, the ledger, `VmHWM`) is marked critic-measured.

## 1. The goal, and what changed

AIZK keeps its public identity (its own repository, the five MCP tools, Logto login) but runs on the same system as mainboard:
the center's single PostgreSQL cluster holds the DuckLake catalog, mainboard's control schema and AIZK's transactional data in
three databases; AIZK's usage history reaches the same DuckLake the evidence lives in; one `mb memory` front, one doctor and one
backup chain cover it. It gets there with one build for the center and one night. Nothing is deployed to crimson beforehand;
crimson's stack stays stopped and untouched as the revert target; the vector, lexical and model changes are judged offline on
scratch restores before the night; and a verified off-host backup comes first, because it is worth having even if nothing else
happens. crimson's GPU 0 returns for non-exclusive work about a week after the night (2026-11-25 at the earliest), not before the
deadlines.

| | version 1 | version 2 |
|---|---|---|
| packages | 22 (U00 to U21) | 13 (U00 to U12) |
| live events on crimson | L-1, L0, L0.5, L2, L3 | one read-only backup proof, the night, the decommission |
| builds | two (an L0 build for crimson, a center build) | one, for the center |
| freeze machinery | maintenance mode, keep journal, replay, typed error (U10) | stop three containers at night, Caddy answers with a message |
| backups | WAL archive, incremental base backups, PITR, per-database dumps | daily dumps of every database, a weekly restore test, offsite copies, an encrypted secrets bundle |
| usage into the lake | spool, versioned JSONL, two streams, five files (U07, U15) | a 20-line pull through DuckDB's postgres scanner |
| center processes | PostgreSQL, 3 AIZK processes, Caddy, frontend, Logto and cloudflared in Docker, oauth2-proxy | the same minus oauth2-proxy, all native; Logto stays a container only if its spike fails |
| migration code | U17 orchestrator, `route.py`, 300-kill chaos tests | a runbook of about nine idempotent steps and one verify script that is also the doctor probe |
| owner decisions | 18 | 7, plus a short veto list |
| gate for AIZK's move | M37 and M32 live | M36 live and soaked 7 days under staged AIZK load, and M38's dump |

## 2. Dispositions of the 36 findings

31 accepted, 5 adapted, 0 rejected as a whole. Eight sub-proposals are rejected (listed after the tables). Reasons are one line
each; the JSON names the package that carries each one.

Critic of simplicity:

| id | finding | disposition | reason |
|---|---|---|---|
| CS1 | decisive facts measured after 14 packages | accept | the RAM, the lexical lane and the find latency decide the plan, so U00 measures them first with written go and no-go numbers |
| CS2 | no intermediate builds on crimson | accept | build once for the center; the vendor-to-pgvector conversion is a one-off tool; SERVE phase 0 runs on its own track; the L-1 baseline moves to a scratch restore because finds are not read-only (CM8) |
| CS3 | delete maintenance mode and the keep journal | accept | stopping the three containers gives the same guarantee with no code, and a night window (CM12) cuts the exposure to about 0.3 keeps |
| CS4 | pull usage instead of a spool | accept | `usage_event` is immutable with a capture key, so a cursor pull has no crash window; `memory_events` has no consumer |
| CS5 | daily dumps instead of PITR | accept | an archive on a 96% full disk can stall all three databases, and today's recovery point is already 24 hours; a shorter one becomes decision D6 |
| CS6 | U17 is a runbook, not a package | accept | one cutover with the owner present does not justify dead code afterward; the runbook and verify script live in U11 |
| CS7 | drop Docker | adapt | cloudflared native and no oauth2-proxy are taken; Logto native only if its spike passes, because the container fallback puts a GUI app on the critical path |
| CS8 | a mechanism for the worker's 2 GiB | accept | heavy jobs run as short-lived subprocesses (the Docling pattern); the 8.5 GiB peak, the pure-Python Louvain and the whole-table `.all()` are in the code |
| CS9 | `pg_restore -L` cannot drop a column | accept | `restore-plan` is deleted; S2 proves the replacement on the real archive |
| CS10 | shadow columns built twice | adapt | no shadow columns or swap; one `reembed` inside the night, or a reuse of vectors by id and text equality from a pre-embedded stage (a copy of rows by `updated_at` would skip derived facts) |
| CS11 | second service supervisor | accept | extend `durable.py`'s renderer with KeepAlive; no `units.py`; services run from a separately locked `memory` env so `mainboard upgrade` never changes a live process |
| CS12 | delete the dead code, U05 first | accept | U01 goes first and absorbs the Docling, ClamAV, GLiNER, web-lane and OpenTelemetry deletions; the benefit is about 2.3k executable lines, not 8,400 |
| CS13 | pgvector in `mainboard.toml` now | accept | conda-forge carries 0.8.6 on all three platforms; no extension directory; tier 2 becomes the arm64 vendor container (CM9) |
| CS14 | disk floor as a number | accept | 30 GiB free before staging (19 GiB today), `max_wal_size` 2 GB, the staging copy dropped after rehearsal |
| CS15 | U14's `aizk[client]` extra does not exist | accept | mount AIZK's existing client app lazily; add `report` to it; no extra and no `memory/client.py` |
| CS16 | cut U19 to U21 and most decisions | accept | U19 and U20 go to the excluded list with triggers, U21 folds into M33's ledger; seven decisions remain |
| CS17 | do not tie the move to the catalog migration | accept | AIZK's databases do not depend on M37; the gate is M36 live plus a 7-day soak and M38's dump |

Critic of migration:

| id | finding | disposition | reason |
|---|---|---|---|
| CM1 | agents authenticate through the live OIDCProxy | accept | HEAD's `auth.py` advertises Logto and Logto has no dynamic registration, so U02 comes first: Logto-direct with a pinned public client by default, decided by an agent matrix, and no secret rotation at the cutover |
| CM2 | live code is the August lineage, 0015 is not additive | accept | the 23-commit jump (verified) never runs in place: crimson is untouched, and 0013 to 0016 run on the center copy, rehearsed on a restored dump |
| CM3 | non-superuser `aizk_admin` breaks the worker | accept | v1 would make the SECURITY DEFINER function fail open; `aizk_admin` is `NOSUPERUSER BYPASSRLS`, a `backup` role dumps, and positive probes prove all three |
| CM4 | backups exist but were never restored | accept | E0 comes first: restore-tested dumps of `aizk` and `logto`, roles, an encrypted secrets bundle, offsite; the nightly job stays on crimson until the soak ends |
| CM5 | the conversion path cannot work as written | accept | `-j` needs `-Fd`, the archive is 1.1 GB, `pgqueuer_log` data is excluded at dump time, and the freeze table is recomputed from measurements |
| CM6 | five changes behind one aggregate gate | accept | a ladder B0 to B5 on clones of one restored snapshot, each rung with its own margin; no L0.5 |
| CM7 | the embedder calibrates graph merges | accept | thresholds are reset at the old operating point from labeled pairs, auto-merge stays off until a shadow run of the last 200 keeps passes |
| CM8 | the gate margins sit inside the noise | accept | scratch restores only; one pooled paired non-inferiority test sized from the pilot; synthetic queries; latency plus 10% under load, n of 200 or more |
| CM9 | the lexical lane is unproven | accept | the A/B runs first on the real 33,268 chunks with OR queries, BM25-weighted sparsevec and Japanese morphology; the arm64 vendor container is the named fallback |
| CM10 | hosted lanes make keep depend on OpenRouter | adapt | bounded retries, timeouts, a spend cap, the dimension probe and a lexical fallback for find; the pending-embedding keep and the generation row are rejected because no live swap remains |
| CM11 | MCP sessions die on restart | accept | HEAD already runs `stateless_http` and `json_response` in the AWS profile; make that the only mode after auditing for server-initiated messages |
| CM12 | the write window is wrong in daytime | adapt | the night window (01:00 to 06:00 JST) and a Caddy 503 message, which is config; the durable freeze file is rejected because it is the keep journal again |
| CM13 | Logto roles, rollback, verification, tunnel token | accept | roles-only dump, JWKS key-id check, an authenticated keep and find on localhost before the flip, the token in an env file, the flip measured on temporary hostnames |
| CM14 | the center is not an unattended server | accept | stated as decision D4 for the mainboard design too (section 9); Docker stays off the critical path |
| CM15 | the disk floor is already broken | accept | a reserved APFS volume for PGDATA, a numeric floor, nothing archived on the internal disk, and the roughly 120 GiB consumer found first |
| CM16 | cluster PITR contradicts M38 | adapt | moot, since no PITR remains; if the owner picks a minutes-level recovery point (D6), PITR covers `aizk` and `logto` only |
| CM17 | order against the waves | accept | M36 renders connections, preload and WAL settings from its first start; the gate is M36 plus M38; a required and a deferred list below |
| CM18 | GPU 0 returns in December | accept | the date is 2026-11-25 at the earliest and is no longer presented as a pre-deadline benefit |
| CM19 | retention would erase the 56 failed builds | accept | the 56 are requeued on the restored copy and their failures recorded; retention is requeue once, then expire; documents with no claims are counted on both sides |

Eight rejected sub-proposals: the keep journal, maintenance mode, the typed retryable error and a durable file that accepts keeps
during the freeze; shadow columns, the swap transaction and an `embedding_generation` row; a keep that stores a pending embedding
(trigger: the first refused keep caused by the provider); copying rows by `updated_at` as the re-embed shortcut; the v1
logical-replication variant (it needs a database restart on crimson, whose stack cannot restart cleanly); cluster-wide WAL
archiving, base backups and PITR (trigger: D6); the "previous images" rollback, which disappears with L0; and L-1 as a "read-only"
baseline on the live database, since each find bumps access counts and enqueues work.

## 3. What the plan stands on, corrected

- Disk: the data volume has 19 GiB free at 96% (checked 00:15 JST today; v1 said 28 GiB at 94%). The 2026-10-06 notes had
  140 to 156 GiB free, so about 120 GiB went in five days, and the critic found `~/Library/Caches` at 18.9 GB and `~/.cache` at
  11.5 GB. PostgreSQL on this disk is the largest new exposure, for mainboard as much as for AIZK.
- Live code: crimson runs the August 8 lineage (critic-measured: image built 2026-09-17, alembic head 0012, fastmcp 3.4.4).
  HEAD is 23 commits later with migrations 0013 to 0016 (verified); 0015 rewrites every RLS policy from a JSON setting to typed
  settings, so old code reading a migrated database sees an empty memory.
- Auth: the live server hosts an OIDC proxy that registers agents dynamically at `aizk.phvv.me`; HEAD's `auth.py` returns a
  `RemoteAuthProvider` naming Logto (verified, commit `c4d7ec1`, 2026-08-13). `state.md` section 4 was wrong about the live path.
- Backups: 11 nightly dumps exist (1.07 to 1.10 GB, critic-measured) but none was ever restored, and `logto`, roles, `.env` and
  the MCP state have no copy. `state.md` F2 said the directory was unreadable; it is readable through `docker exec`.
- Archive size: 1.1 GB, not 350 to 450 MB; `pg_dump -j` needs `-Fd`; `pg_restore -j` cannot read stdin.
- Retrieval constants: `find_max_distance` 0.65, auto-merge 0.9, borderline floor 0.75, RAPTOR redundancy 0.95 and ontology match
  0.85 are calibrated to Qwen3-VL embeddings (verified in `settings.py`).
- MCP sessions: only the AWS profile is stateless (verified), so every restart drops every open session today.
- Center boot posture: FileVault on, `autorestart 1`, no auto-login, only third-party updaters in `/Library/LaunchDaemons`, and the
  settler is the user LaunchAgent `mainboard.monitor.3784a243` (verified).
- Worker: 5.6 GiB resident, 8.5 GiB peak (critic-measured); the code builds a networkx Louvain graph and calls `.all()` on a whole
  embedding table (verified).
- Unchanged from v1: the latency measurements, the three-database layout, placement on the center as one unit, pgvector HNSW with
  relaxed iterative scan, OpenRouter prices, and `mb memory` as the command (`mb serve` is taken).

## 4. The server

One PostgreSQL 18.x cluster per center from the workspace environment (M36), data directory on the internal disk, a short socket
directory, a launchd agent apart from the settler. M36 renders the settings from its first start so no later AIZK event restarts it:
`shared_buffers` 2 GB, `effective_cache_size` 8 GB, `maintenance_work_mem` 1 GB, `max_wal_size` 2 GB, `max_connections` 120,
`shared_preload_libraries = pg_stat_statements`, `archive_mode` off. Pgvector 0.8.6 joins `mainboard.toml` as an ordinary dependency
(the lead's first edit), so `CREATE EXTENSION vector` works with no path settings.

| database | schemas | roles | extensions |
|---|---|---|---|
| `mainboard` | `control`, `ducklake_<LakeId>` (M19, M36, M37, M38) | `mainboard`, `mainboard_rw`, `ducklake_writer`, `ducklake_reader` | contrib only |
| `aizk` | AIZK's `public`, unchanged | `aizk_admin` (NOSUPERUSER, BYPASSRLS, `pg_read_all_stats`), `aizk_app` (NOBYPASSRLS, forced RLS on 16 tables, 42 policies), `aizk_usage_reader` (NOBYPASSRLS, one SELECT policy on `usage_event`), `backup` (NOSUPERUSER, BYPASSRLS, `pg_read_all_data`) | vector, pg_trgm, pgcrypto, pg_stat_statements |
| `logto` | Logto's | `logto`, `logto_tenant_*` (restored from a roles-only dump) | none |

Role posture is checked, not assumed: doctor rows for `aizk_admin.rolbypassrls = true`, `aizk_app.rolbypassrls = false` and no
service superuser, plus positive probes: as `aizk_admin` all 16 tables return the bootstrap count, `artifact_content_blob_attachable`
returns false for a blob attached in another scope, `pg_dump` as `backup` succeeds on the forced tables, and a queued graph job
completes. Per-role timeouts and connection limits are unchanged from v1 (`aizk_app` 40, `aizk_admin` 10, `mainboard_rw` 20,
`ducklake_writer` 20, `logto*` 10); no pooler.

Services, all native launchd units rendered by `durable.py`'s one renderer with KeepAlive, from a separately locked `memory`
environment. The units start in any order because AIZK's connect loop waits for PostgreSQL.

| service | RAM (GiB) |
|---|---|
| PostgreSQL | 3.0 |
| aizk-mcp, aizk-api | 0.43, 0.29 |
| aizk-worker as dispatcher, plus one heavy job at a time (peak 1.5 or less, set by S6) | 0.4, +1.5 transient |
| Caddy, frontend | 0.15 |
| Logto (native Node), cloudflared (pinned binary, token in an env file) | 0.3, 0.05 |
| total | 4.6 steady, 6.1 at a heavy job, no Docker VM |

The center already holds about 9.2 GiB it cannot reclaim and swaps 3.4 GiB with the owner's apps open, so 6.1 GiB is a ceiling to
verify in U00 and not a margin. If Logto stays a container the total gains the Docker VM (about 1.3 GiB) and a GUI app that waits for
a login. Removed: Docling, GLiNER, SeaweedFS, vLLM lanes, the five observability containers, the docs container, ClamAV, oauth2-proxy
and, unless U00 finds a page that needs it, the `console` hostname gate (every `/api/admin/*` route already requires admin). The
compose isolation ("an exploit in the MCP server cannot obtain the owner connection") is lost because all services run as the owner;
the owner password is readable only through the worker's environment file, and the remainder is accepted knowingly.

Disk: PGDATA on a dedicated APFS volume with a reserve and a quota (decision D5), a doctor floor of 30 GiB free before staging,
local dumps limited to the latest two, everything else on `/Volumes/PORTABLE` and crimson's SSD. Steady state is about 8 GB; the
rehearsal peak (a staged copy of 3.7 GB, dumps, restore) is 12 to 15 GB.

## 5. What moves where

| data | goes to |
|---|---|
| documents, chunks, entities, facts, claims, profiles, communities, ontology, queue, watermarks, RLS, vectors, usage serving rows | PostgreSQL `aizk`, converted once (section 8) |
| usage ledger copy | DuckLake `memory_usage`, pulled by the settler from `usage_event` through DuckDB's postgres scanner as `aizk_usage_reader`, cursor from M23, idempotent on capture key |
| structured logs | JSON files with rotation under the services directory, read with `read_json` when an incident needs them; no lake stream |
| golden-set and ladder results | DuckLake receipts via `mb` jobs |
| artifact originals (480 objects, 659 MB) | a content-addressed directory (obstore LocalStore), hash-verified pre-copy, rsynced in the daily backup |
| `pgqueuer_log` (777,695 rows) | not migrated (`--exclude-table-data`); fixed at the source (U08) |
| Logto tenant data and roles | `logto` database, moved by `pg_dump` plus `pg_dumpall --roles-only` so issuer and signing keys survive |
| env files, tunnel token, passfiles, MCP state if a proxy mode is kept | an encrypted bundle in the backup set, key held off the machine |

## 6. Surfaces and auth

- MCP: the five tools and the artifact resource at `https://aizk.phvv.me/mcp`, served stateless (`stateless_http`, `json_response`)
  so restarts, the flip and a rollback are invisible to sessions, once an audit finds no sampling or elicitation in the tools.
- Auth (U02): by default Logto-direct, as HEAD does, with a public Logto application whose client id and callback port are pinned in
  `.mcp.json`, `.agents/mcp.json` and the Claude, Codex and opencode configs of every host. The cost is one re-login per agent
  install (5 client registrations exist, critic-measured). If Codex or opencode cannot use a pinned client, the fallback is to keep an
  OIDC proxy mode with an explicit signing key held apart from the client secret and `mcp-state` copied at the night. Either way no
  secret rotates at the cutover; rotation follows the soak. The doctor probe needs a Logto machine-to-machine client with the `control`
  scope (still pending) and the stale issuer in `scripts/aizk-gate.py` fixed, both before staging.
- CLI: `mb memory up|down|status|logs|doctor` supervises; `mb memory find|keep|status|share|report|login|logout` is AIZK's existing
  client app mounted lazily in its own `App` module, so `cli.py` gains one line and the host runtime never imports fastmcp.
- Observability: no tracing stack; `operator_snapshot`, the usage ledger and JSON logs remain.

## 7. The packages, beside the M waves

Each fits one Codex run in a designated worktree. Acceptance runs on a throwaway PostgreSQL from the workspace environment and fake
services; `mainboard.toml` edits are rollout inputs the lead applies. Detail (files, changes, acceptance, risk) is in the JSON.

| id | package | repo | needs from M | live |
|---|---|---|---|---|
| U00 | W-1 spikes S1 to S9 and the first verified off-host copy | both | none (M36's binaries if present) | reads crimson |
| U01 | Delete what the center will not run (CockroachDB and AWS, Docling, ClamAV, GLiNER, web lane, OpenTelemetry) | aizk | none | no |
| U02 | Auth path, agent configs, stateless MCP, gate issuer | aizk | U00 (S8) | no |
| U03 | AIZK on the workspace PostgreSQL: fixture, extension-free schema, HNSW, role probes | aizk | M00, pgvector in the toml | no |
| U04 | Lexical lane chosen by the A/B | aizk | U00 (S4), U03 | no |
| U05 | Golden set, non-inferiority test and the ladder runner | aizk | M00, U00 | no |
| U06 | Hosted lanes and threshold calibration | aizk | U03, U05 | no |
| U07 | Local content-addressed object store | aizk | U03 | no |
| U08 | Worker memory by subprocess, queue churn, failed-job evidence | aizk | U00 (S6), U03 | no |
| U09 | Shared server layout, role posture, daily dumps and restore test | mainboard | M36 | no |
| U10 | AIZK's services under mainboard, the doctor probe, the client mount and the usage pull | mainboard | M36, M23 for the pull | no |
| U11 | Rehearsal, runbook scripts and the verify script | both | U01 to U10, M36 soaked | reads live |
| U12 | Decommission crimson, GPU 0 | both | U11, M30 | yes |

Merge order (shared files `settings.py`, `0001_init.py`, `vector.py`, compose): U01, U03, U07, U04, U06, U08, then U02 and U05, which
touch other files. SERVE phase 0 (retrieval fixes) is its own track: if it merges before the ladder starts it is rung B1; otherwise it
waits until the soak ends. Nothing that changes ranking merges between the first ladder run and the end of the soak.

| M wave | lands | U beside it |
|---|---|---|
| now | | E0 and U00 (read-only), U01 |
| W0 | M00 fixtures | U03 starts, U05 starts |
| W1 | M01 to M03, M34 | U02, U07 |
| W2 | M06, M10 | U03 merges, U04, U06, U08 |
| W3 | M36 and the PostgreSQL foundation | U09, U10 (units and doctor), E1: stage the center, weekly restore tests begin, the 7-day soak starts |
| W4 | M19 to M24, M37, M38, M32 | U10's usage pull rides M23; AIZK untouched; M37 and M32 are not prerequisites |
| W5 | environments, sealed sources | U11 rehearsals in quiet gaps (twice, plus a reboot drill and a rollback drill with the owner), E3 the night once the gates pass |
| W6 | M30 lease | U12 |
| W7 | M33 | close-out folded into M33's ledger |

Required before the night: U00 to U09, the units and doctor of U10, U11. Deferred after it: U10's client mount and usage pull, U12.
Before the MLSys deadline (2026-10-30) only worktree work and the read-only U00 run, nothing live; the cutok v26 program draws on the
same agents, so if capacity is short the order of sacrifice is U10's deferred half, then U05's synthetic cases, never U00.

Live events:

| event | what | outage |
|---|---|---|
| E0 | this week: restore-tested dumps of `aizk`, `logto`, roles and an encrypted secrets bundle, copied to `/Volumes/PORTABLE` and a second location (S1) | none |
| E1 | after M36 live: empty `aizk` and `logto` databases and roles, units installed stopped, weekly restore test running | none |
| E2 | rehearsals on the center with a fresh dump; the second includes the reboot drill, the flip on temporary hostnames, the copy-back rollback and the agent matrix | none for agents |
| E3 | the night, the first one after 2026-11-17 18:00 JST (the CVPR submission passes) with the gates green and no mainboard wave switch in flight | at most 60 minutes, reads and writes |
| E4 | 7 days of soak with crimson's stack stopped and intact, 14 more days stopped, then archive and decommission; GPU 0 exposed for non-exclusive work from 2026-11-25 | none |

## 8. The migration

Principle: move an exact snapshot taken after the old side has stopped, never acknowledge a keep the surviving side does not hold, and
keep the old side untouched until the new side has soaked.

Prerequisites: E0 green; U11's second rehearsal green with a measured median total of 45 minutes or less on the wired network; the
disk floor met and Ethernet connected (D5); the agent matrix passed; the reboot drill done (D4); objects pre-copied with hash
verification; the Cloudflare route tested on temporary hostnames (D7); the owner present. The order inside the freeze comes from S2's
measured fastest safe path; the default below is the donor conversion on crimson.

| time (JST) | step | budget |
|---|---|---|
| T-1 h | objects pre-copied, last nightly dump verified, Caddy maintenance snippet staged on crimson (not loaded), center stack staged and stopped | |
| 01:00 | load the snippet (503 with "AIZK is moving until HH:MM; hold the note and retry"), `docker stop` server, api and worker, confirm no connection to `aizk` | 2 min |
| T+2 | donor conversion: `CREATE DATABASE aizk_flat TEMPLATE aizk`, drop `chunk.bm25`, `DROP EXTENSION vchord, vchord_bm25, pg_tokenizer CASCADE`; `pg_dump -Fd -j 4 --exclude-table-data=pgqueuer_log`; rsync to the center | 8 min |
| T+10 | `pg_restore -j 4` with `maintenance_work_mem` 1 GB, `alembic upgrade head` (0013 to 0016), HNSW and GIN builds, lexical backfill, object delta | 14 min |
| T+24 | re-embed: `aizk admin reembed` if S5 measured 15 minutes or less, otherwise vector reuse by id and text equality from the pre-embedded stage | 3 to 15 min |
| T+30 | verify script (below) | 5 min |
| T+35 | Logto: stop on crimson, `pg_dump logto` and `pg_dumpall --roles-only`, restore, start, JWKS key ids equal to the live set, an authenticated keep and find on localhost while DNS still points at crimson | 8 min |
| T+43 | flip the routes by S7's method (edit the tunnel's ingress, or a new tunnel and three or four CNAMEs), probe from outside, start the center worker | 5 min |
| T+48 | abort line: anything not through the flip unloads the snippet and starts the three crimson containers | 2 min |
| T+60 | hard end of the budget | |

The sums are estimates until U11 replaces them with medians. If the measured median exceeds 45 minutes the window widens to 90
before any redesign, since the night rate is about 0.1 writes per 25 minutes (critic-measured from the 30-day ledger: 8, 7, 9, 12, 9
and 6 writes in the hours 01 to 06 against 32 to 50 in daytime hours).

Outage budget: reads and writes both down at most 60 minutes, expected 45, at night; the Logto sign-in gap is inside that window;
rollback before the flip costs at most 2 minutes; copy-back rollback after the flip is estimated at 20 minutes of write freeze and
is measured in U11.

Keep-safety. Before 01:00 every acknowledged keep is committed on crimson and therefore in the dump taken after the containers
stopped. From the stop to the flip nothing can be acknowledged because no process answers; an agent sees the Caddy message or a
connection error and holds the note. After the flip every acknowledged keep is a committed transaction on the center, covered by the
daily dump and by the copy-back. At night the expected number of keeps that meet the message is about 0.3; the agent's `aizk` skill
carries one sentence for the day telling it to retry. There is no journal and no replay code to get wrong.

Rollback:
- Before the flip: unload the snippet and `docker start` the three containers. Only an `aizk_flat` database was added on crimson;
  drop it afterward. Nothing is lost.
- After the flip, during the 7-day soak and the 14 stopped days: stop the center's AIZK units, flip the routes back, `docker start`,
  then copy back with `\copy` the `document`, `artifact`, `artifact_content` and `blob` rows (and objects) created after 01:00; the old
  worker rebuilds chunks and graph. Logto changes are not carried back: every agent logs in again, which is budgeted. The expected
  copy is at most about 24 rows in the first day. U11 rehearses this once.
- After that: forward only (fix the center), or the latest daily dump on `/Volumes/PORTABLE` and crimson's SSD, a 24-hour recovery
  point (D6).
- A dead center before the soak ends: restart crimson's stack from the stopped state, then copy back as above, using the last dump
  for rows the dead center never dumped.

Verification (the same script is the doctor probe): alembic head equals the AIZK release; per-table counts and id digests equal to
the dump; documents with no claims counted equal on both sides after processing; 42 policies present, forced RLS, the positive
probes of section 4; five golden queries; an authenticated keep and find on the local route; the JWKS key ids; the agent matrix
(Claude Code, Codex and opencode on the Mac and one remote host, starting from their stored credentials or a fresh login per U02,
plus a server restart under an open session of each); settler pass latency, free disk and resident set within budget for 24 hours.

## 9. The unattended-center question, for AIZK and for the mainboard design

Facts (verified here): FileVault is on, no auto-login user is set, `pmset` has `autorestart 1` (a power cut restarts into the
FileVault unlock screen) and `sleep 0`, `/Library/LaunchDaemons` holds only third-party updaters, and the settler is a user-domain
LaunchAgent. Design v3 chooses a same-user LaunchAgent for PostgreSQL (M36) and an ssh-localhost settler (M24), and says to "retain
user-login/session limitations in doctor". The doctor runs on the center, so it cannot see a center that is locked or down. External
volumes mount only at the GUI login (2026-10-06 note, per the critic).

Consequence for mainboard: after a power cut or an OS restart, PostgreSQL (control and the catalog), the settler and the card leases
stay down until someone unlocks the console. Hosts keep running their jobs and write offline receipts, but `list` and `show` answer
"unavailable", settlement stops, and new submissions wait. Consequence for AIZK: no agent on any host has memory. The mechanism is
identical and the impact differs, so the posture is decided once, in M36, and AIZK inherits it.

| option | what it needs | what it gives |
|---|---|---|
| (a) accept | an external watcher on crimson (curl `/healthz` each minute plus the age of the settler's last pass, alerting the owner's phone), `fdesetup authrestart` for planned restarts, automatic macOS update installs off, a written unlock runbook | a human is needed after a power cut, and everything else is visible |
| (b) system LaunchDaemons with `UserName=pedro` for PostgreSQL, AIZK, Logto and cloudflared | a one-time `sudo` install by the owner; the settler stays an ssh-localhost agent for the external volume | services return after a restart that unlocks FileVault without a GUI login; a power cut still needs the password |
| (c) FileVault off with auto-login | nothing | unattended recovery for everything, at the price of at-rest protection for finance and health notes and every secret: not recommended |

Recommendation: (a) now, because no option removes the password after a power cut short of (c). Run the reboot drill in E2 with
the owner to learn which services return after an authenticated restart; adopt (b) for PostgreSQL and the AIZK units only if the drill
shows agents wait for a login that daemons would not. Docker stays off the critical path either way, and crimson's stopped stack stays
the revert target until one real restart has been recovered cleanly. M36 and M24 should record the chosen posture in the design.

## 10. Decisions for the owner, each with a recommendation

| id | decision | recommendation |
|---|---|---|
| D1 | Go for the slim, native, one-night plan: AIZK on the center as one unit; delete CockroachDB and AWS, Docling, ClamAV, GLiNER, the web lane, the observability stack and oauth2-proxy; run E0 and the read-only spikes now | Yes. Tag the repository first; the web lane and the Cockroach profile stay reachable from the tag. Merges v1 D2, D4, D7, D8 and D18; v1's other defaults are in the veto list below. The web lane recorded zero events in 84 days. |
| D2 | Hosted embedding and reranking, and the privacy line | Yes, behind the ladder and the threshold recalibration. Note text goes to OpenRouter's routed providers with data collection denied (extraction already sends text to DeepSeek); about $4 a month plus $0.25 once. If the ladder rejects hosted models, the lanes stay on crimson's GPU 0 behind an authenticated route and GPU 0 stays AIZK's. |
| D3 | The night and its accepted risks | The first night after 2026-11-17 18:00 JST in the 01:00 to 06:00 window, 60 minutes total outage (reads and writes), about 0.3 refused keeps, one re-login per agent install, copy-back rollback, and GPU 0 back on 2026-11-25 at the earliest. Yes. |
| D4 | The unattended center (section 9), for both designs | (a) accept with a watcher and the reboot drill; (b) only if the drill shows it helps; not (c). |
| D5 | Owner actions on the machine: free the disk to 30 GiB or more (find the 120 GiB consumer first; candidates `~/Library/Caches` and `~/.cache`), create the reserved APFS volume for PGDATA, connect wired Ethernet | Yes, before E1. The volume protects PostgreSQL from every other writer, which a doctor row cannot. |
| D6 | Recovery point | 24 hours: daily dumps, now restore-tested and offsite, which equals today's exposure with a proven restore. A minutes-level point means `pg_receivewal` to `/Volumes/PORTABLE` through the settler context and is added only if a note-loss incident or a higher keep rate asks for it. |
| D7 | Cloudflare and Logto access for the rehearsal and the flip | A scoped Cloudflare API token (zone DNS and tunnel edit) held in the center's `.env` at 0600 and revoked after the soak; otherwise the owner clicks from a printed checklist. Also create the public Logto application and the machine-to-machine `control` grant before E1. |

Defaults taken unless you veto them: AIZK stays a separate repository fronted by mainboard (revisit if more than half of AIZK's commits
also change mainboard for a quarter); one lake with new table names; usage served from PostgreSQL and mirrored; the command is
`mb memory`; retrieval backend order is extension-free first (gated), the arm64 vendor container second, a Linux-hosted database last;
Logto native if its spike passes, otherwise a pinned container; offsite copies on crimson's SSD and `/Volumes/PORTABLE`; crimson's Docker
build cache may be pruned any time (cache only, images wait for the soak).

## 11. What is not settled

- The conversion path (S2), the lexical lane (S4) and the worker's per-job peaks (S6) decide the plan's size; each has a written
  go and no-go in U00. If S4 fails the named fallback is the arm64 vendor container for `aizk`, which breaks "one server" and puts a GUI
  app on the critical path, so it needs the owner's attention before it is adopted.
- Native Logto and native cloudflared, and whether the tunnel's ingress can be edited in place rather than replaced, are unverified (S7).
- Whether Codex and opencode can use a pinned public client (S8). If not, an OIDC proxy mode survives in AIZK.
- The night's total is an estimate summed from a 346 MiB benchmark and the critic's archive size; U11 replaces it.
- OpenRouter's treatment of `dimensions=1024`, its throughput for 23 M tokens and its data-collection routing are unmeasured (S5).
- The gate's power: the margin and n are fixed from the pilot's paired standard deviation before the first ladder run.
- Everything marked critic-measured (the running image, the dumps, the ledger, `VmHWM`) comes from crimson reads I did not repeat.

Files, all under `/private/tmp/claude-501/-Users-pedro-Developer-projects/b52c03f0-d920-5f82-9aac-285c87707d12/scratchpad/codex/aizk/`:
`unify-v2.md`, `unify-v2.json`, and the earlier `unify.md`, `unify.json`, `state.md`, `options.md`, `critic-simplicity.md`,
`critic-migration.md`.

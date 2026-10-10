# Critic: correctness and migration risk of the AIZK-into-mainboard plan

Reviewed 2026-10-11 00:00 to 00:40 JST. Inputs: `unify.md`, `unify.json`, `state.md`, `options.md`, mainboard design v3
(`design.md`, `owner-decisions.md`), against `packages/aizk` (HEAD `6ebeab1`), `packages/mainboard` and the live stack.
Nothing was changed or committed in the repositories.

What I ran, all read-only. On crimson: `docker ps`, `docker exec` into `aizk-db-1` (catalog queries, one aggregate query on
`usage_event`), `aizk-worker-1` (`ls /backups`, `pg_restore -l` and `pg_restore --schema-only -f -` of the newest nightly dump,
stdout only, plus one temporary TOC listing in the container's tmpfs that I deleted) and `aizk-server-1` (file names and counts
under `/var/lib/aizk/mcp`, installed versions, the deployed `auth.py`), `docker inspect` (image ids, creation dates, environment
variable names), `nvidia-smi`, `df`. On the center: `df`, `vm_stat`, `ps`, `pmset`, `fdesetup status`, `docker info`, three public
OAuth discovery documents, the Docker Hub tag API. One `docker inspect` printed the cloudflared tunnel token in the container's
command line; I did not record the value anywhere (finding 13).

## Verdict

The server-side move (snapshot, journal, replay) is sound in principle, but the plan is written against the wrong code and
the wrong auth model. The live stack is the August 8 lineage of AIZK, not HEAD, and it authenticates agents through an
AIZK-hosted OAuth proxy, not through Logto directly. Three things can silently break or lose data and are not in the plan:
the agent login path (1), the migration 0015 rewrite of every RLS policy (2), and the owner role becoming a non-bypass role (3).
Four more make the conversion and the gates unsound as written: the restore plan (5), the confounded quality gate and its
statistics (6, 8), and the embedder thresholds that also drive graph merges (7). Findings 1 to 4 block the first live event
(L0) and any deploy of HEAD-lineage code that agents can reach; finding 5 blocks L2.

## Findings

### 1. Agents do not authenticate with Logto today; the plan's session argument is wrong and L0 would break every agent

Evidence:
- `curl https://aizk.phvv.me/.well-known/oauth-authorization-server` returns issuer `https://aizk.phvv.me/` with
  `registration_endpoint https://aizk.phvv.me/register`; `/.well-known/oauth-protected-resource/mcp` lists
  `authorization_servers: ["https://aizk.phvv.me/"]`. Logto's own discovery (`https://auth.phvv.me/oidc/.well-known/openid-configuration`)
  has no `registration_endpoint` at all.
- The deployed `auth.py` in `aizk-runtime:health-20260917` builds `OIDCProxy(...)` with
  `fastmcp_access_token_expiry_seconds=oauth_reference_token_seconds` (31,536,000 s, one year) on fastmcp 3.4.4. HEAD's `auth.py`
  (since commit `c4d7ec1`, 2026-08-13) returns `RemoteAuthProvider` naming Logto as the authorization server.
- `.mcp.json` and `.agents/mcp.json` give the agents the URL only (Codex adds `oauth_resource` and scopes), no client id. They
  register dynamically with whatever authorization server the server advertises. HEAD's plugin config instead pins a
  pre-registered `clientId` (`plugins/aizk/claude.mcp.json`), which is what a DCR-less Logto requires.
- The agents' session state is `/var/lib/aizk/mcp/oauth-proxy` in the `mcp-state` volume: 5 client registrations, 4 refresh
  tokens, 8 jti mappings, 4 upstream tokens. `provider()` passes no `jwt_signing_key`, so fastmcp derives the proxy signing key
  from `AIZK_OAUTH_CLIENT_SECRET` by HKDF, derives the Fernet key of the store from that, and names the store directory after the
  key's fingerprint (`oauth-proxy/c2cb4ce63a84`, read in the deployed `oauth_proxy/proxy.py`, lines 486 to 536). Its comment says key
  rotation "just causes re-registration": a different secret yields a new empty directory and every client registers and signs in again.
- The plan lists "FastMCP state files (mcp-state 42 kB, oauth 8 kB)" in `migration.scope` but no step copies or tests them, says
  "same issuer, database and signing keys, so agents keep their sessions", and says to "rotate the OpenRouter key and the Logto
  application secrets at the cutover". Rotating `AIZK_OAUTH_CLIENT_SECRET` changes the derived keys and voids every issued token.

Consequence: HEAD-lineage code at `aizk.phvv.me` rejects the AIZK-issued tokens, sends clients to Logto, and Logto cannot
register them. The same happens at L2 if L0 is dropped.

Change:
- Make the auth mode its own first package, before U02/U10: either keep an `OIDCProxy` mode as the self-hosted default (typed
  `auth_mode`), or register a public Logto application and write `clientId` and `callbackPort` into `.mcp.json`,
  `.agents/mcp.json` and the Codex and opencode configs of every host.
- Give the proxy an explicit `jwt_signing_key` and storage key held apart from the OAuth client secret, so the secret can rotate.
- Copy `mcp-state` and the secrets at L2. Add to U16 a staging server on the restored copy that accepts a token minted by the
  live server, and an agent matrix (Claude Code, Codex, opencode, on the Mac and one remote host) that starts from the current
  stored credentials.
- Delete secret rotation from the cutover; rotate after the soak.
- Correct `state.md` section 4 ("Bearer tokens are verified by auth.py (FastMCP RemoteAuthProvider...)") and the `Auth` paragraph in `unify.md`.

### 2. The live code is the August 8 lineage; L0 is a 23-commit jump with a non-additive migration and a false rollback

Evidence:
- Running images: `aizk-runtime:latest` created 2026-09-17 10:11 (previous `pre-health-20260917` 2026-08-08), frontend image
  2026-08-08. Inside: fastmcp 3.4.4, openai 2.45, asyncpg 0.31, alembic head `0012_operator_snapshot_rls` (live `alembic_version`).
  The closest commit is `b0098be` (2026-08-08). HEAD is 23 commits later: 125 files, +4,367 and -1,246 lines under `src/aizk`,
  migrations 0013 to 0016, fastmcp 4.1.0, openai 3.28, asyncpg 0.32.
- Migration `0015_policy_names` runs `PolicyContextTransition.replace` and rewrites every policy and the SECURITY DEFINER
  function from the JSON carrier (live: `current_setting('app.scopes')::jsonb`) to typed settings `app.scopes.read/write/public`.
  The code bound to the old carrier then sees nothing, because an unbound session sees no rows by design.
- `unify.json` L0 says "the migrations are additive" and the rollback is "docker compose up of the previous images against the
  unchanged volumes", with "at most 5 minutes" outage. After 0015 the previous images read an empty memory (a find returns nothing,
  not an error), and while `setup` migrates the still-running old `server` and `api` do the same.

Change:
- L0a: HEAD alone (no U02, U10, U09, U08), first rehearsed on a scratch restore of the nightly dump on crimson itself, with the
  rollback proven there (`alembic downgrade 0012`, or restore of a pre-migration dump into a fresh volume). Production rollback is
  the restore of a dump taken minutes before, not "previous images".
- Stop `server`, `api`, `worker` before `setup` runs (U10 does not exist yet), then bring up.
- Add the agent matrix of finding 1 to L0's verification.
- If L0 is dropped as `critic-simplicity.md` proposes, the same jump lands in the L2 freeze: U16 must then run HEAD code against the
  restored live dump including 0013 to 0016, and the agent matrix, before L2 is scheduled.

### 3. Making `aizk_admin` a non-superuser without BYPASSRLS breaks the worker, a security function and pg_dump

Evidence:
- Live `pg_roles`: `aizk_admin` rolsuper t, rolbypassrls f (superuser bypass only); `aizk` rolsuper t, rolbypassrls t.
  16 of 16 RLS tables are enabled and forced; all 42 policies are `TO {aizk_app}` only. For any other non-bypass role, including a table
  owner, no policy applies and the default is deny.
- `Database.owner()` is documented "privileged ... bypasses RLS" and the compose comment says the worker alone holds the owner
  credential for queue execution, maintenance and backups.
- `artifact_content_blob_attachable` is SECURITY DEFINER owned by `aizk_admin` and ends in
  `IF NOT EXISTS (SELECT 1 FROM artifact_content WHERE blob_id = target_blob) THEN RETURN true`. If RLS applies to its owner, that
  EXISTS sees zero rows and the function returns true for every blob: it fails open (any caller can attach any blob id to new content).
- `object_retirement` and `operator_snapshot` are readable by the owner only through the bypass. `pg_dump` by a role subject to RLS
  fails ("query would be affected by row-level security policy"); U12 and the retired nightly job both need a role that can dump.
- U11 and `layout.roles_rules` require "no AIZK role is a superuser"; the acceptance checks only that `aizk_app` sees zero rows
  without the scope and that no service role is a superuser. No check shows that the owner still sees rows.

Change:
- The bootstrap role runs `ALTER ROLE aizk_admin NOSUPERUSER BYPASSRLS` (only a superuser can grant BYPASSRLS), plus
  `pg_read_all_stats` for the probes and `pg_signal_backend` if the drain terminates sessions.
- Doctor rows: `aizk_admin.rolbypassrls = true`, `aizk_app.rolbypassrls = false`, `aizk_app` has no membership in either.
- Positive probes in U11/U00: as `aizk_admin` every one of the 16 tables returns the bootstrap role's count;
  `artifact_content_blob_attachable` returns false for a blob referenced from another scope; `pg_dump` as the backup role succeeds on
  the forced tables; a queued graph job runs to completion on the throwaway cluster (not just "no error").

### 4. The backup story is better than "none" and worse than the plan assumes; make a verified off-host copy the first live event

Evidence:
- `/backups` in `aizk-worker-1` (readable by `docker exec`; the host directory is uid 10001) holds a daily
  `aizk-YYYYMMDDT020001Z.dump` for 2026-09-30 to 2026-10-10, each 1.07 to 1.10 GB (newest 1,098,211,948 bytes), written 02:00 to
  02:02 (two minutes to dump), 16 GB in the directory on crimson's own 915 GB root disk. `pg_restore -l` of the newest reads clean
  (349 TOC entries, 28 TABLE DATA, dumped by pg_dump 18.4 from server 18.3).
- `backup.py::scheduled_backup` dumps the one configured database. The newest `logto` dump is a manual one from 2026-07-10
  (363 KB). Logto (users, organizations, 1,249 OIDC grants and sessions, 4 `oidc.*` key rows in `logto_configs`, M2M and web
  application secrets) and the cluster roles have no backup; neither do `.env` and `mcp-state`.
- No dump has ever been restored: the archive contains `CREATE EXTENSION vchord, vchord_bm25, pg_tokenizer` and the `bm25vector`
  column (finding 5), so the center cannot test it today. The plan's "verified base backup and offsite copy" is an L1 prerequisite;
  the first time it can exist is U12, while L0 (recreating the database container) may run before 10-30.

Change:
- Add L-2, read-only, this week: pull the newest `aizk` dump (1.1 GB, 80 to 100 s at the measured 11 to 14 MB/s), take a fresh
  `pg_dump -Fc logto` and `pg_dumpall --roles-only`, copy to `/Volumes/PORTABLE` and a second location, and restore both into an
  arm64 `tensorchord/vchord-suite:pg18` container on the center (the `pg18-latest` tag publishes arm64 and amd64 images per the Docker
  Hub tag API; confirm that the pinned digest's index carries arm64). Record `alembic current`, table counts, one keep and one find,
  and the restore wall time. This is also the first real number for the freeze budget.
- Make L0 conditional on L-2 passing.
- Put an encrypted bundle of `.env`, the passfiles and `mcp-state` (key held off the machine) in the backup set; M38's "nonsecret
  recipe" leaves them out.
- The nightly job stays on crimson until L3.

### 5. The conversion path inside the freeze cannot work as written, and the freeze is larger than budgeted

Evidence:
- `pg_restore -L` selects whole TOC entries. The real dump has `CREATE TABLE public.chunk (... bm25 bm25_catalog.bm25vector)`
  (schema lines 459 to 471) and a data entry `COPY chunk (..., bm25)`. No list can drop a column, so U03's `restore-plan` fails at the first
  table with it, and U03's acceptance ("skips the bm25 column ... restores without errors") cannot pass. This matches
  `critic-simplicity.md` finding 9; I checked the real archive but did not run a restore.
- `unify.md` T0 + 1 and `migration.steps` say `pg_dump -Fc -j 4`. `-j` is valid only with `-Fd`; and `pg_restore -j` cannot read stdin, so
  "stream to the center" needs the whole archive on disk before the parallel restore starts.
- The nightly archive is 1.10 GB (gzip custom format), not "350 to 450 MB"; it includes the 427 MB `pgqueuer_log` the plan does not want
  (skipping its data is possible by TOC entry). Transfer is 80 to 100 s at 11 to 14 MB/s and minutes on the Wi-Fi tail. The restore
  benchmark (346 MiB, 117.7 s, random vectors) is a third of the real archive.

Change: choose the conversion before U03 is built, and prove it on the L-2 archive.
- Donor conversion on crimson (writes frozen): `CREATE DATABASE aizk_flat TEMPLATE aizk`, drop the `bm25` column, trigger and
  vendor extensions there, `pg_dump -Fd -j 4 aizk_flat`, copy the directory, `pg_restore -j 4`.
- Or build the schema with `alembic upgrade head` on the center and move data with per-table `COPY (SELECT <non-vendor columns>)`
  streams under one exported snapshot.
- Or avoid it: run `aizk` in the arm64 vendor container on the center (tier 2.5, no re-embed, no lexical change; the Docker VM is
  paid for anyway by Logto and cloudflared) and convert later at leisure.
- Recompute the T0 table from the L-2 restore time and the real archive size.

### 6. Five retrieval changes with one aggregate gate; the baseline is ambiguous and L0.5 waits on the wrong gate

Evidence: before the owner's `find` serves from the center it changes the ranking rules (U02), the embedder (Qwen3-VL-Embedding-2B to
`qwen3-embedding-8b`, a different model with a different instruction format; `embed_instruction_*` wrap every text), the reranker
(Qwen3-Reranker-4B to `qwen3-reranker-8b`), the vector index (vchordrq to HNSW with relaxed iterative scan) and the lexical lane
(bm25 to tsvector). U16 compares the whole new stack against "the live baseline" once. If it fails, the cause is unattributable.
U02 deliberately changes live ranking at L0, so the L-1 baseline and the live stack after L0 are two baselines. L0.5 (freeing
GPU 0) needs U16's go, which needs U03 to U13 and M37, although L0.5's own question (hosted lanes against local lanes on the vendor
path) needs none of them.

Change:
- Run an A/B ladder on one frozen restored snapshot, the same queries, the same day: B0 live code, B1 +U02, B2 +hosted embedder,
  B3 +hosted reranker, B4 +HNSW, B5 +tsvector lane. Each step has its own registered noninferiority margin and stop rule; U02 is
  judged as an improvement separately.
- Make D10's alternative the primary L0.5 gate: a scratch database restored from the nightly dump on crimson, with the golden runner
  pointed at shadow columns before the swap.

### 7. The embedder also calibrates graph merges, and no package recalibrates it

Evidence in `config/settings.py` and `graph/consolidation.py`: `find_max_distance 0.65` ("calibrated on real Qwen3-VL query/document
embeddings: relevant chunks land at cosine distance 0.27 to 0.49, off-corpus questions bottom out at 0.60 to 0.75"),
`consolidation_auto_merge_threshold 0.9` and `consolidation_borderline_floor 0.75` (facts above 0.9 similarity merge without an LLM),
`raptor_redundancy_threshold 0.95`, `raptor_sim_threshold 0.5`, `ontology_match_threshold 0.85`. U08's acceptance and U16's gate test `find`
ranking only; the golden set cannot see a wrong merge. After the swap every keep's consolidation and the 03:30 `dedup` run on the
new geometry.

Change, as an L0.5 precondition: (a) sample about 2,000 labeled fact and entity pairs (duplicate, related, unrelated), compare old
and new cosine distributions, and set each threshold at the old operating point (same precision and recall), with a receipt;
(b) shadow-run consolidation on the last 200 keeps with both embedders and diff the merge decisions; (c) set auto-merge above 1.0
(off) until (a) and (b) pass; (d) calibrate `find_max_distance` with the off-corpus questions.

### 8. The gate statistics cannot detect the margins they declare, and the "read-only" baseline is not read-only

Evidence:
- U01 reports (does not assert) "at least 150 cases, at least 10 per failure class", built from 168 operator reports, i.e. from
  failures, so the set over-represents the six audit classes. The gates are MRR and recall@10 within 0.02 overall and 0.03 per class.
  With a per-case reciprocal-rank spread of 0.3 to 0.4 (an assumption to be measured on the first slice) the standard error is about
  0.025 to 0.03 at n = 150 and about 0.1 at n = 10: the margins are inside the noise. U04's fixture has 60 cases, and one case is 0.0167
  of top-3 hit rate against a 0.02 margin. The find p50 gate "not worse than 5,000 ms" has 0.9% headroom over the measured 4,954 ms.
- L-1 says the 200 finds are read-only apart from usage events. Each find also bumps `fact_claim.access_count` and `last_accessed`
  (`store/models/tables/fact.py:183`), which feed the frequency weight and decay (`0.5 ** (age / half_life) * (1 + access_count)`),
  and enqueues work. A second golden run on the live database sees the first run's effects and nudges the owner's real ranking.

Change:
- Never run golden sets on the live database. Run them on a restored copy in an evaluation mode that skips access and usage writes
  (U10's draining branch is exactly that; reuse it). L-1 becomes "restore the newest dump to scratch and run there".
- One pooled, one-sided noninferiority test (paired bootstrap upper bound on the loss no greater than the margin, margin and n
  registered before the run); per-class numbers are descriptive unless n >= 50.
- Enlarge the case set with synthetic queries generated from sampled documents (gold = the source chunk), which costs cents with the
  extraction model already in use, and stratify by document kind.
- Make the latency gate "no worse than +10% on the same machine under a concurrent-load profile, n >= 200 paired queries", measured
  on the center with the owner's apps and a settler pass running.

### 9. The replacement lexical lane is unproven on the three things that make BM25 work: IDF, subword tokens and query semantics

Evidence:
- The live lane is `tokenizer_catalog.tokenize(coalesce(lexical, text), 'aizk_bm25')` in a trigger, scored by `<&>` and fused by rank.
  Only 5,859 of 33,269 chunks have a non-null `lexical`, so U04's backfill is a real tokenization of the whole table, not a column copy.
- U04 tokenizes with llmlingua2's vocabulary (a subword vocabulary) into a `simple` tsvector with a GIN index and `ts_rank_cd`.
  PostgreSQL text search ranking has no IDF, so very frequent pieces dominate. `plainto_tsquery` is AND semantics: a 10 to 20 token
  agent query matches almost nothing; an OR query matches most of the table and `ts_rank_cd` reads every matched tsvector, so latency
  grows with candidate-set size. None of this is measured. The only existing FTS code (`chunk.py`, Cockroach branch) is an
  unindexed query-time `to_tsvector('english', ...)` that U05 deletes first.
- The gate is a 60-case fixture (finding 8) and the golden set, both after U03 to U13.

Change: run the lexical A/B first and at scale (critic-simplicity finding 1 asks for the same ordering). Tokenize the 33,268 real chunks
from the L-2 restore on the center, measure candidate-set sizes and p95 latency for the query strings of 200 real finds, and score the
overlap@20 against the bm25 lane's own top 20 before any golden set exists. Include in the A/B: BM25-weighted `sparsevec`
(pgvector 0.7 and later is already carried; IDF computed in the application and refreshed on a schedule, inner product gives BM25
semantics), Japanese morphological words (Sudachi or fugashi from the workspace environment) instead of subwords, and OR queries
with a bounded candidate set. Keep the arm64 vendor container (finding 5) as the named fallback instead of building three pgrx
extensions on macOS.

### 10. Hosted lanes make every keep and find depend on OpenRouter, and the swap cannot be atomic with a process setting

Evidence: `keep` embeds before it stores (`extract/ingest.py:353` and `:542` call `EmbedClient...embed(searchable, mode="document")`
in the request path; no deferral), and `find` embeds the query; `EmbedClient.embed` has no fallback. Today the embedder is a container
beside the database. After L0.5 it is a provider call from crimson, and after L2 from a Wi-Fi host with 74 ms spikes and a 50 to 80%
loss history. U08 adds a fallback for the reranker only. An outage, a rate limit or spent credit then fails `keep` (the DEV LOG
path every agent is told to use) and degrades `find`. U08 also "switches the query embedder setting at the same moment" as the swap
transaction; `embed_model` and `embed_url` are process settings (`config/settings.py:286` to `:288`), and no database transaction can
flip three processes at once, so "50 concurrent finds see all old or all new" can only pass if the generation is read from the database
in the same transaction.

Change:
- Keep: store the document and chunks with `embedding NULL` and a pending marker, enqueue the embed, acknowledge. Find: fall back to
  lexical and title lanes when the query embedding fails, and say so in the trace and receipt.
- An `embedding_generation` row (model, dimensions, URL) read inside every find and keep transaction; the swap updates it in the
  swap transaction.
- A daily spend cap, per-call timeouts, and a recorded check that `dimensions=1024` is honored and data-collection is denied for the
  routed providers (the plan marks both unverified).

### 11. MCP sessions are in-process; every restart, flip and rollback drops them

Evidence: `commands/admin.py:138` runs `server.run_http_async(host, port)` with no `stateless_http`; only the AWS profile passes
`stateless_http=True`. The live `aizk-server-1` has no `FASTMCP_STATELESS_HTTP` (only `FASTMCP_HOME` among `FASTMCP_*`), and
`client_label` reads `context.session.client_params`, a per-session value. So each `Mcp-Session-Id` lives in one process. L0, the L2
flip and any rollback invalidate all open sessions. The protocol expects a client to re-initialize on 404, but the plan has no
per-harness test, and its T0 + 25 check opens fresh sessions. I could not probe the live endpoint because `/mcp` requires a token.

Change: run the L0 and center builds with `stateless_http=True` and `json_response=True` if no tool needs server-initiated messages
(I saw none in the five tools but did not audit sampling or elicitation), so restarts become invisible. Otherwise, in staging, kill and
restart the server under an open session of each harness, record what each does on its next call, and put it in the runbook.

### 12. The write-freeze expectation is right on average and wrong where it matters; the typed error has no machine contract

Evidence: the ledger (30 days, JST, `remember_text`, `remember_file`, `share`): 689 text keeps; 17.5% of 25-minute windows contained at
least one write; the busiest window held 13. By hour: night 01 to 06 has 8, 7, 9, 12, 9, 6 writes over 30 days (about 0.1 per
25 minutes); 08 to 17 has 47, 32, 38, 46, 50, 45, 50, 47, 43, 33 (about 0.55 to 0.7 per 25 minutes). The plan says "at most one" and
the owner is present, which in practice means daytime. A "typed retryable error" reaches an MCP caller as tool-error text to a model;
agents call the MCP `keep` directly, not `mb memory keep`, so U14's offline spool does not protect them. The note is lost if the
agent moves on.

Change:
- Choose the window from the ledger and write it in D17 (01:00 to 06:00 JST), or state that the owner accepts a daytime window with
  about a one in three chance of a refused write.
- Put the freeze banner in `status` and in the refusal text ("hold the note, retry in N minutes"), and add that sentence to the
  `aizk` skill for the day.
- Optional: during the freeze, accept text keeps into a durable file on the old side and replay them before thaw (the usage journal
  mechanism already planned), which makes refusals zero without breaking "never acknowledge what the surviving side lacks".
- This holds whether U10 survives or is replaced by stopping the containers.

### 13. Logto and the public route: roles, rollback, verification and the tunnel token

Evidence:
- The `logto` database has 71 tables, 69 with RLS and 137 policies, driven by cluster-level roles `logto_tenant_logto` (no login),
  `logto_tenant_logto_default` and `logto_tenant_logto_admin` (login). A single-database dump (T0 + 15) carries neither the roles nor
  their passwords; U11 `ensure()` creates `logto` but the tenant roles come from Logto's own seeding.
- The rollback replays AIZK keeps only. Logto changes in the 24 hours after the flip (new refresh tokens, grants, users, organization
  edits) are not carried back, and the old Logto does not know refresh tokens the new one issued.
- `scripts/aizk-gate.py` still names the stale issuer `3ado57.logto.app`; the machine-to-machine `control` grant is "still pending".
  T0 + 12's "doctor probe ... over the public route" cannot authenticate, and the fallback is an in-process `Memory.keep` (no auth).
- cloudflared runs as `tunnel ... run --token <token>`: the token is in the container's command line and shows in `docker inspect`
  to any docker-group user of a lab machine. A new tunnel for the center repeats this unless changed.
- "An atomic flip of four hostnames" is four DNS or route records; Cloudflare gives no cross-record atomicity. The browser console and
  frontend also depend on `AIZK_WEB_CLIENT_SECRET` and `AIZK_WEB_SESSION_SECRET` staying identical.

Change:
- Include `pg_dumpall --roles-only` in the L2 payload and in U16; in U16 start Logto 1.41.0 on the restored `logto` database, compare
  its JWKS key ids with the live set, and verify a live-issued access token against the staging JWKS.
- The rollback runbook dumps `logto` back, or says "every agent logs in again" and budgets for it.
- Create the M2M `control` grant and fix the gate's issuer before L1; then at T0 + 12 run a real authenticated keep and find against
  the center on localhost while DNS still points at crimson (the center server fetches JWKS from the still-live `auth.phvv.me`).
- Pass the tunnel token through an 0600 environment file, never argv; delete the old tunnel after the soak.
- Measure the route flip on a temporary hostname pair in the real zone (record propagation and connector-up times) and script the
  order; flip `auth` and `aizk` in one API batch and verify each.

### 14. The center is not an unattended server: FileVault, user-domain launchd and Docker Desktop

Evidence: `fdesetup status` On; no `autoLoginUser`; `pmset` has `autorestart 1` (a power cut restarts into the FileVault login
screen) and `sleep 0`. The settler is a user-domain LaunchAgent in `~/Library/LaunchAgents` (`mainboard.monitor.3784a243.plist`,
installed by `durable.py`), `/Library/LaunchDaemons` holds only third-party updaters, Docker Desktop is a GUI app (running now, 1.26 GB
resident with zero containers), and the 2026-10-06 note says external drives mount only at GUI login. crimson today has 16 days of
uptime under `restart: unless-stopped`. After a macOS update, a panic or a power cut the center, and with it PostgreSQL, AIZK, Logto and
the tunnel, stays down until someone logs in at the console. The doctor runs on the center and cannot see this.

Change: decide explicitly. Either (a) accept it, with an external watcher on crimson that curls `https://aizk.phvv.me/healthz` each
minute and alerts the owner, plus a written unlock runbook and a rule against unattended macOS update restarts; or (b) render PostgreSQL,
the AIZK processes and cloudflared as system LaunchDaemons with `UserName=pedro` (they start after FileVault unlock without a GUI session)
and keep Docker off the critical path (`critic-simplicity.md` finding 7). Keep crimson's stopped stack and current volumes as the revert
target until one unplanned reboot has recovered cleanly on the center; L3 must not delete it before that.

### 15. Disk and WAL: the floor is already broken and the archive is a single point of failure for both databases

Evidence: the data volume has 21 GiB free at 96% (`df`, 2026-10-11 00:05); the plan says 28 GiB and 94%. The 2026-10-06 notes recorded
140 to 156 GiB free, so about 120 GiB went in five days, and `~/Library/Caches` is 18.9 GB and `~/.cache` 11.5 GB. The plan puts PGDATA
(about 3.7 GB for `aizk` after HNSW and GIN, my estimate), `max_wal_size` 4 GB, a WAL archive in `<center-state>/backups/pg/wal`,
base backups, dumps, `aizk_stage` for U16 and the objects on this volume. With `wal_level replica` and archiving on, the 2.7 GB restore
and its index builds write several GB of WAL that must be archived before it is recycled. If anything fills the volume,
`archive_command` fails and `pg_wal` grows until PostgreSQL stops; PostgreSQL cannot cap that, so the plan's "hard cap" is only a doctor
row, and the stop takes `mainboard` control down with AIZK. `critic-simplicity.md` findings 5 and 14 reach the same risk; the trend and
the restore-WAL arithmetic are new here.

Change:
- Put PGDATA and `pg_wal` on a dedicated APFS volume with a reserve and a quota (`diskutil apfs addVolume ... -reserve 20g -quota 40g`;
  check the flags) so other writers cannot starve it.
- Take archiving out of the bulk restore: restore into a stand-alone staging cluster with `archive_mode off`, then take the base backup.
- Keep base backups and WAL off the internal disk, shipped from the settler's `ssh localhost` context.
- Set a numeric floor in the doctor (30 GiB free before staging), and find the 120 GiB consumer before L1.

### 16. Cluster-level PITR contradicts M38 and can leave the lake catalog pointing at files that no longer exist

Evidence: design.md "Backup": "Backups provide restore-point recovery, not a promise of zero-loss PITR", with pins held while files are
copied. The catalog's Parquet files live under DATA_PATH on `/Volumes/PORTABLE`, outside PGDATA. U12 offers `recovery_target_time` for
the whole cluster (base backup plus WAL). Restoring the cluster to time T restores the `ducklake_*` metadata at T while DATA_PATH is at
now; compaction and expiry since T may have deleted files the T catalog references.

Change: limit PITR to the `aizk` and `logto` databases (restore the cluster at T into a scratch instance, dump those two, load them into the live
cluster); for `mainboard` use only M38 checkpoint manifests. U12 acceptance gains a negative test: a PITR request for `mainboard` is refused.

### 17. Order against the waves: L1 precedes its own prerequisite, restarts hide in AIZK events, and capacity is not planned

Evidence:
- `unify.json` aligns U11 and U13 with M wave 3 and L1 "after the M36 live switch", but U13 depends on M24 ("service-manager seam"), which
  design.md places in Wave 4 with M19 to M23, M37, M38, M32.
- L1 restarts the center PostgreSQL "for archive_mode ... inside a drained mainboard switch". `archive_mode`, `max_connections = 120` and
  `shared_preload_libraries` (needed by `pg_stat_statements` in `aizk`) all need a restart, and W3 stages the cluster idle before W4 makes
  it control authority, so the restart belongs to M36's own configuration, not to a later AIZK event during the v26 queue.
- U16 and U17 require M37 though the `aizk` and `logto` databases do not depend on the catalog migration (`critic-simplicity.md` finding 17
  agrees).
- Calendar: L2 not before 11-17 is consistent with the MLSys deadline (10-30 20:00 UTC) and the miyabi stop (10-28 09:00 JST). I found no
  dependency between queued GH200 jobs and crimson's AIZK. U-wave 0 is 12 packages (11 in AIZK) beside the 39 mainboard packages, `unify.md` says the
  earlier Codex runs died when credits ran out, and the cutok v26 program draws on the same agents.
- The L0 build is "cut as a release" from a public repository (`phvv-me/aizk`). Pushes need the owner's authorization; the 10-01 bundle is
  the precedent.

Change:
- Split U13: statically rendered units that need no M24 seam can land in W3; the supervised start waits for M24.
- Render `archive_mode`, `max_connections`, `shared_preload_libraries`, WAL and `extension_control_path` in M36 from the first start.
- Gate U16 and U17 on M36 and M38 only.
- State a credit and calendar cut line: which of the 22 packages are required (findings 1 to 5 first) and which are deferred; no U-wave
  1 to 3 work before MLSys submission outside worktrees.
- Deliver L0 as a bundle or tag under explicit owner authorization.

### 18. GPU 0 returns to research in December at the earliest, and nothing in the plan helps the MLSys window

Evidence: `nvidia-smi` (2026-10-11): GPU 0 holds 20,210 MiB (two vLLM engine cores 10,008 and 7,562 MiB, one 2,620 MiB `python3`, GLiNER) at 0%;
GPU 1 holds 33 MiB. `[hosts.crimson.exports]` pins `CUDA_VISIBLE_DEVICES` to GPU 1's UUID with "GPU 0 hosts AIZK. Research must never displace
that service" and a status note. The plan frees GPU 0 at L0.5 (needs U16's go, hence U03 to U13 and M37), enrolls it in the lease at U18
(after L2, the 24 h running soak, 14 days stopped, and M30 in Wave 6), and keeps it reserved through L0.5's one-week rollback
(restarting 20 GB of lanes). Owner decision 4 makes registered exclusive runs refuse an unmanaged card. So GPU 0 is not usable for the
v26 window under any ordering of the plan.

Change: state the date in D10 and stop presenting GPU 0 as a benefit of the work before the deadlines. If the owner wants it sooner,
use the crimson-scratch L0.5 gate (finding 6), which can follow within days, and after the one-week soak edit `hosts.crimson` to expose
both UUIDs for non-exclusive work, with M30 enrolling the card later. Stopping `gliner` alone (D18) is immediate and frees 2.6 GB of VRAM and
3.8 GiB of RAM but no usable card.

### 19. U09's retention rule would erase the only evidence that 56 documents never got a graph

Evidence: the 79 rows left in `pgqueuer` are all `failed`: 56 graph builds, 10 insights, 10 profile projections, 3 community builds
(state.md F4); `pgqueuer_log` is not migrated; `raptor` last ran 2026-08-02. U09 "gives failed jobs a retention rule instead of leaving 79
failed rows". pgqueuer is the same 1.1.1 live and at HEAD, so the queue schema moves with the dump, and skipping `pgqueuer_log` data by TOC
entry does work. Picked rows after the restore need the recovery pass the plan already names.

Change: before any retention, requeue the 56 graph builds on the restored copy and record why they failed; make retention
"requeue once, then expire" with a count in `operator_snapshot`; add to the cutover verification a count of documents with no claims after
processing, equal on both sides.

## Acceptance criteria that cannot be measured as written

| criterion | why it cannot be measured | fix |
|---|---|---|
| "agents keep their sessions" (verification, T0 + 25) | one fresh session per harness; does not exercise stored credentials or open sessions, and the auth model in the plan is not the live one | finding 1 and 11 matrices |
| U16 MRR and recall@10 within 0.02 overall, 0.03 per class | margins below the standard error at n of 150 (about 10 per class) | finding 8 |
| U04 top-3 hit rate within 0.02 on 60 cases | one case is 0.0167 | finding 9 |
| find p50 not worse than 5,000 ms | 0.9% headroom over 4,954 ms, taken on an idle machine | +10% under load, n >= 200 |
| freeze at most 25 minutes "computed from measured steps" | extrapolated from a 346 MiB random-vector restore, measured on Wi-Fi before the wired port exists (D11) | re-measure on the final network from the L-2 archive |
| settler pass latency during restore at most 1.5 times idle | no PostgreSQL-settler baseline exists until W4 | define the baseline after W4 |
| "at most one acknowledged keep affected" | an expectation (0.1 to 0.4), not a test; the 300-kill property test proves server durability, not agent behavior | finding 12 |
| U13 units "restart after kill, start in dependency order" | stub servers and no launchctl in tests; real user-domain launchd, boot order and FileVault are untested | reboot drill with the owner (finding 14) |
| U03 recall@50 at least 0.98 per table | the dump fixture is trimmed to 5,000 rows a table; HNSW connectivity and scan caps differ at 77,000 rows | repeat at full size in U16 |
| doctor green for a week | the doctor runs on the center and cannot see a locked or down center | external watcher (finding 14) |
| rollback measured within 10 minutes | measured on a fixture; the real one includes the Cloudflare flip and the Logto dump | rehearse on a temporary hostname |

## What I checked and found sound

- Three databases with distinct roles, and the reasons given (hard-coded `public`, per-database extensions, short snapshot for M38).
- Placement on the center as one unit; the latency measurements are consistent with the ledger.
- Server-side "no acknowledged keep lost": snapshot at T0, no acknowledgements between T0 and the flip, a journal in the same transaction after it.
  The weakness is on the agent side (finding 12).
- pgqueuer 1.1.1 on both sides; the queue schema needs no migration, and the data-only skip of `pgqueuer_log` works by TOC entry.
- Relaxed iterative HNSW scan for RLS-filtered queries and the recall test at three selectivities in U03.
- The Docker VM estimate: 1.26 GB resident with no containers against the plan's inferred 1.25.
- The Logto leg itself: same hostname, same database, same container tag (`svhd/logto:1.41.0`) keeps the issuer and the signing keys.
- pg_dump 18.x archives restore into the center's 18.6.

## Order I would run

1. L-2: verified off-host copy of `aizk`, `logto`, roles and secrets, restored in the arm64 vendor container with timings.
2. Auth decision and the agent matrix on a staging server (findings 1, 11), before any HEAD-lineage code reaches production.
3. Owner role posture and the positive probes (finding 3) inside U00 and U11.
4. Golden set on scratch restores only, the A/B ladder, the lexical A/B at scale, threshold calibration (findings 6 to 9).
5. Choose the conversion path on the real archive (finding 5); recompute the freeze table.
6. L0a only if the stack must be recreated before the freeze; otherwise skip L0 and treat F1 inside the freeze.
7. Disk reserve, wired Ethernet and availability decision (findings 14, 15), then L1, then L2 in the night window after 11-17.

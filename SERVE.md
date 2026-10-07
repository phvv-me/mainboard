# mb serve: mainboard and AIZK as one system

Plan of 2026-10-06. Phase 0 can run before the deadlines once the owner approves it; the rest
waits until after November 16 (CVPR).

## Goal

One self-hosted system. mainboard runs the work: environments, jobs, hosts, costs and the lake.
AIZK remembers it: notes, decisions and sources. Both sit behind one login, one PostgreSQL and
one lake, and are reached from the CLI, from agents over MCP, and from an app.

## Starting point

- **mainboard.** A CLI that ships to every job host. It keeps a DuckLake with a SQLite catalog in
  `.mainboard/lake.sqlite` and Parquet in `.mainboard/lake/` (35 GB).
- **AIZK on crimson.** 21 containers:
  - Data: PostgreSQL 18 with VectorChord, SeaweedFS, Logto, Caddy, oauth2-proxy and cloudflared.
  - Observability: Tempo, Loki, VictoriaMetrics, Alloy and Grafana.
  - Services: Docling, ClamAV, GLiNER2, and two vLLM lanes, Qwen3-VL-Embedding-2B and
    Qwen3-Reranker-4B.

  The containers use about 32 GB of RAM. The GPU holds 20 GB at 0% utilization. Extraction runs on
  OpenRouter (DeepSeek V4 Flash), and its total spend since the start is under $3.
- **Load.** Over 30 days: 836 finds, 967 keeps and 73 files.

## What 168 operator reports say (2026-08-07 to 2026-10-06)

164 of the 168 are about `find`. The counts below are keyword matches, so they overlap:

| Failure | Reports | Cause in the code |
|---|---|---|
| A named document or an exact ID is not returned, or comes back in fragments | ~74 | `find` matches titles, never document IDs, and there is no read-by-ID path |
| A stale note outranks its correction | ~66 | recency weight 0.1 with a 30-day half-life (`config/settings.py:560`); corrections do not retire derived claims (the cutok "part of reproducibility" claim has survived since Sep 8) |
| "Current X entities are ..." inventories fill the budget | ~37 | the overview lane is not capped |
| A generic one-word title wins ("latest", "beyond", "review", "stable", "artifact") | ~32 | `Document.named_in_query` (`store/models/tables/document.py:210`) gives first place to any title of 3 or more characters that appears in the query, ahead of the reranker (`retrieval/rerank/rescore.py:58`). 411 of 3,372 documents have one-word titles: 28 "artifact", 8 "latest", 3 "stable" |
| Operator-only reports surface in `find` | 22 | the owner is the operator, and every agent acts as the owner. Reports about failed finds then become top evidence for later finds, against the `report` tool's contract |
| Bulk imported docs (the July 19 Ruff, CodeQL and DataHub imports) crowd out private notes | ~18 | there is no prior favoring authored notes over imported ones |

Retrieval is AIZK's main defect, not its infrastructure. The most common question is "what is
the current state of project X", which is structured (project, kind, newest) rather than semantic.

## Phase 0: fix retrieval (AIZK only, about a week)

1. **Golden set.** Turn the reports into a regression benchmark. Most reports name the query, the
   expected document IDs and the distractors that won. Score recall@k and MRR, run the benchmark
   as an `mb` job, and keep its receipts in the lake.
2. **Title authority** applies only to distinctive titles: two or more words, or an identifier,
   and unique among documents.
3. **Document IDs** named in a query get direct authority and return the full source. Add a
   read-by-ID path for whole documents (TODO ledgers, handoffs).
4. **Reports scope** is excluded from `find` unless the caller names it.
5. **Supersession.** Prefer the newest per (project, kind), where kind is DEV LOG, TODO or
   handoff. Use stronger recency when the query says "current" or "latest". Corrections close the
   claims they refute.
6. **Overview lane** is capped at a fixed share of the budget.
7. **Source prior.** Authored private notes rank above organization notes, which rank above
   imported docs.

Deploying this restarts the crimson stack. Five containers still mount the deleted
`~/projects` path, so recreate the stack from `~/aizk` first (pending since Oct 1).

## Phase 1: slim the stack

- **GLiNER.** Drop it as the extraction gate, since extraction costs pennies. Keep an
  independent local mention detector for the web-egress sanitizer, run on CPU or on Apple MLX.
- **Embeddings.** Either OpenRouter `qwen/qwen3-embedding-8b` ($0.01/M; re-embedding about 28k
  items costs cents and needs the golden set to confirm recall), or the same model run natively
  on the M4 Pro. Docker on macOS has no GPU.
- **Reranker.** Run it natively on the M4 Pro, since OpenRouter lists no reranker.
- **CockroachDB.** Drop that backend, keeping PostgreSQL as the one database.
- **Observability.** Keep the five-container stack only if the RAM budget allows. Otherwise
  structlog events go to the lake.
- **Target.** Under 10 GB of RAM with no GPU, measured after the change.

## Phase 2: one data layer

- The DuckLake catalog moves into AIZK's PostgreSQL, as a separate database on the same server.
  This is the catalog DuckLake recommends for several writers. Parquet stays on the center's
  disk, or moves to SeaweedFS when job hosts write directly.
- AIZK keeps its transactional data in PostgreSQL: the queue, the graph, RLS and vectors. Its
  append-only history (usage and events) moves to DuckLake tables.
- Job hosts write receipts straight to the lake when they can reach it. The local spool stays the
  fallback, because acquisition must never depend on the catalog.

## Phase 3: one login and one API

- `mb login` signs in to AIZK's Logto (device flow). The CLI reaches `mb serve` with that token.
- MCP gains mainboard tools beside `find` and `keep`: job list and show, hosts, costs, and lake
  queries.
- mainboard writes structured DEV LOG entries (project, kind, run handles, commit). "Current state
  of X" is then answered from the lake plus the newest note per project, with no search.
- Thread handoff between Claude, Codex and OpenCode is the "workstreams" item already in AIZK's
  ROADMAP, built on herdr sessions and ACP.

## Phase 4: the app

The jobs, hosts, costs and lake views go into AIZK's existing web frontend first, served through
the same Caddy and cloudflared. A SwiftUI app (iOS, iPadOS, macOS) follows only after those views
prove useful.

## Packaging

- The mainboard CLI stays light. It ships to every job host and to Miyabi's login node, which has
  a 14 GB memory cap.
- AIZK stays its own package and product (PyPI, plugins). `mb serve` is a deployment of AIZK plus
  mainboard's API, installed as a `mainboard[serve]` extra. Merging the repositories is a later
  decision.

## Where it runs

After Phase 1, measure its RAM. If it fits in about 10 GB, it moves to the Mac mini, with Docker's
disk image on `/Volumes/PORTABLE`. Otherwise it stays on crimson. cloudflared serves the public
route from either.

## Owner decisions

1. Start Phase 0 now, including the crimson redeploy and the stack recreation it needs?
2. Keep AIZK a separate product that mainboard depends on, or merge the repositories?
3. Drop CockroachDB support?
4. Prune crimson's 223 GB of Docker build cache? Its disk is 90% full.

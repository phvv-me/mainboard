# What next for mainboard (2026-10-23)

The one live plan. Older plans were deleted on 2026-10-23 and git history keeps them: REDESIGN, UNIFY, SERVE, FIXES,
design v3 with its owner decisions, the AIZK unify plans v1 and v2, the mcmr port audit, and the manifest and
command-map proposal. What they left open is below. Open bugs are in `bugs-ledger.md`, and incidents in `../ISSUES.md`.

## Owner decisions in force

- **2026-10-23:**
  - Memory stays in AIZK for now: fix its bugs and clean it up.
  - A later `mb memory` stores only in DuckLake and PostgreSQL, never in markdown files.
  - ai-memory is not a dependency and LEANN is not used (see "Memory, later").
  - Use more PostgreSQL where rebuilding something inside mainboard would be hard.
- **Still in force from 2026-10-10/11:**
  - PostgreSQL on the center holds the DuckLake catalog and the job-control state, through SQLAlchemy Core with
    psycopg 3.
  - One host-wide card lease that every user of a card takes.
  - Waves merge by drain and switch after the full gate.
  - The center's data lives on /Volumes/PORTABLE, with no watcher.
  - Logging costs nothing inside timed regions.

## Order of work

1. **Bug ledger.**
   - Re-check the 22 open findings and fix each with a test (B02 is mitigated; B08 awaits the owner's call on
     what `mb lint <dir>` fixes). B21, B41 and B42 close with step 2.
   - Then run the safe autofix of `mb rules` over mainboard (291 findings in 91 files) as one reviewable change, and
     convert its 85 `print` calls to the logger.
2. **One parse of every submission (`SubmissionPlan`, `SubmissionId`).**
   - Submit, lane and batch build one validated request, carried to the host unchanged.
   - Records, pulses, captured logs and receipts are keyed by the submission, not by the scheduler's handle. This closes
     B02, B21, B41 and B42 at the source.
   - The CLI splits into one module per noun as each noun moves onto its request type (`job` first), so its verbs become
     plain functions with tests.
3. **The job-state core.**
   - JobReducer and JobStore in the center's PostgreSQL. It is the one place a lifecycle transition and its effects are
     decided, which closes B13 and B16 and the cancel and settle disagreements.
   - The DuckLake catalog moves to the same server.
4. **Apps.** `mb app run`, `results`, `diagnostics` and `devices` replace the `apple` verbs, and the 22 remaining usagi
   tasks move into `[apps.usagiunagi]`.
5. **Commands the week's agent history asked for** (30,142 commands surveyed on 2026-10-23):
   - `run --env VAR=value`;
   - `host status <host>` (queue, GPU processes, disk) and `host cache [--prune]`;
   - `outdated`;
   - `query` over files as well as lake views;
   - `build` over apps, papers and members.
6. **Manifest.**
   - Infer `platforms` and `[system]` from the fleet, recorded in the lock.
   - Drop the unused `rapidsai` and `nvidia` channels.
   - Add a Python 3.15 leg to mainboard's and patos's gates once cupy and torch ship cp315 wheels.

## Patterns behind this week's bugs

| Pattern | Bugs | The change that removes it |
|---|---|---|
| CLI verbs as closures, testable only by patching globals | B14, B39, B43 | step 2's request types and per-noun modules |
| records keyed by a scheduler handle | B02, B11, B21 | `SubmissionId` |
| an empty string meaning both "nothing" and "could not read" | B03 | typed outcomes at every remote read |
| one value type with two formatters | B28 | one owner per type (walltime, duration, size), found by ALL-DUPL0001/0003 |
| a file lock written in two steps | B22 | write, link, rename to reclaim, release by token (`trials/lease.py`), the one primitive for the spool and settlement locks too |

## Memory, later

When `mb memory` resumes, it builds on AIZK's PostgreSQL rather than replacing it:
- the DuckLake catalog lives in the same PostgreSQL server;
- AIZK's `usage_event` and keep history reach the lake through DuckDB's postgres scanner, as a cursor pull;
- lake queries can join a note to the run it describes, by run id.

Rejected on 2026-10-23:
- **Markdown as truth (ai-memory's wiki):** the owner wants DuckLake and PostgreSQL only.
- **ai-memory as a dependency:** it is a Rust server with its own markdown store, a parallel memory rather than a library.
- **LEANN:** it saves storage by recomputing embeddings at query time. AIZK's about 33k chunks need only about 135 MB of
  vectors, and its embeddings are hosted, so it would add a billed call to every find.

Ideas worth taking from ai-memory into AIZK:
- capture through agent hooks;
- a bounded brief at session start;
- upkeep with no LLM calls (decay by use, near-duplicate collapse, contradiction flags).

**Changes from design v2 — owner-decided revision 3, 2026-10-10.**

- Q01/Q02 are final: one center PostgreSQL server holds the DuckLake catalog and a separate control schema. SQLAlchemy Core uses psycopg 3; SQLite is retained only as a legacy input/backup codec, not a new control or card store.
- Q13 is final: drain-and-switch under the existing owner. Stop the settler and every effectful old controller, switch the center editable checkout as a complete reviewed generation, then restart. No separate principal, credential rotation, or claim that a new epoch restrains old code.
- Cut over now, wave by wave after gates pass. W1 delivers M01/M02/M03/M34 independently; W2 delivers M06/M10 before any PostgreSQL migration. Each wave names center and host payloads, drains, legacy PBS/pueue settlement and rollback. v26 GH200 jobs stay queued.
- Q05 is universal: every user of an owned card takes the same host-wide lease, including mainboard jobs, refactor gates and agent runs. There is no development or gate exemption. Unmanaged exclusive runs are refused.
- Add M36 for the workspace-environment PostgreSQL service, M37 for a history-preserving SQLite catalog migration, and M38 for pg_dump plus data-file backup inside lake replicate. Move M32 core drain/restore rehearsal before the first PostgreSQL live switch.
- M19/M22/M23/M24/M30/M32 and their callers now use PostgreSQL transactions, observable advisory sessions, explicit connection-loss recovery and a durable outbox across SQLAlchemy and DuckDB commits. A released advisory lock alone never proves a worker or GPU is gone.
- M00 grows an opt-in-by-fixture throwaway PostgreSQL harness from environment binaries, without delaying the first safety gates. PostgreSQL-owning package acceptance requires real temporary PostgreSQL, not SQLite emulation or mocked locking.
- Current primary docs support PostgreSQL catalogs, inlining and native maintenance. No documented one-command history-preserving SQLite-to-PostgreSQL catalog conversion was found; M37 is an explicit version-qualified catalog export/import with exact type, snapshot and time-travel gates.
- The 36 original package IDs, 55 B mappings, 56 issue dispositions, 118 closure entries, eight abstractions and critique record remain. The plan now has 39 packages. No finding becomes fixed, deployed or historically recovered merely by this revision.

The source of authority is [owner-decisions.md](owner-decisions.md). [design-v3.json](design-v3.json) retains the v2 schema plus `decisions`; it preserves the original issue and closure records and adds M36–M38. This is a design-only run at the c97cb81 baseline. Source, Git metadata, live environments, hosts and lake state were not changed by this run. The lead applies the now-approved incremental deployment policy; package agents remain in temporary workspaces.

**Full plan and ownership.**

Keep one resolved submission plan, sealed SourceSnapshot, complete EnvironmentSolution, JobReducer/JobStore and effect coordinator. One PostgreSQL server on the center holds transactional control and DuckLake metadata in separate schemas; the external volume retains bulk data. Hosts retain schedulers, immutable execution bytes and offline receipts. The same-user drain is an operational deployment discipline. It does not claim to prevent a person deliberately starting an old binary afterward.

| Owner | Responsibility |
| --- | --- |
| JobReducer / JobRecord / JobStore | JobReducer.reduce alone decides lifecycle and effect eligibility; JobStore atomically persists its result and identity under a version check. |
| JobCommand / SubmissionPlan | Parse command structure once, complete preflight once, then use the same typed request for direct, lane, batch and remote execution. |
| SourceSnapshot | Own the exact immutable bytes imported or executed, and establish whether those bytes are approved for this acquisition. |
| EnvironmentSolution | Extend existing Solved/lockfile ownership into one complete EnvironmentSolution, with native solve-only versus frozen install and recoverable publication. |
| Host / Transport / ResourceEnvelope | Normalize capabilities, resource admission and bounded external effects while retaining upstream scheduler ownership. |
| EvidenceLake / EvidenceDelivery | Keep immutable receipt bytes and artifact custody distinct from logical paths and analytical projections. |
| EffectCoordinator / incremental settlement | Collect indexed changes, call JobReducer.reduce and execute its durable intents under one observable owner. |
| CardLease | One host-wide card lease covers every owned-card user: jobs, refactor gates and agent runs, for the whole compute process tree. |

**JobReducer / JobRecord / JobStore — contract.**

- SubmissionId allocated before side effects; workspace_id and HostId; scheduler handle + allocation/generation are external references, never primary keys.
- Immutable command/source/environment/resource/evidence-policy references; version for compare-and-swap; observed_at and observation_source.
- Execution phase, computation outcome, cancellation intent, evidence stage, release stage, freshness, typed blocking reason, pulse keyed by SubmissionId.
- Append-only lifecycle events, idempotency keys, effect receipts and transactional outbox in PostgreSQL control; constrained indexed lookup and complete database backup.
- Observation authority, producer sequence and terminal conflict handling are defined before writer conversion; reducer is a domain method, not a generic state-machine framework. Reuse patos.Lifecycle only for table-edge checks.
- Execution transitions planned → accepted → queued → running → exited, with rejected/cancelled/lost alternatives. Cancellation is requested → acknowledged → observed, never inferred successful from a missing SSH reply. Unknown scheduler states stay unknown/nonterminal until reconciled.
- Computation outcome is pending only before an exit observation; after exit it is succeeded/failed/cancelled/unknown with a reason. Evidence separately progresses required → collecting → verified or blocked. Settlement completes only when required delivery and release obligations close. Failure with verified partial evidence can settle.
- JobReducer.reduce is the sole transition/effect function. Match attempt identity first; verified producer close/exit proof outranks scheduler terminal proof, which outranks running observations. Within an authority use producer sequence, then source time; an equal-rank contradictory terminal proof produces conflict/blocked custody, never arbitrary timestamp tie-breaking. A matching later proof may refine unknown to known; stale running cannot regress exit. Compare semantic projection and destructive-effect counts, excluding observation arrival metadata and record versions.
- A short PostgreSQL transaction writes record, event, effect intent and outbox together. DuckLake projection commit/ack is separate and recoverable; expose projected_through and lag.
- UUID reuse is impossible within the store; handle reuse across hosts and attempts is normal and tested. Ambiguous legacy rows are quarantined, not guessed.
- Persist complete plan, pre-submit retention pins, unique request token and intent before each scheduler/provider effect. Acceptance with lost reply is unknown; reconcile by stable token, or retain unresolved without resubmission. Holds retain SubmissionId/plan; retries change only allowed allocation facts.

**JobCommand / SubmissionPlan — contract.**

- SubmitOptions, target kind, RunnerOptions, exact argv tuple or explicitly requested ShellScript.
- Separate syntax, pytest case enumeration, and side-effect-free trial selection/admission using target facts, input manifest and evidence watermark; carry policy and selected identities through runtime.
- A deterministic serialized transport document and digest; child lane plans change only lane-owned fields.
- Outer options must appear in the outer segment; target and payload boundaries are explicit. Unknown/misplaced submit options, duplicate owned delimiters and zero-selected native tests fail before mirroring, rental allocation or scheduler submit.
- ExecArgv is opaque; pytest owns native payload grammar, options contributed by plugins, and collection. Extend the existing pytest collector to use the exact argv and sealed config/plugins in a compatible bounded environment; unavailable target-compatible collection is a named refusal, never a fabricated proof.
- Nested -- tokens are owned by the segment that consumes them. Payload --help belongs to payload; quoting is a transport concern applied once, never a semantic parse step.
- Registered acquisition defaults to a new attempt. Reuse is an explicit plan result with counts and prior receipt identities, never a queue job that silently runs zero cases. --fresh in a consumer and mainboard's --rerun are not conflated.
- Wait every lane to its requested completion before aggregating exit status; one failure cannot short-circuit observation of remaining handles.
- Evolve Request as the unresolved boundary and add one resolved SubmissionPlan in vocabulary.py. Lane derivation preserves all fields by validated construction. Held retry never reloads defaults, source or solution from today's worktree.

**SourceSnapshot — contract.**

- One SourceSnapshot whose identity is the sealed manifest digest; optional Git/submodule origin proofs do not select a different publication path.
- Capture declared first-party package roots in full by default, including mainboard, namespace portions and pytest plugins. Narrow scientific declarations retain one closure/guard implementation; dynamic dependencies require declarations.
- Read-only source/input bundle plus a distinct per-SubmissionId execution directory and immutable mount table; no active or queued snapshot link is relinked.
- Detached ApprovalRef binds acquisition ID, source manifest digest, solution digest, input digests and policy version outside the hashed bundle; explicit immutable approval creation/selection is owned by M11/M13.
- Capture to immutable staging, verify the entire listing before publication, and transfer from that staging only. Never reread a worktree after sealing. Registered submission compares the complete captured bundle to the approved registration.
- A Git commit without its editable dependency identities is not a complete source identity. A digest verifies bytes without requiring fictitious commit or worktree_dirty values.
- Mounts have disjoint source/output ownership. Reject ambiguous parent-child overlap and symlink escapes before linking. Pins name digests, not only paths.
- GC is reachability over durable submitted jobs, queued scripts, active held claims, running processes and retained receipts. Missing scheduler inventory means unknown, so retain.
- Mutation tools use the same artifact ownership inventory and refuse writes to registered/frozen inputs. Mutable worktrees do not become implicitly frozen merely because an older snapshot exists.
- Declared resource directories include all immutable members, including members ordinary source filters ignore, or preflight enumerates and refuses exclusions. A declared file and that file inside a declared directory have the same inclusion rule.
- An approval is immutable, owner-selected and never silently refreshed by capture/submit; editing the working tree changes the actual digest and fails the existing approval.

**EnvironmentSolution — contract.**

- Normalized manifest input fingerprint, platform matrix, manager/version identities, Pixi lock and every native second-stage lock.
- Resolved distribution providers and source identities; immutable generated artifact manifest; install receipts for addressed prefixes.
- Staged solve transaction, recoverable publication journal, preview diff and current-generation pointer.
- Ecosystem.resolve and install(solution) capabilities for npm/pnpm/yarn/bun, Go and Rust, including target platform limitations and separate disposable tooling; unsupported native forms are refused.
- lock resolves all stages without installing the project prefix. install consumes a complete solution; default install can solve stale inputs, while frozen/locked host install cannot resolve. Existing-prefix run checks every stage fingerprint.
- Solvers execute in disposable staging. Every lock, generated input and manifest change becomes visible through one generation pointer/journal recovery; failure or interruption leaves the previous usable generation current, including when no old lock existed.
- Upgrade is a native solver operation plus explicit policy for stable/non-yanked releases, provider ecosystem and all declared platforms. Calendar versions do not receive fabricated semantic-major upper bounds.
- Remote installation replaces an owned artifact set exactly; it cannot retain retired generated files. No host re-solves or changes a prefix used by running jobs.
- An existing interpreter is accepted by capability and version. Bootstrap is needed only when that capability is absent.
- Provisioner.run/refreshed, nested mb run, activated shells, console scripts and editable .pth metadata select one immutable generation. No reader silently reconciles a prefix still in use.

**Host / Transport / ResourceEnvelope — contract.**

- HostId, transport endpoint, scheduler/provider capability record, runtime release identity, physical GPU identities and observed readiness.
- Evolve dispatch.vocabulary.Resources into the sole ResourceEnvelope; JobSpec and scheduler/provider adapters render it without repeated resource fields.
- ProcessScope for local child/group lifetime; checked transport results distinguishing success, refused, unavailable and unknown.
- Graceful drain versus cancel operations, durable remote tokens and artifact retention.
- Admit only jobs that fit a real allocation including shutdown grace. Duration formatting preserves 24+ hours. The exact same envelope renders CLI flags and script headers.
- release/drain stops new admission and waits for claimed work; cancel targets a specific submission and persists a fence. A failed qdel or spool write is never reported as confirmed cancellation.
- PBS JSON or a tested text adapter yields observations tied to SubmissionId. Logs, exit records, autopsy and sampler framing use the attempt namespace; a bare recycled scheduler handle cannot supply terminal proof.
- Control calls have deadlines; bulk transfer has resumable progress and overall budgets; actual jobs have runner/scheduler limits. Killing local SSH does not prove remote work stopped.
- Host sync stages a new runtime alongside pinned existing releases. Health names the actual executable and daemon generation, not just an answering socket.
- One absolute whole-job deadline spans provision, activation, attestation, capacity wait, execution and reserved cleanup grace. Owned children that cannot be confirmed stopped remain visible obligations; their GPU tokens cannot be released.

**EvidenceLake / EvidenceDelivery — contract.**

- Existing Blobs/Evidence owners expose verified object streams with byte and row/chunk windows. LakeId and resolved LakeHome select one writer authority through a bounded typed process protocol.
- Delivery manifest per SubmissionId, required logs/receipts/artifacts and per-item verified custody; explicit Empty/Unavailable/Unsupported transcript results.
- Versioned raw receipt codecs and typed normalized rows with import watermark; byte-preserving legacy readers.
- Conflict-aware materialization, exact-byte inventory, replica/backup manifest and source/environment reachability.
- Destructive custody decisions require complete chunk/end-to-end verification and a finalized producer or enforced writer coordination. Eviction quarantines the exact candidate, revalidates ownership and bytes, and never unlinks a successor pathname; uncoordinated in-place writers forbid eviction.
- Each SubmissionId has its own execution directory/mount table over immutable shared source/solution. Stage B cannot redirect a late path lookup by A. Delivery finality binds producer termination, a closed manifest and required members; a failed run may be incomplete-but-final without pretending scientific success.
- Restore defaults to report/refuse a changed destination. Boundary traversal checks actual components and disallows symlink races; materialization publishes only inside an owned root.
- Raw receipt bytes, hashes and verification algorithms remain immutable. Native JSON recursively round-trips strings, numbers, arrays and objects without temporal inference. SQL-native types have an explicitly separate projection contract.
- Data and custody manifests commit together in DuckLake where possible. Control-store outbox acknowledgment is a separate recoverable step; replay uses deterministic event/delivery identities and reconciles duplicates.
- Lake availability is an observed bounded result, never a permanent process-global cache. Read-only job status can remain available when an external volume is absent.
- Unknown/future formats are retained unverified without destructive acknowledgment. Old receipt identities use their recorded verifier algorithms, never new source/lock normalization.
- Every metadata access, attach, result iteration, commit and close occurs in the worker. Finite operations return operation ID, bounded typed data/artifact handle and complete/failed/unknown; no live connection, callback or lazy relation escapes.
- LakeId is persistent and canonical across symlinks/workspaces. The same authority grants ingest, local trial close, archiving, maintenance and projection writes. Served writes are disabled in this campaign; write=False cannot evolve a schema.

**EffectCoordinator / incremental settlement — contract.**

- Calls the M19 JobReducer; M23 adds cursors/changed-input acquisition, not another transition authority.
- One center coordinator with generation, PID/start identity, phase, started_at, heartbeat, deadline and queued/pending counts.
- PostgreSQL session advisory ownership, short transactional claims, durable effect identities/cursors/outbox and observable pg_locks/pg_stat_activity holder metadata.
- Supervised lake/transport workers; capture-before-release; cancellation/release reconciliation and startup recovery.
- Keep SQL transactions, row locks and file locks out of remote/volume I/O. A dedicated PostgreSQL advisory ownership session can span supervised work; passive readers never contend for it. Busy replies name holder, age, phase, deadline and last success.
- Only the coordinator grants effects; every publication/ack verifies generation. On owner death, supervised children must be confirmed stopped before a replacement can execute the same effect. TTL expiry alone is not fencing.
- If a kernel-stalled worker cannot be reaped, expose blocked ownership and withhold destructive work; do not pretend Python can guarantee killing uninterruptible filesystem I/O.
- External APIs that cannot deduplicate use durable intent plus query/reconcile-before-retry. A timeout after submission/release is unknown, never an automatic duplicate attempt. Exactly-once execution is not promised across arbitrary providers.
- Lifecycle steady-state work is bounded by changed jobs and new events, with zero object-byte reads for unchanged jobs. A legacy stream is decoded once when imported, not once per run.
- The durable service is tested under its real launch identity. Keep ssh localhost for the current launchd access route until a proven alternative is deployed; expose last successful pass separately from a live heartbeat.
- New grants require reconciliation of previous workers. Unmodified old controllers are drained and their launch routes disabled before the editable checkout switch; no credential/principal boundary is part of this design. Unsafe drain postpones only the affected wave/host.
- Early control facts may come from verified legacy codecs imported once; full normalized analytical tables are not a prerequisite for read-only status or supervision.

**CardLease — contract.**

- Center PostgreSQL resource reservations and advisory sessions, canonical host/GPU/MIG keys, host-wide launch supervisor and atomic multi-card ownership token; no per-workspace allocator database.
- Owner process start/boot identity, job SubmissionId, lifetime/deadline, crash recovery and explicit release.
- Universal admission and pending-capacity reason; exclusive registration refuses unmanaged occupancy or an unverified allocation boundary.
- Aliases, reordered masks, unset masks and overlapping UUID sets resolve to the same physical resources. Atomic acquire-all-or-none prevents partial multi-GPU deadlock.
- Publication is atomic; renew/release compare the ownership token. No former owner can remove a newer claim. Expiry never revokes a still-running process merely because it missed a heartbeat.
- Busy cards keep acquisitions pending before the first trial with a visible deadline. The runtime holds the lease through all workers and cleanup; fixtures consume the granted token.
- Every entry point joins the same lease-exec path, including interactive/agent/gate work. Device ACL/cgroup or exclusive scheduler allocation supplies stronger enforcement where available; polling never proves exclusion of an unmanaged user.
- Exclusive registration is accepted only where enforcement or an exclusive allocation makes the no-overlap guarantee true; otherwise submission fails with the concrete missing host capability.
- Legacy per-universe lease users are nonparticipants: drain them or retain an enforced allocation boundary before claiming exclusive acquisition.
- A disconnected center session releases its advisory lock but not its durable occupied/unknown card record. New grants wait for host process-tree reconciliation; live children keep the card unavailable.

**Current documentation and chosen behavior.**

Primary documentation was read on 2026-10-10. These are documented capabilities and design choices, not successful execution against the carried DuckDB development build. Exact version compatibility remains a named gate. The migration recommendation below is explicitly Mainboard’s design inference: the upstream logical-data migration example is not a history-preserving catalog converter.

**Catalog choice.** DuckLake recommends PostgreSQL for a multi-user lakehouse with potentially remote clients. One center PostgreSQL cluster and one mainboard database; control and ducklake_<LakeId> are separate schemas. Hosts continue to produce immutable files and receipts. [Catalog choice source 1](https://ducklake.select/docs/stable/duckdb/usage/choosing_a_catalog_database).

**Client attachment.** DuckLake accepts a postgres libpq connection string, METADATA_SCHEMA and READ_ONLY. The DuckDB postgres extension also attaches native PostgreSQL schemas; these are different APIs. Lake clients use the DuckLake attach; current job reads use SQLAlchemy directly, or a read-only postgres attachment for explicit analytical joins. Never query physical catalog tables as if they were lake tables. [Client attachment source 1](https://ducklake.select/docs/stable/duckdb/usage/connecting), [Client attachment source 2](https://duckdb.org/docs/current/core_extensions/postgres/overview).

**Catalog migration.** The documented DuckDB-to-DuckLake migration copies logical user data; it is not a promise to preserve an existing SQLite DuckLake catalog history. The specification describes catalog tables and backend-specific inline encodings. Implement a bounded version-qualified metadata migration, preserving IDs, snapshot ranges, inline tables and DATA_PATH; bootstrap target DDL with the matching extension and validate before activation. This is a Mainboard design inference, not an upstream migration command. [Catalog migration source 1](https://ducklake.select/docs/stable/duckdb/migrations/duckdb_to_ducklake), [Catalog migration source 2](https://ducklake.select/docs/stable/specification/introduction), [Catalog migration source 3](https://ducklake.select/docs/stable/specification/data_types).

**Data inlining.** PostgreSQL supports inline inserts/deletes; nested values use text encoding. VARIANT cannot inline with this catalog. Flush converts inline data to Parquet while retaining snapshot semantics. Explicitly disable inlining for blob payload tables and enable a tested small-row limit for supported narrow receipt/projection tables. Bound serialized row bytes as well as row count. Flush in supervised maintenance; never inline bulk evidence payloads onto the internal disk. [Data inlining source 1](https://ducklake.select/docs/stable/duckdb/advanced_features/data_inlining).

**Retries.** DuckLake retries nonlogical commit conflicts without rewriting data files and aborts logical conflicts. Current configuration documents max retries 10, initial wait 100 ms and backoff 1.5; page defaults can differ by version. Record exact versions and set retry limits explicitly within the operation deadline. Retry an entire idempotent application operation only after outcome reconciliation; never blindly repeat a submit or release. [Retries source 1](https://ducklake.select/docs/stable/duckdb/advanced_features/conflict_resolution), [Retries source 2](https://ducklake.select/docs/stable/duckdb/usage/configuration), [Retries source 3](https://ducklake.select/docs/stable/duckdb/advanced_features/transactions).

**Compaction.** Merge-adjacent-files preserves time travel and does not immediately remove old files. auto_compact controls eligibility, not a background scheduler. Keep native DuckLake compaction, with bounded file batches and before/after hash, snapshot and claim-reader tests under a PostgreSQL catalog. [Compaction source 1](https://ducklake.select/docs/stable/duckdb/maintenance/merge_adjacent_files).

**Expiry and cleanup.** Snapshot expiry is catalog-wide; expiry and physical deletion are separate. Cleanup must account for long readers. CHECKPOINT bundles flushing, expiry, compaction and cleanup functions. Do not run an unconditional CHECKPOINT. Select expiry candidates against retained proof, backup and read pins; use dry-run inventories and disable deletion during backup, migration or unresolved ownership. [Expiry and cleanup source 1](https://ducklake.select/docs/stable/duckdb/maintenance/expire_snapshots), [Expiry and cleanup source 2](https://ducklake.select/docs/stable/duckdb/maintenance/cleanup_of_files), [Expiry and cleanup source 3](https://ducklake.select/docs/stable/duckdb/maintenance/checkpoint).

**Advisory locks.** Session advisory locks last until unlock or session end; rollback does not release them. Transaction advisory locks end with the transaction. Advisory locks coordinate only participants and are visible in pg_locks. Dedicated nonpooled sessions own coordinator/card/backup lifetimes. Short transaction locks protect claim changes. Join pg_locks to pg_stat_activity and Mainboard owner metadata; session loss leaves uncertain operations blocked until reconciled. [Advisory locks source 1](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS), [Advisory locks source 2](https://www.postgresql.org/docs/current/monitoring-locks.html).

**Driver.** SQLAlchemy provides the psycopg 3 sync dialect as postgresql+psycopg. Psycopg supports C, binary and Python implementations backed by libpq. Choose synchronous SQLAlchemy Core with psycopg 3, supplied by the workspace environment with conda-forge libpq. Record the loaded implementation/library; use one SQLAlchemy pool for short connections and NullPool for ownership sessions. No ORM or second pool. [Driver source 1](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#module-sqlalchemy.dialects.postgresql.psycopg), [Driver source 2](https://www.psycopg.org/psycopg3/docs/basic/install.html).

**Environment-owned server.** conda-forge distributes PostgreSQL for macOS and Linux. initdb creates a cluster owned by the running user. PostgreSQL supports foreground startup under a service manager. Resolve a stable PostgreSQL release in mainboard.toml, freeze its binary prefix, and keep PGDATA on the internal disk outside environment prefixes. Never install a system server or initialize on an external lake volume. [Environment-owned server source 1](https://anaconda.org/conda-forge/postgresql/files), [Environment-owned server source 2](https://www.postgresql.org/docs/current/app-initdb.html), [Environment-owned server source 3](https://www.postgresql.org/docs/current/server-start.html).

**Service managers.** launchd ProgramArguments/KeepAlive supervise a foreground process; PostgreSQL documents systemd startup, with notify support depending on the build. Use a same-user LaunchAgent on macOS and systemd user service on Linux. Run absolute environment postgres directly; Type=simple plus a readiness probe is portable across builds. Retain the separate ssh-localhost settler path for external data access. [Service managers source 1](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html), [Service managers source 2](https://www.postgresql.org/docs/current/server-start.html).

**Backups.** pg_dump takes a consistent database snapshot and supports an exported snapshot. It is not a cluster-role or external-file backup. DuckLake documents catalog and data recovery separately and cautions about maintenance changing file layout. lake replicate creates one custom-format dump of catalog plus control, a file/proof manifest at the same database snapshot, and verified immutable data copies while maintenance is paused. Restore in a temporary cluster before marking the replica complete. [Backups source 1](https://www.postgresql.org/docs/current/app-pgdump.html), [Backups source 2](https://ducklake.select/docs/stable/duckdb/guides/backups_and_recovery).

**PostgreSQL service, clients, maintenance and backup specification.**

**Layout.** One center_id identifies one internal-disk PostgreSQL cluster. Use database mainboard, schema control and one ducklake_<canonical_LakeId> catalog schema per registered lake. control.lakes binds LakeId, metadata schema, DATA_PATH and authority; path aliases never create another owner. PGDATA defaults to <center-state>/services/postgresql/<major>/data, resolved and checked on the internal disk; socket/log/config directories are adjacent, never in a disposable environment prefix. Parquet and blob bytes stay beneath the configured external DATA_PATH. Unix socket directories must fit platform sockaddr_un limits: use a short center-ID directory in the private runtime/temp area with mode 0700 when an adjacent path is too long, and record/clean only the owned directory.

**Environment.** Declare conda-forge postgresql/libpq and psycopg 3 plus SQLAlchemy in workspace-owned dependency inputs. Select latest stable solvable releases at implementation, record exact locks and loaded library paths on macOS/Linux. No guessed version from current web metadata becomes a pin. Extend environment publication to retain the running server binary generation; environment upgrade cannot replace binaries under a live server. Check PG_VERSION before every start; major upgrade uses a separately rehearsed dump/restore or pg_upgrade, never automatic initdb or an unchecked new major.

**Bootstrap.** Mainboard owns idempotent center service prepare/start/status/stop operations. Resolve initdb/postgres/pg_ctl/pg_isready/pg_dump/pg_restore from the frozen workspace environment. On a genuinely new empty directory, run initdb -D <data> --encoding=UTF8 with an explicitly tested locale and secure local authentication. Refuse a nonempty/unrecognized directory, external/symlink-escaped PGDATA or another postmaster. Same OS owner, restrictive directory/socket permissions, loopback only where TCP is enabled; secrets use protected libpq service/pass files, never printed DSNs. These are connection hygiene, not a new principal fence.

**Macos.** Generate a unique center-ID LaunchAgent with absolute postgres -D arguments, explicit environment, working directory, internal log paths, KeepAlive and throttled restart. It is independent of the periodic settler agent. Prove startup/readiness under its actual service environment and retain user-login/session limitations in doctor. PostgreSQL needs no external-volume access for normal catalog service; the DuckDB worker still does, so keep checked ssh localhost for settler/replicate workers.

**Linux.** Generate a center-ID systemd --user service using absolute postgres -D, Type=simple, Restart=on-failure, bounded readiness/stop checks and PostgreSQL fast-shutdown semantics. Do not assume notify support in the conda build. Show whether user lingering is enabled when unattended operation requires it; no silent privileged configuration. Service-manager adapters reuse durable.py rendering/lifecycle seams without treating a server as a periodic settler pass.

**Driver.** Synchronous SQLAlchemy Core with explicit postgresql+psycopg dialect, psycopg 3 from the environment and libpq provenance check. Use schema-qualified SQL, short READ COMMITTED transactions with versioned conditional UPDATE and UNIQUE/CHECK/FK constraints, stable idempotency keys and RETURNING. Retry bounded SQLSTATE 40001/40P01 transactions by rereading inputs; use SERIALIZABLE only for an invariant requiring it. Pool pre-ping does not recover a lost transaction. Ownership connections are dedicated NullPool connections, never returned locked to a pool. Do not share live engines/connections across fork/spawn. Choose the environment-packaged psycopg 3 C implementation linked to its libpq when the platform solve supports it; record and gate the exact artifact instead of silently substituting psycopg2 or an untested binary wheel.

**Locks.** pg_try_advisory_lock on namespaced stable IDs selects one coordinator per center and one publisher per LakeId. A dedicated session can span an operation lifetime; no SQL transaction, row lock or file lock spans remote/volume I/O. pg_try_advisory_xact_lock is for short claim changes. Durable rows carry operation_id, generation, backend PID/start, local PID/start/boot, phase, deadline, heartbeat and outcome. Inspect pg_locks joined to pg_stat_activity/application_name and these rows. Configure bounded connect/statement/lock timeouts and liveness checks. PostgreSQL releases a dead session only when disconnection is detected; a stopped or partitioned client can remain visible. No lock stealing on TTL, no pg_terminate_backend without reconciliation. Never acquire a session advisory lock through DuckDB postgres_query or a pooled extension connection: ownership must remain on the dedicated psycopg session. The bounded read-only snapshot-export transaction in M38 is an explicit backup exception to short transaction lifetimes; control write transactions remain short.

**Connection loss.** On lost coordinator/advisory connection or server restart, stop new effects and attempt to stop owned workers; mark surviving operations/card reservations unknown. A replacement lock owner must reconcile every prior in-flight operation and prove surviving worker incapacity before conflicting writes/releases. The durable unknown row outlives the advisory lock. If that proof is unavailable, report blocked. This is required even when only a local process crashed; an advisory lock is not a kill switch.

**Transactions.** The same PostgreSQL server does not make SQLAlchemy control commits and DuckDB DuckLake commits one transaction. Persist intent/outbox in control first; commit delivery/projection with a deterministic operation/event identity in one DuckLake transaction; then acknowledge in control. A crash after either commit reconciles the lake operation marker before retry. Do not use PostgreSQL user-table writes to bypass DuckLake metadata semantics, and do not invent distributed two-phase commit.

Illustrative existing-lake/read-only attachment forms, verified by M22 against the exact extensions before use:

```sql
ATTACH 'ducklake:postgres:dbname=mainboard host=/absolute/center/socket' AS lake (METADATA_SCHEMA 'ducklake_example', DATA_PATH '/absolute/external/data', CREATE_IF_NOT_EXISTS false, AUTOMATIC_MIGRATION false, OVERRIDE_DATA_PATH false, READ_ONLY);
ATTACH 'dbname=mainboard host=/absolute/center/socket' AS current_jobs (TYPE postgres, SCHEMA 'control', READ_ONLY);
```

**Attachment policy.** Examples are templates for a registered existing lake. First initialization alone permits catalog creation with an explicit DATA_PATH. Writer attach omits READ_ONLY only inside M22 supervised publication. Preinstall locked ducklake/postgres extensions in the environment-owned cache; ordinary list/show must not download extensions. Existing attaches load stored DATA_PATH or compare it explicitly; override is a restore-only operation on an isolated destination. READ_ONLY plus application query restrictions prevents accidental mutations, not a hostile same-user process. Remote DuckDB clients also need access to DATA_PATH; a PostgreSQL catalog does not serve the Parquet files. Hosts do not attach the live catalog. After any catalog/schema migration, discard all DuckDB workers and native postgres schema caches and reopen fresh clients. Ownership locks stay outside duckdb-postgres connection pooling. Explicit control-query projections cannot expose arbitrary postgres_execute or connection-changing functions through a passive status query.

**Status.** List/show/current-status SQL are passive bounded PostgreSQL readers with observed_at, last successful pass, last attempt, owner/phase and projection watermark. No catalog migration, lake attach, service startup, host probe or reconciliation occurs as a side effect. With PostgreSQL down, return a bounded unavailable/stale response (a last-read display cache, if present, is labeled nonauthoritative). An absent external volume still permits live current-state reads when PostgreSQL is healthy. Refresh/wait explicitly enqueue targeted coordinator work and read progress.

**Inlining.** Set options explicitly for the exact tested extension. Initial policy: blob payload tables data_inlining_row_limit=0; narrow supported receipt/projection tables use a pilot limit of 10 rows with an application byte cap. These pilot values require M27/M38 measurement before raising them. Include null, nested JSON, strings resembling dates, unsigned extremes, decimal, timestamp and binary fixtures. No VARIANT inline assumption. Historical inline rows migrated from SQLite are retained exactly and classified; later verified flushing moves payloads to DATA_PATH.

**Maintenance.** Schedule explicit bounded flush/merge/rewrite operations via the same LakeId publisher, separate from lifecycle polling. Plan snapshot expiry globally against proof/reader/backup pins, and dry-run old/orphan file cleanup with a conservative age beyond every supported reader/writer/recovery window. Compaction may preserve logical history while changing physical files; raw proof retention must retain required exact bytes independently. Refuse destructive maintenance while a migration, backup, unconfirmed writer or unknown-format reachability exists. A PostgreSQL catalog removes SQLite writer contention, not external-disk stalls or file amplification.

**Backup.** M38 extends lake replicate: acquire a durable backup/maintenance exclusion, drain in-flight lake publication for a consistent handoff, export a REPEATABLE READ PostgreSQL snapshot, and use environment pg_dump --format=custom --snapshot=<id> --file=<staging> for the entire mainboard database. The same exported snapshot supplies control/outbox watermarks, every retained catalog file/delete-file reference and inline rows (in the dump). Keep the exporting transaction alive through dump/manifest creation; bound its lifetime. Resume append ingestion after snapshot establishment only if the fixture proves file immutability and maintenance remains blocked; otherwise remain drained. Copy and hash all referenced files, raw proofs, source/solution/runtime retention roots, configuration and a nonsecret role/bootstrap recipe. Pin copying files durably so lock/session loss cannot allow cleanup. pg_dump alone is never a successful lake replica. Publish a complete marker only after hashes and a temporary pg_restore plus DuckLake/claim readback pass. A failed copy leaves an incomplete resumable manifest and retained pins. Restore into a new cluster with effects disabled; never overlay active state. Backups provide restore-point recovery, not a promise of zero-loss PITR. Because one dump includes every registered catalog schema in the database, its completeness manifest must cover every included LakeId and DATA_PATH, or explicitly refuse a purported full-center replica. Restoring one selected lake is a separately scoped export and cannot claim to restore the whole center.

**Test harness.** M00 defines a lazy postgres_cluster fixture; M36 provides its environment dependency and implementation. From an isolated test root, run environment initdb, allocate a private Unix socket and collision-safe loopback port when needed, start foreground postgres under ProcessScope and wait boundedly with pg_isready plus SQL. Supply the DSN only to the test, allow only that local endpoint through the gate, and refuse live DSNs/PGDATA. Use real fsync/durability settings for crash gates. Kill/restart fixture postmasters and clients, retain diagnostic logs, stop/reap in finalizers; no launchctl/systemctl install in integration tests. Missing binaries fail PostgreSQL-required gates explicitly; the earlier safety suite can run without requesting this fixture. Run service-rendering tests on both OSes and separately require exact-service-context rehearsal before each actual deployment. Use a short private socket directory even when pytest/report roots are long; test that cleanup does not follow a substituted socket-directory symlink.

**Catalog migration recipe.**

1. Inventory the source DuckDB build, ducklake/sqlite/postgres extension versions, DuckLake catalog spec, exact catalog schema, tables, indexes, inline data/deletes and DATA_PATH. Record source catalog content/checkpoint digest, snapshot range, next-ID counters, retained proof and all file references. Unknown spec/column/table/type blocks migration; this is a finite converter for inventoried versions.
2. Prepare a consistent SQLite backup with the existing SQLite backup API, including committed WAL content, under drained legacy writes. Preserve the original file read-only and every referenced data/proof byte. Bulk staging may happen earlier from a consistent checkpoint, but its final delta must come from the final drained source; no naïve mtime or latest-snapshot-only copy.
3. Using the same compatible locked DuckLake extension, bootstrap the PostgreSQL target catalog DDL in an isolated candidate schema/database and disposable DATA_PATH, then close every DuckDB attachment. In one exclusive import transaction replace bootstrap metadata with typed source rows, reconstruct dynamic inline tables per table/schema version, and restore all IDs/counters/snapshot validity ranges. SQLAlchemy/psycopg COPY transports data; an explicit versioned mapping validates booleans, BIGINT/unsigned bounds, decimals, timestamps/time zones, bytea and nested-text encodings. Never infer PostgreSQL types from SQLite affinity or use a blind SQLite SQL dump/pgloader default.
4. Preserve data_file/delete_file paths, relative-path flags, partition/schema/column mappings, view definitions, tags/options, scheduled deletion references, inline row IDs and begin/end snapshot values, encryption metadata when present, and the canonical LakeId/DATA_PATH. Do not rewrite Parquet or raw scientific receipts just to move the catalog. Reject unsupported encryption/type/features rather than drop them. Matching-spec conversion and DuckLake spec upgrade are separate gates.
5. Validate table counts plus typed canonical row digests and every retained snapshot/old schema read, current jobs/claim readers and blob hashes. A candidate aimed at the original DATA_PATH is read-only until activation. Write/flush/compact/expiry/restart probes use a cloned temporary DATA_PATH so old and candidate catalogs never concurrently mutate shared files.
6. Import legacy current job rows and raw receipt identities via M19 into control, quarantine ambiguous handles and preserve deterministic submission mappings. Reconcile the final registry/receipt delta while controllers are drained. Candidate creation is not the point of live authority: one journaled center endpoint/schema switch activates it only after M38 restore and M32 switch witnesses pass.
7. Before the first PostgreSQL-authority effect, rollback stops/reaps the candidate and restores the legacy endpoint/tree over untouched SQLite plus its latest checkpoint. After new effects/writes, keep PostgreSQL authoritative and forward-fix or use a tested schema-compatible previous release; never re-enable the stale SQLite writer. Keep legacy input backups and codecs until proof reachability permits retirement.

**Common live switch for every wave.**

**Authority.** Owner decision: deploy each gated wave now via drain-and-switch under the same user. The design/implementation runs never edit the live checkout or real state. The lead performs the reviewable switch; no additional principal/credential approval question is introduced.

**Prepare.** Prepare an immutable copy/checksum of the reviewed candidate tree and all dependency artifacts off the live path. Inventory current launchers, nested environments/.pth, processes, host runtime references, service commands and queued job envelopes. Record prior tree, config/catalog endpoint and backup. Stage everything possible before drain.

**Drain.** Close the center admission window for controllers and prevent service auto-restart. Stop the launchd/systemd settler; wait for or stop all effectful mainboard processes, local trials that ingest/evict at close, notebooks with loaded mainboard, held keepers and their workers. Do not stop unrelated compute merely to simplify deployment. Prove each remaining process is an isolated file producer whose bytes and imports will not change. A surviving unknown writer/effect worker blocks this wave. This is operational drain by the single owner, not a security boundary against someone deliberately launching old code.

**Switch.** While drained, replace the center editable packages/mainboard tree as a complete generation, updating outer entry points, selected nested environment metadata and service configuration in one journaled switch. A filesystem atomic rename/pointer is preferred; otherwise the drain plus checksummed journal makes an interrupted whole-tree copy non-runnable until completed or restored. Never edit live modules one by one with controllers running. Clear inherited PYTHONPATH and prove outer/nested/shell/service imports. Start one settler only after readiness and last-pass gates, then reopen admission.

**Hosts.** Center rollout does not imply host sync. Every host receives only its wave-listed reviewed payload, staged alongside old runtime roots. Existing PATH-based queued PBS/pueue jobs retain their exact aliases/bootstrap/.pth/import roots until they drain or the delayed-start fixture proves isolation. New envelopes later use absolute immutable interpreters. A host whose shared root cannot be preserved waits; it does not cancel queued v26 GH200 work.

**Legacy jobs.** Every wave gate starts actual old generated PBS and pueue commands after the simulated switch, then settles their receipts through the current center. Preserve legacy target/host/handle/submitted_at identity and raw format verifiers; control migration records a deterministic SubmissionId map. Old jobs continue writing their shipped format in their run roots; new settlement can read both. Unknown finality or ambiguous identity stays pending. Never rewrite old receipts into fictitious new proof.

**Pause budget.** 60 s final-delta/entry-point switch budget after offline staging and successful drain, measured per wave. The owner approved the recommendation, not an unmeasured promise. If the final step cannot fit before activation, leave the old tool active and improve/rehearse staging. Drain time is visible separately and can defer a wave.

**Rollback.** Before activation/new effects: stop/reap candidate workers and restore the recorded prior tree/config/endpoint, keeping the latest consistent legacy state. After activation: first stop admission and drain; retain new receipts/jobs and use a tested schema-compatible previous release or forward repair. Never restore a stale database over new events, never infer remote undo from local restore and never restart a pre-PostgreSQL writer against post-switch state. Unknown external effects require reconciliation. Each wave specializes this rule.

**Implementation waves and immediate live adoption.**

The following order is serial within shared files, with semantic prerequisites distinct from integration conflicts. W1 packages are independently shippable micro-releases; W2 follows their safety gates and still precedes PostgreSQL staging. Each later wave also reaches the live tool when its own acceptance passes. Palette selection or an unready host never delays an unrelated completed safety switch. No final all-at-once cutover waits for the MLSys deadline.

**Wave 0: Isolated baseline and wave gate contract.** Bounded import/effect-denial baseline and actual legacy delayed-start fixtures exist; PostgreSQL fixture is lazy and universal card policy applies to any later GPU gate.

| Package | Primary bugs | Acceptance |
| --- | --- | --- |
| M00 — Establish bounded isolated execution and grow the existing gate | Supporting / issue-level | Deny external writes/connects, prove worktree import, classify baseline; grow scenarios with their owners. |

**Center.** No live center switch; deliver the tested gate and review manifest only.

**Hosts.** Hosts receive nothing; queued/running jobs remain untouched.

**Drain.** No live drain. Gate-owned processes are reaped at teardown.

**Legacy jobs.** Existing live settler remains in place; fake legacy scripts define the subsequent-wave test.

**Rollback.** Discard only temporary gate artifacts/processes. No live rollback is necessary.

**Wave 1: Immediate evidence, scheduler, retention and visibility safety.** Each package passes its focused regression plus delayed legacy receipt and drain/switch witness. Ship each package when green; the list is a serial micro-release order, not a requirement to wait for all four.

| Package | Primary bugs | Acceptance |
| --- | --- | --- |
| M01 — Require verified custody and safe destinations for evidence operations | B01, B09, B10, B49 | Corrupt chunks and replacement/in-place races never lose last bytes; inventory writes nothing. |
| M02 — Make scheduler cancellation, drain and waiting states truthful | B04, B13, B16, B18 | Failed/unknown cancel stays unconfirmed; drain preserves claimed work; Locked/Stashed stays waiting. |
| M03 — Protect every referenced source snapshot and prefix | B05 | Queued/held/offline work and historical proof references survive GC; unknown inventory retains. |
| M34 — Make the current settlement pass visible and replay each stream once | Supporting / issue-level | Temporary visible holder plus one replay per stream; no final B23/B32 closure and mandatory deletion. |

**Center.** For each green micro-release switch the center editable tree to the reviewed safety patch set, retaining current catalog/environments/receipt format. Resume the existing settler with M34 holder/freshness and replay containment as soon as M34 lands.

**Hosts.** No host install is required. Center-rendered future cancellation/retention commands carry fixes; do not sync legacy runtime or mainboard source roots used by old queued jobs.

**Drain.** For each micro-release stop the settler and drain center mb commands/workers and local ingest-at-close processes. Remote isolated producers continue. Unknown orphan holder blocks the switch until its process tree is resolved; do not unlink its lock.

**Legacy jobs.** The patched legacy settler still reads original receipts; custody/retention fixes preserve old queued/running data. Delayed PBS/pueue fixture and late fetch pass for every micro-release.

**Rollback.** Drain and restore the prior reviewed tree; no format migration occurs. Preserve every new receipt and state update and retain a replacement safety guard if rolling back would reintroduce destructive behavior.

**Wave 2: Immediate submit grammar and bounded process lifetime.** Known malformed submit is rejected with zero remote effects; native pytest parses positive plugin argv and unsupported preparation refuses explicitly. Whole process-tree deadlines pass.

| Package | Primary bugs | Acceptance |
| --- | --- | --- |
| M06 — Consolidate process lifetime and kill complete fresh-cell work | B38 | One deadline bounds setup through cleanup; owned descendants end or remain visibly blocked. |
| M10 — Parse one typed JobCommand at the submit boundary | B41, B50 | Reject malformed submit before effects; pytest owns full plugin argv; compatible selection has no paid side effects. |

**Center.** Switch the center editable checkout to M06/M10 after drain. Grammar checks run in the existing request path; complete SubmissionPlan/provenance comes later.

**Hosts.** No shared host install; new generated helper payloads use only negotiated legacy-compatible protocol. Old scripts retain existing imports. Host-only lifetime fixes stage separately and wait if a shared bootstrap cannot be drained.

**Drain.** Stop settler and drain center parsing/preparation/dispatch helpers and old mb descendants. Remote queued/running computation stays isolated and proceeds.

**Legacy jobs.** Legacy command envelopes continue through their recorded decoder; original producer deadlines are not rewritten. Both old PBS and pueue late receipts settle on the patched center.

**Rollback.** Drain and restore the prior compatible center tree, retaining receipts and submission identities. No database downgrade. Keep malformed-submit rejection if an unrelated lifetime change needs a forward repair.

**Wave 3: Stage PostgreSQL service and client foundations.** Temporary server, service rendering, exact JSON/query/session and resource gates pass. Exact live service-context verification is required before declaring the staged server ready.

| Package | Primary bugs | Acceptance |
| --- | --- | --- |
| M36 — Manage the center PostgreSQL service from the workspace environment | Supporting / issue-level | Temporary initdb/start/readiness/stop/restart succeeds using workspace environment binaries on macOS and Linux; service command rendering names the exact same frozen prefix and internal data directory. |
| M04 — Make JSON round-trips lossless at the analytical boundary | B20 | Nested JSON types round-trip exactly while native SQL values remain separate. |
| M05 — Use parsed query dependencies and explicit job columns | B52 | DuckDB-native reference detection handles literals/CTEs/quoted names; reads never create a lake. |
| M35 — Give relational sessions an explicit lifetime | Supporting / issue-level | All native relation owners close deterministically; historical shutdown proof stays pending if unreproduced. |
| M17 — Validate host capabilities and runtime provenance | B29, B33 | Actual runtime/daemon capability drives readiness; preserve old executable paths until drain. |
| M18 — Use one resource envelope for PBS and held admission | B17, B27, B28, B47, B48, B51 | One resource model renders all backends; 24+ h, held capacity and interactive paths agree. |

**Center.** Switch reviewed center foundation code and the staged environment generation; provision the same-user PostgreSQL service on the internal disk using environment binaries. The legacy SQLite lake and registry remain authoritative in this wave. PostgreSQL is staged with no production control/catalog import yet.

**Hosts.** Hosts receive capability probes and future resource-script rendering only; retain old runtime aliases and payload decoders. No host PostgreSQL installation or fleet-wide solve.

**Drain.** Drain center mb and the settler for the source/environment switch. Stop any candidate server before changing its binary generation. Existing legacy jobs retain old source/prefix roots.

**Legacy jobs.** The legacy settler continues against the existing lake; query JSON/column fixes use row-specific deltas. Queued v26 and old pueue work starts and returns unchanged receipt formats.

**Rollback.** Drain and restore prior center/environment, then stop the unused candidate PostgreSQL service while retaining its diagnostic data. Legacy state was never redirected; never delete or reinitialize a preexisting cluster.

**Wave 4: Switch control authority and DuckLake catalog to PostgreSQL.** Real temporary PostgreSQL control/concurrency/worker/outbox gates, complete pg_dump restore, exact catalog history migration, delayed legacy settlement and measured final-pause rehearsal all pass. Stage reviewed legacy-compatible host decoder entry points for new envelopes without modifying old roots.

| Package | Primary bugs | Acceptance |
| --- | --- | --- |
| M19 — Define JobReducer and persist its decisions in JobStore | Supporting / issue-level | Real temporary PostgreSQL proves atomic control decisions, observable advisory ownership and recoverable lake outbox; one reducer remains authoritative. |
| M11 — Use a complete SubmissionPlan for every lane and batch | B14, B39, B43 | Direct/lane/held/rented/local plans preserve all fields and approval policy; every lane is observed. |
| M20 — Carry submission identity through every lifecycle boundary | B02, B11, B21 | Every backend survives acceptance/ack loss; held retries and remote logs retain attempt identity. |
| M21 — Read one JobStatus from list, show, wait and SQL | B54 | Text/JSON/show/current SQL agree with PostgreSQL available even when hosts or DATA_PATH are offline; database outage is bounded and explicit. |
| M22 — Convert lake callers to a finite supervised worker protocol | B34 | Finite worker operations bound metadata through close; two workspaces and served clients cannot bypass LakeId authority. |
| M23 — Acquire settlement inputs incrementally through durable cursors | B32 | 1,000 unchanged jobs cause zero replay/object reads; one changed stream consumes only its new range. |
| M24 — Run one observable settler with PostgreSQL ownership and supervised effects | B23 | Dead sessions release advisory locks; surviving or uncertain effects stay blocked and visible; passive status and late legacy settlement continue. |
| M38 — Back up PostgreSQL catalog and control together inside lake replicate | Supporting / issue-level | A full lake replicate fixture restores one pg_dump containing control plus DuckLake catalog and all external referenced data into a second temporary PostgreSQL server; snapshot, outbox, job and evidence identities match. |
| M37 — Migrate the existing DuckLake catalog from SQLite to PostgreSQL without losing history | Supporting / issue-level | An inventoried SQLite DuckLake fixture migrates to a real temporary PostgreSQL catalog with identical retained snapshot IDs, table/column IDs, inline validity ranges, file references and raw evidence hashes. |
| M32 — Rehearse each drain-and-switch and restore the complete center | B07 | Core PostgreSQL restore and actual old-process drain precede activation; later waves extend the same legacy/rollback proof. |

**Center.** Switch the center editable tree, catalog endpoint/schema and current control-store selection together while drained. M37 activates PostgreSQL catalog plus control; M24 starts the single coordinator through the checked service path. list/show become passive. Retire M34 cache/lock bridge. Keep the original SQLite backup frozen as migration input, never dual-write authority.

**Hosts.** Old queued/running jobs receive no changes. New submissions use a versioned decoder/helper in a separate reviewed runtime directory; if a host cannot support that without altering old imports, new admission waits there while old jobs continue. Full immutable solution/runtime automation is M29.

**Drain.** Stop settler and every legacy center/local direct writer, served write endpoint, held control loop and effect worker. Inventory all old mb processes; unconfirmed remote effects or shared host control dependencies block affected activation. Take final legacy catalog/registry delta within the switch budget.

**Legacy jobs.** New settler imports deterministic legacy identity maps and retained codecs. It fetches and settles old SQLite-era PBS/pueue receipts once, preserving queued v26 jobs and old runtime bytes. Missing producer finality remains pending; migration never submits/cancels jobs.

**Rollback.** Before authority activation: stop/reap candidate, restore old tree/endpoint and untouched latest SQLite state. After first PostgreSQL-authority write/effect: do not restart old SQLite controllers; retain PostgreSQL, checkpoint current state and forward-fix or use a tested PostgreSQL-compatible predecessor. Reconcile unknown submit/release effects before retry.

**Wave 5: Complete environment, sealed sources, final delivery and immutable runtimes.** Complete native solution, detached approval, per-attempt mounts/finality, historical verification, PostgreSQL inline/maintenance and actual legacy/new runtime coexistence gates pass. Extend M32 restore/claim parity at each contract change.

| Package | Primary bugs | Acceptance |
| --- | --- | --- |
| M07 — Normalize dependency selectors and release schemes | B44, B45 | Every selector dimension combines correctly; calendar releases get no fabricated semantic cap. |
| M14 — Publish environment edits as recoverable generations | B25 | Every fault exposes an old or new complete generation, never mixed state or a repaired live prefix. |
| M15 — Make lock/install consume one complete EnvironmentSolution | B37 | Native capability matrix precedes implementation; all supported stages lock without project install and install without solve. |
| M16 — Delegate upgrade solving and add a real no-write preview | B31, B35, B36, B55 | Stable native solutions and exact preview locks apply or refuse unchanged; no registry-based solver. |
| M12 — Capture declared roots and make narrow closure guards agree | B15, B40 | Full declared roots and resource directories are complete; narrow closure/guard agrees with imports. |
| M13 — Seal and approve a complete immutable source bundle | B06, B19, B53 | One seal/approval path succeeds for approved bytes and rejects edited/mismatched bundles. |
| M28 — Require verified declared inputs before launching work | B42 | Missing or changed pinned inputs refuse before acquisition; restored digest and carried selection agree at runtime. |
| M25 — Make evidence delivery run-owned and release-gating | B03, B26 | Same-bundle runs never relink each other; closed manifests prove finality/custody before release or explicit cap loss. |
| M26 — Version receipt admission and normalize analytical rows once | Supporting / issue-level | Exact historical verifiers, claim readers and expected-delta parity pass; unknown formats remain retained/unverified. |
| M27 — Batch verified object access and bound materialization | B24 | Byte/row bounded verified streams pass tiny/large/duplicate fixtures and calibrated native I/O/RSS gates. |
| M29 — Install immutable runtime and environment releases on hosts | B30 | Actual delayed legacy scripts/keepers preserve imports; new absolute runtime and nested center paths use immutable generations. |

**Center.** Switch the reviewed center tree plus complete environment-generation selection. Keep PostgreSQL authority and additive/versioned schemas. New submissions bind sealed full source, solution, input identities and approved acquisition; new producers emit final manifests. Maintenance remains explicit and pin-aware.

**Hosts.** Per host, stage complete immutable runtime/environment releases and verified source bundles. New scripts name absolute interpreters. Preserve all legacy PATH aliases, mutable mirrors and prefixes referenced by already dispatched jobs; delay that host default switch if it cannot be isolated.

**Drain.** Drain center settler, submission/preparation processes and writers. On each host drain install/sync/keeper controllers that touch a shared runtime; isolated compute and queued scripts retain their original roots. Maintenance/backup ownership must be quiescent for its relevant changes.

**Legacy jobs.** New settler accepts both producer generations through recorded verifiers and identity maps. Old code need not emit a new final manifest; legacy finality requires termination plus a complete stable inventory. Late old receipts and NULL-Git proofs exercise full claim paths.

**Rollback.** Drain and roll back only to a PostgreSQL/schema-and-envelope-compatible release, keeping both old and newly referenced runtimes. Do not undo solver generations under running processes or remove raw proof/compacted file pins. Irreversible schema/history changes require forward repair or an isolated fully reconciled restore, never in-place stale restore.

**Wave 6: Universal card leases and remaining owned policies.** Job/gate/agent contention and center/host crash tests pass with temporary PostgreSQL. All owned-card entry points are inventoried/enrolled; unmanaged exclusive admission is refused. Q08 profile selection still gates only M09 and does not hold completed resource/lint deployment.

| Package | Primary bugs | Acceptance |
| --- | --- | --- |
| M30 — Make physical-card leases atomic and token-owned | B22 | Every job/gate/agent shares atomic physical-card admission; PostgreSQL session loss cannot free a live or uncertain card. |
| M31 — Reserve GPU capacity before trials and enforce acquisition policy | B12 | Capacity waits precede trials; exclusive acquisition needs enforcement and leases outlive compute children. |
| M08 — Make lint respect artifact ownership | B08 | All rewrite tools preserve pinned/approved bytes, including EOF normalization. |
| M09 — Validate declared plotting policies | B46 | Versioned numerical profile plus positive/negative minimum metric fixtures; unsupported keeps B46 open. |

**Center.** Switch the center lease/control/CLI and policy code. Enable resource admission only for host capabilities that passed enrollment. M08 and M09 can ship as separate gated micro-releases; no relaxed GPU bypass for tests or agents.

**Hosts.** Per host install the same lease-exec supervisor and immutable runtime for jobs, gates, agent commands and interactive users. Shared card keys live outside workspace roots and resource authority is PostgreSQL. Keep old queued PBS/pueue runtimes unchanged; reserve/exclude their cards from shared new work until their nonlease users drain.

**Drain.** Drain center controllers. Before enabling each host, drain existing nonlease GPU users/agent/gate runs and old keepers, or prove scheduler-exclusive isolation; retain active tokens and process-tree ownership while upgrading. Unsafe card or shared alias postpones that host, not queued GH200 jobs.

**Legacy jobs.** Old receipt decoding is unchanged. Legacy queued jobs keep their allocation/runtime and settle; their host remains in legacy-draining status until universal enrollment is possible. No new agent/gate work uses those cards outside the allocation.

**Rollback.** Drain new admissions and retain the lease supervisor and occupied/unknown reservations until all holders end. A rollback release must understand the lease token and PostgreSQL resource rows; otherwise disable new owned-card work and forward-fix. Never fall back to per-universe lock files or unleased GPU runs.

**Wave 7: Cumulative closure and post-wave recovery audit.** All 39 packages, original B/issue/critique obligations and nine incidents have verified evidence or named pending/excluded status; actual deployments remain distinct from code verification.

| Package | Primary bugs | Acceptance |
| --- | --- | --- |
| M33 — Verify completed removals and publish the rollout contract | Supporting / issue-level | Audit every bug/issue/critique and deletion; complete cumulative flows without disguising external/deployment gaps. |

**Center.** Publish closure/deployment/retention manifests. Any final removal affecting executable code is itself a drained reviewed tree switch; there is no single deferred grand cutover.

**Hosts.** Hosts receive removal only for runtimes with zero queued/running/held/receipt-verifier references. Historical proof codecs stay while retained evidence needs them.

**Drain.** No process drain for report-only audit. For executable removal, apply the same center/host drain and prove reference inventory complete; unknown liveness retains bytes.

**Legacy jobs.** Rerun delayed-start/late-receipt fixtures and full PostgreSQL restore after all changes. A real unresolved legacy job keeps its runtime and proof readers regardless of plan-completion status.

**Rollback.** Restore a removed executable from its retained immutable bundle only after verifying reference/schema compatibility. Evidence and current PostgreSQL state are never rolled back to make an audit green.

**Package-by-package implementation contract.**

The wave tables above state every package’s implementation objective and acceptance. The register below supplies semantic prerequisites and replacement boundaries. The full, versioned package records in `design-v3.json` contain every owned file, change, exact acceptance scenario, shared-file integration edge and proposed commit; they are part of this plan. All paths there are relative to the designated worktree and describe future implementation. Environment dependency edits in the live monorepo are lead rollout inputs, never implementation-agent writes. Estimates remain uncalibrated; file surface is not changed LOC or person-time. Each package leaves changes uncommitted and hands the lead one proposed logical commit.

**M00 — Establish bounded isolated execution and grow the existing gate.** Requires: none. Contracts: isolated fixture execution. Remove or replace: . Proposed commit: `test(mainboard): establish bounded isolated execution and grow the existing gate`.

**M01 — Require verified custody and safe destinations for evidence operations.** Requires: M00. Contracts: verified custody and exact-candidate eviction. Remove or replace: presence-only destructive fast path; inventory checksum backfill; path-later unlink. Proposed commit: `fix(mainboard): require verified custody and safe destinations for evidence operations`.

**M02 — Make scheduler cancellation, drain and waiting states truthful.** Requires: M00. Contracts: checked scheduler lifecycle results. Remove or replace: unchecked cancel success; unknown-as-terminal table fallbacks. Proposed commit: `fix(mainboard): make scheduler cancellation, drain and waiting states truthful`.

**M03 — Protect every referenced source snapshot and prefix.** Requires: M00. Contracts: host-readable retention manifest. Remove or replace: pueue-only liveness pruning. Proposed commit: `fix(mainboard): protect every referenced source snapshot and prefix`.

**M34 — Make the current settlement pass visible and replay each stream once.** Requires: M00. Contracts: temporary busy/replay bridge only. Remove or replace: . Proposed commit: `fix(mainboard): make the current settlement pass visible and replay each stream once`.

**M06 — Consolidate process lifetime and kill complete fresh-cell work.** Requires: M00. Contracts: ProcessScope and whole-job budget. Remove or replace: duplicate PID tree termination loops; unbounded runner helper wait. Proposed commit: `fix(mainboard): consolidate process lifetime and kill complete fresh-cell work`.

**M10 — Parse one typed JobCommand at the submit boundary.** Requires: M00, M06. Contracts: JobCommand and side-effect-free trial selection. Remove or replace: delimiter lookahead; punctuation shell inference. Proposed commit: `fix(mainboard): parse one typed jobcommand at the submit boundary`.

**M36 — Manage the center PostgreSQL service from the workspace environment.** Requires: M00, M06. Contracts: CenterPostgres configuration/lifecycle and environment binary provenance; postgres_cluster temporary gate fixture. Remove or replace: No existing PostgreSQL implementation is replaced; add the first environment-owned service without a system-install fallback; Assumption that every center service is a periodic settler pass. Proposed commit: `feat(mainboard): manage an environment-owned center PostgreSQL service`.

**M04 — Make JSON round-trips lossless at the analytical boundary.** Requires: M00. Contracts: JsonValue versus SQL value. Remove or replace: top-level-only string heuristic. Proposed commit: `fix(mainboard): make json round-trips lossless at the analytical boundary`.

**M05 — Use parsed query dependencies and explicit job columns.** Requires: M00. Contracts: parsed read capability. Remove or replace: SQL substring dependency test. Proposed commit: `fix(mainboard): use parsed query dependencies and explicit job columns`.

**M35 — Give relational sessions an explicit lifetime.** Requires: M00, M04. Contracts: owned native relation lifetime. Remove or replace: connection lifetime by interpreter finalization. Proposed commit: `fix(mainboard): give relational sessions an explicit lifetime`.

**M17 — Validate host capabilities and runtime provenance.** Requires: M06. Contracts: observed Host capability. Remove or replace: unconditional Python bootstrap; socket-only daemon readiness. Proposed commit: `fix(mainboard): validate host capabilities and runtime provenance`.

**M18 — Use one resource envelope for PBS and held admission.** Requires: M02, M06, M17. Contracts: Resources → ResourceEnvelope. Remove or replace: JobSpec duplicate resource fields; separate qsub/held resource serializers; gmtime durations. Proposed commit: `fix(mainboard): use one resource envelope for pbs and held admission`.

**M19 — Define JobReducer and persist its decisions in JobStore.** Requires: M00, M36. Contracts: JobReducer/JobRecord/JobStore/outbox. Remove or replace: Cache authoritative whole-record append/update transitions. Proposed commit: `fix(mainboard): define jobreducer and persist its decisions in jobstore`.

**M11 — Use a complete SubmissionPlan for every lane and batch.** Requires: M10, M19. Contracts: Request → SubmissionPlan; ApprovalRef. Remove or replace: partial lane/held Request reconstruction; mixed resolved/unresolved Request meaning; batch first-failure short-circuit. Proposed commit: `fix(mainboard): use a complete submissionplan for every lane and batch`.

**M20 — Carry submission identity through every lifecycle boundary.** Requires: M02, M03, M11, M18, M19. Contracts: SubmissionId at each effect/runtime boundary. Remove or replace: target/handle authoritative lookup; unrecorded ordinary submit; handle-only exit/log paths; quota key coalescing. Proposed commit: `fix(mainboard): carry submission identity through every lifecycle boundary`.

**M21 — Read one JobStatus from list, show, wait and SQL.** Requires: M05, M20. Contracts: shared JobStatus. Remove or replace: reader-triggered settlement/probes; independent verdict reconstruction. Proposed commit: `fix(mainboard): read one jobstatus from list, show, wait and sql`.

**M22 — Convert lake callers to a finite supervised worker protocol.** Requires: M01, M05, M06, M19, M35. Contracts: LakeId authority and finite worker messages. Remove or replace: public live connection/callback escape; permanent lake health cache; unfenced served write attach. Proposed commit: `fix(mainboard): convert lake callers to a finite supervised worker protocol`.

**M23 — Acquire settlement inputs incrementally through durable cursors.** Requires: M20, M22. Contracts: cursor/fact ingestion. Remove or replace: per-run receipt replay; M34 pass-local cache. Proposed commit: `fix(mainboard): acquire settlement inputs incrementally through durable cursors`.

**M24 — Run one observable settler with PostgreSQL ownership and supervised effects.** Requires: M06, M19, M20, M21, M22, M23. Contracts: effect coordinator/grant. Remove or replace: long-held settlement lock; M34 lock metadata bridge; pre-ownership rental expiry cancel. Proposed commit: `fix(mainboard): supervise settlement under PostgreSQL advisory ownership`.

**M38 — Back up PostgreSQL catalog and control together inside lake replicate.** Requires: M19, M22. Contracts: PostgreSQL-plus-DATA_PATH checkpoint manifest in existing lake replicate. Remove or replace: SQLite-only active catalog backup assumption; Catalog-only replica success and live PGDATA file copying. Proposed commit: `feat(mainboard): replicate PostgreSQL state with complete DuckLake data`.

**M37 — Migrate the existing DuckLake catalog from SQLite to PostgreSQL without losing history.** Requires: M19, M22, M24, M38. Contracts: Version-qualified CatalogMigration inside existing center migration orchestration. Remove or replace: lake.sqlite as the active DuckLake catalog after W4; File-path-only catalog identity and unchecked DATA_PATH override. Proposed commit: `feat(mainboard): migrate DuckLake catalog history to PostgreSQL`.

**M32 — Rehearse each drain-and-switch and restore the complete center.** Requires: M21, M24, M37, M38. Contracts: Versioned checkpoint and drain/switch journal shared by every live wave. Remove or replace: hard-coded Project.out-only migration inventory. Proposed commit: `feat(mainboard): rehearse incremental drain-and-switch with PostgreSQL restore`.

**M07 — Normalize dependency selectors and release schemes.** Requires: M00. Contracts: independent selectors/release policy. Remove or replace: exact-path selector guessing; calendar semantic-major cap. Proposed commit: `fix(mainboard): normalize dependency selectors and release schemes`.

**M14 — Publish environment edits as recoverable generations.** Requires: M00. Contracts: recoverable generation publication. Remove or replace: piecemeal contents-only rollback. Proposed commit: `fix(mainboard): publish environment edits as recoverable generations`.

**M15 — Make lock/install consume one complete EnvironmentSolution.** Requires: M14. Contracts: complete Solved and native resolve/install. Remove or replace: Pixi-only Solved authority; regex YAML syntax parser; resolve by project-prefix install. Proposed commit: `fix(mainboard): make lock/install consume one complete environmentsolution`.

**M16 — Delegate upgrade solving and add a real no-write preview.** Requires: M07, M15. Contracts: solved UpgradePlan. Remove or replace: registry version list as compatibility solver. Proposed commit: `fix(mainboard): delegate upgrade solving and add a real no-write preview`.

**M12 — Capture declared roots and make narrow closure guards agree.** Requires: M00, M06. Contracts: declared roots/one narrow closure guard. Remove or replace: default narrow AST-only first-party capture; inconsistent file/directory filtering. Proposed commit: `fix(mainboard): capture declared roots and make narrow closure guards agree`.

**M13 — Seal and approve a complete immutable source bundle.** Requires: M03, M11, M12, M15. Contracts: SourceSnapshot and detached approval. Remove or replace: current-capture equality as approval; mutable transfer reads. Proposed commit: `fix(mainboard): seal and approve a complete immutable source bundle`.

**M28 — Require verified declared inputs before launching work.** Requires: M01, M11, M12, M13, M22. Contracts: verified input readiness. Remove or replace: dropped pin digests; suppressed restore failure. Proposed commit: `fix(mainboard): require verified declared inputs before launching work`.

**M25 — Make evidence delivery run-owned and release-gating.** Requires: M01, M13, M20, M22, M24. Contracts: per-attempt execution image and final delivery. Remove or replace: shared snapshot mount relinking; empty-string unavailable transcript; absence-as-finality. Proposed commit: `fix(mainboard): make evidence delivery run-owned and release-gating`.

**M26 — Version receipt admission and normalize analytical rows once.** Requires: M04, M13, M20, M22, M25, M35. Contracts: format-specific verifiers/normalized projections. Remove or replace: repeated receipt-file Results reconstruction; implicit legacy Monitor finality fallback. Proposed commit: `fix(mainboard): version receipt admission and normalize analytical rows once`.

**M27 — Batch verified object access and bound materialization.** Requires: M01, M22. Contracts: verified byte/row bounded object stream. Remove or replace: per-object/ordinal retrieval loops; unbounded fetchall chunk joining. Proposed commit: `fix(mainboard): batch verified object access and bound materialization`.

**M29 — Install immutable runtime and environment releases on hosts.** Requires: M03, M13, M15, M17, M18, M20, M25, M28. Contracts: versioned runtime envelope and immutable installs. Remove or replace: additive generated-file publication; mutable runtime import roots for new jobs; PATH runtime selection for new scripts. Proposed commit: `fix(mainboard): install immutable runtime and environment releases on hosts`.

**M30 — Make physical-card leases atomic and token-owned.** Requires: M06, M19, M24, M29. Contracts: Universal CardLease / lease-exec grant with PostgreSQL reservation and host process lifetime. Remove or replace: check/unlink/create lease race; release without token. Proposed commit: `fix(mainboard): make physical-card leases atomic and token-owned`.

**M31 — Reserve GPU capacity before trials and enforce acquisition policy.** Requires: M18, M24, M29, M30. Contracts: execution capacity admission. Remove or replace: fixture-time busy card as scientific failure. Proposed commit: `fix(mainboard): reserve gpu capacity before trials and enforce acquisition policy`.

**M08 — Make lint respect artifact ownership.** Requires: M00, M13. Contracts: pre-write immutable ownership. Remove or replace: unconditional frozen-file/EOF rewrite. Proposed commit: `fix(mainboard): make lint respect artifact ownership`.

**M09 — Validate declared plotting policies.** Requires: M00. Contracts: versioned palette profile. Remove or replace: ignored declared palette quality policy. Proposed commit: `fix(mainboard): validate declared plotting policies`.

**M33 — Verify completed removals and publish the rollout contract.** Requires: M08, M09, M16, M28, M32, M36, M37, M38. Contracts: closure evidence index. Remove or replace: . Proposed commit: `docs(mainboard): verify completed removals and publish the rollout contract`.

**Global gates and closure rules.**

- M00 first establishes bounded temporary execution, external-effect denial, import proof and baseline accounting. Each package adds its own golden/fault witnesses to the cumulative suite; no package waits for an unrelated fake-fleet framework. No new expected failures or unclassified skips.
- Each package owns one contract or small leaf fix, plus its immediate call sites and focused scenario. No package hides a fleet rollout, data repair, whole-repository reformat, or new backend. Finish green, leave uncommitted, hand off files and one proposed commit message.
- Record changed source LOC, removed implementations and dependency edges; mandatory end-state: one job reducer, one command parser, one object verification path, one process-lifetime primitive, zero implicit settlement from reads, zero target/handle-only authoritative lookups.
- Use fixed checker scopes/versions established by M00. Ruff remains zero; no new pyrefly/ty/mypy or deterministic mcmr diagnostic outside an explicitly owned baseline. Fix baseline errors in changed files instead of adding ignores. Repository-wide cosmetic mcmr cleanup is excluded.
- Correctness tests use effect counts, byte hashes, reference reachability and crash recovery, not only printed exit codes. Every externally destructive effect requires a durable intent, current ownership token, verified preconditions and a recorded or explicitly unknown outcome.
- All paths below are relative to this worktree. '(new)' is represented separately where relevant; shared-file changes must be dependency-ordered. Test modules added by a package are owned by that package.
- New immutable records/messages use patos.FrozenModel or the established house base; use domain records, recursive JsonValue and a concrete SQL-value union. Audit changed signatures for Any, object and new suppressions, not only checker diagnostics. ABCs belong at substitution boundaries. Frozen updates must revalidate invariants.
- Record implementation, fault/recovery validation, checker repair and integration/review effort separately. Added/modified/deleted LOC and person-time ranges are unknown until M04, M11 and M22 pilots supply evidence; surface LOC is neither effort nor expected deletion.
- The sole lifecycle authority is JobReducer.reduce, introduced by M19 before writer conversion. JobStore persists it atomically; coordinator executes its intents; status and projections never implement transitions.
- Closure requires the expanded issue ledger, every applicable acceptance gate and expected-delta evidence. Plan completion, code verification, deployment verification and historical recovery are separate dimensions.
- Every live wave uses the common drain-and-switch protocol and its explicit rollback; implementation agents remain temporary-only. Live adoption is the owner-directed lead action as soon as gates pass, not an end-of-plan or MLSys deadline task.
- PostgreSQL-owning gates use the real temporary cluster fixture, record exact server/driver/DuckDB/extension versions and test restart, session death, contention, read-only behavior and restore. No SQLite substitute counts as acceptance.
- Every owned-card GPU invocation, including package gates and agent probes, must use the common lease. Until a host implements it, keep its card out of shared gate/agent use or use an already exclusive scheduler allocation with a verified compatible lease wrapper; do not claim universal enforcement from NVML.
- No report claims a measured performance improvement, a successful migration or a production deployment. Unverified current-doc capabilities are explicit implementation gates.
- bug_map names the primary implementation owner; its dependencies and M33 integration gate are required. A B ID is never closed by a plan or a historical report alone.
- M34 reduces current B23/B32 impact early; M24/M23 own final closure and delete the bridge.
- B49's verified inventory API lands in M01; its public CLI lands in M32. B54 gets early column preparation in M05 and closes with shared projection in M21.
- Registered mutable-runtime exclusion is checked in M13; M29 enables immutable host installs. B03 custody gating lands in M25; M24 supplies fenced execution.
- Code verified, deployment verified and historical data recovered are separate statuses. Passing a fake-host test closes only the code finding.
- B01–B55 mappings remain stable. C-SUBMIT-ACK, C-HELD-PLAN, C-REMOTE-ATTEMPT, C-EVICT-RACE and C-PREP-DEADLINE are separate source-traced defects in overview.closure_ledger; RESOURCE-DIRECTORY and NULL-GIT-CONSUMER have explicit dispositions. M33 must consume them too.
- M02's adapter acknowledgment gate is early; complete cancellation intent/effect recovery requires M19/M20/M24. M10 syntax/selection closes only after M11/M13/M28 feed exact identities and M29 provides immutable execution.

Use the required environment PATH in every shell, run from `/Users/pedro/Developer/projects` with worktree source first, and bound long commands with `mainboard proc timeout`. The gate must prevent live provisioning and state mutation; the report’s Python audit guard alone is not an OS sandbox. The actual source import proof for this run was `/Users/pedro/Developer/mb-refactor/src/mainboard/__init__.py`.

| Incident | Packages | Required invariant |
| --- | --- | --- |
| 1 | M19, M20, M21, M22, M24, M34, M36, M37 | Visible holder and read-only status survive orphaned or blocked work; a new owner cannot overlap an unconfirmed old worker. |
| 2 | M23, M26, M27, M34, M37, M38 | Incremental control facts eliminate full-history replay; normalized ingestion and batched object reads remove per-file amplification without assuming the historical CPU profile was reproduced. |
| 3 | M22, M24, M32, M36, M37 | The exact service identity passes a temporary access probe and whole-pass bound; loopback SSH remains until replaced by proven access. |
| 4 | M10, M11 | Malformed segments/empty acquisition selection are refused before remote effects, and explicit reuse is visible. |
| 5 | M12, M13, M15, M29 | Registered full bundle includes mainboard and complete environment; transfer/import never observes later agent edits. |
| 6 | M05, M20, M21, M36, M37 | One identity/status schema carries execution, outcome, evidence and freshness plus host/target. |
| 7 | M18, M30, M31 | Every job, gate and agent takes one host-wide card lease; unmanaged exclusive use is refused before a trial; disconnected ownership remains blocked until process reconciliation. |
| 8 | M07, M14, M15, M16 | All-platform stable solvable generations contain every stage lock and publish atomically. |
| 9 | M13, M25, M26, M32, M37, M38 | Run-owned mounts and versioned proof admit valid Mainboard non-Git fixtures; actual downstream NULL rejection remains consumer-pending until its predicate/fixture is tested. Historical recovery is independent. |

**All original bugs and issue dispositions remain tracked.**

| Bug | Primary owner |
| --- | --- |
| B01 | M01 |
| B02 | M20 |
| B03 | M25 |
| B04 | M02 |
| B05 | M03 |
| B06 | M13 |
| B07 | M32 |
| B08 | M08 |
| B09 | M01 |
| B10 | M01 |
| B11 | M20 |
| B12 | M31 |
| B13 | M02 |
| B14 | M11 |
| B15 | M12 |
| B16 | M02 |
| B17 | M18 |
| B18 | M02 |
| B19 | M13 |
| B20 | M04 |
| B21 | M20 |
| B22 | M30 |
| B23 | M24 |
| B24 | M27 |
| B25 | M14 |
| B26 | M25 |
| B27 | M18 |
| B28 | M18 |
| B29 | M17 |
| B30 | M29 |
| B31 | M16 |
| B32 | M23 |
| B33 | M17 |
| B34 | M22 |
| B35 | M16 |
| B36 | M16 |
| B37 | M15 |
| B38 | M06 |
| B39 | M11 |
| B40 | M12 |
| B41 | M10 |
| B42 | M28 |
| B43 | M11 |
| B44 | M07 |
| B45 | M07 |
| B46 | M09 |
| B47 | M18 |
| B48 | M18 |
| B49 | M01 |
| B50 | M10 |
| B51 | M18 |
| B52 | M05 |
| B53 | M13 |
| B54 | M21 |
| B55 | M16 |

| Issue | Audit status | Packages | Disposition |
| --- | --- | --- | --- |
| host-hold-1 | open | M17 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Sync still invokes unconditional uv Python install; download availability itself not rechecked. |
| host-hold-2 | open | M02 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Unchecked qdel reproduced with a fake remote. |
| host-hold-3 | open | M03 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Temporary held inbox snapshot was actually pruned by the current code. |
| host-hold-4 | open | M18 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Board.interact selects and stages an interactive scheduler command without reading Line or the spool. |
| host-hold-5 | fixed-verified | M06 | Preserve regression; do not infer production verification. login_ask passes timeout=90.0; source catches ProcessTimedOut as HostUnreachable. No live network failure tested. |
| host-hold-6 | fixed-unverified | M00 | Preserve regression; do not infer production verification. Affected context managers now use Generator, but alignment of package/workspace installed Ruff versions was not established. |
| host-hold-7 | open | M18 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Runner receives walltime for PBS as well; top-level JobSpec docstring still describes old backend split. |
| host-hold-8 | open | M18 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Keeper uses login bash and absolute /bin/bash correctly. Pbs.interactive still says command is impossible and rejects it; no live PBS validation attempted. |
| host-hold-9 | open | M29 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. No remote timing measurement performed; dispatch still mirrors/verifies/pins. The reported 70-80 seconds is historical, not a measured current result. |
| host-hold-10 | open | M18, M21 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Held omits allocations whose claims have exits, but ordinary PBS state still relies on qstat. Current wrapper cache behavior not probed. |
| host-hold-11 | open | M18 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Ten-second retry remains; no measurement of the site's interactive quota release gap. |
| cutok-1 | open | M04, M26, M32 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Partial fix verified; follow-ups remain. |
| cutok-2 | fixed-verified | M00 | Preserve regression; do not infer production verification. schema.latest no longer assumes dropped exists. |
| cutok-3 | open | M22 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Connection factory, UTC default, environment settings and query routing are fixed-verified. Effective-setting doctor output and lake override documentation remain open follow-ups. |
| cutok-4 | obsolete |  | Historical/obsolete; no new implementation. The audit target is already 0.5.4, so the concrete 0.5.2 release decision is historical. The release-on-main/version-tag behavior still exists; this audit authorizes no publishing. |
| cutok-5 | fixed-verified | M13 | Preserve regression; do not infer production verification. dispatch_closure.py excludes nested packages/mainboard/__pycache__ bytes without git. |
| cutok-6 | open | M13 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Only source blob mismatch plus file; no mtime or retry advice. |
| cutok-7 | fixed-verified | M16 | Preserve regression; do not infer production verification. Stable 3.14.4 beats 3.15.0rc3 in the isolated test. The function intentionally chooses a prerelease if there are no finals, so this verifies the reported stable-versus-RC incident, not a universal prerelease ban. |
| cutok-8 | open | M16 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. upgrade offers --frozen and --no-install, both different from a no-write preview; no dry_run parameter exists. Feature gap remains. |
| cutok-9 | open | M08 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Mixed entry. Broad directory rewriting and lack of seal protection remain open. The per-owner invocation machinery and pyrefly fallback fix are present. |
| cutok-10 | fixed-unverified | M29 | Preserve regression; do not infer production verification. gzip level1 present; remote throughput not measured. |
| cutok-10a | open | M25, M27 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Transfer mitigations present; large-scope scan/fixed deadline remain. |
| cutok-11 | open | M32 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Operational backlog unverified; no repair authorized. |
| cutok-12 | open | M35 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Leak report not reproduced in small checks; no verified fix identified. |
| cutok-13a | open | M27 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Reads remain per digest and verification loops unique references; shared sessions reduce attach cost but do not batch blob reads. Historical 24 s/process not rebenchmarked. |
| cutok-13b | open | M27 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Exact-file materialization and read-through artifacts are available, but broad materialize eagerly writes every indexed file and has no byte estimate/cap; external Path-based renderers were not audited. |
| cutok-13c | open | M01, M32 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Eviction and external home exist, and the existing unchanged/changed-file test passed. A user-facing inventory of verified tree cache copies is still absent. The newly reproduced corrupt-blob eviction defect means current eviction is not generally safe. |
| cutok-13d | open | M22, M32 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Migration inventories only Project.out, while external Lake.home is elsewhere. Doctor does not include a dedicated lake-home/mount readiness section. |
| cutok-13f2 | open | M22, M24 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Specific TCC and launchd mitigations verified; general unresponsive-volume timeout remains incomplete. |
| cutok-13f3 | fixed-verified | M17, M22 | Preserve regression; do not infer production verification. A temporary mirror without .git ignores a configured center-only absolute lake and returns its generated directory. Does not certify current remote deployment. |
| cutok-13f5 | fixed-verified | M01 | Preserve regression; do not infer production verification. Both focused existing tests passed, including ingest import recording and dedup/materialize/evict relative paths. This does not cover symlink containment. |
| cutok-13f4 | open | M29 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Current named-file shipment does not reconcile retired generated inputs. |
| cutok-13e | open | M09 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. PlotStyle checks shape/basic numeric fields and table checks number of color slots, but no perceptual separation/contrast/color-vision audit is implemented. Feature request, not a reproduced measurement failure. |
| cutok-13f | open |  | Open external/deferred item; no mainboard code fix claimed. An optimization proposal remains unimplemented/unbenchmarked here. SHA-256 remains evidence identity, as required. No claim that xxh3 is currently the bottleneck or a correctness fix. |
| cutok-13g | open | M27 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Current materialization still calls _write per file and chunks queries per ordinal; windowed replication exists but is not reused by materialize. |
| cutok-13h | open | M26 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. View construction inventories/materializes every matching receipt before SQL filtering; project and experiment scope help. No 3–9 minute real-volume benchmark attempted. |
| cutok-13i | open | M28 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Proactive restoration was added, but it ignores the pin hash and swallows restoration failures. Required existence/integrity is not a collection precondition. |
| cutok-13j | open | M25, M26, M32 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. New conventional targets correctly widen fetch to the evidence-store home (fixed-verified); existing recorded handles are not repaired, and shared-output conflicts remain open. |
| cutok-13k | open | M07, M16 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Two local fixes are present; four independent remaining upgrade defects are confirmed by source or isolated witnesses. |
| cutok-13l | open | M14, M15 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Lock/install defect reproduced; rollback handles some common failures but remains incomplete. |
| cutok-13m | open | M24, M34 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open.  |
| cutok-13n | open | M23, M27, M34 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open.  |
| cutok-13 | fixed-verified | M17 | Preserve regression; do not infer production verification. Generated agent launcher executes the workspace prefix Python; source fallback is uv managed Python, with explicit refusal rather than OS python3. Does not certify current remote deployment. |
| cutok-14 | open | M16 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Underlying solver's one-source restriction is intentional per issue. Mainboard forwards solver output and adds no suggested all-path-wheel workaround. Did not re-solve TokTier. |
| cutok-15 | open | M17, M29 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Historical links were reportedly removed by hand. Setup still has no dangling-link reconciliation; the code only tests pueue status, locates/installs pueued and starts it. |
| cutok-16 | open | M17, M29 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. No live daemon inspected or restarted. Setup returns when daemon status answers, without establishing the daemon executable's provenance. |
| cutok-17 | open | M17 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. macmini remains declared as ssh and is surveyed as a profile. Actual current self-resolution was not queried. |
| cutok-18 | open |  | Open external/deferred item; no mainboard code fix claimed. No pedro-home host declaration found. No ssh config or remote state modified; retain/decommission decision is still external to this source audit. |
| cutok-19 | open |  | Open external/deferred item; no mainboard code fix claimed. Historical October 28 stop and queue delays were not revalidated. These are site scheduling facts, not a demonstrated source fix. |
| cutok-19a | open | M28 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Historical cache copy is reported but not verified. Mainboard supports explicit resources/needs/pins; no inference of TokTier's private per-host cache or native-route coverage gate. External campaign follow-up remains open. |
| cutok-20 | open |  | Open external/deferred item; no mainboard code fix claimed. External renderer source is not in c97cb81. Mainboard provides machine facts, but no evidence here verifies v22/v15 renderer changes. Conservative external status. |
| cutok-21 | open | M28 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. External tests/env declaration not contained in this mainboard revision. Explicit resource declarations/restoration exist; cannot mark the historical host contract suite fixed. |
| cutok-21a | open |  | Open external/deferred item; no mainboard code fix claimed. External test/renderer disagreement cannot be established from mainboard code; keep open for owner of that repository. No renderer or test was changed. |
| cutok-22 | open |  | Open external/deferred item; no mainboard code fix claimed. External source and type-check results are outside the c97cb81 repository. Not verified or declared obsolete. |
| cutok-23 | open |  | Open external/deferred item; no mainboard code fix claimed. External upstream packaging action. No current registry/network assertion made; source-build workaround in issue log does not establish upstream wheel availability. |
| cutok-24 | open | M08, M28, M32 | Implement named code/rehearsal scope; excluded operational/consumer portions remain open. Exact-byte repair and -text exclusion are reported externally, not verified here. The generic hygiene adds a final newline unless protected; tool attribution and all-input registration audit remain open. |

The unchanged 118-entry closure ledger in `overview.closure_ledger` also includes C-SUBMIT-ACK (M20), C-HELD-PLAN (M11), C-REMOTE-ATTEMPT (M20), C-EVICT-RACE (M01), C-PREP-DEADLINE (M06), RESOURCE-DIRECTORY (M12) and NULL-GIT-CONSUMER (external consumer owner, M26/M32). They require their original concrete fault/reader witnesses. `cutok-12` remains tied to M35 shutdown evidence; a missing real workload is not a verified fix. Historical 447 s system-time attribution remains an unmeasured incident observation.

**Migration, historical proof and rollback.**

Wave-by-wave drain-and-switch is owner-decided now; this run supplies design only. Temporary rehearsals precede lead deployment. v26 GH200 jobs remain queued and keep their shipped code.

- One active center controller generation, one PostgreSQL control authority and one publisher per canonical LakeId. Until W4 activation the legacy catalog remains sole authority; staged candidate writes are isolated.
- No active editable tree is changed until its effectful mainboard processes and lazy-import consumers drain. Same-user discipline replaces the proposed principal fence; a new epoch is not an enforcement mechanism against an old binary.
- Every wave preserves queued/running PBS/pueue code, old aliases and full runtime/source/environment roots. Old jobs settle through retained decoders and deterministic identity mapping; uncertain finality stays pending.
- Catalog migration preserves retained snapshot and object identities. Raw evidence, historical hash algorithms and NULL-Git verified-tree proof remain intact. New logical normalization never rewrites old proof bytes.
- PostgreSQL control and DuckLake transactions have separate commit boundaries. Durable outbox/operation IDs reconcile every gap; pg_dump includes both schemas and external data copying completes the checkpoint.
- Locks are PostgreSQL advisory ownership sessions plus short row/transaction claims in the new architecture. An ended session or expired heartbeat does not prove a worker/card is safe; durable uncertainty blocks conflicting effects.
- All owned-card compute uses the same host-wide lease, including gates and agents. A host awaiting legacy-user drain cannot grant new unmanaged shared-card work.
- Each wave states rollback and measures the 60 s final switch limit; restore never undoes unknown provider/scheduler effects. After PostgreSQL activation a stale SQLite writer is never restarted.

**S00 — Inventory each wave and preserve old job runtime roots.**

- Prepare an immutable copy/checksum of the reviewed candidate tree and all dependency artifacts off the live path. Inventory current launchers, nested environments/.pth, processes, host runtime references, service commands and queued job envelopes. Record prior tree, config/catalog endpoint and backup. Stage everything possible before drain.
- Center rollout does not imply host sync. Every host receives only its wave-listed reviewed payload, staged alongside old runtime roots. Existing PATH-based queued PBS/pueue jobs retain their exact aliases/bootstrap/.pth/import roots until they drain or the delayed-start fixture proves isolation. New envelopes later use absolute immutable interpreters. A host whose shared root cannot be preserved waits; it does not cancel queued v26 GH200 work.
- Every wave gate starts actual old generated PBS and pueue commands after the simulated switch, then settles their receipts through the current center. Preserve legacy target/host/handle/submitted_at identity and raw format verifiers; control migration records a deterministic SubmissionId map. Old jobs continue writing their shipped format in their run roots; new settlement can read both. Unknown finality or ambiguous identity stays pending. Never rewrite old receipts into fictitious new proof.

Exit gate: Concrete candidate/prior generation, process inventory, queued-job runtime map and complete backup references; unknown ownership retains data and blocks unsafe switch.

**S01 — Stage and rehearse away from the live checkout.**

- M00 defines a lazy postgres_cluster fixture; M36 provides its environment dependency and implementation. From an isolated test root, run environment initdb, allocate a private Unix socket and collision-safe loopback port when needed, start foreground postgres under ProcessScope and wait boundedly with pg_isready plus SQL. Supply the DSN only to the test, allow only that local endpoint through the gate, and refuse live DSNs/PGDATA. Use real fsync/durability settings for crash gates. Kill/restart fixture postmasters and clients, retain diagnostic logs, stop/reap in finalizers; no launchctl/systemctl install in integration tests. Missing binaries fail PostgreSQL-required gates explicitly; the earlier safety suite can run without requesting this fixture. Run service-rendering tests on both OSes and separately require exact-service-context rehearsal before each actual deployment.
- Run the applicable package fault gates and delayed legacy job start/settlement fixtures. W3 stages service, W4 catalog/control conversion and complete restore; subsequent waves extend the same proof.

Exit gate: Real temporary PostgreSQL where required; exact environment/extension versions and no external effects; candidate tree/environment checksums recorded.

**S02 — Drain old controllers under the existing owner.**

- Close the center admission window for controllers and prevent service auto-restart. Stop the launchd/systemd settler; wait for or stop all effectful mainboard processes, local trials that ingest/evict at close, notebooks with loaded mainboard, held keepers and their workers. Do not stop unrelated compute merely to simplify deployment. Prove each remaining process is an isolated file producer whose bytes and imports will not change. A surviving unknown writer/effect worker blocks this wave. This is operational drain by the single owner, not a security boundary against someone deliberately launching old code.
- Retain runtime roots for isolated remote producers. An old held keeper or local trial that can mutate center state must finish or stop before the affected wave.

Exit gate: No old effectful controller or unconfirmed worker can act through any inventoried launch route. No queued v26 job was cancelled.

**S03 — Checkpoint and apply the final delta.**

- M38 extends lake replicate: acquire a durable backup/maintenance exclusion, drain in-flight lake publication for a consistent handoff, export a REPEATABLE READ PostgreSQL snapshot, and use environment pg_dump --format=custom --snapshot=<id> --file=<staging> for the entire mainboard database. The same exported snapshot supplies control/outbox watermarks, every retained catalog file/delete-file reference and inline rows (in the dump). Keep the exporting transaction alive through dump/manifest creation; bound its lifetime. Resume append ingestion after snapshot establishment only if the fixture proves file immutability and maintenance remains blocked; otherwise remain drained. Copy and hash all referenced files, raw proofs, source/solution/runtime retention roots, configuration and a nonsecret role/bootstrap recipe. Pin copying files durably so lock/session loss cannot allow cleanup. pg_dump alone is never a successful lake replica. Publish a complete marker only after hashes and a temporary pg_restore plus DuckLake/claim readback pass. A failed copy leaves an incomplete resumable manifest and retained pins. Restore into a new cluster with effects disabled; never overlay active state. Backups provide restore-point recovery, not a promise of zero-loss PITR.
- Before W4 use a consistent SQLite input backup and M37 final-delta validation; after W4 use M38 PostgreSQL plus external-data checkpoints. Do not run two active catalogs against the same DATA_PATH.

Exit gate: Restorable complete checkpoint and final identity/catalog/control watermarks within the measured switch budget; otherwise keep prior authority active.

**S04 — Switch the complete center editable generation and selected host payloads.**

- While drained, replace the center editable packages/mainboard tree as a complete generation, updating outer entry points, selected nested environment metadata and service configuration in one journaled switch. A filesystem atomic rename/pointer is preferred; otherwise the drain plus checksummed journal makes an interrupted whole-tree copy non-runnable until completed or restored. Never edit live modules one by one with controllers running. Clear inherited PYTHONPATH and prove outer/nested/shell/service imports. Start one settler only after readiness and last-pass gates, then reopen admission.
- 60 s final-delta/entry-point switch budget after offline staging and successful drain, measured per wave. The owner approved the recommendation, not an unmeasured promise. If the final step cannot fit before activation, leave the old tool active and improve/rehearse staging. Drain time is visible separately and can defer a wave.
- Hosts follow their own wave live_switch entry; preserve old aliases/import roots and defer unisolatable host changes.

Exit gate: Outer/nested/shell/service import proof and one current controller; PostgreSQL/database/volume freshness health reflects actual context.

**S05 — Settle late old jobs and observe the new wave.**

- Every wave gate starts actual old generated PBS and pueue commands after the simulated switch, then settles their receipts through the current center. Preserve legacy target/host/handle/submitted_at identity and raw format verifiers; control migration records a deterministic SubmissionId map. Old jobs continue writing their shipped format in their run roots; new settlement can read both. Unknown finality or ambiguous identity stays pending. Never rewrite old receipts into fictitious new proof.
- Verify new submissions use the negotiated current decoder/runtime, late old receipts settle once, status is passive after W4, and universal card consumers all use the same lease after host enrollment.

Exit gate: No duplicate effects, missing proof, false terminal state or unleased owned-card entry; deployment status separate from code gate status.

**S06 — Retire only unreachable executable generations.**

- Review queued/running/held jobs, raw evidence readers, checkpoints and pending effects before deletion. Preserve all historical verifier codecs and exact referenced source/solution/data bytes.
- Repeat the common switch for any executable removal. Unknown reachability retains the generation and prevents destructive cleanup.

Exit gate: Zero active runtime references and explicit retained-proof disposition; audit never deletes a late job dependency.

| Historical format | Verifier obligation | Required references |
| --- | --- | --- |
| wire JSON / mainboard.run.v1 | Retain exact received wire bytes and recorded envelope semantics; decode only supported version. | Submission tuple/map, source manifest, inputs/artifact digests; no finality from absent wire. |
| Parquet receipt fragments / compacted partitions | Retain original object hashes and nested JSON-string field types; decode pre/post-compaction through recorded format. | Full fragment inventory, partition/layout manifest, source raw object IDs. |
| Source TSV / closure listing | Exact historical TSV bytes and SHA-256 algorithm as recorded, without new canonical listing normalization. | All listed source objects and original path/format context. |
| Artifact paths / content hashes / pins | Recorded digest algorithm/size and format-specific path resolver; restore across roots without changing identity. | Exact pinned objects plus original namespace/attempt mapping. |
| Registration / baseline proof | Historical BLAKE2b inputs/parameters and byte construction retained; do not replace with detached-approval digest. | Registration/baseline raw bytes and all referenced source/input proof objects. |
| Legacy environment / prefix identity | Original normalized/truncated identity algorithm retained separately from new Solved normalization. | Original locks/generated manifests, versions and source mappings; executable retention tracked separately. |
| unknown future envelope | Retain exact bytes as unverified, identify format, refuse destructive acknowledgment and automatic admission. | Retain discoverable references; unknown reachability prevents related GC. |

- Preserve exact raw JSON/Parquet/wire bytes, digests, source listings and verification code identity.
- Legacy source proof may be a verified tree with NULL Git fields; type it explicitly. No proof means legacy-unverified, not a default clean commit.
- Canonical event/delivery import identities are stable across retries; natural-key conflicts retain both raw objects and a diagnostic.
- Recover historical missing output only from retained verified snapshots/receipts; never synthesize a receipt to close the backlog.
- Supported-format matrix names decoder, verifier version/algorithm, required raw/referenced objects and unknown-format behavior; new normalization never substitutes for a historical hash algorithm.

**Before new writes.** Before activation/new effects: stop/reap candidate workers and restore the recorded prior tree/config/endpoint, keeping the latest consistent legacy state. After activation: first stop admission and drain; retain new receipts/jobs and use a tested schema-compatible previous release or forward repair. Never restore a stale database over new events, never infer remote undo from local restore and never restart a pre-PostgreSQL writer against post-switch state. Unknown external effects require reconciliation. Each wave specializes this rule.

**After new writes.** Pause admission and drain; preserve current PostgreSQL catalog/control and newly arrived receipts. Roll back code only to a tested schema/envelope-compatible release; otherwise checkpoint and forward-repair. Reverse migration to SQLite is not supplied or assumed.

**External effects.** Submissions/cancels/provider release are not undone by restore. Reconcile stable tokens/receipts, retaining unknown effects. Running and queued jobs keep their runtimes/output roots.

**Lake format.** Catalog-engine migration, DuckLake specification upgrades and application table changes are distinct. M37 preserves the inventoried supported catalog spec. An engine/spec change needs its own retained-snapshot/readback/restore gate and cannot be hidden in W4.

**Per wave.** Every order.waves entry contains a concrete rollback and center/host/drain/legacy switch. Common protocol is in decisions.cutover.

Parity uses the original explicit common schema and legacy identity map. Corrected JSON types, reused-handle attribution, terminal status, held-plan refusal and a proven NULL-consumer change each require enumerated fixture-row expected deltas. New fields are tested separately. SQL parity alone cannot replace `Dataset.passing/status`, Artifact reads and claim pin verification. Unknown future formats remain retained and unverified, blocking destructive acknowledgment.

**Final owner decisions and implementation inputs.**

| Decision | Adopted contract | Status |
| --- | --- | --- |
| Q01 — Where should mutable job state live? | One center_id identifies one internal-disk PostgreSQL cluster. Use database mainboard, schema control and one ducklake_<canonical_LakeId> catalog schema per registered lake. control.lakes binds LakeId, metadata schema, DATA_PATH and authority; path aliases never create another owner. PGDATA defaults to <center-state>/services/postgresql/<major>/data, resolved and checked on the internal disk; socket/log/config directories are adjacent, never in a disposable environment prefix. Parquet and blob bytes stay beneath the configured external DATA_PATH. Unix socket directories must fit platform sockaddr_un limits: use a short center-ID directory in the private runtime/temp area with mode 0700 when an adjacent path is too long, and record/clean only the owned directory. | owner-final-recommendation-adopted |
| Q02 — Should job list/show/query remain implicit settlement commands? | List/show/current-status SQL are passive bounded PostgreSQL readers with observed_at, last successful pass, last attempt, owner/phase and projection watermark. No catalog migration, lake attach, service startup, host probe or reconciliation occurs as a side effect. With PostgreSQL down, return a bounded unavailable/stale response (a last-read display cache, if present, is labeled nonauthoritative). An absent external volume still permits live current-state reads when PostgreSQL is healthy. Refresh/wait explicitly enqueue targeted coordinator work and read progress. | owner-final-recommendation-adopted |
| Q03 — What proves approved acquisition sources? | One sealed SourceSnapshot for all capture origins, plus an immutable detached approval selected by acquisition. Bind full first-party bundle, exact EnvironmentSolution and input identities; never update approval during capture/submit. Use existing registration entry points, with owner-explicit creation. | retained-design-contract-no-approval-created |
| Q04 — What is the default for already-complete registered cases? | A registered acquisition creates a new attempt by default. Reuse must be explicit and produce a local reuse receipt with counts; never submit a zero-work remote pytest job. Preserve consumer-specific freshness flags only inside their owned argv. | owner-final-recommendation-adopted |
| Q05 — How strong must the shared-GPU guarantee be? | Q05 is universal: every user of an owned card takes the same host-wide lease, including mainboard jobs, refactor gates and agent runs. There is no development or gate exemption. Unmanaged exclusive runs are refused. | owner-final-recommendation-adopted |
| Q06 — May a spend cap release a rental before required evidence arrives? | Stop compute at the cap and reserve collection headroom in the admission budget. If the hard financial cap forces release, record evidence_lost and the explicit cap policy; do not certify or silently reuse evidence. | owner-final-recommendation-adopted |
| Q07 — How should local restore conflicts behave? | Refuse overwrite by default and list expected/current digests. A separately explicit replace command first preserves the edited bytes as a recoverable object and records the disposition. | owner-final-recommendation-adopted |
| Q08 — Which palette metrics and thresholds are house policy? | Before M09, select and version a named existing or maintained validator profile and its numerical thresholds. Require contrast, lightness order, chroma bounds, perceptual separation and requested CVD fixtures; unavailable metrics keep B46 open. Do not invent thresholds or count unsupported as success. | owner-final-policy-specific-profile-still-required-before-M09 |
| Q09 — Keep the macOS ssh-localhost service access path? | Keep it initially, with exact-context access checks and complete-pass supervision. Replace it only after an alternative launch identity proves access to the external volume in a manual rollout rehearsal. | owner-final-recommendation-adopted |
| Q10 — How much cutover interruption is acceptable? | Use a measured 60 s maximum final-delta/switch write pause after staging and safe drain, per wave. Isolated file producers continue; unsafe drain defers a wave/host. Abort before activation on overrun; after new effects retain current state and forward-recover. | owner-final-recommendation-adopted |
| Q11 — Which historical runs and fleet entries should be retained or retired? | Inventory first; retain raw evidence and unresolved runs until explicitly dispositioned. Route the self host locally, decide pedro-home separately, and recheck site maintenance/queue facts during rollout. | owner-final-recommendation-adopted |
| Q12 — Should objects move out of DuckLake in this campaign? | No. Batch reads, normalize receipts and separate control-state access first. Reconsider a packed store only if the M27 profile still identifies object-file layout as the limiting cost and a publication/backup/GC pilot passes. | owner-final-recommendation-adopted |
| Q13 — How will unmodified legacy controllers lose effect authority? | Q13 is final: drain-and-switch under the existing owner. Stop the settler and every effectful old controller, switch the center editable checkout as a complete reviewed generation, then restart. No separate principal, credential rotation, or claim that a new epoch restrains old code. | owner-final-recommendation-adopted |
| Q14 — How should incompatible native collection or solve-only forms behave? | Refuse with the missing capability before acquisition effects; use a supported compatible prepare/tool environment only after its exact native behavior is verified. Do not rebuild pytest grammar or a registry-based compatibility solver. | owner-final-recommendation-adopted |
| Q15 — Who closes the NULL-Git admission incident? | Assign the owner of the actual failing consumer and obtain its receipt/predicate fixture. Keep that portion E05 consumer/deployment-pending until its real read path admits verified non-Git proof. | owner-final-recommendation-adopted |

- Exact stable solvable PostgreSQL/psycopg/libpq/DuckDB/extension versions are recorded by M36 against the workspace platform matrix; current docs do not prove the carried development build matches them.
- M37 inventories actual legacy catalog spec/inline schemas from a supplied read-only consistent backup. Unsupported fields or unproven time-travel conversion block only the catalog wave, not early safety releases.
- Q08 still requires the named maintained palette profile and thresholds; owner acceptance of that requirement did not supply a profile. Q15 still requires the actual external NULL-Git consumer owner/predicate fixture.
- Actual production service context, host legacy runtime isolation, universal card-user enrollment and measured 60 s switch remain deployment gates. No live discovery or validation was performed by this report.

**Exclusions and retained critique.**

| ID | Scope | Reason |
| --- | --- | --- |
| E01 | Historical backlog repair/retirement and real mutations during design or package implementation | This run and package agents remain temporary-only. The owner has decided lead deployment now at each safe gated wave; it is no longer excluded or deferred as a plan objective. Historical dispositions still need their actual evidence and no receipt may be invented. |
| E02 | Fast noncryptographic hashing and a packed CAS replacement | Neither is the demonstrated first bottleneck. Keep SHA-256 identity, reuse existing batched reads and native DuckLake maintenance. Any later xxh3/pack experiment needs measured benefit and independent crash/backup/GC tests. |
| E03 | New generic orchestration/solver stacks, Ray, Submitit, PySlurm and wholesale Pixi compiler/rsync replacement | PostgreSQL server/catalog/control is explicitly in scope through M19/M36/M37/M38. Other unrelated stack replacements do not remove the listed contracts and remain excluded. |
| E04 | Live fleet configuration, SSH aliases, current Miyabi maintenance dates/queue waits and upstream PBS quota behavior | These are external facts or owner policy, not source defects established by this worktree. M17/M18 expose truthful capability/freshness information; an operator must validate current host/site facts. |
| E05 | Cutok/TokTier renderer/tests/type fixes, private application cache inference and upstream wheel publication, and the unidentified external NULL-Git admission consumer until an owner/fixture is established | These sources and upstream actions are outside mainboard. M28 guarantees declared inputs, M13 guarantees captured sources, and M21 exposes host facts; application owners still declare/fix their own inputs and behavior. |
| E06 | Whole-repository stylistic cleanup, file-moving campaign, provider/Slurm deletion, logging framework swap, Windows resurrection and generic alias removal | No listed failure requires these changes. Use no-new-diagnostic gates and delete replaced implementations; do not turn 1,908 mcmr findings into unrelated work or remove supported backends without usage evidence. |
| E07 | Publication, tags, commits, Git worktree operations and live dependency upgrade | Git is read-only and the live editable tool is in use. The lead reviews and commits later; package implementation stays in designated worktrees. The old 0.5.2 release question is obsolete. |
| E08 | AIZK-backed serve/login/API/MCP/app product and direct remote catalog writers | A cross-package service product is unrelated to this bug-prevention refactor. Keep center ingestion and offline host receipts; do not introduce a second control plane. Existing served-lake write access is actively disabled or refused by M22; exclusion of a new service does not grandfather an unfenced writer. |

All 32 numbered critique dispositions, seven acceptance subfindings and 36 original package coverage rows remain in the JSON. S10 now reuses PostgreSQL dump/restore plus the legacy SQLite backup input; C02 keeps the old-controller hazard but adopts the owner’s drain proof. Neither a plan nor a fake-host gate certifies production deployment or historical recovery.

| Critique | Disposition | Current reason |
| --- | --- | --- |
| S01 — Effort and implementation surface are uncosted | accept | Add per-package surface/symbol/deletion/contract records and unknown effort ranges; calibrate separate implementation, validation, checker and integration costs with three pilots. |
| S02 — SubmissionPlan and ResourceEnvelope duplicate existing models | accept | Evolve vocabulary.Request/Resources, retain one resolved plan, and retire copied JobSpec fields with direct/lane/held/provider parity. |
| S03 — Lifecycle authority is duplicated and too late | accept | M19 defines JobReducer.reduce before writer conversion; M23 only acquires indexed inputs and JobStore persists decisions. |
| S04 — Every native lock stage needs an explicit interface | accept | Extend Solved and Ecosystem.resolve/install across base/stage/Node/Go/Rust with a verified native capability table and unsupported refusals. |
| S05 — Lake gateway is a cross-process API conversion | accept | Specify finite typed operations and bounded results, keep native resources worker-local, and assign all immediate callers without another queue or lease system. |
| S06 — Shared-file conflicts are being treated as semantic prerequisites | accept | Separate requires_contracts from conflicts_with and serial integration order; prioritize the control slice and bound/delete M34. |
| S07 — Native test validation must reuse pytest | accept | Cyclopts owns Mainboard only; exact pytest argv/plugins use its collector in a compatible environment with explicit refusal when unavailable. |
| S08 — Two sealed-source variants add unnecessary paths | adapt | Use one SourceSnapshot and full declared roots by default, while retaining narrow closure only for explicit scientific declarations and historical format proof. |
| S09 — Dependency reuse must name concrete implementation choices | accept | Use DuckDB extract_statements/get_table_names, PyYAML syntax and native solving; refuse unsupported solver forms instead of reconstructing a solver. |
| S10 — Three migration layers are unnecessary | accept | Use one center orchestrator and checkpoint manifest. M37 supplies the versioned catalog converter; M38 supplies pg_dump plus referenced data backup; SQLite backup remains a legacy input primitive. |
| S11 — Typing rules are absent from acceptance | adapt | Require house immutable models and changed-boundary Any/object/suppression audits; apply model policy at these seams without an unrelated repository-wide dataclass rewrite. |
| S12 — Object I/O bounds omit metadata and have no measured baseline | adapt | Use byte/row/chunk windows on existing Blobs, keep zero idle reads unconditional and treat RSS/throughput tolerances as pilot gates pending measurement. |
| S13 — The initial harness risks becoming a blocking framework | accept | M00 first proves isolation/baseline; each owning package grows existing fixtures and cumulative golden/fault coverage. |
| C01 — Legacy queued scripts do not pin their runtime | accept | Pin new scripts by absolute versioned runtime and preserve old PATH aliases/editables until actual delayed-start, child-import and held-keeper fixtures pass. |
| C02 — New epochs cannot fence unmodified legacy writers | adapt | Owner supersedes the proposed principal fence with a tested process drain and complete editable-tree switch. A surviving old controller blocks handoff; same-user old-code denial is not claimed. |
| C03 — Center launcher switching misses nested environments | accept | Include Provisioner readers, all selected workspace generations, shells/tasks/.pth and outer/nested/service import proofs without the audit guard. |
| C04 — Ordinary scheduler acceptance has an unrecorded-effect window | accept | Track C-SUBMIT-ACK and commit plan/pins/token/intent before every backend; lost replies reconcile or remain unknown without resubmission. |
| C05 — Quota-held requests mutate and collide | accept | Track C-HELD-PLAN and retain a distinct SubmissionId plus sealed resolved plan across retry; legacy unresolved holds require explicit preparation. |
| C06 — Remote logs and exit records alias reused scheduler IDs | accept | Track C-REMOTE-ATTEMPT and carry identity through runtime, script, logs, sampler/framing and scheduler/autopsy APIs before status closure. |
| C07 — Per-run output roots still relink a shared snapshot | accept | Give each SubmissionId an immutable execution directory/mount table and convert agent/dispatcher/claim-path readers; test overlapping identical bundles. |
| C08 — Collect-only cannot decide scientific reuse/fresh policy | accept | Separate syntax/collection from trial admission using target axes/facts, input identities and watermark, then enforce that selection at runtime. |
| C09 — Registered acquisition needs an independent approval authority | accept | Define detached immutable approval binding acquisition/source/solution/inputs, with explicit selection and positive and mismatch tests across all entry paths. |
| C10 — NULL-Git incident and parity are not proven by a codec change | accept | Keep the actual rejecting consumer externally pending and compare common-schema/claim-reader output with row-specific expected corrections. |
| C11 — Historical formats/verifiers need a concrete inventory | accept | Record each raw format and exact verifier family, keep legacy identities beside projections, and test restore/late/compacted/unknown cases. |
| C12 — Supervised I/O conflicts with live connection and callback APIs | accept | Convert all connection-escaping callers after M35, execute/consume operations inside fresh workers and test hangs through commit/close. |
| C13 — Writer authority must follow LakeId and cover served access | accept | Bind canonical LakeId to one control authority, route or refuse other workspaces and disable unsupported served writes including write=False escalation. |
| C14 — Verified lake custody does not prevent local replacement loss | accept | Track C-EVICT-RACE and require finalized/coordinated output with exact-candidate quarantine; never unlink a changed successor or race an uncoordinated writer. |
| C15 — Walltime misses pre-command helpers | accept | Track C-PREP-DEADLINE and propagate one budget plus reserved cleanup grace through provision/activate/attest/wait/command; retain child and GPU ownership. |
| C16 — Delivery needs producer finality, including legacy fragments | accept | Bind closed manifest to termination, support incomplete-but-final failures and retain legacy evidence pending stable inventory proof. |
| C17 — Several acceptance criteria are vacuous or underdefined | adapt | Specify query classes, semantic reducer comparisons, dual I/O limits, measured profiles, minimum palette support and explicit pause/parity gates. |
| C18 — B map alone omits requested closure obligations | accept | Keep B IDs stable and add an issue-level ledger for all original issues, five source-traced defects, resource inclusion and the pending NULL consumer. |
| C19 — Coverage is aligned only after missing paths are owned | accept | Retain all 36 primary package mappings and incorporate the table's call sites, prerequisites and conditional closure; M33 cannot absorb missing behavior. |

**Validation performed for this report.**

The generator checks the v2 top-level schema plus decisions, all original package field types, 39 unique package IDs, 55 unchanged primary B mappings, 56 unchanged issue dispositions, 118 unchanged closure entries, nine incident mappings, dependency ordering and every shared-file integration order. Each of the eight waves has center/host/drain/legacy and rollback records. The semantic longest dependency chain has 12 packages; this is not a duration estimate. Existing source review surface is 110 files and 31,974 physical lines; no source LOC was changed by this run.

No implementation/integration suite, PostgreSQL instance, service installation, migration, GPU or remote operation was executed. PostgreSQL and deployment acceptance above remains work for the named packages. The guarded import proof succeeded; the existing guard wrote only its routine report-local event log. AIZK tools were unavailable after skill/tool discovery, so the design is recorded here. Source commit: none. Proposed report commit: `docs(mainboard): adopt PostgreSQL and incremental drain-and-switch refactor plan`.

Reports: `design-v3.json` and `design-v3.md` under `/private/tmp/claude-501/-Users-pedro-Developer-projects/b52c03f0-d920-5f82-9aac-285c87707d12/scratchpad/codex/mb`.

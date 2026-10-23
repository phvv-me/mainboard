# Bug ledger

The findings of the 2026-10-10 bug hunt still open on 2026-10-23, with the root cause and fix the hunt recorded, then
those already fixed. The hunt's JSON and the plans it fed were deleted on 2026-10-23; git history keeps them.

## Open

### B02: Captured evidence aliases different hosts and submission attempts (data loss)

- Status: mitigated: every submit is auto-labelled `<host>-<digest>`; the `run-<handle>` fallback serves unlabelled legacy rows only.
- Root cause: Captured uses (stream, handle) and default stream is run-<handle>, dropping target and submitted_at. Monitor.capture treats an unavailable/empty transcript as permission to reuse that ambiguous saved log.
- Proposed fix: Give every submission a durable RunId (target, native handle, submission nonce/time). Key captured logs and receipt associations by RunId, and permit fallback only after an exact identity match. Distinguish unavailable log from valid empty output.
- Evidence: src/mainboard/dispatch/state/captured.py:25, src/mainboard/dispatch/state/captured.py:62, src/mainboard/dispatch/state/captured.py:81, src/mainboard/tracking/base.py:19, src/mainboard/monitor.py:196

### B08: Directory lint can rewrite unchanged sealed inputs and analysis sources (data loss)

- Status: re-checked 10-23, mitigated, owner decision: no step rewrites a file under a registered `node.md` (0bade83, `lint/sealed.py`), and `[lint] exclude` names sources sealed from outside their node. What is left changes what `mb lint <dir>` means: fix only changed or untracked files and check the rest, or find every receipt-sealed source through the lake.
- Root cause: Explicit directories expand to all tracked files and writing tools/hygiene run by default; no snapshot/seal registry protects frozen inputs. Exclusions/-text protect known examples but do not enforce immutable ownership generally.
- Proposed fix: Make sealed/registered inputs immutable by contract and lint them read-only. Default broad directory lint to check for unchanged files while applying fixes only to explicitly editable scope.
- Evidence: src/mainboard/cli.py:1608, src/mainboard/lint/linter.py:79, src/mainboard/lint/linter.py:153, src/mainboard/lint/text.py:46

### B19: Registered acquisition does not enforce a selected frozen source identity (wrong verdict)

- Status: open, not yet re-checked.
- Root cause: A captured current-tree identity proves which bytes ran but does not prove they match the acquisition's registered/frozen intended source.
- Proposed fix: Add an immutable acquisition source reference resolved before closure creation and compare it with registered source policy; use captured bundle identity without requiring Git as the universal source authority.
- Evidence: src/mainboard/dispatch/provenance.py:78-85, src/mainboard/dispatch/provenance.py:125-149, src/mainboard/dispatch/shipment.py:99-120

### B20: Relations text preservation stops at top-level keys (wrong verdict)

- Status: open, not yet re-checked.
- Root cause: The VARCHAR override only marks keys whose immediate value is str. DuckDB still infers temporal types inside nested structures, while records advertises JsonValue but returns raw fetchall values.
- Proposed fix: Define the relation boundary explicitly: recursively preserve JSON strings and normalize records into JsonValue, or expose native values under a distinct typed API and audit JSON serialization callers. Do not silently rewrite historical timestamps without preserving their original representation.
- Evidence: src/mainboard/state/relations.py:47, src/mainboard/state/relations.py:54, src/mainboard/state/relations.py:152

### B21: Settling an old reused handle rebuilds and reports the newer submission (wrong verdict)

- Status: open, not yet re-checked.
- Root cause: The registry stores full (target, handle, submitted_at), but Sweep calls Board.job(handle,target) and the final report resolves cache.run(handle,target) again; both choose the newest reused handle.
- Proposed fix: Carry the selected immutable submission through rebuild, capture, transition, and report. Backend calls must prove the scheduler handle still belongs to that attempt; a stale attempt must never operate on a newer job. Report the exact current submission returned by resolve, not another handle lookup.
- Evidence: src/mainboard/monitor.py:133, src/mainboard/board.py:694, src/mainboard/monitor.py:552, src/mainboard/dispatch/state/cache.py:338, src/mainboard/verdicts.py:608

### B24: Artifact reads and materialization perform one lake query sequence per object (silent stall)

- Status: open, not yet re-checked.
- Root cause: materialize calls _write for each row; Blobs.write lists ordinals then queries each chunk. Artifact verification loops digest recalls; query views materialize many receipt files individually. Existing windowed replicate logic is not reused. Historical timings are not re-measured here.
- Proposed fix: Provide bulk verified read/materialize APIs with bounded windows and chunk streaming for large objects. Query selected run/project indexes before touching files; share reads within a pass.
- Evidence: src/mainboard/state/evidence.py:298, src/mainboard/state/evidence.py:350, src/mainboard/state/blobs.py:134, src/mainboard/state/blobs.py:205, src/mainboard/trials/artifacts.py:139, src/mainboard/state/evidence.py:357

### B26: Experiment-wide fetch scans every run and rejects rewritten shared output paths (silent stall)

- Status: open, not yet re-checked.
- Root cause: Widening fetch repairs sibling evidence stores by collecting an entire experiment home, but every run shares that mutable location. Broad per-pass scans and a fixed transfer deadline remain; conflicting rewrites are not versioned per run.
- Proposed fix: Give each submission immutable output/evidence roots identified by RunId and collect only that manifest with resumable object transfers. Keep a separate shared content store addressed by digest.
- Evidence: src/mainboard/jobs/target.py:109, src/mainboard/dispatch/agent/program.py:998, src/mainboard/dispatch/collection/collector.py:91, src/mainboard/dispatch/transport.py:152

### B29: Host setup accepts an answering pueued without validating its executable or tool links (silent stall)

- Status: open, not yet re-checked.
- Root cause: Successful daemon status short-circuits readiness; dangling PATH links and a daemon running a deleted/old binary are not reconciled. Historical hosts were not queried.
- Proposed fix: Model daemon identity/version/executable and tool-link health in readiness; repair invalid links and plan a safe daemon replacement when its queue is idle.
- Evidence: src/mainboard/dispatch/onboard.py:442, src/mainboard/dispatch/onboard.py:448, src/mainboard/dispatch/wrapping.py:23

### B30: Host sync retains retired generated files that poison the environment identity (silent stall)

- Status: open, not yet re-checked.
- Root cause: An artifact that represents a complete generated input set is transferred using additive individual-file semantics, while its consumer hashes all nonhidden regular files.
- Proposed fix: Ship generated artifacts as a versioned complete manifest with explicit owned membership; atomically replace/prune only that owned input set while retaining installed environments and host-specific activation/bookkeeping.
- Evidence: src/mainboard/dispatch/onboard.py:598-601, src/mainboard/dispatch/dispatcher.py:448-460, src/mainboard/dispatch/mirror.py:98-100, src/mainboard/dispatch/agent/program.py:644-654, src/mainboard/engines/compile/generated/files.py:29-41, repro/host_repros.py::test_generated_inputs_continue_to_include_r

### B33: Sync still requires a uv-managed interpreter even when the workspace Python is usable (silent stall)

- Status: open, not yet re-checked.
- Root cause: Bootstrap establishment and incremental sync share an unconditional installation step, rather than sharing a required interpreter capability check.
- Proposed fix: Resolve and validate an existing suitable workspace/managed interpreter before installing; install only when neither meets requirements, then let sync reuse the resolved interpreter.
- Evidence: src/mainboard/dispatch/onboard.py:263-282, src/mainboard/dispatch/onboard.py:618-625, src/mainboard/dispatch/shells.py:40-59, ISSUES.md:11-15

### B34: The lake read probe does not bound attach or later volume stalls (silent stall)

- Status: open, not yet re-checked.
- Root cause: _answering is cached permanently after one successful read; stat/is_file happen before it, and actual DuckLake ATTACH has no process deadline. The TCC-specific first-read guard cannot bound later or metadata I/O hangs.
- Proposed fix: Put each lake operation that can block on volume I/O behind a cancellable process boundary with a total deadline and structured inaccessible-volume result; avoid opening the lake for unrelated verbs.
- Evidence: src/mainboard/state/lake.py:390, src/mainboard/state/lake.py:400, src/mainboard/state/lake.py:712, src/mainboard/state/lake.py:746

### B35: Upgrade chooses conda versions available only on the center's platform (silent stall)

- Status: open, not yet re-checked.
- Root cause: Conda.latest invokes search with channels but no target platform; the chosen floor is applied to a multi-platform manifest.
- Proposed fix: Select a release satisfiable on every declared target platform and validate the whole candidate lock before replacing manifest state.
- Evidence: src/mainboard/deps/indexes.py:71, src/mainboard/deps/curator.py:119

### B37: lock then install leaves second-stage locks stale; existing-prefix run only syncs pixi (silent stall)

- Status: open, not yet re-checked.
- Root cause: lock returns before stage resolution. The subsequent install bases resolve solely on whether that invocation moved pixi.lock, so it chooses frozen Node install against changed package.json. Existing-prefix synchronized also never syncs the second stage.
- Proposed fix: Model and persist separate input/lock/install digests per ecosystem. lock resolves every stage; install checks every stage independently of whether pixi.lock moved.
- Evidence: src/mainboard/engines/compile/provisioner.py:329, src/mainboard/engines/compile/provisioner.py:353, src/mainboard/engines/compile/provisioner.py:284, src/mainboard/engines/compile/ecosystems/node.py:158

### B40: Literal pytest plugin dependencies are not walked transitively (wasted compute)

- Status: open, not yet re-checked.
- Root cause: Pytest-plugin edges use resolve instead of the common transitive dependency walk.
- Proposed fix: Seed resolved plugin modules into Walker.reach with their owning root.
- Evidence: src/mainboard/jobs/closure.py:203-210, src/mainboard/jobs/closure.py:81-94

### B41: Malformed runner segments are accepted until pytest runs on the host (wasted compute)

- Status: open, not yet re-checked.
- Root cause: Runner controls and pytest arguments share an untyped token bag and lose delimiter ownership.
- Proposed fix: Preserve runner and application segments in a parsed invocation; refuse malformed runner-only syntax before shipment while allowing pytest-timeout in the pytest segment.
- Evidence: src/mainboard/board.py:1365, src/mainboard/jobs/call.py:167-170, src/mainboard/jobs/target.py:80-81, src/mainboard/jobs/call.py:142-145

### B42: Pinned-input preflight neither requires restoration nor checks recorded digests (wasted compute)

- Status: open, not yet re-checked.
- Root cause: _pinned keeps only path and ignores sha256; _restored restores missing paths but ignores the result. EvidenceTree.restore suppresses MissionError and leaves missing objects for the eventual reader. An audit can still run expensive work before finding an absent/corrupt pinned input.
- Proposed fix: Resolve every pin by path and digest before acquisition; verify all hashes and fail collection with a complete missing/conflicting-input list. Keep runtime restore separate from a readiness contract.
- Evidence: src/mainboard/trials/pytest_plugin.py:111, src/mainboard/trials/pytest_plugin.py:130, src/mainboard/state/evidence.py:525

### B44: --env/--lang/--dev silently omit matching ecosystem or platform dependencies (misleading output)

- Status: open, not yet re-checked.
- Root cause: searched builds exact candidate paths and defaults an unspecified ecosystem to conda. Thus --env lab omits lab's Python and platform tables; --lang python also excludes environment/overlay paths unless specifically represented.
- Proposed fix: Filter the full declared-slot inventory by independent scope predicates, retaining all unspecified dimensions. Validate --env even when selection yields no requirements.
- Evidence: src/mainboard/deps/curator.py:189, src/mainboard/deps/slots.py:30, src/mainboard/deps/slots.py:76

### B46: Declared palettes load without the requested perceptual validation (misleading output)

- Status: open, not yet re-checked.
- Root cause: PlotStyle validates data shape but does not check contrast, lightness, chroma or distinguishability; no lint step enforces the owner's declared palette policy.
- Proposed fix: Add a style-validation contract with explicit palette checks reused by load and lint, using the existing house validator rather than duplicating rules.
- Evidence: src/mainboard/manifest/schema/plot.py:8, src/mainboard/plots/table.py:81

### B47: Interactive shell cannot diagnose an already-held PBS slot (misleading output)

- Status: open, not yet re-checked.
- Root cause: Board.interact does not inspect the held-line spool before requesting a second interactive allocation.
- Proposed fix: Consult the shared held-allocation inventory and report owner/expiry and supported reuse path before qsub.
- Evidence: src/mainboard/board.py:644, src/mainboard/line.py:131

### B51: PBS interactive API rejects commands that its own keeper can execute (misleading output)

- Status: open, not yet re-checked.
- Root cause: Pbs.interactive claims PBS cannot take a command and refuses it, while the keeper constructs qsub -I -- /bin/bash itself. Documentation and abstraction disagree.
- Proposed fix: Centralize interactive request construction with an absolute executable contract and login-shell PATH policy, and have the keeper reuse it.
- Evidence: src/mainboard/dispatch/schedulers/pbs.py:173, src/mainboard/line.py:166, src/mainboard/line.py:289

### B53: Source mismatch refusal omits change context and recovery action (misleading output)

- Status: open, not yet re-checked.
- Root cause: The refusal says only source blob mismatch and path, leaving the operator unable to distinguish a concurrent edit from stale mirror/corruption without manual inspection.
- Proposed fix: Include expected/actual digest, observed file time and capture time, and an explicit rebuild/retry action; preserve the refusal.
- Evidence: src/mainboard/dispatch/agent/program.py:868, src/mainboard/dispatch/snapshots.py:277

### B55: upgrade has no non-mutating preview (misleading output)

- Status: open, not yet re-checked.
- Root cause: --frozen writes the manifest and --no-install still solves; neither provides the requested dry-run contract.
- Proposed fix: Produce a candidate manifest/lock diff without publishing it, using the same candidate transaction as upgrade.
- Evidence: src/mainboard/cli.py:752, src/mainboard/deps/curator.py:197

## Fixed

| id | finding | how |
|---|---|---|
| B01 | Eviction deletes the last intact copy; re-ingest cannot repair the corrupt lake object | fixed in wave 1 (`evictable` needs an intact lake object) |
| B03 | A transient provider log failure is treated as complete evidence and releases the rental | fixed 10-23: `transcript()` answers None for a failed read; `Monitor.capture` keeps the run pending (test_evidence) |
| B04 | host release kills its running job despite promising to let it finish | fixed in wave 1 (claim before tombstone, `Remote.stop`) |
| B05 | Automatic pruning deletes source snapshots still referenced by held/PBS jobs | fixed in wave 1 (`Retention` inventory: spool inbox, claims, staged scripts, registry; unknown keeps all) |
| B06 | Overlapping needs corrupt the mirror input link | fixed 10-23: the closure keeps only the outermost of nested needs, normally spelled, and the host refuses a need whose parent is reached through a link (test_nested_needs) |
| B07 | Center migration omits an externally configured lake | fixed 10-23: the workspace parcel snapshots `Lake.at(root)`'s catalog and ships its data when the lake is inside the workspace; an external lake is named in `left`, and the destination's `doctor --center` lake row fails until it reaches the catalog (test_center_lake) |
| B09 | Lake path containment is lexical and follows symlinks outside the requested boundary | fixed (commit cites it) |
| B10 | Materialize replaces an edited file without reporting a conflict | fixed (commit cites it) |
| B11 | A new reused handle inherits the old attempt's silence and can be declared stalled immediately | fixed 10-23: pulse key carries `submitted_at` (pulse._key) |
| B12 | GPU leases do not contend across universes or overlapping device masks | fixed 10-23: one lease per card UUID, host-wide in the user cache (trials/lease.py, test_card_lease) |
| B13 | Held cancellation reports success when its durable fence failed | fixed (commit cites it) |
| B14 | Lane submission silently drops execution and data options | fixed 10-23: a lane forwards fetch, needs, env and container (test_lane_submit) |
| B15 | Namespace packages bypass the first-party closure guard | fixed 10-23: the inventory names namespace portions, and the guard passes a portion only when the closure ships it or an installed distribution provides it (test_namespace_guard) |
| B16 | PBS acknowledges cancellation even when qdel refuses | fixed (commit cites it) |
| B17 | PBS command-line select overrides erase GPU requests rendered in the script | fixed 10-23: a scripted `qsub` leaves the header's `select` chunk alone (test_pbs_flags) |
| B18 | Pueue Locked and Stashed tasks are classified as terminal failures | fixed (commit cites it) |
| B22 | Two simultaneous GPU sessions can both acquire the same lease; the former owner can delete the newer lease | fixed 10-23: lease written whole and linked; stale reclaimed by rename; release only by its token |
| B23 | A held settlement lock is invisible while live listing and cached verdict disagree | fixed (commit cites it) |
| B25 | Dependency rollback leaves new locks and misses filesystem failures | fixed 10-23: any failure (an OSError too) puts the manifest and lock back, removing a lock the edit created; the generated shard follows `mb.lock` on its next compile (test_lock_policy) |
| B27 | Held accepts jobs that can never fit any allocation in the line | fixed 10-23: admission (`Facts.admit`) refuses a job whose walltime and grace exceed one allocation, or whose memory exceeds the line's (test_held) |
| B28 | Held allocation lengths at or above 24 hours wrap modulo one day | fixed 10-23: one `walltime()` formatter, hours past 24 kept (runtime/job.py, test_held) |
| B31 | PyPI upgrade ignores packages whose actual provider is conda | fixed 10-23: conda-held Python requirements raised against the conda index (curator `pinned`, 10-23) |
| B32 | Settlement repeatedly replays every event in a shared stream | fixed (commit cites it) |
| B36 | Upgrade selects a yanked PyPI release | fixed 10-23: `_unyanked`: yanked releases skipped (10-23) |
| B38 | Fresh timeout kills pytest but leaves child workers alive | fixed 10-23: a fresh cell runs under `Processes.timeout`, which stops the whole tree (test_fresh_timeout) |
| B39 | Lane --wait skips remaining handles after first failure | fixed 10-23: lane `--wait` waits every handle |
| B43 | Split lanes flatten target and pytest arguments into one target token | fixed 10-23: lane jobs keep the target's own tokens |
| B45 | Calendar versions receive a next-day upper bound | fixed 10-23: a lone date stamp gets a floor only (10-23) |
| B48 | JobSpec documentation still describes the old PBS timeout split | fixed 10-23: already fixed on re-check: `JobSpec.job` documents the one runner deadline beside a queue's own limit, and the class names only what a PBS render requires |
| B49 | Lake has no verified inventory of tree files eligible for eviction | fixed (commit cites it) |
| B50 | Ordinary command help can be intercepted by mainboard | fixed 10-23: a later `--` ends this tool's options only after a job target (`file.py`, `file.py::name`), so `run tool --help -- x` reaches the tool (test_commands) |
| B52 | Query detects lake usage by text substring | fixed 10-23: already fixed on re-check (61c7885): `_connected` reads DuckDB's own tokens, so `"lake".runs` attaches and a literal or comment does not |
| B54 | The jobs query view exposes server while other job surfaces use target or host | fixed 10-23: the `jobs` view names the column `host`, as `job list` does (test_query_stream); one typed job schema behind every surface stays with the SubmissionPlan wave |

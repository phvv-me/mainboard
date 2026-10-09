# Open issues

Found while running real work, kept here to fix later with care rather than in the middle of a
campaign. Each entry says where it showed up, what it costs, and what is already done. Newest
campaign first; move an entry to the bottom section once fixed and verified.

## Found building `host hold` for PBS, 2026-10-09

### Mainboard

1. **`mb host sync miyabi-g` cannot run.** Its first step, `uv python install --no-bin '>=3.14'`,
   fails with `No download found for request: cpython->=3.14-linux-aarch64-gnu`: uv has no
   aarch64 CPython 3.14 build. The sync-only path should skip the managed-Python install when
   the host's environment already runs a satisfying interpreter. Worked around by calling
   `Dispatcher.mirror(plan, root)` directly, which is all a code-only update needs.
2. **`Pbs.cancel` ignores `qdel`'s exit status** (`dispatch/schedulers/pbs.py`), so a refused
   cancel reads as done. `Line` checks its own `qdel`; `Pbs.cancel` should raise unless the job
   had already finished.
3. **`snapshots.queued_here()` only asks the local pueue daemon.** A pinned tree that a queued PBS
   job, or a held job waiting in the spool inbox, will run from is invisible to pruning once the
   age thresholds pass. Include `claims/` and `inbox/` entries of the spool (their `cwd`) and the
   scripts of queued PBS jobs.
4. **`mb shell --on miyabi-g` knows nothing of a held line.** PBS refuses the second interactive
   job with its raw quota text. `Board.interact` could read the spool's `line.json` and say who
   holds the slot and until when.
5. **`login_run` had no deadline**, so a responsive ssh over a hung Lustre read waited forever
   (the review's point 11). Fixed: `login_ask` bounds every probe at 90 s and raises
   `HostUnreachable`.
6. **The workspace ruff (newer than the package's pinned one) flags `Iterator` on context
   managers** in `dispatch/agent/program.py`, `render/human.py` and `trials/flags.py`; the
   package's own `ruff` passes. Align the two versions.
7. Held jobs get no walltime from a queue, so the runner now enforces it under PBS too
   (`jobs/spec.py`); a PBS queue that kills first never lets the runner's deadline fire, but the
   `JobSpec` docstring still describes the old split.
8. **`qsub -I -- <command>` needs an absolute executable.** `pbs_mom` execs the command itself
   with a bare PATH (`/bin:/usr/bin:/usr/local/bin`), so `-- bash script` ends "exec of bash
   failed" while `-- /bin/bash script` runs on the node with a pty as stdin (measured 2026-10-09,
   job 3517082). A tmux pane started by a long-lived server also has no `/opt/pbs/bin` on PATH
   (`zsh: command not found: qsub`), so the keeper runs under `bash -l`. `Pbs.interactive`'s
   docstring still says PBS takes no command; it does, absolutely spelled.
9. **A held submit costs 70-80 s of dispatch** (mirror of a few paths, two login-shell
   activations in `_verify`, the pin) where the old spool took a second. Measure which step
   dominates; a hot line wants a cheaper path for a repeated command.
10. **Miyabi's `qstat` wrapper caches.** A job shows `RUNNING` with a frozen ELAPSE for minutes
    after it ended (3517146 read `00:04:59` after the keeper logged its end; 3516531 stayed
    `01:58:42` for four minutes). `Held` asks PBS only about allocations whose claims have no
    exit artifact, so it is safe, but any wait on a PBS state lags by that cache.
11. **PBS refuses a new `qsub -I` for about 30 s after the previous interactive job ends**
    (`exit 39`, the `njobs_int-g` count). A line's renewal gap is therefore the keeper's retry
    interval (10 s) plus that; measure the real gap before promising continuity.

## Found during the cutok GH200 transfer, 2026-10-07

### Fixed locally, not pushed (mainboard `ef2809d`, root `d3dda40be`)

1. **Every trial's receipt failed to serialize.** `Relations.rows` staged rows as NDJSON and let
   `read_json` infer types, so strings holding an ISO instant, a date, a clock time or a UUID
   became TIMESTAMPTZ/DATE/TIME/UUID in the receipt Parquet. They read back as Python objects
   (instants in the reader's zone, JST on the center) and `ledger.wire` raised at pytest session
   close. Fix: text-valued keys are pinned to VARCHAR. Follow-ups:
   - `state.relations.records` is annotated `JsonValue` but returns raw DuckDB values (datetime,
     UUID, Decimal). Decide: convert there, or change the annotation and audit every caller.
   - Parts written between 840a4ab and the fix hold TIMESTAMPTZ; unioned with new VARCHAR parts
     they read as `2026-10-07 19:00:00+09` text. Find and rewrite or retire them.
   - Other typed `read_json` uses (`results.py` jobs/events, `lake.py` cells) infer the same way.
2. **`mb job show --wait` crashed** with `AttributeError: dropped`: `schema.latest` assumed every
   log table has a `dropped` column; `pulse` does not. Fix: filter drops only where present.
3. **DuckDB settings had no home** (owner request). Done: `state.database.connect` is the one
   factory; `[duckdb]` in `mainboard.toml`, `MB_DUCKDB_<NAME>` from `.env` or the environment,
   UTC default. Follow-ups: `Lake.attach` still forces `TimeZone`/progress bar by itself (by
   design for lake reads, but document it in `mb help`); check `mb query`'s own connection uses
   `connect`; consider exposing `mb doctor` output of the effective settings.
4. Release: mainboard's `publish.yml` runs on every push to `main`. Pushing `ef2809d` needs the
   owner's go-ahead and probably a 0.5.2 bump; the root commit `d3dda40be` points at it, so the
   root cannot be pushed first.

### Mainboard

5. **`__pycache__/*.pyc` under `packages/mainboard` enter job source closures**, so the seal digest
   changes whenever the center runs `mb` between dispatches (`4328e3…` vs `fd9931…` differed only
   by .pyc files). `GitignoreFilter` does not drop them for that submodule. Seals should hash
   source only.
6. Editing any closure file while a dispatch pins sources is refused with `source blob mismatch`
   (correct), but the message could name the file's change time and suggest re-running.
7. **`mb upgrade` raised cutok's Python to `>=3.15.0rc3`**, a pre-release, and still left the lock
   unchanged. Upgrade should skip pre-releases unless asked.
8. **`mb upgrade` has no `--dry-run`**, which `pixi upgrade` has; "mb verbs are pixi's" says the
   flags should match.
9. **`mb lint <dir>` rewrites files the change never touched**, including frozen analysis sources
   (`lifecycle_figures.py`, reverted by hand). Directory lint should check, not fix, untouched
   files, or refuse to rewrite sealed sources. 2026-10-08: it also rewrote
   `cutoken/src/cutoken/types.py` (`from numba import cuda as cuda` to the plain form) although
   cutoken's own `pyproject.toml` exempts that file from PLC0414, because the alias is the
   explicit re-export ten kernel modules import. Lint must honor each package's nearest ruff
   config (hierarchical discovery), not force the root's.
   2026-10-09, pyrefly: `[lint.tools.pyrefly]` runs `pyrefly check .` in each owner, and an
   explicit path bypasses that owner's `project-excludes`: cutoken reports ~500 errors from its
   device modules where its configured run (`pyrefly check`, no path) reports 1. Dropping the `.`
   everywhere is wrong too: an owner without a pyrefly config (`scripts`) then climbs to an
   ancestor config and checks 899 errors' worth of other code, where `.` checks its 0. Fix in the
   runner: no path where the owner's own pyproject has `[tool.pyrefly]`, `.` otherwise (ty
   applies its excludes either way: 682 both forms in research/cuda-tokenization).
   FIXED locally (unreleased): a tool's `path-when-unconfigured` is appended to its command in an
   owner whose own `pyproject.toml` has no `[tool.<name>]` table; the root `mainboard.toml` sets
   it to `.` for pyrefly and drops the `.` from the command. `mb lint --check --only pyrefly` on
   a cutoken device file reports nothing, on a `scripts/` file with a type error reports it as
   before.
10. **Shipping `needs` to miyabi ran at ~2.4 MB/s** through the agent's ssh pipe: 2.1 GB of corpora
    took about 15 minutes per first dispatch. Compression, rsync, or a staged copy would help.
10a. **Collecting from miyabi times out.** A `job list` settle pass on 2026-10-07 22:20 gave up on
    3500519, 3500521 and 3500602 with "ssh collect to 'miyabi-g' timed out after 600s": the
    same slow link as issue 10, now on the way back. Large evidence needs a resumable transfer
    or a longer, size-aware deadline. Root of it: every cutok job's `fetch` is the whole
    `datasets/experiments/current_engine_refresh` folder, 9.4 GB in 78,000 files on miyabi by
    2026-10-08, so one collect pass must sync all of it and a failed pass leaves the host
    "quiet" for the rest of the pass. Settlement then stays pending (cold-native 3500492 ran
    12 of 12 cases but none reached the center). Worked around with a manual `rsync --partial`
    of that folder. Fetch should name a job's own run directories, not the experiment's.
    Fixed locally (unreleased): collection counts paths the lake indexes as known, so an evicted tree no longer
    pulls the whole folder back; the remote side still hashes every known file on the login
    node, which is what makes a pass slow. FIXED locally (2026-10-09, unreleased): `pack.py`
    keeps a remote digest cache (path, size, mtime) under the workspace's `run/`, seeded once
    from a compute node (170,011 files, 20 GB), and runs on the host's workspace-environment
    Python, never the login node's OS `python3`.
    Measured 2026-10-09 on a GH200 node over 182,903 files: the scan cost 2 min of per-file
    metadata (lstat 49 s, resolve 35 s, stat 35 s; now one `lstat` per file through `scandir`,
    containment checked once, `known` sent zlib-compressed: 133 s, 16 MB), and a pass moved
    9.6 GB: 71,701 files under `evidence/artifacts/<run>/<trial>/objects/<sha256>`, each run
    holding its own copy of objects the lake already kept (cutok's artifacts referenced 59.7 GB
    of which 4.1 GB was unique). The 10-09 backlog was moved once as an archive (rsync) and merged.
    FIXED locally (2026-10-09, unreleased, owner-approved):
    - Artifacts by reference: a trial's bytes go once into its node's content store,
      `<node>/evidence/objects/<sha256[:2]>/<sha256>` beside its receipts, shared by every run on
      the host; the receipt references it (digest, size, media type, and `source`, e.g.
      `hf://<repo>@<revision>/<file>`, read from a hub cache path or passed to `log.artifact`).
      Per-run `objects/` of older runs read as before.
    - A host sends no file whose digest the center's lake keeps: `pack` names it in
      `mainboard-held.json` and `Evidence.adopt` indexes its path from the blob. An unchanged
      live event log is not sent again (its `collected-<start>-<end>` name is known).
    - Members are zstd level 3: 1.5 GB of cutok evidence went 3.9x smaller for 7.5 s of one
      GH200 core (level 1: 3.5x, 3.7 s; level 6: 4.3x, 16.6 s).
    - `mb lake dedup` hard-links identical settled files to one copy (verified digests, runs
      quiet for an hour, nothing deleted). 10-09: miyabi 24.4 to 2.4 GiB of cutok evidence
      (104,328 names linked, 225 s on a GH200 node), pedro-cvlab 3.78 to 0.62 GiB, crimson and
      gold 81 MB to 80 MB.
    - Collection keeps each transfer in the lake and writes nothing to the tree; it stages in
      `<out>/tmp/collect-<pid>-*`, and a dead pass's staging is cleared when the next one starts
      (31 root-level `.mb-collect-*` dirs, 7.2 GB, had piled up). A trial session on the center
      moves its run into the lake (kept, then evicted); a dispatch mirror's staging lake keeps
      nothing.
    - A settle pass collects each host's fetch path once, not once per run: the 10-09 pass
      before the fix collected the same folder from miyabi 24 times in 2,798 s; the next pass,
      with it, took 147 s with one collection (76 s alone, of the 600 s deadline; the earlier
      pass had already moved everything, so nothing new crossed).
    Still open: a job's fetch still names the whole experiment folder, whose remote scan
    (~50 s on miyabi's login node) is most of a collection now.
11. **Settlement backlog**: `job list` keeps erroring on old runs ("settlement pending; remote
    evidence retained: result transfer failed" for pedro-cvlab 348, 363, 400, 446, 479 and
    crimson 1862) and on today's 594 and 3500391 ("Expecting value: line 1 column 1"), whose
    receipts were never written (issue 1). Settle or retire them explicitly.
12. DuckDB prints a nanobind leak report at interpreter exit in every job log ("leaked function
    fetchdf ..."): relations or connections alive at exit. Close them, or it hides real errors.
13a. **Reading a cohort's artifacts back is slow on the center.** Re-rendering the admitted v18
    available3 roster report spent about 24 s per process in `Artifact.read`, so 126 processes
    took over 50 minutes before the audit started. Measure where the time goes (lake lookup per
    object, decompression, hashing) and batch it.
13b. **Materializing a whole experiment's datasets** wrote 9.6 GB without reaching the one file a
    renderer needed, on a center disk 93% full. Renderers should read through the lake, or say
    exactly which paths to materialize. Since 0.5.4 `mb lake evict` gives the disk back after a
    materialize, but renderers that read through `Path` still need the tree.
13c. **The center's disk is nearly full** (2026-10-08 03:20: 12 GiB free of 460, 98%). The lake
    keeps 48 GB under `.mainboard/lake` on the internal disk, and tree copies under
    `research/cuda-tokenization/datasets` hold 29 GB, much of it duplicates of lake objects.
    There is no `mb lake evict` to drop tree copies the lake verifiably holds, and no way to see
    which tree files are cached copies. Add both, and consider the lake on `/Volumes/PORTABLE`.
    Done 2026-10-08: the lake moved to PORTABLE (0.5.3, 52 GiB free) and `mb lake evict`
    (0.5.4) dropped 151,166 cutok tree copies (87 GiB free). Still missing: a listing of which
    tree files are cached copies.
13d. **`[workspace] lake` (2026-10-08) is not yet known to center migration.**
    `center/state.py` ships `lake.sqlite` and the data as part of `.mainboard`; with the lake
    elsewhere, `host setup --center` should carry it from `Lake.home` (or leave an external
    volume where it is). `doctor` should also report the lake's home and refuse clearly when the
    volume is unmounted.
13f2. **An unresponsive lake volume hangs every verb, without a timeout** (2026-10-08 ~10:00).
    With `[workspace] lake` on `/Volumes/PORTABLE` and that drive mounted but not answering I/O
    (even `ls` blocks), `mb host sync <host>`, `mb lint` and three concurrent syncs all sat at
    0% CPU for over ten minutes after loading the DuckLake and SQLite extensions. Lint and sync
    should not need the lake at all unless they restore files; where they do, opening the lake
    should time out and name the volume, and `doctor` should report it.
    Cause found: macOS privacy (TCC) denies Full Disk Access to processes Ghostty is
    responsible for, so opening a file on the external volume waits forever while `stat`
    still answers. FIXED locally (unreleased): the manifest consults held rentals only for an
    alias neither declared nor `local`, so sync, lock, lint and run never open the lake.
    2026-10-09: the same wait hit every pytest trial started inside tmux (the agent sessions):
    the tmux server is its own responsible process and holds no Full Disk Access, so attaching
    the catalog never returned. FIXED locally (unreleased): `Lake.attach` reads the lake's home
    once per process first and raises after 10 s, naming the missing access. Granting tmux the
    access would restart the terminal, so agent sessions run lake verbs through `ssh localhost`,
    whose sessions hold Full Disk Access (Remote Login allows it); the message says so.
13f3. **Every job host inherits the center's absolute `[workspace] lake`** (2026-10-08).
    Since the lake moved to `/Volumes/PORTABLE/mainboard-lake` (9ffcc472d) the shipped manifest
    names that path on every host, and a pytest trial there dies in `pytest_configure`
    (`trials/provenance.source` -> `dispatch/provenance.archive` -> `Lake.ready` -> `create`)
    with `Unable to open database "/Volumes/PORTABLE/mainboard-lake/lake.sqlite"` on Linux.
    The key names the center's lake only: sync should ship the manifest without it (a host keeps
    its staging lake in its generated directory, as before), or `lake_home` should apply it on
    the center alone. Registered trials on hosts are blocked until then.
    FIXED locally (unreleased): `lake` applies only on the center, the checkout holding
    `.git`; a dispatch mirror ships without one and keeps its staging lake.
13f5. **`mb lake ingest <relative path>` fails at its end** (2026-10-09). `mainboard lake ingest
    research/cuda-tokenization/datasets` (run from the workspace root through `ssh localhost`)
    raised `ValueError: 'research/cuda-tokenization/datasets' is not in the subpath of
    '/Users/pedro/Developer/projects'` from a `relative_to` on the unresolved argument, after the
    ingest windows had committed (the 611 files were indexed with intact blobs; a following
    `evict` verified and dropped them). Resolve CLI paths against the workspace before any
    `relative_to`, and record the import run.
    FIXED locally (unreleased): `ingest` resolves its paths before it files them, so the import
    run is recorded under the workspace-relative source; `evict`, `materialize` and `dedup` share
    `state.evidence.within`, which resolves a path (relative to where `mb` runs) against the
    workspace and refuses one outside it with a `MissionError` (`dedup` raised a bare
    `ValueError`). `test_every_lake_verb_takes_paths_relative_to_where_it_runs` covers all four.
13f4. **Stale generated inputs on a host break every dispatch's environment pin** (2026-10-08).
    The compiled artifact ships as named files, so a file the center stopped generating stays in
    the host's `.mainboard/envs/default` forever. `GeneratedFiles.inputs` digests every file
    there, so miyabi-g's `package-lock.json` (2026-09-29, from before the pnpm second stage)
    and `activation-windows.json` (2026-10-01) give `df134abbb018105e` against the center's
    `e5b7ed78`, and `job submit` refuses: "could not build default ... describes environment".
    `host sync` and `host setup` both pass, since neither removes them. Fix: prune generated
    inputs the center no longer ships when the artifact lands; until then, delete them by hand.
13e. **Palettes are only validated by hand.** The 2026-10-08 Hugging Face colour change came from
    running the dataviz skill's `validate_palette.js` outside mainboard. `[plots.*]` colours and
    palettes should be checked (lightness band, chroma, colour-blind and normal-vision
    separation, contrast) when a style loads and in `mb lint`.
13f. **Hashing could move to DuckDB with a fast hash** (owner idea, 2026-10-08). xxh3 as a change
    detector for evict, collection and the KeptDigests cache, with SHA-256 kept as the identity
    every receipt, seal and lake object is addressed by. Measure first: evict hashed 151,166
    files in 36 s, while materializing one report folder took over 600 s, so hashing is not the
    current bottleneck. The remote `pack.py` must stay standard-library only. Source: DuckDB's
    `hashfuncs` community extension (Query.Farm) has `xxh3_64`, `xxh3_128` and `xxh3_128_hex`,
    the last matching Python `xxhash.xxh3_128().hexdigest()`; it needs `INSTALL ... FROM
    community`, while `Relations` disables extension autoinstall.
13g. **`mb lake materialize` reads one object per query** (2026-10-09, v25 settlement): about 10
    files/s, so 69,000 files projected past 2 h, and eight parallel runs gained nothing (load
    average above 90). A windowed `Blobs.read` restored the same files in about 2 min; reuse
    `replicate`'s windowed read. Since `Artifact.read` already falls back to the lake by digest,
    an audit needs no materialized tree at all, only pinned path readers do.
13h. **A bare `mb query` over the whole `trials` or `artifacts` view costs 3-9 CPU minutes**
    (2026-10-09). Filter by seal or project, or index the views by seal.
13i. **An audit that needs a pinned file absent from the tree fails after all its work**
    (2026-10-09): the v25 roster audit reached `bulk-baselines-20261006-v17/support-snapshot.json`
    last and wrote `admitted: false`. Resolve every pinned input (from the lake) before the audit
    reads anything.
13j. **A job's `fetch` folder on the host is not scoped to the run** (2026-10-09, cutok smoke 621
    and 622 on pedro-cvlab): a second run of the same test overwrote the first's uncollected
    files. Settling 621 then failed on every pass with "no lake around .../stream-hypotheses-v26
    holds e5abe45f...", since neither the host nor a lake still held what its manifest pinned.
    The rerun's collection was refused as conflicting evidence. Both runs only settled as failed,
    with "result transfer failed". Write each run's fetch under a run-scoped staging folder on
    the host, or collect before the next run of that target starts.
13k. **`mb upgrade` (2026-10-10)**, three defects, two fixed locally:
    - It crashed on `[workspace] members` (`NonExistentKey: llm-head`), since the manifest model
      lists members the text never declares. FIXED: `ManifestText.versioned` answers False for
      an undeclared name.
    - It raised `python` to `>=3.15.0rc3, <4`. FIXED: `_newest` takes the newest final release,
      and a pre-release only when nothing final is listed.
    - Open: a raised entry keeps its comment at its old character offset, so a longer pin breaks
      the column.
    - Also, the manifest is written before the solve, so a failed solve leaves every
      `mb run` re-locking against a manifest that cannot solve until it is restored by hand.
      Write only after a successful lock, or restore the text on failure.
13. The miyabi agent bootstraps with the login node's OS `python3` (3.9) (`python3 -c ...
    mainboard-agent`). It works, but contradicts the rule that nothing runs on the OS interpreter;
    document it as the one exception or ship the agent's own interpreter.
    FIXED locally (unreleased, 2026-10-09): `Dialect.python` runs the env's Python, else uv's CPython.
14. A path wheel on one platform and an index wheel on another is refused ("one source per
    package"), so TokTier's released x86_64 wheel had to be pinned by path too. Fine, but the
    message could suggest that pattern.

### Hosts

15. **Dangling `~/.local/bin/pueue` → `~/mainboard-managed/...`** on pedro-cvlab and purple after the
    October 7 wipe; plumbum resolved it first and the 4090 submit failed with exit 127. Removed by
    hand on both. `host setup` should detect and repair dangling tool links.
16. pedro-cvlab's `pueued` runs from a deleted binary (`~/.mainboard-jobs.aside-r5/...`, up since
    October 1). Restart it under the pixi-global pueue at a quiet moment.
17. `macmini` is a declared host that does not resolve from the Mac mini itself, so `doctor` always
    warns about the fleet. Mark the center's own entry as such, or drop it.
18. pedro-home still answers ssh and sits in `~/.ssh/config`, but left mainboard's hosts with the
    Windows drop. Decide whether to keep it as a plain ssh box or decommission it.
19. Miyabi announces a scheduled stop at 2026-10-28 09:00 JST, and short-g estimated start times of
    10–36 hours for 24 jobs requested with generous walltimes. Request walltimes near the expected
    runtime so jobs backfill.

### Cutok

19a. **TokTier's artifact cache is per host, and GH200 compute nodes are offline.** TokTier routes
    Qwen3, DeepSeek, gpt-oss and Llama 3 natively and needs `~/.cache/toktier/artifacts`;
    miyabi had only a partial cache, so GH200 TokTier input-size allocations 3500491/3500493
    failed at `qwen3-100` with `ArtifactNotFound ... fetching is disabled`. The `72-0-gpt2` gate
    could not catch it, since TokTier routes GPT-2 through HF. The 4090's exact cache (22
    entries) was copied to miyabi and checked file by file. Host setup for an environment
    should stage such caches, and a gate should cover one natively routed model.
20. Analysis renderers were hard-coded to the RTX 4090 (host, UUID, 32 workers, titles). Being made
    machine-aware for v22 (`machines.py`); `bulk_figures.py` (v15) still is.
21. Contract tests error on hosts without the center's datasets (the support pin lives under
    `datasets/`), and `test_scaling_figures.py` cannot import in the cutok environment (no
    seaborn). They only pass on the center.
21a. Three `test_scaling_figures.py` contract tests fail for the input study on the committed
    renderer too (`test_full_grid_and_derived_outputs[input]` expects 3 plots and gets 9, plus
    `test_route_plot_uses_process_observations[input]` and
    `test_late_route_appearance_has_zero_share_in_prior_calls_and_processes[input]`). The
    admitted v20 report draws nine figures (three per document length), so the test fixture is
    probably stale. Check and fix the tests, not the renderer.
22. Pre-existing type errors in `current_engine_refresh`: `os.sched_getaffinity` on macOS,
    incompatible mixin overrides in the TokTier classes, `Log.metrics` given a tuple.
23. TokTier publishes only an x86_64 wheel; the GH200 uses a source build of the v0.2.9 tag
    (`research/cuda-tokenization/wheels/README.md`). Ask upstream for aarch64 wheels.
24. **A pinned upstream input was edited by an end-of-file fixer.** Commit `e72cacd1a`
    (2026-09-25, "retire the lint hooks") gave GPT-2's `encoder.json` under
    `current_engine_refresh/inputs/tiktoken/` a trailing newline. RTX 4090 job 557 (Oct 6) and
    GH200 allocation 3500494 (Oct 7) refused it before measuring; Oct 6 only restored the live
    file, so the committed blob kept the newline and this morning's CRLF rewrite from the index
    brought it back. Fixed in `b5e18d458` (exact bytes, `-text`, lint exclusion
    `**/experiments/*/inputs/`). Still to do: find which tool added the newline, and hash-check
    every other pinned input and resource in the workspace against its registration.

## Fixed and verified

(none yet)

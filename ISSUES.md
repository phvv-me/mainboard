# Open issues

Found while running real work, kept here to fix later with care rather than in the middle of a
campaign. Each entry says where it showed up, what it costs, and what is already done. Newest
campaign first; move an entry to the bottom section once fixed and verified.

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
   files, or refuse to rewrite sealed sources.
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
    exactly which paths to materialize.
13c. **The center's disk is nearly full** (2026-10-08 03:20: 12 GiB free of 460, 98%). The lake
    keeps 48 GB under `.mainboard/lake` on the internal disk, and tree copies under
    `research/cuda-tokenization/datasets` hold 29 GB, much of it duplicates of lake objects.
    There is no `mb lake evict` to drop tree copies the lake verifiably holds, and no way to see
    which tree files are cached copies. Add both, and consider the lake on `/Volumes/PORTABLE`.
13d. **`[workspace] lake` (2026-10-08) is not yet known to center migration.**
    `center/state.py` ships `lake.sqlite` and the data as part of `.mainboard`; with the lake
    elsewhere, `host setup --center` should carry it from `Lake.home` (or leave an external
    volume where it is). `doctor` should also report the lake's home and refuse clearly when the
    volume is unmounted.
13e. **Palettes are only validated by hand.** The 2026-10-08 Hugging Face colour change came from
    running the dataviz skill's `validate_palette.js` outside mainboard. `[plots.*]` colours and
    palettes should be checked (lightness band, chroma, colour-blind and normal-vision
    separation, contrast) when a style loads and in `mb lint`.
13. The miyabi agent bootstraps with the login node's OS `python3` (3.9) (`python3 -c ...
    mainboard-agent`). It works, but contradicts the rule that nothing runs on the OS interpreter;
    document it as the one exception or ship the agent's own interpreter.
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

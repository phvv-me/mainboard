# mainboard

Run batch jobs anywhere without caring about system setup.

One file, `mainboard.toml`, declares your dependencies, your environments,
your container base images, and every machine you run on, from your laptop
to a PBS supercomputer to a cloud GPU provider. One interface runs, submits,
tracks, probes, and profiles across all of them.

```python
from mainboard import Board

board = Board()  # finds the nearest mainboard.toml
board.run("python train.py")  # here, in the activated environment
board.on("gold").run("nvidia-smi")  # any ssh box, same call
job = board.on("miyabi-g").submit(  # a PBS cluster, inside an NGC container,
    "python -m experiments.run",  # with queue policy checked before any ssh
    walltime="06:00:00",
)
job.wait()
print(job.logs())
job.pull()
```

The same surface as a CLI:

```console
$ mainboard run --on gold nvidia-smi -L   # everything from the command on is its own
GPU 0: NVIDIA GB10 (UUID: GPU-6a5c...)
$ mainboard facts --on gold | head -4
{
  "schema_version": 1,
  "hostname": "gold",
  "cpu_name": "10x Arm Cortex-A725 + 10x Arm Cortex-X925",
$ mainboard submit --on miyabi-g --attempt 2 python -m experiments.run
2231259
$ mainboard monitor --json          # one durable pass, what a cron runs
{"running": 1, "finished": [], "failed": [], "unreachable_hosts": [], "changed": false}
$ mainboard jobs                    # every live job as its queue sees it, cells, silence, GPU%
$ mainboard wait 2231259            # cells and a heartbeat on stderr; exit 4 on a stalled job
$ mainboard compute --agent         # every path this workspace can run on
name      kind      access       detail                 usd_hr  credit_usd
local     local     here         1x RTX 4090, 135 GB RAM
gold      ssh       provisioned  cached default: hardware from onboarding
miyabi-g  pbs       unreachable  ssh connect timed out
vast      provider  keyed        1x RTX 4090 Sweden, SE  0.2978  99.9968
```

`mainboard help batch run` opens that command's help. Other queries, such as
`mainboard help Log.read_table` or `mainboard help separate trace pass`, search
command descriptions, this README, and Python API docstrings. Results name their
source locations. The wheel includes the README. API search never imports
the scanned modules. Broad searches show twenty hits and the full match count.

`compute` answers what there is to run on before anything is dispatched: this
machine, every declared host with whether it answers and whether it was set up,
and every provider with whether its credentials are here and what the account
has left. No credential is ever printed, only whether one was found.
`provisioned` means a cached setup record exists, not that a job can currently run.
`observed_at` timestamps the live survey; `cached_at` timestamps the retained host
facts, which may be stale. An SSH echo probe works with POSIX shells, cmd, and
PowerShell without requiring an installed environment. Neither cached hardware nor
a successful login proves GPU availability. For PBS/Slurm, the reachable endpoint
is the login host, not an allocated compute node. Inspect `mainboard jobs` and
`mainboard facts --on <host>` before choosing a target.
A host's `[hosts.<name>.vars]` may contain `status-note` to describe a supported
route or known restriction. This replaces generic setup advice, not the observed
access state, and never makes a host job-ready.


`jobs` shows every dispatched job still in flight before it shows any that
settled, each with what its own scheduler says about it right now, and asks each
host once for all of them: one `qstat`, one `squeue`, one `pueue status`. A
listing that had to leave anything out says so rather than stopping quietly at a
limit, and `--limit` bounds only the settled tail.

`monitor` collects outstanding results into durable job records and study ledgers.
Its reports identify changes and unreachable hosts without repeating unchanged
outcomes on later passes.

`run` executes native file targets locally. Use `submit` for remote jobs.
Collection and help stay local. Plain diagnostic commands use SSH.
On a cluster, SSH reaches the login endpoint. It provides no batch allocation.
Windows diagnostic arguments preserve embedded quotes and empty strings through native process
creation, including `python -c` source. Native Windows collection and direct runs are supported;
queued submissions remain unsupported. HPC-AI catalog prices absent from the provider response
are unknown (`null`), never interpreted as free compute.
`lanes run` refuses a roster containing a Windows host before starting any job; it never
silently omits that host and returns a misleading success for partial coverage.
Onboarding selects Python from the tool's declared runtime requirement, not the host's
older system interpreter. Workspace dependencies still install from the shipped frozen lock.
The environment is installed before starting its queue service. Startup detaches its streams
and waits briefly for readiness, so a fresh host need not have a separate global queue install.
Stopping `wait` does not cancel a job or stop rental billing.

Source snapshots are not deleted automatically. A local job cache cannot prove
that another workstation has no job using a remote snapshot. Monitor inode
usage. Remove a snapshot only after checking all dispatchers and queues.

## Many jobs, many machines, one flow

A batch is declared as data and moves through three verbs, and only the last
one runs anything.

```toml
# fleet.toml
name = "fleet"

[defaults]              # every job inherits these
runtime_s = 1800        # what the command is expected to take, which is what an estimate prices

[[jobs]]
name = "sweep-a"        # the target and its position when left out
target = "miyabi-g"
command = "python -m experiments.run --shard 0"
data = ["corpus/shard-0.npz"]   # what this job needs beyond the mirror
walltime = "06:00:00"
mem_gb = 100
fetch = "results/sweep-a"

[[jobs]]
target = "gold"
command = "python -m experiments.run --shard 1"
```

```console
$ mainboard batch prepare fleet.toml --agent      # what must ship, nothing runs
job       target    files  raw_bytes  wire_bytes  since
sweep-a   miyabi-g  1440   9400549    2435370     2026-08-19T02:03:53+00:00
gold-2    gold      19     106701     33053       2026-08-20T15:49:32+00:00
total               1459   9507250    2468423
$ mainboard batch estimate fleet.toml --agent     # what it will cost, nothing runs
job      target  kind  hardware     wire_bytes  runtime_s  setup_p50_s  setup_p90_s  setup_samples  rate_usd_hr  expected_usd  p90_usd
sweep-a  gold    ssh   129 GB RAM   33053       25.0       2.49         7.53         3              0.0          0.0           0.0
$ mainboard batch run fleet.toml --set repetition=3   # every job to its own target, one knob typed
fleet-db4af53f
$ mainboard batch run fleet.toml --only "sweep-*"    # the jobs that are ready, the rest recorded skipped
fleet-db4af53f
$ mainboard batch wait fleet-db4af53f             # block until every job settles, exit its verdict
$ mainboard interact --on miyabi-g --keep --walltime 02:00:00   # hold a GH200 in tmux, reattach with the same line
```

`prepare` measures compressed changes from the host's workspace mirror, plus
declared input data. `estimate` uses recorded setup times. Its sample count
identifies targets with no timing history. `watch` repeats the monitor sweep.
Automatic result collection and rental release require a running monitor.
Provider outages can delay release. A local execution timeout does not stop billing.

Every state change and cost observation is one NDJSON line under the batch's
own directory, and each verb reads its cursor back out of those lines rather
than out of memory. The topics and payloads are written down in one place,
`batch/receipts.py`, so the file transport can become a broker without anything
downstream noticing.

## One file

```toml
[deps]
python = ">=3.14"

[python.deps]
torch = ">=2.9"

[containers.ngc]
image = "nvcr.io/nvidia/pytorch:25.06-py3"   # fixed off-the-shelf image, never rebuilt
                                             # your env lives on a bound host path inside it

[hosts.gold]
kind = "ssh"
root = "/home/pedro/projects"

[hosts.miyabi-g]
kind = "pbs"
container = "ngc"
account = "xg25g007"
modules = { singularity = "4.2.1" }

[hosts.miyabi-g.queues.short-g]
max-walltime = "07:59:59"       # the scheduler's real rejection boundary, enforced
mem-ceiling-gb = 100            # before your job ever leaves the laptop

[hosts.miyabi-g.defaults]
queue = "debug-g"
mem-gb = "min(100, attempt * 50)"   # retries escalate instead of dying twice
```

Profiles inherit `[hosts.defaults]`, values interpolate (`{{ env('LOCALDIR') }}`,
`{{ num_cpus() }}`), and queue policies are data the tool enforces at submit
time with the error you wish the scheduler gave you.

## One repository tree

A workspace that is a git repository with submodules, nested ones included, is
operated as one repository. Only the repositories whose remote owner is the
root's own or one `[git]` names are ever written; pinned reference code from
anybody else is read to verify the pointers that name it and otherwise left alone.

```toml
[git]
owners = ["phvv-me", "ComputerVisionLaboratory"]
ceiling-mb = 50                              # the default; LFS files are exempt
never-commit = ["**/evidence/artifacts/**"]  # the default; git glob pathspecs
```

```console
$ mainboard git status          # branch or detached, ahead/behind, dirty, published
$ mainboard git pull            # fast-forward only, submodules follow their pointers
$ mainboard git commit -m "…"   # submodules first, then the parents' pointers
$ mainboard git push            # children first, pointers verified, protected main → branch
$ mainboard git check           # everything a clone or the next push would trip on
```

## One lint pass

```toml
[lint]
exclude = ["**/datasets/", "**/references/"]   # never read, never rewritten
owners = ["packages/*", "research/*"]           # beside every dir holding pyproject.toml or .git

[lint.tools.ruff-format]
run = "ruff format --force-exclude {files}"
files = ["*.py", "*.pyi"]
writes = true                                  # fix phase, in declaration order

[lint.tools.pyrefly]
run = "pyrefly check"                          # no {files}: checks the whole owner
files = ["*.py", "*.pyi", "pyproject.toml"]
```

`mainboard lint` repairs text (UTF-8, the newline `.gitattributes` names, no
trailing blanks, one final newline), runs the writing tools in order, then every
check at once, each inside the owner of the files it matched and under the
workspace environment's PATH. With no path it reads what differs from HEAD;
`mainboard lint .` reads everything. `mainboard lint install-hook` makes every
commit run `mainboard lint commit` over the staged files, and `mainboard lint
edit` is the Claude Code PostToolUse hook: it repairs the file an agent just
wrote and hands whatever is left back as context.

## What it replaces

- environment managers that cannot name a host
- dispatch scripts that cannot solve an environment
- container workflows that rebuild an image per dependency change
- profilers that stop at one process on one machine
- the prose wiki page about your cluster's queue limits
- a pre-commit config, an editor hook and a CI job that each lint a different way

Under the facade: pixi-powered multi-ecosystem environments (conda plus PyPI
and friends) that provision inside off-the-shelf containers via bind-mounted
prefixes, an ssh/PBS/SLURM/pueue dispatch core with durable job records and
verdict lifecycles, hardware probing (GPUs, cgroup memory caps, scratch,
InfiniBand fabric) as a versioned wire format, and a profiling stack (spans,
CUPTI, Perfetto merge manifests) that lands multiple machines on one queryable
timeline. Experiment studies group many simultaneous jobs under one identity
with content-addressed run ids and declared data needs.

## Experiments and profiling

A pytest experiment is a native job target:

```console
mainboard run path/to/test_experiment.py::test_measurement -- --collect-only
mainboard submit --on gold path/to/test_experiment.py::test_measurement
```

Pytest owns fixtures, parametrization, assertions, and per-case failures. The
`mainboard.jobs.job` decorator declares literal data needs, source resources,
and a fetch path; Mainboard seals the import closure and dispatches that same
target. Paths must resolve inside the declared environment. A decorator does
not install a project's source package or make mutable input data immutable.

The opt-in `mainboard.trials.pytest_plugin` supplies `trial`, `run`, and `stage`
fixtures after the project provides its trials declaration. Trials own
scientific receipts. A refuted hypothesis is distinct from a failed instrument.
The injected `log` fixture adds diagnostics, metrics, artifacts, and profiling to those trials.

The profiling entry point remains independent of trials:

```python
from mainboard import Profiler, span

with Profiler(features=Profiler.Feature.SPANS) as profiler:
    with span("measurement"):
        work()

profile = profiler.result()
profile.show()
```

For one operation's CUDA activity, use a synchronized window.

```python
answer, profile = Profiler.capture(
    work, activities=Profiler.Activity.KERNEL, device_index=0
)
```

`work` runs once. Capture reuses a compatible active `Profiler` or opens one.
Nested windows share the collector without adding duplicate records to the outer
total. Issue CUDA work serially from one host thread in the selected device's
current context. Checkpoints synchronize all streams in that context.
Unavailable activity support or collection failures raise, including lost records.
An empty window is absent evidence, not proof that a kernel ran.

| Need | Existing API | Boundary |
| --- | --- | --- |
| Python regions | `span`, `Profiler(auto=("package.module",))` | Explicit spans or PEP 669 instrumentation, not statistical sampling |
| GPU execution trace | `Feature.ACTIVITY`, `Profile.perfetto(path)` | Native activity collection; explicitly requested activity cannot silently disappear |
| Process device telemetry | `Feature.DEVICE` | Sampled GPU usage, not kernel execution time |
| Callable timing | `benchmark(fn, sync=barrier)` | Synchronized wall time, not CUDA-event time |
| Stage comparison | `profile_stages(cases, trace=True)` | Untraced timing pass followed by a separate trace pass |
| Fleet telemetry | `mainboard sample`, `facts`, `compute` | Job/machine observations, not a replacement for in-process profiling |

Keep profiling separate from uninstrumented throughput measurements, and keep
the requested collection policy beside each saved profile. An invalid device
index is an error, never permission to sample a different card. Profiling study
exceptions propagate; `Row.has_evidence` describes capture, not success or a
scientific verdict. Use pytest parametrization for independently recorded trials.

Select Python regions with `auto` or `span`. Neither is statistical sampling.
The unused `PYTHON` flag was removed. Existing feature bit values are unchanged.
Historical policies containing that unsupported bit require their original source.
Launch work through `mainboard run` or
`mainboard submit`, with the profiling context inside the target. There is no
profiler attach API. Render the returned `Profile`, not the active `Profiler`.

The `mainboard.trials.pytest_plugin` also injects `log`, backed by the existing
trial identity and profiler. Use `log.info("phase {}", phase)`,
`log.bind(model=model).metrics(loss=loss)`, `log.table(rows)`, and
`log.image(path)`. `log.model(value)` attaches a Pydantic model's JSON bytes with
an inferred name. Retain a captured profile with
`log.model(profile, schema_name="mainboard.Profile")`.

`with log.profile():` opens the declared profiling policy and attaches its result,
even when the body raises. Use `Profiler.capture` for nested operation windows
inside an activity-enabled policy. Do not open a second collector.
Profiling does not replace the experiment's timing protocol. Settle with the
project's vocabulary, such as `log.validated("criterion held", error=error)`.

Identity, output paths, and job fetch paths are inferred from the pytest node.
Tables are Parquet. Other artifacts are content-addressed bytes. Inputs are
explicit `Declaration(inputs={alias: Artifact(...)})` references:
`log.read(alias)` verifies size and SHA-256, and `log.read_table(alias)` reads
Parquet. There is no ambient “latest” lookup or automatic producer execution.
The facade has research pilots on RTX 4090 and Miyabi GH200; this does not establish
coverage for every experiment or provider. Projects enforcing Parquet-only storage permit the node-local
`evidence/artifacts/` subtree for event journals and mixed-format payloads.

For analysis, `Dataset(...).tables(project_root, schema_name="study.reading.v1")`
reads every matching table artifact across runs, verifies its hash, and retains
each row's receipt metadata in the `_trial` JSON column. It does not pick the
newest host, discard failed outcomes, or change scientific units or thresholds.
Use an explicit `run=` to restrict the read. Acquisition inputs still use pinned
`log.read_table(...)` references. Artifact paths remain project-relative through
Mainboard's remote result mounts and after fetching.

### One query surface across machines

```console
mainboard monitor --json
mainboard collect research/reproducibility/datasets/experiments/architecture_error_census --on pedro-home
mainboard query "SELECT project, hardware, count(*) AS runs FROM runs GROUP BY ALL"
mainboard query --project reproducibility "SELECT * FROM metrics ORDER BY recorded_at DESC LIMIT 20"
mainboard query "SELECT server, handle, backend_state, verdict, evidence, settled FROM jobs"
mainboard query "SELECT * FROM runs" --out /tmp/runs.parquet
mainboard query --file queries/inventory.sql --out /tmp/inventory.parquet
mainboard help artifacts
mainboard help batch run
```

`monitor` pulls published results from running jobs as well as finished ones. Run
repeated passes to refresh remote data. A query itself has no network side effects.
DuckDB reads the collected Parquet fragments and event journals directly. There is
no shared database file for different servers to lock, and no database service to deploy.
Each query sees a fresh inventory. It is not a transaction across all servers.

`collect` also imports runs started directly on a node, using the same collection
path as monitor. Remote filesystem operations use Python's standard library and
native `Path`; OpenSSH carries the bytes without remote rsync, tar, Bash, or an
installed Mainboard. The host profile supplies `root` and `python` (default
`python3`). The latter is a trusted interpreter command in the SSH login shell;
quote paths containing spaces as that shell requires. Python 3.9 or later is
needed for collection, independently of the experiment environment.

Artifact references use canonical forward-slash relative paths on every OS.
Queries read collected local bytes, while the original repository path remains
provenance. Older receipt schemas retain missing fields as null. Importing a
copy from another node never changes its recorded acquisition hardware.

Jobs keep the last `backend_state` separate from the command `verdict`. A rental
can report `running` before its successful command is collected and the instance
released. `settled` means the monitor recorded completion; it is not a fresh
provider-liveness check, and it does not turn unverified evidence into verified data.

`--out` exports the selected rows as CSV, Parquet, or JSON according to the filename
extension. It creates parent directories, publishes only a complete file, and refuses
to overwrite an existing destination. Parquet preserves column types; CSV and JSON use
their standard representations. `--json` still prints JSON when no file is requested.
An exact command path passed to `help` opens its documentation. Other words search
command descriptions and API docstrings. Use `help -- --max-usd` to search an option.

Keep reusable SELECT statements in UTF-8 `.sql` files. `--file` replaces the SQL
argument, while Python callers pass a `Path` to `Results.query`. The SQL file and
any paths inside its query resolve from the caller's current directory.

Plot the same SELECT with optional Seaborn and Matplotlib:

```console
uv tool install --from './packages/mainboard[wandb,plot]' mainboard --force
mainboard plot "SELECT hardware, count(*) AS runs FROM runs GROUP BY hardware" --project reproducibility --x hardware --y runs --kind bar --out /tmp/run-inventory.png --out /tmp/run-inventory.pdf --dpi 300
```

This example charts the collected run inventory, not comparative GPU performance.
For measurements, filter the experiment, input regime, and hardware explicitly in SQL.
`scatter` shows individual rows; `line` retains SQL order without estimating a mean or
adding error bars. `bar` requires one row per x/hue group, so aggregate in SQL first.
`--hue` identifies a categorical series column in SQL appearance order. Its legend
sits outside the data axes. `--title` labels the scope. Native plotting settings
live in the workspace manifest and project style files, with no palette package.
Output extensions select Matplotlib formats and DPI controls raster resolution.
Null/nonfinite plotted values and existing output files are refused, not silently dropped
or overwritten. Tables with bespoke plots can use Seaborn directly on
`Results(root).query(sql).to_dict(as_series=False)` or verified `Results(root).table(...)` data.

Define shared styles in `mainboard.toml`. `paper` is the default when declared;
select another named style with `--style`:

```toml
[plots.paper]
palette = ["#745399", "#b7282e", "#1e50a2", "#f8b500"] # or a Seaborn palette name
theme = "default"       # a native Matplotlib style name
figsize = [3.25, 2.1]    # inches; omitted uses the native theme's size
dpi = 300

[plots.paper.rc]
"axes.labelsize" = 8
"font.family" = "sans-serif"
```

Palettes use explicit color lists or Seaborn's names, such as `deep`.
The renderer takes colors in palette order and refuses excess categories.
Group the tail into Other or use separate panels. `rc` contains native Matplotlib
settings, not a second styling language. `--dpi` overrides the style's resolution.
Changing styles does not rebuild an environment.

For a reusable composition, keep its style and figure together in `plots.toml`.
This file overlays the workspace styles without selecting an environment or changing
path resolution. Settings and semantic maps merge by key; lists and scalars replace.
Unspecified project fields retain the shared values.

```toml
[workspace]
name = "project-figures"

[plots.paper]
figsize = [3.25, 2.1]

[figures.inventory]
style = "paper"
out = ["inventory.pdf", "inventory.png"]

[figures.inventory.panels.counts]
file = "queries/inventory.sql"
variables = {x = "hardware", y = "runs"}
layers = [{mark = "Bar"}]
axis = {xlabel = "Hardware", ylabel = "Recorded runs"}
```

The corresponding `queries/inventory.sql` contains
`SELECT hardware, count(*) AS runs FROM runs GROUP BY hardware ORDER BY hardware`.

```console
mainboard plot --config plots.toml --figure inventory
mainboard plot --config plots.toml --figure inventory --out /tmp/inventory.svg
```

Layers use native Seaborn marks and moves. SQL supplies aggregates and interval
bounds; rendering never estimates them. `colors`, `labels`, `markers`, and
`linestyles` in a style provide stable semantic mappings. Shared legend order follows
the explicit `colors` order even when panels contain different subsets.
Native `axis`, `ticks`, `grid`, `legend`, and `rc` settings control presentation.

Declared job outputs are download-only during source mirroring. This includes
current and historically recorded output paths in the local workspace cache,
regardless of host alias. Source files outside those paths still get pruned.
An explicit resource overlapping an output root is refused before upload; materialize
the selected data under a separate pinned input path instead. Ordinary `needs`
remain mutable mirror links. These safeguards do not provide distributed
coordination across separate local caches or simultaneous first submissions
through different aliases to the same endpoint.

GPU leases named `.card.lock` or `.card.lock.*` remain local to their host.
Mirroring neither uploads nor deletes them, even under broader include rules.
Declaring a lease as an explicit source or resource fails before transfer;
missing required source files still fail instead of accepting a partial shipment.

The views are `jobs`, `runs`, `trials`, `events`, `metrics`, and `artifacts`.
Project, run, host, hardware, and source remain explicit; combining storage never
means combining scientific conclusions. `Results(root).table(schema, project=...)`
reads matching published tables with provenance in `_trial`, even before the trial
settles. Missing artifact bytes remain an error, not a silently complete table.
Acquisition dependencies still use explicit pinned `log.read_table(...)` inputs.

Research `log` trials preserve one manifest per node/run from Mainboard's existing
transferred-file listing, including the adjacent captured `node.md`, environment,
inputs, and hardware. No repository, Git executable, clean worktree, commit, or push
is required. Newly written and modified files are ordinary source. Mainboard hashes
their actual bytes with SHA-256, preserves a content-addressed source ZIP under
`.mainboard/source-archives/`, and verifies the listing and bytes before acquisition.
An edit after capture requires a new bundle, not a commit. Optional `.gitignore`
files control discovery without invoking Git; secrets and output protections remain.
Historical Git metadata and inadmissible receipts are retained as historical data,
not relabeled by this policy change. The pilot
experiments no longer maintain a second source-file list or a hash of another seal.
Artifact checksums remain useful for verifying transferred bytes. Historical
registrations and receipts remain valid records of their original instruments.

Receipt parts stay immutable after settlement. Fetches merge run-specific paths
without deleting another server's results and exclude temporary files and mutable
`latest.jsonl` summaries. Use the query views for the combined current picture.
Source sync excludes trial artifact and receipt directories; explicitly sealed
input resources are still transferred. Result downloads stage and validate the
transfer before publishing immutable files without overwriting existing paths.
Conflicting copies fail explicitly, regardless of their timestamps. Live event
prefixes become immutable offset-named snapshots; queries deduplicate overlaps.
Temporary files, derived event heartbeats, and incomplete trailing records are
excluded. Repeating a transfer adds no files when the source is unchanged.
Final receipts supply artifact references when an event
stream is incomplete, without inventing missing events or their timestamps.

Collection verifies transport integrity, not the scientific meaning of results.
Monitor still checks receipt references before declaring evidence delivered;
`Results.table` verifies referenced artifact bytes when reading them. Collect from
the declared storage workspace, not a pinned source tree with external data mounts.
Linked or unreadable evidence is refused. Files publish individually, so a failed
publication may leave a valid subset for the next retry. Current transfers send
the selected scope again and growing snapshots can overlap on disk; incremental
transfer and snapshot compaction remain optimization work.

Collection from native Windows, Linux, and macOS nodes to a Linux client has been exercised,
as has native Windows local execution. Source setup,
snapshot pinning, and scheduler launch still contain Unix-shell paths; collection
support does not establish fully portable remote submission. The next boundary is
one Python executor for the existing job specification, retaining Pueue/PBS/Slurm
as scheduler adapters and Pixi as environment activation rather than adding a
second queue or database service.

September 7 real-use validation covered local RTX 4090, Crimson's reserved RTX
3090 through SSH/pueue, and Miyabi GH200 through PBS. On Crimson, live result
collection and source sync overlapped a registered cache acquisition: its event
inode survived and all byte offsets remained continuous through completion.
This does not establish provider/rental behavior. Keep profiles attached to their
Log provenance: the standalone profile host field remains unpopulated.

## Status

The real-use checks above cover specific hardware and dispatch paths, not every
provider or container configuration. The provider router, `board.on("auto")`,
scoring hosts by fit,
price, and time to result across private clusters and commercial GPU clouds,
is under active development.

`[engines.*]` and `serve` currently stage a declared command in a container on an
owned host. They reuse `run`'s container command construction. Provider-hosted
serving and automatic host selection remain under development.

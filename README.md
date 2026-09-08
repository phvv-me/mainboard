# mainboard

Run batch jobs anywhere without caring about system setup.

One file, `mainboard.toml`, declares your dependencies, your environments,
your container base images, and every machine you run on, from your laptop
to a PBS supercomputer to a cloud GPU provider. One interface runs, submits,
tracks, probes, and profiles across all of them.

```python
from mainboard import Board

board = Board()  # finds mainboard.toml like git finds a repo
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
$ mainboard run --on gold -- nvidia-smi -L
GPU 0: NVIDIA GB10 (UUID: GPU-6a5c...)
$ mainboard facts --on gold | head -4
{
  "schema_version": 1,
  "hostname": "gold",
  "cpu_name": "10x Arm Cortex-A725 + 10x Arm Cortex-X925",
$ mainboard submit --on miyabi-g --attempt 2 -- python -m experiments.run
2231259
$ mainboard monitor --json          # one durable pass, what a cron runs
{"running": 1, "finished": [], "failed": [], "unreachable_hosts": [], "changed": false}
$ mainboard jobs                    # every live job as its own queue sees it right now
$ mainboard compute --agent         # every path this workspace can run on
name      kind      access       detail                 usd_hr  credit_usd
local     local     here         1x RTX 4090, 135 GB RAM
gold      ssh       ready        1x GB10, 129 GB RAM
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


`jobs` shows every dispatched job still in flight before it shows any that
settled, each with what its own scheduler says about it right now, and asks each
host once for all of them: one `qstat`, one `squeue`, one `pueue status`. A
listing that had to leave anything out says so rather than stopping quietly at a
limit, and `--limit` bounds only the settled tail.

`monitor` collects outstanding results into durable job records and study ledgers.
Its reports identify changes and unreachable hosts without repeating unchanged
outcomes on later passes.

`run` executes native file targets locally; use `submit` to ship and allocate a
remote job. Collection and help stay local. Plain commands such as `nvidia-smi`
can run remotely over SSH, which reaches a cluster's login endpoint, not a batch
allocation. Stopping `wait` does not cancel a job or stop rental billing.

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

## What it replaces

- environment managers that cannot name a host
- dispatch scripts that cannot solve an environment
- container workflows that rebuild an image per dependency change
- profilers that stop at one process on one machine
- the prose wiki page about your cluster's queue limits

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
mainboard query "SELECT project, hardware, count(*) AS runs FROM runs GROUP BY ALL"
mainboard query --project reproducibility "SELECT * FROM metrics ORDER BY recorded_at DESC LIMIT 20"
mainboard query "SELECT server, handle, backend_state, verdict, evidence, settled FROM jobs"
mainboard query "SELECT * FROM runs" --out /tmp/runs.parquet
mainboard help artifacts
mainboard help batch run
```

`monitor` pulls published results from running jobs as well as finished ones. Run
repeated passes to refresh remote data. A query itself has no network side effects.
DuckDB reads the collected Parquet fragments and event journals directly. There is
no shared database file for different servers to lock, and no database service to deploy.
Each query sees a fresh inventory. It is not a transaction across all servers.

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

Plot the same SELECT with the optional Seaborn/paleta integration:

```console
uv tool install --from './packages/mainboard[wandb,plot]' --with ./packages/paleta mainboard --force
mainboard plot "SELECT hardware, count(*) AS runs FROM runs GROUP BY hardware" --project reproducibility --x hardware --y runs --kind bar --out /tmp/run-inventory.png --out /tmp/run-inventory.pdf --dpi 300
```

This example charts the collected run inventory, not comparative GPU performance.
For measurements, filter the experiment, input regime, and hardware explicitly in SQL.
`scatter` shows individual rows; `line` retains SQL order without estimating a mean or
adding error bars. `bar` requires one row per x/hue group, so aggregate in SQL first.
`--hue` identifies a categorical series column in SQL appearance order. Its legend
sits outside the data axes. `--title` labels the scope. Paleta supplies native
palette/theme names without copying their definitions.
Output extensions select Matplotlib formats and DPI controls raster resolution.
Null/nonfinite plotted values and existing output files are refused, not silently dropped
or overwritten. Tables with bespoke plots can use Seaborn directly on
`Results(root).query(sql).to_dict(as_series=False)` or verified `Results(root).table(...)` data.

Name additional styles in `mainboard.toml` and select one with `--style paper`:

```toml
[plots.paper]
palette = "paleta-shiho" # also paleta-shiho-dark, paleta-meta, paleta-meta-dark
theme = "paleta-shiho"   # a native Matplotlib style name
figsize = [3.25, 2.1]    # inches; omitted keeps paleta's text-column size
dpi = 300

[plots.paper.rc]
"axes.labelsize" = 8
"font.family" = "sans-serif"
```

Palettes use Seaborn's existing names, including built-ins such as `deep`.
The renderer takes colors in palette order and refuses excess categories.
Group the tail into Other or use separate panels. `rc` contains native Matplotlib
settings, not a second styling language. `--dpi` overrides the style's resolution.
Changing styles does not rebuild an environment.

The views are `jobs`, `runs`, `trials`, `events`, `metrics`, and `artifacts`.
Project, run, host, hardware, and source remain explicit; combining storage never
means combining scientific conclusions. `Results(root).table(schema, project=...)`
reads matching published tables with provenance in `_trial`, even before the trial
settles. Missing artifact bytes remain an error, not a silently complete table.
Acquisition dependencies still use explicit pinned `log.read_table(...)` inputs.

Research `log` trials preserve one manifest per node/run from Mainboard's existing
transferred-file listing, including the adjacent committed `node.md`, environment,
inputs, and hardware. Commit the registration and code before running. The pilot
experiments no longer maintain a second source-file list or a hash of another seal.
Artifact checksums remain useful for verifying transferred bytes. Historical
registrations and receipts remain valid records of their original instruments.

Receipt parts stay immutable after settlement. Fetches merge run-specific paths
without deleting another server's results and exclude temporary files and mutable
`latest.jsonl` summaries. Use the query views for the combined current picture.
Source sync excludes trial artifact and receipt directories; explicitly sealed
input resources are still transferred. Result downloads use rsync's update rule so
an older replica on another host cannot replace a newer local file. This relies
on file timestamps, not distributed conflict resolution; divergent writes to one
run path are unsupported. Final receipts supply artifact references when an event
stream is incomplete, without inventing missing events or their timestamps.

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

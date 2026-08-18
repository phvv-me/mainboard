# mainboard

Run batch jobs anywhere without caring about system setup.

One file, `mainboard.toml`, declares your dependencies, your environments,
your container base images, and every machine you run on, from your laptop
to a PBS supercomputer to a cloud GPU provider. One interface runs, submits,
tracks, probes, and profiles across all of them.

```python
from mainboard import Board

board = Board()                            # finds mainboard.toml like git finds a repo
board.run("python train.py")               # here, in the activated environment
board.on("gold").run("nvidia-smi")         # any ssh box, same call
job = board.on("miyabi-g").submit(         # a PBS cluster, inside an NGC container,
    "python -m experiments.run",           # with queue policy checked before any ssh
    walltime="06:00:00",
)
job.wait(); print(job.logs()); job.pull()
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
```

`monitor` is the sweep that makes a dispatched job's outcome survive the
process that dispatched it. Each pass probes every job still owed an outcome,
pulls back the results of the ones that just finished, records their verdicts
in the study ledgers that own them, and reports only what changed, so a
schedule of passes never announces the same job twice and a host that is down
is one line in the report rather than a failed sweep.

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

## Status

0.1.0. Validated live on x86 and Grace hosts over ssh and pueue; PBS and
container paths covered by the test suite (1332 tests, 100 percent branch
coverage). The provider router, `board.on("auto")`, scoring hosts by fit,
price, and time to result across private clusters and commercial GPU clouds,
is under active development.

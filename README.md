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
$ mainboard compute --agent         # every path this workspace can run on
name      kind      access       detail                 usd_hr  credit_usd
local     local     here         1x RTX 4090, 135 GB RAM
gold      ssh       ready        1x GB10, 129 GB RAM
miyabi-g  pbs       unreachable  ssh connect timed out
vast      provider  keyed        1x RTX 4090 Sweden, SE  0.2978  99.9968
```

`compute` answers what there is to run on before anything is dispatched: this
machine, every declared host with whether it answers and whether it was set up,
and every provider with whether its credentials are here and what the account
has left. No credential is ever printed, only whether one was found.


`monitor` is the sweep that makes a dispatched job's outcome survive the
process that dispatched it. Each pass probes every job still owed an outcome,
pulls back the results of the ones that just finished, records their verdicts
in the study ledgers that own them, and reports only what changed, so a
schedule of passes never announces the same job twice and a host that is down
is one line in the report rather than a failed sweep.

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
$ mainboard batch run fleet.toml                  # every job to its own target
fleet-db4af53f
$ mainboard batch watch fleet-db4af53f --interval 5
```

`prepare` measures the delta rather than the tree: a host already carries the
workspace, so what a job actually sends is what changed since that mirror plus
the data the job names, compressed the way the wire will carry it. `estimate`
prices each row against setup times fitted from this workspace's own recorded
dispatches, and a target nobody has timed says so in its sample count instead
of inventing a number. `watch` drives the same durable sweep a cron runs, so
results come back and provider rentals are cancelled whether or not anyone is
watching.

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

## Status

0.1.0. Validated live on x86 and Grace hosts over ssh and pueue; PBS and
container paths covered by the test suite (1332 tests, 100 percent branch
coverage). The provider router, `board.on("auto")`, scoring hosts by fit,
price, and time to result across private clusters and commercial GPU clouds,
is under active development.

`[engines.*]` and `serve` are a skeleton: a declared command staged through a
declared container, rendered through the same containerize seam `run` already
builds argv with, on an owned host. Staging `serve` onto a rented provider
instance, alongside the same fit/price/time-to-result scoring `board.on("auto")`
is bringing to dispatch, is 0.5 work.

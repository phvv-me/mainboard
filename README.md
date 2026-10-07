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
$ mainboard host list gold --facts | head -4

# facts: gold
hostname	gold
cpu_name	10x Arm Cortex-A725 + 10x Arm Cortex-X925
$ mainboard job submit --on miyabi-g --attempt 2 python -m experiments.run
2231259
$ mainboard job list                    # settles what ended, then every live job as its queue sees it
$ mainboard job show 2231259 --wait     # cells and a heartbeat on stderr; exit 4 on a stalled job
$ mainboard host list                   # every path this workspace can run on
name      kind      access       detail                 usd_hr  credit_usd
local     local     here         1x RTX 4090, 135 GB RAM
gold      ssh       provisioned  cached default: hardware from onboarding
miyabi-g  pbs       unreachable  ssh connect timed out
vast      provider  keyed        1x RTX 4090 Sweden, SE  0.2978  99.9968
```

`mainboard help job submit` opens that command's help. Other queries, such as
`mainboard help Log.read_table` or `mainboard help separate trace pass`, search
command descriptions, this README, and Python API docstrings. Results name their
source locations. The wheel includes the README. API search never imports
the scanned modules. Broad searches show twenty hits and the full match count.

`host list --facts` pairs the hardware with a software census (operating system and version,
shells, filesystem case sensitivity, symlink and long-path support, the git
settings a clone inherits, git, git-lfs, gh, ssh, uv, pixi, tectonic, node,
cargo and nvcc versions, the NVIDIA driver with its CUDA, each
card's compute capability and memory) and judges it against the workspace: a
platform the manifest or its lock does not cover, a driver below `[system] cuda`,
locked CUDA builds the driver cannot run or that carry no kernels for the card,
fewer cards or less memory than the profile's `defaults.gpus` and
`defaults.vram-gb` declare, too little disk. `host setup`, `host list` (its `issues`
column) and `doctor --center` print the same findings from the same census.

`host list` answers what there is to run on before anything is dispatched: this
machine, every declared host with whether it answers and whether it was set up,
and every provider with whether its credentials are here and what the account
has left. No credential is ever printed, only whether one was found.
`provisioned` means a cached setup record exists, not that a job can currently run.
`observed_at` timestamps the live survey; `cached_at` timestamps the retained host
facts, which may be stale. An SSH echo probe works without an installed environment.
Neither cached hardware nor
a successful login proves GPU availability. For PBS/Slurm, the reachable endpoint
is the login host, not an allocated compute node. Inspect `mainboard job list` and
`mainboard host list <host> --facts` before choosing a target.
A host's `[hosts.<name>.vars]` may contain `status-note` to describe a supported
route or known restriction. This replaces generic setup advice, not the observed
access state, and never makes a host job-ready.


`job list` shows every dispatched job still in flight before it shows any that
settled, each with what its own scheduler says about it right now, and asks each
host once for all of them: one `qstat`, one `squeue`, one `pueue status`. A
listing that had to leave anything out says so rather than stopping quietly at a
limit, and `--limit` bounds only the settled tail.

It settles first: outstanding results are collected into durable job records, and a
pass reports changes without repeating unchanged outcomes on
later passes. A host that does not answer is knocked on once per command; its runs
keep what was last recorded, a note names it with what to do about them, and a
wait on another host's job asks it again only every half minute. `job cancel`
settles a run on such a host without stopping it there, and says so.

`run` executes native file targets locally. Use `job submit` for remote jobs.
Collection and help stay local. Plain diagnostic commands use SSH.
On a cluster, SSH reaches the login endpoint. It provides no batch allocation.
HPC-AI catalog prices absent from the provider response are unknown (`null`), never
interpreted as free compute.
Onboarding selects Python from the tool's declared runtime requirement, not the host's
older system interpreter. Workspace dependencies still install from the shipped frozen lock.
The environment is installed before starting its queue service. Startup detaches its streams
and waits briefly for readiness, so a fresh host need not have a separate global queue install.
Stopping `job show --wait` does not cancel a job or stop rental billing.
`host setup` installs the mirror's own environment, the one `run --on` and every
dispatch's preflight enter; the pinned copy a job activates is built by the first
dispatch of each lock (`built <env> on <host>`), from the packages setup fetched.

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
$ mainboard job submit --batch fleet.toml --estimate   # what it ships and costs, nothing runs
job      target  kind  hardware     wire_bytes  runtime_s  setup_p50_s  setup_p90_s  setup_samples  rate_usd_hr  expected_usd  p90_usd
sweep-a  gold    ssh   129 GB RAM   33053       25.0       2.49         7.53         3              0.0          0.0           0.0
$ mainboard job submit --batch fleet.toml --set repetition=3   # every job to its own target
# batch fleet-db4af53f
$ mainboard job submit --batch fleet.toml --only "sweep-*"     # the ready jobs; the rest skipped
$ mainboard job list --batch fleet-db4af53f                    # one batch's jobs, as they settle
$ mainboard job show fleet-db4af53f --wait                     # block until all settle, exit the verdict
$ mainboard shell --on miyabi-g --keep --walltime 02:00:00   # hold a GH200 in tmux, reattach with the same line
```

`job submit --batch <spec> --estimate` measures compressed changes from the host's
workspace mirror, plus declared input data, and prices them with recorded setup
times. Its sample count identifies targets with no timing history. `job list
--batch <id> --watch <seconds>` repeats the sweep. Automatic result collection and
rental release require a sweep to run: a `job list`, a `job show --wait`, or the
periodic pass `job list --every 20m` installs.
Provider outages can delay release. A local execution timeout does not stop billing.

Every state change and cost observation is one NDJSON line under the batch's
own directory, and each verb reads its cursor back out of those lines rather
than out of memory. The topics and payloads are written down in one place,
`batch/receipts.py`, so the file transport can become a broker without anything
downstream noticing.

## Resuming a run that failed

Every job gets `MB_CHECKPOINT`, a directory on its host that outlives the run (one per run
name, beside the mirror), and `MB_ATTEMPT`. Save there, and a failed or cancelled run continues:

```console
$ mb job submit --on gold --name sweep-7 -- python train.py   # saves to $MB_CHECKPOINT
$ mb job submit --resume sweep-7          # same command, name and host; attempt 2; today's code
$ mb query "SELECT handle, verdict FROM lake.runs WHERE name = 'sweep-7'"
```

## Shipping an environment instead of installing it

A lean environment per experiment (`[envs.<name>]` with `no-default = true`) is what makes a
fresh machine fast: 5.7 GB for Python, CUDA torch and cutok against `default`'s 18 GB, installed
cold from the lock in under a minute on a well-connected host. Where a machine should not install
at all (offline, billed by the minute, booted from an image), build the environment into files
on a Linux host, named by the environment's digest:

```console
$ mb pack cutok --on crimson                  # a self-extracting executable (pixi-pack)
$ mb pack cutok --on crimson --image          # plus an OCI image: slim Debian + the environment
$ mb pack cutok --on pedro-cvlab --sif        # plus an Apptainer file for HPC
$ mb pack cutok --on crimson --push ghcr.io/<owner>/mb-cutok   # and to a registry
```

## Holding a rented machine

A rental per job rebuilds the environment every time. A held machine is rented,
named, set up once, and then takes jobs in seconds until its deadline:

```console
$ mainboard host hold vast --gpu-name "RTX 5090" --for 3h --max-usd 4   # rent, alias, onboard, park
$ mainboard job submit --on vast-rtx-5090 --walltime 00:20:00 -- path/to/test_x.py::test_y
$ mainboard host list                                                  # every live rental, held or not
$ mainboard host release vast-rtx-5090                                    # stop billing now
```

The alias is a marked block at the top of `~/.ssh/config`, the host profile is the
provider's own (its sync scope and variables) as an ssh host, and the deadline is
the rental's lease in the run registry, so `job list` releases it on time and
`host list` releases anything past due before it lists. `--max-usd` caps the whole
hold, landing included.

## Papers

A manuscript is declared once and checked on every build:

```toml
[papers.head]
dir = "research/llm-head/papers/iclr-2027-llm-head/latex"
main = "paper.tex"      # the default
limit = 9               # the last page the main text may reach
ends = "Conclusion"     # the section that must end by that page
```

```console
$ mainboard paper build head                        # build, report, exit 1 on any problem
$ mainboard paper build head --show "Pareto front"  # and render that phrase's page to PNG
```

The build is tectonic from the workspace environment. The report names every
error, undefined reference and citation, multiply defined label and overfull box
by `file:line`, the page count, the page each section starts on, and the last
page SyncTeX places any line of `ends` on, unnumbered statements excluded.

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
kind = "ssh"                    # root defaults to ~/.mb-jobs

[hosts.miyabi-g]
kind = "pbs"
root = "/work/xg25g007/x10537/projects"   # a cluster's shared storage, not its home
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

The solve is source: `lock` (and `add`, `remove`, `upgrade`) writes
each environment's pixi lock byte for byte into `mb.lock` beside the manifest,
one sorted section per environment with the digest it was solved from and the
pixi that solved it. Commit it. `install` copies a section into the ignored
state directory, where pixi reads it, and refuses a section the manifest has
moved past. A workspace still on `mainboard.toml` gets `mb.lock` too, since
the file travels with the tree (one already holding `mainboard.lock` keeps
it). A workspace solved before `mb.lock` existed has its cached lock
adopted into it on the next `install` (one line says so), and `doctor` names a
cached lock that `mb.lock` lacks or disagrees with, and the command that
repairs it. `setup` and dispatch ship `mb.lock` to a host beside the cached
copy an older release installs from.

A target never holds a human checkout: Mainboard keeps everything there in one
folder, `~/.mb-jobs` unless the profile names another `root` (a host set up under
the older name keeps its `~/.mainboard-jobs`, so nothing is built twice). `host setup` reads
the home in the host's own shell (`$HOME`) and places
a leading `~` under it, so every consumer uses one absolute path; a host never set
up is refused with the command that fixes it, and a rental's home is read on landing.

## Members

A monorepo is made of projects that each stand on their own. A member is one:
a `pyproject.toml` anybody can `pip install`, naming its dependencies the
normal way (a version or a git URL), and optionally its own `mainboard.toml`
for what only it needs (tasks, papers, lint, system packages). Cloned alone,
that manifest is the workspace. Inside the monorepo, the root composes it:

```toml
[workspace]
name = "life"
members = ["packages/*", "research/*", "!packages/retired"]   # dirs holding a manifest or pyproject
```

Every member joins the one default environment. Its requirements, `[env]`,
named environments, tasks, papers and lint come along, the root layered on top.
Each member with a `pyproject.toml` is installed editable from its directory and
pinned there by a dependency override, so a sibling requiring it by version or
git URL resolves to the local source, the way cargo's `[patch]` does, and no
hand-written path dependency or override is left. Its tasks and papers answer to
`<member>:<name>`, and bare when nobody else takes the name; paths in its
manifest are its own and are rebased on the way in. How to solve and where to
run stay the root's: a member's `[workspace]`, `[system]`, solve settings and
hosts apply only when it stands alone. From inside a member, the workspace
composing it is the root, the way cargo finds one.

```console
$ mainboard doctor --members              # every member, as somebody who clones it alone sees it
$ mainboard doctor --members llm-head     # one: imports, paths, tasks, then a clone installed by uv
```

## One repository tree

A workspace that is a git repository with submodules, nested ones included, is
operated as one repository. Only the repositories whose remote owner is the
root's own or one `[git]` names are ever written; pinned reference code from
anybody else is read to verify the pointers that name it and otherwise left alone.

```toml
[git]
owners = ["phvv-me", "ComputerVisionLaboratory"]
ceiling-mb = 50                              # the default; LFS files are exempt
never-commit = ["**/evidence/**", "**/datasets/**"]  # the default; data lives in the lake
```

Experiment data never enters a commit: `mainboard lake ingest` keeps it, and
`git check` warns while any is still tracked and fails when git knows no author.

```console
$ mainboard git status          # branch or detached, ahead/behind, dirty, published
$ mainboard git pull            # fast-forward only, submodules follow their pointers
$ mainboard git commit -m "…" PATH...  # those paths alone, then the pointers that moved
$ mainboard git commit -m "…"   # what each repository has staged
$ mainboard git commit -m "…" --all  # every change in every owned repository
$ mainboard git push            # children first, pointers verified, protected main → branch
$ mainboard git check           # everything a clone or the next push would trip on
```

## One lint pass

```toml
[lint]
exclude = ["**/datasets/", "**/references/"]   # never read, never rewritten
owners = ["packages/*", "research/*"]           # beside every dir holding pyproject.toml or .git

[lint.tools.ruff-format]
check = "ruff format --check --force-exclude {files}"   # read-only, what --check runs
fix = "ruff format --force-exclude {files}"             # fix phase, in declaration order
files = ["*.py", "*.pyi"]

[lint.tools.pyrefly]
check = "pyrefly check"                        # no {files}: checks the whole owner
files = ["*.py", "*.pyi", "pyproject.toml"]
```

```console
$ mainboard lint                        # what differs from HEAD or is new, submodules too
$ mainboard lint .                      # everything git tracks or would, beneath here
$ mainboard lint --check --json         # CI: write nothing, report it all, fail on any of it
$ mainboard lint src --only ruff-format # one step; `text` names the built-in hygiene
```

`mainboard lint` repairs text (UTF-8, the newline `.gitattributes` names, no
trailing blanks, one final newline), runs each writing tool's `fix` in order,
then every `check` at once, each inside the owner of the files it matched and
under the workspace environment's PATH. `--check` writes nothing: the hygiene
names what it would repair and every tool runs its read-only `check`, so a tree
that passes it is one the writing pass would leave alone. A person, an agent, a
hook and CI all call the same command and read the same exit code. A
`mainboard.toml` holding only `[lint]` is enough to use it in any git
repository.

## Every coding agent alike

Claude Code, Codex and opencode each read their own files in their own format. A workspace writes what they share once, in `.agents`:

```text
AGENTS.md                 the instructions, read by all three (Claude through CLAUDE.md's @AGENTS.md)
.agents/mcp.json          the MCP servers, in the `mcpServers` shape; values reference `${NAME}`
.agents/settings.json     Claude Code's settings, whose `hooks` the others get too
.agents/agents/*.md       the subagents, Markdown under front matter
.agents/skills/           the skills, which Codex and opencode read in place
.agents/codex.toml        settings only Codex reads (opencode.json likewise)
```

```console
$ mainboard agents sync      # render .mcp.json, .codex/, opencode.json and the .claude link
$ mainboard agents check     # drift, servers this machine cannot start, releases, logins
$ mainboard agents update    # install what is missing, bring every harness to its latest release
```

The rendered files are tracked, so a fresh clone works without this tool; edit
`.agents` and sync, never them. Each harness gets the servers in its own
spelling: Codex forwards a variable by name (`env_vars`, `env_http_headers`,
`bearer_token_env_var`) and reads no reference, and opencode writes `{env:NAME}`.
A server's `overrides.<harness>` table is laid over that one harness's entry for
what only it can say. Codex runs the hooks under Claude Code's event names;
opencode runs none, since its lifecycle extensions are JavaScript plugins.

## The center

One machine holds the monorepo, runs this tool and runs the AI agents: the
center. Every other machine is a target that receives only what a job needs. The
verbs that manage the monorepo itself are `git`, `paper`, `doctor --center`,
`doctor --members` and `host setup --center`.

```console
$ mainboard doctor --center                      # is this machine ready to be the center
$ mainboard host setup --center macmini --root ~/Developer/projects    # move it there
```

`doctor --center` is one report, each row with the command that repairs it: this
machine's git tooling (safe git settings applied in place), the machine judged
against the workspace, the `doctor` report, the plan `host list --plan` resolves, a smoke run
of Python, torch and CUDA in the default environment, whether every lint tool can
start, the repository tree, every agent's configuration (`agents check` below), and
the tracked scripts that use a platform-divergent command (`sed -i`,
`find -printf`, `grep -P`, `timeout`, `xargs -r`, `readlink -f`, `stat -c/-f`,
`date -d`, `jq`, `flock`...), each named with its portable replacement.

It also puts the default environment's executable directories on the PATH every
agent shell starts from, and proves it from each shell kind (zsh, bash, sh).
`~/.config/mainboard/path.sh` puts them on PATH and one marked line sources it
from `~/.zshenv` (every zsh, login or not) and `~/.profile` and `~/.bashrc` (bash
and sh). A dotfiles manager should adopt that line.

`host setup --center <alias>` moves the center to any macOS or Linux machine ssh
reaches. It refuses to start while an owned HEAD is on no remote,
puts uv there when missing, runs the same census `host list --facts` uses and stops early on
a platform the workspace or its lock cannot serve, then:

1. signs `gh` in with this machine's login and makes it git's https credential;
2. carries the ssh config blocks the declared and held hosts need (with their
   jump hosts), the keys they name and `known_hosts`, dropping the options the
   destination's client refuses (`UseKeychain` off macOS);
3. clones the monorepo at this HEAD and every owned submodule at its recorded
   pointer (foreign reference submodules are left to fetch on demand);
4. carries what git does not hold: the `.env`, `.mainboard/` (the dispatch
   registry as a consistent SQLite snapshot, ledgers, batches, audits, recovery,
   source archives, holds), every environment's compiled lock and `mb.lock`
   as it stands, committed or not; Claude Code's
   memory re-keyed to the new workspace path, its settings, agents, skills,
   commands and plugin lists, and this workspace's trust and MCP settings in
   `~/.claude.json`; Codex's config (trusted projects re-keyed), auth, MCP OAuth
   store, rules, skills, prompts and memories; opencode's config, auth and MCP
   OAuth store;
5. installs this tool from the cloned source with the extras installed here, the
   fleet's pixi, and the default environment from the carried lock, never solving;
6. runs `doctor --center` on the destination and folds its rows into the report.

Environment prefixes, the hub pin cache and ignored data stay behind by design.
Secrets travel only on ssh's stdin, never on a command line, and are never
printed. Every step converges on what is already there, so re-running continues
after an interruption and re-verifies after success; a destination file that
differs is kept once as `<name>.migrate-backup`.

## The state lake

Every verb reads and writes one DuckLake per workspace, its catalog
`.mb/lake.sqlite` beside a `.mb/lake/` Parquet folder: runs and hosts, batch
events, captured transcripts and receipts, costs and offers, holds, studies, the
pulse memory, digests, source blobs and staged job scripts. `mb query` reads it
as `lake.<table>`. The `lake` group keeps it, the way `uv cache` keeps uv's:

```console
$ mb lake check        # every data file on disk, every evidence object hashed back
$ mb lake compact      # inlined rows to Parquet, small files merged, old snapshots expired
$ mb lake upgrade      # the catalog to the newest DuckLake spec
$ mb lake ingest DIR   # evidence files kept byte for byte, so they can leave git
$ mb lake materialize DIR  # kept evidence written back where it stood
$ mb lake replicate DIR    # every evidence object and the path index, copied to another disk
$ mb lake serve        # this lake over DuckDB's Quack protocol, on localhost:9494
```

`lake ingest` keeps each evidence file once per content in `blobs` (8 MiB
chunks, so a multi-gigabyte object fits) and its path in `lake.evidence`. The
bytes are never re-encoded, so every digest a `receipts.json` or an artifact
reference pins still verifies. Once the files are gone from disk, an artifact
read finds its object by digest, and receipt stores, events and node lookups
list what the lake indexes, materialized on first use into `.mb/evidence/`.

Experiment data therefore lives in the lake, not in git, and gets there on its
own: a trial session keeps its run's receipts and artifacts when it closes, and
collection keeps what it fetches from a job host, which has no lake. Files are
read once, in parallel, and every window of up to 256 MiB lands as one insert,
one data file; a `md5` per chunk lets `lake check` verify everything inside
DuckDB. The tree's copies are a cache and may be deleted: a pytest session
restores from the lake each `@job` resource and each file a node's
`receipts.json` pins before its tests run, and dispatch restores declared
resources and needs before shipping them. Run `lake replicate` after ingesting
data that is leaving git, until which the lake is its only copy.

`lake serve` lets other machines use this lake directly. It listens on localhost
only, so another machine comes in through ssh, then points `mb` at it:

```console
$ ssh -R 9494:localhost:9494 gold                  # from the center
gold$ export MB_LAKE=quack:localhost:9494 MB_LAKE_TOKEN=<.mb/run/lake.token>
gold$ mb job list                                  # the center's runs, read and written live
```

Both ends must run the same DuckDB release (`mb self version`). Maintenance
(`check`, `compact`, `upgrade`) stays on the machine holding the files.

## Portable process chores

Three everyday operations have no binary that behaves the same everywhere, so
they are verbs:

```console
$ mainboard proc timeout 600 pytest -x          # exit 124 when stopped, like GNU timeout
$ mainboard proc kill 4242                       # the process and everything it started
$ mainboard proc wait --port localhost:8000 --timeout 60
$ mainboard proc wait --file results/done.json --pid 4242
```

Everything after the limit is the command's own, its options and any `--` included:
`mb proc timeout 900 mb job submit --on gold -- python train.py` bounds a submit.

## Entering an environment

`mb shell` opens your own shell (zsh, bash, fish, pwsh, on every system) inside
the workspace's environment, and `mb run -- <cmd>` runs one command there:

```console
$ mb shell                            # or: mb shell --env serving
$ mb list torch                       # what is installed, through pixi list
$ mb tree numpy --invert              # who pulls it in, through pixi tree
```

The activation (the host's modules, pixi's own activation, second-stage
binaries) is written into the environment's own directory by `mb install`, so
the state directory keeps no activation scripts, and manifests template with
plain `{{ name }}` and `{{ env('HOME', '') }}` rather than a template engine.

## Logging

One logger for the tool and every workspace using it, loguru's call shape on
structlog: positional arguments format the message, keywords are fields.

```python
from mb import logger

logger.info("epoch {} done", 3, loss=0.12)
```

A terminal gets one readable line per event and anything else (a job's
captured log, CI) one JSON line naming the module and line it came from.
`MB_LOG_LEVEL` sets the floor (`info` by default) and `MB_LOG_FORMAT`
(`console` or `json`) overrides the guess. While a pytest trial runs, every
line logged from any code is also kept in that trial's record. A process that
configured structlog itself keeps its own configuration.

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
mainboard job submit --on gold path/to/test_experiment.py::test_measurement
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
| Fleet telemetry | `mainboard sample`, `host list --facts`, `host list` | Job/machine observations, not a replacement for in-process profiling |

Keep profiling separate from uninstrumented throughput measurements, and keep
the requested collection policy beside each saved profile. An invalid device
index is an error, never permission to sample a different card. Profiling study
exceptions propagate; `Row.has_evidence` describes capture, not success or a
scientific verdict. Use pytest parametrization for independently recorded trials.

Select Python regions with `auto` or `span`. Neither is statistical sampling.
The unused `PYTHON` flag was removed. Existing feature bit values are unchanged.
Historical policies containing that unsupported bit require their original source.
Launch work through `mainboard run` or
`mainboard job submit`, with the profiling context inside the target. There is no
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
mainboard job list --json
mainboard job collect research/reproducibility/datasets/experiments/architecture_error_census --on pedro-cvlab
mainboard query "SELECT project, hardware, count(*) AS runs FROM runs GROUP BY ALL"
mainboard query --project reproducibility "SELECT * FROM metrics ORDER BY recorded_at DESC LIMIT 20"
mainboard query "SELECT server, handle, backend_state, verdict, evidence, settled FROM jobs"
mainboard query "SELECT * FROM runs" --out /tmp/runs.parquet
mainboard query --file queries/inventory.sql --out /tmp/inventory.parquet
mainboard help artifacts
mainboard help job submit
```

`job list` pulls published results from running jobs as well as finished ones. Run
repeated passes to refresh remote data. A query itself has no network side effects.
DuckDB reads the collected Parquet fragments and event journals directly. There is
no shared database file for different servers to lock, and no database service to deploy.
Each query sees a fresh inventory. It is not a transaction across all servers.

`job collect` also imports runs started directly on a node, using the same collection
path as the settling pass. Remote filesystem operations use Python's standard library and
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
uv tool install --reinstall --python 3.14 --from './packages/mainboard[plot]' mainboard --force
mainboard paper plot "SELECT hardware, count(*) AS runs FROM runs GROUP BY hardware" --project reproducibility --x hardware --y runs --kind bar --out /tmp/run-inventory.png --out /tmp/run-inventory.pdf --dpi 300
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
mainboard paper plot --config plots.toml --figure inventory
mainboard paper plot --config plots.toml --figure inventory --out /tmp/inventory.svg
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

Source mirroring needs no rsync, tar or Bash on either end. The host's own Python
(the profile's `python`, 3.9 or later) runs a standard-library agent that
Mainboard streams over the SSH channel on every transfer. Both ends describe their
files by size and SHA-256, remembered by each file's stamp, so only changed files
cross, as one compressed tar stream. Paths the workspace no longer holds are pruned
inside the include paths only, never where an ignore file, the host's `exclude`,
its `protect` rules, a declared output or a card lease applies. Snapshot pins run
in the same agent under a kernel file lock. Where Git answers on the workstation,
each repository in the workspace decides its own files: everything it tracks, plus
the untracked files its own ignore files leave. A parent's ignore file never
reaches into a nested repository or submodule. Without Git the same ignore files
are read directly. Sync patterns read like `.gitignore` lines, except that a
pattern is anchored only by a leading `/` and `dir/***` names a directory with
everything beneath it.

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
An edit after capture requires a new bundle, not a commit. Discovery follows the
same per-repository file set as mirroring, reading `.gitignore` files directly when
no Git answers; secrets and output protections remain. A local trial's bundle also
leaves out `[workspace] data` (default `/datasets/`, `/references/`, `**/evidence/`),
which is pinned through `needs` or `resources`, never archived as source; host mirrors
still ship what their own sync include names. An interrupted archive
leaves one `<digest>.zip.partial` its retry replaces, swept after a day.
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

Collection from Linux and macOS nodes has been exercised. The next boundary is
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
Automatic host selection remains under development.

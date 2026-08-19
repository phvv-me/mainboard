from pathlib import Path

import pytest

from mainboard import Project
from mainboard.dispatch.backends import Credentials

MANIFEST = Project().manifest

_FIXTURE = """
[workspace]
name = "lab"
platforms = ["linux-64", "linux-aarch64"]

[vars]
cuda = "13.0"
scratch = "{{ env('MC_TEST_SCRATCH', '/tmp') }}"
station = "{{ os_name() }}-{{ arch() }}"

[deps]
python = ">=3.14"
pueue = "*"

[python.deps]
torch = ">=2.9"
lab-core = { path = "packages/lab-core", editable = true }

[envs.serving]
system = { cuda = "{{ vars.cuda }}" }

[envs.serving.python.deps]
vllm = "*"

[containers.ngc]
image = "nvcr.io/nvidia/pytorch:25.06-py3"
binds = ["{{ vars.scratch }}"]

[hosts.defaults]
sync = { include = ["packages"], protect = ["results/***"] }

[hosts.defaults.defaults]
walltime = "00:30:00"

[hosts.gold]
kind = "ssh"
env = "serving"

[hosts.miyabi-g]
kind = "pbs"
root = "/work/xg25g007/x10537/projects"
account = "xg25g007"
container = "ngc"
modules = { singularity = "4.2.1" }
scratch = "{{ env('LOCALDIR', '/local') }}"
sync = { exclude = ["data/raw"] }

[hosts.miyabi-g.queues.short-g]
max-walltime = "07:59:59"
mem-ceiling-gb = 100
gpus-per-node = 1

[hosts.miyabi-g.queues.debug-g]
max-walltime = "00:30:00"

[hosts.miyabi-g.defaults]
queue = "debug-g"
mem-gb = "min(100, attempt * 50)"

[tasks]
test = { run = "pytest", dir = "packages/lab-core" }
"""


@pytest.fixture(autouse=True)
def sealed_credentials() -> None:
    """Keep the developer's own workspace `.env` out of every test.

    The credential loader merges that file into the process environment the first time a backend
    looks a key up, so a suite running inside a real workspace would inherit whatever keys the
    machine holds and a test that clears one would watch it come straight back. Marking the
    shared loader spent before each test makes the whole suite read the same on a keyed machine
    as on a bare one, and the loader's own tests unseal it onto a workspace they built.
    """
    Credentials().loaded = True


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace directory holding the full-featured fixture manifest."""
    monkeypatch.setenv("MC_TEST_SCRATCH", "/scratch/lab")
    (tmp_path / MANIFEST).write_text(_FIXTURE)
    return tmp_path

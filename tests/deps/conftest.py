import os
import tomllib
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import Manifest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# A manifest carrying one table of every shape the addressing has to reach: conda and an
# ecosystem, runtime and development, both spellings of a development table, a platform overlay
# and an environment with an overlay of its own. Written the way the real manifest is written,
# values aligned into a column and comments introducing the table below them, so an edit that
# disturbs either shows up here.
_MANIFEST = """\
[workspace]
name      = "lab"
platforms = ["linux-64"]
channels  = ["rapidsai", "conda-forge"]

[deps]
python    = ">=3.14"
pueue     = "*"

[python]
index-url = "https://mirror.internal/simple"

[python.deps]
torch     = ">=2.9"
lab-core  = { path = "packages/lab-core", editable = true }

[dev.deps]
protobuf  = ">=6"

[dev.python.deps]
pytest    = ">=9, <10"

# runtime-keyed toolchains, the table name matches the package in [deps]
[nodejs]
manager = "npm"

[nodejs.deps]
es-toolkit = "^1.49.0"

[nodejs.dev]
"@puppeteer/browsers" = ">=3, <4"

[on.linux-64.deps]
cuda-version = "13.0"

[envs.serving]
system = { cuda = "13.0" }

[envs.serving.deps]
sglang = "*"

[envs.serving.python.deps]
vllm = "*"

[envs.serving.nodejs.dev]
vite = ">=7"

[envs.serving.on.linux-64.python.deps]
flashinfer = "*"
"""


@pytest.fixture
def text() -> str:
    """The fixture manifest exactly as it would sit on disk."""
    return _MANIFEST


@pytest.fixture
def manifest() -> Manifest:
    """The fixture manifest, parsed and validated."""
    return Manifest.model_validate(tomllib.loads(_MANIFEST))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A workspace directory holding the fixture manifest under the tool's own name."""
    (tmp_path / "mainboard.toml").write_text(_MANIFEST, encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def stub_pixi(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Put a `pixi` on plumbum's PATH so a search resolves without one being installed.

    Autouse, because the engine bootstraps pixi when the name is absent, and running the
    installer would eat the fake process a test registered for the call under test.
    """
    bindir = tmp_path_factory.mktemp("bin")
    executable = bindir / "pixi"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    with local.env(PATH=f"{bindir}{os.pathsep}{local.env['PATH']}"):
        yield

import subprocess
from math import inf
from pathlib import Path

import pytest

from mainboard import Board
from mainboard.costs import Ledger

from .....costs.builders import landings


@pytest.fixture
def key(tmp_path: Path) -> Path:
    """The private half of a fresh ssh key pair with no passphrase, its public half beside it."""
    private = tmp_path / "key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
        check=True,
        capture_output=True,
    )
    return private


@pytest.fixture
def board(workspace: Path) -> Board:
    """A workspace whose lake holds Shadeform's record of an H100 and an H200.

    The H100 landed five times in five, the H200 once in three. One slow H200 landing kept a ready
    time that never ended (inf), which would make the H200's median ready time NaN if priced.
    """
    found = Board(workspace)
    ledger = Ledger(found.dispatcher.cache.session)
    rows = [
        *landings("shadeform", gpu="H100", landed=5, ready_s=90.0),
        *landings("shadeform", gpu="H200", landed=1, slow=1, ready_s=600.0),
        *landings("shadeform", gpu="H200", slow=1, ready_s=inf),
    ]
    for row in rows:
        ledger.record(row)
    return found

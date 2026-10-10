import os
import socket
import subprocess
import sysconfig
from pathlib import Path

import pytest


def test_live_paths_symlinks_and_transports_are_refused(tmp_path: Path) -> None:
    tree = Path(__file__).resolve().parents[2]
    alias = tmp_path / "escape"
    alias.symlink_to(tree)
    with pytest.raises(PermissionError, match="M00 denies write"):
        (alias / "M00-forbidden").write_text("forbidden", encoding="utf-8")
    with pytest.raises(PermissionError, match="M00 denies real transport"):
        subprocess.run(["ssh", "gold", "true"], check=True, timeout=5)
    with pytest.raises(PermissionError, match="M00 denies DNS"):
        socket.create_connection(("api.vast.ai", 443), timeout=1)
    assert not (tree / "M00-forbidden").exists()


def test_postgres_is_a_temporary_catalog(postgres: str) -> None:
    psql = Path(sysconfig.get_path("scripts")) / "psql"
    done = subprocess.run(
        [str(psql), postgres, "-XAt", "-v", "ON_ERROR_STOP=1", "-c",
         "CREATE SCHEMA gate; CREATE TABLE gate.proof (id integer PRIMARY KEY); "
         "INSERT INTO gate.proof VALUES (1); SELECT count(*) FROM gate.proof; "
         "SHOW data_directory; SHOW listen_addresses;"],
        check=True, capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert "\n1\n" in done.stdout
    assert os.environ["MB_GATE_ROOT"] in done.stdout
    assert done.stdout.endswith("\n\n")  # listen_addresses is empty: no TCP listener.

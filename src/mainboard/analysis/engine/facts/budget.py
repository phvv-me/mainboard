# What one run's DuckDB work may take.

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import duckdb
from patos import FrozenModel
from pydantic import Field

from ....state.storage import database

# The DuckDB extension that parses source into the syntax trees every fact base reads.
_EXTENSION = "sitting_duck"


class Budget(FrozenModel):
    """What one run's DuckDB work may take, in a folder of its own that goes with the run.

    DuckDB's defaults (80% of memory, a spill of up to 90% of free disk into `.tmp` under the
    working directory, the monorepo root under `mainboard run`) let a few concurrent runs swap
    the machine to a halt. Every setting goes in by `SET`: a cap given to `connect(config=)` is
    reported but not enforced (duckdb/duckdb-python#667). Values measured in
    scratchpad/mcmr-mb/DUCKDB-RESOURCES.md and since: at 2 GB a full check of
    research/reproducibility peaks under 6 GB of process memory (most of it outside DuckDB's
    buffer manager: mcmr's kernel and Arrow copies), at about 3% of the speed of 3 GB.
    """

    memory: str = "2GB"
    spill: str = "2GB"
    threads: int = 8
    folder: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / f"mb-run-{uuid4().hex[:12]}"
    )

    def _statements(self, **settings: str | int) -> list[str]:
        """`settings` with this run's spill folder and spill cap.

        The folder is created, since DuckDB makes only its last component.
        """
        (self.folder / "spill").mkdir(parents=True, exist_ok=True)
        merged = {
            "max_temp_directory_size": self.spill,
            **settings,
            "temp_directory": str(self.folder / "spill"),
        }
        return [f"SET {name} = '{value}'" for name, value in merged.items()]

    def close(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def connect(self, file: Path | None = None) -> duckdb.DuckDBPyConnection:
        """A fact base in `file`, in memory for None.

        Its tables stay compressed on disk and leave memory without spilling (an in-memory base of
        4.7 GB is a 1 GB file), at about 5% of the speed.
        """
        if file:
            file.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(file or ":memory:"))
        for statement in self._statements(memory_limit=self.memory, threads=self.threads):
            connection.execute(statement)
        # Loaded with every base: a macro binds the functions it calls when it is created, and
        # `tree_clean` calls `parse_ast_list` even where no tree is parsed (`refactor fix` reading
        # a cached base failed to define it, 2026-10-23).
        database.load(connection, [_EXTENSION])
        return connection

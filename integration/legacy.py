"""The unmodified c97cb81 release, runnable beside the tree under test."""

import hashlib
import io
import os
import subprocess
import tarfile
import venv
from pathlib import Path

from patos import FrozenModel

_REVISION = "c97cb81daed05e8b9d0ed21f87a7427075487562"


class LegacyRuntime(FrozenModel):
    """The legacy sources, an interpreter that imports only them, and the archive's digest."""

    root: Path
    interpreter: Path
    archive_sha256: str

    @classmethod
    def extract(cls, where: Path) -> LegacyRuntime:
        """Extract the legacy archive below `where` and prove its import comes from there.

        where: an empty temporary directory.
        """
        archive = cls._archive()
        root = where / "legacy"
        root.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as legacy:
            legacy.extractall(root, filter="data")
        prefix = where / "interpreter"
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(prefix)
        runtime = cls(
            root=root,
            interpreter=prefix / "bin/python",
            archive_sha256=hashlib.sha256(archive).hexdigest(),
        )
        proof = runtime.run("import mainboard; print(mainboard.__file__)")
        assert Path(proof.stdout.strip()).resolve() == root / "src/mainboard/__init__.py"
        return runtime

    def run(self, program: str) -> subprocess.CompletedProcess[str]:
        """Run `program` with the legacy sources as the only import root, not the new tree's."""
        environment = {**os.environ, "PYTHONPATH": str(self.root / "src")}
        return subprocess.run(
            [str(self.interpreter), "-c", program],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=True,
        )

    @staticmethod
    def _archive() -> bytes:
        """The immutable git archive of the legacy revision."""
        tree = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["git", "-C", str(tree), "archive", _REVISION],
            capture_output=True,
            timeout=30,
            check=True,
        ).stdout

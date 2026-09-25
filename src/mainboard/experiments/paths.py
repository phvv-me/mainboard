"""The self-contained per-experiment directory layout."""

from pathlib import Path

from patos import FrozenModel


class ExperimentPaths(FrozenModel):
    """Where one experiment keeps its code, raw rows and plots.

    name: the experiment's directory name.
    root: the parent holding every experiment, so files live at root / name.
    """

    name: str
    root: Path

    @classmethod
    def from_file(cls, file: str | Path) -> ExperimentPaths:
        """Derive the layout from an experiment package's own `__file__`."""
        directory = Path(file).resolve().parent
        return cls(name=directory.name, root=directory.parent)

    @property
    def home(self) -> Path:
        """The experiment directory, created on demand."""
        path = self.root / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def raw_dir(self) -> Path:
        """Measurement rows, never hand-edited."""
        return self._ensure("raw")

    @property
    def plots_dir(self) -> Path:
        """Rendered figures."""
        return self._ensure("plots")

    @property
    def results_dir(self) -> Path:
        """Pooled summaries derived from the raw rows."""
        return self._ensure("results")

    def table(self, filename: str) -> Path:
        """A raw table path, given the `.parquet` suffix unless it names a table format already.

        A bare suffix check misreads a model slug such as `Qwen3-1.7B` as an extension.
        """
        if Path(filename).suffix not in (".parquet", ".csv"):
            filename = f"{filename}.parquet"
        return self.raw_dir / filename

    def device_table(self, tag: str, name: str = "results") -> Path:
        """The table for one device, under its own tag so hosts never overwrite each other."""
        path = self.raw_dir / tag / self.table(name).name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def plot(self, filename: str) -> Path:
        """A figure path under the plots directory."""
        return self.plots_dir / filename

    def _ensure(self, child: str) -> Path:
        path = self.home / child
        path.mkdir(parents=True, exist_ok=True)
        return path

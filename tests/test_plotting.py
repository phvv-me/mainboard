from typing import TYPE_CHECKING

import polars as pl
import pytest

from mainboard.cli import build
from mainboard.manifest.schema.plot import PlotStyle

if TYPE_CHECKING:
    from pathlib import Path

plotting = pytest.importorskip("mainboard.plotting", exc_type=ModuleNotFoundError)


@pytest.mark.parametrize("kind", ["scatter", "line", "bar"])
def test_plots_use_complete_outputs_and_preserve_existing_files(tmp_path: Path, kind: str) -> None:
    frame = pl.DataFrame(
        {"width": [1, 2, 1, 2], "error": [1.0, 2.0, 1.5, 3.0], "cell": [0, 0, 1, 1]}
    )
    picture = plotting.Plot(frame)
    paths = (tmp_path / "figures" / "error.png", tmp_path / "error.svg")
    before = plotting.mpl.rcParams["savefig.dpi"]
    assert picture.save(*paths, x="width", y="error", hue="cell", kind=kind, dpi=100) == paths
    assert paths[0].read_bytes().startswith(b"\x89PNG")
    assert b"<svg" in paths[1].read_bytes()
    assert plotting.mpl.rcParams["savefig.dpi"] == before
    assert not plotting.plt.get_fignums()
    saved = paths[0].read_bytes()
    with pytest.raises(FileExistsError):
        picture.save(*paths, x="width", y="error")
    assert paths[0].read_bytes() == saved
    assert sorted(path.name for path in tmp_path.iterdir()) == ["error.svg", "figures"]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pl.DataFrame({"x": [], "y": []}), "without null"),
        (pl.DataFrame({"x": [1], "y": [None]}), "without null"),
        (pl.DataFrame({"x": [1], "y": [float("nan")]}), "finite"),
        (pl.DataFrame({"x": [1], "y": ["not numeric"]}), "numeric"),
    ],
)
def test_invalid_plot_data_is_not_silently_dropped(
    tmp_path: Path, frame: pl.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plotting.Plot(frame).save(tmp_path / "refused.png", x="x", y="y")
    assert not list(tmp_path.iterdir())


def test_bars_require_explicit_sql_aggregation(tmp_path: Path) -> None:
    frame = pl.DataFrame({"x": [1, 1], "y": [2.0, 4.0]})
    with pytest.raises(ValueError, match="aggregate.*SQL"):
        plotting.Plot(frame).save(tmp_path / "refused.png", x="x", y="y", kind="bar")


def test_bad_output_formats_leave_no_partial_render(tmp_path: Path) -> None:
    frame = pl.DataFrame({"x": [1], "y": [2.0]})
    with pytest.raises(ValueError, match="unsupported plot extension"):
        plotting.Plot(frame).save(tmp_path / "ok.png", tmp_path / "bad.xyz", x="x", y="y")
    assert not list(tmp_path.iterdir())
    assert not plotting.plt.get_fignums()


def test_cli_plot_uses_sql_and_requested_dpi(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "plot.png"
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(
            [
                "plot",
                "SELECT * FROM (VALUES (1, 2.25), (2, 3.75)) AS t(width, readings)",
                "--x",
                "width",
                "--y",
                "readings",
                "--out",
                str(path),
                "--dpi",
                "100",
            ]
        )
    assert capsys.readouterr().out.strip() == str(path)
    image = plotting.plt.imread(path)
    assert image.shape[1] == 550


def test_line_chart_retains_every_selected_row_and_sql_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    points = []

    def inspect_lines(canvas, paths) -> None:
        points.extend(canvas.axes[0].lines[0].get_xydata().tolist())

    picture = plotting.Plot(pl.DataFrame({"x": [2, 1, 1], "y": [4.0, 2.0, 3.0]}))
    monkeypatch.setattr(picture, "_publish", inspect_lines)
    picture.save(tmp_path / "line.png", x="x", y="y", kind="line")
    assert points == [[2, 4.0], [1, 2.0], [1, 3.0]]


@pytest.mark.parametrize("kind", ["scatter", "line", "bar"])
def test_numeric_hue_keeps_sql_order_and_exact_palette_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    seen = []

    def inspect_colors(canvas, paths) -> None:
        axis = canvas.axes[0]
        labels = axis.get_legend_handles_labels()[1]
        assert labels == ["20", "10"]
        if kind == "scatter":
            colors = axis.collections[0].get_facecolors()
        elif kind == "line":
            colors = [line.get_color() for line in axis.lines[:2]]
        else:
            colors = [patch.get_facecolor() for patch in axis.patches[:2]]
        seen.extend(plotting.mpl.colors.to_hex(color) for color in colors)

    picture = plotting.Plot(pl.DataFrame({"x": [1, 2], "y": [2.0, 3.0], "cell": [20, 10]}))
    monkeypatch.setattr(picture, "_publish", inspect_colors)
    picture.save(tmp_path / "ordered.png", x="x", y="y", hue="cell", kind=kind)
    assert seen == list(plotting.paleta.palette.categorical[:2])


def test_exhausted_palette_does_not_cycle(tmp_path: Path) -> None:
    frame = pl.DataFrame({"x": range(7), "y": range(7), "cell": range(7)})
    with pytest.raises(ValueError, match="6 slots"):
        plotting.Plot(frame).save(tmp_path / "refused.png", x="x", y="y", hue="cell")
    assert not list(tmp_path.iterdir())


def test_named_style_uses_manifest_dimensions_dpi_and_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mainboard.toml").write_text(
        '[workspace]\nname="demo"\n[plots.paper]\n'
        'palette="paleta-meta"\nfigsize=[3.25,2.0]\ndpi=100\n'
        '[plots.paper.rc]\n"axes.labelsize"=12\n'
    )
    captured = []
    original = plotting.Plot._publish

    def inspect_style(self, canvas, paths) -> None:
        captured.append(canvas.axes[0].xaxis.label.get_fontsize())
        original(self, canvas, paths)

    monkeypatch.setattr(plotting.Plot, "_publish", inspect_style)
    output = tmp_path / "paper.png"
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(
            [
                "plot",
                "SELECT 1 AS x, 2 AS y",
                "--x",
                "x",
                "--y",
                "y",
                "--style",
                "paper",
                "--out",
                str(output),
            ]
        )
    assert plotting.plt.imread(output).shape[:2] == (200, 325)
    assert captured == [12]


def test_unknown_native_rc_parameter_fails_without_publishing(tmp_path: Path) -> None:
    style = PlotStyle(rc={"font.not-a-real-setting": 8})
    with pytest.raises(KeyError):
        plotting.Plot(pl.DataFrame({"x": [1], "y": [2]}), style).save(
            tmp_path / "refused.png", x="x", y="y"
        )
    assert not list(tmp_path.iterdir())

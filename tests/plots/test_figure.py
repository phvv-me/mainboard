from typing import TYPE_CHECKING

import pytest

from mainboard.cli import build
from mainboard.manifest.schema.figures.figure import FigureSpec
from mainboard.manifest.schema.plot import PlotStyle
from mainboard.results import Results

if TYPE_CHECKING:
    from pathlib import Path

rendering = pytest.importorskip("mainboard.plots.figure", exc_type=ModuleNotFoundError)


def test_native_layers_facets_and_supplied_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification = FigureSpec.model_validate(
        {
            "panels": {
                "rates": {
                    "sql": (
                        "SELECT * FROM (VALUES (1, 2., 1., 3., 'a', 'first'), "
                        "(2, 4., 3., 5., 'a', 'second')) t(x,y,low,high,engine,corpus)"
                    ),
                    "variables": {"x": "x", "y": "y", "color": "engine"},
                    "layers": [
                        {"mark": "Dot"},
                        {"mark": "Range", "variables": {"ymin": "low", "ymax": "high"}},
                    ],
                    "facet": {"col": "corpus"},
                    "axis": {"ylabel": "MB/s"},
                }
            },
        }
    )
    picture = rendering.FigurePlot(PlotStyle(figsize=(5, 2), colors={"a": "#7755aa"}))
    captured = []
    original = picture._publish

    def inspect(canvas, paths):
        captured.append(len(canvas.axes))
        original(canvas, paths)

    monkeypatch.setattr(picture, "_publish", inspect)
    output = tmp_path / "facets.png"
    picture.render(specification, Results(tmp_path).query, output, dpi=80)
    assert captured == [2]
    assert rendering.plt.imread(output).shape[:2] == (160, 400)


@pytest.mark.parametrize(
    "layer,message",
    [
        ({"mark": "Range"}, "explicit bounds"),
        ({"mark": "Range", "variables": {"ymin": "low"}}, "both lower"),
        ({"mark": "Range", "variables": {"ymin": "high", "ymax": "low"}}, "lower bound"),
    ],
)
def test_ranges_never_estimate_or_reverse(tmp_path: Path, layer, message: str) -> None:
    spec = FigureSpec.model_validate(
        {
            "panels": {
                "a": {
                    "sql": "SELECT 1 x, 2.0 y, 1.0 low, 3.0 high",
                    "variables": {"x": "x", "y": "y"},
                    "layers": [layer],
                }
            }
        }
    )
    with pytest.raises(ValueError, match=message):
        rendering.FigurePlot().render(spec, Results(tmp_path).query, tmp_path / "bad.png")
    assert not list(tmp_path.glob("*.png"))


def test_cli_explicit_config_does_not_become_execution_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "paper" / "plots.toml"
    config.parent.mkdir()
    config.write_text("""[workspace]
name="figures"
[plots.paper]
figsize=[3,2]
dpi=80
[plots.paper.colors]
ours="#7755aa"
[figures.demo]
style="paper"
out=["demo.png"]
[figures.demo.panels.main]
sql="SELECT 1 x, 2.0 y, 'ours' engine"
variables={x="x", y="y", color="engine"}
layers=[{mark="Dot", kws={pointsize=4}}]
""")
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(["plot", "--config", str(config), "--figure", "demo"])
    assert (tmp_path / "demo.png").exists()
    assert not (config.parent / "demo.png").exists()


def test_nonrectangular_mosaic_is_refused() -> None:
    with pytest.raises(ValueError, match="rectangle"):
        FigureSpec.model_validate(
            {
                "mosaic": [["a", "a"], ["a", "."]],
                "panels": {"a": {"sql": "SELECT 1", "layers": [{"mark": "Dot"}]}},
            }
        )


def test_shared_legend_follows_explicit_style_order_across_panel_subsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = FigureSpec.model_validate(
        {
            "panels": {
                engine: {
                    "sql": f"SELECT 1 x, 2.0 y, '{engine}' engine",
                    "variables": {"x": "x", "y": "y", "color": "engine"},
                    "layers": [{"mark": "Dot"}],
                }
                for engine in ("baseline", "ours")
            }
        }
    )
    style = PlotStyle(
        colors={"ours": "#7755aa", "absent": "#006644", "baseline": "#aa2222"},
        labels={"ours": "Our engine", "baseline": "Baseline"},
    )
    picture = rendering.FigurePlot(style)
    checked = []

    def inspect(canvas, paths):
        [legend] = canvas.legends
        assert [text.get_text() for text in legend.get_texts()] == ["Our engine", "Baseline"]
        assert [handle.get_color() for handle in legend.legend_handles] == ["#7755aa", "#aa2222"]
        checked.append(True)

    monkeypatch.setattr(picture, "_publish", inspect)
    picture.render(spec, Results(tmp_path).query, tmp_path / "ordered.png")
    assert checked == [True]


def test_layer_data_rebinds_inherited_category_and_color(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = FigureSpec.model_validate(
        {
            "panels": {
                "a": {
                    "sql": (
                        "SELECT * FROM (VALUES ('first', 3.0, 'ours', 'en'), "
                        "('second', 4.0, 'ours', 'en'), ('first', 5.0, 'ours', 'zh'), "
                        "('second', 6.0, 'ours', 'zh')) t(model, rate, engine, corpus)"
                    ),
                    "variables": {"x": "model", "y": "rate", "color": "engine"},
                    "order": {"x": ["first", "second"]},
                    "facet": {"col": "corpus", "order": ["en", "zh"]},
                    "layers": [
                        {"mark": "Dot"},
                        {
                            "mark": "Dot",
                            "sql": (
                                "SELECT * FROM (VALUES ('second', 1.5, 'baseline', 'zh'), "
                                "('second', 1.4, 'baseline', 'en')) "
                                "t(model, missing, engine, corpus)"
                            ),
                            "variables": {"y": "missing"},
                            "kws": {"marker": "x"},
                        },
                    ],
                }
            }
        }
    )
    picture = rendering.FigurePlot(PlotStyle(colors={"ours": "#7755aa", "baseline": "#aa2222"}))
    inspected = []

    def inspect(canvas, paths):
        assert len(canvas.axes) == 2
        for axis, expected in zip(canvas.axes, (1.4, 1.5), strict=True):
            marks = axis.collections[-1]
            assert marks.get_offsets().tolist() == [[1.0, expected]]
            assert rendering.mpl.colors.to_hex(marks.get_facecolors()[0]) == "#aa2222"
        assert len(canvas.legends) == 1
        inspected.append(True)

    monkeypatch.setattr(picture, "_publish", inspect)
    picture.render(spec, Results(tmp_path).query, tmp_path / "cross.png")
    assert inspected == [True]


def test_native_dodge_gap_shrinks_caps_without_moving_interval_centers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = FigureSpec.model_validate(
        {
            "panels": {
                "a": {
                    "sql": (
                        "SELECT * FROM (VALUES ('one', 2., 1., 3., 'a'), "
                        "('one', 4., 3., 5., 'b')) t(category, y, low, high, engine)"
                    ),
                    "variables": {"x": "category", "y": "y", "color": "engine"},
                    "layers": [
                        {"mark": "Bar", "kws": {"width": 0.8}, "moves": {"Dodge": {}}},
                        {
                            "mark": "Range",
                            "variables": {"ymin": "low", "ymax": "high"},
                            "moves": {"Dodge": {}},
                        },
                        {
                            "mark": "Dash",
                            "variables": {"y": "high"},
                            "kws": {"width": 0.8},
                            "moves": {"Dodge": {"gap": 0.6}},
                        },
                    ],
                }
            }
        }
    )
    picture = rendering.FigurePlot(PlotStyle(colors={"a": "#7755aa", "b": "#aa2222"}))
    checked = []

    def inspect(canvas, paths):
        axis = canvas.axes[0]
        centers = [bar.get_x() + bar.get_width() / 2 for bar in axis.patches]
        intervals, caps = [collection.get_segments() for collection in axis.collections]
        assert [segment[:, 0].mean() for segment in intervals] == pytest.approx(centers)
        assert [segment[:, 0].mean() for segment in caps] == pytest.approx(centers)
        assert all(
            segment[:, 0].max() - segment[:, 0].min() < axis.patches[0].get_width()
            for segment in caps
        )
        checked.append(True)

    monkeypatch.setattr(picture, "_publish", inspect)
    picture.render(spec, Results(tmp_path).query, tmp_path / "aligned.png")
    assert checked == [True]


def test_heatmap_cells_follow_distinct_axis_values(tmp_path: Path) -> None:
    specification = FigureSpec.model_validate(
        {
            "panels": {
                "grid": {
                    "sql": (
                        "SELECT * FROM (VALUES (82, 827., 16.8, 'a'), (128, 903., 11.0, 'b'), "
                        "(132, 3353., 5.8, 'c')) t(sms, bandwidth, latency, label)"
                    ),
                    "variables": {
                        "x": "sms",
                        "y": "bandwidth",
                        "color": "latency",
                        "text": "label",
                    },
                    "layers": [{"mark": "Heatmap", "kws": {"log": True, "label": "ms"}}],
                    "axis": {"xlabel": "Multiprocessors"},
                }
            },
        }
    )
    output = tmp_path / "heatmap.png"
    rendering.FigurePlot(PlotStyle(figsize=(4, 3))).render(
        specification, Results(tmp_path).query, output, dpi=80
    )
    assert rendering.plt.imread(output).shape[:2] == (240, 320)


def test_heatmap_refuses_repeated_cells(tmp_path: Path) -> None:
    specification = FigureSpec.model_validate(
        {
            "panels": {
                "grid": {
                    "sql": "SELECT * FROM (VALUES (1, 2., 3.), (1, 2., 4.)) t(x, y, v)",
                    "variables": {"x": "x", "y": "y", "color": "v"},
                    "layers": [{"mark": "Heatmap"}],
                }
            },
        }
    )
    with pytest.raises(ValueError, match="cells repeat"):
        rendering.FigurePlot().render(
            specification, Results(tmp_path).query, tmp_path / "bad.png"
        )

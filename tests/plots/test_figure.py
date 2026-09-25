from typing import TYPE_CHECKING

import pytest
from matplotlib.ticker import NullFormatter
from pydantic import JsonValue

from mainboard.cli import build
from mainboard.manifest.schema.figures.figure import FigureSpec
from mainboard.manifest.schema.plot import PlotStyle
from mainboard.results import Results

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure
    from matplotlib.legend import Legend

rendering = pytest.importorskip("mainboard.plots.figure", exc_type=ModuleNotFoundError)

_XY = {"x": "x", "y": "y"}
_ENGINE = {"x": "x", "y": "y", "color": "engine"}
_CELLS = {"x": "x", "y": "y", "color": "v"}
_BY_MODEL = {"x": "model", "y": "rate", "color": "engine"}
_BOUNDED = "SELECT 1 x, 2.0 y, 1.0 low, 3.0 high"
_RANGE = {"mark": "Range", "variables": {"ymin": "low", "ymax": "high"}}
_RATES = (
    "SELECT * FROM (VALUES ('first', 2., 'ours', 'cold'), ('second', 4., 'baseline', 'warm')) "
    "t(model, rate, engine, batch)"
)
_NAMED = PlotStyle(
    colors={"ours": "#7755aa", "baseline": "#aa2222"}, labels={"ours": "Our engine"}
)


def drawn(tmp_path: Path, *panels: dict[str, JsonValue], style: PlotStyle | None = None) -> Figure:
    """Render the panels side by side and hand back the canvas instead of publishing it."""
    picture = rendering.FigurePlot(style)
    canvases: list[Figure] = []
    picture._publish = lambda canvas, paths: canvases.append(canvas)
    specification = FigureSpec.model_validate(
        {"panels": {str(index): panel for index, panel in enumerate(panels)}}
    )
    picture.render(specification, Results(tmp_path).query, tmp_path / "a.png")
    return canvases[0]


def texts(legend: Legend | None) -> list[str]:
    """The entries a legend shows, in order; none when there is no legend."""
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def test_native_layers_facet_with_supplied_ranges(tmp_path: Path) -> None:
    canvas = drawn(
        tmp_path,
        {
            "sql": (
                "SELECT * FROM (VALUES (1, 2., 1., 3., 'a', 'first'), "
                "(2, 4., 3., 5., 'a', 'second')) t(x,y,low,high,engine,corpus)"
            ),
            "variables": _ENGINE,
            "layers": [{"mark": "Dot"}, _RANGE],
            "facet": {"col": "corpus"},
            "axis": {"ylabel": "MB/s"},
        },
        style=PlotStyle(colors={"a": "#7755aa"}),
    )
    assert len(canvas.axes) == 2


def test_facets_share_only_what_the_panel_asks(tmp_path: Path) -> None:
    canvas = drawn(
        tmp_path,
        {
            "sql": (
                "SELECT * FROM (VALUES (1, 2., 'a', 'first'), (2, 4., 'a', 'first'), "
                "(1, 200., 'a', 'second'), (2, 400., 'a', 'second')) t(x,y,engine,corpus)"
            ),
            "variables": _ENGINE,
            "layers": [{"mark": "Line"}],
            "facet": {"col": "corpus"},
            "share": {"y": False},
        },
        style=PlotStyle(colors={"a": "#7755aa"}),
    )
    [first, second] = [axis.get_ylim()[1] for axis in canvas.axes]
    assert first < 10 < second


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


_DOT = {"sql": "SELECT 1", "layers": [{"mark": "Dot"}]}


@pytest.mark.parametrize(
    ("figure", "message"),
    [
        ({"panels": {}}, "at least one panel"),
        ({"mosaic": [["a", "a"], ["a"]], "panels": {"a": _DOT}}, "must be rectangular"),
        ({"mosaic": [["a", "b"]], "panels": {"a": _DOT}}, "exactly its named panels"),
        ({"mosaic": [["a", "a"], ["a", "."]], "panels": {"a": _DOT}}, "not a rectangle"),
        ({"panels": {"a": {"layers": [{"mark": "Dot"}]}}}, "exactly one SQL statement or file"),
        ({"panels": {"a": {"sql": "SELECT 1", "layers": []}}}, "at least one layer"),
        (
            {
                "panels": {
                    "a": {
                        "sql": "SELECT 1",
                        "layers": [{"mark": "Dot", "sql": "SELECT 2", "file": "dots.sql"}],
                    }
                }
            },
            "not both",
        ),
    ],
    ids=[
        "no_panels",
        "ragged_mosaic",
        "unplaced_panel",
        "nonrectangular_panel",
        "sourceless_panel",
        "layerless_panel",
        "doubly_sourced_layer",
    ],
)
def test_an_ambiguous_figure_is_refused_before_any_query(
    figure: dict[str, JsonValue], message: str
) -> None:
    """Placement and data sources are declared exactly once, so nothing is guessed at render."""
    with pytest.raises(ValueError, match=message):
        FigureSpec.model_validate(figure)


def test_shared_legend_follows_explicit_style_order_across_panel_subsets(tmp_path: Path) -> None:
    style = PlotStyle(
        colors={"ours": "#7755aa", "absent": "#006644", "baseline": "#aa2222"},
        labels={"ours": "Our engine", "baseline": "Baseline"},
    )
    canvas = drawn(
        tmp_path,
        *(
            {
                "sql": f"SELECT 1 x, 2.0 y, '{engine}' engine",
                "variables": _ENGINE,
                "layers": [{"mark": "Dot"}],
            }
            for engine in ("baseline", "ours")
        ),
        style=style,
    )
    [legend] = canvas.legends
    assert texts(legend) == ["Our engine", "Baseline"]
    assert [handle.get_color() for handle in legend.legend_handles] == ["#7755aa", "#aa2222"]


def test_layer_data_rebinds_inherited_category_and_color(tmp_path: Path) -> None:
    canvas = drawn(
        tmp_path,
        {
            "sql": (
                "SELECT * FROM (VALUES ('first', 3.0, 'ours', 'en'), "
                "('second', 4.0, 'ours', 'en'), ('first', 5.0, 'ours', 'zh'), "
                "('second', 6.0, 'ours', 'zh')) t(model, rate, engine, corpus)"
            ),
            "variables": _BY_MODEL,
            "order": {"x": ["first", "second"]},
            "facet": {"col": "corpus", "order": ["en", "zh"]},
            "layers": [
                {"mark": "Dot"},
                {
                    "mark": "Dot",
                    "sql": (
                        "SELECT * FROM (VALUES ('second', 1.5, 'baseline', 'zh'), "
                        "('second', 1.4, 'baseline', 'en')) t(model, missing, engine, corpus)"
                    ),
                    "variables": {"y": "missing"},
                    "kws": {"marker": "x"},
                },
            ],
        },
        style=_NAMED,
    )
    assert len(canvas.axes) == 2
    for axis, expected in zip(canvas.axes, (1.4, 1.5), strict=True):
        marks = axis.collections[-1]
        assert marks.get_offsets().tolist() == [[1.0, expected]]
        assert rendering.mpl.colors.to_hex(marks.get_facecolors()[0]) == "#aa2222"
    assert len(canvas.legends) == 1


def test_native_dodge_gap_shrinks_caps_without_moving_interval_centers(tmp_path: Path) -> None:
    canvas = drawn(
        tmp_path,
        {
            "sql": (
                "SELECT * FROM (VALUES ('one', 2., 1., 3., 'a'), "
                "('one', 4., 3., 5., 'b')) t(category, y, low, high, engine)"
            ),
            "variables": {"x": "category", "y": "y", "color": "engine"},
            "layers": [
                {"mark": "Bar", "kws": {"width": 0.8}, "moves": {"Dodge": {}}},
                _RANGE | {"moves": {"Dodge": {}}},
                {
                    "mark": "Dash",
                    "variables": {"y": "high"},
                    "kws": {"width": 0.8},
                    "moves": {"Dodge": {"gap": 0.6}},
                },
            ],
        },
        style=PlotStyle(colors={"a": "#7755aa", "b": "#aa2222"}),
    )
    axis = canvas.axes[0]
    centers = [bar.get_x() + bar.get_width() / 2 for bar in axis.patches]
    intervals, caps = [collection.get_segments() for collection in axis.collections]
    assert [segment[:, 0].mean() for segment in intervals] == pytest.approx(centers)
    assert [segment[:, 0].mean() for segment in caps] == pytest.approx(centers)
    assert all(
        segment[:, 0].max() - segment[:, 0].min() < axis.patches[0].get_width() for segment in caps
    )


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


@pytest.mark.parametrize(
    ("sql", "variables", "layers", "message"),
    [
        (_BOUNDED, _XY, [{"mark": "Range"}], "explicit bounds"),
        (_BOUNDED, _XY, [{"mark": "Range", "variables": {"ymin": "low"}}], "both lower"),
        (
            _BOUNDED,
            _XY,
            [{"mark": "Range", "variables": {"ymin": "high", "ymax": "low"}}],
            "exceeds",
        ),
        ("SELECT 1 x, 2.0 y, 'a' low, 3.0 high", _XY, [_RANGE], "interval bounds must be numeric"),
        ("SELECT 1 x, 5.0 y, 1.0 low, 3.0 high", _XY, [_RANGE], "contain the plotted value"),
        ("SELECT * FROM (VALUES (1, 2.), (1, 3.)) t(x, y)", _XY, [{"mark": "Bar"}], "bar groups"),
        ("SELECT 1 x, 2 y, 3.0 v", _CELLS, [{"mark": "Heatmap"}, {"mark": "Dot"}], "one layer"),
        ("SELECT 1 x, 2 y", _XY, [{"mark": "Heatmap"}], "x, y and color"),
        ("SELECT 'a' x, 2 y, 3.0 v", _CELLS, [{"mark": "Heatmap"}], "columns must be numeric"),
        (
            "SELECT * FROM (VALUES (1, 2., 3.), (1, 2., 4.)) t(x, y, v)",
            _CELLS,
            [{"mark": "Heatmap"}],
            "cells repeat",
        ),
        (
            "SELECT 1 x, 2.0 y, 'unnamed' engine",
            _ENGINE,
            [{"mark": "Dot"}],
            r"no explicit colors for \['unnamed'\]",
        ),
    ],
    ids=[
        "estimated_range",
        "half_range",
        "reversed_range",
        "text_bounds",
        "bounds_miss_the_value",
        "repeated_bar",
        "layered_heatmap",
        "colorless_heatmap",
        "categorical_heatmap",
        "repeated_cell",
        "unnamed_color",
    ],
)
def test_a_panel_refuses_data_it_would_have_to_guess_about(
    tmp_path: Path,
    sql: str,
    variables: dict[str, str],
    layers: list[dict[str, JsonValue]],
    message: str,
) -> None:
    """Nothing is aggregated, reordered or recolored to make the data drawable."""
    with pytest.raises(ValueError, match=message):
        drawn(tmp_path, {"sql": sql, "variables": variables, "layers": layers}, style=_NAMED)
    assert not list(tmp_path.glob("*.png"))


def test_axis_settings_dress_every_axes_the_panel_draws(tmp_path: Path) -> None:
    """Explicit ticks leave no minor labels between them and slanted labels stay on their tick."""
    canvas = drawn(
        tmp_path,
        {
            "sql": _RATES,
            "variables": _BY_MODEL,
            "layers": [{"mark": "Dot"}],
            "axis": {"xticks": [0, 1], "yticks": [2, 4]},
            "ticks": {"labelrotation": 45},
            "grid": {"visible": True},
            "label": {"y": "Rate"},
        },
        style=PlotStyle(labels={"first": "First model"}),
    )
    [axis] = canvas.axes
    assert isinstance(axis.xaxis.get_minor_formatter(), NullFormatter)
    assert isinstance(axis.yaxis.get_minor_formatter(), NullFormatter)
    labels = axis.get_xticklabels()
    assert [label.get_text() for label in labels] == ["First model", "second"]
    assert {(label.get_ha(), label.get_rotation_mode()) for label in labels} == {
        ("right", "anchor")
    }
    assert axis.xaxis.get_gridlines()[0].get_visible()
    assert axis.get_ylabel() == "Rate"
    # Without named colors the palette colors the series and the native key is the shared one.
    assert texts(canvas.legends[0]) == ["ours", "baseline"]


def test_a_heatmap_with_many_values_keeps_the_colorbars_own_ticks(tmp_path: Path) -> None:
    """Past eight values a tick per value would crowd the bar, and no text asks for labels."""
    # Values off the locator's round numbers, so a tick at each value would be told apart.
    cells = [(x, y, 3 * x + y + 0.37) for x in range(3) for y in range(3)]
    rows = ", ".join(f"({x}, {y}, {v})" for x, y, v in cells)
    canvas = drawn(
        tmp_path,
        {
            "sql": f"SELECT * FROM (VALUES {rows}) t(x, y, v)",
            "variables": _CELLS,
            "layers": [{"mark": "Heatmap"}],
        },
    )
    grid, bar = canvas.axes
    assert not grid.texts
    assert not {v for *_, v in cells} & set(bar.get_yticks().tolist())


def test_a_panel_can_drop_its_key_entirely(tmp_path: Path) -> None:
    canvas = drawn(
        tmp_path,
        {"sql": _RATES, "variables": _BY_MODEL, "layers": [{"mark": "Dot"}], "legend": False},
        style=_NAMED,
    )
    assert canvas.legends == []
    assert canvas.axes[0].get_legend() is None


@pytest.mark.parametrize(
    ("on_panel", "local", "shared"),
    [
        (False, ["Our engine", "baseline", "cold", "warm"], []),
        (True, ["cold", "warm"], ["Our engine", "baseline"]),
    ],
    ids=["one_layer_marks", "panel_marks"],
)
def test_a_mark_key_joins_the_colors_only_when_one_layer_alone_carries_it(
    tmp_path: Path, on_panel: bool, local: list[str], shared: list[str]
) -> None:
    """Marks the whole panel carries keep their own key beside the figure's shared colors."""
    marker = {"marker": "batch"}
    canvas = drawn(
        tmp_path,
        {
            "sql": _RATES,
            "variables": _BY_MODEL | (marker if on_panel else {}),
            "layers": [{"mark": "Dot"}, {"mark": "Dot", "variables": {} if on_panel else marker}],
        },
        style=_NAMED,
    )
    assert texts(canvas.axes[0].get_legend()) == local
    assert texts(canvas.legends[0] if canvas.legends else None) == shared


def test_a_panel_key_follows_the_panel_color_order_past_a_colorless_reference(
    tmp_path: Path,
) -> None:
    """A reference layer without a color neither needs a named color nor enters the key."""
    canvas = drawn(
        tmp_path,
        {
            "sql": _RATES,
            "variables": {"x": "model", "y": "rate"},
            "layers": [{"mark": "Dot", "variables": {"color": "engine"}}, {"mark": "Line"}],
            "order": {"color": ["baseline", "ours"]},
            "legend": {"loc": "upper left"},
        },
        style=_NAMED,
    )
    assert texts(canvas.axes[0].get_legend()) == ["baseline", "Our engine"]
    assert canvas.legends == []

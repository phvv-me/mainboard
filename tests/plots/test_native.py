"""Bespoke figures resolve the same native style as declarative Mainboard plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from mainboard.plots.marks import bars
from mainboard.plots.native import (
    apply_style,
    contrasting_text,
    figure_legend,
    load_style,
    save_figure,
    subplots,
)


def test_project_style_overlays_shared_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mainboard.toml").write_text(
        '[workspace]\nname = "test"\n[plots.paper]\n'
        'palette = ["#123456"]\n[plots.paper.colors]\nink = "#111111"\n'
        '[plots.paper.rc]\n"axes.labelsize" = 8\n'
    )
    (tmp_path / "plots.toml").write_text(
        '[workspace]\nname = "project"\n[plots.paper.colors]\naccent = "#654321"\n'
        '[plots.paper.rc]\n"axes.labelsize" = 11\n'
    )
    monkeypatch.chdir(tmp_path)
    style = load_style("plots.toml")
    assert style.palette == ["#123456"]
    assert style.colors == {"ink": "#111111", "accent": "#654321"}
    with plt.rc_context():
        apply_style(style)
        assert plt.rcParams["axes.labelsize"] == 11


def test_native_figure_bundle_preserves_layout(tmp_path: Path) -> None:
    canvas, axis = subplots(3.25, ratio=1.5)
    try:
        drawn = bars(axis, [0, 1], [1, 2], thickness=0.6, color="#745399", label="measured")
        figure_legend(canvas, axis)
        paths = save_figure(canvas, tmp_path / "figure.pdf", tmp_path / "figure.png")
        # A GUI backend snaps the canvas to whole pixels, so the size holds to one pixel.
        pixel = 1 / canvas.dpi
        assert tuple(canvas.get_size_inches()) == pytest.approx((3.25, 3.25 / 1.5), abs=pixel)
        assert all(path.stat().st_size > 0 for path in paths)
        assert [bar.position for bar in drawn] == [0, 1]
    finally:
        plt.close(canvas)


@pytest.mark.parametrize("fill, expected", [("#000000", "#ffffff"), ("#ffffff", "#222222")])
def test_text_contrast_uses_configured_roles(fill: str, expected: str) -> None:
    assert contrasting_text(fill, ink="#222222", surface="#ffffff") == expected

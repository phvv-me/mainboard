import json
import struct
from collections.abc import Sequence
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.apple import Apple, Preview
from mainboard.cli import build
from mainboard.core import Runner, ToolOutput


class FontTool(Runner):
    def __init__(self, fault: str = "") -> None:
        self.fault = fault
        self.calls: list[tuple[str, ...]] = []

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == "/usr/bin/xcrun" and cwd is None
        assert args[0] == "swift" and Path(args[1]).name == "fonts.swift"
        self.calls.append(tuple(args))
        request = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        assert request["text"] == "日本語 ' $(literal)"
        if self.fault == "failure":
            return ToolOutput(returncode=1, stderr="Font lacks preview glyphs")
        if self.fault == "truncated":
            return ToolOutput(stdout="{}", truncated=True)
        if self.fault == "malformed":
            return ToolOutput(stdout="{}")
        folder = Path(request["output"])
        folder.mkdir()
        images = []
        for source in request["fonts"]:
            image = folder / (Path(source).stem + ".png")
            image.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", 13)
                + b"IHDR"
                + struct.pack(">II", 960, 225)
            )
            images.append(
                {
                    "source": source,
                    "image": str(image),
                    "font": "Example",
                    "width": 960,
                    "height": 225,
                }
            )
        if self.fault == "duplicate":
            images.append(images[0])
        elif self.fault == "source":
            images[0]["source"] = str(folder / "unknown.ttf")
        elif self.fault == "path":
            images[0]["image"] = str(folder / "unknown.png")
        elif self.fault == "missing":
            Path(str(images[0]["image"])).unlink()
        elif self.fault == "symlink":
            image = Path(str(images[0]["image"]))
            target = folder / "other.png"
            image.rename(target)
            image.symlink_to(target)
        elif self.fault == "dimensions":
            images[0]["width"] = 1
        elif self.fault == "bytes":
            Path(str(images[0]["image"])).write_bytes(b"not a PNG")
        return ToolOutput(stdout=json.dumps({"images": images}))


@pytest.fixture
def font(tmp_path: Path) -> Path:
    path = tmp_path / "日本 ' $(literal).ttf"
    path.touch()
    return path


def test_font_render_request_and_report_preserve_literal_paths(font: Path, tmp_path: Path) -> None:
    tool = FontTool()
    report = Apple(tool).font_previews(
        (font, font.parent), Preview(text="日本語 ' $(literal)"), tmp_path / "output"
    )
    assert len(tool.calls) == len(report) == 1
    assert report[0].document() == {
        "source": str(font),
        "image": str(tmp_path / "output" / (font.stem + ".png")),
        "font": "Example",
        "width": 960,
        "height": 225,
    }


@pytest.mark.parametrize(
    "fault",
    [
        "failure",
        "truncated",
        "malformed",
        "duplicate",
        "source",
        "path",
        "missing",
        "symlink",
        "dimensions",
        "bytes",
    ],
)
def test_font_preview_rejects_failed_or_inconsistent_output(
    font: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    with pytest.raises(MissionError):
        Apple(FontTool(fault)).font_previews(
            (font,), Preview(text="日本語 ' $(literal)"), tmp_path / "output"
        )


@pytest.mark.parametrize("text", ["", " \n", "a" * 8193], ids=["empty", "blank", "long"])
def test_font_preview_invalid_text_never_starts_a_process(
    font: Path, tmp_path: Path, text: str
) -> None:
    tool = FontTool()
    with pytest.raises(MissionError):
        Apple(tool).font_previews((font,), Preview(text=text), tmp_path / "output")
    assert not tool.calls


@pytest.mark.parametrize("dimension", [0, -1, float("nan"), float("inf")])
def test_font_preview_invalid_dimensions_never_start_a_process(
    font: Path,
    tmp_path: Path,
    dimension: float,
) -> None:
    tool = FontTool()
    for name in ("width", "size", "scale"):
        with pytest.raises(MissionError):
            Apple(tool).font_previews(
                (font,), Preview(text="example", **{name: dimension}), tmp_path / "output"
            )
    assert not tool.calls


def test_font_preview_missing_empty_colliding_and_existing_paths_fail_before_execution(
    font: Path,
    tmp_path: Path,
) -> None:
    tool = FontTool()
    empty = tmp_path / "empty"
    empty.mkdir()
    collision = font.with_suffix(".otf")
    collision.touch()
    for sources, output in [
        ((tmp_path / "missing",), empty / "output"),
        ((empty,), empty / "output"),
        ((font, collision), empty / "output"),
        ((font,), empty),
    ]:
        with pytest.raises(MissionError):
            Apple(tool).font_previews(sources, Preview(text="example"), output)
    assert not tool.calls


def test_font_preview_cli_uses_fresh_output_and_preserves_verdict(
    font: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("mainboard.verbs.apps.apple.Apple", lambda: Apple(FontTool()))
    for _ in range(2):
        with pytest.raises(SystemExit) as stopped:
            build()(
                [
                    "apple",
                    "font-previews",
                    str(font),
                    "--text",
                    "日本語 ' $(literal)",
                    "--results",
                    str(tmp_path),
                    "--json",
                ]
            )
        assert stopped.value.code == 0
        assert len(json.loads(capsys.readouterr().out)) == 1
    assert len(list(tmp_path.glob("fonts-*"))) == 2

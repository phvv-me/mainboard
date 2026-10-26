import json
import math
import string
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.apple import Apple, Preview
from mainboard.core import Runner, ToolOutput

from .constants import XCRUN

_LITERAL = "日本語 ' $(literal)"
_LONGEST = 8192
_PNG = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 960, 225)

type Images = list[dict[str, str | int]]
type Corruption = Callable[[Images, Path], None]


def _intact(images: Images, folder: Path) -> None:
    """Leave the report as the helper made it."""


def _reported_twice(images: Images, folder: Path) -> None:
    images.append(images[0])


def _unknown_source(images: Images, folder: Path) -> None:
    images[0]["source"] = str(folder / "unknown.ttf")


def _unknown_image(images: Images, folder: Path) -> None:
    images[0]["image"] = str(folder / "unknown.png")


def _missing_image(images: Images, folder: Path) -> None:
    Path(str(images[0]["image"])).unlink()


def _linked_image(images: Images, folder: Path) -> None:
    image = Path(str(images[0]["image"]))
    image.rename(folder / "other.png")
    image.symlink_to(folder / "other.png")


def _wrong_dimensions(images: Images, folder: Path) -> None:
    images[0]["width"] = 1


def _not_a_png(images: Images, folder: Path) -> None:
    Path(str(images[0]["image"])).write_bytes(b"not a PNG")


class FontTool(Runner):
    """Renders a PNG per requested font and reports it, as the Swift helper does.

    corrupt: changes the report, or the images it names, before it is answered.
    reply: answers instead of rendering anything.
    """

    def __init__(self, *, corrupt: Corruption = _intact, reply: ToolOutput | None = None) -> None:
        self.corrupt = corrupt
        self.reply = reply
        self.calls: list[Sequence[str]] = []

    def _render(self, fonts: Sequence[str], output: Path) -> Images:
        """The report of rendering every font in `fonts` into the fresh folder `output`."""
        output.mkdir()
        images: Images = []
        for source in fonts:
            image = output / (Path(source).stem + ".png")
            image.write_bytes(_PNG)
            images.append(
                {
                    "source": source,
                    "image": str(image),
                    "font": "Example",
                    "width": 960,
                    "height": 225,
                }
            )
        self.corrupt(images, output)
        return images

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == XCRUN and cwd is None
        assert args[0] == "swift" and Path(args[1]).name == "fonts.swift"
        self.calls.append(args)
        request = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        assert request["text"] == _LITERAL
        if self.reply is not None:
            return self.reply
        images = self._render(request["fonts"], Path(request["output"]))
        return ToolOutput(stdout=json.dumps({"images": images}))


@pytest.fixture
def font(tmp_path: Path) -> Path:
    path = tmp_path / "日本 ' $(literal).ttf"
    path.touch()
    return path


@pytest.fixture
def paths(font: Path) -> Mapping[str, Path]:
    """Named paths: the font, a rival with its stem, an empty folder, and two absent ones."""
    empty = font.parent / "empty"
    empty.mkdir()
    rival = font.with_suffix(".otf")
    rival.touch()
    return {
        "font": font,
        "rival": rival,
        "empty": empty,
        "missing": font.parent / "missing",
        "output": font.parent / "output",
    }


def test_font_render_request_and_report_preserve_literal_paths(font: Path, tmp_path: Path) -> None:
    tool = FontTool()
    report = Apple(tool).font_previews(
        (font, font.parent), Preview(text=_LITERAL), tmp_path / "output"
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
    ("arrange", "match"),
    [
        pytest.param(
            {"reply": ToolOutput(returncode=1, stderr="Font lacks preview glyphs")},
            "Font lacks preview glyphs",
            id="the-helper-fails",
        ),
        pytest.param(
            {"reply": ToolOutput(stdout="{}", truncated=True)},
            "truncated=True",
            id="the-report-is-cut-short",
        ),
        pytest.param(
            {"reply": ToolOutput(stdout="{}")},
            "invalid native font preview report",
            id="the-report-lists-no-image",
        ),
        pytest.param({"corrupt": _reported_twice}, "exactly once", id="a-font-reported-twice"),
        pytest.param({"corrupt": _unknown_source}, "exactly once", id="an-unrequested-font"),
        pytest.param(
            {"corrupt": _unknown_image}, "unexpected or missing image", id="an-image-elsewhere"
        ),
        pytest.param(
            {"corrupt": _missing_image}, "unexpected or missing image", id="an-image-that-is-gone"
        ),
        pytest.param(
            {"corrupt": _linked_image}, "unexpected or missing image", id="an-image-that-is-a-link"
        ),
        pytest.param(
            {"corrupt": _wrong_dimensions},
            "invalid font preview image",
            id="dimensions-the-png-does-not-have",
        ),
        pytest.param(
            {"corrupt": _not_a_png}, "invalid font preview image", id="an-image-that-is-no-png"
        ),
    ],
)
def test_font_preview_rejects_failed_or_inconsistent_output(
    font: Path, tmp_path: Path, arrange, match: str
) -> None:
    with pytest.raises(MissionError, match=match):
        Apple(FontTool(**arrange)).font_previews(
            (font,), Preview(text=_LITERAL), tmp_path / "output"
        )


@pytest.mark.parametrize(
    ("sources", "output", "match"),
    [
        pytest.param(["missing"], "output", "font source does not exist", id="a-missing-source"),
        pytest.param(["empty"], "output", "distinct output stems", id="a-folder-without-fonts"),
        pytest.param(
            ["font", "rival"], "output", "distinct output stems", id="two-fonts-sharing-a-stem"
        ),
        pytest.param(["font"], "empty", "refusing to overwrite", id="an-existing-output-folder"),
    ],
)
def test_font_preview_unusable_sources_and_outputs_fail_before_execution(
    paths: Mapping[str, Path], sources: Sequence[str], output: str, match: str
) -> None:
    tool = FontTool()
    with pytest.raises(MissionError, match=match):
        Apple(tool).font_previews(
            [paths[name] for name in sources], Preview(text=_LITERAL), paths[output]
        )
    assert tool.calls == []


_SHOWN = st.text(min_size=1, max_size=40).filter(str.strip) | st.just("a" * _LONGEST)
_SHOWING = {"text": "a", "width": 1.0, "size": 1.0, "scale": 1.0}
_BLANK = st.text(alphabet=string.whitespace + chr(0x3000), max_size=20)
_TOO_LONG = st.integers(_LONGEST + 1, _LONGEST + 800).map("a".__mul__)
_DRAWABLE = st.floats(min_value=0.001, max_value=1e6)
_UNDRAWABLE = st.floats(max_value=0) | st.sampled_from([math.nan, math.inf])
_VALID = st.fixed_dictionaries(
    {"text": _SHOWN, "width": _DRAWABLE, "size": _DRAWABLE, "scale": _DRAWABLE}
)
_INVALID = st.one_of(
    st.tuples(st.just("text"), _BLANK | _TOO_LONG, st.just("nonblank text")),
    st.tuples(
        st.sampled_from(["width", "size", "scale"]), _UNDRAWABLE, st.just("positive and finite")
    ),
)


@settings(max_examples=200)
@given(fields=_VALID)
def test_a_preview_keeps_text_that_shows_and_sizes_that_can_be_drawn(fields) -> None:
    assert Preview(**fields).model_dump() == fields


@settings(max_examples=200)
@given(fields=_VALID, invalid=_INVALID)
@example(fields=_SHOWING, invalid=("text", "", "nonblank text"))
@example(fields=_SHOWING, invalid=("text", " \n", "nonblank text"))
@example(fields=_SHOWING, invalid=("text", "a" * (_LONGEST + 1), "nonblank text"))
@example(fields=_SHOWING, invalid=("width", 0.0, "positive and finite"))
@example(fields=_SHOWING, invalid=("size", -1.0, "positive and finite"))
@example(fields=_SHOWING, invalid=("scale", math.nan, "positive and finite"))
@example(fields=_SHOWING, invalid=("width", math.inf, "positive and finite"))
def test_a_preview_refuses_blank_or_oversized_text_and_undrawable_sizes(fields, invalid) -> None:
    name, value, reason = invalid
    with pytest.raises(MissionError, match=reason):
        Preview(**{**fields, name: value})


def test_the_font_previews_verb_renders_into_a_fresh_directory_on_every_run(
    cli, font: Path, tmp_path: Path
) -> None:
    apple = Apple(FontTool())
    argv = ["font-previews", str(font), "--text", _LITERAL, "--results", str(tmp_path), "--json"]
    runs = [cli(apple, *argv) for _ in range(2)]
    assert [code for code, _ in runs] == [0, 0]
    assert [len(json.loads(out)) for _, out in runs] == [1, 1]
    assert len(list(tmp_path.glob("fonts-*"))) == 2

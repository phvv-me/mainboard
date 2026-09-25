import codecs
import re
import tomllib
from typing import TYPE_CHECKING

import yaml
from patos import FrozenModel

if TYPE_CHECKING:
    from pathlib import Path

    from .inventory import Attributes

# The byte-order marks a Windows editor or a PowerShell redirect writes UTF-16 text behind. Such
# a file is full of NUL bytes, so the binary test below would otherwise take it for data.
_UTF16 = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)

# Git's own binary test: a NUL byte anywhere in the first 8000 bytes.
_SNIFF = 8000

# Both halves of an unresolved merge, so a lone `=======` underline in a document never counts.
_CONFLICT = (re.compile(r"^<{7}(?: |$)", re.MULTILINE), re.compile(r"^>{7}(?: |$)", re.MULTILINE))

_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY( BLOCK)?-----|PuTTY-User-Key-File-")


def decoded(data: bytes) -> str | None:
    """`data` read as text, or None for a binary file no text step may touch.

    UTF-16 behind its byte-order mark and UTF-8 with or without one all decode, and anything
    else that is not binary raises `UnicodeDecodeError`, since guessing a legacy code page
    would rewrite the file in a meaning nobody chose.
    """
    if data.startswith(_UTF16):
        return data.decode("utf-16")
    if b"\0" in data[:_SNIFF]:
        return None
    return data.decode("utf-8-sig")


def normalized(text: str, newline: str = "\n") -> str:
    """`text` with one newline style, no trailing blanks, and exactly one final newline.

    Every CRLF and lone CR becomes `newline`, spaces and tabs at the end of a line go, and
    trailing blank lines collapse, so a file of nothing but whitespace becomes empty.

    newline: the line ending the file is stored in.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    body = "\n".join(line.rstrip(" \t") for line in lines).rstrip("\n")
    return f"{body}\n".replace("\n", newline) if body else ""


def problems(name: str, text: str) -> list[str]:
    """What only a person can fix in `text`.

    name: the file's name, whose suffix selects the syntax check.
    text: the file's normalized content.
    """
    found: list[str] = []
    if all(marker.search(text) for marker in _CONFLICT):
        found.append("carries unresolved merge conflict markers")
    if _PRIVATE_KEY.search(text):
        found.append("carries a private key")
    if error := _syntax(name, text):
        found.append(error)
    return found


class Examination(FrozenModel):
    """What the text hygiene makes of one file.

    repaired: the bytes the file should hold, None when it holds them already or no text step
        may touch it.
    untidy: what the repair changes, `trailing whitespace` say, empty exactly when `repaired`
        is None.
    problems: what only a person can fix.
    """

    repaired: bytes | None = None
    untidy: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()


def examined(path: Path, attributes: Attributes) -> Examination:
    """Read `path` and say what normalizing it changes and what is left to fix, writing nothing.

    path: an existing regular file.
    """
    if attributes.binary or path.is_symlink():
        return Examination()
    data = path.read_bytes()
    try:
        text = decoded(data)
    except UnicodeDecodeError:
        return Examination(problems=("is not UTF-8 text; save it as UTF-8",))
    if text is None:
        return Examination()
    fixed = normalized(text, attributes.newline)
    untidy = untidiness(data, text, attributes.newline)
    return Examination(
        repaired=fixed.encode() if untidy else None,
        untidy=untidy,
        problems=tuple(problems(path.name, fixed)),
    )


def untidiness(data: bytes, text: str, newline: str = "\n") -> tuple[str, ...]:
    """What `normalized` changes in `text`, each a few words, empty when `data` is already it.

    data: the file's bytes.
    text: `data` decoded.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    trimmed = "\n".join(line.rstrip(" \t") for line in unified.split("\n"))
    found = {
        "an encoding other than plain UTF-8": text.encode() != data,
        "line endings": unified.replace("\n", newline) != text,
        "trailing whitespace": trimmed != unified,
        "blank lines or a missing newline at the end": normalized(trimmed) != trimmed,
    }
    return tuple(change for change, present in found.items() if present)


def _syntax(name: str, text: str) -> str:
    """Why `text` does not parse as the format its suffix names, empty when it does."""
    try:
        if name.endswith(".toml"):
            tomllib.loads(text)
        elif name.endswith((".yaml", ".yml")):
            # Events rather than objects: syntax is the question, and a custom tag such as
            # `!reference` needs no constructor to be well formed.
            list(yaml.parse(text, Loader=yaml.SafeLoader))
    except (tomllib.TOMLDecodeError, yaml.YAMLError) as error:
        return f"does not parse: {' '.join(str(error).split())}"
    return ""

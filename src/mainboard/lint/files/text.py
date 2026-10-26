import codecs
import os
import re
import tomllib
import unicodedata
from typing import TYPE_CHECKING

import yaml
from patos import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .inventory import Attributes

# The byte-order marks UTF-16 text starts with. Such a file is full of NUL bytes, so the binary
# test below would otherwise take it for data.
_UTF16 = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)

# Git's own binary test: a NUL byte anywhere in the first 8000 bytes.
_SNIFF = 8000

# Both halves of an unresolved merge, so a lone `=======` underline in a document never counts.
_CONFLICT = (re.compile(r"^<{7}(?: |$)", re.MULTILINE), re.compile(r"^>{7}(?: |$)", re.MULTILINE))

# A PEM header, or a PuTTY key file's first line. The PuTTY header is written in two pieces and
# anchored to a line start, so neither this file nor a document that quotes the header mid-line
# carries a key.
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY( BLOCK)?-----|^PuTTY-User-" "Key-File-",
    re.MULTILINE,
)


def _decoded(data: bytes) -> str | None:
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


def _normalized(text: str, *, newline: str = "\n") -> str:
    """`text` with one newline style, no trailing blanks, and exactly one final newline.

    Every CRLF and lone CR becomes `newline`, spaces and tabs at the end of a line go, and
    trailing blank lines collapse, so a file of nothing but whitespace becomes empty.

    newline: the line ending the file is stored in.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    body = "\n".join(line.rstrip(" \t") for line in lines).rstrip("\n")
    return f"{body}\n".replace("\n", newline) if body else ""


def _problems(name: str, *, text: str) -> list[str]:
    """What only a person can fix in `text`.

    name: the file's name, whose suffix selects the syntax check.
    text: the file's normalized content.
    """
    found: list[str] = []
    if all(marker.search(text) for marker in _CONFLICT):
        found.append("carries unresolved merge conflict markers")
    if _PRIVATE_KEY.search(text):
        found.append("carries a private key")
    if error := _syntax(name, text=text):
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

    path: an existing file, or a symlink, whose target is all that is asked of it.
    """
    if path.is_symlink():
        return Examination(
            problems=() if path.exists() else (f"is a symlink to nowhere ({os.readlink(path)})",)
        )
    if attributes.binary:
        return Examination()
    data = path.read_bytes()
    try:
        text = _decoded(data)
    except UnicodeDecodeError:
        return Examination(problems=("is not UTF-8 text; save it as UTF-8",))
    if text is None:
        return Examination()
    fixed = _normalized(text, newline=attributes.newline)
    untidy = _untidiness(data, text, newline=attributes.newline)
    return Examination(
        repaired=fixed.encode() if untidy else None,
        untidy=untidy,
        problems=tuple(_problems(path.name, text=fixed)),
    )


def _untidiness(data: bytes, text: str, *, newline: str = "\n") -> tuple[str, ...]:
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
        "blank lines or a missing newline at the end": _normalized(trimmed) != trimmed,
    }
    return tuple(change for change, present in found.items() if present)


def twins(name: str, neighbors: Iterable[str]) -> list[str]:
    """The `neighbors` that differ from `name` only in Unicode form and case.

    A case-insensitive or normalizing disk cannot hold them side by side.

    name: a file name.
    neighbors: the names beside it in its directory.
    """
    folded = _folded(name)
    return sorted(other for other in set(neighbors) if other != name and _folded(other) == folded)


def _folded(name: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())


def _parse(name: str, *, text: str) -> None:
    """Parse `text` as the format its suffix names, doing nothing for a suffix with none."""
    if name.endswith(".toml"):
        tomllib.loads(text)
    elif name.endswith((".yaml", ".yml")):
        # Events rather than objects: syntax is the question, and a custom tag such as
        # `!reference` needs no constructor to be well formed.
        list(yaml.parse(text, Loader=yaml.SafeLoader))


def _syntax(name: str, *, text: str) -> str:
    """Why `text` does not parse as the format its suffix names, empty when it does."""
    try:
        _parse(name, text=text)
    except (tomllib.TOMLDecodeError, yaml.YAMLError) as error:
        return f"does not parse: {' '.join(str(error).split())}"
    return ""

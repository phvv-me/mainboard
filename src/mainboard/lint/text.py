import codecs
import re
import tomllib
from typing import TYPE_CHECKING

import yaml

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
    """What only a person can fix in `text`, each line naming the file it is in.

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


def repair(path: Path, attributes: Attributes) -> list[str]:
    """Rewrite `path` into normalized UTF-8 where needed and return what is left to fix.

    path: an existing regular file.
    attributes: what `.gitattributes` says about it.
    """
    if attributes.binary or path.is_symlink():
        return []
    data = path.read_bytes()
    try:
        text = decoded(data)
    except UnicodeDecodeError:
        return ["is not UTF-8 text; save it as UTF-8"]
    if text is None:
        return []
    fixed = normalized(text, attributes.newline)
    if fixed.encode() != data:
        path.write_bytes(fixed.encode())
    return problems(path.name, fixed)


def _syntax(name: str, text: str) -> str:
    """Why `text` does not parse as the format its suffix names, empty when it does."""
    try:
        if name.endswith(".toml"):
            tomllib.loads(text)
        elif name.endswith((".yaml", ".yml")):
            # Events rather than objects: syntax is the question, and a custom tag such as
            # `!reference` needs no constructor to be well formed.
            for _ in yaml.parse(text, Loader=yaml.SafeLoader):
                pass
    except (tomllib.TOMLDecodeError, yaml.YAMLError) as error:
        return f"does not parse: {' '.join(str(error).split())}"
    return ""

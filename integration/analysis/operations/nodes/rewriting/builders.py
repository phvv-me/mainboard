import json
from collections.abc import Callable, Sequence

from mainboard.analysis.operations.nodes import Node, Rewrite, Site

_SUPPRESS = ("from contextlib import suppress",)
_SUPPRESSED = "with suppress({}):\n    $$$BODY"
_TUPLE_HANDLER = "try:\n    $$$BODY\nexcept ($$$E,):\n    $INERT"
_ONE_HANDLER = "try:\n    $$$BODY\nexcept $E:\n    $INERT"
_BARE_HANDLER = "try:\n    $$$BODY\nexcept:\n    $INERT"
_INERT = {"INERT": ("pass", "...")}
_HEAD = "import shutil\n\n\ndef clean(path):\n"
_CASTS = """\
import typing
from typing import cast


def plain(x):
    return cast(int, x["a"])


def scaled(a, b):
    return typing.cast(int, a + b) * 2


def attribute(a, b):
    return cast(int, a or b).bit_length()


def negated(a, b):
    return -cast(int, a + b)


def raw(x):
    return cast(
        int,
        x,  # the raw value
    )


def spread(xs):
    return cast(int, *xs)


def tall(x):
    return cast(
        int,
        x.method(
            1,
            2,
        ),
    )


def pick(a, b):
    return cast(
        int,
        a
        if b
        else 0,
    )
"""
_BOUNDED = """\
def load(path):
    try:
        text = path.read_text()
        return parse(text)
    except OSError:
        return None
"""
_NOTED_TAIL = """\
def load(path):
    try:
        text = path.read_text()
        return parse(text)  # parsed
    except OSError:
        return None
"""
_GLOBS = """\
from pathlib import Path

SOURCES = Path(__file__).parent.glob("*.py")
LISTED = Path(__file__).parent.glob(
    "*.py",  # the sources
)
"""


def swallows() -> tuple[Rewrite, ...]:
    """Rewrites turning a try that swallows an error into `suppress`, one per handler shape."""
    return (
        Rewrite(
            match=_TUPLE_HANDLER,
            to=_SUPPRESSED.format("$$$E"),
            imports=_SUPPRESS,
            where=_INERT,
        ),
        Rewrite(
            match=_ONE_HANDLER,
            to=_SUPPRESSED.format("$E"),
            imports=_SUPPRESS,
            where=_INERT,
        ),
        Rewrite(
            match=_BARE_HANDLER,
            to=_SUPPRESSED.format("BaseException"),
            imports=_SUPPRESS,
            where=_INERT,
        ),
    )


def sources() -> dict[str, str]:
    """The files the rewrite tests read, by project path: one handler shape or comment each."""
    return {
        "swallow.py": _HEAD
        + "    try:\n        shutil.rmtree(path)\n    except OSError:\n        pass\n",
        "tuple.py": _HEAD
        + "    try:\n        shutil.rmtree(path)\n        path.unlink()\n"
        + "    except (FileNotFoundError, PermissionError):\n        ...\n",
        "bare.py": _HEAD + "    try:\n        shutil.rmtree(path)\n    except:\n        pass\n",
        "busy.py": _HEAD
        + "    try:\n        shutil.rmtree(path)\n    except OSError:\n        print(path)\n",
        "orelse.py": _HEAD
        + "    try:\n        shutil.rmtree(path)\n    except OSError:\n        pass\n"
        + "    else:\n        print(path)\n",
        "handler.py": _HEAD
        + "    try:\n        shutil.rmtree(path)\n    except OSError:  # gone\n        pass\n",
        "leading.py": _HEAD
        + "    try:\n        # note\n        shutil.rmtree(path)\n"
        + "    except OSError:\n        pass\n",
        "closing.py": _HEAD
        + "    try:\n        shutil.rmtree(path)\n        # after\n"
        + "    except OSError:\n        pass\n",
        "between.py": _HEAD
        + "    try:\n        a = 1  # one\n        # two\n        b = 2  # three\n"
        + "    except OSError:\n        pass\n",
        "multibyte.py": _HEAD
        + '    try:\n        print("é日本")  # 日本\n    except OSError:\n        pass\n',
        "string.py": _HEAD
        + '    try:\n        print("""a\nb""")\n    except OSError:\n        pass\n',
        "crlf.py": _HEAD.replace("\n", "\r\n")
        + "    try:\r\n        shutil.rmtree(path)\r\n        path.unlink()\r\n"
        + "    except OSError:\r\n        pass\r\n",
        "bounded.py": _BOUNDED,
        "noted_tail.py": _NOTED_TAIL,
        "casts.py": _CASTS,
        "globs.py": _GLOBS,
    }


def declared(site: Site, rewrites: Sequence[Rewrite], parse: Callable[[str], list[Node]]) -> dict:
    """The `rewrite-node` fix row of declared rewrites, as the rule's SQL states it.

    That is the node's site, the rewrites as JSON and the parse of each pattern.
    """
    return {
        "site": site,
        "rewrites": json.dumps([rewrite.model_dump() for rewrite in rewrites]),
        "patterns": [parse(rewrite.expando()) for rewrite in rewrites],
    }

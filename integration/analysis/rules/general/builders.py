_NESTED = """\
import requests


def f(u):
    return requests.get(requests.get(u).text)
"""
_NESTED_FIXED = """\
import requests


def f(u):
    return requests.get(requests.get(u, timeout=2.5).text, timeout=2.5)
"""
_SPREAD_OUT = """\
import requests


def f(u, h):
    return requests.get(
        u,
        headers=h,
    )
"""
_SPREAD_OUT_FIXED = """\
import requests


def f(u, h):
    return requests.get(u, headers=h, timeout=2.5)
"""
_TUPLE = """\
import requests


def f(a, b):
    return requests.get((a, b))
"""
_DECORATOR = """\
import requests


@requests.get(1)
def f(): ...
"""
_ANNOTATED = """\
import requests


def f(u, h):
    return requests.get(
        u,  # the url
        h,
    )
"""
_ANNOTATED_FIXED = """\
import requests


def f(u, h):
    return requests.get(
        u,  # the url
        h,
        timeout=2.5,
    )
"""
_SPACED = """\
import requests


def f(u):
    return requests.get( u )
"""
_GENERATOR = """\
import subprocess


def f(xs):
    return subprocess.run(x for x in xs)
"""
_STRING = '''\
import requests


def f():
    return requests.get(
        """a
b""",
        verify=False,
    )
'''
_SPLAT = _NESTED.replace("requests.get(requests.get(u).text)", "requests.get(u, **options)")
_STARRED = _NESTED.replace("requests.get(requests.get(u).text)", "requests.get(*u)")
_CLOSING = """\
import requests


def f(u):
    return requests.get(
        u,  # the url
    )
"""


def cases() -> dict[str, tuple[str, str]]:
    """The files the ALL-CALL0002 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "nested.py": (_NESTED, _NESTED_FIXED),
        "carriage_returns.py": (
            _SPREAD_OUT.replace("\n", "\r\n"),
            _SPREAD_OUT_FIXED.replace("\n", "\r\n"),
        ),
        "tuple.py": (_TUPLE, _TUPLE.replace("(a, b))", "(a, b), timeout=2.5)")),
        "decorator.py": (_DECORATOR, _DECORATOR.replace("get(1)", "get(1, timeout=2.5)")),
        "annotated.py": (_ANNOTATED, _ANNOTATED_FIXED),
        "spaced.py": (_SPACED, _SPACED.replace("( u )", "(u, timeout=2.5)")),
        "generator.py": (_GENERATOR, _GENERATOR),
        "multiline.py": (_STRING, _STRING),
        "splat.py": (_SPLAT, _SPLAT),
        "starred.py": (_STARRED, _STARRED),
        "closing.py": (_CLOSING, _CLOSING),
    }

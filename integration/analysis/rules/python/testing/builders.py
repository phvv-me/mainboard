from collections.abc import Sequence

_HEAD = """\
import asyncio


async def fetch(n=0):
    await asyncio.sleep(0)
    return n


"""
_MARKED_HEAD = """\
import asyncio

import pytest


async def fetch(n=0):
    await asyncio.sleep(0)
    return n


"""
_ATTRIBUTE = (
    _HEAD
    + """\
def test_attribute():
    assert asyncio.run(fetch()).real
"""
)
_ATTRIBUTE_FIXED = (
    _MARKED_HEAD
    + """\
@pytest.mark.anyio
async def test_attribute():
    assert (await fetch()).real
"""
)
_SUBSCRIPT = (
    _HEAD
    + """\
def test_subscript():
    assert asyncio.run(fetch([1]))[0]
"""
)
_SUBSCRIPT_FIXED = (
    _MARKED_HEAD
    + """\
@pytest.mark.anyio
async def test_subscript():
    assert (await fetch([1]))[0]
"""
)
_COMPREHENSION = (
    _HEAD
    + """\
def test_comprehension():
    assert [asyncio.run(fetch(i)) for i in range(3)]
"""
)
_COMPREHENSION_FIXED = (
    _MARKED_HEAD
    + """\
@pytest.mark.anyio
async def test_comprehension():
    assert [await fetch(i) for i in range(3)]
"""
)
_TWO = (
    _HEAD
    + """\
def test_one():
    asyncio.run(fetch(1))


def test_two():
    asyncio.run(fetch(2))
    asyncio.run(fetch(3))
"""
)
_TWO_FIXED = (
    _MARKED_HEAD
    + """\
@pytest.mark.anyio
async def test_one():
    await fetch(1)


@pytest.mark.anyio
async def test_two():
    await fetch(2)
    await fetch(3)
"""
)
_DECORATED = (
    _MARKED_HEAD
    + """\
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.slow
def test_decorated(n):
    assert asyncio.run(fetch(n)) == n
"""
)
_DECORATED_FIXED = (
    _MARKED_HEAD
    + """\
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.slow
@pytest.mark.anyio
async def test_decorated(n):
    assert await fetch(n) == n
"""
)
_CLASSED = (
    _MARKED_HEAD
    + """\
class TestFetch:
    @pytest.mark.slow
    def test_method(self):
        asyncio.run(fetch())
"""
)
_CLASSED_FIXED = (
    _MARKED_HEAD
    + """\
class TestFetch:
    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_method(self):
        await fetch()
"""
)
_LAMBDA = (
    _HEAD
    + """\
def test_lambda():
    run = lambda: asyncio.run(fetch())
    assert run()
"""
)
_GENERATOR = (
    _HEAD
    + """\
def test_generator():
    assert sum(asyncio.run(fetch(i)) for i in range(3))
"""
)
_ALIASED = """\
from asyncio import run


async def fetch():
    return 1


def test_aliased():
    run(fetch())
"""
_MARKED = (
    _MARKED_HEAD
    + """\
@pytest.mark.anyio
def test_marked():
    asyncio.run(fetch())
"""
)
_GIVEN = """\
import asyncio

from hypothesis import given
from hypothesis import strategies as st


async def fetch(n):
    return n


@given(st.integers())
def test_given(n):
    asyncio.run(fetch(n))
"""
_TABS = _HEAD + "class TestTabs:\n\tdef test_tabs(self):\n\t\tasyncio.run(fetch())\n"
_NESTED = (
    _HEAD
    + """\
def test_runs():
    asyncio.run(fetch(1))


class TestWrapped:
    def test_nested(self):
        def inner():
            return asyncio.run(fetch())

        assert inner()
"""
)
_NESTED_FIXED = (
    _MARKED_HEAD
    + """\
@pytest.mark.anyio
async def test_runs():
    await fetch(1)


class TestWrapped:
    def test_nested(self):
        def inner():
            return asyncio.run(fetch())

        assert inner()
"""
)


def cases() -> dict[str, tuple[str, str]]:
    """The files the PY-TEST0009 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "test_attribute.py": (_ATTRIBUTE, _ATTRIBUTE_FIXED),
        "test_subscript.py": (_SUBSCRIPT, _SUBSCRIPT_FIXED),
        "test_comprehension.py": (_COMPREHENSION, _COMPREHENSION_FIXED),
        "test_carriage_returns.py": (
            _ATTRIBUTE.replace("\n", "\r\n"),
            _ATTRIBUTE_FIXED.replace("\n", "\r\n"),
        ),
        "test_two.py": (_TWO, _TWO_FIXED),
        "test_decorated.py": (_DECORATED, _DECORATED_FIXED),
        "test_method.py": (_CLASSED, _CLASSED_FIXED),
        "test_lambda.py": (_LAMBDA, _LAMBDA),
        "test_generator.py": (_GENERATOR, _GENERATOR),
        "test_aliased.py": (_ALIASED, _ALIASED),
        "test_marked.py": (_MARKED, _MARKED),
        "test_given.py": (_GIVEN, _GIVEN),
        "test_tabs.py": (_TABS, _TABS),
        "test_nested.py": (_NESTED, _NESTED_FIXED),
    }


def synchronous(forms: Sequence[str]) -> str:
    """A module of one test per form, each running its coroutine with `asyncio.run`."""
    tests = [
        f"def test_{place}():\n    {form.format(f'asyncio.run(fetch({place}))')}\n"
        for place, form in enumerate(forms)
    ]
    return _HEAD + "\n\n".join(tests)


def asynchronous(forms: Sequence[str]) -> str:
    """The module `synchronous` becomes: marked async tests awaiting their coroutines."""
    tests = [
        "@pytest.mark.anyio\n"
        f"async def test_{place}():\n    {form.format(f'await fetch({place})')}\n"
        for place, form in enumerate(forms)
    ]
    return _MARKED_HEAD + "\n\n".join(tests)

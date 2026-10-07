"""Automatic spans follow owned function bodies without importing shared wrappers."""

import sys
from collections import Counter
from types import CodeType, ModuleType

import pytest

from mainboard.profile import Profiler, annotate


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    foreign = ModuleType("_profile_foreign")
    exec(
        """
from mainboard.profile.spans import span

@span("manual_foreign")
def unrelated():
    return 7

class Foreign:
    def inherited(self):
        return 8
""",
        vars(foreign),
    )
    owned = ModuleType("_profile_owned")
    owned.foreign = foreign
    exec(
        """
from mainboard.profile.spans import span

foreign_alias = foreign.unrelated
foreign_class = foreign.Foreign
reexport = span("manual_reexport")(foreign.unrelated)
reexport.__module__ = __name__

def plain():
    return 1

alias = plain

@span("manual_first")
def first():
    return 2

@span("manual_second")
def second():
    return 3

@span
def unlabelled():
    return 9

class Worker(foreign.Foreign):
    borrowed = foreign.unrelated
    borrowed_static = staticmethod(foreign.unrelated)
    borrowed_class = classmethod(foreign.unrelated)

    def regular(self):
        return 4

    @staticmethod
    @span("manual_static")
    def static():
        return 5

    @classmethod
    @span("manual_classed")
    def classed(cls):
        return 6

def nested():
    def values():
        yield 1
        yield 2
    return sum((lambda value: value)(value) for value in values())
""",
        vars(owned),
    )
    monkeypatch.setitem(sys.modules, owned.__name__, owned)
    return owned, foreign


def test_owned_bodies_descriptors_and_foreign_wrappers(modules: tuple[ModuleType, ModuleType]):
    owned, foreign = modules
    assert owned.first.__code__ is owned.second.__code__ is foreign.unrelated.__code__
    selected = Profiler.owned_codes(owned)
    assert len(selected) == len(set(selected))
    assert {code.co_qualname for code in selected} == {
        "plain",
        "first",
        "second",
        "unlabelled",
        "Worker.regular",
        "Worker.static",
        "Worker.classed",
        "nested",
    }


def test_automatic_spans_preserve_explicit_labels(modules: tuple[ModuleType, ModuleType]):
    owned, foreign = modules
    with Profiler(features=Profiler.Feature.SPANS, auto=(owned.__name__,)) as profiler:
        assert owned.plain() == 1
        assert owned.first() == 2
        assert owned.second() == 3
        assert owned.Worker().regular() == 4
        assert owned.Worker.static() == 5
        assert owned.Worker.classed() == 6
        assert foreign.unrelated() == 7
        assert owned.Worker().inherited() == 8
        assert owned.unlabelled() == 9
    assert Counter(row.name for row in profiler.result().summaries) == Counter(
        {
            "plain": 1,
            "manual_first.first": 1,
            "manual_first": 1,
            "manual_second.second": 1,
            "manual_second": 1,
            "Worker.regular": 1,
            "manual_static.Worker.static": 1,
            "manual_static": 1,
            "manual_classed.Worker.classed": 1,
            "manual_classed": 1,
            "manual_foreign": 1,
            "unlabelled": 1,
            "unlabelled.unlabelled": 1,
        }
    )
    assert not annotate.enabled_codes()
    assert not profiler.frames
    assert profiler.result().dropped_spans == 0


@pytest.mark.parametrize("generated", ["<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>"])
def test_nested_functions_survive_generated_parent(
    modules: tuple[ModuleType, ModuleType], generated: str
):
    owned, _ = modules
    code = owned.nested.__code__
    # Python 3.12+ inlines list/set/dict comprehensions; cover their earlier code form too.
    owned.nested.__code__ = code.replace(
        co_consts=tuple(
            item.replace(co_name=generated)
            if isinstance(item, CodeType) and item.co_name == "<genexpr>"
            else item
            for item in code.co_consts
        )
    )
    selected = Profiler.module_codes((owned.__name__,))
    assert {item.co_name for item in selected} >= {"nested", "values", "<lambda>"}
    assert generated not in {item.co_name for item in selected}
    with Profiler(features=Profiler.Feature.SPANS, auto=(owned.__name__,)) as profiler:
        assert owned.nested() == 3
    # Named generators still emit an active segment per resume, including final return.
    assert Counter(row.name for row in profiler.result().summaries) == Counter(
        {
            "nested": 1,
            "nested.nested.<locals>.values": 3,
            "nested.nested.<locals>.<genexpr>.<lambda>": 2,
        }
    )

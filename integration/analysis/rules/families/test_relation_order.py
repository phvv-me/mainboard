"""A fact family's relation answers the same rows whatever order the rows it reads come in.

A tie left to the order of the input rows is settled by whichever row a parallel join emits first,
so the answer changes from one run of the fact base to the next. Each test feeds a relation its
inputs in every order and expects the one answer, the smaller key, framework or name.
"""

from collections.abc import Iterator, Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.analysis import Function, Relation
from mainboard.analysis.catalog import Family
from mainboard.analysis.engine.lint import load_catalog

from .builders import macros, views

_TEXT = "VARCHAR"
_FUNCTION = "function"
_ANCESTORS = [("C", "C", 0), ("C", "P", 1), ("C", "Q", 1), ("P", "P", 0), ("Q", "Q", 0)]
_METHODS = [
    ("C", "__getitem__", "r:2", "method"),
    ("C", "__getitem__", "r:1", "method"),
    ("P", "run", "r:p", "method"),
    ("Q", "run", "r:q", "method"),
]
_SEEDS = [
    ("k", "device", "torch", "annotation a"),
    ("k", "device", "cupy", "annotation b"),
    ("k", "array", "cupy", "annotation c"),
    ("s1", "device", "torch", "seed 1"),
    ("s2", "device", "cupy", "seed 2"),
]
_EDGES = [("s1", "x", None), ("s2", "x", None)]
_LAUNCHES = [("ast:a", "a.py", 10, 1, "torch"), ("ast:a", "a.py", 11, 1, "cupy")]
_ENTRIES = [("f", 1, 14, "z"), ("f", 1, 14, "a"), ("f", 1, 25, "c")]
# Three capture regions call two functions, `g` (100, from `a.py` and `b.py`) and `g2` (101, from
# `c.py`), which both call `h` (200).
_REGIONS = [("f1", "a.py", 1, 1, 10), ("f2", "b.py", 1, 1, 10), ("f3", "c.py", 1, 1, 10)]
_REGION_CALLS = [
    ("f1", "a.py", 5, 1, 3),
    ("f2", "b.py", 5, 1, 9),
    ("f3", "c.py", 5, 1, 2),
    ("g", "g.py", 20, 100, 30),
    ("g", "g.py", 30, 101, 40),
]
_REGION_TARGETS = [
    ("f1", 5, "g", 100, _FUNCTION, "direct"),
    ("f2", 5, "g", 100, _FUNCTION, "direct"),
    ("f3", 5, "g", 101, _FUNCTION, "direct"),
    ("g", 20, "h", 200, _FUNCTION, "direct"),
    ("g", 30, "h", 200, _FUNCTION, "direct"),
]
# The relations the type fixpoint reads besides its seeds and edges, empty.
_SILENT = {
    "cuda_elements": {"source": _TEXT, "element": _TEXT},
    "cuda_variable_sources": {"variable": _TEXT, "source": _TEXT, "literal": "BOOLEAN"},
    "cuda_attribute_reads": {"object_key": _TEXT, "attribute_key": _TEXT, "name": _TEXT},
    "cuda_members": {"receiver": _TEXT, "name": _TEXT, "member": _TEXT, "kind": _TEXT},
    "cuda_method_calls": {
        "receiver_key": _TEXT,
        "call_key": _TEXT,
        "name": _TEXT,
        "node_text": _TEXT,
    },
    "cuda_unpacks": {"variable": _TEXT, "source": _TEXT, "position": "INTEGER"},
    "cuda_field_order": {"class_key": _TEXT, "field": _TEXT, "position": "INTEGER"},
}


@pytest.fixture(scope="module", autouse=True)
def catalog() -> Iterator[None]:
    """Every family, relation and function registered, as a run of the rules has them."""
    load_catalog()
    yield


@given(ancestors=st.permutations(_ANCESTORS), methods=st.permutations(_METHODS))
def test_a_class_answers_to_the_same_member_whatever_order_its_ancestors_and_methods_come_in(
    ancestors: list[tuple], methods: list[tuple]
) -> None:
    connection = macros(
        cuda_ancestors=(
            {"class_key": _TEXT, "ancestor_key": _TEXT, "distance": "INTEGER"},
            ancestors,
        ),
        cuda_fields=({"class_key": _TEXT, "name": _TEXT, "field": _TEXT}, []),
        cuda_methods=(
            {"class_key": _TEXT, "name": _TEXT, "ret": _TEXT, "kind": _TEXT},
            methods,
        ),
    )

    answered = connection.execute(Relation.defined["cuda_members"].query).fetchall()

    assert sorted(answered) == [
        ("C", "__getitem__", "r:1", "method"),
        ("C", "run", "r:p", "method"),
        ("P", "run", "r:p", "method"),
        ("Q", "run", "r:q", "method"),
    ]


@given(seeds=st.permutations(_SEEDS), edges=st.permutations(_EDGES))
def test_a_key_gets_the_same_type_whatever_order_its_seeds_and_edges_come_in(
    seeds: list[tuple], edges: list[tuple]
) -> None:
    connection = macros(
        cuda_type_seeds=({"key": _TEXT, "ty": _TEXT, "fw": _TEXT, "why": _TEXT}, seeds),
        cuda_type_edges=({"src": _TEXT, "dst": _TEXT, "only": _TEXT}, edges),
        **{name: (columns, []) for name, columns in _SILENT.items()},
    )
    Function.defined["cuda_created_array"].create(connection)

    typed = connection.execute(Relation.defined["cuda_types"].query).fetchall()

    assert sorted(typed) == [
        ("k", "device", "cupy", "annotation b"),
        ("s1", "device", "torch", "seed 1"),
        ("s2", "device", "cupy", "seed 2"),
        ("x", "device", "cupy", "flow"),
    ]


@given(launches=st.permutations(_LAUNCHES))
def test_a_function_reaching_work_of_two_frameworks_names_the_same_one_in_every_order(
    launches: Sequence[tuple],
) -> None:
    connection = macros(
        cuda_launches=(
            {
                "fact_id": _TEXT,
                "path": _TEXT,
                "node_id": "INTEGER",
                "scope_id": "INTEGER",
                "framework": _TEXT,
            },
            launches,
        ),
        cuda_calls=(
            {
                "fact_id": _TEXT,
                "path": _TEXT,
                "node_id": "INTEGER",
                "scope_id": "INTEGER",
                "receiver_id": "INTEGER",
                "method": _TEXT,
            },
            [("ast:a", "a.py", 20, 2, None, "f")],
        ),
        cuda_types=({"key": _TEXT, "ty": _TEXT, "fw": _TEXT}, []),
        cuda_resolved_calls=(
            {
                "fact_id": _TEXT,
                "node_id": "INTEGER",
                "definition_fact": _TEXT,
                "definition_id": "INTEGER",
                "kind": _TEXT,
            },
            [("ast:a", 20, "ast:a", 1, _FUNCTION)],
        ),
    )

    work = connection.execute(Relation.defined["cuda_device_work"].query).fetchall()

    assert sorted(work) == [
        ("ast:a", "a.py", 10, 1, "torch", True),
        ("ast:a", "a.py", 11, 1, "cupy", True),
        ("ast:a", "a.py", 20, 2, "cupy", False),
    ]


@given(
    regions=st.permutations(_REGIONS),
    calls=st.permutations(_REGION_CALLS),
    targets=st.permutations(_REGION_TARGETS),
)
def test_a_function_that_two_capture_regions_reach_names_the_same_region_in_every_order(
    regions: Sequence[tuple], calls: Sequence[tuple], targets: Sequence[tuple]
) -> None:
    connection = macros(
        cuda_capture_regions=(
            {
                "fact_id": _TEXT,
                "path": _TEXT,
                "scope_id": "INTEGER",
                "first": "INTEGER",
                "last": "INTEGER",
            },
            regions,
        ),
        cuda_calls=(
            {
                "fact_id": _TEXT,
                "path": _TEXT,
                "node_id": "INTEGER",
                "scope_id": "INTEGER",
                "start_line": "INTEGER",
            },
            calls,
        ),
        cuda_resolved_calls=(
            {
                "fact_id": _TEXT,
                "node_id": "INTEGER",
                "definition_fact": _TEXT,
                "definition_id": "INTEGER",
                "kind": _TEXT,
                "via": _TEXT,
            },
            targets,
        ),
    )

    captured = connection.execute(Relation.defined["cuda_captured"].query).fetchall()

    assert sorted(captured) == [
        ("g", 100, 1, "a.py", 3),
        ("g", 101, 1, "c.py", 2),
        ("h", 200, 2, "a.py", 3),
    ]


@given(entries=st.permutations(_ENTRIES))
def test_the_keys_of_a_projection_come_in_the_same_order_whatever_order_its_entries_do(
    entries: list[tuple],
) -> None:
    connection = views(
        class_projection_entries=(
            {"fact_id": _TEXT, "container": "INTEGER", "entry": "INTEGER", "key": _TEXT},
            entries,
        )
    )

    keys = connection.execute(Family.defined["ClassFact"].relations["projection_keys"]).fetchall()

    assert keys == [("f", 1, ["a", "z", "c"])]

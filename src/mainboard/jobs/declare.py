# What a job file says about itself that its imports cannot: the data it reads and the results
# it writes. Both are read off the file's syntax, never by running it, because a job file imports
# the GPU libraries of the environment it runs in and the machine dispatching it may hold none.

import ast
from contextlib import suppress
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Where the decorator keeps its declaration on the target, for a caller holding the object.
MARK = "__mainboard_job__"

# What the decorator is called, which is how it is recognised in a file that is only parsed.
NAME = "job"


class Declaration(FrozenModel):
    """What a job declared beyond its imports.

    needs: workspace-relative data paths the job reads, reached inside the snapshot by a link
        back to the mirror rather than copied or digested.
    resources: workspace-relative files or directories the job reads by path beside its code, a
        registration or a template, pinned and digested with the code unlike a need.
    fetch: the results path pulled home when the job settles, empty when it declared none.
    """

    needs: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    fetch: str = ""


def job[Declared](
    *, needs: Sequence[str] = (), resources: Sequence[str] = (), fetch: str = ""
) -> Callable[[Declared], Declared]:
    """Declare what a job needs beyond its imports, leaving the target itself untouched.

    A no-op at runtime that keeps the declaration on the target for whoever holds it. What a
    dispatch reads is the decorator's literal keywords in the file's syntax, so all must stay
    literal: a path computed at import time is a path the dispatch cannot see.

    needs: workspace-relative data paths the job reads on the host, linked in.
    resources: workspace-relative files or directories read by path beside the code, pinned in.
    fetch: the results path to pull home when the job settles.
    """
    declared = Declaration(needs=tuple(needs), resources=tuple(resources), fetch=fetch)

    def marked(target: Declared) -> Declared:
        with suppress(AttributeError):
            setattr(target, MARK, declared)
        return target

    return marked


def declared(module: ast.Module, name: str) -> Declaration:
    """The declaration on the target `name` of a parsed module, read without importing it.

    A decorated function or method carries it on its decorator; an application carries it on
    the call that wrapped it, `app = job(needs=...)(App(...))`. A target that declares nothing
    gets the empty declaration.

    module: the parsed job file.
    name: the target inside it, including its class path for a pytest node id.
    """
    parts = name.split("::")
    body = module.body
    for parent in parts[:-1]:
        selected = next(
            (node for node in body if isinstance(node, ast.ClassDef) and node.name == parent),
            None,
        )
        if selected is None:
            return Declaration()
        body = selected.body
    for node in body:
        decoration = _decoration(node, parts[-1])
        if decoration is not None:
            return Declaration.model_validate(
                {keyword.arg: _literal(keyword, name) for keyword in decoration.keywords}
            )
    return Declaration()


def _decoration(node: ast.stmt, name: str) -> ast.Call | None:
    """The `job(...)` call decorating `name` in `node`, None when `node` is not that target."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
        return _job_decorator(node)
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    ):
        wrapped = node.value
        if (
            isinstance(wrapped, ast.Call)
            and isinstance(wrapped.func, ast.Call)
            and _is_job(wrapped.func.func)
        ):
            return wrapped.func
    return None


def _job_decorator(function: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Call | None:
    """The `job(...)` call in `function`'s decorator list, None when none is there."""
    return next(
        (
            decorator
            for decorator in function.decorator_list
            if isinstance(decorator, ast.Call) and _is_job(decorator.func)
        ),
        None,
    )


def _is_job(func: ast.expr) -> bool:
    """Whether `func` spells the decorator, bare or through the module it was imported from."""
    if isinstance(func, ast.Name):
        return func.id == NAME
    return isinstance(func, ast.Attribute) and func.attr == NAME


def _literal(keyword: ast.keyword, name: str) -> str | tuple[str, ...] | list[str]:
    """The literal value of one decorator keyword, refusing anything that needs running."""
    try:
        if keyword.arg is None:
            raise ValueError("a splatted mapping")
        return ast.literal_eval(keyword.value)
    except ValueError as opaque:
        raise MissionError(
            f"the `{NAME}` declaration on {name!r} must be literal, since the file is read "
            f"without being imported; {keyword.arg or '**'} is {opaque}"
        ) from None

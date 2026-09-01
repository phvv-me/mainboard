import ast
from collections.abc import Callable

from ..core.errors import MissionError

_FUNCTIONS = {"min": min, "max": max}
_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a / b,
}


def evaluate(expression: str, *, attempt: int) -> int:
    """A submit-time resource expression as an integer, retry-aware.

    The language is deliberately tiny: integer literals, `attempt` (the
    1-based retry number), `+ - * / //`, and `min`/`max` calls, so
    `min(100, attempt * 50)` escalates a memory request across retries while
    nothing else can execute. Anything outside the language refuses loudly.

    expression: the manifest string, a bare integer also accepted.
    attempt: the 1-based try number this submission is.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise MissionError(f"resource expression {expression!r} does not parse: {error}") from None
    return int(_evaluated(tree.body, attempt=attempt, source=expression))


def _evaluated(node: ast.expr, *, attempt: int, source: str) -> float:
    match node:
        case ast.Constant(value=int() | float() as value):
            return value
        case ast.Name(id="attempt"):
            return attempt
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _OPERATORS:
            return _OPERATORS[type(op)](
                _evaluated(left, attempt=attempt, source=source),
                _evaluated(right, attempt=attempt, source=source),
            )
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _FUNCTIONS:
            return _FUNCTIONS[name](
                *(_evaluated(arg, attempt=attempt, source=source) for arg in args)
            )
        case _:
            raise MissionError(
                f"resource expression {source!r} uses more than integers, attempt, "
                "arithmetic, min and max"
            )

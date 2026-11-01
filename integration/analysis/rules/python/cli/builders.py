from collections.abc import Sequence


def _leaf(depth: int) -> str:
    """A command `depth` words below the root, in the nested groups that lead to it."""
    if depth == 1:
        return "Command(name='leaf')"
    return f"Node(name='g{depth}', help='h', commands=({_leaf(depth - 1)},))"


def _tuple(items: Sequence[str]) -> str:
    return "(" + "".join(f"{item}, " for item in items) + ")"


def tree(*, shown: int = 3, hidden: int = 0, depth: int = 2, default: bool = False) -> str:
    """The module declaring a tree: `shown` root children, `hidden` more, a leaf `depth` down."""
    plain = [f"Command(name='c{number}')" for number in range(shown - 1)]
    quiet = [f"Command(name='q{number}', show=False)" for number in range(hidden)]
    fallback = ["Command(name='mb', default=True)"] if default else []
    top = _leaf(depth)
    commands = [*plain, *quiet, *fallback, *([top] if depth == 1 else [])]
    children = [top] if depth > 1 else []
    declared = f"commands={_tuple(commands)}, children={_tuple(children)}"
    return f"TREE = Node(name='mb', help='h', {declared})\n"

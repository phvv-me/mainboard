from collections.abc import Sequence

# A command's name, whether its node runs it when typed alone, and whether a listing shows it.
type Command = tuple[str, bool, bool]
# A group's name, whether a listing shows it, its commands and its groups.
type Group = tuple[str, bool, Sequence[Command], Sequence[Group]]


def declare(group: Group) -> str:
    """The `Node` call declaring `group`."""
    name, shown, commands, groups = group
    members = "".join(
        f"Command(name={word!r}, default={default}, show={visible}), "
        for word, default, visible in commands
    )
    children = "".join(f"{declare(child)}, " for child in groups)
    return (
        f"Node(name={name!r}, help='h', show={shown}, commands=({members}), children=({children}))"
    )


def expected(group: Group, above: tuple[str, ...] = ()) -> list[tuple[str, str, int, bool]]:
    """The command, parent, depth and visibility of `group` and everything under it."""
    name, shown, commands, groups = group
    words = (*above, name)
    own = [(" ".join(words), " ".join(above), len(above), shown)]
    typed = [
        (" ".join((*words, word)), " ".join(words), len(words), visible)
        for word, default, visible in commands
        if not default
    ]
    return own + typed + [row for child in groups for row in expected(child, words)]

"""Rules over the shape of a project's command tree (mcmr's PY-CLI family)."""

from ....catalog import Example
from ...general.project.filesystem import Ceiling

_ROOT = "TREE = Node(name='mb', help='Tools.', commands=({members}), children=({groups}))\n"
_FILE = "src/pkg/verbs/constants.py"


def _declaration(*, shown: int, depth: int) -> str:
    """A tree whose root shows `shown` commands and whose deepest group sits `depth` levels down."""
    members = ", ".join(f"Command(name='c{number}')" for number in range(shown - 1))
    chain = "Command(name='leaf')"
    for level in reversed(range(2, depth + 1)):
        chain = f"Node(name='g{level}', help='Level.', commands=({chain},))"
    return _ROOT.format(members=members, groups=f"{chain},")


class CommandTreeCeiling(Ceiling, id="PY-CLI0002"):
    """Command nodes that show more than `maximum` children, and commands typed through more
    than `maximum_depth` words after the program's name. A person scans ten names and remembers
    three words; past that a listing stops being read and a command is looked up instead. A
    hidden command (`show=False`) is no child and no depth: nobody scans for it."""

    maximum: int = 10
    maximum_depth: int = 3

    query = """
        WITH counted AS (
          SELECT parent, count(*) AS value FROM cli_facts
          WHERE shown AND parent <> '' GROUP BY parent
        )
        SELECT f.path, f.start_line, f.start_column, f.command,
               'shows ' || c.value || ' children, past the ceiling of ' || $maximum AS description
        FROM cli_facts f JOIN counted c ON c.parent = f.command
        WHERE c.value > $maximum
        UNION ALL
        SELECT path, start_line, start_column, command,
               'sits ' || depth || ' words deep, past the ceiling of ' || $maximum_depth
        FROM cli_facts
        WHERE shown AND depth = $maximum_depth + 1
    """
    message = "`{command}` {description}"
    examples = (
        Example(
            bad=_declaration(shown=11, depth=2), good=_declaration(shown=10, depth=2), path=_FILE
        ),
        Example(
            bad=_declaration(shown=3, depth=4), good=_declaration(shown=3, depth=3), path=_FILE
        ),
    )

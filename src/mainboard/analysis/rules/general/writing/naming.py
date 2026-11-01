"""Rules over how local names read (mcmr's ALL-NAMI family)."""

from ....catalog import Example, Rule
from ....claims import Claim
from ....edits import Edit
from ....findings import Hit, Span
from ....operations import Operation
from ....operations.model import Namer

# What a rename is tried with before the model is asked, so one that cannot be made asks nothing.
_PROBE = "renamed_local"
_QUESTION = """\
A Python function binds a local variable whose name is too short to say what it holds. Rename
`{name}`, bound in `{qualname}` of `{path}`, shown below. Answer one new name in snake_case that
says what the value is, at least {minimum} characters long and not already used in the function.

{function}
"""


class UninformativeLocalName(Rule, id="ALL-NAMI0001", applicability="review"):
    """Local bindings in callables whose name is too short to say what they hold.

    The fix of a Python local asks a small model for its name and renames the local and every
    read of it, the scopes inside the function that read it among them. It leaves the local as it
    is when it is no plain local of the function (a parameter, a `global`, an attribute), when the
    function reads names by their spelling (`locals`, `eval`), when a string spells the name, when
    the model's name is too short or already a name the function or a scope in it uses, and for a
    local of any other language.
    """

    minimum_length: int = 3
    conventional_names: tuple[str, ...] = ("i", "j", "k", "n", "x", "y", "z", "_")

    query = """
        WITH bound AS (
          SELECT a.file_id, a.function_id, a.node_id, a.start_line, a.start_column,
                 string_split(t.name, '.')[-1] AS name
          FROM core_node a
          JOIN core_node t ON t.file_id = a.file_id AND t.parent_id = a.node_id
                          AND t.sibling_index = 0 AND t.type IN ('identifier', 'attribute')
          JOIN core_node p ON p.file_id = a.file_id AND p.node_id = a.parent_id
          WHERE a.type IN ('assignment', 'augmented_assignment') AND p.type <> 'assignment'
            AND a.owner_id = a.function_id
          UNION ALL
          SELECT d.file_id, d.function_id, t.node_id, t.start_line, t.start_column, t.name
          FROM core_node d
          JOIN core_node t ON t.file_id = d.file_id AND t.parent_id = d.node_id
            AND t.type = 'identifier'
          LEFT JOIN core_node k ON k.file_id = d.file_id AND k.parent_id = d.node_id
            AND k.sibling_index = t.sibling_index - 1
          WHERE d.owner_id = d.function_id
            AND (d.type = 'variable_declarator' AND t.sibling_index = 0
                 OR d.type = 'for_in_statement' AND k.type IN ('const', 'let', 'var'))
        )
        SELECT arg_min(f.path, b.node_id) AS path, arg_min(b.start_line, b.node_id) AS start_line,
               arg_min(b.start_column, b.node_id) AS start_column, b.name,
               any_value(d.qualname) AS qualname, any_value(n.start_line) AS def_line,
               any_value(d.end_line) AS end_line, any_value(s.content) AS content
        FROM bound b
        JOIN core_def d ON d.file_id = b.file_id AND d.def_id = b.function_id
        JOIN core_node n ON n.file_id = d.file_id AND n.node_id = d.def_id
        JOIN core_file f ON f.file_id = b.file_id
        JOIN source_files s ON s.path = f.path
        WHERE d.parses AND b.name <> '' AND length(b.name) < $minimum_length
          AND NOT list_contains($conventional_names::VARCHAR[], b.name)
        GROUP BY b.file_id, b.function_id, b.name
    """
    message = (
        "`{qualname}` binds `{name}`, which is shorter than the "
        f"{minimum_length} characters a name needs to say what it holds"
    )
    # The model chooses the new name, so no example states a rename and none may ask it. The first
    # is only flagged. The others are left as they are before any question, one because the
    # function reads names by their spelling, one because a string spells the local and one because
    # an f-string field prints the local's own name (`{ab=}`).
    examples = (
        Example(bad="def f(rows):\n    ab = len(rows)\n    return ab\n"),
        Example(
            bad='def f(rows):\n    ab = len(rows)\n    return locals()["ab"]\n',
            good='def f(rows):\n    ab = len(rows)\n    return locals()["ab"]\n',
        ),
        Example(
            bad='def f(obj):\n    ab = 1\n    return getattr(obj, "ab", ab)\n',
            good='def f(obj):\n    ab = 1\n    return getattr(obj, "ab", ab)\n',
        ),
        Example(
            bad='def f(rows):\n    ab = len(rows)\n    return f"{ab=}"\n',
            good='def f(rows):\n    ab = len(rows)\n    return f"{ab=}"\n',
        ),
    )

    @staticmethod
    def _function(hit: Hit) -> Span:
        """The lines of the function binding the local, from its `def`."""
        return Span(
            path=str(hit.path), start_line=int(hit.def_line or 1), end_line=int(hit.end_line or 1)
        )

    def _question(self, hit: Hit, function: Span) -> str:
        """What the model is asked: the local, where it is bound and the function's own text."""
        lines = str(hit.content).splitlines()[function.start_line - 1 : function.end_line]
        return _QUESTION.format(
            name=hit.name,
            qualname=hit.qualname,
            path=hit.path,
            minimum=self.minimum_length,
            function="\n".join(lines),
        )

    def repair(self, hit: Hit) -> list[Edit] | None:
        """The local renamed to what a small model calls it, every read following.

        The function is claimed, so that two locals of one function are renamed in two passes,
        the second asked about the function with the first's new name in it.
        """
        row = {
            "operation": "rewrite-references",
            "scope": "function",
            "path": hit.path,
            "content": hit.content,
            "line": hit.def_line,
            "name": hit.name,
        }
        if Operation.rendered(row | {"to": _PROBE}) is None:
            return None
        function = UninformativeLocalName._function(hit)
        names = Namer().names(self._question(hit, function), 1)
        if names is None or len(names[0]) < self.minimum_length:
            return None
        renamed = Operation.rendered(row | {"to": names[0]})
        return None if renamed is None else [Claim(span=function), *renamed]

from collections.abc import Sequence

import tomlkit
import tomlkit.items
from tomlkit.container import OutOfOrderTableProxy

from ..core.errors import MissionError

_ASSIGN = len("= ")
# The keys naming where a requirement comes from instead of an index, which pins it there whatever
# version it also states (a binary's `url` and `sha256` beside its `version`).
_SOURCES = {"path", "git", "url"}


class ManifestText:
    """The workspace manifest edited through tomlkit, which keeps comments and quoting.

    Two habits tomlkit does not keep are restored: `=` aligned into a column, and a table's
    heading comment staying with its table rather than the entry appended above it.
    """

    def __init__(self, text: str) -> None:
        self.document = tomlkit.parse(text)

    def constraint(self, path: tuple[str, ...], name: str) -> str:
        """What `name` is pinned to in the table at `path`, its version when written as a
        table, or a source spec as written."""
        declared = self.table(path)[name]
        if isinstance(declared, str):
            return declared
        if isinstance(declared, dict) and "version" in declared:
            return str(declared["version"])
        return tomlkit.dumps({name: declared}).partition("=")[2].strip()

    def versioned(self, path: tuple[str, ...], name: str) -> bool:
        """Whether `name` in the table at `path` names a version to raise: a bare requirement
        or a table carrying one, not a path, git or url source, nor a workspace member the
        manifest declares only through `[workspace] members`, nor an exact pin, which states the
        one release it must be (a registered estimator kept byte-stable) rather than a range."""
        if not self.declares(path, name):
            return False
        declared = self.table(path)[name]
        if isinstance(declared, dict) and not _SOURCES & declared.keys():
            declared = declared.get("version")
        return isinstance(declared, str) and not _exact(declared)

    def declares(self, path: Sequence[str], name: str) -> bool:
        """Whether the table at `path` exists and carries `name`."""
        table = self.document
        for key in path:
            if not isinstance(table, dict) or key not in table:
                return False
            table = table[key]
        return isinstance(table, dict) and name in table

    def drop(self, path: tuple[str, ...], name: str) -> None:
        """Remove `name` from the table at `path`, and any table left empty, undoing a `put`."""
        del self.table(path)[name]
        emptied = path
        while emptied and not self.table(emptied):
            parent = self.document if len(emptied) == 1 else self.table(emptied[:-1])
            del parent[emptied[-1]]
            emptied = emptied[:-1]

    def put(self, path: tuple[str, ...], name: str, *, spec: str) -> None:
        """Declare `name` as `spec` in the table at `path`, creating the table when absent.

        A replaced entry keeps its alignment and its comment's column, and one written as a table
        (`{ version = ..., extras = [...] }`) keeps every field but its version; a new one is
        padded to the column and lands beneath the last requirement, above the trailing heading
        comment.
        """
        table = self.table(path, create=True)
        if name in table:
            declared = table[name]
            width = len(declared.as_string())
            if isinstance(declared, dict):
                declared["version"] = spec
            else:
                table[name] = spec
            # tomlkit keeps the whitespace before a comment, which moves it by the value's growth.
            trivia = table[name].trivia
            if trivia.comment:
                grown = len(table[name].as_string()) - width
                trivia.comment_ws = " " * max(len(trivia.comment_ws) - grown, 1)
            return
        key = tomlkit.key(name)
        key.sep = " " * max(ManifestText._column(table) - len(name) - _ASSIGN, 1) + "= "
        # tomlkit files the trailing whitespace and comments inside the table's body, so lift
        # them off, append, and lay them back down.
        body = table.value.body
        trailing = []
        while body and body[-1][0] is None:
            trailing.append(body.pop())
        table.append(key, tomlkit.item(spec))
        body.extend(reversed(trailing))

    def table(self, path: tuple[str, ...], *, create: bool = False) -> tomlkit.items.Table:
        """The table at `path`, created on request as one `[rust.deps]` heading, not a nest.

        A level whose sub-tables are not contiguous in the file (`[on.linux.deps]`, then
        `[on.unix.python.dev]`, then `[on.linux.python.dev]`) reads as tomlkit's
        `OutOfOrderTableProxy`, which is walked through like a table; the requirements' own
        table at the end is always a real one.
        """
        table: tomlkit.TOMLDocument | tomlkit.items.Table | OutOfOrderTableProxy = self.document
        for at, key in enumerate(path):
            if key not in table:
                if not create:
                    raise MissionError(f"[{'.'.join(path[: at + 1])}] is not in this manifest")
                table[key] = tomlkit.table(is_super_table=at < len(path) - 1)
            nested = table[key]
            if not isinstance(nested, tomlkit.items.Table | OutOfOrderTableProxy):
                raise MissionError(f"[{'.'.join(path[: at + 1])}] is not a table of requirements")
            table = nested
        if not isinstance(table, tomlkit.items.Table):
            raise MissionError(f"[{'.'.join(path)}] is not a table of requirements")
        return table

    def text(self) -> str:
        return tomlkit.dumps(self.document)

    @staticmethod
    def _column(table: tomlkit.items.Table) -> int:
        """The column this table starts its values in: key text plus separator, 0 when empty."""
        return max(
            (len(str(key)) + len(key.sep) for key, _ in table.value.body if key is not None),
            default=0,
        )


def _exact(constraint: str) -> bool:
    """Whether `constraint` is one `==` clause naming a single release (`==1.9.0`), the one
    spelling every ecosystem reads as exact; a bare `0.3.13` is a caret range to cargo."""
    clauses = [clause.strip() for clause in constraint.split(",") if clause.strip()]
    return len(clauses) == 1 and clauses[0].startswith("==") and "*" not in clauses[0]

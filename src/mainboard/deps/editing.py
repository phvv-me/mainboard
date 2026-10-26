from collections.abc import Sequence

import tomlkit
import tomlkit.items
from tomlkit.container import OutOfOrderTableProxy

from ..core.errors import MissionError

_ASSIGN = len("= ")
# The keys naming where a requirement comes from instead of an index, which pins it there whatever
# version it also states (a binary's `url` and `sha256` beside its `version`).
_SOURCES = {"path", "git", "url"}

# One level of the manifest a table path walks through.
type _Level = tomlkit.TOMLDocument | tomlkit.items.Table | OutOfOrderTableProxy


class ManifestText:
    """The workspace manifest edited through tomlkit, which keeps comments and quoting.

    Two habits tomlkit does not keep are restored: `=` aligned into a column, and a table's
    heading comment staying with its table rather than the entry appended above it.
    """

    @staticmethod
    def _append(table: tomlkit.items.Table, name: str, *, spec: str) -> None:
        """Declare a new `name` padded to the column, beneath the last requirement."""
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

    @staticmethod
    def _column(table: tomlkit.items.Table) -> int:
        """The column this table starts its values in: key text plus separator, 0 when empty."""
        return max(
            (len(str(key)) + len(key.sep) for key, _ in table.value.body if key is not None),
            default=0,
        )

    @staticmethod
    def _exact(constraint: str) -> bool:
        """Whether `constraint` is one `==` clause naming a single release (`==1.9.0`).

        That is the one spelling every ecosystem reads as exact; a bare `0.3.13` is a caret range
        to cargo.
        """
        clauses = [clause.strip() for clause in constraint.split(",") if clause.strip()]
        return len(clauses) == 1 and clauses[0].startswith("==") and "*" not in clauses[0]

    @staticmethod
    def _level(table: _Level, path: Sequence[str], made: tomlkit.items.Table | None) -> _Level:
        """The table `path`'s last key names inside `table`, `made` there when absent."""
        key, where = path[-1], f"[{'.'.join(path)}]"
        if key not in table:
            if made is None:
                raise MissionError(f"{where} is not in this manifest")
            table[key] = made
        match table[key]:
            case tomlkit.items.Table() | OutOfOrderTableProxy() as nested:
                return nested
            case _:
                raise MissionError(f"{where} is not a table of requirements")

    @staticmethod
    def _replace(table: tomlkit.items.Table, name: str, *, spec: str) -> None:
        """Point `name` at `spec`, keeping its alignment, its comment's column and its fields.

        One written as a table (`{ version = ..., extras = [...] }`) changes only its version.
        """
        width = len(table[name].as_string())
        match table[name]:
            case tomlkit.items.InlineTable() | tomlkit.items.Table() as fields:
                fields["version"] = spec
            case _:
                table[name] = spec
        # tomlkit keeps the whitespace before a comment, which moves it by the value's growth.
        trivia = table[name].trivia
        if trivia.comment:
            grown = len(table[name].as_string()) - width
            trivia.comment_ws = " " * max(len(trivia.comment_ws) - grown, 1)

    def __init__(self, text: str) -> None:
        self.document = tomlkit.parse(text)

    def _walk(self, path: Sequence[str], *, create: bool = False) -> _Level:
        """The table or out-of-order proxy at `path`, each missing level made on request."""
        table: _Level = self.document
        for at in range(len(path)):
            # Every level but the last is only a heading prefix, never written on its own.
            made = tomlkit.table(is_super_table=at < len(path) - 1) if create else None
            table = ManifestText._level(table, path[: at + 1], made)
        return table

    def constraint(self, path: tuple[str, ...], name: str) -> str:
        """What `name` is pinned to in the table at `path`.

        That is its version when written as a table, or a source spec as written.
        """
        declared = self.table(path)[name]
        match declared.unwrap():
            case str() as constraint:
                return constraint
            case {"version": version}:
                return str(version)
            case _:
                return tomlkit.dumps({name: declared}).partition("=")[2].strip()

    def declares(self, path: tuple[str, ...], name: str) -> bool:
        """Whether the table at `path` exists and carries `name`."""
        try:
            return name in self.table(path)
        except MissionError:
            return False

    def drop(self, path: tuple[str, ...], name: str) -> None:
        """Remove `name` from the table at `path`, and any table left empty, undoing a `put`."""
        del self.table(path)[name]
        emptied = path
        while emptied and not self._walk(emptied):
            parent = self.document if len(emptied) == 1 else self._walk(emptied[:-1])
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
            ManifestText._replace(table, name, spec=spec)
        else:
            ManifestText._append(table, name, spec=spec)

    def table(self, path: tuple[str, ...], *, create: bool = False) -> tomlkit.items.Table:
        """The table at `path`, created on request as one `[rust.deps]` heading, not a nest.

        A level whose sub-tables are not contiguous in the file (`[on.linux.deps]`, then
        `[on.unix.python.dev]`, then `[on.linux.python.dev]`) reads as tomlkit's
        `OutOfOrderTableProxy`, which is walked through like a table; the requirements' own
        table at the end is always a real one.
        """
        match self._walk(path, create=create):
            case tomlkit.items.Table() as table:
                return table
            case _:
                raise MissionError(f"[{'.'.join(path)}] is not a table of requirements")

    def text(self) -> str:
        return tomlkit.dumps(self.document)

    def versioned(self, path: tuple[str, ...], name: str) -> bool:
        """Whether `name` in the table at `path` names a version to raise.

        It does for a bare requirement or a table carrying one, not for a path, git or url source,
        nor a workspace member the manifest declares only through `[workspace] members`, nor an
        exact pin, which states the one release it must be (a registered estimator kept
        byte-stable) rather than a range.
        """
        if not self.declares(path, name):
            return False
        match self.table(path)[name].unwrap():
            case str() as version:
                return not ManifestText._exact(version)
            case {"version": str() as version, **fields} if not _SOURCES & fields.keys():
                return not ManifestText._exact(version)
            case _:
                return False

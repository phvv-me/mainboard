from collections.abc import Sequence

import tomlkit
import tomlkit.items

from ..core.errors import MissionError

_ASSIGN = len("= ")


class ManifestText:
    """The workspace manifest edited through tomlkit, which keeps comments and quoting.

    Two habits tomlkit does not keep are restored: `=` aligned into a column, and a table's
    heading comment staying with its table rather than the entry appended above it.
    """

    def __init__(self, text: str) -> None:
        self.document = tomlkit.parse(text)

    def constraint(self, path: tuple[str, ...], name: str) -> str:
        """What `name` is pinned to in the table at `path`, or a source spec as written."""
        declared = self.table(path)[name]
        if isinstance(declared, str):
            return declared
        return tomlkit.dumps({name: declared}).partition("=")[2].strip()

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

        A replaced entry keeps its alignment and comment; a new one is padded to the column and
        lands beneath the last requirement, above the trailing heading comment.
        """
        table = self.table(path, create=True)
        if name in table:
            table[name] = spec
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
        """The table at `path`, created on request as one `[rust.deps]` heading, not a nest."""
        table = self.document
        for at, key in enumerate(path):
            if key not in table:
                if not create:
                    raise MissionError(f"[{'.'.join(path[: at + 1])}] is not in this manifest")
                table[key] = tomlkit.table(is_super_table=at < len(path) - 1)
            table = table[key]
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

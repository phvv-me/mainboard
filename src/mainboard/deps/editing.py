import tomlkit
import tomlkit.items

from ..core.errors import MissionError

# What `key = value` costs beyond the padding, the `= ` every entry carries.
_ASSIGN = len("= ")


class ManifestText:
    """The workspace manifest as its author wrote it, edited without disturbing the writing.

    tomlkit keeps every comment, blank line and quote style the file already carries, so an
    edit here reads as the same hand-written document with one line added or dropped. Two
    habits of this manifest it does not carry on its own, and both are restored here: entries
    aligning their `=` into a column, and the comment that introduces the next table belonging
    to that table rather than to the entry appended above it.
    """

    def __init__(self, text: str) -> None:
        """text: the manifest exactly as it sits on disk."""
        self.document = tomlkit.parse(text)

    def constraint(self, path: tuple[str, ...], name: str) -> str:
        """What `name` is pinned to in the table at `path`, empty when it declares no version.

        A requirement carrying a source rather than a range, a local path or a git url, has no
        version to report and answers with how it is written instead, since that is the line a
        caller is about to replace.
        """
        declared = self.table(path)[name]
        if isinstance(declared, str):
            return declared
        return tomlkit.dumps({name: declared}).partition("=")[2].strip()

    def declares(self, path: tuple[str, ...], name: str) -> bool:
        """Whether the table at `path` exists and carries `name`."""
        table = self.document
        for key in path:
            if not isinstance(table, dict) or key not in table:
                return False
            table = table[key]
        return isinstance(table, dict) and name in table

    def drop(self, path: tuple[str, ...], name: str) -> None:
        """Remove `name` from the table at `path`, and any table the removal left empty.

        The mirror of `put`, which writes a table the manifest never had. Adding a requirement
        to a new ecosystem and then dropping it again therefore leaves the file exactly as it
        was, instead of an empty `[rust.deps]` heading declaring nothing.
        """
        del self.table(path)[name]
        at = path
        while at and not self.table(at):
            parent = self.document if len(at) == 1 else self.table(at[:-1])
            del parent[at[-1]]
            at = at[:-1]

    def put(self, path: tuple[str, ...], name: str, spec: str) -> None:
        """Declare `name` as `spec` in the table at `path`, creating the table when absent.

        Replacing an existing entry leaves its own alignment and any trailing comment alone,
        since only the value moved. A new entry is padded to the column the table already lines
        its values up in and lands beneath the last requirement rather than beneath the blank
        line and heading comment that trail the table in the file.
        """
        table = self.table(path, create=True)
        if name in table:
            table[name] = spec
            return
        key = tomlkit.key(name)
        key.sep = " " * max(_column(table) - len(name) - _ASSIGN, 1) + "= "
        # tomlkit files the whitespace and comments that follow a table inside that table's own
        # body, so appending outright would put the new requirement below the comment that
        # introduces the next table. Lifting that trailing trivia off, appending, and laying it
        # back down keeps the entry with the requirements it joins and the comment with the
        # table it announces.
        body = table.value.body
        trailing = []
        while body and body[-1][0] is None:
            trailing.append(body.pop())
        table.append(key, tomlkit.item(spec))
        body.extend(reversed(trailing))

    def table(self, path: tuple[str, ...], *, create: bool = False) -> tomlkit.items.Table:
        """The table at `path`, created as the manifest would spell it when asked for.

        A table the manifest never had is written as one heading rather than a nest of empty
        ones, `[rust.deps]` and not `[rust]` followed by `[rust.deps]`, which is how every
        dependency table already in the file is written.
        """
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
        """The whole manifest as it should now be written."""
        return tomlkit.dumps(self.document)


def _column(table: tomlkit.items.Table) -> int:
    """The column this table starts its values in, so a new entry lines up with the rest.

    tomlkit keeps a key's padding in the key text and its `= ` in the separator, so where a
    value begins is the two measured together, and a table with nothing in it yet begins
    nowhere and gets the single space every ordinary entry would have carried.
    """
    return max(
        (len(str(key)) + len(key.sep) for key, _ in table.value.body if key is not None),
        default=0,
    )

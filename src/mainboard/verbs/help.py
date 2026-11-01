"""The root of the command line: the options it answers before any command word, and the
listing of the root or a group typed alone.

None of it imports a verb module: a listing reads `TREE`, and only a command's own clause, which
`TREE` does not hold, resolves the group that command sits in.
"""

__lazy_modules__ = ["importlib", "json", "mainboard.guide.search", "mainboard.render", "rich.tree"]

import json
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from functools import reduce
from importlib import metadata
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import App, Parameter, default_name_transform
from patos import FrozenOpenModel
from rich.text import Text
from rich.tree import Tree

from ..core.errors import MissionError
from ..core.project import Project
from ..guide.search import Help
from ..log import logger
from ..render import Audience, Terminal, mode_of, record
from .enums import Accent
from .node import Command, Node

if TYPE_CHECKING:
    from .node import Words

# How wide a listing row prints at most, and the narrowest its name column is.
_WIDTH = 75
_COLUMN = 10
# What a listing ends with, the way further.
_FURTHER = "mb <path> lists a group · add --help for arguments · mb --tree shows all"
# What ends a help's first clause.
_CLAUSE = ("; ", ": ", ". ")


class Root:
    """The options cyclopts' meta app runs before any command word, and the listing of the root
    or of a group typed alone.

    A person sees groups in their accent and clauses dim; an agent the same words, plain.
    """

    @classmethod
    def cli(cls, node: Node, argv: Sequence[str] | None = None) -> App:
        """`node`'s CLI application, run as `app.meta(argv)` so the root's options answer first.

        argv: as `Node.cli` reads it.

        Raises ValueError for a root option declared in `node` and bound to no verb here, or the
        reverse.
        """
        root = cls(node, node.cli(argv))
        verbs = (root.completion, root.help_, root.tree, root.version)
        bound = {f"--{default_name_transform(verb.__name__)}": verb for verb in verbs}
        if stray := sorted(set(bound) ^ {option.name for option in node.options}):
            raise ValueError(f"the root options {stray} are declared or bound alone")
        for name, verb in bound.items():
            # `-h` is cyclopts' own short help flag, so the root keeps it beside `--help`.
            root.app.meta.command(verb, name=(name, "-h") if name == "--help" else name)
        root.app.meta.default(root.launch)
        return root.app

    @staticmethod
    def _depths(node: Node, path: Words) -> dict[int, int]:
        """How many shown commands sit at each depth in words under `node`, which sits at `path`.

        A node run alone counts as a command.
        """
        counted = Counter(
            len(words)
            for words, entry in node.entries(path)
            if entry.show and (isinstance(entry, Command) or entry.ran)
        )
        return dict(sorted(counted.items()))

    @staticmethod
    def _members(node: Node) -> list[Node | Command]:
        """`node`'s commands, then its groups, hidden ones too, in the order its listing shows."""
        return [*(command for command in node.commands if not command.default), *node.children]

    def __init__(self, node: Node, app: App) -> None:
        """node: the declaration `app` was built from."""
        self.node = node
        self.app = app

    def _described(self, member: Node | Command, words: Words) -> str:
        """The first clause of `member`'s help, its command's docstring when the tree has none."""
        said = member.help or reduce(App.__getitem__, words, self.app).help
        first = said.strip().partition("\n")[0]
        for end in _CLAUSE:
            first = first.partition(end)[0]
        lead, _, rest = first.rstrip(".").partition(" ")
        return " ".join(filter(None, (lead.lower() if lead[1:].islower() else lead, rest)))

    def _group(self, path: Sequence[str]) -> Node:
        """The node `path` names; a MissionError when it names none."""
        try:
            return self.node.node(" ".join(path))
        except KeyError:
            raise MissionError(
                f"no group `{' '.join(('mb', *path))}`; `mb --tree` shows them all"
            ) from None

    def _grown(
        self, branch: Tree, node: Node, path: Words, *, accent: Accent | None, every: bool
    ) -> Tree:
        """`branch` with every member of `node` grown under it, groups recursively."""
        members = [member for member in Root._members(node) if every or member.show]
        width = max((len(member.label) for member in members), default=0) + 2
        for member in members:
            words = (*path, member.name)
            tone = (member.accent if isinstance(member, Node) else None) or accent
            ran = isinstance(member, Command) or member.ran
            said = (self._described(member, words) if ran else "", "dim")
            label = Text.assemble(
                self._labelled(member, tone, width), said, no_wrap=True, overflow="ellipsis"
            )
            if not member.show:
                label.stylize("dim")
            twig = branch.add(label)
            if isinstance(member, Node):
                self._grown(twig, member, words, accent=tone, every=every)
        return branch

    def _labelled(self, member: Node | Command, tone: Accent | None, width: int) -> Text:
        """`member`'s name padded to `width`: a group in bold `tone`, a command in bold."""
        group = isinstance(member, Node) and not member.ran and tone
        return Text(f"{member.label:<{width}}", style=f"bold {tone}" if group else "bold")

    def _listed(self, path: Words, node: Node) -> None:
        """Print `node`'s listing: what it is, a row per shown member, and the way further."""
        named = " ".join((self.app.name[0], *path))
        version = "" if path else f" {metadata.version(Project().package)}"
        rows = [Text(f"{named}{version} · {self._described(node, path)}", style="dim")]
        accent = self.node.node(path[0]).accent if path else None
        members = [member for member in Root._members(node) if member.show]
        width = max(_COLUMN, *(len(member.label) + 2 for member in members))
        for member in members:
            tone = (member.accent if isinstance(member, Node) else None) or accent
            label = self._labelled(member, tone, width)
            rows.append(Text.assemble("  ", label, self._said(member, (*path, member.name), tone)))
        rows.append(Text(_FURTHER, style="dim"))
        console = Terminal.current().console()
        for row in rows:
            row.truncate(_WIDTH, overflow="ellipsis")
            console.print(row, no_wrap=True, crop=False)

    def _said(self, member: Node | Command, words: Words, tone: Accent | None) -> Text:
        """What a listing row says of `member`: its clause, and a group's own members."""
        said = Text()
        if isinstance(member, Command) or member.ran:
            said.append(self._described(member, words), style="dim")
        members = Root._members(member) if isinstance(member, Node) else []
        inner = [child for child in members if child.show]
        if said and inner:
            said.append(" · ", style="dim")
        for each, child in enumerate(inner):
            group = isinstance(child, Node) and not child.ran and tone
            said.append(f"{' ' * bool(each)}{child.label}", style=str(tone) if group else "")
        return said

    def _shown(self, grown: Tree) -> None:
        """Print `grown`: rich guides in its accents for a person, plain indented lines else."""
        if Terminal.current().audience is Audience.HUMAN:
            Terminal.current().console().print(grown)
            return
        logger.result("\n".join(_plain(grown)))

    def completion(self, shell: Literal["bash", "zsh", "fish"]) -> None:
        """Print the script that completes this tool's commands and options in a shell.

        Keep it in the shell's startup: `eval "$(mb --completion zsh)"` (zsh, bash), or
        `mb --completion fish | source` in fish.

        Args:
            shell: the shell the script is for.
        """
        whole = Root.cli(self.node)
        logger.result(whole.generate_completion(prog_name=whole.name[0], shell=shell))

    def help_(self, *query: Annotated[str, Parameter(allow_leading_hyphen=True)]) -> None:
        """Show a command's help, or search the docs, the commands and the Python docstrings.

        Args:
            query: an exact command path such as `job submit`, or words such as `Log.read_table`.
        """
        # A search reads every command, the root's own too, which only the whole CLI mounts.
        search = Help(Root.cli(self.node) if query else self.app)
        try:
            found = search.show(" ".join(query))
        except LookupError as nothing:
            raise MissionError(str(nothing)) from None
        logger.result(found, end="")

    def launch(
        self, *tokens: Annotated[str, Parameter(allow_leading_hyphen=True, show=False)]
    ) -> int | None:
        """List the group `tokens` name, the root when they name none; run anything else.

        Returns what the command run returns, its exit status when it states one.
        """
        try:
            node = self.node.node(" ".join(tokens))
        except KeyError:
            return self.app(tokens)
        if node.ran:
            return self.app(tokens)
        self._listed(tokens, node)
        return None

    def tree(
        self,
        *path: str,
        all_: Annotated[bool, Parameter(name="--all")] = False,
        json_: Annotated[bool, Parameter(name="--json")] = False,
    ) -> None:
        """Show every command under a group, each with the first clause of its help.

        A person sees the tree's guides in each group's accent, an agent plain indented lines.

        Args:
            path: the group's words, the root when omitted.
            all_: also the hidden verbs that jobs and keepers run, and the aliases, dimmed.
            json_: print the declaration and how many shown commands sit at each depth instead.
        """
        node = self._group(path)
        if json_:
            declared = node.model_dump(mode="json", exclude_defaults=True)
            logger.result(json.dumps({"tree": declared, "depths": Root._depths(node, path)}))
            return
        accent = self.node.node(path[0]).accent if path else None
        root = Text(node.label if path else self.app.name[0], style=f"bold {accent or ''}")
        grown = self._grown(Tree(root), node, path, accent=accent, every=all_)
        aliases = {" ".join(words): target for words, target in self.node.spellings}
        width = max(map(len, aliases), default=0) + 2
        for alias, target in aliases.items() if all_ and not path else ():
            grown.add(Text(f"{alias:<{width}}(alias of mb {' '.join(target)})", style="dim"))
        self._shown(grown)

    def version(self, *, json_: Annotated[bool, Parameter(name="--json")] = False) -> None:
        """Show this installation: its version, where it runs from, and the engines it pins.

        `source` is where it was installed from, the checkout of an editable install. DuckDB is
        named because a served lake needs the same release at both ends.

        Args:
            json_: print canonical JSON instead of the compact rows.
        """
        installed = metadata.distribution(Project().package)
        origin = _Origin.model_validate_json(installed.read_text("direct_url.json") or "{}")
        found = {
            "version": installed.version,
            "python": sys.executable,
            "source": origin.url.removeprefix("file://"),
            "duckdb": metadata.version("duckdb"),
        }
        record(found, mode=mode_of(json_mode=json_), fields=(), title="version")


class _Origin(FrozenOpenModel):
    """Where an installer took a distribution from, as its `direct_url.json` says (PEP 610)."""

    url: str = ""


def _plain(tree: Tree, depth: int = 0) -> Iterator[str]:
    """`tree` as indented lines, two spaces a level, with no guide and no color."""
    yield f"{'  ' * depth}{tree.label}".rstrip()
    for child in tree.children:
        yield from _plain(child, depth + 1)

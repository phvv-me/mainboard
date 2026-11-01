"""What a command tree is made of, how it registers on cyclopts, and how a typed command line
is read against one.

A `Node` registers on cyclopts lazily: a group's home module (`module`) defines it as the
attribute named after it, its App built there by `TREE.node(path).app(...)` from that module's own
verbs, so every method reference stays where the method lives. A root command names its home
module, which binds its verbs for the root as `mb = TREE.bind(...)`, and the root's options run on
cyclopts' meta app (`verbs.help.Root`). Before cyclopts parses, an alias is spelled as its path
and a word naming nothing is refused with the path it moved to or the nearest names (`Typed`).
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from difflib import SequenceMatcher
from importlib import import_module
from itertools import takewhile
from types import FunctionType, MethodType

from cyclopts import App, Parameter, default_name_transform
from patos import FrozenModel

from ..core.project import Project
from ..render import Terminal
from .enums import Accent, Scope

# A verb as its home module binds it: a bound method or a module-level function.
type Verb = Callable[..., int | None]
# A command's words below the root, `("data", "lake", "ingest")`.
type Words = tuple[str, ...]

# The help and version flags every command answers but a trailing one's version; the root's own
# are its options, which no command could inherit. Typed anywhere, they name no command and run
# nothing.
_HELP = ("--help", "-h")
_VERSION = ("--version",)
_META = frozenset((*_HELP, *_VERSION))

# How close an unknown word must read to a command name for the refusal to suggest it, how much
# less close than the nearest name another may read and still be suggested beside it, and the
# most paths a refusal suggests.
_NEAR = 0.6
_TIE = 0.1
_GUESSES = 3

# The columns help is laid out in when standard output is not a terminal to measure.
_PIPED_WIDTH = 160


class Command(FrozenModel):
    """A verb under its node: what cyclopts cannot read from the method itself.

    aliases: words typed in place of this command's path, each a spelling sealed research runs.
    default: run when its node is typed alone; it is then named after the node.
    help: the line a listing shows without importing the verb, its docstring's first line; a
        command registered lazily (a root command, or one with its own `module`) needs it.
    module: its home module when that is not its node's: imported only when it is typed.
    moved: the paths 0.5.7 typed it at, which a refusal answers with this one's, where neither
        its name nor the words typed lead there (`self update` for `host update`).
    name: the word it is typed as.
    scope: who besides the owner may run it.
    show: listed in help; a verb only a dispatched job or a keeper runs is not.
    trailing: the command typed after it passes through verbatim, `--version` included.
    """

    aliases: tuple[str, ...] = ()
    default: bool = False
    help: str = ""
    module: str = ""
    moved: tuple[str, ...] = ()
    name: str
    scope: Scope | None = None
    show: bool = True
    trailing: bool = False

    @property
    def flags(self) -> dict[str, Sequence[str]]:
        """What cyclopts registers it with beyond its name, its help and version flags."""
        return {"help_flags": _HELP, "version_flags": () if self.trailing else _VERSION}

    @property
    def label(self) -> str:
        """Its name as a listing shows it."""
        return self.name

    def defer(self, app: App) -> None:
        """Register it on `app` as its home module's attribute named after it, imported if typed.

        A root command registers so only for its listing line: `Node.cli` mounts a typed one
        itself, and `main` refuses any other word before cyclopts could resolve one.
        """
        target = f"{self.module}:{self.name.replace('-', '_')}"
        app.command(target, name=self.name, help=self.help or None, show=self.show, **self.flags)

    def register(self, app: App, verb: Verb) -> None:
        """Register `verb` on `app` as this command."""
        if self.default:
            app.default(verb)
            return
        app.command(verb, name=self.name, show=self.show, **self.flags)


class Node(FrozenModel):
    """A command group.

    accent: the color its subtree prints in for a person, set on root groups.
    aliases: words typed in place of its path, as a command's are.
    children: its subgroups, each registered lazily.
    commands: its verbs, in the order a listing shows them.
    help: the only home of its help text; a node run alone (`ran`) answers `--help` with its
        default command's docstring, and lists this line only.
    module: the module defining it as the attribute named after it, imported only when typed;
        the root has none, its commands each naming theirs.
    name: the word it is typed as.
    options: the root's own options, each a command of cyclopts' meta app typed before any word.
    show: listed in help.
    """

    accent: Accent | None = None
    aliases: tuple[str, ...] = ()
    children: tuple[Node, ...] = ()
    commands: tuple[Command, ...] = ()
    help: str
    module: str = ""
    name: str
    options: tuple[Command, ...] = ()
    show: bool = True

    @property
    def label(self) -> str:
        """Its name as a listing shows it: a group's ends in `/`, a node run alone's does not."""
        return self.name if self.ran else f"{self.name}/"

    @property
    def ran(self) -> bool:
        """Whether typing it alone runs its default command."""
        return any(command.default for command in self.commands)

    @property
    def spellings(self) -> list[tuple[Words, Words]]:
        """Every alias as its words and the path it stands for, the longest alias first."""
        options = (((option.name,), option) for option in self.options)
        return sorted(
            (
                (tuple(alias.split()), path)
                for path, entry in (*self.entries(), *options)
                for alias in entry.aliases
            ),
            key=lambda pair: -len(pair[0]),
        )

    @staticmethod
    def _named(verb: Verb) -> str:
        """The command `verb` is as cyclopts names it; a TypeError for a callable with no name."""
        if not isinstance(verb, FunctionType | MethodType):
            raise TypeError(f"a verb is a function or bound method, not {type(verb).__name__}")
        return default_name_transform(verb.__name__)

    def _bound(self, app: App, verbs: Mapping[str, Verb]) -> App:
        """`app` holding every command bound to its verb and the rest registered lazily."""
        self._register_commands(app, verbs)
        self._register_children(app)
        return app

    def _register_children(self, app: App) -> None:
        """Register every subgroup on `app` as its home module's attribute, imported if typed."""
        for child in self.children:
            app.command(
                f"{child.module}:{child.name.replace('-', '_')}",
                name=child.name,
                help=child.help,
                show=child.show,
            )

    def _register_commands(self, app: App, verbs: Mapping[str, Verb]) -> None:
        """Register each command on `app`, bound to its verb or, homed elsewhere, lazily.

        Raises ValueError for a command of this node's own module that no verb binds, a TREE and
        its home module out of step.
        """
        for command in self.commands:
            if command.name in verbs:
                command.register(app, verbs[command.name])
            elif command.module:
                command.defer(app)
            else:
                raise ValueError(f"`{self.name} {command.name}` is bound by no verb")

    def app(self, *verbs: Verb, **named: Verb) -> App:
        """This node as a cyclopts App, built once by its home module from that module's verbs."""
        app = App(name=self.name, help=None if self.ran else self.help, show=self.show)
        return self._bound(app, self.bind(*verbs, **named))

    def bind(self, *verbs: Verb, **named: Verb) -> dict[str, Verb]:
        """Each verb under the name of the command it is, as a home module states it.

        verbs: callables named as cyclopts names them (`show` is `show`, `help_` is `help`).
        named: callables under the command they are typed as, `_` for `-` (`font_previews`).

        Raises ValueError for a verb this node declares no command for.
        """
        bound = {self._named(verb): verb for verb in verbs}
        bound |= {default_name_transform(name): verb for name, verb in named.items()}
        if stray := sorted(set(bound) - {command.name for command in self.commands}):
            raise ValueError(f"`{self.name}` declares no command {stray} in TREE")
        return bound

    def cli(self, argv: Sequence[str] | None = None) -> App:
        """The root as the CLI application, importing only the verb module `argv` types.

        argv: the command line; when None every root command's module is imported, as help search
            and completion read them all.

        Groups register lazily. A root command does only when not typed: resolved by cyclopts,
        its help would be the listing line rather than its whole docstring. The root answers no
        `--help` or `--version` itself: they are among its options, which `Root` mounts.
        """
        project, terminal = Project(), Terminal.current()
        width = None if terminal.tty else _PIPED_WIDTH
        app = App(
            name=project.name,
            help=self.help,
            help_flags=(),
            version_flags=(),
            # Plain help for an agent: box drawing and colour cost it tokens and tell it nothing.
            help_formatter=terminal.formatter(),
            # Plain rows for an agent too: the panel of a bad option is a box of border characters.
            error_formatter=terminal.error_formatter(),
            # No `--no-<flag>` twin for every boolean: every flag here defaults to off.
            default_parameter=Parameter(negative=()),
            console=terminal.console(width=width),
            # Its own, since cyclopts derives an error console that detects color for itself.
            error_console=terminal.error_console(width=width),
        )
        typed = None if argv is None else next((t for t in argv if not t.startswith("-")), "")
        verbs = {
            command.name: getattr(import_module(command.module), self.name)[command.name]
            for command in self.commands
            if typed is None or command.name == typed
        }
        return self._bound(app, verbs)

    def entries(self, path: Words = ()) -> Iterator[tuple[Words, Node | Command]]:
        """Every node and typed command under this one, each with its words below this node.

        path: the words of this node itself, prefixed to every yielded path.
        """
        for command in self.commands:
            if not command.default:
                yield (*path, command.name), command
        for child in self.children:
            yield (*path, child.name), child
            yield from child.entries((*path, child.name))

    def node(self, path: str) -> Node:
        """The node at `path`, its words below this one (`data lake`); a KeyError for none."""
        found = self
        for word in path.split():
            found = {child.name: child for child in found.children}[word]
        return found

    def rewritten(self, argv: Sequence[str]) -> list[str]:
        """`argv` with a leading alias spelled as the path of the command it stands for."""
        for words, path in self.spellings:
            if tuple(argv[: len(words)]) == words:
                return [*path, *argv[len(words) :]]
        return list(argv)


class Typed(FrozenModel):
    """A typed command line read against a tree before cyclopts parses it: whether it runs the
    owner's act, and why it names nothing with what to type instead.

    The answer is the path a command moved to from the path typed, else the declared paths ending
    in the words typed from the unknown one on (`lake check` is `data lake check`), else the shown
    names nearest that word.

    node: the tree's root.
    """

    node: Node

    def _ending(self, words: Sequence[str], *, before: Sequence[str]) -> tuple[Words, list[Words]]:
        """The longest run of `words` some declared paths end in, and those paths.

        before: the words typed ahead of `words`; paths under them come first, then shown ones.
        """
        for end in range(len(words), 0, -1):
            tail = tuple(words[:end])
            found = sorted(
                (not entry.show, path[: len(before)] != tuple(before), path)
                for path, entry in self.node.entries()
                if path[-end:] == tail
            )
            if found:
                return (*before, *tail), [path for *_, path in found[:_GUESSES]]
        return (), []

    def _moved(self, argv: Sequence[str]) -> tuple[Words, list[Words]]:
        """The 0.5.7 path `argv` starts with and the path its command moved to, the longest."""
        options = (((option.name,), option) for option in self.node.options)
        found = sorted(
            (-len(words), words, path)
            for path, entry in (*self.node.entries(), *options)
            if isinstance(entry, Command)
            for words in (tuple(spelling.split()) for spelling in entry.moved)
            if tuple(argv[: len(words)]) == words
        )
        return (found[0][1], [found[0][2]]) if found else ((), [])

    def _nearest(self, word: str, under: Sequence[str]) -> list[Words]:
        """The shown paths whose last word reads most like `word`, at most three.

        Only those as near as the nearest, give or take `_TIE`, and at least `_NEAR`: `ingets`
        is `data lake ingest`, not `job digest` too. Of names equally near, those `under` the node
        it was typed at come first: `host lst` is `host list` before `job list`.
        """
        scored = sorted(
            (
                (SequenceMatcher(None, word, path[-1]).ratio(), path[:-1] == tuple(under), path)
                for path, entry in self.node.entries()
                if entry.show
            ),
            reverse=True,
        )
        near = max(_NEAR, (scored[0][0] if scored else 1.0) - _TIE)
        return [path for score, _, path in scored[:_GUESSES] if score >= near]

    def _unknown(self, argv: Sequence[str], at: int) -> str:
        """The refusal of `argv`, whose word `at` names nothing under the words before it."""
        words = list(takewhile(lambda word: not word.startswith("-"), argv[at:]))
        typed, guesses = self._moved(argv)
        if not guesses:
            typed, guesses = self._ending(words, before=argv[:at])
        if not guesses:
            typed, guesses = (*argv[: at + 1],), self._nearest(argv[at], argv[:at])
        nearest = " or ".join(f"`{' '.join((self.node.name, *path))}`" for path in guesses)
        listing = " ".join((self.node.name, *argv[:at], "--help"))
        guess = f"did you mean {nearest}?" if nearest else f"`{listing}` lists them"
        return f"no command `{' '.join((self.node.name, *typed))}`; {guess}"

    def owned(self, argv: Sequence[str]) -> str:
        """The words of the owner-scoped command `argv` runs, empty for any other or for help.

        Answered from the declaration alone, like `refusal`; a default command is named by its
        node's path.
        """
        if _META.intersection(argv):
            return ""
        node, rest = self.node, list(argv)
        while rest and (child := next((c for c in node.children if c.name == rest[0]), None)):
            node, rest = child, rest[1:]
        named = next((c for c in node.commands if rest and c.name == rest[0]), None)
        command = named or next((c for c in node.commands if c.default), None)
        if command is None or command.scope is not Scope.OWNER:
            return ""
        return " ".join(argv[: len(argv) - len(rest) + bool(named)])

    def refusal(self, argv: Sequence[str]) -> str:
        """Why `argv` names nothing, with where to look instead; empty when it names something.

        Answered from the declaration alone, before any verb module is imported: a node run
        alone takes whatever follows it, and the help and version flags are cyclopts' own.
        """
        node = self.node
        for at, token in enumerate(argv):
            if token in _META or node.ran:
                return ""
            child = next((child for child in node.children if child.name == token), None)
            if child is None:
                known = any(entry.name == token for entry in (*node.commands, *node.options))
                return "" if known else self._unknown(argv, at)
            node = child
        return ""

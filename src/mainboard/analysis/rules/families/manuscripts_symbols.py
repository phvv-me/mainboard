"""Reading the symbols one span of LaTeX mathematics names, as mcmr's kernel does.

A symbol is a single letter or a control word that does not shape mathematics, kept with its
styling (`\\mathbf{x}`), its subscript (`x_{i}`) and, as symbols of their own, the symbols of its
superscript. One regular expression cuts a span into tokens and SQL names each token.
"""

from ...catalog import Function
from .manuscripts_reader import GROUP, INSIDE, quoted

STRUCTURAL = [
    "Big",
    "Bigg",
    "Biggl",
    "Biggm",
    "Biggr",
    "Bigl",
    "Bigm",
    "Bigr",
    "Leftarrow",
    "Rightarrow",
    "approx",
    "big",
    "binom",
    "bigg",
    "bmatrix",
    "bmod",
    "cdot",
    "cdots",
    "circ",
    "colon",
    "coloneqq",
    "cup",
    "cap",
    "dfrac",
    "displaystyle",
    "dots",
    "biggl",
    "biggm",
    "biggr",
    "bigl",
    "bigm",
    "bigr",
    "emptyset",
    "equiv",
    "exists",
    "forall",
    "frac",
    "ge",
    "geq",
    "gets",
    "gg",
    "hat",
    "hbox",
    "in",
    "infty",
    "int",
    "iff",
    "implies",
    "label",
    "land",
    "langle",
    "lceil",
    "lfloor",
    "ldots",
    "le",
    "left",
    "leftarrow",
    "leq",
    "ll",
    "lor",
    "mapsto",
    "mathbb",
    "mathbf",
    "mathcal",
    "mathfrak",
    "mathit",
    "mathrm",
    "mathsf",
    "max",
    "mid",
    "min",
    "mspace",
    "neg",
    "ne",
    "neq",
    "nonumber",
    "not",
    "notin",
    "operatorname",
    "overline",
    "partial",
    "pmatrix",
    "pm",
    "prod",
    "propto",
    "qquad",
    "quad",
    "rangle",
    "rceil",
    "rfloor",
    "right",
    "rightarrow",
    "setminus",
    "sim",
    "simeq",
    "sqrt",
    "star",
    "subset",
    "subseteq",
    "substack",
    "sum",
    "text",
    "textstyle",
    "textrm",
    "tfrac",
    "tilde",
    "times",
    "to",
    "top",
    "underbrace",
    "underline",
    "vdots",
    "vec",
    "widehat",
    "widetilde",
] + ["ensuremath", "xspace"]  # mcmr counts these two macros' wrappers as symbols
OPERATORS = [
    "arccos",
    "arcsin",
    "arctan",
    "cos",
    "cosh",
    "det",
    "dim",
    "exp",
    "gcd",
    "inf",
    "ker",
    "lim",
    "liminf",
    "limsup",
    "ln",
    "log",
    "sin",
    "sinh",
    "sup",
    "tan",
    "tanh",
]
TEXTUAL = [
    "begin",
    "end",
    "hbox",
    "label",
    "mbox",
    "mathrm",
    "operatorname",
    "text",
    "textbf",
    "textit",
    "textrm",
    "textsc",
    "textup",
    "textcolor",
    "color",
]
STYLED = [
    "bar",
    "boldsymbol",
    "hat",
    "mathbb",
    "mathbf",
    "mathcal",
    "mathfrak",
    "mathit",
    "mathsf",
    "overline",
    "tilde",
    "vec",
    "widehat",
    "widetilde",
]

_GROUP = GROUP
_INSIDE = rf"\{{({INSIDE})\}}"
_ARGUMENT = r"(?:\\[A-Za-z]+|[A-Za-z])"
_SCRIPT_PIECE = (
    rf"[_^](?:{_GROUP}|\\(?:{'|'.join(STYLED)})\b\s*(?:{_GROUP}|{_ARGUMENT})"
    rf"|\\(?:{'|'.join(TEXTUAL)})\b\s*(?:{_GROUP})?|\\[A-Za-z]+\s*|(?s:.))"
)
_SCRIPTS = rf"(?:\s*{_SCRIPT_PIECE})*"
_SHAPING = [*STRUCTURAL, *OPERATORS, *TEXTUAL]


def _words(names: list[str]) -> str:
    return "(?:" + "|".join(names) + r")\b"


# What a span is cut at: a text command (with its group), a styled symbol, any other control word
# with the scripts behind it (a structural one leaves its scripts to be read as plain symbols,
# as mcmr does), a letter, a backslash alone. The first to match wins.
TOKEN = "|".join(
    [
        rf"\\{_words(TEXTUAL)}\s*(?:{_GROUP})?",
        rf"\\{_words(STYLED)}\s*{_GROUP}{_SCRIPTS}",
        rf"\\{_words(STYLED)}\s*{_ARGUMENT}{_SCRIPTS}",
        rf"\\(?:{'|'.join([*STRUCTURAL, *OPERATORS])})(?:[^A-Za-z\\]|$)",
        rf"\\[A-Za-z]+{_SCRIPTS}",
        rf"[A-Za-z]{_SCRIPTS}",
        r"\\",
    ]
)


def _headed(pattern: str) -> str:
    """The SQL pattern for a token that is the control word `name`, then `pattern`; matching the
    name itself keeps a shorter run of letters from standing in for it."""
    return f"('^' || chr(92) || chr(92) || name || {quoted(chr(92) + 's*' + pattern)})"


def _in(column: str, words: list[str]) -> str:
    return f"{column} IN ({', '.join(map(quoted, words))})"


def _let(name: str, value: str, body: str) -> str:
    """`body` with `name` standing for `value`, which SQL then evaluates once."""
    return f"list_transform([{value}], lambda {name}: {body})[1]"


def _lite() -> str:
    """The symbols of `text` without scripts: what a group nested two levels deep needs."""
    brace = f"regexp_extract(m, {_headed(_INSIDE)}, 1)"
    argument = f"regexp_extract(m, {_headed(f'({_ARGUMENT})')}, 1)"
    named = (
        "CASE WHEN NOT starts_with(m, chr(92)) THEN m "
        f"WHEN {_in('name', STYLED)} AND regexp_matches(m, {_headed(chr(92) + '{')}) "
        f"THEN chr(92) || name || '{{' || trim({brace}, ' ' || chr(9) || chr(10) || chr(13)) || '}}' "
        f"WHEN {_in('name', STYLED)} AND regexp_matches(m, {_headed(_ARGUMENT)}) "
        f"THEN chr(92) || name || '{{' || {argument} || '}}' "
        f"WHEN name = '' OR {_in('name', _SHAPING)} THEN '' ELSE chr(92) || name END"
    )
    tokens = "|".join(
        [
            rf"\\{_words(TEXTUAL)}\s*(?:{_GROUP})?",
            rf"\\{_words(STYLED)}\s*{_GROUP}",
            rf"\\{_words(STYLED)}\s*{_ARGUMENT}",
            r"\\[A-Za-z]+",
            r"[A-Za-z]",
            r"\\",
        ]
    )
    control = f"regexp_extract(m, {quoted(r'^\\([A-Za-z]+)')}, 1)"
    return (
        f"list_filter(list_transform(regexp_extract_all(text, {quoted(tokens)}), "
        f"lambda m: {_let('name', control, named)}), lambda x: x <> '')"
    )


def _named(inner: str) -> str:
    """The expression naming the symbols of `text`; `inner` names the macro reading the groups
    nested in a token's scripts and arguments, called once per token on all of them."""
    brace = f"regexp_extract(m, {_headed(_INSIDE)}, 1)"
    argument = f"regexp_extract(m, {_headed(f'({_ARGUMENT})')}, 1)"
    shaping_rest = f"regexp_replace(m, {quoted(r'^\\[A-Za-z]+')}, '')"
    # The token read before its nested groups are named: its base (null when it names nothing),
    # what is left of it for scripts, and the group a styled symbol wraps.
    token = (
        "CASE "
        "WHEN NOT starts_with(m, chr(92)) THEN "
        "{'base': left(m, 1), 'rest': substr(m, 2), 'wrapped': '', 'styled': false} "
        f"WHEN {_in('name', STYLED)} AND regexp_matches(m, {_headed(chr(92) + '{')}) THEN "
        f"{{'base': NULL, 'rest': regexp_replace(m, {_headed(_GROUP)}, ''), "
        f"'wrapped': {brace}, 'styled': true}} "
        f"WHEN {_in('name', STYLED)} AND regexp_matches(m, {_headed(_ARGUMENT)}) THEN "
        f"{{'base': CASE WHEN NOT (starts_with({argument}, chr(92)) "
        f"AND {_in(f'substr({argument}, 2)', _SHAPING)}) "
        f"THEN chr(92) || name || '{{' || {argument} || '}}' END, "
        f"'rest': regexp_replace(m, {_headed(_ARGUMENT)}, ''), 'wrapped': '', "
        "'styled': false} "
        f"WHEN name = '' OR {_in('name', TEXTUAL)} THEN "
        "{'base': NULL, 'rest': '', 'wrapped': '', 'styled': false} "
        f"WHEN {_in('name', [*STRUCTURAL, *OPERATORS])} THEN "
        f"{{'base': NULL, 'rest': {shaping_rest}, 'wrapped': '', 'styled': false}} "
        f"ELSE {{'base': chr(92) || name, 'rest': {shaping_rest}, 'wrapped': '', "
        "'styled': false} END"
    )
    pieces = f"regexp_extract_all(t.rest, {quoted(_SCRIPT_PIECE)})"
    # The text of the last subscript that needs its symbols named (a group, or a control word
    # with its argument), and of each superscript.
    subscript = f"list_filter({pieces}, lambda p: starts_with(p, '_'))[-1]"
    power_text = (
        "list_transform(list_filter(" + pieces + ", lambda p: starts_with(p, '^')), "
        "lambda p: CASE WHEN starts_with(substr(p, 2), '{') AND ends_with(p, '}') "
        "THEN substr(p, 3, length(p) - 3) ELSE substr(p, 2) END)"
    )
    kids = (
        f"list_concat([t.wrapped, CASE WHEN u IS NULL THEN '' "
        f"WHEN starts_with(substr(u, 2), '{{') THEN substr(u, 3, length(u) - 3) "
        f"WHEN starts_with(substr(u, 2), chr(92)) THEN substr(u, 2) ELSE '' END, "
        f"CASE WHEN t.base IS NULL AND NOT t.styled THEN t.rest ELSE '' END], "
        f"list_filter({power_text}, lambda g: g NOT IN "
        f"('\\mathsf{{T}}', '\\mathrm{{T}}', '\\top')))"
    )
    # Subscript `u` (a piece such as `_{i}`), `a` the names of its text.
    script = (
        "CASE WHEN u IS NULL THEN '' "
        "WHEN starts_with(substr(u, 2), chr(92)) THEN '_{' || coalesce(r[2][1], substr(u, 2)) || '}' "
        "WHEN starts_with(substr(u, 2), '{') THEN "
        + _let(
            "w",
            "CASE WHEN len(r[2]) = 1 AND contains(substr(u, 3, length(u) - 3), chr(92)) "
            "THEN r[2][1] ELSE regexp_replace(substr(u, 3, length(u) - 3), '\\s', '', 'g') END",
            "CASE WHEN length(w) = 1 THEN '_' || w ELSE '_{' || w || '}' END",
        )
        + " ELSE '_' || substr(u, 2) END"
    )
    wrapped = (
        "CASE WHEN t.styled THEN chr(92) || name || '{' || CASE WHEN len(r[1]) = 1 THEN r[1][1] "
        "ELSE trim(t.wrapped, ' ' || chr(9) || chr(10) || chr(13)) END || '}' "
        "ELSE t.base END"
    )
    named = (
        "CASE WHEN t.base IS NULL AND NOT t.styled THEN r[3] ELSE "
        f"list_concat([{wrapped} || {script}], flatten(list_slice(r, 4, len(r)))) END"
    )
    control = f"regexp_extract(m, {quoted(r'^\\([A-Za-z]+)')}, 1)"
    read = _let(
        "u",
        subscript,
        _let("r", f"list_transform({kids}, lambda x: {inner}(x))", named),
    )
    return (
        f"list_filter(flatten(list_transform(regexp_extract_all(text, {quoted(TOKEN)}), "
        f"lambda m: {_let('name', control, _let('t', token, read))})), lambda x: x <> '')"
    )


class ManuscriptSymbolsLite(Function):
    """The symbols of a group nested two levels deep, without their scripts."""

    text: str

    expression = _lite()


class ManuscriptSymbolsLeaf(Function):
    """The symbols of a group nested in a script, reading the groups inside it without scripts."""

    text: str

    expression = _named("manuscript_symbols_lite")


class ManuscriptSymbols(Function):
    """The symbols one span of mathematics names, each spelled the way a reader meets it; the
    groups nested in its scripts are read two levels deep."""

    text: str

    expression = _named("manuscript_symbols_leaf")


class ManuscriptNamesNothing(Function):
    """Whether a text names no symbol: after the words of text-mode commands (with their groups)
    and the structural control words are dropped, no letter or backslash is left."""

    text: str

    expression = (
        "NOT regexp_matches(regexp_replace(regexp_replace(text, "
        f"{quoted(rf'\\{_words(TEXTUAL)}\s*(?:{_GROUP})?')}, '', 'g'), "
        f"{quoted(rf'\\{_words([*STRUCTURAL, *OPERATORS])}')}, '', 'g'), '[A-Za-z\\\\]')"
    )

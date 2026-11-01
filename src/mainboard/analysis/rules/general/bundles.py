"""The rules that were several: each bundle reports what its parts report, so a project holds one
id and one table per family and configures a part as `[tool.mcmr.rules."ID".part_name]`."""

from ...catalog import Bundle, Example
from ..python.style.comprehensions import ComprehensionLoopCount
from ..python.testing.suite.coverage import OwnedTestStatementCount
from .classes.shape import AncestorCount, DeclaredFieldCount
from .control.flow import DeeplyNestedBody
from .dependencies.bindings import CrossLanguageBoundaryWidth, UnreachedCrossLanguageArtifact
from .dependencies.modules import ModuleLineCount, ModuleMemberCount, ModuleStatementCount
from .functions.flow import CognitiveComplexity
from .functions.size import FunctionStatementCount
from .idioms.overrides.rules import (
    AbstractMemberLeftUnimplemented,
    InheritedAttributeHidesAMethod,
    InitializerCalledOnAStranger,
    OverridingMethodAcceptsDifferentArguments,
    OverridingMethodChangesItsCallProtocol,
)
from .manuscript.layout.order import (
    ForwardReferenceToUnreadMaterial,
    SymbolUsedBeforeItIsIntroduced,
    TermUsedBeforeItIsMarked,
)
from .manuscript.notation import (
    NotationEntryAbsentFromTheBody,
    SymbolIntroducedUnderTwoMeanings,
    SymbolMissingFromTheNotationIndex,
)
from .manuscript.prose import (
    ParagraphLongerThanAReaderHolds,
    SectionTitleThatIsNotANounPhrase,
    SentenceLengthOutsideHouseBounds,
)
from .testing import TestModuleMemberCount
from .writing.comments_content import CommentLength

_SENTENCE = "This sentence is long enough to pass the lower bound of ten words easily"
_THEOREM = r"""\begin{theorem}\label{thm:main}
The claim holds in every case that we consider here today.
\end{theorem}"""


def _limited(bundle: str, *, part: str, **limits: int) -> dict[str, str]:
    """The project file that sets `limits` on one part of `bundle`."""
    settings = "".join(f"{name} = {value}\n" for name, value in limits.items())
    return {"pyproject.toml": f'[tool.mcmr.rules."{bundle}".{part}]\n{settings}'}


def _paper(body: str) -> str:
    """A LaTeX article whose body is `body`."""
    return f"\\documentclass{{article}}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"


class Ceiling(Bundle, id="ALL-CEIL0001"):
    """A count of code held at or under its maximum: a body's statements or complexity, a type's
    fields or ancestors, a module's lines, members or statements, the nesting of a body, the
    longest comment, the members of a test module."""

    ancestors: AncestorCount = AncestorCount()
    cognitive_complexity: CognitiveComplexity = CognitiveComplexity()
    comment_length: CommentLength = CommentLength()
    declared_fields: DeclaredFieldCount = DeclaredFieldCount()
    function_statements: FunctionStatementCount = FunctionStatementCount()
    module_lines: ModuleLineCount = ModuleLineCount()
    module_members: ModuleMemberCount = ModuleMemberCount()
    module_statements: ModuleStatementCount = ModuleStatementCount()
    nesting_depth: DeeplyNestedBody = DeeplyNestedBody()
    test_module_members: TestModuleMemberCount = TestModuleMemberCount()

    examples = (
        Example(
            bad="def run():\n"
            + "".join(f"    v{n} = {n}\n" for n in range(11))
            + "    return v0\n",
            good="def run():\n    return sum(range(11))\n",
        ),
        Example(
            bad="def run(a, b):\n    if a:\n        if b:\n            return 1\n    return 2\n",
            good="def run(a):\n    return a\n",
            context=_limited("ALL-CEIL0001", part="cognitive_complexity", maximum=1),
        ),
        Example(
            bad="class Row:\n    a: int\n    b: int\n",
            good="class Row:\n    a: int\n",
            context=_limited("ALL-CEIL0001", part="declared_fields", maximum=1),
        ),
        Example(
            bad="class A:\n    pass\n\n\nclass B(A):\n    pass\n\n\nclass C(B):\n    pass\n",
            good="class A:\n    pass\n\n\nclass B(A):\n    pass\n",
            context=_limited("ALL-CEIL0001", part="ancestors", maximum=1),
        ),
        Example(
            bad="# one two three four five six seven eight nine ten\n# eleven twelve\nVALUE = 1\n",
            good="# short\nVALUE = 1\n",
            context=_limited("ALL-CEIL0001", part="comment_length", maximum=5),
        ),
        Example(
            bad="# one two three four five six\n# seven eight nine ten\nVALUE = 1\n",
            good="# one two three four five six\nVALUE = 1  # seven eight nine ten\n",
            context=_limited("ALL-CEIL0001", part="comment_length", maximum=5),
        ),
        Example(
            bad=(
                """def run(a, b):
    if a:
        for x in b:
            if x:
                return x
    return None
"""
            ),
            good="def run(a):\n    if a:\n        return a\n    return None\n",
            context=_limited("ALL-CEIL0001", part="nesting_depth", maximum_depth=1),
        ),
        Example(
            bad="A = 1\nB = 2\nC = 3\nD = 4\n",
            good="A = 1\nB = 2\n",
            context=_limited("ALL-CEIL0001", part="module_lines", maximum=3),
        ),
        Example(
            bad="def a():\n    return 1\n\n\ndef b():\n    return 2\n",
            good="def a():\n    return 1\n",
            context=_limited("ALL-CEIL0001", part="module_members", maximum=1),
        ),
        Example(
            bad="A = 1\nB = 2\nC = 3\n",
            good="A = 1\n",
            context=_limited("ALL-CEIL0001", part="module_statements", maximum=2),
        ),
        Example(
            bad="def test_a():\n    assert 1\n\n\ndef test_b():\n    assert 2\n",
            good="def test_a():\n    assert 1\n",
            context=_limited("ALL-CEIL0001", part="test_module_members", maximum=1),
        ),
    )


class DefinedBeforeUse(Bundle, id="ALL-MANU0017"):
    """What a manuscript's reader meets before the manuscript has set it down: a reference to
    unread material, a symbol, a term."""

    forward_references: ForwardReferenceToUnreadMaterial = ForwardReferenceToUnreadMaterial()
    symbols: SymbolUsedBeforeItIsIntroduced = SymbolUsedBeforeItIsIntroduced()
    terms: TermUsedBeforeItIsMarked = TermUsedBeforeItIsMarked()

    examples = (
        Example(
            bad=_paper(
                f"\\section{{Method}}\nWe rely on Theorem~\\ref{{thm:main}}. {_SENTENCE}.\n{_THEOREM}"
            ),
            good=_paper(
                f"\\section{{Method}}\n{_THEOREM}\nWe rely on Theorem~\\ref{{thm:main}}. {_SENTENCE}."
            ),
            path="paper.tex",
        ),
        Example(
            bad=_paper(
                r"""\section{Method}
The quantity $\alpha$ grows, $\alpha$ shrinks and $\alpha$ stays put.
Let $\alpha$ be the rate."""
            ),
            good=_paper(
                r"""\section{Method}
Let $\alpha$ be the rate.
The quantity $\alpha$ grows, $\alpha$ shrinks and $\alpha$ stays put."""
            ),
            path="paper.tex",
        ),
        Example(
            bad=_paper(
                r"""\section{Method}
A lemniscate is used here, and a lemniscate is used again in this sentence.
We call this the \term{lemniscate} from now on."""
            ),
            good=_paper(
                r"""\section{Method}
We call this the \term{lemniscate} from now on.
A lemniscate is used here, and a lemniscate is used again in this sentence."""
            ),
            path="paper.tex",
        ),
    )


class InteropReach(Bundle, id="ALL-BIND0003", applicability="review"):
    """Artifacts one language declares that another language reaches too little or too widely."""

    boundary_width: CrossLanguageBoundaryWidth = CrossLanguageBoundaryWidth()
    unreached_artifacts: UnreachedCrossLanguageArtifact = UnreachedCrossLanguageArtifact()

    examples = (
        Example(
            bad="__device__ int one() { return 1; }\n\n__global__ void dead(int *p) {\n    p[0] = one();\n}\n",
            good="__device__ int one() { return 1; }\n",
            path="kernel.cu",
        ),
        Example(
            bad="__global__ void dead(int *p) {\n    p[0] = 1;\n}\n\nvoid run(int *p) {\n    dead<<<1, 1>>>(p);\n}\n",
            good="__global__ void dead(int *p) {\n    p[0] = 1;\n}\n\nvoid run(int *p) {\n    dead<<<1, 1>>>(p);\n}\n",
            path="kernel.cu",
        ),
        Example(
            bad="__global__ void dead(int *p) { p[0] = 1; } void other() {}\n",
            good="__global__ void dead(int *p) { p[0] = 1; } void other() {}\n",
            path="kernel.cu",
        ),
    )


class NotationIndex(Bundle, id="ALL-MANU0018"):
    """Where a manuscript's notation index and its body disagree: a symbol the index lacks, a row
    the body never sets, a symbol with two meanings."""

    absent_entries: NotationEntryAbsentFromTheBody = NotationEntryAbsentFromTheBody()
    double_meanings: SymbolIntroducedUnderTwoMeanings = SymbolIntroducedUnderTwoMeanings()
    missing_symbols: SymbolMissingFromTheNotationIndex = SymbolMissingFromTheNotationIndex()


class OverrideContract(Bundle, id="ALL-OVER0011"):
    """What an override breaks of the contract its base states: the arguments, the call protocol,
    an abstract member left open, an initializer run on a stranger, a method hidden by data."""

    abstract_members: AbstractMemberLeftUnimplemented = AbstractMemberLeftUnimplemented()
    call_protocol: OverridingMethodChangesItsCallProtocol = (
        OverridingMethodChangesItsCallProtocol()
    )
    different_arguments: OverridingMethodAcceptsDifferentArguments = (
        OverridingMethodAcceptsDifferentArguments()
    )
    hidden_methods: InheritedAttributeHidesAMethod = InheritedAttributeHidesAMethod()
    initializer_calls: InitializerCalledOnAStranger = InitializerCalledOnAStranger()

    examples = (
        Example(
            bad=(
                """class Base:
    def send(self, payload):
        return payload


class Child(Base):
    def send(self):
        return None
"""
            ),
            good=(
                """class Base:
    def send(self, payload):
        return payload


class Child(Base):
    def send(self, payload):
        return None
"""
            ),
        ),
        Example(
            bad=(
                """class Base:
    @property
    def size(self):
        return 1


class Child(Base):
    def size(self):
        return 2
"""
            ),
            good=(
                """class Base:
    @property
    def size(self):
        return 1


class Child(Base):
    @property
    def size(self):
        return 2
"""
            ),
        ),
        Example(
            bad=(
                """from abc import abstractmethod


class Base:
    @abstractmethod
    def run(self):
        raise NotImplementedError


class Child(Base):
    pass
"""
            ),
            good=(
                """from abc import abstractmethod


class Base:
    @abstractmethod
    def run(self):
        raise NotImplementedError


class Child(Base):
    def run(self):
        return 1
"""
            ),
        ),
        Example(
            bad=(
                """class Other:
    def __init__(self):
        self.a = 1


class Base:
    pass


class Child(Base):
    def __init__(self):
        Other.__init__(self)
"""
            ),
            good=(
                """class Other:
    def __init__(self):
        self.a = 1


class Base:
    pass


class Child(Base):
    def __init__(self):
        Base.__init__(self)
"""
            ),
        ),
        Example(
            bad=(
                """class Base:
    run = 1


class Child(Base):
    def run(self):
        return 2
"""
            ),
            good=(
                """class Base:
    run = 1


class Child(Base):
    def go(self):
        return 2
"""
            ),
        ),
    )


class ProseBound(Bundle, id="ALL-MANU0016"):
    """Prose outside the house bounds: a sentence, a paragraph or a section heading."""

    headings: SectionTitleThatIsNotANounPhrase = SectionTitleThatIsNotANounPhrase()
    paragraphs: ParagraphLongerThanAReaderHolds = ParagraphLongerThanAReaderHolds()
    sentences: SentenceLengthOutsideHouseBounds = SentenceLengthOutsideHouseBounds()

    examples = (
        Example(
            bad=_paper(f"\\section{{What is the method}}\n{_SENTENCE} for a reader."),
            good=_paper(f"\\section{{Method}}\n{_SENTENCE} for a reader."),
            path="paper.tex",
        ),
        Example(
            bad=_paper(f"\\section{{Method}}\n{_SENTENCE}. {_SENTENCE} again."),
            good=_paper(f"\\section{{Method}}\n{_SENTENCE}."),
            path="paper.tex",
            context=_limited("ALL-MANU0016", part="paragraphs", maximum_words=15),
        ),
        Example(
            bad=_paper(f"\\section{{Method}}\nShort one. {_SENTENCE}."),
            good=_paper(f"\\section{{Method}}\n{_SENTENCE}."),
            path="paper.tex",
        ),
    )


class PythonCeiling(Bundle, id="PY-CEIL0001"):
    """A count of Python code held at or under its maximum: the loops of a comprehension and the
    statements of a test."""

    comprehension_loops: ComprehensionLoopCount = ComprehensionLoopCount()
    test_statements: OwnedTestStatementCount = OwnedTestStatementCount()

    examples = (
        Example(
            bad="def pairs(a, b):\n    return [(x, y) for x in a for y in b]\n",
            good="def pairs(a):\n    return [x for x in a]\n",
            context=_limited("PY-CEIL0001", part="comprehension_loops", maximum=1),
        ),
        Example(
            bad="def test_run():\n    a = 1\n    b = 2\n    assert a + b == 3\n",
            good="def test_run():\n    assert 1 + 2 == 3\n",
            context=_limited("PY-CEIL0001", part="test_statements", maximum=2),
        ),
    )

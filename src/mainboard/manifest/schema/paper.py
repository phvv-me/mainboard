from ...core.base import Declared


class Paper(Declared):
    """One manuscript `center paper` builds and checks, in place of a hand-written build task.

    A venue's page rule is two numbers the build can check on every pass: how many pages the
    main text may take, and which section closes it. Declaring both here is what turns "count
    the pages by eye after each edit" into an exit status.

    dir: the directory holding the manuscript, workspace-relative; the build lands in `build/`
        inside it.
    main: the root `.tex` file inside `dir`.
    limit: the last page the main text may reach, 0 for a manuscript with no page rule.
    ends: the title of the section that must end on or before page `limit`, the conclusion
        of a main text whose references and appendix may follow it; empty checks no section.
    """

    dir: str
    main: str = "paper.tex"
    limit: int = 0
    ends: str = ""

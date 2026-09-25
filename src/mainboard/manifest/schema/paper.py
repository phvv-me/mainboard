from ...core.base import Declared


class Paper(Declared):
    """One manuscript `center paper` builds and checks against its venue's page rule.

    dir: workspace-relative; the build lands in its `build/`.
    main: the root `.tex` file inside `dir`.
    limit: the last page the main text may reach, 0 for no page rule.
    ends: the section title (the conclusion) that must end by page `limit`, empty for none.
    """

    dir: str
    main: str = "paper.tex"
    limit: int = 0
    ends: str = ""

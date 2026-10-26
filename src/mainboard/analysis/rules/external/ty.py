"""ty's type diagnostics as findings: the inference mcmr says only a type checker can do."""

from ...catalog import Example, Rule


class TypeDiagnostics(Rule, id="PY-TY0001"):
    """Type errors and warnings ty finds, each named by its ty rule."""

    ignored: tuple[str, ...] = ()

    query = """
        SELECT * FROM ty_diagnostics WHERE NOT list_contains($ignored::VARCHAR[], code)
    """
    message = "{code}: {message}"
    examples = (Example(bad='x: int = "text"\n'),)

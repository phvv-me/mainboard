"""Rules over command-line construction (mcmr's PY-CLI family)."""

from ...catalog import Rule


class ArgparseCliConstruction(Rule, id="PY-CLI0001"):
    """CLI parsers that bypass the configured Cyclopts foundation."""

    query = """
        SELECT path, start_line, start_column FROM core_call
        WHERE qualified_name = 'argparse.ArgumentParser'
    """
    message = (
        "`argparse.ArgumentParser` builds a second CLI schema instead of exposing typed "
        "callables through Cyclopts"
    )

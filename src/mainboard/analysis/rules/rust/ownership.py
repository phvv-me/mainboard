"""Rules over where Rust modules copy values (mcmr's RS-OWNE family)."""

from ...catalog import Rule

_CLONES = """
    SELECT f.path, c.line AS start_line, 0 AS start_column,
           CASE WHEN c.owner = '' THEN f.path ELSE c.owner END AS owner,
           CASE WHEN c.receiver = '' THEN 'a value' ELSE c.receiver END AS receiver
    FROM rust_surface_clones c JOIN rust_surface_facts f USING (fact_order, fact_id)
"""


class CloneInsideLoop(Rule, id="RS-OWNE0001"):
    """Clones made inside a loop, paying for the copy again on every pass."""

    maximum: int | None = None

    query = (
        f"{_CLONES} WHERE c.loop_depth > 0 "
        "QUALIFY count(*) OVER (PARTITION BY c.fact_id) > $maximum"
    )
    message = (
        "`{owner}` copies `{receiver}` inside a loop, so the copy is paid again on every pass"
    )


class CloneCallCount(Rule, id="RS-OWNE0002"):
    """Every explicit clone, counted per module."""

    maximum: int | None = None

    query = f"{_CLONES} WHERE f.clones_length > $maximum"
    message = "`{owner}` copies `{receiver}` explicitly"

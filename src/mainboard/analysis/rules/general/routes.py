"""Rules over the routes a repository declares (mcmr's ALL-ROUT family)."""

from ...catalog import Relation, Rule


class RoutesDeclared(Relation):
    """The `routes_declared()` relation."""

    query = """
    SELECT fact_id, record_id, ordinal, method, path AS route, declared_in, line,
           is_prefix_composed, references_length
    FROM route_routes
"""


class DuplicateRouteDeclaration(Rule, id="ALL-ROUT0001"):
    """Routes declared more than once for the same method and path."""

    query = """
        SELECT arg_min(declared_in, ordinal) AS path, arg_min(line, ordinal) AS start_line,
               0 AS start_column, method, route, count(*) AS count,
               string_agg(declared_in || ':' || line, '`, `' ORDER BY ordinal) AS sites
        FROM routes_declared() WHERE NOT is_prefix_composed
        GROUP BY fact_id, method, route HAVING count(*) > 1
    """
    message = "`{method} {route}` is declared {count} times in `{sites}`"


class UnreachedDeclaredRoute(Rule, id="ALL-ROUT0002"):
    """Routes no client in this repository names, where other routes are named."""

    query = """
        WITH judged AS (
          SELECT *, max(references_length) OVER (PARTITION BY fact_id) AS most
          FROM routes_declared() WHERE NOT is_prefix_composed AND NOT contains(route, '{'))
        SELECT declared_in AS path, line AS start_line, 0 AS start_column, method, route
        FROM judged WHERE references_length = 0 AND most > 0
    """
    message = "`{method} {route}` is declared here but no retained client reference names its path"


class InconsistentRoutePathStyle(Rule, id="ALL-ROUT0003"):
    """Route segments spelled against the convention the rest of the paths follow."""

    query = """
        WITH spelled AS (
          SELECT fact_id, record_id, count(*) FILTER (WHERE contains(segment, '-')) AS hyphenated,
                 count(*) FILTER (WHERE contains(segment, '_')) AS underscored
          FROM (SELECT fact_id, record_id, unnest(str_split(route, '/')) AS segment
                FROM routes_declared())
          WHERE segment <> '' AND NOT contains(segment, '{')
          GROUP BY ALL
        ), contested AS (
          SELECT *, hyphenated_total > underscored_total AS hyphens_win FROM (
            SELECT *, sum(hyphenated) OVER (PARTITION BY fact_id) AS hyphenated_total,
                   sum(underscored) OVER (PARTITION BY fact_id) AS underscored_total
            FROM spelled)
          WHERE hyphenated_total > 0 AND underscored_total > 0
            AND hyphenated_total <> underscored_total)
        SELECT r.declared_in AS path, r.line AS start_line, 0 AS start_column, r.route,
               CASE WHEN hyphens_win THEN '_' ELSE '-' END AS unexpected,
               CASE WHEN hyphens_win THEN '-' ELSE '_' END AS expected,
               CASE WHEN hyphens_win THEN underscored ELSE hyphenated END AS amount
        FROM routes_declared() r JOIN contested USING (fact_id, record_id)
        WHERE CASE WHEN hyphens_win THEN underscored ELSE hyphenated END > 0
    """
    message = (
        "`{route}` uses `{unexpected}` in {amount} segments "
        "while the repository route convention uses `{expected}`"
    )

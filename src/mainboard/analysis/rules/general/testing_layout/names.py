"""ALL-TEST0006: tests whose name states too few words to state a behavior."""

from ....catalog import Example, Rule
from .constants import ANCHOR

_PYTHON = """\
def test_it_works():
    pass


class TestLake:
    def test_reads(self):
        pass


def check_it():
    pass
"""
_PYTHON_FIXED = """\
def test_a_lake_verb_refuses_a_path_outside_the_workspace():
    pass


class TestLake:
    def test_a_lake_verb_reads_a_path_inside_the_workspace(self):
        pass


def check_it():
    pass
"""
_SWIFT = """\
import XCTest
import Testing

final class FooTests: XCTestCase {
    func testWorks() {}
    func testReturnsTheEmptyListWhenNoItemsAreGiven() {}
}

@Test func works() {}
@Test("returns the empty list when no items are given") func empty() {}
func helper() {}
"""
_SWIFT_FIXED = """\
import XCTest
import Testing

final class FooTests: XCTestCase {
    func testReturnsTheSumOfTwoNumbersGiven() {}
    func testReturnsTheEmptyListWhenNoItemsAreGiven() {}
}

@Test func returnsTheEmptyListWhenNoItemsAreGiven() {}
@Test("returns the empty list when no items are given") func empty() {}
func helper() {}
"""
_RUST = """\
fn helper() {}

#[cfg(test)]
mod tests {
    #[test]
    fn works() {}

    #[test]
    #[should_panic]
    fn adds_two_numbers_together_and_returns_the_sum() {}
}
"""
_RUST_FIXED = """\
fn helper() {}

#[cfg(test)]
mod tests {
    #[test]
    fn adds_two_numbers_together_and_returns_the_sum_once() {}

    #[test]
    #[should_panic]
    fn adds_two_numbers_together_and_returns_the_sum() {}
}
"""
_TYPESCRIPT = """\
describe("foo", () => {
  it("works", () => {});
  it.each([1, 2])("handles %i", () => {});
  test("returns the sum of two numbers in the list", () => {});
});
"""
_TYPESCRIPT_FIXED = """\
describe("foo", () => {
  it("returns the sum of two numbers in the list", () => {});
  it.each([1, 2])("handles the number %i given to the sum", () => {});
  test("returns the sum of two numbers in the list", () => {});
});
"""


class TestNameWords(Rule, id="ALL-TEST0006"):
    """Collected tests whose name, after the framework's prefix, states fewer than
    `minimum_words` words. A test name is a sentence about one behavior
    (`test_a_lake_verb_refuses_a_path_outside_the_workspace`), so a stranger reads what broke from
    the failure line alone; `test_it_works` says nothing.

    Python reads the collected tests (`test_` prefix, words split on `_`), Swift the XCTest
    `func testX` of an `XCTestCase` and the `@Test` functions of Swift Testing (the display name
    when it has one, else the camel-cased words of the name), Rust the `#[test]` functions
    (words split on `_`) and TypeScript the titles of `it(...)` and `test(...)` calls in test
    files. There is no fix: the words that state the behavior are a sentence only the author can
    write, and a generated one would pass the count while saying nothing."""

    minimum_words: int = 5

    query = r"""
        WITH python AS (
          SELECT t.path, t.node_span_start_line AS start_line,
                 t.node_span_start_column AS start_column, t.name,
                 string_split(regexp_replace(t.name, '^test_', ''), '_') AS words
          FROM test_function_tests t WHERE t.is_collected
        ), swift AS (
          SELECT f.path, f.start_line, f.start_column, f.name,
                 string_split(regexp_replace(regexp_replace(
                   coalesce(d.title, regexp_replace(f.name, '^test', '')),
                   '([a-z0-9])([A-Z])', '\1 \2', 'g'), '[^A-Za-z0-9]+', ' ', 'g'), ' ') AS words
          FROM ast_nodes f
          LEFT JOIN (SELECT a.fact_id, m.parent_id AS function, any_value(s.name) AS title
                     FROM ast_nodes a
                     JOIN ast_nodes m ON m.fact_id = a.fact_id AND m.node_id = a.parent_id
                                     AND m.type = 'modifiers'
                     JOIN ast_nodes u ON u.fact_id = a.fact_id AND u.parent_id = a.node_id
                                     AND u.type = 'user_type' AND u.name = 'Test'
                     LEFT JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
                                          AND l.type = 'line_string_literal'
                     LEFT JOIN ast_nodes s ON s.fact_id = a.fact_id AND s.parent_id = l.node_id
                                          AND s.type = 'line_str_text'
                     WHERE a.type = 'attribute'
                     GROUP BY a.fact_id, m.parent_id) d
            ON d.fact_id = f.fact_id AND d.function = f.node_id
          WHERE f.type = 'function_declaration'
            AND (d.function IS NOT NULL
                 OR (regexp_matches(f.name, '^test[A-Z0-9_]')
                     AND EXISTS (SELECT 1 FROM ast_nodes b
                                 JOIN ast_nodes c ON c.fact_id = b.fact_id
                                                 AND c.node_id = b.parent_id
                                                 AND c.type = 'class_declaration'
                                 JOIN ast_nodes i ON i.fact_id = c.fact_id
                                                 AND i.parent_id = c.node_id
                                                 AND i.type = 'inheritance_specifier'
                                 JOIN ast_nodes u ON u.fact_id = c.fact_id
                                                 AND u.parent_id = i.node_id
                                                 AND u.type = 'user_type'
                                 JOIN ast_nodes n ON n.fact_id = c.fact_id
                                                 AND n.parent_id = u.node_id
                                                 AND n.name = 'XCTestCase'
                                 WHERE b.fact_id = f.fact_id AND b.node_id = f.parent_id)))
        ), rust AS (
          SELECT f.path, f.start_line, f.start_column, f.name,
                 string_split(regexp_replace(f.name, '^test_', ''), '_') AS words
          FROM ast_nodes f
          WHERE f.type = 'function_item' AND EXISTS (
            SELECT 1 FROM ast_nodes a
            WHERE a.fact_id = f.fact_id AND a.parent_id = f.parent_id
              AND a.type = 'attribute_item' AND a.sibling_index < f.sibling_index
              AND regexp_matches(a.name, '^#\[(?:[A-Za-z_]+::)*test(?:\(.*\))?\]$')
              AND NOT EXISTS (SELECT 1 FROM ast_nodes b
                              WHERE b.fact_id = f.fact_id AND b.parent_id = f.parent_id
                                AND b.sibling_index > a.sibling_index
                                AND b.sibling_index < f.sibling_index
                                AND b.type NOT IN ('attribute_item', 'line_comment',
                                                   'block_comment')))
        ), callees AS (
          SELECT c.fact_id, c.node_id AS call, f.name AS root
          FROM ast_nodes c
          JOIN ast_nodes f ON f.fact_id = c.fact_id AND f.parent_id = c.node_id
                          AND f.sibling_index = 0 AND f.type = 'identifier'
          WHERE c.type = 'call_expression'
          UNION ALL
          SELECT c.fact_id, c.node_id, i.name
          FROM ast_nodes c
          JOIN ast_nodes m ON m.fact_id = c.fact_id AND m.parent_id = c.node_id
                          AND m.sibling_index = 0 AND m.type = 'member_expression'
          JOIN ast_nodes i ON i.fact_id = c.fact_id AND i.parent_id = m.node_id
                          AND i.sibling_index = 0 AND i.type = 'identifier'
          WHERE c.type = 'call_expression'
          UNION ALL
          SELECT c.fact_id, c.node_id, i.name
          FROM ast_nodes c
          JOIN ast_nodes e ON e.fact_id = c.fact_id AND e.parent_id = c.node_id
                          AND e.sibling_index = 0 AND e.type = 'call_expression'
          JOIN ast_nodes m ON m.fact_id = c.fact_id AND m.parent_id = e.node_id
                          AND m.sibling_index = 0 AND m.type = 'member_expression'
          JOIN ast_nodes i ON i.fact_id = c.fact_id AND i.parent_id = m.node_id
                          AND i.sibling_index = 0 AND i.type = 'identifier'
          WHERE c.type = 'call_expression'
        ), typescript AS (
          SELECT c.path, c.start_line, c.start_column, substr(t.name, 2, len(t.name) - 2) AS name,
                 string_split(trim(substr(t.name, 2, len(t.name) - 2)), ' ') AS words
          FROM callees k
          JOIN ast_nodes c ON c.fact_id = k.fact_id AND c.node_id = k.call
          JOIN ast_nodes a ON a.fact_id = c.fact_id AND a.parent_id = c.node_id
                          AND a.type = 'arguments'
          JOIN ast_nodes t ON t.fact_id = c.fact_id AND t.parent_id = a.node_id
                          AND t.type IN ('string', 'template_string') AND t.sibling_index = 1
          WHERE k.root IN ('it', 'test')
            AND regexp_matches(c.path, '(\.(test|spec)\.[cm]?[jt]sx?$)|(__tests__/)')
        )
        SELECT path, start_line, start_column, name,
               len(list_filter(words, lambda word: word <> '')) AS words
        FROM (SELECT * FROM python UNION ALL SELECT * FROM swift UNION ALL SELECT * FROM rust
              UNION ALL SELECT * FROM typescript)
        WHERE len(list_filter(words, lambda word: word <> '')) < $minimum_words
    """
    message = (
        "`{name}` states its behavior in {words} words, fewer than the {minimum_words} needed"
    )
    examples = (
        Example(bad=_PYTHON, good=_PYTHON_FIXED, path="tests/test_core.py"),
        Example(
            bad=_SWIFT,
            good=_SWIFT_FIXED,
            path="Tests/FooTests/FooTests.swift",
            context=ANCHOR,
        ),
        Example(bad=_RUST, good=_RUST_FIXED, path="src/lib.rs", context=ANCHOR),
        Example(
            bad=_TYPESCRIPT,
            good=_TYPESCRIPT_FIXED,
            path="web/a.test.ts",
            context=ANCHOR,
        ),
    )

"""Agent remediation stays available offline and does not imply an automatic migration."""

import json
from collections.abc import Mapping

import pytest


def _catalog(mb) -> list[dict]:
    """Every rule `explain --json` lists."""
    listed = mb("code", "rules", "explain", "--json")
    assert listed.code == 0, listed.err
    return json.loads(listed.out)["rules"]


@pytest.fixture
def rule(mb) -> dict:
    """The first rule of the catalog that cites references."""
    return next(entry for entry in _catalog(mb) if entry["references"])


def test_explain_without_a_rule_lists_the_catalog_and_some_rules_cite_references(mb) -> None:
    assert any(entry["references"] for entry in _catalog(mb))


def test_explain_as_json_reports_the_rule_as_the_catalog_lists_it(mb, rule: Mapping) -> None:
    explained = mb("code", "rules", "explain", rule["id"], "--json")
    assert explained.code == 0, explained.err
    assert json.loads(explained.out)["rule"] == rule


def test_explain_reports_rule_evidence_and_repair_policy(mb, rule: Mapping) -> None:
    text = mb("code", "rules", "explain", rule["id"])
    assert text.code == 0, text.err
    assert rule["summary"] in text.out
    assert rule["guidance"] in text.out
    assert all(url in text.out for url in rule["references"])
    assert f"fix: {rule['fix']}" in text.out


def test_unknown_rule_explanation_fails_without_a_traceback(mb) -> None:
    result = mb("code", "rules", "explain", "SW-NO-SUCH-RULE")
    assert result.code == 1
    assert "no rule" in result.err and "`mb code rules explain` lists them" in result.err
    assert "Traceback" not in result.err

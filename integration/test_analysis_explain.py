"""Agent remediation stays available offline and does not imply an automatic migration."""

import json


def test_explain_reports_rule_evidence_and_repair_policy(mb) -> None:
    listed = mb("rules", "catalog", "--json")
    assert listed.code == 0, listed.err
    entries = json.loads(listed.out)["rules"]
    rule = next(entry for entry in entries if entry["references"])
    explained = mb("rules", "explain", rule["id"], "--json")
    assert explained.code == 0, explained.err
    assert json.loads(explained.out)["rule"] == rule
    text = mb("rules", "explain", rule["id"])
    assert text.code == 0, text.err
    assert rule["summary"] in text.out
    assert rule["guidance"] in text.out
    assert all(url in text.out for url in rule["references"])
    assert f"fix: {rule['fix']}" in text.out


def test_unknown_rule_explanation_fails_without_a_traceback(mb) -> None:
    result = mb("rules", "explain", "SW-NO-SUCH-RULE")
    assert result.code == 1
    assert "no rule" in result.err and "rules catalog" in result.err
    assert "Traceback" not in result.err

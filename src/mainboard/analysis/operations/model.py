# The names a small model gives: the one judgment an operation cannot derive from the code, asked
# through the `claude` CLI under a JSON schema, so the edits around a name stay deterministic.

import json
import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the claude CLI with fixed argv since=2026-10-25
import tempfile
from hashlib import sha256
from pathlib import Path

# The model asked, and how long one question may take before it counts as unanswered.
_MODEL = "claude-haiku-5-5"
_SECONDS = 180
# Every answer by the digest of the model and its question, so a fix pass asks each once, and
# the names given so far to each literal, so a value means the same name across a project.
_CACHE = Path.home() / ".cache/mainboard/analysis/names.json"


class Namer:
    """Identifiers asked of a small model, one question answering a list of them in order."""

    @staticmethod
    def _asked(question: str, count: int) -> list[str] | None:
        """The model's `count` names for `question`, None when it gives no such answer."""
        items = {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"}
        names = {"type": "array", "items": items, "minItems": count, "maxItems": count}
        schema = {"type": "object", "properties": {"names": names}, "required": ["names"]}
        command = ["claude", "--print", "--model", _MODEL, "--output-format", "json"]
        command += ["--json-schema", json.dumps(schema), "--tools", "", "--strict-mcp-config"]
        command += ["--safe-mode", "--no-session-persistence"]
        with tempfile.TemporaryDirectory(prefix="mb-namer-") as folder:
            try:
                ran = subprocess.run(
                    command,
                    input=question,
                    capture_output=True,
                    text=True,
                    cwd=folder,
                    timeout=_SECONDS,
                    check=True,
                )
            except FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired:
                return None
        answer = (json.loads(ran.stdout).get("structured_output") or {}).get("names")
        return answer if isinstance(answer, list) and len(answer) == count else None

    def __init__(self) -> None:
        try:
            stored = json.loads(_CACHE.read_text(encoding="utf-8"))
        except FileNotFoundError, json.JSONDecodeError:
            stored = {}
        self.answers: dict[str, list[str]] = stored.get("answers", {})
        self.given: dict[str, list[str]] = stored.get("given", {})

    def _save(self) -> None:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        stored = {"answers": self.answers, "given": self.given}
        staged = _CACHE.with_suffix(f".{os.getpid()}.tmp")
        staged.write_text(json.dumps(stored, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(staged, _CACHE)

    def names(self, question: str, count: int) -> list[str] | None:
        """`count` names for `question`, from the cache or else the model."""
        key = sha256(f"{_MODEL}\n{count}\n{question}".encode()).hexdigest()
        if key not in self.answers:
            answer = Namer._asked(question, count)
            if answer is None:
                return None
            self.answers[key] = answer
            self._save()
        return self.answers[key]

    def remember(self, literal: str, *, name: str) -> None:
        """Record that `literal` was named `name`, for the questions about it that follow."""
        if name not in self.given.setdefault(literal, []):
            self.given[literal].append(name)
            self._save()

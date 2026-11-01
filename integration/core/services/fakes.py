from collections.abc import Mapping, Sequence
from pathlib import Path


class Scripted:
    """The service managers of one machine as a double for `Shell`, every command recorded.

    A command the script names is answered as scripted. Any other is answered by the managers
    as they remember what they were told: launchd holds a label from a `bootstrap` until a
    `bootout`, systemd reports a unit active from an `enable --now` until a `stop` or
    `disable`, `loginctl` says this user lingers, and the rest say nothing and refuse nothing.

    answers: an argv prefix to the `(status, output)` its commands get.
    loaded: the labels launchd holds right now.
    active: the unit names systemd reports active right now.
    """

    def __call__(self, command: Sequence[str]) -> tuple[int, str]:
        said = list(command)
        self.asked.append(said)
        return self.scripted(said) or self.answer(said)

    def __init__(self, answers: Mapping[tuple[str, ...], tuple[int, str]] | None = None) -> None:
        self.answers = dict(answers or {})
        self.asked: list[list[str]] = []
        self.loaded: set[str] = set()
        self.active: set[str] = set()

    def _bootout(self, label: str) -> tuple[int, str]:
        if label not in self.loaded:
            return 3, "Boot-out failed: 3: No such process"
        self.loaded.discard(label)
        return 0, ""

    def _bootstrap(self, label: str) -> tuple[int, str]:
        if label in self.loaded:
            return 5, "Bootstrap failed: 5: Input/output error"
        self.loaded.add(label)
        return 0, ""

    def _launchd(self, verb: str, *words: str) -> tuple[int, str]:
        """A launchd command, its label the last word of its target or the stem of its plist."""
        label = Path(words[-1]).stem if verb == "bootstrap" else words[-1].rpartition("/")[2]
        if verb == "bootstrap":
            return self._bootstrap(label)
        if verb == "bootout":
            return self._bootout(label)
        # A domain is `gui/501`, a service `gui/501/label`.
        absent = verb == "print" and words[0].count("/") > 1 and label not in self.loaded
        return (113, "Could not find service") if absent else (0, "")

    def _systemd(self, verb: str, *words: str) -> tuple[int, str]:
        names = [word for word in words if not word.startswith("--")]
        if verb == "enable":
            self.active.update(names)
        elif verb in {"disable", "stop"}:
            self.active.difference_update(names)
        elif verb == "show":
            return 0, f"ActiveState={'active' if names[0] in self.active else 'inactive'}\n"
        return 0, ""

    def answer(self, said: Sequence[str]) -> tuple[int, str]:
        """What the managers say to a command the script names no answer for."""
        program, *words = said
        if program == "launchctl":
            return self._launchd(*words)
        if program == "systemctl":
            return self._systemd(*words[1:])
        return (0, "Linger=yes") if program == "loginctl" else (0, "")

    def scripted(self, said: Sequence[str]) -> tuple[int, str] | None:
        """The answer the script names for `said`, None when it names none."""
        return next(
            (
                answer
                for prefix, answer in self.answers.items()
                if list(said[: len(prefix)]) == list(prefix)
            ),
            None,
        )

    def user_verbs(self) -> list[str]:
        """The verb of every `systemctl --user` command asked, in order."""
        return [said[2] for said in self.asked if said[:2] == ["systemctl", "--user"]]

    def verbs(self, program: str) -> list[str]:
        """The first argument of every command `program` was asked, in order."""
        return [said[1] for said in self.asked if said[0] == program and len(said) > 1]

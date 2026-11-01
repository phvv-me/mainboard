"""The dispatch constants other modules share."""

from ..core.project import Project

COMMIT_VAR = Project().variable("SOURCE_COMMIT")

# Where a job keeps what it resumes from, the same directory on every attempt of one run name,
# and which attempt this is (1 first).
CHECKPOINT_VAR = Project().variable("CHECKPOINT")
ATTEMPT_VAR = Project().variable("ATTEMPT")

# Host/device coordination is never transferable, even under an explicit include rule, and a
# mirror never removes one a host holds.
CARD_LEASES = (".card.lock", ".card.lock.*")

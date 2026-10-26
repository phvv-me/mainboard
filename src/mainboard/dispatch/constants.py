"""The dispatch constants other modules share."""

from ..core.project import Project

# How often a waiter asks a job's host for its state, in seconds.
POLL_SECONDS = 5.0
# How long a `wait` blocks before handing the shell back with the job still in flight. Finite
# because an unbounded waiter once sat two days on a job owed by a host that had left the
# network; 0 means forever.
WAIT_SECONDS = 3600.0

# The variables a run reads its provenance from, here because the dispatch exporting them and the
# receipt reading them sit at opposite ends of the package. Each is exported under every name the
# tool answers to, since the job may import an older release than the one that dispatched it.
SOURCE_VAR = Project().variable("SOURCE")
COMMIT_VAR = Project().variable("SOURCE_COMMIT")
DIGEST_VAR = Project().variable("SOURCE_DIGEST")
CLOSURE_VAR = Project().variable("CLOSURE")
FIRST_PARTY_VAR = Project().variable("FIRST_PARTY")
DEFERRED_VAR = Project().variable("DEFERRED")

# Where a job keeps what it resumes from, the same directory on every attempt of one run name,
# and which attempt this is (1 first).
CHECKPOINT_VAR = Project().variable("CHECKPOINT")
ATTEMPT_VAR = Project().variable("ATTEMPT")

# The variable naming a dispatched job's receipts file.
RECEIPTS_VAR = Project().variable("RECEIPTS")

# Host/device coordination is never transferable, even under an explicit include rule, and a
# mirror never removes one a host holds.
CARD_LEASES = (".card.lock", ".card.lock.*")

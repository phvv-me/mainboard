"""The constants the modules of `mainboard` share."""

from .core.project import Project

# The context key naming an event's sink, and every registered sink by its key.
SINK = "mb_sink"

# The alias every attach uses, so view definitions and callers' SQL name one catalog.
ALIAS = "lake"
# What a read of table artifacts answers when none holds the schema: no rows, and the `_trial`
# provenance column every such read adds beside the table's own.
NO_TABLE = "SELECT NULL::VARCHAR AS _trial WHERE false"
# The event an ask is spooled as, and the attributes naming the act it asks for and its target:
# the policy files asks under these names, and the telemetry lake's `asks` view reads them back.
ASK_EVENT = "mb.policy.ask"
ASK_ACT = "mb.ask.act"
ASK_TARGET = "mb.ask.target"

# The longest a grant may last, in hours: a year, past which a grant is a standing allowance the
# owner declares in the file by hand.
LONGEST_GRANT_HOURS = 24 * 365

# Whether the reap pass ends and binds the rentals it proves ours and orphaned, or only logs what
# it would do. The owner flips it, with the `reap` service's `off`, once the canaries have shown
# that a rented machine's label comes back from every provider's listing as it was sent.
REAP_DELETES = False

# The attribute a spool line carries its event's id in, OpenTelemetry's own for a log record's
# unique id, so the lake holds it as the sink stored it and a delivery is checked against it.
EVENT_ID = "log.record.uid"
# What a job's output says as its host starts emitting, followed by the spool file's name: the
# center reads the output it already keeps to learn which file belongs to which run, since the
# runner cannot know the scheduler's handle (pueue exports no task id to its tasks).
FRAMES_MARK = "mainboard: frames "

# What a workspace holds as data rather than source, in the sync filter language: never sealed or
# archived as a local trial's source, and pinned through `needs` or `resources` by what reads it.
# The same places `[git] never-commit` keeps out of commits, so a source package named `datasets`
# or `evidence` stays source.
DATA = ("/datasets/", "/research/*/datasets/", "/references/", "**/experiments/**/evidence/")

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
DIGEST_VAR = Project().variable("SOURCE_DIGEST")
CLOSURE_VAR = Project().variable("CLOSURE")
FIRST_PARTY_VAR = Project().variable("FIRST_PARTY")
DEFERRED_VAR = Project().variable("DEFERRED")
# The variable naming a dispatched job's receipts file.
RECEIPTS_VAR = Project().variable("RECEIPTS")

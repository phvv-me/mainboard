"""The constants of `monitor` other modules share."""

from typing import Final

from ...dispatch.jobs import vocabulary

# The routing answer for the schedulers reached over ssh, the one family whose whole host can be
# asked about in a single query. A provider has no such listing and is asked run by run.
QUEUED: Final = "ssh-family"

# A provider creation with no confirmed handle is neither probed nor resubmitted, only reported.
UNCREATED: Final[dict[str, str]] = {
    vocabulary.Word.PREPARED: (
        "provider creation prepared; no create attempted; cancel if abandoned"
    ),
    vocabulary.Word.SUBMITTING: "provider creation has no confirmed handle; reconcile its exact "
    "label before retrying; no automatic resubmission or release",
}

# The delivery statuses that say a run's evidence already arrived.
COPIED: Final = frozenset({"copied", "verified"})

"""The constants of `reaping` other modules share."""

from datetime import timedelta
from typing import Final

# What the reap pass leaves to the process that owns a creation or a deadline before it steps in:
# a create answers in seconds, and `Deadlines` ends an expired lease on the settler's next pass.
GRACE: Final = timedelta(minutes=10)

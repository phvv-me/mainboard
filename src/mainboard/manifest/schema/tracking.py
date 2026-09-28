from ...core.base import Declared


class Tracking(Declared):
    """The live machine series a dispatched job publishes into its own receipts.

    interval: seconds between live machine samples, 0 for none (and no attestation either).
    """

    interval: float = 10.0

    @property
    def on(self) -> bool:
        """Whether a dispatched job samples and attests its own machine."""
        return self.interval > 0

from mainboard.costs import Observation, Offer, Verdict


def landings(
    kind: str, *, gpu: str = "H100", ready_s: float = 60.0, **ended: int
) -> list[Observation]:
    """Recorded landings of provider `kind`, `ended` counting them by how each ended.

    `landings("p", landed=3, slow=1)` is three that delivered and one that billed, then failed. A
    refused create allocated nothing, so it carries no ready time.
    """
    return [
        Observation(
            provider=kind,
            gpu=gpu,
            t_submit=0.0,
            verdict=Verdict(verdict.replace("_", "-")),
            ready_s=0.0 if verdict == Verdict.REFUSED else ready_s,
        )
        for verdict, count in ended.items()
        for _ in range(count)
    ]


def offer(
    kind: str,
    *,
    rate_usd_hr: float = 2.0,
    gpu_count: int = 1,
    minimum_s: int = 0,
    gpu: str = "H100",
) -> Offer:
    """One offer of provider `kind`, priced per machine."""
    return Offer(
        provider=kind,
        gpu=gpu,
        gpu_count=gpu_count,
        rate_usd_hr=rate_usd_hr,
        minimum_s=minimum_s,
    )

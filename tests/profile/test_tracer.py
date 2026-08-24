import pytest

from mainboard.profile import Activity, CallbackSession, TraceCollector, Tracer, Vendor


class SupportingTracer(Tracer):
    """A tracer reporting a fixed KERNEL|MEMCPY support set, to exercise `resolve`."""

    def supported(self) -> Activity:
        return Activity.KERNEL | Activity.MEMCPY


class FullTracer(Tracer):
    """A tracer that supports every activity kind, so `ALL` needs no adaptation."""

    def supported(self) -> Activity:
        return Activity.ALL


def test_tracer_base_is_an_unavailable_noop() -> None:
    """The base tracer annotates nothing, supports nothing, and is never available."""
    tracer = Tracer()
    assert Tracer.is_available() is False
    tracer.push("x")
    tracer.pop()
    tracer.mark("x")
    tracer.start("x")()
    assert tracer.supported() == Activity(0)
    assert isinstance(tracer.timestamp(), int)
    assert isinstance(tracer.collect(), TraceCollector)
    assert isinstance(tracer.callbacks(), CallbackSession)


@pytest.mark.parametrize(
    ("tracer", "kinds", "expected"),
    [
        (Tracer(), Activity.KERNEL, Activity.KERNEL),
        (SupportingTracer(), Activity.ALL, Activity.KERNEL | Activity.MEMCPY),
        (FullTracer(), Activity.ALL, Activity.ALL),
        (SupportingTracer(), Activity.KERNEL, Activity.KERNEL),
    ],
    ids=["no_support_trusts_the_caller", "all_adapts_down", "all_when_nothing_drops", "explicit"],
)
def test_tracer_resolve_adapts_all_and_passes_explicit_kinds_through(
    tracer: Tracer, kinds: Activity, expected: Activity
) -> None:
    """`ALL` means whatever the device offers, and a supported explicit request is kept.

    A backend reporting no support at all is not second-guessed, since the no-op base has
    no deep trace to reconcile against.
    """
    assert tracer.resolve(kinds) == expected


def test_tracer_resolve_fails_fast_on_explicit_unsupported_kind() -> None:
    """An explicit unsupported kind is an error, not a silent omission."""
    with pytest.raises(ValueError, match="not supported"):
        SupportingTracer().resolve(Activity.MEMORY)


def test_tracer_detect_prefers_a_matching_vendor_then_anything_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detection prefers a backend for a vendor on this host, and falls back to the no-op base.

    An annotation library remains useful even when its vendor is not in `present`, so an
    available backend is still chosen over annotating nothing. Both stand-ins are declared
    inside the test because a `Tracer` subclass registers itself for the life of the
    process, and one that claims to be available would then be detected by every other test.
    """

    class MatchingTracer(Tracer):
        vendor = Vendor.NVIDIA

        @classmethod
        def is_available(cls) -> bool:
            return True

    class AvailableTracer(Tracer):
        @classmethod
        def is_available(cls) -> bool:
            return True

    cases = [
        ([Tracer], frozenset[str](), Tracer),
        ([MatchingTracer], frozenset({"nvidia"}), MatchingTracer),
        ([AvailableTracer], frozenset({"nvidia"}), AvailableTracer),
    ]
    for registered, present, expected in cases:
        monkeypatch.setattr(Tracer, "registry", classmethod(lambda cls, found=registered: found))
        assert type(Tracer.detect(present=present)) is expected

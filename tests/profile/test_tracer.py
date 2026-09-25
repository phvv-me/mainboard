import pytest

from mainboard.profile import Activity, CallbackSession, TraceCollector, Tracer, Vendor


class SupportingTracer(Tracer):
    def supported(self) -> Activity:
        return Activity.KERNEL | Activity.MEMCPY


class FullTracer(Tracer):
    def supported(self) -> Activity:
        return Activity.ALL


def test_tracer_base_is_an_unavailable_noop() -> None:
    tracer = Tracer()
    assert Tracer.is_available() is False
    tracer.push("x")
    tracer.pop()
    tracer.mark("x")
    tracer.start("x")()
    assert tracer.supported() == Activity(0)
    assert isinstance(tracer.timestamp(), int)
    with pytest.raises(ValueError, match="no activity collector available"):
        tracer.collect()
    assert isinstance(tracer.callbacks(), CallbackSession)
    assert type(tracer.open(Activity.KERNEL)) is TraceCollector


@pytest.mark.parametrize(
    ("tracer", "kinds", "expected"),
    [
        (SupportingTracer(), Activity.ALL, Activity.KERNEL | Activity.MEMCPY),
        (FullTracer(), Activity.ALL, Activity.ALL),
        (SupportingTracer(), Activity.KERNEL, Activity.KERNEL),
    ],
    ids=["all_adapts_down", "all_when_nothing_drops", "explicit"],
)
def test_tracer_resolve_adapts_all_and_passes_explicit_kinds_through(
    tracer: Tracer, kinds: Activity, expected: Activity
) -> None:
    assert tracer.resolve(kinds) == expected


def test_tracer_resolve_fails_fast_on_explicit_unsupported_kind() -> None:
    with pytest.raises(ValueError, match="not supported"):
        SupportingTracer().resolve(Activity.MEMORY)


@pytest.mark.parametrize("kinds", [Activity.KERNEL, Activity.ALL])
def test_a_backend_without_activity_support_never_returns_a_noop_trace(kinds: Activity) -> None:
    """Even ALL requires a real collector; an empty capability set is not a trace."""
    with pytest.raises(ValueError, match="no activity collector available"):
        Tracer().collect(kinds)


def test_tracer_detect_prefers_a_matching_vendor_then_anything_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An available backend beats annotating nothing even for an absent vendor. The stand-ins
    are local because a `Tracer` subclass registers itself for the life of the process."""

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

import pytest
from mainboard.profile import Activity, CallbackSession, TraceCollector, Tracer, Vendor


def test_tracer_base_is_an_unavailable_noop() -> None:
    """The base tracer annotates nothing, supports nothing, and is never available."""
    tracer = Tracer()
    assert Tracer.is_available() is False
    tracer.push("x")
    tracer.pop()
    tracer.mark("x")
    assert tracer.supported() == Activity(0)
    assert isinstance(tracer.timestamp(), int)
    assert isinstance(tracer.collect(), TraceCollector)
    assert isinstance(tracer.callbacks(), CallbackSession)


def test_tracer_resolve_passes_through_when_unsupported() -> None:
    """With no device support, resolve trusts the caller rather than dropping kinds."""
    assert Tracer().resolve(Activity.KERNEL) == Activity.KERNEL


class _SupportingTracer(Tracer):
    """A tracer reporting a fixed support set, to exercise `resolve`."""

    def supported(self) -> Activity:
        return Activity.KERNEL | Activity.MEMCPY


def test_tracer_resolve_adapts_all_to_supported() -> None:
    """`Activity.ALL` adapts down to exactly what the device supports."""
    assert _SupportingTracer().resolve(Activity.ALL) == (Activity.KERNEL | Activity.MEMCPY)


def test_tracer_resolve_all_with_full_support() -> None:
    class FullTracer(Tracer):
        def supported(self) -> Activity:
            return Activity.ALL

    assert FullTracer().resolve(Activity.ALL) is Activity.ALL


def test_tracer_resolve_fails_fast_on_explicit_unsupported_kind() -> None:
    """An explicit unsupported kind is an error, not a silent omission."""
    with pytest.raises(ValueError, match="not supported"):
        _SupportingTracer().resolve(Activity.MEMORY)


def test_tracer_detect_with_no_backend_available_is_the_no_op_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`detect` returns the no-op base when no backend library is available."""
    monkeypatch.setattr(Tracer, "registry", classmethod(lambda cls: [Tracer]))
    assert type(Tracer.detect()) is Tracer


def test_tracer_detect_prefers_a_backend_matching_present_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MatchingTracer(Tracer):
        vendor = Vendor.NVIDIA

        @classmethod
        def is_available(cls) -> bool:
            return True

    monkeypatch.setattr(Tracer, "registry", classmethod(lambda cls: [MatchingTracer]))
    assert isinstance(Tracer.detect(present=frozenset({"nvidia"})), MatchingTracer)


def test_tracer_detect_uses_available_backend_without_matching_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An annotation library remains useful even when its vendor is not in `present`."""

    class AvailableTracer(Tracer):
        @classmethod
        def is_available(cls) -> bool:
            return True

    monkeypatch.setattr(Tracer, "registry", classmethod(lambda cls: [AvailableTracer]))
    assert isinstance(Tracer.detect(present=frozenset()), AvailableTracer)

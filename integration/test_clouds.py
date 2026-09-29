"""The cloud backends against their APIs' recorded answers: what each call sends, and what a
machine reads as, without a key or a bill. A live rent needs the owner's key (`host hold`)."""

import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from mainboard.core.errors import MissionError
from mainboard.dispatch.arch import arch, capability, card, sm
from mainboard.dispatch.backends import HpcAiBackend, LambdaBackend, RunPodBackend, VastBackend
from mainboard.dispatch.backends.cloud import CapacityGone, _waiter
from mainboard.dispatch.rentals import LAUNCH
from mainboard.dispatch.sync import outermost
from mainboard.dispatch.vocabulary import Resources
from mainboard.reliability import reliability


class Recorded:
    """A transport answering each (method, path suffix) from a table, keeping every request."""

    def __init__(self, answers: dict[tuple[str, str], object]) -> None:
        self.answers = answers
        self.sent: list[tuple[str, str, dict | None]] = []

    def __call__(self, request):
        body = json.loads(request.data) if request.data else None
        self.sent.append((request.get_method(), request.full_url, body))
        for (method, suffix), answer in self.answers.items():
            if request.get_method() == method and request.full_url.endswith(suffix):
                if isinstance(answer, HTTPError):
                    raise answer
                return SimpleNamespace(status=200, read=lambda a=answer: json.dumps(a).encode())
        raise AssertionError(f"unexpected {request.get_method()} {request.full_url}")


def refusal(code: int, said: str) -> HTTPError:
    return HTTPError("https://x", code, said, {}, io.BytesIO(said.encode()))


@pytest.fixture(autouse=True)
def keys(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-runpod")
    monkeypatch.setenv("LAMBDA_API_KEY", "test-lambda")


def offer(**fields) -> SimpleNamespace:
    return SimpleNamespace(
        **{"instance_name": "x", "location": "r", "gpu_count": 1, "spot": False, **fields}
    )


def test_a_card_reads_the_same_however_it_is_spelled() -> None:
    assert card("NVIDIA_GeForce_RTX_4090") == card("rtx 4090") == "RTX4090"


def test_runpod_creates_a_pod_that_opens_ssh_with_the_key() -> None:
    wire = Recorded({("POST", "/pods"): {"id": "pod-1"}})
    handle = RunPodBackend(transport=wire).create(
        offer(instance_name="NVIDIA GeForce RTX 4090"),
        name="hold-a",
        public_key="ssh-ed25519 K",
        disk_gb=80,
    )
    _, url, body = wire.sent[0]
    assert handle == "pod-1" and url == "https://rest.runpod.io/v1/pods"
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"] and body["ports"] == ["22/tcp"]
    assert "sshd" in body["dockerStartCmd"][-1] and "ssh-ed25519 K" in body["dockerStartCmd"][-1]


def test_runpod_reads_a_running_pod_by_its_mapped_ssh_port() -> None:
    pod = {
        "id": "pod-1",
        "desiredStatus": "RUNNING",
        "publicIp": "1.2.3.4",
        "portMappings": {"22": 40022},
    }
    found = RunPodBackend(transport=Recorded({("GET", "/pods/pod-1"): pod})).machine("pod-1")
    assert (found.status, found.host, found.port, found.user) == (
        "running",
        "1.2.3.4",
        40022,
        "root",
    )


def test_runpod_out_of_capacity_is_a_retry_not_a_failure() -> None:
    wire = Recorded(
        {("POST", "/pods"): refusal(400, "There are no longer any instances available")}
    )
    with pytest.raises(CapacityGone):
        RunPodBackend(transport=wire).create(offer(), name="n", public_key="k", disk_gb=10)


def test_a_gone_pod_reads_as_gone_and_ends_quietly() -> None:
    wire = Recorded(
        {("GET", "/pods/p"): refusal(404, "{}"), ("DELETE", "/pods/p"): refusal(404, "{}")}
    )
    backend = RunPodBackend(transport=wire)
    assert backend.machine("p").status == "gone"
    backend.terminate("p")


def test_lambda_registers_the_key_once_then_launches_with_it() -> None:
    wire = Recorded(
        {
            ("GET", "/ssh-keys"): {"data": []},
            ("POST", "/ssh-keys"): {"data": {}},
            ("POST", "/instance-operations/launch"): {"data": {"instance_ids": ["i-1"]}},
        }
    )
    handle = LambdaBackend(transport=wire).create(
        offer(instance_name="gpu_1x_h100_pcie", location="us-east-1"),
        name="hold-b",
        public_key="ssh-ed25519 K",
        disk_gb=0,
    )
    launch = next(body for method, url, body in wire.sent if url.endswith("/launch"))
    registered = next(
        body for method, url, body in wire.sent if method == "POST" and url.endswith("/ssh-keys")
    )
    assert handle == "i-1" and launch["ssh_key_names"] == [registered["name"]]
    assert (
        launch["instance_type_name"] == "gpu_1x_h100_pcie" and launch["region_name"] == "us-east-1"
    )


def test_lambda_reads_an_active_instance_as_ubuntu_over_ssh() -> None:
    active = {
        "data": {
            "id": "i-1",
            "status": "active",
            "ip": "5.6.7.8",
            "instance_type": {"price_cents_per_hour": 249},
        }
    }
    found = LambdaBackend(transport=Recorded({("GET", "/instances/i-1"): active})).machine("i-1")
    assert (found.status, found.host, found.user, found.usd_hr) == (
        "running",
        "5.6.7.8",
        "ubuntu",
        2.49,
    )


def test_lambda_insufficient_capacity_is_a_retry() -> None:
    wire = Recorded(
        {
            ("GET", "/ssh-keys"): {"data": [{"name": "x"}]},
            ("POST", "/ssh-keys"): {"data": {}},
            ("POST", "/instance-operations/launch"): refusal(
                400, '{"error": {"code": "instance-operations/launch/insufficient-capacity"}}'
            ),
        }
    )
    with pytest.raises(CapacityGone):
        LambdaBackend(transport=wire).create(offer(), name="n", public_key="k", disk_gb=0)


def test_offers_list_every_cloud_and_mark_what_mb_rents(mb) -> None:
    ran = mb("host", "offers", "RTX 4090", "--limit", "40", timeout=300)
    assert ran.code == 0, ran.said
    rows = ran.out.splitlines()
    assert rows[0].startswith("provider") and any("vast" in row for row in rows[1:])


def test_an_architecture_matches_the_capability_its_kernels_load_on() -> None:
    rtx = arch("sm_120")
    assert {(each.low, each.high) for each in (rtx, arch("12.0"), arch("SM120"))} == {
        ((12, 0), (12, 0))
    }
    assert rtx.holds(capability("RTX PRO 4500")) and rtx.holds(capability("RTX 5090"))
    assert not rtx.holds(capability("B200")) and not rtx.holds(capability("GB10"))
    assert arch("blackwell-dc").holds(capability("B300"))  # sm_103, which gpuhunt lists as sm_100
    assert arch("sm_90+").holds(capability("B200")) and not arch("sm_90+").holds(
        capability("L40S")
    )
    assert sm(capability("RTX 4090")) == "sm_89" and capability("no such card") is None
    with pytest.raises(MissionError):
        arch("volta-ish")


def test_vast_asks_its_own_capability_field_for_an_architecture(monkeypatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "test-vast")
    wire = Recorded({("POST", "/bundles/"): {"offers": []}})
    VastBackend(transport=wire).search(gpus=1, arch=arch("sm_120"))
    query = wire.sent[0][2]
    assert query["compute_cap"] == {"gte": 1200, "lte": 1200} and "gpu_name" not in query


def hpcai_types(*rows: tuple[str, int, float, bool]) -> dict:
    """A `/resource/user/instance/list` answer, one family per (gpu, count, rate, in stock)."""
    return {
        "instanceInfos": [
            {
                "gpuName": gpu,
                "regionInfos": [
                    {
                        "regionName": "us-west-1",
                        "regionId": f"region-{gpu}",
                        "instanceTypeInfos": [
                            {
                                "gpuNum": count,
                                "instanceTypeId": f"type-{gpu}",
                                "stockStatus": "InStock" if stocked else "OutOfStock",
                                "price": [{"chargeMode": "perHour", "price": rate}],
                            }
                        ],
                    }
                ],
            }
            for gpu, count, rate, stocked in rows
        ]
    }


def test_hpcai_picks_the_cheapest_type_in_stock_for_the_architecture(monkeypatch) -> None:
    monkeypatch.setenv("HPCAI_API_KEY", "test-hpcai")
    catalog = hpcai_types(
        ("", 0, 0.24, True),
        ("RTX-5090", 8, 5.2, True),
        ("B200-SXM-180GB-SPOT", 8, 11.92, True),
        ("B200-SXM-180GB", 8, 24.4, True),
    )
    backend = HpcAiBackend(transport=Recorded({("POST", "/resource/user/instance/list"): catalog}))
    plan = SimpleNamespace(profile=SimpleNamespace(vars={"image-id": "img"}))
    spot_dc = backend.chosen(plan, Resources(gpus=8, arch="sm_100", spot=True))
    assert (spot_dc["instance_type_id"], spot_dc["spot"]) == ("type-B200-SXM-180GB-SPOT", True)
    assert backend.chosen(plan, Resources(gpus=1, arch="sm_120"))["instance_type_id"] == (
        "type-RTX-5090"
    )
    assert backend.chosen(plan, Resources())["instance_type_id"] == "type-"
    with pytest.raises(MissionError, match="no 8x H100"):
        backend.chosen(plan, Resources(gpus=8, gpu_name="H100"))


def test_a_cloud_job_reads_its_verdict_from_the_exit_file_the_waiter_leaves(monkeypatch) -> None:
    running = {
        "id": "pod-1",
        "desiredStatus": "RUNNING",
        "publicIp": "1.2.3.4",
        "portMappings": {"22": 1},
    }
    backend = RunPodBackend(transport=Recorded({("GET", "/pods/pod-1"): running}))
    monkeypatch.setattr(backend, "read", lambda handle, path: "")
    assert backend.state("pod-1").verdict == "running"
    monkeypatch.setattr(backend, "read", lambda handle, path: "3\n")
    finished = backend.state("pod-1")
    assert (finished.verdict, finished.exit_code) == ("failed", 3)
    assert LAUNCH in _waiter() and "/tmp/mainboard.exit" in _waiter()


def test_hpcai_gives_up_on_an_image_that_will_not_pull(monkeypatch) -> None:
    monkeypatch.setenv("HPCAI_API_KEY", "test-hpcai")
    stuck = {
        "instanceMetadata": {"instanceId": "nb-1"},
        "instanceRuntimeInfo": {
            "status": "Starting",
            "phase": "DownloadImage",
            "diagnosisReason": "BackOff",
            "diagnosisMessage": "Back-off restarting failed container download-image",
        },
        "instanceSpecInfo": {"nodePorts": [{"port": 22, "nodePort": 0}]},
    }
    listing = {"instances": [stuck], "pager": {"totalEntries": 1}}
    slept: list[float] = []
    backend = HpcAiBackend(
        transport=Recorded({("POST", "/instance/list"): listing}), sleeper=slept.append
    )
    with pytest.raises(MissionError, match="DownloadImage BackOff.*image-id"):
        backend.endpoint("nb-1")
    assert len(slept) < 20  # minutes of back-off, not the whole quarter hour


def test_a_rate_limited_vast_create_waits_and_asks_again(monkeypatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "test-vast")
    answers = iter([refusal(429, "{}"), refusal(429, "{}"), {"new_contract": 7}])

    def wire(request):
        answer = next(answers)
        if isinstance(answer, HTTPError):
            raise answer
        return SimpleNamespace(status=200, read=lambda: json.dumps(answer).encode())

    slept: list[float] = []
    created = VastBackend(transport=wire, sleeper=slept.append).created({"id": 1}, {})
    assert created == {"new_contract": 7} and slept == [2.0, 5.0]


def test_reliability_charges_the_machine_not_the_job() -> None:
    def run(
        target: str, verdict: str, *, exit_code: int | None = None, evidence: str = ""
    ) -> object:
        return SimpleNamespace(
            target=target, kind="vast", verdict=verdict, exit_code=exit_code, evidence=evidence
        )

    rows = reliability(
        [
            run("vast", "ok", exit_code=0),
            run("vast", "failed", exit_code=1),  # the job's own failure: the machine delivered
            run("vast", "failed", evidence="not_started"),  # the landing never started it
            run("vast", "vanished"),  # taken back under it
            run("vast", "cancelled"),  # judges nothing
            run("held", "running"),
        ]
    )
    vast = next(row for row in rows if row.target == "vast")
    assert (vast.ran, vast.unstarted, vast.lost, vast.unsettled) == (2, 1, 1, 1)
    assert vast.delivered == 0.5 and rows[-1].target == "held" and rows[-1].delivered is None


def test_a_path_under_another_root_is_listed_once() -> None:
    roots = [
        "mb.toml",
        "research/x/cutoken",
        "research/x/cutoken/pyproject.toml",
        "pkg/pyproject.toml",
    ]
    assert outermost(roots) == ["mb.toml", "research/x/cutoken", "pkg/pyproject.toml"]
    assert outermost([".", "a/b"]) == ["."]

from typing import TYPE_CHECKING

from mainboard.probe import Fabric

if TYPE_CHECKING:
    from pathlib import Path


def make_port(
    root: Path, device: str, port: str, *, state: str, rate: str, link_layer: str
) -> None:
    port_dir = root / device / "ports" / port
    port_dir.mkdir(parents=True)
    (port_dir / "state").write_text(state)
    (port_dir / "rate").write_text(rate)
    (port_dir / "link_layer").write_text(link_layer)


def test_probe_reads_every_device_and_port(tmp_path: Path) -> None:
    """Every numbered port under every device directory becomes one `FabricPort`."""
    make_port(
        tmp_path,
        "mlx5_0",
        "1",
        state="4: ACTIVE",
        rate="400 Gb/sec (4X NDR)",
        link_layer="InfiniBand",
    )
    make_port(
        tmp_path,
        "mlx5_1",
        "1",
        state="4: ACTIVE",
        rate="200 Gb/sec (2X NDR)",
        link_layer="Ethernet",
    )

    ports = Fabric.probe(root=tmp_path)
    assert len(ports) == 2
    ib, roce = ports[0], ports[1]
    assert ib.device == "mlx5_0"
    assert ib.port == 1
    assert ib.state == "4: ACTIVE"
    assert ib.rate == "400 Gb/sec (4X NDR)"
    assert ib.link_layer == "InfiniBand"
    assert roce.link_layer == "Ethernet"


def test_probe_orders_ports_numerically(tmp_path: Path) -> None:
    """Ports sort by their numeric port number, not lexically (`2` before `10`)."""
    for port in ("10", "2", "1"):
        make_port(
            tmp_path, "mlx5_0", port, state="4: ACTIVE", rate="100 Gb/sec", link_layer="InfiniBand"
        )
    ports = Fabric.probe(root=tmp_path)
    assert [p.port for p in ports] == [1, 2, 10]


def test_probe_is_empty_when_sysfs_tree_is_absent(tmp_path: Path) -> None:
    """A host with no `/sys/class/infiniband` (no HCA) yields no ports, never raises."""
    assert Fabric.probe(root=tmp_path / "absent") == ()


def test_probe_skips_a_device_with_no_ports_directory(tmp_path: Path) -> None:
    """A device directory missing its `ports` subdirectory contributes nothing."""
    (tmp_path / "mlx5_0").mkdir()
    assert Fabric.probe(root=tmp_path) == ()


def test_probe_ignores_non_numeric_port_entries(tmp_path: Path) -> None:
    """A non-numeric entry under `ports` (e.g. a stray file) is not treated as a port."""
    ports_dir = tmp_path / "mlx5_0" / "ports"
    ports_dir.mkdir(parents=True)
    (ports_dir / "README").write_text("not a port")
    assert Fabric.probe(root=tmp_path) == ()


def test_probe_tolerates_unreadable_field_files(tmp_path: Path) -> None:
    """A port directory with no field files still yields a port with empty fields."""
    (tmp_path / "mlx5_0" / "ports" / "1").mkdir(parents=True)
    (port,) = Fabric.probe(root=tmp_path)
    assert (port.state, port.rate, port.link_layer) == ("", "", "")

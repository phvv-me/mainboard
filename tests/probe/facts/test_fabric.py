from pathlib import Path

from mainboard.probe import Fabric


def make_port(root: Path, device: str, port: str, **fields: str) -> None:
    """Create one `<root>/<device>/ports/<port>` directory holding the named sysfs files.

    root: the fake `infiniband` class directory.
    device: the HCA device name, e.g. `mlx5_0`.
    port: the port directory name, numeric for a real port.
    fields: sysfs file name to contents, left out entirely to model an unreadable port.
    """
    port_dir = root / device / "ports" / port
    port_dir.mkdir(parents=True)
    for name, value in fields.items():
        (port_dir / name).write_text(value)


def test_probe_reads_every_numbered_port_in_numeric_order_and_skips_what_is_not_one(
    tmp_path: Path,
) -> None:
    """Ports sort numerically and a torn entry degrades quietly.

    `2` comes before `10`, and a stray file under `ports`, a device with no `ports`
    directory at all, and a port whose field files cannot be read each contribute nothing
    instead of taking the whole scan down.
    """
    for port in ("10", "2", "1"):
        make_port(
            tmp_path,
            "mlx5_0",
            port,
            state="4: ACTIVE",
            rate="400 Gb/sec (4X NDR)",
            link_layer="InfiniBand",
        )
    make_port(tmp_path, "mlx5_1", "1", state="4: ACTIVE", rate="200 Gb/sec", link_layer="Ethernet")
    make_port(tmp_path, "mlx5_3", "1")  # a port whose field files are all absent
    (tmp_path / "mlx5_0" / "ports" / "README").write_text("not a port")
    (tmp_path / "mlx5_2").mkdir()  # a device directory with no `ports` subdirectory

    make_port(tmp_path, "mlx5_10", "1", state="4: ACTIVE", link_layer="InfiniBand")

    ports = Fabric.probe(root=tmp_path)
    assert [(p.device, p.port, p.link_layer) for p in ports] == [
        ("mlx5_0", 1, "InfiniBand"),
        ("mlx5_0", 2, "InfiniBand"),
        ("mlx5_0", 10, "InfiniBand"),
        ("mlx5_1", 1, "Ethernet"),  # RoCE reports the same way over an Ethernet link layer
        ("mlx5_3", 1, ""),
        ("mlx5_10", 1, "InfiniBand"),  # devices sort by number too, so this is last, not second
    ]
    assert (ports[0].state, ports[0].rate) == ("4: ACTIVE", "400 Gb/sec (4X NDR)")
    unreadable = next(port for port in ports if port.device == "mlx5_3")
    assert (unreadable.state, unreadable.rate) == ("", "")


def test_probe_is_empty_on_a_host_with_no_fabric_tree(tmp_path: Path) -> None:
    """A machine with no HCA has no `/sys/class/infiniband` at all, which reads as no ports."""
    assert Fabric.probe(root=tmp_path / "absent") == ()

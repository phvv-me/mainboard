from pathlib import Path

from mainboard.observe.files import FrameFile


def test_an_event_stream_archived_under_a_quoted_path_reads_back_exactly(tmp_path: Path) -> None:
    stream = tmp_path / "it's ''quoted''.events"
    stream.write_bytes(b'{"a": 1}\n' * 3 + b'{"b"')

    archive = FrameFile(stream).archive()

    assert archive.name == f"{stream.name}.parquet"
    assert FrameFile(archive).read_bytes() == stream.read_bytes()

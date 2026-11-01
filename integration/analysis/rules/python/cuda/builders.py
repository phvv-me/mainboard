_NUMBA = """\
from numba import cuda


@cuda.jit
def kernel(x):
    x[0] = 1


"""
_PATOS = """\
from patos.cuda.typed import kernel


@kernel(threads=256)
def place(x) -> None:
    x[0] = 1


"""
_LAUNCH = _NUMBA + "def run(x, stream):\n    kernel[1, 32](x)\n"
_LAUNCHED = _LAUNCH.replace("kernel[1, 32](x)", "kernel[1, 32, stream](x)")
_TWICE = _NUMBA + "def run(x, stream):\n    kernel[1, 32](x)\n    kernel[2, 64](x)\n"
_MULTILINE = _LAUNCH.replace("kernel[1, 32](x)", "kernel[\n        1,\n        32,\n    ](x)")
_BOTH = _NUMBA + "def run(x):\n    stream = cuda.stream()\n    kernel[1, 32](x)\n"
_CREATED = _NUMBA + "def run(host):\n    stream = cuda.stream()\n    return cuda.to_device(host)\n"
_HOST = (
    _NUMBA
    + "def run():\n    stream = cuda.stream()\n    return cuda.device_array(4).copy_to_host()\n"
)
_PATOS_LAUNCH = _PATOS + "def run(x, stream):\n    place[1](x)\n"
_BARRIER = "from numba import cuda\n\n\ndef run(stream):\n    cuda.synchronize()\n"
_WAITED = "def run(stream):\n    stream.synchronize()\n"
_DEVICE_BARRIER = """\
import torch


def run(device):
    stream = torch.cuda.Stream()
    torch.cuda.synchronize(device)
"""
_FOREIGN_BARRIER = """\
import torch
from numba import cuda


def run():
    stream = torch.cuda.Stream()
    cuda.synchronize()
"""
_TWO_BARRIER = _BARRIER.replace("    cuda", "    other = cuda.stream()\n    cuda")
_AFTER = _NUMBA + "def run(x):\n    kernel[1, 32](x)\n    stream = cuda.stream()\n"
_TWO_STREAMS = _NUMBA + "def run(x, stream):\n    other = cuda.stream()\n    kernel[1, 32](x)\n"
_TORCH_STREAM = "import torch\n" + _BOTH.replace("cuda.stream()", "torch.cuda.Stream()")
_NAMED = _LAUNCH.replace("x, stream", 'x, s: "cuda.Stream"')
_ONE_ITEM = _LAUNCH.replace("[1, 32]", "[1]")


def cases() -> dict[str, tuple[str, str]]:
    """The files the PY-CUDA0003 and PY-CUDA0004 fixes read, by project path, each with its result.

    A file the fixes must refuse is its own expectation.
    """
    return {
        "tuple_grid.py": (
            _LAUNCH.replace("[1,", "[(2, 3),"),
            _LAUNCHED.replace("[1,", "[(2, 3),"),
        ),
        "trailing_comma.py": (_LAUNCH.replace("32]", "32,]"), _LAUNCHED),
        "twice.py": (_TWICE, _TWICE.replace("](x)", ", stream](x)")),
        "multiline.py": (_MULTILINE, _LAUNCHED),
        "carriage_returns.py": (_LAUNCH.replace("\n", "\r\n"), _LAUNCHED.replace("\n", "\r\n")),
        "both_rules.py": (_BOTH, _BOTH.replace("32](x)", "32, stream](x)")),
        "copy.py": (
            _CREATED,
            _CREATED.replace("to_device(host)", "to_device(host, stream=stream)"),
        ),
        "patos.py": (_PATOS_LAUNCH, _PATOS_LAUNCH.replace("[1](x)", "[1](x, stream=stream)")),
        "barrier.py": (_BARRIER, _WAITED),
        "barrier_device.py": (_DEVICE_BARRIER, _DEVICE_BARRIER),
        "barrier_foreign.py": (_FOREIGN_BARRIER, _FOREIGN_BARRIER),
        "barrier_two.py": (_TWO_BARRIER, _TWO_BARRIER),
        "host.py": (_HOST, _HOST),
        "after.py": (_AFTER, _AFTER),
        "two_streams.py": (_TWO_STREAMS, _TWO_STREAMS),
        "other_name.py": (_NAMED, _NAMED),
        "torch_stream.py": (_TORCH_STREAM, _TORCH_STREAM),
        "one_item.py": (_ONE_ITEM, _ONE_ITEM),
    }

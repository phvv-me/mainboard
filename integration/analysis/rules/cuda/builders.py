_SCALE = "__global__ void scale(float* values) {\n    values[threadIdx.x] *= 2.0f;\n}\n\n"
_PING = "__global__ void ping() {}\n\n"
_PARAMETER = _SCALE + "void run(float* d, cudaStream_t s) {\n    scale<<<1, 32>>>(d);\n}\n"
_LOCAL = _SCALE + "void run(float* d) {\n    cudaStream_t s;\n    scale<<<1, 32>>>(d);\n}\n"
_TWO = (
    _SCALE
    + """\
void run(float* d, cudaStream_t s) {
    scale<<<1, 32>>>(d);
    scale<<<2, 64, 8>>>(d);
}
"""
)
_AFTER = _SCALE + "void run(float* d) {\n    scale<<<1, 32>>>(d);\n    cudaStream_t s;\n}\n"
_ELSEWHERE = (
    _SCALE
    + """\
void make() {
    cudaStream_t s;
}

void run(float* d) {
    scale<<<1, 32>>>(d);
}
"""
)
_GLOBAL = _SCALE + "cudaStream_t g;\n\nvoid run(float* d) {\n    scale<<<1, 32>>>(d);\n}\n"
_TEMPLATE = """\
template <typename T>
__global__ void scale(T* values) {
    values[0] = 1;
}

void run(float* d, cudaStream_t s) {
    scale<float><<<1, 32>>>(d);
}
"""
_BARE = _PING + "void run(cudaStream_t s) {\n    ping<<<1, 1>>>();\n}\n"
_EXPRESSIONS = _PARAMETER.replace("<<<1, 32>>>", "<<<(n + 255) / 256, dim3(8, 8)>>>")
_TWO_DECLARED = _PARAMETER.replace("cudaStream_t s", "cudaStream_t s, cudaStream_t t")
_POINTER = _PARAMETER.replace("cudaStream_t s", "cudaStream_t* s")
_PROTOTYPE = (
    _SCALE + "void other(cudaStream_t s);\n\nvoid run(float* d) {\n    scale<<<1, 32>>>(d);\n}\n"
)
_COMMENTED = _PARAMETER.replace("<<<1, 32>>>", "<<<1, /* block */ 32>>>")
_NAMED = _PARAMETER.replace("<<<1, 32>>>", "<<<1, 32, 0, 0>>>")


def cases() -> dict[str, tuple[str, str]]:
    """The files the CU-LAUN0002 fix reads, by project path, each with the text it must leave.

    A file the fix must refuse is its own expectation.
    """
    return {
        "parameter.cu": (_PARAMETER, _PARAMETER.replace("<<<1, 32>>>", "<<<1, 32, 0, s>>>")),
        "local.cu": (_LOCAL, _LOCAL.replace("<<<1, 32>>>", "<<<1, 32, 0, s>>>")),
        "two.cu": (
            _TWO,
            _TWO.replace("<<<1, 32>>>", "<<<1, 32, 0, s>>>").replace(
                "<<<2, 64, 8>>>", "<<<2, 64, 8, s>>>"
            ),
        ),
        "global.cu": (_GLOBAL, _GLOBAL.replace("<<<1, 32>>>", "<<<1, 32, 0, g>>>")),
        "template.cu": (_TEMPLATE, _TEMPLATE.replace("<<<1, 32>>>", "<<<1, 32, 0, s>>>")),
        "bare.cu": (_BARE, _BARE.replace("<<<1, 1>>>", "<<<1, 1, 0, s>>>")),
        "expressions.cu": (
            _EXPRESSIONS,
            _EXPRESSIONS.replace("dim3(8, 8)>>>", "dim3(8, 8), 0, s>>>"),
        ),
        "carriage_returns.cu": (
            _PARAMETER.replace("\n", "\r\n"),
            _PARAMETER.replace("<<<1, 32>>>", "<<<1, 32, 0, s>>>").replace("\n", "\r\n"),
        ),
        "after.cu": (_AFTER, _AFTER),
        "elsewhere.cu": (_ELSEWHERE, _ELSEWHERE),
        "two_declared.cu": (_TWO_DECLARED, _TWO_DECLARED),
        "pointer.cu": (_POINTER, _POINTER),
        "prototype.cu": (_PROTOTYPE, _PROTOTYPE),
        "commented.cu": (_COMMENTED, _COMMENTED),
        "named.cu": (_NAMED, _NAMED),
    }

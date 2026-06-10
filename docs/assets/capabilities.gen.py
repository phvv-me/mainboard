"""Generate ``capabilities.svg`` — the profiler feature + roadmap map.

Run: ``python docs/assets/capabilities.gen.py``. Flip a status from ``R`` (roadmap) to
``S`` (shipped) as features land and regenerate. Status: ``S`` shipped, ``P`` partial,
``R`` roadmap. The map is CUDA-first; macOS/AMD coverage is noted in the footer.
"""

from __future__ import annotations

from pathlib import Path

GREEN, AMBER, GRAY, INK, SUB, LINE, BG = (
    "#16a34a",
    "#d97706",
    "#9ca3af",
    "#0f172a",
    "#475569",
    "#e2e8f0",
    "#ffffff",
)
MARK = {"S": GREEN, "P": AMBER, "R": GRAY}


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


SECTIONS = [
    (
        "Annotate",
        [
            ("region() & @profile", "S"),
            ("auto-annotate (PEP 669)", "S"),
            ("AST source rewrite", "S"),
            ("NVTX → Nsight timeline", "S"),
        ],
    ),
    (
        "Telemetry sampling",
        [
            ("memory used / peak", "S"),
            ("power & energy", "S"),
            ("utilization", "S"),
            ("temperature & clocks", "S"),
        ],
    ),
    (
        "CUPTI Activity trace",
        [
            ("kernels", "S"),
            ("memcpy / bandwidth", "S"),
            ("memset", "S"),
            ("driver & runtime API", "S"),
            ("memory alloc / free", "S"),
            ("JIT", "S"),
            ("overhead", "S"),
            ("synchronization", "S"),
            ("memory pool", "S"),
            ("NVTX records", "R"),
        ],
    ),
    (
        "CUPTI advanced",
        [
            ("Callback API (API counts)", "S"),
            ("PC sampling", "R"),
            ("Range / PM profiling", "R"),
            ("SASS / metric counters", "R"),
        ],
    ),
    (
        "Nsight ingest",
        [
            ("Nsight Compute (ncu_report)", "R"),
            ("Nsight Systems (sqlite)", "R"),
            ("occupancy & roofline metrics", "R"),
        ],
    ),
    (
        "Analyze & export",
        [
            ("rich report", "S"),
            ("bottlenecks", "S"),
            ("profile diff", "S"),
            ("Perfetto / Chrome trace", "S"),
            ("save / load", "S"),
            ("CLI", "S"),
            ("roofline (hardware-aware)", "R"),
            ("memory timeline", "R"),
        ],
    ),
]

COLS, COL_W, GAP, PAD, ROW_H, HEAD = 2, 470, 24, 28, 23, 34
TITLE_H, LEGEND_H, FOOT_H = 64, 26, 30


def card(x: float, y: float, title: str, rows: list[tuple[str, str]]) -> tuple[str, float]:
    h = HEAD + len(rows) * ROW_H + 12
    out = [
        f'<rect x="{x}" y="{y}" width="{COL_W}" height="{h}" rx="12" '
        f'fill="{BG}" stroke="{LINE}" stroke-width="1.5"/>',
        f'<text x="{x + 18}" y="{y + 23}" font-size="16" font-weight="700" fill="{INK}">'
        f"{esc(title)}</text>",
        f'<line x1="{x + 16}" y1="{y + HEAD - 4}" x2="{x + COL_W - 16}" y2="{y + HEAD - 4}" '
        f'stroke="{LINE}"/>',
    ]
    for i, (label, status) in enumerate(rows):
        ry = y + HEAD + i * ROW_H + 12
        color = MARK[status]
        fill = color if status == "S" else BG
        out.append(
            f'<circle cx="{x + 26}" cy="{ry - 4}" r="6" fill="{fill}" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        label_color = INK if status == "S" else SUB
        out.append(
            f'<text x="{x + 42}" y="{ry}" font-size="14" fill="{label_color}">{esc(label)}</text>'
        )
    return "\n".join(out), h


def build() -> str:
    col_x = [PAD, PAD + COL_W + GAP]
    col_y = [TITLE_H + LEGEND_H, TITLE_H + LEGEND_H]
    body = []
    for idx, (title, rows) in enumerate(SECTIONS):
        col = idx % COLS
        svg, h = card(col_x[col], col_y[col], title, rows)
        body.append(svg)
        col_y[col] += h + GAP
    width = PAD * 2 + COLS * COL_W + GAP
    height = max(col_y) + FOOT_H
    legend = (
        f'<circle cx="{PAD + 4}" cy="{TITLE_H + 6}" r="6" fill="{GREEN}"/>'
        f'<text x="{PAD + 16}" y="{TITLE_H + 11}" font-size="13" fill="{SUB}">shipped</text>'
        f'<circle cx="{PAD + 92}" cy="{TITLE_H + 6}" r="6" fill="{BG}" '
        f'stroke="{GRAY}" stroke-width="2"/>'
        f'<text x="{PAD + 104}" y="{TITLE_H + 11}" font-size="13" fill="{SUB}">roadmap</text>'
    )
    foot = (
        f'<text x="{PAD}" y="{height - 10}" font-size="12" fill="{SUB}">'
        "CUDA-first. macOS (os_signpost + snapshots) and AMD (ROCTx + snapshots) ship "
        "annotate and telemetry today; their deep trace is on the roadmap.</text>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="-apple-system, Segoe UI, sans-serif">\n'
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>\n'
        f'<text x="{PAD}" y="34" font-size="22" font-weight="800" fill="{INK}">'
        f"mainboard · profiler capabilities</text>\n"
        f"{legend}\n" + "\n".join(body) + f"\n{foot}\n</svg>\n"
    )


if __name__ == "__main__":
    Path(__file__).with_name("capabilities.svg").write_text(build())
    print("wrote capabilities.svg")

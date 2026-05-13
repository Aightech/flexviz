#!/usr/bin/env python3
"""Profile the IPC viewer's load sequence on a real PCB.

Mirrors what `ipc_open_viewer.py` does (minus the GL window), so we can see
which phase dominates and decide whether caching helps.

Usage:
    python3 scripts/profile_viewer_load.py [path/to/board.kicad_pcb]
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

# Make the plugin importable from a checkout (no install required)
HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent / "plugins" / "com_github_aightech_flexviz"
sys.path.insert(0, str(PLUGIN_DIR))

# Local imports
from kicad_parser import KiCadPCB  # noqa: E402
from geometry import extract_geometry  # noqa: E402
from markers import detect_fold_markers  # noqa: E402
from config import FlexConfig  # noqa: E402
from stiffener import extract_stiffeners  # noqa: E402
from mesh import (  # noqa: E402
    compute_regions,
    build_board_layer,
    build_traces_layer,
    build_pads_layer,
    build_components_layer,
    build_3d_models_layer,
    build_stiffeners_layer,
    precompute_all_layers,
)


class Phase:
    """Context manager that times a code block and records the result."""

    def __init__(self, label: str, results: list[tuple[str, float]]):
        self.label = label
        self.results = results

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt_ms = (time.perf_counter() - self.t0) * 1000.0
        self.results.append((self.label, dt_ms))
        return False


def fmt_row(label: str, ms: float, total_ms: float) -> str:
    pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
    bar = "#" * int(pct / 2)  # 2% per char, up to 50 chars
    return f"  {label:<32} {ms:>8.1f} ms   {pct:>5.1f}%  {bar}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pcb",
        nargs="?",
        default="/home/adev/dev/neubond/pcb/neubondV4/neubondV4.kicad_pcb",
    )
    parser.add_argument(
        "--bend-subdivisions",
        type=int,
        default=8,
        help="Match FlexConfig.bend_subdivisions (default 8)",
    )
    args = parser.parse_args()

    pcb_path = Path(args.pcb)
    if not pcb_path.exists():
        print(f"error: {pcb_path} not found", file=sys.stderr)
        return 1

    print(f"Profiling: {pcb_path}  ({pcb_path.stat().st_size / 1024:.0f} KB)")
    print()

    results: list[tuple[str, float]] = []

    # --- Synchronous phases (block the viewer's first paint) -----------------
    with Phase("read file (str I/O)", results):
        board_str = pcb_path.read_text()

    with Phase("KiCadPCB.parse", results):
        pcb = KiCadPCB.parse(board_str)

    with Phase("extract_geometry", results):
        board = extract_geometry(pcb)

    with Phase("FlexConfig.load_for_pcb", results):
        config = FlexConfig.load_for_pcb(pcb_path)

    with Phase("detect_fold_markers", results):
        markers = detect_fold_markers(pcb, layer=config.marker_layer)

    with Phase("extract_stiffeners", results):
        stiffeners = (
            extract_stiffeners(pcb, config) if config.has_stiffener else None
        )

    sync_ms = sum(ms for _, ms in results)

    # --- Background phases (run after the frame appears) ---------------------
    num_subs = config.bend_subdivisions
    bend_enabled = True
    pcb_dir = str(pcb_path.parent)

    with Phase("compute_regions", results):
        regions, active_regions = compute_regions(board, markers, num_subs, bend_enabled)

    with Phase("build_board_layer", results):
        build_board_layer(board, markers, 1.0, num_subs, False, bend_enabled)

    with Phase("build_traces_layer", results):
        build_traces_layer(board, active_regions, markers, num_subs)

    with Phase("build_pads_layer", results):
        build_pads_layer(board, active_regions)

    with Phase("build_components_layer", results):
        build_components_layer(board, active_regions)

    with Phase("build_stiffeners_layer", results):
        build_stiffeners_layer(board, regions, stiffeners, bend_enabled)

    with Phase("build_3d_models_layer", results):
        try:
            build_3d_models_layer(board, active_regions, pcb_dir, pcb)
        except Exception as e:
            print(f"  (build_3d_models_layer failed: {e})", file=sys.stderr)

    with Phase("precompute_all_layers", results):
        precompute_all_layers(
            board, markers, num_subs, subdivide_length=1.0, debug_regions=False
        )

    # --- Report --------------------------------------------------------------
    total_ms = sum(ms for _, ms in results)

    print(f"  {'phase':<32} {'time':>8}        {'share':>5}")
    print(f"  {'-' * 32} {'-' * 8}        {'-' * 5}")
    for label, ms in results:
        print(fmt_row(label, ms, total_ms))
    print(f"  {'-' * 32} {'-' * 8}        {'-' * 5}")
    print(f"  {'TOTAL (incl. background)':<32} {total_ms:>8.1f} ms")
    print(f"  {'SYNC ONLY (before frame shown)':<32} {sync_ms:>8.1f} ms")
    print()
    print(f"Board: {len(board.outline) if board.outline else 0} outline pts, "
          f"{len(getattr(board, 'traces', []))} traces, "
          f"{len(getattr(board, 'pads', []))} pads, "
          f"{len(getattr(board, 'components', []))} components")
    print(f"Markers: {len(markers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

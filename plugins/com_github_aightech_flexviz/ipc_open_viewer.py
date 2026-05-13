#!/usr/bin/env python3
"""
IPC API entrypoint: Open Fold Viewer

Launched by KiCad as a separate process via the IPC plugin system.
Reads the board as an S-expression string via IPC (no save-to-disk hack),
parses it, extracts geometry and markers, and opens the 3D viewer.
"""

import sys
import os

# Add plugin directory to path for local imports
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

try:
    from kipy import KiCad
except ImportError:
    print("Error: kicad-python package is required. Install with: pip install kicad-python")
    sys.exit(1)

from kicad_parser import KiCadPCB
from geometry import extract_geometry
from markers import detect_fold_markers
from config import FlexConfig


def main():
    kicad = KiCad()
    board = kicad.get_board()

    # Get board as S-expression string directly from KiCad — no disk I/O needed
    board_str = board.get_as_string()
    pcb = KiCadPCB.parse(board_str)

    # Resolve the FULL board path so per-PCB config (<name>.flex_config.json)
    # loads and 3D model paths (relative to the .kicad_pcb directory) resolve.
    # board.name returns only the basename over IPC; the full directory comes
    # from board.document.project.path.
    board_path = None
    try:
        doc = board.document
        proj_path = doc.project.path
        filename = doc.board_filename or board.name
        if proj_path and filename:
            board_path = os.path.join(proj_path, filename)
    except Exception:
        pass
    if not board_path:
        # Fallback: at least give viewer the basename so per-PCB config can
        # work in the cwd. 3D model loading will still fail without a dir.
        n = getattr(board, "name", None)
        board_path = n if n else None

    geom = extract_geometry(pcb)
    config = FlexConfig.load_for_pcb(board_path) if board_path else FlexConfig()
    markers = detect_fold_markers(pcb, layer=config.marker_layer)

    # Check OpenGL availability before importing viewer
    try:
        from viewer import OPENGL_AVAILABLE
    except ImportError:
        OPENGL_AVAILABLE = False

    import wx

    if not OPENGL_AVAILABLE:
        app = wx.App()
        wx.MessageBox(
            "PyOpenGL is required for the 3D viewer.\n\n"
            "Install with: pip install PyOpenGL",
            "Flex Viewer - Missing Dependency",
            wx.OK | wx.ICON_ERROR,
        )
        sys.exit(1)

    from viewer import FlexViewerFrame

    app = wx.App()
    frame = FlexViewerFrame(
        None, geom, markers,
        pcb=pcb,
        pcb_filepath=board_path,  # From board.name (IPC); None for unsaved boards
        kicad_ipc=kicad,          # Pass IPC connection for refresh
    )
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()

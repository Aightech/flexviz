"""
KiCad Flex Viewer - Action Plugin Registration

This module registers the plugin buttons in KiCad PCB Editor.
"""

import os
import sys
import pcbnew
import wx

# Add plugin directory to path for imports
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)


class FlexViewerTestAction(pcbnew.ActionPlugin):
    """Test action to verify plugin is loaded correctly."""

    def defaults(self):
        self.name = "Flex Viewer - Test"
        self.category = "Flex PCB"
        self.description = "Test that the Flex Viewer plugin is installed correctly"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(PLUGIN_DIR, "resources", "icon_test.png")

    def Run(self):
        # Import our modules to verify they work
        try:
            from .kicad_parser import KiCadPCB
            from .markers import detect_fold_markers
            from .geometry import extract_geometry

            # Get current board
            board = pcbnew.GetBoard()
            if board is None:
                wx.MessageBox(
                    "No board is currently open.",
                    "Flex Viewer Test",
                    wx.OK | wx.ICON_WARNING
                )
                return

            # Get board file path
            board_path = board.GetFileName()
            if not board_path:
                wx.MessageBox(
                    "Board has not been saved yet.\nPlease save the board first.",
                    "Flex Viewer Test",
                    wx.OK | wx.ICON_WARNING
                )
                return

            # Parse the board
            pcb = KiCadPCB.load(board_path)
            geom = extract_geometry(pcb)
            markers = detect_fold_markers(pcb)

            # Show results
            msg = f"""Flex Viewer Plugin Test - SUCCESS

Board: {os.path.basename(board_path)}

Parsed Data:
  - Outline vertices: {len(geom.outline)}
  - Board thickness: {geom.thickness}mm
  - Traces: {len(geom.all_traces)}
  - Components: {len(geom.components)}
  - Fold markers: {len(markers)}"""

            if markers:
                msg += "\n\nFold Markers Found:"
                for i, m in enumerate(markers, 1):
                    msg += f"\n  {i}. Angle: {m.angle_degrees}°, Radius: {m.radius:.2f}mm"

            if len(geom.outline) > 0:
                bbox = geom.bounding_box
                msg += f"\n\nBoard size: {bbox.width:.1f}mm x {bbox.height:.1f}mm"

            wx.MessageBox(msg, "Flex Viewer Test", wx.OK | wx.ICON_INFORMATION)

        except Exception as e:
            import traceback
            error_msg = f"Error loading Flex Viewer modules:\n\n{str(e)}\n\n{traceback.format_exc()}"
            wx.MessageBox(error_msg, "Flex Viewer Test - Error", wx.OK | wx.ICON_ERROR)


class CreateFoldAction(pcbnew.ActionPlugin):
    """Action to create a new fold marker."""

    def defaults(self):
        self.name = "Create Fold"
        self.category = "Flex PCB"
        self.description = "Create a fold marker for flex PCB visualization"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(PLUGIN_DIR, "resources", "icon_create_fold.png")

    def Run(self):
        try:
            from .fold_placer import run_fold_placer
            run_fold_placer()
        except Exception as e:
            import traceback
            error_msg = f"Error running Create Fold:\n\n{str(e)}\n\n{traceback.format_exc()}"
            wx.MessageBox(error_msg, "Create Fold - Error", wx.OK | wx.ICON_ERROR)


class OpenViewerAction(pcbnew.ActionPlugin):
    """Action to open the 3D fold viewer."""

    def defaults(self):
        self.name = "Open Fold Viewer"
        self.category = "Flex PCB"
        self.description = "Open the 3D viewer to visualize the folded flex PCB"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(PLUGIN_DIR, "resources", "icon_open_viewer.png")

    def Run(self):
        try:
            from .kicad_parser import KiCadPCB
            from .markers import detect_fold_markers
            from .geometry import extract_geometry
            from .viewer import FlexViewerFrame

            # Get current board
            board = pcbnew.GetBoard()
            if board is None:
                wx.MessageBox(
                    "No board is currently open.",
                    "Flex Viewer",
                    wx.OK | wx.ICON_WARNING
                )
                return

            # Get board file path
            board_path = board.GetFileName()
            if not board_path:
                wx.MessageBox(
                    "Board has not been saved yet.\nPlease save the board first.",
                    "Flex Viewer",
                    wx.OK | wx.ICON_WARNING
                )
                return

            # Parse the board
            pcb = KiCadPCB.load(board_path)
            geom = extract_geometry(pcb)
            markers = detect_fold_markers(pcb)

            # Open viewer window
            frame = FlexViewerFrame(None, geom, markers)
            frame.Show()

        except Exception as e:
            import traceback
            error_msg = f"Error opening Flex Viewer:\n\n{str(e)}\n\n{traceback.format_exc()}"
            wx.MessageBox(error_msg, "Flex Viewer - Error", wx.OK | wx.ICON_ERROR)


# Note: Registration is done in __init__.py, not here
# This allows proper error handling if imports fail

#!/usr/bin/env python3
"""
IPC API entrypoint: Create Fold Marker

Launched by KiCad as a separate process via the IPC plugin system.
Reads the current selection, shows a parameter dialog, creates fold
marker geometry, and optionally lets the user position it interactively.
"""

import sys
import os
import math

# Add plugin directory to path for local imports
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

from fold_math import compute_fold_marker_geometry

try:
    from kipy import KiCad
    from kipy.board_types import BoardSegment, BoardLayer, LeaderDimension, BoardText
    from kipy.geometry import Vector2
    from kipy.util.units import from_mm, to_mm
    from kipy.proto.common.types.base_types_pb2 import StrokeLineStyle
except ImportError:
    print("Error: kicad-python package is required. Install with: pip install kicad-python")
    sys.exit(1)

import wx


# ---- Selection helpers -------------------------------------------------------

def get_selected_line_ipc(board):
    """Get a selected line segment from the board via IPC. Returns ((x,y), (x,y)) in mm or None."""
    selection = board.get_selection()

    for item in selection:
        if isinstance(item, BoardSegment):
            return (
                (to_mm(item.start.x), to_mm(item.start.y)),
                (to_mm(item.end.x), to_mm(item.end.y)),
            )

    return None


def get_selected_points_ipc(board):
    """Get two selected points (pads, vias, etc.) via IPC. Returns ((x,y), (x,y)) in mm or None."""
    from kipy.board_types import Pad, Via

    selection = board.get_selection()
    points = []

    for item in selection:
        if isinstance(item, Pad):
            pos = item.position
            points.append((to_mm(pos.x), to_mm(pos.y)))
        elif isinstance(item, Via):
            pos = item.position
            points.append((to_mm(pos.x), to_mm(pos.y)))
        elif isinstance(item, BoardSegment):
            # Use midpoint of segment as a point
            mx = (to_mm(item.start.x) + to_mm(item.end.x)) / 2
            my = (to_mm(item.start.y) + to_mm(item.end.y)) / 2
            points.append((mx, my))

    if len(points) >= 2:
        return points[0], points[1]
    return None


# ---- Fold parameter dialog (reuses the same UI as SWIG path) ----------------

class FoldParametersDialog(wx.Dialog):
    """Simple dialog for fold angle and radius."""

    def __init__(self, parent, line_length_mm):
        super().__init__(parent, title="Fold Parameters",
                         style=wx.DEFAULT_DIALOG_STYLE)
        self.angle = 90.0
        self.radius = 1.0

        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        info = wx.StaticText(panel, label=f"Fold line detected: {line_length_mm:.1f} mm")
        main_sizer.Add(info, 0, wx.ALL, 10)

        grid = wx.FlexGridSizer(3, 2, 5, 10)

        grid.Add(wx.StaticText(panel, label="Angle (degrees):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_angle = wx.SpinCtrlDouble(panel, value="90", min=-360, max=360, inc=5)
        grid.Add(self.spin_angle, 0, wx.EXPAND)

        grid.Add(wx.StaticText(panel, label="Radius (mm):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_radius = wx.SpinCtrlDouble(panel, value="1.0", min=0.1, max=50, inc=0.5)
        self.spin_radius.SetDigits(2)
        grid.Add(self.spin_radius, 0, wx.EXPAND)

        grid.Add(wx.StaticText(panel, label="Zone width:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.label_zone = wx.StaticText(panel, label="")
        grid.Add(self.label_zone, 0, wx.EXPAND)

        main_sizer.Add(grid, 0, wx.ALL | wx.EXPAND, 10)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        main_sizer.Add(btn_sizer, 0, wx.ALL | wx.EXPAND, 10)

        panel.SetSizer(main_sizer)
        main_sizer.Fit(self)

        self.spin_angle.Bind(wx.EVT_SPINCTRLDOUBLE, self._update_zone)
        self.spin_radius.Bind(wx.EVT_SPINCTRLDOUBLE, self._update_zone)
        self._update_zone(None)

    def _update_zone(self, event):
        angle_rad = math.radians(abs(self.spin_angle.GetValue()))
        radius = self.spin_radius.GetValue()
        zone_w = radius * angle_rad if angle_rad > 0.001 else 0.0
        self.label_zone.SetLabel(f"{zone_w:.2f} mm")

    def get_result(self):
        return self.spin_angle.GetValue(), self.spin_radius.GetValue()


# ---- Main -------------------------------------------------------------------

def main():
    kicad = KiCad()
    board = kicad.get_board()

    # Read selection
    line_points = get_selected_line_ipc(board)
    if line_points is None:
        line_points = get_selected_points_ipc(board)

    if line_points is None:
        app = wx.App()
        wx.MessageBox(
            "Please select a line or two points first.\n\n"
            "How to use:\n"
            "1. Draw a line where you want the fold\n"
            "2. Select the line\n"
            "3. Run this tool\n\n"
            "Or select two pads/vias to define endpoints.",
            "Create Fold - Select First",
            wx.OK | wx.ICON_INFORMATION,
        )
        return

    point_a_mm, point_b_mm = line_points

    dx = point_b_mm[0] - point_a_mm[0]
    dy = point_b_mm[1] - point_a_mm[1]
    length_mm = math.sqrt(dx * dx + dy * dy)

    if length_mm < 0.1:
        app = wx.App()
        wx.MessageBox("Selected line is too short.", "Create Fold", wx.OK | wx.ICON_WARNING)
        return

    # Show parameter dialog
    app = wx.App()
    dlg = FoldParametersDialog(None, length_mm)
    if dlg.ShowModal() != wx.ID_OK:
        dlg.Destroy()
        return
    angle_degrees, radius_mm = dlg.get_result()
    dlg.Destroy()

    # Compute fold marker geometry in mm
    geom = compute_fold_marker_geometry(point_a_mm, point_b_mm, angle_degrees, radius_mm)
    if geom is None:
        wx.MessageBox("Failed to compute fold geometry.", "Create Fold", wx.OK | wx.ICON_ERROR)
        return

    # Create board items via IPC
    line1 = BoardSegment()
    line1.start = Vector2.from_xy(from_mm(geom.line1_start[0]), from_mm(geom.line1_start[1]))
    line1.end = Vector2.from_xy(from_mm(geom.line1_end[0]), from_mm(geom.line1_end[1]))
    line1.layer = BoardLayer.BL_User_1
    line1.attributes.stroke.width = from_mm(0.15)
    line1.attributes.stroke.style = StrokeLineStyle.SLS_DOT

    line2 = BoardSegment()
    line2.start = Vector2.from_xy(from_mm(geom.line2_start[0]), from_mm(geom.line2_start[1]))
    line2.end = Vector2.from_xy(from_mm(geom.line2_end[0]), from_mm(geom.line2_end[1]))
    line2.layer = BoardLayer.BL_User_1
    line2.attributes.stroke.width = from_mm(0.15)
    line2.attributes.stroke.style = StrokeLineStyle.SLS_DOT

    # Dimension annotation
    dim = LeaderDimension()
    dim.start = Vector2.from_xy(from_mm(geom.dim_start[0]), from_mm(geom.dim_start[1]))
    dim.end = Vector2.from_xy(from_mm(geom.dim_end[0]), from_mm(geom.dim_end[1]))
    dim.layer = BoardLayer.BL_User_1
    dim.override_text_enabled = True
    dim.override_text = geom.angle_text

    # Create in a single undo transaction
    commit = board.begin_commit()
    try:
        created = board.create_items([line1, line2, dim])
        board.push_commit(commit, "Create fold marker")
    except Exception:
        board.drop_commit(commit)
        raise

    # Let user position the marker interactively
    if created:
        try:
            board.interactive_move([item.id for item in created])
        except Exception:
            pass  # interactive_move may not be supported in all versions


if __name__ == "__main__":
    main()

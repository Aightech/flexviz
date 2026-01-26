"""
Fold Placer - Interactive fold marker creation for KiCad.

Workflow:
1. Select a line on the board (this will be the fold centerline)
2. Run the "Create Fold" tool
3. Enter angle and radius in the simple dialog
4. Done!

Alternative: Select two points/pads to define the fold line endpoints.
"""

import math
import pcbnew
import wx


# KiCad internal units are nanometers
# 1 mm = 1,000,000 nm
IU_PER_MM = 1000000


def mm_to_iu(mm):
    """Convert millimeters to KiCad internal units."""
    return int(mm * IU_PER_MM)


def iu_to_mm(iu):
    """Convert KiCad internal units to millimeters."""
    return iu / IU_PER_MM


class FoldParametersDialog(wx.Dialog):
    """Simple dialog for fold angle and radius - points come from selection."""

    def __init__(self, parent, line_length_mm):
        super().__init__(parent, title="Fold Parameters",
                         style=wx.DEFAULT_DIALOG_STYLE)

        self.angle = 90.0
        self.radius = 1.0

        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Show the detected line info
        info = wx.StaticText(
            panel,
            label=f"Fold line detected: {line_length_mm:.1f} mm"
        )
        font = info.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        info.SetFont(font)
        main_sizer.Add(info, 0, wx.ALL, 10)

        # Angle
        angle_sizer = wx.BoxSizer(wx.HORIZONTAL)
        angle_label = wx.StaticText(panel, label="Fold Angle:")
        self.angle_ctrl = wx.SpinCtrlDouble(panel, value="90", min=-180, max=180, inc=5)
        self.angle_ctrl.SetDigits(1)
        angle_unit = wx.StaticText(panel, label="degrees")
        angle_sizer.Add(angle_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        angle_sizer.Add(self.angle_ctrl, 1, wx.RIGHT, 5)
        angle_sizer.Add(angle_unit, 0, wx.ALIGN_CENTER_VERTICAL)
        main_sizer.Add(angle_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        main_sizer.AddSpacer(10)

        # Radius
        radius_sizer = wx.BoxSizer(wx.HORIZONTAL)
        radius_label = wx.StaticText(panel, label="Bend Radius:")
        self.radius_ctrl = wx.SpinCtrlDouble(panel, value="1.0", min=0.1, max=50, inc=0.5)
        self.radius_ctrl.SetDigits(2)
        radius_unit = wx.StaticText(panel, label="mm")
        radius_sizer.Add(radius_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        radius_sizer.Add(self.radius_ctrl, 1, wx.RIGHT, 5)
        radius_sizer.Add(radius_unit, 0, wx.ALIGN_CENTER_VERTICAL)
        main_sizer.Add(radius_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        main_sizer.AddSpacer(5)

        # Zone width (calculated, read-only)
        self.zone_label = wx.StaticText(panel, label="Bend zone: 1.57 mm")
        self.zone_label.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(self.zone_label, 0, wx.LEFT | wx.RIGHT, 10)

        main_sizer.AddSpacer(10)

        # Help text
        help_text = wx.StaticText(
            panel,
            label="+ angle = fold toward you\n"
                  "- angle = fold away from you"
        )
        help_text.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(help_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(panel, wx.ID_OK, "Create")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL)
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(cancel_btn)
        btn_sizer.Realize()
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(main_sizer)
        main_sizer.Fit(self)
        self.Centre()

        # Bind events
        self.angle_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        self.radius_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        self._update_zone_display()

    def _on_change(self, event):
        self._update_zone_display()

    def _update_zone_display(self):
        angle = abs(self.angle_ctrl.GetValue())
        radius = self.radius_ctrl.GetValue()
        if angle > 0.1:
            zone = radius * math.radians(angle)
            self.zone_label.SetLabel(f"Bend zone: {zone:.2f} mm")
        else:
            self.zone_label.SetLabel("Bend zone: -- (flat)")

    def get_result(self):
        return self.angle_ctrl.GetValue(), self.radius_ctrl.GetValue()


class FoldMarkerCreator:
    """Creates fold marker geometry on the PCB."""

    MARKER_LAYER = pcbnew.User_1
    LINE_WIDTH_MM = 0.15

    def __init__(self, board):
        self.board = board

    def create_fold_marker(self, point_a, point_b, angle_degrees, radius_mm):
        """Create a fold marker with two parallel lines and a dimension."""
        created_items = []

        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1:
            return created_items

        # Unit perpendicular vector
        perp_x = -dy / length
        perp_y = dx / length

        # Calculate zone width
        angle_rad = math.radians(abs(angle_degrees))
        zone_width = radius_mm * angle_rad if angle_rad > 0.001 else 0.1

        # Offset for parallel lines
        offset = mm_to_iu(zone_width / 2)

        # Create two parallel dotted lines
        line1 = self._create_line(
            (point_a[0] + perp_x * offset, point_a[1] + perp_y * offset),
            (point_b[0] + perp_x * offset, point_b[1] + perp_y * offset)
        )
        created_items.append(line1)

        line2 = self._create_line(
            (point_a[0] - perp_x * offset, point_a[1] - perp_y * offset),
            (point_b[0] - perp_x * offset, point_b[1] - perp_y * offset)
        )
        created_items.append(line2)

        # Create dimension annotation
        dim = self._create_dimension(point_a, point_b, angle_degrees, offset)
        if dim:
            created_items.append(dim)

        for item in created_items:
            self.board.Add(item)

        return created_items

    def _create_line(self, start, end):
        """Create a dotted line on the marker layer."""
        line = pcbnew.PCB_SHAPE(self.board)
        line.SetShape(pcbnew.SHAPE_T_SEGMENT)
        line.SetStart(pcbnew.VECTOR2I(int(start[0]), int(start[1])))
        line.SetEnd(pcbnew.VECTOR2I(int(end[0]), int(end[1])))
        line.SetLayer(self.MARKER_LAYER)
        line.SetWidth(mm_to_iu(self.LINE_WIDTH_MM))

        try:
            stroke = line.GetStroke()
            stroke.SetLineStyle(pcbnew.LINE_STYLE_DASH)
            line.SetStroke(stroke)
        except AttributeError:
            pass

        return line

    def _create_dimension(self, point_a, point_b, angle_degrees, offset):
        """Create a leader dimension annotation showing the fold angle."""
        try:
            mid_x = (point_a[0] + point_b[0]) / 2
            mid_y = (point_a[1] + point_b[1]) / 2

            dx = point_b[0] - point_a[0]
            dy = point_b[1] - point_a[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length < 1:
                return None
        
            perp_x = -dy / length
            perp_y = dx / length

            dim_offset = offset + mm_to_iu(2.0)
            text_pos_x = int(mid_x + dx)
            text_pos_y = int(mid_y + dy)

            # Use leader dimension (arrow with text)
            dim = pcbnew.PCB_DIM_LEADER(self.board)
            dim.SetStart(pcbnew.VECTOR2I(int(mid_x), int(mid_y)))
            dim.SetEnd(pcbnew.VECTOR2I(text_pos_x, text_pos_y))
            dim.SetLayer(self.MARKER_LAYER)

            angle_text = f"{angle_degrees:.0f}°"
            dim.SetOverrideTextEnabled(True)
            dim.SetOverrideText(angle_text)

            # Explicitly set the text position
            dim.SetTextPos(pcbnew.VECTOR2I(text_pos_x, text_pos_y))

            dim.Update()
            return dim

        except Exception:
            return self._create_angle_text(point_a, point_b, angle_degrees, offset)

    def _create_angle_text(self, point_a, point_b, angle_degrees, offset):
        """Create a text annotation as fallback."""
        try:
            mid_x = (point_a[0] + point_b[0]) / 2
            mid_y = (point_a[1] + point_b[1]) / 2

            dx = point_b[0] - point_a[0]
            dy = point_b[1] - point_a[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length < 1:
                return None

            perp_x = -dy / length
            perp_y = dx / length
            text_offset = offset + mm_to_iu(1.5)

            text = pcbnew.PCB_TEXT(self.board)
            text.SetText(f"{angle_degrees:+.1f}")
            text.SetPosition(pcbnew.VECTOR2I(
                int(mid_x + perp_x * text_offset),
                int(mid_y + perp_y * text_offset)
            ))
            text.SetLayer(self.MARKER_LAYER)
            text.SetTextSize(pcbnew.VECTOR2I(mm_to_iu(1.5), mm_to_iu(1.5)))

            angle_rad = math.atan2(dy, dx)
            text.SetTextAngle(pcbnew.EDA_ANGLE(math.degrees(angle_rad), pcbnew.DEGREES_T))

            return text

        except Exception:
            return None


def get_selected_line(board):
    """Get a selected line from the board. Returns (point_a, point_b) or None."""

    # Check board drawings
    for item in board.GetDrawings():
        if item.IsSelected():
            if hasattr(item, 'GetStart') and hasattr(item, 'GetEnd'):
                start = item.GetStart()
                end = item.GetEnd()
                return (start.x, start.y), (end.x, end.y)

    # Check footprint graphics
    for fp in board.GetFootprints():
        for item in fp.GraphicalItems():
            if item.IsSelected():
                if hasattr(item, 'GetStart') and hasattr(item, 'GetEnd'):
                    start = item.GetStart()
                    end = item.GetEnd()
                    return (start.x, start.y), (end.x, end.y)

    return None


def get_selected_points(board):
    """Get two selected points (pads, vias, or item positions). Returns (point_a, point_b) or None."""
    points = []

    # Check pads
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.IsSelected():
                pos = pad.GetPosition()
                points.append((pos.x, pos.y))

    # Check vias
    for track in board.GetTracks():
        if track.IsSelected() and track.GetClass() == "PCB_VIA":
            pos = track.GetPosition()
            points.append((pos.x, pos.y))

    # Check selected drawings (use center/position)
    for item in board.GetDrawings():
        if item.IsSelected():
            if hasattr(item, 'GetPosition'):
                pos = item.GetPosition()
                points.append((pos.x, pos.y))
            elif hasattr(item, 'GetCenter'):
                pos = item.GetCenter()
                points.append((pos.x, pos.y))

    if len(points) >= 2:
        return points[0], points[1]

    return None


def run_fold_placer():
    """Run the fold placer tool."""
    board = pcbnew.GetBoard()
    if board is None:
        wx.MessageBox("No board is open.", "Create Fold", wx.OK | wx.ICON_WARNING)
        return

    # Try to get line from selection
    line_points = get_selected_line(board)

    if line_points is None:
        # Try to get two points
        line_points = get_selected_points(board)

    if line_points is None:
        wx.MessageBox(
            "Please select a line or two points first.\n\n"
            "How to use:\n"
            "1. Draw a line where you want the fold\n"
            "2. Select the line\n"
            "3. Run this tool\n\n"
            "Or select two pads/vias to define endpoints.",
            "Create Fold - Select First",
            wx.OK | wx.ICON_INFORMATION
        )
        return

    point_a, point_b = line_points

    # Calculate line length
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    length_mm = iu_to_mm(math.sqrt(dx * dx + dy * dy))

    if length_mm < 0.1:
        wx.MessageBox(
            "Selected line is too short.\n"
            "Please select a longer line.",
            "Create Fold",
            wx.OK | wx.ICON_WARNING
        )
        return

    # Get main window
    frame = None
    for w in wx.GetTopLevelWindows():
        if 'PCB' in w.GetTitle() or 'Pcbnew' in w.GetTitle():
            frame = w
            break

    # Show simple parameters dialog
    dlg = FoldParametersDialog(frame, length_mm)

    if dlg.ShowModal() == wx.ID_OK:
        angle, radius = dlg.get_result()

        creator = FoldMarkerCreator(board)
        items = creator.create_fold_marker(point_a, point_b, angle, radius)

        if items:
            pcbnew.Refresh()
        else:
            wx.MessageBox(
                "Failed to create fold marker.",
                "Create Fold",
                wx.OK | wx.ICON_ERROR
            )

    dlg.Destroy()

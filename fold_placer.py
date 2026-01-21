"""
Fold Placer - Interactive fold marker creation for KiCad.

This module provides an interactive tool for placing fold markers on PCB boards.
The user clicks two points to define the fold line, then enters the fold angle.
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


class FoldPlacerDialog(wx.Dialog):
    """Dialog for entering fold angle after placing points."""

    def __init__(self, parent, point_a, point_b):
        super().__init__(parent, title="Create Fold Marker",
                         style=wx.DEFAULT_DIALOG_STYLE)

        self.point_a = point_a
        self.point_b = point_b
        self.angle = 90.0
        self.radius = 1.0

        self._create_ui()
        self.Centre()

    def _create_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Info about the placement
        dist = math.sqrt(
            (self.point_b[0] - self.point_a[0]) ** 2 +
            (self.point_b[1] - self.point_a[1]) ** 2
        )
        info_text = wx.StaticText(
            panel,
            label=f"Fold line length: {iu_to_mm(dist):.2f} mm"
        )
        main_sizer.Add(info_text, 0, wx.ALL, 10)

        # Angle input
        angle_sizer = wx.BoxSizer(wx.HORIZONTAL)
        angle_label = wx.StaticText(panel, label="Fold Angle (degrees):")
        self.angle_ctrl = wx.SpinCtrlDouble(
            panel, value="90", min=-180, max=180, inc=5
        )
        self.angle_ctrl.SetDigits(1)
        angle_sizer.Add(angle_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        angle_sizer.Add(self.angle_ctrl, 1, wx.EXPAND)
        main_sizer.Add(angle_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Radius input
        radius_sizer = wx.BoxSizer(wx.HORIZONTAL)
        radius_label = wx.StaticText(panel, label="Bend Radius (mm):")
        self.radius_ctrl = wx.SpinCtrlDouble(
            panel, value="1.0", min=0.1, max=50, inc=0.5
        )
        self.radius_ctrl.SetDigits(2)
        radius_sizer.Add(radius_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        radius_sizer.Add(self.radius_ctrl, 1, wx.EXPAND)
        main_sizer.Add(radius_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Help text
        help_text = wx.StaticText(
            panel,
            label="Positive angle: fold towards you\n"
                  "Negative angle: fold away from you\n"
                  "Radius defines the bend zone width"
        )
        help_text.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(help_text, 0, wx.ALL, 10)

        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(panel, wx.ID_OK)
        cancel_btn = wx.Button(panel, wx.ID_CANCEL)
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(cancel_btn)
        btn_sizer.Realize()
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(main_sizer)
        main_sizer.Fit(self)

        ok_btn.Bind(wx.EVT_BUTTON, self._on_ok)

    def _on_ok(self, event):
        self.angle = self.angle_ctrl.GetValue()
        self.radius = self.radius_ctrl.GetValue()
        self.EndModal(wx.ID_OK)

    def get_angle(self):
        return self.angle

    def get_radius(self):
        return self.radius


class FoldMarkerCreator:
    """Creates fold marker geometry on the PCB."""

    # Layer for fold markers (User.1)
    MARKER_LAYER = pcbnew.User_1

    # Line style for fold markers
    LINE_WIDTH_MM = 0.15

    def __init__(self, board):
        self.board = board

    def create_fold_marker(self, point_a, point_b, angle_degrees, radius_mm):
        """
        Create a fold marker with two parallel lines and a dimension.

        Args:
            point_a: First point (x, y) in internal units
            point_b: Second point (x, y) in internal units
            angle_degrees: Fold angle in degrees
            radius_mm: Bend radius in millimeters

        Returns:
            List of created PCB items
        """
        created_items = []

        # Calculate the direction perpendicular to the fold line
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1:  # Too short
            return created_items

        # Unit perpendicular vector
        perp_x = -dy / length
        perp_y = dx / length

        # Offset for parallel lines (half the bend zone width)
        offset = mm_to_iu(radius_mm / 2)

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

        # Create dimension annotation showing the angle
        dim = self._create_dimension(point_a, point_b, angle_degrees, offset)
        if dim:
            created_items.append(dim)

        # Add all items to the board
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

        # Set line style to dotted
        # In KiCad 7+, use SetStroke with LINE_STYLE
        try:
            stroke = line.GetStroke()
            stroke.SetLineStyle(pcbnew.LINE_STYLE_DASH)
            line.SetStroke(stroke)
        except AttributeError:
            # Fallback for older KiCad versions
            pass

        return line

    def _create_dimension(self, point_a, point_b, angle_degrees, offset):
        """Create a dimension annotation showing the fold angle."""
        try:
            # Calculate dimension position (centered, offset from fold line)
            mid_x = (point_a[0] + point_b[0]) / 2
            mid_y = (point_a[1] + point_b[1]) / 2

            # Calculate perpendicular direction for dimension placement
            dx = point_b[0] - point_a[0]
            dy = point_b[1] - point_a[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length < 1:
                return None

            perp_x = -dy / length
            perp_y = dx / length

            # Place dimension above the fold line
            dim_offset = offset + mm_to_iu(2.0)  # 2mm above the line

            # Create aligned dimension
            dim = pcbnew.PCB_DIM_ALIGNED(self.board)
            dim.SetStart(pcbnew.VECTOR2I(int(point_a[0]), int(point_a[1])))
            dim.SetEnd(pcbnew.VECTOR2I(int(point_b[0]), int(point_b[1])))
            dim.SetLayer(self.MARKER_LAYER)

            # Set the dimension height (perpendicular offset)
            dim.SetHeight(int(dim_offset))

            # Override the text to show angle instead of length
            angle_text = f"{angle_degrees:+.1f}°"
            dim.SetOverrideTextEnabled(True)
            dim.SetOverrideText(angle_text)

            # Update the dimension
            dim.Update()

            return dim

        except Exception as e:
            # Dimension creation failed, create a text item instead
            return self._create_angle_text(point_a, point_b, angle_degrees, offset)

    def _create_angle_text(self, point_a, point_b, angle_degrees, offset):
        """Create a text annotation as fallback for dimension."""
        try:
            # Calculate text position
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
            text.SetText(f"{angle_degrees:+.1f}°")
            text.SetPosition(pcbnew.VECTOR2I(
                int(mid_x + perp_x * text_offset),
                int(mid_y + perp_y * text_offset)
            ))
            text.SetLayer(self.MARKER_LAYER)

            # Set text size
            text.SetTextSize(pcbnew.VECTOR2I(mm_to_iu(1.5), mm_to_iu(1.5)))

            # Calculate rotation to align with fold line
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            text.SetTextAngle(pcbnew.EDA_ANGLE(angle_deg, pcbnew.DEGREES_T))

            return text

        except Exception:
            return None


def run_fold_placer():
    """
    Run the interactive fold placer.

    This is called from the CreateFoldAction plugin.
    """
    board = pcbnew.GetBoard()
    if board is None:
        wx.MessageBox(
            "No board is currently open.",
            "Create Fold",
            wx.OK | wx.ICON_WARNING
        )
        return

    # Get the main KiCad window
    frame = wx.GetTopLevelWindows()[0] if wx.GetTopLevelWindows() else None

    # Show instructions
    result = wx.MessageBox(
        "Click two points on the board to define the fold line.\n\n"
        "After clicking, you'll be prompted to enter the fold angle.\n\n"
        "Click OK to start, or Cancel to abort.",
        "Create Fold - Instructions",
        wx.OK | wx.CANCEL | wx.ICON_INFORMATION
    )

    if result != wx.OK:
        return

    # For now, use a simple dialog-based approach to get coordinates
    # A full interactive mouse-based approach requires deeper KiCad integration
    point_a = _get_point_from_user("Enter Point A", "First point of fold line")
    if point_a is None:
        return

    point_b = _get_point_from_user("Enter Point B", "Second point of fold line")
    if point_b is None:
        return

    # Convert to internal units
    point_a_iu = (mm_to_iu(point_a[0]), mm_to_iu(point_a[1]))
    point_b_iu = (mm_to_iu(point_b[0]), mm_to_iu(point_b[1]))

    # Show angle dialog
    dlg = FoldPlacerDialog(frame, point_a_iu, point_b_iu)
    if dlg.ShowModal() == wx.ID_OK:
        angle = dlg.get_angle()
        radius = dlg.get_radius()

        # Create the fold marker
        creator = FoldMarkerCreator(board)
        items = creator.create_fold_marker(point_a_iu, point_b_iu, angle, radius)

        if items:
            # Refresh the view
            pcbnew.Refresh()

            wx.MessageBox(
                f"Fold marker created!\n\n"
                f"Angle: {angle:+.1f}°\n"
                f"Radius: {radius:.2f} mm\n"
                f"Items created: {len(items)}",
                "Create Fold",
                wx.OK | wx.ICON_INFORMATION
            )
        else:
            wx.MessageBox(
                "Failed to create fold marker.\n"
                "Please check the points are valid.",
                "Create Fold - Error",
                wx.OK | wx.ICON_ERROR
            )

    dlg.Destroy()


def _get_point_from_user(title, message):
    """Get a point from the user via dialog."""
    dlg = PointInputDialog(None, title, message)
    result = dlg.ShowModal()

    if result == wx.ID_OK:
        point = dlg.get_point()
        dlg.Destroy()
        return point

    dlg.Destroy()
    return None


class PointInputDialog(wx.Dialog):
    """Dialog for entering X, Y coordinates."""

    def __init__(self, parent, title, message):
        super().__init__(parent, title=title, style=wx.DEFAULT_DIALOG_STYLE)

        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Message
        msg_text = wx.StaticText(panel, label=message)
        main_sizer.Add(msg_text, 0, wx.ALL, 10)

        # X coordinate
        x_sizer = wx.BoxSizer(wx.HORIZONTAL)
        x_label = wx.StaticText(panel, label="X (mm):")
        self.x_ctrl = wx.SpinCtrlDouble(panel, value="0", min=-1000, max=1000, inc=1)
        self.x_ctrl.SetDigits(3)
        x_sizer.Add(x_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        x_sizer.Add(self.x_ctrl, 1, wx.EXPAND)
        main_sizer.Add(x_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Y coordinate
        y_sizer = wx.BoxSizer(wx.HORIZONTAL)
        y_label = wx.StaticText(panel, label="Y (mm):")
        self.y_ctrl = wx.SpinCtrlDouble(panel, value="0", min=-1000, max=1000, inc=1)
        self.y_ctrl.SetDigits(3)
        y_sizer.Add(y_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        y_sizer.Add(self.y_ctrl, 1, wx.EXPAND)
        main_sizer.Add(y_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Tip
        tip_text = wx.StaticText(
            panel,
            label="Tip: You can read coordinates from KiCad's\n"
                  "status bar when hovering over the board."
        )
        tip_text.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(tip_text, 0, wx.ALL, 10)

        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(panel, wx.ID_OK)
        cancel_btn = wx.Button(panel, wx.ID_CANCEL)
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(cancel_btn)
        btn_sizer.Realize()
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(main_sizer)
        main_sizer.Fit(self)
        self.Centre()

    def get_point(self):
        return (self.x_ctrl.GetValue(), self.y_ctrl.GetValue())

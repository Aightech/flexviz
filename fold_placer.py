"""
Fold Placer - Interactive fold marker creation for KiCad.

This module provides an interactive tool for placing fold markers on PCB boards.
Users can either:
1. Select an existing line on User.1 layer to use as fold centerline
2. Click "Capture" buttons to grab cursor position from the board
3. Manually enter coordinates
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
    """Interactive dialog for creating fold markers."""

    def __init__(self, parent, board):
        super().__init__(parent, title="Create Fold Marker",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.board = board
        self.point_a = None
        self.point_b = None
        self.angle = 90.0
        self.radius = 1.0

        # Try to get points from selection
        self._init_from_selection()

        self._create_ui()
        self.Centre()

    def _init_from_selection(self):
        """Initialize points from selected items."""
        selection = []

        # Get selected items
        for item in self.board.GetDrawings():
            if item.IsSelected():
                selection.append(item)

        # Also check footprints for selected items
        for fp in self.board.GetFootprints():
            for item in fp.GraphicalItems():
                if item.IsSelected():
                    selection.append(item)

        if not selection:
            return

        # If a single line is selected, use its endpoints
        if len(selection) == 1:
            item = selection[0]
            if hasattr(item, 'GetStart') and hasattr(item, 'GetEnd'):
                start = item.GetStart()
                end = item.GetEnd()
                self.point_a = (start.x, start.y)
                self.point_b = (end.x, end.y)
                return

        # If two items are selected, use their positions/centers
        if len(selection) == 2:
            positions = []
            for item in selection:
                if hasattr(item, 'GetPosition'):
                    pos = item.GetPosition()
                    positions.append((pos.x, pos.y))
                elif hasattr(item, 'GetStart'):
                    # Use midpoint for lines
                    start = item.GetStart()
                    end = item.GetEnd()
                    positions.append(((start.x + end.x) // 2, (start.y + end.y) // 2))

            if len(positions) == 2:
                self.point_a = positions[0]
                self.point_b = positions[1]

    def _create_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Instructions
        instructions = wx.StaticText(
            panel,
            label="Define fold line by selecting a line, capturing cursor positions,\n"
                  "or entering coordinates manually."
        )
        main_sizer.Add(instructions, 0, wx.ALL, 10)

        # === Point A ===
        point_a_box = wx.StaticBox(panel, label="Point A (Start)")
        point_a_sizer = wx.StaticBoxSizer(point_a_box, wx.VERTICAL)

        a_coord_sizer = wx.BoxSizer(wx.HORIZONTAL)

        a_x_label = wx.StaticText(panel, label="X:")
        self.a_x_ctrl = wx.SpinCtrlDouble(panel, value="0", min=-1000, max=1000, inc=1)
        self.a_x_ctrl.SetDigits(3)

        a_y_label = wx.StaticText(panel, label="Y:")
        self.a_y_ctrl = wx.SpinCtrlDouble(panel, value="0", min=-1000, max=1000, inc=1)
        self.a_y_ctrl.SetDigits(3)

        self.capture_a_btn = wx.Button(panel, label="📍 Capture Cursor")
        self.capture_a_btn.SetToolTip("Click here, then position cursor on board and press Enter")

        a_coord_sizer.Add(a_x_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        a_coord_sizer.Add(self.a_x_ctrl, 1, wx.RIGHT, 10)
        a_coord_sizer.Add(a_y_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        a_coord_sizer.Add(self.a_y_ctrl, 1, wx.RIGHT, 10)
        a_coord_sizer.Add(self.capture_a_btn, 0)

        point_a_sizer.Add(a_coord_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(point_a_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # === Point B ===
        point_b_box = wx.StaticBox(panel, label="Point B (End)")
        point_b_sizer = wx.StaticBoxSizer(point_b_box, wx.VERTICAL)

        b_coord_sizer = wx.BoxSizer(wx.HORIZONTAL)

        b_x_label = wx.StaticText(panel, label="X:")
        self.b_x_ctrl = wx.SpinCtrlDouble(panel, value="0", min=-1000, max=1000, inc=1)
        self.b_x_ctrl.SetDigits(3)

        b_y_label = wx.StaticText(panel, label="Y:")
        self.b_y_ctrl = wx.SpinCtrlDouble(panel, value="0", min=-1000, max=1000, inc=1)
        self.b_y_ctrl.SetDigits(3)

        self.capture_b_btn = wx.Button(panel, label="📍 Capture Cursor")
        self.capture_b_btn.SetToolTip("Click here, then position cursor on board and press Enter")

        b_coord_sizer.Add(b_x_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        b_coord_sizer.Add(self.b_x_ctrl, 1, wx.RIGHT, 10)
        b_coord_sizer.Add(b_y_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        b_coord_sizer.Add(self.b_y_ctrl, 1, wx.RIGHT, 10)
        b_coord_sizer.Add(self.capture_b_btn, 0)

        point_b_sizer.Add(b_coord_sizer, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(point_b_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # === Fold Parameters ===
        params_box = wx.StaticBox(panel, label="Fold Parameters")
        params_sizer = wx.StaticBoxSizer(params_box, wx.VERTICAL)

        # Line length display
        self.length_label = wx.StaticText(panel, label="Line length: -- mm")
        params_sizer.Add(self.length_label, 0, wx.ALL, 5)

        # Angle
        angle_sizer = wx.BoxSizer(wx.HORIZONTAL)
        angle_label = wx.StaticText(panel, label="Fold Angle (°):")
        self.angle_ctrl = wx.SpinCtrlDouble(panel, value="90", min=-180, max=180, inc=5)
        self.angle_ctrl.SetDigits(1)
        angle_sizer.Add(angle_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        angle_sizer.Add(self.angle_ctrl, 1)
        params_sizer.Add(angle_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Radius
        radius_sizer = wx.BoxSizer(wx.HORIZONTAL)
        radius_label = wx.StaticText(panel, label="Bend Radius (mm):")
        self.radius_ctrl = wx.SpinCtrlDouble(panel, value="1.0", min=0.1, max=50, inc=0.5)
        self.radius_ctrl.SetDigits(2)
        radius_sizer.Add(radius_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        radius_sizer.Add(self.radius_ctrl, 1)
        params_sizer.Add(radius_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Zone width display
        self.zone_width_label = wx.StaticText(panel, label="Bend zone width: -- mm")
        params_sizer.Add(self.zone_width_label, 0, wx.ALL, 5)

        main_sizer.Add(params_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Help text
        help_text = wx.StaticText(
            panel,
            label="• Positive angle: fold towards viewer (top side inward)\n"
                  "• Negative angle: fold away from viewer (bottom side inward)\n"
                  "• Tip: Select a line on User.1 layer before running this tool"
        )
        help_text.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(help_text, 0, wx.ALL, 10)

        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        self.ok_btn = wx.Button(panel, wx.ID_OK, "Create Fold")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL)
        btn_sizer.AddButton(self.ok_btn)
        btn_sizer.AddButton(cancel_btn)
        btn_sizer.Realize()
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(main_sizer)
        main_sizer.Fit(self)

        # Set initial values if we got them from selection
        if self.point_a:
            self.a_x_ctrl.SetValue(iu_to_mm(self.point_a[0]))
            self.a_y_ctrl.SetValue(iu_to_mm(self.point_a[1]))
        if self.point_b:
            self.b_x_ctrl.SetValue(iu_to_mm(self.point_b[0]))
            self.b_y_ctrl.SetValue(iu_to_mm(self.point_b[1]))

        # Bind events
        self.capture_a_btn.Bind(wx.EVT_BUTTON, self._on_capture_a)
        self.capture_b_btn.Bind(wx.EVT_BUTTON, self._on_capture_b)
        self.ok_btn.Bind(wx.EVT_BUTTON, self._on_ok)
        self.a_x_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_coord_change)
        self.a_y_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_coord_change)
        self.b_x_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_coord_change)
        self.b_y_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_coord_change)
        self.angle_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_param_change)
        self.radius_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_param_change)

        # Update displays
        self._update_length_display()
        self._update_zone_width_display()

    def _get_cursor_position(self):
        """Get current cursor position from KiCad board view."""
        try:
            # Try to get the PCB frame and its cursor position
            frame = None
            for w in wx.GetTopLevelWindows():
                if 'PCB' in w.GetTitle() or 'Pcbnew' in w.GetTitle():
                    frame = w
                    break

            if frame and hasattr(frame, 'GetScreen'):
                screen = frame.GetScreen()
                if screen and hasattr(screen, 'GetCrossHairPosition'):
                    pos = screen.GetCrossHairPosition()
                    return (pos.x, pos.y)

            # Alternative: try pcbnew module directly
            if hasattr(pcbnew, 'GetBoard'):
                # Get design settings which may have cursor info
                ds = self.board.GetDesignSettings()
                # This is a fallback - may not give cursor position

        except Exception as e:
            print(f"Error getting cursor position: {e}")

        return None

    def _on_capture_a(self, event):
        """Capture cursor position for Point A."""
        self._capture_point_interactive('A')

    def _on_capture_b(self, event):
        """Capture cursor position for Point B."""
        self._capture_point_interactive('B')

    def _capture_point_interactive(self, point_name):
        """Show instructions and capture point."""
        # Minimize dialog to let user see the board
        self.Iconize(True)

        # Show instruction popup
        wx.MessageBox(
            f"Position your cursor on the board where you want Point {point_name}.\n\n"
            "Look at the coordinates in KiCad's status bar (bottom of window),\n"
            "then click OK to enter them.",
            f"Capture Point {point_name}",
            wx.OK | wx.ICON_INFORMATION
        )

        # Restore dialog
        self.Iconize(False)
        self.Raise()

        # Try to get cursor position automatically
        pos = self._get_cursor_position()
        if pos:
            if point_name == 'A':
                self.a_x_ctrl.SetValue(iu_to_mm(pos[0]))
                self.a_y_ctrl.SetValue(iu_to_mm(pos[1]))
            else:
                self.b_x_ctrl.SetValue(iu_to_mm(pos[0]))
                self.b_y_ctrl.SetValue(iu_to_mm(pos[1]))
            self._update_length_display()
        else:
            # Manual entry fallback
            dlg = QuickCoordDialog(self, f"Enter Point {point_name} Coordinates")
            if dlg.ShowModal() == wx.ID_OK:
                x, y = dlg.get_coords()
                if point_name == 'A':
                    self.a_x_ctrl.SetValue(x)
                    self.a_y_ctrl.SetValue(y)
                else:
                    self.b_x_ctrl.SetValue(x)
                    self.b_y_ctrl.SetValue(y)
                self._update_length_display()
            dlg.Destroy()

    def _on_coord_change(self, event):
        """Handle coordinate value change."""
        self._update_length_display()

    def _on_param_change(self, event):
        """Handle parameter value change."""
        self._update_zone_width_display()

    def _update_length_display(self):
        """Update the line length display."""
        ax = self.a_x_ctrl.GetValue()
        ay = self.a_y_ctrl.GetValue()
        bx = self.b_x_ctrl.GetValue()
        by = self.b_y_ctrl.GetValue()

        length = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        self.length_label.SetLabel(f"Line length: {length:.2f} mm")

    def _update_zone_width_display(self):
        """Update the bend zone width display."""
        angle = abs(self.angle_ctrl.GetValue())
        radius = self.radius_ctrl.GetValue()

        if angle > 0.1:
            zone_width = radius * math.radians(angle)
            self.zone_width_label.SetLabel(f"Bend zone width: {zone_width:.2f} mm")
        else:
            self.zone_width_label.SetLabel("Bend zone width: -- (flat)")

    def _on_ok(self, event):
        """Handle OK button."""
        # Validate points are different
        ax = self.a_x_ctrl.GetValue()
        ay = self.a_y_ctrl.GetValue()
        bx = self.b_x_ctrl.GetValue()
        by = self.b_y_ctrl.GetValue()

        length = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        if length < 0.1:
            wx.MessageBox(
                "Points A and B are too close together.\n"
                "Please define a longer fold line.",
                "Invalid Fold Line",
                wx.OK | wx.ICON_WARNING
            )
            return

        self.point_a = (mm_to_iu(ax), mm_to_iu(ay))
        self.point_b = (mm_to_iu(bx), mm_to_iu(by))
        self.angle = self.angle_ctrl.GetValue()
        self.radius = self.radius_ctrl.GetValue()

        self.EndModal(wx.ID_OK)

    def get_result(self):
        """Get the fold marker parameters."""
        return self.point_a, self.point_b, self.angle, self.radius


class QuickCoordDialog(wx.Dialog):
    """Quick dialog for entering coordinates from status bar."""

    def __init__(self, parent, title):
        super().__init__(parent, title=title, style=wx.DEFAULT_DIALOG_STYLE)

        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        info = wx.StaticText(
            panel,
            label="Enter the coordinates from KiCad's status bar:"
        )
        main_sizer.Add(info, 0, wx.ALL, 10)

        # Coordinate entry
        coord_sizer = wx.BoxSizer(wx.HORIZONTAL)

        x_label = wx.StaticText(panel, label="X:")
        self.x_ctrl = wx.TextCtrl(panel, value="0", style=wx.TE_PROCESS_ENTER)
        y_label = wx.StaticText(panel, label="Y:")
        self.y_ctrl = wx.TextCtrl(panel, value="0", style=wx.TE_PROCESS_ENTER)

        coord_sizer.Add(x_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        coord_sizer.Add(self.x_ctrl, 1, wx.RIGHT, 10)
        coord_sizer.Add(y_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        coord_sizer.Add(self.y_ctrl, 1)

        main_sizer.Add(coord_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Tip
        tip = wx.StaticText(
            panel,
            label="Tip: You can paste values like '12.5' or '12,5'"
        )
        tip.SetForegroundColour(wx.Colour(128, 128, 128))
        main_sizer.Add(tip, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

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

        # Bind enter key
        self.x_ctrl.Bind(wx.EVT_TEXT_ENTER, lambda e: self.y_ctrl.SetFocus())
        self.y_ctrl.Bind(wx.EVT_TEXT_ENTER, lambda e: self.EndModal(wx.ID_OK))

    def get_coords(self):
        """Get coordinates, handling comma/dot decimal separators."""
        x_str = self.x_ctrl.GetValue().replace(',', '.')
        y_str = self.y_ctrl.GetValue().replace(',', '.')
        try:
            return float(x_str), float(y_str)
        except ValueError:
            return 0.0, 0.0


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

        # Calculate zone width from radius and angle
        angle_rad = math.radians(abs(angle_degrees))
        if angle_rad > 0.001:
            zone_width = radius_mm * angle_rad
        else:
            zone_width = 0.1  # Minimum zone width

        # Offset for parallel lines (half the bend zone width)
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
        try:
            stroke = line.GetStroke()
            stroke.SetLineStyle(pcbnew.LINE_STYLE_DASH)
            line.SetStroke(stroke)
        except AttributeError:
            pass

        return line

    def _create_dimension(self, point_a, point_b, angle_degrees, offset):
        """Create a dimension annotation showing the fold angle."""
        try:
            # Place dimension above the fold line
            dim_offset = offset + mm_to_iu(2.0)

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

            dim.Update()

            return dim

        except Exception as e:
            # Dimension creation failed, create a text item instead
            return self._create_angle_text(point_a, point_b, angle_degrees, offset)

    def _create_angle_text(self, point_a, point_b, angle_degrees, offset):
        """Create a text annotation as fallback for dimension."""
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
            text.SetText(f"{angle_degrees:+.1f}°")
            text.SetPosition(pcbnew.VECTOR2I(
                int(mid_x + perp_x * text_offset),
                int(mid_y + perp_y * text_offset)
            ))
            text.SetLayer(self.MARKER_LAYER)
            text.SetTextSize(pcbnew.VECTOR2I(mm_to_iu(1.5), mm_to_iu(1.5)))

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
    frame = None
    for w in wx.GetTopLevelWindows():
        if 'PCB' in w.GetTitle() or 'Pcbnew' in w.GetTitle():
            frame = w
            break

    # Show the fold placer dialog
    dlg = FoldPlacerDialog(frame, board)

    if dlg.ShowModal() == wx.ID_OK:
        point_a, point_b, angle, radius = dlg.get_result()

        # Create the fold marker
        creator = FoldMarkerCreator(board)
        items = creator.create_fold_marker(point_a, point_b, angle, radius)

        if items:
            # Refresh the view
            pcbnew.Refresh()

            # Calculate zone width for display
            angle_rad = math.radians(abs(angle))
            zone_width = radius * angle_rad if angle_rad > 0.001 else 0

            wx.MessageBox(
                f"Fold marker created on User.1 layer!\n\n"
                f"Angle: {angle:+.1f}°\n"
                f"Bend radius: {radius:.2f} mm\n"
                f"Zone width: {zone_width:.2f} mm\n"
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

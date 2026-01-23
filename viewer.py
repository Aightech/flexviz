"""
3D Viewer for flex PCB visualization.

Uses wxPython + OpenGL (wx.glcanvas) for rendering.
No external dependencies required - uses packages bundled with KiCad.
"""

import os
import wx
import wx.glcanvas as glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
import math

try:
    from .mesh import Mesh, create_board_geometry_mesh
    from .bend_transform import FoldDefinition, create_fold_definitions
    from .geometry import BoardGeometry, extract_geometry
    from .markers import FoldMarker, detect_fold_markers
    from .config import FlexConfig, FLEX_THICKNESS_PRESETS, STIFFENER_THICKNESS_PRESETS
    from .kicad_parser import KiCadPCB
    from .stiffener import extract_stiffeners
    from .validation import validate_design, get_fold_radius_status, ValidationResult
except ImportError:
    from mesh import Mesh, create_board_geometry_mesh
    from bend_transform import FoldDefinition, create_fold_definitions
    from geometry import BoardGeometry, extract_geometry
    from markers import FoldMarker, detect_fold_markers
    from config import FlexConfig, FLEX_THICKNESS_PRESETS, STIFFENER_THICKNESS_PRESETS
    from kicad_parser import KiCadPCB
    from stiffener import extract_stiffeners
    from validation import validate_design, get_fold_radius_status, ValidationResult


class GLCanvas(glcanvas.GLCanvas):
    """OpenGL canvas for 3D rendering."""

    def __init__(self, parent):
        attribs = [
            glcanvas.WX_GL_RGBA,
            glcanvas.WX_GL_DOUBLEBUFFER,
            glcanvas.WX_GL_DEPTH_SIZE, 24,
            glcanvas.WX_GL_STENCIL_SIZE, 8,
            0
        ]
        super().__init__(parent, attribList=attribs)

        self.context = glcanvas.GLContext(self)
        self.initialized = False

        # Camera state
        self.camera_distance = 100.0
        self.camera_rot_x = 30.0  # Pitch
        self.camera_rot_z = 45.0  # Yaw
        self.camera_target = [0.0, 0.0, 0.0]

        # Mouse state
        self.last_mouse_pos = None
        self.mouse_mode = None  # 'rotate', 'pan', 'zoom'

        # Mesh data
        self.mesh = None
        self.display_list = None

        # Display options
        self.show_wireframe = False
        self.show_faces = True

        # Bind events
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.on_mouse)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_wheel)

    def init_gl(self):
        """Initialize OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Light position
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

        # Background color (dark gray)
        glClearColor(0.2, 0.2, 0.2, 1.0)

        # Enable smooth shading
        glShadeModel(GL_SMOOTH)

        self.initialized = True

    def set_mesh(self, mesh: Mesh):
        """Set the mesh to display."""
        self.mesh = mesh

        # Delete old display list
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
            self.display_list = None

        # Auto-center camera on mesh
        if mesh and mesh.vertices:
            xs = [v[0] for v in mesh.vertices]
            ys = [v[1] for v in mesh.vertices]
            zs = [v[2] for v in mesh.vertices]

            self.camera_target = [
                (min(xs) + max(xs)) / 2,
                (min(ys) + max(ys)) / 2,
                (min(zs) + max(zs)) / 2
            ]

            # Set camera distance based on mesh size
            size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            self.camera_distance = size * 2

        self.Refresh()

    def build_display_list(self):
        """Build OpenGL display list for the mesh."""
        if self.mesh is None:
            return

        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)

        # Draw faces
        if self.show_faces:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            for i, face in enumerate(self.mesh.faces):
                # Get color for this face
                if i < len(self.mesh.colors):
                    r, g, b = self.mesh.colors[i]
                    glColor3f(r / 255.0, g / 255.0, b / 255.0)
                else:
                    glColor3f(0.2, 0.6, 0.2)  # Default green

                # Get normal
                if i < len(self.mesh.normals):
                    glNormal3fv(self.mesh.normals[i])

                # Draw polygon
                if len(face) == 3:
                    glBegin(GL_TRIANGLES)
                elif len(face) == 4:
                    glBegin(GL_QUADS)
                else:
                    glBegin(GL_POLYGON)

                for vi in face:
                    if vi < len(self.mesh.vertices):
                        glVertex3fv(self.mesh.vertices[vi])

                glEnd()

        # Draw wireframe overlay
        if self.show_wireframe:
            glDisable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.0)

            for face in self.mesh.faces:
                if len(face) == 3:
                    glBegin(GL_TRIANGLES)
                elif len(face) == 4:
                    glBegin(GL_QUADS)
                else:
                    glBegin(GL_POLYGON)

                for vi in face:
                    if vi < len(self.mesh.vertices):
                        glVertex3fv(self.mesh.vertices[vi])

                glEnd()

            glEnable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glEndList()

    def on_paint(self, event):
        """Handle paint event."""
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)

        if not self.initialized:
            self.init_gl()

        # Build display list if needed
        if self.mesh is not None and self.display_list is None:
            self.build_display_list()

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up projection
        width, height = self.GetClientSize()
        if height == 0:
            height = 1

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / height, 0.1, 10000.0)

        # Set up modelview (camera)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera transformation
        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_rot_x, 1, 0, 0)
        glRotatef(self.camera_rot_z, 0, 0, 1)
        # Flip Y axis: KiCad Y points down, OpenGL Y points up
        glScalef(1.0, -1.0, 1.0)
        glTranslatef(-self.camera_target[0], -self.camera_target[1], -self.camera_target[2])

        # Draw mesh
        if self.display_list is not None:
            glCallList(self.display_list)

        # Draw axes for reference
        self.draw_axes()

        self.SwapBuffers()

    def draw_axes(self):
        """Draw coordinate axes."""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)

        axis_length = self.camera_distance * 0.1

        glBegin(GL_LINES)
        # X axis - red
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)

        # Y axis - green
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)

        # Z axis - blue
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        glEnd()

        glEnable(GL_LIGHTING)

    def on_size(self, event):
        """Handle resize event."""
        self.Refresh()

    def on_mouse(self, event):
        """Handle mouse events."""
        if event.LeftDown():
            self.last_mouse_pos = event.GetPosition()
            self.mouse_mode = 'rotate'
            self.CaptureMouse()

        elif event.MiddleDown():
            self.last_mouse_pos = event.GetPosition()
            self.mouse_mode = 'pan'
            self.CaptureMouse()

        elif event.RightDown():
            self.last_mouse_pos = event.GetPosition()
            self.mouse_mode = 'zoom'
            self.CaptureMouse()

        elif event.LeftUp() or event.MiddleUp() or event.RightUp():
            self.mouse_mode = None
            if self.HasCapture():
                self.ReleaseMouse()

        elif event.Dragging() and self.last_mouse_pos is not None:
            pos = event.GetPosition()
            dx = pos.x - self.last_mouse_pos.x
            dy = pos.y - self.last_mouse_pos.y

            if self.mouse_mode == 'rotate':
                self.camera_rot_z += dx * 0.5
                self.camera_rot_x += dy * 0.5
                # Allow full rotation (no clamping) to view from below

            elif self.mouse_mode == 'pan':
                scale = self.camera_distance * 0.002
                # Pan in screen space
                rad_z = math.radians(self.camera_rot_z)
                self.camera_target[0] -= (dx * math.cos(rad_z) + dy * math.sin(rad_z)) * scale
                self.camera_target[1] -= (-dx * math.sin(rad_z) + dy * math.cos(rad_z)) * scale

            elif self.mouse_mode == 'zoom':
                self.camera_distance *= 1.0 + dy * 0.01
                self.camera_distance = max(1.0, min(10000.0, self.camera_distance))

            self.last_mouse_pos = pos
            self.Refresh()

    def on_wheel(self, event):
        """Handle mouse wheel for zoom."""
        rotation = event.GetWheelRotation()
        if rotation > 0:
            self.camera_distance *= 0.9
        else:
            self.camera_distance *= 1.1

        self.camera_distance = max(1.0, min(10000.0, self.camera_distance))
        self.Refresh()

    def set_wireframe(self, show: bool):
        """Toggle wireframe display."""
        self.show_wireframe = show
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
            self.display_list = None
        self.Refresh()

    def refresh_mesh(self):
        """Force mesh display list rebuild."""
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
            self.display_list = None
        self.Refresh()


class FoldSlider(wx.Panel):
    """A labeled control for setting fold angle."""

    def __init__(self, parent, fold_index: int, initial_angle: float, callback):
        super().__init__(parent)

        self.fold_index = fold_index
        self.callback = callback

        # Layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Status indicator (colored circle)
        self.status_indicator = wx.StaticText(self, label="●", size=(15, -1))
        self.status_indicator.SetForegroundColour(wx.Colour(0, 180, 0))  # Default green
        sizer.Add(self.status_indicator, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)

        # Label
        self.label = wx.StaticText(self, label=f"Fold {fold_index + 1}:")
        sizer.Add(self.label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        # Spin control (-360 to 360 degrees)
        self.spin = wx.SpinCtrlDouble(
            self,
            value=str(initial_angle),
            min=-360.0,
            max=360.0,
            inc=1.0,
            size=(80, -1)
        )
        self.spin.SetDigits(1)
        sizer.Add(self.spin, 0, wx.RIGHT, 5)

        # Degree symbol
        sizer.Add(wx.StaticText(self, label="°"), 0, wx.ALIGN_CENTER_VERTICAL)

        self.SetSizer(sizer)

        # Bind event
        self.spin.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_spin)

    def on_spin(self, event):
        """Handle spin control change."""
        value = self.spin.GetValue()
        self.callback(self.fold_index, value)

    def get_angle(self) -> float:
        """Get current angle in degrees."""
        return self.spin.GetValue()

    def set_angle(self, angle: float):
        """Set angle in degrees."""
        self.spin.SetValue(angle)

    def set_status(self, status: str, tooltip: str = ""):
        """Set the status indicator color: 'green', 'yellow', or 'red'."""
        colors = {
            "green": wx.Colour(0, 180, 0),
            "yellow": wx.Colour(220, 180, 0),
            "red": wx.Colour(220, 50, 50),
        }
        self.status_indicator.SetForegroundColour(colors.get(status, colors["green"]))
        if tooltip:
            self.status_indicator.SetToolTip(tooltip)
        self.Refresh()


class FlexViewerFrame(wx.Frame):
    """Main viewer window."""

    def __init__(self, parent=None, board_geometry=None, fold_markers=None,
                 config: FlexConfig = None, pcb: KiCadPCB = None, pcb_filepath: str = None):
        super().__init__(
            parent,
            title="Flex PCB Viewer",
            size=(1100, 750),
            style=wx.DEFAULT_FRAME_STYLE
        )

        self.board_geometry = board_geometry
        self.fold_markers = fold_markers or []
        self.folds = create_fold_definitions(self.fold_markers)
        self.fold_sliders = []

        # Config and PCB reference
        self.pcb = pcb
        self.pcb_filepath = pcb_filepath

        # Load saved config or use provided/default
        if config:
            self.config = config
        elif pcb_filepath:
            # Try to load saved settings for this PCB
            self.config = FlexConfig.load_for_pcb(pcb_filepath)
        else:
            self.config = FlexConfig()

        # Always get flex thickness from PCB board settings (read-only)
        if self.pcb:
            board_info = self.pcb.get_board_info()
            if board_info.thickness > 0:
                self.config.flex_thickness = board_info.thickness

        # Get available user layers from PCB
        self.available_layers = []
        if self.pcb:
            self.available_layers = self.pcb.get_user_layers()
        if not self.available_layers:
            self.available_layers = ["User.1", "User.2", "User.3", "User.4"]

        self.init_ui()
        self.update_mesh()

        self.Centre()

    def init_ui(self):
        """Initialize the UI."""
        # Main splitter
        splitter = wx.SplitterWindow(self)

        # Left panel - 3D view
        self.canvas = GLCanvas(splitter)

        # Right panel - controls (two columns)
        control_panel = wx.Panel(splitter)
        control_columns = wx.BoxSizer(wx.HORIZONTAL)

        # Left column - Fold angles
        left_column = wx.BoxSizer(wx.VERTICAL)

        # Fold angle controls
        fold_box = wx.StaticBox(control_panel, label="Fold Angles")
        fold_sizer = wx.StaticBoxSizer(fold_box, wx.VERTICAL)
        self.fold_panel = control_panel  # Store reference for refresh
        self.fold_sizer = fold_sizer  # Store reference for refresh

        for i, marker in enumerate(self.fold_markers):
            slider = FoldSlider(
                control_panel,
                i,
                marker.angle_degrees,
                self.on_fold_angle_changed
            )
            self.fold_sliders.append(slider)
            fold_sizer.Add(slider, 0, wx.EXPAND | wx.ALL, 5)

        if not self.fold_markers:
            no_folds_label = wx.StaticText(control_panel, label="No fold markers found.\nAdd markers on User.1 layer.")
            fold_sizer.Add(no_folds_label, 0, wx.ALL, 10)

        left_column.Add(fold_sizer, 0, wx.EXPAND | wx.ALL, 5)
        control_columns.Add(left_column, 0, wx.EXPAND)

        # Right column - Settings and options
        right_column = wx.BoxSizer(wx.VERTICAL)

        # Hidden controls (keep as attributes but don't show)
        self.cb_bend = wx.CheckBox(control_panel, label="Bend")
        self.cb_bend.SetValue(True)
        self.cb_bend.Hide()

        self.cb_debug_regions = wx.CheckBox(control_panel, label="Debug Regions")
        self.cb_debug_regions.SetValue(False)
        self.cb_debug_regions.Hide()

        # Display options
        display_box = wx.StaticBox(control_panel, label="Display Options")
        display_sizer = wx.StaticBoxSizer(display_box, wx.VERTICAL)

        self.cb_wireframe = wx.CheckBox(control_panel, label="Show Wireframe")
        self.cb_wireframe.Bind(wx.EVT_CHECKBOX, self.on_wireframe_toggle)
        display_sizer.Add(self.cb_wireframe, 0, wx.ALL, 3)

        self.cb_traces = wx.CheckBox(control_panel, label="Show Traces")
        self.cb_traces.SetValue(True)
        self.cb_traces.Bind(wx.EVT_CHECKBOX, self.on_display_option_changed)
        display_sizer.Add(self.cb_traces, 0, wx.ALL, 3)

        self.cb_pads = wx.CheckBox(control_panel, label="Show Pads")
        self.cb_pads.SetValue(True)
        self.cb_pads.Bind(wx.EVT_CHECKBOX, self.on_display_option_changed)
        display_sizer.Add(self.cb_pads, 0, wx.ALL, 3)

        self.cb_components = wx.CheckBox(control_panel, label="Show Components")
        self.cb_components.SetValue(False)
        self.cb_components.Bind(wx.EVT_CHECKBOX, self.on_display_option_changed)
        display_sizer.Add(self.cb_components, 0, wx.ALL, 3)

        self.cb_stiffeners = wx.CheckBox(control_panel, label="Show Stiffeners")
        self.cb_stiffeners.SetValue(True)
        self.cb_stiffeners.Bind(wx.EVT_CHECKBOX, self.on_display_option_changed)
        display_sizer.Add(self.cb_stiffeners, 0, wx.ALL, 3)

        self.cb_3d_models = wx.CheckBox(control_panel, label="Show 3D Models")
        self.cb_3d_models.SetValue(False)
        self.cb_3d_models.Bind(wx.EVT_CHECKBOX, self.on_display_option_changed)
        display_sizer.Add(self.cb_3d_models, 0, wx.ALL, 3)

        right_column.Add(display_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # PCB Settings
        settings_box = wx.StaticBox(control_panel, label="PCB Settings")
        settings_sizer = wx.StaticBoxSizer(settings_box, wx.VERTICAL)

        # PCB thickness (read-only, from board settings)
        flex_sizer = wx.BoxSizer(wx.HORIZONTAL)
        flex_sizer.Add(wx.StaticText(control_panel, label="PCB thickness:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.label_flex_thickness = wx.StaticText(control_panel, label=f"{self.config.flex_thickness:.2f} mm")
        self.label_flex_thickness.SetFont(self.label_flex_thickness.GetFont().Bold())
        flex_sizer.Add(self.label_flex_thickness, 0, wx.ALIGN_CENTER_VERTICAL)
        settings_sizer.Add(flex_sizer, 0, wx.ALL, 3)

        # Bend subdivisions
        subdiv_sizer = wx.BoxSizer(wx.HORIZONTAL)
        subdiv_sizer.Add(wx.StaticText(control_panel, label="Bend quality:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.spin_subdivisions = wx.SpinCtrl(
            control_panel,
            value=str(self.config.bend_subdivisions),
            min=1, max=32,
            size=(50, -1)
        )
        self.spin_subdivisions.Bind(wx.EVT_SPINCTRL, self.on_settings_changed)
        subdiv_sizer.Add(self.spin_subdivisions, 0, wx.RIGHT, 3)
        subdiv_sizer.Add(wx.StaticText(control_panel, label="strips"), 0, wx.ALIGN_CENTER_VERTICAL)
        settings_sizer.Add(subdiv_sizer, 0, wx.ALL, 3)

        right_column.Add(settings_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Stiffener Settings
        stiffener_box = wx.StaticBox(control_panel, label="Stiffener")
        stiffener_sizer = wx.StaticBoxSizer(stiffener_box, wx.VERTICAL)

        # Stiffener thickness
        stiff_thick_sizer = wx.BoxSizer(wx.HORIZONTAL)
        stiff_thick_sizer.Add(wx.StaticText(control_panel, label="Thickness:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.spin_stiffener_thickness = wx.SpinCtrlDouble(
            control_panel,
            value=str(self.config.stiffener_thickness),
            min=0.0, max=2.0, inc=0.1,
            size=(60, -1)
        )
        self.spin_stiffener_thickness.SetDigits(2)
        self.spin_stiffener_thickness.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_settings_changed)
        stiff_thick_sizer.Add(self.spin_stiffener_thickness, 0, wx.RIGHT, 3)
        stiff_thick_sizer.Add(wx.StaticText(control_panel, label="mm"), 0, wx.ALIGN_CENTER_VERTICAL)
        stiffener_sizer.Add(stiff_thick_sizer, 0, wx.ALL, 3)

        # Top stiffener layer
        layer_choices_with_none = ["(none)"] + self.available_layers
        stiff_top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        stiff_top_sizer.Add(wx.StaticText(control_panel, label="Top layer:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.choice_stiffener_layer_top = wx.Choice(control_panel, choices=layer_choices_with_none, size=(80, -1))
        if self.config.stiffener_layer_top and self.config.stiffener_layer_top in self.available_layers:
            self.choice_stiffener_layer_top.SetSelection(self.available_layers.index(self.config.stiffener_layer_top) + 1)
        else:
            self.choice_stiffener_layer_top.SetSelection(0)  # (none)
        self.choice_stiffener_layer_top.Bind(wx.EVT_CHOICE, self.on_settings_changed)
        stiff_top_sizer.Add(self.choice_stiffener_layer_top, 0)
        stiffener_sizer.Add(stiff_top_sizer, 0, wx.ALL, 3)

        # Bottom stiffener layer
        stiff_bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        stiff_bottom_sizer.Add(wx.StaticText(control_panel, label="Bottom layer:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.choice_stiffener_layer_bottom = wx.Choice(control_panel, choices=layer_choices_with_none, size=(80, -1))
        if self.config.stiffener_layer_bottom and self.config.stiffener_layer_bottom in self.available_layers:
            self.choice_stiffener_layer_bottom.SetSelection(self.available_layers.index(self.config.stiffener_layer_bottom) + 1)
        else:
            self.choice_stiffener_layer_bottom.SetSelection(0)  # (none)
        self.choice_stiffener_layer_bottom.Bind(wx.EVT_CHOICE, self.on_settings_changed)
        stiff_bottom_sizer.Add(self.choice_stiffener_layer_bottom, 0)
        stiffener_sizer.Add(stiff_bottom_sizer, 0, wx.ALL, 3)

        # Stiffener status label
        self.label_stiffener_status = wx.StaticText(control_panel, label="")
        self.label_stiffener_status.SetForegroundColour(wx.Colour(128, 128, 128))
        stiffener_sizer.Add(self.label_stiffener_status, 0, wx.LEFT | wx.BOTTOM, 5)

        right_column.Add(stiffener_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Validation Panel
        validation_box = wx.StaticBox(control_panel, label="Validation")
        validation_sizer = wx.StaticBoxSizer(validation_box, wx.VERTICAL)

        self.validation_text = wx.StaticText(control_panel, label="No issues detected")
        self.validation_text.SetForegroundColour(wx.Colour(0, 128, 0))
        validation_sizer.Add(self.validation_text, 0, wx.ALL, 5)

        # Validation details (hidden by default, shown when there are issues)
        self.validation_details = wx.TextCtrl(
            control_panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_NO_VSCROLL,
            size=(-1, 60)
        )
        self.validation_details.Hide()
        validation_sizer.Add(self.validation_details, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        right_column.Add(validation_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn_refresh = wx.Button(control_panel, label="Refresh")
        btn_refresh.Bind(wx.EVT_BUTTON, self.on_refresh)
        btn_sizer.Add(btn_refresh, 1, wx.ALL, 3)

        btn_reset = wx.Button(control_panel, label="Reset View")
        btn_reset.Bind(wx.EVT_BUTTON, self.on_reset_view)
        btn_sizer.Add(btn_reset, 1, wx.ALL, 3)

        right_column.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 2)

        # Export buttons
        export_sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn_export_obj = wx.Button(control_panel, label="Export OBJ")
        btn_export_obj.Bind(wx.EVT_BUTTON, self.on_export_obj)
        export_sizer.Add(btn_export_obj, 1, wx.ALL, 3)

        btn_export_stl = wx.Button(control_panel, label="Export STL")
        btn_export_stl.Bind(wx.EVT_BUTTON, self.on_export_stl)
        export_sizer.Add(btn_export_stl, 1, wx.ALL, 3)

        right_column.Add(export_sizer, 0, wx.EXPAND | wx.ALL, 2)

        # Save settings button
        btn_save_settings = wx.Button(control_panel, label="Save Settings")
        btn_save_settings.Bind(wx.EVT_BUTTON, self.on_save_settings)
        right_column.Add(btn_save_settings, 0, wx.ALL | wx.EXPAND, 5)

        # Help text
        help_text = wx.StaticText(control_panel, label=(
            "Controls:\n"
            "  Left drag: Rotate\n"
            "  Middle drag: Pan\n"
            "  Scroll: Zoom"
        ))
        right_column.Add(help_text, 0, wx.ALL, 5)

        # Add right column to control_columns
        control_columns.Add(right_column, 1, wx.EXPAND)

        control_panel.SetSizer(control_columns)

        # Set up splitter
        splitter.SplitVertically(self.canvas, control_panel)
        splitter.SetSashPosition(700)
        splitter.SetMinimumPaneSize(350)

    def update_mesh(self):
        """Regenerate mesh with current fold angles."""
        if self.board_geometry is None:
            return

        # Update fold definitions and markers with current slider values
        for i, slider in enumerate(self.fold_sliders):
            angle_deg = slider.get_angle()
            if i < len(self.folds):
                self.folds[i].angle = math.radians(angle_deg)
            if i < len(self.fold_markers):
                self.fold_markers[i].angle_degrees = angle_deg

        # Extract stiffeners if PCB, config available, and display enabled
        stiffeners = None
        show_stiffeners = self.cb_stiffeners.GetValue() if hasattr(self, 'cb_stiffeners') else True

        # Update stiffener status label
        if hasattr(self, 'label_stiffener_status'):
            if not self.pcb:
                self.label_stiffener_status.SetLabel("(No PCB loaded)")
                self.label_stiffener_status.SetForegroundColour(wx.Colour(128, 128, 128))
            elif not self.config.has_stiffener:
                self.label_stiffener_status.SetLabel("(Set thickness > 0)")
                self.label_stiffener_status.SetForegroundColour(wx.Colour(128, 128, 128))
            elif not show_stiffeners:
                self.label_stiffener_status.SetLabel("(Display disabled)")
                self.label_stiffener_status.SetForegroundColour(wx.Colour(128, 128, 128))
            else:
                stiffeners = extract_stiffeners(self.pcb, self.config)
                if stiffeners:
                    top_count = sum(1 for s in stiffeners if s.side == "top")
                    bottom_count = sum(1 for s in stiffeners if s.side == "bottom")
                    parts = []
                    if top_count > 0:
                        parts.append(f"{top_count} top")
                    if bottom_count > 0:
                        parts.append(f"{bottom_count} bottom")
                    self.label_stiffener_status.SetLabel(f"Found: {', '.join(parts)}")
                    self.label_stiffener_status.SetForegroundColour(wx.Colour(0, 128, 0))  # Green
                else:
                    layers = []
                    if self.config.stiffener_layer_top:
                        layers.append(self.config.stiffener_layer_top)
                    if self.config.stiffener_layer_bottom:
                        layers.append(self.config.stiffener_layer_bottom)
                    self.label_stiffener_status.SetLabel(f"No shapes on {', '.join(layers)}")
                    self.label_stiffener_status.SetForegroundColour(wx.Colour(200, 100, 0))  # Orange
        elif self.pcb and self.config.has_stiffener and show_stiffeners:
            stiffeners = extract_stiffeners(self.pcb, self.config)

        # Check if bending is enabled
        bend_enabled = self.cb_bend.GetValue() if hasattr(self, 'cb_bend') else True

        # Determine PCB directory for 3D model path resolution
        pcb_dir = os.path.dirname(self.pcb_filepath) if self.pcb_filepath else None

        # Check if 3D models should be shown
        include_3d_models = self.cb_3d_models.GetValue() if hasattr(self, 'cb_3d_models') else False

        # Generate mesh
        mesh = create_board_geometry_mesh(
            self.board_geometry,
            markers=self.fold_markers,
            include_traces=self.cb_traces.GetValue() if hasattr(self, 'cb_traces') else True,
            include_pads=self.cb_pads.GetValue() if hasattr(self, 'cb_pads') else True,
            include_components=self.cb_components.GetValue() if hasattr(self, 'cb_components') else False,
            num_bend_subdivisions=self.config.bend_subdivisions if hasattr(self, 'config') else 1,
            stiffeners=stiffeners,
            debug_regions=self.cb_debug_regions.GetValue() if hasattr(self, 'cb_debug_regions') else False,
            apply_bend=bend_enabled,
            include_3d_models=include_3d_models,
            pcb_dir=pcb_dir,
            pcb=self.pcb
        )

        self.canvas.set_mesh(mesh)

        # Run validation
        self.run_validation(stiffeners)

    def on_fold_angle_changed(self, fold_index: int, angle: float):
        """Handle fold angle slider change."""
        if fold_index < len(self.folds):
            self.folds[fold_index].angle = math.radians(angle)
        # Also update the fold marker (used by mesh generation)
        if fold_index < len(self.fold_markers):
            self.fold_markers[fold_index].angle_degrees = angle
        self.update_mesh()

    def run_validation(self, stiffeners: list = None):
        """Run validation checks and update UI."""
        if not hasattr(self, 'validation_text'):
            return

        # Get stiffeners if not provided
        if stiffeners is None and self.pcb and self.config.has_stiffener:
            stiffeners = extract_stiffeners(self.pcb, self.config)

        stiffeners = stiffeners or []

        # Run validation
        result = validate_design(
            self.fold_markers,
            self.board_geometry,
            stiffeners,
            self.config
        )

        # Update fold slider status indicators
        for i, slider in enumerate(self.fold_sliders):
            if i < len(self.fold_markers):
                status = get_fold_radius_status(self.fold_markers[i], self.config)

                # Check if this fold has stiffener conflicts
                fold_errors = [w for w in result.get_by_category("stiffener")
                              if w.details.get("fold_index") == i]
                if fold_errors:
                    status = "red"
                    tooltip = fold_errors[0].message
                else:
                    # Get radius warning if any
                    radius_warnings = [w for w in result.get_by_category("bend_radius")
                                      if w.details.get("fold_index") == i]
                    if radius_warnings:
                        tooltip = radius_warnings[0].message
                    else:
                        tooltip = f"Radius: {self.fold_markers[i].radius:.2f}mm"

                slider.set_status(status, tooltip)

        # Update validation panel
        if result.has_errors or result.has_warnings:
            error_count = result.error_count
            warning_count = result.warning_count

            if error_count > 0:
                self.validation_text.SetLabel(f"⚠ {error_count} error(s), {warning_count} warning(s)")
                self.validation_text.SetForegroundColour(wx.Colour(220, 50, 50))
            else:
                self.validation_text.SetLabel(f"⚠ {warning_count} warning(s)")
                self.validation_text.SetForegroundColour(wx.Colour(220, 150, 0))

            # Show details
            details = []
            for w in result.warnings:
                if w.severity != "info":
                    prefix = "❌" if w.severity == "error" else "⚠"
                    details.append(f"{prefix} {w.message}")

            self.validation_details.SetValue("\n".join(details[:5]))  # Show max 5
            self.validation_details.Show()
        else:
            self.validation_text.SetLabel("✓ No issues detected")
            self.validation_text.SetForegroundColour(wx.Colour(0, 128, 0))
            self.validation_details.Hide()

        # Force layout update
        self.validation_text.GetParent().Layout()

    def on_wireframe_toggle(self, event):
        """Handle wireframe toggle."""
        self.canvas.set_wireframe(self.cb_wireframe.GetValue())

    def on_display_option_changed(self, event):
        """Handle display option change."""
        self.update_mesh()

    def on_settings_changed(self, event):
        """Handle PCB settings change."""
        # Update config from UI (flex_thickness is read-only from board settings)

        # Top stiffener layer (index 0 = "(none)")
        top_idx = self.choice_stiffener_layer_top.GetSelection()
        if top_idx > 0 and (top_idx - 1) < len(self.available_layers):
            self.config.stiffener_layer_top = self.available_layers[top_idx - 1]
        else:
            self.config.stiffener_layer_top = ""

        # Bottom stiffener layer (index 0 = "(none)")
        bottom_idx = self.choice_stiffener_layer_bottom.GetSelection()
        if bottom_idx > 0 and (bottom_idx - 1) < len(self.available_layers):
            self.config.stiffener_layer_bottom = self.available_layers[bottom_idx - 1]
        else:
            self.config.stiffener_layer_bottom = ""

        self.config.stiffener_thickness = self.spin_stiffener_thickness.GetValue()
        self.config.bend_subdivisions = self.spin_subdivisions.GetValue()

        # Auto-save settings if we have a PCB filepath
        if self.pcb_filepath:
            try:
                self.config.save_for_pcb(self.pcb_filepath)
            except Exception:
                pass  # Silently ignore save errors during auto-save

        # Refresh mesh with new settings
        self.update_mesh()

    def on_save_settings(self, event):
        """Save settings to file."""
        if self.pcb_filepath:
            try:
                self.config.save_for_pcb(self.pcb_filepath)
                wx.MessageBox(
                    f"Settings saved for:\n{self.pcb_filepath}",
                    "Settings Saved",
                    wx.OK | wx.ICON_INFORMATION
                )
            except Exception as e:
                wx.MessageBox(
                    f"Error saving settings:\n{e}",
                    "Save Error",
                    wx.OK | wx.ICON_ERROR
                )
        else:
            # No PCB file, save to a chosen location
            with wx.FileDialog(
                self,
                "Save Settings",
                wildcard="JSON files (*.json)|*.json",
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
            ) as dialog:
                if dialog.ShowModal() == wx.ID_OK:
                    try:
                        self.config.save(dialog.GetPath())
                        wx.MessageBox(
                            f"Settings saved to:\n{dialog.GetPath()}",
                            "Settings Saved",
                            wx.OK | wx.ICON_INFORMATION
                        )
                    except Exception as e:
                        wx.MessageBox(
                            f"Error saving settings:\n{e}",
                            "Save Error",
                            wx.OK | wx.ICON_ERROR
                        )

    def on_refresh(self, event):
        """Handle refresh button - reload PCB file from disk."""
        if not self.pcb_filepath:
            # No file path - just update mesh with current data
            self.update_mesh()
            return

        try:
            # Reload PCB from disk
            self.pcb = KiCadPCB.load(self.pcb_filepath)
            self.board_geometry = extract_geometry(self.pcb)
            new_markers = detect_fold_markers(self.pcb)

            # Preserve current fold angles if marker count matches
            old_angles = [s.get_angle() for s in self.fold_sliders]

            # Update fold markers
            self.fold_markers = new_markers
            self.folds = create_fold_definitions(new_markers)

            # Rebuild fold sliders if marker count changed
            if len(new_markers) != len(self.fold_sliders):
                # Clear old sliders
                for slider in self.fold_sliders:
                    slider.Destroy()
                self.fold_sliders = []

                # Create new sliders
                for i, marker in enumerate(new_markers):
                    slider = FoldSlider(
                        self.fold_panel, i, marker.angle_degrees,
                        self.on_fold_angle_changed
                    )
                    self.fold_sizer.Add(slider, 0, wx.EXPAND | wx.ALL, 2)
                    self.fold_sliders.append(slider)

                self.fold_panel.Layout()
            else:
                # Restore old angles to sliders
                for i, angle in enumerate(old_angles):
                    if i < len(self.fold_sliders):
                        self.fold_sliders[i].set_angle(angle)

            # Update mesh
            self.update_mesh()

        except Exception as e:
            wx.MessageBox(
                f"Error refreshing from file:\n{e}",
                "Refresh Error",
                wx.OK | wx.ICON_ERROR
            )

    def on_reset_view(self, event):
        """Reset camera to default view."""
        self.canvas.camera_rot_x = 30.0
        self.canvas.camera_rot_z = 45.0
        if self.canvas.mesh and self.canvas.mesh.vertices:
            xs = [v[0] for v in self.canvas.mesh.vertices]
            ys = [v[1] for v in self.canvas.mesh.vertices]
            zs = [v[2] for v in self.canvas.mesh.vertices]
            self.canvas.camera_target = [
                (min(xs) + max(xs)) / 2,
                (min(ys) + max(ys)) / 2,
                (min(zs) + max(zs)) / 2
            ]
            size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            self.canvas.camera_distance = size * 2
        self.canvas.Refresh()

    def on_export_obj(self, event):
        """Export to OBJ file."""
        if self.canvas.mesh is None:
            wx.MessageBox("No mesh to export.", "Export Error", wx.OK | wx.ICON_WARNING)
            return

        with wx.FileDialog(
            self,
            "Export OBJ",
            wildcard="OBJ files (*.obj)|*.obj",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                path = dialog.GetPath()
                self.canvas.mesh.to_obj(path)
                wx.MessageBox(f"Exported to:\n{path}", "Export Complete", wx.OK | wx.ICON_INFORMATION)

    def on_export_stl(self, event):
        """Export to STL file."""
        if self.canvas.mesh is None:
            wx.MessageBox("No mesh to export.", "Export Error", wx.OK | wx.ICON_WARNING)
            return

        with wx.FileDialog(
            self,
            "Export STL",
            wildcard="STL files (*.stl)|*.stl",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                path = dialog.GetPath()
                self.canvas.mesh.to_stl(path)
                wx.MessageBox(f"Exported to:\n{path}", "Export Complete", wx.OK | wx.ICON_INFORMATION)


def show_viewer(
    board_geometry: BoardGeometry,
    fold_markers: list = None,
    standalone: bool = False,
    config: FlexConfig = None,
    pcb: KiCadPCB = None,
    pcb_filepath: str = None
):
    """
    Show the flex viewer window.

    Args:
        board_geometry: Board geometry to display
        fold_markers: List of fold markers
        standalone: If True, create wx.App (for standalone testing)
        config: Flex configuration (optional, uses defaults if not provided)
        pcb: Parsed KiCad PCB for layer queries (optional)
        pcb_filepath: Path to PCB file for saving settings (optional)
    """
    if standalone:
        app = wx.App()
        frame = FlexViewerFrame(
            None, board_geometry, fold_markers,
            config=config, pcb=pcb, pcb_filepath=pcb_filepath
        )
        frame.Show()
        app.MainLoop()
    else:
        frame = FlexViewerFrame(
            None, board_geometry, fold_markers,
            config=config, pcb=pcb, pcb_filepath=pcb_filepath
        )
        frame.Show()


# Standalone test
if __name__ == "__main__":
    from kicad_parser import KiCadPCB
    from markers import detect_fold_markers
    from geometry import extract_geometry
    from config import FlexConfig

    # Test with sample file
    import os
    test_file = os.path.join(os.path.dirname(__file__), "tests/test_data/with_fold.kicad_pcb")

    if os.path.exists(test_file):
        pcb = KiCadPCB.load(test_file)
        geom = extract_geometry(pcb)
        markers = detect_fold_markers(pcb)

        # Load or create config
        config = FlexConfig.load_for_pcb(test_file)

        print(f"Loaded: {test_file}")
        print(f"  Outline: {len(geom.outline)} vertices")
        print(f"  Folds: {len(markers)}")
        print(f"  User layers: {pcb.get_user_layers()}")
        print(f"  Config: flex={config.flex_thickness}mm, stiffener={config.stiffener_thickness}mm")

        show_viewer(geom, markers, standalone=True, config=config, pcb=pcb, pcb_filepath=test_file)
    else:
        print(f"Test file not found: {test_file}")

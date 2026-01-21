"""
3D Viewer for flex PCB visualization.

Uses wxPython + OpenGL (wx.glcanvas) for rendering.
No external dependencies required - uses packages bundled with KiCad.
"""

import wx
import wx.glcanvas as glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
import math

try:
    from .mesh import Mesh, create_board_geometry_mesh
    from .bend_transform import FoldDefinition, create_fold_definitions
    from .geometry import BoardGeometry
    from .markers import FoldMarker
except ImportError:
    from mesh import Mesh, create_board_geometry_mesh
    from bend_transform import FoldDefinition, create_fold_definitions
    from geometry import BoardGeometry
    from markers import FoldMarker


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
                self.camera_rot_x = max(-90, min(90, self.camera_rot_x))

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
    """A labeled slider for controlling fold angle."""

    def __init__(self, parent, fold_index: int, initial_angle: float, callback):
        super().__init__(parent)

        self.fold_index = fold_index
        self.callback = callback

        # Layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Label
        self.label = wx.StaticText(self, label=f"Fold {fold_index + 1}:")
        sizer.Add(self.label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        # Slider (-180 to 180 degrees)
        self.slider = wx.Slider(
            self,
            value=int(initial_angle),
            minValue=-180,
            maxValue=180,
            style=wx.SL_HORIZONTAL
        )
        sizer.Add(self.slider, 1, wx.EXPAND | wx.RIGHT, 5)

        # Value display
        self.value_text = wx.StaticText(self, label=f"{initial_angle:.0f}°", size=(50, -1))
        sizer.Add(self.value_text, 0, wx.ALIGN_CENTER_VERTICAL)

        self.SetSizer(sizer)

        # Bind event
        self.slider.Bind(wx.EVT_SLIDER, self.on_slider)

    def on_slider(self, event):
        """Handle slider change."""
        value = self.slider.GetValue()
        self.value_text.SetLabel(f"{value}°")
        self.callback(self.fold_index, value)

    def get_angle(self) -> float:
        """Get current angle in degrees."""
        return float(self.slider.GetValue())


class FlexViewerFrame(wx.Frame):
    """Main viewer window."""

    def __init__(self, parent=None, board_geometry=None, fold_markers=None):
        super().__init__(
            parent,
            title="Flex PCB Viewer",
            size=(1000, 700),
            style=wx.DEFAULT_FRAME_STYLE
        )

        self.board_geometry = board_geometry
        self.fold_markers = fold_markers or []
        self.folds = create_fold_definitions(self.fold_markers)
        self.fold_sliders = []

        self.init_ui()
        self.update_mesh()

        self.Centre()

    def init_ui(self):
        """Initialize the UI."""
        # Main splitter
        splitter = wx.SplitterWindow(self)

        # Left panel - 3D view
        self.canvas = GLCanvas(splitter)

        # Right panel - controls
        control_panel = wx.Panel(splitter)
        control_sizer = wx.BoxSizer(wx.VERTICAL)

        # Fold angle controls
        fold_box = wx.StaticBox(control_panel, label="Fold Angles")
        fold_sizer = wx.StaticBoxSizer(fold_box, wx.VERTICAL)

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
            no_folds_label = wx.StaticText(control_panel, label="No fold markers found.\nAdd fold markers on User.1 layer.")
            fold_sizer.Add(no_folds_label, 0, wx.ALL, 10)

        control_sizer.Add(fold_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Display options
        display_box = wx.StaticBox(control_panel, label="Display Options")
        display_sizer = wx.StaticBoxSizer(display_box, wx.VERTICAL)

        self.cb_wireframe = wx.CheckBox(control_panel, label="Show Wireframe")
        self.cb_wireframe.Bind(wx.EVT_CHECKBOX, self.on_wireframe_toggle)
        display_sizer.Add(self.cb_wireframe, 0, wx.ALL, 5)

        self.cb_traces = wx.CheckBox(control_panel, label="Show Traces")
        self.cb_traces.SetValue(True)
        self.cb_traces.Bind(wx.EVT_CHECKBOX, self.on_display_option_changed)
        display_sizer.Add(self.cb_traces, 0, wx.ALL, 5)

        self.cb_pads = wx.CheckBox(control_panel, label="Show Pads")
        self.cb_pads.SetValue(True)
        self.cb_pads.Bind(wx.EVT_CHECKBOX, self.on_display_option_changed)
        display_sizer.Add(self.cb_pads, 0, wx.ALL, 5)

        self.cb_components = wx.CheckBox(control_panel, label="Show Components")
        self.cb_components.SetValue(False)
        self.cb_components.Bind(wx.EVT_CHECKBOX, self.on_display_option_changed)
        display_sizer.Add(self.cb_components, 0, wx.ALL, 5)

        control_sizer.Add(display_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn_refresh = wx.Button(control_panel, label="Refresh")
        btn_refresh.Bind(wx.EVT_BUTTON, self.on_refresh)
        btn_sizer.Add(btn_refresh, 1, wx.ALL, 5)

        btn_reset = wx.Button(control_panel, label="Reset View")
        btn_reset.Bind(wx.EVT_BUTTON, self.on_reset_view)
        btn_sizer.Add(btn_reset, 1, wx.ALL, 5)

        control_sizer.Add(btn_sizer, 0, wx.EXPAND)

        # Export buttons
        export_sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn_export_obj = wx.Button(control_panel, label="Export OBJ")
        btn_export_obj.Bind(wx.EVT_BUTTON, self.on_export_obj)
        export_sizer.Add(btn_export_obj, 1, wx.ALL, 5)

        btn_export_stl = wx.Button(control_panel, label="Export STL")
        btn_export_stl.Bind(wx.EVT_BUTTON, self.on_export_stl)
        export_sizer.Add(btn_export_stl, 1, wx.ALL, 5)

        control_sizer.Add(export_sizer, 0, wx.EXPAND)

        # Help text
        help_text = wx.StaticText(control_panel, label=(
            "\nControls:\n"
            "  Left drag: Rotate\n"
            "  Middle drag: Pan\n"
            "  Right drag: Zoom\n"
            "  Scroll: Zoom"
        ))
        control_sizer.Add(help_text, 0, wx.ALL, 10)

        control_panel.SetSizer(control_sizer)

        # Set up splitter
        splitter.SplitVertically(self.canvas, control_panel)
        splitter.SetSashPosition(700)
        splitter.SetMinimumPaneSize(200)

    def update_mesh(self):
        """Regenerate mesh with current fold angles."""
        if self.board_geometry is None:
            return

        # Update fold definitions with current slider values
        for i, slider in enumerate(self.fold_sliders):
            if i < len(self.folds):
                self.folds[i].angle = math.radians(slider.get_angle())

        # Generate mesh
        mesh = create_board_geometry_mesh(
            self.board_geometry,
            folds=self.folds,
            markers=self.fold_markers,
            include_traces=self.cb_traces.GetValue() if hasattr(self, 'cb_traces') else True,
            include_pads=self.cb_pads.GetValue() if hasattr(self, 'cb_pads') else True,
            include_components=self.cb_components.GetValue() if hasattr(self, 'cb_components') else False
        )

        self.canvas.set_mesh(mesh)

    def on_fold_angle_changed(self, fold_index: int, angle: float):
        """Handle fold angle slider change."""
        if fold_index < len(self.folds):
            self.folds[fold_index].angle = math.radians(angle)
            self.update_mesh()

    def on_wireframe_toggle(self, event):
        """Handle wireframe toggle."""
        self.canvas.set_wireframe(self.cb_wireframe.GetValue())

    def on_display_option_changed(self, event):
        """Handle display option change."""
        self.update_mesh()

    def on_refresh(self, event):
        """Handle refresh button."""
        self.update_mesh()

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


def show_viewer(board_geometry: BoardGeometry, fold_markers: list = None, standalone: bool = False):
    """
    Show the flex viewer window.

    Args:
        board_geometry: Board geometry to display
        fold_markers: List of fold markers
        standalone: If True, create wx.App (for standalone testing)
    """
    if standalone:
        app = wx.App()
        frame = FlexViewerFrame(None, board_geometry, fold_markers)
        frame.Show()
        app.MainLoop()
    else:
        frame = FlexViewerFrame(None, board_geometry, fold_markers)
        frame.Show()


# Standalone test
if __name__ == "__main__":
    from kicad_parser import KiCadPCB
    from markers import detect_fold_markers
    from geometry import extract_geometry

    # Test with sample file
    import os
    test_file = os.path.join(os.path.dirname(__file__), "tests/test_data/with_fold.kicad_pcb")

    if os.path.exists(test_file):
        pcb = KiCadPCB.load(test_file)
        geom = extract_geometry(pcb)
        markers = detect_fold_markers(pcb)

        print(f"Loaded: {test_file}")
        print(f"  Outline: {len(geom.outline)} vertices")
        print(f"  Folds: {len(markers)}")

        show_viewer(geom, markers, standalone=True)
    else:
        print(f"Test file not found: {test_file}")

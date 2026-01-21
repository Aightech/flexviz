"""Unit tests for mesh module."""

import pytest
import math
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh import (
    Mesh,
    create_board_mesh,
    create_trace_mesh,
    create_pad_mesh,
    create_component_mesh,
    create_board_geometry_mesh,
    COLOR_BOARD,
    COLOR_COPPER,
    COLOR_PAD,
)
from geometry import (
    Polygon, LineSegment, PadGeometry, ComponentGeometry,
    BoundingBox, BoardGeometry
)
from bend_transform import FoldDefinition


class TestMesh:
    """Tests for Mesh class."""

    def test_create_empty(self):
        """Test creating empty mesh."""
        mesh = Mesh()
        assert len(mesh.vertices) == 0
        assert len(mesh.faces) == 0

    def test_add_vertex(self):
        """Test adding vertices."""
        mesh = Mesh()
        idx = mesh.add_vertex((1.0, 2.0, 3.0))
        assert idx == 0
        assert mesh.vertices[0] == (1.0, 2.0, 3.0)

        idx2 = mesh.add_vertex((4.0, 5.0, 6.0))
        assert idx2 == 1

    def test_add_triangle(self):
        """Test adding triangle."""
        mesh = Mesh()
        v0 = mesh.add_vertex((0, 0, 0))
        v1 = mesh.add_vertex((1, 0, 0))
        v2 = mesh.add_vertex((0, 1, 0))

        mesh.add_triangle(v0, v1, v2)

        assert len(mesh.faces) == 1
        assert mesh.faces[0] == [0, 1, 2]

    def test_add_quad(self):
        """Test adding quad."""
        mesh = Mesh()
        v0 = mesh.add_vertex((0, 0, 0))
        v1 = mesh.add_vertex((1, 0, 0))
        v2 = mesh.add_vertex((1, 1, 0))
        v3 = mesh.add_vertex((0, 1, 0))

        mesh.add_quad(v0, v1, v2, v3)

        assert len(mesh.faces) == 1
        assert mesh.faces[0] == [0, 1, 2, 3]

    def test_merge(self):
        """Test merging meshes."""
        mesh1 = Mesh()
        mesh1.add_vertex((0, 0, 0))
        mesh1.add_vertex((1, 0, 0))
        mesh1.add_triangle(0, 1, 0)

        mesh2 = Mesh()
        mesh2.add_vertex((2, 0, 0))
        mesh2.add_vertex((3, 0, 0))
        mesh2.add_triangle(0, 1, 0)

        mesh1.merge(mesh2)

        assert len(mesh1.vertices) == 4
        assert len(mesh1.faces) == 2
        # Second face should have offset indices
        assert mesh1.faces[1] == [2, 3, 2]

    def test_compute_normals(self):
        """Test normal computation."""
        mesh = Mesh()
        # XY plane triangle
        mesh.add_vertex((0, 0, 0))
        mesh.add_vertex((1, 0, 0))
        mesh.add_vertex((0, 1, 0))
        mesh.add_triangle(0, 1, 2)

        mesh.compute_normals()

        assert len(mesh.normals) == 1
        # Normal should point in Z direction
        assert abs(mesh.normals[0][2]) > 0.9

    def test_to_obj(self):
        """Test OBJ export."""
        mesh = Mesh()
        mesh.add_vertex((0, 0, 0))
        mesh.add_vertex((1, 0, 0))
        mesh.add_vertex((0, 1, 0))
        mesh.add_triangle(0, 1, 2)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            filename = f.name

        try:
            mesh.to_obj(filename)
            assert os.path.exists(filename)

            with open(filename, 'r') as f:
                content = f.read()
                assert 'v 0.000000' in content
                assert 'f 1 2 3' in content
        finally:
            os.unlink(filename)

    def test_to_stl(self):
        """Test STL export."""
        mesh = Mesh()
        mesh.add_vertex((0, 0, 0))
        mesh.add_vertex((1, 0, 0))
        mesh.add_vertex((0, 1, 0))
        mesh.add_triangle(0, 1, 2)

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            mesh.to_stl(filename)
            assert os.path.exists(filename)

            with open(filename, 'r') as f:
                content = f.read()
                assert 'solid kicad_flex_viewer' in content
                assert 'facet normal' in content
                assert 'vertex' in content
        finally:
            os.unlink(filename)


class TestCreateBoardMesh:
    """Tests for board mesh creation."""

    def test_simple_rectangle(self):
        """Test creating mesh from rectangle."""
        outline = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        mesh = create_board_mesh(outline, thickness=1.6)

        # Should have vertices and faces
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_empty_outline(self):
        """Test with empty outline."""
        outline = Polygon([])
        mesh = create_board_mesh(outline, thickness=1.6)

        assert len(mesh.vertices) == 0
        assert len(mesh.faces) == 0

    def test_with_fold(self):
        """Test board mesh with fold."""
        outline = Polygon([(0, 0), (100, 0), (100, 30), (0, 30)])
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        mesh = create_board_mesh(outline, thickness=0.2, folds=[fold])

        # Should have vertices with non-zero z after fold
        z_values = [v[2] for v in mesh.vertices]
        assert max(z_values) > 0 or min(z_values) < -0.2


class TestCreateTraceMesh:
    """Tests for trace mesh creation."""

    def test_simple_trace(self):
        """Test creating trace mesh."""
        segment = LineSegment((10, 15), (90, 15), width=0.5)
        mesh = create_trace_mesh(segment, z_offset=0.01)

        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_trace_with_fold(self):
        """Test trace mesh with fold."""
        segment = LineSegment((10, 15), (90, 15), width=0.5)
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        mesh = create_trace_mesh(segment, z_offset=0.01, folds=[fold])

        # Vertices should have varying z
        z_values = [v[2] for v in mesh.vertices]
        assert len(set(z_values)) > 1  # Not all same z


class TestCreatePadMesh:
    """Tests for pad mesh creation."""

    def test_rect_pad(self):
        """Test rectangular pad mesh."""
        pad = PadGeometry(
            center=(20, 15),
            shape='rect',
            size=(1.0, 0.5)
        )

        mesh = create_pad_mesh(pad, z_offset=0.02)

        assert len(mesh.vertices) == 4
        assert len(mesh.faces) > 0

    def test_circle_pad(self):
        """Test circular pad mesh."""
        pad = PadGeometry(
            center=(20, 15),
            shape='circle',
            size=(1.0, 1.0)
        )

        mesh = create_pad_mesh(pad, z_offset=0.02)

        # Circle is approximated with 16 vertices
        assert len(mesh.vertices) == 16


class TestCreateComponentMesh:
    """Tests for component mesh creation."""

    def test_component_box(self):
        """Test component box mesh."""
        comp = ComponentGeometry(
            reference="R1",
            value="10k",
            center=(20, 15),
            angle=0,
            bounding_box=BoundingBox(18, 14, 22, 16),
            pads=[],
            layer="F.Cu"
        )

        mesh = create_component_mesh(comp, height=2.0)

        # Box should have 8 vertices (4 top + 4 bottom)
        assert len(mesh.vertices) == 8
        assert len(mesh.faces) > 0


class TestCreateBoardGeometryMesh:
    """Tests for complete board geometry mesh."""

    def test_simple_board(self):
        """Test creating mesh from board geometry."""
        board = BoardGeometry(
            outline=Polygon([(0, 0), (100, 0), (100, 50), (0, 50)]),
            thickness=1.6,
            traces={
                'F.Cu': [
                    LineSegment((10, 25), (90, 25), width=0.5),
                    LineSegment((10, 30), (90, 30), width=0.25)
                ]
            }
        )

        mesh = create_board_geometry_mesh(
            board,
            include_traces=True,
            include_pads=False,
            include_components=False
        )

        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_board_with_pads(self):
        """Test board with pads."""
        pad = PadGeometry(
            center=(20, 15),
            shape='rect',
            size=(1.0, 0.5)
        )

        comp = ComponentGeometry(
            reference="R1",
            value="10k",
            center=(20, 15),
            angle=0,
            bounding_box=BoundingBox(18, 14, 22, 16),
            pads=[pad],
            layer="F.Cu"
        )

        board = BoardGeometry(
            outline=Polygon([(0, 0), (100, 0), (100, 50), (0, 50)]),
            thickness=1.6,
            components=[comp]
        )

        mesh = create_board_geometry_mesh(
            board,
            include_traces=False,
            include_pads=True,
            include_components=False
        )

        # Should include pad mesh
        assert len(mesh.vertices) > 8  # More than just the board

    def test_board_with_components(self):
        """Test board with component boxes."""
        comp = ComponentGeometry(
            reference="U1",
            value="IC",
            center=(50, 25),
            angle=0,
            bounding_box=BoundingBox(45, 20, 55, 30),
            pads=[],
            layer="F.Cu"
        )

        board = BoardGeometry(
            outline=Polygon([(0, 0), (100, 0), (100, 50), (0, 50)]),
            thickness=1.6,
            components=[comp]
        )

        mesh = create_board_geometry_mesh(
            board,
            include_traces=False,
            include_pads=False,
            include_components=True
        )

        # Should include component box (8 vertices)
        assert len(mesh.vertices) >= 8

    def test_board_with_folds(self):
        """Test board mesh with fold applied."""
        board = BoardGeometry(
            outline=Polygon([(0, 0), (100, 0), (100, 30), (0, 30)]),
            thickness=0.2
        )

        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        mesh = create_board_geometry_mesh(
            board,
            folds=[fold],
            include_traces=False,
            include_pads=False
        )

        # Check for non-zero z values
        z_values = [v[2] for v in mesh.vertices]
        z_range = max(z_values) - min(z_values)
        assert z_range > 0.1  # Should have some depth variation

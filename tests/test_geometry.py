"""Unit tests for geometry module."""

import pytest
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kicad_parser import KiCadPCB
from geometry import (
    Point2D,
    Point3D,
    BoundingBox,
    Polygon,
    LineSegment,
    PadGeometry,
    ComponentGeometry,
    BoardGeometry,
    extract_geometry,
    subdivide_polygon,
    offset_polygon,
    line_segment_to_ribbon,
    pad_to_polygon,
    component_to_box,
)


class TestPoint2D:
    """Tests for Point2D class."""

    def test_create(self):
        """Test creating a point."""
        p = Point2D(3.0, 4.0)
        assert p.x == 3.0
        assert p.y == 4.0

    def test_iteration(self):
        """Test iterating over point."""
        p = Point2D(3.0, 4.0)
        coords = list(p)
        assert coords == [3.0, 4.0]

    def test_to_tuple(self):
        """Test converting to tuple."""
        p = Point2D(3.0, 4.0)
        assert p.to_tuple() == (3.0, 4.0)


class TestPoint3D:
    """Tests for Point3D class."""

    def test_create(self):
        """Test creating a 3D point."""
        p = Point3D(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_to_tuple(self):
        """Test converting to tuple."""
        p = Point3D(1.0, 2.0, 3.0)
        assert p.to_tuple() == (1.0, 2.0, 3.0)


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_create(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(0, 0, 100, 50)
        assert bbox.min_x == 0
        assert bbox.min_y == 0
        assert bbox.max_x == 100
        assert bbox.max_y == 50

    def test_width_height(self):
        """Test width and height properties."""
        bbox = BoundingBox(10, 20, 110, 70)
        assert bbox.width == 100
        assert bbox.height == 50

    def test_center(self):
        """Test center property."""
        bbox = BoundingBox(0, 0, 100, 50)
        assert bbox.center == (50, 25)

    def test_contains(self):
        """Test contains method."""
        bbox = BoundingBox(0, 0, 100, 50)
        assert bbox.contains(50, 25)
        assert bbox.contains(0, 0)
        assert bbox.contains(100, 50)
        assert not bbox.contains(-1, 25)
        assert not bbox.contains(50, 51)

    def test_expand(self):
        """Test expand method."""
        bbox = BoundingBox(10, 20, 90, 80)
        expanded = bbox.expand(5)
        assert expanded.min_x == 5
        assert expanded.min_y == 15
        assert expanded.max_x == 95
        assert expanded.max_y == 85


class TestPolygon:
    """Tests for Polygon class."""

    def test_create(self):
        """Test creating a polygon."""
        vertices = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(vertices)
        assert len(poly) == 4

    def test_iteration(self):
        """Test iterating over vertices."""
        vertices = [(0, 0), (10, 0), (10, 10)]
        poly = Polygon(vertices)
        assert list(poly) == vertices

    def test_indexing(self):
        """Test vertex indexing."""
        vertices = [(0, 0), (10, 0), (10, 10)]
        poly = Polygon(vertices)
        assert poly[0] == (0, 0)
        assert poly[1] == (10, 0)
        assert poly[2] == (10, 10)

    def test_bounding_box(self):
        """Test bounding box calculation."""
        vertices = [(5, 10), (15, 10), (15, 30), (5, 30)]
        poly = Polygon(vertices)
        bbox = poly.bounding_box
        assert bbox.min_x == 5
        assert bbox.min_y == 10
        assert bbox.max_x == 15
        assert bbox.max_y == 30

    def test_centroid(self):
        """Test centroid calculation."""
        vertices = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(vertices)
        assert poly.centroid == (5, 5)

    def test_edges(self):
        """Test edge generation."""
        vertices = [(0, 0), (10, 0), (10, 10)]
        poly = Polygon(vertices)
        edges = poly.edges()
        assert len(edges) == 3
        assert edges[0] == ((0, 0), (10, 0))
        assert edges[1] == ((10, 0), (10, 10))
        assert edges[2] == ((10, 10), (0, 0))

    def test_empty_polygon(self):
        """Test empty polygon handling."""
        poly = Polygon([])
        assert len(poly) == 0
        assert poly.centroid == (0, 0)
        bbox = poly.bounding_box
        assert bbox.min_x == 0


class TestLineSegment:
    """Tests for LineSegment class."""

    def test_create(self):
        """Test creating a line segment."""
        seg = LineSegment((0, 0), (10, 0), 0.5)
        assert seg.start == (0, 0)
        assert seg.end == (10, 0)
        assert seg.width == 0.5

    def test_length(self):
        """Test length calculation."""
        seg = LineSegment((0, 0), (3, 4))
        assert abs(seg.length - 5.0) < 0.01

    def test_midpoint(self):
        """Test midpoint calculation."""
        seg = LineSegment((0, 0), (10, 20))
        assert seg.midpoint == (5, 10)

    def test_angle(self):
        """Test angle calculation."""
        seg = LineSegment((0, 0), (10, 0))
        assert abs(seg.angle) < 0.01  # Horizontal = 0

        seg = LineSegment((0, 0), (0, 10))
        assert abs(seg.angle - math.pi / 2) < 0.01  # Vertical = 90°


class TestPadGeometry:
    """Tests for PadGeometry class."""

    def test_create_circle(self):
        """Test creating circular pad."""
        pad = PadGeometry(
            center=(10, 20),
            shape='circle',
            size=(1.0, 1.0)
        )
        assert pad.center == (10, 20)
        assert pad.shape == 'circle'

    def test_create_rect(self):
        """Test creating rectangular pad."""
        pad = PadGeometry(
            center=(10, 20),
            shape='rect',
            size=(2.0, 1.0),
            angle=45
        )
        assert pad.shape == 'rect'
        assert pad.angle == 45


class TestExtractGeometry:
    """Tests for geometry extraction."""

    def test_extract_from_rectangle(self, rectangle_pcb_path):
        """Test extracting geometry from rectangle board."""
        if not rectangle_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(rectangle_pcb_path)
        geom = extract_geometry(pcb)

        # Check outline
        assert len(geom.outline) == 4
        bbox = geom.outline.bounding_box
        assert bbox.width == 100
        assert bbox.height == 50

        # Check thickness
        assert geom.thickness == 1.6

        # Check traces
        assert 'F.Cu' in geom.traces
        assert len(geom.traces['F.Cu']) == 2

    def test_extract_from_fold_pcb(self, fold_pcb_path):
        """Test extracting geometry from PCB with components."""
        if not fold_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(fold_pcb_path)
        geom = extract_geometry(pcb)

        # Check components
        assert len(geom.components) == 1
        comp = geom.components[0]
        assert comp.reference == "R1"
        assert len(comp.pads) == 2

    def test_extract_empty_board(self, minimal_pcb_path):
        """Test extracting from empty board."""
        if not minimal_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(minimal_pcb_path)
        geom = extract_geometry(pcb)

        assert len(geom.outline) == 0
        assert geom.thickness == 1.6
        assert len(geom.all_traces) == 0
        assert len(geom.components) == 0


class TestSubdividePolygon:
    """Tests for polygon subdivision."""

    def test_subdivide_long_edge(self):
        """Test subdividing a polygon with long edges."""
        # Square with 20-unit edges
        vertices = [(0, 0), (20, 0), (20, 20), (0, 20)]
        poly = Polygon(vertices)

        # Subdivide with max edge length of 5
        subdivided = subdivide_polygon(poly, 5.0)

        # Each edge should be split into ~4 segments
        # Original 4 vertices + 3 new per edge = 16 vertices
        assert len(subdivided) > 4
        assert len(subdivided) >= 16

    def test_no_subdivision_needed(self):
        """Test polygon that doesn't need subdivision."""
        vertices = [(0, 0), (2, 0), (2, 2), (0, 2)]
        poly = Polygon(vertices)

        # Max edge length larger than any edge
        subdivided = subdivide_polygon(poly, 10.0)

        # Should be unchanged
        assert len(subdivided) == 4

    def test_subdivide_empty(self):
        """Test subdividing empty polygon."""
        poly = Polygon([])
        subdivided = subdivide_polygon(poly, 5.0)
        assert len(subdivided) == 0


class TestOffsetPolygon:
    """Tests for polygon offset."""

    def test_expand_square(self):
        """Test expanding a square polygon."""
        vertices = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(vertices)

        expanded = offset_polygon(poly, 1.0)

        # Expanded polygon should be larger
        assert expanded.bounding_box.width > poly.bounding_box.width
        assert expanded.bounding_box.height > poly.bounding_box.height

    def test_shrink_square(self):
        """Test shrinking a square polygon."""
        vertices = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(vertices)

        shrunk = offset_polygon(poly, -1.0)

        # Shrunk polygon should be smaller
        assert shrunk.bounding_box.width < poly.bounding_box.width
        assert shrunk.bounding_box.height < poly.bounding_box.height


class TestLineSegmentToRibbon:
    """Tests for line segment to ribbon conversion."""

    def test_horizontal_ribbon(self):
        """Test converting horizontal segment to ribbon."""
        seg = LineSegment((0, 0), (10, 0), 2.0)
        ribbon = line_segment_to_ribbon(seg)

        # Should be 4-vertex polygon
        assert len(ribbon) == 4

        # Width should be 2 (±1 from centerline)
        bbox = ribbon.bounding_box
        assert abs(bbox.height - 2.0) < 0.01
        assert abs(bbox.width - 10.0) < 0.01

    def test_vertical_ribbon(self):
        """Test converting vertical segment to ribbon."""
        seg = LineSegment((0, 0), (0, 10), 2.0)
        ribbon = line_segment_to_ribbon(seg)

        bbox = ribbon.bounding_box
        assert abs(bbox.width - 2.0) < 0.01
        assert abs(bbox.height - 10.0) < 0.01

    def test_degenerate_segment(self):
        """Test converting zero-length segment."""
        seg = LineSegment((5, 5), (5, 5), 2.0)
        ribbon = line_segment_to_ribbon(seg)

        # Should return small square
        assert len(ribbon) == 4


class TestPadToPolygon:
    """Tests for pad to polygon conversion."""

    def test_rect_pad(self):
        """Test converting rectangular pad."""
        pad = PadGeometry(
            center=(10, 20),
            shape='rect',
            size=(4, 2)
        )
        poly = pad_to_polygon(pad)

        assert len(poly) == 4
        bbox = poly.bounding_box
        assert abs(bbox.width - 4.0) < 0.01
        assert abs(bbox.height - 2.0) < 0.01

    def test_circle_pad(self):
        """Test converting circular pad."""
        pad = PadGeometry(
            center=(10, 20),
            shape='circle',
            size=(2, 2)
        )
        poly = pad_to_polygon(pad)

        # Should be approximated with 16 vertices
        assert len(poly) == 16

    def test_oval_pad(self):
        """Test converting oval pad."""
        pad = PadGeometry(
            center=(10, 20),
            shape='oval',
            size=(4, 2)
        )
        poly = pad_to_polygon(pad)

        # Should be approximated with 16 vertices
        assert len(poly) == 16

    def test_rotated_pad(self):
        """Test converting rotated pad."""
        pad = PadGeometry(
            center=(0, 0),
            shape='rect',
            size=(4, 2),
            angle=90
        )
        poly = pad_to_polygon(pad)

        # After 90° rotation, width and height should swap
        bbox = poly.bounding_box
        assert abs(bbox.width - 2.0) < 0.1
        assert abs(bbox.height - 4.0) < 0.1


class TestComponentToBox:
    """Tests for component bounding box."""

    def test_component_box(self):
        """Test getting component bounding box polygon."""
        comp = ComponentGeometry(
            reference="R1",
            value="10k",
            center=(20, 15),
            angle=0,
            bounding_box=BoundingBox(18, 14, 22, 16),
            pads=[],
            layer="F.Cu"
        )
        box = component_to_box(comp)

        assert len(box) == 4
        assert box.bounding_box.width == 4
        assert box.bounding_box.height == 2


class TestBoardGeometry:
    """Tests for BoardGeometry class."""

    def test_all_traces(self):
        """Test all_traces property."""
        geom = BoardGeometry(
            outline=Polygon([]),
            thickness=1.6,
            traces={
                'F.Cu': [LineSegment((0, 0), (10, 0))],
                'B.Cu': [LineSegment((0, 5), (10, 5))]
            }
        )

        assert len(geom.all_traces) == 2

    def test_all_pads(self):
        """Test all_pads property."""
        pad1 = PadGeometry((0, 0), 'rect', (1, 1))
        pad2 = PadGeometry((5, 0), 'rect', (1, 1))

        comp = ComponentGeometry(
            reference="U1",
            value="IC",
            center=(2.5, 0),
            angle=0,
            bounding_box=BoundingBox(0, -1, 5, 1),
            pads=[pad1, pad2],
            layer="F.Cu"
        )

        geom = BoardGeometry(
            outline=Polygon([]),
            thickness=1.6,
            components=[comp]
        )

        assert len(geom.all_pads) == 2

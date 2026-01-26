"""Unit tests for markers module."""

import pytest
import math
from pathlib import Path

from kicad_parser import KiCadPCB, GraphicLine, Dimension
from markers import (
    FoldMarker,
    find_dotted_lines,
    find_line_pairs,
    associate_dimensions,
    create_fold_marker,
    detect_fold_markers,
    sort_markers_by_position,
    _line_angle,
    _line_midpoint,
    _line_length,
    _distance_point_to_line,
    _lines_parallel,
    _distance_between_parallel_lines,
    _parse_angle_from_text,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_line_angle_horizontal(self):
        """Test angle of horizontal line."""
        line = GraphicLine(0, 0, 10, 0, "User.1")
        angle = _line_angle(line)
        assert abs(angle) < 0.01  # ~0 radians

    def test_line_angle_vertical(self):
        """Test angle of vertical line."""
        line = GraphicLine(0, 0, 0, 10, "User.1")
        angle = _line_angle(line)
        assert abs(angle - math.pi / 2) < 0.01  # ~90 degrees

    def test_line_angle_diagonal(self):
        """Test angle of 45-degree line."""
        line = GraphicLine(0, 0, 10, 10, "User.1")
        angle = _line_angle(line)
        assert abs(angle - math.pi / 4) < 0.01  # ~45 degrees

    def test_line_midpoint(self):
        """Test midpoint calculation."""
        line = GraphicLine(0, 0, 10, 20, "User.1")
        mid = _line_midpoint(line)
        assert mid == (5, 10)

    def test_line_length(self):
        """Test length calculation."""
        line = GraphicLine(0, 0, 3, 4, "User.1")
        length = _line_length(line)
        assert abs(length - 5.0) < 0.01  # 3-4-5 triangle

    def test_distance_point_to_line(self):
        """Test point-to-line distance."""
        line = GraphicLine(0, 0, 10, 0, "User.1")
        # Point directly above middle of line
        dist = _distance_point_to_line((5, 5), line)
        assert abs(dist - 5.0) < 0.01

    def test_distance_point_to_line_endpoint(self):
        """Test point-to-line distance near endpoint."""
        line = GraphicLine(0, 0, 10, 0, "User.1")
        # Point beyond end of line
        dist = _distance_point_to_line((15, 0), line)
        assert abs(dist - 5.0) < 0.01

    def test_lines_parallel_horizontal(self):
        """Test parallel detection for horizontal lines."""
        line1 = GraphicLine(0, 0, 10, 0, "User.1")
        line2 = GraphicLine(0, 5, 10, 5, "User.1")
        assert _lines_parallel(line1, line2)

    def test_lines_parallel_vertical(self):
        """Test parallel detection for vertical lines."""
        line1 = GraphicLine(0, 0, 0, 10, "User.1")
        line2 = GraphicLine(5, 0, 5, 10, "User.1")
        assert _lines_parallel(line1, line2)

    def test_lines_not_parallel(self):
        """Test non-parallel lines."""
        line1 = GraphicLine(0, 0, 10, 0, "User.1")
        line2 = GraphicLine(0, 0, 10, 10, "User.1")
        assert not _lines_parallel(line1, line2)

    def test_lines_parallel_opposite_direction(self):
        """Test parallel lines drawn in opposite directions."""
        line1 = GraphicLine(0, 0, 10, 0, "User.1")
        line2 = GraphicLine(10, 5, 0, 5, "User.1")  # Reversed direction
        assert _lines_parallel(line1, line2)

    def test_distance_between_parallel_lines(self):
        """Test distance between parallel lines."""
        line1 = GraphicLine(0, 0, 10, 0, "User.1")
        line2 = GraphicLine(0, 5, 10, 5, "User.1")
        dist = _distance_between_parallel_lines(line1, line2)
        assert abs(dist - 5.0) < 0.01


class TestParseAngle:
    """Tests for angle parsing from text."""

    def test_parse_simple_integer(self):
        """Test parsing simple integer angle."""
        assert _parse_angle_from_text("90") == 90.0

    def test_parse_with_degree_symbol(self):
        """Test parsing angle with degree symbol."""
        assert _parse_angle_from_text("90°") == 90.0

    def test_parse_positive_sign(self):
        """Test parsing positive angle."""
        assert _parse_angle_from_text("+90°") == 90.0

    def test_parse_negative_angle(self):
        """Test parsing negative angle."""
        assert _parse_angle_from_text("-45°") == -45.0

    def test_parse_decimal(self):
        """Test parsing decimal angle."""
        assert _parse_angle_from_text("45.5") == 45.5

    def test_parse_with_deg_suffix(self):
        """Test parsing with 'deg' suffix."""
        assert _parse_angle_from_text("90 deg") == 90.0

    def test_parse_empty(self):
        """Test parsing empty string."""
        assert _parse_angle_from_text("") is None

    def test_parse_invalid(self):
        """Test parsing invalid text."""
        assert _parse_angle_from_text("not a number") is None


class TestFoldMarker:
    """Tests for FoldMarker dataclass."""

    def test_create_fold_marker(self):
        """Test creating fold marker."""
        line_a = GraphicLine(40, 0, 40, 30, "User.1")
        line_b = GraphicLine(45, 0, 45, 30, "User.1")

        marker = create_fold_marker(line_a, line_b, 90.0)

        assert marker.angle_degrees == 90.0
        assert abs(marker.zone_width - 5.0) < 0.01
        # radius = zone_width / angle_radians = 5 / (pi/2) ≈ 3.18
        assert abs(marker.radius - 5.0 / (math.pi / 2)) < 0.01

    def test_fold_marker_axis(self):
        """Test fold marker axis calculation."""
        line_a = GraphicLine(40, 0, 40, 30, "User.1")  # Vertical line
        line_b = GraphicLine(45, 0, 45, 30, "User.1")

        marker = create_fold_marker(line_a, line_b, 90.0)

        # Axis should be along the lines (vertical)
        assert abs(marker.axis[0]) < 0.01  # X component ~0
        assert abs(abs(marker.axis[1]) - 1.0) < 0.01  # Y component ~1

    def test_fold_marker_center(self):
        """Test fold marker center calculation."""
        line_a = GraphicLine(40, 0, 40, 30, "User.1")
        line_b = GraphicLine(45, 0, 45, 30, "User.1")

        marker = create_fold_marker(line_a, line_b, 90.0)

        # Center should be at (42.5, 15)
        assert abs(marker.center[0] - 42.5) < 0.01
        assert abs(marker.center[1] - 15.0) < 0.01

    def test_fold_marker_angle_radians(self):
        """Test angle_radians property."""
        line_a = GraphicLine(0, 0, 0, 10, "User.1")
        line_b = GraphicLine(5, 0, 5, 10, "User.1")

        marker = create_fold_marker(line_a, line_b, 90.0)
        assert abs(marker.angle_radians - math.pi / 2) < 0.01

        marker = create_fold_marker(line_a, line_b, 180.0)
        assert abs(marker.angle_radians - math.pi) < 0.01

    def test_fold_marker_negative_angle(self):
        """Test fold marker with negative angle."""
        line_a = GraphicLine(0, 0, 0, 10, "User.1")
        line_b = GraphicLine(5, 0, 5, 10, "User.1")

        marker = create_fold_marker(line_a, line_b, -90.0)
        assert marker.angle_degrees == -90.0
        # Radius uses absolute angle
        assert marker.radius > 0


class TestFindLinePairs:
    """Tests for line pairing algorithm."""

    def test_find_single_pair(self):
        """Test finding a single pair of parallel lines."""
        lines = [
            GraphicLine(0, 0, 0, 30, "User.1"),
            GraphicLine(5, 0, 5, 30, "User.1"),
        ]

        pairs = find_line_pairs(lines)
        assert len(pairs) == 1

    def test_find_multiple_pairs(self):
        """Test finding multiple pairs."""
        lines = [
            GraphicLine(0, 0, 0, 30, "User.1"),
            GraphicLine(5, 0, 5, 30, "User.1"),
            GraphicLine(50, 0, 50, 30, "User.1"),
            GraphicLine(55, 0, 55, 30, "User.1"),
        ]

        pairs = find_line_pairs(lines)
        assert len(pairs) == 2

    def test_no_pairs_perpendicular(self):
        """Test no pairs for perpendicular lines."""
        lines = [
            GraphicLine(0, 0, 10, 0, "User.1"),  # Horizontal
            GraphicLine(5, 0, 5, 10, "User.1"),  # Vertical
        ]

        pairs = find_line_pairs(lines)
        assert len(pairs) == 0

    def test_no_pairs_too_far(self):
        """Test no pairs for lines too far apart."""
        lines = [
            GraphicLine(0, 0, 0, 10, "User.1"),
            GraphicLine(100, 0, 100, 10, "User.1"),  # Very far
        ]

        pairs = find_line_pairs(lines)
        assert len(pairs) == 0

    def test_ignore_very_different_lengths(self):
        """Test that very different length lines aren't paired."""
        lines = [
            GraphicLine(0, 0, 0, 10, "User.1"),  # Short
            GraphicLine(5, 0, 5, 100, "User.1"),  # 10x longer
        ]

        pairs = find_line_pairs(lines)
        assert len(pairs) == 0


class TestDetectFoldMarkers:
    """Tests for full fold marker detection."""

    def test_detect_from_pcb_file(self, fold_pcb_path):
        """Test detecting fold markers from PCB file."""
        if not fold_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(fold_pcb_path)
        markers = detect_fold_markers(pcb)

        assert len(markers) == 2

        # Check angles
        angles = sorted([m.angle_degrees for m in markers])
        assert -45 in angles or -45.0 in angles
        assert 90 in angles or 90.0 in angles

    def test_detect_empty_pcb(self, minimal_pcb_path):
        """Test detecting from PCB with no markers."""
        if not minimal_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(minimal_pcb_path)
        markers = detect_fold_markers(pcb)

        assert len(markers) == 0

    def test_detect_pcb_without_dimensions(self, rectangle_pcb_path):
        """Test detecting from PCB with lines but no dimensions."""
        if not rectangle_pcb_path.exists():
            pytest.skip("Test data file not found")

        pcb = KiCadPCB.load(rectangle_pcb_path)
        markers = detect_fold_markers(pcb)

        # No markers on User.1 layer
        assert len(markers) == 0


class TestSortMarkers:
    """Tests for marker sorting."""

    def test_sort_by_x(self):
        """Test sorting markers by X position."""
        line1a = GraphicLine(10, 0, 10, 30, "User.1")
        line1b = GraphicLine(15, 0, 15, 30, "User.1")
        line2a = GraphicLine(50, 0, 50, 30, "User.1")
        line2b = GraphicLine(55, 0, 55, 30, "User.1")

        marker1 = create_fold_marker(line1a, line1b, 90)
        marker2 = create_fold_marker(line2a, line2b, 45)

        # Pass in reverse order
        sorted_markers = sort_markers_by_position([marker2, marker1], axis='x')

        assert sorted_markers[0].center[0] < sorted_markers[1].center[0]

    def test_sort_by_y(self):
        """Test sorting markers by Y position."""
        line1a = GraphicLine(0, 10, 30, 10, "User.1")
        line1b = GraphicLine(0, 15, 30, 15, "User.1")
        line2a = GraphicLine(0, 50, 30, 50, "User.1")
        line2b = GraphicLine(0, 55, 30, 55, "User.1")

        marker1 = create_fold_marker(line1a, line1b, 90)
        marker2 = create_fold_marker(line2a, line2b, 45)

        sorted_markers = sort_markers_by_position([marker2, marker1], axis='y')

        assert sorted_markers[0].center[1] < sorted_markers[1].center[1]

    def test_sort_auto_axis(self):
        """Test automatic axis detection for sorting."""
        # Markers spread more along X axis
        line1a = GraphicLine(10, 0, 10, 10, "User.1")
        line1b = GraphicLine(15, 0, 15, 10, "User.1")
        line2a = GraphicLine(100, 0, 100, 10, "User.1")
        line2b = GraphicLine(105, 0, 105, 10, "User.1")

        marker1 = create_fold_marker(line1a, line1b, 90)
        marker2 = create_fold_marker(line2a, line2b, 45)

        sorted_markers = sort_markers_by_position([marker2, marker1], axis='auto')

        # Should sort by X since that has more variation
        assert sorted_markers[0].center[0] < sorted_markers[1].center[0]

    def test_sort_empty_list(self):
        """Test sorting empty list."""
        result = sort_markers_by_position([])
        assert result == []

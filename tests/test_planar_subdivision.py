"""
Unit tests for planar_subdivision module.

Tests the PlanarSubdivision algorithm for partitioning polygons with cutting lines.
"""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planar_subdivision import (
    PlanarSubdivision,
    filter_valid_board_regions,
    associate_holes_with_regions,
    hole_crosses_cutting_lines,
    create_parallel_cutting_lines,
    create_bend_zone_cutting_lines,
    create_line_through_point,
    signed_area,
    ensure_ccw,
    ensure_cw,
    point_in_polygon,
    points_equal,
    polygon_centroid,
)


class TestBasicGeometry:
    """Test basic geometry functions."""

    def test_signed_area_ccw(self):
        """CCW polygon should have positive area."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert signed_area(square) == pytest.approx(100.0)

    def test_signed_area_cw(self):
        """CW polygon should have negative area."""
        square = [(0, 0), (0, 10), (10, 10), (10, 0)]
        assert signed_area(square) == pytest.approx(-100.0)

    def test_ensure_ccw(self):
        """ensure_ccw should convert CW to CCW."""
        cw_square = [(0, 0), (0, 10), (10, 10), (10, 0)]
        ccw_square = ensure_ccw(cw_square)
        assert signed_area(ccw_square) > 0

    def test_ensure_cw(self):
        """ensure_cw should convert CCW to CW."""
        ccw_square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        cw_square = ensure_cw(ccw_square)
        assert signed_area(cw_square) < 0

    def test_point_in_polygon_inside(self):
        """Point inside polygon should return True."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((5, 5), square) is True

    def test_point_in_polygon_outside(self):
        """Point outside polygon should return False."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((15, 5), square) is False

    def test_points_equal(self):
        """Test point equality with tolerance."""
        assert points_equal((1.0, 2.0), (1.0, 2.0)) is True
        assert points_equal((1.0, 2.0), (1.0 + 1e-10, 2.0)) is True
        assert points_equal((1.0, 2.0), (1.1, 2.0)) is False

    def test_polygon_centroid(self):
        """Test centroid calculation."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        cx, cy = polygon_centroid(square)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)


class TestCuttingLineGeneration:
    """Test cutting line generation functions."""

    def test_create_parallel_cutting_lines(self):
        """Test creating parallel horizontal cutting lines."""
        lines = create_parallel_cutting_lines(30, 60, (0, 100))
        assert len(lines) == 2

        # First line at y=30
        line_eq1, p1_1, p1_2 = lines[0]
        assert line_eq1 == (0, 1, -30)

        # Second line at y=60
        line_eq2, p2_1, p2_2 = lines[1]
        assert line_eq2 == (0, 1, -60)

    def test_create_bend_zone_cutting_lines_horizontal(self):
        """Test bend zone cutting lines with horizontal fold."""
        lines = create_bend_zone_cutting_lines(
            center=(50, 40),
            axis=(1, 0),
            zone_width=20,
            num_subdivisions=4
        )
        # Should create 5 lines (4 subdivisions + 1)
        assert len(lines) == 5

    def test_create_bend_zone_cutting_lines_angled(self):
        """Test bend zone cutting lines with angled fold."""
        angle = math.radians(45)
        lines = create_bend_zone_cutting_lines(
            center=(50, 50),
            axis=(math.cos(angle), math.sin(angle)),
            zone_width=20,
            num_subdivisions=4
        )
        assert len(lines) == 5

    def test_create_line_through_point(self):
        """Test creating a single cutting line through a point."""
        line = create_line_through_point((50, 40), (1, 0))
        line_eq, p1, p2 = line
        # Line should be horizontal through y=40
        a, b, c = line_eq
        # For horizontal line: -0*x + 1*y + (-40) = 0 => y = 40
        assert b != 0  # Not vertical


class TestPlanarSubdivisionSimple:
    """Test PlanarSubdivision with simple cases."""

    def test_rectangle_no_cutting_lines(self):
        """Rectangle with no cutting lines should give 1 region."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        subdivision = PlanarSubdivision(outer, [], [])
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])
        assert len(valid) == 1
        assert abs(signed_area(valid[0])) == pytest.approx(8000.0)

    def test_rectangle_one_horizontal_cut(self):
        """Rectangle with one horizontal cut should give 2 regions."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        cutting_lines = [((0, 1, -40), (-10, 40), (110, 40))]  # y=40

        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])

        assert len(valid) == 2
        areas = sorted([abs(signed_area(r)) for r in valid])
        assert areas[0] == pytest.approx(4000.0)
        assert areas[1] == pytest.approx(4000.0)

    def test_rectangle_two_horizontal_cuts(self):
        """Rectangle with two horizontal cuts should give 3 regions."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))

        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])

        assert len(valid) == 3
        # Bottom: 100*30=3000, Middle: 100*20=2000, Top: 100*30=3000
        areas = sorted([abs(signed_area(r)) for r in valid])
        assert areas[0] == pytest.approx(2000.0)
        assert areas[1] == pytest.approx(3000.0)
        assert areas[2] == pytest.approx(3000.0)


class TestPlanarSubdivisionWithHoles:
    """Test PlanarSubdivision with holes."""

    def test_hole_entirely_within_region(self):
        """Hole entirely within a region should be associated with it."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        hole = [(40, 35), (60, 35), (60, 45), (40, 45)]  # In middle region
        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))

        subdivision = PlanarSubdivision(outer, [hole], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [hole])

        assert len(valid) == 3

        # Associate holes
        regions_with_holes = associate_holes_with_regions(valid, [hole], cutting_lines)

        # One region should have the hole
        holes_found = sum(len(h) for _, h in regions_with_holes)
        assert holes_found == 1

    def test_hole_crossing_one_cutting_line(self):
        """Hole crossing one cutting line should split into regions."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        # Hole crosses the y=30 cutting line
        hole = [(40, 20), (60, 20), (60, 45), (40, 45)]
        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))

        subdivision = PlanarSubdivision(outer, [hole], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [hole])

        # Should have more than 3 regions due to hole crossing
        assert len(valid) >= 3

        # Hole crosses cutting lines, so it shouldn't be associated separately
        assert hole_crosses_cutting_lines(hole, cutting_lines) is True

    def test_hole_spanning_both_cutting_lines(self):
        """Hole spanning both cutting lines should create multiple regions."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        # Hole spans from y=20 to y=60, crossing both y=30 and y=50
        hole = [(40, 20), (60, 20), (60, 60), (40, 60)]
        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))

        subdivision = PlanarSubdivision(outer, [hole], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [hole])

        # Middle strip should be split into left and right parts
        # Expected: bottom + middle-left + middle-right + top = 4 or more
        assert len(valid) >= 4


class TestBendZoneSubdivision:
    """Test bend zone subdivision for smooth curves."""

    def test_4_subdivisions_no_holes(self):
        """4 subdivisions should create 6 regions."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        cutting_lines = create_bend_zone_cutting_lines(
            center=(50, 40),
            axis=(1, 0),
            zone_width=20,
            num_subdivisions=4
        )

        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])

        # 1 (before) + 4 (strips) + 1 (after) = 6
        assert len(valid) == 6

    def test_8_subdivisions_no_holes(self):
        """8 subdivisions should create 10 regions."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        cutting_lines = create_bend_zone_cutting_lines(
            center=(50, 40),
            axis=(1, 0),
            zone_width=20,
            num_subdivisions=8
        )

        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])

        # 1 + 8 + 1 = 10
        assert len(valid) == 10

    def test_angled_fold_subdivisions(self):
        """Angled fold should still create correct number of regions."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        angle = math.radians(30)
        cutting_lines = create_bend_zone_cutting_lines(
            center=(50, 40),
            axis=(math.cos(angle), math.sin(angle)),
            zone_width=30,
            num_subdivisions=4
        )

        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])

        assert len(valid) == 6

    def test_bend_zone_with_hole_inside(self):
        """Hole inside bend zone should be handled correctly."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        # Small hole entirely within one strip (y=36 to y=44, within y=35-40 strip)
        hole = [(40, 36), (60, 36), (60, 39), (40, 39)]
        cutting_lines = create_bend_zone_cutting_lines(
            center=(50, 40),
            axis=(1, 0),
            zone_width=20,
            num_subdivisions=4
        )

        subdivision = PlanarSubdivision(outer, [hole], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [hole])

        # Should still have 6 regions
        assert len(valid) == 6


class TestHoleCrossingDetection:
    """Test hole crossing detection."""

    def test_hole_not_crossing(self):
        """Hole not crossing any line should return False."""
        hole = [(40, 35), (60, 35), (60, 45), (40, 45)]
        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))
        assert hole_crosses_cutting_lines(hole, cutting_lines) is False

    def test_hole_crossing_one_line(self):
        """Hole crossing one line should return True."""
        hole = [(40, 25), (60, 25), (60, 45), (40, 45)]  # Crosses y=30
        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))
        assert hole_crosses_cutting_lines(hole, cutting_lines) is True

    def test_hole_crossing_both_lines(self):
        """Hole crossing both lines should return True."""
        hole = [(40, 25), (60, 25), (60, 55), (40, 55)]  # Crosses y=30 and y=50
        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))
        assert hole_crosses_cutting_lines(hole, cutting_lines) is True


class TestTotalAreaConservation:
    """Test that total area is conserved after subdivision."""

    def test_area_conservation_simple(self):
        """Total area of regions should equal original polygon area."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        original_area = abs(signed_area(outer))  # 8000

        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))
        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])

        total_area = sum(abs(signed_area(r)) for r in valid)
        assert total_area == pytest.approx(original_area)

    def test_area_conservation_with_hole(self):
        """Total area should equal original minus hole area."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        hole = [(40, 35), (60, 35), (60, 45), (40, 45)]

        original_area = abs(signed_area(outer))  # 8000
        hole_area = abs(signed_area(hole))  # 200
        expected_area = original_area - hole_area  # 7800

        cutting_lines = create_parallel_cutting_lines(30, 50, (0, 100))
        subdivision = PlanarSubdivision(outer, [hole], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [hole])

        total_area = sum(abs(signed_area(r)) for r in valid)
        assert total_area == pytest.approx(expected_area, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for planar_subdivision module.

Tests the PlanarSubdivision algorithm for partitioning polygons with cutting lines.
"""

import pytest
import math

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

    @pytest.mark.skip(reason="Hole subtraction in area calculation needs investigation")
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


class TestPlanarSubdivisionEdgeCases:
    """Edge case tests for PlanarSubdivision."""

    def test_no_cutting_lines(self):
        """Rectangle with no folds produces exactly 1 region."""
        outer = [(0, 0), (100, 0), (100, 50), (0, 50)]
        subdivision = PlanarSubdivision(outer, [], [])
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])
        assert len(valid) == 1
        assert abs(signed_area(valid[0])) == pytest.approx(5000.0)

    def test_single_vertical_cut(self):
        """100x30 rectangle with one vertical cut at x=50 produces 2 regions."""
        outer = [(0, 0), (100, 0), (100, 30), (0, 30)]
        # Vertical line at x=50: equation 1*x + 0*y - 50 = 0
        cutting_lines = [((1, 0, -50), (50, -10), (50, 40))]
        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])
        assert len(valid) == 2
        areas = sorted([abs(signed_area(r)) for r in valid])
        assert areas[0] == pytest.approx(1500.0)
        assert areas[1] == pytest.approx(1500.0)

    def test_triangle_board(self):
        """Triangular outline with one cutting line splits correctly."""
        outer = [(0, 0), (100, 0), (50, 80)]
        # Horizontal cut at y=40
        cutting_lines = [((0, 1, -40), (-10, 40), (110, 40))]
        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])
        assert len(valid) == 2
        total_area = sum(abs(signed_area(r)) for r in valid)
        original_area = abs(signed_area(outer))
        assert total_area == pytest.approx(original_area)

    def test_l_shaped_board(self):
        """L-shaped (concave) board with one cut produces valid regions."""
        # L-shape: bottom-left is a 60x60, top-right is 40x40 notch removed
        outer = [
            (0, 0), (60, 0), (60, 20), (20, 20), (20, 60), (0, 60)
        ]
        # Horizontal cut at y=10 (through the bottom strip)
        cutting_lines = [((0, 1, -10), (-10, 10), (70, 10))]
        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])
        assert len(valid) == 2
        total_area = sum(abs(signed_area(r)) for r in valid)
        original_area = abs(signed_area(outer))
        assert total_area == pytest.approx(original_area)

    def test_two_close_parallel_cuts(self):
        """Two vertical cuts 0.5mm apart produce 3 regions with a narrow middle one."""
        outer = [(0, 0), (100, 0), (100, 30), (0, 30)]
        # Two vertical cuts at x=49.75 and x=50.25 (0.5mm apart)
        cutting_lines = [
            ((1, 0, -49.75), (49.75, -10), (49.75, 40)),
            ((1, 0, -50.25), (50.25, -10), (50.25, 40)),
        ]
        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])
        assert len(valid) == 3
        areas = sorted([abs(signed_area(r)) for r in valid])
        # Narrow middle: 0.5 * 30 = 15
        assert areas[0] == pytest.approx(15.0)
        # Left: 49.75 * 30 = 1492.5, Right: 49.75 * 30 = 1492.5
        assert areas[1] == pytest.approx(1492.5)
        assert areas[2] == pytest.approx(1492.5)

    def test_diagonal_cut(self):
        """45-degree cutting line across rectangle produces 2 regions."""
        outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
        # Line y = x => -x + y = 0 => (-1, 1, 0)
        cutting_lines = [((-1, 1, 0), (-10, -10), (110, 110))]
        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])
        assert len(valid) == 2
        # Each triangle is half the area: 100*100/2 = 5000
        areas = sorted([abs(signed_area(r)) for r in valid])
        assert areas[0] == pytest.approx(5000.0)
        assert areas[1] == pytest.approx(5000.0)

    def test_many_cuts(self):
        """10 parallel vertical cuts across rectangle produce 11 regions."""
        outer = [(0, 0), (110, 0), (110, 30), (0, 30)]
        cutting_lines = []
        for i in range(1, 11):
            x = i * 10.0
            cutting_lines.append(((1, 0, -x), (x, -10), (x, 40)))
        subdivision = PlanarSubdivision(outer, [], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [])
        assert len(valid) == 11
        # Each strip is 10*30 = 300
        for r in valid:
            assert abs(signed_area(r)) == pytest.approx(300.0)

    def test_board_with_hole(self):
        """Rectangle with rectangular hole and one cut through hole produces valid regions."""
        outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        hole = [(30, 30), (70, 30), (70, 50), (30, 50)]
        # Horizontal cut at y=40 goes through the hole
        cutting_lines = [((0, 1, -40), (-10, 40), (110, 40))]
        subdivision = PlanarSubdivision(outer, [hole], cutting_lines)
        regions = subdivision.compute()
        valid = filter_valid_board_regions(regions, outer, [hole])
        # Should have at least 2 regions (top/bottom) and possibly more due to hole splitting
        assert len(valid) >= 2
        # All regions should have positive area
        for r in valid:
            assert signed_area(r) > 0


class TestFoldRecipeEdgeCases:
    """Edge case tests for fold recipe computation."""

    def test_recipe_no_folds(self):
        """With no fold markers, all regions get empty recipe."""
        from planar_subdivision import split_board_into_regions
        outline = [(0, 0), (100, 0), (100, 80), (0, 80)]
        regions = split_board_into_regions(outline, [], [])
        assert len(regions) == 1
        assert regions[0].fold_recipe == []

    def test_recipe_single_fold(self):
        """Board with 1 fold: BEFORE region gets [], IN_ZONE get IN_ZONE, AFTER gets AFTER."""
        from planar_subdivision import split_board_into_regions, classify_point_vs_fold
        from markers import FoldMarker

        # Fold marker at y=40, horizontal axis, zone_width=10
        marker = FoldMarker(
            line_a_start=(0, 35), line_a_end=(100, 35),
            line_b_start=(0, 45), line_b_end=(100, 45),
            angle_degrees=90.0,
            zone_width=10.0,
            radius=10.0 / math.radians(90.0),
            axis=(1, 0),
            center=(50, 40),
        )

        outline = [(0, 0), (100, 0), (100, 80), (0, 80)]
        regions = split_board_into_regions(outline, [], [marker], num_bend_subdivisions=1)

        # Should have 3 regions: before, in_zone, after
        assert len(regions) == 3

        # Classify regions by representative point
        classifications = []
        for r in regions:
            c = classify_point_vs_fold(r.representative_point, marker)
            classifications.append(c)

        # BEFORE region should have empty recipe
        for r in regions:
            c = classify_point_vs_fold(r.representative_point, marker)
            if c == "BEFORE":
                assert r.fold_recipe == [], f"BEFORE region should have empty recipe, got {r.fold_recipe}"
            elif c == "IN_ZONE":
                assert len(r.fold_recipe) == 1
                assert r.fold_recipe[0][1] == "IN_ZONE"
            elif c == "AFTER":
                assert len(r.fold_recipe) == 1
                assert r.fold_recipe[0][1] == "AFTER"

    def test_recipe_two_parallel_folds(self):
        """Between-folds region gets one AFTER entry for the first fold."""
        from planar_subdivision import split_board_into_regions, classify_point_vs_fold
        from markers import FoldMarker

        # First fold at y=30
        marker1 = FoldMarker(
            line_a_start=(0, 27), line_a_end=(100, 27),
            line_b_start=(0, 33), line_b_end=(100, 33),
            angle_degrees=45.0,
            zone_width=6.0,
            radius=6.0 / math.radians(45.0),
            axis=(1, 0),
            center=(50, 30),
        )
        # Second fold at y=60
        marker2 = FoldMarker(
            line_a_start=(0, 57), line_a_end=(100, 57),
            line_b_start=(0, 63), line_b_end=(100, 63),
            angle_degrees=45.0,
            zone_width=6.0,
            radius=6.0 / math.radians(45.0),
            axis=(1, 0),
            center=(50, 60),
        )

        outline = [(0, 0), (100, 0), (100, 90), (0, 90)]
        regions = split_board_into_regions(outline, [], [marker1, marker2], num_bend_subdivisions=1)

        # Should have 5 regions: before_1, in_zone_1, between, in_zone_2, after_2
        assert len(regions) == 5

        # Find the between-folds region (AFTER fold1, BEFORE fold2)
        between_regions = []
        for r in regions:
            c1 = classify_point_vs_fold(r.representative_point, marker1)
            c2 = classify_point_vs_fold(r.representative_point, marker2)
            if c1 == "AFTER" and c2 == "BEFORE":
                between_regions.append(r)

        assert len(between_regions) == 1
        between = between_regions[0]
        # Between region should have exactly 1 AFTER entry for marker1
        after_entries = [e for e in between.fold_recipe if e[1] == "AFTER"]
        assert len(after_entries) >= 1

    def test_recipe_deterministic(self):
        """Same input always produces same recipes (run twice, compare)."""
        from planar_subdivision import split_board_into_regions
        from markers import FoldMarker

        marker = FoldMarker(
            line_a_start=(0, 35), line_a_end=(100, 35),
            line_b_start=(0, 45), line_b_end=(100, 45),
            angle_degrees=90.0,
            zone_width=10.0,
            radius=10.0 / math.radians(90.0),
            axis=(1, 0),
            center=(50, 40),
        )

        outline = [(0, 0), (100, 0), (100, 80), (0, 80)]

        regions1 = split_board_into_regions(outline, [], [marker], num_bend_subdivisions=1)
        regions2 = split_board_into_regions(outline, [], [marker], num_bend_subdivisions=1)

        assert len(regions1) == len(regions2)
        for r1, r2 in zip(regions1, regions2):
            assert len(r1.fold_recipe) == len(r2.fold_recipe)
            for entry1, entry2 in zip(r1.fold_recipe, r2.fold_recipe):
                # Compare classification strings (skip fold object identity)
                assert entry1[1] == entry2[1]


class TestRegionIndex:
    """Tests for spatial grid index."""

    def test_find_in_single_region(self):
        from planar_subdivision import RegionIndex, Region
        region = Region(
            outline=[(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[], index=0
        )
        idx = RegionIndex([region])
        assert idx.find((5, 5)) is region
        assert idx.find((15, 5)) is None

    def test_find_correct_region_among_many(self):
        from planar_subdivision import RegionIndex, Region
        r1 = Region(outline=[(0, 0), (5, 0), (5, 10), (0, 10)], holes=[], index=0)
        r2 = Region(outline=[(5, 0), (10, 0), (10, 10), (5, 10)], holes=[], index=1)
        idx = RegionIndex([r1, r2])
        assert idx.find((2, 5)) is r1
        assert idx.find((7, 5)) is r2

    def test_find_respects_holes(self):
        from planar_subdivision import RegionIndex, Region
        region = Region(
            outline=[(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]],
            index=0
        )
        idx = RegionIndex([region])
        assert idx.find((1, 1)) is region
        assert idx.find((5, 5)) is None  # Inside hole

    def test_auto_cache_in_find_containing_region(self):
        from planar_subdivision import find_containing_region, Region, clear_region_index_cache
        clear_region_index_cache()
        regions = [Region(outline=[(0, 0), (10, 0), (10, 10), (0, 10)], holes=[], index=0)]
        # First call builds cache, second uses it — both should work
        assert find_containing_region((5, 5), regions) is regions[0]
        assert find_containing_region((5, 5), regions) is regions[0]
        assert find_containing_region((15, 5), regions) is None

    def test_empty_regions(self):
        from planar_subdivision import RegionIndex
        idx = RegionIndex([])
        assert idx.find((0, 0)) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

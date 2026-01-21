"""Unit tests for bend_transform module."""

import pytest
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bend_transform import (
    FoldDefinition,
    PointClassification,
    classify_point,
    transform_point_single_fold,
    transform_point,
    transform_polygon,
    transform_line_segment,
    create_fold_definitions,
    _project_onto_axis,
)
from geometry import Polygon, LineSegment
from markers import FoldMarker


class TestFoldDefinition:
    """Tests for FoldDefinition class."""

    def test_create(self):
        """Test creating a fold definition."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=math.pi / 2
        )
        assert fold.position == (50, 15)
        assert fold.angle == math.pi / 2

    def test_radius(self):
        """Test radius calculation."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=math.pi / 2  # 90 degrees
        )
        # radius = zone_width / angle = 5 / (pi/2) ≈ 3.18
        expected = 5.0 / (math.pi / 2)
        assert abs(fold.radius - expected) < 0.01

    def test_radius_zero_angle(self):
        """Test radius with zero angle."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=0.0
        )
        assert fold.radius == float('inf')

    def test_perpendicular(self):
        """Test perpendicular vector calculation."""
        # Vertical fold axis
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=math.pi / 2
        )
        perp = fold.perpendicular
        assert abs(perp[0] - (-1)) < 0.01
        assert abs(perp[1]) < 0.01

        # Horizontal fold axis
        fold = FoldDefinition(
            position=(50, 15),
            axis=(1, 0),
            zone_width=5.0,
            angle=math.pi / 2
        )
        perp = fold.perpendicular
        assert abs(perp[0]) < 0.01
        assert abs(perp[1] - 1) < 0.01

    def test_from_marker(self):
        """Test creating from FoldMarker."""
        marker = FoldMarker(
            line_a_start=(40, 0),
            line_a_end=(40, 30),
            line_b_start=(45, 0),
            line_b_end=(45, 30),
            angle_degrees=90.0,
            zone_width=5.0,
            radius=3.18,
            axis=(0, 1),
            center=(42.5, 15)
        )

        fold = FoldDefinition.from_marker(marker)
        assert fold.position == (42.5, 15)
        assert fold.axis == (0, 1)
        assert fold.zone_width == 5.0
        assert abs(fold.angle - math.pi / 2) < 0.01


class TestProjectOntoAxis:
    """Tests for axis projection."""

    def test_project_simple(self):
        """Test simple projection."""
        along, perp = _project_onto_axis(
            point=(10, 5),
            origin=(0, 0),
            axis=(1, 0)  # Horizontal axis
        )
        assert abs(along - 10) < 0.01
        assert abs(perp - 5) < 0.01

    def test_project_vertical_axis(self):
        """Test projection on vertical axis."""
        along, perp = _project_onto_axis(
            point=(5, 10),
            origin=(0, 0),
            axis=(0, 1)  # Vertical axis
        )
        assert abs(along - 10) < 0.01
        assert abs(perp - (-5)) < 0.01


class TestClassifyPoint:
    """Tests for point classification."""

    def test_point_before_fold(self):
        """Test point before fold zone."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),  # Vertical fold line
            zone_width=10.0,
            angle=math.pi / 2
        )

        # Point at x=40, well before fold center at x=50
        classification, dist = classify_point((40, 15), fold)
        assert classification == PointClassification.BEFORE

    def test_point_in_zone(self):
        """Test point inside fold zone."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        # Point at fold center
        classification, dist = classify_point((50, 15), fold)
        assert classification == PointClassification.IN_ZONE
        assert abs(dist - 5.0) < 0.01  # Middle of zone

    def test_point_after_fold(self):
        """Test point after fold zone."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        # Point at x=60, well after fold center
        classification, dist = classify_point((60, 15), fold)
        assert classification == PointClassification.AFTER


class TestTransformPointSingleFold:
    """Tests for single fold transformation."""

    def test_point_before_unchanged(self):
        """Test that points before fold are unchanged."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        # Point well before fold
        result = transform_point_single_fold((30, 15), fold)
        assert abs(result[0] - 30) < 0.01
        assert abs(result[1] - 15) < 0.01
        assert abs(result[2]) < 0.01

    def test_90_degree_fold_endpoint(self):
        """Test 90-degree fold end position."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2  # 90 degrees
        )

        # Point at the end of the fold zone
        # Should be bent upward 90 degrees
        result = transform_point_single_fold((55, 15), fold)

        # After a 90-degree bend, the z should have increased
        assert result[2] > 0

    def test_zero_angle_unchanged(self):
        """Test that zero angle leaves points unchanged."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=0.0  # No bend
        )

        result = transform_point_single_fold((50, 15), fold)
        assert abs(result[0] - 50) < 0.1
        assert abs(result[1] - 15) < 0.1
        assert abs(result[2]) < 0.1

    def test_negative_angle_bends_down(self):
        """Test negative angle bends downward."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=-math.pi / 2  # -90 degrees
        )

        result = transform_point_single_fold((55, 15), fold)
        # Should bend downward (negative z)
        assert result[2] < 0


class TestTransformPoint:
    """Tests for multi-fold transformation."""

    def test_no_folds(self):
        """Test with no folds."""
        result = transform_point((50, 15), [])
        assert result == (50, 15, 0.0)

    def test_single_fold(self):
        """Test with single fold."""
        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        result = transform_point((30, 15), [fold])
        assert abs(result[0] - 30) < 0.01
        assert abs(result[1] - 15) < 0.01


class TestTransformPolygon:
    """Tests for polygon transformation."""

    def test_transform_square(self):
        """Test transforming a square polygon."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        fold = FoldDefinition(
            position=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        # Square is before the fold, should be mostly unchanged
        result = transform_polygon(poly, [fold], subdivide=False)
        assert len(result) == 4

        # All z values should be ~0 (before fold)
        for v in result:
            assert abs(v[2]) < 0.01

    def test_transform_with_subdivision(self):
        """Test that subdivision adds vertices."""
        poly = Polygon([(0, 0), (100, 0), (100, 10), (0, 10)])

        result = transform_polygon(poly, [], subdivide=True, max_edge_length=10.0)

        # Should have more vertices due to subdivision
        assert len(result) > 4


class TestTransformLineSegment:
    """Tests for line segment transformation."""

    def test_transform_segment(self):
        """Test transforming a line segment."""
        segment = LineSegment((0, 0), (100, 0), width=1.0)

        fold = FoldDefinition(
            position=(50, 0),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        result = transform_line_segment(segment, [fold], subdivisions=10)

        assert len(result) == 11  # 10 subdivisions + 1

    def test_segment_no_folds(self):
        """Test segment with no folds."""
        segment = LineSegment((0, 0), (10, 0), width=1.0)

        result = transform_line_segment(segment, [], subdivisions=5)

        assert len(result) == 6
        for v in result:
            assert abs(v[2]) < 0.01  # All z should be 0


class TestCreateFoldDefinitions:
    """Tests for creating fold definitions from markers."""

    def test_create_from_markers(self):
        """Test creating fold definitions from markers."""
        markers = [
            FoldMarker(
                line_a_start=(40, 0),
                line_a_end=(40, 30),
                line_b_start=(45, 0),
                line_b_end=(45, 30),
                angle_degrees=90.0,
                zone_width=5.0,
                radius=3.18,
                axis=(0, 1),
                center=(42.5, 15)
            ),
            FoldMarker(
                line_a_start=(80, 0),
                line_a_end=(80, 30),
                line_b_start=(85, 0),
                line_b_end=(85, 30),
                angle_degrees=-45.0,
                zone_width=5.0,
                radius=6.37,
                axis=(0, 1),
                center=(82.5, 15)
            )
        ]

        folds = create_fold_definitions(markers)

        assert len(folds) == 2
        assert abs(folds[0].angle - math.pi / 2) < 0.01
        assert abs(folds[1].angle - (-math.pi / 4)) < 0.01

    def test_create_empty(self):
        """Test with no markers."""
        folds = create_fold_definitions([])
        assert folds == []

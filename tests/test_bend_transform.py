"""Unit tests for bend_transform module."""

import pytest
import math
from pathlib import Path

from bend_transform import (
    FoldDefinition,
    transform_point,
    compute_normal,
    transform_point_and_normal,
    create_fold_definitions,
)
from markers import FoldMarker


class TestFoldDefinition:
    """Tests for FoldDefinition class."""

    def test_create(self):
        """Test creating a fold definition."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=math.pi / 2
        )
        assert fold.center == (50, 15)
        assert fold.angle == math.pi / 2

    def test_radius(self):
        """Test radius calculation."""
        fold = FoldDefinition(
            center=(50, 15),
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
            center=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=0.0
        )
        assert fold.radius == float('inf')

    def test_perp(self):
        """Test perpendicular vector calculation."""
        # Vertical fold axis
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=5.0,
            angle=math.pi / 2
        )
        perp = fold.perp
        assert abs(perp[0] - (-1)) < 0.01
        assert abs(perp[1]) < 0.01

        # Horizontal fold axis
        fold = FoldDefinition(
            center=(50, 15),
            axis=(1, 0),
            zone_width=5.0,
            angle=math.pi / 2
        )
        perp = fold.perp
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
        assert fold.center == (42.5, 15)
        assert fold.axis == (0, 1)
        assert fold.zone_width == 5.0
        assert abs(fold.angle - math.pi / 2) < 0.01


class TestTransformPoint:
    """Tests for recipe-based point transformation."""

    def test_empty_recipe(self):
        """Test with empty recipe returns flat point."""
        result = transform_point((50, 15), [])
        assert result == (50, 15, 0.0)

    def test_in_zone_transformation(self):
        """Test point in fold zone gets curved."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),  # Vertical fold
            zone_width=10.0,
            angle=math.pi / 2  # 90 degrees
        )

        # Recipe: point is IN_ZONE
        recipe = [(fold, "IN_ZONE", False)]
        result = transform_point((50, 15), recipe)

        # Point at center of zone should have some z displacement
        # For 90-degree fold at midpoint (45 degrees), z > 0
        assert result[2] > 0

    def test_after_transformation(self):
        """Test point after fold zone gets rotated."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),  # Vertical fold
            zone_width=10.0,
            angle=math.pi / 2  # 90 degrees
        )

        # Recipe: point is AFTER (past the fold zone)
        recipe = [(fold, "AFTER", False)]
        result = transform_point((60, 15), recipe)

        # After 90-degree fold, point should have significant z displacement
        assert abs(result[2]) > 1

    def test_negative_angle_opposite_direction(self):
        """Test negative angle bends in opposite direction."""
        fold_pos = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )
        fold_neg = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=-math.pi / 2
        )

        recipe_pos = [(fold_pos, "AFTER", False)]
        recipe_neg = [(fold_neg, "AFTER", False)]

        result_pos = transform_point((60, 15), recipe_pos)
        result_neg = transform_point((60, 15), recipe_neg)

        # Opposite angles should produce opposite z directions
        assert result_pos[2] * result_neg[2] < 0

    def test_back_entry_mirroring(self):
        """Test back entry mirrors the transformation."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),
            zone_width=10.0,
            angle=math.pi / 2
        )

        # Same position, different entry direction
        recipe_front = [(fold, "AFTER", False)]
        recipe_back = [(fold, "AFTER", True)]

        result_front = transform_point((60, 15), recipe_front)
        result_back = transform_point((40, 15), recipe_back)

        # Back entry should produce mirrored perpendicular position
        # Both should have z displacement (same sign due to mirroring)
        assert abs(result_front[2]) > 1
        assert abs(result_back[2]) > 1


class TestComputeNormal:
    """Tests for normal computation."""

    def test_flat_normal(self):
        """Test normal with no folds is (0, 0, 1)."""
        normal = compute_normal((50, 15), [])
        assert abs(normal[0]) < 0.01
        assert abs(normal[1]) < 0.01
        assert abs(normal[2] - 1.0) < 0.01

    def test_after_fold_normal_rotated(self):
        """Test normal after fold is rotated."""
        fold = FoldDefinition(
            center=(50, 15),
            axis=(0, 1),  # Vertical fold
            zone_width=10.0,
            angle=math.pi / 2  # 90 degrees
        )

        recipe = [(fold, "AFTER", False)]
        normal = compute_normal((60, 15), recipe)

        # After 90-degree fold, normal should be significantly rotated from (0,0,1)
        # The exact direction depends on fold axis orientation
        assert abs(normal[0]) > 0.9 or abs(normal[2]) < 0.1  # Normal is rotated


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

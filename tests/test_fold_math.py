"""Tests for fold_math module — pure geometry computation."""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"))

from fold_math import compute_fold_marker_geometry, FoldMarkerGeometry


class TestComputeFoldMarkerGeometry:
    """Tests for compute_fold_marker_geometry()."""

    def test_returns_geometry(self):
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 90, 1.0)
        assert isinstance(geom, FoldMarkerGeometry)

    def test_degenerate_line_returns_none(self):
        assert compute_fold_marker_geometry((5, 5), (5, 5), 90, 1.0) is None

    def test_zone_width_formula(self):
        """zone_width = R * |angle_rad|"""
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 90, 2.0)
        expected = 2.0 * math.radians(90)
        assert geom.zone_width == pytest.approx(expected, abs=1e-6)

    def test_zone_width_negative_angle(self):
        geom = compute_fold_marker_geometry((0, 0), (10, 0), -90, 2.0)
        expected = 2.0 * math.radians(90)
        assert geom.zone_width == pytest.approx(expected, abs=1e-6)

    def test_parallel_lines_symmetric(self):
        """Lines should be equidistant from centerline."""
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 90, 1.0)
        # Horizontal line → perpendicular is vertical
        # line1 offset up, line2 offset down
        mid_y_1 = (geom.line1_start[1] + geom.line1_end[1]) / 2
        mid_y_2 = (geom.line2_start[1] + geom.line2_end[1]) / 2
        assert mid_y_1 == pytest.approx(-mid_y_2, abs=1e-6)

    def test_lines_parallel_to_input(self):
        """Output lines should be parallel to input line."""
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 90, 1.0)
        # line1 should be horizontal (y constant)
        assert geom.line1_start[1] == pytest.approx(geom.line1_end[1], abs=1e-6)
        assert geom.line2_start[1] == pytest.approx(geom.line2_end[1], abs=1e-6)

    def test_diagonal_line(self):
        """Test with 45-degree input line."""
        geom = compute_fold_marker_geometry((0, 0), (10, 10), 45, 1.0)
        assert geom is not None
        # Lines should still be offset perpendicular to the diagonal
        dx1 = geom.line1_end[0] - geom.line1_start[0]
        dy1 = geom.line1_end[1] - geom.line1_start[1]
        angle1 = math.atan2(dy1, dx1)
        assert angle1 == pytest.approx(math.pi / 4, abs=0.01)

    def test_angle_text_format(self):
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 90, 1.0)
        assert geom.angle_text == "90°"

    def test_angle_text_negative(self):
        geom = compute_fold_marker_geometry((0, 0), (10, 0), -45, 1.0)
        assert geom.angle_text == "-45°"

    def test_dim_start_is_midpoint(self):
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 90, 1.0)
        assert geom.dim_start == pytest.approx((5, 0), abs=1e-6)

    def test_perp_vector_unit_length(self):
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 90, 1.0)
        px, py = geom.perp
        length = math.sqrt(px * px + py * py)
        assert length == pytest.approx(1.0, abs=1e-6)

    def test_small_angle(self):
        """Very small angle should not crash."""
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 0.1, 5.0)
        assert geom is not None
        assert geom.zone_width > 0

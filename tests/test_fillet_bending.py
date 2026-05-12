"""Tests for Phase 6: Fillet bending — adaptive arc subdivision near fold zones."""

import math
import pytest
import sys
from pathlib import Path

# Add plugin directory to path
PLUGIN_DIR = Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"
sys.path.insert(0, str(PLUGIN_DIR))

from geometry import (
    OutlineSegment,
    Polygon,
    arc_crosses_fold_zone,
    _refine_arc_segment,
    refine_outline_for_folds,
)
from markers import FoldMarker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arc_segment(cx, cy, r, start_deg, end_deg):
    """Create an OutlineSegment of type 'arc' from center, radius, and angle range.

    Angles are in degrees, measured counter-clockwise from +X.
    """
    sa = math.radians(start_deg)
    ea = math.radians(end_deg)
    mid_deg = (start_deg + end_deg) / 2
    ma = math.radians(mid_deg)

    start = (cx + r * math.cos(sa), cy + r * math.sin(sa))
    mid = (cx + r * math.cos(ma), cy + r * math.sin(ma))
    end = (cx + r * math.cos(ea), cy + r * math.sin(ea))

    return OutlineSegment(
        type="arc",
        start=start,
        end=end,
        center=(cx, cy),
        radius=r,
        mid=mid,
    )


def _make_fold_marker(center, axis, zone_width, angle_degrees=90.0):
    """Create a minimal FoldMarker for testing."""
    angle_rad = math.radians(abs(angle_degrees))
    radius = zone_width / angle_rad if angle_rad > 1e-6 else float('inf')

    # Build dummy line endpoints (not critical for these tests)
    perp = (-axis[1], axis[0])
    hw = zone_width / 2
    line_a_start = (center[0] + perp[0] * hw - axis[0] * 10,
                    center[1] + perp[1] * hw - axis[1] * 10)
    line_a_end = (center[0] + perp[0] * hw + axis[0] * 10,
                  center[1] + perp[1] * hw + axis[1] * 10)
    line_b_start = (center[0] - perp[0] * hw - axis[0] * 10,
                    center[1] - perp[1] * hw - axis[1] * 10)
    line_b_end = (center[0] - perp[0] * hw + axis[0] * 10,
                  center[1] - perp[1] * hw + axis[1] * 10)

    return FoldMarker(
        line_a_start=line_a_start,
        line_a_end=line_a_end,
        line_b_start=line_b_start,
        line_b_end=line_b_end,
        angle_degrees=angle_degrees,
        zone_width=zone_width,
        radius=radius,
        axis=axis,
        center=center,
    )


def _make_rounded_rect_outline(x0, y0, w, h, fillet_r):
    """Build a Polygon with 4 arc corners and 4 line edges (rounded rectangle).

    Returns Polygon whose .segments list contains 8 OutlineSegments
    (alternating line / arc).
    """
    segments = []
    vertices = []

    # Corner centers
    corners = [
        (x0 + fillet_r, y0 + fillet_r, 180, 270),        # bottom-left
        (x0 + w - fillet_r, y0 + fillet_r, 270, 360),    # bottom-right
        (x0 + w - fillet_r, y0 + h - fillet_r, 0, 90),   # top-right
        (x0 + fillet_r, y0 + h - fillet_r, 90, 180),     # top-left
    ]

    # Build edges: bottom, right, top, left — each preceded by its start-corner arc
    for i, (ccx, ccy, sa, ea) in enumerate(corners):
        # Arc segment for this corner
        arc = _make_arc_segment(ccx, ccy, fillet_r, sa, ea)
        segments.append(arc)

        # Line to next corner
        next_i = (i + 1) % 4
        ncx, ncy, nsa, _ = corners[next_i]
        line_start = arc.end
        line_end_angle = math.radians(nsa)
        line_end = (ncx + fillet_r * math.cos(line_end_angle),
                    ncy + fillet_r * math.sin(line_end_angle))
        segments.append(OutlineSegment(type="line", start=line_start, end=line_end))

    # Linearize to get vertices (coarse, like the parser does at 2 mm)
    for seg in segments:
        if seg.type == "arc":
            pts = _refine_arc_segment(seg, max_seg_length=2.0)
            vertices.extend(pts[:-1])
        else:
            vertices.append(seg.start)

    return Polygon(vertices, segments)


# ============================================================================
# Tests for arc_crosses_fold_zone
# ============================================================================

class TestArcCrossesFoldZone:
    """Test detection of arc / fold-zone intersection."""

    def test_arc_inside_fold_zone_positive(self):
        """Arc center is squarely inside the fold zone -> True."""
        arc = _make_arc_segment(50, 50, 3, 0, 90)
        # Horizontal fold at y=50, zone_width=10 => zone from y=45..55
        assert arc_crosses_fold_zone(arc, (50, 50), (1, 0), 10) is True

    def test_arc_far_from_fold_zone_negative(self):
        """Arc well outside the fold zone -> False."""
        arc = _make_arc_segment(50, 50, 3, 0, 90)
        # Fold zone far away at y=200
        assert arc_crosses_fold_zone(arc, (50, 200), (1, 0), 4) is False

    def test_arc_partially_crossing(self):
        """Arc sits on the boundary of the fold zone."""
        # Arc center at (50, 50), radius 3, from 0..90 degrees
        # Fold zone at y=53, width 4 => zone from y=51..55
        # Arc point at 90 deg = (50, 53) is on the zone boundary
        arc = _make_arc_segment(50, 50, 3, 0, 90)
        result = arc_crosses_fold_zone(arc, (50, 53), (1, 0), 4)
        assert result is True

    def test_arc_with_vertical_fold(self):
        """Arc crossing a vertical fold zone."""
        arc = _make_arc_segment(50, 50, 5, 0, 90)
        # Vertical fold at x=54, zone_width=6 => zone from x=51..57
        # Arc point at 0 deg = (55, 50) is inside
        result = arc_crosses_fold_zone(arc, (54, 50), (0, 1), 6)
        assert result is True


# ============================================================================
# Tests for _refine_arc_segment
# ============================================================================

class TestRefineArcSegment:
    """Test fine-resolution arc re-sampling."""

    def test_produces_more_points_than_coarse(self):
        """Fine re-linearization should produce many more points than 2 mm coarse."""
        arc = _make_arc_segment(0, 0, 10, 0, 90)
        coarse = _refine_arc_segment(arc, max_seg_length=2.0)
        fine = _refine_arc_segment(arc, max_seg_length=0.25)
        assert len(fine) > len(coarse)
        # 90 deg arc of radius 10 => arc_length ~ 15.7 mm
        # At 0.25 mm => ~63 segments => 64 points
        assert len(fine) >= 60

    def test_endpoints_match(self):
        """First and last points should be close to the original start/end."""
        arc = _make_arc_segment(10, 20, 5, 45, 180)
        pts = _refine_arc_segment(arc, max_seg_length=0.5)
        assert math.dist(pts[0], arc.start) < 0.01
        assert math.dist(pts[-1], arc.end) < 0.01

    def test_points_on_circle(self):
        """All generated points should lie on the arc's circle."""
        arc = _make_arc_segment(0, 0, 8, 30, 270)
        pts = _refine_arc_segment(arc, max_seg_length=0.5)
        for pt in pts:
            dist = math.dist(pt, arc.center)
            assert abs(dist - 8.0) < 0.01, f"Point {pt} off circle: dist={dist}"

    def test_degenerate_no_center(self):
        """Arc without center falls back to [start, end]."""
        seg = OutlineSegment(type="arc", start=(0, 0), end=(10, 0), center=None, radius=0)
        pts = _refine_arc_segment(seg, max_seg_length=0.25)
        assert len(pts) == 2

    def test_minimum_four_segments(self):
        """Even a tiny arc should have at least 4 segments (5 points)."""
        arc = _make_arc_segment(0, 0, 0.5, 0, 10)  # Very small arc
        pts = _refine_arc_segment(arc, max_seg_length=0.25)
        assert len(pts) >= 5


# ============================================================================
# Tests for refine_outline_for_folds
# ============================================================================

class TestRefineOutlineForFolds:
    """Test the top-level outline refinement function."""

    def test_no_markers_still_tessellates_arcs(self):
        """With no fold markers, arcs are still re-tessellated at base sagitta
        so the rendered polygon shows smooth curvature instead of chord
        approximations. The vertex count grows when the input was coarse."""
        outline = _make_rounded_rect_outline(0, 0, 100, 50, 5)
        original_count = len(outline.vertices)
        result = refine_outline_for_folds(outline, [])
        # Helper produced coarse 2 mm-chord arcs; base-sagitta (0.05 mm) is
        # much denser, so we expect more vertices, and the segments list is
        # preserved.
        assert len(result.vertices) > original_count
        assert result.segments is outline.segments

    def test_no_segments_returns_same(self):
        """Outline with no segment info is returned as-is."""
        outline = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        marker = _make_fold_marker((5, 5), (1, 0), 4)
        result = refine_outline_for_folds(outline, [marker])
        assert result is outline

    def test_no_arcs_returns_same(self):
        """Outline with only line segments should be returned unchanged."""
        segments = [
            OutlineSegment(type="line", start=(0, 0), end=(10, 0)),
            OutlineSegment(type="line", start=(10, 0), end=(10, 10)),
            OutlineSegment(type="line", start=(10, 10), end=(0, 10)),
            OutlineSegment(type="line", start=(0, 10), end=(0, 0)),
        ]
        outline = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], segments)
        marker = _make_fold_marker((5, 5), (1, 0), 4)
        result = refine_outline_for_folds(outline, [marker])
        # No arcs to refine, so polygon is returned unchanged
        assert result is outline

    def test_arc_crossing_fold_produces_more_vertices(self):
        """Rounded rect with fold crossing a fillet -> more vertices."""
        # Rounded rect 100 x 50 with 5 mm corner fillets
        outline = _make_rounded_rect_outline(0, 0, 100, 50, 5)
        original_count = len(outline.vertices)

        # Fold at x=5 (crosses the bottom-left fillet at x=0..5, y=0..5)
        marker = _make_fold_marker(center=(5, 25), axis=(0, 1), zone_width=6)
        refined = refine_outline_for_folds(outline, [marker])

        assert len(refined.vertices) > original_count

    def test_arc_far_from_fold_uses_base_sagitta(self):
        """Non-crossing arcs are still tessellated, just at the looser base
        sagitta (denser than the helper's coarse 2 mm chord, less dense than
        the fold-crossing case)."""
        outline = _make_rounded_rect_outline(0, 0, 100, 50, 5)
        original_count = len(outline.vertices)

        # Fold at x=50 — well away from all corner fillets
        marker = _make_fold_marker(center=(50, 25), axis=(0, 1), zone_width=4)
        refined_far = refine_outline_for_folds(outline, [marker])

        # And a fold that does cross fillets, for comparison
        marker_near = _make_fold_marker(center=(5, 25), axis=(0, 1), zone_width=6)
        refined_near = refine_outline_for_folds(outline, [marker_near])

        # Non-crossing case: more vertices than the coarse input, fewer than
        # when a fold actually crosses.
        assert len(refined_far.vertices) > original_count
        assert len(refined_far.vertices) < len(refined_near.vertices)

    def test_segments_preserved(self):
        """The segment list is preserved on the refined polygon."""
        outline = _make_rounded_rect_outline(0, 0, 100, 50, 5)
        marker = _make_fold_marker(center=(5, 25), axis=(0, 1), zone_width=6)
        refined = refine_outline_for_folds(outline, [marker])

        assert refined.segments is outline.segments


# ============================================================================
# Integration-level test
# ============================================================================

class TestFilletBendingIntegration:
    """Verify that the refinement integrates with the mesh pipeline."""

    def test_precompute_board_mesh_with_rounded_outline(self):
        """precompute_board_mesh accepts a refined (more-vertices) outline."""
        from board_mesh import precompute_board_mesh

        outline = _make_rounded_rect_outline(0, 0, 60, 30, 3)
        marker = _make_fold_marker(center=(3, 15), axis=(0, 1), zone_width=4)

        # Should not raise — refinement happens inside precompute_board_mesh
        precomputed = precompute_board_mesh(
            outline,
            thickness=1.6,
            markers=[marker],
            subdivide_length=1.0,
            num_bend_subdivisions=2,
        )

        # Basic sanity: at least one region was produced
        assert len(precomputed.region_data) >= 1

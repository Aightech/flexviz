"""
Region Splitter - Divide board into regions along fold lines.

This module splits a board outline into separate regions based on fold markers.
Each region can then be triangulated independently, ensuring no triangles
cross fold boundaries.
"""

from dataclasses import dataclass
from typing import Optional
import math

try:
    from .geometry import Polygon, BoundingBox
    from .markers import FoldMarker
except ImportError:
    from geometry import Polygon, BoundingBox
    from markers import FoldMarker


@dataclass
class Region:
    """A region of the board between fold lines."""
    outline: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]]
    # Index of the region (0 = first, 1 = after first fold, etc.)
    index: int
    # The fold that starts this region (None for first region)
    fold_before: Optional[FoldMarker] = None
    # The fold that ends this region (None for last region)
    fold_after: Optional[FoldMarker] = None


def _cross_2d(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _line_intersection(p1, p2, p3, p4):
    """
    Find intersection point of line segments p1-p2 and p3-p4.

    Returns (x, y, t) where t is the parameter along p1-p2 (0 to 1 if on segment),
    or None if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y, t)


def _point_side_of_line(point, line_point, line_dir):
    """
    Determine which side of a line a point is on.

    Returns:
        > 0: point is on the left side
        < 0: point is on the right side
        = 0: point is on the line
    """
    # Use cross product to determine side
    px, py = point
    lx, ly = line_point
    dx, dy = line_dir

    return (px - lx) * dy - (py - ly) * dx


def _get_fold_axis_line(marker: FoldMarker):
    """
    Get the fold axis as a point and direction.

    The fold axis passes through the marker center, parallel to the marker lines.

    Returns:
        (point, direction): A point on the axis and a unit direction vector
    """
    # The axis direction is stored in the marker
    return (marker.center, marker.axis)


def _extend_line_to_bbox(point, direction, bbox: BoundingBox, margin=10.0):
    """
    Extend a line (point + t*direction) to cover the bounding box.

    Returns two points defining a line segment that spans the bbox.
    """
    px, py = point
    dx, dy = direction

    # Extend bbox by margin
    min_x = bbox.min_x - margin
    max_x = bbox.max_x + margin
    min_y = bbox.min_y - margin
    max_y = bbox.max_y + margin

    # Find t values where line intersects bbox edges
    t_values = []

    if abs(dx) > 1e-10:
        # Intersections with vertical edges
        t1 = (min_x - px) / dx
        t2 = (max_x - px) / dx
        t_values.extend([t1, t2])

    if abs(dy) > 1e-10:
        # Intersections with horizontal edges
        t3 = (min_y - py) / dy
        t4 = (max_y - py) / dy
        t_values.extend([t3, t4])

    if not t_values:
        # Line is a point, shouldn't happen
        return (point, point)

    t_min = min(t_values)
    t_max = max(t_values)

    p1 = (px + t_min * dx, py + t_min * dy)
    p2 = (px + t_max * dx, py + t_max * dy)

    return (p1, p2)


def split_polygon_by_line(polygon: list[tuple[float, float]],
                          line_point: tuple[float, float],
                          line_dir: tuple[float, float]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Split a polygon into two parts along an infinite line.

    Args:
        polygon: List of (x, y) vertices
        line_point: A point on the splitting line
        line_dir: Direction vector of the splitting line

    Returns:
        Tuple of (left_polygon, right_polygon) where left/right are relative
        to the line direction.
    """
    if len(polygon) < 3:
        return (polygon, [])

    n = len(polygon)
    left_points = []
    right_points = []

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        side1 = _point_side_of_line(p1, line_point, line_dir)
        side2 = _point_side_of_line(p2, line_point, line_dir)

        # Add current point to appropriate polygon(s)
        if side1 >= 0:
            left_points.append(p1)
        if side1 <= 0:
            right_points.append(p1)

        # Check if edge crosses the line
        if (side1 > 0 and side2 < 0) or (side1 < 0 and side2 > 0):
            # Find intersection point
            # Line through p1, p2
            # Splitting line: line_point + t * line_dir
            # We need to find where the edge intersects the splitting line

            # Edge direction
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]

            # Solve: p1 + s * e = line_point + t * line_dir
            # This is a 2D line intersection
            denom = ex * line_dir[1] - ey * line_dir[0]

            if abs(denom) > 1e-10:
                s = ((line_point[0] - p1[0]) * line_dir[1] - (line_point[1] - p1[1]) * line_dir[0]) / denom

                # Intersection point
                ix = p1[0] + s * ex
                iy = p1[1] + s * ey
                intersection = (ix, iy)

                # Add intersection to both polygons
                left_points.append(intersection)
                right_points.append(intersection)

    return (left_points, right_points)


def _polygon_centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    """Calculate the centroid of a polygon."""
    if not polygon:
        return (0, 0)
    x = sum(p[0] for p in polygon) / len(polygon)
    y = sum(p[1] for p in polygon) / len(polygon)
    return (x, y)


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _clean_polygon(polygon: list[tuple[float, float]], tolerance=1e-6) -> list[tuple[float, float]]:
    """Remove duplicate consecutive vertices and degenerate edges."""
    if len(polygon) < 3:
        return polygon

    cleaned = []
    for p in polygon:
        if not cleaned:
            cleaned.append(p)
        else:
            # Check if different from last point
            last = cleaned[-1]
            if abs(p[0] - last[0]) > tolerance or abs(p[1] - last[1]) > tolerance:
                cleaned.append(p)

    # Check first and last
    if len(cleaned) > 1:
        first = cleaned[0]
        last = cleaned[-1]
        if abs(first[0] - last[0]) < tolerance and abs(first[1] - last[1]) < tolerance:
            cleaned.pop()

    return cleaned if len(cleaned) >= 3 else []


def split_board_into_regions(
    outline: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]],
    markers: list[FoldMarker]
) -> list[Region]:
    """
    Split a board outline into regions based on fold markers.

    Args:
        outline: Board outline polygon vertices
        holes: List of hole polygons
        markers: List of fold markers (will be sorted by position)

    Returns:
        List of Region objects, one for each region between folds
    """
    if not markers:
        # No folds - single region
        return [Region(
            outline=list(outline),
            holes=[list(h) for h in holes],
            index=0,
            fold_before=None,
            fold_after=None
        )]

    # Calculate bounding box for line extension
    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]
    bbox = BoundingBox(min(xs), min(ys), max(xs), max(ys))

    # Sort markers by position along their perpendicular axis
    # (perpendicular to the fold direction)
    def marker_sort_key(m):
        # Project center onto perpendicular axis
        # Perpendicular to axis is (-axis[1], axis[0])
        perp = (-m.axis[1], m.axis[0])
        return m.center[0] * perp[0] + m.center[1] * perp[1]

    sorted_markers = sorted(markers, key=marker_sort_key)

    # Start with the full outline as the current polygon to split
    current_polygons = [list(outline)]

    # Split along each fold axis
    all_regions = []

    for i, marker in enumerate(sorted_markers):
        line_point, line_dir = _get_fold_axis_line(marker)

        new_polygons = []

        for poly in current_polygons:
            left, right = split_polygon_by_line(poly, line_point, line_dir)

            left = _clean_polygon(left)
            right = _clean_polygon(right)

            if left:
                new_polygons.append(left)
            if right:
                new_polygons.append(right)

        current_polygons = new_polygons

    # Now assign each resulting polygon as a region
    # We need to determine the order based on position
    def polygon_sort_key(poly):
        centroid = _polygon_centroid(poly)
        # Sort by position perpendicular to first fold
        if sorted_markers:
            m = sorted_markers[0]
            perp = (-m.axis[1], m.axis[0])
            return centroid[0] * perp[0] + centroid[1] * perp[1]
        return centroid[0]

    current_polygons.sort(key=polygon_sort_key)

    # Create regions and assign holes
    regions = []
    for idx, poly in enumerate(current_polygons):
        # Find holes that belong to this region
        region_holes = []
        for hole in holes:
            hole_centroid = _polygon_centroid(hole)
            if _point_in_polygon(hole_centroid, poly):
                region_holes.append(list(hole))

        # Determine fold_before and fold_after
        fold_before = sorted_markers[idx - 1] if idx > 0 else None
        fold_after = sorted_markers[idx] if idx < len(sorted_markers) else None

        regions.append(Region(
            outline=poly,
            holes=region_holes,
            index=idx,
            fold_before=fold_before,
            fold_after=fold_after
        ))

    return regions


def get_region_for_point(
    point: tuple[float, float],
    regions: list[Region]
) -> Optional[Region]:
    """Find which region contains a given point."""
    for region in regions:
        if _point_in_polygon(point, region.outline):
            # Also check it's not in a hole
            in_hole = False
            for hole in region.holes:
                if _point_in_polygon(point, hole):
                    in_hole = True
                    break
            if not in_hole:
                return region
    return None

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


def _find_edge_intersection(p1, p2, line_point, line_dir):
    """Find intersection point of edge p1-p2 with splitting line."""
    ex, ey = p2[0] - p1[0], p2[1] - p1[1]
    denom = ex * line_dir[1] - ey * line_dir[0]

    if abs(denom) < 1e-10:
        return None

    s = ((line_point[0] - p1[0]) * line_dir[1] - (line_point[1] - p1[1]) * line_dir[0]) / denom

    # Check if intersection is on the edge (with small tolerance)
    if s < -1e-10 or s > 1.0 + 1e-10:
        return None

    ix = p1[0] + s * ex
    iy = p1[1] + s * ey
    return (ix, iy)


def _project_onto_line(point, line_point, line_dir):
    """Project a point onto a line and return the parameter t."""
    dx = point[0] - line_point[0]
    dy = point[1] - line_point[1]
    return dx * line_dir[0] + dy * line_dir[1]


def split_polygon_by_line(polygon: list[tuple[float, float]],
                          line_point: tuple[float, float],
                          line_dir: tuple[float, float]) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """
    Split a polygon into multiple parts along an infinite line.

    For complex shapes (U-shaped, etc.) that cross the line multiple times,
    this returns multiple separate polygons for each side.

    Args:
        polygon: List of (x, y) vertices
        line_point: A point on the splitting line
        line_dir: Direction vector of the splitting line

    Returns:
        Tuple of (left_polygons, right_polygons) where each is a list of polygons.
        Left/right are relative to the line direction.
    """
    if len(polygon) < 3:
        return ([polygon], [])

    n = len(polygon)
    tolerance = 1e-6

    # Step 1: Find all intersection points and build augmented vertex list
    # Each entry is (point, edge_index, is_intersection, t_along_line)
    augmented = []
    intersections_on_line = []  # (t_value, point) for sorting

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Add the vertex
        side1 = _point_side_of_line(p1, line_point, line_dir)
        augmented.append((p1, i, False, side1))

        # Check for intersection
        side2 = _point_side_of_line(p2, line_point, line_dir)

        if (side1 > tolerance and side2 < -tolerance) or (side1 < -tolerance and side2 > tolerance):
            intersection = _find_edge_intersection(p1, p2, line_point, line_dir)
            if intersection:
                t = _project_onto_line(intersection, line_point, line_dir)
                augmented.append((intersection, i, True, 0.0))  # side = 0 (on line)
                intersections_on_line.append((t, intersection))

    if len(intersections_on_line) < 2:
        # No proper split - return original polygon on the appropriate side
        centroid = _polygon_centroid(polygon)
        side = _point_side_of_line(centroid, line_point, line_dir)
        if side >= 0:
            return ([polygon], [])
        else:
            return ([], [polygon])

    # Sort intersection points along the splitting line
    intersections_on_line.sort(key=lambda x: x[0])

    def points_equal(p1, p2):
        return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance

    # Step 2: Build index-based partner mapping for intersections
    # Pair consecutive intersections along the line (0-1, 2-3, 4-5, etc.)
    # Map augmented list index -> partner augmented list index
    intersection_indices = []  # List of augmented indices that are intersections
    for aug_idx, (pt, _, is_intersection, _) in enumerate(augmented):
        if is_intersection:
            intersection_indices.append(aug_idx)

    # Sort intersection indices by their t-value along the line
    intersection_indices.sort(key=lambda idx: _project_onto_line(augmented[idx][0], line_point, line_dir))

    partner_map = {}  # augmented_idx -> partner_augmented_idx
    for i in range(0, len(intersection_indices) - 1, 2):
        idx1 = intersection_indices[i]
        idx2 = intersection_indices[i + 1]
        partner_map[idx1] = idx2
        partner_map[idx2] = idx1

    # Step 3: Walk the augmented polygon and extract closed regions
    # We need to trace both directions from intersection pairs to get both sides
    left_polygons = []
    right_polygons = []

    # Track which (intersection_idx, direction) pairs have been used
    used_walks = set()  # (start_aug_idx, building_left)

    def trace_polygon(start_aug_idx, building_left):
        """Trace a closed polygon starting from an intersection."""
        start_pt = augmented[start_aug_idx][0]
        current_polygon = [start_pt]
        current_aug_idx = (start_aug_idx + 1) % len(augmented)

        max_steps = len(augmented) * 2
        steps = 0

        while steps < max_steps:
            steps += 1
            pt, edge_idx, is_intersection, side = augmented[current_aug_idx]

            if is_intersection:
                if current_aug_idx == start_aug_idx:
                    break

                current_polygon.append(pt)

                partner_aug_idx = partner_map.get(current_aug_idx)
                if partner_aug_idx is not None:
                    partner_pt = augmented[partner_aug_idx][0]
                    current_polygon.append(partner_pt)

                    if partner_aug_idx == start_aug_idx:
                        break

                    current_aug_idx = (partner_aug_idx + 1) % len(augmented)
                else:
                    current_aug_idx = (current_aug_idx + 1) % len(augmented)
            else:
                # Regular vertex - add it if on correct side
                if (building_left and side >= -tolerance) or (not building_left and side <= tolerance):
                    current_polygon.append(pt)
                current_aug_idx = (current_aug_idx + 1) % len(augmented)

        return current_polygon

    # For each intersection pair, trace polygons for both sides
    for i in range(0, len(intersection_indices) - 1, 2):
        idx1 = intersection_indices[i]
        idx2 = intersection_indices[i + 1]

        # Determine which side each direction goes to
        # From idx1 going forward: check next non-intersection vertex
        for j in range(1, len(augmented)):
            test_idx = (idx1 + j) % len(augmented)
            if not augmented[test_idx][2]:  # not an intersection
                side_from_idx1 = augmented[test_idx][3]
                break
        else:
            side_from_idx1 = 0

        # From idx2 going forward
        for j in range(1, len(augmented)):
            test_idx = (idx2 + j) % len(augmented)
            if not augmented[test_idx][2]:
                side_from_idx2 = augmented[test_idx][3]
                break
        else:
            side_from_idx2 = 0

        # Trace from idx1 (goes to side_from_idx1)
        building_left = side_from_idx1 > 0
        if (idx1, building_left) not in used_walks:
            poly = trace_polygon(idx1, building_left)
            if len(poly) >= 3:
                if building_left:
                    left_polygons.append(poly)
                else:
                    right_polygons.append(poly)
            used_walks.add((idx1, building_left))

        # Trace from idx2 (goes to side_from_idx2)
        building_left = side_from_idx2 > 0
        if (idx2, building_left) not in used_walks:
            poly = trace_polygon(idx2, building_left)
            if len(poly) >= 3:
                if building_left:
                    left_polygons.append(poly)
                else:
                    right_polygons.append(poly)
            used_walks.add((idx2, building_left))

    # Deduplicate polygons (same vertices in same or different order)
    def remove_consecutive_duplicates(poly):
        """Remove consecutive duplicate vertices."""
        if not poly:
            return poly
        result = [poly[0]]
        for pt in poly[1:]:
            last = result[-1]
            if abs(pt[0] - last[0]) > tolerance or abs(pt[1] - last[1]) > tolerance:
                result.append(pt)
        # Check first and last
        if len(result) > 1:
            if abs(result[0][0] - result[-1][0]) < tolerance and abs(result[0][1] - result[-1][1]) < tolerance:
                result.pop()
        return result

    def polygon_signature(poly):
        """Create a hashable signature for a polygon."""
        if not poly:
            return None
        # First clean the polygon
        cleaned = remove_consecutive_duplicates(poly)
        if len(cleaned) < 3:
            return None
        # Normalize: find min point, rotate to start there
        min_idx = 0
        min_pt = cleaned[0]
        for i, pt in enumerate(cleaned):
            if (pt[0], pt[1]) < (min_pt[0], min_pt[1]):
                min_pt = pt
                min_idx = i
        rotated = cleaned[min_idx:] + cleaned[:min_idx]
        # Round to avoid floating point issues
        return tuple((round(p[0], 6), round(p[1], 6)) for p in rotated)

    def deduplicate(polygons):
        seen = set()
        result = []
        for poly in polygons:
            sig = polygon_signature(poly)
            if sig and sig not in seen:
                seen.add(sig)
                result.append(poly)
        return result

    return (deduplicate(left_polygons), deduplicate(right_polygons))


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
            left_polys, right_polys = split_polygon_by_line(poly, line_point, line_dir)

            # Add all left-side polygons
            for left in left_polys:
                cleaned = _clean_polygon(left)
                if cleaned:
                    new_polygons.append(cleaned)

            # Add all right-side polygons
            for right in right_polys:
                cleaned = _clean_polygon(right)
                if cleaned:
                    new_polygons.append(cleaned)

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

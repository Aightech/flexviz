"""
Planar Subdivision Module

Partitions a polygon (with holes) into regions using cutting lines.
Used for creating triangulatable regions along fold axes in flex PCBs.

Algorithm:
1. Treat all edges (outer boundary, holes, cutting lines) as a planar graph
2. Build vertex adjacency structure with edges sorted by angle
3. Trace region boundaries using "next clockwise edge" rule
4. Filter to valid board material regions (CCW winding, inside outer, outside holes)

See docs/bend_zone_triangulation.md for detailed algorithm description.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional


# Type aliases
Point = tuple[float, float]
Polygon = list[Point]
LineEquation = tuple[float, float, float]  # (a, b, c) for ax + by + c = 0
CuttingLine = tuple[LineEquation, Point, Point]  # (line_eq, p1, p2)


# =============================================================================
# Basic Geometry Functions
# =============================================================================

def signed_area(polygon: Polygon) -> float:
    """
    Calculate signed area of polygon using shoelace formula.

    Returns:
        Positive for CCW winding, negative for CW winding.
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0


def ensure_ccw(polygon: Polygon) -> Polygon:
    """Ensure polygon has counter-clockwise winding order."""
    if signed_area(polygon) < 0:
        return list(reversed(polygon))
    return list(polygon)


def ensure_cw(polygon: Polygon) -> Polygon:
    """Ensure polygon has clockwise winding order."""
    if signed_area(polygon) > 0:
        return list(reversed(polygon))
    return list(polygon)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.

    Args:
        point: (x, y) coordinates
        polygon: List of (x, y) vertices

    Returns:
        True if point is inside polygon.
    """
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


def points_equal(p1: Point, p2: Point, eps: float = 1e-9) -> bool:
    """Check if two points are equal within epsilon tolerance."""
    return abs(p1[0] - p2[0]) < eps and abs(p1[1] - p2[1]) < eps


def signed_distance_to_line(point: Point, line: LineEquation) -> float:
    """
    Compute signed distance from point to line.

    Args:
        point: (x, y) coordinates
        line: (a, b, c) where ax + by + c = 0

    Returns:
        Signed distance (positive on one side, negative on the other).
    """
    a, b, c = line
    return a * point[0] + b * point[1] + c


def segment_line_intersection(
    seg_start: Point,
    seg_end: Point,
    line: LineEquation
) -> Optional[tuple[float, Point]]:
    """
    Find intersection of line segment with infinite line.

    Args:
        seg_start: Start point of segment
        seg_end: End point of segment
        line: (a, b, c) line equation

    Returns:
        (t, point) where t is parameter [0,1] on segment, or None if no intersection.
    """
    d1 = signed_distance_to_line(seg_start, line)
    d2 = signed_distance_to_line(seg_end, line)

    # Both on same side - no intersection
    if d1 * d2 > 1e-10:
        return None

    # Both on the line (collinear)
    if abs(d1) < 1e-10 and abs(d2) < 1e-10:
        return None

    # One endpoint on line
    if abs(d1) < 1e-10:
        return (0.0, seg_start)
    if abs(d2) < 1e-10:
        return (1.0, seg_end)

    # Proper intersection
    t = d1 / (d1 - d2)
    px = seg_start[0] + t * (seg_end[0] - seg_start[0])
    py = seg_start[1] + t * (seg_end[1] - seg_start[1])
    return (t, (px, py))


def polygon_centroid(polygon: Polygon) -> Point:
    """Calculate centroid of a polygon."""
    if len(polygon) == 0:
        return (0.0, 0.0)
    cx = sum(v[0] for v in polygon) / len(polygon)
    cy = sum(v[1] for v in polygon) / len(polygon)
    return (cx, cy)


def get_interior_test_point(region: Polygon) -> Point:
    """
    Get a point that is definitely inside the region.

    Uses the midpoint of an edge, offset slightly inward based on winding order.
    Falls back to centroid if edge-based method fails.
    """
    if len(region) < 3:
        return polygon_centroid(region)

    area = signed_area(region)

    # Try multiple edges to find a good test point
    for i in range(len(region)):
        p1 = region[i]
        p2 = region[(i + 1) % len(region)]

        # Midpoint of edge
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        # Edge direction
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            continue

        # Normal direction (perpendicular to edge)
        # For CCW polygon (positive area), inward is to the left: (-dy, dx)
        # For CW polygon (negative area), inward is to the right: (dy, -dx)
        if area > 0:
            nx, ny = -dy / length, dx / length
        else:
            nx, ny = dy / length, -dx / length

        # Offset slightly inward
        offset = min(length * 0.01, 0.1)
        test_x = mid_x + nx * offset
        test_y = mid_y + ny * offset

        # Verify this point is inside the region
        if point_in_polygon((test_x, test_y), region):
            return (test_x, test_y)

    # Fallback to centroid
    return polygon_centroid(region)


# =============================================================================
# Planar Subdivision Class
# =============================================================================

class PlanarSubdivision:
    """
    Computes regions by partitioning a polygon (with holes) using cutting lines.

    Uses boundary tracing: treats all edges (outer, holes, cutting lines) as a
    planar graph and traces faces using the "next clockwise edge" rule.

    Example:
        >>> outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
        >>> holes = [[(30, 30), (50, 30), (50, 50), (30, 50)]]
        >>> cutting_lines = [((0, 1, -40), (-10, 40), (110, 40))]  # y=40
        >>> subdivision = PlanarSubdivision(outer, holes, cutting_lines)
        >>> regions = subdivision.compute()
    """

    def __init__(
        self,
        outer: Polygon,
        holes: list[Polygon],
        cutting_lines: list[CuttingLine]
    ):
        """
        Initialize planar subdivision.

        Args:
            outer: Outer boundary polygon (will be converted to CCW)
            holes: List of hole polygons (will be converted to CW)
            cutting_lines: List of (line_eq, p1, p2) tuples where:
                - line_eq is (a, b, c) for ax + by + c = 0
                - p1, p2 define the extent of the cutting line
        """
        self.outer = ensure_ccw(outer)
        self.holes = [ensure_cw(h) for h in holes]
        self.cutting_lines = cutting_lines

        # Populated by compute()
        self.vertices: list[Point] = []
        self.vertex_to_idx: dict[tuple[float, float], int] = {}
        self.edges: list[tuple[int, int]] = []
        self.vertex_edges: dict[int, list[tuple[float, int, str]]] = {}
        self.regions: list[Polygon] = []

    def _add_vertex(self, point: Point) -> int:
        """Add vertex and return its index."""
        # Round to avoid floating point issues
        key = (round(point[0], 9), round(point[1], 9))
        if key in self.vertex_to_idx:
            return self.vertex_to_idx[key]
        idx = len(self.vertices)
        self.vertices.append(point)
        self.vertex_to_idx[key] = idx
        return idx

    def _add_edge(self, start_idx: int, end_idx: int) -> None:
        """Add edge between two vertices."""
        if start_idx == end_idx:
            return
        # Avoid duplicate edges
        edge = (min(start_idx, end_idx), max(start_idx, end_idx))
        existing = [(min(e[0], e[1]), max(e[0], e[1])) for e in self.edges]
        if edge not in existing:
            self.edges.append((start_idx, end_idx))

    def _collect_polygon_edges(
        self,
        polygon: Polygon,
        cutting_lines: list[CuttingLine]
    ) -> list[tuple[Point, Point]]:
        """
        Collect edges from a polygon, splitting at cutting line intersections.
        """
        edges = []
        n = len(polygon)

        for i in range(n):
            seg_start = polygon[i]
            seg_end = polygon[(i + 1) % n]

            # Find all intersections with cutting lines
            intersections = []
            for line_eq, _, _ in cutting_lines:
                result = segment_line_intersection(seg_start, seg_end, line_eq)
                if result is not None:
                    t, point = result
                    if 0 < t < 1:  # Proper intersection (not at endpoints)
                        intersections.append((t, point))

            # Sort by parameter t
            intersections.sort(key=lambda x: x[0])

            # Create sub-edges
            current = seg_start
            for _, int_point in intersections:
                if not points_equal(current, int_point):
                    edges.append((current, int_point))
                current = int_point
            if not points_equal(current, seg_end):
                edges.append((current, seg_end))

        return edges

    def _collect_cutting_line_segments(
        self,
        line_eq: LineEquation,
        line_p1: Point,
        line_p2: Point,
        all_polygons: list[Polygon]
    ) -> list[tuple[Point, Point]]:
        """
        Collect segments of cutting line that are inside the board.

        Only considers intersections within the finite extent defined by
        line_p1 and line_p2. This ensures cutting lines don't extend beyond
        where the actual fold marker lines exist.
        """
        # Find all intersections of cutting line with all polygon edges
        intersections = []

        for polygon in all_polygons:
            n = len(polygon)
            for i in range(n):
                seg_start = polygon[i]
                seg_end = polygon[(i + 1) % n]
                result = segment_line_intersection(seg_start, seg_end, line_eq)
                if result is not None:
                    _, point = result
                    intersections.append(point)

        if len(intersections) < 2:
            return []

        # Sort intersections along the line direction
        line_dir = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
        line_len_sq = line_dir[0]**2 + line_dir[1]**2

        if line_len_sq < 1e-10:
            return []

        def project(p: Point) -> float:
            return (p[0] - line_p1[0]) * line_dir[0] + (p[1] - line_p1[1]) * line_dir[1]

        intersections.sort(key=project)

        # Filter intersections to only include those within the finite extent [line_p1, line_p2]
        # Project normalized to [0, 1] range where 0 is at line_p1 and 1 is at line_p2
        filtered = []
        tolerance = 0.01  # Small tolerance to avoid floating point issues
        for p in intersections:
            proj_normalized = project(p) / line_len_sq
            if -tolerance <= proj_normalized <= 1.0 + tolerance:
                filtered.append(p)

        if len(filtered) < 2:
            return []

        intersections = filtered

        # Remove duplicates
        unique = [intersections[0]]
        for p in intersections[1:]:
            if not points_equal(p, unique[-1]):
                unique.append(p)

        # Create segments between consecutive intersections that are inside the board
        segments = []
        for i in range(len(unique) - 1):
            p1, p2 = unique[i], unique[i + 1]
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            # Check if midpoint is inside outer and outside all holes
            if point_in_polygon(mid, self.outer):
                inside_hole = False
                for hole in self.holes:
                    if point_in_polygon(mid, hole):
                        inside_hole = True
                        break
                if not inside_hole:
                    segments.append((p1, p2))

        return segments

    def compute(self, debug: bool = False) -> list[Polygon]:
        """
        Compute the planar subdivision and extract regions.

        Args:
            debug: If True, print debug information.

        Returns:
            List of region polygons (including invalid ones - use filter_valid_board_regions).
        """
        # Step 1: Collect all edges
        all_polygons = [self.outer] + self.holes

        # Edges from outer boundary
        outer_edges = self._collect_polygon_edges(self.outer, self.cutting_lines)

        # Edges from holes
        hole_edges = []
        for hole in self.holes:
            hole_edges.extend(self._collect_polygon_edges(hole, self.cutting_lines))

        # Edges from cutting lines
        cutting_edges = []
        for line_eq, p1, p2 in self.cutting_lines:
            segments = self._collect_cutting_line_segments(line_eq, p1, p2, all_polygons)
            cutting_edges.extend(segments)

        if debug:
            print(f"  Outer edges: {len(outer_edges)}")
            print(f"  Hole edges: {len(hole_edges)}")
            print(f"  Cutting edges: {len(cutting_edges)}")

        # Step 2: Build vertex and edge lists
        all_edges = outer_edges + hole_edges + cutting_edges

        for start, end in all_edges:
            start_idx = self._add_vertex(start)
            end_idx = self._add_vertex(end)
            self._add_edge(start_idx, end_idx)

        if debug:
            print(f"  Total vertices: {len(self.vertices)}")
            print(f"  Total edges: {len(self.edges)}")

        # Step 3: Build adjacency - for each vertex, list edges in angular order
        self.vertex_edges = {i: [] for i in range(len(self.vertices))}

        for edge_idx, (start_idx, end_idx) in enumerate(self.edges):
            start = self.vertices[start_idx]
            end = self.vertices[end_idx]

            # Angle from start to end
            angle_forward = math.atan2(end[1] - start[1], end[0] - start[0])
            # Angle from end to start
            angle_backward = math.atan2(start[1] - end[1], start[0] - end[0])

            self.vertex_edges[start_idx].append((angle_forward, edge_idx, 'forward'))
            self.vertex_edges[end_idx].append((angle_backward, edge_idx, 'backward'))

        # Sort edges at each vertex by angle
        for v_idx in self.vertex_edges:
            self.vertex_edges[v_idx].sort(key=lambda x: x[0])

        # Step 4: Trace regions
        self._trace_regions()

        return self.regions

    def _trace_regions(self) -> None:
        """Trace all region boundaries using the 'next CW edge' rule."""
        used: set[tuple[int, str]] = set()

        for edge_idx in range(len(self.edges)):
            for direction in ['forward', 'backward']:
                if (edge_idx, direction) in used:
                    continue

                region = self._trace_one_region(edge_idx, direction, used)
                if region and len(region) >= 3:
                    self.regions.append(region)

    def _trace_one_region(
        self,
        start_edge_idx: int,
        start_direction: str,
        used: set[tuple[int, str]]
    ) -> Polygon:
        """
        Trace one region boundary starting from given edge and direction.

        Uses standard planar subdivision face tracing: at each vertex,
        take the next edge in CLOCKWISE order (the rightmost turn).
        """
        boundary: list[Point] = []

        current_edge = start_edge_idx
        current_dir = start_direction

        max_steps = len(self.edges) * 2 + 10
        steps = 0

        while steps < max_steps:
            steps += 1

            if (current_edge, current_dir) in used:
                if (current_edge == start_edge_idx and
                    current_dir == start_direction and
                    len(boundary) > 0):
                    break
                else:
                    break

            used.add((current_edge, current_dir))

            # Get start and end vertices based on direction
            edge_start, edge_end = self.edges[current_edge]
            if current_dir == 'forward':
                from_v, to_v = edge_start, edge_end
            else:
                from_v, to_v = edge_end, edge_start

            boundary.append(self.vertices[from_v])

            # Find next edge: at to_v, find the next edge in CLOCKWISE order
            incoming_angle = math.atan2(
                self.vertices[to_v][1] - self.vertices[from_v][1],
                self.vertices[to_v][0] - self.vertices[from_v][0]
            )

            reversed_incoming = incoming_angle + math.pi

            edges_at_v = self.vertex_edges[to_v]
            if len(edges_at_v) == 0:
                break

            best_edge_idx = None
            best_dir = None
            best_angle_diff = float('inf')

            for angle, e_idx, e_dir in edges_at_v:
                # Skip the edge we just came from (reverse direction)
                if e_idx == current_edge:
                    if current_dir == 'forward' and e_dir == 'backward':
                        continue
                    if current_dir == 'backward' and e_dir == 'forward':
                        continue

                # Calculate CW distance from reversed_incoming to this edge's angle
                diff = reversed_incoming - angle
                while diff < 0:
                    diff += 2 * math.pi
                while diff >= 2 * math.pi:
                    diff -= 2 * math.pi

                if diff > 1e-9 and diff < best_angle_diff:
                    best_angle_diff = diff
                    best_edge_idx = e_idx
                    best_dir = e_dir

            if best_edge_idx is None:
                # Fallback: take any edge that's not going back
                for angle, e_idx, e_dir in edges_at_v:
                    if e_idx != current_edge:
                        best_edge_idx = e_idx
                        best_dir = e_dir
                        break
                if best_edge_idx is None:
                    break

            current_edge = best_edge_idx
            current_dir = best_dir

            if current_edge == start_edge_idx and current_dir == start_direction:
                break

        return boundary


# =============================================================================
# Region Filtering and Hole Association
# =============================================================================

def filter_valid_board_regions(
    regions: list[Polygon],
    outer: Polygon,
    holes: list[Polygon]
) -> list[Polygon]:
    """
    Filter regions to only include valid board material regions.

    A valid region must:
    - Have positive signed area (CCW winding = filled region)
    - Have a test point inside the outer boundary
    - Have a test point outside all holes

    Args:
        regions: All traced regions from PlanarSubdivision
        outer: Original outer boundary
        holes: Original hole polygons

    Returns:
        List of valid board material regions.
    """
    outer_ccw = ensure_ccw(outer)
    holes_cw = [ensure_cw(h) for h in holes]

    valid_regions = []
    for region in regions:
        if len(region) < 3:
            continue

        # Only keep CCW regions (positive area = filled interior)
        area = signed_area(region)
        if area <= 0:
            continue

        # Use an interior test point
        test_point = get_interior_test_point(region)

        # Must be inside outer boundary
        if not point_in_polygon(test_point, outer_ccw):
            continue

        # Must be outside all holes
        inside_hole = False
        for hole in holes_cw:
            if point_in_polygon(test_point, hole):
                inside_hole = True
                break

        if not inside_hole:
            valid_regions.append(region)

    return valid_regions


def hole_crosses_cutting_lines(
    hole: Polygon,
    cutting_lines: list[CuttingLine]
) -> bool:
    """
    Check if a hole crosses any of the cutting lines.

    A hole crosses a cutting line if any of its edges have endpoints
    on opposite sides of the line.
    """
    for line_eq, _, _ in cutting_lines:
        n = len(hole)
        for i in range(n):
            seg_start = hole[i]
            seg_end = hole[(i + 1) % n]
            d1 = signed_distance_to_line(seg_start, line_eq)
            d2 = signed_distance_to_line(seg_end, line_eq)
            if d1 * d2 < -1e-10:
                return True
    return False


def associate_holes_with_regions(
    valid_regions: list[Polygon],
    original_holes: list[Polygon],
    cutting_lines: list[CuttingLine]
) -> list[tuple[Polygon, list[Polygon]]]:
    """
    Associate holes that don't cross cutting lines with their containing region.

    Holes that cross cutting lines have their boundaries already incorporated
    into the region edges, so they don't need separate association.

    Args:
        valid_regions: Filtered valid board regions
        original_holes: Original hole polygons
        cutting_lines: Cutting lines used for subdivision

    Returns:
        List of (region, [holes]) tuples.
    """
    region_holes: list[list[Polygon]] = [[] for _ in valid_regions]

    for hole in original_holes:
        # Skip holes that cross cutting lines - they're already incorporated
        if hole_crosses_cutting_lines(hole, cutting_lines):
            continue

        # Find which region contains this hole
        hole_center = polygon_centroid(hole)

        for i, region in enumerate(valid_regions):
            if point_in_polygon(hole_center, region):
                region_holes[i].append(hole)
                break

    return list(zip(valid_regions, region_holes))


# =============================================================================
# Cutting Line Generation
# =============================================================================

def create_parallel_cutting_lines(
    y1: float,
    y2: float,
    x_extent: tuple[float, float]
) -> list[CuttingLine]:
    """
    Create two horizontal parallel cutting lines at y=y1 and y=y2.

    Args:
        y1: Y-coordinate of first cutting line
        y2: Y-coordinate of second cutting line
        x_extent: (x_min, x_max) extent of the lines

    Returns:
        List of cutting line tuples (line_eq, p1, p2).
    """
    lines = []

    # Line at y = y1: equation is 0*x + 1*y - y1 = 0
    line1: LineEquation = (0, 1, -y1)
    p1_1 = (x_extent[0] - 10, y1)
    p1_2 = (x_extent[1] + 10, y1)
    lines.append((line1, p1_1, p1_2))

    # Line at y = y2
    line2: LineEquation = (0, 1, -y2)
    p2_1 = (x_extent[0] - 10, y2)
    p2_2 = (x_extent[1] + 10, y2)
    lines.append((line2, p2_1, p2_2))

    return lines


def create_bend_zone_cutting_lines(
    center: Point,
    axis: tuple[float, float],
    zone_width: float,
    num_subdivisions: int,
    extent: float = 200.0
) -> list[CuttingLine]:
    """
    Create cutting lines for a bend zone with subdivisions.

    The bend zone is centered at `center` and extends `zone_width/2` in each
    direction perpendicular to the fold axis. The zone is divided into
    `num_subdivisions` strips.

    Args:
        center: Center point of the bend zone
        axis: Direction vector of the fold axis (will be normalized)
        zone_width: Total width of the bend zone
        num_subdivisions: Number of strips in the bend zone
        extent: How far to extend the cutting lines

    Returns:
        List of cutting line tuples. Creates num_subdivisions + 1 lines
        (boundaries at both edges plus internal divisions).
    """
    # Normalize axis
    ax, ay = axis
    length = math.sqrt(ax * ax + ay * ay)
    if length < 1e-10:
        raise ValueError("Axis vector cannot be zero")
    ax, ay = ax / length, ay / length

    # Perpendicular direction (for offsetting lines)
    perp = (-ay, ax)

    half_width = zone_width / 2
    lines = []

    for i in range(num_subdivisions + 1):
        # Parameter along perpendicular direction
        t = -half_width + (i / num_subdivisions) * zone_width

        # Point on this cutting line
        px = center[0] + t * perp[0]
        py = center[1] + t * perp[1]

        # Line equation: -ay*x + ax*y + (ay*px - ax*py) = 0
        # This is a line through (px, py) with direction (ax, ay)
        a, b, c = -ay, ax, ay * px - ax * py
        line_eq: LineEquation = (a, b, c)

        # Extend line for visualization/intersection
        p1 = (px - extent * ax, py - extent * ay)
        p2 = (px + extent * ax, py + extent * ay)

        lines.append((line_eq, p1, p2))

    return lines


@dataclass
class Region:
    """A region of the board between fold lines."""
    outline: Polygon
    holes: list[Polygon]
    index: int
    fold_before: Optional[object] = None  # FoldMarker, but avoid circular import
    fold_after: Optional[object] = None

    # Ordered list of (fold_marker, classification) pairs
    # Classification is "IN_ZONE" or "AFTER"
    # This determines exactly which folds affect this region and in what order
    fold_recipe: list[tuple[object, str]] = None  # Will be initialized to []

    # Representative point inside this region (for fold classification)
    representative_point: Optional[Point] = None

    def __post_init__(self):
        if self.fold_recipe is None:
            self.fold_recipe = []
        if self.representative_point is None:
            self.representative_point = get_interior_test_point(self.outline)


def split_board_into_regions(
    outline: Polygon,
    holes: list[Polygon],
    markers: list,  # list[FoldMarker]
    num_bend_subdivisions: int = 1
) -> list[Region]:
    """
    Split a board into regions along fold markers.

    This is a high-level wrapper around PlanarSubdivision that provides
    a simpler interface for the mesh generation code.

    Args:
        outline: Board outer boundary (list of (x, y) tuples)
        holes: List of hole polygons
        markers: List of FoldMarker objects defining fold lines
        num_bend_subdivisions: Number of subdivisions per bend zone (1 = no extra subdivision)

    Returns:
        List of Region objects, sorted by position along the fold axis.
    """
    if not markers:
        # No folds - return single region
        return [Region(outline=list(outline), holes=[list(h) for h in holes], index=0)]

    # Sort markers by position along the perpendicular to first marker's axis
    # This ensures consistent ordering of regions
    ref_marker = markers[0]
    perp = (-ref_marker.axis[1], ref_marker.axis[0])

    def marker_sort_key(m):
        return m.center[0] * perp[0] + m.center[1] * perp[1]

    sorted_markers = sorted(markers, key=marker_sort_key)

    # Create cutting lines from markers using actual marker line segments
    # This ensures cutting lines only extend where the fold markers actually exist
    # (finite segments instead of infinite lines)
    cutting_lines: list[CuttingLine] = []

    for marker in sorted_markers:
        # Use the new function that creates finite cutting lines from marker segments
        lines = create_cutting_lines_from_marker_segments(
            marker,
            num_subdivisions=max(1, num_bend_subdivisions)
        )
        cutting_lines.extend(lines)

    # Run planar subdivision
    subdivision = PlanarSubdivision(outline, holes, cutting_lines)
    all_regions = subdivision.compute()
    valid_regions = filter_valid_board_regions(all_regions, outline, holes)
    regions_with_holes = associate_holes_with_regions(valid_regions, holes, cutting_lines)

    # Sort regions by position along perpendicular direction
    def region_sort_key(rh_tuple):
        region, _ = rh_tuple
        centroid = polygon_centroid(region)
        return centroid[0] * perp[0] + centroid[1] * perp[1]

    sorted_regions_with_holes = sorted(regions_with_holes, key=region_sort_key)

    # Create Region objects
    result: list[Region] = []
    for idx, (region_outline, region_holes) in enumerate(sorted_regions_with_holes):
        # Determine fold_before and fold_after
        # This is approximate - based on position relative to markers
        fold_before = None
        fold_after = None

        region_pos = region_sort_key((region_outline, region_holes))

        for i, marker in enumerate(sorted_markers):
            marker_pos = marker_sort_key(marker)
            if region_pos < marker_pos:
                fold_after = marker
                if i > 0:
                    fold_before = sorted_markers[i - 1]
                break
        else:
            # Region is after all markers
            if sorted_markers:
                fold_before = sorted_markers[-1]

        result.append(Region(
            outline=list(region_outline),
            holes=[list(h) for h in region_holes],
            index=idx,
            fold_before=fold_before,
            fold_after=fold_after
        ))

    # Compute fold recipes using region adjacency and BFS
    # Pass board outline to enable finite fold segment checking
    if markers:
        adjacency = build_region_adjacency(result)
        compute_fold_recipes(result, adjacency, markers, board_outline=outline)

    return result


def create_cutting_lines_from_marker_segments(
    marker,
    num_subdivisions: int = 1
) -> list[CuttingLine]:
    """
    Create cutting lines from actual fold marker line segments.

    Instead of extending lines infinitely, uses the actual marker line
    endpoints to determine the finite extent of cutting lines.

    Args:
        marker: FoldMarker with line_a_start, line_a_end, line_b_start, line_b_end
        num_subdivisions: Number of strips in the bend zone

    Returns:
        List of cutting line tuples. Creates num_subdivisions + 1 lines
        (boundaries at both edges plus internal divisions).
    """
    # Get marker properties
    center = marker.center
    axis = marker.axis
    zone_width = getattr(marker, 'zone_width', 0)

    # Get the actual marker line endpoints
    line_a_start = marker.line_a_start
    line_a_end = marker.line_a_end
    line_b_start = marker.line_b_start
    line_b_end = marker.line_b_end

    # Normalize axis
    ax, ay = axis
    length = math.sqrt(ax * ax + ay * ay)
    if length < 1e-10:
        return []
    ax, ay = ax / length, ay / length

    # Calculate the extent along the axis direction by projecting all marker endpoints
    def project_along_axis(p: Point) -> float:
        return (p[0] - center[0]) * ax + (p[1] - center[1]) * ay

    all_points = [line_a_start, line_a_end, line_b_start, line_b_end]
    projections = [project_along_axis(p) for p in all_points]

    min_proj = min(projections)
    max_proj = max(projections)

    # Perpendicular direction (for offsetting lines)
    perp = (-ay, ax)

    half_width = zone_width / 2
    lines = []

    for i in range(num_subdivisions + 1):
        # Parameter along perpendicular direction
        t = -half_width + (i / num_subdivisions) * zone_width

        # Point on this cutting line (at center position along axis)
        px = center[0] + t * perp[0]
        py = center[1] + t * perp[1]

        # Line equation: -ay*x + ax*y + (ay*px - ax*py) = 0
        # This is a line through (px, py) with direction (ax, ay)
        a, b, c = -ay, ax, ay * px - ax * py
        line_eq: LineEquation = (a, b, c)

        # Use the actual marker extent (finite segment, not infinite)
        p1 = (px + min_proj * ax, py + min_proj * ay)
        p2 = (px + max_proj * ax, py + max_proj * ay)

        lines.append((line_eq, p1, p2))

    return lines


def create_line_through_point(
    point: Point,
    direction: tuple[float, float],
    extent: float = 200.0
) -> CuttingLine:
    """
    Create a cutting line through a point with given direction.

    Args:
        point: Point the line passes through
        direction: Direction vector of the line
        extent: How far to extend the line endpoints

    Returns:
        Cutting line tuple (line_eq, p1, p2).
    """
    px, py = point
    dx, dy = direction

    # Normalize direction
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-10:
        raise ValueError("Direction vector cannot be zero")
    dx, dy = dx / length, dy / length

    # Line equation: -dy*x + dx*y + (dy*px - dx*py) = 0
    a, b, c = -dy, dx, dy * px - dx * py
    line_eq: LineEquation = (a, b, c)

    # Extend line
    p1 = (px - extent * dx, py - extent * dy)
    p2 = (px + extent * dx, py + extent * dy)

    return (line_eq, p1, p2)


# =============================================================================
# Region Adjacency and Fold Recipe Computation
# =============================================================================

def compute_fold_extent_on_board(
    fold_marker,
    board_outline: Polygon
) -> Optional[tuple[Point, Point]]:
    """
    Get the finite extent of a fold based on the actual marker line endpoints,
    but only if the fold actually intersects the board.

    Uses the fold marker's actual line endpoints (line_a and line_b) to determine
    the fold's reach. Returns None if the fold doesn't intersect the board.

    Args:
        fold_marker: FoldMarker with line_a_start, line_a_end, line_b_start, line_b_end
        board_outline: Board outline polygon

    Returns:
        (p1, p2) endpoints defining the fold's extent along its axis,
        or None if fold doesn't intersect the board
    """
    # Get all four endpoints of the fold marker lines
    marker_points = [
        fold_marker.line_a_start,
        fold_marker.line_a_end,
        fold_marker.line_b_start,
        fold_marker.line_b_end
    ]

    # Check if fold center is inside the board (most reliable check)
    center = fold_marker.center
    center_inside = point_in_polygon(center, board_outline)

    # Check if ANY marker point is inside the board
    any_inside = False
    for p in marker_points:
        if point_in_polygon(p, board_outline):
            any_inside = True
            break

    # Also check if marker lines cross the board outline
    marker_lines = [
        (fold_marker.line_a_start, fold_marker.line_a_end),
        (fold_marker.line_b_start, fold_marker.line_b_end)
    ]

    crosses_board = False
    for line_start, line_end in marker_lines:
        n = len(board_outline)
        for i in range(n):
            edge_start = board_outline[i]
            edge_end = board_outline[(i + 1) % n]
            if segments_intersect(line_start, line_end, edge_start, edge_end):
                crosses_board = True
                break
        if crosses_board:
            break

    # If fold doesn't touch the board at all, return None
    if not center_inside and not any_inside and not crosses_board:
        return None

    axis = fold_marker.axis
    # center already defined above

    # Project all points onto the fold axis to find extent
    def project_along_axis(p: Point) -> float:
        return (p[0] - center[0]) * axis[0] + (p[1] - center[1]) * axis[1]

    projections = [(project_along_axis(p), p) for p in marker_points]
    projections.sort(key=lambda x: x[0])

    # Get min and max extent points
    min_proj, _ = projections[0]
    max_proj, _ = projections[-1]

    # Convert back to actual points on the fold axis
    p1 = (
        center[0] + min_proj * axis[0],
        center[1] + min_proj * axis[1]
    )
    p2 = (
        center[0] + max_proj * axis[0],
        center[1] + max_proj * axis[1]
    )

    return (p1, p2)


def segments_intersect(
    p1: Point, p2: Point,
    p3: Point, p4: Point
) -> bool:
    """
    Check if line segment (p1, p2) intersects line segment (p3, p4).

    Uses cross product to determine if segments cross each other.
    """
    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    # Check if segments straddle each other
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Check for collinear cases (endpoint on segment)
    eps = 1e-9
    if abs(d1) < eps and _on_segment(p3, p1, p4):
        return True
    if abs(d2) < eps and _on_segment(p3, p2, p4):
        return True
    if abs(d3) < eps and _on_segment(p1, p3, p2):
        return True
    if abs(d4) < eps and _on_segment(p1, p4, p2):
        return True

    return False


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """Check if point q lies on segment pr."""
    return (min(p[0], r[0]) <= q[0] + 1e-9 <= max(p[0], r[0]) + 1e-9 and
            min(p[1], r[1]) <= q[1] + 1e-9 <= max(p[1], r[1]) + 1e-9)


def fold_segment_crosses_region_boundary(
    fold_extent: tuple[Point, Point],
    region_a: Region,
    region_b: Region
) -> bool:
    """
    Check if a finite fold segment crosses the shared boundary between two regions.

    Args:
        fold_extent: (p1, p2) endpoints of fold on board
        region_a: First region
        region_b: Second region

    Returns:
        True if the fold segment intersects the shared edge(s) between regions.
    """
    if fold_extent is None:
        return False

    fold_p1, fold_p2 = fold_extent

    # Get shared edges
    edges_a = _edges_from_polygon(region_a.outline)
    edges_b = _edges_from_polygon(region_b.outline)
    shared_edges = edges_a & edges_b

    if not shared_edges:
        return False

    # Check if fold segment crosses any shared edge
    for edge in shared_edges:
        edge_p1, edge_p2 = edge
        if segments_intersect(fold_p1, fold_p2, edge_p1, edge_p2):
            return True

    # Also check if fold segment is close to shared edges (within tolerance)
    # This handles cases where fold runs along the boundary
    for edge in shared_edges:
        edge_p1, edge_p2 = edge
        edge_mid = ((edge_p1[0] + edge_p2[0]) / 2, (edge_p1[1] + edge_p2[1]) / 2)

        # Check if edge midpoint is close to the fold line
        # Signed distance from edge_mid to fold line
        fold_dir = (fold_p2[0] - fold_p1[0], fold_p2[1] - fold_p1[1])
        fold_len = math.sqrt(fold_dir[0]**2 + fold_dir[1]**2)
        if fold_len < 1e-9:
            continue

        # Perpendicular direction
        perp = (-fold_dir[1] / fold_len, fold_dir[0] / fold_len)
        dist_to_line = abs(
            (edge_mid[0] - fold_p1[0]) * perp[0] +
            (edge_mid[1] - fold_p1[1]) * perp[1]
        )

        # If edge midpoint is very close to fold line, consider it crossed
        if dist_to_line < 0.1:  # Small tolerance
            # Also check that edge midpoint is within fold extent
            t = ((edge_mid[0] - fold_p1[0]) * fold_dir[0] +
                 (edge_mid[1] - fold_p1[1]) * fold_dir[1]) / (fold_len * fold_len)
            if -0.1 < t < 1.1:  # With some tolerance
                return True

    return False


def _edges_from_polygon(polygon: Polygon) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    """Extract all edges from a polygon as normalized tuples (sorted endpoints)."""
    edges = set()
    n = len(polygon)
    for i in range(n):
        p1 = (round(polygon[i][0], 6), round(polygon[i][1], 6))
        p2 = (round(polygon[(i + 1) % n][0], 6), round(polygon[(i + 1) % n][1], 6))
        # Normalize edge direction for comparison
        edge = (min(p1, p2), max(p1, p2))
        edges.add(edge)
    return edges


def build_region_adjacency(regions: list[Region]) -> dict[int, list[int]]:
    """
    Build adjacency list for regions based on shared edges.

    Two regions are adjacent if they share at least one edge.

    Args:
        regions: List of Region objects

    Returns:
        Dictionary mapping region index to list of adjacent region indices.
    """
    # Extract edges for each region
    region_edges: list[set] = []
    for region in regions:
        edges = _edges_from_polygon(region.outline)
        region_edges.append(edges)

    # Build adjacency by finding shared edges
    adjacency: dict[int, list[int]] = {r.index: [] for r in regions}

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            # Check if regions share any edge
            shared = region_edges[i] & region_edges[j]
            if shared:
                adjacency[regions[i].index].append(regions[j].index)
                adjacency[regions[j].index].append(regions[i].index)

    return adjacency


def classify_point_vs_fold(point: Point, fold_marker) -> str:
    """
    Classify a point relative to a fold marker.

    Args:
        point: (x, y) coordinates
        fold_marker: FoldMarker object with center, axis, zone_width

    Returns:
        "BEFORE", "IN_ZONE", or "AFTER"
    """
    # Get fold properties
    center = fold_marker.center
    axis = fold_marker.axis
    zone_width = getattr(fold_marker, 'zone_width', 0)
    half_width = zone_width / 2

    # Perpendicular direction
    perp = (-axis[1], axis[0])

    # Signed distance from fold center along perpendicular
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    perp_dist = dx * perp[0] + dy * perp[1]

    if perp_dist < -half_width:
        return "BEFORE"
    elif perp_dist > half_width:
        return "AFTER"
    else:
        return "IN_ZONE"


def find_anchor_region(regions: list[Region], fold_markers: list) -> Optional[Region]:
    """
    Find an anchor region that is BEFORE all folds.

    If no such region exists, return the region affected by fewest folds.

    Args:
        regions: List of Region objects
        fold_markers: List of FoldMarker objects

    Returns:
        The anchor region, or None if no regions exist.
    """
    if not regions:
        return None

    if not fold_markers:
        return regions[0]

    # Find region that is BEFORE all folds
    for region in regions:
        point = region.representative_point
        all_before = True
        for fold in fold_markers:
            if classify_point_vs_fold(point, fold) != "BEFORE":
                all_before = False
                break
        if all_before:
            return region

    # No region is before all folds - find one with fewest affecting folds
    best_region = regions[0]
    best_count = len(fold_markers) + 1

    for region in regions:
        point = region.representative_point
        count = 0
        for fold in fold_markers:
            if classify_point_vs_fold(point, fold) != "BEFORE":
                count += 1
        if count < best_count:
            best_count = count
            best_region = region

    return best_region


def get_shared_edge_midpoint(region_a: Region, region_b: Region) -> Optional[Point]:
    """
    Get the midpoint of the shared edge between two adjacent regions.

    Returns None if regions don't share an edge.
    """
    edges_a = _edges_from_polygon(region_a.outline)
    edges_b = _edges_from_polygon(region_b.outline)
    shared = edges_a & edges_b

    if not shared:
        return None

    # Use first shared edge
    edge = next(iter(shared))
    p1, p2 = edge
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def region_in_fold_column(fold_marker, region: Region) -> bool:
    """
    Check if a region is in the "column" of a fold - the band of board
    that the fold affects, extending infinitely along the fold direction.

    This is a broader check than fold_reaches_region: it returns True for
    both IN_ZONE regions (touched by marker lines) AND AFTER regions
    (beyond the marker lines but in the same band).

    For a vertical fold (markers oriented vertically):
    - The column is defined by the X-range of the marker endpoints
    - A region is in the column if its bounding box X-range overlaps

    Args:
        fold_marker: FoldMarker with line_a_start, line_a_end, etc.
        region: Region to check

    Returns:
        True if the region is in the fold's column
    """
    # Get fold marker line endpoints
    marker_xs = [
        fold_marker.line_a_start[0], fold_marker.line_a_end[0],
        fold_marker.line_b_start[0], fold_marker.line_b_end[0]
    ]
    marker_ys = [
        fold_marker.line_a_start[1], fold_marker.line_a_end[1],
        fold_marker.line_b_start[1], fold_marker.line_b_end[1]
    ]

    # Get the marker bounding box
    marker_x_min, marker_x_max = min(marker_xs), max(marker_xs)
    marker_y_min, marker_y_max = min(marker_ys), max(marker_ys)

    # Get region bounding box
    region_xs = [p[0] for p in region.outline]
    region_ys = [p[1] for p in region.outline]
    region_x_min, region_x_max = min(region_xs), max(region_xs)
    region_y_min, region_y_max = min(region_ys), max(region_ys)

    # Determine fold orientation by which axis the marker lines span more.
    # If markers span more in X (horizontal lines), the fold affects a vertical
    # "column" defined by that X range.
    # If markers span more in Y (vertical lines), the fold affects a horizontal
    # "band" defined by that Y range.
    marker_x_span = marker_x_max - marker_x_min
    marker_y_span = marker_y_max - marker_y_min

    # Small tolerance for floating point
    eps = 0.1

    if marker_x_span > marker_y_span:
        # Markers are horizontal (span X direction) - fold is a horizontal crease
        # The fold's column is defined by its X range
        # Regions must overlap in X to be in the column
        return not (region_x_max < marker_x_min - eps or region_x_min > marker_x_max + eps)
    else:
        # Markers are vertical (span Y direction) - fold is a vertical crease
        # The fold's column is defined by its Y range
        # Regions must overlap in Y to be in the column
        return not (region_y_max < marker_y_min - eps or region_y_min > marker_y_max + eps)


def fold_reaches_region(fold_marker, region: Region) -> bool:
    """
    Check if a fold's marker lines actually touch a region.

    A fold reaches a region if:
    1. Any endpoint of the fold's marker lines is inside the region, OR
    2. Any of the fold's marker lines intersect any edge of the region

    This is the key check for H-shaped boards: a fold on one arm
    shouldn't affect regions on the other arm.

    Args:
        fold_marker: FoldMarker with line_a_start, line_a_end, etc.
        region: Region to check

    Returns:
        True if the fold physically touches the region
    """
    # Get fold marker line endpoints
    marker_points = [
        fold_marker.line_a_start,
        fold_marker.line_a_end,
        fold_marker.line_b_start,
        fold_marker.line_b_end
    ]

    # Check if any marker endpoint is inside the region
    for p in marker_points:
        if point_in_polygon(p, region.outline):
            return True

    # Check if marker lines intersect region edges
    marker_lines = [
        (fold_marker.line_a_start, fold_marker.line_a_end),
        (fold_marker.line_b_start, fold_marker.line_b_end)
    ]

    n = len(region.outline)
    for line_start, line_end in marker_lines:
        for i in range(n):
            edge_start = region.outline[i]
            edge_end = region.outline[(i + 1) % n]
            if segments_intersect(line_start, line_end, edge_start, edge_end):
                return True

    return False


def detect_crossed_folds(
    region_from: Region,
    region_to: Region,
    fold_markers: list,
    already_in_recipe_with_class: list,
    fold_extents: dict = None
) -> list[tuple[object, str]]:
    """
    Detect which folds are crossed when moving from region_from to region_to.

    Uses finite fold segments: a fold is only considered crossed if its
    finite extent (where it intersects the board) actually crosses the
    boundary between the two regions.

    Entry Direction Detection:
    - Front entry (entered_from_back=False): crossing from BEFORE into IN_ZONE or AFTER
    - Back entry (entered_from_back=True): crossing from AFTER into IN_ZONE or BEFORE

    Back entry occurs in multi-fold configurations where BFS traversal reaches
    a fold from its "far side" (geometrically past the fold). The entry direction
    is tracked so the transformation can mirror appropriately.

    Args:
        region_from: Source region
        region_to: Destination region
        fold_markers: List of all FoldMarker objects
        already_in_recipe_with_class: List of (fold, classification, entered_from_back) tuples
        fold_extents: Dict mapping fold marker id to (p1, p2) extent on board

    Returns:
        List of (fold_marker, classification, entered_from_back) tuples for newly crossed folds.
    """
    crossed = []

    point_from = region_from.representative_point
    point_to = region_to.representative_point

    # Extract just the fold objects for presence checking
    already_in_recipe = [entry[0] for entry in already_in_recipe_with_class]

    for fold in fold_markers:
        # Check if fold is already in recipe using identity comparison
        already_present = any(f is fold for f in already_in_recipe)

        # FINITE FOLD CHECK: Skip folds that don't affect the destination region
        # BUT only apply this for NEW folds, not for folds already in recipe that
        # need to upgrade from IN_ZONE to AFTER.
        if not already_present and fold_extents is not None:
            fold_extent = fold_extents.get(id(fold))

            # If fold has no extent (doesn't intersect board), skip it entirely
            if fold_extent is None:
                continue

            # Check if the fold affects the destination region:
            # - fold_reaches_region: marker lines touch the region boundary
            # - region_in_fold_column: region is in the fold's band of influence
            # A region can be fully inside the fold zone without touching markers
            if not fold_reaches_region(fold, region_to) and not region_in_fold_column(fold, region_to):
                continue

        class_to = classify_point_vs_fold(point_to, fold)

        # For folds not yet in recipe: add when crossing into the fold
        # Recipe format: (fold, classification, entered_from_back)
        if not already_present:
            class_from = classify_point_vs_fold(point_from, fold)
            # Detect crossing from BEFORE into fold (normal entry)
            if class_from == "BEFORE" and class_to != "BEFORE":
                crossed.append((fold, class_to, False))  # entered_from_back = False
            # Also detect crossing from AFTER into fold (back entry)
            elif class_from == "AFTER" and class_to == "IN_ZONE":
                crossed.append((fold, "IN_ZONE", True))  # entered_from_back = True
            elif class_from == "AFTER" and class_to == "BEFORE":
                # Crossed entire fold from back - treat as AFTER
                crossed.append((fold, "AFTER", True))  # entered_from_back = True
        else:
            # For folds already in recipe: upgrade IN_ZONE to AFTER
            # Use the RECIPE classification (not point-based) to handle cases
            # where hole geometry causes regions to reconnect unexpectedly.
            current_class = None
            current_back = False
            for entry in already_in_recipe_with_class:
                f = entry[0]
                c = entry[1]
                back = entry[2] if len(entry) > 2 else False
                if f is fold:
                    current_class = c
                    current_back = back
                    break

            # Upgrade if currently IN_ZONE and destination is not IN_ZONE
            if current_class == "IN_ZONE" and class_to != "IN_ZONE":
                crossed.append((fold, "AFTER", current_back))  # preserve entry direction

    return crossed


def compute_fold_recipes(
    regions: list[Region],
    adjacency: dict[int, list[int]],
    fold_markers: list,
    board_outline: Polygon = None
) -> None:
    """
    Compute fold_recipe for each region using BFS from anchor.

    Uses finite fold segments: a fold only affects regions that its
    extent (where it intersects the board) actually reaches.

    Modifies regions in-place to set their fold_recipe field.

    Args:
        regions: List of Region objects
        adjacency: Adjacency dict from build_region_adjacency()
        fold_markers: List of FoldMarker objects
        board_outline: Board outline polygon for computing fold extents
    """
    if not regions:
        return

    # Create index lookup
    region_by_index = {r.index: r for r in regions}

    # Compute fold extents (where each fold intersects the board outline)
    fold_extents = {}
    if board_outline is not None:
        for fold in fold_markers:
            extent = compute_fold_extent_on_board(fold, board_outline)
            if extent is not None:
                fold_extents[id(fold)] = extent

    # Find anchor region
    anchor = find_anchor_region(regions, fold_markers)
    if anchor is None:
        return

    # BFS from anchor
    from collections import deque
    queue = deque([anchor])
    visited = {anchor.index}
    anchor.fold_recipe = []

    while queue:
        current = queue.popleft()

        for neighbor_idx in adjacency.get(current.index, []):
            if neighbor_idx in visited:
                continue

            neighbor = region_by_index[neighbor_idx]

            # Detect newly crossed folds (using finite fold segments)
            # Pass full recipe with classifications for proper IN_ZONE → AFTER upgrade
            crossed = detect_crossed_folds(
                current, neighbor, fold_markers, current.fold_recipe, fold_extents
            )

            # Neighbor's recipe inherits ALL folds from current.
            # Once you've crossed a fold, it stays in your recipe - you don't
            # "uncross" it by moving sideways to a different column.
            neighbor.fold_recipe = list(current.fold_recipe)

            # Add crossed folds, handling upgrades from IN_ZONE to AFTER
            # Recipe format: (fold, classification, entered_from_back)
            for fold, classification, entered_from_back in crossed:
                # Check if this fold is already in recipe (as IN_ZONE)
                existing_idx = None
                for i, entry in enumerate(neighbor.fold_recipe):
                    f = entry[0]
                    if f is fold:
                        existing_idx = i
                        break

                if existing_idx is not None:
                    # Upgrade classification if needed, preserve entry direction
                    if classification == "AFTER":
                        old_entry = neighbor.fold_recipe[existing_idx]
                        old_back = old_entry[2] if len(old_entry) > 2 else False
                        neighbor.fold_recipe[existing_idx] = (fold, "AFTER", old_back)
                else:
                    neighbor.fold_recipe.append((fold, classification, entered_from_back))

            visited.add(neighbor_idx)
            queue.append(neighbor)

    # Handle any disconnected regions (shouldn't happen normally)
    for region in regions:
        if region.index not in visited:
            # Classify directly based on representative point
            # But only for folds that actually intersect the board
            region.fold_recipe = []
            for fold in fold_markers:
                # Skip folds that don't intersect the board
                if fold_extents and id(fold) not in fold_extents:
                    continue

                classification = classify_point_vs_fold(
                    region.representative_point, fold
                )
                if classification != "BEFORE":
                    # For disconnected regions, assume normal entry (not from back)
                    region.fold_recipe.append((fold, classification, False))


def find_containing_region(point: Point, regions: list[Region]) -> Optional[Region]:
    """
    Find the region that contains a given point.

    Args:
        point: (x, y) coordinates
        regions: List of Region objects

    Returns:
        The containing Region, or None if point is outside all regions.
    """
    for region in regions:
        if point_in_polygon(point, region.outline):
            # Check it's not inside a hole
            in_hole = False
            for hole in region.holes:
                if point_in_polygon(point, hole):
                    in_hole = True
                    break
            if not in_hole:
                return region

    return None

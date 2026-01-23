"""
Stiffener region extraction and queries.

Stiffeners are rigid areas bonded to flex PCBs that cannot bend.
This module extracts stiffener polygons from KiCad layers and provides
geometry queries for validation.
"""

from dataclasses import dataclass
from typing import Optional

try:
    from .kicad_parser import KiCadPCB
    from .config import FlexConfig
except ImportError:
    from kicad_parser import KiCadPCB
    from config import FlexConfig


Point = tuple[float, float]
Polygon = list[Point]


@dataclass
class StiffenerRegion:
    """
    A stiffener region on the flex PCB.

    Attributes:
        outline: Polygon vertices (CCW winding for outer boundary)
        cutouts: List of hole polygons inside this stiffener (CW winding)
        layer: Source KiCad layer name
        thickness: Stiffener material thickness in mm
        side: Which side of the flex the stiffener is bonded to
    """
    outline: Polygon
    cutouts: list[Polygon]
    layer: str
    thickness: float
    side: str  # "top" or "bottom"

    @property
    def area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(self.outline)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.outline[i][0] * self.outline[j][1]
            area -= self.outline[j][0] * self.outline[i][1]
        return abs(area) / 2.0

    @property
    def centroid(self) -> Point:
        """Calculate polygon centroid."""
        n = len(self.outline)
        if n == 0:
            return (0.0, 0.0)
        if n < 3:
            # For degenerate cases, return average of points
            cx = sum(p[0] for p in self.outline) / n
            cy = sum(p[1] for p in self.outline) / n
            return (cx, cy)

        # Centroid of polygon using signed area
        signed_area = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(n):
            j = (i + 1) % n
            cross = (self.outline[i][0] * self.outline[j][1] -
                     self.outline[j][0] * self.outline[i][1])
            signed_area += cross
            cx += (self.outline[i][0] + self.outline[j][0]) * cross
            cy += (self.outline[i][1] + self.outline[j][1]) * cross

        signed_area /= 2.0
        if abs(signed_area) < 1e-10:
            # Degenerate polygon
            cx = sum(p[0] for p in self.outline) / n
            cy = sum(p[1] for p in self.outline) / n
            return (cx, cy)

        cx /= (6.0 * signed_area)
        cy /= (6.0 * signed_area)
        return (cx, cy)


def _signed_polygon_area(polygon: Polygon) -> float:
    """
    Calculate signed area of polygon using shoelace formula.

    Positive area = CCW winding (outer boundary)
    Negative area = CW winding (hole/cutout)
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


def _polygon_centroid(polygon: Polygon) -> Point:
    """Calculate centroid of a polygon."""
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return (cx, cy)


def _extract_stiffeners_from_layer(
    pcb: KiCadPCB,
    layer: str,
    thickness: float,
    side: str
) -> list[StiffenerRegion]:
    """
    Extract stiffener polygons from a specific layer.

    Detects cutouts (holes) within stiffener regions by analyzing polygon
    containment. A polygon is a hole only if it's contained within another
    polygon.

    Args:
        pcb: Parsed KiCad PCB
        layer: KiCad layer name to extract from
        thickness: Stiffener thickness in mm
        side: "top" or "bottom"

    Returns:
        List of StiffenerRegion objects with associated cutouts
    """
    polygons = pcb.get_layer_polygon_vertices(layer)

    if not polygons:
        return []

    # Filter out degenerate polygons
    valid_polygons = [p for p in polygons if len(p) >= 3]
    if not valid_polygons:
        return []

    # Determine which polygons are holes by checking containment
    # A polygon is a hole if its centroid is inside another polygon
    # and it has smaller area (to handle concentric shapes)
    n = len(valid_polygons)
    is_hole = [False] * n
    hole_parent = [-1] * n  # Index of containing polygon for each hole

    # Pre-compute areas
    areas = [abs(_signed_polygon_area(p)) for p in valid_polygons]

    for i in range(n):
        center_i = _polygon_centroid(valid_polygons[i])
        for j in range(n):
            if i == j:
                continue
            if point_in_polygon(center_i, valid_polygons[j]):
                # Polygon i's center is inside polygon j
                # It's a hole if it has smaller area (nested inside larger)
                if areas[i] < areas[j]:
                    is_hole[i] = True
                    hole_parent[i] = j
                    break

    # Collect outer boundaries (non-holes)
    outer_indices = [i for i in range(n) if not is_hole[i]]

    # Build mapping: outer_index -> list of holes
    outer_to_holes: dict[int, list[Polygon]] = {i: [] for i in outer_indices}

    for i in range(n):
        if is_hole[i] and hole_parent[i] in outer_to_holes:
            outer_to_holes[hole_parent[i]].append(valid_polygons[i])

    # Create StiffenerRegion objects
    # Ensure outer boundaries have CCW winding
    result = []
    for i in outer_indices:
        outline = valid_polygons[i]
        # Ensure CCW winding for outer boundary
        if _signed_polygon_area(outline) < 0:
            outline = list(reversed(outline))

        # Ensure CW winding for holes
        cutouts = []
        for hole in outer_to_holes[i]:
            if _signed_polygon_area(hole) > 0:
                hole = list(reversed(hole))
            cutouts.append(hole)

        result.append(StiffenerRegion(
            outline=outline,
            cutouts=cutouts,
            layer=layer,
            thickness=thickness,
            side=side
        ))

    return result


def extract_stiffeners(pcb: KiCadPCB, config: FlexConfig) -> list[StiffenerRegion]:
    """
    Extract stiffener polygons from configured top and bottom layers.

    Args:
        pcb: Parsed KiCad PCB
        config: Flex configuration specifying stiffener layers and thickness

    Returns:
        List of StiffenerRegion objects with associated cutouts
    """
    if not config.has_stiffener:
        return []

    result = []

    # Extract top stiffeners
    if config.has_top_stiffener:
        result.extend(_extract_stiffeners_from_layer(
            pcb, config.stiffener_layer_top, config.stiffener_thickness, "top"
        ))

    # Extract bottom stiffeners
    if config.has_bottom_stiffener:
        result.extend(_extract_stiffeners_from_layer(
            pcb, config.stiffener_layer_bottom, config.stiffener_thickness, "bottom"
        ))

    return result


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Check if a point is inside a polygon using ray casting.

    Args:
        point: (x, y) point to test
        polygon: List of (x, y) vertices

    Returns:
        True if point is inside or on boundary
    """
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if point is on a horizontal edge
        if abs(yi - yj) < 1e-10 and abs(y - yi) < 1e-10:
            if min(xi, xj) <= x <= max(xi, xj):
                return True

        if ((yi > y) != (yj > y)):
            slope = (x - xi) * (yj - yi) - (xj - xi) * (y - yi)
            if abs(slope) < 1e-10:
                return True  # On edge
            if (slope < 0) != (yj < yi):
                inside = not inside

        j = i

    return inside


def point_in_stiffener(point: Point, stiffeners: list[StiffenerRegion]) -> Optional[StiffenerRegion]:
    """
    Check if a point is inside any stiffener region.

    Args:
        point: (x, y) point to test
        stiffeners: List of stiffener regions

    Returns:
        The StiffenerRegion containing the point, or None if not in any stiffener
    """
    for stiffener in stiffeners:
        if point_in_polygon(point, stiffener.outline):
            return stiffener
    return None


def segment_intersects_polygon(p1: Point, p2: Point, polygon: Polygon) -> bool:
    """
    Check if a line segment intersects a polygon boundary.

    Args:
        p1, p2: Segment endpoints
        polygon: Polygon vertices

    Returns:
        True if segment crosses polygon boundary
    """
    n = len(polygon)
    if n < 3:
        return False

    for i in range(n):
        j = (i + 1) % n
        if _segments_intersect(p1, p2, polygon[i], polygon[j]):
            return True

    return False


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    Check if two line segments intersect (excluding endpoints touching).

    Uses cross product method.
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

    # Check for collinear cases
    eps = 1e-10
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
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def line_intersects_stiffener(
    p1: Point,
    p2: Point,
    stiffeners: list[StiffenerRegion]
) -> Optional[StiffenerRegion]:
    """
    Check if a line segment crosses any stiffener region.

    A line "crosses" if it intersects the stiffener boundary or
    passes through the interior.

    Args:
        p1, p2: Line segment endpoints
        stiffeners: List of stiffener regions

    Returns:
        The StiffenerRegion that the line crosses, or None if no crossing
    """
    for stiffener in stiffeners:
        # Check if segment intersects polygon boundary
        if segment_intersects_polygon(p1, p2, stiffener.outline):
            return stiffener

        # Check if segment is entirely inside polygon
        # (both endpoints inside, no boundary intersection)
        if (point_in_polygon(p1, stiffener.outline) and
            point_in_polygon(p2, stiffener.outline)):
            return stiffener

    return None


def fold_line_in_stiffener(
    center: Point,
    axis: tuple[float, float],
    extent: float,
    stiffeners: list[StiffenerRegion]
) -> Optional[StiffenerRegion]:
    """
    Check if a fold line (infinite line segment through center) crosses any stiffener.

    Args:
        center: Fold center point
        axis: Fold axis direction (normalized)
        extent: How far to extend the line in each direction
        stiffeners: List of stiffener regions

    Returns:
        The StiffenerRegion that the fold line crosses, or None
    """
    # Create line segment extending from center along axis
    dx, dy = axis
    p1 = (center[0] - dx * extent, center[1] - dy * extent)
    p2 = (center[0] + dx * extent, center[1] + dy * extent)

    return line_intersects_stiffener(p1, p2, stiffeners)


def get_stiffener_at_point(
    x: float,
    y: float,
    stiffeners: list[StiffenerRegion]
) -> Optional[StiffenerRegion]:
    """
    Get the stiffener region at a specific point (convenience wrapper).

    Args:
        x, y: Point coordinates
        stiffeners: List of stiffener regions

    Returns:
        StiffenerRegion at point, or None
    """
    return point_in_stiffener((x, y), stiffeners)

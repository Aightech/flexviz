"""
Fold marker detection and parsing.

Fold markers consist of:
- Two parallel dotted lines on User.1 layer (defining the bend zone)
- A dimension annotation between them (specifying the bend angle)
"""

from dataclasses import dataclass
from typing import Optional
import math
import re

try:
    from .kicad_parser import KiCadPCB, GraphicLine, Dimension
except ImportError:
    from kicad_parser import KiCadPCB, GraphicLine, Dimension


# Configuration
FOLD_MARKER_LAYER = "User.1"
PARALLEL_TOLERANCE_DEG = 5.0  # Lines considered parallel if angle differs by less than this
ASSOCIATION_DISTANCE_FACTOR = 3.0  # Dimension must be within this factor of line spacing


@dataclass
class FoldMarker:
    """
    A detected fold marker with all parameters needed for bend transformation.
    """
    # Line A (start of bend zone)
    line_a_start: tuple[float, float]
    line_a_end: tuple[float, float]

    # Line B (end of bend zone)
    line_b_start: tuple[float, float]
    line_b_end: tuple[float, float]

    # Bend parameters
    angle_degrees: float  # Positive = fold toward viewer, negative = away
    zone_width: float  # Distance between lines (arc length)
    radius: float  # Calculated: zone_width / angle_radians

    # Fold axis (unit vector along the fold lines)
    axis: tuple[float, float]

    # Fold center position (midpoint between the two lines)
    center: tuple[float, float]

    @property
    def angle_radians(self) -> float:
        return math.radians(self.angle_degrees)

    def __repr__(self):
        return (
            f"FoldMarker(angle={self.angle_degrees}°, "
            f"radius={self.radius:.2f}mm, "
            f"center=({self.center[0]:.2f}, {self.center[1]:.2f}))"
        )


def _line_angle(line: GraphicLine) -> float:
    """Calculate the angle of a line in radians (0 to pi)."""
    dx = line.end_x - line.start_x
    dy = line.end_y - line.start_y
    angle = math.atan2(dy, dx)
    # Normalize to 0 to pi (direction doesn't matter for parallelism)
    if angle < 0:
        angle += math.pi
    if angle >= math.pi:
        angle -= math.pi
    return angle


def _line_midpoint(line: GraphicLine) -> tuple[float, float]:
    """Get the midpoint of a line."""
    return (
        (line.start_x + line.end_x) / 2,
        (line.start_y + line.end_y) / 2
    )


def _line_length(line: GraphicLine) -> float:
    """Get the length of a line."""
    dx = line.end_x - line.start_x
    dy = line.end_y - line.start_y
    return math.sqrt(dx * dx + dy * dy)


def _distance_point_to_line(point: tuple[float, float], line: GraphicLine) -> float:
    """Calculate perpendicular distance from a point to a line segment."""
    x0, y0 = point
    x1, y1 = line.start_x, line.start_y
    x2, y2 = line.end_x, line.end_y

    # Line direction
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq < 1e-10:
        # Line is a point
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    # Project point onto line
    t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)


def _lines_parallel(line1: GraphicLine, line2: GraphicLine, tolerance_deg: float = PARALLEL_TOLERANCE_DEG) -> bool:
    """Check if two lines are approximately parallel."""
    angle1 = _line_angle(line1)
    angle2 = _line_angle(line2)

    diff = abs(angle1 - angle2)
    # Handle wrap-around at pi
    if diff > math.pi / 2:
        diff = math.pi - diff

    return diff < math.radians(tolerance_deg)


def _distance_between_parallel_lines(line1: GraphicLine, line2: GraphicLine) -> float:
    """Calculate the perpendicular distance between two parallel lines."""
    # Use midpoint of line2 and measure to line1
    mid2 = _line_midpoint(line2)
    return _distance_point_to_line(mid2, line1)


def _point_between_lines(point: tuple[float, float], line1: GraphicLine, line2: GraphicLine) -> bool:
    """Check if a point is in the region between two parallel lines."""
    d1 = _distance_point_to_line(point, line1)
    d2 = _distance_point_to_line(point, line2)
    line_dist = _distance_between_parallel_lines(line1, line2)

    # Point is between if sum of distances is approximately equal to line distance
    return abs(d1 + d2 - line_dist) < line_dist * 0.5


def _parse_angle_from_text(text: str) -> Optional[float]:
    """
    Parse angle value from dimension text.

    Handles formats:
    - "90"
    - "90°"
    - "+90°"
    - "-45.5°"
    - "90 deg"
    """
    if not text:
        return None

    # Remove common suffixes and whitespace
    text = text.strip()
    text = text.replace('°', '').replace('deg', '').replace('DEG', '').strip()

    # Try to parse as float
    match = re.match(r'^([+-]?\d+\.?\d*)$', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def find_dotted_lines(pcb: KiCadPCB, layer: str = FOLD_MARKER_LAYER) -> list[GraphicLine]:
    """
    Find all dotted/dashed lines on the specified layer.

    Lines with stroke_type 'dash', 'dot', 'dash_dot', or 'dash_dot_dot'
    are considered fold marker lines.
    """
    dotted_types = {'dash', 'dot', 'dash_dot', 'dash_dot_dot', 'default'}

    lines = pcb.get_graphic_lines(layer=layer)
    return [l for l in lines if l.stroke_type in dotted_types or l.stroke_type == 'solid']
    # Note: For initial implementation, we accept solid lines too
    # since the user might not set stroke type. Can be made stricter later.


def find_line_pairs(lines: list[GraphicLine]) -> list[tuple[GraphicLine, GraphicLine]]:
    """
    Find pairs of parallel lines that could be fold markers.

    Returns list of (line_a, line_b) tuples where:
    - Lines are approximately parallel
    - Lines have similar lengths (within 50%)
    - Lines are close together (relative to their length)
    """
    pairs = []
    used = set()

    for i, line1 in enumerate(lines):
        if i in used:
            continue

        best_match = None
        best_distance = float('inf')

        for j, line2 in enumerate(lines):
            if j <= i or j in used:
                continue

            # Check parallelism
            if not _lines_parallel(line1, line2):
                continue

            # Check similar length
            len1 = _line_length(line1)
            len2 = _line_length(line2)
            if len1 < 1e-6 or len2 < 1e-6:
                continue
            length_ratio = min(len1, len2) / max(len1, len2)
            if length_ratio < 0.5:
                continue

            # Check distance
            distance = _distance_between_parallel_lines(line1, line2)
            avg_length = (len1 + len2) / 2

            # Lines should be reasonably close (distance < 2x average length)
            if distance < avg_length * 2 and distance < best_distance:
                best_match = j
                best_distance = distance

        if best_match is not None:
            pairs.append((line1, lines[best_match]))
            used.add(i)
            used.add(best_match)

    return pairs


def _point_between_parallel_lines(point: tuple, line1: GraphicLine, line2: GraphicLine, tolerance: float = 0.5) -> bool:
    """
    Check if a point is between two parallel lines.

    The point is considered "between" if the sum of distances to both lines
    is approximately equal to the distance between the lines (with tolerance).
    """
    d1 = _distance_point_to_line(point, line1)
    d2 = _distance_point_to_line(point, line2)
    line_dist = _distance_between_parallel_lines(line1, line2)

    # Point is between if sum of distances ≈ line distance
    # tolerance is a fraction of line_dist
    return abs(d1 + d2 - line_dist) < line_dist * tolerance


def _point_along_lines(point: tuple, line1: GraphicLine, line2: GraphicLine) -> bool:
    """
    Check if a point is within the length span of the parallel lines.
    Projects the point onto the line direction and checks if it's within bounds.
    """
    # Use line1's direction
    dx = line1.end_x - line1.start_x
    dy = line1.end_y - line1.start_y
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1e-6:
        return False

    # Unit direction vector
    ux, uy = dx / length, dy / length

    # Project point onto line direction (relative to line1 start)
    px = point[0] - line1.start_x
    py = point[1] - line1.start_y
    proj = px * ux + py * uy

    # Get projection range from both lines
    min_proj = min(0, length)
    max_proj = max(0, length)

    # Also check line2's extent
    p2x = line2.start_x - line1.start_x
    p2y = line2.start_y - line1.start_y
    proj2_start = p2x * ux + p2y * uy

    p2ex = line2.end_x - line1.start_x
    p2ey = line2.end_y - line1.start_y
    proj2_end = p2ex * ux + p2ey * uy

    min_proj = min(min_proj, proj2_start, proj2_end)
    max_proj = max(max_proj, proj2_start, proj2_end)

    # Add some tolerance (20% of line length)
    margin = length * 0.2
    return (min_proj - margin) <= proj <= (max_proj + margin)


def find_containing_line_pair(
    point: tuple,
    lines: list[GraphicLine]
) -> Optional[tuple[GraphicLine, GraphicLine]]:
    """
    Find the pair of parallel lines that contains the given point.

    Returns the pair where:
    - Lines are parallel
    - Point is between the two lines
    - Point is within the length span of the lines
    - If multiple pairs match, returns the one with smallest line distance
    """
    best_pair = None
    best_distance = float('inf')

    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if j <= i:
                continue

            # Check parallelism
            if not _lines_parallel(line1, line2):
                continue

            # Check similar length
            len1 = _line_length(line1)
            len2 = _line_length(line2)
            if len1 < 1e-6 or len2 < 1e-6:
                continue
            length_ratio = min(len1, len2) / max(len1, len2)
            if length_ratio < 0.5:
                continue

            # Check if point is between the lines
            if not _point_between_parallel_lines(point, line1, line2):
                continue

            # Check if point is along the lines (within their span)
            if not _point_along_lines(point, line1, line2):
                continue

            # This pair contains the point - check if it's the closest
            distance = _distance_between_parallel_lines(line1, line2)
            if distance < best_distance:
                best_pair = (line1, line2)
                best_distance = distance

    return best_pair


def _get_line_pair_bbox(line1: GraphicLine, line2: GraphicLine, margin: float = 0) -> tuple:
    """
    Get bounding box for a line pair.
    Returns (min_x, min_y, max_x, max_y) with optional margin.
    """
    all_x = [line1.start_x, line1.end_x, line2.start_x, line2.end_x]
    all_y = [line1.start_y, line1.end_y, line2.start_y, line2.end_y]

    return (
        min(all_x) - margin,
        min(all_y) - margin,
        max(all_x) + margin,
        max(all_y) + margin
    )


def _boxes_overlap(box1: tuple, box2: tuple) -> bool:
    """
    Check if two bounding boxes overlap.
    Each box is (min_x, min_y, max_x, max_y).
    """
    return not (
        box1[2] < box2[0] or  # box1 is left of box2
        box1[0] > box2[2] or  # box1 is right of box2
        box1[3] < box2[1] or  # box1 is above box2
        box1[1] > box2[3]     # box1 is below box2
    )


def _point_in_box(point: tuple, box: tuple) -> bool:
    """Check if a point is inside a bounding box."""
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def associate_dimensions(
    line_pairs: list[tuple[GraphicLine, GraphicLine]],
    dimensions: list[Dimension]
) -> list[tuple[tuple[GraphicLine, GraphicLine], Dimension]]:
    """
    Associate dimension annotations with line pairs.

    A dimension is associated with a line pair if it is at least
    partially inside the bounding box formed by the two lines.
    """
    associations = []
    used_dims = set()

    for pair in line_pairs:
        line1, line2 = pair

        # Get bounding box of the line pair with margin
        line_dist = _distance_between_parallel_lines(line1, line2)
        margin = line_dist * 2  # Allow dimension to be somewhat outside
        pair_bbox = _get_line_pair_bbox(line1, line2, margin)

        best_dim = None
        best_dist = float('inf')

        for i, dim in enumerate(dimensions):
            if i in used_dims:
                continue

            # Dimension bounding box
            dim_bbox = (
                min(dim.start_x, dim.end_x),
                min(dim.start_y, dim.end_y),
                max(dim.start_x, dim.end_x),
                max(dim.start_y, dim.end_y)
            )

            # Check if boxes overlap or dimension center/endpoints are inside
            dim_center = ((dim.start_x + dim.end_x) / 2, (dim.start_y + dim.end_y) / 2)

            overlaps = (
                _boxes_overlap(pair_bbox, dim_bbox) or
                _point_in_box(dim_center, pair_bbox) or
                _point_in_box((dim.start_x, dim.start_y), pair_bbox) or
                _point_in_box((dim.end_x, dim.end_y), pair_bbox)
            )

            if overlaps:
                # Use distance to center of line pair as tiebreaker
                pair_center = (
                    (line1.start_x + line1.end_x + line2.start_x + line2.end_x) / 4,
                    (line1.start_y + line1.end_y + line2.start_y + line2.end_y) / 4
                )
                dist = math.sqrt(
                    (dim_center[0] - pair_center[0]) ** 2 +
                    (dim_center[1] - pair_center[1]) ** 2
                )
                if dist < best_dist:
                    best_dim = dim
                    best_dist = dist

        if best_dim is not None:
            associations.append((pair, best_dim))
            used_dims.add(dimensions.index(best_dim))

    return associations


def create_fold_marker(
    line_a: GraphicLine,
    line_b: GraphicLine,
    angle_degrees: float
) -> FoldMarker:
    """
    Create a FoldMarker from two lines and an angle.

    Calculates derived properties like radius, axis, and center.
    """
    # Calculate zone width (perpendicular distance between lines)
    zone_width = _distance_between_parallel_lines(line_a, line_b)

    # Calculate radius from zone width and angle
    angle_rad = math.radians(abs(angle_degrees))
    if angle_rad > 1e-6:
        radius = zone_width / angle_rad
    else:
        radius = float('inf')  # Flat, no bend

    # Calculate axis direction (along the lines)
    dx = line_a.end_x - line_a.start_x
    dy = line_a.end_y - line_a.start_y
    length = math.sqrt(dx * dx + dy * dy)
    if length > 1e-6:
        axis = (dx / length, dy / length)
    else:
        axis = (1.0, 0.0)

    # Normalize axis direction so parallel folds have consistent perp directions.
    # For mostly horizontal folds: ensure axis points in +X direction
    # For mostly vertical folds: ensure axis points in +Y direction
    # This makes perp = (-axis.y, axis.x) consistent across parallel folds.
    if abs(axis[0]) >= abs(axis[1]):
        # Mostly horizontal - ensure axis.x is positive
        if axis[0] < 0:
            axis = (-axis[0], -axis[1])
    else:
        # Mostly vertical - ensure axis.y is positive
        if axis[1] < 0:
            axis = (-axis[0], -axis[1])

    # Calculate center (midpoint between the two line midpoints)
    mid_a = _line_midpoint(line_a)
    mid_b = _line_midpoint(line_b)
    center = ((mid_a[0] + mid_b[0]) / 2, (mid_a[1] + mid_b[1]) / 2)

    return FoldMarker(
        line_a_start=(line_a.start_x, line_a.start_y),
        line_a_end=(line_a.end_x, line_a.end_y),
        line_b_start=(line_b.start_x, line_b.start_y),
        line_b_end=(line_b.end_x, line_b.end_y),
        angle_degrees=angle_degrees,
        zone_width=zone_width,
        radius=radius,
        axis=axis,
        center=center
    )


def detect_fold_markers(pcb: KiCadPCB, layer: str = FOLD_MARKER_LAYER) -> list[FoldMarker]:
    """
    Detect all fold markers in a KiCad PCB.

    Detection approach:
    1. Start from each dimension annotation on the layer
    2. Use the dimension's start point to find containing parallel lines
    3. The closest pair of parallel lines that contains the point is the match

    This avoids confusion when multiple fold markers are close together.

    Args:
        pcb: Parsed KiCad PCB
        layer: Layer to search for markers (default: User.1)

    Returns:
        List of detected FoldMarker objects
    """
    # Find candidate lines
    lines = find_dotted_lines(pcb, layer)
    if len(lines) < 2:
        return []

    # Get dimensions on the same layer
    dimensions = pcb.get_dimensions(layer=layer)
    if not dimensions:
        return []

    # Track which lines have been used
    used_lines = set()
    markers = []

    # For each dimension, find the containing line pair
    for dim in dimensions:
        # Use dimension start point as anchor
        dim_start = (dim.start_x, dim.start_y)

        # Filter out already-used lines
        available_lines = [l for i, l in enumerate(lines) if i not in used_lines]
        if len(available_lines) < 2:
            break

        # Find the line pair containing this dimension's start
        pair = find_containing_line_pair(dim_start, available_lines)

        if pair is None:
            continue

        line_a, line_b = pair

        # Mark these lines as used
        for i, l in enumerate(lines):
            if l is line_a or l is line_b:
                used_lines.add(i)

        # Parse angle from dimension
        angle = _parse_angle_from_text(dim.text)
        if angle is None:
            angle = dim.value  # Fall back to parsed numeric value

        if angle is not None:
            marker = create_fold_marker(line_a, line_b, angle)
            markers.append(marker)

    return markers


def sort_markers_by_position(
    markers: list[FoldMarker],
    axis: str = 'auto'
) -> list[FoldMarker]:
    """
    Sort fold markers by their position along an axis.

    Args:
        markers: List of fold markers
        axis: 'x', 'y', or 'auto' (detect dominant direction)

    Returns:
        Sorted list of markers
    """
    if not markers:
        return markers

    if axis == 'auto':
        # Determine which axis has more variation
        x_vals = [m.center[0] for m in markers]
        y_vals = [m.center[1] for m in markers]
        x_range = max(x_vals) - min(x_vals) if x_vals else 0
        y_range = max(y_vals) - min(y_vals) if y_vals else 0
        axis = 'x' if x_range >= y_range else 'y'

    if axis == 'x':
        return sorted(markers, key=lambda m: m.center[0])
    else:
        return sorted(markers, key=lambda m: m.center[1])

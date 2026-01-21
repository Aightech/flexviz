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


def associate_dimensions(
    line_pairs: list[tuple[GraphicLine, GraphicLine]],
    dimensions: list[Dimension]
) -> list[tuple[tuple[GraphicLine, GraphicLine], Dimension]]:
    """
    Associate dimension annotations with line pairs.

    A dimension is associated with a line pair if:
    - It's positioned between or near the two lines
    - Its orientation roughly matches the lines
    """
    associations = []

    for pair in line_pairs:
        line1, line2 = pair
        line_dist = _distance_between_parallel_lines(line1, line2)

        best_dim = None
        best_score = float('inf')

        for dim in dimensions:
            # Check if dimension center is near the line pair
            dim_center = ((dim.start_x + dim.end_x) / 2, (dim.start_y + dim.end_y) / 2)

            d1 = _distance_point_to_line(dim_center, line1)
            d2 = _distance_point_to_line(dim_center, line2)

            # Score based on how centered the dimension is between lines
            # and how close it is to the lines overall
            score = abs(d1 - d2) + min(d1, d2)

            # Dimension should be within reasonable range of the lines
            max_dist = line_dist * ASSOCIATION_DISTANCE_FACTOR
            if d1 < max_dist and d2 < max_dist and score < best_score:
                best_dim = dim
                best_score = score

        if best_dim is not None:
            associations.append((pair, best_dim))

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

    Fold markers consist of:
    - Two parallel dotted lines on the marker layer
    - A dimension annotation between them specifying the angle

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

    # Find line pairs
    pairs = find_line_pairs(lines)
    if not pairs:
        return []

    # Get dimensions on the same layer
    dimensions = pcb.get_dimensions(layer=layer)

    # Associate dimensions with line pairs
    associations = associate_dimensions(pairs, dimensions)

    # Create fold markers
    markers = []

    for (line_a, line_b), dim in associations:
        # Parse angle from dimension
        angle = _parse_angle_from_text(dim.text)
        if angle is None:
            angle = dim.value  # Fall back to parsed numeric value

        if angle is not None:
            marker = create_fold_marker(line_a, line_b, angle)
            markers.append(marker)

    # Also create markers for line pairs without dimensions (default angle)
    paired_lines = set()
    for (line_a, line_b), _ in associations:
        paired_lines.add(id(line_a))
        paired_lines.add(id(line_b))

    for line_a, line_b in pairs:
        if id(line_a) not in paired_lines:
            # No dimension found - could warn user or use default
            # For now, skip these
            pass

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

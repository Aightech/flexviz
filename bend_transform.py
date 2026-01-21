"""
Bend transformation mathematics.

Transforms 2D PCB geometry into 3D bent geometry based on fold markers.
"""

from dataclasses import dataclass
from typing import Optional
import math

try:
    from .markers import FoldMarker
    from .geometry import Polygon, LineSegment
except ImportError:
    from markers import FoldMarker
    from geometry import Polygon, LineSegment


@dataclass
class FoldDefinition:
    """
    Defines a fold for bend transformation.

    The fold is defined by:
    - A fold line (position and direction)
    - The bend zone width (arc length)
    - The bend angle (positive = fold up, negative = fold down)
    """
    # Position of the fold line center (x, y)
    position: tuple[float, float]

    # Direction of the fold axis (unit vector along the fold line)
    axis: tuple[float, float]

    # Width of the bend zone (distance between the two marker lines)
    zone_width: float

    # Bend angle in radians (positive = up/toward viewer, negative = down)
    angle: float

    @property
    def radius(self) -> float:
        """Bend radius calculated from zone width and angle."""
        if abs(self.angle) < 1e-6:
            return float('inf')
        return self.zone_width / abs(self.angle)

    @property
    def perpendicular(self) -> tuple[float, float]:
        """Unit vector perpendicular to the fold axis (in the bend direction)."""
        # Rotate axis 90 degrees
        return (-self.axis[1], self.axis[0])

    @classmethod
    def from_marker(cls, marker: FoldMarker) -> 'FoldDefinition':
        """Create FoldDefinition from a FoldMarker."""
        return cls(
            position=marker.center,
            axis=marker.axis,
            zone_width=marker.zone_width,
            angle=marker.angle_radians
        )


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Dot product of 2D vectors."""
    return a[0] * b[0] + a[1] * b[1]


def _project_onto_axis(
    point: tuple[float, float],
    origin: tuple[float, float],
    axis: tuple[float, float]
) -> tuple[float, float]:
    """
    Project a point onto an axis and return (along_axis, perpendicular) distances.

    Args:
        point: The point to project
        origin: Origin point on the axis
        axis: Unit vector along the axis

    Returns:
        (distance_along_axis, distance_perpendicular)
    """
    # Vector from origin to point
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]

    # Distance along axis
    along = dx * axis[0] + dy * axis[1]

    # Distance perpendicular to axis (using perpendicular vector)
    perp_axis = (-axis[1], axis[0])
    perp = dx * perp_axis[0] + dy * perp_axis[1]

    return (along, perp)


class PointClassification:
    """Classification of a point relative to a fold."""
    BEFORE = 0  # Before the fold zone (unchanged)
    IN_ZONE = 1  # Inside the fold zone (on the arc)
    AFTER = 2   # After the fold zone (rotated)


def classify_point(
    point: tuple[float, float],
    fold: FoldDefinition
) -> tuple[int, float]:
    """
    Classify a point relative to a fold zone.

    Args:
        point: 2D point (x, y)
        fold: Fold definition

    Returns:
        (classification, distance_into_zone)
        - classification: BEFORE, IN_ZONE, or AFTER
        - distance_into_zone: How far into the zone (0 at start, zone_width at end)
    """
    # Get perpendicular distance from fold center
    _, perp_dist = _project_onto_axis(point, fold.position, fold.axis)

    # Half the zone width on each side of center
    half_width = fold.zone_width / 2

    if perp_dist < -half_width:
        # Before the fold zone
        return (PointClassification.BEFORE, perp_dist + half_width)
    elif perp_dist > half_width:
        # After the fold zone
        return (PointClassification.AFTER, perp_dist - half_width)
    else:
        # Inside the fold zone
        # Convert to distance from start of zone (0 to zone_width)
        dist_into_zone = perp_dist + half_width
        return (PointClassification.IN_ZONE, dist_into_zone)


def transform_point_single_fold(
    point: tuple[float, float],
    fold: FoldDefinition,
    z: float = 0.0
) -> tuple[float, float, float]:
    """
    Transform a 2D point through a single fold to 3D.

    Args:
        point: 2D point (x, y)
        fold: Fold definition
        z: Initial Z coordinate (for multi-fold, carries over)

    Returns:
        3D point (x, y, z) after the bend transformation
    """
    classification, dist = classify_point(point, fold)

    # Get position along the fold axis (this coordinate stays constant)
    along_axis, perp_dist = _project_onto_axis(point, fold.position, fold.axis)

    # Perpendicular direction
    perp = fold.perpendicular

    if classification == PointClassification.BEFORE:
        # Point is before the fold - unchanged except for existing z
        return (point[0], point[1], z)

    elif classification == PointClassification.IN_ZONE:
        # Point is in the bend zone - map to cylindrical arc
        R = fold.radius

        # Angle at this position in the arc
        # dist goes from 0 to zone_width
        arc_fraction = dist / fold.zone_width
        theta = arc_fraction * fold.angle

        # Start of the bend zone in 2D
        half_width = fold.zone_width / 2
        zone_start_x = fold.position[0] - half_width * perp[0]
        zone_start_y = fold.position[1] - half_width * perp[1]

        # Position along the fold axis relative to fold center
        axis_offset_x = along_axis * fold.axis[0]
        axis_offset_y = along_axis * fold.axis[1]

        # Calculate 3D position on the arc
        # The arc bends in the perpendicular direction
        if abs(fold.angle) < 1e-6:
            # Nearly flat, just offset
            new_x = zone_start_x + axis_offset_x + dist * perp[0]
            new_y = zone_start_y + axis_offset_y + dist * perp[1]
            new_z = z
        else:
            # Bend center is at radius R in the -z direction from the start
            # (for positive angle bending upward)
            sign = 1 if fold.angle >= 0 else -1

            # Arc position
            arc_y_offset = R * math.sin(abs(theta))
            arc_z_offset = R * (1 - math.cos(abs(theta))) * sign

            new_x = zone_start_x + axis_offset_x + arc_y_offset * perp[0]
            new_y = zone_start_y + axis_offset_y + arc_y_offset * perp[1]
            new_z = z + arc_z_offset

        return (new_x, new_y, new_z)

    else:  # AFTER
        # Point is after the fold - apply full rotation
        R = fold.radius

        # End of the bend zone in 2D
        half_width = fold.zone_width / 2
        zone_end_x = fold.position[0] + half_width * perp[0]
        zone_end_y = fold.position[1] + half_width * perp[1]

        # Position along the fold axis relative to fold center
        axis_offset_x = along_axis * fold.axis[0]
        axis_offset_y = along_axis * fold.axis[1]

        # Full arc endpoint
        sign = 1 if fold.angle >= 0 else -1

        if abs(fold.angle) < 1e-6:
            # Nearly flat
            new_x = point[0]
            new_y = point[1]
            new_z = z
        else:
            # End of arc position
            arc_end_y = R * math.sin(abs(fold.angle))
            arc_end_z = R * (1 - math.cos(abs(fold.angle))) * sign

            # Distance beyond the fold zone
            beyond = dist  # This is the distance past the zone end

            # Direction after the bend (rotated by the bend angle)
            # Original direction was along perp, now rotated by angle
            cos_a = math.cos(fold.angle)
            sin_a = math.sin(fold.angle)

            # Rotated perpendicular direction (in the perp-z plane)
            # After bending, "forward" direction is rotated
            rotated_perp_y = cos_a  # Component in original perp direction
            rotated_perp_z = sin_a  # Component in z direction

            # Start position (end of arc)
            start_x = fold.position[0] + axis_offset_x + (half_width + arc_end_y) * perp[0] - half_width * perp[0]
            start_y = fold.position[1] + axis_offset_y + (half_width + arc_end_y) * perp[1] - half_width * perp[1]
            start_z = z + arc_end_z

            # Simplified: just offset from arc end in rotated direction
            new_x = start_x + beyond * perp[0] * rotated_perp_y
            new_y = start_y + beyond * perp[1] * rotated_perp_y
            new_z = start_z + beyond * rotated_perp_z

        return (new_x, new_y, new_z)


def transform_point(
    point: tuple[float, float],
    folds: list[FoldDefinition]
) -> tuple[float, float, float]:
    """
    Transform a 2D point through multiple folds to 3D.

    Folds are applied in order based on their position.

    Args:
        point: 2D point (x, y)
        folds: List of fold definitions

    Returns:
        3D point (x, y, z) after all bend transformations
    """
    if not folds:
        return (point[0], point[1], 0.0)

    # Sort folds by perpendicular distance from point
    # (apply closest folds first)
    # For now, just apply in given order

    result = (point[0], point[1], 0.0)

    for fold in folds:
        # Transform through this fold
        result = transform_point_single_fold((result[0], result[1]), fold, result[2])

    return result


def transform_polygon(
    polygon: Polygon,
    folds: list[FoldDefinition],
    subdivide: bool = True,
    max_edge_length: float = 1.0
) -> list[tuple[float, float, float]]:
    """
    Transform a 2D polygon through folds to 3D vertices.

    Args:
        polygon: 2D polygon
        folds: List of fold definitions
        subdivide: Whether to subdivide edges for smoother bends
        max_edge_length: Maximum edge length when subdividing

    Returns:
        List of 3D vertices
    """
    if subdivide:
        # Subdivide polygon for smoother bends
        from .geometry import subdivide_polygon
        polygon = subdivide_polygon(polygon, max_edge_length)

    # Transform each vertex
    vertices_3d = []
    for vertex in polygon.vertices:
        v3d = transform_point(vertex, folds)
        vertices_3d.append(v3d)

    return vertices_3d


def transform_line_segment(
    segment: LineSegment,
    folds: list[FoldDefinition],
    subdivisions: int = 10
) -> list[tuple[float, float, float]]:
    """
    Transform a 2D line segment through folds to 3D points.

    Args:
        segment: 2D line segment
        folds: List of fold definitions
        subdivisions: Number of subdivisions along the segment

    Returns:
        List of 3D points along the transformed segment
    """
    points = []

    for i in range(subdivisions + 1):
        t = i / subdivisions
        x = segment.start[0] + t * (segment.end[0] - segment.start[0])
        y = segment.start[1] + t * (segment.end[1] - segment.start[1])

        point_3d = transform_point((x, y), folds)
        points.append(point_3d)

    return points


def create_fold_definitions(markers: list[FoldMarker]) -> list[FoldDefinition]:
    """
    Create fold definitions from fold markers.

    Args:
        markers: List of fold markers detected from the PCB

    Returns:
        List of fold definitions ready for transformation
    """
    return [FoldDefinition.from_marker(m) for m in markers]

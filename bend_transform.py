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


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Dot product of 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """Cross product of 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a 3D vector."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < 1e-10:
        return (0.0, 0.0, 1.0)
    return (v[0]/length, v[1]/length, v[2]/length)


def _mat3_identity() -> list[list[float]]:
    """Return 3x3 identity matrix."""
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _mat3_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 3x3 matrices."""
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _mat3_vec3(m: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Multiply 3x3 matrix by 3D vector."""
    return (
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2]
    )


def _rotation_matrix_axis_angle(axis: tuple[float, float, float], angle: float) -> list[list[float]]:
    """
    Create rotation matrix for rotation around arbitrary axis by angle (radians).
    Uses Rodrigues' rotation formula.
    """
    ax, ay, az = _normalize3(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    return [
        [t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay],
        [t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax],
        [t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c   ]
    ]


def _rotate_point_around_axis(
    point: tuple[float, float, float],
    pivot: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle: float
) -> tuple[float, float, float]:
    """
    Rotate a 3D point around an axis passing through a pivot point.

    Args:
        point: The point to rotate
        pivot: A point on the rotation axis
        axis: Direction of the rotation axis (will be normalized)
        angle: Rotation angle in radians

    Returns:
        Rotated point
    """
    # Translate point to origin (pivot becomes origin)
    p = (point[0] - pivot[0], point[1] - pivot[1], point[2] - pivot[2])

    # Create rotation matrix and apply
    rot = _rotation_matrix_axis_angle(axis, angle)
    rotated = _mat3_vec3(rot, p)

    # Translate back
    return (
        rotated[0] + pivot[0],
        rotated[1] + pivot[1],
        rotated[2] + pivot[2]
    )


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

            # Start position (end of arc) - must match where IN_ZONE ends
            # The arc starts at zone_start = fold.position - half_width * perp
            # and ends at zone_start + arc_end_y * perp = fold.position + (arc_end_y - half_width) * perp
            start_x = fold.position[0] + axis_offset_x + (arc_end_y - half_width) * perp[0]
            start_y = fold.position[1] + axis_offset_y + (arc_end_y - half_width) * perp[1]
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

    Handles both parallel and non-parallel fold axes.
    Classification uses ORIGINAL 2D coordinates, transformations compound properly.

    Args:
        point: 2D point (x, y)
        folds: List of fold definitions

    Returns:
        3D point (x, y, z) after all bend transformations
    """
    if not folds:
        return (point[0], point[1], 0.0)

    # Sort folds by their perpendicular distance from first fold's reference
    if len(folds) > 1:
        ref_perp = folds[0].perpendicular
        def fold_sort_key(f):
            return f.position[0] * ref_perp[0] + f.position[1] * ref_perp[1]
        folds = sorted(folds, key=fold_sort_key)

    # Start with the 2D point at z=0
    pos = [point[0], point[1], 0.0]

    for i, fold in enumerate(folds):
        classification, dist = classify_point(point, fold)

        if classification == PointClassification.BEFORE:
            # This fold doesn't affect the point
            continue

        half_width = fold.zone_width / 2
        perp = fold.perpendicular
        R = fold.radius
        sign = 1 if fold.angle >= 0 else -1

        # Get position along this fold's axis
        along_axis, _ = _project_onto_axis(point, fold.position, fold.axis)

        # Compute fold axis and pivot in 3D (transformed by earlier folds)
        if i == 0:
            # First fold - axis and pivot are in the XY plane
            fold_axis_3d = (fold.axis[0], fold.axis[1], 0.0)
            pivot_3d = (
                fold.position[0] + along_axis * fold.axis[0] - half_width * perp[0],
                fold.position[1] + along_axis * fold.axis[1] - half_width * perp[1],
                0.0
            )
            perp_3d = (perp[0], perp[1], 0.0)
            up_3d = (0.0, 0.0, 1.0)
        else:
            # Transform axis and pivot through earlier folds
            axis_p1 = fold.position
            axis_p2 = (fold.position[0] + fold.axis[0], fold.position[1] + fold.axis[1])
            axis_p1_3d = transform_point(axis_p1, folds[:i])
            axis_p2_3d = transform_point(axis_p2, folds[:i])
            fold_axis_3d = _normalize3((
                axis_p2_3d[0] - axis_p1_3d[0],
                axis_p2_3d[1] - axis_p1_3d[1],
                axis_p2_3d[2] - axis_p1_3d[2]
            ))

            pivot_2d = (
                fold.position[0] + along_axis * fold.axis[0] - half_width * perp[0],
                fold.position[1] + along_axis * fold.axis[1] - half_width * perp[1]
            )
            pivot_3d = transform_point(pivot_2d, folds[:i])

            # Perpendicular direction in 3D
            perp_point_2d = (pivot_2d[0] + perp[0], pivot_2d[1] + perp[1])
            perp_point_3d = transform_point(perp_point_2d, folds[:i])
            perp_3d = _normalize3((
                perp_point_3d[0] - pivot_3d[0],
                perp_point_3d[1] - pivot_3d[1],
                perp_point_3d[2] - pivot_3d[2]
            ))

            # Up direction: perpendicular to both axis and perp
            up_3d = _normalize3(_cross3(fold_axis_3d, perp_3d))

        # Ensure up direction is consistent with fold direction
        # For the first fold, up should point +z for positive angles
        if i == 0:
            if sign < 0:
                up_3d = (0.0, 0.0, -1.0)
        else:
            # Check if up is pointing the right way relative to fold direction
            # up should be in the direction of the bend
            if sign < 0:
                up_3d = (-up_3d[0], -up_3d[1], -up_3d[2])

        if classification == PointClassification.IN_ZONE:
            # Point is on the cylindrical arc
            arc_fraction = dist / fold.zone_width
            theta = arc_fraction * fold.angle

            if abs(fold.angle) < 1e-6:
                arc_perp = dist
                arc_up = 0.0
            else:
                arc_perp = R * math.sin(abs(theta))
                arc_up = R * (1 - math.cos(abs(theta))) * sign

            pos = [
                pivot_3d[0] + arc_perp * perp_3d[0] + arc_up * up_3d[0],
                pivot_3d[1] + arc_perp * perp_3d[1] + arc_up * up_3d[1],
                pivot_3d[2] + arc_perp * perp_3d[2] + arc_up * up_3d[2]
            ]
            # Point is in a zone - done processing
            break

        else:  # AFTER
            # Compute arc end position (same as IN_ZONE with full angle)
            if abs(fold.angle) < 1e-6:
                arc_perp = fold.zone_width
                arc_up = 0.0
            else:
                arc_perp = R * math.sin(abs(fold.angle))
                arc_up = R * (1 - math.cos(abs(fold.angle))) * sign

            arc_end = (
                pivot_3d[0] + arc_perp * perp_3d[0] + arc_up * up_3d[0],
                pivot_3d[1] + arc_perp * perp_3d[1] + arc_up * up_3d[1],
                pivot_3d[2] + arc_perp * perp_3d[2] + arc_up * up_3d[2]
            )

            # Rotated perpendicular direction (after the fold)
            # The surface continues in a direction rotated by the fold angle
            cos_a = math.cos(fold.angle)
            sin_a = math.sin(fold.angle)

            # Rotated direction = cos(angle) * perp + sin(angle) * up
            rotated_dir = (
                cos_a * perp_3d[0] + sin_a * up_3d[0],
                cos_a * perp_3d[1] + sin_a * up_3d[1],
                cos_a * perp_3d[2] + sin_a * up_3d[2]
            )

            # Position = arc_end + flat_offset * rotated_direction
            pos = [
                arc_end[0] + dist * rotated_dir[0],
                arc_end[1] + dist * rotated_dir[1],
                arc_end[2] + dist * rotated_dir[2]
            ]

    return (pos[0], pos[1], pos[2])


def compute_surface_normal(
    point: tuple[float, float],
    folds: list[FoldDefinition]
) -> tuple[float, float, float]:
    """
    Compute the surface normal at a point after bend transformation.

    Handles both parallel and non-parallel fold axes using rotation matrices.

    Args:
        point: 2D point (x, y)
        folds: List of fold definitions

    Returns:
        Unit normal vector (nx, ny, nz) pointing away from the surface
    """
    if not folds:
        return (0.0, 0.0, 1.0)

    # Sort folds consistently with transform_point
    if len(folds) > 1:
        ref_perp = folds[0].perpendicular
        def fold_sort_key(f):
            return f.position[0] * ref_perp[0] + f.position[1] * ref_perp[1]
        folds = sorted(folds, key=fold_sort_key)

    # Track cumulative rotation matrix
    cumulative_rotation = _mat3_identity()
    normal = (0.0, 0.0, 1.0)

    for fold in folds:
        classification, dist = classify_point(point, fold)
        fold_axis_3d = (fold.axis[0], fold.axis[1], 0.0)

        if classification == PointClassification.BEFORE:
            # Normal unchanged by this fold
            pass

        elif classification == PointClassification.IN_ZONE:
            # On cylindrical surface - normal rotated by partial angle
            arc_fraction = dist / fold.zone_width
            theta = arc_fraction * fold.angle

            # Create rotation for this partial angle - axis must be rotated for non-parallel folds
            rotated_fold_axis = _mat3_vec3(cumulative_rotation, fold_axis_3d)
            partial_rotation = _rotation_matrix_axis_angle(rotated_fold_axis, theta)

            # Apply cumulative rotation then partial rotation
            total_rotation = _mat3_multiply(partial_rotation, cumulative_rotation)
            normal = _mat3_vec3(total_rotation, (0.0, 0.0, 1.0))
            break  # Point is in zone, don't process further folds

        else:  # AFTER
            # Full rotation from this fold - axis must be rotated for non-parallel folds
            rotated_fold_axis = _mat3_vec3(cumulative_rotation, fold_axis_3d)
            fold_rotation = _rotation_matrix_axis_angle(rotated_fold_axis, fold.angle)
            cumulative_rotation = _mat3_multiply(fold_rotation, cumulative_rotation)
            normal = _mat3_vec3(cumulative_rotation, (0.0, 0.0, 1.0))

    # Normalize
    length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if length > 1e-10:
        normal = (normal[0]/length, normal[1]/length, normal[2]/length)

    return normal


def transform_point_with_normal(
    point: tuple[float, float],
    folds: list[FoldDefinition]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Transform a 2D point and compute its surface normal.

    Args:
        point: 2D point (x, y)
        folds: List of fold definitions

    Returns:
        Tuple of (position, normal) where both are (x, y, z) tuples
    """
    pos = transform_point(point, folds)
    normal = compute_surface_normal(point, folds)
    return (pos, normal)


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

"""
Bend Transformation Module

Transforms 2D PCB coordinates to 3D based on fold recipes from region subdivision.

The transformation is purely recipe-based: each region has a fold_recipe that
specifies exactly which folds affect it and how (IN_ZONE or AFTER).
No geometric re-classification is performed.
"""

from dataclasses import dataclass
import math


@dataclass
class FoldDefinition:
    """Defines a fold for transformation."""
    center: tuple[float, float]  # Fold line center
    axis: tuple[float, float]    # Unit vector along fold line
    zone_width: float            # Width of bend zone
    angle: float                 # Bend angle in radians (positive = up)

    @property
    def radius(self) -> float:
        """Bend radius: R = arc_length / angle."""
        if abs(self.angle) < 1e-9:
            return float('inf')
        return self.zone_width / abs(self.angle)

    @property
    def perp(self) -> tuple[float, float]:
        """Perpendicular to axis (direction of bending)."""
        return (-self.axis[1], self.axis[0])

    @classmethod
    def from_marker(cls, marker) -> 'FoldDefinition':
        """Create from a FoldMarker object."""
        return cls(
            center=marker.center,
            axis=marker.axis,
            zone_width=marker.zone_width,
            angle=marker.angle_radians
        )


# Type alias for fold recipe
FoldRecipe = list[tuple[FoldDefinition, str]]  # [(fold, "IN_ZONE" or "AFTER"), ...]


def transform_point(
    point: tuple[float, float],
    recipe: FoldRecipe
) -> tuple[float, float, float]:
    """
    Transform a 2D point to 3D using a fold recipe.

    The recipe determines exactly how each fold affects this point.
    No geometric classification is performed - we trust the recipe.

    Args:
        point: 2D point (x, y)
        recipe: List of (FoldDefinition, classification) tuples

    Returns:
        3D point (x, y, z)
    """
    if not recipe:
        return (point[0], point[1], 0.0)

    # Current 3D position and coordinate frame
    pos = [point[0], point[1], 0.0]

    # Track cumulative rotation for chaining folds
    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Identity

    for fold, classification in recipe:
        # Project point onto fold's coordinate system
        dx = point[0] - fold.center[0]
        dy = point[1] - fold.center[1]

        # Distance along fold axis (preserved)
        along = dx * fold.axis[0] + dy * fold.axis[1]

        # Distance perpendicular to fold axis
        perp_dist = dx * fold.perp[0] + dy * fold.perp[1]

        # Half-width of bend zone
        hw = fold.zone_width / 2

        if classification == "AFTER":
            # Point is after this fold - apply full rotation
            R = fold.radius

            if abs(fold.angle) < 1e-9:
                # No bend - straight through
                arc_end_perp = fold.zone_width
                arc_end_up = 0.0
            else:
                arc_end_perp = R * math.sin(abs(fold.angle))
                arc_end_up = R * (1 - math.cos(abs(fold.angle)))
                if fold.angle < 0:
                    arc_end_up = -arc_end_up

            # Distance beyond the zone
            dist_from_start = perp_dist + hw
            beyond = dist_from_start - fold.zone_width
            if beyond < 0:
                beyond = 0

            # Direction after bend (rotated by fold angle)
            cos_a = math.cos(fold.angle)
            sin_a = math.sin(fold.angle)

            # Position in fold's local system
            local_perp = arc_end_perp + beyond * cos_a - hw
            local_up = arc_end_up + beyond * sin_a

            # Transform to 3D with current rotation
            base_x = fold.center[0] + along * fold.axis[0] + local_perp * fold.perp[0]
            base_y = fold.center[1] + along * fold.axis[1] + local_perp * fold.perp[1]
            base_z = local_up

            pos = _apply_rotation(rot, (base_x, base_y, base_z))

            # Update rotation for subsequent folds
            fold_rot = _rotation_matrix_around_axis(
                (fold.axis[0], fold.axis[1], 0),
                fold.angle
            )
            rot = _multiply_matrices(fold_rot, rot)

        elif classification == "IN_ZONE":
            # Point is in the bend zone - map to cylindrical arc
            dist_into_zone = perp_dist + hw
            dist_into_zone = max(0, min(dist_into_zone, fold.zone_width))

            arc_fraction = dist_into_zone / fold.zone_width if fold.zone_width > 0 else 0
            theta = arc_fraction * fold.angle

            R = fold.radius

            if abs(fold.angle) < 1e-9:
                local_perp = dist_into_zone - hw
                local_up = 0.0
            else:
                local_perp = R * math.sin(abs(theta)) - hw
                local_up = R * (1 - math.cos(abs(theta)))
                if fold.angle < 0:
                    local_up = -local_up

            base_x = fold.center[0] + along * fold.axis[0] + local_perp * fold.perp[0]
            base_y = fold.center[1] + along * fold.axis[1] + local_perp * fold.perp[1]
            base_z = local_up

            pos = _apply_rotation(rot, (base_x, base_y, base_z))
            break  # IN_ZONE is terminal

    return (pos[0], pos[1], pos[2])


def compute_normal(
    point: tuple[float, float],
    recipe: FoldRecipe
) -> tuple[float, float, float]:
    """
    Compute surface normal at a point using a fold recipe.

    Args:
        point: 2D point (x, y)
        recipe: List of (FoldDefinition, classification) tuples

    Returns:
        Unit normal vector (nx, ny, nz)
    """
    if not recipe:
        return (0.0, 0.0, 1.0)

    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for fold, classification in recipe:
        if classification == "AFTER":
            fold_rot = _rotation_matrix_around_axis(
                (fold.axis[0], fold.axis[1], 0),
                fold.angle
            )
            rot = _multiply_matrices(fold_rot, rot)

        elif classification == "IN_ZONE":
            dx = point[0] - fold.center[0]
            dy = point[1] - fold.center[1]
            perp_dist = dx * fold.perp[0] + dy * fold.perp[1]
            hw = fold.zone_width / 2

            dist_into_zone = perp_dist + hw
            dist_into_zone = max(0, min(dist_into_zone, fold.zone_width))

            arc_fraction = dist_into_zone / fold.zone_width if fold.zone_width > 0 else 0
            theta = arc_fraction * fold.angle

            fold_rot = _rotation_matrix_around_axis(
                (fold.axis[0], fold.axis[1], 0),
                theta
            )
            rot = _multiply_matrices(fold_rot, rot)
            break

    normal = _apply_rotation(rot, (0.0, 0.0, 1.0))

    length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if length > 1e-10:
        normal = (normal[0]/length, normal[1]/length, normal[2]/length)

    return normal


def transform_point_and_normal(
    point: tuple[float, float],
    recipe: FoldRecipe
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Transform point and compute normal in one call."""
    return (transform_point(point, recipe), compute_normal(point, recipe))


def create_fold_definitions(markers: list) -> list[FoldDefinition]:
    """
    Create fold definitions from fold markers.

    Args:
        markers: List of FoldMarker objects

    Returns:
        List of FoldDefinition objects
    """
    return [FoldDefinition.from_marker(m) for m in markers]


# =============================================================================
# Matrix Helpers
# =============================================================================

def _rotation_matrix_around_axis(
    axis: tuple[float, float, float],
    angle: float
) -> list[list[float]]:
    """Create 3x3 rotation matrix for rotation around axis by angle."""
    length = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if length < 1e-10:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    ax, ay, az = axis[0]/length, axis[1]/length, axis[2]/length
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c

    return [
        [t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay],
        [t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax],
        [t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c]
    ]


def _multiply_matrices(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 3x3 matrices."""
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _apply_rotation(
    rot: list[list[float]],
    vec: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply 3x3 rotation matrix to vector."""
    return (
        rot[0][0]*vec[0] + rot[0][1]*vec[1] + rot[0][2]*vec[2],
        rot[1][0]*vec[0] + rot[1][1]*vec[1] + rot[1][2]*vec[2],
        rot[2][0]*vec[0] + rot[2][1]*vec[1] + rot[2][2]*vec[2]
    )

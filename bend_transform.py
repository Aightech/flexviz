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

    For multiple folds, we track both the cumulative rotation AND translation
    (affine transformation) through each fold.

    Key insight: The perpendicular distance from a fold to a point is preserved
    through previous bends (isometric transformation), so we use ORIGINAL 2D
    coordinates to compute along/perp_dist, but track the 3D origin correctly.

    Args:
        point: 2D point (x, y)
        recipe: List of (FoldDefinition, classification) tuples

    Returns:
        3D point (x, y, z)
    """
    if not recipe:
        return (point[0], point[1], 0.0)

    # Track cumulative affine transformation: rot (rotation) and origin (translation)
    # A point p transforms as: rot @ p + origin
    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Identity rotation
    origin = (0.0, 0.0, 0.0)  # No translation initially

    for fold, classification in recipe:
        hw = fold.zone_width / 2
        R = fold.radius

        # Use ORIGINAL 2D coordinates to compute distances (preserved through bending)
        dx = point[0] - fold.center[0]
        dy = point[1] - fold.center[1]
        along = dx * fold.axis[0] + dy * fold.axis[1]
        perp_dist = dx * fold.perp[0] + dy * fold.perp[1]

        # Transform fold's local coordinate frame through cumulative rotation
        fold_axis_3d = _apply_rotation(rot, (fold.axis[0], fold.axis[1], 0.0))
        fold_perp_3d = _apply_rotation(rot, (fold.perp[0], fold.perp[1], 0.0))
        up_3d = _apply_rotation(rot, (0.0, 0.0, 1.0))

        # Compute fold center in 3D using the affine transformation
        fold_center_3d = (
            rot[0][0] * fold.center[0] + rot[0][1] * fold.center[1] + origin[0],
            rot[1][0] * fold.center[0] + rot[1][1] * fold.center[1] + origin[1],
            rot[2][0] * fold.center[0] + rot[2][1] * fold.center[1] + origin[2]
        )

        if classification == "AFTER":
            # Point is after this fold - apply pure rigid rotation around fold axis
            # All points in an AFTER region rotate by the same angle around the fold line
            # This ensures the entire region remains planar

            cos_a = math.cos(fold.angle)
            sin_a = math.sin(fold.angle)

            # Pure rotation: perp_dist rotates in the perp-up plane
            local_perp = perp_dist * cos_a
            local_up = perp_dist * sin_a

            # Final 3D position
            pos_3d = (
                fold_center_3d[0] + along * fold_axis_3d[0] + local_perp * fold_perp_3d[0] + local_up * up_3d[0],
                fold_center_3d[1] + along * fold_axis_3d[1] + local_perp * fold_perp_3d[1] + local_up * up_3d[1],
                fold_center_3d[2] + along * fold_axis_3d[2] + local_perp * fold_perp_3d[2] + local_up * up_3d[2]
            )

            # Update cumulative affine transformation for subsequent folds
            # New rotation: fold_rot @ rot
            fold_rot = _rotation_matrix_around_axis(fold_axis_3d, fold.angle)
            new_rot = _multiply_matrices(fold_rot, rot)

            # New origin: use fold center as reference point (it's on the rotation axis, so stays put)
            # We need: new_rot @ P + new_origin = correct_3d_position for any 2D point P
            # For fold_center: new_rot @ fold_center + new_origin = fold_center_3d
            rotated_fold_center = (
                new_rot[0][0] * fold.center[0] + new_rot[0][1] * fold.center[1],
                new_rot[1][0] * fold.center[0] + new_rot[1][1] * fold.center[1],
                new_rot[2][0] * fold.center[0] + new_rot[2][1] * fold.center[1]
            )
            origin = (
                fold_center_3d[0] - rotated_fold_center[0],
                fold_center_3d[1] - rotated_fold_center[1],
                fold_center_3d[2] - rotated_fold_center[2]
            )
            rot = new_rot

        elif classification == "IN_ZONE":
            # Point is in the bend zone - map to cylindrical arc
            dist_into_zone = max(0, min(perp_dist + hw, fold.zone_width))

            arc_fraction = dist_into_zone / fold.zone_width if fold.zone_width > 0 else 0
            theta = arc_fraction * fold.angle

            if abs(fold.angle) < 1e-9:
                local_perp = dist_into_zone - hw
                local_up = 0.0
            else:
                local_perp = R * math.sin(abs(theta)) - hw
                local_up = R * (1 - math.cos(abs(theta)))
                if fold.angle < 0:
                    local_up = -local_up

            pos_3d = (
                fold_center_3d[0] + along * fold_axis_3d[0] + local_perp * fold_perp_3d[0] + local_up * up_3d[0],
                fold_center_3d[1] + along * fold_axis_3d[1] + local_perp * fold_perp_3d[1] + local_up * up_3d[1],
                fold_center_3d[2] + along * fold_axis_3d[2] + local_perp * fold_perp_3d[2] + local_up * up_3d[2]
            )
            return pos_3d  # IN_ZONE is terminal

    return pos_3d


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
        # Transform fold axis through cumulative rotation
        fold_axis_3d = _apply_rotation(rot, (fold.axis[0], fold.axis[1], 0.0))

        if classification == "AFTER":
            fold_rot = _rotation_matrix_around_axis(fold_axis_3d, fold.angle)
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

            fold_rot = _rotation_matrix_around_axis(fold_axis_3d, theta)
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

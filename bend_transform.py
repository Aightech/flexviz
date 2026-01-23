"""
Bend Transformation Module

Transforms 2D PCB coordinates to 3D based on fold recipes from region subdivision.

================================================================================
OVERVIEW
================================================================================

The transformation is purely recipe-based: each region has a fold_recipe that
specifies exactly which folds affect it and how (IN_ZONE or AFTER).
No geometric re-classification is performed during transformation.

Recipe format: [(fold, classification, entered_from_back), ...]
  - fold: FoldDefinition with center, axis, zone_width, angle
  - classification: "IN_ZONE" (point is in bend zone) or "AFTER" (point is past bend)
  - entered_from_back: True if fold was entered from the AFTER side during BFS

================================================================================
FOLD GEOMETRY
================================================================================

A fold is defined by two parallel marking lines separated by zone_width.
The fold center is a reference point on the fold line (midway between markers).

Coordinate system relative to fold:
  - axis: unit vector along the fold line
  - perp: unit vector perpendicular to fold (direction of bending)
  - along: signed distance along the fold axis from center
  - perp_dist: signed distance perpendicular to fold from center
    * perp_dist < -hw: BEFORE the fold (hw = half_width = zone_width/2)
    * -hw <= perp_dist <= +hw: IN_ZONE (inside the bend)
    * perp_dist > +hw: AFTER the fold

The fold axis (rotation axis for the cylinder) is at the BEFORE boundary,
not at the fold center. This means:
  - At perp_dist = -hw: point is at the fold axis, no rotation yet
  - At perp_dist = +hw: point has rotated by the full fold angle

================================================================================
TRANSFORMATION MATH
================================================================================

1. IN_ZONE TRANSFORMATION (Cylindrical Mapping)
-----------------------------------------------
Points in the bend zone are mapped onto a cylindrical arc.

Given:
  - R = zone_width / |angle|  (bend radius)
  - dist_into_zone = perp_dist + hw  (distance from BEFORE boundary)
  - arc_fraction = dist_into_zone / zone_width
  - theta = arc_fraction * angle  (rotation angle at this point)

The point maps to local coordinates:
  - local_perp = R * sin(|theta|) - hw
  - local_up = R * (1 - cos(|theta|))  [negated if angle < 0]

At the BEFORE boundary (theta=0): local_perp = -hw, local_up = 0 (original position)
At the AFTER boundary (theta=angle): local_perp = R*sin(|angle|) - hw, local_up = R*(1-cos(|angle|))

2. AFTER TRANSFORMATION (Rotated Plane)
---------------------------------------
Points past the bend zone continue on a flat plane rotated by the fold angle.

The plane starts at the end of the IN_ZONE cylinder:
  - zone_end_perp = R * sin(|angle|) - hw
  - zone_end_up = R * (1 - cos(|angle|))

The excess distance beyond the zone is rotated:
  - excess = perp_dist - hw
  - local_perp = zone_end_perp + excess * cos(angle)
  - local_up = zone_end_up + excess * sin(angle)

3. CUMULATIVE TRANSFORMATION (Multiple Folds)
---------------------------------------------
For multiple folds, we track an affine transformation: pos_3d = rot @ pos_2d + origin

Each AFTER fold updates the cumulative transformation:
  - rot: 3x3 rotation matrix (composition of all fold rotations)
  - origin: 3D translation (accounts for cylindrical offsets)

The fold's local coordinate frame is transformed through the cumulative rotation:
  - fold_axis_3d = rot @ (axis.x, axis.y, 0)
  - fold_perp_3d = rot @ (perp.x, perp.y, 0)
  - up_3d = rot @ (0, 0, 1)

================================================================================
BACK ENTRY HANDLING
================================================================================

When BFS traverses the region graph, it may enter a fold from the "back" (AFTER side)
rather than the front (BEFORE side). This happens in multi-fold configurations where
the traversal path goes around the fold.

For back entry, we mirror the transformation:
  1. Negate perp_dist: measures distance from the opposite side
  2. Negate local_perp (IN_ZONE): cylinder axis is at AFTER boundary instead of BEFORE
  3. Negate zone_end_perp (AFTER): plane starts at the correct mirrored position

The fold angle is NOT negated - the bend direction remains consistent with the
physical fold angle, regardless of which side we entered from.

This ensures:
  - Continuity at IN_ZONE/AFTER boundary for back entry
  - Consistent bend direction (positive angle = bend up) regardless of entry side
  - Correct spatial positioning through perpendicular mirroring
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
# Format: [(fold, classification, entered_from_back), ...]
# - fold: FoldDefinition
# - classification: "IN_ZONE" or "AFTER"
# - entered_from_back: bool (optional, defaults to False)
FoldRecipe = list[tuple[FoldDefinition, str, bool]]


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

    for entry in recipe:
        # Recipe format: (fold, classification, entered_from_back) or (fold, classification)
        fold = entry[0]
        classification = entry[1]
        entered_from_back = entry[2] if len(entry) > 2 else False

        hw = fold.zone_width / 2
        R = fold.radius

        # Use ORIGINAL 2D coordinates to compute distances (preserved through bending)
        dx = point[0] - fold.center[0]
        dy = point[1] - fold.center[1]
        along = dx * fold.axis[0] + dy * fold.axis[1]
        perp_dist = dx * fold.perp[0] + dy * fold.perp[1]

        # When entered from back, the transformation is mirrored:
        # - Fold axis is at the opposite boundary (perp_dist = +hw instead of -hw)
        # - Distance into zone is measured from the opposite side
        # The angle sign is preserved (bend direction is consistent with fold angle)
        effective_angle = fold.angle
        if entered_from_back:
            perp_dist = -perp_dist
            effective_angle = effective_angle


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
            # Point is after this fold - first map to end of IN_ZONE (cylindrical),
            # then rotate the excess distance beyond the zone.
            # This ensures continuity with IN_ZONE at the boundary.

            # For back entry, use -angle for rotation direction
            # (point is geometrically BEFORE fold, on the un-rotated side)
            rotation_angle = -effective_angle+math.pi   if entered_from_back else effective_angle
            cos_a = math.cos(rotation_angle)
            sin_a = math.sin(rotation_angle)

            # Position at end of IN_ZONE (perp_dist = hw)
            zone_end_perp = R * math.sin(abs(effective_angle)) - hw
            zone_end_up = R * (1 - math.cos(abs(effective_angle)))
            if effective_angle < 0:
                zone_end_up = -zone_end_up

            # For back entry, mirror zone_end_perp to match IN_ZONE negation
            if entered_from_back:
                zone_end_perp = -zone_end_perp

            # Excess distance beyond the zone
            excess = perp_dist - hw

            # Rotate the excess in the rotated perp-up plane
            local_perp = zone_end_perp + excess * cos_a
            local_up = zone_end_up + excess * sin_a

            # Final 3D position
            pos_3d = (
                fold_center_3d[0] + along * fold_axis_3d[0] + local_perp * fold_perp_3d[0] + local_up * up_3d[0],
                fold_center_3d[1] + along * fold_axis_3d[1] + local_perp * fold_perp_3d[1] + local_up * up_3d[1],
                fold_center_3d[2] + along * fold_axis_3d[2] + local_perp * fold_perp_3d[2] + local_up * up_3d[2]
            )

            # Update cumulative affine transformation for subsequent folds
            # New rotation: fold_rot @ rot
            # For back entry, use -angle (consistent with compute_normal)
            fold_rot = _rotation_matrix_around_axis(fold_axis_3d, rotation_angle)
            new_rot = _multiply_matrices(fold_rot, rot)

            # New origin: The cylindrical offset means fold center also moves.
            # Compute where fold center (perp_dist=0) ends up:
            center_local_perp = zone_end_perp + (0 - hw) * cos_a
            center_local_up = zone_end_up + (0 - hw) * sin_a
            fold_center_transformed = (
                fold_center_3d[0] + center_local_perp * fold_perp_3d[0] + center_local_up * up_3d[0],
                fold_center_3d[1] + center_local_perp * fold_perp_3d[1] + center_local_up * up_3d[1],
                fold_center_3d[2] + center_local_perp * fold_perp_3d[2] + center_local_up * up_3d[2]
            )

            # We need: new_rot @ P + new_origin = correct_3d_position for any 2D point P
            # For fold_center: new_rot @ fold_center + new_origin = fold_center_transformed
            rotated_fold_center = (
                new_rot[0][0] * fold.center[0] + new_rot[0][1] * fold.center[1],
                new_rot[1][0] * fold.center[0] + new_rot[1][1] * fold.center[1],
                new_rot[2][0] * fold.center[0] + new_rot[2][1] * fold.center[1]
            )
            origin = (
                fold_center_transformed[0] - rotated_fold_center[0],
                fold_center_transformed[1] - rotated_fold_center[1],
                fold_center_transformed[2] - rotated_fold_center[2]
            )
            rot = new_rot

        elif classification == "IN_ZONE":
            # Point is in the bend zone - map to cylindrical arc
            dist_into_zone = max(0, min(perp_dist + hw, fold.zone_width))

            arc_fraction = dist_into_zone / fold.zone_width if fold.zone_width > 0 else 0
            theta = arc_fraction * effective_angle

            if abs(effective_angle) < 1e-9:
                local_perp = dist_into_zone - hw
                local_up = 0.0
            else:
                local_perp = R * math.sin(abs(theta)) - hw
                local_up = R * (1 - math.cos(abs(theta)))
                if effective_angle < 0:
                    local_up = -local_up

            # For back entry, mirror local_perp around 0
            # The cylinder axis is at AFTER boundary (+hw) instead of BEFORE (-hw)
            if entered_from_back:
                local_perp = -local_perp

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
        recipe: List of (FoldDefinition, classification, entered_from_back) tuples

    Returns:
        Unit normal vector (nx, ny, nz)
    """
    if not recipe:
        return (0.0, 0.0, 1.0)

    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for entry in recipe:
        fold = entry[0]
        classification = entry[1]
        entered_from_back = entry[2] if len(entry) > 2 else False

        # Transform fold axis through cumulative rotation
        fold_axis_3d = _apply_rotation(rot, (fold.axis[0], fold.axis[1], 0.0))

        if classification == "AFTER":
            if not entered_from_back:  
                fold_rot = _rotation_matrix_around_axis(fold_axis_3d, fold.angle)
            else:
                fold_rot = _rotation_matrix_around_axis(fold_axis_3d, -fold.angle)
            rot = _multiply_matrices(fold_rot, rot)

        elif classification == "IN_ZONE":
            dx = point[0] - fold.center[0]
            dy = point[1] - fold.center[1]
            perp_dist = dx * fold.perp[0] + dy * fold.perp[1]
            hw = fold.zone_width / 2

            # For back entry, negate perp_dist to measure from opposite side                                            
            if entered_from_back:                                                                                       
                perp_dist = -perp_dist

            # Normal depends on geometric position only, NOT entry direction.
            # The normal represents the physical "up" direction of the surface,
            # which is the same regardless of how we reached this point.
            dist_into_zone = perp_dist + hw
            dist_into_zone = max(0, min(dist_into_zone, fold.zone_width))

            arc_fraction = dist_into_zone / fold.zone_width if fold.zone_width > 0 else 0
            theta = arc_fraction * fold.angle

            # For back entry, negate theta to flip normal direction                                                     
            # (cylinder axis is on opposite side, so outward direction is reversed)                                     
            if entered_from_back:                                                                                       
               theta = -theta 

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

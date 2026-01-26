"""
Validation module for flex PCB design rules.

Checks for:
- Fold lines crossing stiffener regions
- Minimum bend radius violations
- Components in bend zones
"""

from dataclasses import dataclass, field
from typing import Optional
import math

try:
    from .markers import FoldMarker
    from .stiffener import StiffenerRegion, segment_intersects_polygon, point_in_polygon
    from .geometry import BoardGeometry, ComponentGeometry, BoundingBox
    from .config import FlexConfig
except ImportError:
    from markers import FoldMarker
    from stiffener import StiffenerRegion, segment_intersects_polygon, point_in_polygon
    from geometry import BoardGeometry, ComponentGeometry, BoundingBox
    from config import FlexConfig


@dataclass
class ValidationWarning:
    """A single validation warning."""
    category: str  # "stiffener", "bend_radius", "component"
    severity: str  # "error", "warning", "info"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Results of all validation checks."""
    warnings: list[ValidationWarning] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(w.severity == "error" for w in self.warnings)

    @property
    def has_warnings(self) -> bool:
        return any(w.severity == "warning" for w in self.warnings)

    @property
    def error_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "warning")

    def get_by_category(self, category: str) -> list[ValidationWarning]:
        return [w for w in self.warnings if w.category == category]


def _extend_fold_line(marker: FoldMarker, extent: float = 1000.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Get extended fold line endpoints for intersection testing.

    The fold line runs through the center perpendicular to the fold axis.
    """
    # Fold axis is along the fold lines, so perpendicular is the fold direction
    ax, ay = marker.axis
    cx, cy = marker.center

    # Extend in both directions along the fold axis
    p1 = (cx - ax * extent, cy - ay * extent)
    p2 = (cx + ax * extent, cy + ay * extent)

    return p1, p2


def _get_bend_zone_polygon(marker: FoldMarker) -> list[tuple[float, float]]:
    """
    Get the bend zone as a polygon (rectangle between the two fold lines).
    """
    # Use the actual line endpoints to form the bend zone rectangle
    return [
        marker.line_a_start,
        marker.line_a_end,
        marker.line_b_end,
        marker.line_b_start,
    ]


def _point_in_bend_zone(point: tuple[float, float], marker: FoldMarker) -> bool:
    """Check if a point is within the bend zone."""
    bend_zone = _get_bend_zone_polygon(marker)
    return point_in_polygon(point, bend_zone)


def _bbox_overlaps_bend_zone(bbox: BoundingBox, marker: FoldMarker) -> bool:
    """Check if a bounding box overlaps the bend zone."""
    bend_zone = _get_bend_zone_polygon(marker)

    # Check if any corner of the bbox is in the bend zone
    corners = [
        (bbox.min_x, bbox.min_y),
        (bbox.max_x, bbox.min_y),
        (bbox.max_x, bbox.max_y),
        (bbox.min_x, bbox.max_y),
    ]

    for corner in corners:
        if point_in_polygon(corner, bend_zone):
            return True

    # Check if any edge of the bend zone crosses the bbox
    for i in range(len(bend_zone)):
        j = (i + 1) % len(bend_zone)
        p1, p2 = bend_zone[i], bend_zone[j]

        # Check intersection with bbox edges
        bbox_edges = [
            ((bbox.min_x, bbox.min_y), (bbox.max_x, bbox.min_y)),
            ((bbox.max_x, bbox.min_y), (bbox.max_x, bbox.max_y)),
            ((bbox.max_x, bbox.max_y), (bbox.min_x, bbox.max_y)),
            ((bbox.min_x, bbox.max_y), (bbox.min_x, bbox.min_y)),
        ]

        for edge_p1, edge_p2 in bbox_edges:
            if _segments_intersect(p1, p2, edge_p1, edge_p2):
                return True

    # Check if bend zone is entirely inside bbox
    if (bbox.min_x <= bend_zone[0][0] <= bbox.max_x and
        bbox.min_y <= bend_zone[0][1] <= bbox.max_y):
        return True

    return False


def _segments_intersect(p1: tuple[float, float], p2: tuple[float, float],
                        p3: tuple[float, float], p4: tuple[float, float]) -> bool:
    """Check if two line segments intersect."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def check_fold_stiffener_conflicts(
    markers: list[FoldMarker],
    stiffeners: list[StiffenerRegion]
) -> list[ValidationWarning]:
    """
    Check if any fold line crosses a stiffener region.

    Stiffeners are rigid and cannot be bent, so fold lines must not pass through them.
    """
    warnings = []

    for i, marker in enumerate(markers):
        fold_name = f"Fold {i + 1}"

        # Get the two fold lines
        line_a = (marker.line_a_start, marker.line_a_end)
        line_b = (marker.line_b_start, marker.line_b_end)

        for j, stiffener in enumerate(stiffeners):
            stiff_name = f"Stiffener {j + 1} ({stiffener.side})"

            # Check if either fold line intersects the stiffener
            line_a_intersects = segment_intersects_polygon(line_a[0], line_a[1], stiffener.outline)
            line_b_intersects = segment_intersects_polygon(line_b[0], line_b[1], stiffener.outline)

            # Also check if fold center is inside stiffener
            center_inside = point_in_polygon(marker.center, stiffener.outline)

            if line_a_intersects or line_b_intersects or center_inside:
                warnings.append(ValidationWarning(
                    category="stiffener",
                    severity="error",
                    message=f"{fold_name} crosses {stiff_name}",
                    details={
                        "fold_index": i,
                        "stiffener_index": j,
                        "stiffener_side": stiffener.side,
                        "fold_center": marker.center,
                    }
                ))

    return warnings


def check_bend_radius(
    markers: list[FoldMarker],
    config: FlexConfig
) -> list[ValidationWarning]:
    """
    Check if bend radius meets minimum requirements.

    Minimum bend radius = min_bend_radius_factor × flex_thickness
    """
    warnings = []
    min_radius = config.min_bend_radius

    for i, marker in enumerate(markers):
        fold_name = f"Fold {i + 1}"

        # Skip if angle is 0 (no actual bend)
        if abs(marker.angle_degrees) < 0.1:
            continue

        actual_radius = marker.radius

        if actual_radius < min_radius:
            ratio = actual_radius / config.flex_thickness if config.flex_thickness > 0 else 0
            severity = "error" if actual_radius < min_radius * 0.5 else "warning"

            warnings.append(ValidationWarning(
                category="bend_radius",
                severity=severity,
                message=f"{fold_name}: radius {actual_radius:.2f}mm < min {min_radius:.2f}mm",
                details={
                    "fold_index": i,
                    "actual_radius": actual_radius,
                    "min_radius": min_radius,
                    "radius_factor": ratio,
                    "min_factor": config.min_bend_radius_factor,
                    "flex_thickness": config.flex_thickness,
                }
            ))
        elif actual_radius < min_radius * 1.5:
            # Close to minimum - info level
            warnings.append(ValidationWarning(
                category="bend_radius",
                severity="info",
                message=f"{fold_name}: radius {actual_radius:.2f}mm near minimum",
                details={
                    "fold_index": i,
                    "actual_radius": actual_radius,
                    "min_radius": min_radius,
                }
            ))

    return warnings


def check_components_in_bend_zones(
    markers: list[FoldMarker],
    board: BoardGeometry
) -> list[ValidationWarning]:
    """
    Check if any components are placed in bend zones.

    Components in bend zones can cause:
    - Solder joint stress and cracking
    - Component damage during bending
    - Interference with the bend
    """
    warnings = []

    for i, marker in enumerate(markers):
        fold_name = f"Fold {i + 1}"

        for comp in board.components:
            if _bbox_overlaps_bend_zone(comp.bounding_box, marker):
                warnings.append(ValidationWarning(
                    category="component",
                    severity="warning",
                    message=f"{comp.reference} in {fold_name} bend zone",
                    details={
                        "fold_index": i,
                        "component_ref": comp.reference,
                        "component_value": comp.value,
                        "component_center": comp.center,
                        "component_layer": comp.layer,
                    }
                ))

    return warnings


def validate_design(
    markers: list[FoldMarker],
    board: BoardGeometry,
    stiffeners: list[StiffenerRegion],
    config: FlexConfig
) -> ValidationResult:
    """
    Run all validation checks on the flex PCB design.

    Args:
        markers: List of fold markers
        board: Board geometry with components
        stiffeners: List of stiffener regions
        config: Flex configuration

    Returns:
        ValidationResult with all warnings
    """
    result = ValidationResult()

    # Check fold-stiffener conflicts
    result.warnings.extend(check_fold_stiffener_conflicts(markers, stiffeners))

    # Check bend radius
    result.warnings.extend(check_bend_radius(markers, config))

    # Check components in bend zones
    result.warnings.extend(check_components_in_bend_zones(markers, board))

    return result


def get_fold_radius_status(marker: FoldMarker, config: FlexConfig) -> str:
    """
    Get the status color for a fold's bend radius.

    Returns: "green", "yellow", or "red"
    """
    if abs(marker.angle_degrees) < 0.1:
        return "green"  # No bend

    min_radius = config.min_bend_radius
    actual_radius = marker.radius

    if actual_radius >= min_radius:
        return "green"
    elif actual_radius >= min_radius * 0.5:
        return "yellow"
    else:
        return "red"

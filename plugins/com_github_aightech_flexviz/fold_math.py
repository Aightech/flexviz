"""
Pure-math fold marker geometry computation.

No KiCad (pcbnew) or IPC (kipy) dependencies — shared by both SWIG and IPC paths.
All coordinates are in the caller's unit system (typically mm for IPC, IU for SWIG).
"""

import math
from dataclasses import dataclass


@dataclass
class FoldMarkerGeometry:
    """Computed fold marker positions. All coordinates in caller's units."""
    # Two parallel boundary lines
    line1_start: tuple[float, float]
    line1_end: tuple[float, float]
    line2_start: tuple[float, float]
    line2_end: tuple[float, float]
    # Dimension annotation
    dim_start: tuple[float, float]  # Fold midpoint
    dim_end: tuple[float, float]    # Text position
    angle_text: str                 # e.g., "90°"
    # Fallback text annotation position
    text_pos: tuple[float, float]
    text_angle_deg: float           # Rotation angle for text
    # Derived values
    zone_width: float
    perp: tuple[float, float]       # Unit perpendicular vector


def compute_fold_marker_geometry(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    angle_degrees: float,
    radius: float,
    line_spacing_factor: float = 1.0,
) -> FoldMarkerGeometry | None:
    """
    Compute fold marker line and annotation positions.

    All inputs and outputs are in the same unit system (mm, IU, nm — whatever
    the caller uses). No unit conversion is performed here.

    Args:
        point_a: First endpoint of fold centerline
        point_b: Second endpoint of fold centerline
        angle_degrees: Bend angle in degrees
        radius: Bend radius (same units as points)
        line_spacing_factor: Extra scale for offset (default 1.0)

    Returns:
        FoldMarkerGeometry, or None if the line is degenerate
    """
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1e-9:
        return None

    # Unit perpendicular vector
    perp_x = -dy / length
    perp_y = dx / length

    # Zone width = R * |angle_rad|
    angle_rad = math.radians(abs(angle_degrees))
    zone_width = radius * angle_rad if angle_rad > 0.001 else radius * 0.001

    # Offset for parallel lines (half zone width)
    offset = (zone_width / 2) * line_spacing_factor

    # Two parallel lines offset from centerline
    line1_start = (point_a[0] + perp_x * offset, point_a[1] + perp_y * offset)
    line1_end = (point_b[0] + perp_x * offset, point_b[1] + perp_y * offset)
    line2_start = (point_a[0] - perp_x * offset, point_a[1] - perp_y * offset)
    line2_end = (point_b[0] - perp_x * offset, point_b[1] - perp_y * offset)

    # Dimension annotation: from fold midpoint outward
    mid_x = (point_a[0] + point_b[0]) / 2
    mid_y = (point_a[1] + point_b[1]) / 2
    dim_start = (mid_x, mid_y)
    dim_end = (mid_x + dx, mid_y + dy)

    angle_text = f"{angle_degrees:.0f}°"

    # Fallback text position (offset perpendicular from midpoint)
    text_offset = offset + length * 0.05  # Small extra spacing
    text_pos = (mid_x + perp_x * text_offset, mid_y + perp_y * text_offset)
    text_angle_deg = math.degrees(math.atan2(dy, dx))

    return FoldMarkerGeometry(
        line1_start=line1_start,
        line1_end=line1_end,
        line2_start=line2_start,
        line2_end=line2_end,
        dim_start=dim_start,
        dim_end=dim_end,
        angle_text=angle_text,
        text_pos=text_pos,
        text_angle_deg=text_angle_deg,
        zone_width=zone_width,
        perp=(perp_x, perp_y),
    )

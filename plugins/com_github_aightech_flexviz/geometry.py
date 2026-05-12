"""
Geometry extraction from KiCad PCB.

Extracts and organizes board geometry into structures suitable for
3D rendering and bend transformation.
"""

from dataclasses import dataclass, field
from typing import Optional
import math

try:
    from .kicad_parser import KiCadPCB, TraceSegment, Footprint, Pad, CircleCutout
except ImportError:
    from kicad_parser import KiCadPCB, TraceSegment, Footprint, Pad, CircleCutout


@dataclass
class Point2D:
    """A 2D point."""
    x: float
    y: float

    def __iter__(self):
        yield self.x
        yield self.y

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Point3D:
    """A 3D point."""
    x: float
    y: float
    z: float

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def expand(self, margin: float) -> 'BoundingBox':
        return BoundingBox(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin
        )


@dataclass
class OutlineSegment:
    """A segment of an outline that can be either a line or an arc."""
    type: str  # "line" or "arc"
    start: tuple[float, float]
    end: tuple[float, float]
    # For arcs only:
    center: tuple[float, float] = None
    radius: float = 0.0
    mid: tuple[float, float] = None  # Mid point on arc (for 3-point arc definition)


@dataclass
class Polygon:
    """A 2D polygon defined by vertices, optionally with arc segment info."""
    vertices: list[tuple[float, float]]
    segments: list[OutlineSegment] = field(default_factory=list)  # Optional segment info with arcs

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, index):
        return self.vertices[index]

    def __iter__(self):
        return iter(self.vertices)

    @property
    def bounding_box(self) -> BoundingBox:
        if not self.vertices:
            return BoundingBox(0, 0, 0, 0)
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.vertices:
            return (0, 0)
        x = sum(v[0] for v in self.vertices) / len(self.vertices)
        y = sum(v[1] for v in self.vertices) / len(self.vertices)
        return (x, y)

    def edges(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return list of edges as (start, end) point tuples."""
        if len(self.vertices) < 2:
            return []
        edges = []
        for i in range(len(self.vertices)):
            j = (i + 1) % len(self.vertices)
            edges.append((self.vertices[i], self.vertices[j]))
        return edges


@dataclass
class LineSegment:
    """A line segment with optional width."""
    start: tuple[float, float]
    end: tuple[float, float]
    width: float = 0.1
    layer: str = "F.Cu"  # Copper layer (F.Cu or B.Cu)

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def midpoint(self) -> tuple[float, float]:
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2
        )

    @property
    def angle(self) -> float:
        """Angle in radians."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.atan2(dy, dx)


@dataclass
class PadGeometry:
    """Geometry for a component pad."""
    center: tuple[float, float]
    shape: str  # 'circle', 'rect', 'oval', 'roundrect'
    size: tuple[float, float]  # (width, height)
    angle: float = 0  # rotation in degrees
    drill: float = 0  # drill hole diameter (0 for SMD)
    layer: str = "F.Cu"  # Primary layer (F.Cu or B.Cu)


@dataclass
class Model3DGeometry:
    """3D model reference for a component."""
    path: str
    offset: tuple[float, float, float] = (0, 0, 0)
    scale: tuple[float, float, float] = (1, 1, 1)
    rotate: tuple[float, float, float] = (0, 0, 0)
    hide: bool = False


@dataclass
class ComponentGeometry:
    """Geometry for a component (footprint)."""
    reference: str
    value: str
    center: tuple[float, float]
    angle: float  # rotation in degrees
    bounding_box: BoundingBox
    pads: list[PadGeometry]
    layer: str
    model_path: str = ""
    models: list[Model3DGeometry] = field(default_factory=list)


@dataclass
class BoardGeometry:
    """
    Complete board geometry extracted from KiCad PCB.

    Contains all geometric elements needed for 3D visualization.
    """
    # Board outline (main polygon)
    outline: Polygon

    # Board thickness in mm
    thickness: float

    # Copper traces by layer
    traces: dict[str, list[LineSegment]] = field(default_factory=dict)

    # Components
    components: list[ComponentGeometry] = field(default_factory=list)

    # Cutouts (holes in the board)
    cutouts: list[Polygon] = field(default_factory=list)

    @property
    def bounding_box(self) -> BoundingBox:
        return self.outline.bounding_box

    @property
    def all_traces(self) -> list[LineSegment]:
        """All traces across all layers."""
        result = []
        for layer_traces in self.traces.values():
            result.extend(layer_traces)
        return result

    @property
    def all_pads(self) -> list[PadGeometry]:
        """All pads from all components."""
        result = []
        for comp in self.components:
            result.extend(comp.pads)
        return result


def extract_geometry(pcb: KiCadPCB) -> BoardGeometry:
    """
    Extract all geometry from a KiCad PCB.

    Args:
        pcb: Parsed KiCad PCB

    Returns:
        BoardGeometry with outline, traces, components, etc.
    """
    # Get board info
    board_info = pcb.get_board_info()

    # Get board outline with arc info
    outline_points, outline_segs = pcb.get_board_outline_with_arcs()

    # Convert segment dicts to OutlineSegment objects
    segments = []
    for seg in outline_segs:
        segments.append(OutlineSegment(
            type=seg['type'],
            start=seg['start'],
            end=seg['end'],
            center=seg.get('center'),
            radius=seg.get('radius', 0.0),
            mid=seg.get('mid')
        ))

    outline = Polygon(outline_points, segments) if outline_points else Polygon([], [])

    # Get traces by layer
    traces = {}
    all_traces = pcb.get_traces()
    for trace in all_traces:
        layer = trace.layer
        if layer not in traces:
            traces[layer] = []
        traces[layer].append(LineSegment(
            start=(trace.start_x, trace.start_y),
            end=(trace.end_x, trace.end_y),
            width=trace.width,
            layer=layer
        ))

    # Get components
    components = []
    for fp in pcb.get_footprints():
        # Calculate bounding box from pads (with footprint rotation applied)
        # KiCad uses Y-down coords: negate angle for standard math rotation
        if fp.pads:
            angle_rad = math.radians(-fp.at_angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            pad_xs = [p.at_x * cos_a - p.at_y * sin_a + fp.at_x for p in fp.pads]
            pad_ys = [p.at_x * sin_a + p.at_y * cos_a + fp.at_y for p in fp.pads]
            pad_sizes = [max(p.size_x, p.size_y) / 2 for p in fp.pads]

            min_x = min(x - s for x, s in zip(pad_xs, pad_sizes))
            max_x = max(x + s for x, s in zip(pad_xs, pad_sizes))
            min_y = min(y - s for y, s in zip(pad_ys, pad_sizes))
            max_y = max(y + s for y, s in zip(pad_ys, pad_sizes))
            bbox = BoundingBox(min_x, min_y, max_x, max_y)
        else:
            # No pads, create small bounding box at center
            bbox = BoundingBox(
                fp.at_x - 1, fp.at_y - 1,
                fp.at_x + 1, fp.at_y + 1
            )

        # Convert pads
        pads = []
        for p in fp.pads:
            # Transform pad position by component position and rotation
            # KiCad uses Y-down coords: negate angle for standard math rotation
            angle_rad = math.radians(-fp.at_angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Rotate pad position around component center
            px = p.at_x * cos_a - p.at_y * sin_a + fp.at_x
            py = p.at_x * sin_a + p.at_y * cos_a + fp.at_y

            # Determine pad layer from pad's layers list or component layer
            # Through-hole pads use "*.Cu" wildcard matching all copper layers
            pad_layer = fp.layer
            if p.layers:
                has_all_cu = any(l == "*.Cu" for l in p.layers)
                has_f_cu = "F.Cu" in p.layers or has_all_cu
                has_b_cu = "B.Cu" in p.layers or has_all_cu
                if has_b_cu and not has_f_cu:
                    pad_layer = "B.Cu"
                elif has_f_cu:
                    pad_layer = "F.Cu"

            pads.append(PadGeometry(
                center=(px, py),
                shape=p.shape,
                size=(p.size_x, p.size_y),
                angle=-p.at_angle,  # File stores global angle; negate for Y-down
                drill=p.drill,
                layer=pad_layer
            ))

        # Convert 3D models
        models = [
            Model3DGeometry(
                path=m.path,
                offset=m.offset,
                scale=m.scale,
                rotate=m.rotate,
                hide=m.hide
            )
            for m in fp.models
        ]

        components.append(ComponentGeometry(
            reference=fp.reference,
            value=fp.value,
            center=(fp.at_x, fp.at_y),
            angle=fp.at_angle,
            bounding_box=bbox,
            pads=pads,
            layer=fp.layer,
            model_path=fp.model_path,
            models=models
        ))

    # Get cutouts from Edge.Cuts
    cutouts = []

    # Circle cutouts (gr_circle elements)
    for circle in pcb.get_circle_cutouts():
        cutout_poly = circle_to_polygon(
            circle.center_x,
            circle.center_y,
            circle.radius
        )
        cutouts.append(cutout_poly)

    # Polygon cutouts (closed polygons from gr_line/gr_arc that aren't the outline)
    for poly_verts in pcb.get_polygon_cutouts():
        cutouts.append(Polygon(poly_verts))

    # Drill holes as cutouts (through-hole pads and mounting holes)
    for hole in pcb.get_drill_holes():
        hole_poly = circle_to_polygon(
            hole.center_x,
            hole.center_y,
            hole.diameter / 2  # radius = diameter / 2
        )
        cutouts.append(hole_poly)

    return BoardGeometry(
        outline=outline,
        thickness=board_info.thickness,
        traces=traces,
        components=components,
        cutouts=cutouts
    )


def subdivide_polygon(polygon: Polygon, max_edge_length: float) -> Polygon:
    """
    Subdivide polygon edges that exceed max_edge_length.

    Useful for smoother bending of long edges.
    """
    if not polygon.vertices:
        return polygon

    new_vertices = []

    for i in range(len(polygon.vertices)):
        v1 = polygon.vertices[i]
        v2 = polygon.vertices[(i + 1) % len(polygon.vertices)]

        new_vertices.append(v1)

        # Calculate edge length
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length > max_edge_length:
            # Subdivide this edge
            num_segments = math.ceil(length / max_edge_length)
            for j in range(1, num_segments):
                t = j / num_segments
                new_vertices.append((
                    v1[0] + t * dx,
                    v1[1] + t * dy
                ))

    return Polygon(new_vertices)


def offset_polygon(polygon: Polygon, offset: float) -> Polygon:
    """
    Create an offset (parallel) polygon.

    Positive offset expands outward, negative shrinks inward.
    Simple implementation using vertex normal averaging.
    """
    if len(polygon.vertices) < 3:
        return polygon

    n = len(polygon.vertices)
    new_vertices = []

    for i in range(n):
        # Get adjacent vertices
        v_prev = polygon.vertices[(i - 1) % n]
        v_curr = polygon.vertices[i]
        v_next = polygon.vertices[(i + 1) % n]

        # Calculate edge normals
        e1 = (v_curr[0] - v_prev[0], v_curr[1] - v_prev[1])
        e2 = (v_next[0] - v_curr[0], v_next[1] - v_curr[1])

        # Normal is perpendicular to edge (rotated 90 degrees)
        n1 = (-e1[1], e1[0])
        n2 = (-e2[1], e2[0])

        # Normalize
        len1 = math.sqrt(n1[0] ** 2 + n1[1] ** 2)
        len2 = math.sqrt(n2[0] ** 2 + n2[1] ** 2)

        if len1 > 1e-10:
            n1 = (n1[0] / len1, n1[1] / len1)
        if len2 > 1e-10:
            n2 = (n2[0] / len2, n2[1] / len2)

        # Average normals
        avg_n = ((n1[0] + n2[0]) / 2, (n1[1] + n2[1]) / 2)
        avg_len = math.sqrt(avg_n[0] ** 2 + avg_n[1] ** 2)

        if avg_len > 1e-10:
            avg_n = (avg_n[0] / avg_len, avg_n[1] / avg_len)

        # Offset vertex
        new_vertices.append((
            v_curr[0] + offset * avg_n[0],
            v_curr[1] + offset * avg_n[1]
        ))

    return Polygon(new_vertices)


def line_segment_to_ribbon(segment: LineSegment) -> Polygon:
    """
    Convert a line segment with width to a ribbon polygon.

    Returns a 4-vertex polygon representing the trace ribbon.
    """
    dx = segment.end[0] - segment.start[0]
    dy = segment.end[1] - segment.start[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1e-10:
        # Degenerate segment, return small square
        hw = segment.width / 2
        cx, cy = segment.start
        return Polygon([
            (cx - hw, cy - hw),
            (cx + hw, cy - hw),
            (cx + hw, cy + hw),
            (cx - hw, cy + hw)
        ])

    # Perpendicular unit vector
    px = -dy / length
    py = dx / length

    hw = segment.width / 2

    return Polygon([
        (segment.start[0] + px * hw, segment.start[1] + py * hw),
        (segment.end[0] + px * hw, segment.end[1] + py * hw),
        (segment.end[0] - px * hw, segment.end[1] - py * hw),
        (segment.start[0] - px * hw, segment.start[1] - py * hw)
    ])


def pad_to_polygon(pad: PadGeometry) -> Polygon:
    """
    Convert a pad geometry to a polygon approximation.
    """
    cx, cy = pad.center
    hw, hh = pad.size[0] / 2, pad.size[1] / 2
    angle_rad = math.radians(pad.angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    def rotate(x, y):
        return (
            cx + x * cos_a - y * sin_a,
            cy + x * sin_a + y * cos_a
        )

    if pad.shape == 'circle':
        # Approximate circle with 16-gon
        n = 16
        r = max(hw, hh)
        vertices = []
        for i in range(n):
            theta = 2 * math.pi * i / n
            vertices.append(rotate(r * math.cos(theta), r * math.sin(theta)))
        return Polygon(vertices)

    elif pad.shape in ('rect', 'roundrect'):
        # Rectangle (roundrect corners ignored for simplicity)
        return Polygon([
            rotate(-hw, -hh),
            rotate(hw, -hh),
            rotate(hw, hh),
            rotate(-hw, hh)
        ])

    elif pad.shape == 'oval':
        # Approximate oval with polygon
        n = 16
        vertices = []
        for i in range(n):
            theta = 2 * math.pi * i / n
            vertices.append(rotate(hw * math.cos(theta), hh * math.sin(theta)))
        return Polygon(vertices)

    else:
        # Unknown shape, use rectangle
        return Polygon([
            rotate(-hw, -hh),
            rotate(hw, -hh),
            rotate(hw, hh),
            rotate(-hw, hh)
        ])


def component_to_box(comp: ComponentGeometry) -> Polygon:
    """
    Get the bounding box polygon for a component.
    """
    bbox = comp.bounding_box
    return Polygon([
        (bbox.min_x, bbox.min_y),
        (bbox.max_x, bbox.min_y),
        (bbox.max_x, bbox.max_y),
        (bbox.min_x, bbox.max_y)
    ])


def circle_to_polygon(center_x: float, center_y: float, radius: float, segments: int = 0) -> Polygon:
    """
    Convert a circle to a polygon approximation.

    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center
        radius: Circle radius
        segments: Number of segments. 0 (default) = adaptive based on radius
                  with ~2mm max segment length, minimum 8 segments.

    Returns:
        Polygon approximation of the circle
    """
    if segments <= 0:
        # Adaptive: ~2mm max chord length, minimum 8 segments
        circumference = 2 * math.pi * radius
        segments = max(8, int(math.ceil(circumference / 2.0)))
    vertices = []
    for i in range(segments):
        theta = 2 * math.pi * i / segments
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        vertices.append((x, y))
    return Polygon(vertices)


# ============================================================================
# Fillet bending — adaptive arc subdivision near fold zones
# ============================================================================

def arc_crosses_fold_zone(
    arc_seg: OutlineSegment,
    marker_center: tuple[float, float],
    marker_axis: tuple[float, float],
    marker_zone_width: float,
) -> bool:
    """
    Check if an arc outline segment crosses or enters a fold zone.

    The fold zone is a strip of width *marker_zone_width* centered on
    *marker_center*, extending perpendicular to *marker_axis*.

    We sample several points on the arc (start, mid, end, and two
    quarter-points if center/radius are known) and check whether any of
    them falls inside the fold zone (with 50 % margin).

    Args:
        arc_seg: An OutlineSegment of type "arc".
        marker_center: (x, y) centre of the fold marker.
        marker_axis: Unit vector along the fold line.
        marker_zone_width: Distance between the two fold lines.

    Returns:
        True if the arc crosses or enters the fold zone.
    """
    hw = marker_zone_width / 2
    perp = (-marker_axis[1], marker_axis[0])

    # Collect sample points
    sample_points = [arc_seg.start, arc_seg.end]
    if arc_seg.mid is not None:
        sample_points.append(arc_seg.mid)

    # If centre/radius are known, add quarter-arc points for better coverage
    if arc_seg.center is not None and arc_seg.radius > 0 and arc_seg.mid is not None:
        cx, cy = arc_seg.center
        r = arc_seg.radius

        start_angle = math.atan2(arc_seg.start[1] - cy, arc_seg.start[0] - cx)
        mid_angle = math.atan2(arc_seg.mid[1] - cy, arc_seg.mid[0] - cx)
        end_angle = math.atan2(arc_seg.end[1] - cy, arc_seg.end[0] - cx)

        def _norm(a):
            while a < 0:
                a += 2 * math.pi
            while a >= 2 * math.pi:
                a -= 2 * math.pi
            return a

        def _angle_between(a1, a2):
            d = a2 - a1
            while d < 0:
                d += 2 * math.pi
            while d >= 2 * math.pi:
                d -= 2 * math.pi
            return d

        sa = _norm(start_angle)
        ma = _norm(mid_angle)
        ea = _norm(end_angle)

        ccw_to_mid = _angle_between(sa, ma)
        ccw_to_end = _angle_between(sa, ea)

        if ccw_to_mid < ccw_to_end:
            sweep = ccw_to_end
            direction = 1
        else:
            sweep = 2 * math.pi - ccw_to_end
            direction = -1

        for frac in (0.25, 0.75):
            angle = sa + direction * sweep * frac
            sample_points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    # Check each sample point against the fold zone
    for pt in sample_points:
        if pt is None:
            continue
        dx = pt[0] - marker_center[0]
        dy = pt[1] - marker_center[1]
        perp_dist = abs(dx * perp[0] + dy * perp[1])
        if perp_dist <= hw * 1.5:  # 50 % margin
            return True

    return False


def _refine_arc_segment(
    seg: OutlineSegment,
    max_seg_length: float = 0.25,
    max_sagitta: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Re-linearize an arc OutlineSegment at a finer resolution.

    Returns a list of (x, y) points along the arc from *seg.start* to
    *seg.end* (inclusive).  If the arc geometry is degenerate, falls back
    to [start, end].

    Args:
        seg: An OutlineSegment with type "arc", center, radius, and mid set.
        max_seg_length: Maximum chord length between consecutive points
            (used when max_sagitta is 0).
        max_sagitta: When > 0, choose subdivision so each chord deflects from
            the true arc by at most this many mm. Takes precedence over
            max_seg_length.

    Returns:
        List of (x, y) tuples along the arc.
    """
    if seg.center is None or seg.radius <= 0 or seg.mid is None:
        return [seg.start, seg.end]

    cx, cy = seg.center
    r = seg.radius

    # Compute angles
    start_angle = math.atan2(seg.start[1] - cy, seg.start[0] - cx)
    mid_angle = math.atan2(seg.mid[1] - cy, seg.mid[0] - cx)
    end_angle = math.atan2(seg.end[1] - cy, seg.end[0] - cx)

    def _norm(a):
        while a < 0:
            a += 2 * math.pi
        while a >= 2 * math.pi:
            a -= 2 * math.pi
        return a

    def _angle_between(a1, a2):
        d = a2 - a1
        while d < 0:
            d += 2 * math.pi
        while d >= 2 * math.pi:
            d -= 2 * math.pi
        return d

    sa = _norm(start_angle)
    ma = _norm(mid_angle)
    ea = _norm(end_angle)

    ccw_to_mid = _angle_between(sa, ma)
    ccw_to_end = _angle_between(sa, ea)

    if ccw_to_mid < ccw_to_end:
        sweep = ccw_to_end
        direction = 1
    else:
        sweep = 2 * math.pi - ccw_to_end
        direction = -1

    if max_sagitta > 0 and r > 0 and max_sagitta < r:
        # Adaptive: angle per chord such that sagitta ≤ max_sagitta.
        theta_per_seg = 2.0 * math.acos(1.0 - max_sagitta / r)
        n_segments = max(4, int(math.ceil(sweep / theta_per_seg)))
    else:
        arc_length = r * sweep
        n_segments = max(4, int(math.ceil(arc_length / max_seg_length)))

    points = []
    for i in range(n_segments + 1):
        t = i / n_segments
        angle = sa + direction * sweep * t
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    return points


def refine_outline_for_folds(
    outline: Polygon,
    fold_markers: list,
    fine_max_seg_length: float = 0.25,
    base_sagitta: float = 0.05,
) -> Polygon:
    """
    Re-linearize arc segments at appropriate resolution for rendering.

    Each arc segment is tessellated:
    - At ``fine_max_seg_length`` chord (default 0.25 mm) if it crosses any
      fold zone — the bend needs many subdivisions for smooth deformation.
    - At ``base_sagitta`` adaptive sagitta (default 0.05 mm) otherwise —
      enough density for smooth rendering at any radius without bloating
      the vertex count.

    Without re-tessellating non-crossing arcs we would lose the dense
    samples produced by the parser (since this function rebuilds the
    vertex list from .segments).

    Args:
        outline: Board outline Polygon with .segments populated.
        fold_markers: List of FoldMarker objects (need .center, .axis,
            .zone_width attributes).
        fine_max_seg_length: Chord length for arcs crossing fold zones (mm).
        base_sagitta: Target sagitta for arcs not crossing folds (mm).

    Returns:
        A new Polygon with arc segments tessellated. If there are no arcs
        the original Polygon is returned unchanged.
    """
    if not outline.segments:
        return outline

    fold_zones = [
        (m.center, m.axis, m.zone_width) for m in fold_markers
    ] if fold_markers else []

    new_vertices = []
    changed = False

    for seg in outline.segments:
        if seg.type == "arc" and seg.center is not None and seg.mid is not None:
            crosses = any(
                arc_crosses_fold_zone(seg, fz[0], fz[1], fz[2])
                for fz in fold_zones
            )
            if crosses:
                arc_pts = _refine_arc_segment(seg, max_seg_length=fine_max_seg_length)
            else:
                arc_pts = _refine_arc_segment(seg, max_sagitta=base_sagitta)
            # Add all points except the last (will be the start of the next segment)
            new_vertices.extend(arc_pts[:-1])
            changed = True
            continue

        # Line segments contribute only their start vertex (end is the
        # start of the next segment).
        new_vertices.append(seg.start)

    if not changed:
        return outline

    return Polygon(new_vertices, outline.segments)

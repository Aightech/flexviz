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
class Polygon:
    """A 2D polygon defined by vertices."""
    vertices: list[tuple[float, float]]

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

    # Get board outline
    outline_points = pcb.get_board_outline()
    outline = Polygon(outline_points) if outline_points else Polygon([])

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
        # Calculate bounding box from pads
        if fp.pads:
            pad_xs = [fp.at_x + p.at_x for p in fp.pads]
            pad_ys = [fp.at_y + p.at_y for p in fp.pads]
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
            angle_rad = math.radians(fp.at_angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Rotate pad position around component center
            px = p.at_x * cos_a - p.at_y * sin_a + fp.at_x
            py = p.at_x * sin_a + p.at_y * cos_a + fp.at_y

            # Determine pad layer from pad's layers list or component layer
            # SMD pads: use pad's layer if available, else component layer
            # Through-hole: use component layer for visualization
            pad_layer = fp.layer
            if p.layers:
                if "B.Cu" in p.layers and "F.Cu" not in p.layers:
                    pad_layer = "B.Cu"
                elif "F.Cu" in p.layers:
                    pad_layer = "F.Cu"

            pads.append(PadGeometry(
                center=(px, py),
                shape=p.shape,
                size=(p.size_x, p.size_y),
                angle=p.at_angle + fp.at_angle,
                drill=p.drill,
                layer=pad_layer
            ))

        components.append(ComponentGeometry(
            reference=fp.reference,
            value=fp.value,
            center=(fp.at_x, fp.at_y),
            angle=fp.at_angle,
            bounding_box=bbox,
            pads=pads,
            layer=fp.layer,
            model_path=fp.model_path
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


def circle_to_polygon(center_x: float, center_y: float, radius: float, segments: int = 24) -> Polygon:
    """
    Convert a circle to a polygon approximation.

    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center
        radius: Circle radius
        segments: Number of segments (more = smoother)

    Returns:
        Polygon approximation of the circle
    """
    vertices = []
    for i in range(segments):
        theta = 2 * math.pi * i / segments
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        vertices.append((x, y))
    return Polygon(vertices)

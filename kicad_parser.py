"""
KiCad .kicad_pcb file parser

Parses S-expression format used by KiCad into a tree structure
that can be queried for board data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Iterator
import re


@dataclass
class SExpr:
    """
    Represents an S-expression node.

    Can be either:
    - A list node with a name and children: (name child1 child2 ...)
    - An atom (string, number, or symbol)
    """
    name: str
    children: list = field(default_factory=list)

    def __getitem__(self, key: Union[int, str]) -> Union['SExpr', str, None]:
        """
        Access children by index or find child by name.

        expr[0] -> first child
        expr['layer'] -> first child named 'layer'
        """
        if isinstance(key, int):
            if key < len(self.children):
                return self.children[key]
            return None
        else:
            for child in self.children:
                if isinstance(child, SExpr) and child.name == key:
                    return child
            return None

    def find_all(self, name: str) -> Iterator['SExpr']:
        """Find all direct children with the given name."""
        for child in self.children:
            if isinstance(child, SExpr) and child.name == name:
                yield child

    def get_value(self, index: int = 0) -> Union[str, float, None]:
        """Get atom value at index (after the name)."""
        if index < len(self.children):
            val = self.children[index]
            if isinstance(val, str):
                return val
            elif isinstance(val, SExpr):
                return val.name
        return None

    def get_float(self, index: int = 0) -> float:
        """Get float value at index."""
        val = self.get_value(index)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return 0.0

    def get_string(self, index: int = 0) -> str:
        """Get string value at index."""
        val = self.get_value(index)
        return str(val) if val is not None else ""

    def __repr__(self):
        if not self.children:
            return f"SExpr({self.name!r})"
        return f"SExpr({self.name!r}, {self.children!r})"


class SExprTokenizer:
    """Tokenize S-expression input."""

    TOKEN_PATTERN = re.compile(
        r'''
        (?P<LPAREN>\()|
        (?P<RPAREN>\))|
        (?P<STRING>"(?:[^"\\]|\\.)*")|
        (?P<ATOM>[^\s()]+)
        ''',
        re.VERBOSE
    )

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens = list(self._tokenize())
        self.token_pos = 0

    def _tokenize(self) -> Iterator[tuple[str, str]]:
        """Generate tokens from input text."""
        for match in self.TOKEN_PATTERN.finditer(self.text):
            kind = match.lastgroup
            value = match.group()
            if kind == 'STRING':
                # Remove quotes and unescape
                value = value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
            yield (kind, value)

    def peek(self) -> tuple[str, str] | None:
        """Look at next token without consuming."""
        if self.token_pos < len(self.tokens):
            return self.tokens[self.token_pos]
        return None

    def next(self) -> tuple[str, str] | None:
        """Consume and return next token."""
        token = self.peek()
        if token:
            self.token_pos += 1
        return token

    def expect(self, kind: str) -> str:
        """Consume token of expected kind, raise if not found."""
        token = self.next()
        if token is None:
            raise ValueError(f"Unexpected end of input, expected {kind}")
        if token[0] != kind:
            raise ValueError(f"Expected {kind}, got {token[0]}: {token[1]}")
        return token[1]


def parse_sexpr(tokenizer: SExprTokenizer) -> SExpr:
    """Parse an S-expression from tokenizer."""
    tokenizer.expect('LPAREN')

    # First element is the name
    token = tokenizer.next()
    if token is None:
        raise ValueError("Unexpected end of input in S-expression")

    if token[0] == 'LPAREN':
        # Nested expression as first element - unusual but handle it
        tokenizer.token_pos -= 1
        name = ""
    else:
        name = token[1]

    children = []

    while True:
        token = tokenizer.peek()
        if token is None:
            raise ValueError("Unexpected end of input, missing closing paren")

        if token[0] == 'RPAREN':
            tokenizer.next()
            break
        elif token[0] == 'LPAREN':
            children.append(parse_sexpr(tokenizer))
        else:
            tokenizer.next()
            children.append(token[1])

    return SExpr(name, children)


def parse_kicad_pcb(text: str) -> SExpr:
    """Parse a .kicad_pcb file content into S-expression tree."""
    tokenizer = SExprTokenizer(text)
    return parse_sexpr(tokenizer)


def load_kicad_pcb(filepath: Union[str, Path]) -> SExpr:
    """Load and parse a .kicad_pcb file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    text = filepath.read_text(encoding='utf-8')
    return parse_kicad_pcb(text)


@dataclass
class BoardInfo:
    """Basic board information extracted from kicad_pcb."""
    thickness: float = 1.6  # mm
    layers: list = field(default_factory=list)
    title: str = ""
    date: str = ""


@dataclass
class LayerPolygon:
    """A closed polygon from a user/graphic layer."""
    vertices: list  # list of (x, y) tuples
    layer: str


@dataclass
class GraphicLine:
    """A graphic line element."""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    layer: str
    width: float = 0.1
    stroke_type: str = "solid"  # solid, dash, dot, etc.


@dataclass
class Dimension:
    """A dimension annotation."""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    layer: str
    value: float  # The dimension value
    text: str  # The displayed text (may include units, degree symbol)


@dataclass
class CircleCutout:
    """A circular cutout/hole on Edge.Cuts layer."""
    center_x: float
    center_y: float
    radius: float


@dataclass
class TraceSegment:
    """A copper trace segment."""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    width: float
    layer: str
    net: int = 0


@dataclass
class Pad:
    """A component pad."""
    number: str
    pad_type: str  # thru_hole, smd, np_thru_hole
    shape: str  # circle, rect, oval, roundrect, custom
    at_x: float
    at_y: float
    at_angle: float = 0
    size_x: float = 0
    size_y: float = 0
    drill: float = 0
    layers: list = field(default_factory=list)


@dataclass
class Footprint:
    """A component footprint."""
    library: str
    name: str
    at_x: float
    at_y: float
    at_angle: float = 0
    layer: str = "F.Cu"
    reference: str = ""
    value: str = ""
    pads: list = field(default_factory=list)
    model_path: str = ""


class KiCadPCB:
    """
    Parsed KiCad PCB with extracted data.

    Usage:
        pcb = KiCadPCB.load("board.kicad_pcb")
        outline = pcb.get_board_outline()
        traces = pcb.get_traces()
    """

    def __init__(self, root: SExpr):
        self.root = root
        self._board_info = None
        self._outline = None
        self._polygon_cutouts = None  # Polygon cutouts from Edge.Cuts
        self._circle_cutouts = None   # Circle cutouts from Edge.Cuts
        self._graphic_lines = None
        self._dimensions = None
        self._traces = None
        self._footprints = None
        self._layer_polygons_cache = {}  # Cache for layer polygon extraction

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'KiCadPCB':
        """Load a .kicad_pcb file."""
        root = load_kicad_pcb(filepath)
        return cls(root)

    @classmethod
    def parse(cls, text: str) -> 'KiCadPCB':
        """Parse .kicad_pcb content from string."""
        root = parse_kicad_pcb(text)
        return cls(root)

    def get_board_info(self) -> BoardInfo:
        """Extract basic board information."""
        if self._board_info is not None:
            return self._board_info

        info = BoardInfo()

        # Get general section
        general = self.root['general']
        if general:
            thickness = general['thickness']
            if thickness:
                info.thickness = thickness.get_float(0)

        # Get layers
        layers_section = self.root['layers']
        if layers_section:
            for child in layers_section.children:
                if isinstance(child, SExpr):
                    layer_name = child.get_string(0)
                    if layer_name:
                        info.layers.append(layer_name)

        # Get title block info
        title_block = self.root['title_block']
        if title_block:
            title = title_block['title']
            if title:
                info.title = title.get_string(0)
            date = title_block['date']
            if date:
                info.date = date.get_string(0)

        self._board_info = info
        return info

    def get_board_outline(self) -> list[tuple[float, float]]:
        """
        Extract board outline from Edge.Cuts layer.

        Returns list of (x, y) vertices forming the outline polygon (largest polygon).
        """
        self._parse_edge_cuts_polygons()
        return self._outline

    def get_polygon_cutouts(self) -> list[list[tuple[float, float]]]:
        """
        Get polygon cutouts from Edge.Cuts layer.

        Returns list of polygon vertex lists representing internal cutouts
        (all closed polygons except the largest/outer one).
        """
        self._parse_edge_cuts_polygons()
        return self._polygon_cutouts

    def _linearize_arc(
        self,
        start: tuple[float, float],
        mid: tuple[float, float],
        end: tuple[float, float],
        segments_per_90deg: int = 8
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """
        Linearize an arc defined by 3 points into line segments.

        Args:
            start: Start point of arc
            mid: Middle point of arc (on the arc)
            end: End point of arc
            segments_per_90deg: Number of segments per 90 degrees of arc

        Returns:
            List of line segments [(p1, p2), ...]
        """
        import math

        if mid is None:
            # No mid point, just return straight line
            return [(start, end)]

        # Calculate circle center from 3 points
        # Using the perpendicular bisector method
        ax, ay = start
        bx, by = mid
        cx, cy = end

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            # Points are collinear, return straight line
            return [(start, end)]

        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

        center = (ux, uy)
        radius = math.sqrt((ax - ux)**2 + (ay - uy)**2)

        # Calculate angles
        angle_start = math.atan2(ay - uy, ax - ux)
        angle_mid = math.atan2(by - uy, bx - ux)
        angle_end = math.atan2(cy - uy, cx - ux)

        # Determine arc direction (clockwise or counter-clockwise)
        # by checking if mid is on the shorter or longer arc
        def normalize_angle(a):
            while a < 0:
                a += 2 * math.pi
            while a >= 2 * math.pi:
                a -= 2 * math.pi
            return a

        angle_start = normalize_angle(angle_start)
        angle_mid = normalize_angle(angle_mid)
        angle_end = normalize_angle(angle_end)

        # Check if going CCW from start through mid to end
        def angle_between(a1, a2):
            diff = a2 - a1
            while diff < 0:
                diff += 2 * math.pi
            while diff >= 2 * math.pi:
                diff -= 2 * math.pi
            return diff

        ccw_to_mid = angle_between(angle_start, angle_mid)
        ccw_to_end = angle_between(angle_start, angle_end)

        if ccw_to_mid < ccw_to_end:
            # CCW direction
            total_angle = ccw_to_end
            direction = 1
        else:
            # CW direction
            total_angle = 2 * math.pi - ccw_to_end
            direction = -1

        # Calculate number of segments
        num_segments = max(2, int(total_angle / (math.pi / 2) * segments_per_90deg))

        # Generate points along arc
        segments = []
        prev_point = start

        for i in range(1, num_segments + 1):
            t = i / num_segments
            angle = angle_start + direction * total_angle * t
            x = ux + radius * math.cos(angle)
            y = uy + radius * math.sin(angle)
            point = (x, y)
            segments.append((prev_point, point))
            prev_point = point

        return segments

    def _order_segments(self, segments: list, tolerance: float = 0.01) -> list[list[tuple[float, float]]]:
        """
        Order line segments into continuous polygons.
        Returns list of all closed polygons found.
        """
        if not segments:
            return []

        def point_eq(p1, p2):
            return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance

        remaining = list(segments)
        all_polygons = []

        while remaining:
            # Start a new polygon with the first remaining segment
            ordered = [remaining[0][0], remaining[0][1]]
            remaining.pop(0)

            # Keep adding segments that connect
            max_iterations = len(segments) * 2
            iterations = 0

            while remaining and iterations < max_iterations:
                iterations += 1
                found = False

                for i, seg in enumerate(remaining):
                    # Check if segment connects to end of ordered list
                    if point_eq(seg[0], ordered[-1]):
                        ordered.append(seg[1])
                        remaining.pop(i)
                        found = True
                        break
                    elif point_eq(seg[1], ordered[-1]):
                        ordered.append(seg[0])
                        remaining.pop(i)
                        found = True
                        break
                    # Check if segment connects to start of ordered list
                    elif point_eq(seg[1], ordered[0]):
                        ordered.insert(0, seg[0])
                        remaining.pop(i)
                        found = True
                        break
                    elif point_eq(seg[0], ordered[0]):
                        ordered.insert(0, seg[1])
                        remaining.pop(i)
                        found = True
                        break

                if not found:
                    # No connecting segment found, polygon is complete
                    break

            # Remove duplicate end point if polygon is closed
            if len(ordered) > 1 and point_eq(ordered[0], ordered[-1]):
                ordered.pop()

            # Store this polygon if it has at least 3 vertices
            if len(ordered) >= 3:
                all_polygons.append(ordered)

        return all_polygons

    def _parse_edge_cuts_polygons(self):
        """Parse all polygons from Edge.Cuts layer."""
        if self._outline is not None:
            return  # Already parsed

        # Collect all Edge.Cuts segments
        segments = []

        for line in self.root.find_all('gr_line'):
            layer = line['layer']
            if layer and layer.get_string(0) == 'Edge.Cuts':
                start = line['start']
                end = line['end']
                if start and end:
                    segments.append((
                        (start.get_float(0), start.get_float(1)),
                        (end.get_float(0), end.get_float(1))
                    ))

        for arc in self.root.find_all('gr_arc'):
            layer = arc['layer']
            if layer and layer.get_string(0) == 'Edge.Cuts':
                start = arc['start']
                mid = arc['mid']
                end = arc['end']
                if start and end:
                    arc_segments = self._linearize_arc(
                        (start.get_float(0), start.get_float(1)),
                        (mid.get_float(0), mid.get_float(1)) if mid else None,
                        (end.get_float(0), end.get_float(1))
                    )
                    segments.extend(arc_segments)

        for rect in self.root.find_all('gr_rect'):
            layer = rect['layer']
            if layer and layer.get_string(0) == 'Edge.Cuts':
                start = rect['start']
                end = rect['end']
                if start and end:
                    x1, y1 = start.get_float(0), start.get_float(1)
                    x2, y2 = end.get_float(0), end.get_float(1)
                    segments.append(((x1, y1), (x2, y1)))
                    segments.append(((x2, y1), (x2, y2)))
                    segments.append(((x2, y2), (x1, y2)))
                    segments.append(((x1, y2), (x1, y1)))

        # Order segments into polygons
        all_polygons = self._order_segments(segments)

        if not all_polygons:
            self._outline = []
            self._polygon_cutouts = []
            return

        # Calculate area for each polygon
        def polygon_area(vertices):
            n = len(vertices)
            if n < 3:
                return 0
            area = 0
            for i in range(n):
                j = (i + 1) % n
                area += vertices[i][0] * vertices[j][1]
                area -= vertices[j][0] * vertices[i][1]
            return abs(area) / 2

        # Sort by area (largest first)
        all_polygons.sort(key=polygon_area, reverse=True)

        # Largest is the outline, rest are cutouts
        self._outline = all_polygons[0]
        self._polygon_cutouts = all_polygons[1:] if len(all_polygons) > 1 else []

    def get_circle_cutouts(self) -> list[CircleCutout]:
        """
        Get circular cutouts from Edge.Cuts layer.

        Returns list of CircleCutout objects representing holes in the board.
        """
        if self._circle_cutouts is not None:
            return self._circle_cutouts

        import math

        self._circle_cutouts = []

        for circle in self.root.find_all('gr_circle'):
            layer = circle['layer']
            if layer and layer.get_string(0) == 'Edge.Cuts':
                center = circle['center']
                end = circle['end']

                if center and end:
                    cx = center.get_float(0)
                    cy = center.get_float(1)
                    ex = end.get_float(0)
                    ey = end.get_float(1)

                    # Calculate radius from center to end point
                    radius = math.sqrt((ex - cx)**2 + (ey - cy)**2)

                    if radius > 0.01:  # Ignore very small circles
                        self._circle_cutouts.append(CircleCutout(
                            center_x=cx,
                            center_y=cy,
                            radius=radius
                        ))

        return self._circle_cutouts

    def get_graphic_lines(self, layer: str = None) -> list[GraphicLine]:
        """
        Get graphic lines, optionally filtered by layer.

        Args:
            layer: Filter by layer name (e.g., 'User.1'), or None for all
        """
        if self._graphic_lines is None:
            self._graphic_lines = []

            for line in self.root.find_all('gr_line'):
                layer_expr = line['layer']
                if not layer_expr:
                    continue

                line_layer = layer_expr.get_string(0)
                start = line['start']
                end = line['end']

                if not start or not end:
                    continue

                stroke = line['stroke']
                width = 0.1
                stroke_type = "solid"

                if stroke:
                    width_expr = stroke['width']
                    if width_expr:
                        width = width_expr.get_float(0)
                    type_expr = stroke['type']
                    if type_expr:
                        stroke_type = type_expr.get_string(0)
                else:
                    # Legacy format
                    width_expr = line['width']
                    if width_expr:
                        width = width_expr.get_float(0)

                self._graphic_lines.append(GraphicLine(
                    start_x=start.get_float(0),
                    start_y=start.get_float(1),
                    end_x=end.get_float(0),
                    end_y=end.get_float(1),
                    layer=line_layer,
                    width=width,
                    stroke_type=stroke_type
                ))

        if layer is None:
            return self._graphic_lines

        return [l for l in self._graphic_lines if l.layer == layer]

    def get_dimensions(self, layer: str = None) -> list[Dimension]:
        """
        Get dimension annotations, optionally filtered by layer.

        Args:
            layer: Filter by layer name (e.g., 'User.1'), or None for all
        """
        if self._dimensions is None:
            self._dimensions = []

            for dim in self.root.find_all('dimension'):
                layer_expr = dim['layer']
                if not layer_expr:
                    continue

                dim_layer = layer_expr.get_string(0)

                # Get dimension type (KiCad 7+)
                dim_type = dim['type']

                # Get points - format varies by KiCad version and dimension type
                pts = dim['pts']
                start_x, start_y = 0, 0
                end_x, end_y = 0, 0

                if pts:
                    xy_list = list(pts.find_all('xy'))
                    if len(xy_list) >= 2:
                        start_x = xy_list[0].get_float(0)
                        start_y = xy_list[0].get_float(1)
                        end_x = xy_list[1].get_float(0)
                        end_y = xy_list[1].get_float(1)

                # Get the text/value
                gr_text = dim['gr_text']
                text = ""
                value = 0.0

                if gr_text:
                    text = gr_text.get_string(0)
                    # Try to parse numeric value from text
                    # Handle formats like "90", "90°", "+90°", "-45.5°"
                    match = re.search(r'([+-]?\d+\.?\d*)', text)
                    if match:
                        try:
                            value = float(match.group(1))
                        except ValueError:
                            pass

                self._dimensions.append(Dimension(
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    layer=dim_layer,
                    value=value,
                    text=text
                ))

        if layer is None:
            return self._dimensions

        return [d for d in self._dimensions if d.layer == layer]

    def get_traces(self, layer: str = None) -> list[TraceSegment]:
        """
        Get copper trace segments, optionally filtered by layer.

        Args:
            layer: Filter by layer name (e.g., 'F.Cu'), or None for all
        """
        if self._traces is None:
            self._traces = []

            for segment in self.root.find_all('segment'):
                layer_expr = segment['layer']
                if not layer_expr:
                    continue

                seg_layer = layer_expr.get_string(0)
                start = segment['start']
                end = segment['end']
                width = segment['width']
                net = segment['net']

                if not start or not end:
                    continue

                self._traces.append(TraceSegment(
                    start_x=start.get_float(0),
                    start_y=start.get_float(1),
                    end_x=end.get_float(0),
                    end_y=end.get_float(1),
                    width=width.get_float(0) if width else 0.25,
                    layer=seg_layer,
                    net=int(net.get_float(0)) if net else 0
                ))

        if layer is None:
            return self._traces

        return [t for t in self._traces if t.layer == layer]

    def get_footprints(self) -> list[Footprint]:
        """Get all footprints (components) on the board."""
        if self._footprints is not None:
            return self._footprints

        self._footprints = []

        for fp in self.root.find_all('footprint'):
            lib_name = fp.get_string(0)
            parts = lib_name.split(':')
            library = parts[0] if len(parts) > 1 else ""
            name = parts[-1]

            at = fp['at']
            at_x, at_y, at_angle = 0, 0, 0
            if at:
                at_x = at.get_float(0)
                at_y = at.get_float(1)
                at_angle = at.get_float(2)

            layer_expr = fp['layer']
            layer = layer_expr.get_string(0) if layer_expr else "F.Cu"

            # Get reference and value
            reference = ""
            value = ""
            for prop in fp.find_all('property'):
                prop_name = prop.get_string(0)
                if prop_name == "Reference":
                    reference = prop.get_string(1)
                elif prop_name == "Value":
                    value = prop.get_string(1)

            # Legacy format for reference/value
            if not reference:
                ref_expr = fp['fp_text']
                # Would need to iterate to find 'reference' type

            # Get 3D model path
            model_path = ""
            model = fp['model']
            if model:
                model_path = model.get_string(0)

            # Get pads
            pads = []
            for pad_expr in fp.find_all('pad'):
                pad_num = pad_expr.get_string(0)
                pad_type = pad_expr.get_string(1)
                pad_shape = pad_expr.get_string(2)

                pad_at = pad_expr['at']
                pad_x, pad_y, pad_angle = 0, 0, 0
                if pad_at:
                    pad_x = pad_at.get_float(0)
                    pad_y = pad_at.get_float(1)
                    pad_angle = pad_at.get_float(2)

                pad_size = pad_expr['size']
                size_x, size_y = 0, 0
                if pad_size:
                    size_x = pad_size.get_float(0)
                    size_y = pad_size.get_float(1)

                drill_expr = pad_expr['drill']
                drill = 0
                if drill_expr:
                    drill = drill_expr.get_float(0)

                pad_layers = []
                layers_expr = pad_expr['layers']
                if layers_expr:
                    pad_layers = [l for l in layers_expr.children if isinstance(l, str)]

                pads.append(Pad(
                    number=pad_num,
                    pad_type=pad_type,
                    shape=pad_shape,
                    at_x=pad_x,
                    at_y=pad_y,
                    at_angle=pad_angle,
                    size_x=size_x,
                    size_y=size_y,
                    drill=drill,
                    layers=pad_layers
                ))

            self._footprints.append(Footprint(
                library=library,
                name=name,
                at_x=at_x,
                at_y=at_y,
                at_angle=at_angle,
                layer=layer,
                reference=reference,
                value=value,
                pads=pads,
                model_path=model_path
            ))

        return self._footprints

    def get_available_layers(self) -> list[str]:
        """
        Get list of all layer names defined in the PCB.

        Returns:
            List of layer names (e.g., ['F.Cu', 'B.Cu', 'User.1', 'User.2', ...])
        """
        info = self.get_board_info()
        return info.layers

    def get_user_layers(self) -> list[str]:
        """
        Get list of User layers (User.1, User.2, etc.).

        These are commonly used for stiffener outlines and other annotations.
        """
        return [l for l in self.get_available_layers() if l.startswith("User.")]

    def get_layer_polygons(self, layer: str) -> list[LayerPolygon]:
        """
        Extract closed polygons from a specific layer.

        Supports:
        - gr_poly: Explicit polygons
        - gr_rect: Rectangles (converted to 4-vertex polygon)
        - gr_line/gr_arc: Connected line/arc segments forming closed shapes

        Args:
            layer: Layer name (e.g., 'User.2')

        Returns:
            List of LayerPolygon objects
        """
        if layer in self._layer_polygons_cache:
            return self._layer_polygons_cache[layer]

        polygons = []

        # 1. Extract explicit polygons (gr_poly)
        for poly in self.root.find_all('gr_poly'):
            layer_expr = poly['layer']
            if not layer_expr or layer_expr.get_string(0) != layer:
                continue

            pts = poly['pts']
            if pts:
                vertices = []
                for xy in pts.find_all('xy'):
                    vertices.append((xy.get_float(0), xy.get_float(1)))
                if len(vertices) >= 3:
                    polygons.append(LayerPolygon(vertices=vertices, layer=layer))

        # 2. Extract rectangles (gr_rect) and convert to polygons
        for rect in self.root.find_all('gr_rect'):
            layer_expr = rect['layer']
            if not layer_expr or layer_expr.get_string(0) != layer:
                continue

            start = rect['start']
            end = rect['end']
            if start and end:
                x1, y1 = start.get_float(0), start.get_float(1)
                x2, y2 = end.get_float(0), end.get_float(1)
                vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                polygons.append(LayerPolygon(vertices=vertices, layer=layer))

        # 3. Extract connected line/arc segments and form closed polygons
        segments = []

        for line in self.root.find_all('gr_line'):
            layer_expr = line['layer']
            if not layer_expr or layer_expr.get_string(0) != layer:
                continue

            start = line['start']
            end = line['end']
            if start and end:
                segments.append((
                    (start.get_float(0), start.get_float(1)),
                    (end.get_float(0), end.get_float(1))
                ))

        for arc in self.root.find_all('gr_arc'):
            layer_expr = arc['layer']
            if not layer_expr or layer_expr.get_string(0) != layer:
                continue

            start = arc['start']
            mid = arc['mid']
            end = arc['end']
            if start and end:
                arc_segments = self._linearize_arc(
                    (start.get_float(0), start.get_float(1)),
                    (mid.get_float(0), mid.get_float(1)) if mid else None,
                    (end.get_float(0), end.get_float(1))
                )
                segments.extend(arc_segments)

        # Order segments into closed polygons
        if segments:
            ordered_polygons = self._order_segments(segments)
            for vertices in ordered_polygons:
                if len(vertices) >= 3:
                    polygons.append(LayerPolygon(vertices=vertices, layer=layer))

        self._layer_polygons_cache[layer] = polygons
        return polygons

    def get_layer_polygon_vertices(self, layer: str) -> list[list[tuple[float, float]]]:
        """
        Get polygon vertices from a layer as simple lists.

        Convenience method that returns just the vertex lists without LayerPolygon wrapper.

        Args:
            layer: Layer name (e.g., 'User.2')

        Returns:
            List of vertex lists, each representing a closed polygon
        """
        return [poly.vertices for poly in self.get_layer_polygons(layer)]

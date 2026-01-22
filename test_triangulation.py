"""
Test script for triangulation with holes.
Implements ear clipping from Eberly's "Triangulation by Ear Clipping" paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import random
import math


# =============================================================================
# Polygon Generation
# =============================================================================

def generate_random_convex_polygon(center, radius, num_vertices, irregularity=0.3):
    """Generate a random convex polygon."""
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
    vertices = []
    for angle in angles:
        r = radius * (1 + random.uniform(-irregularity, irregularity))
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        vertices.append((x, y))
    return vertices


def generate_random_concave_polygon(center, radius, num_vertices, concavity=0.3):
    """Generate a random concave polygon."""
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    vertices = []
    for i, angle in enumerate(angles):
        if i % 3 == 1:
            r = radius * (1 - concavity * random.uniform(0.5, 1.0))
        else:
            r = radius * (1 + random.uniform(-0.1, 0.1))
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        vertices.append((x, y))
    return vertices


# =============================================================================
# Basic Geometry Functions
# =============================================================================

def signed_area(polygon):
    """
    Calculate signed area of polygon.
    Positive = CCW, Negative = CW (in standard Y-up coordinates)
    """
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0


def ensure_ccw(polygon):
    """Ensure polygon is counter-clockwise ordered."""
    if signed_area(polygon) < 0:
        return list(reversed(polygon))
    return list(polygon)


def ensure_cw(polygon):
    """Ensure polygon is clockwise ordered."""
    if signed_area(polygon) > 0:
        return list(reversed(polygon))
    return list(polygon)


def cross_product_2d(o, a, b):
    """
    Cross product of vectors OA and OB.
    Positive = B is to the left of OA (CCW turn)
    Negative = B is to the right of OA (CW turn)
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex_vertex(prev_v, curr_v, next_v):
    """
    Check if curr_v is a convex vertex (interior angle < 180 degrees).
    For CCW polygon, convex means cross product > 0.
    """
    return cross_product_2d(prev_v, curr_v, next_v) > 0


def is_reflex_vertex(prev_v, curr_v, next_v):
    """
    Check if curr_v is a reflex vertex (interior angle > 180 degrees).
    For CCW polygon, reflex means cross product < 0.
    """
    return cross_product_2d(prev_v, curr_v, next_v) < 0


def point_in_triangle(p, a, b, c):
    """Check if point p is strictly inside triangle abc."""
    d1 = cross_product_2d(a, b, p)
    d2 = cross_product_2d(b, c, p)
    d3 = cross_product_2d(c, a, p)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# =============================================================================
# Ear Clipping Algorithm (Section 2 of PDF)
# =============================================================================

def find_reflex_vertices(polygon):
    """Find all reflex vertex indices in a CCW polygon."""
    n = len(polygon)
    reflex = []
    for i in range(n):
        prev_v = polygon[(i - 1) % n]
        curr_v = polygon[i]
        next_v = polygon[(i + 1) % n]
        if is_reflex_vertex(prev_v, curr_v, next_v):
            reflex.append(i)
    return reflex


def is_ear(polygon, vertex_idx, reflex_indices):
    """
    Check if vertex at vertex_idx is an ear.
    An ear must be convex and have no reflex vertices inside its triangle.
    """
    n = len(polygon)
    prev_idx = (vertex_idx - 1) % n
    next_idx = (vertex_idx + 1) % n

    prev_v = polygon[prev_idx]
    curr_v = polygon[vertex_idx]
    next_v = polygon[next_idx]

    # Must be convex
    if not is_convex_vertex(prev_v, curr_v, next_v):
        return False

    # Check that no reflex vertex is inside the triangle
    for r_idx in reflex_indices:
        if r_idx in (prev_idx, vertex_idx, next_idx):
            continue
        if point_in_triangle(polygon[r_idx], prev_v, curr_v, next_v):
            return False

    return True


def ear_clip_triangulate(polygon):
    """
    Triangulate a simple polygon using ear clipping.

    Args:
        polygon: List of (x, y) vertices in CCW order

    Returns:
        List of triangles, each as (i, j, k) indices into original polygon
    """
    polygon = ensure_ccw(polygon)
    n = len(polygon)

    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Work with indices into the original polygon
    indices = list(range(n))
    triangles = []

    # Find initial reflex vertices
    reflex_set = set(find_reflex_vertices(polygon))

    max_iterations = n * n
    iteration = 0

    while len(indices) > 3 and iteration < max_iterations:
        iteration += 1
        ear_found = False

        for i in range(len(indices)):
            idx = indices[i]
            prev_idx = indices[(i - 1) % len(indices)]
            next_idx = indices[(i + 1) % len(indices)]

            prev_v = polygon[prev_idx]
            curr_v = polygon[idx]
            next_v = polygon[next_idx]

            # Check if convex
            if not is_convex_vertex(prev_v, curr_v, next_v):
                continue

            # Check no reflex vertex inside triangle
            is_valid_ear = True
            for r_idx in reflex_set:
                if r_idx in (prev_idx, idx, next_idx):
                    continue
                if r_idx not in indices:
                    continue
                if point_in_triangle(polygon[r_idx], prev_v, curr_v, next_v):
                    is_valid_ear = False
                    break

            if is_valid_ear:
                # Found an ear - clip it
                triangles.append((prev_idx, idx, next_idx))
                indices.remove(idx)

                # Update reflex status of adjacent vertices
                if len(indices) >= 3:
                    # Find new positions of prev and next in remaining indices
                    new_prev_pos = indices.index(prev_idx)
                    new_next_pos = indices.index(next_idx)

                    # Check if prev vertex changed from reflex to convex
                    pp = indices[(new_prev_pos - 1) % len(indices)]
                    pn = indices[(new_prev_pos + 1) % len(indices)]
                    if is_reflex_vertex(polygon[pp], polygon[prev_idx], polygon[pn]):
                        reflex_set.add(prev_idx)
                    else:
                        reflex_set.discard(prev_idx)

                    # Check if next vertex changed from reflex to convex
                    np_ = indices[(new_next_pos - 1) % len(indices)]
                    nn = indices[(new_next_pos + 1) % len(indices)]
                    if is_reflex_vertex(polygon[np_], polygon[next_idx], polygon[nn]):
                        reflex_set.add(next_idx)
                    else:
                        reflex_set.discard(next_idx)

                ear_found = True
                break

        if not ear_found:
            # Fallback: just clip any vertex
            if len(indices) > 3:
                i = 0
                prev_idx = indices[(i - 1) % len(indices)]
                idx = indices[i]
                next_idx = indices[(i + 1) % len(indices)]
                triangles.append((prev_idx, idx, next_idx))
                indices.remove(idx)

    # Final triangle
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return triangles


# =============================================================================
# Finding Mutually Visible Vertices (Section 4 of PDF)
# =============================================================================

def find_mutually_visible_vertex(M, outer_polygon):
    """
    Find a vertex in outer_polygon that is mutually visible with point M.

    Algorithm from Section 4 of Eberly's paper:
    1. Cast ray M + t(1,0) in positive x direction
    2. Find closest intersection I with outer polygon edge
    3. If I is a vertex, return that vertex
    4. Otherwise, P = endpoint of edge with max x-value
    5. If reflex vertices inside triangle <M,I,P>, return one with min angle
    6. Otherwise return P

    Args:
        M: Point (x, y) - typically the rightmost vertex of a hole
        outer_polygon: List of vertices (CCW ordered)

    Returns:
        Index of mutually visible vertex in outer_polygon
    """
    mx, my = M
    n = len(outer_polygon)

    # Step 2: Find closest intersection with ray M + t(1,0)
    min_t = float('inf')
    hit_edge_start = -1
    intersection_point = None

    for i in range(n):
        j = (i + 1) % n
        vi = outer_polygon[i]
        vj = outer_polygon[j]

        # Only consider edges where vi is below (or on) ray and vj is above (or on)
        # This handles the directed edge requirement from the paper
        if not ((vi[1] <= my < vj[1]) or (vj[1] <= my < vi[1])):
            continue

        # Skip horizontal edges
        if abs(vj[1] - vi[1]) < 1e-10:
            continue

        # Calculate intersection x coordinate
        t_edge = (my - vi[1]) / (vj[1] - vi[1])
        x_intersect = vi[0] + t_edge * (vj[0] - vi[0])

        # Must be to the right of M
        if x_intersect <= mx:
            continue

        t = x_intersect - mx
        if t < min_t:
            min_t = t
            hit_edge_start = i
            intersection_point = (x_intersect, my)

    if hit_edge_start == -1:
        # Fallback: find closest vertex to the right
        best_idx = 0
        best_dist = float('inf')
        for i, v in enumerate(outer_polygon):
            if v[0] > mx:
                dist = (v[0] - mx) ** 2 + (v[1] - my) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        return best_idx

    I = intersection_point
    vi = outer_polygon[hit_edge_start]
    vj = outer_polygon[(hit_edge_start + 1) % n]

    # Step 3: Check if I is a vertex
    eps = 1e-9
    if abs(I[0] - vi[0]) < eps and abs(I[1] - vi[1]) < eps:
        return hit_edge_start
    if abs(I[0] - vj[0]) < eps and abs(I[1] - vj[1]) < eps:
        return (hit_edge_start + 1) % n

    # Step 4: P = endpoint with maximum x-value
    if vi[0] > vj[0]:
        P_idx = hit_edge_start
    else:
        P_idx = (hit_edge_start + 1) % n
    P = outer_polygon[P_idx]

    # Step 5: Find reflex vertices inside triangle <M, I, P>
    reflex_in_triangle = []
    for i in range(n):
        if i == P_idx:
            continue
        prev_v = outer_polygon[(i - 1) % n]
        curr_v = outer_polygon[i]
        next_v = outer_polygon[(i + 1) % n]

        if is_reflex_vertex(prev_v, curr_v, next_v):
            if point_in_triangle(curr_v, M, I, P):
                reflex_in_triangle.append(i)

    # Step 6: If no reflex vertices in triangle, P is visible
    if not reflex_in_triangle:
        return P_idx

    # Step 7: Find reflex vertex with minimum angle to ray direction
    best_idx = reflex_in_triangle[0]
    best_angle = float('inf')

    for idx in reflex_in_triangle:
        R = outer_polygon[idx]
        dx = R[0] - mx
        dy = abs(R[1] - my)
        if dx > eps:
            angle = dy / dx  # tan(angle) - smaller is better
            if angle < best_angle:
                best_angle = angle
                best_idx = idx

    return best_idx


# =============================================================================
# Polygon with Holes (Section 3 & 5 of PDF)
# =============================================================================

def merge_hole_into_polygon(outer, hole):
    """
    Merge a hole into the outer polygon to create a pseudosimple polygon.

    From Section 3 of the paper:
    - Find M = vertex with max x-value in hole
    - Find V = mutually visible vertex in outer
    - Create bridge with edges <V,M> and <M,V>

    The merged polygon format (from equation 2, page 8):
    {outer_before_V, V, hole_from_M, M, V, outer_after_V}

    Args:
        outer: Outer polygon vertices (CCW)
        hole: Hole polygon vertices (CW)

    Returns:
        Merged pseudosimple polygon
    """
    # Find M = vertex with maximum x-value in hole
    m_idx = max(range(len(hole)), key=lambda i: hole[i][0])
    M = hole[m_idx]

    # Find V = mutually visible vertex in outer polygon
    v_idx = find_mutually_visible_vertex(M, outer)

    # Reorder hole to start from M
    hole_from_M = hole[m_idx:] + hole[:m_idx]

    # Build merged polygon:
    # {outer[0..v_idx], hole_from_M, M, V, outer[v_idx+1..end]}
    merged = (
        outer[:v_idx + 1] +      # outer up to and including V
        hole_from_M +             # hole starting from M
        [M] +                     # M again (bridge return)
        [outer[v_idx]] +          # V again (bridge return)
        outer[v_idx + 1:]         # rest of outer
    )

    return merged


def triangulate_with_holes(outer, holes):
    """
    Triangulate a polygon with holes using ear clipping.

    Algorithm from Section 5:
    1. Ensure outer is CCW, holes are CW
    2. Sort holes by maximum x-value (rightmost first)
    3. Merge each hole into the outer polygon
    4. Triangulate the resulting pseudosimple polygon

    Args:
        outer: Outer polygon vertices
        holes: List of hole polygons

    Returns:
        Tuple of (triangles, merged_polygon)
        triangles: List of (i, j, k) indices into merged_polygon
    """
    if not holes:
        outer_ccw = ensure_ccw(outer)
        return ear_clip_triangulate(outer_ccw), outer_ccw

    # Step 1: Ensure correct winding
    outer_ccw = ensure_ccw(outer)
    holes_cw = [ensure_cw(h) for h in holes]

    # Step 2: Sort holes by maximum x-value (process rightmost first)
    hole_data = []
    for hole in holes_cw:
        max_x = max(v[0] for v in hole)
        hole_data.append((max_x, hole))
    hole_data.sort(key=lambda x: x[0], reverse=True)

    # Step 3: Merge each hole
    result = outer_ccw
    for _, hole in hole_data:
        result = merge_hole_into_polygon(result, hole)

    # Step 4: Triangulate
    triangles = ear_clip_triangulate(result)

    return triangles, result


# =============================================================================
# Visualization
# =============================================================================

def plot_polygon_with_holes(outer, holes, title="Polygon with Holes", filename=None):
    """Plot outer polygon and holes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot outer polygon
    outer_closed = list(outer) + [outer[0]]
    xs, ys = zip(*outer_closed)
    ax.plot(xs, ys, 'b-', linewidth=2, label='Outer')
    ax.fill(xs, ys, alpha=0.2, color='blue')

    # Plot holes
    colors = ['red', 'green', 'orange', 'purple']
    for i, hole in enumerate(holes):
        hole_closed = list(hole) + [hole[0]]
        xs, ys = zip(*hole_closed)
        color = colors[i % len(colors)]
        ax.plot(xs, ys, '-', color=color, linewidth=2, label=f'Hole {i+1}')
        ax.fill(xs, ys, alpha=0.5, color='white')

    # Mark vertices
    for i, v in enumerate(outer):
        ax.plot(v[0], v[1], 'bo', markersize=6)
        ax.annotate(f'{i}', (v[0]+0.5, v[1]+0.5), fontsize=8, color='blue')

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved to {filename}")
    plt.show()


def plot_triangulation(polygon, triangles, title="Triangulation", filename=None):
    """Plot the triangulated polygon."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot each triangle
    colors = plt.cm.Set3(np.linspace(0, 1, len(triangles)))
    for i, (a, b, c) in enumerate(triangles):
        tri_verts = [polygon[a], polygon[b], polygon[c]]
        tri = MplPolygon(tri_verts, closed=True,
                         facecolor=colors[i], edgecolor='black',
                         linewidth=1, alpha=0.7)
        ax.add_patch(tri)

    # Plot polygon outline
    poly_closed = list(polygon) + [polygon[0]]
    xs, ys = zip(*poly_closed)
    ax.plot(xs, ys, 'b-', linewidth=2)

    # Mark vertices
    for i, v in enumerate(polygon):
        ax.plot(v[0], v[1], 'ko', markersize=5)
        ax.annotate(f'{i}', (v[0]+0.3, v[1]+0.3), fontsize=7)

    ax.set_aspect('equal')
    ax.set_title(f"{title} ({len(triangles)} triangles)")
    ax.grid(True, alpha=0.3)
    ax.autoscale()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved to {filename}")
    plt.show()


def generate_hole_inside_polygon(outer, hole_radius, max_attempts=100):
    """Generate a random hole polygon that fits inside the outer polygon."""
    min_x = min(v[0] for v in outer)
    max_x = max(v[0] for v in outer)
    min_y = min(v[1] for v in outer)
    max_y = max(v[1] for v in outer)

    for _ in range(max_attempts):
        cx = random.uniform(min_x + hole_radius * 1.5, max_x - hole_radius * 1.5)
        cy = random.uniform(min_y + hole_radius * 1.5, max_y - hole_radius * 1.5)

        if not point_in_polygon((cx, cy), outer):
            continue

        hole = generate_random_convex_polygon((cx, cy), hole_radius, 8)
        all_inside = all(point_in_polygon(v, outer) for v in hole)
        if all_inside:
            return hole

    return None


# =============================================================================
# Planar Subdivision for Bend Zone Regions
# =============================================================================

def line_from_two_points(p1, p2):
    """
    Create line equation ax + by + c = 0 from two points.
    Returns (a, b, c) normalized so that (a, b) is unit length.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Normal to line direction
    a = -dy
    b = dx
    length = math.sqrt(a*a + b*b)
    if length < 1e-10:
        return (0, 0, 0)
    a /= length
    b /= length
    c = -(a * p1[0] + b * p1[1])
    return (a, b, c)


def signed_distance_to_line(point, line):
    """Signed distance from point to line (a, b, c)."""
    a, b, c = line
    return a * point[0] + b * point[1] + c


def segment_line_intersection(seg_start, seg_end, line):
    """
    Find intersection of segment with infinite line.
    Returns (t, point) where t is parameter [0,1] on segment, or None if no intersection.
    """
    d1 = signed_distance_to_line(seg_start, line)
    d2 = signed_distance_to_line(seg_end, line)

    # Both on same side - no intersection
    if d1 * d2 > 1e-10:
        return None

    # Both on the line
    if abs(d1) < 1e-10 and abs(d2) < 1e-10:
        return None  # Collinear, treat as no single intersection

    # One endpoint on line
    if abs(d1) < 1e-10:
        return (0.0, seg_start)
    if abs(d2) < 1e-10:
        return (1.0, seg_end)

    # Proper intersection
    t = d1 / (d1 - d2)
    px = seg_start[0] + t * (seg_end[0] - seg_start[0])
    py = seg_start[1] + t * (seg_end[1] - seg_start[1])
    return (t, (px, py))


def points_equal(p1, p2, eps=1e-9):
    """Check if two points are equal within epsilon."""
    return abs(p1[0] - p2[0]) < eps and abs(p1[1] - p2[1]) < eps


def normalize_angle(angle):
    """Normalize angle to [0, 2*pi)."""
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle


class PlanarSubdivision:
    """
    Computes regions by partitioning a polygon (with holes) using cutting lines.

    Uses boundary tracing: treats all edges (outer, holes, cutting lines) as a
    planar graph and traces faces.
    """

    def __init__(self, outer, holes, cutting_lines):
        """
        Args:
            outer: Outer boundary polygon (CCW)
            holes: List of hole polygons (CW)
            cutting_lines: List of (line_eq, p1, p2) where line_eq is (a,b,c)
                          and p1,p2 are points defining extent
        """
        self.outer = ensure_ccw(outer)
        self.holes = [ensure_cw(h) for h in holes]
        self.cutting_lines = cutting_lines

        # Will be populated by compute()
        self.vertices = []  # List of (x, y) vertices
        self.vertex_to_idx = {}  # (x, y) -> index
        self.edges = []  # List of (start_idx, end_idx)
        self.vertex_edges = {}  # vertex_idx -> list of (angle, edge_idx, direction)
        self.regions = []  # List of region boundaries

    def _add_vertex(self, point):
        """Add vertex and return its index."""
        # Round to avoid floating point issues
        key = (round(point[0], 9), round(point[1], 9))
        if key in self.vertex_to_idx:
            return self.vertex_to_idx[key]
        idx = len(self.vertices)
        self.vertices.append(point)
        self.vertex_to_idx[key] = idx
        return idx

    def _add_edge(self, start_idx, end_idx):
        """Add edge between two vertices."""
        if start_idx == end_idx:
            return
        # Avoid duplicate edges
        edge = (min(start_idx, end_idx), max(start_idx, end_idx))
        if edge not in [(min(e[0], e[1]), max(e[0], e[1])) for e in self.edges]:
            self.edges.append((start_idx, end_idx))

    def _collect_polygon_edges(self, polygon, cutting_lines):
        """
        Collect edges from a polygon, splitting at cutting line intersections.
        Returns list of (start_point, end_point) edges.
        """
        edges = []
        n = len(polygon)

        for i in range(n):
            seg_start = polygon[i]
            seg_end = polygon[(i + 1) % n]

            # Find all intersections with cutting lines
            intersections = []
            for line_eq, _, _ in cutting_lines:
                result = segment_line_intersection(seg_start, seg_end, line_eq)
                if result is not None:
                    t, point = result
                    if 0 < t < 1:  # Proper intersection (not at endpoints)
                        intersections.append((t, point))

            # Sort by parameter t
            intersections.sort(key=lambda x: x[0])

            # Create sub-edges
            current = seg_start
            for _, int_point in intersections:
                if not points_equal(current, int_point):
                    edges.append((current, int_point))
                current = int_point
            if not points_equal(current, seg_end):
                edges.append((current, seg_end))

        return edges

    def _collect_cutting_line_segments(self, line_eq, line_p1, line_p2, all_polygons):
        """
        Collect segments of cutting line that are inside the board.
        The cutting line is clipped to polygon intersections.
        """
        # Find all intersections of cutting line with all polygon edges
        intersections = []

        for polygon in all_polygons:
            n = len(polygon)
            for i in range(n):
                seg_start = polygon[i]
                seg_end = polygon[(i + 1) % n]
                result = segment_line_intersection(seg_start, seg_end, line_eq)
                if result is not None:
                    _, point = result
                    intersections.append(point)

        if len(intersections) < 2:
            return []

        # Sort intersections along the line direction
        line_dir = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
        def project(p):
            return (p[0] - line_p1[0]) * line_dir[0] + (p[1] - line_p1[1]) * line_dir[1]

        intersections.sort(key=project)

        # Remove duplicates
        unique = [intersections[0]]
        for p in intersections[1:]:
            if not points_equal(p, unique[-1]):
                unique.append(p)

        # Create segments between consecutive intersections that are inside the board
        segments = []
        for i in range(len(unique) - 1):
            p1, p2 = unique[i], unique[i + 1]
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            # Check if midpoint is inside outer and outside all holes
            if point_in_polygon(mid, self.outer):
                inside_hole = False
                for hole in self.holes:
                    if point_in_polygon(mid, hole):
                        inside_hole = True
                        break
                if not inside_hole:
                    segments.append((p1, p2))

        return segments

    def compute(self, debug=False):
        """Compute the planar subdivision and extract regions."""
        # Step 1: Collect all edges
        all_polygons = [self.outer] + self.holes

        # Edges from outer boundary
        outer_edges = self._collect_polygon_edges(self.outer, self.cutting_lines)

        # Edges from holes
        hole_edges = []
        for hole in self.holes:
            hole_edges.extend(self._collect_polygon_edges(hole, self.cutting_lines))

        # Edges from cutting lines
        cutting_edges = []
        for line_eq, p1, p2 in self.cutting_lines:
            segments = self._collect_cutting_line_segments(line_eq, p1, p2, all_polygons)
            cutting_edges.extend(segments)

        if debug:
            print(f"  Outer edges: {len(outer_edges)}")
            print(f"  Hole edges: {len(hole_edges)}")
            print(f"  Cutting edges: {len(cutting_edges)}")
            for i, (s, e) in enumerate(cutting_edges):
                print(f"    Cut {i}: ({s[0]:.1f}, {s[1]:.1f}) -> ({e[0]:.1f}, {e[1]:.1f})")

        # Step 2: Build vertex and edge lists
        all_edges = outer_edges + hole_edges + cutting_edges

        for start, end in all_edges:
            start_idx = self._add_vertex(start)
            end_idx = self._add_vertex(end)
            self._add_edge(start_idx, end_idx)

        if debug:
            print(f"  Total vertices: {len(self.vertices)}")
            print(f"  Total edges: {len(self.edges)}")

        # Step 3: Build adjacency - for each vertex, list edges in angular order
        self.vertex_edges = {i: [] for i in range(len(self.vertices))}

        for edge_idx, (start_idx, end_idx) in enumerate(self.edges):
            start = self.vertices[start_idx]
            end = self.vertices[end_idx]

            # Angle from start to end
            angle_forward = math.atan2(end[1] - start[1], end[0] - start[0])
            # Angle from end to start
            angle_backward = math.atan2(start[1] - end[1], start[0] - end[0])

            self.vertex_edges[start_idx].append((angle_forward, edge_idx, 'forward'))
            self.vertex_edges[end_idx].append((angle_backward, edge_idx, 'backward'))

        # Sort edges at each vertex by angle
        for v_idx in self.vertex_edges:
            self.vertex_edges[v_idx].sort(key=lambda x: x[0])

        if debug:
            print("  Vertex adjacency:")
            for v_idx, edges in self.vertex_edges.items():
                if len(edges) > 2:  # Only show interesting vertices
                    print(f"    Vertex {v_idx} ({self.vertices[v_idx][0]:.1f}, {self.vertices[v_idx][1]:.1f}): {len(edges)} edges")

        # Step 4: Trace regions
        self._trace_regions()

        return self.regions

    def _trace_regions(self):
        """Trace all region boundaries using the 'next CCW edge' rule."""
        # Track which (edge, direction) pairs have been used
        used = set()

        for edge_idx in range(len(self.edges)):
            for direction in ['forward', 'backward']:
                if (edge_idx, direction) in used:
                    continue

                # Trace a region starting from this edge
                region = self._trace_one_region(edge_idx, direction, used)
                if region and len(region) >= 3:
                    self.regions.append(region)

    def _trace_one_region(self, start_edge_idx, start_direction, used):
        """
        Trace one region boundary starting from given edge and direction.

        Uses the standard planar subdivision face tracing: at each vertex,
        take the next edge in CLOCKWISE order (the rightmost turn).
        This ensures we trace the face to the RIGHT of each directed edge.
        """
        boundary = []

        current_edge = start_edge_idx
        current_dir = start_direction

        max_steps = len(self.edges) * 2 + 10
        steps = 0

        while steps < max_steps:
            steps += 1

            if (current_edge, current_dir) in used:
                if current_edge == start_edge_idx and current_dir == start_direction and len(boundary) > 0:
                    # Completed the loop
                    break
                else:
                    # Hit an already-used edge - shouldn't happen in valid planar graph
                    break

            used.add((current_edge, current_dir))

            # Get start and end vertices based on direction
            edge_start, edge_end = self.edges[current_edge]
            if current_dir == 'forward':
                from_v, to_v = edge_start, edge_end
            else:
                from_v, to_v = edge_end, edge_start

            boundary.append(self.vertices[from_v])

            # Find next edge: at to_v, find the next edge in CLOCKWISE order
            # (this traces the face to the right of our current directed edge)
            incoming_angle = math.atan2(
                self.vertices[to_v][1] - self.vertices[from_v][1],
                self.vertices[to_v][0] - self.vertices[from_v][0]
            )

            # We want to find the edge that, when leaving to_v, is the "rightmost turn"
            # This is the edge whose angle is just BEFORE our reversed incoming angle (CW)
            # Reversed incoming = incoming + π = direction we came FROM as seen from to_v
            reversed_incoming = incoming_angle + math.pi

            edges_at_v = self.vertex_edges[to_v]
            if len(edges_at_v) == 0:
                break

            # Sort edges by their angular distance from reversed_incoming, going CW (decreasing angle)
            # The next edge CW is the one with largest angle that is still < reversed_incoming
            # Or if none, wrap around to the largest angle overall

            best_edge_idx = None
            best_dir = None
            best_angle_diff = float('inf')

            for angle, e_idx, e_dir in edges_at_v:
                # Skip the edge we just came from
                if e_idx == current_edge:
                    # Check if this is the reverse direction (we don't want to go back)
                    actual_dir = e_dir
                    if current_dir == 'forward' and actual_dir == 'backward':
                        continue
                    if current_dir == 'backward' and actual_dir == 'forward':
                        continue

                # Calculate CW distance from reversed_incoming to this edge's angle
                # CW distance = (reversed_incoming - angle) mod 2π
                diff = reversed_incoming - angle
                while diff < 0:
                    diff += 2 * math.pi
                while diff >= 2 * math.pi:
                    diff -= 2 * math.pi

                # We want the smallest positive CW distance (but > 0 to not include the reverse)
                if diff > 1e-9 and diff < best_angle_diff:
                    best_angle_diff = diff
                    best_edge_idx = e_idx
                    best_dir = e_dir

            if best_edge_idx is None:
                # Fallback: take any edge that's not going back
                for angle, e_idx, e_dir in edges_at_v:
                    if e_idx != current_edge:
                        best_edge_idx = e_idx
                        best_dir = e_dir
                        break
                if best_edge_idx is None:
                    break

            # Move to next edge
            current_edge = best_edge_idx
            current_dir = best_dir

            # Check if we've returned to start
            if current_edge == start_edge_idx and current_dir == start_direction:
                break

        return boundary


def create_parallel_cutting_lines(y1, y2, x_extent):
    """
    Create two horizontal parallel cutting lines at y=y1 and y=y2.

    Returns list of (line_eq, p1, p2) tuples.
    """
    lines = []

    # Line at y = y1: equation is 0*x + 1*y - y1 = 0 -> (0, 1, -y1)
    line1 = (0, 1, -y1)
    p1_1 = (x_extent[0] - 10, y1)
    p1_2 = (x_extent[1] + 10, y1)
    lines.append((line1, p1_1, p1_2))

    # Line at y = y2
    line2 = (0, 1, -y2)
    p2_1 = (x_extent[0] - 10, y2)
    p2_2 = (x_extent[1] + 10, y2)
    lines.append((line2, p2_1, p2_2))

    return lines


def region_centroid(region):
    """Calculate centroid of a region."""
    if len(region) == 0:
        return (0, 0)
    cx = sum(v[0] for v in region) / len(region)
    cy = sum(v[1] for v in region) / len(region)
    return (cx, cy)


def get_interior_test_point(region):
    """
    Get a point that is definitely inside the region.

    Uses the midpoint of the first edge, offset slightly inward.
    """
    if len(region) < 3:
        return region_centroid(region)

    # For a CCW polygon, "inward" is to the left of each edge
    # For a CW polygon, "inward" is to the right
    # We determine this by the signed area

    area = signed_area(region)

    # Try multiple edges to find a good test point
    for i in range(len(region)):
        p1 = region[i]
        p2 = region[(i + 1) % len(region)]

        # Midpoint of edge
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        # Edge direction
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-10:
            continue

        # Normal direction (perpendicular to edge)
        # For CCW polygon (positive area), inward is to the left: (-dy, dx)
        # For CW polygon (negative area), inward is to the right: (dy, -dx)
        if area > 0:
            nx, ny = -dy / length, dx / length
        else:
            nx, ny = dy / length, -dx / length

        # Offset slightly inward (1% of edge length or 0.1 units)
        offset = min(length * 0.01, 0.1)
        test_x = mid_x + nx * offset
        test_y = mid_y + ny * offset

        # Verify this point is inside the region
        if point_in_polygon((test_x, test_y), region):
            return (test_x, test_y)

    # Fallback to centroid
    return region_centroid(region)


def filter_valid_board_regions(regions, outer, holes):
    """
    Filter regions to only include valid board material regions.

    A valid region must:
    - Have positive signed area (CCW winding = filled region)
    - Have a test point that is inside the outer boundary
    - Have a test point that is outside all holes
    """
    outer_ccw = ensure_ccw(outer)
    holes_cw = [ensure_cw(h) for h in holes]

    valid_regions = []
    for region in regions:
        if len(region) < 3:
            continue

        # Only keep CCW regions (positive area = filled interior)
        area = signed_area(region)
        if area <= 0:
            continue

        # Use an interior test point
        test_point = get_interior_test_point(region)

        # Must be inside outer boundary
        if not point_in_polygon(test_point, outer_ccw):
            continue

        # Must be outside all holes
        inside_hole = False
        for hole in holes_cw:
            if point_in_polygon(test_point, hole):
                inside_hole = True
                break

        if not inside_hole:
            valid_regions.append(region)

    return valid_regions


def hole_crosses_cutting_lines(hole, cutting_lines):
    """Check if a hole crosses any of the cutting lines."""
    for line_eq, _, _ in cutting_lines:
        # Check if any edge of the hole crosses the line
        n = len(hole)
        for i in range(n):
            seg_start = hole[i]
            seg_end = hole[(i + 1) % n]
            d1 = signed_distance_to_line(seg_start, line_eq)
            d2 = signed_distance_to_line(seg_end, line_eq)
            # If signs differ, the edge crosses the line
            if d1 * d2 < -1e-10:
                return True
    return False


def associate_holes_with_regions(valid_regions, original_holes, cutting_lines):
    """
    Associate holes that don't cross cutting lines with their containing region.

    Returns a list of (region, [holes]) tuples.
    """
    # For each region, find holes that are entirely inside it
    region_holes = [[] for _ in valid_regions]

    for hole in original_holes:
        # Skip holes that cross cutting lines - they're already incorporated
        if hole_crosses_cutting_lines(hole, cutting_lines):
            continue

        # Find which region contains this hole
        hole_centroid = region_centroid(hole)

        for i, region in enumerate(valid_regions):
            if point_in_polygon(hole_centroid, region):
                region_holes[i].append(hole)
                break

    return list(zip(valid_regions, region_holes))


def triangulate_regions(regions_with_holes):
    """
    Triangulate each region with its associated holes.

    Args:
        regions_with_holes: List of (region_boundary, [holes]) tuples

    Returns:
        List of (region_boundary, holes, triangles, vertices) tuples
    """
    results = []

    for region, holes in regions_with_holes:
        if len(region) < 3:
            continue

        # Triangulate using ear clipping with holes
        triangles, merged_polygon = triangulate_with_holes(region, holes)

        results.append({
            'boundary': region,
            'holes': holes,
            'triangles': triangles,
            'vertices': merged_polygon,
            'area': signed_area(region)
        })

    return results


def plot_triangulated_regions(triangulation_results, cutting_lines, title="Triangulated Regions", filename=None):
    """Plot triangulated regions with different colors per region."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Color palette for regions
    region_colors = plt.cm.tab10(np.linspace(0, 1, max(len(triangulation_results), 10)))

    for region_idx, result in enumerate(triangulation_results):
        vertices = result['vertices']
        triangles = result['triangles']
        region_color = region_colors[region_idx % len(region_colors)]

        # Plot triangles
        for tri_idx, (a, b, c) in enumerate(triangles):
            tri_verts = [vertices[a], vertices[b], vertices[c]]
            # Slightly vary the color for each triangle
            shade = 0.7 + 0.3 * (tri_idx % 3) / 3
            tri = MplPolygon(tri_verts, closed=True,
                           facecolor=(*region_color[:3], 0.4 * shade),
                           edgecolor=(*region_color[:3], 0.8),
                           linewidth=0.5)
            ax.add_patch(tri)

        # Plot region boundary
        boundary = result['boundary']
        boundary_closed = list(boundary) + [boundary[0]]
        xs, ys = zip(*boundary_closed)
        ax.plot(xs, ys, '-', color=region_color, linewidth=2,
                label=f'Region {region_idx+1} ({len(triangles)} tris)')

        # Plot holes
        for hole in result['holes']:
            hole_closed = list(hole) + [hole[0]]
            xs, ys = zip(*hole_closed)
            ax.plot(xs, ys, '--', color=region_color, linewidth=1.5)

    # Plot cutting lines
    line_labels_added = False
    for line_eq, p1, p2 in cutting_lines:
        label = 'Cutting line' if not line_labels_added else None
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=2, label=label)
        line_labels_added = True

    ax.set_aspect('equal')
    total_tris = sum(len(r['triangles']) for r in triangulation_results)
    ax.set_title(f"{title} ({len(triangulation_results)} regions, {total_tris} triangles)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.autoscale()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved to {filename}")
    plt.show()


def plot_regions(regions, cutting_lines, title="Regions", filename=None, outer=None, holes=None):
    """Plot regions with different colors and show cutting lines."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Color palette for regions
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(regions), 10)))

    for i, region in enumerate(regions):
        if len(region) < 3:
            continue

        # Calculate region area
        area = signed_area(region)

        # Plot region
        region_closed = list(region) + [region[0]]
        xs, ys = zip(*region_closed)

        color = colors[i % len(colors)]
        ax.fill(xs, ys, alpha=0.4, color=color, label=f'Region {i+1} (area={area:.1f})')
        ax.plot(xs, ys, '-', color=color, linewidth=2)

        # Mark vertices with small dots
        for v in region:
            ax.plot(v[0], v[1], 'o', color=color, markersize=4)

        # Mark centroid
        centroid = region_centroid(region)
        ax.plot(centroid[0], centroid[1], 'x', color=color, markersize=10, markeredgewidth=2)

    # Plot cutting lines
    line_labels_added = False
    for line_eq, p1, p2 in cutting_lines:
        label = 'Cutting line' if not line_labels_added else None
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=2, label=label)
        line_labels_added = True

    # Plot outer boundary if provided
    if outer is not None:
        outer_closed = list(outer) + [outer[0]]
        xs, ys = zip(*outer_closed)
        ax.plot(xs, ys, 'b-', linewidth=3, alpha=0.5, label='Outer boundary')

    # Plot holes if provided
    if holes is not None:
        for hole in holes:
            hole_closed = list(hole) + [hole[0]]
            xs, ys = zip(*hole_closed)
            ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.5)

    ax.set_aspect('equal')
    ax.set_title(f"{title} ({len(regions)} regions)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.autoscale()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved to {filename}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    # Generate outer polygon (concave)
    outer = generate_random_concave_polygon(
        center=(50, 50),
        radius=40,
        num_vertices=12,
        concavity=0.9
    )

    print(f"Outer polygon: {len(outer)} vertices")

    # Generate two holes
    holes = []
    hole1 = generate_hole_inside_polygon(outer, hole_radius=8)
    if hole1:
        holes.append(hole1)
        print(f"Hole 1: {len(hole1)} vertices")

    hole2 = generate_hole_inside_polygon(outer, hole_radius=6)
    if hole2:
        # Check distance from hole1
        if hole1:
            c1 = (sum(v[0] for v in hole1)/len(hole1), sum(v[1] for v in hole1)/len(hole1))
            c2 = (sum(v[0] for v in hole2)/len(hole2), sum(v[1] for v in hole2)/len(hole2))
            dist = math.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)
            if dist > 15:
                holes.append(hole2)
                print(f"Hole 2: {len(hole2)} vertices")
        else:
            holes.append(hole2)
            print(f"Hole 2: {len(hole2)} vertices")

    print(f"\nTotal holes: {len(holes)}")
    original_holes = holes.copy()  # Save a copy for later use

    # Plot original
    plot_polygon_with_holes(outer, holes, "Board with Holes", "/tmp/01_board_outline.png")

    # Triangulate
    triangles, merged = triangulate_with_holes(outer, holes)

    print(f"\nMerged polygon: {len(merged)} vertices")
    print(f"Triangles: {len(triangles)}")

    # Plot triangulation
    plot_triangulation(merged, triangles, "Triangulation", "/tmp/02_triangulation.png")

    # Verify: check that no triangle centroid is inside a hole
    holes_cw = [ensure_cw(h) for h in holes]
    bad_triangles = 0
    for tri in triangles:
        v0, v1, v2 = merged[tri[0]], merged[tri[1]], merged[tri[2]]
        cx = (v0[0] + v1[0] + v2[0]) / 3
        cy = (v0[1] + v1[1] + v2[1]) / 3
        for hole in holes_cw:
            if point_in_polygon((cx, cy), hole):
                bad_triangles += 1
                print(f"  BAD: Triangle centroid ({cx:.2f}, {cy:.2f}) inside hole")
                break

    print(f"\nBad triangles (inside holes): {bad_triangles}")
    print(f"Result: {'PASS' if bad_triangles == 0 else 'FAIL'}")

    # =========================================================================
    # Test Planar Subdivision with Cutting Lines
    # =========================================================================
    print("\n" + "="*60)
    print("Testing Planar Subdivision with Cutting Lines")
    print("="*60)

    # Create a simple rectangular board with a cutout
    board_outer = [
        (10, 10),
        (90, 10),
        (90, 70),
        (10, 70)
    ]

    # Cutout that spans across one cutting line
    board_hole = [
        (35, 35),
        (55, 35),
        (55, 55),
        (35, 55)
    ]

    # Create two parallel horizontal cutting lines
    x_extent = (0, 100)
    cutting_lines = create_parallel_cutting_lines(y1=30, y2=60, x_extent=x_extent)

    print(f"Board outer: {len(board_outer)} vertices")
    print(f"Board hole: {len(board_hole)} vertices")
    print(f"Cutting lines: {len(cutting_lines)}")
    print(f"  Line 1: y = 30")
    print(f"  Line 2: y = 60")

    # Compute planar subdivision
    subdivision = PlanarSubdivision(board_outer, [board_hole], cutting_lines)
    regions = subdivision.compute(debug=False)

    print(f"\nComputed {len(regions)} total regions:")
    for i, region in enumerate(regions):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    # Filter to valid board regions only
    valid_regions = filter_valid_board_regions(regions, board_outer, [board_hole])
    print(f"\nFiltered to {len(valid_regions)} valid board regions:")
    for i, region in enumerate(valid_regions):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    # Plot the regions
    plot_regions(valid_regions, cutting_lines, "Board with Cutting Lines",
                 "/tmp/03_regions.png", outer=board_outer, holes=[board_hole])

    # Associate holes and triangulate
    regions_with_holes = associate_holes_with_regions(valid_regions, [board_hole], cutting_lines)
    print(f"\nHole association:")
    for i, (region, holes) in enumerate(regions_with_holes):
        print(f"  Region {i+1}: {len(holes)} holes")

    triangulation_results = triangulate_regions(regions_with_holes)
    print(f"\nTriangulation:")
    for i, result in enumerate(triangulation_results):
        print(f"  Region {i+1}: {len(result['triangles'])} triangles")

    plot_triangulated_regions(triangulation_results, cutting_lines,
                              "Triangulated Board", "/tmp/03_triangulated.png")

    # =========================================================================
    # Test with hole crossing a cutting line
    # =========================================================================
    print("\n" + "="*60)
    print("Testing with Hole Crossing Cutting Line")
    print("="*60)

    # Cutout that crosses the lower cutting line (y=30)
    board_hole_crossing = [
        (60, 20),
        (80, 20),
        (80, 45),
        (60, 45)
    ]

    subdivision2 = PlanarSubdivision(board_outer, [board_hole_crossing], cutting_lines)
    regions2 = subdivision2.compute()

    print(f"\nComputed {len(regions2)} total regions:")
    for i, region in enumerate(regions2):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    valid_regions2 = filter_valid_board_regions(regions2, board_outer, [board_hole_crossing])
    print(f"\nFiltered to {len(valid_regions2)} valid board regions:")
    for i, region in enumerate(valid_regions2):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    plot_regions(valid_regions2, cutting_lines, "Hole Crossing Cutting Line",
                 "/tmp/04_regions_crossing.png", outer=board_outer, holes=[board_hole_crossing])

    # =========================================================================
    # Test with multiple holes
    # =========================================================================
    print("\n" + "="*60)
    print("Testing with Multiple Holes")
    print("="*60)

    holes_multi = [
        [(20, 15), (30, 15), (30, 25), (20, 25)],  # Below first cut
        [(60, 20), (75, 20), (75, 45), (60, 45)],  # Crosses first cut
        [(25, 62), (40, 62), (40, 68), (25, 68)],  # Above second cut
    ]

    subdivision3 = PlanarSubdivision(board_outer, holes_multi, cutting_lines)
    regions3 = subdivision3.compute()

    print(f"\nComputed {len(regions3)} total regions:")
    for i, region in enumerate(regions3):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    valid_regions3 = filter_valid_board_regions(regions3, board_outer, holes_multi)
    print(f"\nFiltered to {len(valid_regions3)} valid board regions:")
    for i, region in enumerate(valid_regions3):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    plot_regions(valid_regions3, cutting_lines, "Multiple Holes",
                 "/tmp/05_regions_multi.png", outer=board_outer, holes=holes_multi)

    # =========================================================================
    # Test with Cutout Spanning Both Cutting Lines
    # =========================================================================
    print("\n" + "="*60)
    print("Testing with Cutout Spanning Both Cutting Lines")
    print("="*60)

    # Cutout that spans both cutting lines (y=30 and y=60)
    # This should split the middle region into left and right parts
    board_hole_spanning = [
        (40, 20),   # Below first cut
        (60, 20),
        (60, 65),   # Above second cut
        (40, 65)
    ]

    print(f"Hole spans from y=20 to y=65 (crosses both y=30 and y=60)")

    subdivision4 = PlanarSubdivision(board_outer, [board_hole_spanning], cutting_lines)
    regions4 = subdivision4.compute(debug=False)

    print(f"\nComputed {len(regions4)} total regions:")
    for i, region in enumerate(regions4):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    valid_regions4 = filter_valid_board_regions(regions4, board_outer, [board_hole_spanning])
    print(f"\nFiltered to {len(valid_regions4)} valid board regions:")
    for i, region in enumerate(valid_regions4):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    plot_regions(valid_regions4, cutting_lines, "Cutout Spanning Both Lines",
                 "/tmp/06_regions_spanning.png", outer=board_outer, holes=[board_hole_spanning])

    # Associate holes and triangulate
    regions_with_holes4 = associate_holes_with_regions(valid_regions4, [board_hole_spanning], cutting_lines)
    print(f"\nHole association:")
    for i, (region, holes) in enumerate(regions_with_holes4):
        print(f"  Region {i+1}: {len(holes)} holes")

    triangulation_results4 = triangulate_regions(regions_with_holes4)
    print(f"\nTriangulation:")
    for i, result in enumerate(triangulation_results4):
        print(f"  Region {i+1}: {len(result['triangles'])} triangles")

    plot_triangulated_regions(triangulation_results4, cutting_lines,
                              "Triangulated Spanning Cutout", "/tmp/06_triangulated.png")

    # =========================================================================
    # Test on Original Random Board with Holes
    # =========================================================================
    print("\n" + "="*60)
    print("Testing on Original Random Board with Holes")
    print("="*60)

    # Use the random board generated earlier
    print(f"Outer polygon: {len(outer)} vertices")
    print(f"Holes: {len(original_holes)}")

    # Create cutting lines based on the board's bounding box
    min_y = min(v[1] for v in outer)
    max_y = max(v[1] for v in outer)
    min_x = min(v[0] for v in outer)
    max_x = max(v[0] for v in outer)

    # Two horizontal cutting lines at 1/3 and 2/3 of the board height
    y1 = min_y + (max_y - min_y) * 0.35
    y2 = min_y + (max_y - min_y) * 0.65

    print(f"Board Y range: {min_y:.1f} to {max_y:.1f}")
    print(f"Cutting lines at y={y1:.1f} and y={y2:.1f}")

    cutting_lines_random = create_parallel_cutting_lines(y1, y2, (min_x - 10, max_x + 10))

    subdivision5 = PlanarSubdivision(outer, original_holes, cutting_lines_random)
    regions5 = subdivision5.compute(debug=False)

    print(f"\nComputed {len(regions5)} total regions:")
    for i, region in enumerate(regions5):
        area = signed_area(region)
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")

    valid_regions5 = filter_valid_board_regions(regions5, outer, original_holes)
    print(f"\nFiltered to {len(valid_regions5)} valid board regions:")
    total_area = 0
    for i, region in enumerate(valid_regions5):
        area = signed_area(region)
        total_area += area
        print(f"  Region {i+1}: {len(region)} vertices, area = {area:.1f}")
    print(f"Total area of valid regions: {total_area:.1f}")

    plot_regions(valid_regions5, cutting_lines_random, "Random Board with Cutting Lines",
                 "/tmp/07_regions_random.png", outer=outer, holes=original_holes)

    # Associate holes and triangulate
    regions_with_holes5 = associate_holes_with_regions(valid_regions5, original_holes, cutting_lines_random)
    print(f"\nHole association:")
    for i, (region, region_holes) in enumerate(regions_with_holes5):
        print(f"  Region {i+1}: {len(region_holes)} holes")

    triangulation_results5 = triangulate_regions(regions_with_holes5)
    print(f"\nTriangulation:")
    for i, result in enumerate(triangulation_results5):
        print(f"  Region {i+1}: {len(result['triangles'])} triangles")

    plot_triangulated_regions(triangulation_results5, cutting_lines_random,
                              "Triangulated Random Board", "/tmp/07_triangulated.png")


if __name__ == "__main__":
    main()

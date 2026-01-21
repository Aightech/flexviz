"""
Mesh generation for 3D visualization.

Creates 3D meshes from board geometry with bend transformations applied.
"""

from dataclasses import dataclass, field
from typing import Optional
import math

try:
    from .geometry import (
        Polygon, LineSegment, BoardGeometry, PadGeometry,
        ComponentGeometry, subdivide_polygon, line_segment_to_ribbon,
        pad_to_polygon, component_to_box
    )
    from .bend_transform import FoldDefinition, transform_point, transform_polygon
    from .markers import FoldMarker
    from .region_splitter import split_board_into_regions, Region
except ImportError:
    from geometry import (
        Polygon, LineSegment, BoardGeometry, PadGeometry,
        ComponentGeometry, subdivide_polygon, line_segment_to_ribbon,
        pad_to_polygon, component_to_box
    )
    from bend_transform import FoldDefinition, transform_point, transform_polygon
    from markers import FoldMarker
    from region_splitter import split_board_into_regions, Region


# =============================================================================
# Triangulation - Ear Clipping Algorithm
# Based on "Triangulation by Ear Clipping" by David Eberly
# =============================================================================

def signed_area(polygon: list[tuple[float, float]]) -> float:
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


def ensure_ccw(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Ensure polygon is counter-clockwise ordered."""
    if signed_area(polygon) < 0:
        return list(reversed(polygon))
    return list(polygon)


def ensure_cw(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Ensure polygon is clockwise ordered."""
    if signed_area(polygon) > 0:
        return list(reversed(polygon))
    return list(polygon)


def cross_product_2d(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """
    Cross product of vectors OA and OB.
    Positive = B is to the left of OA (CCW turn)
    Negative = B is to the right of OA (CW turn)
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex_vertex(prev_v: tuple[float, float], curr_v: tuple[float, float], next_v: tuple[float, float]) -> bool:
    """
    Check if curr_v is a convex vertex (interior angle < 180 degrees).
    For CCW polygon, convex means cross product > 0.
    """
    return cross_product_2d(prev_v, curr_v, next_v) > 0


def is_reflex_vertex_pts(prev_v: tuple[float, float], curr_v: tuple[float, float], next_v: tuple[float, float]) -> bool:
    """
    Check if curr_v is a reflex vertex (interior angle > 180 degrees).
    For CCW polygon, reflex means cross product < 0.
    """
    return cross_product_2d(prev_v, curr_v, next_v) < 0


def point_in_triangle(p: tuple[float, float],
                      a: tuple[float, float],
                      b: tuple[float, float],
                      c: tuple[float, float]) -> bool:
    """Check if point p is inside triangle abc."""
    d1 = cross_product_2d(a, b, p)
    d2 = cross_product_2d(b, c, p)
    d3 = cross_product_2d(c, a, p)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
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


def find_mutually_visible_vertex(M: tuple[float, float],
                                  polygon: list[tuple[float, float]]) -> int:
    """
    Find a vertex in polygon that is mutually visible with point M.

    Implements the algorithm from Eberly's "Triangulation by Ear Clipping":
    1. Cast ray from M in +X direction
    2. Find closest intersection point I on polygon edge
    3. If I is a vertex, return that vertex
    4. Otherwise, P = endpoint of edge with max x-value
    5. If reflex vertices exist inside triangle <M, I, P>, return the one
       with minimum angle to ray direction
    6. Otherwise return P

    Args:
        M: The point (typically rightmost vertex of a hole)
        polygon: The outer polygon (CCW ordered)

    Returns:
        Index of the mutually visible vertex in polygon
    """
    mx, my = M
    n = len(polygon)

    # Step 1-2: Cast ray M + t*(1,0) and find closest intersection
    min_t = float('inf')
    hit_edge_idx = -1
    intersection_point = None

    for i in range(n):
        j = (i + 1) % n
        vi, vj = polygon[i], polygon[j]

        # Only consider edges where vi is below (or on) ray and vj is above (or on)
        if not ((vi[1] <= my < vj[1]) or (vj[1] <= my < vi[1])):
            continue

        # Skip horizontal edges
        if abs(vj[1] - vi[1]) < 1e-10:
            continue

        # Calculate intersection x
        t_edge = (my - vi[1]) / (vj[1] - vi[1])
        x_int = vi[0] + t_edge * (vj[0] - vi[0])

        # Must be to the right of M
        if x_int <= mx:
            continue

        t = x_int - mx  # Distance along ray

        if t < min_t:
            min_t = t
            hit_edge_idx = i
            intersection_point = (x_int, my)

    if hit_edge_idx == -1:
        # Fallback: find closest vertex to the right
        best_idx = 0
        best_dist = float('inf')
        for i, v in enumerate(polygon):
            if v[0] > mx:
                dist = (v[0] - mx) ** 2 + (v[1] - my) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        return best_idx

    I = intersection_point
    vi = polygon[hit_edge_idx]
    vj = polygon[(hit_edge_idx + 1) % n]

    # Step 3: Check if I is (very close to) a vertex
    eps = 1e-9
    if abs(I[0] - vi[0]) < eps and abs(I[1] - vi[1]) < eps:
        return hit_edge_idx
    if abs(I[0] - vj[0]) < eps and abs(I[1] - vj[1]) < eps:
        return (hit_edge_idx + 1) % n

    # Step 4: P = endpoint with maximum x-value
    if vi[0] > vj[0]:
        P_idx = hit_edge_idx
    else:
        P_idx = (hit_edge_idx + 1) % n
    P = polygon[P_idx]

    # Step 5: Find reflex vertices inside triangle <M, I, P>
    reflex_in_triangle = []
    for i in range(n):
        if i == P_idx:
            continue
        prev_v = polygon[(i - 1) % n]
        curr_v = polygon[i]
        next_v = polygon[(i + 1) % n]

        if is_reflex_vertex_pts(prev_v, curr_v, next_v):
            if point_in_triangle(curr_v, M, I, P):
                reflex_in_triangle.append(i)

    # Step 6: If no reflex vertices in triangle, P is visible
    if not reflex_in_triangle:
        return P_idx

    # Step 7: Find reflex vertex R that minimizes angle between <M,I> and <M,R>
    best_idx = reflex_in_triangle[0]
    best_angle = float('inf')

    for idx in reflex_in_triangle:
        R = polygon[idx]
        dx = R[0] - mx
        dy = abs(R[1] - my)
        if dx > eps:
            angle = dy / dx  # tan(angle) - smaller is better
            if angle < best_angle:
                best_angle = angle
                best_idx = idx

    return best_idx


def merge_hole_into_polygon(
    outer: list[tuple[float, float]],
    hole: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    Merge a hole into the outer polygon to create a pseudosimple polygon.

    From Section 3 of Eberly's paper:
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


def triangulate_with_holes(
    outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]]
) -> tuple[list[tuple[int, int, int]], list[tuple[float, float]]]:
    """
    Triangulate a polygon with holes using ear clipping.

    Algorithm from Eberly's "Triangulation by Ear Clipping" Section 5:
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
        return triangulate_polygon(outer_ccw), outer_ccw

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
    triangles = triangulate_polygon(result)

    return triangles, result


def find_reflex_vertices(polygon: list[tuple[float, float]]) -> list[int]:
    """Find all reflex vertex indices in a CCW polygon."""
    n = len(polygon)
    reflex = []
    for i in range(n):
        prev_v = polygon[(i - 1) % n]
        curr_v = polygon[i]
        next_v = polygon[(i + 1) % n]
        if is_reflex_vertex_pts(prev_v, curr_v, next_v):
            reflex.append(i)
    return reflex


def triangulate_polygon(vertices: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """
    Triangulate a simple polygon using ear clipping.

    Based on Eberly's "Triangulation by Ear Clipping" algorithm.

    Args:
        vertices: List of (x, y) vertices in order

    Returns:
        List of triangles, each as (i, j, k) indices into vertices
    """
    polygon = ensure_ccw(vertices)
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
                    if is_reflex_vertex_pts(polygon[pp], polygon[prev_idx], polygon[pn]):
                        reflex_set.add(prev_idx)
                    else:
                        reflex_set.discard(prev_idx)

                    # Check if next vertex changed from reflex to convex
                    np_ = indices[(new_next_pos - 1) % len(indices)]
                    nn = indices[(new_next_pos + 1) % len(indices)]
                    if is_reflex_vertex_pts(polygon[np_], polygon[next_idx], polygon[nn]):
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


@dataclass
class Mesh:
    """
    A 3D mesh with vertices and faces.

    Vertices are (x, y, z) tuples.
    Faces are lists of vertex indices (triangles or quads).
    """
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[list[int]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)

    # Optional per-face colors (r, g, b) 0-255
    colors: list[tuple[int, int, int]] = field(default_factory=list)

    def add_vertex(self, v: tuple[float, float, float]) -> int:
        """Add a vertex and return its index."""
        self.vertices.append(v)
        return len(self.vertices) - 1

    def add_face(self, indices: list[int], color: tuple[int, int, int] = None):
        """Add a face (list of vertex indices)."""
        self.faces.append(indices)
        if color:
            self.colors.append(color)

    def add_triangle(self, v0: int, v1: int, v2: int, color: tuple[int, int, int] = None):
        """Add a triangle face."""
        self.add_face([v0, v1, v2], color)

    def add_quad(self, v0: int, v1: int, v2: int, v3: int, color: tuple[int, int, int] = None):
        """Add a quad face (will be split into two triangles for some formats)."""
        self.add_face([v0, v1, v2, v3], color)

    def merge(self, other: 'Mesh'):
        """Merge another mesh into this one."""
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        for face in other.faces:
            self.faces.append([i + offset for i in face])
        self.colors.extend(other.colors)

    def compute_normals(self):
        """Compute face normals."""
        self.normals = []
        for face in self.faces:
            if len(face) >= 3:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]

                # Two edge vectors
                e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
                e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])

                # Cross product
                nx = e1[1] * e2[2] - e1[2] * e2[1]
                ny = e1[2] * e2[0] - e1[0] * e2[2]
                nz = e1[0] * e2[1] - e1[1] * e2[0]

                # Normalize
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 1e-10:
                    self.normals.append((nx/length, ny/length, nz/length))
                else:
                    self.normals.append((0, 0, 1))
            else:
                self.normals.append((0, 0, 1))

    def to_obj(self, filename: str):
        """Export mesh to OBJ file format."""
        with open(filename, 'w') as f:
            f.write("# KiCad Flex Viewer - OBJ Export\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n\n")

            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # Write faces (OBJ uses 1-based indexing)
            for face in self.faces:
                indices = " ".join(str(i + 1) for i in face)
                f.write(f"f {indices}\n")

    def to_stl(self, filename: str):
        """Export mesh to STL file format (ASCII)."""
        if not self.normals:
            self.compute_normals()

        with open(filename, 'w') as f:
            f.write("solid kicad_flex_viewer\n")

            for i, face in enumerate(self.faces):
                if len(face) < 3:
                    continue

                normal = self.normals[i] if i < len(self.normals) else (0, 0, 1)

                # Triangulate if needed (fan triangulation for convex polygons)
                for j in range(1, len(face) - 1):
                    v0 = self.vertices[face[0]]
                    v1 = self.vertices[face[j]]
                    v2 = self.vertices[face[j + 1]]

                    f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                    f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                    f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")

            f.write("endsolid kicad_flex_viewer\n")


# Color constants (RGB 0-255)
COLOR_BOARD = (34, 139, 34)      # Forest green (PCB color)
COLOR_COPPER = (184, 115, 51)    # Copper/bronze
COLOR_PAD = (255, 215, 0)        # Gold
COLOR_COMPONENT = (128, 128, 128) # Gray


def create_board_mesh(
    outline: Polygon,
    thickness: float,
    folds: list[FoldDefinition] = None,
    subdivide_length: float = 1.0,
    cutouts: list[Polygon] = None
) -> Mesh:
    """
    Create a 3D mesh for the board outline.

    Args:
        outline: Board outline polygon
        thickness: Board thickness in mm
        folds: List of fold definitions to apply
        subdivide_length: Maximum edge length for subdivision
        cutouts: List of cutout polygons (holes in the board)

    Returns:
        Mesh representing the board
    """
    if not outline.vertices:
        return Mesh()

    mesh = Mesh()
    folds = folds or []
    cutouts = cutouts or []

    # Subdivide for smooth bending
    subdivided = subdivide_polygon(outline, subdivide_length)

    # Transform top surface vertices
    top_vertices = []
    for v in subdivided.vertices:
        if folds:
            v3d = transform_point(v, folds)
        else:
            v3d = (v[0], v[1], 0.0)
        top_vertices.append(v3d)

    # Transform bottom surface vertices (offset by thickness in local -Z)
    bottom_vertices = []
    for v in top_vertices:
        bottom_vertices.append((v[0], v[1], v[2] - thickness))

    # Add outline vertices to mesh
    n = len(top_vertices)
    top_indices = [mesh.add_vertex(v) for v in top_vertices]
    bottom_indices = [mesh.add_vertex(v) for v in bottom_vertices]

    # Process cutouts - transform vertices and add to mesh
    cutout_top_indices = []
    cutout_bottom_indices = []
    cutout_2d_verts = []

    for cutout in cutouts:
        cutout_top_verts = []
        cutout_bottom_verts = []
        cutout_2d = []

        for v in cutout.vertices:
            if folds:
                v3d = transform_point(v, folds)
            else:
                v3d = (v[0], v[1], 0.0)
            cutout_top_verts.append(v3d)
            cutout_bottom_verts.append((v3d[0], v3d[1], v3d[2] - thickness))
            cutout_2d.append((v[0], v[1]))

        # Add cutout vertices to mesh
        ct_indices = [mesh.add_vertex(v) for v in cutout_top_verts]
        cb_indices = [mesh.add_vertex(v) for v in cutout_bottom_verts]

        cutout_top_indices.append(ct_indices)
        cutout_bottom_indices.append(cb_indices)
        cutout_2d_verts.append(cutout_2d)

    # Create top/bottom faces with holes
    # Use 2D coordinates for triangulation
    outline_2d = [(v[0], v[1]) for v in subdivided.vertices]

    if cutouts:
        # Triangulate with hole filtering (removes triangles inside holes)
        triangles, merged = triangulate_with_holes(outline_2d, cutout_2d_verts)

        # Build a mapping from merged 2D vertices to mesh vertex indices
        # First, create lookup for outline and cutout vertices
        all_2d = list(outline_2d)
        all_top_indices = list(top_indices)
        all_bottom_indices = list(bottom_indices)

        for cutout_2d, ct_idx, cb_idx in zip(cutout_2d_verts, cutout_top_indices, cutout_bottom_indices):
            all_2d.extend(cutout_2d)
            all_top_indices.extend(ct_idx)
            all_bottom_indices.extend(cb_idx)

        # Map merged vertices to mesh indices
        merged_top_indices = []
        merged_bottom_indices = []
        for v2d in merged:
            # Find matching vertex in all_2d
            found = False
            for i, av in enumerate(all_2d):
                if abs(av[0] - v2d[0]) < 0.001 and abs(av[1] - v2d[1]) < 0.001:
                    merged_top_indices.append(all_top_indices[i])
                    merged_bottom_indices.append(all_bottom_indices[i])
                    found = True
                    break
            if not found:
                # Bridge vertices - create new mesh vertices
                if folds:
                    v3d = transform_point(v2d, folds)
                else:
                    v3d = (v2d[0], v2d[1], 0.0)
                ti = mesh.add_vertex(v3d)
                bi = mesh.add_vertex((v3d[0], v3d[1], v3d[2] - thickness))
                merged_top_indices.append(ti)
                merged_bottom_indices.append(bi)

        # Create top faces
        for tri in triangles:
            mesh.add_triangle(
                merged_top_indices[tri[0]],
                merged_top_indices[tri[1]],
                merged_top_indices[tri[2]],
                COLOR_BOARD
            )

        # Create bottom faces (reversed winding)
        for tri in triangles:
            mesh.add_triangle(
                merged_bottom_indices[tri[0]],
                merged_bottom_indices[tri[2]],
                merged_bottom_indices[tri[1]],
                COLOR_BOARD
            )
    else:
        # No cutouts - simple triangulation
        triangles = triangulate_polygon(outline_2d)

        for tri in triangles:
            mesh.add_triangle(top_indices[tri[0]], top_indices[tri[1]], top_indices[tri[2]], COLOR_BOARD)

        for tri in triangles:
            mesh.add_triangle(bottom_indices[tri[0]], bottom_indices[tri[2]], bottom_indices[tri[1]], COLOR_BOARD)

    # Create side faces for outline (quads connecting top and bottom)
    for i in range(n):
        j = (i + 1) % n
        mesh.add_quad(
            top_indices[i], top_indices[j],
            bottom_indices[j], bottom_indices[i],
            COLOR_BOARD
        )

    # Create side faces for cutouts (inner walls)
    for ct_indices, cb_indices in zip(cutout_top_indices, cutout_bottom_indices):
        nc = len(ct_indices)
        for i in range(nc):
            j = (i + 1) % nc
            # Reversed winding for inner walls (facing inward)
            mesh.add_quad(
                ct_indices[j], ct_indices[i],
                cb_indices[i], cb_indices[j],
                COLOR_CUTOUT
            )

    return mesh


def create_board_mesh_with_regions(
    outline: Polygon,
    thickness: float,
    folds: list[FoldDefinition] = None,
    markers: list[FoldMarker] = None,
    subdivide_length: float = 1.0,
    cutouts: list[Polygon] = None
) -> Mesh:
    """
    Create a 3D mesh for the board, split by fold regions.

    This function splits the board into regions along fold lines and
    triangulates each region separately. This ensures no triangles cross
    fold boundaries, preventing visual artifacts when bending.

    Args:
        outline: Board outline polygon
        thickness: Board thickness in mm
        folds: List of fold definitions to apply (for 3D transform)
        markers: List of fold markers (for region splitting)
        subdivide_length: Maximum edge length for subdivision
        cutouts: List of cutout polygons (holes in the board)

    Returns:
        Mesh representing the board
    """
    if not outline.vertices:
        return Mesh()

    # If no markers, fall back to standard meshing
    if not markers:
        return create_board_mesh(outline, thickness, folds, subdivide_length, cutouts)

    mesh = Mesh()
    folds = folds or []
    cutouts = cutouts or []

    # Convert outline and cutouts to lists of tuples
    outline_verts = [(v[0], v[1]) for v in outline.vertices]
    cutout_verts = [[(v[0], v[1]) for v in c.vertices] for c in cutouts]

    # Split into regions
    regions = split_board_into_regions(outline_verts, cutout_verts, markers)

    # Process each region
    for region in regions:
        # Subdivide the region outline
        region_poly = Polygon(region.outline)
        subdivided = subdivide_polygon(region_poly, subdivide_length)
        subdivided_verts = [(v[0], v[1]) for v in subdivided.vertices]

        # Subdivide holes in this region
        region_holes_2d = []
        for hole in region.holes:
            hole_poly = Polygon(hole)
            sub_hole = subdivide_polygon(hole_poly, subdivide_length)
            region_holes_2d.append([(v[0], v[1]) for v in sub_hole.vertices])

        # Transform vertices to 3D
        top_vertices = []
        for v in subdivided_verts:
            if folds:
                v3d = transform_point(v, folds)
            else:
                v3d = (v[0], v[1], 0.0)
            top_vertices.append(v3d)

        bottom_vertices = [(v[0], v[1], v[2] - thickness) for v in top_vertices]

        # Add outline vertices to mesh
        n = len(top_vertices)
        top_indices = [mesh.add_vertex(v) for v in top_vertices]
        bottom_indices = [mesh.add_vertex(v) for v in bottom_vertices]

        # Process holes for this region
        hole_top_indices = []
        hole_bottom_indices = []
        hole_2d_lists = []

        for hole_2d in region_holes_2d:
            hole_top_verts = []
            hole_bottom_verts = []

            for v in hole_2d:
                if folds:
                    v3d = transform_point(v, folds)
                else:
                    v3d = (v[0], v[1], 0.0)
                hole_top_verts.append(v3d)
                hole_bottom_verts.append((v3d[0], v3d[1], v3d[2] - thickness))

            ht_indices = [mesh.add_vertex(v) for v in hole_top_verts]
            hb_indices = [mesh.add_vertex(v) for v in hole_bottom_verts]

            hole_top_indices.append(ht_indices)
            hole_bottom_indices.append(hb_indices)
            hole_2d_lists.append(hole_2d)

        # Triangulate this region
        if region_holes_2d:
            triangles, merged = triangulate_with_holes(subdivided_verts, region_holes_2d)

            # Build vertex mapping
            all_2d = list(subdivided_verts)
            all_top = list(top_indices)
            all_bottom = list(bottom_indices)

            for h2d, ht_idx, hb_idx in zip(hole_2d_lists, hole_top_indices, hole_bottom_indices):
                all_2d.extend(h2d)
                all_top.extend(ht_idx)
                all_bottom.extend(hb_idx)

            # Map merged vertices to mesh indices
            merged_top = []
            merged_bottom = []
            for v2d in merged:
                found = False
                for i, av in enumerate(all_2d):
                    if abs(av[0] - v2d[0]) < 0.001 and abs(av[1] - v2d[1]) < 0.001:
                        merged_top.append(all_top[i])
                        merged_bottom.append(all_bottom[i])
                        found = True
                        break
                if not found:
                    if folds:
                        v3d = transform_point(v2d, folds)
                    else:
                        v3d = (v2d[0], v2d[1], 0.0)
                    ti = mesh.add_vertex(v3d)
                    bi = mesh.add_vertex((v3d[0], v3d[1], v3d[2] - thickness))
                    merged_top.append(ti)
                    merged_bottom.append(bi)

            # Add triangles
            for tri in triangles:
                mesh.add_triangle(merged_top[tri[0]], merged_top[tri[1]], merged_top[tri[2]], COLOR_BOARD)
            for tri in triangles:
                mesh.add_triangle(merged_bottom[tri[0]], merged_bottom[tri[2]], merged_bottom[tri[1]], COLOR_BOARD)
        else:
            # No holes in this region
            triangles = triangulate_polygon(subdivided_verts)
            for tri in triangles:
                mesh.add_triangle(top_indices[tri[0]], top_indices[tri[1]], top_indices[tri[2]], COLOR_BOARD)
            for tri in triangles:
                mesh.add_triangle(bottom_indices[tri[0]], bottom_indices[tri[2]], bottom_indices[tri[1]], COLOR_BOARD)

        # Side faces for region outline
        for i in range(n):
            j = (i + 1) % n
            mesh.add_quad(top_indices[i], top_indices[j], bottom_indices[j], bottom_indices[i], COLOR_BOARD)

        # Side faces for holes
        for ht_idx, hb_idx in zip(hole_top_indices, hole_bottom_indices):
            nh = len(ht_idx)
            for i in range(nh):
                j = (i + 1) % nh
                mesh.add_quad(ht_idx[j], ht_idx[i], hb_idx[i], hb_idx[j], COLOR_CUTOUT)

    return mesh


def create_trace_mesh(
    segment: LineSegment,
    z_offset: float,
    folds: list[FoldDefinition] = None,
    subdivisions: int = 20
) -> Mesh:
    """
    Create a 3D mesh for a copper trace.

    Args:
        segment: Trace line segment with width
        z_offset: Z offset for the trace (on top of board)
        folds: List of fold definitions
        subdivisions: Number of subdivisions along the trace

    Returns:
        Mesh representing the trace
    """
    mesh = Mesh()
    folds = folds or []

    # Convert segment to ribbon polygon
    ribbon = line_segment_to_ribbon(segment)

    # Get the 4 corners of the ribbon
    if len(ribbon.vertices) != 4:
        return mesh

    # Subdivide the ribbon along its length
    v0, v1, v2, v3 = ribbon.vertices

    # v0-v1 and v3-v2 are the long edges (along trace)
    # v0-v3 and v1-v2 are the short edges (across trace)

    # Create subdivided points along both long edges
    edge1_points = []  # v0 to v1
    edge2_points = []  # v3 to v2

    for i in range(subdivisions + 1):
        t = i / subdivisions
        p1 = (v0[0] + t * (v1[0] - v0[0]), v0[1] + t * (v1[1] - v0[1]))
        p2 = (v3[0] + t * (v2[0] - v3[0]), v3[1] + t * (v2[1] - v3[1]))

        # Transform to 3D
        if folds:
            p1_3d = transform_point(p1, folds)
            p2_3d = transform_point(p2, folds)
        else:
            p1_3d = (p1[0], p1[1], 0.0)
            p2_3d = (p2[0], p2[1], 0.0)

        # Add z offset
        p1_3d = (p1_3d[0], p1_3d[1], p1_3d[2] + z_offset)
        p2_3d = (p2_3d[0], p2_3d[1], p2_3d[2] + z_offset)

        edge1_points.append(p1_3d)
        edge2_points.append(p2_3d)

    # Add vertices
    edge1_indices = [mesh.add_vertex(p) for p in edge1_points]
    edge2_indices = [mesh.add_vertex(p) for p in edge2_points]

    # Create quads between the two edges
    for i in range(subdivisions):
        mesh.add_quad(
            edge1_indices[i], edge1_indices[i + 1],
            edge2_indices[i + 1], edge2_indices[i],
            COLOR_COPPER
        )

    return mesh


def create_pad_mesh(
    pad: PadGeometry,
    z_offset: float,
    folds: list[FoldDefinition] = None
) -> Mesh:
    """
    Create a 3D mesh for a pad.

    Args:
        pad: Pad geometry
        z_offset: Z offset for the pad
        folds: List of fold definitions

    Returns:
        Mesh representing the pad
    """
    mesh = Mesh()
    folds = folds or []

    # Convert pad to polygon
    poly = pad_to_polygon(pad)

    # Transform vertices
    vertices_3d = []
    for v in poly.vertices:
        if folds:
            v3d = transform_point(v, folds)
        else:
            v3d = (v[0], v[1], 0.0)
        vertices_3d.append((v3d[0], v3d[1], v3d[2] + z_offset))

    # Add vertices
    indices = [mesh.add_vertex(v) for v in vertices_3d]

    # Create face (fan triangulation)
    n = len(indices)
    for i in range(1, n - 1):
        mesh.add_triangle(indices[0], indices[i], indices[i + 1], COLOR_PAD)

    return mesh


def create_component_mesh(
    component: ComponentGeometry,
    height: float,
    folds: list[FoldDefinition] = None
) -> Mesh:
    """
    Create a 3D mesh for a component (as a box).

    Args:
        component: Component geometry
        height: Component height
        folds: List of fold definitions

    Returns:
        Mesh representing the component
    """
    mesh = Mesh()
    folds = folds or []

    # Get bounding box polygon
    box = component_to_box(component)

    # Transform bottom vertices
    bottom_3d = []
    for v in box.vertices:
        if folds:
            v3d = transform_point(v, folds)
        else:
            v3d = (v[0], v[1], 0.0)
        bottom_3d.append(v3d)

    # Create top vertices (offset by height in local up direction)
    # Simplified: just offset in global Z
    top_3d = [(v[0], v[1], v[2] + height) for v in bottom_3d]

    # Add vertices
    n = len(bottom_3d)
    bottom_indices = [mesh.add_vertex(v) for v in bottom_3d]
    top_indices = [mesh.add_vertex(v) for v in top_3d]

    # Bottom face
    for i in range(1, n - 1):
        mesh.add_triangle(bottom_indices[0], bottom_indices[i + 1], bottom_indices[i], COLOR_COMPONENT)

    # Top face
    for i in range(1, n - 1):
        mesh.add_triangle(top_indices[0], top_indices[i], top_indices[i + 1], COLOR_COMPONENT)

    # Side faces
    for i in range(n):
        j = (i + 1) % n
        mesh.add_quad(
            bottom_indices[i], bottom_indices[j],
            top_indices[j], top_indices[i],
            COLOR_COMPONENT
        )

    return mesh


# Color for cutout walls (darker than board)
COLOR_CUTOUT = (25, 100, 25)


def create_cutout_mesh(
    cutout: Polygon,
    thickness: float,
    folds: list[FoldDefinition] = None
) -> Mesh:
    """
    Create a 3D mesh for a cutout hole (cylinder walls only).

    Args:
        cutout: Cutout polygon (approximated circle)
        thickness: Board thickness in mm
        folds: List of fold definitions

    Returns:
        Mesh representing the cutout walls
    """
    if len(cutout.vertices) < 3:
        return Mesh()

    mesh = Mesh()
    folds = folds or []

    # Transform vertices to 3D
    top_vertices = []
    for v in cutout.vertices:
        if folds:
            v3d = transform_point(v, folds)
        else:
            v3d = (v[0], v[1], 0.0)
        top_vertices.append(v3d)

    # Bottom vertices (offset by thickness)
    bottom_vertices = []
    for v in top_vertices:
        bottom_vertices.append((v[0], v[1], v[2] - thickness))

    # Add vertices to mesh
    n = len(top_vertices)
    top_indices = [mesh.add_vertex(v) for v in top_vertices]
    bottom_indices = [mesh.add_vertex(v) for v in bottom_vertices]

    # Create side faces (cylinder walls)
    # Note: facing inward so winding is reversed compared to board outline
    for i in range(n):
        j = (i + 1) % n
        mesh.add_quad(
            top_indices[j], top_indices[i],
            bottom_indices[i], bottom_indices[j],
            COLOR_CUTOUT
        )

    return mesh


def create_board_geometry_mesh(
    board: BoardGeometry,
    folds: list[FoldDefinition] = None,
    markers: list[FoldMarker] = None,
    include_traces: bool = True,
    include_pads: bool = True,
    include_components: bool = False,
    component_height: float = 2.0,
    subdivide_length: float = 1.0
) -> Mesh:
    """
    Create a complete 3D mesh from board geometry.

    Args:
        board: Board geometry
        folds: List of fold definitions (for 3D transform)
        markers: List of fold markers (for region-based triangulation)
        include_traces: Include copper traces
        include_pads: Include pads
        include_components: Include component boxes
        component_height: Height for component boxes
        subdivide_length: Maximum edge length for subdivision

    Returns:
        Complete mesh
    """
    mesh = Mesh()
    folds = folds or []

    # Board outline with cutouts - use region-based meshing if markers provided
    if board.outline.vertices:
        board_mesh = create_board_mesh_with_regions(
            board.outline,
            board.thickness,
            folds,
            markers=markers,
            subdivide_length=subdivide_length,
            cutouts=board.cutouts
        )
        mesh.merge(board_mesh)

    # Traces
    if include_traces:
        z_offset = 0.01  # Slightly above board surface
        for layer, traces in board.traces.items():
            for trace in traces:
                trace_mesh = create_trace_mesh(trace, z_offset, folds)
                mesh.merge(trace_mesh)

    # Pads
    if include_pads:
        z_offset = 0.02  # Above traces
        for pad in board.all_pads:
            pad_mesh = create_pad_mesh(pad, z_offset, folds)
            mesh.merge(pad_mesh)

    # Components
    if include_components:
        for comp in board.components:
            comp_mesh = create_component_mesh(comp, component_height, folds)
            mesh.merge(comp_mesh)

    mesh.compute_normals()
    return mesh

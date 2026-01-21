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


if __name__ == "__main__":
    main()

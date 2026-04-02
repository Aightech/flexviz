"""
Trace, pad, component, and stiffener mesh generation.

Creates 3D meshes for copper traces, pads, component boxes/3D models,
and stiffener regions with bend transformations applied.
"""

import math

try:
    from .geometry import (
        Polygon, LineSegment, BoardGeometry, PadGeometry,
        ComponentGeometry, subdivide_polygon, line_segment_to_ribbon,
        pad_to_polygon, component_to_box
    )
    from .bend_transform import FoldDefinition, FoldRecipe, transform_point, transform_point_and_normal, recipe_from_region
    from .markers import FoldMarker
    from .planar_subdivision import Region, find_containing_region, build_region_index
    from .model_loader import load_model, expand_kicad_vars, get_loader_status
    from .triangulation import triangulate_polygon, triangulate_with_holes
    from .mesh_types import (
        Mesh, COLOR_COPPER, COLOR_PAD, COLOR_COMPONENT, COLOR_STIFFENER,
        COLOR_CUTOUT, COLOR_MODEL_3D, get_region_recipe, PrecomputedTraceData
    )
except ImportError:
    from geometry import (
        Polygon, LineSegment, BoardGeometry, PadGeometry,
        ComponentGeometry, subdivide_polygon, line_segment_to_ribbon,
        pad_to_polygon, component_to_box
    )
    from bend_transform import FoldDefinition, FoldRecipe, transform_point, transform_point_and_normal, recipe_from_region
    from markers import FoldMarker
    from planar_subdivision import Region, find_containing_region, build_region_index
    from model_loader import load_model, expand_kicad_vars, get_loader_status
    from triangulation import triangulate_polygon, triangulate_with_holes
    from mesh_types import (
        Mesh, COLOR_COPPER, COLOR_PAD, COLOR_COMPONENT, COLOR_STIFFENER,
        COLOR_CUTOUT, COLOR_MODEL_3D, get_region_recipe, PrecomputedTraceData
    )


def _compute_fold_crossing_t_values(
    start: tuple[float, float],
    end: tuple[float, float],
    markers: list[FoldMarker],
    num_bend_subdivisions: int = 1
) -> list[float]:
    """
    Compute parametric t values where a line segment crosses fold zone boundaries
    and internal bend subdivision lines.

    For each fold marker, the zone spans perpendicular distance [-hw, +hw] from
    the fold center. When num_bend_subdivisions > 1, internal subdivision
    boundaries within the zone are also included so that trace quads align with
    the board mesh facets (preventing traces from cutting through the surface).

    Returns:
        Sorted list of t values in (0, 1) where the line crosses boundaries.
    """
    crossings = []
    for marker in markers:
        hw = marker.zone_width / 2
        perp = (-marker.axis[1], marker.axis[0])

        # Perpendicular distance of start and end from fold center
        ds = (start[0] - marker.center[0]) * perp[0] + (start[1] - marker.center[1]) * perp[1]
        de = (end[0] - marker.center[0]) * perp[0] + (end[1] - marker.center[1]) * perp[1]

        denom = de - ds
        if abs(denom) < 1e-12:
            continue

        # Compute all subdivision boundaries within the bend zone
        # With N subdivisions, there are N+1 boundaries from -hw to +hw
        n_subs = max(1, num_bend_subdivisions)
        for i in range(n_subs + 1):
            boundary = -hw + (i / n_subs) * marker.zone_width
            t = (boundary - ds) / denom
            if 0.001 < t < 0.999:
                crossings.append(t)

    return sorted(crossings)


def _dot3(a, b):
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def create_trace_mesh(
    segment: LineSegment,
    z_offset: float,
    regions: list[Region] = None,
    subdivisions: int = 20,
    pcb_thickness: float = 0.0,
    markers: list[FoldMarker] = None,
    num_bend_subdivisions: int = 1
) -> Mesh:
    """
    Create a 3D mesh for a copper trace.

    Args:
        segment: Trace line segment with width
        z_offset: Z offset for the trace (along surface normal)
        regions: List of Region objects for region-based transformation
        subdivisions: Number of subdivisions along the trace
        pcb_thickness: PCB thickness for back layer trace positioning
        markers: List of fold markers for adaptive subdivision at fold boundaries
        num_bend_subdivisions: Number of strips per bend zone (must match board mesh)

    Returns:
        Mesh representing the trace
    """
    mesh = Mesh()

    # Convert segment to ribbon polygon
    ribbon = line_segment_to_ribbon(segment)

    # Get the 4 corners of the ribbon
    if len(ribbon.vertices) != 4:
        return mesh

    # Determine if trace is on back layer
    is_back_layer = segment.layer == "B.Cu"

    # Subdivide the ribbon along its length
    v0, v1, v2, v3 = ribbon.vertices

    # v0-v1 and v3-v2 are the long edges (along trace)
    # v0-v3 and v1-v2 are the short edges (across trace)

    # Build t-values: uniform subdivisions + fold zone boundary crossings
    t_values = [i / subdivisions for i in range(subdivisions + 1)]
    if regions and markers:
        # Use trace centerline for crossing detection
        center_start = ((v0[0] + v3[0]) / 2, (v0[1] + v3[1]) / 2)
        center_end = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)
        crossings = _compute_fold_crossing_t_values(center_start, center_end, markers, num_bend_subdivisions)
        t_values.extend(crossings)
        t_values = sorted(set(t_values))

    # Create subdivided points along both long edges
    edge1_points = []  # v0 to v1
    edge2_points = []  # v3 to v2

    last_valid_region_1 = None
    last_valid_region_2 = None
    last_n1 = None
    last_n2 = None

    for t in t_values:
        p1 = (v0[0] + t * (v1[0] - v0[0]), v0[1] + t * (v1[1] - v0[1]))
        p2 = (v3[0] + t * (v2[0] - v3[0]), v3[1] + t * (v2[1] - v3[1]))

        # Find region for each point with fallback to last valid region
        region_recipe_1 = []
        region_recipe_2 = []
        if regions:
            containing_region_1 = find_containing_region(p1, regions)
            if containing_region_1:
                last_valid_region_1 = containing_region_1
            elif last_valid_region_1:
                containing_region_1 = last_valid_region_1
            if containing_region_1:
                region_recipe_1 = get_region_recipe(containing_region_1)

            containing_region_2 = find_containing_region(p2, regions)
            if containing_region_2:
                last_valid_region_2 = containing_region_2
            elif last_valid_region_2:
                containing_region_2 = last_valid_region_2
            if containing_region_2:
                region_recipe_2 = get_region_recipe(containing_region_2)

        # Transform to 3D with normal for proper offset
        p1_3d, n1 = transform_point_and_normal(p1, region_recipe_1)
        p2_3d, n2 = transform_point_and_normal(p2, region_recipe_2)

        # Normal consistency check: if normal flips relative to previous point,
        # something went wrong at a region boundary — flip to match
        if last_n1 and _dot3(n1, last_n1) < 0:
            n1 = (-n1[0], -n1[1], -n1[2])
        if last_n2 and _dot3(n2, last_n2) < 0:
            n2 = (-n2[0], -n2[1], -n2[2])
        last_n1 = n1
        last_n2 = n2

        if is_back_layer:
            # Back layer: offset from bottom surface (negative normal direction)
            total_offset = -(pcb_thickness + z_offset)
        else:
            # Front layer: offset from top surface (positive normal direction)
            total_offset = z_offset

        # Apply offset along normal
        p1_3d = (p1_3d[0] + n1[0] * total_offset, p1_3d[1] + n1[1] * total_offset, p1_3d[2] + n1[2] * total_offset)
        p2_3d = (p2_3d[0] + n2[0] * total_offset, p2_3d[1] + n2[1] * total_offset, p2_3d[2] + n2[2] * total_offset)

        edge1_points.append(p1_3d)
        edge2_points.append(p2_3d)

    # Add vertices
    edge1_indices = [mesh.add_vertex(p) for p in edge1_points]
    edge2_indices = [mesh.add_vertex(p) for p in edge2_points]

    # Create quads between the two edges
    # Reverse winding for back layer so normals face outward
    num_quads = len(t_values) - 1
    for i in range(num_quads):
        if is_back_layer:
            mesh.add_quad(
                edge2_indices[i], edge2_indices[i + 1],
                edge1_indices[i + 1], edge1_indices[i],
                COLOR_COPPER
            )
        else:
            mesh.add_quad(
                edge1_indices[i], edge1_indices[i + 1],
                edge2_indices[i + 1], edge2_indices[i],
                COLOR_COPPER
            )

    return mesh


def precompute_trace_mesh(
    segment: LineSegment,
    regions: list[Region] = None,
    subdivisions: int = 20,
    markers: list[FoldMarker] = None,
    num_bend_subdivisions: int = 1
) -> PrecomputedTraceData:
    """
    Precompute angle-independent trace data: ribbon geometry, t-values,
    region assignments for each point.

    Args:
        segment: Trace line segment with width
        regions: List of Region objects for region-based transformation
        subdivisions: Number of subdivisions along the trace
        markers: List of fold markers for adaptive subdivision at fold boundaries
        num_bend_subdivisions: Number of strips per bend zone

    Returns:
        PrecomputedTraceData with all angle-independent data, or None if invalid
    """
    ribbon = line_segment_to_ribbon(segment)
    if len(ribbon.vertices) != 4:
        return None

    is_back_layer = segment.layer == "B.Cu"
    v0, v1, v2, v3 = ribbon.vertices

    # Build t-values: uniform subdivisions + fold zone boundary crossings
    t_values = [i / subdivisions for i in range(subdivisions + 1)]
    if regions and markers:
        center_start = ((v0[0] + v3[0]) / 2, (v0[1] + v3[1]) / 2)
        center_end = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)
        crossings = _compute_fold_crossing_t_values(center_start, center_end, markers, num_bend_subdivisions)
        t_values.extend(crossings)
        t_values = sorted(set(t_values))

    # For each t-value, compute 2D points and their region recipes
    point_data = []
    last_valid_region_1 = None
    last_valid_region_2 = None

    for t in t_values:
        p1 = (v0[0] + t * (v1[0] - v0[0]), v0[1] + t * (v1[1] - v0[1]))
        p2 = (v3[0] + t * (v2[0] - v3[0]), v3[1] + t * (v2[1] - v3[1]))

        # Find region for each point with fallback
        region_1 = None
        region_2 = None
        if regions:
            containing_region_1 = find_containing_region(p1, regions)
            if containing_region_1:
                last_valid_region_1 = containing_region_1
            elif last_valid_region_1:
                containing_region_1 = last_valid_region_1
            region_1 = containing_region_1

            containing_region_2 = find_containing_region(p2, regions)
            if containing_region_2:
                last_valid_region_2 = containing_region_2
            elif last_valid_region_2:
                containing_region_2 = last_valid_region_2
            region_2 = containing_region_2

        point_data.append((p1, p2, region_1, region_2))

    return PrecomputedTraceData(
        t_values=t_values,
        v0=v0, v1=v1, v2=v2, v3=v3,
        is_back_layer=is_back_layer,
        point_data=point_data,
    )


def transform_trace_mesh(
    precomputed: PrecomputedTraceData,
    z_offset: float,
    pcb_thickness: float = 0.0,
) -> Mesh:
    """
    Transform precomputed trace data to 3D mesh using current fold angles.

    This is the fast path: skips ribbon computation, t-value computation,
    and region lookups.

    Args:
        precomputed: PrecomputedTraceData from precompute_trace_mesh()
        z_offset: Z offset for the trace (along surface normal)
        pcb_thickness: PCB thickness for back layer trace positioning

    Returns:
        Mesh representing the trace
    """
    mesh = Mesh()
    is_back_layer = precomputed.is_back_layer

    edge1_points = []
    edge2_points = []
    last_n1 = None
    last_n2 = None

    for p1, p2, region_1, region_2 in precomputed.point_data:
        region_recipe_1 = get_region_recipe(region_1) if region_1 else []
        region_recipe_2 = get_region_recipe(region_2) if region_2 else []

        # Transform to 3D with normal
        p1_3d, n1 = transform_point_and_normal(p1, region_recipe_1)
        p2_3d, n2 = transform_point_and_normal(p2, region_recipe_2)

        # Normal consistency check
        if last_n1 and _dot3(n1, last_n1) < 0:
            n1 = (-n1[0], -n1[1], -n1[2])
        if last_n2 and _dot3(n2, last_n2) < 0:
            n2 = (-n2[0], -n2[1], -n2[2])
        last_n1 = n1
        last_n2 = n2

        if is_back_layer:
            total_offset = -(pcb_thickness + z_offset)
        else:
            total_offset = z_offset

        p1_3d = (p1_3d[0] + n1[0] * total_offset, p1_3d[1] + n1[1] * total_offset, p1_3d[2] + n1[2] * total_offset)
        p2_3d = (p2_3d[0] + n2[0] * total_offset, p2_3d[1] + n2[1] * total_offset, p2_3d[2] + n2[2] * total_offset)

        edge1_points.append(p1_3d)
        edge2_points.append(p2_3d)

    # Add vertices
    edge1_indices = [mesh.add_vertex(p) for p in edge1_points]
    edge2_indices = [mesh.add_vertex(p) for p in edge2_points]

    # Create quads
    num_quads = len(precomputed.t_values) - 1
    for i in range(num_quads):
        if is_back_layer:
            mesh.add_quad(
                edge2_indices[i], edge2_indices[i + 1],
                edge1_indices[i + 1], edge1_indices[i],
                COLOR_COPPER
            )
        else:
            mesh.add_quad(
                edge1_indices[i], edge1_indices[i + 1],
                edge2_indices[i + 1], edge2_indices[i],
                COLOR_COPPER
            )

    return mesh


def create_pad_mesh(
    pad: PadGeometry,
    z_offset: float,
    regions: list[Region] = None,
    pcb_thickness: float = 0.0
) -> Mesh:
    """
    Create a 3D mesh for a pad.

    For through-hole pads with drill holes, creates an annular ring.
    For SMD pads (no drill), creates a solid pad.

    Args:
        pad: Pad geometry
        z_offset: Z offset for the pad (along surface normal)
        regions: List of Region objects for region-based transformation
        pcb_thickness: PCB thickness for back layer pad positioning

    Returns:
        Mesh representing the pad
    """
    mesh = Mesh()

    # Convert pad to polygon
    poly = pad_to_polygon(pad)

    # Find a fallback region for the pad (used when per-vertex lookup fails)
    fallback_region = None

    if regions:
        # First try the pad center
        fallback_region = find_containing_region(pad.center, regions)

        # If center is in a cutout (drill hole), try small offsets
        if not fallback_region:
            cx, cy = pad.center
            offset = max(pad.drill / 2, 0.3) + 0.1 if pad.drill > 0 else 0.3
            for dx, dy in [(offset, 0), (-offset, 0), (0, offset), (0, -offset)]:
                fallback_region = find_containing_region((cx + dx, cy + dy), regions)
                if fallback_region:
                    break

        # Still not found? Try pad polygon vertices
        if not fallback_region and poly.vertices:
            for v in poly.vertices:
                fallback_region = find_containing_region((v[0], v[1]), regions)
                if fallback_region:
                    break

    # Skip pads completely outside the board (no region found after all fallbacks)
    if regions and not fallback_region:
        return mesh

    fallback_recipe = get_region_recipe(fallback_region) if fallback_region else []

    # Determine if pad is on back layer
    is_back_layer = pad.layer == "B.Cu"

    # Calculate z offset based on layer
    if is_back_layer:
        total_offset = -(pcb_thickness + z_offset)
    else:
        total_offset = z_offset

    # Get 2D vertices for outer pad
    outer_2d = [(v[0], v[1]) for v in poly.vertices]

    # Check if this pad has a drill hole
    has_drill = pad.drill > 0
    hole_2d = []

    if has_drill:
        # Create circular hole at pad center
        cx, cy = pad.center
        r = pad.drill / 2
        n_hole = 12  # Number of vertices for hole circle
        for i in range(n_hole):
            theta = 2 * math.pi * i / n_hole
            hole_2d.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))

    def _transform_vertex(v):
        """Transform a 2D vertex using per-vertex region lookup with fallback."""
        recipe = fallback_recipe
        if regions:
            r = find_containing_region(v, regions)
            if r:
                recipe = get_region_recipe(r)
        v3d, normal = transform_point_and_normal(v, recipe)
        return (
            v3d[0] + normal[0] * total_offset,
            v3d[1] + normal[1] * total_offset,
            v3d[2] + normal[2] * total_offset
        )

    # Transform outer vertices to 3D (per-vertex region lookup)
    outer_3d = []
    outer_indices = []
    for v in outer_2d:
        v3d_offset = _transform_vertex(v)
        outer_3d.append(v3d_offset)
        outer_indices.append(mesh.add_vertex(v3d_offset))

    if has_drill and hole_2d:
        # Transform hole vertices to 3D (per-vertex region lookup)
        hole_3d = []
        hole_indices = []
        for v in hole_2d:
            v3d_offset = _transform_vertex(v)
            hole_3d.append(v3d_offset)
            hole_indices.append(mesh.add_vertex(v3d_offset))

        # Triangulate with hole
        triangles, merged = triangulate_with_holes(outer_2d, [hole_2d])

        # Build mapping from 2D vertices to mesh indices
        all_2d = list(outer_2d) + list(hole_2d)
        all_indices = list(outer_indices) + list(hole_indices)

        # Map merged polygon vertices to mesh indices
        merged_indices = []
        for v2d in merged:
            found = False
            for i, av in enumerate(all_2d):
                if abs(av[0] - v2d[0]) < 0.001 and abs(av[1] - v2d[1]) < 0.001:
                    merged_indices.append(all_indices[i])
                    found = True
                    break
            if not found:
                # Fallback: transform and add
                v3d_offset = _transform_vertex(v2d)
                merged_indices.append(mesh.add_vertex(v3d_offset))

        # Add triangles
        for tri in triangles:
            if is_back_layer:
                mesh.add_triangle(merged_indices[tri[0]], merged_indices[tri[2]], merged_indices[tri[1]], COLOR_PAD)
            else:
                mesh.add_triangle(merged_indices[tri[0]], merged_indices[tri[1]], merged_indices[tri[2]], COLOR_PAD)
    else:
        # No drill hole - simple fan triangulation
        n = len(outer_indices)
        if is_back_layer:
            for i in range(1, n - 1):
                mesh.add_triangle(outer_indices[0], outer_indices[i + 1], outer_indices[i], COLOR_PAD)
        else:
            for i in range(1, n - 1):
                mesh.add_triangle(outer_indices[0], outer_indices[i], outer_indices[i + 1], COLOR_PAD)

    return mesh


def create_component_mesh(
    component: ComponentGeometry,
    height: float,
    regions: list[Region] = None,
    pcb_thickness: float = 0.0
) -> Mesh:
    """
    Create a 3D mesh for a component (as a box).

    Args:
        component: Component geometry
        height: Component height (along surface normal)
        regions: List of Region objects for region-based transformation
        pcb_thickness: PCB thickness for back layer component positioning

    Returns:
        Mesh representing the component
    """
    mesh = Mesh()

    # Get bounding box polygon
    box = component_to_box(component)

    # Find which region the component is in
    # Try center, nearby offsets, box corners, then pad positions
    region_recipe = []

    if regions:
        containing_region = find_containing_region(component.center, regions)

        # If center is in a cutout (drill hole), try small offsets
        if not containing_region:
            cx, cy = component.center
            for dx, dy in [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]:
                containing_region = find_containing_region((cx + dx, cy + dy), regions)
                if containing_region:
                    break

        if not containing_region and box.vertices:
            for v in box.vertices:
                containing_region = find_containing_region((v[0], v[1]), regions)
                if containing_region:
                    break

        # Try pad positions for through-hole components
        if not containing_region:
            for pad in component.pads:
                containing_region = find_containing_region(pad.center, regions)
                if containing_region:
                    break

        if containing_region:
            region_recipe = get_region_recipe(containing_region)

    # Determine if component is on back layer
    is_back_layer = component.layer == "B.Cu"

    # Transform vertices with per-vertex region lookup
    base_3d = []
    normals = []
    for v in box.vertices:
        recipe = region_recipe
        if regions:
            v_region = find_containing_region((v[0], v[1]), regions)
            if v_region:
                recipe = get_region_recipe(v_region)
        v3d, normal = transform_point_and_normal(v, recipe)
        base_3d.append(v3d)
        normals.append(normal)

    if is_back_layer:
        # Back layer: component extends downward from PCB bottom
        # PCB surface is at base_3d, bottom surface is at -pcb_thickness
        # Component goes from bottom surface to bottom surface - height
        top_3d = [
            (v[0] - n[0] * pcb_thickness, v[1] - n[1] * pcb_thickness, v[2] - n[2] * pcb_thickness)
            for v, n in zip(base_3d, normals)
        ]
        bottom_3d = [
            (v[0] - n[0] * (pcb_thickness + height), v[1] - n[1] * (pcb_thickness + height), v[2] - n[2] * (pcb_thickness + height))
            for v, n in zip(base_3d, normals)
        ]
    else:
        # Front layer: component extends upward from PCB top
        bottom_3d = base_3d
        top_3d = [
            (v[0] + n[0] * height, v[1] + n[1] * height, v[2] + n[2] * height)
            for v, n in zip(base_3d, normals)
        ]

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


def create_stiffener_mesh(
    outline: list[tuple[float, float]],
    stiffener_thickness: float,
    pcb_thickness: float,
    side: str,
    regions: list[Region] = None,
    cutouts: list[list[tuple[float, float]]] = None
) -> Mesh:
    """
    Create a 3D mesh for a stiffener region.

    Stiffeners are rigid areas bonded to the flex PCB that add thickness.
    They are rendered on top or bottom of the PCB.

    Args:
        outline: Stiffener polygon vertices (list of (x, y) tuples)
        stiffener_thickness: Thickness of the stiffener in mm
        pcb_thickness: PCB thickness in mm (to position stiffener)
        side: "top" or "bottom" - which side the stiffener is on
        regions: List of Region objects for region-based transformation
        cutouts: List of hole polygons within the stiffener

    Returns:
        Mesh representing the stiffener
    """
    if len(outline) < 3 or stiffener_thickness <= 0:
        return Mesh()

    mesh = Mesh()

    # Find which region the stiffener is in
    # Try centroid first, then outline vertices if centroid is in a hole
    region_recipe = []

    if regions:
        # Try centroid first
        cx = sum(v[0] for v in outline) / len(outline)
        cy = sum(v[1] for v in outline) / len(outline)
        containing_region = find_containing_region((cx, cy), regions)

        # If centroid is in a hole (board cutout), try outline vertices
        if not containing_region:
            for v in outline:
                containing_region = find_containing_region(v, regions)
                if containing_region:
                    break

        if containing_region:
            region_recipe = get_region_recipe(containing_region)

    # Transform outline vertices to 3D with normals using recipe-based function
    top_vertices_pcb = []
    vertex_normals = []

    for v in outline:
        v3d, normal = transform_point_and_normal(v, region_recipe)
        top_vertices_pcb.append(v3d)
        vertex_normals.append(normal)

    # Calculate stiffener vertices based on side
    # Stiffener sits on top or bottom of the PCB
    if side == "top":
        # Stiffener on top: from PCB top surface to PCB top + stiffener thickness
        stiffener_bottom = top_vertices_pcb  # Same as PCB top
        stiffener_top = []
        for v3d, normal in zip(top_vertices_pcb, vertex_normals):
            stiffener_top.append((
                v3d[0] + normal[0] * stiffener_thickness,
                v3d[1] + normal[1] * stiffener_thickness,
                v3d[2] + normal[2] * stiffener_thickness
            ))
    else:  # bottom
        # Stiffener on bottom: from PCB bottom to PCB bottom - stiffener thickness
        # PCB bottom is top - pcb_thickness along negative normal
        stiffener_top = []
        stiffener_bottom = []
        for v3d, normal in zip(top_vertices_pcb, vertex_normals):
            # PCB bottom surface
            pcb_bottom = (
                v3d[0] - normal[0] * pcb_thickness,
                v3d[1] - normal[1] * pcb_thickness,
                v3d[2] - normal[2] * pcb_thickness
            )
            stiffener_top.append(pcb_bottom)
            # Stiffener bottom (further down)
            stiffener_bottom.append((
                pcb_bottom[0] - normal[0] * stiffener_thickness,
                pcb_bottom[1] - normal[1] * stiffener_thickness,
                pcb_bottom[2] - normal[2] * stiffener_thickness
            ))

    # Add vertices to mesh for outer boundary
    n = len(outline)
    top_indices = [mesh.add_vertex(v) for v in stiffener_top]
    bottom_indices = [mesh.add_vertex(v) for v in stiffener_bottom]

    # Build combined 2D vertex list and corresponding mesh indices
    # for triangulation with holes
    all_2d = list(outline)
    all_top_indices = list(top_indices)
    all_bottom_indices = list(bottom_indices)

    # Process cutouts - transform to 3D and add to mesh
    if cutouts:
        for cutout in cutouts:
            if len(cutout) < 3:
                continue

            cutout_top_indices = []
            cutout_bottom_indices = []

            for v in cutout:
                v3d, normal = transform_point_and_normal(v, region_recipe)

                if side == "top":
                    ct = (
                        v3d[0] + normal[0] * stiffener_thickness,
                        v3d[1] + normal[1] * stiffener_thickness,
                        v3d[2] + normal[2] * stiffener_thickness
                    )
                    cb = v3d  # PCB top surface
                else:  # bottom
                    pcb_bottom = (
                        v3d[0] - normal[0] * pcb_thickness,
                        v3d[1] - normal[1] * pcb_thickness,
                        v3d[2] - normal[2] * pcb_thickness
                    )
                    ct = pcb_bottom
                    cb = (
                        pcb_bottom[0] - normal[0] * stiffener_thickness,
                        pcb_bottom[1] - normal[1] * stiffener_thickness,
                        pcb_bottom[2] - normal[2] * stiffener_thickness
                    )

                ti = mesh.add_vertex(ct)
                bi = mesh.add_vertex(cb)
                cutout_top_indices.append(ti)
                cutout_bottom_indices.append(bi)

            # Add to combined lists for triangulation mapping
            all_2d.extend(cutout)
            all_top_indices.extend(cutout_top_indices)
            all_bottom_indices.extend(cutout_bottom_indices)

            # Create wall faces around the cutout
            # Winding is reversed for inner walls (facing inward)
            nc = len(cutout)
            for i in range(nc):
                j = (i + 1) % nc
                mesh.add_quad(
                    cutout_top_indices[j], cutout_top_indices[i],
                    cutout_bottom_indices[i], cutout_bottom_indices[j],
                    COLOR_STIFFENER
                )

    # Triangulate the stiffener top and bottom faces
    outline_2d = [(v[0], v[1]) for v in outline]

    if cutouts:
        # Use triangulation with holes
        triangles, merged = triangulate_with_holes(outline_2d, cutouts)

        # Map merged vertices back to mesh indices
        merged_top = []
        merged_bottom = []
        for v2d in merged:
            found = False
            for i, av in enumerate(all_2d):
                if abs(av[0] - v2d[0]) < 0.001 and abs(av[1] - v2d[1]) < 0.001:
                    merged_top.append(all_top_indices[i])
                    merged_bottom.append(all_bottom_indices[i])
                    found = True
                    break
            if not found:
                # This shouldn't happen, but handle gracefully
                v3d, normal = transform_point_and_normal(v2d, region_recipe)
                if side == "top":
                    ct = (v3d[0] + normal[0] * stiffener_thickness,
                          v3d[1] + normal[1] * stiffener_thickness,
                          v3d[2] + normal[2] * stiffener_thickness)
                    cb = v3d
                else:
                    pcb_bottom = (v3d[0] - normal[0] * pcb_thickness,
                                  v3d[1] - normal[1] * pcb_thickness,
                                  v3d[2] - normal[2] * pcb_thickness)
                    ct = pcb_bottom
                    cb = (pcb_bottom[0] - normal[0] * stiffener_thickness,
                          pcb_bottom[1] - normal[1] * stiffener_thickness,
                          pcb_bottom[2] - normal[2] * stiffener_thickness)
                merged_top.append(mesh.add_vertex(ct))
                merged_bottom.append(mesh.add_vertex(cb))

        # Top face
        for tri in triangles:
            mesh.add_triangle(merged_top[tri[0]], merged_top[tri[1]], merged_top[tri[2]], COLOR_STIFFENER)

        # Bottom face (reversed winding)
        for tri in triangles:
            mesh.add_triangle(merged_bottom[tri[0]], merged_bottom[tri[2]], merged_bottom[tri[1]], COLOR_STIFFENER)
    else:
        # No cutouts - simple triangulation
        triangles = triangulate_polygon(outline_2d)

        # Top face
        for tri in triangles:
            mesh.add_triangle(top_indices[tri[0]], top_indices[tri[1]], top_indices[tri[2]], COLOR_STIFFENER)

        # Bottom face (reversed winding)
        for tri in triangles:
            mesh.add_triangle(bottom_indices[tri[0]], bottom_indices[tri[2]], bottom_indices[tri[1]], COLOR_STIFFENER)

    # Side faces (outer boundary)
    for i in range(n):
        j = (i + 1) % n
        mesh.add_quad(
            top_indices[i], top_indices[j],
            bottom_indices[j], bottom_indices[i],
            COLOR_STIFFENER
        )

    return mesh


def create_component_3d_model_mesh(
    component: ComponentGeometry,
    pcb_dir: str,
    pcb_thickness: float,
    regions: list[Region] = None,
    pcb=None
) -> Mesh:
    """
    Create a 3D mesh for a component from its 3D model file.

    Args:
        component: Component geometry with models list
        pcb_dir: Directory of PCB file for path resolution
        pcb_thickness: PCB thickness for back layer positioning
        regions: List of Region objects for region-based transformation
        pcb: KiCadPCB object for extracting embedded models

    Returns:
        Transformed mesh or empty mesh if no model loaded
    """
    mesh = Mesh()

    if not hasattr(component, 'models') or not component.models:
        return mesh

    # Find which region the component is in
    # Try center first, then nearby offsets (for drill hole avoidance),
    # then bounding box corners, then pad positions
    region_recipe = []
    if regions:
        containing_region = find_containing_region(component.center, regions)

        # If center is in a cutout (drill hole), try small offsets around center
        if not containing_region:
            cx, cy = component.center
            for dx, dy in [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]:
                containing_region = find_containing_region((cx + dx, cy + dy), regions)
                if containing_region:
                    break

        # Try bounding box corners
        if not containing_region:
            box = component_to_box(component)
            if box.vertices:
                for v in box.vertices:
                    containing_region = find_containing_region((v[0], v[1]), regions)
                    if containing_region:
                        break

        # Try pad positions (most reliable for through-hole components)
        if not containing_region:
            for pad in component.pads:
                containing_region = find_containing_region(pad.center, regions)
                if containing_region:
                    break
                # Try pad polygon vertices if pad center is in drill hole
                poly = pad_to_polygon(pad)
                for v in poly.vertices:
                    containing_region = find_containing_region(v, regions)
                    if containing_region:
                        break
                if containing_region:
                    break

        if containing_region:
            region_recipe = get_region_recipe(containing_region)

    # Determine if component is on back layer
    is_back_layer = component.layer == "B.Cu"

    # Try each model until one loads successfully
    for model_ref in component.models:
        if model_ref.hide:
            continue

        # Resolve path (supports ${VAR}, relative paths, and kicad-embed://)
        resolved_path = expand_kicad_vars(model_ref.path, pcb_dir, pcb)
        if not resolved_path:
            continue

        # Load model
        loaded = load_model(resolved_path)
        if not loaded:
            continue

        # Apply transforms to each vertex
        # Transform order (KiCad convention):
        # 1. Scale the model
        # 2. Rotate the model (model's own rotation)
        # 3. Translate by model offset
        # 4. Rotate by component angle
        # 5. If back layer, mirror and offset
        # 6. Transform using region recipe (bend)
        # 7. Translate to component position

        # Pre-compute rotation matrices
        def rot_x(angle):
            c, s = math.cos(angle), math.sin(angle)
            return lambda x, y, z: (x, y * c - z * s, y * s + z * c)

        def rot_y(angle):
            c, s = math.cos(angle), math.sin(angle)
            return lambda x, y, z: (x * c + z * s, y, -x * s + z * c)

        def rot_z(angle):
            c, s = math.cos(angle), math.sin(angle)
            return lambda x, y, z: (x * c - y * s, x * s + y * c, z)

        # Model rotations (convert to radians)
        rx = rot_x(math.radians(model_ref.rotate[0]))
        ry = rot_y(math.radians(model_ref.rotate[1]))
        rz = rot_z(math.radians(model_ref.rotate[2]))

        # Component rotation
        comp_rz = rot_z(math.radians(-component.angle))  # KiCad uses clockwise positive

        # Get surface normal at component center (after bend)
        _, surface_normal = transform_point_and_normal(component.center, region_recipe)

        # Track vertex index mapping (old index -> new index)
        vertex_map = {}

        for old_idx, v in enumerate(loaded.mesh.vertices):
            x, y, z = v

            # 1. Scale
            x *= model_ref.scale[0]
            y *= model_ref.scale[1]
            z *= model_ref.scale[2]

            # 2. Model rotation (ZYX order)
            x, y, z = rz(x, y, z)
            x, y, z = ry(x, y, z)
            x, y, z = rx(x, y, z)

            # 3. Model offset
            x += model_ref.offset[0]
            y += model_ref.offset[1]
            z += model_ref.offset[2]

            # 4. Component rotation
            x, y, z = comp_rz(x, y, z)

            # 5. Back layer handling (flip and offset)
            if is_back_layer:
                z = -z - pcb_thickness

            # Now apply bend transform to (x, y) position relative to component center
            # The Z offset is applied along the transformed surface normal
            local_x = component.center[0] + x
            local_y = component.center[1] + y

            # Per-vertex region lookup for correct fold transform
            v_recipe = region_recipe
            if regions:
                v_region = find_containing_region((local_x, local_y), regions)
                if v_region:
                    v_recipe = get_region_recipe(v_region)

            # Transform the base position using the vertex's region recipe
            base_3d, normal = transform_point_and_normal((local_x, local_y), v_recipe)

            # Apply Z offset along the surface normal
            final_x = base_3d[0] + normal[0] * z
            final_y = base_3d[1] + normal[1] * z
            final_z = base_3d[2] + normal[2] * z

            new_idx = mesh.add_vertex((final_x, final_y, final_z))
            vertex_map[old_idx] = new_idx

        # Copy faces with proper winding, using mapped indices
        for i, face in enumerate(loaded.mesh.faces):
            color = loaded.mesh.colors[i] if i < len(loaded.mesh.colors) else COLOR_MODEL_3D
            # Map old indices to new indices
            mapped_face = [vertex_map[idx] for idx in face]
            if len(mapped_face) == 3:
                if is_back_layer:
                    mesh.add_triangle(mapped_face[0], mapped_face[2], mapped_face[1], color)
                else:
                    mesh.add_triangle(mapped_face[0], mapped_face[1], mapped_face[2], color)
            elif len(mapped_face) == 4:
                if is_back_layer:
                    mesh.add_quad(mapped_face[0], mapped_face[3], mapped_face[2], mapped_face[1], color)
                else:
                    mesh.add_quad(mapped_face[0], mapped_face[1], mapped_face[2], mapped_face[3], color)

        # Return after first successful model load
        return mesh

    return mesh

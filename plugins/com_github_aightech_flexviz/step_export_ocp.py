"""
STEP file export using build123d/OCP (OpenCASCADE) backend.

Produces exact B-Rep geometry: circular arcs stay as true circles,
bend zones are exact cylindrical surfaces via revolve, and fillets
in the revolve profile become exact toroidal surfaces.

Requires: build123d >= 0.5.0  (pip install build123d)

This is a parallel export path to step_export.py (pure Python).
Use when running outside KiCad's restricted Python environment
(IPC API mode or command-line).
"""

import math
from typing import Optional, List, Tuple

# ---------- conditional imports -----------------------------------------

try:
    from build123d import (
        Edge, Wire, Face, Solid, Compound, Shell,
        Vector, Axis, Plane, Location,
        export_step,
    )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid,
    )
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Pnt, gp_Dir, gp_Vec, gp_Trsf
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
    from OCP.BRep import BRep_Tool
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.ShapeFix import ShapeFix_Solid
    OCP_AVAILABLE = True
except ImportError:
    OCP_AVAILABLE = False

try:
    from .bend_transform import (
        FoldDefinition, transform_point, compute_normal,
        recipe_from_region,
        _rotation_matrix_around_axis, _apply_rotation, _multiply_matrices,
    )
    from .planar_subdivision import split_board_into_regions, find_containing_region
    from .step_export import (
        _recover_arcs, _point_on_arc, _compute_region_normal, _is_bend_region,
        _recipe_with_fold_defs, TaggedEdge, _snap_to_plane,
        _sample_arc_2d,
        _resolve_model_path_for_step, _compute_component_placement,
    )
    from .stiffener import extract_stiffeners
    from .step_writer import _normalize, _cross, _sub, _add, _scale, _dot
except ImportError:
    from bend_transform import (
        FoldDefinition, transform_point, compute_normal,
        recipe_from_region,
        _rotation_matrix_around_axis, _apply_rotation, _multiply_matrices,
    )
    from planar_subdivision import split_board_into_regions, find_containing_region
    from step_export import (
        _recover_arcs, _point_on_arc, _compute_region_normal, _is_bend_region,
        _recipe_with_fold_defs, TaggedEdge, _snap_to_plane,
        _sample_arc_2d,
        _resolve_model_path_for_step, _compute_component_placement,
    )
    from stiffener import extract_stiffeners
    from step_writer import _normalize, _cross, _sub, _add, _scale, _dot


# =====================================================================
# Public API
# =====================================================================

def is_ocp_available() -> bool:
    """Check if build123d/OCP backend is available."""
    return OCP_AVAILABLE


def board_to_step_ocp(board_geometry, markers, filename, config=None,
                      pcb=None, stiffeners=None, pcb_dir=None,
                      include_models=False, include_wrl_models=False,
                      bbox_models=False):
    """
    Export board geometry to STEP using build123d/OCP for exact B-Rep.

    Same signature as board_to_step_native for drop-in replacement.
    Returns True on success, False on error.
    """
    if not OCP_AVAILABLE:
        raise ImportError(
            "build123d is required for OCP export. "
            "Install with: pip install build123d"
        )

    try:
        thickness = board_geometry.thickness
        original_segments = (
            board_geometry.outline.segments
            if board_geometry.outline.segments else []
        )

        # Gather circle cutout / drill hole info for exact hole geometry
        circle_holes = []  # list of (cx, cy, radius)
        if pcb is not None:
            try:
                for cc in (pcb.get_circle_cutouts() or []):
                    circle_holes.append((cc.center_x, cc.center_y, cc.radius))
            except Exception:
                pass
            try:
                for dh in (pcb.get_drill_holes() or []):
                    circle_holes.append((dh.center_x, dh.center_y, dh.diameter / 2))
            except Exception:
                pass

        # Gather cutout segment info for arc recovery in holes
        cutout_segments = []
        for c in board_geometry.cutouts:
            cutout_segments.append(c.segments if c.segments else [])

        outline_2d = _tessellate_arcs(
            list(board_geometry.outline.vertices), original_segments)
        if len(outline_2d) < 3:
            return False
        holes_2d = [list(c.vertices) for c in board_geometry.cutouts]

        regions = split_board_into_regions(
            outline_2d, holes_2d, markers,
            num_bend_subdivisions=1,
            arc_segments=original_segments,
        )

        # --- build unified PCB solid (single shell, no boolean fuse) ------
        pcb_solid = _build_unified_pcb_ocp(
            regions, thickness, original_segments, circle_holes,
            cutout_segments)

        if pcb_solid is None:
            return False

        all_solids = [pcb_solid]

        # --- stiffeners ---------------------------------------------------
        if stiffeners is None and config is not None and pcb is not None:
            if config.has_stiffener:
                stiffeners = extract_stiffeners(pcb, config)

        if stiffeners:
            for stiff in stiffeners:
                s = _build_stiffener_ocp(
                    stiff, regions, thickness)
                if s is not None:
                    all_solids.append(s)

        # --- mirror Y: KiCad Y-down → CAD Y-up ----------------------------
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.BRepLib import BRepLib
        mirror_trsf = gp_Trsf()
        mirror_trsf.SetMirror(gp_Ax2(
            gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))  # mirror across XZ plane
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        mirrored_shapes = []
        for s in all_solids:
            xform = BRepBuilderAPI_Transform(s.wrapped, mirror_trsf, True)
            xform.Build()
            shape = xform.Shape()
            # Mirror flips face normals → reverse if volume is negative
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape, props)
            if props.Mass() < 0:
                shape.Reverse()
            mirrored_shapes.append(shape)

        # --- export -------------------------------------------------------
        if include_models and board_geometry.components:
            _write_step_with_models(
                mirrored_shapes, board_geometry, regions, thickness,
                filename, pcb_dir=pcb_dir, pcb=pcb,
                include_wrl=include_wrl_models,
                mirror_trsf=mirror_trsf,
                bbox_models=bbox_models,
            )
        else:
            _write_step_assembly(mirrored_shapes, filename)

        return True

    except Exception as e:
        print(f"OCP STEP export error: {e}")
        import traceback
        traceback.print_exc()
        return False


# =====================================================================
# Helpers: OCP wrappers
# =====================================================================

def _vec(pt):
    """Tuple → build123d Vector."""
    return Vector(pt[0], pt[1], pt[2])


def _make_wire(edges) -> "Wire":
    """Build a Wire from edges using OCP's tolerant wire builder."""
    builder = BRepBuilderAPI_MakeWire()
    for e in edges:
        builder.Add(e.wrapped)
    builder.Build()
    return Wire(builder.Wire())


def _make_face(outer_wire, hole_wires=None) -> "Face":
    """Create a Face from an outer Wire and optional hole Wires (OCP)."""
    maker = BRepBuilderAPI_MakeFace(outer_wire.wrapped)
    if hole_wires:
        for hw in hole_wires:
            maker.Add(hw.wrapped)
    maker.Build()
    if not maker.IsDone():
        return None
    face_shape = maker.Face()
    if face_shape.IsNull():
        return None
    return Face(face_shape)


def _tessellate_arcs(vertices, segments, max_chord=0.5):
    """Add intermediate vertices on arc segments for smooth bend zones.

    Inserts points on the arc circle so that each sub-chord is at most
    ``max_chord`` mm.  The original arc segment metadata is preserved
    for downstream arc recovery on flat regions.
    """
    if not segments:
        return vertices
    arc_segs = [s for s in segments if s.type == "arc" and s.center is not None]
    if not arc_segs:
        return vertices

    result = []
    n = len(vertices)
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        result.append(v1)

        # Check if this edge lies on an arc
        for arc in arc_segs:
            cx, cy = arc.center
            d1 = math.sqrt((v1[0] - cx) ** 2 + (v1[1] - cy) ** 2)
            d2 = math.sqrt((v2[0] - cx) ** 2 + (v2[1] - cy) ** 2)
            if abs(d1 - arc.radius) < arc.radius * 0.15 and \
               abs(d2 - arc.radius) < arc.radius * 0.15:
                # Both endpoints on this arc — tessellate
                a1 = math.atan2(v1[1] - cy, v1[0] - cx)
                a2 = math.atan2(v2[1] - cy, v2[0] - cx)
                # Determine sweep using cross product (short arc)
                sx, sy = v1[0] - cx, v1[1] - cy
                ex, ey = v2[0] - cx, v2[1] - cy
                ccw = (sx * ey - sy * ex) > 0
                if ccw:
                    sweep = a2 - a1
                    if sweep <= 0:
                        sweep += 2 * math.pi
                else:
                    sweep = a2 - a1
                    if sweep >= 0:
                        sweep -= 2 * math.pi

                # Number of sub-segments based on chord length
                arc_len = abs(sweep) * arc.radius
                n_sub = max(1, int(math.ceil(arc_len / max_chord)))
                if n_sub > 1:
                    for k in range(1, n_sub):
                        frac = k / n_sub
                        angle = a1 + sweep * frac
                        px = cx + arc.radius * math.cos(angle)
                        py = cy + arc.radius * math.sin(angle)
                        result.append((px, py))
                break

    return result


def _extrude_face(face: "Face", direction: tuple) -> "Solid":
    """Extrude a Face into a Solid along *direction* (x,y,z)."""
    prism = BRepPrimAPI_MakePrism(
        face.wrapped,
        gp_Vec(direction[0], direction[1], direction[2]),
    )
    prism.Build()
    return Solid(prism.Shape())


def _revolve_face(face: "Face", axis_origin: tuple, axis_dir: tuple,
                  angle_rad: float) -> "Solid":
    """Revolve a Face around an axis by *angle_rad* radians."""
    ax = gp_Ax1(
        gp_Pnt(axis_origin[0], axis_origin[1], axis_origin[2]),
        gp_Dir(axis_dir[0], axis_dir[1], axis_dir[2]),
    )
    revol = BRepPrimAPI_MakeRevol(face.wrapped, ax, angle_rad)
    revol.Build()
    return Solid(revol.Shape())


# =====================================================================
# Wire construction from tagged edges
# =====================================================================

def _arc_midpoint_2d(start, end, center, radius, ccw):
    """Compute the midpoint of a 2D circular arc."""
    cx, cy = center
    a_start = math.atan2(start[1] - cy, start[0] - cx)
    a_end = math.atan2(end[1] - cy, end[0] - cx)
    if ccw:
        sweep = a_end - a_start
        if sweep <= 0:
            sweep += 2 * math.pi
    else:
        sweep = a_end - a_start
        if sweep >= 0:
            sweep -= 2 * math.pi
    a_mid = a_start + sweep / 2
    return (cx + radius * math.cos(a_mid), cy + radius * math.sin(a_mid))


def _make_3d_wire(tagged_edges, recipe, normal):
    """
    Create a 3D Wire from tagged 2D edges.

    Line edges → Edge.make_line
    Arc edges  → Edge.make_three_point_arc  (exact circle through 3 pts)
    """
    n = _normalize(normal)
    edges: list = []

    for te in tagged_edges:
        start_3d = transform_point(te.start, recipe)
        end_3d = transform_point(te.end, recipe)

        d = _sub(end_3d, start_3d)
        if math.sqrt(_dot(d, d)) < 1e-9:
            continue

        if te.type == "arc" and te.center is not None:
            mid_2d = _arc_midpoint_2d(
                te.start, te.end, te.center, te.radius, te.ccw)
            mid_3d = transform_point(mid_2d, recipe)

            # Snap the three points to the face plane for planarity
            pts = _snap_to_plane([start_3d, mid_3d, end_3d], n)
            try:
                edges.append(Edge.make_three_point_arc(
                    _vec(pts[0]), _vec(pts[1]), _vec(pts[2])))
                continue
            except Exception:
                pass  # fall through to line

        edges.append(Edge.make_line(_vec(start_3d), _vec(end_3d)))

    if not edges:
        return None
    return _make_wire(edges)


def _make_polygon_wire_3d(vertices_2d, recipe):
    """Create a 3D Wire of all-line edges from a 2D polygon."""
    pts_3d = [transform_point(v, recipe) for v in vertices_2d]
    edges: list = []
    n = len(pts_3d)
    for i in range(n):
        p1 = pts_3d[i]
        p2 = pts_3d[(i + 1) % n]
        d = _sub(p2, p1)
        if math.sqrt(_dot(d, d)) < 1e-9:
            continue
        edges.append(Edge.make_line(_vec(p1), _vec(p2)))
    if not edges:
        return None
    return _make_wire(edges)


def _match_circle_hole(hole_verts, circle_holes, tolerance=0.1):
    """Check if a polygon hole matches a known circle cutout/drill.

    Returns (cx, cy, radius) if matched, else None.
    """
    if len(hole_verts) < 4 or not circle_holes:
        return None
    # Compute centroid and average radius from vertices
    n = len(hole_verts)
    cx = sum(p[0] for p in hole_verts) / n
    cy = sum(p[1] for p in hole_verts) / n
    dists = [math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in hole_verts]
    avg_r = sum(dists) / len(dists)
    max_dev = max(abs(d - avg_r) for d in dists)
    # Only match if vertices are roughly circular
    if max_dev > avg_r * 0.15:
        return None
    # Find matching circle
    for ccx, ccy, cr in circle_holes:
        if abs(ccx - cx) < tolerance and abs(ccy - cy) < tolerance and abs(cr - avg_r) < tolerance:
            return (ccx, ccy, cr)
    return None


def _make_circle_wire_3d(center_2d, radius, recipe, normal):
    """Create an exact 3D circle wire from a 2D center + radius.

    The circle is constructed as two semicircular arcs in the
    transformed 3D plane.
    """
    n = _normalize(normal)
    c3 = transform_point(center_2d, recipe)
    # Two points on the circle (along a direction in the plane)
    p_east_2d = (center_2d[0] + radius, center_2d[1])
    p_west_2d = (center_2d[0] - radius, center_2d[1])
    p_north_2d = (center_2d[0], center_2d[1] + radius)
    p_south_2d = (center_2d[0], center_2d[1] - radius)

    pe = transform_point(p_east_2d, recipe)
    pw = transform_point(p_west_2d, recipe)
    pn = transform_point(p_north_2d, recipe)
    ps = transform_point(p_south_2d, recipe)

    # Snap to plane
    pts = _snap_to_plane([pe, pn, pw, ps], n)
    pe, pn, pw, ps = pts

    try:
        arc1 = Edge.make_three_point_arc(_vec(pe), _vec(pn), _vec(pw))
        arc2 = Edge.make_three_point_arc(_vec(pw), _vec(ps), _vec(pe))
        return _make_wire([arc1, arc2])
    except Exception:
        return None


# =====================================================================
# Flat region solid
# =====================================================================

def _build_flat_solid_ocp(region, recipe_defs, thickness, original_segments,
                          circle_holes=None):
    """
    Build an extruded solid for a flat (non-bend) region.

    Arc edges from the original board outline are preserved as exact
    circular arcs.  Circle cutouts / drill holes are exact circles.
    """
    normal = _compute_region_normal(region, recipe_defs)
    n = _normalize(normal)

    # Recover arc info for outer boundary
    if original_segments:
        tagged_edges = _recover_arcs(region.outline, original_segments)
    else:
        nv = len(region.outline)
        tagged_edges = [
            TaggedEdge(type="line",
                       start=region.outline[i],
                       end=region.outline[(i + 1) % nv])
            for i in range(nv)
        ]

    outer_wire = _make_3d_wire(tagged_edges, recipe_defs, normal)
    if outer_wire is None:
        return None

    # Hole wires — try exact circles first, then arc recovery, then lines
    hole_wires: list = []
    for hole in region.holes:
        hw = None
        # 1) Try matching to a known circle cutout / drill hole
        if circle_holes:
            match = _match_circle_hole(hole, circle_holes)
            if match:
                cx, cy, r = match
                hw = _make_circle_wire_3d((cx, cy), r, recipe_defs, normal)

        # 2) Try arc recovery from original cutout segments
        if hw is None and original_segments:
            hole_tagged = _recover_arcs(hole, original_segments)
            if any(e.type == "arc" for e in hole_tagged):
                # Reverse winding for CW hole
                rev = list(reversed(hole_tagged))
                rev_tagged = [
                    TaggedEdge(type=e.type, start=e.end, end=e.start,
                               center=e.center, radius=e.radius,
                               ccw=(not e.ccw) if e.type == "arc" else e.ccw)
                    for e in rev
                ]
                hw = _make_3d_wire(rev_tagged, recipe_defs, normal)

        # 3) Fallback: all-line polygon (reversed for CW)
        if hw is None:
            hw = _make_polygon_wire_3d(list(reversed(hole)), recipe_defs)

        if hw is not None:
            hole_wires.append(hw)

    face = _make_face(outer_wire, hole_wires)
    if face is None:
        return None

    # Extrude along -normal by thickness
    direction = (-n[0] * thickness, -n[1] * thickness, -n[2] * thickness)
    return _extrude_face(face, direction)


# =====================================================================
# Bend region solid  (revolve)
# =====================================================================

def _build_bend_solid_ocp(region, recipe_defs, thickness, original_segments):
    """
    Build a cylindrical solid for a bend zone via revolve.

    The cross-section profile is constructed at theta_min in the
    (cyl_axis × radial) plane, then revolved around the cylinder axis
    by the sweep angle.  Straight profile edges produce exact cylindrical
    surfaces; arc profile edges produce exact toroidal surfaces.
    """
    # --- extract fold definition ----------------------------------------
    fold_def = None
    entered_from_back = False
    for entry in reversed(recipe_defs):
        if entry[1] == "IN_ZONE":
            fold_def = entry[0]
            entered_from_back = entry[2] if len(entry) > 2 else False
            break

    if fold_def is None:
        return _build_flat_solid_ocp(
            region, recipe_defs, thickness, original_segments)

    prior_recipe = recipe_defs[:-1]
    hw = fold_def.zone_width / 2
    R = fold_def.radius

    # --- perp / along extents ------------------------------------------
    perp_dists = []
    along_dists = []
    for p in region.outline:
        dx = p[0] - fold_def.center[0]
        dy = p[1] - fold_def.center[1]
        perp_dists.append(dx * fold_def.perp[0] + dy * fold_def.perp[1])
        along_dists.append(dx * fold_def.axis[0] + dy * fold_def.axis[1])

    perp_min, perp_max = min(perp_dists), max(perp_dists)
    along_min, along_max = min(along_dists), max(along_dists)

    # --- theta range ---------------------------------------------------
    # Compute raw fractions first (same regardless of entry direction).
    # The axis_sign negation already handles the sweep direction for
    # back entry — theta reversal is applied *after* axis setup.
    zw = fold_def.zone_width
    frac_min = max(0, perp_min + hw) / zw if zw > 0 else 0
    frac_max = min(zw, perp_max + hw) / zw if zw > 0 else 0

    sign_angle = 1.0 if fold_def.angle >= 0 else -1.0

    # --- 3-D cylinder parameters (same logic as step_export.py) --------
    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for entry in prior_recipe:
        f, c = entry[0], entry[1]
        back = entry[2] if len(entry) > 2 else False
        if c == "AFTER":
            f_ax = _apply_rotation(rot, (f.axis[0], f.axis[1], 0.0))
            ra = -f.angle if back else f.angle
            rot = _multiply_matrices(
                _rotation_matrix_around_axis(f_ax, ra), rot)

    fold_axis_3d = _apply_rotation(rot, (fold_def.axis[0], fold_def.axis[1], 0.0))
    fold_perp_3d = _apply_rotation(rot, (fold_def.perp[0], fold_def.perp[1], 0.0))
    up_3d = _apply_rotation(rot, (0.0, 0.0, 1.0))

    ref_2d = (fold_def.center[0] + hw * fold_def.perp[0],
              fold_def.center[1] + hw * fold_def.perp[1])
    ref_3d = transform_point(ref_2d, prior_recipe)
    fold_center_3d = _sub(ref_3d, _scale(fold_perp_3d, hw))

    if entered_from_back:
        entry_3d = _add(fold_center_3d, _scale(fold_perp_3d, hw))
    else:
        entry_3d = _sub(fold_center_3d, _scale(fold_perp_3d, hw))

    cyl_origin = _add(entry_3d, _scale(up_3d, sign_angle * R))
    cyl_ref = _normalize(_scale(up_3d, -sign_angle))
    axis_sign = -sign_angle if entered_from_back else sign_angle
    cyl_axis = _normalize(_scale(fold_axis_3d, axis_sign))

    # Apply theta reversal for back entry AFTER axis setup (same as
    # step_export.py lines 569-574).  The axis_sign negation already
    # reverses the sweep direction; the theta reversal maps the perp
    # fractions onto the back-entry cylinder parametrisation.
    if entered_from_back:
        theta_min = abs(fold_def.angle) - frac_max * abs(fold_def.angle)
        theta_max = abs(fold_def.angle) - frac_min * abs(fold_def.angle)
    else:
        theta_min = frac_min * abs(fold_def.angle)
        theta_max = frac_max * abs(fold_def.angle)

    sweep = theta_max - theta_min
    if sweep < 1e-9:
        return None

    if sign_angle >= 0:
        inner_r, outer_r = R, R + thickness
    else:
        inner_r, outer_r = R - thickness, R

    if axis_sign >= 0:
        al_min, al_max = along_min, along_max
    else:
        al_min, al_max = -along_max, -along_min

    # --- build profile at theta_min ------------------------------------
    tangent = _normalize(_cross(cyl_axis, cyl_ref))
    cos_t = math.cos(theta_min)
    sin_t = math.sin(theta_min)
    radial = _add(_scale(cyl_ref, cos_t), _scale(tangent, sin_t))

    def _prof_pt(along, r):
        return _add(_add(cyl_origin, _scale(cyl_axis, along)),
                    _scale(radial, r))

    # Try to include arcs in the along-boundary profile edges.
    # If construction fails, fall back to rectangular profile.
    profile_edges = _profile_with_arcs(
        region, fold_def, original_segments, recipe_defs,
        cyl_origin, cyl_axis, radial, inner_r, outer_r, al_min, al_max,
    )

    if profile_edges is None:
        # Simple rectangular profile
        p1 = _prof_pt(al_min, inner_r)
        p2 = _prof_pt(al_max, inner_r)
        p3 = _prof_pt(al_max, outer_r)
        p4 = _prof_pt(al_min, outer_r)
        profile_edges = [
            Edge.make_line(_vec(p1), _vec(p2)),   # inner (along axis)
            Edge.make_line(_vec(p2), _vec(p3)),   # end cap
            Edge.make_line(_vec(p3), _vec(p4)),   # outer (along axis)
            Edge.make_line(_vec(p4), _vec(p1)),   # start cap
        ]

    try:
        profile_wire = _make_wire(profile_edges)
        profile_face = _make_face(profile_wire)
        if profile_face is None:
            raise ValueError("profile face failed")
        return _revolve_face(profile_face, cyl_origin, cyl_axis, sweep)
    except Exception:
        # Fall back to rectangular profile if arc profile fails
        p1 = _prof_pt(al_min, inner_r)
        p2 = _prof_pt(al_max, inner_r)
        p3 = _prof_pt(al_max, outer_r)
        p4 = _prof_pt(al_min, outer_r)
        rect_edges = [
            Edge.make_line(_vec(p1), _vec(p2)),
            Edge.make_line(_vec(p2), _vec(p3)),
            Edge.make_line(_vec(p3), _vec(p4)),
            Edge.make_line(_vec(p4), _vec(p1)),
        ]
        rw = _make_wire(rect_edges)
        rf = _make_face(rw)
        return _revolve_face(rf, cyl_origin, cyl_axis, sweep)


def _profile_with_arcs(region, fold_def, original_segments, recipe_defs,
                       cyl_origin, cyl_axis, radial,
                       inner_r, outer_r, al_min, al_max):
    """
    Build profile edges that include exact arcs where the board outline
    has fillets intersecting the bend zone.

    Returns a list of Edge, or None to fall back to a rectangle.
    """
    if not original_segments:
        return None

    tagged = _recover_arcs(region.outline, original_segments)
    has_arcs = any(e.type == "arc" for e in tagged)
    if not has_arcs:
        return None

    # Project each outline vertex into (along, perp) fold-local coords
    fold_c = fold_def.center
    verts_ap = []
    for p in region.outline:
        dx = p[0] - fold_c[0]
        dy = p[1] - fold_c[1]
        al = dx * fold_def.axis[0] + dy * fold_def.axis[1]
        pe = dx * fold_def.perp[0] + dy * fold_def.perp[1]
        verts_ap.append((al, pe))

    hw = fold_def.zone_width / 2

    # Classify outline edges as "along inner", "along outer", or "perp boundary"
    # Inner boundary: edges near perp_min; Outer boundary: edges near perp_max
    # Perp boundaries: edges near along_min or along_max
    perp_vals = [v[1] for v in verts_ap]
    pmin, pmax = min(perp_vals), max(perp_vals)
    tol = fold_def.zone_width * 0.1

    # Collect the "inner" edges (near pmin) and "outer" edges (near pmax)
    inner_edges: list = []  # profile inner boundary (along axis, at inner_r)
    outer_edges: list = []  # profile outer boundary (along axis, at outer_r)

    for i, te in enumerate(tagged):
        ap1 = verts_ap[i]
        ap2 = verts_ap[(i + 1) % len(verts_ap)]
        avg_perp = (ap1[1] + ap2[1]) / 2

        if abs(avg_perp - pmin) < tol:
            inner_edges.append((te, ap1, ap2))
        elif abs(avg_perp - pmax) < tol:
            outer_edges.append((te, ap1, ap2))

    if not inner_edges and not outer_edges:
        return None

    # Helper to make a profile edge from a tagged edge
    def _make_profile_edge(te, r):
        # Map the 2D edge to the profile plane (along, radial)
        al_s = (te.start[0] - fold_c[0]) * fold_def.axis[0] + \
               (te.start[1] - fold_c[1]) * fold_def.axis[1]
        al_e = (te.end[0] - fold_c[0]) * fold_def.axis[0] + \
               (te.end[1] - fold_c[1]) * fold_def.axis[1]

        start_3d = _add(_add(cyl_origin, _scale(cyl_axis, al_s)),
                        _scale(radial, r))
        end_3d = _add(_add(cyl_origin, _scale(cyl_axis, al_e)),
                      _scale(radial, r))

        if te.type == "arc" and te.center is not None:
            al_m_2d = _arc_midpoint_2d(
                te.start, te.end, te.center, te.radius, te.ccw)
            al_m = (al_m_2d[0] - fold_c[0]) * fold_def.axis[0] + \
                   (al_m_2d[1] - fold_c[1]) * fold_def.axis[1]
            mid_3d = _add(_add(cyl_origin, _scale(cyl_axis, al_m)),
                          _scale(radial, r))
            try:
                return Edge.make_three_point_arc(
                    _vec(start_3d), _vec(mid_3d), _vec(end_3d))
            except Exception:
                pass
        return Edge.make_line(_vec(start_3d), _vec(end_3d))

    # Build profile: inner edges → end cap → outer edges (reversed) → start cap
    try:
        edges: list = []

        # Sort inner edges by along coordinate
        inner_sorted = sorted(inner_edges, key=lambda x: x[1][0])
        for te, _, _ in inner_sorted:
            edges.append(_make_profile_edge(te, inner_r))

        # End cap at along_max
        last_inner = _add(_add(cyl_origin, _scale(cyl_axis, al_max)),
                          _scale(radial, inner_r))
        first_outer = _add(_add(cyl_origin, _scale(cyl_axis, al_max)),
                           _scale(radial, outer_r))
        edges.append(Edge.make_line(_vec(last_inner), _vec(first_outer)))

        # Outer edges (reversed direction: along_max → along_min)
        outer_sorted = sorted(outer_edges, key=lambda x: x[1][0], reverse=True)
        for te, _, _ in outer_sorted:
            # Reverse the edge direction
            rev_te = TaggedEdge(
                type=te.type, start=te.end, end=te.start,
                center=te.center, radius=te.radius,
                ccw=not te.ccw if te.type == "arc" else te.ccw,
            )
            edges.append(_make_profile_edge(rev_te, outer_r))

        # Start cap at along_min
        last_outer = _add(_add(cyl_origin, _scale(cyl_axis, al_min)),
                          _scale(radial, outer_r))
        first_inner = _add(_add(cyl_origin, _scale(cyl_axis, al_min)),
                           _scale(radial, inner_r))
        edges.append(Edge.make_line(_vec(last_outer), _vec(first_inner)))

        return edges
    except Exception:
        return None


# =====================================================================
# Stiffener solid
# =====================================================================

def _stiffener_segments_to_outline_segments(seg_dicts):
    """Convert stiffener segment dicts to OutlineSegment objects.

    This allows reusing ``_recover_arcs`` (same arc matching as the PCB
    outline) instead of heuristic arc detection.
    """
    try:
        from .geometry import OutlineSegment
    except ImportError:
        from geometry import OutlineSegment

    result = []
    for s in seg_dicts:
        if s['type'] == 'arc' and s.get('center'):
            result.append(OutlineSegment(
                type='arc',
                start=s['start'],
                end=s['end'],
                center=s['center'],
                radius=s['radius'],
                mid=s.get('mid'),
            ))
        else:
            result.append(OutlineSegment(
                type='line',
                start=s['start'],
                end=s['end'],
            ))
    return result


def _build_stiffener_ocp(stiff, regions, pcb_thickness):
    """Build a flat extruded solid for a stiffener.

    When the stiffener has arc/line segment metadata from KiCad
    (``stiff.segments``), uses ``_recover_arcs`` for exact arc edges
    (same approach as the PCB outline).  Otherwise falls back to
    straight-line polygon or heuristic arc detection.
    """
    containing = find_containing_region(stiff.centroid, regions)
    if containing is None:
        return None

    recipe_defs = _recipe_with_fold_defs(containing.fold_recipe)
    normal = _compute_region_normal(containing, recipe_defs)
    n = _normalize(normal)

    if stiff.side == "bottom":
        offset = _scale(n, -(pcb_thickness + stiff.thickness))
    else:
        offset = (0, 0, 0)
    extr_dir = _scale(n, stiff.thickness)

    # Helper: transform + offset a 2D point
    def _xform(p2d):
        p3 = transform_point(p2d, recipe_defs)
        return _add(p3, offset) if stiff.side == "bottom" else p3

    outline = stiff.outline
    if not isinstance(outline, (list, tuple)) or len(outline) < 3:
        return None

    # Convert segment metadata to OutlineSegment objects if available
    stiff_segs = None
    if stiff.segments:
        stiff_segs = _stiffener_segments_to_outline_segments(stiff.segments)

    # Build outer wire — use _recover_arcs if we have segment metadata
    outer_wire = None
    if stiff_segs:
        tagged = _recover_arcs(outline, stiff_segs)
        # Build wire using same path as PCB outline: transform + arc edges
        # We need a custom recipe that includes the offset for bottom stiffeners
        outer_wire = _make_3d_wire_xform(tagged, _xform, n)

    if outer_wire is None:
        outer_wire = _make_polygon_wire(_xform, outline, n)
    if outer_wire is None:
        return None

    # Build cutout hole wires (reversed winding for CW)
    hole_wires: list = []
    for ci, cutout in enumerate(stiff.cutouts):
        cut_verts = cutout if isinstance(cutout, (list, tuple)) else list(cutout)
        if len(cut_verts) < 3:
            continue

        hw = None

        # Try arc recovery from cutout segment metadata
        cut_seg_dicts = None
        if (stiff.cutout_segments and ci < len(stiff.cutout_segments)
                and stiff.cutout_segments[ci]):
            cut_seg_dicts = stiff.cutout_segments[ci]

        if cut_seg_dicts:
            cut_segs = _stiffener_segments_to_outline_segments(cut_seg_dicts)
            # Cutout vertices are already CW (ensured by stiffener extraction)
            # — use them as-is, no reversal needed for OCP hole wires
            tagged = _recover_arcs(cut_verts, cut_segs)
            hw = _make_3d_wire_xform(tagged, _xform, n)

        # Fallback: check if it's a circle
        if hw is None:
            nc = len(cut_verts)
            cx = sum(p[0] for p in cut_verts) / nc
            cy = sum(p[1] for p in cut_verts) / nc
            dists = [math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                     for p in cut_verts]
            avg_r = sum(dists) / len(dists)
            max_dev = max(abs(d - avg_r) for d in dists)

            if nc >= 6 and max_dev < avg_r * 0.08:
                pe = _xform((cx + avg_r, cy))
                pw = _xform((cx - avg_r, cy))
                pn = _xform((cx, cy + avg_r))
                ps = _xform((cx, cy - avg_r))
                pts = _snap_to_plane([pe, pn, pw, ps], n)
                try:
                    a1 = Edge.make_three_point_arc(
                        _vec(pts[0]), _vec(pts[1]), _vec(pts[2]))
                    a2 = Edge.make_three_point_arc(
                        _vec(pts[2]), _vec(pts[3]), _vec(pts[0]))
                    hw = _make_wire([a1, a2])
                except Exception:
                    pass

        # Final fallback: polygon (already CW, no reversal)
        if hw is None:
            hw = _make_polygon_wire(_xform, cut_verts, n)

        if hw is not None:
            hole_wires.append(hw)

    face = _make_face(outer_wire, hole_wires if hole_wires else None)
    if face is None:
        return None
    return _extrude_face(face, extr_dir)


def _make_3d_wire_xform(tagged_edges, transform_fn, normal):
    """Build a 3D wire from tagged edges using a custom transform function.

    Like ``_make_3d_wire`` but uses ``transform_fn(p2d) -> p3d`` instead
    of ``transform_point(p2d, recipe)``.
    """
    n = _normalize(normal)
    edges: list = []

    for te in tagged_edges:
        start_3d = transform_fn(te.start)
        end_3d = transform_fn(te.end)

        d = _sub(end_3d, start_3d)
        if math.sqrt(_dot(d, d)) < 1e-9:
            continue

        if te.type == "arc" and te.center is not None:
            mid_2d = _arc_midpoint_2d(
                te.start, te.end, te.center, te.radius, te.ccw)
            mid_3d = transform_fn(mid_2d)
            pts = _snap_to_plane([start_3d, mid_3d, end_3d], n)
            try:
                edges.append(Edge.make_three_point_arc(
                    _vec(pts[0]), _vec(pts[1]), _vec(pts[2])))
                continue
            except Exception:
                pass

        edges.append(Edge.make_line(_vec(start_3d), _vec(end_3d)))

    if not edges:
        return None
    return _make_wire(edges)


def _make_polygon_wire(transform_fn, verts_2d, normal):
    """Build a 3D polygon wire from 2D vertices — straight edges only."""
    n = _normalize(normal)
    pts_3d = [transform_fn(v) for v in verts_2d]
    nv = len(pts_3d)
    if nv < 3:
        return None
    edges: list = []
    for i in range(nv):
        p1 = pts_3d[i]
        p2 = pts_3d[(i + 1) % nv]
        d = _sub(p2, p1)
        if math.sqrt(_dot(d, d)) < 1e-9:
            continue
        edges.append(Edge.make_line(_vec(p1), _vec(p2)))
    if not edges:
        return None
    return _make_wire(edges)


def _make_arc_recovered_wire(verts_2d, transform_fn, normal):
    """Build a 3D wire, auto-detecting circular arcs from vertex sequences.

    Uses the transform_fn(p2d) → p3d to map each vertex.
    Detects arcs by checking if consecutive vertices are co-circular.
    """
    n = _normalize(normal)
    pts_3d = [transform_fn(v) for v in verts_2d]
    nv = len(pts_3d)
    if nv < 3:
        return None

    edges: list = []
    i = 0
    while i < nv:
        p1 = pts_3d[i]
        p2 = pts_3d[(i + 1) % nv]
        d = _sub(p2, p1)
        if math.sqrt(_dot(d, d)) < 1e-9:
            i += 1
            continue

        # Try to find an arc span: check if 3+ consecutive points are co-circular
        arc_end = _find_arc_span(pts_3d, i, n)
        if arc_end is not None and arc_end > i + 1:
            # Use 3-point arc: start, mid, end
            mid_idx = (i + arc_end) // 2
            start = pts_3d[i]
            mid = pts_3d[mid_idx]
            end = pts_3d[arc_end % nv]
            snapped = _snap_to_plane([start, mid, end], n)
            try:
                edges.append(Edge.make_three_point_arc(
                    _vec(snapped[0]), _vec(snapped[1]), _vec(snapped[2])))
                i = arc_end
                continue
            except Exception:
                pass

        # Straight line segment
        edges.append(Edge.make_line(_vec(p1), _vec(p2)))
        i += 1

    if not edges:
        return None
    return _make_wire(edges)


def _find_arc_span(pts_3d, start_idx, normal):
    """Find the longest arc span starting at start_idx.

    Returns the end index if 4+ consecutive points lie on a circle,
    else None.
    """
    n_pts = len(pts_3d)
    if n_pts < 4:
        return None

    # We need at least 3 consecutive points to define a circle,
    # then check if subsequent points also lie on it.
    i0 = start_idx
    i1 = (i0 + 1) % n_pts
    i2 = (i0 + 2) % n_pts

    p0, p1, p2 = pts_3d[i0], pts_3d[i1], pts_3d[i2]

    # Compute circle center from 3 points (in the normal plane)
    # Use the circumcenter formula
    a = _sub(p1, p0)
    b = _sub(p2, p0)
    axb = _cross(a, b)
    denom = 2.0 * _dot(axb, axb)
    if denom < 1e-20:
        return None  # collinear

    aa = _dot(a, a)
    bb = _dot(b, b)
    center = _add(p0, _scale(
        _add(_scale(_cross(axb, a), bb), _scale(_cross(b, axb), aa)),
        1.0 / denom))
    radius = math.sqrt(_dot(_sub(p0, center), _sub(p0, center)))

    if radius > 200:  # too large = probably not an arc
        return None

    # Extend: check how many subsequent points also lie on this circle
    tolerance = radius * 0.02  # 2% tolerance
    last_good = i2
    for k in range(3, min(n_pts, 100)):
        idx = (i0 + k) % n_pts
        if idx == i0:
            break
        pt = pts_3d[idx]
        dist = math.sqrt(_dot(_sub(pt, center), _sub(pt, center)))
        if abs(dist - radius) > tolerance:
            break
        last_good = idx

    # Need at least 4 points on the arc (3 define it, 4th confirms)
    span = last_good - i0 if last_good >= i0 else last_good + n_pts - i0
    if span < 3:
        return None

    # Sagitta check: reject arcs whose bulge from the chord is negligible
    # (nearly-collinear points on a huge-radius circle).
    chord = _sub(pts_3d[last_good % n_pts], pts_3d[i0])
    chord_len = math.sqrt(_dot(chord, chord))
    if chord_len > 0 and radius > chord_len * 0.5:
        sagitta = radius - math.sqrt(radius * radius - (chord_len / 2) ** 2)
        if sagitta < 0.01:  # less than 10 µm bulge → straight line
            return None

    return last_good


# =====================================================================
# Unified PCB solid construction (shell sewing — no boolean fuse)
# =====================================================================

def _edge_key_2d(p1, p2, decimals=5):
    """Canonical key for a 2D edge, order-independent."""
    a = (round(p1[0], decimals), round(p1[1], decimals))
    b = (round(p2[0], decimals), round(p2[1], decimals))
    return (a, b) if a <= b else (b, a)


def _build_edge_frequency(regions):
    """Count how many regions each outline edge appears in.

    Perimeter edges (board outline, cutout walls) appear once.
    Internal edges (fold zone boundaries) appear twice.
    """
    freq: dict = {}
    for region in regions:
        nv = len(region.outline)
        for i in range(nv):
            key = _edge_key_2d(region.outline[i], region.outline[(i + 1) % nv])
            freq[key] = freq.get(key, 0) + 1
    return freq


def _translation_trsf(vec):
    """Create a gp_Trsf for translation by a 3D vector tuple."""
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(vec[0], vec[1], vec[2]))
    return trsf


def _build_flat_skin_faces(region, recipe_defs, thickness,
                           outline_segments, cutout_segments, circle_holes):
    """Build top face, bottom face, and cutout wall faces for a flat region.

    ``outline_segments`` is used for outer boundary arc recovery.
    ``cutout_segments`` is used for hole arc recovery (kept separate to
    prevent cross-contamination that causes wild arc faces).

    No side faces are created here — those are handled by
    ``_build_perimeter_side_faces`` for outline edges that lie on
    the board perimeter.
    """
    normal = _compute_region_normal(region, recipe_defs)
    n = _normalize(normal)

    # --- Outer wire (top) with arc recovery (outline segments only) ---
    if outline_segments:
        tagged_edges = _recover_arcs(region.outline, outline_segments)
    else:
        nv = len(region.outline)
        tagged_edges = [
            TaggedEdge(type="line",
                       start=region.outline[i],
                       end=region.outline[(i + 1) % nv])
            for i in range(nv)
        ]

    outer_wire_top = _make_3d_wire(tagged_edges, recipe_defs, normal)
    if outer_wire_top is None:
        return []

    # --- Hole wires (top) — circles, arc-recovered, or polygon fallback ---
    hole_wires_top: list = []
    for hole in region.holes:
        hw = None

        # 1) Try exact circle match
        if circle_holes:
            match = _match_circle_hole(hole, circle_holes)
            if match:
                cx, cy, r = match
                hw = _make_circle_wire_3d((cx, cy), r, recipe_defs, normal)

        # 2) Try arc recovery from cutout segments only
        if hw is None and cutout_segments:
            hole_tagged = _recover_arcs(hole, cutout_segments)
            if any(e.type == "arc" for e in hole_tagged):
                rev = list(reversed(hole_tagged))
                rev_tagged = [
                    TaggedEdge(type=e.type, start=e.end, end=e.start,
                               center=e.center, radius=e.radius,
                               ccw=(not e.ccw) if e.type == "arc" else e.ccw)
                    for e in rev
                ]
                hw = _make_3d_wire(rev_tagged, recipe_defs, normal)

        # 3) Fallback: all-line polygon (reversed for CW)
        if hw is None:
            hw = _make_polygon_wire_3d(list(reversed(hole)), recipe_defs)

        if hw is not None:
            hole_wires_top.append(hw)

    # --- Top face ---
    top_face = _make_face(outer_wire_top,
                          hole_wires_top if hole_wires_top else None)
    if top_face is None:
        return []

    # --- Bottom face (translated by -n*thickness, reversed winding) ---
    offset_vec = (-n[0] * thickness, -n[1] * thickness, -n[2] * thickness)
    loc = TopLoc_Location(_translation_trsf(offset_vec))

    outer_wire_bot = Wire(TopoDS.Wire_s(
        outer_wire_top.wrapped.Moved(loc).Reversed()))

    hole_wires_bot: list = []
    for hw in hole_wires_top:
        hole_wires_bot.append(Wire(TopoDS.Wire_s(
            hw.wrapped.Moved(loc).Reversed())))

    bottom_face = _make_face(outer_wire_bot,
                             hole_wires_bot if hole_wires_bot else None)
    if bottom_face is None:
        return []

    faces = [top_face, bottom_face]

    # --- Cutout wall faces (extrude each hole wire in -normal direction) ---
    extrude_vec = gp_Vec(offset_vec[0], offset_vec[1], offset_vec[2])
    for hw in hole_wires_top:
        try:
            prism = BRepPrimAPI_MakePrism(hw.wrapped, extrude_vec)
            prism.Build()
            if prism.IsDone():
                exp = TopExp_Explorer(prism.Shape(), TopAbs_FACE)
                while exp.More():
                    faces.append(Face(exp.Current()))
                    exp.Next()
        except Exception:
            pass

    return faces


def _bend_cylinder_params(region, recipe_defs):
    """Extract cylinder axis, reference direction, theta_min, and sweep.

    Returns ``(cyl_axis, cyl_ref, theta_min, sweep)`` or *None*.
    """
    fold_def = None
    entered_from_back = False
    for entry in reversed(recipe_defs):
        if entry[1] == "IN_ZONE":
            fold_def = entry[0]
            entered_from_back = entry[2] if len(entry) > 2 else False
            break
    if fold_def is None:
        return None

    prior_recipe = recipe_defs[:-1]
    hw = fold_def.zone_width / 2

    # perp extents → theta range
    perp_dists = []
    for p in region.outline:
        dx = p[0] - fold_def.center[0]
        dy = p[1] - fold_def.center[1]
        perp_dists.append(dx * fold_def.perp[0] + dy * fold_def.perp[1])
    perp_min, perp_max = min(perp_dists), max(perp_dists)

    zw = fold_def.zone_width
    frac_min = max(0, perp_min + hw) / zw if zw > 0 else 0
    frac_max = min(zw, perp_max + hw) / zw if zw > 0 else 0
    sign_angle = 1.0 if fold_def.angle >= 0 else -1.0

    # Build accumulated rotation from prior folds
    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for entry in prior_recipe:
        f, c = entry[0], entry[1]
        back = entry[2] if len(entry) > 2 else False
        if c == "AFTER":
            f_ax = _apply_rotation(rot, (f.axis[0], f.axis[1], 0.0))
            ra = -f.angle if back else f.angle
            rot = _multiply_matrices(
                _rotation_matrix_around_axis(f_ax, ra), rot)

    fold_axis_3d = _apply_rotation(rot, (fold_def.axis[0], fold_def.axis[1], 0.0))
    fold_perp_3d = _apply_rotation(rot, (fold_def.perp[0], fold_def.perp[1], 0.0))
    up_3d = _apply_rotation(rot, (0.0, 0.0, 1.0))

    ref_2d = (fold_def.center[0] + hw * fold_def.perp[0],
              fold_def.center[1] + hw * fold_def.perp[1])
    ref_3d = transform_point(ref_2d, prior_recipe)
    fold_center_3d = _sub(ref_3d, _scale(fold_perp_3d, hw))

    if entered_from_back:
        entry_3d = _add(fold_center_3d, _scale(fold_perp_3d, hw))
    else:
        entry_3d = _sub(fold_center_3d, _scale(fold_perp_3d, hw))

    R = fold_def.radius
    cyl_origin = _add(entry_3d, _scale(up_3d, sign_angle * R))
    cyl_ref = _normalize(_scale(up_3d, -sign_angle))
    axis_sign = -sign_angle if entered_from_back else sign_angle
    cyl_axis = _normalize(_scale(fold_axis_3d, axis_sign))

    # Theta reversal after axis setup (matches step_export.py)
    if entered_from_back:
        theta_min = abs(fold_def.angle) - frac_max * abs(fold_def.angle)
        theta_max_val = abs(fold_def.angle) - frac_min * abs(fold_def.angle)
    else:
        theta_min = frac_min * abs(fold_def.angle)
        theta_max_val = frac_max * abs(fold_def.angle)

    sweep = theta_max_val - theta_min
    if sweep < 1e-9:
        return None

    return (cyl_axis, cyl_ref, theta_min, sweep)


def _extract_skin_faces(solid, cyl_axis, cyl_ref, theta_min, sweep):
    """Extract all faces from a revolved solid *except* the theta-boundary faces.

    The two planar faces at theta_min and theta_min + sweep are internal
    boundaries (they connect to adjacent flat regions).  They are identified
    by having a planar surface whose normal aligns with the cylinder tangent
    at the start or end angle.
    """
    tangent0 = _normalize(_cross(cyl_axis, cyl_ref))

    cos_s = math.cos(theta_min)
    sin_s = math.sin(theta_min)
    rad_start = _add(_scale(cyl_ref, cos_s), _scale(tangent0, sin_s))
    tan_start = _normalize(_cross(cyl_axis, rad_start))

    cos_e = math.cos(theta_min + sweep)
    sin_e = math.sin(theta_min + sweep)
    rad_end = _add(_scale(cyl_ref, cos_e), _scale(tangent0, sin_e))
    tan_end = _normalize(_cross(cyl_axis, rad_end))

    faces: list = []
    explorer = TopExp_Explorer(solid.wrapped, TopAbs_FACE)
    while explorer.More():
        face_shape = explorer.Current()
        is_theta_face = False

        try:
            surf_handle = BRep_Tool.Surface_s(face_shape)
            adaptor = GeomAdaptor_Surface(surf_handle)
            if adaptor.GetType() == GeomAbs_Plane:
                plane = adaptor.Plane()
                fn = plane.Axis().Direction()
                fn_vec = _normalize((fn.X(), fn.Y(), fn.Z()))

                if abs(_dot(fn_vec, tan_start)) > 0.99:
                    is_theta_face = True
                elif abs(_dot(fn_vec, tan_end)) > 0.99:
                    is_theta_face = True
        except Exception:
            pass

        if not is_theta_face:
            faces.append(Face(face_shape))

        explorer.Next()

    return faces


def _build_bend_skin_faces(region, recipe_defs, thickness, original_segments,
                           edge_freq=None):
    """Build skin faces for a bend region using B-spline surfaces.

    Samples a regular grid via ``transform_point`` (same math as the viewer)
    and fits a B-spline surface through the grid.  The surface is trimmed to
    the outline boundary, producing a smooth curved face that correctly
    follows arc fillets crossing the bend zone.

    Also creates the offset bottom surface and perimeter side faces.
    """
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.Approx import Approx_ParametrizationType

    fold_def = None
    entered_from_back = False
    for entry in reversed(recipe_defs):
        if entry[1] == "IN_ZONE":
            fold_def = entry[0]
            entered_from_back = entry[2] if len(entry) > 2 else False
            break
    if fold_def is None:
        return _build_flat_skin_faces(region, recipe_defs, thickness,
                                      original_segments, [], None)

    hw = fold_def.zone_width / 2
    cx, cy = fold_def.center
    ax_d, ay_d = fold_def.axis
    px_d, py_d = fold_def.perp

    # Compute along/perp extents from region outline
    along_vals = []
    perp_vals = []
    for p in region.outline:
        dx, dy = p[0] - cx, p[1] - cy
        along_vals.append(dx * ax_d + dy * ay_d)
        perp_vals.append(dx * px_d + dy * py_d)
    al_min, al_max = min(along_vals), max(along_vals)
    pe_min, pe_max = min(perp_vals), max(perp_vals)

    # Sample a regular grid in (along, perp) and transform to 3D
    n_along = max(4, int((al_max - al_min) / 0.5) + 1)
    n_perp = max(4, int((pe_max - pe_min) / 0.3) + 1)

    def _sample(al, pe, bottom=False):
        """Sample a point on the bend surface via transform_point."""
        x = cx + al * ax_d + pe * px_d
        y = cy + al * ay_d + pe * py_d
        pt = transform_point((x, y), recipe_defs)
        if bottom:
            # Per-point normal offset (varies across the bend zone)
            nrm = compute_normal((x, y), recipe_defs)
            pt = _add(pt, _scale(nrm, -thickness))
        return pt

    faces: list = []

    # Build B-spline surfaces for top and bottom skins
    for bottom, reverse in [(False, False), (True, True)]:
        grid = TColgp_Array2OfPnt(1, n_along, 1, n_perp)
        for i in range(n_along):
            al = al_min + (al_max - al_min) * i / (n_along - 1)
            for j in range(n_perp):
                pe = pe_min + (pe_max - pe_min) * j / (n_perp - 1)
                pt = _sample(al, pe, bottom=bottom)
                grid.SetValue(i + 1, j + 1,
                              gp_Pnt(pt[0], pt[1], pt[2]))

        try:
            surf_maker = GeomAPI_PointsToBSplineSurface(
                grid, 3, 8, GeomAbs_C2, 0.001)
            if not surf_maker.IsDone():
                continue
            surface = surf_maker.Surface()

            # Build trimming wire with B-spline edges for perimeter chains
            # and straight lines for fold-boundary edges.
            from OCP.GeomAPI import GeomAPI_PointsToBSpline as _PtsBSpl
            from OCP.TColgp import TColgp_Array1OfPnt as _Arr1Pnt

            nv = len(region.outline)

            def _outline_pt(idx):
                p2d = region.outline[idx % nv]
                pt = transform_point(p2d, recipe_defs)
                if bottom:
                    nrm = compute_normal(p2d, recipe_defs)
                    pt = _add(pt, _scale(nrm, -thickness))
                return pt

            # Classify edges as perimeter vs fold-boundary
            is_perim = []
            for i in range(nv):
                p1 = region.outline[i]
                p2 = region.outline[(i + 1) % nv]
                key = _edge_key_2d(p1, p2)
                is_perim.append(
                    (edge_freq.get(key, 1) < 2) if edge_freq else True)

            # Walk the outline in order (CCW for top, CW for bottom)
            if reverse:
                order = list(range(nv - 1, -1, -1))
            else:
                order = list(range(nv))

            wire_edges = []
            i = 0
            while i < len(order):
                idx = order[i]
                edge_i = idx if not reverse else (idx - 1) % nv
                # Check perim status of the ORIGINAL edge
                perim = is_perim[min(idx, (idx + 1) % nv) if not reverse
                                 else min((idx - 1) % nv, idx)]
                # For perimeter edges: collect a contiguous chain and B-spline it
                if is_perim[order[i] if not reverse else (order[i] - 1) % nv]:
                    chain_pts = [_outline_pt(order[i])]
                    while i < len(order):
                        ei = order[i] if not reverse else (order[i] - 1) % nv
                        if not is_perim[ei]:
                            break
                        chain_pts.append(_outline_pt(
                            order[(i + 1) % len(order)]))
                        i += 1
                    # Build B-spline edge through chain points
                    if len(chain_pts) >= 3:
                        arr = _Arr1Pnt(1, len(chain_pts))
                        for k, pt in enumerate(chain_pts):
                            arr.SetValue(k + 1, gp_Pnt(pt[0], pt[1], pt[2]))
                        bsp = _PtsBSpl(arr, 3, 8)
                        if bsp.IsDone():
                            from OCP.BRepBuilderAPI import \
                                BRepBuilderAPI_MakeEdge as _MkE
                            emk = _MkE(bsp.Curve())
                            emk.Build()
                            if emk.IsDone():
                                wire_edges.append(Edge(emk.Shape()))
                                continue
                    # Fallback: line segments
                    for k in range(len(chain_pts) - 1):
                        d = _sub(chain_pts[k + 1], chain_pts[k])
                        if math.sqrt(_dot(d, d)) > 1e-9:
                            wire_edges.append(Edge.make_line(
                                _vec(chain_pts[k]),
                                _vec(chain_pts[k + 1])))
                else:
                    # Fold-boundary edge: straight line
                    p1 = _outline_pt(order[i])
                    p2 = _outline_pt(order[(i + 1) % len(order)])
                    d = _sub(p2, p1)
                    if math.sqrt(_dot(d, d)) > 1e-9:
                        wire_edges.append(Edge.make_line(_vec(p1), _vec(p2)))
                    i += 1

            wire = _make_wire(wire_edges)
            face_maker = BRepBuilderAPI_MakeFace(surface, wire.wrapped, True)
            face_maker.Build()
            if face_maker.IsDone() and not face_maker.Face().IsNull():
                faces.append(Face(face_maker.Face()))
        except Exception:
            pass

    # Side faces: for each contiguous chain of perimeter edges (not fold
    # boundaries), fit B-spline curves through the top and bottom points
    # and loft a single smooth face between them.
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.TColgp import TColgp_Array1OfPnt

    nv = len(region.outline)

    def _top_bot(idx):
        p2d = region.outline[idx % nv]
        t = transform_point(p2d, recipe_defs)
        nrm = compute_normal(p2d, recipe_defs)
        b = _add(t, _scale(nrm, -thickness))
        return t, b

    # Classify each edge as perimeter or internal
    is_perimeter = []
    for i in range(nv):
        p1 = region.outline[i]
        p2 = region.outline[(i + 1) % nv]
        key = _edge_key_2d(p1, p2)
        is_perim = (edge_freq.get(key, 1) < 2) if edge_freq else True
        is_perimeter.append(is_perim)

    # Extract contiguous perimeter chains (sequences of consecutive
    # perimeter edges, collecting the vertex indices)
    chains = []
    current_chain = []
    for i in range(nv):
        if is_perimeter[i]:
            if not current_chain:
                current_chain.append(i)
            current_chain.append((i + 1) % nv)
        else:
            if current_chain:
                chains.append(current_chain)
                current_chain = []
    if current_chain:
        # Check if it wraps around and connects to the first chain
        if chains and is_perimeter[(nv - 1)] and is_perimeter[0]:
            chains[0] = current_chain + chains[0][1:]
        else:
            chains.append(current_chain)

    def _bspline_wire(pts_3d):
        """Fit a B-spline through 3D points and return a single-edge Wire."""
        if len(pts_3d) < 2:
            return None
        if len(pts_3d) == 2:
            # Just a line
            return _make_wire([Edge.make_line(_vec(pts_3d[0]), _vec(pts_3d[1]))])
        arr = TColgp_Array1OfPnt(1, len(pts_3d))
        for k, pt in enumerate(pts_3d):
            arr.SetValue(k + 1, gp_Pnt(pt[0], pt[1], pt[2]))
        bsp = GeomAPI_PointsToBSpline(arr, 3, 8)
        if not bsp.IsDone():
            return None
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        edge_mk = BRepBuilderAPI_MakeEdge(bsp.Curve())
        edge_mk.Build()
        if not edge_mk.IsDone():
            return None
        return _make_wire([Edge(edge_mk.Shape())])

    for chain in chains:
        top_pts = [_top_bot(idx)[0] for idx in chain]
        bot_pts = [_top_bot(idx)[1] for idx in chain]

        top_wire = _bspline_wire(top_pts)
        bot_wire = _bspline_wire(bot_pts)
        if top_wire is None or bot_wire is None:
            continue
        try:
            loft = BRepOffsetAPI_ThruSections(False, False, 1e-6)
            loft.AddWire(top_wire.wrapped)
            loft.AddWire(bot_wire.wrapped)
            loft.Build()
            if loft.IsDone():
                exp = TopExp_Explorer(loft.Shape(), TopAbs_FACE)
                while exp.More():
                    faces.append(Face(exp.Current()))
                    exp.Next()
        except Exception:
            pass

    return faces


def _build_top_edge_3d(p1_2d, p2_2d, tagged_edges, edge_idx,
                       recipe_defs, normal):
    """Build a 3D OCP Edge for an outline edge, preserving arc geometry."""
    n = _normalize(normal)

    start_3d = transform_point(p1_2d, recipe_defs)
    end_3d = transform_point(p2_2d, recipe_defs)
    d = _sub(end_3d, start_3d)
    if math.sqrt(_dot(d, d)) < 1e-9:
        return None

    if tagged_edges is not None and edge_idx < len(tagged_edges):
        te = tagged_edges[edge_idx]
        if te.type == "arc" and te.center is not None:
            mid_2d = _arc_midpoint_2d(
                te.start, te.end, te.center, te.radius, te.ccw)
            mid_3d = transform_point(mid_2d, recipe_defs)
            pts = _snap_to_plane([start_3d, mid_3d, end_3d], n)
            try:
                return Edge.make_three_point_arc(
                    _vec(pts[0]), _vec(pts[1]), _vec(pts[2]))
            except Exception:
                pass

    return Edge.make_line(_vec(start_3d), _vec(end_3d))


def _build_perimeter_side_faces(regions, thickness, outline_segments, edge_freq):
    """Build side faces for perimeter outline edges of flat regions.

    Bend regions' along-axis end caps are already included via
    ``_build_bend_skin_faces`` (revolved end-cap edges).
    Only flat-region perimeter edges need explicit side walls.

    Each edge is matched against original arc segments directly (not via
    ``_recover_arcs`` which merges consecutive arcs and breaks the 1:1
    index correspondence with the region outline).
    """
    # Pre-filter arc segments for efficiency
    arc_segments = ([s for s in outline_segments
                     if s.type == "arc" and s.center is not None]
                    if outline_segments else [])

    faces: list = []

    for region in regions:
        recipe_defs = _recipe_with_fold_defs(region.fold_recipe)

        if _is_bend_region(region.fold_recipe):
            continue

        normal = _compute_region_normal(region, recipe_defs)
        n = _normalize(normal)
        offset_vec = gp_Vec(-n[0] * thickness, -n[1] * thickness,
                            -n[2] * thickness)

        nv = len(region.outline)
        for i in range(nv):
            p1 = region.outline[i]
            p2 = region.outline[(i + 1) % nv]
            key = _edge_key_2d(p1, p2)

            if edge_freq.get(key, 0) >= 2:
                continue  # internal edge

            start_3d = transform_point(p1, recipe_defs)
            end_3d = transform_point(p2, recipe_defs)
            d = _sub(end_3d, start_3d)
            if math.sqrt(_dot(d, d)) < 1e-9:
                continue

            # Match this edge directly against original arc segments
            top_edge = None
            for arc in arc_segments:
                if (_point_on_arc(p1, arc) and _point_on_arc(p2, arc)):
                    # Compute ccw for THIS edge (p1→p2)
                    cx, cy = arc.center
                    sx, sy = p1[0] - cx, p1[1] - cy
                    ex, ey = p2[0] - cx, p2[1] - cy
                    ccw = (sx * ey - sy * ex) > 0

                    mid_2d = _arc_midpoint_2d(
                        p1, p2, arc.center, arc.radius, ccw)
                    mid_3d = transform_point(mid_2d, recipe_defs)
                    pts = _snap_to_plane([start_3d, mid_3d, end_3d], n)
                    try:
                        top_edge = Edge.make_three_point_arc(
                            _vec(pts[0]), _vec(pts[1]), _vec(pts[2]))
                    except Exception:
                        pass
                    break

            if top_edge is None:
                top_edge = Edge.make_line(_vec(start_3d), _vec(end_3d))

            try:
                prism = BRepPrimAPI_MakePrism(top_edge.wrapped, offset_vec)
                prism.Build()
                if prism.IsDone():
                    faces.append(Face(prism.Shape()))
            except Exception:
                pass

    return faces


def _sew_to_solid(faces, tolerance=1e-4):
    """Sew faces into a closed shell and convert to a Solid.

    Returns a ``Solid`` on success, *None* on failure.
    """
    if not faces:
        return None

    sewer = BRepBuilderAPI_Sewing(tolerance)
    for f in faces:
        sewer.Add(f.wrapped)
    sewer.Perform()
    sewn = sewer.SewedShape()

    # Extract shells
    shells: list = []
    explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
    while explorer.More():
        shells.append(TopoDS.Shell_s(explorer.Current()))
        explorer.Next()

    if not shells:
        return None

    maker = BRepBuilderAPI_MakeSolid()
    for shell in shells:
        maker.Add(shell)

    if not maker.IsDone():
        return None

    raw_solid = maker.Solid()

    # Fix face orientations
    try:
        fixer = ShapeFix_Solid(raw_solid)
        fixer.Perform()
        raw_solid = fixer.Solid()
    except Exception:
        pass

    # Ensure positive volume (negative = inside-out normals)
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(raw_solid, props)
    if props.Mass() < 0:
        raw_solid.Reverse()

    return Solid(raw_solid)


def _build_unified_pcb_ocp(regions, thickness, original_segments,
                           circle_holes, cutout_segments):
    """Build the PCB as a single solid via face construction + sewing.

    Instead of building separate closed solids per region and fusing them
    (fragile when faces are coincident), this function constructs only the
    *exterior* skin faces and side walls, then sews them into one solid:

    * Flat regions → top + bottom planar faces, cutout wall faces.
    * Bend regions → inner/outer cylindrical faces + along-axis end caps
      (theta-boundary faces are discarded as they are internal).
    * Perimeter outline edges → side wall faces connecting top to bottom.

    ``original_segments`` is used for outline arc recovery (side faces,
    flat face outer boundaries).  ``cutout_segments`` is used only for
    hole arc recovery, keeping them separate to avoid cross-contamination.
    """
    # Flatten cutout segments for hole arc recovery (kept separate from outline)
    flat_cutout_segs: list = []
    if cutout_segments:
        for segs in cutout_segments:
            flat_cutout_segs.extend(segs)

    edge_freq = _build_edge_frequency(regions)

    all_faces: list = []
    for region in regions:
        recipe_defs = _recipe_with_fold_defs(region.fold_recipe)

        if _is_bend_region(region.fold_recipe):
            faces = _build_bend_skin_faces(
                region, recipe_defs, thickness, original_segments,
                edge_freq)
        else:
            faces = _build_flat_skin_faces(
                region, recipe_defs, thickness,
                original_segments, flat_cutout_segs, circle_holes)
        all_faces.extend(faces)

    # Side walls for perimeter outline edges — use outline segments ONLY
    side_faces = _build_perimeter_side_faces(
        regions, thickness, original_segments, edge_freq)
    all_faces.extend(side_faces)

    # Sew into a single solid
    solid = _sew_to_solid(all_faces)

    if solid is not None:
        return solid

    # Fallback: individual solids as a compound (no fuse)
    solids: list = []
    for region in regions:
        recipe_defs = _recipe_with_fold_defs(region.fold_recipe)
        if _is_bend_region(region.fold_recipe):
            s = _build_bend_solid_ocp(
                region, recipe_defs, thickness, original_segments)
        else:
            s = _build_flat_solid_ocp(
                region, recipe_defs, thickness, original_segments,
                circle_holes=circle_holes)
        if s is not None:
            solids.append(s)
    return Compound(solids) if solids else None


# =====================================================================
# XDE assembly writers
# =====================================================================

def _unwrap(shape):
    """Get raw OCP TopoDS_Shape from either a build123d wrapper or raw shape."""
    return shape.wrapped if hasattr(shape, 'wrapped') else shape


def _write_step_assembly(shapes, filename):
    """Write STEP with named bodies: FLEX_PCB + STIFFENER_0 … STIFFENER_N.

    ``shapes`` is a list of OCP TopoDS_Shape or build123d wrappers.
    First element is the PCB body, rest are stiffeners.
    """
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDataStd import TDataStd_Name
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_StepModelType
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.TopLoc import TopLoc_Location as TL

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XDE"))
    app.InitDocument(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    # Root assembly
    builder = BRep_Builder()
    empty = TopoDS_Compound()
    builder.MakeCompound(empty)
    root = shape_tool.AddShape(empty, True)
    TDataStd_Name.Set_s(root, TCollection_ExtendedString("FlexPCB_Assembly"))

    # PCB body
    pcb_label = shape_tool.AddShape(_unwrap(shapes[0]), False)
    TDataStd_Name.Set_s(pcb_label, TCollection_ExtendedString("FLEX_PCB"))
    shape_tool.AddComponent(root, pcb_label, TL())

    # Stiffeners
    for idx, s in enumerate(shapes[1:]):
        lbl = shape_tool.AddShape(_unwrap(s), False)
        TDataStd_Name.Set_s(
            lbl, TCollection_ExtendedString(f"STIFFENER_{idx}"))
        shape_tool.AddComponent(root, lbl, TL())

    shape_tool.UpdateAssemblies()

    # Disable pcurve output — reduces file size by ~70% with no geometry loss
    from OCP.Interface import Interface_Static
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 0)

    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
    status = writer.Write(filename)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed with status {status}")


# =====================================================================
# XDE assembly writer  (shared geometry for component models)
# =====================================================================

def _write_step_with_models(pcb_solids, board_geometry, regions, thickness,
                            filename, pcb_dir=None, pcb=None, include_wrl=False,
                            mirror_trsf=None, bbox_models=False):
    """
    Write STEP using XCAF/XDE assembly structure.

    Each unique STEP model is embedded once; instances reference it via
    placement transforms.  ``mirror_trsf`` (Y-flip) is baked into each
    placement so the entire assembly is in CAD Y-up coordinates.

    With ``bbox_models=True``, components are represented as lightweight
    bounding boxes instead of full 3D geometry.
    """
    import os
    from build123d import import_step as bd_import_step, Box, Compound
    from OCP.gp import gp_Trsf, gp_Ax3
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDataStd import TDataStd_Name
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_StepModelType
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    # --- create XDE document -------------------------------------------
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XDE"))
    app.InitDocument(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    # --- root assembly (empty compound) --------------------------------
    builder = BRep_Builder()
    empty_compound = TopoDS_Compound()
    builder.MakeCompound(empty_compound)
    root_label = shape_tool.AddShape(empty_compound, True)
    TDataStd_Name.Set_s(
        root_label, TCollection_ExtendedString("FlexPCB_Assembly"))

    # --- add PCB body + stiffeners (already mirrored) ------------------
    pcb_label = shape_tool.AddShape(_unwrap(pcb_solids[0]), False)
    TDataStd_Name.Set_s(pcb_label, TCollection_ExtendedString("FLEX_PCB"))
    shape_tool.AddComponent(root_label, pcb_label, TopLoc_Location())

    for idx, s in enumerate(pcb_solids[1:]):
        lbl = shape_tool.AddShape(_unwrap(s), False)
        TDataStd_Name.Set_s(
            lbl, TCollection_ExtendedString(f"STIFFENER_{idx}"))
        shape_tool.AddComponent(root_label, lbl, TopLoc_Location())

    # --- collect component model instances -----------------------------
    step_cache = {}        # resolved_path -> Shape
    bbox_cache = {}        # resolved_path -> (dx, dy, dz) bounding box dims
    proto_labels = {}      # resolved_path -> TDF_Label (XDE prototype)

    for comp in board_geometry.components:
        if not comp.models:
            continue

        for model_info in comp.models:
            if model_info.hide:
                continue

            resolved = _resolve_model_path_for_step(
                model_info.path, pcb_dir, pcb, include_wrl)
            if resolved is None:
                continue

            ext = os.path.splitext(resolved)[1].lower()
            if ext not in ('.step', '.stp'):
                continue

            # Import unique model once
            if resolved not in step_cache:
                try:
                    step_cache[resolved] = bd_import_step(resolved)
                except Exception:
                    step_cache[resolved] = None
                    continue

            model_shape = step_cache[resolved]
            if model_shape is None:
                continue

            # For bbox mode: compute bounding box from imported model
            if bbox_models:
                if resolved not in bbox_cache:
                    bb = Bnd_Box()
                    BRepBndLib.Add_s(model_shape.wrapped, bb)
                    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
                    bbox_cache[resolved] = (
                        xmax - xmin, ymax - ymin, zmax - zmin,
                        xmin, ymin, zmin)

                # Create a box prototype for this model
                proto_key = resolved + "_bbox"
                if proto_key not in proto_labels:
                    dx, dy, dz, ox, oy, oz = bbox_cache[resolved]
                    if dx > 0.01 and dy > 0.01 and dz > 0.01:
                        box_shape = BRepPrimAPI_MakeBox(
                            gp_Pnt(ox, oy, oz), dx, dy, dz).Shape()
                        lbl = shape_tool.AddShape(box_shape, False)
                        name = os.path.splitext(
                            os.path.basename(resolved))[0] + "_bbox"
                        TDataStd_Name.Set_s(
                            lbl, TCollection_ExtendedString(name))
                        proto_labels[proto_key] = lbl
                resolved_key = proto_key
            else:
                # Register full model prototype in XDE once
                if resolved not in proto_labels:
                    lbl = shape_tool.AddShape(model_shape.wrapped, False)
                    name = os.path.splitext(os.path.basename(resolved))[0]
                    TDataStd_Name.Set_s(
                        lbl, TCollection_ExtendedString(name))
                    proto_labels[resolved] = lbl
                resolved_key = resolved

            if resolved_key not in proto_labels:
                continue

            # Compute placement
            placement = _compute_component_placement(
                comp, model_info, thickness, regions)
            if placement is None:
                continue
            origin, z_axis, x_axis = placement

            # Apply Y-mirror to the placement (KiCad Y-down → CAD Y-up)
            if mirror_trsf is not None:
                origin = (origin[0], -origin[1], origin[2])
                z_axis = (z_axis[0], -z_axis[1], z_axis[2])
                x_axis = (x_axis[0], -x_axis[1], x_axis[2])

            try:
                # Build the model→world affine matrix directly.
                # Columns = where model's X, Y, Z axes end up; translation = origin.
                y_axis = _normalize(_cross(z_axis, x_axis))
                trsf = gp_Trsf()
                trsf.SetValues(
                    x_axis[0], y_axis[0], z_axis[0], origin[0],
                    x_axis[1], y_axis[1], z_axis[1], origin[1],
                    x_axis[2], y_axis[2], z_axis[2], origin[2])

                shape_tool.AddComponent(
                    root_label, proto_labels[resolved_key],
                    TopLoc_Location(trsf))
            except Exception:
                continue

            break  # first valid model per component

    shape_tool.UpdateAssemblies()

    # --- write STEP ----------------------------------------------------
    # Disable pcurve output — reduces file size by ~70% with no geometry loss
    from OCP.Interface import Interface_Static
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 0)

    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
    status = writer.Write(filename)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed with status {status}")

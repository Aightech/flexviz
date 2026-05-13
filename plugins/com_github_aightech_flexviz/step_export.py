"""
STEP file export for flex PCB visualization.

Pure Python — no external dependencies. Uses StepWriter to emit B-Rep geometry
(PLANE + CYLINDRICAL_SURFACE) directly to STEP AP214 format.
"""

import math
import os
from dataclasses import dataclass

try:
    from .step_writer import StepWriter, _normalize, _cross, _sub, _add, _scale, _dot
    from .bend_transform import (
        FoldDefinition, transform_point, compute_normal,
        _rotation_matrix_around_axis, _apply_rotation, _multiply_matrices,
        recipe_from_region,
    )
    from .planar_subdivision import split_board_into_regions, find_containing_region
    from .stiffener import extract_stiffeners
except ImportError:
    from step_writer import StepWriter, _normalize, _cross, _sub, _add, _scale, _dot
    from bend_transform import (
        FoldDefinition, transform_point, compute_normal,
        _rotation_matrix_around_axis, _apply_rotation, _multiply_matrices,
        recipe_from_region,
    )
    from planar_subdivision import split_board_into_regions, find_containing_region
    from stiffener import extract_stiffeners


# =====================================================================
# TaggedEdge: edge with arc/line type info preserved after subdivision
# =====================================================================

@dataclass
class TaggedEdge:
    """An edge that knows whether it's a line or arc segment."""
    type: str          # "line" or "arc"
    start: tuple       # (x, y) 2D
    end: tuple         # (x, y) 2D
    center: tuple = None  # (x, y) for arcs
    radius: float = 0.0
    ccw: bool = True       # True = CCW arc from start to end


def _point_on_arc(pt, arc_segment, tolerance=0.01):
    """
    Check if a 2D point lies on an OutlineSegment arc.

    Verifies:
    1. Distance from point to arc center matches radius (within tolerance)
    2. Point's angle is within the arc's angular span (using mid point for sweep direction)

    The tolerance is automatically increased for points that may have been
    created by chord-based subdivision (fold zone cutting).  The maximum
    sagitta for a chord of length L on a circle of radius R is
    R - sqrt(R² - (L/2)²).  We use 15% of the radius as the upper bound.
    """
    if arc_segment.type != "arc" or arc_segment.center is None:
        return False

    cx, cy = arc_segment.center
    dx = pt[0] - cx
    dy = pt[1] - cy
    dist = math.sqrt(dx * dx + dy * dy)

    # Adaptive tolerance: allow up to 15% of radius for chord-displaced points
    effective_tol = max(tolerance, arc_segment.radius * 0.15)
    if abs(dist - arc_segment.radius) > effective_tol:
        return False

    # Check angular span using start, mid, end points
    angle_pt = math.atan2(dy, dx)
    angle_start = math.atan2(arc_segment.start[1] - cy, arc_segment.start[0] - cx)
    angle_end = math.atan2(arc_segment.end[1] - cy, arc_segment.end[0] - cx)

    if arc_segment.mid is not None:
        angle_mid = math.atan2(arc_segment.mid[1] - cy, arc_segment.mid[0] - cx)
    else:
        # Without mid, assume short arc from start to end CCW
        angle_mid = (angle_start + angle_end) / 2

    # Determine sweep direction from start -> mid -> end
    # Normalize all angles to [0, 2*pi)
    def norm_angle(a):
        return a % (2 * math.pi)

    a_s = norm_angle(angle_start)
    a_m = norm_angle(angle_mid)
    a_e = norm_angle(angle_end)
    a_p = norm_angle(angle_pt)

    # Check if mid is in the CCW sweep from start to end
    def in_ccw_sweep(start, end, test):
        """Check if test angle is in CCW sweep from start to end."""
        if start <= end:
            return start <= test <= end
        else:  # wraps around 2*pi
            return test >= start or test <= end

    # Determine if the arc goes CCW or CW by checking mid
    if in_ccw_sweep(a_s, a_e, a_m):
        # CCW from start to end
        return in_ccw_sweep(a_s, a_e, a_p)
    else:
        # CW from start to end = CCW from end to start
        return in_ccw_sweep(a_e, a_s, a_p)


def _recover_arcs(region_outline, original_segments, tolerance=0.01):
    """
    Recover arc info for edges in a region outline after planar subdivision.

    For each consecutive edge (v[i] -> v[i+1]), check if both endpoints lie on
    the same original arc segment. If yes, tag as arc with parent's center/radius.

    Args:
        region_outline: list of (x, y) vertices from subdivision
        original_segments: list of OutlineSegment from the original board outline

    Returns:
        list of TaggedEdge
    """
    n = len(region_outline)
    tagged = []

    # Filter to just arc segments for efficiency
    arc_segments = [s for s in original_segments if s.type == "arc" and s.center is not None]

    for i in range(n):
        p1 = region_outline[i]
        p2 = region_outline[(i + 1) % n]

        # Check if this edge lies on any original arc
        matched_arc = None
        for arc in arc_segments:
            if _point_on_arc(p1, arc, tolerance) and _point_on_arc(p2, arc, tolerance):
                matched_arc = arc
                break

        if matched_arc is not None:
            # Determine sweep direction for THIS edge (p1→p2) by checking
            # which direction keeps the midpoint on the original arc.
            # The cross-product approach only works for arcs < 180°;
            # using the original arc's angular span handles all cases.
            cx, cy = matched_arc.center
            a_s = math.atan2(p1[1] - cy, p1[0] - cx)
            a_e = math.atan2(p2[1] - cy, p2[0] - cx)
            r = matched_arc.radius

            # Compute CCW midpoint and check if it's on the original arc
            sweep_ccw = a_e - a_s
            if sweep_ccw <= 0:
                sweep_ccw += 2 * math.pi
            a_mid_ccw = a_s + sweep_ccw / 2
            mid_ccw = (cx + r * math.cos(a_mid_ccw),
                       cy + r * math.sin(a_mid_ccw))
            ccw = _point_on_arc(mid_ccw, matched_arc, tolerance)
            tagged.append(TaggedEdge(
                type="arc",
                start=p1,
                end=p2,
                center=matched_arc.center,
                radius=matched_arc.radius,
                ccw=ccw,
            ))
        else:
            tagged.append(TaggedEdge(
                type="line",
                start=p1,
                end=p2,
            ))

    # Merge consecutive arcs on the same circle into single arcs
    merged = _merge_consecutive_arcs(tagged, tolerance)
    return merged


def _merge_consecutive_arcs(tagged, tolerance=0.01):
    """Merge consecutive TaggedEdges that are arcs on the same circle.

    Adjacent arcs with matching center and radius are combined into a single
    arc edge spanning from the first arc's start to the last arc's end.
    """
    if not tagged:
        return tagged

    merged = []
    i = 0
    while i < len(tagged):
        e = tagged[i]
        if e.type != 'arc':
            merged.append(e)
            i += 1
            continue

        # Start a run of arcs on the same circle
        run_start = e.start
        run_end = e.end
        cx, cy, r = e.center[0], e.center[1], e.radius
        ccw = e.ccw
        j = i + 1
        while j < len(tagged):
            nxt = tagged[j]
            if (nxt.type == 'arc' and
                    abs(nxt.center[0] - cx) < tolerance and
                    abs(nxt.center[1] - cy) < tolerance and
                    abs(nxt.radius - r) < tolerance and
                    nxt.ccw == ccw):
                run_end = nxt.end
                j += 1
            else:
                break

        merged.append(TaggedEdge(
            type='arc',
            start=run_start,
            end=run_end,
            center=e.center,
            radius=e.radius,
            ccw=ccw,
        ))
        i = j

    return merged


def _recover_circle_holes(region_holes, circle_cutouts, drill_holes, tolerance=0.01):
    """
    Identify which region holes are actually circles (from cutouts or drill holes).

    For each hole polygon, check if all vertices are equidistant from a known circle
    center. If so, return it as a circle descriptor instead of a polygon.

    Args:
        region_holes: list of hole vertex lists from subdivision
        circle_cutouts: list of CircleCutout objects (center_x, center_y, radius)
        drill_holes: list of DrillHole objects (center_x, center_y, diameter)

    Returns:
        list of ('circle', (cx, cy), radius) or ('polygon', [TaggedEdge...])
    """
    # Build list of known circles: (cx, cy, radius)
    known_circles = []
    if circle_cutouts:
        for c in circle_cutouts:
            known_circles.append((c.center_x, c.center_y, c.radius))
    if drill_holes:
        for h in drill_holes:
            known_circles.append((h.center_x, h.center_y, h.diameter / 2))

    result = []
    for hole in region_holes:
        if len(hole) < 3:
            result.append(('polygon', [TaggedEdge(type="line", start=hole[i], end=hole[(i+1) % len(hole)]) for i in range(len(hole))]))
            continue

        matched = False
        for (cx, cy, r) in known_circles:
            all_match = True
            for v in hole:
                dx = v[0] - cx
                dy = v[1] - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if abs(dist - r) > tolerance:
                    all_match = False
                    break
            if all_match:
                result.append(('circle', (cx, cy), r))
                matched = True
                break

        if not matched:
            # Return as polygon with line edges
            n = len(hole)
            edges = [TaggedEdge(type="line", start=hole[i], end=hole[(i+1) % n]) for i in range(n)]
            result.append(('polygon', edges))

    return result


def is_step_export_available():
    """STEP export is always available (pure Python)."""
    return True


def _marker_to_fold_def(marker):
    """Convert a FoldMarker to a FoldDefinition."""
    return FoldDefinition.from_marker(marker)


def _recipe_with_fold_defs(recipe):
    """Convert a region fold_recipe to use FoldDefinitions. Delegates to bend_transform.recipe_from_region."""
    # Create a minimal object with fold_recipe attribute for the canonical function
    class _Holder:
        fold_recipe = recipe
    return recipe_from_region(_Holder())


def _transform_polygon_3d(polygon_2d, recipe):
    """Transform a 2D polygon to 3D using a fold recipe."""
    return [transform_point(p, recipe) for p in polygon_2d]


def _transform_tagged_edges_3d(tagged_edges_2d, recipe, face_normal):
    """Transform 2D tagged edges to 3D edge dicts for StepWriter.

    For arcs, also transforms center and computes axis/ref_dir.
    Applies coplanar snapping to eliminate floating-point drift.

    Args:
        tagged_edges_2d: list of TaggedEdge (2D)
        recipe: fold recipe for 3D transformation
        face_normal: 3D face normal (used as cylinder axis for arcs)

    Returns:
        list of edge dicts suitable for StepWriter._make_mixed_loop
    """
    n = _normalize(face_normal)

    # First pass: transform all points to 3D
    raw_points = []
    for edge in tagged_edges_2d:
        raw_points.append(transform_point(edge.start, recipe))
    # Snap line endpoints to plane (arc endpoints stay on the arc)
    snapped = _snap_to_plane(raw_points, n)

    result = []
    for i, edge in enumerate(tagged_edges_2d):
        start_3d = snapped[i]
        end_3d = snapped[(i + 1) % len(snapped)]

        if edge.type == 'arc' and edge.center is not None:
            center_3d = transform_point(edge.center, recipe)
            # Snap center to same plane
            count = len(snapped)
            cx = sum(v[0] for v in snapped) / count
            cy = sum(v[1] for v in snapped) / count
            cz = sum(v[2] for v in snapped) / count
            plane_center = (cx, cy, cz)
            diff = _sub(center_3d, plane_center)
            dist = _dot(diff, n)
            center_3d = _sub(center_3d, _scale(n, dist))

            ref_dir = _normalize(_sub(start_3d, center_3d))
            result.append({
                'type': 'arc',
                'start': start_3d,
                'end': end_3d,
                'center': center_3d,
                'axis': n,
                'ref_dir': ref_dir,
                'radius': edge.radius,
                'ccw': edge.ccw,
            })
        else:
            result.append({
                'type': 'line',
                'start': start_3d,
                'end': end_3d,
            })
    return result


def _sample_arc_2d(start, end, center, radius, ccw, n_samples=32):
    """Sample a 2D arc at n_samples evenly-spaced points.

    Returns list of (x, y) points from start to end, inclusive.
    """
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

    points = []
    for i in range(n_samples):
        t = i / (n_samples - 1) if n_samples > 1 else 0.0
        angle = a_start + t * sweep
        points.append((cx + radius * math.cos(angle),
                        cy + radius * math.sin(angle)))
    return points


def _transform_tagged_edges_3d_bend(tagged_edges_2d, recipe, n_arc_samples=32):
    """Transform 2D tagged edges to 3D edge dicts for bend zone regions.

    For arcs, samples the arc in 2D and transforms each sample point
    through the full fold recipe, producing a B-spline edge dict.
    For lines, transforms start and end points normally.

    Args:
        tagged_edges_2d: list of TaggedEdge (2D)
        recipe: fold recipe for 3D transformation
        n_arc_samples: number of sample points for arc curves

    Returns:
        list of edge dicts suitable for StepWriter._make_mixed_loop
        (types: "line" or "bspline")
    """
    result = []
    for edge in tagged_edges_2d:
        start_3d = transform_point(edge.start, recipe)
        end_3d = transform_point(edge.end, recipe)

        if edge.type == 'arc' and edge.center is not None:
            # Sample the arc at many 2D points, then transform each to 3D
            pts_2d = _sample_arc_2d(
                edge.start, edge.end, edge.center, edge.radius,
                edge.ccw, n_arc_samples
            )
            pts_3d = [transform_point(p, recipe) for p in pts_2d]
            # Ensure start/end match the vertex-snapped positions
            pts_3d[0] = start_3d
            pts_3d[-1] = end_3d
            result.append({
                'type': 'bspline',
                'start': start_3d,
                'end': end_3d,
                'sample_points': pts_3d,
            })
        else:
            result.append({
                'type': 'line',
                'start': start_3d,
                'end': end_3d,
            })
    return result


def _compute_region_normal(region, recipe):
    """Compute the surface normal for a region."""
    if region.representative_point:
        return compute_normal(region.representative_point, recipe)
    # Fallback: use centroid
    if region.outline:
        n = len(region.outline)
        cx = sum(p[0] for p in region.outline) / n
        cy = sum(p[1] for p in region.outline) / n
        return compute_normal((cx, cy), recipe)
    return (0.0, 0.0, 1.0)


def _is_bend_region(recipe):
    """Check if the last entry in the recipe is IN_ZONE (bend region)."""
    if not recipe:
        return False
    last_entry = recipe[-1]
    return last_entry[1] == "IN_ZONE"


def _get_bend_info(recipe):
    """Get the fold definition and entry info for a bend region's active fold."""
    if not recipe:
        return None, None, False
    last_entry = recipe[-1]
    if last_entry[1] != "IN_ZONE":
        return None, None, False
    marker = last_entry[0]
    entered_from_back = last_entry[2] if len(last_entry) > 2 else False
    return _marker_to_fold_def(marker), last_entry[1], entered_from_back


def _build_bend_region_solid(writer, region, recipe_with_defs, thickness,
                             original_segments=None):
    """
    Build a cylindrical solid for a bend zone region.

    For a bend region, the last recipe entry is IN_ZONE. We compute the
    cylinder geometry from the fold definition and region outline.

    When original_segments are provided and arcs are found, arc edges in the
    outline are exported as B-spline curves (sampled and transformed to 3D).
    """
    fold_def = None
    entered_from_back = False
    for entry in reversed(recipe_with_defs):
        if entry[1] == "IN_ZONE":
            fold_def = entry[0]
            entered_from_back = entry[2] if len(entry) > 2 else False
            break

    if fold_def is None:
        # Fallback to flat
        return _build_flat_region_solid(writer, region, recipe_with_defs, thickness)

    # The recipe up to (but not including) the last IN_ZONE entry gives us
    # the cumulative transformation before entering the bend zone.
    prior_recipe = recipe_with_defs[:-1]

    hw = fold_def.zone_width / 2
    R = fold_def.radius

    # Find the min/max perpendicular distances of region vertices from fold center
    perp_dists = []
    along_dists = []
    for p in region.outline:
        dx = p[0] - fold_def.center[0]
        dy = p[1] - fold_def.center[1]
        perp = dx * fold_def.perp[0] + dy * fold_def.perp[1]
        along = dx * fold_def.axis[0] + dy * fold_def.axis[1]
        perp_dists.append(perp)
        along_dists.append(along)

    perp_min = min(perp_dists)
    perp_max = max(perp_dists)
    along_min = min(along_dists)
    along_max = max(along_dists)

    # Arc fractions — the physical cylinder is the same regardless of entry
    # direction, so we always use the raw perpendicular distances.
    dist_into_zone_min = max(0, perp_min + hw)
    dist_into_zone_max = min(fold_def.zone_width, perp_max + hw)
    frac_min = dist_into_zone_min / fold_def.zone_width if fold_def.zone_width > 0 else 0
    frac_max = dist_into_zone_max / fold_def.zone_width if fold_def.zone_width > 0 else 0
    # Always use positive theta: the cyl_axis negation already handles
    # the sweep direction for positive angles. For negative angles the
    # physical sweep is opposite, but abs(angle) with the same negated axis
    # produces the correct geometry (tangent goes in the right direction).
    theta_min = frac_min * abs(fold_def.angle)
    theta_max = frac_max * abs(fold_def.angle)

    # Track cumulative rotation through prior recipe (for 3D directions).
    # We do NOT track the origin here — instead we derive fold_center_3d
    # from transform_point on a reference point that avoids edge cases.
    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for entry in prior_recipe:
        f = entry[0]
        c = entry[1]
        back = entry[2] if len(entry) > 2 else False
        if c == "AFTER":
            f_axis_3d = _apply_rotation(rot, (f.axis[0], f.axis[1], 0.0))
            rotation_angle = -f.angle if back else f.angle
            fold_rot = _rotation_matrix_around_axis(f_axis_3d, rotation_angle)
            rot = _multiply_matrices(fold_rot, rot)

    fold_axis_3d = _apply_rotation(rot, (fold_def.axis[0], fold_def.axis[1], 0.0))
    fold_perp_3d = _apply_rotation(rot, (fold_def.perp[0], fold_def.perp[1], 0.0))
    up_3d = _apply_rotation(rot, (0.0, 0.0, 1.0))

    # Derive fold_center_3d from a reference point on the +hw boundary.
    # Using the +hw boundary avoids a parallel-fold edge case: when two folds
    # share the same perpendicular line (e.g., y=8 on both bars of an H-shape),
    # the fold center (perp=0 from the other fold) gives negative excess in
    # transform_point's AFTER step, producing wrong results. The +hw boundary
    # has excess=0 (exactly at the zone edge), which is handled correctly.
    ref_2d = (fold_def.center[0] + hw * fold_def.perp[0],
              fold_def.center[1] + hw * fold_def.perp[1])
    ref_3d = transform_point(ref_2d, prior_recipe)
    fold_center_3d = _sub(ref_3d, _scale(fold_perp_3d, hw))

    sign_angle = 1.0 if fold_def.angle >= 0 else -1.0

    # For back entry, the BFS enters from the +hw side. The cylinder's
    # entry surface is at +hw and it sweeps toward -hw. This reverses the
    # cylinder axis direction (tangent = -perp instead of +perp).
    if entered_from_back:
        surface_entry_3d = _add(fold_center_3d, _scale(fold_perp_3d, hw))
    else:
        surface_entry_3d = _sub(fold_center_3d, _scale(fold_perp_3d, hw))

    # Rotation center is displaced from entry surface by R * up_3d
    cyl_origin_3d = _add(surface_entry_3d, _scale(up_3d, sign_angle * R))

    # cyl_ref points FROM rotation center TO entry surface (theta=0)
    cyl_ref = _normalize(_scale(up_3d, -sign_angle))

    # Cylinder axis: for back entry, negate to reverse the sweep direction
    axis_sign = -sign_angle if entered_from_back else sign_angle
    cyl_axis = _normalize(_scale(fold_axis_3d, axis_sign))

    # For back entry, theta range is reversed: perp=-hw maps to theta=|angle|
    # in the original parametrization, but to theta_step=0 on the back-entry
    # cylinder. So theta_step = |angle| - theta_original.
    if entered_from_back:
        theta_min = abs(fold_def.angle) - frac_max * abs(fold_def.angle)
        theta_max = abs(fold_def.angle) - frac_min * abs(fold_def.angle)
    else:
        theta_min = frac_min * abs(fold_def.angle)
        theta_max = frac_max * abs(fold_def.angle)

    # Inner/outer radius depends on angle sign (which side the center is on)
    if sign_angle >= 0:
        inner_radius = R
        outer_radius = R + thickness
    else:
        inner_radius = R - thickness
        outer_radius = R

    # Along values in fold_axis direction; project onto cyl_axis
    if axis_sign >= 0:
        along_min_cyl = along_min
        along_max_cyl = along_max
    else:
        along_min_cyl = -along_max
        along_max_cyl = -along_min

    # _cyl_point: center + along*cyl_axis + r*(cos(theta)*cyl_ref + sin(theta)*tangent)
    # where tangent = cross(cyl_axis, cyl_ref) now points in the fold_perp direction
    def _cyl_point(along, theta, radius):
        tangent = _normalize(_cross(cyl_axis, cyl_ref))
        return _add(
            _add(cyl_origin_3d, _scale(cyl_axis, along)),
            _add(_scale(cyl_ref, radius * math.cos(theta)),
                 _scale(tangent, radius * math.sin(theta)))
        )

    # Check for arcs in the bend region outline
    tagged_edges = None
    has_arcs = False
    if original_segments:
        tagged_edges = _recover_arcs(region.outline, original_segments)
        has_arcs = any(e.type == 'arc' for e in tagged_edges)

    if has_arcs and tagged_edges is not None:
        # Mixed-edge path: arcs become B-splines through the fold recipe
        inner_edges_3d = _transform_tagged_edges_3d_bend(
            tagged_edges, recipe_with_defs, n_arc_samples=32
        )

        # Offset for outer (bottom) surface
        # For the outer surface, we transform at a z-offset = -thickness
        # In practice, we compute outer edges by offsetting inner edges
        # along the radial direction (away from cyl_origin)
        outer_edges_3d = []
        for edge in inner_edges_3d:
            oe = dict(edge)
            if edge['type'] == 'bspline':
                # Offset each sample point radially outward by thickness
                out_pts = []
                for pt in edge['sample_points']:
                    # Project onto cylinder axis to find the axial component
                    diff = _sub(pt, cyl_origin_3d)
                    along_ax = _dot(diff, cyl_axis)
                    axial_pt = _add(cyl_origin_3d, _scale(cyl_axis, along_ax))
                    radial = _sub(pt, axial_pt)
                    radial_len = math.sqrt(_dot(radial, radial))
                    if radial_len > 1e-12:
                        radial_dir = _normalize(radial)
                        out_pts.append(_add(pt, _scale(radial_dir, thickness)))
                    else:
                        out_pts.append(pt)
                oe['start'] = out_pts[0]
                oe['end'] = out_pts[-1]
                oe['sample_points'] = out_pts
            else:
                # Line edge: offset start and end radially
                for key in ('start', 'end'):
                    pt = edge[key]
                    diff = _sub(pt, cyl_origin_3d)
                    along_ax = _dot(diff, cyl_axis)
                    axial_pt = _add(cyl_origin_3d, _scale(cyl_axis, along_ax))
                    radial = _sub(pt, axial_pt)
                    radial_len = math.sqrt(_dot(radial, radial))
                    if radial_len > 1e-12:
                        radial_dir = _normalize(radial)
                        oe[key] = _add(pt, _scale(radial_dir, thickness))
                    else:
                        oe[key] = pt
            outer_edges_3d.append(oe)

        # Build inner face on cylindrical surface
        inner_face = writer.add_cylindrical_face_mixed(
            cyl_origin_3d, cyl_axis, cyl_ref, inner_radius,
            inner_edges_3d, face_same_sense=False
        )

        # Build outer face (reversed winding) on outer cylindrical surface
        outer_edges_reversed = []
        for e in reversed(outer_edges_3d):
            re_e = dict(e)
            re_e['start'] = e['end']
            re_e['end'] = e['start']
            if e['type'] == 'bspline':
                re_e['sample_points'] = list(reversed(e['sample_points']))
            outer_edges_reversed.append(re_e)

        outer_face = writer.add_cylindrical_face_mixed(
            cyl_origin_3d, cyl_axis, cyl_ref, outer_radius,
            outer_edges_reversed, face_same_sense=True
        )

        # Build side faces connecting inner and outer boundaries
        side_face_ids = writer._build_side_faces_mixed(
            inner_edges_3d, outer_edges_3d,
            _normalize(up_3d)
        )

        all_faces = [inner_face, outer_face] + side_face_ids
        shell_id = writer.closed_shell(all_faces)
        return writer.manifold_solid_brep(shell_id)

    # Default rectangular patch path (no arcs)
    # Inner surface (PCB top, at radius R from rotation center)
    inner_corners = [
        _cyl_point(along_min_cyl, theta_min, inner_radius),
        _cyl_point(along_min_cyl, theta_max, inner_radius),
        _cyl_point(along_max_cyl, theta_max, inner_radius),
        _cyl_point(along_max_cyl, theta_min, inner_radius),
    ]

    # Outer surface (PCB bottom, at radius R + thickness)
    outer_corners = [
        _cyl_point(along_min_cyl, theta_min, outer_radius),
        _cyl_point(along_min_cyl, theta_max, outer_radius),
        _cyl_point(along_max_cyl, theta_max, outer_radius),
        _cyl_point(along_max_cyl, theta_min, outer_radius),
    ]

    return writer.build_bend_solid(
        inner_corners, outer_corners,
        cyl_origin_3d, cyl_axis, cyl_ref,
        inner_radius, outer_radius,
        end_cap_pairs=None
    )


def _snap_to_plane(vertices_3d, normal):
    """Snap 3D vertices to a best-fit plane to eliminate floating-point drift.

    Uses the centroid and normal to define the plane, then projects all points onto it.
    """
    if not vertices_3d or len(vertices_3d) < 3:
        return vertices_3d

    n = _normalize(normal)
    # Compute centroid
    count = len(vertices_3d)
    cx = sum(v[0] for v in vertices_3d) / count
    cy = sum(v[1] for v in vertices_3d) / count
    cz = sum(v[2] for v in vertices_3d) / count
    center = (cx, cy, cz)

    # Project each vertex onto the plane defined by (center, n)
    result = []
    for v in vertices_3d:
        diff = _sub(v, center)
        dist = _dot(diff, n)
        projected = _sub(v, _scale(n, dist))
        result.append(projected)
    return result


def _build_flat_region_solid(writer, region, recipe_with_defs, thickness,
                             original_segments=None, circle_cutouts=None,
                             drill_holes=None):
    """Build a flat extruded solid for a region.

    When original_segments are provided, recovers arc info for exact geometry.
    Returns MANIFOLD_SOLID_BREP entity ID.
    """
    normal = _compute_region_normal(region, recipe_with_defs)

    # Try mixed (exact arc) path if original segment info is available
    if original_segments:
        tagged_edges = _recover_arcs(region.outline, original_segments)
        has_arcs = any(e.type == 'arc' for e in tagged_edges)

        if has_arcs:
            top_edges_3d = _transform_tagged_edges_3d(tagged_edges, recipe_with_defs, normal)

            # Holes stay as tessellated polygons (no circle hole recovery for now)
            hole_data = None
            if region.holes:
                hole_data = []
                for hole in region.holes:
                    n_h = len(hole)
                    hole_edges_2d = [TaggedEdge(type="line", start=hole[k],
                                                end=hole[(k+1) % n_h])
                                     for k in range(n_h)]
                    hole_edges_3d = _transform_tagged_edges_3d(
                        hole_edges_2d, recipe_with_defs, normal)
                    hole_data.append(('polygon', hole_edges_3d))

            return writer.build_flat_solid_mixed(
                top_edges_3d, normal, thickness,
                hole_data=hole_data
            )

    # Fallback: all-line path (original behavior)
    outline_3d = _snap_to_plane(
        _transform_polygon_3d(region.outline, recipe_with_defs), normal
    )
    holes_3d = [_snap_to_plane(_transform_polygon_3d(h, recipe_with_defs), normal)
                for h in region.holes]
    return writer.build_flat_solid(outline_3d, holes_3d if holes_3d else None, normal, thickness)



def board_to_step_native(board_geometry, markers, filename, config=None,
                         pcb=None, stiffeners=None, pcb_dir=None,
                         include_models=False, include_wrl_models=False):
    """
    Export board geometry to STEP file using pure Python.

    Args:
        board_geometry: BoardGeometry from geometry.py
        markers: list of FoldMarker objects
        filename: output .step file path
        config: FlexConfig (optional, for stiffener settings)
        pcb: KiCadPCB (optional, for stiffener extraction)
        stiffeners: pre-extracted stiffener list (optional)
        pcb_dir: directory of PCB file for resolving model paths
        include_models: if True, export 3D component models
        include_wrl_models: if True, also export WRL-only models (no STEP equivalent)

    Returns:
        True on success, False on error
    """
    try:
        writer = StepWriter()
        thickness = board_geometry.thickness

        # Capture original arc segments for arc recovery
        original_segments = board_geometry.outline.segments if board_geometry.outline.segments else []

        # Capture circle cutout and drill hole info for exact circle recovery
        circle_cutouts = None
        drill_holes = None
        if pcb is not None:
            try:
                circle_cutouts = pcb.get_circle_cutouts()
                drill_holes = pcb.get_drill_holes()
            except Exception:
                pass  # Non-critical: falls back to polygon path

        # Get outline as list of (x,y) tuples
        outline_2d = list(board_geometry.outline.vertices)
        if len(outline_2d) < 3:
            return False
        holes_2d = [list(c.vertices) for c in board_geometry.cutouts]

        # Split board into regions (1 subdivision for STEP — smooth cylinders)
        regions = split_board_into_regions(
            outline_2d, holes_2d, markers,
            num_bend_subdivisions=1
        )

        # Build each region as its own independent solid.
        # Reset the edge cache between regions to prevent cross-region
        # edge sharing which causes non-manifold topology (edges used
        # by more than 2 faces) when faces from different regions
        # reference the same edge entity.
        for region in regions:
            writer._edge_cache = {}
            recipe_defs = _recipe_with_fold_defs(region.fold_recipe)

            if _is_bend_region(region.fold_recipe):
                brep_id = _build_bend_region_solid(
                    writer, region, recipe_defs, thickness,
                    original_segments=original_segments,
                )
            else:
                brep_id = _build_flat_region_solid(
                    writer, region, recipe_defs, thickness,
                    original_segments=original_segments,
                    circle_cutouts=circle_cutouts,
                    drill_holes=drill_holes,
                )
            if brep_id is not None:
                writer.add_body(brep_id, "FLEX_PCB")

        # Handle stiffeners
        if stiffeners is None and config is not None and pcb is not None:
            if config.has_stiffener:
                stiffeners = extract_stiffeners(pcb, config)

        if stiffeners:
            for stiff_idx, stiff in enumerate(stiffeners):
                # Find which region contains the stiffener
                containing_region = find_containing_region(stiff.centroid, regions)
                if containing_region is None:
                    continue

                recipe_defs = _recipe_with_fold_defs(containing_region.fold_recipe)
                stiff_normal = _compute_region_normal(containing_region, recipe_defs)

                # Stiffener is on top or bottom of PCB
                if stiff.side == "bottom":
                    # Offset below PCB (in -normal direction by pcb thickness + stiffener thickness)
                    n = _normalize(stiff_normal)
                    offset_vec = _scale(n, -(thickness + stiff.thickness))
                    outline_3d = [_add(transform_point(p, recipe_defs), offset_vec)
                                  for p in stiff.outline]
                    cutouts_3d = [[_add(transform_point(p, recipe_defs), offset_vec)
                                   for p in cut] for cut in stiff.cutouts]
                    stiff_normal_out = _scale(n, -1.0)
                else:
                    # On top - offset above PCB
                    outline_3d = [transform_point(p, recipe_defs) for p in stiff.outline]
                    cutouts_3d = [[transform_point(p, recipe_defs) for p in cut]
                                  for cut in stiff.cutouts]
                    stiff_normal_out = stiff_normal

                brep_id = writer.build_flat_solid(
                    outline_3d,
                    cutouts_3d if cutouts_3d else None,
                    stiff_normal_out,
                    stiff.thickness
                )
                writer.add_body(brep_id, f"STIFFENER_{stiff_idx}")

        # Handle 3D component models
        if include_models and board_geometry.components:
            _export_component_models(
                writer, board_geometry, regions, thickness,
                pcb_dir=pcb_dir, pcb=pcb,
                include_wrl=include_wrl_models,
            )

        writer.write(filename)
        return True

    except Exception as e:
        print(f"STEP export error: {e}")
        import traceback
        traceback.print_exc()
        return False


def _resolve_model_path_for_step(model_path, pcb_dir=None, pcb=None,
                                  include_wrl=False):
    """Resolve a component model path, preferring STEP over WRL.

    For WRL models, checks if a STEP version exists in the same folder.
    Only returns WRL paths if include_wrl is True and no STEP alternative exists.

    Returns:
        Resolved absolute path, or None if not found/not allowed.
    """
    try:
        from .model_loader import expand_kicad_vars
    except ImportError:
        from model_loader import expand_kicad_vars

    resolved = expand_kicad_vars(model_path, pcb_dir, pcb)
    if resolved is None:
        return None

    ext = os.path.splitext(resolved)[1].lower()

    if ext in ('.step', '.stp'):
        return resolved

    if ext in ('.wrl', '.vrml'):
        # Check if a STEP version exists in the same folder
        base = os.path.splitext(resolved)[0]
        for step_ext in ('.step', '.stp'):
            step_path = base + step_ext
            if os.path.exists(step_path):
                return step_path

        # No STEP alternative — only return WRL if explicitly allowed
        if include_wrl:
            return resolved
        return None

    # Other formats — skip for STEP export
    return None


def _compute_component_placement(comp, model_info, thickness, regions):
    """Compute the STEP placement (origin, z_axis, x_axis) for a component.

    Uses the same transform chain as the viewer (model_loader.py):
    scale → model rotation → model offset → component rotation →
    back mirror → component position → fold recipe.

    Derives the placement by transforming 4 reference points (origin +
    unit axis tips) through the full chain, then extracting the
    AXIS2_PLACEMENT_3D from the result.

    Returns:
        (origin, z_axis, x_axis) as 3-tuples, or None if placement fails.
    """
    is_back = comp.layer == "B.Cu"

    # Same rotation lambdas as apply_model_transform
    def rot_x(a):
        c, s = math.cos(a), math.sin(a)
        return lambda x, y, z: (x, y*c - z*s, y*s + z*c)
    def rot_y(a):
        c, s = math.cos(a), math.sin(a)
        return lambda x, y, z: (x*c + z*s, y, -x*s + z*c)
    def rot_z(a):
        c, s = math.cos(a), math.sin(a)
        return lambda x, y, z: (x*c - y*s, x*s + y*c, z)

    rx = rot_x(math.radians(model_info.rotate[0]))
    ry = rot_y(math.radians(model_info.rotate[1]))
    rz = rot_z(math.radians(model_info.rotate[2]))
    comp_rz = rot_z(math.radians(-comp.angle))

    def _flat_transform(x, y, z):
        """Apply model → component transform (same as viewer, no fold)."""
        # 1. Scale
        x *= model_info.scale[0]
        y *= model_info.scale[1]
        z *= model_info.scale[2]
        # 2. Model rotation (ZYX)
        x, y, z = rz(x, y, z)
        x, y, z = ry(x, y, z)
        x, y, z = rx(x, y, z)
        # 3. Model offset
        # KiCad stores model offset in the model's Y-up frame (OpenGL/STEP/VRML
        # convention used by KiCad's 3D viewer); this transform chain operates
        # in the PCB's Y-down frame, so the Y component must be negated.
        x += model_info.offset[0]
        y -= model_info.offset[1]
        z += model_info.offset[2]
        # 4. Component rotation
        x, y, z = comp_rz(x, y, z)
        # 5. Back layer
        if is_back:
            z = -z - thickness
        # 6. Component position
        x += comp.center[0]
        y += comp.center[1]
        return (x, y, z)

    # Transform 4 reference points to get the full placement
    origin_flat = _flat_transform(0, 0, 0)
    x_tip_flat = _flat_transform(1, 0, 0)
    y_tip_flat = _flat_transform(0, 1, 0)
    z_tip_flat = _flat_transform(0, 0, 1)

    # Apply fold recipe
    containing_region = find_containing_region(comp.center, regions)
    if containing_region is not None and containing_region.fold_recipe:
        recipe_defs = _recipe_with_fold_defs(containing_region.fold_recipe)

        origin_3d = transform_point((origin_flat[0], origin_flat[1]), recipe_defs)
        normal = _compute_region_normal(containing_region, recipe_defs)
        n = _normalize(normal)
        origin_3d = _add(origin_3d, _scale(n, origin_flat[2]))

        # Build cumulative fold rotation for axis directions
        cum_rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for entry in recipe_defs:
            f = entry[0]
            c = entry[1]
            back = entry[2] if len(entry) > 2 else False
            if c == "AFTER":
                f_axis_3d = _apply_rotation(cum_rot, (f.axis[0], f.axis[1], 0.0))
                rotation_angle = -f.angle if back else f.angle
                fold_rot = _rotation_matrix_around_axis(f_axis_3d, rotation_angle)
                cum_rot = _multiply_matrices(fold_rot, cum_rot)

        # Transform axis directions through fold rotation
        x_dir_flat = _sub(x_tip_flat, origin_flat)
        z_dir_flat = _sub(z_tip_flat, origin_flat)
        x_axis = _normalize(_apply_rotation(cum_rot, x_dir_flat))
        z_axis = _normalize(_apply_rotation(cum_rot, z_dir_flat))
        origin = origin_3d
    else:
        origin = origin_flat
        x_axis = _normalize(_sub(x_tip_flat, origin_flat))
        z_axis = _normalize(_sub(z_tip_flat, origin_flat))

    return origin, z_axis, x_axis


def _export_component_models(writer, board_geometry, regions, thickness,
                              pcb_dir=None, pcb=None, include_wrl=False):
    """Export 3D component models in the STEP file.

    For STEP source models: embeds the original B-Rep entities directly
    (preserving exact geometry) and positions via REPRESENTATION_RELATIONSHIP.
    For WRL source models: tessellates to triangle mesh as fallback.

    Identical STEP models are embedded once and reused across all instances.
    """
    try:
        from .model_loader import load_model, apply_model_transform
    except ImportError:
        from model_loader import load_model, apply_model_transform

    # Cache: resolved_path -> list of brep_ids (for STEP) or LoadedModel (for WRL)
    step_embed_cache = {}   # path -> [brep_id, ...]
    wrl_mesh_cache = {}     # path -> LoadedModel or None

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
            is_step = ext in ('.step', '.stp')

            if is_step:
                # Embed STEP B-Rep directly (exact geometry, much smaller)
                if resolved not in step_embed_cache:
                    brep_ids = writer.embed_step_file(resolved)
                    step_embed_cache[resolved] = brep_ids

                brep_ids = step_embed_cache[resolved]
                if brep_ids:
                    # Compute placement transform for this component
                    placement = _compute_component_placement(
                        comp, model_info, thickness, regions)
                    if placement:
                        origin, z_axis, x_axis = placement
                        writer.add_sub_assembly(brep_ids, comp.reference,
                                                origin, z_axis, x_axis)
            else:
                # WRL: tessellate to triangles (fallback)
                if resolved not in wrl_mesh_cache:
                    wrl_mesh_cache[resolved] = load_model(resolved)
                loaded = wrl_mesh_cache[resolved]
                if loaded is None:
                    continue

                is_back = comp.layer == "B.Cu"
                transformed_mesh = apply_model_transform(
                    loaded.mesh, comp.center, comp.angle,
                    model_info.offset, model_info.scale,
                    model_info.rotate, thickness, is_back,
                )

                if not transformed_mesh.vertices or not transformed_mesh.faces:
                    continue

                # Apply fold recipe if component is in a folded region
                containing_region = find_containing_region(comp.center, regions)
                if containing_region is not None and containing_region.fold_recipe:
                    recipe_defs = _recipe_with_fold_defs(containing_region.fold_recipe)
                    normal = _compute_region_normal(containing_region, recipe_defs)
                    n = _normalize(normal)
                    verts_3d = []
                    for v in transformed_mesh.vertices:
                        base_3d = transform_point((v[0], v[1]), recipe_defs)
                        verts_3d.append(_add(base_3d, _scale(n, v[2])))
                else:
                    verts_3d = [(v[0], v[1], v[2])
                                for v in transformed_mesh.vertices]

                tri_faces = []
                for face in transformed_mesh.faces:
                    if len(face) == 3:
                        tri_faces.append(face)
                    elif len(face) == 4:
                        tri_faces.append((face[0], face[1], face[2]))
                        tri_faces.append((face[0], face[2], face[3]))

                brep_id = writer.build_tessellated_solid(verts_3d, tri_faces)
                if brep_id is not None:
                    writer.add_body(brep_id, comp.reference)

            break  # Only use first valid model per component

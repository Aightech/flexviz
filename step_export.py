"""
STEP file export for flex PCB visualization.

Uses build123d/OCC to create solid geometry from the mesh and export to STEP format.
Supports true cylindrical surfaces for bend zones to reduce file size.
"""

import os
import math
from typing import Optional, List, Tuple

# Check for build123d availability
_build123d_available = False
_occ_cylindrical_available = False
try:
    from build123d import (
        Solid, Shell, Face, Wire, Edge, Vertex,
        Vector, Plane, Location,
        export_step, import_step,
        Compound, Part
    )
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeShell
    )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    from OCP.BRep import BRep_Builder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_Face, TopoDS_Shape
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_Circ
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    _build123d_available = True
    _occ_cylindrical_available = True
except ImportError:
    try:
        # Fallback - basic build123d without cylindrical support
        from build123d import (
            Solid, Shell, Face, Wire, Edge, Vertex,
            Vector, Plane, Location,
            export_step, import_step,
            Compound, Part
        )
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakePolygon,
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_Sewing
        )
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid, TopoDS_Compound
        from OCP.gp import gp_Pnt
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        _build123d_available = True
    except ImportError:
        pass


def is_step_export_available() -> bool:
    """Check if STEP export is available."""
    return _build123d_available


def mesh_to_step(mesh, filename: str, tolerance: float = 0.01, max_faces: int = 5000) -> bool:
    """
    Export a mesh to STEP format.

    Creates faces from mesh and exports to STEP. For meshes with many faces,
    sewing is skipped to avoid extremely long processing times.

    Args:
        mesh: Mesh object with vertices and faces
        filename: Output STEP file path
        tolerance: Sewing tolerance for joining faces (only used for small meshes)
        max_faces: Maximum number of faces to export (for performance)

    Returns:
        True if export successful, False otherwise
    """
    if not _build123d_available:
        print("STEP export not available: build123d/OCC not installed")
        return False

    if not mesh or not mesh.vertices or not mesh.faces:
        print("Cannot export empty mesh")
        return False

    total_faces = len(mesh.faces)
    if total_faces > max_faces:
        print(f"Warning: Mesh has {total_faces} faces, limiting to {max_faces} for performance")
        print("Consider reducing bend subdivisions or using coarser mesh for STEP export")

    # Sewing is O(n²) and freezes with many faces - only use for small meshes
    use_sewing = total_faces <= 200

    try:
        faces_to_process = mesh.faces[:max_faces] if total_faces > max_faces else mesh.faces
        num_to_process = len(faces_to_process)

        if use_sewing:
            # Create a sewing object to join all faces (only for small meshes)
            sewing = BRepBuilderAPI_Sewing(tolerance)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)

        faces_added = 0

        for idx, face_indices in enumerate(faces_to_process):
            # Progress indicator for large meshes
            if num_to_process > 100 and idx % 500 == 0:
                print(f"Processing faces: {idx}/{num_to_process} ({100*idx//num_to_process}%)")
            if len(face_indices) < 3:
                continue

            # Get vertices for this face
            try:
                points = [mesh.vertices[i] for i in face_indices]
            except IndexError:
                continue

            # Create polygon wire
            polygon = BRepBuilderAPI_MakePolygon()
            for p in points:
                polygon.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
            polygon.Close()

            if not polygon.IsDone():
                continue

            wire = polygon.Wire()

            # Create face from wire
            face_maker = BRepBuilderAPI_MakeFace(wire, True)  # True = only plane
            if face_maker.IsDone():
                face = face_maker.Face()
                if use_sewing:
                    sewing.Add(face)
                else:
                    builder.Add(compound, face)
                faces_added += 1

        if faces_added == 0:
            print("No valid faces could be created")
            return False

        if use_sewing:
            # Perform sewing for small meshes
            print("Sewing faces...")
            sewing.Perform()
            sewn_shape = sewing.SewedShape()

            # Export using build123d
            if hasattr(sewn_shape, 'wrapped'):
                shape_to_export = sewn_shape
            else:
                # Wrap in Compound for export
                compound = TopoDS_Compound()
                builder.MakeCompound(compound)
                builder.Add(compound, sewn_shape)
                shape_to_export = Compound(compound)
        else:
            # Skip sewing for large meshes - export as compound of faces
            shape_to_export = Compound(compound)

        # Export to STEP
        export_step(shape_to_export, filename)

        print(f"Exported {faces_added} faces to {filename}")
        return True

    except Exception as e:
        print(f"STEP export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def board_geometry_to_step(
    board_geometry,
    markers: list = None,
    filename: str = "output.step",
    include_traces: bool = False,
    include_pads: bool = False,
    num_bend_subdivisions: int = 8
) -> bool:
    """
    Export board geometry to STEP format with bend transformations.

    This creates proper solid geometry by extruding the board outline
    and applying bend transformations.

    Args:
        board_geometry: BoardGeometry object
        markers: List of FoldMarker objects
        filename: Output STEP file path
        include_traces: Include copper traces (adds complexity)
        include_pads: Include pads (adds complexity)
        num_bend_subdivisions: Subdivisions for bend zones

    Returns:
        True if export successful, False otherwise
    """
    if not _build123d_available:
        print("STEP export not available: build123d/OCC not installed")
        return False

    try:
        # Import mesh generation functions
        try:
            from .mesh import create_board_geometry_mesh
        except ImportError:
            from mesh import create_board_geometry_mesh

        # Generate mesh with higher subdivision for smooth STEP output
        mesh = create_board_geometry_mesh(
            board_geometry,
            markers=markers,
            include_traces=include_traces,
            include_pads=include_pads,
            include_components=False,
            subdivide_length=0.5,  # Finer subdivision for STEP
            num_bend_subdivisions=num_bend_subdivisions,
            apply_bend=True
        )

        return mesh_to_step(mesh, filename)

    except Exception as e:
        print(f"Board geometry STEP export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def unify_same_domain(shape: TopoDS_Shape, tolerance: float = 1e-6) -> TopoDS_Shape:
    """
    Merge adjacent coplanar faces and tangent edges into single faces/edges.

    This significantly reduces STEP file size by combining triangulated planar
    regions into single faces with proper geometric representation.

    Args:
        shape: Input OCC shape (compound, shell, solid)
        tolerance: Tolerance for considering faces coplanar

    Returns:
        Unified shape with merged faces
    """
    if not _occ_cylindrical_available:
        return shape

    try:
        unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        unifier.SetAngularTolerance(0.01)  # ~0.5 degrees
        unifier.SetLinearTolerance(tolerance)
        unifier.Build()
        return unifier.Shape()
    except Exception as e:
        print(f"Face unification failed: {e}")
        return shape


def board_to_step_direct(
    board_geometry,
    markers: list = None,
    filename: str = "output.step",
    num_bend_subdivisions: int = 1,
    stiffeners: list = None
) -> bool:
    """
    Export board geometry to STEP using direct CAD construction.

    Instead of triangulating and converting, this creates native CAD geometry:
    - Flat regions: extruded 2D profiles (planar faces)
    - Bend zones: swept profiles along circular arcs (cylindrical faces)
    - Stiffeners: extruded profiles offset from PCB surface

    This produces much smaller STEP files with proper parametric surfaces.

    Args:
        board_geometry: BoardGeometry object
        markers: List of FoldMarker objects
        filename: Output STEP file path
        num_bend_subdivisions: Subdivisions in bend zones (1 = single arc)
        stiffeners: List of StiffenerRegion objects

    Returns:
        True if export successful, False otherwise
    """
    if not _build123d_available:
        print("STEP export not available: build123d/OCC not installed")
        return False

    try:
        from build123d import (
            BuildPart, BuildSketch, BuildLine,
            Polygon as B3DPolygon, Rectangle, Circle,
            extrude, sweep, make_face, add,
            Axis, Mode, Align, Rotation,
            Polyline, Line, ThreePointArc, Spline
        )

        # Import our modules
        try:
            from .planar_subdivision import split_board_into_regions, Region
            from .bend_transform import FoldDefinition, transform_point, compute_normal
            from .bend_transform import _rotation_matrix_around_axis, _apply_rotation
        except ImportError:
            from planar_subdivision import split_board_into_regions, Region
            from bend_transform import FoldDefinition, transform_point, compute_normal
            from bend_transform import _rotation_matrix_around_axis, _apply_rotation

        outline = board_geometry.outline
        thickness = board_geometry.thickness
        cutouts = board_geometry.cutouts or []

        if not outline.vertices or len(outline.vertices) < 3:
            print("Invalid board outline")
            return False

        # Convert to 2D coordinate lists
        outline_verts = [(v[0], v[1]) for v in outline.vertices]
        cutout_verts = [[(v[0], v[1]) for v in c.vertices] for c in cutouts]

        # Split board into regions
        if markers:
            regions = split_board_into_regions(
                outline_verts, cutout_verts, markers,
                num_bend_subdivisions=num_bend_subdivisions
            )
        else:
            # No bends - single flat region
            single_region = Region(
                index=0,
                outline=outline_verts,
                holes=cutout_verts,
                fold_recipe=[]
            )
            regions = [single_region]

        print(f"Processing {len(regions)} regions...")

        all_parts = []

        for region in regions:
            # Build fold recipe with FoldDefinitions
            recipe = []
            if hasattr(region, 'fold_recipe') and region.fold_recipe:
                for entry in region.fold_recipe:
                    fm = entry[0]
                    classification = entry[1]
                    entered_from_back = entry[2] if len(entry) > 2 else False
                    recipe.append((FoldDefinition.from_marker(fm), classification, entered_from_back))

            # Check if this is a bend zone
            is_bend_zone = any(entry[1] == "IN_ZONE" for entry in recipe)

            if is_bend_zone:
                # Create bend zone geometry using sweep
                part = _create_bend_zone_solid(region, recipe, thickness)
                if part:
                    all_parts.append(part)
            else:
                # Create flat region geometry using extrusion
                part = _create_flat_region_solid(region, recipe, thickness)
                if part:
                    all_parts.append(part)

        # Process stiffeners
        if stiffeners:
            print(f"Processing {len(stiffeners)} stiffeners...")

            # Import region finder
            try:
                from .planar_subdivision import find_containing_region
            except ImportError:
                from planar_subdivision import find_containing_region

            for stiffener in stiffeners:
                # Find which region the stiffener is in (for fold transformation)
                stiff_centroid = stiffener.centroid
                containing_region = find_containing_region(stiff_centroid, regions)

                # Build recipe for this stiffener
                stiff_recipe = []
                if containing_region and hasattr(containing_region, 'fold_recipe') and containing_region.fold_recipe:
                    for entry in containing_region.fold_recipe:
                        fm = entry[0]
                        classification = entry[1]
                        entered_from_back = entry[2] if len(entry) > 2 else False
                        stiff_recipe.append((FoldDefinition.from_marker(fm), classification, entered_from_back))

                # Create stiffener solid
                part = _create_stiffener_solid(
                    stiffener.outline,
                    stiffener.cutouts,
                    stiffener.thickness,
                    thickness,  # PCB thickness
                    stiffener.side,
                    stiff_recipe
                )
                if part:
                    all_parts.append(part)

        if not all_parts:
            print("No geometry created")
            return False

        # Combine all parts
        print(f"Combining {len(all_parts)} parts...")

        if len(all_parts) == 1:
            result = all_parts[0]
        else:
            # Create compound of all parts
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for part in all_parts:
                if hasattr(part, 'wrapped'):
                    builder.Add(compound, part.wrapped)
                else:
                    builder.Add(compound, part)
            result = Compound(compound)

        # Export
        export_step(result, filename)
        file_size = os.path.getsize(filename)
        print(f"Exported to {filename} ({file_size:,} bytes)")

        return True

    except Exception as e:
        print(f"Direct STEP export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _create_flat_region_solid(region, recipe: list, thickness: float):
    """
    Create a solid for a flat region by extruding its 2D outline.

    For flat regions (no bend or after a bend), we create the geometry
    directly in 3D space using the transformed vertices.
    """
    try:
        # Import transform functions
        try:
            from .bend_transform import transform_point, compute_normal
        except ImportError:
            from bend_transform import transform_point, compute_normal

        # Get the 2D outline
        outline_2d = [(v[0], v[1]) for v in region.outline]
        holes_2d = [[(v[0], v[1]) for v in h] for h in region.holes] if region.holes else []

        if len(outline_2d) < 3:
            return None

        # Transform all vertices to 3D
        top_3d = []
        bottom_3d = []
        for v in outline_2d:
            p3d = transform_point(v, recipe)
            normal = compute_normal(v, recipe)
            top_3d.append(p3d)
            bottom_3d.append((
                p3d[0] - normal[0] * thickness,
                p3d[1] - normal[1] * thickness,
                p3d[2] - normal[2] * thickness
            ))

        # Transform holes
        holes_top_3d = []
        holes_bottom_3d = []
        for hole in holes_2d:
            hole_top = []
            hole_bottom = []
            for v in hole:
                p3d = transform_point(v, recipe)
                normal = compute_normal(v, recipe)
                hole_top.append(p3d)
                hole_bottom.append((
                    p3d[0] - normal[0] * thickness,
                    p3d[1] - normal[1] * thickness,
                    p3d[2] - normal[2] * thickness
                ))
            holes_top_3d.append(hole_top)
            holes_bottom_3d.append(hole_bottom)

        # Create faces using OCC directly
        builder = BRep_Builder()
        shell = TopoDS_Shell()
        builder.MakeShell(shell)

        # Top face
        top_face = _make_polygon_face(top_3d, holes_top_3d)
        if top_face:
            builder.Add(shell, top_face)

        # Bottom face (reversed winding)
        bottom_face = _make_polygon_face(list(reversed(bottom_3d)),
                                         [list(reversed(h)) for h in holes_bottom_3d])
        if bottom_face:
            builder.Add(shell, bottom_face)

        # Side faces (outer boundary)
        n = len(top_3d)
        for i in range(n):
            j = (i + 1) % n
            side_pts = [top_3d[i], top_3d[j], bottom_3d[j], bottom_3d[i]]
            side_face = _make_polygon_face(side_pts)
            if side_face:
                builder.Add(shell, side_face)

        # Side faces for holes (inner walls)
        for hole_top, hole_bottom in zip(holes_top_3d, holes_bottom_3d):
            nh = len(hole_top)
            for i in range(nh):
                j = (i + 1) % nh
                # Reversed winding for inner walls
                side_pts = [hole_top[j], hole_top[i], hole_bottom[i], hole_bottom[j]]
                side_face = _make_polygon_face(side_pts)
                if side_face:
                    builder.Add(shell, side_face)

        return shell

    except Exception as e:
        print(f"Flat region creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _make_polygon_face(vertices_3d: list, holes_3d: list = None) -> TopoDS_Face:
    """Create a planar face from 3D vertices with optional holes."""
    if len(vertices_3d) < 3:
        return None

    try:
        # Create outer wire
        polygon = BRepBuilderAPI_MakePolygon()
        for p in vertices_3d:
            polygon.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        polygon.Close()

        if not polygon.IsDone():
            return None

        wire = polygon.Wire()

        # Create face
        face_maker = BRepBuilderAPI_MakeFace(wire, True)  # True = planar
        if not face_maker.IsDone():
            return None

        face = face_maker.Face()

        # Add holes if any
        if holes_3d:
            for hole in holes_3d:
                if len(hole) >= 3:
                    hole_poly = BRepBuilderAPI_MakePolygon()
                    for p in hole:
                        hole_poly.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
                    hole_poly.Close()
                    if hole_poly.IsDone():
                        hole_wire = hole_poly.Wire()
                        face_maker = BRepBuilderAPI_MakeFace(face, hole_wire)
                        if face_maker.IsDone():
                            face = face_maker.Face()

        return face

    except Exception:
        return None


def _create_stiffener_solid(
    outline: list,
    cutouts: list,
    stiffener_thickness: float,
    pcb_thickness: float,
    side: str,
    recipe: list
):
    """
    Create a solid for a stiffener region.

    Stiffeners are flat extruded profiles that sit on top or bottom of the PCB.

    Args:
        outline: Stiffener outline vertices [(x,y), ...]
        cutouts: List of hole polygons
        stiffener_thickness: Thickness of the stiffener
        pcb_thickness: PCB thickness (for bottom offset)
        side: "top" or "bottom"
        recipe: Fold recipe for 3D transformation

    Returns:
        OCC shell representing the stiffener
    """
    try:
        # Import transform functions
        try:
            from .bend_transform import transform_point, compute_normal
        except ImportError:
            from bend_transform import transform_point, compute_normal

        outline_2d = [(v[0], v[1]) for v in outline]
        holes_2d = [[(v[0], v[1]) for v in h] for h in cutouts] if cutouts else []

        if len(outline_2d) < 3:
            return None

        # Transform all vertices to 3D
        # For stiffeners, we need to offset from the PCB surface
        top_3d = []
        bottom_3d = []

        for v in outline_2d:
            p3d = transform_point(v, recipe)
            normal = compute_normal(v, recipe)

            if side == "top":
                # Stiffener on top: from PCB top surface upward
                stiff_bottom = p3d
                stiff_top = (
                    p3d[0] + normal[0] * stiffener_thickness,
                    p3d[1] + normal[1] * stiffener_thickness,
                    p3d[2] + normal[2] * stiffener_thickness
                )
            else:  # bottom
                # Stiffener on bottom: from PCB bottom surface downward
                pcb_bottom = (
                    p3d[0] - normal[0] * pcb_thickness,
                    p3d[1] - normal[1] * pcb_thickness,
                    p3d[2] - normal[2] * pcb_thickness
                )
                stiff_top = pcb_bottom
                stiff_bottom = (
                    pcb_bottom[0] - normal[0] * stiffener_thickness,
                    pcb_bottom[1] - normal[1] * stiffener_thickness,
                    pcb_bottom[2] - normal[2] * stiffener_thickness
                )

            top_3d.append(stiff_top)
            bottom_3d.append(stiff_bottom)

        # Transform holes
        holes_top_3d = []
        holes_bottom_3d = []
        for hole in holes_2d:
            hole_top = []
            hole_bottom = []
            for v in hole:
                p3d = transform_point(v, recipe)
                normal = compute_normal(v, recipe)

                if side == "top":
                    ht = (
                        p3d[0] + normal[0] * stiffener_thickness,
                        p3d[1] + normal[1] * stiffener_thickness,
                        p3d[2] + normal[2] * stiffener_thickness
                    )
                    hb = p3d
                else:
                    pcb_bottom = (
                        p3d[0] - normal[0] * pcb_thickness,
                        p3d[1] - normal[1] * pcb_thickness,
                        p3d[2] - normal[2] * pcb_thickness
                    )
                    ht = pcb_bottom
                    hb = (
                        pcb_bottom[0] - normal[0] * stiffener_thickness,
                        pcb_bottom[1] - normal[1] * stiffener_thickness,
                        pcb_bottom[2] - normal[2] * stiffener_thickness
                    )

                hole_top.append(ht)
                hole_bottom.append(hb)
            holes_top_3d.append(hole_top)
            holes_bottom_3d.append(hole_bottom)

        # Create faces using OCC directly
        builder = BRep_Builder()
        shell = TopoDS_Shell()
        builder.MakeShell(shell)

        # Top face
        top_face = _make_polygon_face(top_3d, holes_top_3d)
        if top_face:
            builder.Add(shell, top_face)

        # Bottom face (reversed winding)
        bottom_face = _make_polygon_face(list(reversed(bottom_3d)),
                                         [list(reversed(h)) for h in holes_bottom_3d])
        if bottom_face:
            builder.Add(shell, bottom_face)

        # Side faces (outer boundary)
        n = len(top_3d)
        for i in range(n):
            j = (i + 1) % n
            side_pts = [top_3d[i], top_3d[j], bottom_3d[j], bottom_3d[i]]
            side_face = _make_polygon_face(side_pts)
            if side_face:
                builder.Add(shell, side_face)

        # Side faces for holes (inner walls)
        for hole_top, hole_bottom in zip(holes_top_3d, holes_bottom_3d):
            nh = len(hole_top)
            for i in range(nh):
                j = (i + 1) % nh
                # Reversed winding for inner walls
                side_pts = [hole_top[j], hole_top[i], hole_bottom[i], hole_bottom[j]]
                side_face = _make_polygon_face(side_pts)
                if side_face:
                    builder.Add(shell, side_face)

        return shell

    except Exception as e:
        print(f"Stiffener creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _create_bend_zone_solid(region, recipe: list, thickness: float):
    """
    Create a solid for a bend zone by sweeping a profile along an arc.

    The bend zone is a cylindrical shell section created by sweeping
    the board cross-section along a circular arc.
    """
    from build123d import (
        BuildPart, BuildSketch, BuildLine,
        Rectangle, sweep, ThreePointArc,
        Plane, Vector, Align
    )

    try:
        # Find the IN_ZONE fold
        in_zone_entry = next((e for e in recipe if e[1] == "IN_ZONE"), None)
        if not in_zone_entry:
            return None

        fold = in_zone_entry[0]
        entered_from_back = in_zone_entry[2]

        # Get previous AFTER folds for cumulative transformation
        recipe_before = [(e[0], e[1], e[2] if len(e) > 2 else False)
                         for e in recipe if e[1] == "AFTER"]

        # Import transform functions
        try:
            from .bend_transform import (
                transform_point, compute_normal,
                _rotation_matrix_around_axis, _multiply_matrices, _apply_rotation
            )
        except ImportError:
            from bend_transform import (
                transform_point, compute_normal,
                _rotation_matrix_around_axis, _multiply_matrices, _apply_rotation
            )

        # Compute bend parameters
        R = fold.radius  # Inner radius
        angle = fold.angle  # Bend angle in radians
        hw = fold.zone_width / 2

        # Compute width of this region along the fold axis
        min_along = float('inf')
        max_along = float('-inf')
        for v in region.outline:
            dx = v[0] - fold.center[0]
            dy = v[1] - fold.center[1]
            along = dx * fold.axis[0] + dy * fold.axis[1]
            min_along = min(min_along, along)
            max_along = max(max_along, along)

        width = max_along - min_along
        if width <= 0:
            return None

        # Center along the axis
        center_along = (min_along + max_along) / 2

        # Compute cumulative rotation matrix from previous folds
        rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for entry in recipe_before:
            prev_fold = entry[0]
            classification = entry[1]
            efb = entry[2] if len(entry) > 2 else False
            if classification == "AFTER":
                fold_axis_3d = _apply_rotation(rot, (prev_fold.axis[0], prev_fold.axis[1], 0.0))
                rotation_angle = -prev_fold.angle if efb else prev_fold.angle
                fold_rot = _rotation_matrix_around_axis(fold_axis_3d, rotation_angle)
                rot = _multiply_matrices(fold_rot, rot)

        # Get 3D directions
        axis_3d = _apply_rotation(rot, (fold.axis[0], fold.axis[1], 0.0))
        perp_3d = _apply_rotation(rot, (fold.perp[0], fold.perp[1], 0.0))
        up_3d = _apply_rotation(rot, (0.0, 0.0, 1.0))

        # Get the 3D transformation at the start of the bend (centered)
        # The bend starts at perp_dist = -hw from fold center
        start_2d = (
            fold.center[0] - hw * fold.perp[0] + center_along * fold.axis[0],
            fold.center[1] - hw * fold.perp[1] + center_along * fold.axis[1]
        )
        start_3d = transform_point(start_2d, recipe_before)

        # Arc path points (on the centerline of the board)
        cyl_center = Vector(start_3d[0], start_3d[1], start_3d[2])

        # Handle back entry
        effective_angle = angle
        if entered_from_back:
            effective_angle = -angle

        # Middle point of arc (at half angle)
        mid_angle = effective_angle / 2
        mid_offset_perp = R * math.sin(abs(mid_angle))
        mid_offset_up = R * (1 - math.cos(abs(mid_angle)))
        if effective_angle < 0:
            mid_offset_up = -mid_offset_up

        arc_mid = Vector(
            cyl_center.X + mid_offset_perp * perp_3d[0] + mid_offset_up * up_3d[0],
            cyl_center.Y + mid_offset_perp * perp_3d[1] + mid_offset_up * up_3d[1],
            cyl_center.Z + mid_offset_perp * perp_3d[2] + mid_offset_up * up_3d[2]
        )

        # End point of arc (at full angle)
        end_offset_perp = R * math.sin(abs(effective_angle))
        end_offset_up = R * (1 - math.cos(abs(effective_angle)))
        if effective_angle < 0:
            end_offset_up = -end_offset_up

        arc_end = Vector(
            cyl_center.X + end_offset_perp * perp_3d[0] + end_offset_up * up_3d[0],
            cyl_center.Y + end_offset_perp * perp_3d[1] + end_offset_up * up_3d[1],
            cyl_center.Z + end_offset_perp * perp_3d[2] + end_offset_up * up_3d[2]
        )

        # Create the sweep path (three-point arc)
        with BuildLine() as path:
            ThreePointArc(cyl_center, arc_mid, arc_end)

        # Create the profile (rectangle in the plane perpendicular to path start)
        # Profile plane: normal is perp_3d (sweep direction at start)
        # X direction: axis_3d (along fold)
        # Y direction: up_3d (board thickness direction)
        profile_plane = Plane(
            origin=cyl_center,
            x_dir=Vector(axis_3d[0], axis_3d[1], axis_3d[2]),
            z_dir=Vector(perp_3d[0], perp_3d[1], perp_3d[2])
        )

        # Profile centered on the path, extending width/2 in each direction along axis
        # and thickness in the -up direction (board thickness)
        with BuildSketch(profile_plane) as profile:
            Rectangle(width, thickness, align=(Align.CENTER, Align.MAX))

        # Sweep the profile along the arc
        with BuildPart() as part:
            sweep(profile.sketch, path=path.line)

        return part.part

    except Exception as e:
        print(f"Bend zone creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def board_geometry_to_step_optimized(
    board_geometry,
    markers: list = None,
    filename: str = "output.step",
    include_traces: bool = False,
    include_pads: bool = False,
    merge_faces: bool = True,
    num_bend_subdivisions: int = 4
) -> bool:
    """
    Export board geometry to STEP with optimized face merging.

    Creates mesh-based geometry but with coplanar face merging for flat regions,
    reducing STEP file size while maintaining accurate bent geometry.

    Note: Face merging is most effective with fewer subdivisions (1-4).
    Higher subdivisions create many small faces that can't be easily merged.

    Args:
        board_geometry: BoardGeometry object
        markers: List of FoldMarker objects
        filename: Output STEP file path
        include_traces: Include copper traces
        include_pads: Include pads
        merge_faces: Apply face unification to reduce file size
        num_bend_subdivisions: Subdivisions for bend zones (fewer = smaller file)

    Returns:
        True if export successful, False otherwise
    """
    if not _build123d_available:
        print("STEP export not available: build123d/OCC not installed")
        return False

    try:
        # Import mesh generation
        try:
            from .mesh import create_board_geometry_mesh
        except ImportError:
            from mesh import create_board_geometry_mesh

        # Generate mesh with specified subdivisions
        mesh = create_board_geometry_mesh(
            board_geometry,
            markers=markers,
            include_traces=include_traces,
            include_pads=include_pads,
            include_components=False,
            subdivide_length=2.0,  # Coarser for smaller files
            num_bend_subdivisions=num_bend_subdivisions,
            apply_bend=True
        )

        if not mesh.vertices or not mesh.faces:
            print("Empty mesh generated")
            return False

        print(f"Generated mesh: {len(mesh.faces)} faces")

        # Use unified export with face merging
        return mesh_to_step_unified(
            mesh, filename,
            max_faces=len(mesh.faces),  # Use all faces
            merge_faces=merge_faces
        )

    except Exception as e:
        print(f"Optimized STEP export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def mesh_to_step_unified(
    mesh,
    filename: str,
    tolerance: float = 0.01,
    max_faces: int = 10000,
    merge_faces: bool = True
) -> bool:
    """
    Export mesh to STEP with optional face merging.

    Like mesh_to_step but with post-processing to merge coplanar faces,
    which can significantly reduce file size for triangulated meshes.

    The process:
    1. Create individual faces from mesh triangles
    2. Sew faces together to create topological connections
    3. Apply ShapeUpgrade_UnifySameDomain to merge coplanar faces

    Args:
        mesh: Mesh object with vertices and faces
        filename: Output STEP file path
        tolerance: Sewing tolerance
        max_faces: Maximum faces to process
        merge_faces: Apply face unification

    Returns:
        True if successful
    """
    if not _build123d_available:
        print("STEP export not available: build123d/OCC not installed")
        return False

    if not mesh or not mesh.vertices or not mesh.faces:
        print("Cannot export empty mesh")
        return False

    total_faces = len(mesh.faces)
    if total_faces > max_faces:
        print(f"Warning: Mesh has {total_faces} faces, limiting to {max_faces}")

    # For very large meshes, sewing + unification can be slow
    # but it's still much faster than the O(n²) sewing we avoided before
    # because unification is O(n) after the sewing
    use_sewing = merge_faces and total_faces <= 2000

    try:
        faces_to_process = mesh.faces[:max_faces] if total_faces > max_faces else mesh.faces
        num_to_process = len(faces_to_process)

        if use_sewing:
            sewing = BRepBuilderAPI_Sewing(tolerance)
        else:
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)

        faces_added = 0

        for idx, face_indices in enumerate(faces_to_process):
            if num_to_process > 100 and idx % 1000 == 0:
                print(f"Processing faces: {idx}/{num_to_process} ({100*idx//num_to_process}%)")

            if len(face_indices) < 3:
                continue

            try:
                points = [mesh.vertices[i] for i in face_indices]
            except IndexError:
                continue

            polygon = BRepBuilderAPI_MakePolygon()
            for p in points:
                polygon.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
            polygon.Close()

            if not polygon.IsDone():
                continue

            wire = polygon.Wire()
            face_maker = BRepBuilderAPI_MakeFace(wire, True)
            if face_maker.IsDone():
                if use_sewing:
                    sewing.Add(face_maker.Face())
                else:
                    builder.Add(compound, face_maker.Face())
                faces_added += 1

        if faces_added == 0:
            print("No valid faces could be created")
            return False

        if use_sewing:
            print("Sewing faces...")
            sewing.Perform()
            shape_to_export = sewing.SewedShape()

            # Apply face unification after sewing
            if merge_faces and _occ_cylindrical_available:
                print("Merging adjacent coplanar faces...")
                shape_to_export = unify_same_domain(shape_to_export)
        else:
            shape_to_export = compound
            if merge_faces:
                print("Note: Skipping face merge for large mesh (>2000 faces)")

        # Wrap in Compound if needed
        if hasattr(shape_to_export, 'wrapped'):
            pass  # Already wrapped
        else:
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            builder.Add(compound, shape_to_export)
            shape_to_export = compound

        export_step(Compound(shape_to_export), filename)

        print(f"Exported {faces_added} faces to {filename}")
        if merge_faces and use_sewing:
            print("(Faces sewn and coplanar faces merged)")

        return True

    except Exception as e:
        print(f"STEP export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_flat_board_step(
    board_geometry,
    filename: str = "board_flat.step"
) -> bool:
    """
    Export the flat (unbent) board to STEP format.

    Creates a proper extruded solid from the 2D board outline.

    Args:
        board_geometry: BoardGeometry object
        filename: Output STEP file path

    Returns:
        True if export successful, False otherwise
    """
    if not _build123d_available:
        print("STEP export not available: build123d/OCC not installed")
        return False

    try:
        from build123d import (
            BuildPart, BuildSketch, Sketch,
            extrude, make_face, Polygon as B3DPolygon,
            add, Locations, Mode
        )

        outline = board_geometry.outline
        thickness = board_geometry.thickness
        cutouts = board_geometry.cutouts or []

        if not outline.vertices or len(outline.vertices) < 3:
            print("Invalid board outline")
            return False

        # Create the board as an extruded solid
        with BuildPart() as board:
            with BuildSketch() as sketch:
                # Create outer boundary
                outer_pts = [(v[0], v[1]) for v in outline.vertices]
                B3DPolygon(outer_pts, align=None)

                # Subtract cutouts
                for cutout in cutouts:
                    if len(cutout.vertices) >= 3:
                        cutout_pts = [(v[0], v[1]) for v in cutout.vertices]
                        with Locations((0, 0)):
                            B3DPolygon(cutout_pts, align=None, mode=Mode.SUBTRACT)

            # Extrude to board thickness
            extrude(amount=thickness)

        # Export
        export_step(board.part, filename)
        print(f"Exported flat board to {filename}")
        return True

    except Exception as e:
        print(f"Flat board STEP export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

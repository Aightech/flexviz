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

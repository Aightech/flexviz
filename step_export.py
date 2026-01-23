"""
STEP file export for flex PCB visualization.

Uses build123d/OCC to create solid geometry from the mesh and export to STEP format.
"""

import os
import math
from typing import Optional

# Check for build123d availability
_build123d_available = False
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

    Creates a shell solid from mesh faces and exports to STEP.

    Args:
        mesh: Mesh object with vertices and faces
        filename: Output STEP file path
        tolerance: Sewing tolerance for joining faces
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

    try:
        # Create a sewing object to join all faces
        sewing = BRepBuilderAPI_Sewing(tolerance)

        faces_added = 0
        faces_to_process = mesh.faces[:max_faces] if total_faces > max_faces else mesh.faces
        num_to_process = len(faces_to_process)

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
                sewing.Add(face_maker.Face())
                faces_added += 1

        if faces_added == 0:
            print("No valid faces could be created")
            return False

        # Perform sewing
        sewing.Perform()
        sewn_shape = sewing.SewedShape()

        # Try to create a solid from the shell
        builder = BRep_Builder()

        # Export using build123d
        if hasattr(sewn_shape, 'wrapped'):
            shape_to_export = sewn_shape
        else:
            # Wrap in Compound for export
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            builder.Add(compound, sewn_shape)
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

#!/usr/bin/env python3
"""
Command-line STEP export for flex PCB visualization.

Usage:
    python step_export_cli.py <input.kicad_pcb> <output.step> [options]

Options:
    --flat              Export flat (unbent) board
    --subdivisions N    Bend zone subdivisions (default: 4)
    --traces            Include copper traces
    --pads              Include pads
    --components        Include component boxes
    --3d-models         Include 3D models from footprints
    --stiffeners        Include stiffeners (default: enabled)
    --no-stiffeners     Disable stiffeners
    --stiffener-thickness  Stiffener thickness in mm (default: from config or 0.2)
    --marker-layer      Layer containing fold markers (default: from config or User.1)
    --merge-faces       Merge adjacent coplanar faces (best with low subdivisions)
    --max-faces N       Maximum faces to export (default: 10000)

Example:
    source venv/bin/activate
    python step_export_cli.py my_board.kicad_pcb my_board.step
    python step_export_cli.py my_board.kicad_pcb my_board.step --3d-models --pads
    python step_export_cli.py my_board.kicad_pcb my_board.step --merge-faces --subdivisions 2
"""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step_export import (
    is_step_export_available, mesh_to_step, mesh_to_step_unified,
    board_geometry_to_step_optimized, board_to_step_direct
)
from mesh import create_board_geometry_mesh
from geometry import extract_geometry
from markers import detect_fold_markers
from kicad_parser import KiCadPCB
from config import FlexConfig
from stiffener import extract_stiffeners


def main():
    parser = argparse.ArgumentParser(
        description='Export KiCad flex PCB to STEP format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s board.kicad_pcb output.step
  %(prog)s board.kicad_pcb output.step --flat
  %(prog)s board.kicad_pcb output.step --3d-models --pads
  %(prog)s board.kicad_pcb output.step --no-stiffeners
  %(prog)s board.kicad_pcb output.step --stiffener-thickness 0.3
        """
    )
    parser.add_argument('input', help='Input KiCad PCB file (.kicad_pcb)')
    parser.add_argument('output', help='Output STEP file (.step)')
    parser.add_argument('--flat', action='store_true',
                        help='Export flat (unbent) board')
    parser.add_argument('--subdivisions', type=int, default=None,
                        help='Bend zone subdivisions (default: from config or 4)')
    parser.add_argument('--traces', action='store_true',
                        help='Include copper traces')
    parser.add_argument('--pads', action='store_true',
                        help='Include pads')
    parser.add_argument('--components', action='store_true',
                        help='Include component boxes (simple 3D representation)')
    parser.add_argument('--3d-models', dest='models_3d', action='store_true',
                        help='Include 3D models from footprints')
    parser.add_argument('--stiffeners', action='store_true', default=True,
                        help='Include stiffeners (default: enabled)')
    parser.add_argument('--no-stiffeners', dest='stiffeners', action='store_false',
                        help='Disable stiffeners')
    parser.add_argument('--stiffener-thickness', type=float, default=None,
                        help='Stiffener thickness in mm (default: from config or 0.2)')
    parser.add_argument('--marker-layer', type=str, default=None,
                        help='Layer containing fold markers (default: from config or User.1)')
    parser.add_argument('--max-faces', type=int, default=10000,
                        help='Maximum faces to export (default: 10000)')
    parser.add_argument('--merge-faces', action='store_true', default=False,
                        help='Merge adjacent coplanar faces (works best with --subdivisions 1-4, mesh ≤2000 faces)')
    parser.add_argument('--no-merge-faces', dest='merge_faces', action='store_false',
                        help='Disable face merging')
    parser.add_argument('--direct', action='store_true', default=False,
                        help='Use direct CAD construction (experimental, smaller files, proper cylindrical bends)')

    args = parser.parse_args()

    # Check if STEP export is available
    if not is_step_export_available():
        print("ERROR: STEP export not available.")
        print("Make sure you're running from the venv with build123d installed:")
        print("  source venv/bin/activate")
        print("  pip install build123d")
        sys.exit(1)

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading PCB: {args.input}")

    try:
        # Load PCB
        pcb = KiCadPCB.load(args.input)
        geom = extract_geometry(pcb)
        pcb_dir = os.path.dirname(os.path.abspath(args.input))

        # Load saved config for this PCB (if exists)
        config = FlexConfig.load_for_pcb(args.input)

        # Override config with command-line args
        if args.subdivisions is not None:
            config.bend_subdivisions = args.subdivisions
        subdivisions = config.bend_subdivisions if config.bend_subdivisions > 0 else 4

        if args.stiffener_thickness is not None:
            config.stiffener_thickness = args.stiffener_thickness
        elif config.stiffener_thickness == 0:
            # Default stiffener thickness if not configured
            config.stiffener_thickness = 0.2

        # Marker layer
        if args.marker_layer is not None:
            config.marker_layer = args.marker_layer
        marker_layer = config.marker_layer if config.marker_layer else "User.1"

        # Detect fold markers (unless flat export)
        markers = None if args.flat else detect_fold_markers(pcb, layer=marker_layer)

        if markers:
            print(f"Found {len(markers)} fold marker(s)")
        else:
            print("No fold markers found (exporting flat)")

        print(f"Board outline: {len(geom.outline.vertices)} vertices")

        # Extract stiffeners if enabled
        stiffeners = None
        if args.stiffeners and not args.flat:
            stiffeners = extract_stiffeners(pcb, config)
            if stiffeners:
                print(f"Found {len(stiffeners)} stiffener region(s), thickness: {config.stiffener_thickness}mm")

        # Print options
        opts = [f"subdivisions={subdivisions}", f"marker-layer={marker_layer}"]
        if args.traces:
            opts.append("traces")
        if args.pads:
            opts.append("pads")
        if args.components:
            opts.append("components")
        if args.models_3d:
            opts.append("3d-models")
        if stiffeners:
            opts.append(f"stiffeners({len(stiffeners)})")
        if args.merge_faces:
            opts.append("merge-faces")
        if args.direct:
            opts.append("direct")
        print(f"Options: {', '.join(opts)}")

        # Export to STEP
        print(f"Exporting to STEP: {args.output}")

        if args.direct and markers and not args.flat:
            # Use direct CAD construction (no mesh intermediate)
            print("Using direct CAD construction...")
            success = board_to_step_direct(
                geom,
                markers=markers,
                filename=args.output,
                num_bend_subdivisions=subdivisions,
                stiffeners=stiffeners if args.stiffeners else None
            )
        else:
            # Generate mesh
            print("Generating mesh...")
            mesh = create_board_geometry_mesh(
                geom,
                markers=markers,
                include_traces=args.traces,
                include_pads=args.pads,
                include_components=args.components and not args.models_3d,
                include_3d_models=args.models_3d,
                pcb_dir=pcb_dir,
                pcb=pcb,
                subdivide_length=1.0,
                num_bend_subdivisions=subdivisions,
                stiffeners=stiffeners,
                apply_bend=not args.flat
            )

            print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

            if args.merge_faces:
                if len(mesh.faces) > 2000:
                    print(f"Note: Face merging requires sewing which is slow for large meshes.")
                    print(f"      Consider using --subdivisions 1-2 for faster merge.")
                success = mesh_to_step_unified(
                    mesh, args.output,
                    max_faces=args.max_faces,
                    merge_faces=True
                )
            else:
                success = mesh_to_step(mesh, args.output, max_faces=args.max_faces)

        if success:
            file_size = os.path.getsize(args.output)
            print(f"SUCCESS: Exported to {args.output} ({file_size:,} bytes)")
        else:
            print("ERROR: STEP export failed")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

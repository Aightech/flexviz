#!/usr/bin/env python3
"""
Command-line STEP export for flex PCB visualization.

Usage:
    python step_export_cli.py <input.kicad_pcb> <output.step> [options]

Options:
    --flat          Export flat (unbent) board
    --subdivisions  Bend zone subdivisions (default: 4)
    --traces        Include copper traces
    --pads          Include pads

Example:
    source venv/bin/activate
    python step_export_cli.py my_board.kicad_pcb my_board.step
"""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step_export import is_step_export_available, mesh_to_step, board_geometry_to_step
from mesh import create_board_geometry_mesh
from geometry import extract_geometry
from markers import detect_fold_markers
from kicad_parser import KiCadPCB


def main():
    parser = argparse.ArgumentParser(
        description='Export KiCad flex PCB to STEP format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s board.kicad_pcb output.step
  %(prog)s board.kicad_pcb output.step --flat
  %(prog)s board.kicad_pcb output.step --subdivisions 8 --pads
        """
    )
    parser.add_argument('input', help='Input KiCad PCB file (.kicad_pcb)')
    parser.add_argument('output', help='Output STEP file (.step)')
    parser.add_argument('--flat', action='store_true',
                        help='Export flat (unbent) board')
    parser.add_argument('--subdivisions', type=int, default=4,
                        help='Bend zone subdivisions (default: 4)')
    parser.add_argument('--traces', action='store_true',
                        help='Include copper traces')
    parser.add_argument('--pads', action='store_true',
                        help='Include pads')
    parser.add_argument('--max-faces', type=int, default=5000,
                        help='Maximum faces to export (default: 5000)')

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

        # Detect fold markers (unless flat export)
        markers = None if args.flat else detect_fold_markers(pcb)

        if markers:
            print(f"Found {len(markers)} fold marker(s)")
        else:
            print("No fold markers found (exporting flat)")

        print(f"Board outline: {len(geom.outline.vertices)} vertices")
        print(f"Options: subdivisions={args.subdivisions}, traces={args.traces}, pads={args.pads}")

        # Generate mesh
        print("Generating mesh...")
        mesh = create_board_geometry_mesh(
            geom,
            markers=markers,
            include_traces=args.traces,
            include_pads=args.pads,
            include_components=False,
            subdivide_length=1.0,
            num_bend_subdivisions=args.subdivisions,
            apply_bend=not args.flat
        )

        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Export to STEP
        print(f"Exporting to STEP: {args.output}")
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

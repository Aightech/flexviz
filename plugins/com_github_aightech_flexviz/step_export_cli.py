#!/usr/bin/env python3
"""Command-line STEP export for flex PCB visualization.

Usage:
    python step_export_cli.py <input.kicad_pcb> <output.step> [options]

Examples:
    python step_export_cli.py board.kicad_pcb board.step
    python step_export_cli.py board.kicad_pcb board.step --stiffener-layer User.2 --stiffener-thickness 0.2
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kicad_parser import KiCadPCB
from geometry import extract_geometry
from markers import detect_fold_markers
from config import FlexConfig
from stiffener import extract_stiffeners
from step_export import board_to_step_native

try:
    from step_export_ocp import board_to_step_ocp, is_ocp_available
except ImportError:
    def board_to_step_ocp(*a, **kw):
        raise ImportError("build123d required for OCP backend")
    def is_ocp_available():
        return False


def main():
    parser = argparse.ArgumentParser(description='Export KiCad flex PCB to STEP')
    parser.add_argument('input', help='Input .kicad_pcb file')
    parser.add_argument('output', help='Output .step file')
    parser.add_argument('--backend', choices=['native', 'ocp', 'auto'],
                        default='auto',
                        help='STEP export backend (default: auto — '
                             'prefers ocp if build123d is installed)')
    parser.add_argument('--marker-layer', default='User.1',
                        help='Fold marker layer (default: User.1)')
    parser.add_argument('--stiffener-layer', default=None,
                        help='Stiffener layer (default: none)')
    parser.add_argument('--stiffener-thickness', type=float, default=0.2,
                        help='Stiffener thickness in mm (default: 0.2)')
    parser.add_argument('--stiffener-side', default='bottom',
                        choices=['top', 'bottom'],
                        help='Stiffener side (default: bottom)')
    parser.add_argument('--include-models', action='store_true',
                        help='Include 3D component models in STEP export')
    parser.add_argument('--include-wrl-models', action='store_true',
                        help='Also include WRL-only models (no STEP equivalent)')
    parser.add_argument('--bbox-models', action='store_true',
                        help='Use bounding box approximations instead of full 3D models')
    parser.add_argument('--compress', action='store_true',
                        help='Write gzip-compressed .stpZ (supported by FreeCAD, KiCad)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    pcb = KiCadPCB.load(args.input)
    geom = extract_geometry(pcb)
    markers = detect_fold_markers(pcb, layer=args.marker_layer)
    print(f"Outline: {len(geom.outline.vertices)} verts, {len(geom.cutouts)} cutouts, {len(markers)} folds")

    config = FlexConfig()
    stiffeners = None
    if args.stiffener_layer:
        config.stiffener_thickness = args.stiffener_thickness
        if args.stiffener_side == 'top':
            config.stiffener_layer_top = args.stiffener_layer
        else:
            config.stiffener_layer_bottom = args.stiffener_layer
        stiffeners = extract_stiffeners(pcb, config)
        print(f"Stiffeners: {len(stiffeners) if stiffeners else 0} ({args.stiffener_side}, {args.stiffener_thickness}mm)")

    pcb_dir = os.path.dirname(os.path.abspath(args.input))

    # Select backend
    backend = args.backend
    if backend == 'auto':
        backend = 'ocp' if is_ocp_available() else 'native'

    export_fn = board_to_step_ocp if backend == 'ocp' else board_to_step_native
    print(f"Backend: {backend}" + (" (build123d/OCP)" if backend == 'ocp' else " (pure Python)"))

    ok = export_fn(
        geom, markers, args.output, pcb=pcb, config=config,
        stiffeners=stiffeners, pcb_dir=pcb_dir,
        include_models=args.include_models or args.bbox_models,
        include_wrl_models=args.include_wrl_models,
        bbox_models=args.bbox_models,
    )
    if ok:
        # Compress to .stpZ if requested
        if args.compress:
            import gzip
            with open(args.output, 'rb') as f_in:
                raw = f_in.read()
            out_z = args.output
            if not out_z.lower().endswith('.stpz'):
                out_z = os.path.splitext(out_z)[0] + '.stpZ'
            with gzip.open(out_z, 'wb', compresslevel=9) as f_out:
                f_out.write(raw)
            os.remove(args.output)
            size = os.path.getsize(out_z)
            ratio = len(raw) / size
            print(f"Exported: {out_z} ({size:,} bytes, {ratio:.1f}x compressed)")
        else:
            size = os.path.getsize(args.output)
            print(f"Exported: {args.output} ({size:,} bytes)")
    else:
        print("ERROR: export failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

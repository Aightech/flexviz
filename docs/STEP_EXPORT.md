# STEP Export Guide

The STEP export feature converts the bent flex PCB mesh to solid CAD geometry using `build123d` (OpenCASCADE bindings).

## Important: Command-Line Only

**STEP export must be run from the command line**, not from within KiCad. This is because build123d's dependencies (vtk, numpy) conflict with KiCad's bundled libraries and cause crashes.

Clicking "Export STEP" in the viewer will show instructions for command-line export.

## Quick Start

```bash
cd /path/to/flexviz
source venv/bin/activate
python step_export_cli.py your_board.kicad_pcb output.step
```

## CLI Options

```
Usage: step_export_cli.py <input.kicad_pcb> <output.step> [options]

Board Options:
  --flat                   Export flat (unbent) board
  --subdivisions N         Bend zone subdivisions (default: from config or 4)

Content Options:
  --traces                 Include copper traces
  --pads                   Include pads
  --components             Include component boxes (simple 3D)
  --3d-models              Include 3D models from footprints

Stiffener Options:
  --stiffeners             Include stiffeners (default: enabled)
  --no-stiffeners          Disable stiffeners
  --stiffener-thickness N  Stiffener thickness in mm (default: from config or 0.2)

Performance Options:
  --max-faces N            Maximum faces to export (default: 5000)

Examples:
  python step_export_cli.py board.kicad_pcb output.step
  python step_export_cli.py board.kicad_pcb flat.step --flat
  python step_export_cli.py board.kicad_pcb detailed.step --3d-models --pads
  python step_export_cli.py board.kicad_pcb output.step --no-stiffeners
  python step_export_cli.py board.kicad_pcb output.step --stiffener-thickness 0.3
```

The CLI automatically loads saved settings from the viewer (if you saved them), so subdivisions and stiffener thickness will use your configured values by default.

## Installation

### Using the Project Virtual Environment (Recommended)

The project venv should already have build123d. If not:

```bash
cd /path/to/flexviz
source venv/bin/activate
pip install build123d
```

### Installing build123d on Different Systems

If you need to set up a new environment:

#### Linux (Ubuntu/Debian)

```bash
python3 -m venv venv
source venv/bin/activate
pip install build123d
```

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
pip install build123d
```

#### macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install build123d
```

## Why Not Inside KiCad?

build123d depends on:
- **vtk** - Visualization toolkit (conflicts with KiCad's OpenGL)
- **numpy** - Numerical library (version conflicts)
- **OCC** - OpenCASCADE (different version than KiCad's)

Installing these in the system Python or KiCad's Python causes crashes. The solution is to use an isolated virtual environment and run exports from the command line.

## Performance Notes

- STEP export converts mesh triangles to solid faces
- Export time: ~1-2 seconds per 100 faces
- Complex meshes (>5000 faces) are automatically limited
- For faster exports, use lower `--subdivisions` value

## Output Compatibility

Exported STEP files (AP214) are compatible with:
- FreeCAD
- Fusion 360
- SolidWorks
- CATIA
- Onshape
- Any CAD software supporting STEP

## Troubleshooting

### "STEP export not available"

Make sure you've activated the venv:
```bash
source venv/bin/activate
python -c "from build123d import Box; print('OK')"
```

### Slow Export

Reduce mesh complexity:
```bash
python step_export_cli.py board.kicad_pcb output.step --subdivisions 2
```

### Memory Errors

Limit the number of faces:
```bash
python step_export_cli.py board.kicad_pcb output.step --max-faces 2000
```

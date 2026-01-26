# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KiCad Flex Viewer is a KiCad Python plugin for visualizing folded/bent flex PCBs in 3D. It provides toolbar buttons in KiCad PCB Editor to create fold markers and view the bent PCB in an interactive 3D window.

## Commands

### Running Tests

```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_kicad_parser.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Installation (Development)

```bash
./install.sh  # Creates symlink to KiCad plugins directory
```

### Dependencies

```bash
pip install pytest pytest-cov numpy pyvista
```

## Repository Structure

```
flexviz/
├── plugins/com_github_aightech_flexviz/  # KiCad plugin (installed to KiCad)
│   ├── __init__.py          # Plugin registration
│   ├── plugin.py            # ActionPlugin classes
│   ├── kicad_parser.py      # .kicad_pcb file parser
│   ├── geometry.py          # Board geometry extraction
│   ├── markers.py           # Fold marker detection
│   ├── bend_transform.py    # 3D transformation
│   ├── mesh.py              # Triangle mesh generation
│   ├── viewer.py            # wxPython + OpenGL viewer
│   ├── step_export.py       # STEP CAD export
│   └── resources/           # Icons
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── install.sh               # Installation script
```

## Architecture

### Data Flow Pipeline

```
.kicad_pcb file → KiCadPCB parser → BoardGeometry + FoldMarkers → FoldDefinitions → Mesh → OpenGL render
```

1. **kicad_parser.py**: S-expression tokenizer/parser that reads `.kicad_pcb` files into an `SExpr` tree structure. The `KiCadPCB` class wraps this with convenience methods.

2. **geometry.py**: Extracts board geometry (outline, traces, pads, components) from parsed PCB data into a `BoardGeometry` dataclass.

3. **markers.py**: Detects fold markers from a configurable User layer (default: User.1). Uses dimension-first detection: starts from each dimension's start point and finds the containing parallel line pair. This avoids mismatch when markers are close together. Fold axis direction is normalized for consistency (horizontal folds: +X, vertical folds: +Y) to ensure parallel folds have consistent perpendicular directions.

4. **bend_transform.py**: Transforms 2D flat geometry into 3D bent geometry using recipe-based fold application. Each region has a fold_recipe specifying which folds affect it and how (IN_ZONE or AFTER).

5. **mesh.py**: Generates triangle meshes for board outline, traces, and pads. Handles subdivision of geometry crossing bend zones for smooth curves.

6. **viewer.py**: wxPython + OpenGL viewer window (`FlexViewerFrame`). Uses `wx.glcanvas` for rendering - no external 3D libraries needed since these are bundled with KiCad.

7. **plugin.py**: KiCad `ActionPlugin` registration. Three actions: Test, Create Fold, Open Viewer.

8. **step_export.py**: Exports bent geometry to STEP format using build123d/OpenCASCADE. Creates true CAD solids with named bodies (FLEX_PCB, STIFFENER_N).

### KiCad Plugin Entry Point

`plugins/com_github_aightech_flexviz/__init__.py` registers the action plugins. If imports fail, it logs errors to `flex_viewer_error.log` and registers a dummy error-reporting plugin instead.

### Fold Marker Convention

Fold markers on a User layer (configurable via viewer or config) consist of:
- Two parallel dotted lines defining the bend zone boundaries
- A dimension object between them showing the bend angle (positive = toward viewer)
- Bend radius is derived from: `R = line_distance / angle_in_radians`
- Marker detection uses dimension-first approach: finds the dimension start point, then locates containing parallel lines

### Configuration

`config.py` manages user preferences via `FlexConfig` class:
- `marker_layer`: Layer for fold markers (default: "User.1")
- `bend_subdivisions`: Number of subdivisions in bend zones
- `stiffener_layer_top/bottom`: Layers for stiffener regions
- `stiffener_thickness`: Thickness of stiffeners in mm
- Settings are saved per-PCB in `<pcb_name>.flexviewer.json`

### Test Data

Test fixtures in `tests/conftest.py` provide paths to test PCB files in `tests/test_data/`. The `minimal_pcb_content` fixture provides inline S-expression content for simple tests.

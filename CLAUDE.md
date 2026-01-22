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

## Architecture

### Data Flow Pipeline

```
.kicad_pcb file → KiCadPCB parser → BoardGeometry + FoldMarkers → FoldDefinitions → Mesh → OpenGL render
```

1. **kicad_parser.py**: S-expression tokenizer/parser that reads `.kicad_pcb` files into an `SExpr` tree structure. The `KiCadPCB` class wraps this with convenience methods.

2. **geometry.py**: Extracts board geometry (outline, traces, pads, components) from parsed PCB data into a `BoardGeometry` dataclass.

3. **markers.py**: Detects fold markers from User.1 layer (dotted line pairs + dimension text) and builds `FoldMarker` objects containing angle and position data.

4. **bend_transform.py**: Transforms 2D flat geometry into 3D bent geometry. `FoldDefinition` encapsulates fold parameters. Points are classified as before/in/after the bend zone and transformed accordingly (cylindrical mapping in bend zone, rotation+translation after).

5. **mesh.py**: Generates triangle meshes for board outline, traces, and pads. Handles subdivision of geometry crossing bend zones for smooth curves.

6. **viewer.py**: wxPython + OpenGL viewer window (`FlexViewerFrame`). Uses `wx.glcanvas` for rendering - no external 3D libraries needed since these are bundled with KiCad.

7. **plugin.py**: KiCad `ActionPlugin` registration. Three actions: Test, Create Fold, Open Viewer.

### KiCad Plugin Entry Point

`__init__.py` registers the action plugins. If imports fail, it logs errors to `flex_viewer_error.log` and registers a dummy error-reporting plugin instead.

### Fold Marker Convention

Fold markers on User.1 layer consist of:
- Two parallel dotted lines defining the bend zone boundaries
- A dimension object between them showing the bend angle (positive = toward viewer)
- Bend radius is derived from: `R = line_distance / angle_in_radians`

### Test Data

Test fixtures in `tests/conftest.py` provide paths to test PCB files in `tests/test_data/`. The `minimal_pcb_content` fixture provides inline S-expression content for simple tests.

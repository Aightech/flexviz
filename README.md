# KiCad Flex Viewer

A KiCad Python extension for visualizing folded/bent flex PCBs in an interactive 3D window.

## Overview

KiCad does not natively support flex PCB bend visualization in its 3D viewer. This plugin fills that gap by:

1. Providing a marker system to define fold lines directly in the PCB layout
2. Parsing the board geometry from `.kicad_pcb` files (no STEP export needed)
3. Applying mathematical bend transformations
4. Rendering the folded PCB in a separate interactive 3D window

## Features

- **Direct KiCad integration**: Two action buttons in PCB Editor toolbar
- **Visual fold markers**: Define bends using selection-based placement
- **Real-time preview**: Adjust bend angles with interactive controls
- **Performance modes**: From fast outline-only to full component rendering
- **3D model support**: Load and display component 3D models (WRL/STEP)
- **Stiffener regions**: Define and visualize stiffener areas
- **STEP export**: Export bent geometry to STEP format for CAD tools
- **Configurable marker layer**: Use any User layer for fold markers
- **Design validation**: Warnings for bend radius, stiffener conflicts, components in bend zones

## Plugin Buttons

### Button 1: Create Fold

Selection-based workflow:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Draw a graphic line on User layer where you want the     │
│    fold (the line defines fold position and direction)      │
│                                                             │
│ 2. Select the line in KiCad                                 │
│                                                             │
│ 3. Click "Create Fold" button                               │
│    └─► Dialog opens with angle and zone width inputs        │
│    └─► Creates two parallel dotted lines + dimension        │
│    └─► Original line is replaced with fold marker           │
└─────────────────────────────────────────────────────────────┘
```

**Alternative:** If no line is selected, a dialog prompts for manual coordinate entry.

### Button 2: Open Viewer

Opens the 3D visualization window with:
- Folded PCB rendering
- Angle adjustment sliders (per fold)
- Display mode checkboxes (outline/traces/pads/components)
- Export options

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KiCad PCB Editor                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Toolbar: [Create Fold] [Open Viewer]                │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Board with fold markers on User layer (configurable) │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │ "Open Viewer" clicked
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Flex Viewer Window                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────┬──────────────────────┐ │
│  │                                 │ Controls:            │ │
│  │                                 │ ┌──────────────────┐ │ │
│  │      3D Viewport                │ │ Fold 1: [====●==]│ │ │
│  │      (PyVista/vispy)            │ │ Fold 2: [==●====]│ │ │
│  │                                 │ └──────────────────┘ │ │
│  │                                 │ Display:             │ │
│  │                                 │ ☑ Outline            │ │
│  │                                 │ ☑ Traces             │ │
│  │                                 │ ☐ Pads               │ │
│  │                                 │ ☐ Components         │ │
│  │                                 │                      │ │
│  │                                 │ [Refresh] [Export]   │ │
│  └─────────────────────────────────┴──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Fold Marker System

Fold markers are placed on a User layer (configurable, default: User.1) and consist of three elements:

```
        ┆       ┆
        ┆       ┆
═══════╪════════╪═══════════════  PCB Edge
        ┆       ┆
        ┆ +90°  ┆   ← Dimension (positive = fold toward viewer)
        ┆       ┆
═══════╪════════╪═══════════════
        ┆       ┆
      Line A   Line B
      (dotted) (dotted)
```

### Marker Elements

| Element | Description |
|---------|-------------|
| **Line A** | Dotted line on User.1 - start of bend zone |
| **Line B** | Dotted line on User.1 - end of bend zone |
| **Dimension** | Angle in degrees between the lines |

### Bend Parameters

- **Angle**: From dimension text (signed: +up/-down from PCB plane)
- **Bend radius**: Derived from line spacing: `R = line_distance / angle_in_radians`
- **Fold axis**: Perpendicular to line A-B direction
- **Order**: Not required - folds are independent based on position

## Rendering Modes

| Mode | Elements Rendered | Performance | Use Case |
|------|-------------------|-------------|----------|
| **Outline** | Board edge only | Very fast | Real-time angle adjustment |
| **Traces** | Outline + copper traces | Fast | Design verification |
| **Pads** | Above + pads | Medium | Component placement check |
| **Components** | Above + component bounding boxes | Medium | Collision detection |
| **Full 3D** | Above + 3D models | Slow | Final visualization |

## Installation

### Quick Install (Development)

```bash
cd kicad_flex_viewer
./install.sh
```

The install script will:
- Auto-detect your KiCad version (7.0, 8.0, 9.0)
- Create a symlink for development (or copy for distribution)
- Show you where the plugin was installed

### Manual Install

```bash
# For KiCad 9.x
ln -s /path/to/kicad_flex_viewer ~/.local/share/kicad/9.0/scripting/plugins/kicad_flex_viewer

# For KiCad 8.x
ln -s /path/to/kicad_flex_viewer ~/.local/share/kicad/8.0/scripting/plugins/kicad_flex_viewer

# For KiCad 7.x
ln -s /path/to/kicad_flex_viewer ~/.local/share/kicad/7.0/scripting/plugins/kicad_flex_viewer
```

### Verify Installation

1. Restart KiCad PCB Editor
2. Go to **Tools → External Plugins**
3. You should see:
   - **Flex Viewer - Test** (verifies plugin works)
   - **Create Fold** (placeholder)
   - **Open Fold Viewer** (placeholder)

### Dependencies (for Phase 3+)

```bash
pip install vispy numpy
```

## Usage

1. **Create folds**: Click "Create Fold" button → click point A → click point B → set angle
2. **Repeat** for additional folds as needed
3. **Visualize**: Click "Open Viewer" to see the folded PCB in 3D
4. **Adjust**: Use sliders to fine-tune angles, checkboxes for display options
5. **Export**: Save bent geometry as OBJ/STL if needed

## File Structure

```
kicad_flex_viewer/
├── __init__.py                # Package init
├── plugin.py                  # KiCad action plugin registration
├── kicad_parser.py            # .kicad_pcb file parser (S-expression)
├── geometry.py                # Geometry extraction (outline, traces, pads)
├── markers.py                 # Fold marker detection and parsing
├── bend_transform.py          # Bend transformation mathematics (Phase 2)
├── fold_placer.py             # Interactive placement logic (Phase 4)
├── viewer.py                  # 3D visualization window (Phase 3)
├── ui_controls.py             # Sidebar controls (Phase 3)
├── install.sh                 # Installation script
├── tests/                     # Unit tests
│   ├── test_kicad_parser.py
│   ├── test_markers.py
│   ├── test_geometry.py
│   └── test_data/             # Test PCB files
└── resources/
    ├── icon_test.png          # Test button icon
    ├── icon_create_fold.png   # Create Fold button icon
    └── icon_open_viewer.png   # Open Viewer button icon
```

## Bend Transformation

The bend transformation maps points from flat PCB space to bent 3D space:

```
For a fold at position P, with axis direction A, radius R, and angle θ:

1. Points before fold zone: unchanged (z = 0)

2. Points in fold zone (0 < d < arc_length):
   - Map to cylindrical coordinates
   - x' = x
   - y' = R × sin(d / R)
   - z' = R × (1 - cos(d / R))

3. Points after fold zone:
   - Apply rotation matrix around fold axis
   - Translate to end of arc
```

## Limitations

- Components are displayed as bounding boxes or reference points (full 3D model bending is computationally expensive)
- Single-sided view (no layer stackup simulation)
- Bend radius must be larger than board thickness for realistic results

## Future Enhancements

- [ ] Fold animation / stepped preview
- [x] ~~Export to STEP~~ (completed via build123d)
- [ ] Collision detection between folded sections
- [ ] Integration with DRC for bend radius rules
- [ ] Support for rigid-flex zone definitions
- [ ] Edit existing fold markers (move, change angle)
- [ ] Delete fold marker tool
- [ ] Silkscreen layer display
- [ ] Copper zone fill visualization

## License

MIT License

## Contributing

Contributions welcome! Please open an issue to discuss proposed changes.

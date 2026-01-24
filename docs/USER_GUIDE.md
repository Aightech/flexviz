# KiCad Flex Viewer - User Guide

A complete guide to visualizing bent flex PCBs in KiCad.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Creating Fold Markers](#creating-fold-markers)
4. [Using the 3D Viewer](#using-the-3d-viewer)
5. [Stiffener Regions](#stiffener-regions)
6. [Exporting](#exporting)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Installation

### Requirements

- KiCad 7.0, 8.0, or 9.0
- Python 3.8+ (bundled with KiCad)

### Quick Install (Recommended)

```bash
cd /path/to/kicad_flex_viewer
./install.sh
```

The script automatically:
- Detects your KiCad version
- Creates a symlink in the plugins directory
- Reports the installation location

### Manual Install

#### Linux

```bash
# For KiCad 9.x
ln -s /path/to/kicad_flex_viewer ~/.local/share/kicad/9.0/scripting/plugins/kicad_flex_viewer

# For KiCad 8.x
ln -s /path/to/kicad_flex_viewer ~/.local/share/kicad/8.0/scripting/plugins/kicad_flex_viewer

# For KiCad 7.x
ln -s /path/to/kicad_flex_viewer ~/.local/share/kicad/7.0/scripting/plugins/kicad_flex_viewer
```

#### Windows

Copy the `kicad_flex_viewer` folder to:
- KiCad 9.x: `%APPDATA%\kicad\9.0\scripting\plugins\`
- KiCad 8.x: `%APPDATA%\kicad\8.0\scripting\plugins\`
- KiCad 7.x: `%APPDATA%\kicad\7.0\scripting\plugins\`

#### macOS

```bash
# For KiCad 9.x
ln -s /path/to/kicad_flex_viewer ~/Library/Preferences/kicad/9.0/scripting/plugins/kicad_flex_viewer
```

### Verify Installation

1. Restart KiCad PCB Editor
2. Go to **Tools > External Plugins**
3. You should see:
   - **Create Fold** - Creates fold markers
   - **Open Fold Viewer** - Opens 3D viewer

---

## Quick Start

### Step 1: Open Your PCB

Open your flex PCB design in KiCad PCB Editor.

### Step 2: Create a Fold Marker

1. Draw a line on a User layer (default: User.1) where you want the fold
2. Select the line
3. Click **Create Fold** in the toolbar
4. Enter the fold angle (e.g., 90 for a right-angle bend)
5. The plugin creates two parallel dotted lines with a dimension showing the angle

### Step 3: View in 3D

1. Click **Open Fold Viewer** in the toolbar
2. The 3D viewer opens showing your bent PCB
3. Adjust angles using the spin controls
4. Toggle display options (traces, pads, components)

### Step 4: Export (Optional)

- **OBJ/STL**: Click export buttons in the viewer
- **STEP**: Use the command-line tool (see [STEP Export](#step-export))

---

## Creating Fold Markers

### Marker Structure

Each fold marker consists of three elements on a User layer:

```
    ┆       ┆
    ┆       ┆
════╪═══════╪════  PCB Edge
    ┆       ┆
    ┆ +90°  ┆   ← Dimension (angle)
    ┆       ┆
════╪═══════╪════
    ┆       ┆
  Line A  Line B
```

### Using Create Fold Tool

**Method 1: Selection-Based (Recommended)**

1. Draw a graphic line on User.1 (or your marker layer) where you want the fold
2. Select the line in KiCad
3. Click **Create Fold** button
4. In the dialog:
   - **Angle**: Bend angle in degrees (positive = toward viewer)
   - **Zone width**: Width of the bend zone (affects bend radius)
5. Click OK

**Method 2: Manual Entry**

1. Click **Create Fold** without selecting anything
2. Enter coordinates for point A and point B
3. Enter angle and zone width

### Marker Parameters

| Parameter | Description |
|-----------|-------------|
| **Angle** | Bend angle in degrees. Positive folds toward you, negative folds away. |
| **Zone width** | Distance between the two marker lines. Determines bend radius. |
| **Bend radius** | Calculated as: `R = zone_width / angle_in_radians` |

### Changing Marker Layer

By default, fold markers are detected on **User.1**. To use a different layer:

1. Open the viewer
2. In **PCB Settings**, change **Marker layer**
3. Click **Save Settings** to remember for future sessions

### Tips for Marker Placement

- Place fold lines perpendicular to the flex direction
- Ensure marker lines cross the entire board width
- Keep adequate spacing between multiple folds
- Avoid placing folds through component areas

---

## Using the 3D Viewer

### Camera Controls

| Action | Mouse Control |
|--------|---------------|
| Rotate | Left-click + drag |
| Pan | Middle-click + drag |
| Zoom | Scroll wheel |

### Control Panel

#### Fold Angles

Each fold has a spin control showing the current angle:
- Adjust from -360° to +360°
- Status indicator shows validation: green (OK), yellow (warning), red (error)
- Hover over the indicator for details

#### Display Options

| Option | Description |
|--------|-------------|
| **Show Wireframe** | Overlay mesh edges |
| **Show Traces** | Display copper traces |
| **Show Pads** | Display component pads and holes |
| **Show Components** | Display component bounding boxes |
| **Show Stiffeners** | Display stiffener regions |
| **Show 3D Models** | Load footprint 3D models (slower) |

#### PCB Settings

| Setting | Description |
|---------|-------------|
| **PCB thickness** | Read from board settings (display only) |
| **Bend quality** | Number of segments in bend zones (1-32) |
| **Marker layer** | Layer containing fold markers |

#### Stiffener Settings

| Setting | Description |
|---------|-------------|
| **Thickness** | Stiffener material thickness (mm) |
| **Top layer** | Layer with top-side stiffener outlines |
| **Bottom layer** | Layer with bottom-side stiffener outlines |

### Validation Panel

The viewer performs automatic design checks:

- **Bend radius warnings**: Alerts if radius is too small for the PCB thickness
- **Stiffener conflicts**: Errors if a fold line crosses a stiffener region
- **Component in bend zone**: Warnings if components are in bend areas

---

## Stiffener Regions

Stiffeners are rigid areas added to flex PCBs for mechanical support.

### Defining Stiffeners

1. Draw closed polygons on a User layer (e.g., User.2 for top, User.3 for bottom)
2. In the viewer, set **Stiffener > Top layer** and/or **Bottom layer**
3. Set the **Thickness** (typically 0.1-0.3mm)
4. Stiffeners appear in the 3D view

### Validation

The viewer checks for conflicts:
- Folds crossing stiffener regions (error - would break the stiffener)
- Components placed on stiffeners (informational)

---

## Exporting

### OBJ Export

Wavefront OBJ format, compatible with:
- Blender
- MeshLab
- Most 3D software

Click **Export OBJ** in the viewer.

### STL Export

STL format, suitable for:
- 3D printing
- CAD import

Click **Export STL** in the viewer.

### STEP Export

STEP format (AP214), compatible with:
- FreeCAD
- Fusion 360
- SolidWorks
- CATIA
- Onshape

**Important**: STEP export must be run from command line due to library conflicts with KiCad.

#### Using the CLI Tool

```bash
cd /path/to/kicad_flex_viewer
source venv/bin/activate
python step_export_cli.py your_board.kicad_pcb output.step
```

#### CLI Options

```
Options:
  --flat                   Export flat (unbent) board
  --subdivisions N         Bend zone subdivisions (default: from config)
  --marker-layer LAYER     Layer containing fold markers
  --traces                 Include copper traces
  --pads                   Include pads
  --components             Include component boxes
  --3d-models              Include 3D models from footprints
  --stiffeners             Include stiffeners (default: enabled)
  --no-stiffeners          Disable stiffeners
  --stiffener-thickness N  Stiffener thickness in mm
```

#### Examples

```bash
# Basic export
python step_export_cli.py board.kicad_pcb output.step

# Flat export (no bending)
python step_export_cli.py board.kicad_pcb flat.step --flat

# With 3D models and pads
python step_export_cli.py board.kicad_pcb detailed.step --3d-models --pads

# Custom marker layer
python step_export_cli.py board.kicad_pcb output.step --marker-layer User.5
```

---

## Troubleshooting

### Plugin doesn't appear in KiCad

1. Check installation path matches your KiCad version
2. Restart KiCad completely (not just the PCB editor)
3. Check for errors in `flex_viewer_error.log` in the plugin directory
4. Verify Python can import the modules:
   ```bash
   cd /path/to/kicad_flex_viewer
   python -c "from kicad_parser import KiCadPCB; print('OK')"
   ```

### "No fold markers found"

1. Verify markers are on the correct layer (check Marker layer setting)
2. Ensure each fold has:
   - Two parallel dotted lines
   - A dimension annotation between them
3. Check that marker lines cross the board outline

### 3D view shows broken geometry

1. Check fold marker orientations are consistent
2. Ensure parallel folds use consistent line directions
3. Try reducing bend angles to isolate the problem
4. Check for overlapping fold zones

### STEP export fails

1. Ensure you're running from command line, not inside KiCad
2. Activate the virtual environment: `source venv/bin/activate`
3. Install build123d if missing: `pip install build123d`
4. Try with fewer features (e.g., without 3D models)

### Viewer is slow

1. Reduce **Bend quality** (subdivisions)
2. Disable **Show 3D Models**
3. Disable **Show Components**
4. For large boards, try viewing outline only first

### Settings not saved

1. Click **Save Settings** button explicitly
2. Check write permissions for PCB directory
3. Settings are stored in `<pcb_name>.flexviewer.json`

---

## FAQ

### Q: What bend angles are supported?

A: Any angle from -360° to +360°. Common values:
- 90° - right angle fold
- 180° - fold flat (U-turn)
- 45° - shallow bend

### Q: How is bend radius calculated?

A: `Radius = zone_width / angle_in_radians`

For example, with zone_width = 5mm and angle = 90° (π/2 radians):
`R = 5 / (π/2) ≈ 3.18mm`

### Q: Can I have multiple folds?

A: Yes. Create multiple fold markers. The viewer handles sequential folds automatically, calculating cumulative transformations.

### Q: What's the minimum bend radius?

A: Depends on your flex PCB specifications. The viewer warns if the calculated radius is less than 6× the PCB thickness (configurable via `min_bend_radius_factor` in config).

### Q: Can I edit fold markers after creation?

A: Currently, edit markers directly in KiCad:
- Move/resize the dotted lines
- Edit the dimension value
- Delete and recreate if needed

### Q: Why is STEP export separate from the viewer?

A: The build123d library (used for STEP) conflicts with KiCad's bundled libraries. Running from command line uses an isolated environment.

### Q: How do I visualize back-side copper?

A: Traces on B.Cu (back copper) are automatically rendered on the bottom of the PCB and follow bend transformations.

### Q: Can I animate the folding?

A: Not yet. Animation export is planned for a future release. Currently, adjust angles manually to preview different states.

### Q: What User layers can I use for markers?

A: Any User layer (User.1 through User.9). Select the layer in the viewer's **Marker layer** dropdown.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| R | Reset view |
| W | Toggle wireframe |
| Escape | Close viewer |

---

## Getting Help

- **Issues**: Report bugs at https://github.com/anthropics/claude-code/issues
- **Source**: Plugin source code in the installation directory
- **Logs**: Check `flex_viewer_error.log` for startup errors

---

*KiCad Flex Viewer - Visualize your flex PCB designs in 3D*

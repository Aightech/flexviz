# Implementation Checklist

## Phase 1: Core Foundation (No KiCad Required)

This phase can be developed and tested entirely outside of KiCad.

### 1.1 KiCad Parser (`kicad_parser.py`)

- [ ] Parse `.kicad_pcb` file structure (S-expression tokenizer)
- [ ] Extract board general info (thickness, layers)
- [ ] Extract Edge.Cuts layer → board outline polygon
- [ ] Extract graphic lines from User.1 layer
- [ ] Extract dimension objects from User.1 layer
- [ ] Handle multi-polygon boards (cutouts)

**Unit Tests (`tests/test_kicad_parser.py`):**
- [ ] Test S-expression tokenizer with sample strings
- [ ] Test parsing minimal valid `.kicad_pcb` file
- [ ] Test board outline extraction (rectangle)
- [ ] Test board outline extraction (complex polygon)
- [ ] Test board with cutouts
- [ ] Test graphic line extraction
- [ ] Test dimension object extraction
- [ ] Test error handling for malformed files

### 1.2 Marker Detection (`markers.py`)

- [ ] Find dotted lines on User.1 layer
- [ ] Find dimension objects on User.1 layer
- [ ] Group lines into pairs (by proximity/parallelism)
- [ ] Associate dimension with line pair
- [ ] Parse angle from dimension text (handle +/- signs)
- [ ] Calculate bend radius from line spacing
- [ ] Build `FoldMarker` data structure

**Unit Tests (`tests/test_markers.py`):**
- [ ] Test dotted line detection
- [ ] Test line pairing algorithm
- [ ] Test dimension association
- [ ] Test angle parsing (positive, negative, with/without degree symbol)
- [ ] Test radius calculation
- [ ] Test multiple fold markers in same file
- [ ] Test invalid/incomplete markers (missing dimension, single line)

### 1.3 Geometry Extraction (`geometry.py`)

- [ ] Convert board outline to vertex list
- [ ] Extract copper traces (segments with width)
- [ ] Extract pads (position, shape, size)
- [ ] Extract component positions (reference point, rotation)
- [ ] Extract component bounding boxes
- [ ] Optional: extract 3D model paths

**Unit Tests (`tests/test_geometry.py`):**
- [ ] Test outline vertex extraction
- [ ] Test trace segment extraction
- [ ] Test pad extraction (various shapes: rect, circle, oval)
- [ ] Test component position extraction
- [ ] Test bounding box calculation
- [ ] Test empty board handling

---

### Testing Checkpoint 1

**Manual Testing:**
- [ ] Parse a real `.kicad_pcb` file from your project
- [ ] Verify extracted outline matches KiCad display
- [ ] Manually add fold markers in KiCad, verify detection
- [ ] Print extracted geometry data, sanity check values

**Deliverable:** Parser can read real KiCad files and extract geometry + markers.

---

## Phase 2: Bend Mathematics

### 2.1 Bend Transform (`bend_transform.py`)

- [ ] Define `FoldDefinition` class (position, axis, radius, angle)
- [ ] Implement point classification (before/in/after bend zone)
- [ ] Implement cylindrical mapping for points in bend zone
- [ ] Implement rotation + translation for points after bend
- [ ] Handle multiple sequential folds
- [ ] Transform line segments (bend both endpoints)
- [ ] Transform polygons (bend all vertices, handle subdivision)

**Unit Tests (`tests/test_bend_transform.py`):**
- [ ] Test point classification
- [ ] Test 90° bend on simple points
- [ ] Test 45° bend
- [ ] Test 180° bend (fold flat)
- [ ] Test negative angle (opposite direction)
- [ ] Test point exactly on fold line
- [ ] Test point in middle of bend zone
- [ ] Test multiple folds
- [ ] Test line segment transformation
- [ ] Test polygon transformation
- [ ] Test identity transform (0° bend)

### 2.2 Mesh Generation (`mesh.py`)

- [ ] Generate flat board mesh from outline
- [ ] Generate bent board mesh using transforms
- [ ] Subdivide edges crossing bend zones for smooth curves
- [ ] Generate trace meshes (as ribbons with width)
- [ ] Generate pad meshes
- [ ] Generate component placeholder boxes

**Unit Tests (`tests/test_mesh.py`):**
- [ ] Test flat mesh generation (vertex count, face count)
- [ ] Test mesh topology (no holes, correct winding)
- [ ] Test bend zone subdivision
- [ ] Test trace ribbon generation
- [ ] Test pad mesh generation

---

### Testing Checkpoint 2

**Manual Testing:**
- [ ] Create test board with single fold, export bent mesh to OBJ
- [ ] Open OBJ in Blender/MeshLab, verify geometry looks correct
- [ ] Test with 45°, 90°, 135°, 180° bends
- [ ] Test with negative angles
- [ ] Create board with two folds, verify both apply correctly

**Deliverable:** Bend math produces correct 3D geometry.

---

## Phase 3: 3D Viewer

### 3.1 Viewer Window (`viewer.py`)

- [ ] Create vispy window
- [ ] Load and display mesh
- [ ] Implement camera controls (orbit, pan, zoom)
- [ ] Set up lighting
- [ ] Color coding (board, traces, pads)
- [ ] Wireframe toggle

**Unit Tests (`tests/test_viewer.py`):**
- [ ] Test window creation (may need to skip in CI)
- [ ] Test mesh loading
- [ ] Test color assignment

### 3.2 UI Controls (`ui_controls.py`)

- [ ] Create sidebar panel (wxPython or Qt)
- [ ] Add angle slider per fold (linked to fold marker)
- [ ] Add display checkboxes (outline, traces, pads, components)
- [ ] Add refresh button
- [ ] Add export button (OBJ/STL)
- [ ] Real-time update on slider change

**Unit Tests (`tests/test_ui_controls.py`):**
- [ ] Test slider value binding
- [ ] Test checkbox state management
- [ ] Test callback triggering

### 3.3 Viewer Integration

- [ ] Connect parser → geometry → transform → mesh → viewer pipeline
- [ ] Implement refresh: re-read file, rebuild mesh
- [ ] Implement export: save current mesh to file
- [ ] Handle viewer window close gracefully

---

### Testing Checkpoint 3

**Manual Testing:**
- [ ] Open viewer with test board
- [ ] Adjust angle slider, verify real-time update
- [ ] Toggle display options, verify elements show/hide
- [ ] Test orbit/pan/zoom controls
- [ ] Export to OBJ, open in external tool
- [ ] Test with complex board (many traces, pads)

**Deliverable:** Standalone viewer works with test files.

---

## Phase 4: KiCad Integration

### 4.1 Plugin Registration (`__init__.py`)

- [ ] Register "Create Fold" action plugin
- [ ] Register "Open Viewer" action plugin
- [ ] Set up toolbar icons
- [ ] Configure plugin metadata (name, description, icon)

**Manual Testing:**
- [ ] Plugin appears in Tools → External Plugins
- [ ] Toolbar buttons appear
- [ ] Clicking buttons triggers actions

### 4.2 Create Fold Action (`action_create_fold.py`)

- [ ] Implement action plugin class
- [ ] Launch fold placer on button click
- [ ] Handle plugin lifecycle

### 4.3 Fold Placer (`fold_placer.py`)

- [ ] Enter placement mode on activation
- [ ] Capture mouse click for point A
- [ ] Draw temporary marker at point A
- [ ] Capture mouse move, draw preview line
- [ ] Capture mouse click for point B
- [ ] Handle ESC key to cancel
- [ ] Handle right-click to cancel
- [ ] Show angle input dialog after point B
- [ ] Create final geometry (2 dotted lines + dimension)
- [ ] Clean up temporary graphics
- [ ] Update status bar messages

**Unit Tests (`tests/test_fold_placer.py`):**
- [ ] Test state machine transitions
- [ ] Test geometry creation logic
- [ ] Test cancellation handling

**Manual Testing:**
- [ ] Click Create Fold, click two points, verify lines created
- [ ] Test ESC cancellation at various stages
- [ ] Test right-click cancellation
- [ ] Verify angle dialog appears and value is used
- [ ] Verify lines are on User.1 layer
- [ ] Verify lines are dotted style
- [ ] Verify dimension shows correct angle

### 4.4 Open Viewer Action (`action_open_viewer.py`)

- [ ] Get current board file path
- [ ] Launch viewer window with board
- [ ] Handle case: no board open
- [ ] Handle case: board not saved

**Manual Testing:**
- [ ] Open viewer from KiCad with board loaded
- [ ] Verify viewer shows correct board
- [ ] Test with unsaved board (should prompt or warn)

---

### Testing Checkpoint 4

**Integration Testing:**
- [ ] Full workflow: Create fold → Open viewer → Adjust → Export
- [ ] Test with real project board
- [ ] Test creating multiple folds
- [ ] Test viewer updates when angle changed in KiCad (via Refresh)
- [ ] Test on KiCad 7.x
- [ ] Test on KiCad 8.x

**Deliverable:** Plugin works end-to-end in KiCad.

---

## Phase 5: Polish & Edge Cases

### 5.1 Error Handling

- [ ] Handle missing User.1 layer
- [ ] Handle no fold markers found
- [ ] Handle invalid dimension text
- [ ] Handle non-parallel fold lines
- [ ] Handle fold lines outside board outline
- [ ] Display user-friendly error messages

### 5.2 Performance Optimization

- [ ] Profile with large board (1000+ traces)
- [ ] Optimize mesh generation if needed
- [ ] Add progress indicator for slow operations
- [ ] Implement level-of-detail rendering

### 5.3 Edge Cases

- [ ] Board with holes/cutouts
- [ ] Circular board outline
- [ ] Fold line at board edge
- [ ] Very small bend radius (< board thickness)
- [ ] Very large angle (> 180°)
- [ ] Overlapping fold zones

---

### Testing Checkpoint 5

**Stress Testing:**
- [ ] Test with largest board in project
- [ ] Test with many folds (5+)
- [ ] Test error conditions trigger appropriate messages
- [ ] Test recovery from errors

---

## Phase 6: Documentation

### 6.1 User Documentation

- [ ] Installation guide (KiCad 7.x and 8.x)
- [ ] Quick start tutorial with screenshots
- [ ] Marker placement guide
- [ ] Viewer controls reference
- [ ] Troubleshooting section
- [ ] FAQ

### 6.2 Developer Documentation

- [ ] Architecture overview
- [ ] API documentation (docstrings)
- [ ] Contributing guide
- [ ] Build/test instructions

### 6.3 Release Preparation

- [ ] Version number
- [ ] Changelog
- [ ] License file
- [ ] Package for KiCad Plugin Manager (PCM)
- [ ] Create release on GitHub

---

## Test Infrastructure

### Test Setup

```bash
# Create test environment
cd kicad_flex_viewer
python -m venv venv
source venv/bin/activate
pip install pytest pytest-cov numpy pyvista

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Data

```
tests/
├── conftest.py              # Shared fixtures
├── test_data/
│   ├── minimal.kicad_pcb    # Minimal valid board
│   ├── rectangle.kicad_pcb  # Simple rectangle board
│   ├── complex.kicad_pcb    # Board with cutouts
│   ├── with_fold.kicad_pcb  # Board with fold markers
│   └── real_project.kicad_pcb  # Copy of actual project board
├── test_kicad_parser.py
├── test_markers.py
├── test_geometry.py
├── test_bend_transform.py
├── test_mesh.py
├── test_viewer.py
├── test_ui_controls.py
└── test_fold_placer.py
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install pytest pytest-cov numpy
      - run: pytest tests/ -v --ignore=tests/test_viewer.py  # Skip GUI tests in CI
```

---

## Summary Timeline

| Phase | Components | Testing |
|-------|------------|---------|
| **Phase 1** | Parser, Markers, Geometry | Unit tests + manual file parsing |
| **Phase 2** | Bend Transform, Mesh | Unit tests + visual OBJ inspection |
| **Phase 3** | Viewer, UI Controls | Unit tests + manual viewer testing |
| **Phase 4** | KiCad Actions, Fold Placer | Integration testing in KiCad |
| **Phase 5** | Error Handling, Performance | Stress testing, edge cases |
| **Phase 6** | Documentation, Release | Final review |

Each phase builds on the previous. Complete all unit tests before moving to the next phase.

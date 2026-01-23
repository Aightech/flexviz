# Implementation Checklist

## Phase 1: Core Foundation (No KiCad Required)

This phase can be developed and tested entirely outside of KiCad.

### 1.1 KiCad Parser (`kicad_parser.py`)

- [x] Parse `.kicad_pcb` file structure (S-expression tokenizer)
- [x] Extract board general info (thickness, layers)
- [x] Extract Edge.Cuts layer → board outline polygon
- [x] Extract graphic lines from User.1 layer
- [x] Extract dimension objects from User.1 layer
- [x] Handle multi-polygon boards (cutouts)

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

- [x] Find dotted lines on User.1 layer
- [x] Find dimension objects on User.1 layer
- [x] Group lines into pairs (by proximity/parallelism)
- [x] Associate dimension with line pair
- [x] Parse angle from dimension text (handle +/- signs)
- [x] Calculate bend radius from line spacing
- [x] Build `FoldMarker` data structure

**Unit Tests (`tests/test_markers.py`):**
- [ ] Test dotted line detection
- [ ] Test line pairing algorithm
- [ ] Test dimension association
- [ ] Test angle parsing (positive, negative, with/without degree symbol)
- [ ] Test radius calculation
- [ ] Test multiple fold markers in same file
- [ ] Test invalid/incomplete markers (missing dimension, single line)

### 1.3 Geometry Extraction (`geometry.py`)

- [x] Convert board outline to vertex list
- [x] Extract copper traces (segments with width)
- [x] Extract pads (position, shape, size)
- [x] Extract component positions (reference point, rotation)
- [x] Extract component bounding boxes
- [x] Optional: extract 3D model paths

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
- [x] Parse a real `.kicad_pcb` file from your project
- [x] Verify extracted outline matches KiCad display
- [x] Manually add fold markers in KiCad, verify detection
- [x] Print extracted geometry data, sanity check values

**Deliverable:** Parser can read real KiCad files and extract geometry + markers. ✅

---

## Phase 2: Bend Mathematics

### 2.1 Bend Transform (`bend_transform.py`)

- [x] Define `FoldDefinition` class (position, axis, radius, angle)
- [x] Implement point classification (before/in/after bend zone)
- [x] Implement cylindrical mapping for points in bend zone
- [x] Implement rotation + translation for points after bend
- [x] Handle multiple sequential folds
- [x] Transform line segments (bend both endpoints)
- [x] Transform polygons (bend all vertices, handle subdivision)

**Unit Tests (`tests/test_bend_transform.py`):**
- [ ] Test point classification
- [x] Test 90° bend on simple points (test_arch_board.py)
- [x] Test 45° bend (test_arch_board.py)
- [ ] Test 180° bend (fold flat)
- [ ] Test negative angle (opposite direction)
- [ ] Test point exactly on fold line
- [ ] Test point in middle of bend zone
- [x] Test multiple folds (test_arch_board.py with 3-fold configurations)
- [ ] Test line segment transformation
- [ ] Test polygon transformation
- [ ] Test identity transform (0° bend)

### 2.2 Mesh Generation (`mesh.py`)

- [x] Generate flat board mesh from outline
- [x] Generate bent board mesh using transforms
- [x] Subdivide edges crossing bend zones for smooth curves
- [x] Generate trace meshes (as ribbons with width)
- [x] Generate pad meshes
- [x] Generate component placeholder boxes
- [x] Triangulate polygons with holes (ear clipping algorithm from Eberly's paper)
- [x] Region-based triangulation (split board by fold lines before triangulation)

**Unit Tests (`tests/test_mesh.py`):**
- [ ] Test flat mesh generation (vertex count, face count)
- [ ] Test mesh topology (no holes, correct winding)
- [ ] Test bend zone subdivision
- [ ] Test trace ribbon generation
- [ ] Test pad mesh generation
- [x] Test triangulation with holes (test_triangulation.py)

---

### Testing Checkpoint 2

**Manual Testing:**
- [x] Create test board with single fold, export bent mesh to OBJ
- [x] Open OBJ in Blender/MeshLab, verify geometry looks correct
- [x] Test with 45°, 90°, 135°, 180° bends
- [x] Test with negative angles
- [x] Create board with two folds, verify both apply correctly

**Deliverable:** Bend math produces correct 3D geometry. ✅

---

## Phase 3: 3D Viewer

### 3.1 Viewer Window (`viewer.py`)

- [x] Create vispy window
- [x] Load and display mesh
- [x] Implement camera controls (orbit, pan, zoom)
- [x] Set up lighting
- [x] Color coding (board, traces, pads)
- [x] Wireframe toggle

**Unit Tests (`tests/test_viewer.py`):**
- [ ] Test window creation (may need to skip in CI)
- [ ] Test mesh loading
- [ ] Test color assignment

### 3.2 UI Controls (`ui_controls.py`)

- [x] Create sidebar panel (wxPython or Qt)
- [x] Add angle slider per fold (linked to fold marker)
- [x] Add display checkboxes (outline, traces, pads, components)
- [x] Add refresh button
- [x] Add export button (OBJ/STL)
- [x] Real-time update on slider change

**Unit Tests (`tests/test_ui_controls.py`):**
- [ ] Test slider value binding
- [ ] Test checkbox state management
- [ ] Test callback triggering

### 3.3 Viewer Integration

- [x] Connect parser → geometry → transform → mesh → viewer pipeline
- [x] Implement refresh: re-read file, rebuild mesh
- [x] Implement export: save current mesh to file
- [x] Handle viewer window close gracefully

---

### Testing Checkpoint 3

**Manual Testing:**
- [x] Open viewer with test board
- [x] Adjust angle slider, verify real-time update
- [x] Toggle display options, verify elements show/hide
- [x] Test orbit/pan/zoom controls
- [x] Export to OBJ, open in external tool
- [x] Test with complex board (many traces, pads)

**Deliverable:** Standalone viewer works with test files. ✅

---

## Phase 4: KiCad Integration

### 4.1 Plugin Registration (`__init__.py`)

- [x] Register "Create Fold" action plugin
- [x] Register "Open Viewer" action plugin
- [x] Set up toolbar icons
- [x] Configure plugin metadata (name, description, icon)

**Manual Testing:**
- [x] Plugin appears in Tools → External Plugins
- [x] Toolbar buttons appear
- [x] Clicking buttons triggers actions

### 4.2 Create Fold Action (`action_create_fold.py`)

- [x] Implement action plugin class
- [x] Launch fold placer on button click
- [x] Handle plugin lifecycle

### 4.3 Fold Placer (`fold_placer.py`)

- [x] Enter placement mode on activation
- [x] Capture point A coordinates (dialog-based for reliability)
- [x] Capture point B coordinates (dialog-based for reliability)
- [x] Show angle input dialog after point B
- [x] Create final geometry (2 dotted lines + dimension)
- [ ] Interactive mouse-based placement (future enhancement)
- [ ] Draw temporary marker at point A
- [ ] Capture mouse move, draw preview line
- [ ] Handle ESC key to cancel
- [ ] Handle right-click to cancel
- [ ] Clean up temporary graphics
- [ ] Update status bar messages

**Unit Tests (`tests/test_fold_placer.py`):**
- [ ] Test state machine transitions
- [ ] Test geometry creation logic
- [ ] Test cancellation handling

**Manual Testing:**
- [ ] Click Create Fold, enter two points, verify lines created
- [ ] Verify angle dialog appears and value is used
- [ ] Verify lines are on User.1 layer
- [ ] Verify lines are dotted style
- [ ] Verify dimension shows correct angle

### 4.4 Open Viewer Action (`action_open_viewer.py`)

- [x] Get current board file path
- [x] Launch viewer window with board
- [x] Handle case: no board open
- [x] Handle case: board not saved

**Manual Testing:**
- [x] Open viewer from KiCad with board loaded
- [x] Verify viewer shows correct board
- [x] Test with unsaved board (should prompt or warn)

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

- [x] Board with holes/cutouts (fixed: ear clipping triangulation from Eberly's paper)
- [ ] Circular board outline
- [ ] Fold line at board edge
- [ ] Very small bend radius (< board thickness)
- [ ] Very large angle (> 180°)
- [ ] Overlapping fold zones
- [x] Multi-fold boards with back entry (arch-shaped configurations)

---

### Testing Checkpoint 5

**Stress Testing:**
- [ ] Test with largest board in project
- [x] Test with many folds (3-fold arch board with various angle combinations)
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

| Phase | Components | Testing | Status |
|-------|------------|---------|--------|
| **Phase 1** | Parser, Markers, Geometry | Unit tests + manual file parsing | ✅ Complete |
| **Phase 2** | Bend Transform, Mesh | Unit tests + visual OBJ inspection | ✅ Complete |
| **Phase 3** | Viewer, UI Controls | Unit tests + manual viewer testing | ✅ Complete |
| **Phase 4** | KiCad Actions, Fold Placer | Integration testing in KiCad | 🔄 In Progress |
| **Phase 5** | Error Handling, Performance | Stress testing, edge cases | 🔄 In Progress |
| **Phase 6** | Documentation, Release | Final review | ⏳ Pending |

Each phase builds on the previous. Complete all unit tests before moving to the next phase.

---

## Recent Updates

**2026-01-21**: Added region-based triangulation
- New `region_splitter.py` module splits board into regions along fold axes
- `split_polygon_by_line()` splits polygon along an infinite line using cross product for side detection
- `split_board_into_regions()` divides board by all fold markers, assigns cutouts to regions
- Updated `mesh.py` to triangulate each region separately via `create_board_mesh_with_regions()`
- Test result: 0 triangles crossing fold lines
- Prevents visual artifacts when board is bent in 3D (triangles no longer "jump" between planes)

**2026-01-21**: Implemented Create Fold tool (Phase 4.2 & 4.3)
- Added `fold_placer.py` with interactive fold marker creation
- Dialog-based coordinate input for reliability across KiCad versions
- Creates two parallel dotted lines on User.1 layer
- Creates dimension annotation showing fold angle
- FoldPlacerDialog for angle and radius input
- FoldMarkerCreator class for PCB geometry creation

**2026-01-21**: Fixed board triangulation with cutouts
- Implemented clean ear clipping algorithm from Eberly's "Triangulation by Ear Clipping" paper
- Key functions: `ear_clip_triangulate`, `find_mutually_visible_vertex`, `merge_hole_into_polygon`, `triangulate_with_holes`
- Test result: 0 bad triangles (triangles inside holes)
- Removed Y-coordinate flip hack - algorithm now works correctly with proper winding order enforcement

**2026-01-23**: Multi-fold transformation with back entry handling
- **Problem**: Arch-shaped boards with multiple folds were incorrectly transformed when BFS traversal crosses a fold from the "back" (AFTER side) instead of the "front" (BEFORE side)
- **Solution**: Extended fold recipe format to 3-tuples `(fold, classification, entered_from_back)` to track entry direction
- **Key changes**:
  - `planar_subdivision.py`: `detect_crossed_folds()` now detects back entry when crossing AFTER→IN_ZONE or AFTER→BEFORE
  - `bend_transform.py`: Comprehensive back entry handling:
    - BEFORE classification: negate `perp_dist` to flip perpendicular coordinate
    - IN_ZONE: mirror `local_perp` for cylindrical mapping
    - AFTER: use rotation angle `-effective_angle + π` instead of `effective_angle`
  - `compute_normal()`: Fixed for back entry cases (negate perp_dist/theta for IN_ZONE, negate angle for AFTER)
  - `mesh.py`: Updated `get_region_recipe()` to handle 3-tuple format with backwards compatibility
- **Added documentation**: Detailed transformation math explanation in `bend_transform.py` including:
  - Coordinate system and fold axis positioning
  - Recipe propagation via BFS
  - Cylindrical mapping formulas for IN_ZONE
  - Rotation/translation for AFTER regions
  - Back entry coordinate flipping rationale

**2026-01-23**: Fixed pad/component extrusion on bent surfaces
- **Problem**: Pads and components appeared flat on bent regions instead of following the surface
- **Solution**: Use `transform_point_and_normal()` and offset vertices along surface normal instead of global Z
- **Key changes**:
  - `create_pad_mesh()`: Top/bottom vertices now offset along transformed normal direction
  - `create_component_mesh()`: Box top vertices offset along normal for each base vertex

**2026-01-23**: Fixed viewer angle slider
- **Problem**: Angle slider in KiCad extension had no effect on 3D model
- **Cause**: `on_fold_angle_changed()` updated `self.folds` but mesh generation used `self.fold_markers`
- **Solution**: Update both `self.folds[i].angle` and `self.fold_markers[i].angle_degrees` when slider changes

**2026-01-23**: Enhanced arch board test suite
- Added `tests/test_arch_board.py` with comprehensive multi-fold configurations
- Test configs include 0°, 45°, 90°, 135° angles for left, top, and right folds
- Generates visual test results in `tests/results/` with PNG renders and OBJ exports
- Tests verify surface continuity and correct back entry transformation

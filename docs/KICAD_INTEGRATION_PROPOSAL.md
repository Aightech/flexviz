# KiCad Core Integration Feasibility Study

This document assesses the feasibility of integrating flex PCB bend visualization directly into KiCad's core functionality.

## Executive Summary

Integrating flex PCB visualization into KiCad is technically feasible and would provide significant value to users designing flexible circuits. The main challenges are the C++ port of bend transformation algorithms and integration with KiCad's OpenCASCADE-based 3D viewer.

**Recommendation**: Start with a proposal to the KiCad development team, focusing on the file format extension first, followed by 3D viewer integration.

---

## 1. Current Plugin Architecture

### What the Plugin Does

1. **Marker Detection**: Parses User layer for fold markers (parallel lines + dimension)
2. **Geometry Extraction**: Reads board outline, traces, pads, components from `.kicad_pcb`
3. **Bend Transformation**: Applies cylindrical mapping to transform 2D geometry to 3D
4. **Mesh Generation**: Creates triangulated mesh with proper subdivision at bend zones
5. **Rendering**: Displays in wxPython/OpenGL viewer window
6. **Export**: Outputs to OBJ, STL, STEP formats

### Key Algorithms

- **Dimension-first marker detection**: Finds parallel lines containing dimension anchor points
- **Planar subdivision**: Splits board into regions along fold axes using BFS
- **Cylindrical bend mapping**: Transforms points in bend zone to arc surface
- **Region-based triangulation**: Triangulates each region separately to prevent artifacts
- **Multi-fold transformation**: Handles cumulative transformations with back-entry detection

---

## 2. KiCad Architecture Overview

### Technology Stack

| Component | Technology |
|-----------|------------|
| Language | C++17 |
| UI Framework | wxWidgets 3.x |
| 3D Geometry | OpenCASCADE (OCC/OCE) |
| 3D Rendering | OpenGL via wxGLCanvas |
| Build System | CMake |
| Version Control | GitLab |

### Relevant Modules

- **pcbnew**: PCB editor, contains board data structures
- **3d-viewer**: 3D visualization using OpenCASCADE
- **common**: Shared utilities, geometry primitives
- **libs/kimath**: Math utilities, vectors, matrices

### File Format

`.kicad_pcb` uses S-expression format:
```lisp
(kicad_pcb
  (general ...)
  (layers ...)
  (gr_line ...)      ; Graphic elements
  (dimension ...)    ; Dimension annotations
  (footprint ...)    ; Components
  ...)
```

---

## 3. Integration Points

### 3.1 File Format Extension

**Proposal**: Add native flex zone definition to `.kicad_pcb` schema.

```lisp
(flex_zone
  (axis 0 1)              ; Fold axis direction
  (center 50 0)           ; Fold center position
  (width 5)               ; Bend zone width
  (angle 90)              ; Bend angle in degrees
  (radius 3.18)           ; Calculated bend radius
)
```

**Benefits**:
- Native storage of flex parameters
- No need for marker detection heuristics
- Versioned with board file
- Accessible to other tools (DRC, etc.)

**Implementation Complexity**: Low
- Add new S-expression token handlers
- Extend `BOARD` class with flex zone list
- Update file format version

### 3.2 PCB Editor Integration

**Proposal**: Add "Create Flex Zone" tool to PCB Editor.

**Features**:
- Two-click placement (like dimension tool)
- Property dialog for angle, radius
- Visual feedback showing bend zone
- Integration with DRC for bend radius rules

**Implementation Complexity**: Medium
- New tool class inheriting from `PCB_TOOL_BASE`
- UI dialog for parameters
- Drawing code for zone visualization
- DRC rule integration

### 3.3 3D Viewer Integration

**Proposal**: Extend 3D viewer to apply bend transformations.

**Current 3D Viewer Pipeline**:
```
BOARD → BOARD_ADAPTER → 3D_SHAPES (OCC) → OpenGL Render
```

**Proposed Pipeline**:
```
BOARD → BOARD_ADAPTER → FLEX_TRANSFORM → 3D_SHAPES (OCC) → OpenGL Render
                              ↓
                     Apply cylindrical mapping
                     to board and components
```

**Key Classes to Modify**:
- `BOARD_ADAPTER`: Add flex zone handling
- `BVH_CONTAINER_3D`: Modify bounding volume for bent geometry
- `RENDER_3D_BASE`: Apply transformations during rendering

**Implementation Complexity**: High
- Requires understanding OCC geometry transformations
- Need to handle component models (STEP files)
- Performance considerations for real-time preview

---

## 4. Technical Requirements

### 4.1 C++ Port of Algorithms

**Bend Transform** (bend_transform.py → bend_transform.cpp)
- Point classification (BEFORE/IN_ZONE/AFTER)
- Cylindrical mapping math
- Rotation matrix computation
- ~500 lines of C++

**Mesh Generation** (mesh.py → integrated with OCC)
- Can leverage existing OCC triangulation
- Need to add subdivision at bend boundaries

**Region Splitting** (planar_subdivision.py → region_splitter.cpp)
- Polygon splitting by line
- BFS region traversal
- ~800 lines of C++

### 4.2 OpenCASCADE Integration

OCC provides:
- `BRepBuilderAPI_Transform` for geometry transformation
- `gp_Trsf` for affine transformations
- `gp_Ax1` for rotation axes

**Challenge**: OCC doesn't have built-in cylindrical bend transformation. Options:

1. **Point-wise transformation**: Transform mesh vertices, rebuild topology
   - Simpler but loses BREP precision

2. **Surface deformation**: Use OCC's `BRepOffsetAPI_MakePipeShell`
   - More complex but preserves solid geometry

3. **Hybrid approach**: Transform as mesh for display, solid for export

### 4.3 Performance Considerations

| Operation | Current (Python) | Target (C++) |
|-----------|------------------|--------------|
| Marker detection | ~50ms | ~5ms |
| Mesh generation | ~200ms | ~20ms |
| Bend transform | ~100ms | ~10ms |
| Full refresh | ~500ms | ~50ms |

C++ implementation should achieve 10x performance improvement.

---

## 5. Development Roadmap

### Phase 1: File Format (2-3 weeks)
- [ ] Draft S-expression schema for flex zones
- [ ] Implement parser/writer in pcbnew
- [ ] Add flex zone to BOARD class
- [ ] Unit tests for serialization

### Phase 2: PCB Editor Tool (3-4 weeks)
- [ ] Create FLEX_ZONE_TOOL class
- [ ] Implement two-click placement
- [ ] Add property dialog
- [ ] Zone visualization on board canvas
- [ ] Integration with undo/redo

### Phase 3: 3D Viewer - Basic (4-6 weeks)
- [ ] Port bend transformation to C++
- [ ] Integrate with BOARD_ADAPTER
- [ ] Transform board outline
- [ ] Handle multiple flex zones
- [ ] Basic rendering

### Phase 4: 3D Viewer - Complete (4-6 weeks)
- [ ] Transform copper layers
- [ ] Transform component models
- [ ] Performance optimization
- [ ] Real-time angle adjustment
- [ ] Export to STEP

### Phase 5: DRC Integration (2-3 weeks)
- [ ] Minimum bend radius rule
- [ ] Component in flex zone warning
- [ ] Stiffener conflict detection

---

## 6. Risks and Mitigations

### Risk: OCC Complexity
**Impact**: High
**Mitigation**: Start with mesh-based transformation, upgrade to solid geometry later.

### Risk: Performance on Large Boards
**Impact**: Medium
**Mitigation**: Implement LOD (level of detail), cache transformed geometry.

### Risk: Component Model Handling
**Impact**: Medium
**Mitigation**: Initially skip 3D models in flex zones, add support incrementally.

### Risk: KiCad Release Cycle
**Impact**: Low
**Mitigation**: Target feature branch, plan for multi-release integration.

---

## 7. Community Engagement

### Steps

1. **Forum Discussion**: Post proposal on KiCad developers forum
2. **GitLab Issue**: Create feature request with this document
3. **Proof of Concept**: Submit minimal C++ implementation of bend transform
4. **Code Review**: Work with maintainers on architecture decisions
5. **Incremental PRs**: Submit small, reviewable changes

### Key Contacts

- KiCad Lead Developers (via GitLab)
- 3D Viewer Maintainers
- PCBnew Module Owners

---

## 8. Alternative Approaches

### A. Improved Plugin Architecture

Instead of core integration, enhance the plugin:
- Better KiCad API integration
- Persistent settings in board file (custom fields)
- Tighter UI integration via wxPython

**Pros**: No C++ required, faster development
**Cons**: Limited integration, separate window

### B. External Tool Integration

Create standalone flex visualization tool:
- Read KiCad files directly
- Web-based or Qt interface
- IPC communication with KiCad

**Pros**: Independent development, cross-platform
**Cons**: Not integrated, requires external launch

### C. FreeCAD Workbench

Leverage FreeCAD's CAD capabilities:
- Import KiCad boards
- Use FreeCAD's bend/flex tools
- Export back to KiCad

**Pros**: Powerful CAD features
**Cons**: Separate tool, complex workflow

---

## 9. Conclusion

Core integration into KiCad is the best long-term solution for flex PCB visualization. The main technical challenge is the 3D viewer integration with OpenCASCADE.

**Recommended next steps**:
1. Engage with KiCad community via forum/GitLab
2. Prototype C++ bend transformation
3. Submit file format proposal first (lowest risk)
4. Incrementally add 3D viewer support

The existing Python plugin provides a working reference implementation that can guide the C++ port.

---

## Appendix: Code Mapping

| Python Module | C++ Target | Notes |
|---------------|------------|-------|
| kicad_parser.py | pcbnew/io_kicad | Already exists, extend for flex zones |
| markers.py | pcbnew/flex_zone.cpp | New file |
| geometry.py | pcbnew/board_adapter.cpp | Extend existing |
| bend_transform.py | common/flex_transform.cpp | New file, ~500 LOC |
| mesh.py | 3d-viewer/occ_triangulator.cpp | Extend existing |
| viewer.py | 3d-viewer/render_3d_flex.cpp | New renderer mode |
| validation.py | pcbnew/drc_flex.cpp | New DRC rules |

---

*Document Version: 1.0*
*Date: January 2026*

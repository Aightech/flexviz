# Implementation Plan: Thickness & Stiffener Support

## Overview

Add realistic thickness handling using the neutral axis approach, plus stiffener region support with fold validation.

**Also includes:** Extraction of PlanarSubdivision algorithm from `test_triangulation.py` into production code.

---

## Phase 0: Region Segmentation (PlanarSubdivision)

### Current State

The `PlanarSubdivision` class in `test_triangulation.py` implements a robust algorithm for partitioning a board into regions along cutting lines (fold axes). This is documented in `docs/bend_zone_triangulation.md`.

**Key features:**
- Treats all edges (outer, holes, cutting lines) as a planar graph
- Traces region boundaries using "next clockwise edge" rule
- Properly handles holes that:
  - Are entirely within a region
  - Cross one cutting line
  - Span multiple cutting lines (creates split regions)

### Task: Extract to Production Module

**File: `planar_subdivision.py` (new)**

Extract from `test_triangulation.py`:
- [ ] `PlanarSubdivision` class
- [ ] Helper functions:
  - `segment_line_intersection()`
  - `signed_distance_to_line()`
  - `points_equal()`
  - `filter_valid_board_regions()`
  - `associate_holes_with_regions()`
  - `hole_crosses_cutting_lines()`
  - `get_interior_test_point()`
  - `region_centroid()`
- [ ] Add type hints and docstrings
- [ ] Add unit tests in `tests/test_planar_subdivision.py`

**Update: `mesh.py`**

- [ ] Replace `split_board_into_regions()` usage with `PlanarSubdivision`
- [ ] Update `create_board_mesh_with_regions()` to use new module
- [ ] Convert FoldMarkers to cutting lines format

**Interface:**

```python
from planar_subdivision import PlanarSubdivision, filter_valid_board_regions, associate_holes_with_regions

def create_cutting_lines_from_markers(markers: list[FoldMarker], bbox) -> list[tuple]:
    """Convert fold markers to cutting line format (line_eq, p1, p2)."""
    lines = []
    for marker in markers:
        # Line equation: ax + by + c = 0
        # For line through point P with direction D: -Dy*x + Dx*y + (Dy*Px - Dx*Py) = 0
        px, py = marker.center
        dx, dy = marker.axis
        a, b, c = -dy, dx, dy*px - dx*py

        # Extend line to bbox
        p1, p2 = extend_line_to_bbox((px, py), (dx, dy), bbox)
        lines.append(((a, b, c), p1, p2))
    return lines

# Usage:
subdivision = PlanarSubdivision(outer, holes, cutting_lines)
all_regions = subdivision.compute()
valid_regions = filter_valid_board_regions(all_regions, outer, holes)
regions_with_holes = associate_holes_with_regions(valid_regions, holes, cutting_lines)

for region, region_holes in regions_with_holes:
    triangles, merged = triangulate_with_holes(region, region_holes)
```

### Test Cases (from docs/bend_zone_triangulation.md)

| Case | Description | Expected |
|------|-------------|----------|
| 1 | Hole entirely within region | 3 regions, hole associated with middle |
| 2 | Hole crossing one cutting line | Hole boundary incorporated into 2 regions |
| 3 | Hole spanning both cutting lines | 4 regions (middle split left/right) |
| 4 | Complex non-rectangular board | Handles concave outer + multiple holes |

### Bend Zone Subdivision for Smooth Curves

**Problem:** A single fold creates 3 regions (before, bend zone, after). When bent, the middle "bend zone" region is a single flat strip that gets cylindrically mapped. For smooth rendering, this needs subdivision into multiple thin strips.

**Solution:** Generate additional cutting lines within the bend zone.

```
Before subdivision:         After subdivision (n=4):

┌───────────────┐           ┌───────────────┐
│   Region 1    │           │   Region 1    │
├───────────────┤ ← fold    ├───────────────┤
│               │   line    │  bend strip 1 │
│   Bend Zone   │   pair    │  bend strip 2 │
│               │           │  bend strip 3 │
├───────────────┤           │  bend strip 4 │
│   Region 2    │           ├───────────────┤
└───────────────┘           │   Region 2    │
                            └───────────────┘
```

**Add to test_triangulation.py:**

```python
def test_bend_zone_subdivision():
    """Test subdividing the bend zone into multiple thin strips for smooth bending."""
    # Board: 100x80 rectangle
    outer = [(0, 0), (100, 0), (100, 80), (0, 80)]
    holes = []

    # Fold zone between y=30 and y=50 (width=20)
    # Subdivide into 4 strips (width=5 each)
    num_subdivisions = 4
    y_start, y_end = 30, 50
    step = (y_end - y_start) / num_subdivisions

    cutting_lines = []
    for i in range(num_subdivisions + 1):
        y = y_start + i * step
        line_eq = (0, 1, -y)  # y = const
        p1, p2 = (-10, y), (110, y)
        cutting_lines.append((line_eq, p1, p2))

    subdivision = PlanarSubdivision(outer, holes, cutting_lines)
    regions = subdivision.compute()
    valid = filter_valid_board_regions(regions, outer, holes)

    # Should have: 1 (before) + 4 (bend strips) + 1 (after) = 6 regions
    assert len(valid) == 6

    # Verify strip heights
    for region in valid:
        # Each bend strip should have height ≈ 5
        # Before/after strips have different heights
        pass
```

**Integration with FoldMarker:**

```python
def create_bend_zone_cutting_lines(marker: FoldMarker, num_subdivisions: int = 8) -> list:
    """
    Create cutting lines for a bend zone with subdivisions.

    The bend zone spans from -zone_width/2 to +zone_width/2 perpendicular
    to the fold axis. This creates num_subdivisions + 2 cutting lines
    (boundaries plus internal divisions).
    """
    lines = []
    half_width = marker.zone_width / 2

    # Generate lines perpendicular to fold axis
    for i in range(num_subdivisions + 1):
        t = -half_width + (i / num_subdivisions) * marker.zone_width
        # Offset from center along perpendicular direction
        perp = (-marker.axis[1], marker.axis[0])
        point = (
            marker.center[0] + t * perp[0],
            marker.center[1] + t * perp[1]
        )
        # Line through point, parallel to axis
        lines.append(create_line_through_point(point, marker.axis))

    return lines
```

---

## Geometry Concepts

### Neutral Axis Bending

```
        Outer surface (stretched)
    ════════════════════════════════
    ─ ─ ─ ─ ─ Neutral axis ─ ─ ─ ─ ─   ← No stretch/compression
    ════════════════════════════════
        Inner surface (compressed)

    |←───── thickness t ─────→|
```

- Current mesh represents the **neutral axis** (middle of PCB thickness)
- Extrude ±t/2 perpendicular to local surface normal
- At bends, this automatically creates correct arc length differences

### Stiffener Regions

```
    ┌─────────────────────────────────┐
    │  Flex region      │ Stiffener  │
    │  (can bend)       │ (rigid)    │
    │                   │████████████│ ← Extra thickness on one side
    └─────────────────────────────────┘
```

- Stiffeners are rigid areas that cannot bend
- Typically FR4 or aluminum bonded to flex
- Must validate no fold lines cross stiffener regions

---

## Phase A: Configuration & Settings

### A.1 New Configuration Data Structure

**File: `config.py` (new)**

```python
@dataclass
class FlexConfig:
    # Flex PCB parameters
    flex_thickness: float = 0.11  # mm (typical: 0.11-0.2mm)

    # Stiffener parameters
    stiffener_layer: str = "User.2"  # KiCad layer containing stiffener outlines
    stiffener_thickness: float = 0.0  # mm (0 = no stiffener)
    stiffener_side: str = "bottom"  # "top" or "bottom"

    # Visualization
    show_thickness: bool = True  # 3D extrusion vs flat surface

    # Validation
    min_bend_radius_factor: float = 6.0  # Min bend radius = factor × thickness
```

### A.2 Settings UI

**Update: `viewer.py` - Add settings panel**

- [ ] Add "PCB Settings" section to sidebar
- [ ] Flex thickness input (spinner, 0.05-0.5mm range, 0.01mm step)
- [ ] Stiffener layer dropdown (populated from available layers)
- [ ] Stiffener thickness input (spinner, 0-2mm range)
- [ ] Stiffener side radio buttons (Top/Bottom)
- [ ] "Show thickness" checkbox
- [ ] Settings persistence (save/load from file or PCB metadata)

### A.3 Parser Updates

**Update: `kicad_parser.py`**

- [ ] Add method to list available User layers
- [ ] Add method to extract polygons from arbitrary layer
- [ ] Return layer names in a consistent format

---

## Phase B: Stiffener Detection & Validation

### B.1 Stiffener Region Extraction

**File: `stiffener.py` (new)**

```python
@dataclass
class StiffenerRegion:
    outline: list[tuple[float, float]]  # Polygon vertices
    layer: str  # Source layer name
    thickness: float  # mm
    side: str  # "top" or "bottom"

def extract_stiffeners(pcb: KiCadPCB, config: FlexConfig) -> list[StiffenerRegion]:
    """Extract stiffener polygons from configured layer."""
    pass

def point_in_stiffener(point: tuple, stiffeners: list[StiffenerRegion]) -> bool:
    """Check if a point is inside any stiffener region."""
    pass

def line_intersects_stiffener(p1: tuple, p2: tuple, stiffeners: list[StiffenerRegion]) -> bool:
    """Check if a line segment crosses any stiffener region."""
    pass
```

### B.2 Fold Validation

**File: `validation.py` (new)**

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]  # Critical issues (fold in stiffener)
    warnings: list[str]  # Non-critical issues (tight bend radius)

def validate_folds(
    markers: list[FoldMarker],
    stiffeners: list[StiffenerRegion],
    config: FlexConfig
) -> ValidationResult:
    """
    Validate all fold markers against constraints.

    Checks:
    1. Fold line doesn't intersect any stiffener
    2. Bend radius >= min_bend_radius_factor × thickness
    3. Fold zones don't overlap
    4. Fold line is within board outline
    """
    pass

def validate_single_fold(
    marker: FoldMarker,
    stiffeners: list[StiffenerRegion],
    config: FlexConfig
) -> tuple[bool, list[str]]:
    """Validate a single fold marker."""
    pass
```

### B.3 Validation Integration

**Update: `viewer.py`**

- [ ] Run validation when loading board
- [ ] Run validation when settings change
- [ ] Display validation errors/warnings in UI
- [ ] Highlight invalid fold lines in red
- [ ] Option to ignore warnings and proceed

**Update: `fold_placer.py`**

- [ ] Validate fold position before creating geometry
- [ ] Warn user if placing fold in stiffener region
- [ ] Show stiffener regions as overlay during placement (future)

---

## Phase C: 3D Mesh with Thickness

### C.1 Surface Normal Calculation

**Update: `mesh.py`**

```python
def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex normals by averaging adjacent face normals.

    For flat regions: normal = (0, 0, 1) or (0, 0, -1)
    For bent regions: normal points radially outward from bend axis
    """
    pass

def compute_face_normal(v0, v1, v2) -> np.ndarray:
    """Compute normal of a triangle face."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    return normal / np.linalg.norm(normal)
```

### C.2 Thickness Extrusion

**Update: `mesh.py`**

```python
def extrude_mesh_with_thickness(
    neutral_vertices: np.ndarray,
    neutral_faces: np.ndarray,
    normals: np.ndarray,
    thickness: float,
    stiffener_mask: np.ndarray = None,  # Per-vertex: is this in a stiffener?
    stiffener_thickness: float = 0.0,
    stiffener_side: str = "bottom"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrude a neutral-axis mesh to create a solid with thickness.

    Returns:
        vertices: Combined top + bottom + edge vertices
        faces: All faces (top, bottom, edges)

    Algorithm:
    1. Create top surface: neutral_vertices + normals * (t/2)
    2. Create bottom surface: neutral_vertices - normals * (t/2)
    3. For stiffener vertices, add extra thickness on configured side
    4. Create edge faces connecting top and bottom at boundaries
    5. Create edge faces at holes/cutouts
    """
    pass

def create_edge_faces(
    top_boundary: list[int],  # Vertex indices forming boundary loop
    bottom_boundary: list[int],
    top_vertices: np.ndarray,
    bottom_vertices: np.ndarray
) -> np.ndarray:
    """Create quad faces (as triangles) connecting top and bottom boundaries."""
    pass
```

### C.3 Stiffener Thickness Handling

**Update: `mesh.py`**

```python
def compute_stiffener_mask(
    vertices: np.ndarray,
    stiffeners: list[StiffenerRegion]
) -> np.ndarray:
    """
    Create boolean mask indicating which vertices are in stiffener regions.

    Returns:
        mask: np.ndarray of bool, shape (n_vertices,)
    """
    pass

def apply_stiffener_offset(
    vertices: np.ndarray,
    normals: np.ndarray,
    mask: np.ndarray,
    stiffener_thickness: float,
    side: str  # "top" or "bottom"
) -> np.ndarray:
    """
    Offset vertices in stiffener regions by additional thickness.

    If side="bottom": bottom surface moves down by stiffener_thickness
    If side="top": top surface moves up by stiffener_thickness
    """
    pass
```

### C.4 Mesh Pipeline Update

**Update: `mesh.py` - `create_board_mesh_with_regions()`**

Current pipeline:
```
regions → triangulate each → transform (bend) → combine → flat mesh
```

New pipeline:
```
regions → triangulate each → transform (bend) → combine
    → compute normals → compute stiffener mask
    → extrude with thickness → solid mesh
```

---

## Phase D: Visualization Updates

### D.1 Material/Color Differentiation

**Update: `viewer.py`**

- [ ] Different color for stiffener regions (e.g., darker green or brown)
- [ ] Option to show top/bottom surfaces in different colors
- [ ] Wireframe mode shows both surfaces

### D.2 Cross-Section View (Optional/Future)

- [ ] Slice plane to show internal structure
- [ ] Useful for verifying thickness and bend geometry

### D.3 Stiffener Overlay in 2D

**Update: `viewer.py`**

- [ ] When viewing flat (all angles = 0), show stiffener regions as overlay
- [ ] Helps user understand which areas are rigid

---

## Phase E: Edge Cases & Validation

### E.1 Critical Validations

| Check | Action if Failed |
|-------|------------------|
| Fold line crosses stiffener | Error: Block fold creation |
| Bend radius < min allowed | Warning: Show but highlight |
| Stiffener in bend zone | Error: Geometry would be invalid |
| Self-intersection after bend | Warning: Show potential collision |

### E.2 Geometric Edge Cases

- [ ] Board edge at fold line (edge faces need special handling)
- [ ] Hole/cutout crossing fold line (already handled by region splitting)
- [ ] Stiffener touching board edge
- [ ] Multiple stiffeners
- [ ] Stiffener with holes/cutouts

### E.3 Numerical Stability

- [ ] Very thin flex (< 0.05mm): may cause z-fighting in OpenGL
- [ ] Very thick stiffener: ensure mesh doesn't self-intersect
- [ ] Sharp bends: normal interpolation at bend transitions

---

## Implementation Order

### Sprint 0: Region Segmentation (1-2 days)
1. [ ] **Add bend zone subdivision test** to `test_triangulation.py`:
   - Test subdividing middle strip into N thin bands
   - Verify correct number of regions created
   - Verify holes handled correctly with multiple cutting lines
2. [ ] Extract `PlanarSubdivision` class to `planar_subdivision.py`
3. [ ] Extract helper functions (intersection, filtering, hole association)
4. [ ] Add `create_bend_zone_cutting_lines()` function
5. [ ] Add type hints and comprehensive docstrings
6. [ ] Create `tests/test_planar_subdivision.py` with test cases from docs
7. [ ] Update `mesh.py` to use `PlanarSubdivision` instead of `region_splitter.py`
8. [ ] Verify all existing triangulation tests pass

### Sprint 1: Configuration & Basic Stiffener (1-2 days)
7. [ ] Create `config.py` with FlexConfig dataclass
8. [ ] Update `kicad_parser.py` to extract polygons from any layer
9. [ ] Create `stiffener.py` with extraction functions
10. [ ] Add basic settings UI to viewer

### Sprint 2: Validation (1 day)
11. [ ] Create `validation.py` with fold validation
12. [ ] Integrate validation into viewer (show errors/warnings)
13. [ ] Add validation to fold placer

### Sprint 3: Thickness Extrusion (2-3 days)
14. [ ] Implement `compute_vertex_normals()`
15. [ ] Implement `extrude_mesh_with_thickness()` for uniform thickness
16. [ ] Test with simple rectangular board
17. [ ] Add stiffener mask computation
18. [ ] Implement variable thickness for stiffener regions
19. [ ] Create edge faces for board boundary and holes

### Sprint 4: Polish & Testing (1-2 days)
20. [ ] Add color differentiation for stiffeners
21. [ ] Test with real flex PCB designs
22. [ ] Handle edge cases
23. [ ] Update documentation

---

## Testing Checklist

### Unit Tests

- [ ] `test_config.py`: Config serialization, validation
- [ ] `test_stiffener.py`: Polygon extraction, point-in-stiffener
- [ ] `test_validation.py`: Fold validation logic
- [ ] `test_mesh_thickness.py`: Normal computation, extrusion

### Visual Tests

- [ ] Flat board with thickness renders correctly
- [ ] Single bend shows correct arc on both surfaces
- [ ] Stiffener region shows extra thickness
- [ ] Board with holes has closed edges
- [ ] Multiple bends work correctly
- [ ] Stiffener on bent region shows error

### Integration Tests

- [ ] Load real flex PCB with stiffener layer
- [ ] Validate fold placement in KiCad
- [ ] Export STL/OBJ with thickness, verify in external tool

---

## File Summary

| File | Status | Description |
|------|--------|-------------|
| `planar_subdivision.py` | New | PlanarSubdivision class, region tracing |
| `config.py` | New | FlexConfig dataclass, settings |
| `stiffener.py` | New | Stiffener extraction and queries |
| `validation.py` | New | Fold validation logic |
| `kicad_parser.py` | Update | Add layer polygon extraction |
| `mesh.py` | Update | Use PlanarSubdivision, add thickness extrusion |
| `viewer.py` | Update | Settings UI, validation display |
| `fold_placer.py` | Update | Validation during placement |
| `region_splitter.py` | Deprecate | Replace with planar_subdivision.py |
| `test_triangulation.py` | Update | Add bend zone subdivision tests |
| `tests/test_planar_subdivision.py` | New | Unit tests for region segmentation |

---

## Open Questions

1. **Stiffener layer convention**: Should we support multiple stiffener layers? (e.g., top stiffener on User.2, bottom on User.3)

2. **Neutral axis position**: For multi-layer flex with asymmetric stackup, neutral axis isn't at center. Do we need to support offset?

3. **Adhesive thickness**: Stiffeners are bonded with adhesive (typically 0.05mm). Include in total thickness calculation?

4. **Visualization LOD**: For performance, should we skip thickness extrusion when zoomed out?

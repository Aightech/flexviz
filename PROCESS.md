# Flex PCB 3D Visualization Process

## Overview

The process transforms a flat 2D PCB into a 3D bent visualization.

```
.kicad_pcb → Parse → Detect Markers → Split Regions → Transform to 3D → Render
```

## Step-by-Step Process

### 1. Parse PCB File
**File:** `kicad_parser.py`
- `KiCadPCB.from_file()` - Parse .kicad_pcb file into S-expression tree

### 2. Extract Board Geometry
**File:** `geometry.py`
- `extract_board_geometry()` - Extract outline, traces, pads, components from parsed PCB
- Returns: `BoardGeometry` dataclass

### 3. Detect Fold Markers
**File:** `markers.py`
- `detect_fold_markers()` - Find fold markers on User.1 layer (parallel dotted lines + dimension)
- Returns: List of `FoldMarker` objects with center, axis, zone_width, angle

### 4. Split Board into Regions
**File:** `planar_subdivision.py`
- `split_board_into_regions()` - Split board outline along fold lines
- `compute_fold_recipes()` - Assign fold recipe to each region (BFS from anchor)
- Returns: List of `Region` objects, each with outline and fold_recipe

### 5. Generate 3D Mesh
**File:** `mesh.py`
- `create_board_geometry_mesh()` - Main entry point
- `create_board_mesh_with_regions()` - Create mesh for board with fold regions
- For each region:
  - `get_region_recipe()` - Get fold recipe for region
  - `transform_vertices_with_thickness()` - Transform 2D vertices to 3D
  - `triangulate_polygon()` - Triangulate region
  - Add top/bottom/side faces

### 6. Transform Points to 3D
**File:** `bend_transform.py`
- `transform_point()` - Transform 2D point using fold recipe
- `compute_normal()` - Compute surface normal at point
- `transform_point_and_normal()` - Both in one call

### 7. Render
**File:** `viewer.py`
- `FlexViewerFrame` - Main viewer window
- `GLCanvas` - OpenGL rendering canvas
- `update_mesh()` - Regenerate mesh when settings change

---

## Required Functions by File

### bend_transform.py
| Function | Used | Purpose |
|----------|------|---------|
| `FoldDefinition` | YES | Dataclass for fold parameters |
| `FoldDefinition.from_marker()` | YES | Create from FoldMarker |
| `transform_point()` | YES | Core 2D→3D transformation |
| `compute_normal()` | YES | Compute surface normal |
| `transform_point_and_normal()` | YES | Convenience wrapper |
| `create_fold_definitions()` | YES | Create list from markers |

### planar_subdivision.py
| Function | Used | Purpose |
|----------|------|---------|
| `Region` | YES | Dataclass for board region |
| `split_board_into_regions()` | YES | Main entry point |
| `create_cutting_lines_from_marker_segments()` | YES | Create finite cut lines |
| `compute_fold_recipes()` | YES | Assign recipes via BFS |
| `find_anchor_region()` | YES | Find BEFORE region for BFS start |
| `build_region_adjacency()` | YES | Build region neighbor graph |
| `detect_crossed_folds()` | YES | Detect folds between regions |
| `classify_point_vs_fold()` | YES | BEFORE/IN_ZONE/AFTER |
| `find_containing_region()` | YES | Find region containing point |

### mesh.py
| Function | Used | Purpose |
|----------|------|---------|
| `Mesh` | YES | Mesh data structure |
| `create_board_geometry_mesh()` | YES | Main entry point |
| `create_board_mesh_with_regions()` | YES | Board mesh with regions |
| `get_region_recipe()` | YES | Get recipe for region |
| `transform_vertex()` | YES | Transform single vertex |
| `transform_vertices_with_thickness()` | YES | Transform vertex list |
| `triangulate_polygon()` | YES | Ear-clipping triangulation |
| `triangulate_with_holes()` | YES | Triangulate with cutouts |
| `create_trace_mesh()` | YES | Mesh for copper traces |
| `create_pad_mesh()` | YES | Mesh for pads |
| `create_component_mesh()` | YES | Mesh for components |
| `create_stiffener_mesh()` | YES | Mesh for stiffeners |
| `create_cutout_mesh()` | NO | Dead code - DELETE |
| `create_board_mesh()` | MAYBE | Fallback without regions |

### geometry.py
| Function | Used | Purpose |
|----------|------|---------|
| `BoardGeometry` | YES | Main geometry container |
| `Polygon` | YES | Polygon data structure |
| `extract_board_geometry()` | YES | Extract from parsed PCB |
| `subdivide_polygon()` | YES | Subdivide edges for smooth bends |

### markers.py
| Function | Used | Purpose |
|----------|------|---------|
| `FoldMarker` | YES | Fold marker data |
| `detect_fold_markers()` | YES | Find markers on User.1 |

---

## Functions to Review/Delete

### Definitely Unused
- `mesh.py:create_cutout_mesh()` - Never called

### Potentially Redundant
- Complex hole processing in `create_board_mesh_with_regions()` (lines 1030-1102)
- Multiple triangulation paths that could be unified

---

## Data Flow

```
FoldMarker (from markers.py)
    ↓
FoldDefinition (bend_transform.py) ← created via from_marker()
    ↓
Region.fold_recipe = [(FoldMarker, "IN_ZONE"|"AFTER"), ...]
    ↓
get_region_recipe() → [(FoldDefinition, "IN_ZONE"|"AFTER"), ...]
    ↓
transform_point(point_2d, recipe) → point_3d
```

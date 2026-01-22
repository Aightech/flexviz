# Plan: Region-Based Fold Transformation

## Problem Statement

For non-parallel folds on different "branches" of a board (e.g., U-shape with perpendicular folds), the current linear fold ordering doesn't work. We need region-based transformation where each region knows exactly which folds affect it and in what order.

## Example Scenario

```
    A─────B─────C
    │           │
    │    ═══════│═══  ← Fold 1 (horizontal)
    │           │
    D───E       F
        │       │
        │   ════│════  ← Fold 2 (vertical, right arm only)
        │       │
        G       H
```

Regions and their fold recipes:
- **R_ABC**: BEFORE F1, BEFORE F2 → no transformation
- **R_DE**: AFTER F1, BEFORE F2 → apply F1 only
- **R_EF**: AFTER F1, BEFORE F2 → apply F1 only
- **R_G**: AFTER F1, BEFORE F2 → apply F1 only (left arm)
- **R_H**: AFTER F1, AFTER F2 → apply F1, then F2 (right arm)

Key insight: F2 only affects regions on the right arm, not the left arm.

## Architecture Changes

### 1. Extend Region Dataclass

```python
@dataclass
class Region:
    outline: Polygon
    holes: list[Polygon]
    index: int

    # NEW: Ordered list of (fold, classification) pairs
    # Classification is "IN_ZONE" or "AFTER"
    fold_recipe: list[tuple[FoldMarker, str]] = field(default_factory=list)

    # For debugging/visualization
    representative_point: Optional[Point] = None
```

### 2. Build Region Connectivity Graph

After planar subdivision, build a graph where:
- Nodes = regions
- Edges = regions that share a boundary

```python
def build_region_adjacency(regions: list[Region]) -> dict[int, list[int]]:
    """Build adjacency list: region_index -> [neighbor_indices]"""
    ...
```

### 3. Compute Fold Recipes via BFS

Starting from an "anchor" region (BEFORE all folds), traverse the graph:

```python
def compute_fold_recipes(
    regions: list[Region],
    adjacency: dict[int, list[int]],
    folds: list[FoldMarker]
) -> None:
    """
    Compute fold_recipe for each region using BFS from anchor.

    When crossing from region A to region B:
    - Check which fold zones the shared boundary crosses
    - If crossing fold F's zone: B's recipe = A's recipe + [(F, classification)]
    """

    # 1. Find anchor region (BEFORE all folds)
    anchor = find_anchor_region(regions, folds)

    # 2. BFS from anchor
    queue = [anchor]
    visited = {anchor.index}
    anchor.fold_recipe = []

    while queue:
        current = queue.pop(0)
        for neighbor_idx in adjacency[current.index]:
            if neighbor_idx in visited:
                continue

            neighbor = regions[neighbor_idx]

            # Determine what fold(s) we cross to reach neighbor
            crossed_folds = detect_crossed_folds(current, neighbor, folds)

            # Neighbor's recipe = current's recipe + crossed folds
            neighbor.fold_recipe = current.fold_recipe.copy()
            for fold, classification in crossed_folds:
                neighbor.fold_recipe.append((fold, classification))

            visited.add(neighbor_idx)
            queue.append(neighbor)
```

### 4. Detect Fold Crossings Between Adjacent Regions

```python
def detect_crossed_folds(
    region_a: Region,
    region_b: Region,
    folds: list[FoldMarker]
) -> list[tuple[FoldMarker, str]]:
    """
    Determine which folds are crossed when moving from region_a to region_b.

    Uses the shared boundary between regions and checks if it crosses
    any fold zone.
    """
    # Get shared boundary edge(s)
    shared_edge = find_shared_boundary(region_a, region_b)

    crossed = []
    for fold in folds:
        # Check if shared edge crosses this fold's zone
        if edge_crosses_fold_zone(shared_edge, fold):
            # Determine if B is IN_ZONE or AFTER
            classification = classify_region_vs_fold(region_b, fold)
            if classification != "BEFORE":
                crossed.append((fold, classification))

    return crossed
```

### 5. Modify transform_point to Use Region Info

```python
def transform_point_with_regions(
    point: tuple[float, float],
    regions: list[Region],
    folds: list[FoldDefinition]
) -> tuple[float, float, float]:
    """
    Transform a point using region-based fold recipes.

    1. Find which region contains the point
    2. Use that region's fold_recipe for transformation
    """
    # Find containing region
    region = find_containing_region(point, regions)
    if region is None:
        return (point[0], point[1], 0.0)

    # Get fold definitions for this region's recipe
    applicable_folds = []
    for fold_marker, classification in region.fold_recipe:
        fold_def = FoldDefinition.from_marker(fold_marker)
        applicable_folds.append(fold_def)

    # Transform using only applicable folds (in order)
    return transform_point(point, applicable_folds)
```

## Implementation Steps

### Step 1: Add region adjacency computation
- Extend `planar_subdivision.py` with `build_region_adjacency()`
- Detect shared edges between regions

### Step 2: Add fold recipe computation
- Add `compute_fold_recipes()` function
- Implement BFS traversal from anchor region
- Detect fold zone crossings between adjacent regions

### Step 3: Extend Region dataclass
- Add `fold_recipe` field
- Add `representative_point` field for debugging

### Step 4: Create region-aware transform function
- Add `find_containing_region()` function
- Add `transform_point_with_regions()` function

### Step 5: Integrate with mesh generation
- Modify `mesh.py` to use region-based transformation
- Pass region info through the pipeline

### Step 6: Test with perpendicular folds
- Create test case with U-shaped board
- Verify each region gets correct fold recipe
- Verify transformation continuity at boundaries

## Edge Cases to Handle

1. **Multiple folds crossed at once**: If moving from A to B crosses both F1 and F2 zones, determine order based on which zone is entered first

2. **IN_ZONE regions**: Regions entirely within a fold zone need special handling

3. **Disconnected regions**: If board has holes that separate regions, ensure BFS can still reach all regions

4. **Fold zones that don't reach all branches**: A fold on one arm shouldn't affect other arms

## Files to Modify

1. `planar_subdivision.py`: Add adjacency and recipe computation
2. `bend_transform.py`: Add region-aware transformation
3. `mesh.py`: Use region info for transformation
4. `viewer.py`: Pass region info through pipeline

## Testing Strategy

1. Unit tests for adjacency detection
2. Unit tests for fold recipe computation
3. Visual test with U-shaped board and perpendicular folds
4. Verify no discontinuities at region boundaries

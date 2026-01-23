"""
Test suite for arch-shaped board with fold combinations.

Tests the region-based fold recipe system where folds only affect
regions they physically touch (via connectivity graph traversal).

Board geometry:
- 100x100 bounding box
- Left leg: x=0-20, y=0-80
- Top: x=0-100, y=80-100
- Right leg: x=80-100, y=0-80
- Inner cutout: x=20-80, y=0-80

Folds (10 wide each):
- Left fold: horizontal at y=40, spans x=0-20 (left leg only)
- Top fold: vertical at x=50, spans y=80-100 (top only)
- Right fold: horizontal at y=40, spans x=80-100 (right leg only)
"""

import math
import sys
import os
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from planar_subdivision import (
    split_board_into_regions,
    find_containing_region,
    signed_area,
    Region,
    build_region_adjacency,
)
from bend_transform import FoldDefinition, transform_point


# =============================================================================
# Setup
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Mock FoldMarker with finite extent
# =============================================================================

@dataclass
class MockFoldMarker:
    """
    Mock FoldMarker with finite line extents for testing.

    The key difference from FoldDefinition: includes line_a/b endpoints
    that define where the fold physically exists on the board.
    """
    # Line A (start of bend zone) - defines spatial extent
    line_a_start: tuple[float, float]
    line_a_end: tuple[float, float]

    # Line B (end of bend zone)
    line_b_start: tuple[float, float]
    line_b_end: tuple[float, float]

    # Bend parameters
    angle_degrees: float
    zone_width: float

    # Fold axis (unit vector along the fold lines)
    axis: tuple[float, float]

    # Fold center position
    center: tuple[float, float]

    @property
    def angle_radians(self) -> float:
        return math.radians(self.angle_degrees)

    @property
    def radius(self) -> float:
        if abs(self.angle_radians) < 1e-9:
            return float('inf')
        return self.zone_width / abs(self.angle_radians)


# =============================================================================
# Arch Board Geometry
# =============================================================================

def get_arch_outline():
    """Arch/U-shaped board outline (CCW winding)."""
    return [
        (0, 0),
        (20, 0),
        (20, 80),
        (80, 80),
        (80, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    ]


def get_all_holes():
    """All three holes."""
    return [
        [(7.5, 37.5), (12.5, 37.5), (12.5, 42.5), (7.5, 42.5)],      # Left: 5x5
        [(50, 85), (60, 85), (60, 95), (50, 95)],                     # Top: 10x10
        [(82.5, 32.5), (97.5, 32.5), (97.5, 47.5), (82.5, 47.5)],    # Right: 15x15
    ]


def get_interesting_points():
    """Key points to track through transformations."""
    # Note: fold centers (10,40), (50,90), (90,40) are inside holes,
    # so we use points just outside the holes for display
    return {
        "left_bottom": (10, 10),
        "left_above_fold": (10, 60),
        "top_left_corner": (5, 90),
        "top_right_corner": (95, 90),
        "right_above_fold": (90, 60),
        "right_bottom": (90, 10),
    }


# =============================================================================
# Fold Marker Creation with Finite Extents
# =============================================================================

def create_left_fold_marker(angle_deg: float, zone_width: float = 10.0) -> MockFoldMarker:
    """
    Create fold marker for left leg.
    Horizontal fold at y=40, spans x=0-20 (left leg width only).
    """
    hw = zone_width / 2
    return MockFoldMarker(
        line_a_start=(0, 40 - hw),
        line_a_end=(20, 40 - hw),
        line_b_start=(0, 40 + hw),
        line_b_end=(20, 40 + hw),
        angle_degrees=angle_deg,
        zone_width=zone_width,
        axis=(1, 0),  # Fold line along X
        center=(10, 40),
    )


def create_top_fold_marker(angle_deg: float, zone_width: float = 10.0) -> MockFoldMarker:
    """
    Create fold marker for top section.
    Vertical fold at x=50, spans y=80-100 (top section only).
    """
    hw = zone_width / 2
    return MockFoldMarker(
        line_a_start=(50 - hw, 80),
        line_a_end=(50 - hw, 100),
        line_b_start=(50 + hw, 80),
        line_b_end=(50 + hw, 100),
        angle_degrees=angle_deg,
        zone_width=zone_width,
        axis=(0, 1),  # Fold line along Y
        center=(50, 90),
    )


def create_right_fold_marker(angle_deg: float, zone_width: float = 10.0) -> MockFoldMarker:
    """
    Create fold marker for right leg.
    Horizontal fold at y=40, spans x=80-100 (right leg width only).
    """
    hw = zone_width / 2
    return MockFoldMarker(
        line_a_start=(80, 40 - hw),
        line_a_end=(100, 40 - hw),
        line_b_start=(80, 40 + hw),
        line_b_end=(100, 40 + hw),
        angle_degrees=angle_deg,
        zone_width=zone_width,
        axis=(1, 0),  # Fold line along X
        center=(90, 40),
    )


# =============================================================================
# Region-based transformation
# =============================================================================

def get_regions_with_recipes(outline, holes, markers):
    """
    Get regions with properly computed fold recipes using connectivity graph.
    """
    if not any(m.angle_degrees != 0 for m in markers):
        # No active folds - single flat region
        return [Region(outline=list(outline), holes=[list(h) for h in holes], index=0)]

    # Filter to active folds only
    active_markers = [m for m in markers if m.angle_degrees != 0]

    # Use the planar subdivision system which computes recipes via BFS
    regions = split_board_into_regions(
        outline,
        holes,
        active_markers,
        num_bend_subdivisions=4
    )
    return regions


def transform_point_with_regions(point, regions):
    """
    Transform a point using the region-based recipe system.
    """
    region = find_containing_region(point, regions)
    if region is None:
        return (point[0], point[1], 0.0)

    # Convert region's fold_recipe to FoldDefinition-based recipe
    recipe = []
    for entry in region.fold_recipe:
        marker = entry[0]
        classification = entry[1]
        entered_from_back = entry[2] if len(entry) > 2 else False
        fold_def = FoldDefinition(
            center=marker.center,
            axis=marker.axis,
            zone_width=marker.zone_width,
            angle=marker.angle_radians,
        )
        recipe.append((fold_def, classification, entered_from_back))

    return transform_point(point, recipe)


def transform_polygon_with_regions(polygon, regions):
    """Transform a 2D polygon to 3D using region-based recipes."""
    return [transform_point_with_regions(p, regions) for p in polygon]


def transform_point_with_recipe(point, fold_recipe):
    """Transform a point using a specific fold recipe (not region lookup)."""
    recipe = []
    for entry in fold_recipe:
        marker = entry[0]
        classification = entry[1]
        entered_from_back = entry[2] if len(entry) > 2 else False
        fold_def = FoldDefinition(
            center=marker.center,
            axis=marker.axis,
            zone_width=marker.zone_width,
            angle=marker.angle_radians,
        )
        recipe.append((fold_def, classification, entered_from_back))
    return transform_point(point, recipe)


def transform_polygon_with_recipe(polygon, fold_recipe):
    """Transform a polygon using a specific fold recipe."""
    return [transform_point_with_recipe(p, fold_recipe) for p in polygon]


# =============================================================================
# Plotting
# =============================================================================

def save_3d_plot(filename: str, config: dict, markers: list, regions: list, title: str):
    """Save 3D visualization showing outline, holes, and regions."""
    filepath = RESULTS_DIR / filename

    outline = get_arch_outline()
    holes = get_all_holes()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color palette for regions
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(regions), 1)))

    # Plot each region using its OWN recipe (not region lookup)
    # Subdivide edges for smoother 3D rendering
    def subdivide_polygon(outline, recipe, num_subdivisions=5):
        """Subdivide polygon edges for smoother 3D transformation."""
        result = []
        for i in range(len(outline)):
            p1 = outline[i]
            p2 = outline[(i + 1) % len(outline)]
            for j in range(num_subdivisions):
                t = j / num_subdivisions
                p2d = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                p3d = transform_point_with_recipe(p2d, recipe)
                result.append(p3d)
        return result

    for i, region in enumerate(regions):
        region_3d = subdivide_polygon(region.outline, region.fold_recipe)
        region_3d_closed = region_3d + [region_3d[0]]

        xs = [p[0] for p in region_3d_closed]
        ys = [p[1] for p in region_3d_closed]
        zs = [p[2] for p in region_3d_closed]

        # Plot region outline
        ax.plot(xs, ys, zs, 'k-', linewidth=0.5, alpha=0.7)

        # Fill region
        verts = [list(zip(xs[:-1], ys[:-1], zs[:-1]))]
        poly = Poly3DCollection(verts, alpha=0.4, facecolor=colors[i % len(colors)],
                                 edgecolor='black', linewidth=0.5)
        ax.add_collection3d(poly)

    # Note: Board outline and holes are already visible as region boundaries
    # Drawing them separately causes distortion at region boundaries

    # Plot interesting points - use the point's containing region's recipe
    for name, p2d in get_interesting_points().items():
        region = find_containing_region(p2d, regions)
        if region:
            p3d = transform_point_with_recipe(p2d, region.fold_recipe)
        else:
            p3d = (p2d[0], p2d[1], 0.0)
        ax.scatter([p3d[0]], [p3d[1]], [p3d[2]], c='green', s=40, marker='o')
        ax.text(p3d[0], p3d[1], p3d[2], f'  {name}', fontsize=6)

    # Note: Fold marker lines are not drawn in 3D because they lie on region
    # boundaries and inside holes, making region lookup unreliable.
    # The fold zones are visible as the curved IN_ZONE region stripes.

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    ax.set_xlim(-20, 120)
    ax.set_ylim(-20, 120)
    ax.set_zlim(-20, 80)

    # Config text
    config_text = f"Folds: L={config['left']}°, T={config['top']}°, R={config['right']}°"
    fig.text(0.02, 0.02, config_text, fontsize=10, family='monospace')
    fig.text(0.02, 0.05, f"Regions: {len(regions)}", fontsize=10, family='monospace')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_graph_plot(filename: str, config: dict, markers: list, regions: list):
    """
    Save a visualization of the region adjacency graph.

    Nodes = regions (colored to match 3D plot)
    Edges = adjacency between regions
    Labels = region index + fold recipe
    """
    filepath = RESULTS_DIR / filename

    # Build adjacency
    adjacency = build_region_adjacency(regions)

    # Compute region centroids for node positions
    positions = {}
    for region in regions:
        xs = [p[0] for p in region.outline]
        ys = [p[1] for p in region.outline]
        positions[region.index] = (sum(xs) / len(xs), sum(ys) / len(ys))

    # Color palette matching 3D plot
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(regions), 1)))

    fig, ax = plt.subplots(figsize=(14, 12))

    # Draw board outline for context (faint)
    outline = get_arch_outline()
    outline_closed = list(outline) + [outline[0]]
    xs = [p[0] for p in outline_closed]
    ys = [p[1] for p in outline_closed]
    ax.plot(xs, ys, 'k-', alpha=0.2, linewidth=1)

    # Draw holes for context
    for hole in get_all_holes():
        hole_closed = list(hole) + [hole[0]]
        hxs = [p[0] for p in hole_closed]
        hys = [p[1] for p in hole_closed]
        ax.plot(hxs, hys, 'k--', alpha=0.2, linewidth=1)

    # Draw fold marker lines
    for marker in markers:
        if marker.angle_radians != 0:
            ax.plot([marker.line_a_start[0], marker.line_a_end[0]],
                   [marker.line_a_start[1], marker.line_a_end[1]],
                   'r-', linewidth=2, alpha=0.5)
            ax.plot([marker.line_b_start[0], marker.line_b_end[0]],
                   [marker.line_b_start[1], marker.line_b_end[1]],
                   'r-', linewidth=2, alpha=0.5)

    # Draw edges (adjacency)
    for idx, neighbors in adjacency.items():
        x1, y1 = positions[idx]
        for neighbor_idx in neighbors:
            if neighbor_idx > idx:  # Draw each edge once
                x2, y2 = positions[neighbor_idx]
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=1)

    # Draw nodes
    node_size = 800
    for region in regions:
        x, y = positions[region.index]
        color = colors[region.index % len(colors)]
        ax.scatter([x], [y], c=[color], s=node_size, zorder=5, edgecolors='black', linewidth=1)

    # Draw labels with recipe info
    for region in regions:
        x, y = positions[region.index]

        # Format recipe
        if not region.fold_recipe:
            recipe_str = "flat"
        else:
            parts = []
            for entry in region.fold_recipe:
                fold = entry[0]
                classification = entry[1]
                entered_from_back = entry[2] if len(entry) > 2 else False
                # Identify fold by its center
                fold_id = f"({int(fold.center[0])},{int(fold.center[1])})"
                class_short = classification[0] if classification else "?"  # I, A, B
                if entered_from_back:
                    class_short += "'"  # Mark back entry with prime
                parts.append(f"{fold_id}:{class_short}")
            recipe_str = "\n".join(parts)
        print("R", region.index, recipe_str)



        label = f"R{region.index}\n{recipe_str}"
        ax.annotate(label, (x, y), ha='center', va='center', fontsize=4,
                   fontweight='bold', zorder=10)

    # Title and labels
    title = f"Region Adjacency Graph: L={config['left']}° T={config['top']}° R={config['right']}°"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Legend for fold markers
    legend_text = "Fold centers: Left=(10,40), Top=(50,90), Right=(90,40)"
    fig.text(0.02, 0.02, legend_text, fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_text_results(filename: str, config: dict, markers: list, regions: list):
    """Save text results showing point transformations."""
    filepath = RESULTS_DIR / filename
    interesting = get_interesting_points()

    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ARCH BOARD FOLD TRANSFORMATION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("FOLD CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Left:  {config['left']}°\n")
        f.write(f"  Top:   {config['top']}°\n")
        f.write(f"  Right: {config['right']}°\n\n")

        f.write(f"REGIONS: {len(regions)}\n")
        f.write("-" * 40 + "\n")
        for region in regions:
            area = abs(signed_area(region.outline))
            recipe_str = " -> ".join(
                f"{e[0].center}:{e[1]}{'(B)' if (len(e) > 2 and e[2]) else ''}"
                for e in region.fold_recipe
            ) if region.fold_recipe else "flat"
            f.write(f"  Region {region.index}: area={area:.0f}, recipe={recipe_str}\n")
        f.write("\n")

        f.write("POINT TRANSFORMATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Point':<20} {'2D':<12} {'3D':<30} {'Recipe':<30}\n")
        f.write("-" * 70 + "\n")

        for name, p2d in interesting.items():
            p3d = transform_point_with_regions(p2d, regions)
            region = find_containing_region(p2d, regions)

            p2d_str = f"({p2d[0]:.0f},{p2d[1]:.0f})"
            p3d_str = f"({p3d[0]:.1f}, {p3d[1]:.1f}, {p3d[2]:.1f})"

            if region and region.fold_recipe:
                recipe_str = " -> ".join(f"{entry[1]}" for entry in region.fold_recipe)
            else:
                recipe_str = "flat"

            f.write(f"{name:<20} {p2d_str:<12} {p3d_str:<30} {recipe_str}\n")


# =============================================================================
# Test Configurations
# =============================================================================

TEST_CONFIGS = [
    ("L0_T0_R0", 0, 0, 0),
    ("L90_T0_R0", 90, 0, 0),
    ("L90_T90_R0", 90, 90, 0),
    ("L0_T90_R0", 0, 90, 0),
    ("L0_T90_R90", 0, 90, 90),
    ("L0_T0_R90", 0, 0, 90),
    ("L90_T90_R90", 90, 90, 90),
]


def generate_all_results():
    """Generate results for all test configurations."""
    print(f"Saving results to: {RESULTS_DIR}")
    RESULTS_DIR.mkdir(exist_ok=True)

    outline = get_arch_outline()
    holes = get_all_holes()

    for config_name, left_deg, top_deg, right_deg in TEST_CONFIGS:
        print(f"Generating {config_name}...")

        markers = [
            create_left_fold_marker(left_deg),
            create_top_fold_marker(top_deg),
            create_right_fold_marker(right_deg),
        ]

        config = {"left": left_deg, "top": top_deg, "right": right_deg}

        # Get regions with recipes computed via connectivity
        regions = get_regions_with_recipes(outline, holes, markers)

        # Save outputs
        save_text_results(f"{config_name}.txt", config, markers, regions)
        title = f"Arch Board: L={left_deg}°, T={top_deg}°, R={right_deg}°"
        save_3d_plot(f"{config_name}.png", config, markers, regions, title)
        save_graph_plot(f"{config_name}_graph.png", config, markers, regions)

    print(f"Generated {len(TEST_CONFIGS)} result sets")
    generate_summary()


def generate_summary():
    """Generate summary with pass/fail checks."""
    summary_file = RESULTS_DIR / "SUMMARY.txt"
    outline = get_arch_outline()
    holes = get_all_holes()

    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ARCH BOARD TRANSFORMATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Key points
        key_points = [
            ("left_bottom", (10, 10)),
            ("left_above_fold", (10, 60)),
            ("top_left", (5, 90)),
            ("top_right", (95, 90)),
            ("right_above_fold", (90, 60)),
            ("right_bottom", (90, 10)),
        ]

        f.write("Z-coordinates by configuration:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Config':<15}")
        for name, _ in key_points:
            f.write(f"{name:<12}")
        f.write("\n")
        f.write("-" * 90 + "\n")

        for config_name, left_deg, top_deg, right_deg in TEST_CONFIGS:
            markers = [
                create_left_fold_marker(left_deg),
                create_top_fold_marker(top_deg),
                create_right_fold_marker(right_deg),
            ]
            regions = get_regions_with_recipes(outline, holes, markers)

            f.write(f"{config_name:<15}")
            for _, p2d in key_points:
                p3d = transform_point_with_regions(p2d, regions)
                f.write(f"z={p3d[2]:<9.1f}")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("VALIDATION CHECKS\n")
        f.write("=" * 80 + "\n\n")

        checks = [
            ("L0_T0_R0", "All z=0 (flat)", check_flat),
            ("L90_T0_R0", "Left leg lifts, right stays flat", check_left_only),
            ("L0_T90_R0", "Top folds, legs stay flat", check_top_only),
            ("L0_T0_R90", "Right leg lifts, left stays flat", check_right_only),
        ]

        for config_name, desc, check_fn in checks:
            result, detail = check_fn(outline, holes)
            status = "[PASS]" if result else "[FAIL]"
            f.write(f"{status} {config_name}: {desc}\n")
            f.write(f"       {detail}\n\n")

    print(f"Summary saved to: {summary_file}")


# =============================================================================
# Validation Checks
# =============================================================================

def check_flat(outline, holes):
    markers = [
        create_left_fold_marker(0),
        create_top_fold_marker(0),
        create_right_fold_marker(0),
    ]
    regions = get_regions_with_recipes(outline, holes, markers)

    max_z = 0
    for name, p2d in get_interesting_points().items():
        p3d = transform_point_with_regions(p2d, regions)
        max_z = max(max_z, abs(p3d[2]))

    if max_z < 0.1:
        return True, f"max |z| = {max_z:.2f}"
    return False, f"max |z| = {max_z:.2f}, expected ~0"


def check_left_only(outline, holes):
    markers = [
        create_left_fold_marker(90),
        create_top_fold_marker(0),
        create_right_fold_marker(0),
    ]
    regions = get_regions_with_recipes(outline, holes, markers)

    # Left above fold should lift
    p3d = transform_point_with_regions((10, 60), regions)
    if p3d[2] < 10:
        return False, f"left_above_fold z={p3d[2]:.1f}, should be >10"

    # Right leg should stay flat (fold doesn't reach it)
    p3d = transform_point_with_regions((90, 60), regions)
    if abs(p3d[2]) > 1:
        return False, f"right_above_fold z={p3d[2]:.1f}, should be ~0 (left fold shouldn't reach right leg)"

    # Right fold center should stay flat
    p3d = transform_point_with_regions((90, 40), regions)
    if abs(p3d[2]) > 1:
        return False, f"right_fold_center z={p3d[2]:.1f}, should be ~0"

    return True, "Left fold affects only left leg and connected regions"


def check_top_only(outline, holes):
    markers = [
        create_left_fold_marker(0),
        create_top_fold_marker(90),
        create_right_fold_marker(0),
    ]
    regions = get_regions_with_recipes(outline, holes, markers)

    # Left leg should stay flat (not connected through top fold)
    p3d = transform_point_with_regions((10, 10), regions)
    if abs(p3d[2]) > 1:
        return False, f"left_bottom z={p3d[2]:.1f}, should be ~0"

    p3d = transform_point_with_regions((10, 60), regions)
    if abs(p3d[2]) > 1:
        return False, f"left_above_fold z={p3d[2]:.1f}, should be ~0 (top fold shouldn't affect left leg)"

    # Right leg should stay flat
    p3d = transform_point_with_regions((90, 10), regions)
    if abs(p3d[2]) > 1:
        return False, f"right_bottom z={p3d[2]:.1f}, should be ~0"

    return True, "Top fold affects only top section"


def check_right_only(outline, holes):
    markers = [
        create_left_fold_marker(0),
        create_top_fold_marker(0),
        create_right_fold_marker(90),
    ]
    regions = get_regions_with_recipes(outline, holes, markers)

    # Right above fold should lift
    p3d = transform_point_with_regions((90, 60), regions)
    if p3d[2] < 10:
        return False, f"right_above_fold z={p3d[2]:.1f}, should be >10"

    # Left leg should stay flat
    p3d = transform_point_with_regions((10, 60), regions)
    if abs(p3d[2]) > 1:
        return False, f"left_above_fold z={p3d[2]:.1f}, should be ~0 (right fold shouldn't reach left leg)"

    return True, "Right fold affects only right leg and connected regions"


if __name__ == "__main__":
    generate_all_results()

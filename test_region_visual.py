#!/usr/bin/env python3
"""
Visual test for region splitting and triangulation.
Plots board outline, fold lines, regions, and triangulation.
"""

import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import numpy as np

from region_splitter import split_polygon_by_line, split_board_into_regions, _clean_polygon
from markers import FoldMarker
from mesh import ear_clip_triangulate, triangulate_with_holes


def create_test_board():
    """Create a test board outline (rectangle with notch)."""
    return [
        (0, 0),
        (100, 0),
        (100, 40),
        (70, 40),
        (70, 30),
        (30, 30),
        (30, 40),
        (0, 40),
    ]


def create_random_fold_markers(board_outline, num_folds=2):
    """Create random parallel fold markers across the board."""
    # Get board bounds
    xs = [p[0] for p in board_outline]
    ys = [p[1] for p in board_outline]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Random fold direction (normalized)
    angle = random.uniform(0, math.pi)  # Random angle
    axis = (math.cos(angle), math.sin(angle))

    # Perpendicular direction for spacing folds
    perp = (-axis[1], axis[0])

    # Project board corners onto perpendicular to find range
    projs = [p[0] * perp[0] + p[1] * perp[1] for p in board_outline]
    min_proj, max_proj = min(projs), max(projs)

    # Create fold positions evenly spaced
    margin = (max_proj - min_proj) * 0.2
    fold_range = max_proj - min_proj - 2 * margin

    markers = []
    for i in range(num_folds):
        # Position along perpendicular
        t = (i + 1) / (num_folds + 1)
        proj_pos = min_proj + margin + t * fold_range

        # Center point of fold
        center_x = proj_pos * perp[0]
        center_y = proj_pos * perp[1]

        # Adjust to be inside board
        center_x = min_x + (max_x - min_x) / 2 + (proj_pos - (min_proj + max_proj) / 2) * perp[0]
        center_y = min_y + (max_y - min_y) / 2 + (proj_pos - (min_proj + max_proj) / 2) * perp[1]

        # Create line endpoints (extend beyond board)
        extent = max(max_x - min_x, max_y - min_y)
        line_start = (center_x - axis[0] * extent, center_y - axis[1] * extent)
        line_end = (center_x + axis[0] * extent, center_y + axis[1] * extent)

        # Create fold marker
        radius = 2.0
        offset = radius / 2

        marker = FoldMarker(
            line_a_start=(line_start[0] + perp[0] * offset, line_start[1] + perp[1] * offset),
            line_a_end=(line_end[0] + perp[0] * offset, line_end[1] + perp[1] * offset),
            line_b_start=(line_start[0] - perp[0] * offset, line_start[1] - perp[1] * offset),
            line_b_end=(line_end[0] - perp[0] * offset, line_end[1] - perp[1] * offset),
            angle_degrees=90.0,
            zone_width=radius,
            radius=radius,
            axis=axis,
            center=(center_x, center_y),
        )
        markers.append(marker)

    return markers


def plot_polygon(ax, polygon, color='blue', alpha=0.3, edgecolor='black', linewidth=1):
    """Plot a polygon."""
    if not polygon:
        return
    poly = plt.Polygon(polygon, facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(poly)


def plot_triangulation(ax, polygon, color='green', alpha=0.2):
    """Triangulate and plot a polygon."""
    if len(polygon) < 3:
        return

    try:
        triangles = ear_clip_triangulate(polygon)
        for tri in triangles:
            tri_poly = plt.Polygon(tri, facecolor=color, alpha=alpha, edgecolor='gray', linewidth=0.5)
            ax.add_patch(tri_poly)
    except Exception as e:
        print(f"Triangulation error: {e}")


def plot_fold_line(ax, marker, board_bounds, color='red', linewidth=2):
    """Plot a fold line across the board."""
    min_x, max_x, min_y, max_y = board_bounds

    # Extend line to cover board
    center = marker.center
    axis = marker.axis
    extent = max(max_x - min_x, max_y - min_y) * 1.5

    p1 = (center[0] - axis[0] * extent, center[1] - axis[1] * extent)
    p2 = (center[0] + axis[0] * extent, center[1] + axis[1] * extent)

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, linestyle='--', label='Fold line')


def main():
    random.seed(42)  # For reproducibility

    # Create test board
    board = create_test_board()

    # Create random fold markers
    markers = create_random_fold_markers(board, num_folds=2)

    # Get board bounds
    xs = [p[0] for p in board]
    ys = [p[1] for p in board]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    board_bounds = (min_x, max_x, min_y, max_y)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Original board with fold lines
    ax1 = axes[0]
    ax1.set_title('Original Board + Fold Lines')
    plot_polygon(ax1, board, color='lightblue', alpha=0.5, edgecolor='blue', linewidth=2)
    for i, marker in enumerate(markers):
        plot_fold_line(ax1, marker, board_bounds, color=['red', 'orange'][i % 2])
        ax1.plot(*marker.center, 'o', color=['red', 'orange'][i % 2], markersize=8)
    ax1.set_xlim(min_x - 10, max_x + 10)
    ax1.set_ylim(min_y - 10, max_y + 10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Split into regions
    regions = split_board_into_regions(board, [], markers)
    print(f"Number of regions: {len(regions)}")
    for i, region in enumerate(regions):
        print(f"  Region {i}: {len(region.outline)} vertices")

    # Plot 2: Regions (colored differently)
    ax2 = axes[1]
    ax2.set_title(f'Split into {len(regions)} Regions')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    for i, region in enumerate(regions):
        cleaned = _clean_polygon(region.outline)
        if cleaned:
            plot_polygon(ax2, cleaned, color=colors[i % len(colors)], alpha=0.6, edgecolor='black', linewidth=1.5)
            # Mark centroid
            cx = sum(p[0] for p in cleaned) / len(cleaned)
            cy = sum(p[1] for p in cleaned) / len(cleaned)
            ax2.text(cx, cy, f'R{i}', ha='center', va='center', fontsize=12, fontweight='bold')

    for i, marker in enumerate(markers):
        plot_fold_line(ax2, marker, board_bounds, color=['red', 'orange'][i % 2])

    ax2.set_xlim(min_x - 10, max_x + 10)
    ax2.set_ylim(min_y - 10, max_y + 10)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Triangulated regions
    ax3 = axes[2]
    ax3.set_title('Triangulated Regions')
    for i, region in enumerate(regions):
        cleaned = _clean_polygon(region.outline)
        if cleaned:
            plot_triangulation(ax3, cleaned, color=colors[i % len(colors)], alpha=0.5)

    for i, marker in enumerate(markers):
        plot_fold_line(ax3, marker, board_bounds, color=['red', 'orange'][i % 2])

    ax3.set_xlim(min_x - 10, max_x + 10)
    ax3.set_ylim(min_y - 10, max_y + 10)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/region_test.png', dpi=150)
    print(f"\nSaved plot to /tmp/region_test.png")
    plt.show()


if __name__ == '__main__':
    main()

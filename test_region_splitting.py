"""
Test script for region splitting with cutouts/holes.

When a fold line crosses a hole, the region boundary should go around the hole,
not through it. This script tests and validates the region splitting algorithm.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import random
import math
import os

# Import from existing modules
from region_splitter import (
    split_polygon_by_line, _clean_polygon, _point_side_of_line,
    _polygon_centroid, _point_in_polygon
)


def signed_area(polygon):
    """Calculate signed area. Positive = CCW, Negative = CW."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0


def ensure_ccw(polygon):
    """Ensure polygon is counter-clockwise."""
    if signed_area(polygon) < 0:
        return list(reversed(polygon))
    return list(polygon)


def ensure_cw(polygon):
    """Ensure polygon is clockwise."""
    if signed_area(polygon) > 0:
        return list(reversed(polygon))
    return list(polygon)


def point_in_polygon(point, polygon):
    """Ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def line_segment_intersection(p1, p2, p3, p4):
    """
    Find intersection of line segments p1-p2 and p3-p4.
    Returns intersection point or None.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def clip_polygon_to_halfplane(polygon, line_point, line_dir, keep_left=True):
    """
    Clip a polygon to one side of a line using Sutherland-Hodgman style algorithm.

    Args:
        polygon: List of (x, y) vertices
        line_point: Point on the clipping line
        line_dir: Direction of the clipping line
        keep_left: If True, keep points where side > 0 (left of line direction)

    Returns:
        Clipped polygon vertices
    """
    if len(polygon) < 3:
        return []

    def point_side(p):
        return _point_side_of_line(p, line_point, line_dir)

    def is_inside(p):
        side = point_side(p)
        if keep_left:
            return side >= -1e-9
        else:
            return side <= 1e-9

    def line_intersect(p1, p2):
        """Find intersection of edge p1-p2 with the clipping line."""
        d1 = point_side(p1)
        d2 = point_side(p2)

        if abs(d2 - d1) < 1e-10:
            return None

        t = d1 / (d1 - d2)
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        return (x, y)

    output = []
    n = len(polygon)

    for i in range(n):
        current = polygon[i]
        next_v = polygon[(i + 1) % n]

        current_inside = is_inside(current)
        next_inside = is_inside(next_v)

        if current_inside:
            output.append(current)

            if not next_inside:
                # Exiting - add intersection
                intersection = line_intersect(current, next_v)
                if intersection:
                    output.append(intersection)
        else:
            if next_inside:
                # Entering - add intersection
                intersection = line_intersect(current, next_v)
                if intersection:
                    output.append(intersection)

    return output


def split_polygon_with_holes(outer, holes, line_point, line_dir):
    """
    Split a polygon with holes by a line.

    Returns two lists of (outline, holes) tuples for left and right sides.
    """
    # Split outer polygon
    left_outers, right_outers = split_polygon_by_line(outer, line_point, line_dir)

    # Clean the outer polygons
    left_outers = [_clean_polygon(p) for p in left_outers if _clean_polygon(p)]
    right_outers = [_clean_polygon(p) for p in right_outers if _clean_polygon(p)]

    # For each hole, clip it to each side
    left_holes = []
    right_holes = []

    for hole in holes:
        # Clip hole to left side
        left_hole = clip_polygon_to_halfplane(hole, line_point, line_dir, keep_left=True)
        if left_hole and len(left_hole) >= 3:
            left_holes.append(left_hole)

        # Clip hole to right side
        right_hole = clip_polygon_to_halfplane(hole, line_point, line_dir, keep_left=False)
        if right_hole and len(right_hole) >= 3:
            right_holes.append(right_hole)

    # Match holes to their containing outer polygons
    left_regions = []
    for outer_poly in left_outers:
        region_holes = []
        for hole in left_holes:
            hole_center = _polygon_centroid(hole)
            if point_in_polygon(hole_center, outer_poly):
                region_holes.append(hole)
        left_regions.append((outer_poly, region_holes))

    right_regions = []
    for outer_poly in right_outers:
        region_holes = []
        for hole in right_holes:
            hole_center = _polygon_centroid(hole)
            if point_in_polygon(hole_center, outer_poly):
                region_holes.append(hole)
        right_regions.append((outer_poly, region_holes))

    return left_regions, right_regions


def split_board_with_holes(outer, holes, fold_lines):
    """
    Split a board (outer polygon with holes) by multiple fold lines.

    Returns list of (outline, holes) tuples for each region.
    """
    # Start with single region
    current_regions = [(outer, holes)]

    for line_point, line_dir in fold_lines:
        new_regions = []

        for region_outer, region_holes in current_regions:
            left_regions, right_regions = split_polygon_with_holes(
                region_outer, region_holes, line_point, line_dir
            )
            new_regions.extend(left_regions)
            new_regions.extend(right_regions)

        current_regions = new_regions

    return current_regions


def generate_test_shapes():
    """Generate various test configurations."""
    test_cases = []

    # Test 1: Simple rectangle with one hole, horizontal fold through hole
    test_cases.append({
        'name': 'rect_hole_horizontal',
        'outer': [(0, 0), (100, 0), (100, 60), (0, 60)],
        'holes': [[(30, 20), (50, 20), (50, 40), (30, 40)]],
        'fold_lines': [((50, 30), (1, 0))]  # Horizontal line at y=30
    })

    # Test 2: Rectangle with hole, vertical fold through hole
    test_cases.append({
        'name': 'rect_hole_vertical',
        'outer': [(0, 0), (100, 0), (100, 60), (0, 60)],
        'holes': [[(30, 20), (70, 20), (70, 40), (30, 40)]],
        'fold_lines': [((50, 30), (0, 1))]  # Vertical line at x=50
    })

    # Test 3: Rectangle with two holes, one fold between them
    test_cases.append({
        'name': 'rect_two_holes_between',
        'outer': [(0, 0), (100, 0), (100, 60), (0, 60)],
        'holes': [
            [(10, 20), (30, 20), (30, 40), (10, 40)],
            [(70, 20), (90, 20), (90, 40), (70, 40)]
        ],
        'fold_lines': [((50, 30), (0, 1))]  # Vertical line at x=50
    })

    # Test 4: Rectangle with hole, two parallel folds, one through hole
    test_cases.append({
        'name': 'rect_hole_two_folds',
        'outer': [(0, 0), (100, 0), (100, 60), (0, 60)],
        'holes': [[(35, 15), (65, 15), (65, 45), (35, 45)]],
        'fold_lines': [
            ((50, 20), (1, 0)),  # Horizontal at y=20 (crosses hole)
            ((50, 40), (1, 0))   # Horizontal at y=40 (crosses hole)
        ]
    })

    # Test 5: L-shaped board with hole at corner
    test_cases.append({
        'name': 'l_shape_hole',
        'outer': [(0, 0), (60, 0), (60, 40), (40, 40), (40, 60), (0, 60)],
        'holes': [[(15, 35), (35, 35), (35, 55), (15, 55)]],
        'fold_lines': [((30, 30), (1, 0))]  # Horizontal at y=30
    })

    # Test 6: Rectangle with diagonal fold through hole
    test_cases.append({
        'name': 'rect_hole_diagonal',
        'outer': [(0, 0), (100, 0), (100, 60), (0, 60)],
        'holes': [[(40, 20), (60, 20), (60, 40), (40, 40)]],
        'fold_lines': [((50, 30), (1, 0.5))]  # Diagonal line
    })

    # Test 7: Rectangle with multiple holes, multiple folds
    test_cases.append({
        'name': 'rect_multi_holes_folds',
        'outer': [(0, 0), (120, 0), (120, 80), (0, 80)],
        'holes': [
            [(20, 30), (40, 30), (40, 50), (20, 50)],
            [(60, 20), (80, 20), (80, 60), (60, 60)],
            [(90, 35), (110, 35), (110, 55), (90, 55)]
        ],
        'fold_lines': [
            ((40, 40), (0, 1)),  # Vertical at x=40
            ((80, 40), (0, 1))   # Vertical at x=80
        ]
    })

    return test_cases


def plot_board_with_regions(outer, holes, fold_lines, regions, title, filename):
    """Plot board, holes, fold lines, and resulting regions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Original board with fold lines
    ax1 = axes[0]

    # Plot outer
    outer_closed = list(outer) + [outer[0]]
    xs, ys = zip(*outer_closed)
    ax1.fill(xs, ys, alpha=0.3, color='blue')
    ax1.plot(xs, ys, 'b-', linewidth=2, label='Board outline')

    # Plot holes
    for i, hole in enumerate(holes):
        hole_closed = list(hole) + [hole[0]]
        xs, ys = zip(*hole_closed)
        ax1.fill(xs, ys, color='white')
        ax1.plot(xs, ys, 'r-', linewidth=2, label=f'Hole {i+1}' if i == 0 else '')

    # Plot fold lines
    all_x = [v[0] for v in outer]
    all_y = [v[1] for v in outer]
    margin = 10
    min_x, max_x = min(all_x) - margin, max(all_x) + margin
    min_y, max_y = min(all_y) - margin, max(all_y) + margin

    for i, (lp, ld) in enumerate(fold_lines):
        t_vals = []
        if abs(ld[0]) > 0.01:
            t_vals.extend([(min_x - lp[0]) / ld[0], (max_x - lp[0]) / ld[0]])
        if abs(ld[1]) > 0.01:
            t_vals.extend([(min_y - lp[1]) / ld[1], (max_y - lp[1]) / ld[1]])
        if t_vals:
            t_min, t_max = min(t_vals), max(t_vals)
            x1, y1 = lp[0] + t_min * ld[0], lp[1] + t_min * ld[1]
            x2, y2 = lp[0] + t_max * ld[0], lp[1] + t_max * ld[1]
            ax1.plot([x1, x2], [y1, y2], 'g--', linewidth=2,
                    label=f'Fold {i+1}' if i == 0 else '')

    ax1.set_aspect('equal')
    ax1.set_title('Original Board with Fold Lines')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)

    # Right plot: Resulting regions
    ax2 = axes[1]

    colors = ['cyan', 'lightgreen', 'orange', 'pink', 'yellow', 'lightblue', 'salmon']

    for i, (region_outer, region_holes) in enumerate(regions):
        if len(region_outer) < 3:
            continue

        color = colors[i % len(colors)]

        # Plot region outline
        region_closed = list(region_outer) + [region_outer[0]]
        xs, ys = zip(*region_closed)
        ax2.fill(xs, ys, alpha=0.5, color=color, label=f'Region {i+1}')
        ax2.plot(xs, ys, 'k-', linewidth=2)

        # Plot region holes
        for hole in region_holes:
            if len(hole) >= 3:
                hole_closed = list(hole) + [hole[0]]
                hxs, hys = zip(*hole_closed)
                ax2.fill(hxs, hys, color='white')
                ax2.plot(hxs, hys, 'r-', linewidth=2)

        # Mark vertices
        for j, (x, y) in enumerate(region_outer):
            ax2.plot(x, y, 'ko', markersize=4)

    # Plot fold lines on region plot too
    for i, (lp, ld) in enumerate(fold_lines):
        t_vals = []
        if abs(ld[0]) > 0.01:
            t_vals.extend([(min_x - lp[0]) / ld[0], (max_x - lp[0]) / ld[0]])
        if abs(ld[1]) > 0.01:
            t_vals.extend([(min_y - lp[1]) / ld[1], (max_y - lp[1]) / ld[1]])
        if t_vals:
            t_min, t_max = min(t_vals), max(t_vals)
            x1, y1 = lp[0] + t_min * ld[0], lp[1] + t_min * ld[1]
            x2, y2 = lp[0] + t_max * ld[0], lp[1] + t_max * ld[1]
            ax2.plot([x1, x2], [y1, y2], 'g--', linewidth=1, alpha=0.5)

    ax2.set_aspect('equal')
    ax2.set_title(f'Resulting Regions ({len(regions)} regions)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def validate_regions(outer, holes, regions, fold_lines):
    """
    Validate that regions are correctly split.

    Checks:
    1. Each region outline is a valid closed polygon
    2. Region holes are inside their region
    3. No region outline crosses a fold line (except at intersection points)
    4. Total area of regions equals board area minus holes
    """
    issues = []

    # Calculate expected area
    outer_area = abs(signed_area(outer))
    holes_area = sum(abs(signed_area(h)) for h in holes)
    expected_area = outer_area - holes_area

    # Calculate actual area
    actual_area = 0
    for region_outer, region_holes in regions:
        if len(region_outer) < 3:
            issues.append(f"Region has < 3 vertices")
            continue

        region_area = abs(signed_area(region_outer))
        region_holes_area = sum(abs(signed_area(h)) for h in region_holes if len(h) >= 3)
        actual_area += region_area - region_holes_area

        # Check holes are inside region
        for hole in region_holes:
            if len(hole) < 3:
                continue
            hole_center = _polygon_centroid(hole)
            if not point_in_polygon(hole_center, region_outer):
                issues.append(f"Hole centroid not inside region")

    # Check area match (allow 1% tolerance)
    area_diff = abs(actual_area - expected_area) / expected_area if expected_area > 0 else 0
    if area_diff > 0.01:
        issues.append(f"Area mismatch: expected {expected_area:.1f}, got {actual_area:.1f} ({area_diff*100:.1f}% diff)")

    # Check no region edges cross fold lines improperly
    for region_outer, _ in regions:
        if len(region_outer) < 3:
            continue
        for i in range(len(region_outer)):
            p1 = region_outer[i]
            p2 = region_outer[(i + 1) % len(region_outer)]

            for fold_point, fold_dir in fold_lines:
                side1 = _point_side_of_line(p1, fold_point, fold_dir)
                side2 = _point_side_of_line(p2, fold_point, fold_dir)

                # Edge crosses fold if points are on opposite sides (not on line)
                tolerance = 0.1
                if (side1 > tolerance and side2 < -tolerance) or (side1 < -tolerance and side2 > tolerance):
                    # This edge crosses the fold line internally (not at endpoints)
                    issues.append(f"Region edge ({p1[0]:.1f},{p1[1]:.1f})->({p2[0]:.1f},{p2[1]:.1f}) crosses fold line")

    return issues


def run_tests():
    """Run all test cases and generate reports."""
    output_dir = "/tmp/region_tests"
    os.makedirs(output_dir, exist_ok=True)

    test_cases = generate_test_shapes()

    print("="*70)
    print("REGION SPLITTING TESTS WITH HOLES/CUTOUTS")
    print("="*70)

    all_results = []

    for tc in test_cases:
        name = tc['name']
        outer = tc['outer']
        holes = tc['holes']
        fold_lines = tc['fold_lines']

        print(f"\n{'='*50}")
        print(f"Test: {name}")
        print(f"{'='*50}")
        print(f"Outer polygon: {len(outer)} vertices")
        print(f"Holes: {len(holes)}")
        print(f"Fold lines: {len(fold_lines)}")

        # Split into regions
        regions = split_board_with_holes(outer, holes, fold_lines)

        print(f"\nResulting regions: {len(regions)}")
        for i, (region_outer, region_holes) in enumerate(regions):
            area = abs(signed_area(region_outer)) if len(region_outer) >= 3 else 0
            holes_area = sum(abs(signed_area(h)) for h in region_holes if len(h) >= 3)
            print(f"  Region {i+1}: {len(region_outer)} verts, {len(region_holes)} holes, net area={area-holes_area:.1f}")

        # Validate
        issues = validate_regions(outer, holes, regions, fold_lines)

        if issues:
            print(f"\nISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
            result = "FAIL"
        else:
            print(f"\nValidation: PASS")
            result = "PASS"

        all_results.append((name, result, len(issues)))

        # Generate plot
        plot_file = os.path.join(output_dir, f"{name}.png")
        plot_board_with_regions(outer, holes, fold_lines, regions,
                               f"Test: {name}", plot_file)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, r, _ in all_results if r == "PASS")
    failed = sum(1 for _, r, _ in all_results if r == "FAIL")

    for name, result, num_issues in all_results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}" + (f" ({num_issues} issues)" if num_issues > 0 else ""))

    print(f"\nTotal: {passed} passed, {failed} failed")
    print(f"\nPlots saved to: {output_dir}/")

    return all_results


if __name__ == "__main__":
    run_tests()

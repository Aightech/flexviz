"""
Test that the region_splitter module correctly handles holes when splitting by fold lines.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from region_splitter import split_board_into_regions, _clean_polygon
from markers import FoldMarker


def signed_area(polygon):
    """Calculate signed area. Positive = CCW, Negative = CW."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0


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


def create_mock_fold_marker(center, axis, angle_degrees=90.0, radius=2.0):
    """Create a mock FoldMarker for testing."""
    return FoldMarker(
        line_a_start=(0, 0),
        line_a_end=(0, 0),
        line_b_start=(0, 0),
        line_b_end=(0, 0),
        angle_degrees=angle_degrees,
        zone_width=radius,
        radius=radius,
        axis=axis,
        center=center,
    )


def run_module_test():
    """Test the actual split_board_into_regions function from the module."""
    print("="*70)
    print("MODULE INTEGRATION TEST - split_board_into_regions()")
    print("="*70)

    # Test case: Rectangle with hole, vertical fold through hole
    outer = [(0, 0), (100, 0), (100, 60), (0, 60)]
    holes = [[(30, 20), (70, 20), (70, 40), (30, 40)]]

    # Create mock fold marker at x=50 (vertical line)
    fold_marker = create_mock_fold_marker(
        center=(50, 30),
        axis=(0, 1),  # Vertical direction
        angle_degrees=90.0
    )

    print(f"\nTest: Rectangle with hole, vertical fold through hole")
    print(f"Outer polygon: {len(outer)} vertices")
    print(f"Holes: {len(holes)}")
    print(f"Fold at x=50 (vertical)")

    # Split into regions
    regions = split_board_into_regions(outer, holes, [fold_marker])

    print(f"\nResulting regions: {len(regions)}")

    issues = []

    for i, region in enumerate(regions):
        area = abs(signed_area(region.outline)) if len(region.outline) >= 3 else 0
        holes_area = sum(abs(signed_area(h)) for h in region.holes if len(h) >= 3)
        print(f"  Region {i}: {len(region.outline)} verts, {len(region.holes)} holes, net area={area-holes_area:.1f}")

        # Verify holes are inside their region
        for hole in region.holes:
            if len(hole) < 3:
                continue
            hole_center = (sum(p[0] for p in hole)/len(hole), sum(p[1] for p in hole)/len(hole))
            if not point_in_polygon(hole_center, region.outline):
                issues.append(f"Hole centroid {hole_center} not inside region {i}")

            # Verify all hole vertices are inside the region outline
            for vi, v in enumerate(hole):
                if not point_in_polygon(v, region.outline):
                    # Check if it's on the boundary (within tolerance)
                    on_boundary = False
                    for j in range(len(region.outline)):
                        p1 = region.outline[j]
                        p2 = region.outline[(j+1) % len(region.outline)]
                        # Check if point is on edge p1-p2
                        d1 = abs((v[0]-p1[0])*(p2[1]-p1[1]) - (v[1]-p1[1])*(p2[0]-p1[0]))
                        length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
                        if length > 0 and d1/length < 0.1:
                            # Point is on the line - check if between p1 and p2
                            t1 = (v[0]-p1[0])*(p2[0]-p1[0]) + (v[1]-p1[1])*(p2[1]-p1[1])
                            t_max = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
                            if -0.01*t_max <= t1 <= 1.01*t_max:
                                on_boundary = True
                                break
                    if not on_boundary:
                        issues.append(f"Hole vertex {vi} ({v[0]:.1f}, {v[1]:.1f}) outside region {i}")

    # Check total area
    outer_area = abs(signed_area(outer))
    holes_area = sum(abs(signed_area(h)) for h in holes)
    expected_area = outer_area - holes_area

    actual_area = 0
    for region in regions:
        region_area = abs(signed_area(region.outline)) if len(region.outline) >= 3 else 0
        region_holes_area = sum(abs(signed_area(h)) for h in region.holes if len(h) >= 3)
        actual_area += region_area - region_holes_area

    area_diff = abs(actual_area - expected_area) / expected_area if expected_area > 0 else 0
    if area_diff > 0.01:
        issues.append(f"Area mismatch: expected {expected_area:.1f}, got {actual_area:.1f}")

    print(f"\nExpected total area: {expected_area:.1f}")
    print(f"Actual total area: {actual_area:.1f}")

    if issues:
        print(f"\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        result = "FAIL"
    else:
        print(f"\nValidation: PASS")
        result = "PASS"

    # Generate plot
    output_dir = "/tmp/region_tests"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Original
    ax1 = axes[0]
    outer_closed = list(outer) + [outer[0]]
    xs, ys = zip(*outer_closed)
    ax1.fill(xs, ys, alpha=0.3, color='blue')
    ax1.plot(xs, ys, 'b-', linewidth=2)

    for hole in holes:
        hole_closed = list(hole) + [hole[0]]
        xs, ys = zip(*hole_closed)
        ax1.fill(xs, ys, color='white')
        ax1.plot(xs, ys, 'r-', linewidth=2)

    # Fold line
    ax1.axvline(x=50, color='green', linestyle='--', linewidth=2, label='Fold line')
    ax1.set_aspect('equal')
    ax1.set_title('Original Board with Fold Line')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Regions
    ax2 = axes[1]
    colors = ['cyan', 'orange', 'lightgreen', 'pink']

    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        outline_closed = list(region.outline) + [region.outline[0]]
        xs, ys = zip(*outline_closed)
        ax2.fill(xs, ys, alpha=0.5, color=color, label=f'Region {i}')
        ax2.plot(xs, ys, 'k-', linewidth=2)

        for hole in region.holes:
            if len(hole) >= 3:
                hole_closed = list(hole) + [hole[0]]
                hxs, hys = zip(*hole_closed)
                ax2.fill(hxs, hys, color='white')
                ax2.plot(hxs, hys, 'r-', linewidth=2)

    ax2.axvline(x=50, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_aspect('equal')
    ax2.set_title(f'Resulting Regions ({len(regions)} regions)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/module_integration_test.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to: {output_dir}/module_integration_test.png")

    return result


def run_multiple_folds_test():
    """Test with multiple folds crossing a hole."""
    print("\n" + "="*70)
    print("MULTIPLE FOLDS TEST - Two folds crossing same hole")
    print("="*70)

    outer = [(0, 0), (100, 0), (100, 60), (0, 60)]
    holes = [[(35, 15), (65, 15), (65, 45), (35, 45)]]

    # Two horizontal folds at y=20 and y=40
    markers = [
        create_mock_fold_marker(center=(50, 20), axis=(1, 0)),
        create_mock_fold_marker(center=(50, 40), axis=(1, 0)),
    ]

    print(f"\nOuter polygon: {len(outer)} vertices")
    print(f"Holes: {len(holes)}")
    print(f"Folds at y=20 and y=40 (horizontal)")

    regions = split_board_into_regions(outer, holes, markers)

    print(f"\nResulting regions: {len(regions)}")

    issues = []
    for i, region in enumerate(regions):
        area = abs(signed_area(region.outline)) if len(region.outline) >= 3 else 0
        holes_area = sum(abs(signed_area(h)) for h in region.holes if len(h) >= 3)
        print(f"  Region {i}: {len(region.outline)} verts, {len(region.holes)} holes, net area={area-holes_area:.1f}")

        # Verify holes are inside their region
        for hole in region.holes:
            if len(hole) < 3:
                continue
            hole_center = (sum(p[0] for p in hole)/len(hole), sum(p[1] for p in hole)/len(hole))
            if not point_in_polygon(hole_center, region.outline):
                issues.append(f"Hole centroid not inside region {i}")

    # Check total area
    outer_area = abs(signed_area(outer))
    holes_area_total = sum(abs(signed_area(h)) for h in holes)
    expected_area = outer_area - holes_area_total

    actual_area = 0
    for region in regions:
        region_area = abs(signed_area(region.outline)) if len(region.outline) >= 3 else 0
        region_holes_area = sum(abs(signed_area(h)) for h in region.holes if len(h) >= 3)
        actual_area += region_area - region_holes_area

    print(f"\nExpected total area: {expected_area:.1f}")
    print(f"Actual total area: {actual_area:.1f}")

    area_diff = abs(actual_area - expected_area) / expected_area if expected_area > 0 else 0
    if area_diff > 0.01:
        issues.append(f"Area mismatch: expected {expected_area:.1f}, got {actual_area:.1f}")

    if issues:
        print(f"\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        result = "FAIL"
    else:
        print(f"\nValidation: PASS")
        result = "PASS"

    # Generate plot
    output_dir = "/tmp/region_tests"
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Original
    ax1 = axes[0]
    outer_closed = list(outer) + [outer[0]]
    xs, ys = zip(*outer_closed)
    ax1.fill(xs, ys, alpha=0.3, color='blue')
    ax1.plot(xs, ys, 'b-', linewidth=2)

    for hole in holes:
        hole_closed = list(hole) + [hole[0]]
        xs, ys = zip(*hole_closed)
        ax1.fill(xs, ys, color='white')
        ax1.plot(xs, ys, 'r-', linewidth=2)

    ax1.axhline(y=20, color='green', linestyle='--', linewidth=2)
    ax1.axhline(y=40, color='orange', linestyle='--', linewidth=2)
    ax1.set_aspect('equal')
    ax1.set_title('Original Board with Fold Lines')
    ax1.grid(True, alpha=0.3)

    # Right: Regions
    ax2 = axes[1]
    colors = ['cyan', 'orange', 'lightgreen', 'pink']

    for i, region in enumerate(regions):
        color = colors[i % len(colors)]
        outline_closed = list(region.outline) + [region.outline[0]]
        xs, ys = zip(*outline_closed)
        ax2.fill(xs, ys, alpha=0.5, color=color, label=f'Region {i}')
        ax2.plot(xs, ys, 'k-', linewidth=2)

        for hole in region.holes:
            if len(hole) >= 3:
                hole_closed = list(hole) + [hole[0]]
                hxs, hys = zip(*hole_closed)
                ax2.fill(hxs, hys, color='white')
                ax2.plot(hxs, hys, 'r-', linewidth=2)

    ax2.axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=40, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_aspect('equal')
    ax2.set_title(f'Resulting Regions ({len(regions)} regions)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/module_multi_folds_test.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to: {output_dir}/module_multi_folds_test.png")

    return result


if __name__ == "__main__":
    results = []
    results.append(("Module integration test", run_module_test()))
    results.append(("Multiple folds test", run_multiple_folds_test()))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")

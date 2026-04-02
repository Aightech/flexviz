"""
Core mesh data types and transformation helpers.

Provides the Mesh dataclass for 3D mesh representation and utility functions
for vertex transformation with thickness, debug coloring, and plane snapping.
"""

from dataclasses import dataclass, field
import math

try:
    from .bend_transform import (
        FoldRecipe, transform_point, transform_point_and_normal,
        recipe_from_region, transform_points_batch, compute_normal,
    )
except ImportError:
    from bend_transform import (
        FoldRecipe, transform_point, transform_point_and_normal,
        recipe_from_region, transform_points_batch, compute_normal,
    )


@dataclass
class Mesh:
    """
    A 3D mesh with vertices and faces.

    Vertices are (x, y, z) tuples.
    Faces are lists of vertex indices (triangles or quads).
    """
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[list[int]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)

    # Optional per-face colors (r, g, b) 0-255
    colors: list[tuple[int, int, int]] = field(default_factory=list)

    def add_vertex(self, v: tuple[float, float, float]) -> int:
        """Add a vertex and return its index."""
        self.vertices.append(v)
        return len(self.vertices) - 1

    def add_face(self, indices: list[int], color: tuple[int, int, int] = None):
        """Add a face (list of vertex indices)."""
        self.faces.append(indices)
        if color:
            self.colors.append(color)

    def add_triangle(self, v0: int, v1: int, v2: int, color: tuple[int, int, int] = None):
        """Add a triangle face."""
        self.add_face([v0, v1, v2], color)

    def add_quad(self, v0: int, v1: int, v2: int, v3: int, color: tuple[int, int, int] = None):
        """Add a quad face (will be split into two triangles for some formats)."""
        self.add_face([v0, v1, v2, v3], color)

    def merge(self, other: 'Mesh'):
        """Merge another mesh into this one."""
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        for face in other.faces:
            self.faces.append([i + offset for i in face])
        self.colors.extend(other.colors)

    def compute_normals(self):
        """Compute face normals."""
        self.normals = []
        for face in self.faces:
            if len(face) >= 3:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]

                # Two edge vectors
                e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
                e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])

                # Cross product
                nx = e1[1] * e2[2] - e1[2] * e2[1]
                ny = e1[2] * e2[0] - e1[0] * e2[2]
                nz = e1[0] * e2[1] - e1[1] * e2[0]

                # Normalize
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 1e-10:
                    self.normals.append((nx/length, ny/length, nz/length))
                else:
                    self.normals.append((0, 0, 1))
            else:
                self.normals.append((0, 0, 1))

    def to_obj(self, filename: str):
        """Export mesh to OBJ file format."""
        with open(filename, 'w') as f:
            f.write("# KiCad Flex Viewer - OBJ Export\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n\n")

            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # Write faces (OBJ uses 1-based indexing)
            for face in self.faces:
                indices = " ".join(str(i + 1) for i in face)
                f.write(f"f {indices}\n")

    def to_stl(self, filename: str):
        """Export mesh to STL file format (ASCII)."""
        if not self.normals:
            self.compute_normals()

        with open(filename, 'w') as f:
            f.write("solid kicad_flex_viewer\n")

            for i, face in enumerate(self.faces):
                if len(face) < 3:
                    continue

                normal = self.normals[i] if i < len(self.normals) else (0, 0, 1)

                # Triangulate if needed (fan triangulation for convex polygons)
                for j in range(1, len(face) - 1):
                    v0 = self.vertices[face[0]]
                    v1 = self.vertices[face[j]]
                    v2 = self.vertices[face[j + 1]]

                    f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                    f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                    f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")

            f.write("endsolid kicad_flex_viewer\n")


# Color constants (RGB 0-255)
COLOR_BOARD = (34, 139, 34)      # Forest green (PCB color)
COLOR_COPPER = (184, 115, 51)    # Copper/bronze
COLOR_PAD = (255, 215, 0)        # Gold
COLOR_COMPONENT = (128, 128, 128) # Gray
COLOR_STIFFENER = (139, 90, 43)  # Saddle brown (FR4 stiffener color)
COLOR_CUTOUT = (50, 50, 50)      # Dark gray for cutout walls
COLOR_MODEL_3D = (180, 180, 180) # Light gray for 3D models

# Debug colors for regions (used when debug_regions=True)
DEBUG_REGION_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring green
    (255, 128, 128),  # Light red
    (128, 255, 128),  # Light green
    (128, 128, 255),  # Light blue
]


# =============================================================================
# Helper Functions for Transformation
# =============================================================================

# Backward-compatible alias — canonical implementation now in bend_transform.py
get_region_recipe = recipe_from_region


def transform_vertices_with_thickness(
    vertices_2d: list[tuple[float, float]],
    recipe: FoldRecipe,
    thickness: float
) -> tuple[list, list, list]:
    """
    Transform 2D vertices to 3D top/bottom surfaces.

    Uses vectorised batch transform (numpy) when available for 5-10x speedup.

    Args:
        vertices_2d: List of 2D points
        recipe: Fold recipe from region
        thickness: Board thickness for bottom surface offset

    Returns:
        (top_vertices, bottom_vertices, normals) tuple
    """
    # Batch transform positions (vectorised with numpy when >= 8 points)
    top_vertices = transform_points_batch(vertices_2d, recipe)

    # Normals still computed per-vertex (recipe-dependent direction)
    normals = [compute_normal(v, recipe) for v in vertices_2d]

    bottom_vertices = [
        (v3d[0] - n[0] * thickness,
         v3d[1] - n[1] * thickness,
         v3d[2] - n[2] * thickness)
        for v3d, n in zip(top_vertices, normals)
    ]

    return top_vertices, bottom_vertices, normals


def get_debug_color(region_index: int) -> tuple[int, int, int]:
    """Get a debug color for a region based on its index."""
    return DEBUG_REGION_COLORS[region_index % len(DEBUG_REGION_COLORS)]


# =============================================================================
# Precomputed Data Structures (for decoupled angle updates)
# =============================================================================

@dataclass
class PrecomputedRegionData:
    """Angle-independent precomputed data for a single board region."""
    region: object  # Region object (has fold_recipe)
    subdivided_verts: list  # Subdivided 2D outline vertices
    region_holes_2d: list  # Subdivided 2D hole vertices per hole
    triangles: list  # Triangle indices from triangulation
    merged_2d: list  # Merged 2D vertices (from triangulate_with_holes), or None
    has_holes: bool
    color: tuple  # (r, g, b)


@dataclass
class PrecomputedBoardData:
    """Angle-independent precomputed data for the board mesh."""
    region_data: list  # list of PrecomputedRegionData
    thickness: float
    debug_regions: bool = False


@dataclass
class PrecomputedTraceData:
    """Angle-independent precomputed data for a single trace mesh."""
    t_values: list  # Parametric t-values along trace length
    v0: tuple  # Ribbon corner 0
    v1: tuple  # Ribbon corner 1
    v2: tuple  # Ribbon corner 2
    v3: tuple  # Ribbon corner 3
    is_back_layer: bool
    # Per-t-value data: list of (p1_2d, p2_2d, region_recipe_1, region_recipe_2)
    point_data: list


@dataclass
class PrecomputedLayerData:
    """Angle-independent precomputed data for all mesh layers."""
    board: PrecomputedBoardData = None
    traces: list = None  # list of PrecomputedTraceData
    regions: list = None  # The Region objects (needed for recipe lookup)
    markers: list = None  # FoldMarker objects (snapshot at precompute time)
    num_bend_subdivisions: int = 1
    board_thickness: float = 0.0


def snap_to_plane(vertices: list[tuple[float, float, float]],
                  normals: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """
    Snap vertices to their best-fit plane for coplanarity.

    For regions that should be flat (all normals pointing same direction),
    this ensures all vertices lie exactly on the same plane, avoiding
    numerical precision issues that cause non-flat triangles.

    Args:
        vertices: List of 3D vertices
        normals: List of surface normals at each vertex

    Returns:
        Adjusted vertices snapped to the common plane
    """
    if len(vertices) < 3:
        return vertices

    # Check if this is a flat region (all normals approximately equal)
    n0 = normals[0]
    is_flat = True
    for n in normals[1:]:
        dot = n0[0]*n[0] + n0[1]*n[1] + n0[2]*n[2]
        if abs(dot - 1.0) > 0.01:  # Normals differ by more than ~5 degrees
            is_flat = False
            break

    if not is_flat:
        # This is a curved region (in bend zone), don't snap
        return vertices

    # Average normal for the plane
    avg_nx = sum(n[0] for n in normals) / len(normals)
    avg_ny = sum(n[1] for n in normals) / len(normals)
    avg_nz = sum(n[2] for n in normals) / len(normals)

    # Normalize
    length = math.sqrt(avg_nx**2 + avg_ny**2 + avg_nz**2)
    if length < 1e-10:
        return vertices
    avg_nx /= length
    avg_ny /= length
    avg_nz /= length

    # Compute average point (centroid) to define the plane
    avg_x = sum(v[0] for v in vertices) / len(vertices)
    avg_y = sum(v[1] for v in vertices) / len(vertices)
    avg_z = sum(v[2] for v in vertices) / len(vertices)

    # Project each vertex onto the plane
    # Plane equation: n · (p - p0) = 0
    # Projection: p' = p - (n · (p - p0)) * n
    result = []
    for v in vertices:
        # Vector from centroid to vertex
        dx = v[0] - avg_x
        dy = v[1] - avg_y
        dz = v[2] - avg_z

        # Distance from plane (signed)
        dist = dx * avg_nx + dy * avg_ny + dz * avg_nz

        # Project onto plane
        result.append((
            v[0] - dist * avg_nx,
            v[1] - dist * avg_ny,
            v[2] - dist * avg_nz
        ))

    return result

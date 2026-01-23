"""
3D Model loader for KiCad component models.

Graceful degradation strategy:
1. OCC (OpenCASCADE) - Best for STEP files, may be available in KiCad environment
2. trimesh - Good fallback, supports STEP (with cascadio), WRL, and many formats
3. Native WRL parser - No dependencies, handles KiCad's VRML files
4. Placeholder boxes - Always works, used when no loader available

Handles KiCad environment variable expansion for model paths.
"""

import os
import re
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


def _get_mesh_class():
    """Lazily import Mesh to avoid circular imports."""
    try:
        from .mesh import Mesh
    except ImportError:
        from mesh import Mesh
    return Mesh


# Track available loaders
_occ_available = False
_trimesh_available = False

# Try to import OCC (OpenCASCADE)
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    _occ_available = True
except ImportError:
    pass

# Try to import trimesh
try:
    import trimesh
    _trimesh_available = True
except ImportError:
    pass


def get_loader_status() -> dict:
    """
    Get status of available model loaders.

    Returns:
        Dict with loader availability and capabilities
    """
    return {
        'occ': _occ_available,
        'trimesh': _trimesh_available,
        'native_wrl': True,  # Always available
        'step_support': _occ_available or _trimesh_available,
        'wrl_support': True,
        'best_loader': 'occ' if _occ_available else ('trimesh' if _trimesh_available else 'native_wrl'),
    }


@dataclass
class LoadedModel:
    """A loaded 3D model."""
    mesh: 'Mesh'  # Forward reference to avoid circular import
    source_path: str
    loader_used: str = "unknown"
    # Bounding box in model space (mm)
    min_point: tuple[float, float, float] = (0, 0, 0)
    max_point: tuple[float, float, float] = (0, 0, 0)


# KiCad environment variables and their typical paths
import base64
import tempfile

# Try to import zstd for embedded file decompression
_zstd_available = False
try:
    import zstandard as zstd
    _zstd_available = True
except ImportError:
    try:
        import zstd
        _zstd_available = True
    except ImportError:
        pass

# Cache for extracted embedded models (avoid re-extracting)
_embedded_model_cache = {}


def extract_embedded_model(embed_url: str, pcb, pcb_dir: str = None) -> Optional[str]:
    """
    Extract an embedded model from a PCB file.

    KiCad 8+ supports embedding 3D models directly in the PCB file.
    These are referenced with kicad-embed://filename URLs.

    Note: KiCad compresses embedded files with zstd. If zstd/zstandard
    is not installed, compressed embedded models cannot be extracted.

    Args:
        embed_url: URL like "kicad-embed://model.step"
        pcb: KiCadPCB object containing the embedded files
        pcb_dir: Directory for caching extracted files

    Returns:
        Path to extracted model file, or None if extraction fails
    """
    if not pcb:
        return None

    # Parse the URL to get the filename
    # Format: kicad-embed://filename.ext
    if not embed_url.startswith("kicad-embed://"):
        return None

    model_name = embed_url[len("kicad-embed://"):]

    # Check cache first
    cache_key = (id(pcb), model_name)
    if cache_key in _embedded_model_cache:
        cached_path = _embedded_model_cache[cache_key]
        if os.path.exists(cached_path):
            return cached_path

    # Find the embedded file in the PCB
    try:
        root = pcb.root
        for child in root.children:
            if hasattr(child, 'name') and child.name == 'embedded_files':
                for file_entry in child.children:
                    if not hasattr(file_entry, 'name') or file_entry.name != 'file':
                        continue

                    # Parse file entry
                    file_name = None
                    file_data = None

                    for item in file_entry.children:
                        if hasattr(item, 'name'):
                            if item.name == 'name' and item.children:
                                file_name = item.children[0]
                            elif item.name == 'data' and item.children:
                                file_data = item.children[0]

                    if file_name == model_name and file_data:
                        # KiCad embedded files are base64-encoded and often zstd-compressed
                        # Compressed data starts with '|' prefix
                        try:
                            is_compressed = file_data.startswith('|')
                            if is_compressed:
                                if not _zstd_available:
                                    print(f"Cannot extract embedded model {model_name}: "
                                          "zstd compression but no zstd library. "
                                          "Install with: pip install zstandard")
                                    return None
                                # Remove '|' prefix and decode
                                raw_data = base64.b64decode(file_data[1:])
                                # Decompress
                                if hasattr(zstd, 'decompress'):
                                    model_bytes = zstd.decompress(raw_data)
                                else:
                                    dctx = zstd.ZstdDecompressor()
                                    model_bytes = dctx.decompress(raw_data)
                            else:
                                model_bytes = base64.b64decode(file_data)
                        except Exception as e:
                            print(f"Failed to decode embedded model {model_name}: {e}")
                            continue

                        # Write to temp file (or cache dir)
                        ext = os.path.splitext(model_name)[1].lower()
                        if pcb_dir:
                            cache_dir = os.path.join(pcb_dir, '.kicad_embed_cache')
                            os.makedirs(cache_dir, exist_ok=True)
                            output_path = os.path.join(cache_dir, model_name)
                        else:
                            fd, output_path = tempfile.mkstemp(suffix=ext)
                            os.close(fd)

                        with open(output_path, 'wb') as f:
                            f.write(model_bytes)

                        _embedded_model_cache[cache_key] = output_path
                        return output_path

    except Exception as e:
        print(f"Failed to extract embedded model {model_name}: {e}")

    return None


KICAD_ENV_VARS = {
    "KICAD9_3DMODEL_DIR": [
        "/usr/share/kicad/3dmodels",
        "/usr/local/share/kicad/3dmodels",
        os.path.expanduser("~/.local/share/kicad/9.0/3dmodels"),
        "C:/Program Files/KiCad/9.0/share/kicad/3dmodels",
    ],
    "KICAD8_3DMODEL_DIR": [
        "/usr/share/kicad/3dmodels",
        os.path.expanduser("~/.local/share/kicad/8.0/3dmodels"),
        "C:/Program Files/KiCad/8.0/share/kicad/3dmodels",
    ],
    "KICAD7_3DMODEL_DIR": [
        "/usr/share/kicad/3dmodels",
        os.path.expanduser("~/.local/share/kicad/7.0/3dmodels"),
    ],
    "KICAD6_3DMODEL_DIR": [
        "/usr/share/kicad/3dmodels",
        os.path.expanduser("~/.local/share/kicad/6.0/3dmodels"),
    ],
    "KISYS3DMOD": [
        "/usr/share/kicad/3dmodels",
        os.path.expanduser("~/.local/share/kicad/3dmodels"),
    ],
}

# Additional search paths for user libraries
# Maps folder names (from relative paths) to actual locations
USER_3DMODEL_PATHS = {
    "00_lcsc.3dshapes": [
        os.path.expanduser("~/KICAD/kicad_libs/easyeda_library/00_lcsc.3dshapes"),
        os.path.expanduser("~/.local/share/kicad/3rdparty/00_lcsc.3dshapes"),
    ],
    "step": [
        os.path.expanduser("~/KICAD/kicad_libs/custom_kicad_library/step"),
        os.path.expanduser("~/.local/share/kicad/3rdparty/step"),
    ],
    # Add more mappings as needed
}


def expand_kicad_vars(path: str, pcb_dir: str = None, pcb=None) -> Optional[str]:
    """
    Expand KiCad environment variables in a model path.

    Args:
        path: Model path potentially containing ${VAR} syntax
        pcb_dir: Directory of the PCB file for relative path resolution
        pcb: KiCadPCB object for extracting embedded models

    Returns:
        Resolved absolute path, or None if not found
    """
    # Handle kicad-embed:// URLs (models embedded in PCB file)
    if path.startswith("kicad-embed://"):
        return extract_embedded_model(path, pcb, pcb_dir)

    # Handle relative paths
    if path.startswith("../") or path.startswith("./"):
        # First try relative to PCB directory
        if pcb_dir:
            resolved = os.path.normpath(os.path.join(pcb_dir, path))
            if os.path.exists(resolved):
                return resolved

        # Try user library search paths
        # Extract the folder name from the path (e.g., "00_lcsc.3dshapes" from "../00_lcsc.3dshapes/file.wrl")
        path_parts = path.replace("\\", "/").split("/")
        for i, part in enumerate(path_parts):
            if part in USER_3DMODEL_PATHS:
                # Get the filename portion after the matched folder
                filename = "/".join(path_parts[i + 1:])
                for search_path in USER_3DMODEL_PATHS[part]:
                    resolved = os.path.join(search_path, filename)
                    if os.path.exists(resolved):
                        return resolved
                break

        return None

    # Find and expand environment variables
    pattern = r'\$\{([^}]+)\}'
    match = re.search(pattern, path)

    if not match:
        # No variable, check if absolute path exists
        if os.path.isabs(path) and os.path.exists(path):
            return path
        return None

    var_name = match.group(1)
    var_pattern = match.group(0)

    # First try actual environment variable
    env_value = os.environ.get(var_name)
    if env_value:
        resolved = path.replace(var_pattern, env_value)
        if os.path.exists(resolved):
            return resolved

    # Try known paths for KiCad variables
    if var_name in KICAD_ENV_VARS:
        for base_path in KICAD_ENV_VARS[var_name]:
            resolved = path.replace(var_pattern, base_path)
            if os.path.exists(resolved):
                return resolved

    return None


def get_model_paths(component, pcb_dir: str = None) -> list[tuple[str, dict]]:
    """
    Get resolved model paths for a component.

    Args:
        component: ComponentGeometry with models list
        pcb_dir: Directory of the PCB file

    Returns:
        List of (resolved_path, model_info) tuples for models that exist
    """
    result = []

    for model in component.models:
        if model.hide:
            continue

        resolved = expand_kicad_vars(model.path, pcb_dir)
        if resolved:
            result.append((resolved, {
                'offset': model.offset,
                'scale': model.scale,
                'rotate': model.rotate,
            }))

    return result


# =============================================================================
# OCC (OpenCASCADE) Loader - Best for STEP files
# =============================================================================

def load_step_occ(path: str) -> Optional[LoadedModel]:
    """
    Load a STEP file using OpenCASCADE.

    Args:
        path: Path to STEP file

    Returns:
        LoadedModel or None if loading fails
    """
    if not _occ_available:
        return None

    try:
        # Read STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)

        if status != IFSelect_RetDone:
            return None

        reader.TransferRoots()
        shape = reader.OneShape()

        if shape.IsNull():
            return None

        # Mesh the shape
        mesh_algo = BRepMesh_IncrementalMesh(shape, 0.1)  # 0.1mm tolerance
        mesh_algo.Perform()

        # Extract triangles
        Mesh = _get_mesh_class()
        mesh = Mesh()
        vertex_map = {}

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)

            if triangulation is not None:
                # Get transformation
                trsf = location.Transformation()

                # Extract vertices
                nodes = triangulation.Nodes()
                for i in range(1, triangulation.NbNodes() + 1):
                    pt = nodes.Value(i)
                    pt_transformed = pt.Transformed(trsf)
                    v = (pt_transformed.X(), pt_transformed.Y(), pt_transformed.Z())
                    if v not in vertex_map:
                        vertex_map[v] = mesh.add_vertex(v)

                # Extract triangles
                triangles = triangulation.Triangles()
                for i in range(1, triangulation.NbTriangles() + 1):
                    tri = triangles.Value(i)
                    n1, n2, n3 = tri.Get()

                    pt1 = nodes.Value(n1).Transformed(trsf)
                    pt2 = nodes.Value(n2).Transformed(trsf)
                    pt3 = nodes.Value(n3).Transformed(trsf)

                    v1 = (pt1.X(), pt1.Y(), pt1.Z())
                    v2 = (pt2.X(), pt2.Y(), pt2.Z())
                    v3 = (pt3.X(), pt3.Y(), pt3.Z())

                    idx1 = vertex_map[v1]
                    idx2 = vertex_map[v2]
                    idx3 = vertex_map[v3]

                    mesh.add_triangle(idx1, idx2, idx3, (180, 180, 180))

            explorer.Next()

        if not mesh.vertices:
            return None

        # Get bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        return LoadedModel(
            mesh=mesh,
            source_path=path,
            loader_used='occ',
            min_point=(xmin, ymin, zmin),
            max_point=(xmax, ymax, zmax)
        )

    except Exception as e:
        print(f"OCC loader failed for {path}: {e}")
        return None


# =============================================================================
# trimesh Loader - Fallback for STEP and other formats
# =============================================================================

def load_model_trimesh(path: str) -> Optional[LoadedModel]:
    """
    Load a 3D model using trimesh.

    Args:
        path: Path to STEP, WRL, or other supported format

    Returns:
        LoadedModel or None if loading fails
    """
    if not _trimesh_available:
        return None

    try:
        scene = trimesh.load(path)

        # Convert scene to single mesh
        if isinstance(scene, trimesh.Scene):
            if len(scene.geometry) == 0:
                return None
            meshes = list(scene.geometry.values())
            combined = trimesh.util.concatenate(meshes)
        else:
            combined = scene

        # Convert to our Mesh format
        Mesh = _get_mesh_class()
        mesh = Mesh()
        for v in combined.vertices:
            mesh.add_vertex((float(v[0]), float(v[1]), float(v[2])))

        for face in combined.faces:
            if len(face) == 3:
                mesh.add_triangle(int(face[0]), int(face[1]), int(face[2]), (180, 180, 180))

        # Get bounding box
        bounds = combined.bounds
        min_pt = tuple(float(x) for x in bounds[0])
        max_pt = tuple(float(x) for x in bounds[1])

        return LoadedModel(
            mesh=mesh,
            source_path=path,
            loader_used='trimesh',
            min_point=min_pt,
            max_point=max_pt
        )

    except Exception as e:
        print(f"trimesh loader failed for {path}: {e}")
        return None


# =============================================================================
# Native WRL (VRML) Parser - No dependencies
# =============================================================================

def parse_wrl_native(path: str) -> Optional[LoadedModel]:
    """
    Parse a VRML 2.0 (.wrl) file natively without external dependencies.

    Handles various WRL formats:
    - KiCad standard library format
    - EasyEDA/LCSC library format
    - Other common VRML 2.0 structures

    KiCad WRL files use units where 1 unit = 0.1 inch = 2.54mm,
    so we apply a scale factor to convert to mm.

    Args:
        path: Path to WRL file

    Returns:
        LoadedModel or None if parsing fails
    """
    # KiCad WRL files use 0.1 inch units, need to convert to mm
    WRL_SCALE = 2.54  # 0.1 inch to mm

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        Mesh = _get_mesh_class()
        mesh = Mesh()

        # Strategy: Find all coordinate arrays and coordIndex arrays separately,
        # then pair them up based on their positions in the file

        # Find all point arrays: coord [DEF name] Coordinate { point [ ... ] }
        # Handles both "coord Coordinate {" and "coord DEF name Coordinate {"
        point_pattern = re.compile(
            r'coord\s+(?:DEF\s+\w+\s+)?Coordinate\s*\{\s*point\s*\[\s*([^\]]+)\s*\]',
            re.DOTALL
        )

        # Find all coordIndex arrays: coordIndex [ ... ]
        index_pattern = re.compile(
            r'coordIndex\s*\[\s*([^\]]+)\s*\]',
            re.DOTALL
        )

        point_matches = list(point_pattern.finditer(content))
        index_matches = list(index_pattern.finditer(content))

        if not point_matches or not index_matches:
            return None

        # Pair point arrays with coordIndex arrays by position
        # Each IndexedFaceSet has one coord and one coordIndex
        # They can appear in either order, so find the closest match
        pairs = []
        used_indices = set()

        for pm in point_matches:
            # Find the nearest coordIndex (before or after) this point array
            best_im = None
            best_dist = float('inf')
            for im in index_matches:
                if im.start() in used_indices:
                    continue
                # Distance can be negative (index before point) or positive (index after)
                dist = abs(im.start() - pm.end())
                if dist < best_dist:
                    best_dist = dist
                    best_im = im
            if best_im:
                pairs.append((pm.group(1), best_im.group(1)))
                used_indices.add(best_im.start())

        if not pairs:
            return None

        vertex_offset = 0

        for points_str, coord_indices_str in pairs:
            # Parse vertices: "x y z, x y z, ..."
            vertices = []
            point_parts = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', points_str)

            for i in range(0, len(point_parts) - 2, 3):
                x = float(point_parts[i]) * WRL_SCALE
                y = float(point_parts[i + 1]) * WRL_SCALE
                z = float(point_parts[i + 2]) * WRL_SCALE
                vertices.append((x, y, z))
                mesh.add_vertex((x, y, z))

            # Parse face indices: "0,1,2,-1,3,4,5,-1,..."
            index_parts = re.findall(r'-?\d+', coord_indices_str)
            current_face = []

            for idx_str in index_parts:
                idx = int(idx_str)
                if idx == -1:
                    # End of face
                    if len(current_face) >= 3:
                        # Adjust indices with offset
                        adjusted = [i + vertex_offset for i in current_face]
                        if len(adjusted) == 3:
                            mesh.add_triangle(adjusted[0], adjusted[1], adjusted[2], (180, 180, 180))
                        elif len(adjusted) == 4:
                            mesh.add_quad(adjusted[0], adjusted[1], adjusted[2], adjusted[3], (180, 180, 180))
                        else:
                            # Fan triangulation for polygons
                            for i in range(1, len(adjusted) - 1):
                                mesh.add_triangle(adjusted[0], adjusted[i], adjusted[i + 1], (180, 180, 180))
                    current_face = []
                else:
                    current_face.append(idx)

            vertex_offset += len(vertices)

        if not mesh.vertices:
            return None

        # Calculate bounding box
        xs = [v[0] for v in mesh.vertices]
        ys = [v[1] for v in mesh.vertices]
        zs = [v[2] for v in mesh.vertices]

        return LoadedModel(
            mesh=mesh,
            source_path=path,
            loader_used='native_wrl',
            min_point=(min(xs), min(ys), min(zs)),
            max_point=(max(xs), max(ys), max(zs))
        )

    except Exception as e:
        print(f"Native WRL parser failed for {path}: {e}")
        return None


# =============================================================================
# Main Loader Function - Graceful Degradation
# =============================================================================

def load_model(path: str) -> Optional[LoadedModel]:
    """
    Load a 3D model from file using best available loader.

    Graceful degradation order:
    1. For STEP files: OCC → trimesh → try WRL fallback
    2. For WRL files: trimesh → native parser
    3. Returns None if all loaders fail (caller uses placeholder)

    Args:
        path: Path to model file

    Returns:
        LoadedModel or None if loading fails
    """
    ext = os.path.splitext(path)[1].lower()

    # STEP files
    if ext in ('.step', '.stp'):
        # Try OCC first (best for STEP)
        if _occ_available:
            result = load_step_occ(path)
            if result:
                return result

        # Try trimesh (needs cascadio for STEP)
        if _trimesh_available:
            result = load_model_trimesh(path)
            if result:
                return result

        # Fallback: try WRL version (KiCad often provides both)
        wrl_path = os.path.splitext(path)[0] + '.wrl'
        if os.path.exists(wrl_path):
            result = parse_wrl_native(wrl_path)
            if result:
                return result

        return None

    # WRL/VRML files
    if ext in ('.wrl', '.vrml'):
        # Try trimesh first (better material handling)
        if _trimesh_available:
            result = load_model_trimesh(path)
            if result:
                return result

        # Fall back to native parser
        result = parse_wrl_native(path)
        if result:
            return result

        return None

    # Other formats - try trimesh
    if _trimesh_available:
        return load_model_trimesh(path)

    return None


# =============================================================================
# Transform Utilities
# =============================================================================

def apply_model_transform(
    mesh: 'Mesh',
    component_pos: tuple[float, float],
    component_angle: float,
    model_offset: tuple[float, float, float],
    model_scale: tuple[float, float, float],
    model_rotate: tuple[float, float, float],
    pcb_thickness: float = 0,
    is_back_layer: bool = False
) -> 'Mesh':
    """
    Apply KiCad model transforms to a mesh.

    Transform order (KiCad convention):
    1. Scale the model
    2. Rotate the model (model's own rotation)
    3. Translate by model offset
    4. Rotate by component angle
    5. Translate to component position
    6. If back layer, mirror and offset

    Args:
        mesh: Source mesh to transform
        component_pos: (x, y) position of component
        component_angle: Component rotation in degrees
        model_offset: Model offset in mm
        model_scale: Model scale factors
        model_rotate: Model rotation in degrees (x, y, z)
        pcb_thickness: Board thickness for back layer positioning
        is_back_layer: Whether component is on back layer

    Returns:
        New transformed mesh
    """
    Mesh = _get_mesh_class()
    result = Mesh()

    # Pre-compute rotation matrices
    def rot_x(angle):
        c, s = math.cos(angle), math.sin(angle)
        return lambda x, y, z: (x, y * c - z * s, y * s + z * c)

    def rot_y(angle):
        c, s = math.cos(angle), math.sin(angle)
        return lambda x, y, z: (x * c + z * s, y, -x * s + z * c)

    def rot_z(angle):
        c, s = math.cos(angle), math.sin(angle)
        return lambda x, y, z: (x * c - y * s, x * s + y * c, z)

    # Model rotations (convert to radians)
    rx = rot_x(math.radians(model_rotate[0]))
    ry = rot_y(math.radians(model_rotate[1]))
    rz = rot_z(math.radians(model_rotate[2]))

    # Component rotation
    comp_rz = rot_z(math.radians(-component_angle))  # KiCad uses clockwise positive

    for v in mesh.vertices:
        x, y, z = v

        # 1. Scale
        x *= model_scale[0]
        y *= model_scale[1]
        z *= model_scale[2]

        # 2. Model rotation (ZYX order)
        x, y, z = rz(x, y, z)
        x, y, z = ry(x, y, z)
        x, y, z = rx(x, y, z)

        # 3. Model offset
        x += model_offset[0]
        y += model_offset[1]
        z += model_offset[2]

        # 4. Component rotation
        x, y, z = comp_rz(x, y, z)

        # 5. Back layer handling
        if is_back_layer:
            z = -z - pcb_thickness

        # 6. Component position
        x += component_pos[0]
        y += component_pos[1]

        result.add_vertex((x, y, z))

    # Copy faces and colors
    for i, face in enumerate(mesh.faces):
        color = mesh.colors[i] if i < len(mesh.colors) else (180, 180, 180)
        if len(face) == 3:
            # Reverse winding for back layer
            if is_back_layer:
                result.add_triangle(face[0], face[2], face[1], color)
            else:
                result.add_triangle(face[0], face[1], face[2], color)
        elif len(face) == 4:
            if is_back_layer:
                result.add_quad(face[0], face[3], face[2], face[1], color)
            else:
                result.add_quad(face[0], face[1], face[2], face[3], color)

    return result


def create_component_model_mesh(
    component,
    pcb_dir: str = None,
    pcb_thickness: float = 0
) -> Optional['Mesh']:
    """
    Create a mesh for a component from its 3D model.

    Uses graceful degradation to load models with available loaders.

    Args:
        component: ComponentGeometry with models list
        pcb_dir: Directory of PCB file for relative paths
        pcb_thickness: Board thickness for positioning

    Returns:
        Transformed Mesh or None if no model could be loaded
    """
    model_paths = get_model_paths(component, pcb_dir)

    for resolved_path, model_info in model_paths:
        loaded = load_model(resolved_path)
        if loaded:
            # Apply transforms
            is_back = component.layer == "B.Cu"
            transformed = apply_model_transform(
                loaded.mesh,
                component.center,
                component.angle,
                model_info['offset'],
                model_info['scale'],
                model_info['rotate'],
                pcb_thickness,
                is_back
            )
            return transformed

    return None

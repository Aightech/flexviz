"""
3D Model loader for KiCad component models.

Supports:
- STEP files (.step, .stp)
- WRL/VRML files (.wrl)

Handles KiCad environment variable expansion for model paths.
"""

import os
import re
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from .mesh import Mesh
except ImportError:
    from mesh import Mesh


@dataclass
class LoadedModel:
    """A loaded 3D model."""
    mesh: Mesh
    source_path: str
    # Bounding box in model space
    min_point: tuple[float, float, float] = (0, 0, 0)
    max_point: tuple[float, float, float] = (0, 0, 0)


# KiCad environment variables and their typical paths
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


def expand_kicad_vars(path: str, pcb_dir: str = None) -> Optional[str]:
    """
    Expand KiCad environment variables in a model path.

    Args:
        path: Model path potentially containing ${VAR} syntax
        pcb_dir: Directory of the PCB file for relative path resolution

    Returns:
        Resolved absolute path, or None if not found
    """
    # Handle relative paths
    if path.startswith("../") or path.startswith("./"):
        if pcb_dir:
            resolved = os.path.normpath(os.path.join(pcb_dir, path))
            if os.path.exists(resolved):
                return resolved
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


# Optional: trimesh-based loading
_trimesh_available = False
try:
    import trimesh
    _trimesh_available = True
except ImportError:
    pass


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
            min_point=min_pt,
            max_point=max_pt
        )

    except Exception as e:
        print(f"Warning: Could not load model {path}: {e}")
        return None


def load_model(path: str) -> Optional[LoadedModel]:
    """
    Load a 3D model from file.

    Tries available loaders in order of preference.

    Args:
        path: Path to model file

    Returns:
        LoadedModel or None if loading fails
    """
    # Try trimesh first (supports many formats including STEP with cascadio)
    if _trimesh_available:
        result = load_model_trimesh(path)
        if result:
            return result

    # No loader available
    return None


def create_component_model_mesh(
    component,
    pcb_dir: str = None,
    fallback_box: bool = True
) -> Optional[Mesh]:
    """
    Create a mesh for a component from its 3D model.

    Args:
        component: ComponentGeometry
        pcb_dir: Directory of PCB file for relative paths
        fallback_box: If True, return placeholder box if model can't be loaded

    Returns:
        Mesh or None
    """
    # Try to load actual 3D model
    model_paths = get_model_paths(component, pcb_dir)

    for resolved_path, model_info in model_paths:
        loaded = load_model(resolved_path)
        if loaded:
            # Apply transforms: scale, rotate, offset
            mesh = loaded.mesh

            # TODO: Apply model_info transforms
            # For now, return the mesh as-is (transforms need matrix operations)

            return mesh

    # Fallback to placeholder box
    if fallback_box:
        return None  # Let caller use default box

    return None


def apply_model_transform(
    mesh: Mesh,
    component_pos: tuple[float, float],
    component_angle: float,
    model_offset: tuple[float, float, float],
    model_scale: tuple[float, float, float],
    model_rotate: tuple[float, float, float],
    pcb_thickness: float = 0,
    is_back_layer: bool = False
) -> Mesh:
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

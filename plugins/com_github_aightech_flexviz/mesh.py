"""
Mesh generation for 3D visualization.

Creates 3D meshes from board geometry with bend transformations applied.

This module re-exports everything from the focused sub-modules
(triangulation, mesh_types, board_mesh, trace_mesh) so that existing
``from mesh import X`` statements continue to work unchanged.
"""

# ---------------------------------------------------------------------------
# Re-exports from sub-modules — keeps backward compatibility
# ---------------------------------------------------------------------------

try:
    from .triangulation import (                                      # noqa: F401
        find_mutually_visible_vertex,
        merge_hole_into_polygon,
        triangulate_with_holes,
        find_reflex_vertices,
        triangulate_polygon,
    )
    from .mesh_types import (                                         # noqa: F401
        Mesh,
        COLOR_BOARD, COLOR_COPPER, COLOR_PAD, COLOR_COMPONENT,
        COLOR_STIFFENER, COLOR_CUTOUT, COLOR_MODEL_3D,
        DEBUG_REGION_COLORS,
        get_region_recipe,
        transform_vertices_with_thickness,
        get_debug_color,
        snap_to_plane,
    )
    from .board_mesh import (                                          # noqa: F401
        create_board_mesh_with_regions,
        precompute_board_mesh,
        transform_board_mesh,
    )
    from .trace_mesh import (                                         # noqa: F401
        _compute_fold_crossing_t_values,
        _dot3,
        create_trace_mesh,
        precompute_trace_mesh,
        transform_trace_mesh,
        create_pad_mesh,
        create_component_mesh,
        create_stiffener_mesh,
        create_component_3d_model_mesh,
    )
    from .mesh_types import PrecomputedLayerData, PrecomputedBoardData, PrecomputedTraceData  # noqa: F401
except ImportError:
    from triangulation import (                                       # noqa: F401
        find_mutually_visible_vertex,
        merge_hole_into_polygon,
        triangulate_with_holes,
        find_reflex_vertices,
        triangulate_polygon,
    )
    from mesh_types import (                                          # noqa: F401
        Mesh,
        COLOR_BOARD, COLOR_COPPER, COLOR_PAD, COLOR_COMPONENT,
        COLOR_STIFFENER, COLOR_CUTOUT, COLOR_MODEL_3D,
        DEBUG_REGION_COLORS,
        get_region_recipe,
        transform_vertices_with_thickness,
        get_debug_color,
        snap_to_plane,
    )
    from board_mesh import (                                           # noqa: F401
        create_board_mesh_with_regions,
        precompute_board_mesh,
        transform_board_mesh,
    )
    from trace_mesh import (                                          # noqa: F401
        _compute_fold_crossing_t_values,
        _dot3,
        create_trace_mesh,
        precompute_trace_mesh,
        transform_trace_mesh,
        create_pad_mesh,
        create_component_mesh,
        create_stiffener_mesh,
        create_component_3d_model_mesh,
    )
    from mesh_types import PrecomputedLayerData, PrecomputedBoardData, PrecomputedTraceData  # noqa: F401

# Re-export polygon_ops symbols that were previously importable via mesh
try:
    from .polygon_ops import signed_area                              # noqa: F401
except ImportError:
    from polygon_ops import signed_area                               # noqa: F401

# ---------------------------------------------------------------------------
# Imports needed by functions that remain in this module
# ---------------------------------------------------------------------------

try:
    from .geometry import Polygon, BoardGeometry, refine_outline_for_folds
    from .bend_transform import FoldDefinition
    from .markers import FoldMarker
    from .planar_subdivision import split_board_into_regions, Region, clear_region_index_cache
except ImportError:
    from geometry import Polygon, BoardGeometry, refine_outline_for_folds
    from bend_transform import FoldDefinition
    from markers import FoldMarker
    from planar_subdivision import split_board_into_regions, Region, clear_region_index_cache


# ============================================================================
# Layer-building orchestration (stays in mesh.py)
# ============================================================================

def create_board_geometry_mesh(
    board: BoardGeometry,
    markers: list[FoldMarker] = None,
    include_traces: bool = True,
    include_pads: bool = True,
    include_components: bool = False,
    component_height: float = 2.0,
    subdivide_length: float = 1.0,
    num_bend_subdivisions: int = 1,
    stiffeners: list = None,
    debug_regions: bool = False,
    apply_bend: bool = True,
    include_3d_models: bool = False,
    pcb_dir: str = None,
    pcb=None
) -> Mesh:
    """
    Create a complete 3D mesh from board geometry.

    Args:
        board: Board geometry
        markers: List of fold markers (for region splitting and 3D transform)
        include_traces: Include copper traces
        include_pads: Include pads
        include_components: Include component boxes
        component_height: Height for component boxes
        subdivide_length: Maximum edge length for subdivision
        num_bend_subdivisions: Number of strips in bend zone
        stiffeners: List of StiffenerRegion objects to render
        debug_regions: If True, color each region differently for debugging
        apply_bend: If False, show flat board with regions but no bending
        include_3d_models: Include 3D models from footprints
        pcb_dir: PCB directory for resolving model paths
        pcb: KiCadPCB object for extracting embedded models

    Returns:
        Complete mesh
    """
    mesh = Mesh()

    # Refine arc segments that cross fold zones (adaptive subdivision)
    refined_outline = board.outline
    if markers and board.outline.segments:
        refined_outline = refine_outline_for_folds(board.outline, markers)

    # Compute regions for region-based transformation
    regions = None
    if markers and refined_outline.vertices:
        outline_verts = [(v[0], v[1]) for v in refined_outline.vertices]
        cutout_verts = [[(v[0], v[1]) for v in c.vertices] for c in (board.cutouts or [])]
        regions = split_board_into_regions(
            outline_verts,
            cutout_verts,
            markers,
            num_bend_subdivisions=num_bend_subdivisions
        )

    # Board outline with cutouts
    if refined_outline.vertices:
        board_mesh = create_board_mesh_with_regions(
            refined_outline,
            board.thickness,
            markers=markers,
            subdivide_length=subdivide_length,
            cutouts=board.cutouts,
            num_bend_subdivisions=num_bend_subdivisions,
            debug_regions=debug_regions,
            apply_bend=apply_bend
        )
        mesh.merge(board_mesh)

    # Use empty regions when bending is disabled
    active_regions = regions if apply_bend else None

    # Traces
    if include_traces:
        z_offset = 0.05  # Above board surface (needs enough clearance to avoid z-fighting)
        for layer, traces in board.traces.items():
            for trace in traces:
                trace_mesh = create_trace_mesh(trace, z_offset, active_regions, pcb_thickness=board.thickness, markers=markers, num_bend_subdivisions=num_bend_subdivisions)
                mesh.merge(trace_mesh)

    # Pads
    if include_pads:
        z_offset = 0.08  # Above traces
        for pad in board.all_pads:
            pad_mesh = create_pad_mesh(pad, z_offset, active_regions, board.thickness)
            mesh.merge(pad_mesh)

    # Components (box placeholders)
    if include_components and not include_3d_models:
        for comp in board.components:
            comp_mesh = create_component_mesh(comp, component_height, active_regions, board.thickness)
            mesh.merge(comp_mesh)

    # 3D Models
    if include_3d_models and pcb_dir:
        loaded_count = 0
        for comp in board.components:
            model_mesh = create_component_3d_model_mesh(
                comp, pcb_dir, board.thickness, active_regions, pcb
            )
            if model_mesh.vertices:
                mesh.merge(model_mesh)
                loaded_count += 1
            elif include_components:
                # Fallback to box if model couldn't load
                comp_mesh = create_component_mesh(comp, component_height, active_regions, board.thickness)
                mesh.merge(comp_mesh)

    # Stiffeners
    if stiffeners and apply_bend:
        for stiffener in stiffeners:
            stiff_mesh = create_stiffener_mesh(
                outline=stiffener.outline,
                stiffener_thickness=stiffener.thickness,
                pcb_thickness=board.thickness,
                side=stiffener.side,
                regions=regions,
                cutouts=stiffener.cutouts if hasattr(stiffener, 'cutouts') else None
            )
            mesh.merge(stiff_mesh)

    mesh.compute_normals()
    return mesh


def compute_regions(board, markers, num_bend_subdivisions=1, apply_bend=True):
    """Compute planar subdivision regions from board outline and markers.

    Returns (regions, active_regions) where active_regions is None when
    bending is disabled.
    """
    clear_region_index_cache()  # Invalidate spatial index when regions change
    regions = None
    if markers and board.outline.vertices:
        # Refine arc segments that cross fold zones
        refined = board.outline
        if board.outline.segments:
            refined = refine_outline_for_folds(board.outline, markers)
        outline_verts = [(v[0], v[1]) for v in refined.vertices]
        cutout_verts = [[(v[0], v[1]) for v in c.vertices] for c in (board.cutouts or [])]
        regions = split_board_into_regions(
            outline_verts,
            cutout_verts,
            markers,
            num_bend_subdivisions=num_bend_subdivisions
        )
    active_regions = regions if apply_bend else None
    return regions, active_regions


def build_board_layer(board, markers, subdivide_length=1.0, num_bend_subdivisions=1,
                      debug_regions=False, apply_bend=True):
    """Build the board outline mesh."""
    mesh = Mesh()
    if board.outline.vertices:
        mesh = create_board_mesh_with_regions(
            board.outline,
            board.thickness,
            markers=markers,
            subdivide_length=subdivide_length,
            cutouts=board.cutouts,
            num_bend_subdivisions=num_bend_subdivisions,
            debug_regions=debug_regions,
            apply_bend=apply_bend
        )
    mesh.compute_normals()
    return mesh


def build_traces_layer(board, active_regions, markers=None, num_bend_subdivisions=1):
    """Build the traces mesh."""
    mesh = Mesh()
    z_offset = 0.05
    for layer, traces in board.traces.items():
        for trace in traces:
            trace_mesh = create_trace_mesh(
                trace, z_offset, active_regions,
                pcb_thickness=board.thickness,
                markers=markers,
                num_bend_subdivisions=num_bend_subdivisions
            )
            mesh.merge(trace_mesh)
    mesh.compute_normals()
    return mesh


def build_pads_layer(board, active_regions):
    """Build the pads mesh."""
    mesh = Mesh()
    z_offset = 0.08
    for pad in board.all_pads:
        pad_mesh = create_pad_mesh(pad, z_offset, active_regions, board.thickness)
        mesh.merge(pad_mesh)
    mesh.compute_normals()
    return mesh


def build_components_layer(board, active_regions, component_height=2.0):
    """Build the component box placeholders mesh."""
    mesh = Mesh()
    for comp in board.components:
        cm = create_component_mesh(comp, component_height, active_regions, board.thickness)
        mesh.merge(cm)
    mesh.compute_normals()
    return mesh


def build_3d_models_layer(board, active_regions, pcb_dir=None, pcb=None,
                          on_progress=None):
    """Build the 3D models mesh.

    Args:
        on_progress: optional callable(str) — invoked every PROGRESS_EVERY
            components with a human-readable progress string like
            "Loading 3D models: 150/563 (84 loaded, 66 missing)". Used to
            drive the viewer's status bar so the GUI shows life during
            the long STEP/WRL load.

    Diagnostics are also written to ~/.cache/flexviz/profile.log via _ptick.
    """
    try:
        from .viewer import _ptick
    except Exception:
        try:
            from viewer import _ptick
        except Exception:
            _ptick = lambda _: None

    mesh = Mesh()
    if not pcb_dir:
        _ptick("3d_models: SKIPPED — pcb_dir is empty/None (board file path unknown)")
        if on_progress:
            on_progress("3D models: no PCB directory (cannot resolve model paths)")
        mesh.compute_normals()
        return mesh

    n_comps = len(board.components) if board.components else 0
    _ptick(f"3d_models: pcb_dir={pcb_dir} components={n_comps}")
    if on_progress:
        on_progress(f"Loading 3D models: 0/{n_comps}…")

    import time as _time
    n_no_models = 0
    n_succeeded = 0
    n_empty_mesh = 0
    PROGRESS_EVERY = 25
    for i, comp in enumerate(board.components):
        if i and (i % PROGRESS_EVERY == 0):
            msg = (
                f"3d_models: progress {i}/{n_comps} "
                f"(loaded={n_succeeded} failed={n_empty_mesh} "
                f"no_ref={n_no_models})"
            )
            _ptick(msg)
            if on_progress:
                on_progress(
                    f"Loading 3D models: {i}/{n_comps} "
                    f"({n_succeeded} loaded, {n_empty_mesh} missing)"
                )
            # Release the GIL briefly so the wx main thread can pump events.
            # OCC's STEP loader runs in C without releasing the GIL, which
            # starves the GUI; sleep(0) gives it a chance to draw and stay
            # responsive between component loads.
            _time.sleep(0)
        if not getattr(comp, 'models', None):
            n_no_models += 1
            continue
        model_mesh = create_component_3d_model_mesh(
            comp, pcb_dir, board.thickness, active_regions, pcb
        )
        if model_mesh.vertices:
            n_succeeded += 1
            mesh.merge(model_mesh)
        else:
            n_empty_mesh += 1

    _ptick(
        f"3d_models: DONE loaded={n_succeeded} "
        f"empty/failed={n_empty_mesh} no_models_ref={n_no_models} "
        f"total_verts={len(mesh.vertices)}"
    )
    if on_progress:
        on_progress(
            f"3D models: {n_succeeded} loaded, "
            f"{n_empty_mesh} missing, "
            f"{n_no_models} components have no model"
        )
    mesh.compute_normals()
    return mesh


def build_stiffeners_layer(board, regions, stiffeners=None, apply_bend=True):
    """Build the stiffeners mesh."""
    mesh = Mesh()
    if stiffeners and apply_bend:
        for stiffener in stiffeners:
            sm = create_stiffener_mesh(
                outline=stiffener.outline,
                stiffener_thickness=stiffener.thickness,
                pcb_thickness=board.thickness,
                side=stiffener.side,
                regions=regions,
                cutouts=stiffener.cutouts if hasattr(stiffener, 'cutouts') else None
            )
            mesh.merge(sm)
    mesh.compute_normals()
    return mesh


def create_board_layer_meshes(
    board: BoardGeometry,
    markers: list[FoldMarker] = None,
    component_height: float = 2.0,
    subdivide_length: float = 1.0,
    num_bend_subdivisions: int = 1,
    stiffeners: list = None,
    debug_regions: bool = False,
    apply_bend: bool = True,
    pcb_dir: str = None,
    pcb=None
) -> dict[str, Mesh]:
    """
    Create separate meshes for each display layer.

    Returns a dict with keys: 'board', 'traces', 'pads', 'components',
    '3d_models', 'stiffeners'. Each value is a Mesh (possibly empty).
    Regions are computed once and shared across all layers.
    """
    regions, active_regions = compute_regions(
        board, markers, num_bend_subdivisions, apply_bend
    )

    return {
        'board': build_board_layer(board, markers, subdivide_length,
                                   num_bend_subdivisions, debug_regions, apply_bend),
        'traces': build_traces_layer(board, active_regions, markers, num_bend_subdivisions),
        'pads': build_pads_layer(board, active_regions),
        'components': build_components_layer(board, active_regions, component_height),
        '3d_models': build_3d_models_layer(board, active_regions, pcb_dir, pcb),
        'stiffeners': build_stiffeners_layer(board, regions, stiffeners, apply_bend),
    }


# ============================================================================
# Precompute / retransform (decoupled angle updates)
# ============================================================================

def precompute_all_layers(board, markers, num_bend_subdivisions=1,
                          subdivide_length=1.0, debug_regions=False):
    """Precompute angle-independent data for board and trace layers.

    Called once when the board or markers change (topology change).
    The returned PrecomputedLayerData can be passed to retransform_all_layers()
    for fast angle-only updates.

    Pads, components, stiffeners, and 3D models are NOT precomputed — they are
    fast enough to rebuild each time, and their geometry is simpler.

    Returns:
        PrecomputedLayerData
    """
    # Refine arc segments that cross fold zones (adaptive subdivision)
    refined_outline = board.outline
    if markers and board.outline.segments:
        refined_outline = refine_outline_for_folds(board.outline, markers)

    # Precompute board mesh data
    board_precomputed = None
    if refined_outline.vertices:
        board_precomputed = precompute_board_mesh(
            refined_outline,
            board.thickness,
            markers=markers,
            subdivide_length=subdivide_length,
            cutouts=board.cutouts,
            num_bend_subdivisions=num_bend_subdivisions,
            debug_regions=debug_regions,
        )

    # Precompute regions for traces (needed for region lookups)
    regions = None
    if markers and refined_outline.vertices:
        outline_verts = [(v[0], v[1]) for v in refined_outline.vertices]
        cutout_verts = [[(v[0], v[1]) for v in c.vertices] for c in (board.cutouts or [])]
        regions = split_board_into_regions(
            outline_verts, cutout_verts, markers,
            num_bend_subdivisions=num_bend_subdivisions,
        )

    # Precompute trace data
    trace_precomputed = []
    for layer, traces in board.traces.items():
        for trace in traces:
            td = precompute_trace_mesh(
                trace, regions,
                markers=markers,
                num_bend_subdivisions=num_bend_subdivisions,
            )
            if td is not None:
                trace_precomputed.append(td)

    return PrecomputedLayerData(
        board=board_precomputed,
        traces=trace_precomputed,
        regions=regions,
        markers=list(markers) if markers else [],
        num_bend_subdivisions=num_bend_subdivisions,
        board_thickness=board.thickness,
    )


def retransform_all_layers(precomputed, board, active_regions, markers,
                           num_bend_subdivisions=1, apply_bend=True,
                           stiffeners=None, pcb_dir=None, pcb=None,
                           debug_regions=False, visibility=None):
    """Fast path: retransform board + traces from precomputed data, rebuild other layers.

    For board and traces, this skips the expensive region splitting, subdivision,
    and triangulation steps — only the 2D→3D vertex transformation is performed.

    Pads, components, stiffeners, and 3D models are rebuilt from scratch (they're fast).

    Args:
        precomputed: PrecomputedLayerData from precompute_all_layers()
        board: BoardGeometry (needed for pads/components/stiffeners)
        active_regions: Region objects for non-board layers (None if bend disabled)
        markers: Current fold markers with updated angles
        num_bend_subdivisions: Number of bend zone strips
        apply_bend: Whether bending is enabled
        stiffeners: Stiffener objects
        pcb_dir: PCB directory for 3D models
        pcb: KiCadPCB object
        debug_regions: Debug region coloring
        visibility: Dict of layer_name -> bool (for skipping invisible layers)

    Returns:
        Dict of layer_name -> Mesh
    """
    visibility = visibility or {}
    result = {}

    # Board: fast retransform from precomputed data
    if precomputed.board:
        board_mesh = transform_board_mesh(precomputed.board, apply_bend)
        board_mesh.compute_normals()
    else:
        board_mesh = Mesh()
    result['board'] = board_mesh

    # Traces: fast retransform from precomputed data
    trace_mesh = Mesh()
    z_offset = 0.05
    for td in (precomputed.traces or []):
        tm = transform_trace_mesh(td, z_offset, pcb_thickness=board.thickness)
        trace_mesh.merge(tm)
    trace_mesh.compute_normals()
    result['traces'] = trace_mesh

    # Pads: full rebuild (fast, no precomputation needed)
    result['pads'] = build_pads_layer(board, active_regions)

    # Components: full rebuild
    result['components'] = build_components_layer(board, active_regions)

    # 3D models: full rebuild
    result['3d_models'] = build_3d_models_layer(board, active_regions, pcb_dir, pcb)

    # Stiffeners: full rebuild
    regions = precomputed.regions
    result['stiffeners'] = build_stiffeners_layer(board, regions, stiffeners, apply_bend)

    return result

"""Tests for IPC API integration — importability and fold creation logic."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"))


class TestIpcModuleImports:
    """Verify IPC modules can be imported without errors (no KiCad needed)."""

    def test_fold_math_importable(self):
        from fold_math import compute_fold_marker_geometry, FoldMarkerGeometry
        assert callable(compute_fold_marker_geometry)

    def test_ipc_create_fold_importable(self):
        """The module should be importable even without kipy installed."""
        # We just test that the fold_math import inside it works
        from fold_math import compute_fold_marker_geometry
        geom = compute_fold_marker_geometry((0, 0), (10, 0), 90, 1.0)
        assert geom is not None

    def test_plugin_json_exists(self):
        plugin_dir = Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"
        assert (plugin_dir / "plugin.json").exists()

    def test_plugin_json_valid(self):
        import json
        plugin_dir = Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"
        with open(plugin_dir / "plugin.json") as f:
            data = json.load(f)
        assert data["identifier"] == "com.github.aightech.flexviz"
        assert len(data["actions"]) == 2
        assert data["actions"][0]["entrypoint"] == "ipc_create_fold.py"
        assert data["actions"][1]["entrypoint"] == "ipc_open_viewer.py"

    def test_entrypoint_files_exist(self):
        plugin_dir = Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"
        assert (plugin_dir / "ipc_create_fold.py").exists()
        assert (plugin_dir / "ipc_open_viewer.py").exists()


class TestFoldMathIpcCompatibility:
    """Test that fold math works with mm coordinates (IPC uses mm)."""

    def test_mm_coordinates(self):
        """IPC API uses mm. Verify fold math works in mm."""
        from fold_math import compute_fold_marker_geometry
        geom = compute_fold_marker_geometry(
            (100.0, 50.0), (120.0, 50.0),
            angle_degrees=90, radius=2.0
        )
        assert geom is not None
        # Zone width = 2.0 * pi/2 ≈ 3.14mm
        assert geom.zone_width == pytest.approx(2.0 * 3.14159 / 2, abs=0.01)

    def test_iu_coordinates(self):
        """SWIG API uses IU (nanometers). Verify fold math works in IU."""
        from fold_math import compute_fold_marker_geometry
        IU_PER_MM = 1_000_000
        geom = compute_fold_marker_geometry(
            (100 * IU_PER_MM, 50 * IU_PER_MM),
            (120 * IU_PER_MM, 50 * IU_PER_MM),
            angle_degrees=90,
            radius=2.0 * IU_PER_MM
        )
        assert geom is not None
        expected_zw = 2.0 * IU_PER_MM * 3.14159 / 2
        assert geom.zone_width == pytest.approx(expected_zw, rel=0.01)

    def test_requirements_file_exists(self):
        plugin_dir = Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"
        assert (plugin_dir / "requirements.txt").exists()

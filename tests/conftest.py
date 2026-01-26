"""Pytest fixtures for kicad_flex_viewer tests."""

import pytest
import sys
from pathlib import Path

# Add plugin directory to path for imports
PLUGIN_DIR = Path(__file__).parent.parent / "plugins" / "com_github_aightech_flexviz"
sys.path.insert(0, str(PLUGIN_DIR))


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def minimal_pcb_path(test_data_dir) -> Path:
    """Return path to minimal test PCB."""
    return test_data_dir / "minimal.kicad_pcb"


@pytest.fixture
def rectangle_pcb_path(test_data_dir) -> Path:
    """Return path to rectangle test PCB."""
    return test_data_dir / "rectangle.kicad_pcb"


@pytest.fixture
def fold_pcb_path(test_data_dir) -> Path:
    """Return path to PCB with fold markers."""
    return test_data_dir / "with_fold.kicad_pcb"


@pytest.fixture
def minimal_pcb_content() -> str:
    """Return minimal PCB content as string."""
    return '''(kicad_pcb
  (version 20240108)
  (generator "test")
  (general
    (thickness 1.6)
  )
  (layers
    (0 "F.Cu" signal)
  )
)'''


@pytest.fixture
def simple_sexpr() -> str:
    """Return simple S-expression for testing."""
    return '(test (a 1) (b "hello") (c 3.14))'

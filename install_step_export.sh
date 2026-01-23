#!/bin/bash
# Install build123d for STEP export support
# This installs into the project's virtual environment (NOT system Python)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== Flex Viewer STEP Export Setup ==="
echo ""
echo "This will install build123d into the project's virtual environment."
echo "STEP export runs from the command line, not inside KiCad."
echo ""

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install build123d
echo "Installing build123d..."
pip install --upgrade pip
pip install build123d

# Verify
echo ""
echo "Verifying installation..."
python -c "from build123d import Box; print('SUCCESS: build123d is installed')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To export STEP files, use:"
echo "  source venv/bin/activate"
echo "  python step_export_cli.py your_board.kicad_pcb output.step"
echo ""
echo "See docs/STEP_EXPORT.md for more options."

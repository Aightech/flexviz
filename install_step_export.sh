#!/bin/bash
# Install build123d for STEP export support
# Run this script to enable STEP export in the KiCad Flex Viewer plugin

set -e

echo "=== Flex Viewer STEP Export Setup ==="
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo ""

case $OS in
    linux)
        echo "Installing build123d for system Python (used by KiCad on Linux)..."
        echo ""

        # Check if we're on a PEP 668 system (Debian 12+, Ubuntu 23.04+)
        if pip3 install --user --dry-run build123d 2>&1 | grep -q "externally-managed"; then
            echo "Detected PEP 668 protected environment (modern Debian/Ubuntu)"
            echo "Using --break-system-packages flag for user install..."
            echo ""
            echo "Running: pip3 install --user --break-system-packages build123d"
            pip3 install --user --break-system-packages build123d
        else
            echo "Running: pip3 install --user build123d"
            pip3 install --user build123d
        fi

        echo ""
        echo "Verifying installation..."
        python3 -c "from build123d import Box; print('SUCCESS: build123d is installed')"
        ;;

    macos)
        KICAD_PYTHON="/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3"
        if [ -f "$KICAD_PYTHON" ]; then
            echo "Found KiCad Python at: $KICAD_PYTHON"
            echo ""
            echo "Installing build123d..."
            "$KICAD_PYTHON" -m pip install --user build123d
            echo ""
            echo "Verifying installation..."
            "$KICAD_PYTHON" -c "from build123d import Box; print('SUCCESS: build123d is installed')"
        else
            echo "ERROR: KiCad Python not found at expected location"
            echo "Please install manually. See docs/STEP_EXPORT.md"
            exit 1
        fi
        ;;

    windows)
        echo "On Windows, please run these commands in an Administrator terminal:"
        echo ""
        echo 'For KiCad 9.0:'
        echo '  "C:\Program Files\KiCad\9.0\bin\python.exe" -m pip install build123d'
        echo ""
        echo 'For KiCad 8.0:'
        echo '  "C:\Program Files\KiCad\8.0\bin\python.exe" -m pip install build123d'
        echo ""
        echo "See docs/STEP_EXPORT.md for detailed instructions."
        ;;

    *)
        echo "Unknown OS. Please see docs/STEP_EXPORT.md for manual installation instructions."
        exit 1
        ;;
esac

echo ""
echo "=== Setup Complete ==="
echo "Restart KiCad and try the STEP export feature."

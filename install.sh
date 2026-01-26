#!/bin/bash
# KiCad Flex Viewer - Installation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_NAME="com_github_aightech_flexviz"
PLUGIN_SRC="$SCRIPT_DIR/plugins/$PLUGIN_NAME"

# Detect KiCad version and plugin directory
detect_kicad_plugin_dir() {
    # Check for common KiCad plugin locations
    # KiCad 9.0+ uses 3rdparty/plugins instead of scripting/plugins
    local KICAD_DIRS=(
        "$HOME/.local/share/kicad/9.0/3rdparty/plugins"
        "$HOME/.local/share/kicad/8.0/3rdparty/plugins"
        "$HOME/.local/share/kicad/8.0/scripting/plugins"
        "$HOME/.local/share/kicad/7.0/scripting/plugins"
        "$HOME/.kicad/scripting/plugins"
        "$HOME/.kicad_plugins"
    )

    for dir in "${KICAD_DIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            echo "$dir"
            return 0
        fi
    done

    # Default to KiCad 9.0 location
    echo "$HOME/.local/share/kicad/9.0/3rdparty/plugins"
}

PLUGIN_DIR=$(detect_kicad_plugin_dir)
TARGET_DIR="$PLUGIN_DIR/$PLUGIN_NAME"

echo "KiCad Flex Viewer - Installation"
echo "================================"
echo ""
echo "Source directory: $PLUGIN_SRC"
echo "Target directory: $TARGET_DIR"
echo ""

# Check source exists
if [[ ! -d "$PLUGIN_SRC" ]]; then
    echo "Error: Plugin source not found at $PLUGIN_SRC"
    exit 1
fi

# Check if already installed
if [[ -e "$TARGET_DIR" ]]; then
    echo "Plugin already installed at: $TARGET_DIR"
    read -p "Remove and reinstall? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET_DIR"
        echo "Removed existing installation."
    else
        echo "Installation cancelled."
        exit 0
    fi
fi

# Create plugins directory if needed
mkdir -p "$PLUGIN_DIR"

# Option 1: Symlink (for development)
# Option 2: Copy (for distribution)

read -p "Install as symlink (development) or copy (distribution)? [S/c] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Cc]$ ]]; then
    # Copy files
    echo "Copying files..."
    cp -r "$PLUGIN_SRC" "$TARGET_DIR"
    echo "Files copied to: $TARGET_DIR"
else
    # Create symlink
    echo "Creating symlink..."
    ln -s "$PLUGIN_SRC" "$TARGET_DIR"
    echo "Symlink created: $TARGET_DIR -> $PLUGIN_SRC"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Restart KiCad PCB Editor (if it's open)"
echo "2. Go to Tools -> External Plugins"
echo "3. You should see 'Flex Viewer - Test', 'Create Fold', and 'Open Fold Viewer'"
echo ""
echo "To uninstall: rm -rf \"$TARGET_DIR\""

#!/bin/bash
# KiCad Flex Viewer - Installation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_NAME="com_github_aightech_flexviz"
PLUGIN_SRC="$SCRIPT_DIR/plugins/$PLUGIN_NAME"

# ---- Plugin directory detection ----

detect_swig_plugin_dir() {
    local KICAD_DIRS=(
        "$HOME/.local/share/kicad/10.0/3rdparty/plugins"
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
    echo "$HOME/.local/share/kicad/10.0/3rdparty/plugins"
}

detect_ipc_plugin_dir() {
    # IPC API plugins live in <kicad_data>/<version>/plugins/
    # (same root as SWIG but different subdirectory)
    local KICAD_DIRS=(
        "$HOME/.local/share/kicad/10.0/plugins"
        "$HOME/.local/share/kicad/9.0/plugins"
    )
    for dir in "${KICAD_DIRS[@]}"; do
        if [[ -d "$(dirname "$dir")" ]]; then
            echo "$dir"
            return 0
        fi
    done
    echo "$HOME/.local/share/kicad/10.0/plugins"
}

# ---- Main ----

echo "KiCad Flex Viewer - Installation"
echo "================================"
echo ""

# Check source exists
if [[ ! -d "$PLUGIN_SRC" ]]; then
    echo "Error: Plugin source not found at $PLUGIN_SRC"
    exit 1
fi

# Choose installation mode
echo "Installation modes:"
echo "  [1] SWIG (classic)  - In-process plugin via __init__.py (KiCad 7-10)"
echo "  [2] IPC API (new)   - Out-of-process plugin via plugin.json (KiCad 9-10+)"
echo "  [3] Both            - Install to both locations"
echo ""
read -p "Choose mode [1/2/3] (default: 1): " -n 1 -r MODE
echo ""

case "$MODE" in
    2) MODES=("ipc") ;;
    3) MODES=("swig" "ipc") ;;
    *) MODES=("swig") ;;
esac

# Choose install method
read -p "Install as symlink (development) or copy (distribution)? [S/c] " -n 1 -r METHOD
echo ""

do_install() {
    local TARGET_DIR="$1"
    local LABEL="$2"

    # Remove existing
    if [[ -e "$TARGET_DIR" ]]; then
        echo "  Removing existing installation at $TARGET_DIR"
        rm -rf "$TARGET_DIR"
    fi

    mkdir -p "$(dirname "$TARGET_DIR")"

    if [[ $METHOD =~ ^[Cc]$ ]]; then
        cp -r "$PLUGIN_SRC" "$TARGET_DIR"
        echo "  [$LABEL] Copied to: $TARGET_DIR"
    else
        ln -s "$PLUGIN_SRC" "$TARGET_DIR"
        echo "  [$LABEL] Symlinked: $TARGET_DIR -> $PLUGIN_SRC"
    fi
}

echo ""
for mode in "${MODES[@]}"; do
    if [[ "$mode" == "swig" ]]; then
        SWIG_DIR=$(detect_swig_plugin_dir)
        do_install "$SWIG_DIR/$PLUGIN_NAME" "SWIG"
    elif [[ "$mode" == "ipc" ]]; then
        IPC_DIR=$(detect_ipc_plugin_dir)
        do_install "$IPC_DIR/$PLUGIN_NAME" "IPC"
    fi
done

echo ""
echo "Installation complete!"
echo ""

if [[ " ${MODES[*]} " =~ " swig " ]]; then
    echo "SWIG mode:"
    echo "  1. Restart KiCad PCB Editor"
    echo "  2. Go to Tools -> External Plugins"
    echo "  3. You should see 'Create Fold' and 'Open Fold Viewer'"
    echo ""
fi

if [[ " ${MODES[*]} " =~ " ipc " ]]; then
    echo "IPC mode:"
    echo "  1. Enable the API: Preferences -> Plugins -> Enable API"
    echo "  2. Restart KiCad PCB Editor"
    echo "  3. The plugin appears in Tools -> External Plugins"
    echo "  Note: kicad-python is auto-installed into the plugin's venv by KiCad"
    echo ""
fi

echo "To uninstall:"
for mode in "${MODES[@]}"; do
    if [[ "$mode" == "swig" ]]; then
        echo "  rm -rf \"$(detect_swig_plugin_dir)/$PLUGIN_NAME\""
    elif [[ "$mode" == "ipc" ]]; then
        echo "  rm -rf \"$(detect_ipc_plugin_dir)/$PLUGIN_NAME\""
    fi
done

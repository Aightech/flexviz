#!/bin/bash
# Create a KiCad PCM-compatible release ZIP
#
# Usage: ./scripts/create_release.sh [version]
# Example: ./scripts/create_release.sh 1.0.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERSION="${1:-1.0.0}"
ZIP_NAME="flexviz-${VERSION}.zip"

echo "Creating release: $ZIP_NAME"
echo "================================"

# Clean up any existing release
rm -f "$PROJECT_DIR/$ZIP_NAME"

# Create temp directory for packaging
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create PCM structure:
# ├── metadata.json
# └── plugins/
#     └── (plugin files)
mkdir -p "$TEMP_DIR/plugins"

# Copy plugin files (excluding __pycache__ and .pyc)
cp -r "$PROJECT_DIR/plugins/com_github_aightech_flexviz"/* "$TEMP_DIR/plugins/"
find "$TEMP_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -name "*.pyc" -delete 2>/dev/null || true

# Copy metadata.json
cp "$PROJECT_DIR/metadata.json" "$TEMP_DIR/"

# Create ZIP
cd "$TEMP_DIR"
zip -r "$PROJECT_DIR/$ZIP_NAME" .

echo ""
echo "================================"
echo "Release created: $ZIP_NAME"
echo ""

# Calculate and display metadata values
cd "$PROJECT_DIR"
SHA256=$(sha256sum "$ZIP_NAME" | cut -d' ' -f1)
DOWNLOAD_SIZE=$(stat -c%s "$ZIP_NAME")
INSTALL_SIZE=$(du -sb "$TEMP_DIR/plugins" | cut -f1)

echo "Update metadata.json with these values:"
echo ""
echo "  \"download_sha256\": \"$SHA256\","
echo "  \"download_size\": $DOWNLOAD_SIZE,"
echo "  \"install_size\": $INSTALL_SIZE"
echo ""
echo "ZIP contents:"
unzip -l "$ZIP_NAME" | head -20

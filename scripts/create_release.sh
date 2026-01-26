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

# Calculate install size before adding metadata
INSTALL_SIZE=$(du -sb "$TEMP_DIR/plugins" | cut -f1)

# Create initial metadata.json with placeholder SHA256
sed -e "s/{{VERSION}}/$VERSION/g" \
    -e "s/{{SHA256}}/PLACEHOLDER/g" \
    -e "s/{{DOWNLOAD_SIZE}}/0/g" \
    -e "s/{{INSTALL_SIZE}}/$INSTALL_SIZE/g" \
    "$PROJECT_DIR/metadata.template.json" > "$TEMP_DIR/metadata.json"

# Create ZIP (first pass to get size)
cd "$TEMP_DIR"
zip -r "$PROJECT_DIR/$ZIP_NAME" . > /dev/null

# Get actual values
cd "$PROJECT_DIR"
SHA256=$(sha256sum "$ZIP_NAME" | cut -d' ' -f1)
DOWNLOAD_SIZE=$(stat -c%s "$ZIP_NAME")

# Regenerate metadata.json with correct values
sed -e "s/{{VERSION}}/$VERSION/g" \
    -e "s/{{SHA256}}/$SHA256/g" \
    -e "s/{{DOWNLOAD_SIZE}}/$DOWNLOAD_SIZE/g" \
    -e "s/{{INSTALL_SIZE}}/$INSTALL_SIZE/g" \
    "$PROJECT_DIR/metadata.template.json" > "$TEMP_DIR/metadata.json"

# Also update the project's metadata.json
cp "$TEMP_DIR/metadata.json" "$PROJECT_DIR/metadata.json"

# Recreate ZIP with final metadata
rm "$PROJECT_DIR/$ZIP_NAME"
cd "$TEMP_DIR"
zip -r "$PROJECT_DIR/$ZIP_NAME" . > /dev/null

# Recalculate final SHA256 (it changed slightly due to updated metadata)
cd "$PROJECT_DIR"
FINAL_SHA256=$(sha256sum "$ZIP_NAME" | cut -d' ' -f1)
FINAL_SIZE=$(stat -c%s "$ZIP_NAME")

# Update metadata.json one more time with final SHA256
sed -i "s/$SHA256/$FINAL_SHA256/g" "$PROJECT_DIR/metadata.json"
sed -i "s/\"download_size\": $DOWNLOAD_SIZE/\"download_size\": $FINAL_SIZE/g" "$PROJECT_DIR/metadata.json"

echo ""
echo "================================"
echo "Release created: $ZIP_NAME"
echo ""
echo "metadata.json updated with:"
echo "  version: $VERSION"
echo "  download_sha256: $FINAL_SHA256"
echo "  download_size: $FINAL_SIZE"
echo "  install_size: $INSTALL_SIZE"
echo ""
echo "ZIP contents:"
unzip -l "$ZIP_NAME" | head -15
echo ""
echo "Next steps:"
echo "  1. git add metadata.json && git commit -m 'Update metadata for v$VERSION release'"
echo "  2. git tag v$VERSION"
echo "  3. git push origin main --tags"
echo "  4. Create GitHub release and upload $ZIP_NAME"

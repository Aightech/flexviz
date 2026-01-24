#!/usr/bin/env python3
"""
Build KiCad Plugin Manager (PCM) package for distribution.

This script creates a ZIP file suitable for distribution via KiCad's PCM
or manual installation.

Usage:
    python scripts/build_pcm_package.py [--version X.Y.Z]
"""

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path


# Files to include in the package
INCLUDE_FILES = [
    "__init__.py",
    "plugin.py",
    "viewer.py",
    "kicad_parser.py",
    "geometry.py",
    "markers.py",
    "bend_transform.py",
    "mesh.py",
    "fold_placer.py",
    "config.py",
    "stiffener.py",
    "validation.py",
    "planar_subdivision.py",
    "region_splitter.py",
    "model_loader.py",
    "step_export.py",
    "step_export_cli.py",
    "metadata.json",
]

INCLUDE_DIRS = [
    "resources",
    "docs",
]

# Files to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    "venv",
    "tests",
    "*.bkp",
    ".DS_Store",
]


def should_exclude(path: str) -> bool:
    """Check if a path should be excluded."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if path.endswith(pattern[1:]):
                return True
        elif pattern in path:
            return True
    return False


def get_file_sha256(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def build_package(version: str, output_dir: str = "dist"):
    """Build the PCM package."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / output_dir
    output_dir.mkdir(exist_ok=True)

    package_name = f"kicad-flex-viewer-{version}"
    zip_path = output_dir / f"{package_name}.zip"

    print(f"Building package: {zip_path}")

    # Create ZIP file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add individual files
        for filename in INCLUDE_FILES:
            src_path = project_root / filename
            if src_path.exists():
                # Files go into plugins/kicad_flex_viewer/
                arcname = f"plugins/kicad_flex_viewer/{filename}"
                zf.write(src_path, arcname)
                print(f"  Added: {arcname}")
            else:
                print(f"  Warning: {filename} not found")

        # Add directories
        for dirname in INCLUDE_DIRS:
            src_dir = project_root / dirname
            if src_dir.exists():
                for root, dirs, files in os.walk(src_dir):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if not should_exclude(d)]

                    for file in files:
                        if should_exclude(file):
                            continue
                        src_path = Path(root) / file
                        rel_path = src_path.relative_to(project_root)
                        arcname = f"plugins/kicad_flex_viewer/{rel_path}"
                        zf.write(src_path, arcname)
                        print(f"  Added: {arcname}")

    # Calculate file info
    file_size = zip_path.stat().st_size
    file_hash = get_file_sha256(str(zip_path))

    print(f"\nPackage created: {zip_path}")
    print(f"  Size: {file_size:,} bytes")
    print(f"  SHA256: {file_hash}")

    # Update metadata.json with size and hash
    metadata_path = project_root / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Update the latest version entry
        if metadata.get("versions"):
            metadata["versions"][0]["version"] = version
            metadata["versions"][0]["download_size"] = file_size
            metadata["versions"][0]["install_size"] = file_size * 2  # Approximate
            metadata["versions"][0]["download_sha256"] = file_hash

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"\nUpdated metadata.json with package info")

    return str(zip_path)


def main():
    parser = argparse.ArgumentParser(description="Build KiCad PCM package")
    parser.add_argument(
        "--version", "-v",
        default="1.0.0",
        help="Package version (default: 1.0.0)"
    )
    parser.add_argument(
        "--output", "-o",
        default="dist",
        help="Output directory (default: dist)"
    )

    args = parser.parse_args()
    build_package(args.version, args.output)


if __name__ == "__main__":
    main()

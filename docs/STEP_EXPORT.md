# STEP Export Setup Guide

The STEP export feature requires the `build123d` package, which provides OpenCASCADE (OCC) bindings for creating solid CAD geometry.

## Why Special Setup is Needed

KiCad plugins run inside KiCad's bundled Python environment, which is separate from your system Python. To use STEP export, you need to install `build123d` into KiCad's Python environment.

## Installation by Operating System

### Linux (Ubuntu/Debian)

KiCad on Linux uses the system Python. Installation method depends on your distro version:

#### Ubuntu 23.04+ / Debian 12+ (PEP 668 systems)

Modern systems protect the system Python. Use the `--break-system-packages` flag for user installs:

```bash
pip3 install --user --break-system-packages build123d
```

This is safe because `--user` installs to `~/.local/` (your home directory), not system-wide.

#### Older Ubuntu/Debian

```bash
pip3 install --user build123d
```

#### Using the install script

The easiest method - the script auto-detects your system:

```bash
./install_step_export.sh
```

3. **Verify installation:**
   - Open KiCad PCB Editor
   - Go to Tools → Scripting Console
   - Type:
     ```python
     from build123d import Box
     print("build123d OK")
     ```

### Linux (Fedora/RHEL)

```bash
# Install dependencies first
sudo dnf install python3-pip

# Install build123d
pip3 install --user build123d
```

### Windows

1. **Find KiCad's Python:**
   - KiCad 8.x/9.x includes Python at:
     ```
     C:\Program Files\KiCad\8.0\bin\python.exe
     # or
     C:\Program Files\KiCad\9.0\bin\python.exe
     ```

2. **Open Command Prompt as Administrator:**
   - Press Win+X → "Terminal (Admin)" or "Command Prompt (Admin)"

3. **Install build123d:**
   ```cmd
   "C:\Program Files\KiCad\9.0\bin\python.exe" -m pip install build123d
   ```

   For KiCad 8.0:
   ```cmd
   "C:\Program Files\KiCad\8.0\bin\python.exe" -m pip install build123d
   ```

4. **If pip is not found:**
   ```cmd
   "C:\Program Files\KiCad\9.0\bin\python.exe" -m ensurepip
   "C:\Program Files\KiCad\9.0\bin\python.exe" -m pip install --upgrade pip
   "C:\Program Files\KiCad\9.0\bin\python.exe" -m pip install build123d
   ```

5. **Verify in KiCad:**
   - Open KiCad PCB Editor
   - Tools → Scripting Console
   - Type:
     ```python
     from build123d import Box
     print("build123d OK")
     ```

### macOS

1. **Find KiCad's Python:**
   - KiCad 8.x/9.x Python is at:
     ```
     /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3
     ```

2. **Open Terminal and install:**
   ```bash
   /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install build123d
   ```

3. **If permission denied:**
   ```bash
   /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install --user build123d
   ```

4. **Verify in KiCad Scripting Console:**
   ```python
   from build123d import Box
   print("build123d OK")
   ```

## Troubleshooting

### "No module named pip"

Install pip first:
```bash
# Linux/macOS
python3 -m ensurepip

# Windows (in admin terminal)
"C:\Program Files\KiCad\9.0\bin\python.exe" -m ensurepip
```

### Permission Errors

- **Linux/macOS:** Add `--user` flag to pip install
- **Windows:** Run terminal as Administrator

### Build Errors During Installation

build123d requires compilation of OCC bindings. If you see build errors:

1. **Install build tools:**
   - **Linux:** `sudo apt install build-essential python3-dev`
   - **Windows:** Install Visual Studio Build Tools
   - **macOS:** `xcode-select --install`

2. **Try installing pre-built wheels:**
   ```bash
   pip install --only-binary :all: build123d
   ```

### KiCad Can't Find the Package

If you installed build123d but KiCad still can't find it:

1. Check where pip installed it:
   ```bash
   pip3 show build123d
   ```

2. Ensure the install location is in KiCad's Python path. In KiCad Scripting Console:
   ```python
   import sys
   print(sys.path)
   ```

3. If needed, add the path manually in your plugin or KiCad's Python startup.

## Alternative: Standalone Export

If you cannot install build123d in KiCad's Python, you can export from the command line:

```bash
# Activate your venv with build123d installed
source /path/to/flexviz/venv/bin/activate

# Run the export script
python3 -c "
from step_export import board_geometry_to_step
from geometry import extract_geometry
from markers import detect_fold_markers
from kicad_parser import KiCadPCB

pcb = KiCadPCB.load('/path/to/your/board.kicad_pcb')
geom = extract_geometry(pcb)
markers = detect_fold_markers(pcb)

board_geometry_to_step(geom, markers, 'output.step')
"
```

## Checking STEP Export Availability

In KiCad's Scripting Console, you can check if STEP export is available:

```python
try:
    from build123d import Box
    print("STEP export: Available")
except ImportError as e:
    print(f"STEP export: Not available - {e}")
```

## Performance Notes

- STEP export converts mesh triangles to solid faces, which can be slow for complex meshes
- For faster exports, reduce "Bend quality" (subdivisions) in the viewer before exporting
- Meshes with more than 5000 faces will be automatically limited
- Export time: ~1-2 seconds per 100 faces

## File Compatibility

The exported STEP files are compatible with:
- FreeCAD
- Fusion 360
- SolidWorks
- CATIA
- Any CAD software supporting STEP AP214

# Changelog

All notable changes to KiCad Flex Viewer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-23

### Added

#### Core Features
- **3D Fold Visualization**: Interactive 3D viewer for bent flex PCBs
- **Fold Marker System**: Create fold markers using selection-based placement
- **Multi-fold Support**: Handle multiple sequential folds with proper transformation chaining
- **Real-time Preview**: Adjust bend angles interactively with spin controls

#### Display Options
- Board outline, copper traces, pads, component boxes
- 3D component models from footprint definitions (WRL/STEP)
- Stiffener region visualization
- Wireframe overlay mode

#### Validation
- Bend radius warnings based on PCB thickness
- Stiffener-fold conflict detection
- Component in bend zone warnings
- Color-coded status indicators (green/yellow/red)

#### Export Formats
- OBJ (Wavefront) for general 3D software
- STL for 3D printing
- STEP for CAD tools (via command-line)

#### Configuration
- Configurable marker layer (any User layer)
- Adjustable bend quality (subdivision count)
- Stiffener thickness and layer settings
- Per-PCB settings persistence

### Technical Details

#### Geometry Processing
- Region-based triangulation preventing artifacts at fold boundaries
- Dimension-first marker detection for reliable fold identification
- Axis normalization ensuring consistent fold directions
- Back-entry handling for complex fold configurations

#### 3D Model Support
- Native WRL/VRML parser (no external dependencies)
- KiCad environment variable expansion for model paths
- Automatic WRL fallback when STEP unavailable
- Proper component positioning and rotation

#### KiCad Integration
- Works with KiCad 7.x, 8.x, and 9.x
- Toolbar buttons for Create Fold and Open Viewer
- Hot-reload support for development

---

## Development History

### Pre-release Development

#### Phase 1: Core Foundation
- S-expression parser for `.kicad_pcb` files
- Board geometry extraction (outline, traces, pads)
- Fold marker detection from User layer

#### Phase 2: Bend Mathematics
- `FoldDefinition` class for fold parameters
- Point classification (before/in/after bend zone)
- Cylindrical mapping for bend transformation
- Multi-fold sequential transformation

#### Phase 3: 3D Viewer
- wxPython + OpenGL viewer window
- Camera controls (orbit, pan, zoom)
- Mesh rendering with display list optimization
- Per-face coloring and normals

#### Phase 4: KiCad Integration
- Action plugin registration
- Create Fold tool with selection workflow
- Open Viewer action with board loading
- Error handling and user feedback

#### Phase 5: Polish & Edge Cases
- Ear-clipping triangulation for boards with cutouts
- Region splitting along fold axes
- Back-entry transformation handling

#### Phase 6: Documentation
- User guide with installation instructions
- API documentation in docstrings
- Architecture documentation (CLAUDE.md)

#### Phase 7: Validation & Warnings
- Design rule checking framework
- Bend radius validation
- Stiffener conflict detection
- Component placement warnings

#### Phase 8: Visual Enhancements
- Drill holes as board cutouts
- 3D model loading (WRL parser)
- Dual-layer stiffener support
- Back copper trace rendering

#### Phase 9: Export
- STEP export via build123d
- Command-line export tool
- Auto-generated export commands with settings

#### Phase 10: Release Preparation
- Comprehensive user documentation
- KiCad PCM package structure
- Tooltips for all UI controls
- CHANGELOG and LICENSE files

---

## Version Numbering

- **Major** (X.0.0): Breaking changes, major new features
- **Minor** (0.X.0): New features, backwards compatible
- **Patch** (0.0.X): Bug fixes, minor improvements

---

[1.0.0]: https://github.com/flexviz/kicad-flex-viewer/releases/tag/v1.0.0

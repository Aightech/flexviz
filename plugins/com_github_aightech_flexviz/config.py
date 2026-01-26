"""
Configuration for flex PCB visualization.

Defines settings for flex thickness, stiffener regions, and validation parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class FlexConfig:
    """
    Configuration for flex PCB visualization with thickness and stiffener support.

    Attributes:
        flex_thickness: PCB thickness in mm (read from board settings by default)
        marker_layer: KiCad layer containing fold markers (default: User.1)
        stiffener_layer_top: KiCad layer for top stiffeners (empty = none)
        stiffener_layer_bottom: KiCad layer for bottom stiffeners (empty = none)
        stiffener_thickness: Stiffener material thickness in mm (0 = no stiffener)
        show_thickness: Whether to render 3D extrusion or flat surface
        min_bend_radius_factor: Minimum bend radius = factor × flex_thickness
        bend_subdivisions: Number of strips to subdivide bend zone for smooth curves
    """
    # PCB parameters
    flex_thickness: float = 1.6  # mm (default PCB thickness, overridden from board settings)

    # Fold marker layer
    marker_layer: str = "User.1"  # KiCad layer for fold markers

    # Stiffener parameters (separate layers for top and bottom)
    stiffener_layer_top: str = ""  # KiCad layer for top stiffeners (empty = none)
    stiffener_layer_bottom: str = "User.2"  # KiCad layer for bottom stiffeners
    stiffener_thickness: float = 0.0  # mm (0 = no stiffener)

    # Visualization
    show_thickness: bool = True  # 3D extrusion vs flat surface

    # Validation
    min_bend_radius_factor: float = 6.0  # Min bend radius = factor × thickness

    # Mesh quality
    bend_subdivisions: int = 8  # Number of strips in bend zone for smooth rendering

    def validate(self) -> list[str]:
        """
        Validate configuration values.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if self.flex_thickness <= 0:
            errors.append(f"flex_thickness must be positive, got {self.flex_thickness}")

        if self.stiffener_thickness < 0:
            errors.append(f"stiffener_thickness cannot be negative, got {self.stiffener_thickness}")

        if self.min_bend_radius_factor < 1.0:
            errors.append(f"min_bend_radius_factor should be >= 1.0, got {self.min_bend_radius_factor}")

        if self.bend_subdivisions < 1:
            errors.append(f"bend_subdivisions must be >= 1, got {self.bend_subdivisions}")
        if self.bend_subdivisions > 32:
            errors.append(f"bend_subdivisions {self.bend_subdivisions} is excessive (max recommended: 32)")

        return errors

    @property
    def min_bend_radius(self) -> float:
        """Calculate minimum allowed bend radius in mm."""
        return self.min_bend_radius_factor * self.flex_thickness

    @property
    def has_stiffener(self) -> bool:
        """Check if any stiffener is configured."""
        return self.stiffener_thickness > 0 and bool(self.stiffener_layer_top or self.stiffener_layer_bottom)

    @property
    def has_top_stiffener(self) -> bool:
        """Check if top stiffener is configured."""
        return self.stiffener_thickness > 0 and bool(self.stiffener_layer_top)

    @property
    def has_bottom_stiffener(self) -> bool:
        """Check if bottom stiffener is configured."""
        return self.stiffener_thickness > 0 and bool(self.stiffener_layer_bottom)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "flex_thickness": self.flex_thickness,
            "marker_layer": self.marker_layer,
            "stiffener_layer_top": self.stiffener_layer_top,
            "stiffener_layer_bottom": self.stiffener_layer_bottom,
            "stiffener_thickness": self.stiffener_thickness,
            "show_thickness": self.show_thickness,
            "min_bend_radius_factor": self.min_bend_radius_factor,
            "bend_subdivisions": self.bend_subdivisions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlexConfig":
        """Create from dictionary."""
        # Handle legacy config with stiffener_layer + stiffener_side
        if "stiffener_layer" in data and "stiffener_side" in data:
            layer = data.get("stiffener_layer", "User.2")
            side = data.get("stiffener_side", "bottom")
            layer_top = layer if side == "top" else ""
            layer_bottom = layer if side == "bottom" else ""
        else:
            layer_top = data.get("stiffener_layer_top", "")
            layer_bottom = data.get("stiffener_layer_bottom", "User.2")

        return cls(
            flex_thickness=data.get("flex_thickness", 0.11),
            marker_layer=data.get("marker_layer", "User.1"),
            stiffener_layer_top=layer_top,
            stiffener_layer_bottom=layer_bottom,
            stiffener_thickness=data.get("stiffener_thickness", 0.0),
            show_thickness=data.get("show_thickness", True),
            min_bend_radius_factor=data.get("min_bend_radius_factor", 6.0),
            bend_subdivisions=data.get("bend_subdivisions", 8),
        )

    def save(self, filepath: Path | str) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> "FlexConfig":
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            return cls()  # Return defaults if file doesn't exist

        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_for_pcb(cls, pcb_filepath: Path | str) -> "FlexConfig":
        """
        Load configuration for a specific PCB file.

        Looks for <pcb_name>.flex_config.json next to the PCB file.
        Returns defaults if config file doesn't exist.
        """
        pcb_path = Path(pcb_filepath)
        config_path = pcb_path.with_suffix('.flex_config.json')
        return cls.load(config_path)

    def save_for_pcb(self, pcb_filepath: Path | str) -> None:
        """
        Save configuration for a specific PCB file.

        Saves as <pcb_name>.flex_config.json next to the PCB file.
        """
        pcb_path = Path(pcb_filepath)
        config_path = pcb_path.with_suffix('.flex_config.json')
        self.save(config_path)


# Common flex PCB thickness presets
FLEX_THICKNESS_PRESETS = {
    "1-layer": 0.11,   # Single-layer flex
    "2-layer": 0.15,   # 2-layer flex
    "4-layer": 0.20,   # 4-layer flex (thicker)
}

# Common stiffener thickness presets
STIFFENER_THICKNESS_PRESETS = {
    "none": 0.0,
    "thin_fr4": 0.2,    # Thin FR4 stiffener
    "standard_fr4": 0.4,  # Standard FR4 stiffener
    "thick_fr4": 0.8,   # Thick FR4 stiffener
    "aluminum": 0.5,    # Aluminum stiffener
}

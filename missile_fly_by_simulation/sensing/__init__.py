"""
Sensor models and camera utilities.

This module simulates how cameras observe the world, including:
- Camera projection models (3D world → 2D pixels)
- Field-of-view calculations and geometric checks
- Visibility and occlusion testing
"""

from missile_fly_by_simulation.sensing.camera_model import PinholeCameraModel
from missile_fly_by_simulation.sensing.field_of_view import (
    check_point_in_fov,
    is_pixel_in_image,
    check_line_of_sight,
    compute_fov_corners,
    compute_fov_boundary,
)

__all__ = [
    # Camera models
    'PinholeCameraModel',
    
    # Field of view utilities
    'check_point_in_fov',
    'is_pixel_in_image',
    'check_line_of_sight',
    'compute_fov_corners',
    'compute_fov_boundary',
]
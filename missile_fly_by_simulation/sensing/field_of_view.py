"""
Field of view utilities for cameras.

This module provides helper functions for field-of-view calculations,
geometric checks, and camera-related utilities.

Functions
---------
check_point_in_fov
    Check if a point (in camera frame) is inside FOV cone
is_pixel_in_image
    Check if pixel coordinates are inside image bounds
compute_fov_corners
    Compute the four corner rays of the camera FOV
compute_fov_boundary
    Compute points along the FOV boundary
"""

from typing import Tuple
import numpy as np
import numpy.typing as npt

# Import from satellite_simulation package
from missile_fly_by_simulation.domain.satellite import CameraSpecification


def check_point_in_fov(
    point_camera: npt.NDArray[np.float64],
    fov_horizontal_deg: float,
    fov_vertical_deg: float,
    min_depth: float = 0.0
) -> bool:
    """
    Check if point (in camera frame) is inside field-of-view cone.
    
    This performs an angular check - is the point within the FOV angles?
    
    Parameters
    ----------
    point_camera : ndarray of shape (3,)
        Point in camera coordinates (x=right, y=up, z=forward)
    fov_horizontal_deg : float
        Horizontal field of view [degrees] (full angle, not half)
    fov_vertical_deg : float
        Vertical field of view [degrees] (full angle, not half)
    min_depth : float, optional
        Minimum depth (z-coordinate) for point to be visible, default 0
        
    Returns
    -------
    in_fov : bool
        True if point is inside FOV cone
        
    Notes
    -----
    The FOV is defined as a cone/pyramid with apex at camera center.
    A point is inside if:
    - z > min_depth (in front of camera)
    - |atan2(x, z)| < fov_horizontal / 2
    - |atan2(y, z)| < fov_vertical / 2
    
    Examples
    --------
    >>> # Point straight ahead
    >>> point = np.array([0, 0, 100])
    >>> in_fov = check_point_in_fov(point, 30.0, 20.0)
    >>> print(in_fov)
    True
    >>> 
    >>> # Point to the side (outside FOV)
    >>> point = np.array([100, 0, 100])
    >>> in_fov = check_point_in_fov(point, 30.0, 20.0)
    >>> print(in_fov)
    False
    """
    point_camera = np.asarray(point_camera, dtype=np.float64)
    
    if point_camera.shape != (3,):
        raise ValueError(f"Point must be 3D, got shape {point_camera.shape}")
    
    x, y, z = point_camera
    
    # Check if in front of camera
    if z <= min_depth:
        return False
    
    # Compute angles
    angle_horizontal = np.abs(np.arctan2(x, z))
    angle_vertical = np.abs(np.arctan2(y, z))
    
    # Convert FOV to radians and get half-angles
    max_angle_h = np.radians(fov_horizontal_deg / 2)
    max_angle_v = np.radians(fov_vertical_deg / 2)
    
    # Check if within FOV cone
    return angle_horizontal <= max_angle_h and angle_vertical <= max_angle_v


def is_pixel_in_image(
    pixel: Tuple[float, float],
    resolution: Tuple[int, int]
) -> bool:
    """
    Check if pixel coordinates are inside image bounds.
    
    Parameters
    ----------
    pixel : tuple of (float, float)
        Pixel coordinates (u, v)
    resolution : tuple of (int, int)
        Image dimensions (width, height) in pixels
        
    Returns
    -------
    inside : bool
        True if 0 <= u < width and 0 <= v < height
        
    Examples
    --------
    >>> is_pixel_in_image((100.5, 200.3), (1024, 720))
    True
    >>> is_pixel_in_image((1100, 200), (1024, 720))
    False
    >>> is_pixel_in_image((-10, 200), (1024, 720))
    False
    """
    u, v = pixel
    width, height = resolution
    
    return 0.0 <= u < width and 0.0 <= v < height


def check_line_of_sight(
    observer_position: npt.NDArray[np.float64],
    target_position: npt.NDArray[np.float64],
    earth_radius: float = 6.378e6
) -> bool:
    """
    Check if observer has line-of-sight to target (not blocked by Earth).
    
    This is a geometric check - does the line segment between observer
    and target pass through Earth?
    
    Parameters
    ----------
    observer_position : ndarray of shape (3,)
        Observer position in ECI frame [m]
    target_position : ndarray of shape (3,)
        Target position in ECI frame [m]
    earth_radius : float, optional
        Earth's radius [m], default 6.378e6
        
    Returns
    -------
    visible : bool
        True if line-of-sight exists, False if blocked by Earth
        
    Notes
    -----
    The algorithm finds the closest point on the line segment to Earth's
    center and checks if that distance is greater than Earth's radius.
    
    This is a purely geometric test - it does not account for:
    - Atmospheric refraction
    - Earth oblateness
    - Camera field-of-view limits
    
    Examples
    --------
    >>> # Satellite at 600 km altitude
    >>> sat_pos = np.array([7.0e6, 0, 0])
    >>> 
    >>> # Target on opposite side of Earth (no line of sight)
    >>> target_pos = np.array([-7.0e6, 0, 0])
    >>> visible = check_line_of_sight(sat_pos, target_pos)
    >>> print(visible)
    False
    >>> 
    >>> # Target nearby (line of sight exists)
    >>> target_pos = np.array([7.0e6, 1.0e5, 0])
    >>> visible = check_line_of_sight(sat_pos, target_pos)
    >>> print(visible)
    True
    """
    obs_pos = np.asarray(observer_position, dtype=np.float64)
    tgt_pos = np.asarray(target_position, dtype=np.float64)
    
    # Check if either point is inside Earth
    if np.linalg.norm(obs_pos) < earth_radius:
        return False
    if np.linalg.norm(tgt_pos) < earth_radius:
        return False
    
    # Line direction and length
    line_dir = tgt_pos - obs_pos
    line_length = np.linalg.norm(line_dir)
    
    if line_length < 1e-6:
        # Positions are coincident
        return True
    
    line_dir_normalized = line_dir / line_length
    
    # Find parameter t for closest point on line to Earth's center
    # Closest point: P(t) = obs_pos + t × line_dir
    # Minimize |P(t)|² → solve d/dt |P(t)|² = 0
    t = -np.dot(obs_pos, line_dir_normalized)
    
    # Clamp t to line segment [0, line_length]
    if t < 0:
        closest_point = obs_pos
    elif t > line_length:
        closest_point = tgt_pos
    else:
        closest_point = obs_pos + t * line_dir_normalized
    
    # Check if closest point is outside Earth
    closest_distance = np.linalg.norm(closest_point)
    
    return closest_distance > earth_radius


# =============================================================================
# FOV GEOMETRY FUNCTIONS
# =============================================================================

def compute_fov_corners(
    camera_spec: CameraSpecification
) -> npt.NDArray[np.float64]:
    """
    Compute the four corner ray directions of the camera FOV.
    
    Returns unit vectors pointing to the four corners of the image plane.
    
    Parameters
    ----------
    camera_spec : CameraSpecification
        Camera specification (resolution, FOV)
        
    Returns
    -------
    corners : ndarray of shape (4, 3)
        Four corner ray directions (in camera frame):
        - [0]: Top-left
        - [1]: Top-right
        - [2]: Bottom-right
        - [3]: Bottom-left
        
    Notes
    -----
    The rays are in camera coordinates (x=right, y=up, z=forward).
    All rays are normalized to unit length.
    
    Useful for:
    - Visualizing FOV in 3D plots
    - Computing FOV boundary
    - Checking if regions are visible
    
    Examples
    --------
    >>> spec = CameraSpecification(
    ...     resolution=(1024, 720),
    ...     fov_horizontal_deg=30.0,
    ...     fps=30
    ... )
    >>> corners = compute_fov_corners(spec)
    >>> print(corners.shape)
    (4, 3)
    >>> 
    >>> # Visualize FOV in 3D
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> for corner in corners:
    ...     ax.plot([0, corner[0]], [0, corner[1]], [0, corner[2]])
    """
    # Half-angles
    fov_h_rad = np.radians(camera_spec.fov_horizontal_deg)
    fov_v_rad = np.radians(camera_spec.fov_vertical_deg)
    
    half_h = fov_h_rad / 2
    half_v = fov_v_rad / 2
    
    # Tangent of half-angles
    tan_h = np.tan(half_h)
    tan_v = np.tan(half_v)
    
    # Four corners (at unit depth z=1)
    # Camera frame: x=right, y=up, z=forward
    corners_normalized = np.array([
        [-tan_h,  tan_v, 1.0],  # Top-left
        [ tan_h,  tan_v, 1.0],  # Top-right
        [ tan_h, -tan_v, 1.0],  # Bottom-right
        [-tan_h, -tan_v, 1.0],  # Bottom-left
    ], dtype=np.float64)
    
    # Normalize to unit vectors
    corners = corners_normalized / np.linalg.norm(corners_normalized, axis=1, keepdims=True)
    
    return corners


def compute_fov_boundary(
    camera_spec: CameraSpecification,
    num_points: int = 100,
    depth: float = 1.0
) -> npt.NDArray[np.float64]:
    """
    Compute points along the boundary of the FOV.
    
    Returns points forming a rectangular boundary at a specified depth.
    
    Parameters
    ----------
    camera_spec : CameraSpecification
        Camera specification
    num_points : int, optional
        Number of points along boundary (divisible by 4), default 100
    depth : float, optional
        Depth at which to compute boundary [m], default 1.0
        
    Returns
    -------
    boundary : ndarray of shape (num_points, 3)
        Points along FOV boundary in camera frame
        
    Notes
    -----
    The boundary is a rectangle at the specified depth, following
    the edges of the field of view.
    
    Examples
    --------
    >>> spec = CameraSpecification(
    ...     resolution=(1024, 720),
    ...     fov_horizontal_deg=30.0,
    ...     fps=30
    ... )
    >>> boundary = compute_fov_boundary(spec, num_points=100, depth=100.0)
    >>> # Plot boundary
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(boundary[:, 0], boundary[:, 1])
    """
    # Ensure num_points is divisible by 4 (four sides)
    num_points = (num_points // 4) * 4
    points_per_side = num_points // 4
    
    # Get corner directions
    corners = compute_fov_corners(camera_spec)
    
    # Scale corners to specified depth
    # corners are unit vectors, scale them so z-component = depth
    corners_scaled = corners * (depth / corners[:, 2:3])
    
    # Generate points along each side
    boundary_points = []
    
    for i in range(4):
        start = corners_scaled[i]
        end = corners_scaled[(i + 1) % 4]
        
        # Linearly interpolate between corners
        t = np.linspace(0, 1, points_per_side, endpoint=(i == 3))
        for ti in t:
            point = (1 - ti) * start + ti * end
            boundary_points.append(point)
    
    return np.array(boundary_points, dtype=np.float64)


def compute_fov_frustum_vertices(
    camera_spec: CameraSpecification,
    near_depth: float = 1.0,
    far_depth: float = 1000.0
) -> npt.NDArray[np.float64]:
    """
    Compute vertices of the view frustum (truncated pyramid).
    
    The frustum is the 3D volume visible to the camera between
    near and far depth planes.
    
    Parameters
    ----------
    camera_spec : CameraSpecification
        Camera specification
    near_depth : float, optional
        Near clipping plane depth [m], default 1.0
    far_depth : float, optional
        Far clipping plane depth [m], default 1000.0
        
    Returns
    -------
    vertices : ndarray of shape (8, 3)
        Eight vertices of the frustum:
        - [0:4]: Near plane corners (TL, TR, BR, BL)
        - [4:8]: Far plane corners (TL, TR, BR, BL)
        
    Notes
    -----
    Useful for 3D visualization of camera FOV.
    
    Examples
    --------
    >>> spec = CameraSpecification(
    ...     resolution=(1024, 720),
    ...     fov_horizontal_deg=30.0,
    ...     fps=30
    ... )
    >>> vertices = compute_fov_frustum_vertices(spec, 10.0, 1000.0)
    >>> print(vertices.shape)
    (8, 3)
    """
    # Get corner directions
    corners = compute_fov_corners(camera_spec)
    
    # Scale to near plane
    near_corners = corners * (near_depth / corners[:, 2:3])
    
    # Scale to far plane
    far_corners = corners * (far_depth / corners[:, 2:3])
    
    # Combine
    vertices = np.vstack([near_corners, far_corners])
    
    return vertices


def pixel_coordinates_to_normalized(
    pixel: Tuple[float, float],
    resolution: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to normalized device coordinates.
    
    Normalized coordinates are in [-1, 1] range with origin at center.
    
    Parameters
    ----------
    pixel : tuple of (float, float)
        Pixel coordinates (u, v)
    resolution : tuple of (int, int)
        Image dimensions (width, height)
        
    Returns
    -------
    normalized : tuple of (float, float)
        Normalized coordinates (x_norm, y_norm) in [-1, 1]
        
    Examples
    --------
    >>> # Image center
    >>> normalized = pixel_coordinates_to_normalized((512, 360), (1024, 720))
    >>> print(normalized)
    (0.0, 0.0)
    >>> 
    >>> # Top-left corner
    >>> normalized = pixel_coordinates_to_normalized((0, 0), (1024, 720))
    >>> print(normalized)
    (-1.0, 1.0)
    """
    u, v = pixel
    width, height = resolution
    
    # Map [0, width] → [-1, 1]
    x_norm = 2.0 * u / width - 1.0
    
    # Map [0, height] → [1, -1] (flip y)
    y_norm = 1.0 - 2.0 * v / height
    
    return (x_norm, y_norm)


def normalized_to_pixel_coordinates(
    normalized: Tuple[float, float],
    resolution: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Convert normalized device coordinates to pixel coordinates.
    
    Inverse of pixel_coordinates_to_normalized.
    
    Parameters
    ----------
    normalized : tuple of (float, float)
        Normalized coordinates (x_norm, y_norm) in [-1, 1]
    resolution : tuple of (int, int)
        Image dimensions (width, height)
        
    Returns
    -------
    pixel : tuple of (float, float)
        Pixel coordinates (u, v)
        
    Examples
    --------
    >>> # Center
    >>> pixel = normalized_to_pixel_coordinates((0.0, 0.0), (1024, 720))
    >>> print(pixel)
    (512.0, 360.0)
    """
    x_norm, y_norm = normalized
    width, height = resolution
    
    # Map [-1, 1] → [0, width]
    u = (x_norm + 1.0) * width / 2.0
    
    # Map [1, -1] → [0, height]
    v = (1.0 - y_norm) * height / 2.0
    
    return (u, v)


def compute_pixel_solid_angle(
    camera_spec: CameraSpecification,
    pixel: Tuple[float, float]
) -> float:
    """
    Compute the solid angle subtended by a pixel.
    
    This is useful for radiometric calculations and understanding
    how pixel size varies across the image.
    
    Parameters
    ----------
    camera_spec : CameraSpecification
        Camera specification
    pixel : tuple of (float, float)
        Pixel coordinates (u, v)
        
    Returns
    -------
    solid_angle : float
        Solid angle [steradians]
        
    Notes
    -----
    Pixels near the edge of the image subtend larger solid angles
    than pixels at the center (for perspective projection).
    
    Examples
    --------
    >>> spec = CameraSpecification(
    ...     resolution=(1024, 720),
    ...     fov_horizontal_deg=30.0,
    ...     fps=30
    ... )
    >>> # Center pixel
    >>> omega_center = compute_pixel_solid_angle(spec, (512, 360))
    >>> # Corner pixel
    >>> omega_corner = compute_pixel_solid_angle(spec, (0, 0))
    >>> print(f"Corner/Center ratio: {omega_corner/omega_center:.2f}")
    """
    width, height = camera_spec.resolution
    
    # Compute focal lengths
    fov_h_rad = np.radians(camera_spec.fov_horizontal_deg)
    fov_v_rad = np.radians(camera_spec.fov_vertical_deg)
    
    fx = width / (2.0 * np.tan(fov_h_rad / 2))
    fy = height / (2.0 * np.tan(fov_v_rad / 2))
    
    # Pixel position relative to principal point
    u, v = pixel
    cx, cy = width / 2.0, height / 2.0
    
    x = (u - cx) / fx
    y = (v - cy) / fy
    
    # Distance from optical axis (normalized)
    r_squared = x**2 + y**2
    
    # Solid angle (approximate for small pixels)
    # Exact formula: dΩ = dA / r³ where r = √(1 + x² + y²)
    r_cubed = (1 + r_squared) ** 1.5
    
    # Pixel area in normalized coordinates
    pixel_area = (1.0 / fx) * (1.0 / fy)
    
    solid_angle = pixel_area / r_cubed
    
    return solid_angle
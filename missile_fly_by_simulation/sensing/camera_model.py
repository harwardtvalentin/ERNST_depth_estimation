"""
Camera models for optical sensing.

This module implements camera projection models that simulate how
3D world points are observed as 2D pixel coordinates.

Classes
-------
PinholeCameraModel
    Ideal pinhole camera model with perspective projection

Notes
-----
This module handles the FORWARD problem: world → image.
The inverse problem (image → world depth) is handled by estimation/.
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

# Import from our domain module
from missile_fly_by_simulation.domain import CameraSpecification, SatelliteState


class PinholeCameraModel:
    """
    Ideal pinhole camera model for perspective projection.
    
    This class models a camera sensor that projects 3D world points
    to 2D image pixels using the pinhole camera model (perspective
    projection with no lens distortion).
    
    The camera model includes:
    - Intrinsic parameters (focal length, principal point)
    - Image resolution and field of view
    - Projection from 3D world to 2D pixels
    - Inverse projection from 2D pixels to 3D rays
    
    Attributes
    ----------
    spec : CameraSpecification
        Camera hardware specification
    focal_length_x : float
        Focal length in x-direction [pixels]
    focal_length_y : float
        Focal length in y-direction [pixels]
    principal_point_x : float
        Principal point x-coordinate [pixels]
    principal_point_y : float
        Principal point y-coordinate [pixels]
    
    Notes
    -----
    Coordinate conventions:
    - Camera frame: x=right, y=up, z=forward (in front of camera)
    - Image frame: u=right, v=down (standard computer vision)
    - World frame: ECI (Earth-Centered Inertial)
    
    The pinhole projection equation:
        [u]   [fx  0  cx] [X/Z]
        [v] = [ 0 fy  cy] [Y/Z]
        [1]   [ 0  0   1] [ 1 ]
    
    where (X,Y,Z) is the point in camera frame.
    
    References
    ----------
    .. [1] Hartley & Zisserman, "Multiple View Geometry"
    .. [2] Szeliski, "Computer Vision: Algorithms and Applications"
    
    Examples
    --------
    >>> from domain_satellite import CameraSpecification
    >>> spec = CameraSpecification(
    ...     resolution=(1024, 720),
    ...     fov_horizontal_deg=30.0,
    ...     fps=30
    ... )
    >>> camera = PinholeCameraModel(spec)
    >>> 
    >>> # Project a 3D point
    >>> point = np.array([7e6, 1e5, 0])
    >>> pixel = camera.project_to_image(point, satellite_state)
    >>> if pixel:
    ...     print(f"Pixel: {pixel}")
    """
    
    def __init__(self, spec: CameraSpecification):
        """
        Initialize camera model from specification.
        
        Parameters
        ----------
        spec : CameraSpecification
            Camera hardware properties (resolution, FOV, fps)
        """
        self.spec = spec
        
        # Image dimensions
        self.width = spec.resolution[0]
        self.height = spec.resolution[1]
        
        # Compute focal lengths from field of view
        # For pinhole camera: tan(FOV/2) = (sensor_size/2) / focal_length
        # Rearranging: focal_length = (image_size/2) / tan(FOV/2)
        
        fov_h_rad = np.radians(spec.fov_horizontal_deg)
        fov_v_rad = np.radians(spec.fov_vertical_deg)
        
        self.focal_length_x = self.width / (2.0 * np.tan(fov_h_rad / 2))
        self.focal_length_y = self.height / (2.0 * np.tan(fov_v_rad / 2))
        
        # Principal point (optical center) - typically at image center
        self.principal_point_x = self.width / 2.0
        self.principal_point_y = self.height / 2.0
    
    def project_to_image(
        self,
        point_world: npt.NDArray[np.float64],
        camera_state: SatelliteState
    ) -> Optional[Tuple[float, float]]:
        """
        Project 3D world point to 2D pixel coordinates.
        
        This is the main projection function - simulates what the camera sees.
        
        Parameters
        ----------
        point_world : ndarray of shape (3,)
            3D point in world frame (ECI) [m]
        camera_state : SatelliteState
            Camera position and orientation (must have attitude)
            
        Returns
        -------
        pixel : tuple of (float, float) or None
            Pixel coordinates (u, v) if visible, None otherwise
            
        Notes
        -----
        Returns None if:
        - Point is behind the camera (z <= 0 in camera frame)
        - Point is outside the field of view
        - Camera state has no attitude
        
        The projection steps are:
        1. Transform point from world frame to camera frame
        2. Check if point is in front of camera
        3. Apply pinhole projection
        4. Check if pixel is inside image bounds
        
        Examples
        --------
        >>> point_world = np.array([7.0e6, 1.0e5, 0])
        >>> pixel = camera.project_to_image(point_world, sat_state)
        >>> if pixel:
        ...     u, v = pixel
        ...     print(f"Detected at pixel ({u:.1f}, {v:.1f})")
        ... else:
        ...     print("Not visible")
        """
        # Validate inputs
        if camera_state.attitude is None:
            raise ValueError("Camera state must have attitude to project points")
        
        point_world = np.asarray(point_world, dtype=np.float64)
        if point_world.shape != (3,):
            raise ValueError(f"Point must be 3D, got shape {point_world.shape}")
        
        # Step 1: Transform point from world frame to camera frame
        # Vector from camera to target in world coordinates
        vector_world = point_world - camera_state.position
        
        # Transform to camera frame using attitude
        # Camera frame: x=right, y=up, z=forward
        point_camera = camera_state.attitude.world_to_satellite(vector_world)
        
        x_cam = point_camera[0]  # right
        y_cam = point_camera[1]  # up
        z_cam = point_camera[2]  # forward
        
        # Step 2: Check if point is in front of camera
        if z_cam <= 0:
            return None  # Behind camera or on camera plane
        
        # Step 3: Pinhole projection
        # Normalize by depth (z) and apply intrinsics
        u = self.focal_length_x * (x_cam / z_cam) + self.principal_point_x
        
        # Note: Image y-axis typically points DOWN (computer vision convention)
        # So we negate y_cam to flip the vertical axis
        v = -self.focal_length_y * (y_cam / z_cam) + self.principal_point_y
        
        # Step 4: Check if pixel is inside image bounds
        if 0.0 <= u < self.width and 0.0 <= v < self.height:
            # Return exact sub-pixel coordinates; noise is added by the caller
            return (u, v)
        else:
            return None  # Outside field of view
    
    def pixel_to_ray(
        self,
        pixel: Tuple[float, float]
    ) -> npt.NDArray[np.float64]:
        """
        Convert pixel to 3D ray direction in camera frame.
        
        This is the inverse projection (without depth information).
        Given a pixel, compute the 3D ray from the camera center through
        that pixel into the scene.
        
        Parameters
        ----------
        pixel : tuple of (float, float)
            Pixel coordinates (u, v)
            
        Returns
        -------
        ray : ndarray of shape (3,)
            Normalized ray direction in camera frame
            
        Notes
        -----
        The ray is in camera coordinates (x=right, y=up, z=forward).
        To get the ray in world frame, transform using attitude:
            ray_world = attitude.satellite_to_world(ray_camera)
        
        This function is used by depth estimation algorithms that
        triangulate 3D position from multiple pixel observations.
        
        Examples
        --------
        >>> pixel = (512.0, 360.0)  # Image center
        >>> ray = camera.pixel_to_ray(pixel)
        >>> print(ray)
        [0. 0. 1.]  # Points forward along optical axis
        >>> 
        >>> # Get ray in world frame
        >>> ray_world = satellite_state.attitude.satellite_to_world(ray)
        """
        u, v = pixel
        
        # Inverse projection: pixel → normalized image coordinates
        x = (u - self.principal_point_x) / self.focal_length_x
        
        # Note: Flip y back (we flipped it in projection)
        y = -(v - self.principal_point_y) / self.focal_length_y
        
        # Ray direction in camera frame
        # z=1 corresponds to unit depth (will be normalized anyway)
        ray = np.array([x, y, 1.0], dtype=np.float64)
        
        # Normalize to unit vector
        ray = ray / np.linalg.norm(ray)
        
        return ray
    
    def is_in_fov(
        self,
        point_world: npt.NDArray[np.float64],
        camera_state: SatelliteState
    ) -> bool:
        """
        Check if 3D point is inside camera's field of view.
        
        This is faster than full projection - just checks angular bounds.
        
        Parameters
        ----------
        point_world : ndarray of shape (3,)
            3D point in world frame [m]
        camera_state : SatelliteState
            Camera position and orientation
            
        Returns
        -------
        in_fov : bool
            True if point is visible (inside FOV cone and in front)
            
        Examples
        --------
        >>> in_fov = camera.is_in_fov(missile_position, sat_state)
        >>> if in_fov:
        ...     print("Target is visible")
        """
        if camera_state.attitude is None:
            return False
        
        # Transform to camera frame
        vector_world = point_world - camera_state.position
        point_camera = camera_state.attitude.world_to_satellite(vector_world)
        
        x, y, z = point_camera
        
        # Check if in front
        if z <= 0:
            return False
        
        # Check angular bounds
        # Horizontal angle
        angle_h = np.abs(np.arctan2(x, z))
        max_angle_h = np.radians(self.spec.fov_horizontal_deg / 2)
        
        # Vertical angle
        angle_v = np.abs(np.arctan2(y, z))
        max_angle_v = np.radians(self.spec.fov_vertical_deg / 2)
        
        return angle_h <= max_angle_h and angle_v <= max_angle_v
    
    def batch_project(
        self,
        points_world: npt.NDArray[np.float64],
        camera_state: SatelliteState
    ) -> list:
        """
        Project multiple 3D points to pixels (batch processing).
        
        Parameters
        ----------
        points_world : ndarray of shape (N, 3)
            N points in world frame [m]
        camera_state : SatelliteState
            Camera position and orientation
            
        Returns
        -------
        pixels : list of tuple or None
            List of pixel coordinates or None for each point
            
        Examples
        --------
        >>> points = np.array([
        ...     [7.0e6, 0, 0],
        ...     [7.0e6, 1e5, 0],
        ...     [7.0e6, 2e5, 0]
        ... ])
        >>> pixels = camera.batch_project(points, sat_state)
        >>> for i, pixel in enumerate(pixels):
        ...     if pixel:
        ...         print(f"Point {i}: {pixel}")
        """
        pixels = []
        for point in points_world:
            pixel = self.project_to_image(point, camera_state)
            pixels.append(pixel)
        return pixels
    
    # =========================================================================
    # PROPERTIES (Derived quantities)
    # =========================================================================
    
    @property
    def focal_length_pixels(self) -> Tuple[float, float]:
        """
        Focal length in pixels.
        
        Returns
        -------
        (fx, fy) : tuple of float
            Focal lengths in x and y directions [pixels]
        """
        return (self.focal_length_x, self.focal_length_y)
    
    @property
    def principal_point(self) -> Tuple[float, float]:
        """
        Principal point (optical center).
        
        Returns
        -------
        (cx, cy) : tuple of float
            Principal point coordinates [pixels]
        """
        return (self.principal_point_x, self.principal_point_y)
    
    @property
    def intrinsic_matrix(self) -> npt.NDArray[np.float64]:
        """
        Camera intrinsic matrix K.
        
        Returns
        -------
        K : ndarray of shape (3, 3)
            Intrinsic calibration matrix
            
        Notes
        -----
        The intrinsic matrix is:
            K = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        
        Used in projection equation: s*[u,v,1]^T = K * [X/Z, Y/Z, 1]^T
        """
        K = np.array([
            [self.focal_length_x, 0.0, self.principal_point_x],
            [0.0, self.focal_length_y, self.principal_point_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return K
    
    @property
    def aspect_ratio(self) -> float:
        """
        Image aspect ratio (width / height).
        
        Returns
        -------
        aspect : float
            Width divided by height
        """
        return self.width / self.height
    
    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PinholeCameraModel("
            f"resolution={self.spec.resolution}, "
            f"fov_h={self.spec.fov_horizontal_deg:.1f}°, "
            f"fov_v={self.spec.fov_vertical_deg:.1f}°)"
        )
    
    def __str__(self) -> str:
        """Human-readable description."""
        return (
            f"Pinhole Camera Model:\n"
            f"  Resolution: {self.width} × {self.height} pixels\n"
            f"  Field of View: {self.spec.fov_horizontal_deg:.1f}° × "
            f"{self.spec.fov_vertical_deg:.1f}°\n"
            f"  Focal Length: fx={self.focal_length_x:.1f}, "
            f"fy={self.focal_length_y:.1f} pixels\n"
            f"  Principal Point: ({self.principal_point_x:.1f}, "
            f"{self.principal_point_y:.1f})\n"
            f"  Frame Rate: {self.spec.fps} fps"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_projection_matrix(
    camera: PinholeCameraModel,
    camera_state: SatelliteState
) -> npt.NDArray[np.float64]:
    """
    Compute full projection matrix (world → image).
    
    Combines intrinsics (K) and extrinsics (R, t) into a single 3×4 matrix.
    
    Parameters
    ----------
    camera : PinholeCameraModel
        Camera model
    camera_state : SatelliteState
        Camera position and orientation
        
    Returns
    -------
    P : ndarray of shape (3, 4)
        Projection matrix such that s*[u,v,1]^T = P * [X,Y,Z,1]^T
        
    Notes
    -----
    The projection matrix is: P = K * [R | t]
    where K is intrinsics, R is rotation, t is translation.
    
    Examples
    --------
    >>> P = compute_projection_matrix(camera, sat_state)
    >>> point_homogeneous = np.array([7e6, 0, 0, 1])
    >>> pixel_homogeneous = P @ point_homogeneous
    >>> u = pixel_homogeneous[0] / pixel_homogeneous[2]
    >>> v = pixel_homogeneous[1] / pixel_homogeneous[2]
    """
    if camera_state.attitude is None:
        raise ValueError("Camera state must have attitude")
    
    # Intrinsic matrix
    K = camera.intrinsic_matrix
    
    # Extrinsic matrix [R | t]
    R = camera_state.attitude.rotation_matrix
    t = -R @ camera_state.position  # Translation in camera frame
    
    # Combine: extrinsic = [R | t]
    extrinsic = np.hstack([R, t.reshape(3, 1)])
    
    # Full projection matrix
    P = K @ extrinsic
    
    return P
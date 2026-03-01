"""
Attitude dynamics computations.

This module computes satellite orientation (attitude) for camera pointing.

Classes
-------
NadirPointingController
    Computes satellite attitude to point camera at target

Notes
-----
The attitude is represented as an orthonormal basis (right, up, forward)
that defines the satellite body frame relative to the ECI frame.
"""

from typing import Optional
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.domain import Attitude
from missile_fly_by_simulation.constants import (
    EARTH_RADIUS_M,
    ATTITUDE_NOISE_DEG
)


class NadirPointingController:
    """
    Compute satellite attitude to point camera at target.
    
    This controller computes the orientation (attitude) of a satellite
    such that its camera points toward a specified target while maintaining
    a stable "up" direction.
    
    The attitude is defined by three orthonormal vectors:
    - forward: points from satellite toward target (camera look direction)
    - right: perpendicular to orbital plane
    - up: completes the right-handed coordinate system
    
    Notes
    -----
    The algorithm prioritizes:
    1. Forward direction (must point at target)
    2. Right direction (tries to be perpendicular to velocity)
    3. Up direction (adjusted to ensure orthonormality)
    
    This is a simple geometric controller, not a dynamics simulation.
    It assumes instantaneous attitude changes (no angular momentum).
    
    Examples
    --------
    >>> controller = NadirPointingController()
    >>> 
    >>> # Satellite position and velocity
    >>> sat_pos = np.array([7.0e6, 0, 0])
    >>> sat_vel = np.array([0, 7.5e3, 0])
    >>> 
    >>> # Target position
    >>> target_pos = np.array([7.0e6, 1.0e5, 0])
    >>> 
    >>> # Compute attitude
    >>> attitude = controller.compute_attitude(sat_pos, target_pos, sat_vel)
    >>> print(attitude.forward)
    [0. 1. 0.]
    """
    
    def __init__(self):
        """Initialize attitude controller."""
        pass
    
    def compute_attitude(
        self,
        satellite_position,
        target_position,
        satellite_velocity,
        nadir_fallback=True,
        attitude_noise_deg=ATTITUDE_NOISE_DEG,  # ← pulls from constants.py
    ) -> Attitude:
        """
        Compute satellite attitude to point at target.
        
        Parameters
        ----------
        satellite_position : ndarray of shape (3,)
            Satellite position in ECI frame [m]
        target_position : ndarray of shape (3,)
            Target position in ECI frame [m]
        satellite_velocity : ndarray of shape (3,)
            Satellite velocity in ECI frame [m/s]
        nadir_fallback : bool, optional
            If True, use nadir pointing when target not available
            
        Returns
        -------
        Attitude
            Satellite orientation (right, up, forward vectors)
            
        Raises
        ------
        ValueError
            If vectors are parallel or degenerate
            
        Notes
        -----
        The attitude computation follows these steps:
        1. Forward: from satellite to target (normalized)
        2. Right: velocity × (-position) for Earth-facing right
        3. Up: forward × right (completes orthonormal system)
        4. Re-orthogonalize to ensure numerical accuracy
        
        Examples
        --------
        >>> controller = NadirPointingController()
        >>> attitude = controller.compute_attitude(
        ...     satellite_position=np.array([7e6, 0, 0]),
        ...     target_position=np.array([7e6, 1e5, 0]),
        ...     satellite_velocity=np.array([0, 7.5e3, 0])
        ... )
        """
        # Convert to numpy arrays
        sat_pos = np.asarray(satellite_position, dtype=np.float64)
        target_pos = np.asarray(target_position, dtype=np.float64)
        sat_vel = np.asarray(satellite_velocity, dtype=np.float64)
        
        # Validate inputs
        if sat_pos.shape != (3,) or target_pos.shape != (3,) or sat_vel.shape != (3,):
            raise ValueError("All input vectors must be 3D")
        
        # Step 1: Compute forward vector (satellite → target)
        forward = target_pos - sat_pos
        forward_norm = np.linalg.norm(forward)
        
        if forward_norm < 1e-6:
            raise ValueError("Target and satellite positions are coincident")
        
        forward = forward / forward_norm
        
        # Step 2: Compute right vector
        # Use velocity × (-position) to get a vector perpendicular to orbital plane
        # pointing "right" from the satellite's perspective
        nadir = -sat_pos  # Points toward Earth center
        right = np.cross(sat_vel, nadir)
        right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            # Velocity and nadir are parallel - degenerate case
            # This can happen at orbital poles
            # Use a random perpendicular vector
            right = self._get_perpendicular_vector(forward)
        else:
            right = right / right_norm
        
        # Check if forward and right are parallel
        if abs(np.dot(forward, right)) > 0.99:
            # Nearly parallel - adjust right to be perpendicular
            right = self._get_perpendicular_vector(forward)
        
        # Step 3: Re-orthogonalize (CRITICAL: matches original algorithm)
        # This is the key difference - we do TWO cross products to ensure
        # forward is prioritized and the basis is exactly orthonormal
        
        # First: Recalculate up from forward and right
        # This DISCARDS the initial velocity-based up and ensures orthogonality
        up = np.cross(forward, right)
        up_norm = np.linalg.norm(up)
        
        if up_norm < 1e-6:
            raise ValueError("Failed to compute orthonormal basis (degenerate geometry)")
        
        up = up / up_norm
        
        # Second: Recalculate right from up and forward
        # This ensures right is exactly perpendicular to both
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        # Step 4: Final normalization (all three vectors)
        # (Already normalized above, but doing it again to match original)
        right = right / np.linalg.norm(right)
        up = up / np.linalg.norm(up)
        forward = forward / np.linalg.norm(forward)
        
        # Artificially add attitude noise
        if attitude_noise_deg > 0.0:
            # Sample a random rotation axis (uniform on unit sphere)
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)

            # Sample a rotation angle from N(0, σ²)
            angle_rad = np.deg2rad(np.random.normal(0.0, attitude_noise_deg))

            # Rodrigues' rotation formula (already available in this file)
            forward = self._rotate_vector_around_axis(forward, axis, np.rad2deg(angle_rad))
            forward /= np.linalg.norm(forward)

            # Re-orthogonalise the basis around the perturbed forward
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)
            up /= np.linalg.norm(up)

        # Create and return attitude
        return Attitude(right=right, up=up, forward=forward)
    
    def compute_nadir_pointing_attitude(
        self,
        satellite_position: npt.NDArray[np.float64],
        satellite_velocity: npt.NDArray[np.float64],
        look_angle_deg: float = 30.0
    ) -> Attitude:
        """
        Compute nadir-pointing attitude (camera points toward Earth).
        
        This is used when no target is available. The camera points
        toward Earth at a specified angle from nadir.
        
        Parameters
        ----------
        satellite_position : ndarray of shape (3,)
            Satellite position in ECI frame [m]
        satellite_velocity : ndarray of shape (3,)
            Satellite velocity in ECI frame [m/s]
        look_angle_deg : float, optional
            Angle from nadir (toward flight direction) [degrees], default 30°
            
        Returns
        -------
        Attitude
            Nadir-pointing orientation
            
        Notes
        -----
        The forward vector is tilted from nadir by look_angle_deg
        in the direction of satellite motion.
        
        Examples
        --------
        >>> controller = NadirPointingController()
        >>> attitude = controller.compute_nadir_pointing_attitude(
        ...     satellite_position=np.array([7e6, 0, 0]),
        ...     satellite_velocity=np.array([0, 7.5e3, 0]),
        ...     look_angle_deg=30.0
        ... )
        """
        sat_pos = np.asarray(satellite_position, dtype=np.float64)
        sat_vel = np.asarray(satellite_velocity, dtype=np.float64)
        
        # Nadir direction (toward Earth center)
        nadir = -sat_pos
        nadir = nadir / np.linalg.norm(nadir)
        
        # Right vector (perpendicular to orbital plane)
        right = np.cross(sat_vel, nadir)
        right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            # Degenerate case
            right = self._get_perpendicular_vector(nadir)
        else:
            right = right / right_norm
        
        # Rotate nadir by look_angle toward velocity direction
        forward = self._rotate_vector_around_axis(
            vector=sat_pos,  # Note: rotation of +position, not nadir
            axis=right,
            angle_deg=look_angle_deg
        )
        forward = forward / np.linalg.norm(forward)
        
        # Compute up
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        
        # Re-orthogonalize
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        return Attitude(right=right, up=up, forward=forward)
    
    @staticmethod
    def _get_perpendicular_vector(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Get a vector perpendicular to v.
        
        Parameters
        ----------
        v : ndarray of shape (3,)
            Input vector
            
        Returns
        -------
        perp : ndarray of shape (3,)
            Normalized vector perpendicular to v
            
        Notes
        -----
        Uses the cross product with [1,0,0] or [0,1,0] depending on
        which gives a non-zero result.
        """
        v = v / np.linalg.norm(v)
        
        # Try cross product with [1, 0, 0]
        perp = np.cross(v, np.array([1, 0, 0]))
        
        if np.linalg.norm(perp) < 1e-6:
            # v is parallel to [1,0,0], try [0,1,0]
            perp = np.cross(v, np.array([0, 1, 0]))
        
        return perp / np.linalg.norm(perp)
    
    @staticmethod
    def _rotate_vector_around_axis(
        vector: npt.NDArray[np.float64],
        axis: npt.NDArray[np.float64],
        angle_deg: float
    ) -> npt.NDArray[np.float64]:
        """
        Rotate a vector around an axis using Rodrigues' rotation formula.
        
        Parameters
        ----------
        vector : ndarray of shape (3,)
            Vector to rotate
        axis : ndarray of shape (3,)
            Rotation axis (will be normalized)
        angle_deg : float
            Rotation angle [degrees]
            
        Returns
        -------
        rotated : ndarray of shape (3,)
            Rotated vector
            
        Notes
        -----
        Rodrigues' formula:
        v_rot = v×cos(θ) + (k×v)×sin(θ) + k×(k·v)×(1-cos(θ))
        
        where k is the unit axis vector.
        
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        """
        # Normalize axis
        k = axis / np.linalg.norm(axis)
        
        # Convert angle to radians
        theta = np.radians(angle_deg)
        
        # Rodrigues' formula components
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # v_rot = v×cos(θ) + (k×v)×sin(θ) + k×(k·v)×(1-cos(θ))
        term1 = vector * cos_theta
        term2 = np.cross(k, vector) * sin_theta
        term3 = k * np.dot(k, vector) * (1 - cos_theta)
        
        rotated = term1 + term2 + term3
        
        return rotated


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_line_of_sight(
    observer_position: npt.NDArray[np.float64],
    target_position: npt.NDArray[np.float64],
    earth_radius: float = EARTH_RADIUS_M
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


def are_vectors_parallel(
    v1: npt.NDArray[np.float64],
    v2: npt.NDArray[np.float64],
    tolerance: float = 1e-8
) -> bool:
    """
    Check if two vectors are parallel (or anti-parallel).
    
    Parameters
    ----------
    v1 : ndarray of shape (3,)
        First vector
    v2 : ndarray of shape (3,)
        Second vector
    tolerance : float, optional
        Tolerance for parallelism check
        
    Returns
    -------
    parallel : bool
        True if vectors are (anti-)parallel
        
    Notes
    -----
    Two vectors are parallel if their normalized dot product is ±1.
    
    Examples
    --------
    >>> v1 = np.array([1, 0, 0])
    >>> v2 = np.array([2, 0, 0])
    >>> are_vectors_parallel(v1, v2)
    True
    >>> 
    >>> v3 = np.array([0, 1, 0])
    >>> are_vectors_parallel(v1, v3)
    False
    """
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    dot_product = np.dot(v1_norm, v2_norm)
    
    return np.isclose(abs(dot_product), 1.0, rtol=tolerance)
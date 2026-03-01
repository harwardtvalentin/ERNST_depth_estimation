"""
Domain objects for satellites.

This module defines what satellites ARE, not what they DO.
Contains only data structures and simple accessors - no computation.

Classes
-------
Value Objects (Immutable):
    Attitude - Satellite orientation in 3D space
    SatelliteState - Complete state snapshot at one instant
    OrbitalElements - Keplerian orbit description

Specifications (Immutable):
    CameraSpecification - Camera hardware properties
    SatelliteSpecification - Complete satellite design

Entities (Mutable):
    Satellite - Unique satellite with state history

Functions
---------
    validate_satellite_trajectory - Check trajectory sanity
    interpolate_state - Linear interpolation between states
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.constants import EARTH_RADIUS_M, EARTH_MU


# =============================================================================
# VALUE OBJECTS (Immutable snapshots and descriptions)
# =============================================================================

@dataclass(frozen=True)
class Attitude:
    """
    Satellite orientation in 3D space.
    
    Represents a right-handed orthonormal coordinate frame defined by
    three unit vectors: right, up, and forward.
    
    Attributes
    ----------
    right : ndarray of shape (3,)
        Unit vector pointing "right" from satellite's perspective
        (Typically perpendicular to orbital plane)
    up : ndarray of shape (3,)
        Unit vector pointing "up" from satellite's perspective
        (Typically completes right-handed system)
    forward : ndarray of shape (3,)
        Unit vector pointing "forward" (camera look direction)
        (Typically toward nadir or target)
        
    Notes
    -----
    The three vectors must form an orthonormal basis:
    - Each vector has magnitude 1
    - Vectors are mutually perpendicular
    - right × up = forward (right-handed convention)
    
    Examples
    --------
    >>> import numpy as np
    >>> att = Attitude(
    ...     right=np.array([1, 0, 0]),
    ...     up=np.array([0, 1, 0]),
    ...     forward=np.array([0, 0, 1])
    ... )
    >>> print(att.rotation_matrix)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    """
    right: npt.NDArray[np.float64]
    up: npt.NDArray[np.float64]
    forward: npt.NDArray[np.float64]
    
    def __post_init__(self):
        """Validate that vectors are orthonormal."""
        # Convert to numpy arrays if needed
        object.__setattr__(self, 'right', np.asarray(self.right, dtype=np.float64))
        object.__setattr__(self, 'up', np.asarray(self.up, dtype=np.float64))
        object.__setattr__(self, 'forward', np.asarray(self.forward, dtype=np.float64))
        
        # Check shapes
        for name, vec in [('right', self.right), ('up', self.up), ('forward', self.forward)]:
            if vec.shape != (3,):
                raise ValueError(f"{name} must be 3D vector, got shape {vec.shape}")
        
        # Check unit vectors (with tolerance)
        tol = 1e-6
        for name, vec in [('right', self.right), ('up', self.up), ('forward', self.forward)]:
            norm = np.linalg.norm(vec)
            if abs(norm - 1.0) > tol:
                raise ValueError(f"{name} must be unit vector, |{name}| = {norm:.6f}")
        
        # Check orthogonality
        if abs(np.dot(self.right, self.up)) > tol:
            raise ValueError(f"right and up must be orthogonal, dot product = {np.dot(self.right, self.up):.6e}")
        if abs(np.dot(self.right, self.forward)) > tol:
            raise ValueError(f"right and forward must be orthogonal, dot product = {np.dot(self.right, self.forward):.6e}")
        if abs(np.dot(self.up, self.forward)) > tol:
            raise ValueError(f"up and forward must be orthogonal, dot product = {np.dot(self.up, self.forward):.6e}")
    
    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """
        3x3 rotation matrix (rows are right, up, forward).
        
        This matrix transforms vectors from world frame to satellite frame.
        
        Returns
        -------
        ndarray of shape (3, 3)
            Rotation matrix where row i is the i-th basis vector
        """
        return np.vstack([self.right, self.up, self.forward])
    
    @property
    def rotation_matrix_inverse(self) -> npt.NDArray[np.float64]:
        """
        Inverse rotation (world ← satellite frame).
        
        For orthonormal matrices, inverse = transpose.
        
        Returns
        -------
        ndarray of shape (3, 3)
            Inverse rotation matrix
        """
        return self.rotation_matrix.T
    
    def world_to_satellite(self, vector_world: npt.NDArray) -> npt.NDArray:
        """
        Transform vector from world frame to satellite frame.
        
        Parameters
        ----------
        vector_world : ndarray of shape (3,)
            Vector in world coordinates (ECI frame)
            
        Returns
        -------
        ndarray of shape (3,)
            Same vector in satellite body coordinates
            
        Examples
        --------
        >>> att = Attitude(
        ...     right=np.array([1, 0, 0]),
        ...     up=np.array([0, 1, 0]),
        ...     forward=np.array([0, 0, 1])
        ... )
        >>> world_vec = np.array([1, 2, 3])
        >>> sat_vec = att.world_to_satellite(world_vec)
        >>> print(sat_vec)
        [1. 2. 3.]
        """
        return self.rotation_matrix @ vector_world
    
    def satellite_to_world(self, vector_satellite: npt.NDArray) -> npt.NDArray:
        """
        Transform vector from satellite frame to world frame.
        
        Parameters
        ----------
        vector_satellite : ndarray of shape (3,)
            Vector in satellite body coordinates
            
        Returns
        -------
        ndarray of shape (3,)
            Same vector in world coordinates (ECI frame)
            
        Examples
        --------
        >>> att = Attitude(
        ...     right=np.array([1, 0, 0]),
        ...     up=np.array([0, 1, 0]),
        ...     forward=np.array([0, 0, 1])
        ... )
        >>> sat_vec = np.array([1, 0, 0])
        >>> world_vec = att.satellite_to_world(sat_vec)
        >>> print(world_vec)
        [1. 0. 0.]
        """
        return self.rotation_matrix_inverse @ vector_satellite


@dataclass(frozen=True)
class SatelliteState:
    """
    Complete satellite state at a single instant in time.
    
    This is a snapshot - immutable data representing everything about
    the satellite at one moment. Think of it like a photograph.
    
    Attributes
    ----------
    timestamp : datetime
        Time instant this state represents
    position : ndarray of shape (3,)
        Satellite position in ECI (Earth-Centered Inertial) frame [m]
    velocity : ndarray of shape (3,)
        Satellite velocity in ECI frame [m/s]
    attitude : Attitude, optional
        Satellite orientation (None if not yet computed)
        
    Notes
    -----
    Immutable (frozen=True) so we can safely share states without
    worrying about accidental modification.
    
    ECI frame: X points to vernal equinox, Z points to North pole,
    Y completes right-handed system.
    
    Examples
    --------
    >>> from datetime import datetime
    >>> state = SatelliteState(
    ...     timestamp=datetime(2026, 2, 9, 14, 30, 0),
    ...     position=np.array([7.0e6, 0, 0]),
    ...     velocity=np.array([0, 7.5e3, 0]),
    ...     attitude=None
    ... )
    >>> print(f"Altitude: {state.altitude/1000:.1f} km")
    Altitude: 622.0 km
    """
    timestamp: datetime
    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    attitude: Optional[Attitude] = None
    
    def __post_init__(self):
        """Validate state data."""
        # Convert to numpy arrays
        object.__setattr__(self, 'position', np.asarray(self.position, dtype=np.float64))
        object.__setattr__(self, 'velocity', np.asarray(self.velocity, dtype=np.float64))
        
        # Check shapes
        if self.position.shape != (3,):
            raise ValueError(f"Position must be 3D, got shape {self.position.shape}")
        if self.velocity.shape != (3,):
            raise ValueError(f"Velocity must be 3D, got shape {self.velocity.shape}")
    
    @property
    def altitude(self, earth_radius: float = EARTH_RADIUS_M) -> float:
        """
        Altitude above Earth's surface [m].
        
        Parameters
        ----------
        earth_radius : float, optional
            Earth's radius [m], default is 6.378e6
            
        Returns
        -------
        float
            Altitude in meters
        """
        distance_from_center = np.linalg.norm(self.position)
        return distance_from_center - earth_radius
    
    @property
    def speed(self) -> float:
        """
        Velocity magnitude [m/s].
        
        Returns
        -------
        float
            Speed in meters per second
        """
        return np.linalg.norm(self.velocity)
    
    def with_attitude(self, attitude: Attitude) -> 'SatelliteState':
        """
        Create new state with updated attitude.
        
        Since states are immutable, we create a new one.
        
        Parameters
        ----------
        attitude : Attitude
            New orientation
            
        Returns
        -------
        SatelliteState
            New state with same position/velocity but updated attitude
            
        Examples
        --------
        >>> state1 = SatelliteState(...)
        >>> attitude = Attitude(...)
        >>> state2 = state1.with_attitude(attitude)
        >>> assert state1.attitude is None
        >>> assert state2.attitude is attitude
        """
        return SatelliteState(
            timestamp=self.timestamp,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=attitude
        )


@dataclass(frozen=True)
class OrbitalElements:
    """
    Keplerian orbital elements defining an orbit.
    
    These are the six classical elements that uniquely define
    an orbit in the two-body problem (satellite around Earth).
    
    Attributes
    ----------
    semi_major_axis : float
        Semi-major axis [m] - defines orbit size
    eccentricity : float
        Eccentricity [dimensionless] - defines orbit shape (0 = circular, <1 = ellipse)
    inclination : float
        Inclination [degrees] - tilt of orbital plane relative to equator
    raan : float
        Right Ascension of Ascending Node [degrees] - rotates orbital plane around Z axis
    arg_perigee : float
        Argument of perigee [degrees] - rotation within orbital plane
    true_anomaly : float, optional
        Initial position in orbit [degrees], default 0
        
    Notes
    -----
    Validation in __post_init__ ensures physical validity.
    
    References
    ----------
    .. [1] Curtis, H. D. "Orbital Mechanics for Engineering Students"
    
    Examples
    --------
    >>> elements = OrbitalElements(
    ...     semi_major_axis=6.9e6,
    ...     eccentricity=0.01,
    ...     inclination=97.5,
    ...     raan=0.0,
    ...     arg_perigee=0.0
    ... )
    >>> print(f"Period: {elements.period/60:.1f} minutes")
    Period: 97.2 minutes
    """
    semi_major_axis: float      # a [m]
    eccentricity: float         # e [dimensionless]
    inclination: float          # i [deg]
    raan: float                 # Ω (Omega) [deg]
    arg_perigee: float          # ω (omega) [deg]
    true_anomaly: float = 0.0   # ν (nu) [deg]
    
    def __post_init__(self):
        """Validate orbital elements are physically reasonable."""
        if self.semi_major_axis <= 0:
            raise ValueError(f"Semi-major axis must be positive: {self.semi_major_axis}")
        
        if not 0 <= self.eccentricity < 1:
            raise ValueError(f"Eccentricity must be in [0,1) for elliptical orbit: {self.eccentricity}")
        
        if not 0 <= self.inclination <= 180:
            raise ValueError(f"Inclination must be in [0,180] degrees: {self.inclination}")
    
    @classmethod
    def from_apogee_perigee(
        cls,
        apogee_altitude: float,
        perigee_altitude: float,
        inclination: float,
        raan: float = 0.0,
        arg_perigee: float = 0.0,
        true_anomaly: float = 0.0,
        earth_radius: float = EARTH_RADIUS_M
    ) -> 'OrbitalElements':
        """
        Factory method: Create OrbitalElements from apogee and perigee altitudes.
        
        This is more convenient when you have altitude data instead of
        semi-major axis and eccentricity.
        
        Parameters
        ----------
        apogee_altitude : float
            Highest altitude above Earth's surface [m]
        perigee_altitude : float
            Lowest altitude above Earth's surface [m]
        inclination : float
            Orbital inclination [degrees]
        raan : float, optional
            Right ascension of ascending node [degrees], default 0
        arg_perigee : float, optional
            Argument of perigee [degrees], default 0
        true_anomaly : float, optional
            Initial true anomaly [degrees], default 0
        earth_radius : float, optional
            Earth's radius [m], default 6.378e6
            
        Returns
        -------
        OrbitalElements
            Constructed orbital elements
            
        Examples
        --------
        >>> elements = OrbitalElements.from_apogee_perigee(
        ...     apogee_altitude=519986.0,
        ...     perigee_altitude=514905.0,
        ...     inclination=97.5
        ... )
        >>> print(f"Semi-major axis: {elements.semi_major_axis/1e6:.3f} Mm")
        Semi-major axis: 6.895 Mm
        """
        # Validate inputs
        if apogee_altitude < 0:
            raise ValueError(f"Apogee altitude cannot be negative: {apogee_altitude}")
        if perigee_altitude < 0:
            raise ValueError(f"Perigee altitude cannot be negative: {perigee_altitude}")
        if apogee_altitude < perigee_altitude:
            raise ValueError(
                f"Apogee altitude ({apogee_altitude}) must be >= perigee altitude ({perigee_altitude})"
            )
        
        # Convert altitudes to radii (from Earth's center)
        r_apogee = apogee_altitude + earth_radius
        r_perigee = perigee_altitude + earth_radius
        
        # Compute semi-major axis
        a = (r_apogee + r_perigee) / 2
        
        # Compute eccentricity
        e = (r_apogee - r_perigee) / (r_apogee + r_perigee)
        
        # Call the standard constructor
        return cls(
            semi_major_axis=a,
            eccentricity=e,
            inclination=inclination,
            raan=raan,
            arg_perigee=arg_perigee,
            true_anomaly=true_anomaly
        )
    
    @property
    def apogee_radius(self) -> float:
        """
        Apogee distance from Earth's center [m].
        
        Formula: r_apogee = a × (1 + e)
        
        Returns
        -------
        float
            Apogee radius in meters
        """
        return self.semi_major_axis * (1 + self.eccentricity)
    
    @property
    def perigee_radius(self) -> float:
        """
        Perigee distance from Earth's center [m].
        
        Formula: r_perigee = a × (1 - e)
        
        Returns
        -------
        float
            Perigee radius in meters
        """
        return self.semi_major_axis * (1 - self.eccentricity)
    
    @property
    def apogee_altitude(self, earth_radius: float = EARTH_RADIUS_M) -> float:
        """
        Apogee altitude above Earth's surface [m].
        
        Returns
        -------
        float
            Apogee altitude in meters
        """
        return self.apogee_radius - earth_radius
    
    @property
    def perigee_altitude(self, earth_radius: float = EARTH_RADIUS_M) -> float:
        """
        Perigee altitude above Earth's surface [m].
        
        Returns
        -------
        float
            Perigee altitude in meters
        """
        return self.perigee_radius - earth_radius
    
    @property
    def period(self, mu: float = EARTH_MU) -> float:
        """
        Orbital period [seconds].
        
        Uses Kepler's third law: T = 2π√(a³/μ)
        
        Parameters
        ----------
        mu : float, optional
            Earth's gravitational parameter [m³/s²], default 3.986004418e14
            
        Returns
        -------
        float
            Orbital period in seconds
        """
        return 2 * np.pi * np.sqrt(self.semi_major_axis**3 / mu)
    
    @property
    def mean_motion(self, mu: float = EARTH_MU) -> float:
        """
        Mean motion [rad/s].
        
        Formula: n = 2π / T = √(μ / a³)
        
        Parameters
        ----------
        mu : float, optional
            Earth's gravitational parameter [m³/s²]
            
        Returns
        -------
        float
            Mean motion in radians per second
        """
        return np.sqrt(mu / self.semi_major_axis**3)


# =============================================================================
# SPECIFICATIONS (Immutable blueprints)
# =============================================================================

@dataclass(frozen=True)
class CameraSpecification:
    """
    Physical specification of the satellite's camera.
    
    This describes the camera's properties (like a manufacturer's spec sheet),
    not how it projects images (that's in sensing/camera_model.py).
    
    Attributes
    ----------
    resolution : tuple of int
        Image resolution (width, height) in pixels
    fov_horizontal_deg : float
        Horizontal field of view [degrees]
    fps : int, optional
        Frame rate [frames per second], default 30
    sensor_width_mm : float, optional
        Physical sensor width [mm] (if known)
    sensor_height_mm : float, optional
        Physical sensor height [mm] (if known)
    
    Notes
    -----
    The vertical FOV is derived from horizontal FOV and aspect ratio,
    assuming square pixels and a pinhole camera model.
    
    Examples
    --------
    >>> camera = CameraSpecification(
    ...     resolution=(1024, 720),
    ...     fov_horizontal_deg=30.0,
    ...     fps=30
    ... )
    >>> print(f"Vertical FOV: {camera.fov_vertical_deg:.2f}°")
    Vertical FOV: 21.34°
    """
    resolution: Tuple[int, int]
    fov_horizontal_deg: float
    fps: int = 30
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    
    def __post_init__(self):
        """Validate camera parameters."""
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError(f"Resolution must be positive: {self.resolution}")
        
        if not 0 < self.fov_horizontal_deg < 180:
            raise ValueError(f"FOV must be in (0,180) degrees: {self.fov_horizontal_deg}")
        
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive: {self.fps}")
    
    @property
    def aspect_ratio(self) -> float:
        """
        Image aspect ratio (height / width).
        
        Returns
        -------
        float
            Aspect ratio
        """
        return self.resolution[1] / self.resolution[0]
    
    @property
    def fov_vertical_deg(self) -> float:
        """
        Vertical field of view [degrees].
        
        Computed from horizontal FOV and aspect ratio assuming
        square pixels and pinhole camera model.
        
        Formula: tan(VFOV/2) = tan(HFOV/2) × (H / W)
        
        Returns
        -------
        float
            Vertical FOV in degrees
        """
        hfov_rad = np.radians(self.fov_horizontal_deg)
        vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * self.aspect_ratio)
        return np.degrees(vfov_rad)
    
    @property
    def frame_duration(self) -> float:
        """
        Duration of one frame [seconds].
        
        Returns
        -------
        float
            Frame duration in seconds
        """
        return 1.0 / self.fps


@dataclass(frozen=True)
class SatelliteSpecification:
    """
    Complete specification of a satellite.
    
    This describes everything ABOUT the satellite that doesn't change
    over time. Think of this as the "design document" or "blueprint".
    
    Attributes
    ----------
    name : str
        Satellite identifier (e.g., "ERNST")
    camera : CameraSpecification
        Onboard camera specification
    orbital_elements : OrbitalElements
        Initial orbital parameters
    
    Notes
    -----
    This is immutable (frozen) because a satellite's design doesn't
    change during the simulation.
    
    Examples
    --------
    >>> spec = SatelliteSpecification(
    ...     name="ERNST",
    ...     camera=CameraSpecification(
    ...         resolution=(1024, 720),
    ...         fov_horizontal_deg=30.0,
    ...         fps=30
    ...     ),
    ...     orbital_elements=OrbitalElements.from_apogee_perigee(
    ...         apogee_altitude=519986.0,
    ...         perigee_altitude=514905.0,
    ...         inclination=97.5
    ...     )
    ... )
    >>> print(spec.name)
    ERNST
    """
    name: str
    camera: CameraSpecification
    orbital_elements: OrbitalElements
    
    def __post_init__(self):
        """Validate satellite specification."""
        
        if not self.name:
            raise ValueError("Satellite name cannot be empty")


# =============================================================================
# ENTITY (Mutable container for state history)
# =============================================================================

class Satellite:
    """
    Satellite entity - manages state history.
    
    This class is a CONTAINER for satellite data. It does NOT:
    - Compute orbital mechanics
    - Calculate attitudes
    - Project to camera
    - Estimate anything
    
    It ONLY:
    - Stores the satellite's specification
    - Stores the satellite's state history (list of snapshots)
    - Provides simple accessors to that data
    
    Attributes
    ----------
    spec : SatelliteSpecification
        Time-invariant satellite properties
    states : list of SatelliteState
        Trajectory history (chronologically ordered snapshots)
    
    Examples
    --------
    >>> spec = SatelliteSpecification(...)
    >>> satellite = Satellite(spec)
    >>> 
    >>> # Later, physics engine adds states:
    >>> state = SatelliteState(
    ...     timestamp=datetime.now(),
    ...     position=np.array([7000e3, 0, 0]),
    ...     velocity=np.array([0, 7500, 0])
    ... )
    >>> satellite.add_state(state)
    >>> print(f"Satellite has {satellite.num_states} states")
    Satellite has 1 states
    """
    
    def __init__(self, spec: SatelliteSpecification):
        """
        Initialize satellite with specification.
        
        Parameters
        ----------
        spec : SatelliteSpecification
            Complete satellite design/properties
        """
        self.spec = spec
        self.states: List[SatelliteState] = []
    
    # =========================================================================
    # STATE MANAGEMENT (Simple CRUD operations)
    # =========================================================================
    
    def add_state(self, state: SatelliteState) -> None:
        """
        Add a state snapshot to history.
        
        States should be added in chronological order.
        
        Parameters
        ----------
        state : SatelliteState
            Snapshot to add
        """
        self.states.append(state)
    
    def add_states(self, states: List[SatelliteState]) -> None:
        """
        Add multiple states at once.
        
        Parameters
        ----------
        states : list of SatelliteState
            Snapshots to add
        """
        self.states.extend(states)
    
    def clear_states(self) -> None:
        """Remove all state history."""
        self.states.clear()
    
    def get_state_at_index(self, index: int) -> SatelliteState:
        """
        Get state at specific index.
        
        Parameters
        ----------
        index : int
            Index into state history
            
        Returns
        -------
        SatelliteState
            State at that index
            
        Raises
        ------
        IndexError
            If index out of range
        """
        return self.states[index]
    
    def get_state_at_time(
        self,
        timestamp: datetime,
        tolerance_seconds: float = 1.0
    ) -> Optional[SatelliteState]:
        """
        Get state at specific time (nearest match).
        
        This is a SIMPLE LOOKUP, not interpolation.
        For interpolation, use interpolate_state() function.
        
        Parameters
        ----------
        timestamp : datetime
            Desired time
        tolerance_seconds : float, optional
            Maximum time difference to accept [seconds], default 1.0
            
        Returns
        -------
        SatelliteState or None
            Nearest state within tolerance, or None if none found
            
        Examples
        --------
        >>> state = satellite.get_state_at_time(
        ...     datetime(2026, 2, 9, 14, 30, 0),
        ...     tolerance_seconds=0.5
        ... )
        """
        if not self.states:
            return None
        
        # Find nearest state
        nearest_state = min(
            self.states,
            key=lambda s: abs((s.timestamp - timestamp).total_seconds())
        )
        
        # Check tolerance
        time_diff = abs((nearest_state.timestamp - timestamp).total_seconds())
        if time_diff <= tolerance_seconds:
            return nearest_state
        else:
            return None
    
    # =========================================================================
    # DERIVED PROPERTIES (Simple computations from existing data)
    # =========================================================================
    
    @property
    def num_states(self) -> int:
        """Number of states in history."""
        return len(self.states)
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Time of first state, or None if no states."""
        return self.states[0].timestamp if self.states else None
    
    @property
    def end_time(self) -> Optional[datetime]:
        """Time of last state, or None if no states."""
        return self.states[-1].timestamp if self.states else None
    
    @property
    def duration(self) -> Optional[float]:
        """
        Total duration of trajectory [seconds].
        
        Returns None if fewer than 2 states.
        """
        if len(self.states) < 2:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """
        All positions as (N, 3) array.
        
        Returns
        -------
        ndarray of shape (N, 3)
            Position history, where N = number of states
        """
        if not self.states:
            return np.array([]).reshape(0, 3)
        return np.array([state.position for state in self.states])
    
    @property
    def velocities(self) -> npt.NDArray[np.float64]:
        """
        All velocities as (N, 3) array.
        
        Returns
        -------
        ndarray of shape (N, 3)
            Velocity history
        """
        if not self.states:
            return np.array([]).reshape(0, 3)
        return np.array([state.velocity for state in self.states])
    
    @property
    def timestamps(self) -> List[datetime]:
        """All timestamps as list."""
        return [state.timestamp for state in self.states]
    
    @property
    def has_attitudes(self) -> bool:
        """Check if all states have attitudes computed."""
        if not self.states:
            return False
        return all(state.attitude is not None for state in self.states)
    
    # =========================================================================
    # CONVENIENCE METHODS (Still just data access)
    # =========================================================================
    
    def get_position_at_time(
        self,
        timestamp: datetime,
        tolerance_seconds: float = 1.0
    ) -> Optional[npt.NDArray[np.float64]]:
        """Get position at specific time."""
        state = self.get_state_at_time(timestamp, tolerance_seconds)
        return state.position if state else None
    
    def get_velocity_at_time(
        self,
        timestamp: datetime,
        tolerance_seconds: float = 1.0
    ) -> Optional[npt.NDArray[np.float64]]:
        """Get velocity at specific time."""
        state = self.get_state_at_time(timestamp, tolerance_seconds)
        return state.velocity if state else None
    
    def get_attitude_at_time(
        self,
        timestamp: datetime,
        tolerance_seconds: float = 1.0
    ) -> Optional[Attitude]:
        """Get attitude at specific time."""
        state = self.get_state_at_time(timestamp, tolerance_seconds)
        return state.attitude if state else None
    
    # =========================================================================
    # STRING REPRESENTATION (Debugging)
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Satellite(name='{self.spec.name}', "
            f"num_states={self.num_states}"
        )
    
    def __str__(self) -> str:
        """Human-readable description."""
        if not self.states:
            return f"Satellite '{self.spec.name}' (no trajectory data)"
        
        altitude_start = self.states[0].altitude / 1e3
        altitude_end = self.states[-1].altitude / 1e3
        
        return (
            f"Satellite '{self.spec.name}':\n"
            f"  Camera: {self.spec.camera.resolution[0]}x{self.spec.camera.resolution[1]} "
            f"@ {self.spec.camera.fps} fps\n"
            f"  Trajectory: {self.num_states} states\n"
            f"  Duration: {self.duration:.1f} seconds\n"
            f"  Altitudes: {altitude_start:.1f} - {altitude_end:.1f} km"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_satellite_trajectory(
    satellite: Satellite,
    earth_radius: float = EARTH_RADIUS_M
) -> List[str]:
    """
    Validate that satellite trajectory is physically reasonable.
    
    This is a VALIDATION function - checks data quality, doesn't compute.
    
    Parameters
    ----------
    satellite : Satellite
        Satellite to validate
    earth_radius : float, optional
        Earth's radius [m], default 6.378e6
        
    Returns
    -------
    list of str
        List of validation errors (empty if valid)
        
    Examples
    --------
    >>> errors = validate_satellite_trajectory(satellite)
    >>> if errors:
    ...     print("Validation failed:")
    ...     for error in errors:
    ...         print(f"  - {error}")
    ... else:
    ...     print("Trajectory is valid")
    """
    errors = []
    
    if not satellite.states:
        errors.append("No trajectory data")
        return errors
    
    # Check that states are chronologically ordered
    for i in range(len(satellite.states) - 1):
        if satellite.states[i+1].timestamp <= satellite.states[i].timestamp:
            errors.append(f"States not in chronological order at index {i}")
    
    # Check altitudes are above Earth's surface
    for i, state in enumerate(satellite.states):
        altitude = state.altitude
        if altitude < 0:
            errors.append(
                f"State {i}: satellite below Earth's surface "
                f"(altitude={altitude/1e3:.1f} km)"
            )
    
    # Check speeds are reasonable (< escape velocity)
    escape_velocity = 11.2e3  # m/s
    for i, state in enumerate(satellite.states):
        if state.speed > escape_velocity:
            errors.append(
                f"State {i}: speed exceeds escape velocity "
                f"({state.speed/1e3:.1f} km/s > {escape_velocity/1e3:.1f} km/s)"
            )
    
    return errors


def interpolate_state(
    state1: SatelliteState,
    state2: SatelliteState,
    timestamp: datetime
) -> SatelliteState:
    """
    Linearly interpolate between two states.
    
    This is a UTILITY FUNCTION - simple math on data, not physics.
    
    Parameters
    ----------
    state1 : SatelliteState
        Earlier state
    state2 : SatelliteState
        Later state
    timestamp : datetime
        Desired time (must be between state1 and state2)
        
    Returns
    -------
    SatelliteState
        Interpolated state
        
    Raises
    ------
    ValueError
        If timestamp is outside [state1.timestamp, state2.timestamp]
        
    Notes
    -----
    For attitude, uses nearest neighbor (no spherical interpolation).
    For true orbital mechanics, use a proper propagator instead.
    
    Examples
    --------
    >>> state_interp = interpolate_state(
    ...     state_at_t0, state_at_t1,
    ...     timestamp_between
    ... )
    """
    # Validate order
    if not state1.timestamp <= timestamp <= state2.timestamp:
        raise ValueError(
            f"Timestamp {timestamp} not in range "
            f"[{state1.timestamp}, {state2.timestamp}]"
        )
    
    # Compute interpolation factor
    total_dt = (state2.timestamp - state1.timestamp).total_seconds()
    partial_dt = (timestamp - state1.timestamp).total_seconds()
    alpha = partial_dt / total_dt if total_dt > 0 else 0.0
    
    # Interpolate position and velocity
    position = (1 - alpha) * state1.position + alpha * state2.position
    velocity = (1 - alpha) * state1.velocity + alpha * state2.velocity
    
    # For attitude, take nearest (proper method would be SLERP)
    attitude = state1.attitude if alpha < 0.5 else state2.attitude
    
    return SatelliteState(
        timestamp=timestamp,
        position=position,
        velocity=velocity,
        attitude=attitude
    )
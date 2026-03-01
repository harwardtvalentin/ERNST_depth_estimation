"""
Domain objects for missiles.

This module defines what missiles ARE, not what they DO.
Contains only data structures and simple accessors - no computation.

Classes
-------
Value Objects (Immutable):
    MissileState - Complete state snapshot at one instant

Entities (Mutable):
    Missile - Unique missile with state history

Functions
---------
    validate_missile_trajectory - Check trajectory sanity
    interpolate_missile_state - Linear interpolation between states
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.constants import EARTH_RADIUS_M


# =============================================================================
# VALUE OBJECTS (Immutable snapshots and descriptions)
# =============================================================================

@dataclass(frozen=True)
class MissileState:
    """
    Complete missile state at a single instant in time.
    
    This is a snapshot - immutable data representing everything about
    the missile at one moment.
    
    Attributes
    ----------
    timestamp : datetime
        Time instant this state represents
    position : ndarray of shape (3,)
        Missile position in ECI (Earth-Centered Inertial) frame [m]
    velocity : ndarray of shape (3,)
        Missile velocity in ECI frame [m/s]
        
    Notes
    -----
    Immutable (frozen=True) so we can safely share states without
    worrying about accidental modification.
    
    For ballistic missiles, velocity is often constant (no propulsion phase).
    For more complex missiles, velocity would vary over time.
    
    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> state = MissileState(
    ...     timestamp=datetime(2026, 2, 9, 14, 30, 0),
    ...     position=np.array([7.0e6, 0, 0]),
    ...     velocity=np.array([100, 50, 0])
    ... )
    >>> print(f"Speed: {state.speed:.1f} m/s")
    Speed: 111.8 m/s
    """
    timestamp: datetime
    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    
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


# =============================================================================
# ENTITY (Mutable container for state history)
# =============================================================================

class Missile:
    """
    Missile entity - manages state history.
    
    This class is a CONTAINER for missile data. It does NOT:
    - Compute trajectories
    - Perform physics simulations
    - Estimate anything
    
    It ONLY:
    - Stores the missile's state history (list of snapshots)
    - Provides simple accessors to that data
    
    Attributes
    ----------
    name : str
        Missile identifier
    states : list of MissileState
        Trajectory history (chronologically ordered snapshots)
    
    Examples
    --------
    >>> missile = Missile(name="Target-1")
    >>> 
    >>> # Later, physics engine adds states:
    >>> state = MissileState(
    ...     timestamp=datetime.now(),
    ...     position=np.array([7000e3, 0, 0]),
    ...     velocity=np.array([100, 50, 0])
    ... )
    >>> missile.add_state(state)
    >>> print(f"Missile has {missile.num_states} states")
    Missile has 1 states
    """
    
    def __init__(
        self,
        name: str = "Target"
    ):
        """
        Initialize missile.
        
        Parameters
        ----------
        name : str, optional
            Missile identifier, default "Target"
        """
        self.name = name
        self.states: List[MissileState] = []
    
    # =========================================================================
    # STATE MANAGEMENT (Simple CRUD operations)
    # =========================================================================
    
    def add_state(self, state: MissileState) -> None:
        """
        Add a state snapshot to history.
        
        States should be added in chronological order.
        
        Parameters
        ----------
        state : MissileState
            Snapshot to add
        """
        self.states.append(state)
    
    def add_states(self, states: List[MissileState]) -> None:
        """
        Add multiple states at once.
        
        Parameters
        ----------
        states : list of MissileState
            Snapshots to add
        """
        self.states.extend(states)
    
    def clear_states(self) -> None:
        """Remove all state history."""
        self.states.clear()
    
    def get_state_at_index(self, index: int) -> MissileState:
        """
        Get state at specific index.
        
        Parameters
        ----------
        index : int
            Index into state history
            
        Returns
        -------
        MissileState
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
    ) -> Optional[MissileState]:
        """
        Get state at specific time (nearest match).
        
        This is a SIMPLE LOOKUP, not interpolation.
        For interpolation, use interpolate_missile_state() function.
        
        Parameters
        ----------
        timestamp : datetime
            Desired time
        tolerance_seconds : float, optional
            Maximum time difference to accept [seconds], default 1.0
            
        Returns
        -------
        MissileState or None
            Nearest state within tolerance, or None if none found
            
        Examples
        --------
        >>> state = missile.get_state_at_time(
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
    
    # =========================================================================
    # STRING REPRESENTATION (Debugging)
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Missile(name='{self.name}', "
            f"num_states={self.num_states})"
        )
    
    def __str__(self) -> str:
        """Human-readable description."""
        if not self.states:
            return f"Missile '{self.name}' (no trajectory data)"
        
        altitude_start = self.states[0].altitude / 1e3
        altitude_end = self.states[-1].altitude / 1e3
        speed_avg = np.mean([s.speed for s in self.states])
        
        lines = [
            f"Missile '{self.name}':",
            f"  Trajectory: {self.num_states} states",
        ]
        
        if self.duration:
            lines.append(f"  Duration: {self.duration:.1f} seconds")
        
        lines.append(f"  Altitudes: {altitude_start:.1f} - {altitude_end:.1f} km")
        lines.append(f"  Average speed: {speed_avg:.1f} m/s")
        
        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# I THINK THIS FUNCTION IS NOT REQUIRED!
def validate_missile_trajectory(
    missile: Missile,
    earth_radius: float = EARTH_RADIUS_M
) -> List[str]:
    """
    Validate that missile trajectory is physically reasonable.
    
    This is a VALIDATION function - checks data quality, doesn't compute.
    
    Parameters
    ----------
    missile : Missile
        Missile to validate
    earth_radius : float, optional
        Earth's radius [m], default 6.378e6
        
    Returns
    -------
    list of str
        List of validation errors (empty if valid)
        
    Examples
    --------
    >>> errors = validate_missile_trajectory(missile)
    >>> if errors:
    ...     print("Validation failed:")
    ...     for error in errors:
    ...         print(f"  - {error}")
    ... else:
    ...     print("Trajectory is valid")
    """
    errors = []
    
    if not missile.states:
        errors.append("No trajectory data")
        return errors
    
    # Check that states are chronologically ordered
    for i in range(len(missile.states) - 1):
        if missile.states[i+1].timestamp <= missile.states[i].timestamp:
            errors.append(f"States not in chronological order at index {i}")
    
    # Check altitudes (missiles can be below surface briefly, but warn)
    num_below_surface = 0
    for i, state in enumerate(missile.states):
        altitude = state.altitude
        if altitude < 0:
            num_below_surface += 1
    
    if num_below_surface > 0:
        errors.append(
            f"{num_below_surface} states below Earth's surface "
            f"(may indicate impact or data error)"
        )
    
    # Check speeds are reasonable (< escape velocity, > 0)
    escape_velocity = 11.2e3  # m/s
    for i, state in enumerate(missile.states):
        if state.speed > escape_velocity:
            errors.append(
                f"State {i}: speed exceeds escape velocity "
                f"({state.speed/1e3:.1f} km/s > {escape_velocity/1e3:.1f} km/s)"
            )
        
        if state.speed < 0.1:
            errors.append(
                f"State {i}: speed suspiciously low ({state.speed:.3f} m/s)"
            )
    
    return errors


def interpolate_missile_state(
    state1: MissileState,
    state2: MissileState,
    timestamp: datetime
) -> MissileState:
    """
    Linearly interpolate missile state between two timestamps.
    
    For a ballistic missile with constant velocity, this is exact.
    For more complex motion, this is an approximation.
    
    Parameters
    ----------
    state1 : MissileState
        Earlier state
    state2 : MissileState
        Later state
    timestamp : datetime
        Desired time (must be between state1 and state2)
        
    Returns
    -------
    MissileState
        Interpolated state at timestamp
        
    Raises
    ------
    ValueError
        If timestamp is outside [state1.timestamp, state2.timestamp]
        
    Notes
    -----
    Linear interpolation is exact for constant-velocity trajectories.
    For accelerating missiles, use a proper ballistic propagator.
    
    Examples
    --------
    >>> state_interp = interpolate_missile_state(
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
    
    if total_dt == 0:
        return state1  # Same timestamp
    
    alpha = partial_dt / total_dt  # 0 to 1
    
    # Linear interpolation
    position = (1 - alpha) * state1.position + alpha * state2.position
    velocity = (1 - alpha) * state1.velocity + alpha * state2.velocity
    
    return MissileState(
        timestamp=timestamp,
        position=position,
        velocity=velocity
    )
"""
Orbital mechanics computations.

This module implements physics equations for satellite orbital motion
using Keplerian (two-body) mechanics.

Classes
-------
KeplerianOrbitPropagator
    Computes satellite trajectories from orbital elements

Notes
-----
Uses simplified two-body mechanics (Earth + satellite only).
Neglects:
- Atmospheric drag
- J2 perturbations (Earth oblateness)
- Third-body effects (Sun, Moon)
- Solar radiation pressure

These simplifications are acceptable for short-duration simulations
and typical LEO satellites.
"""

from datetime import datetime, timedelta
from typing import List
import numpy as np
import sys

# Import from our domain module
from missile_fly_by_simulation.domain import OrbitalElements, SatelliteState

# Physical constants
from missile_fly_by_simulation.constants import EARTH_MU




class KeplerianOrbitPropagator:
    """
    Propagate satellite orbit using Keplerian (two-body) mechanics.
    
    This computes satellite position and velocity at specified times
    using classical orbital mechanics equations.
    
    Attributes
    ----------
    elements : OrbitalElements
        Orbital elements defining the orbit
    mu : float
        Gravitational parameter [m³/s²], default is Earth's μ
    
    Notes
    -----
    The propagation is done in the following steps:
    1. Compute mean anomaly from time: M = n × t
    2. Solve Kepler's equation for eccentric anomaly: M = E - e×sin(E)
    3. Compute true anomaly from eccentric anomaly
    4. Compute position and velocity in orbital plane
    5. Rotate to ECI (Earth-Centered Inertial) frame
    
    References
    ----------
    .. [1] Curtis, H. D. "Orbital Mechanics for Engineering Students"
    .. [2] Vallado, D. A. "Fundamentals of Astrodynamics and Applications"
    
    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> elements = OrbitalElements.from_apogee_perigee(
    ...     apogee_altitude=519986.0,
    ...     perigee_altitude=514905.0,
    ...     inclination=97.5
    ... )
    >>> propagator = KeplerianOrbitPropagator(elements)
    >>> 
    >>> # Generate timestamps
    >>> start = datetime(2026, 2, 9, 14, 0, 0)
    >>> timestamps = [start + timedelta(seconds=i) for i in range(100)]
    >>> 
    >>> # Propagate orbit
    >>> states = propagator.propagate(timestamps)
    >>> print(f"Generated {len(states)} states")
    Generated 100 states
    """
    
    def __init__(
        self,
        elements: OrbitalElements,
        mu: float = EARTH_MU
    ):
        """
        Initialize orbit propagator.
        
        Parameters
        ----------
        elements : OrbitalElements
            Keplerian orbital elements
        mu : float, optional
            Gravitational parameter [m³/s²], default is Earth's μ
        """
        self.elements = elements
        self.mu = mu
        
        # Compute derived quantities
        self.mean_motion = np.sqrt(mu / elements.semi_major_axis**3)  # [rad/s]
        self.period = 2 * np.pi / self.mean_motion  # [s]
        
        # Precompute rotation matrix (orbital plane → ECI)
        self._rotation_matrix = self._compute_rotation_matrix()
    
    def propagate(
        self,
        timestamps: List[datetime],
        reference_time: datetime = None,
        show_progress: bool = False
    ) -> List[SatelliteState]:
        """
        Propagate orbit to generate states at set of specified times.
        
        Parameters
        ----------
        timestamps : list of datetime
            Times at which to compute satellite state
        reference_time : datetime, optional
            Reference time (epoch) for orbit. If None, uses first timestamp.
        show_progress : bool, optional
            If True, display progress bar
            
        Returns
        -------
        list of SatelliteState
            Satellite states at each timestamp
            
        Examples
        --------
        >>> states = propagator.propagate(
        ...     timestamps=[t0, t1, t2, ...],
        ...     reference_time=t0,
        ...     show_progress=True
        ... )
        """
        if not timestamps:
            return []
        
        if reference_time is None:
            reference_time = timestamps[0]
        
        states = []
        
        for i, timestamp in enumerate(timestamps):
            # Time since epoch [seconds]
            dt = (timestamp - reference_time).total_seconds()
            
            # Compute position and velocity
            position, velocity = self._compute_state_at_time(dt)
            
            # Create state object (no attitude yet)
            state = SatelliteState(
                timestamp=timestamp,
                position=position,
                velocity=velocity,
                attitude=None
            )
            states.append(state)
            
            # Progress bar
            if show_progress and (i % 10000 == 0 or i + 1 == len(timestamps)):
                self._print_progress(i + 1, len(timestamps), 'Propagating orbit')
        
        if show_progress:
            print()  # Newline after progress bar
        
        return states
    
    def _compute_state_at_time(self, t: float) -> tuple:
        """
        Compute position and velocity at time t.
        
        Parameters
        ----------
        t : float
            Time since epoch [seconds]
            
        Returns
        -------
        position : ndarray of shape (3,)
            Position in ECI frame [m]
        velocity : ndarray of shape (3,)
            Velocity in ECI frame [m/s]
        """
        # Extract orbital elements
        a = self.elements.semi_major_axis
        e = self.elements.eccentricity
        n = self.mean_motion
        
        # Step 1: Compute mean anomaly
        M = n * t  # [rad]
        
        # Step 2: Solve Kepler's equation for eccentric anomaly
        E = self._solve_kepler_equation(M, e)
        
        # Step 3: Compute true anomaly
        theta = 2 * np.arctan(
            np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)
        )
        
        # Step 4: Compute radius
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        
        # Step 5: Position in orbital plane
        x_orbital = r * np.cos(theta)
        y_orbital = r * np.sin(theta)
        z_orbital = 0.0
        position_orbital = np.array([x_orbital, y_orbital, z_orbital])
        
        # Step 6: Velocity in orbital plane
        # Angular momentum
        h = np.sqrt(self.mu * a * (1 - e**2))
        
        # Velocity components
        v_r = h * e * np.sin(theta) / (r * (1 - e**2))  # Radial
        v_theta = h / r  # Tangential
        
        vx_orbital = v_r * np.cos(theta) - v_theta * np.sin(theta)
        vy_orbital = v_r * np.sin(theta) + v_theta * np.cos(theta)
        vz_orbital = 0.0
        velocity_orbital = np.array([vx_orbital, vy_orbital, vz_orbital])
        
        # Step 7: Rotate to ECI frame
        position_eci = self._rotation_matrix @ position_orbital
        velocity_eci = self._rotation_matrix @ velocity_orbital
        
        return position_eci, velocity_eci
    
    def _solve_kepler_equation(
        self,
        M: float,
        e: float,
        tolerance: float = 1e-8,
        max_iterations: int = 100
    ) -> float:
        """
        Solve Kepler's equation M = E - e×sin(E) for eccentric anomaly E.
        
        Uses Newton-Raphson iteration.
        
        Parameters
        ----------
        M : float
            Mean anomaly [rad]
        e : float
            Eccentricity
        tolerance : float, optional
            Convergence tolerance
        max_iterations : int, optional
            Maximum number of iterations
            
        Returns
        -------
        E : float
            Eccentric anomaly [rad]
            
        Raises
        ------
        RuntimeError
            If iteration doesn't converge
            
        Notes
        -----
        Newton-Raphson formula:
        E_{n+1} = E_n - f(E_n) / f'(E_n)
        where f(E) = E - e×sin(E) - M
              f'(E) = 1 - e×cos(E)
        """
        # Initial guess (mean anomaly is good for low eccentricity)
        E = M
        
        for iteration in range(max_iterations):
            # Function and its derivative
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            
            # Newton-Raphson update
            E_next = E - f / f_prime
            
            # Check convergence
            if abs(E_next - E) < tolerance:
                return E_next
            
            E = E_next
        
        # Failed to converge
        raise RuntimeError(
            f"Kepler equation didn't converge after {max_iterations} iterations. "
            f"M={M:.6f}, e={e:.6f}, final E={E:.6f}"
        )
    
    def _compute_rotation_matrix(self) -> np.ndarray:
        """
        Compute rotation matrix from orbital plane to ECI frame.
        
        The rotation is: R = R_Ω × R_i × R_ω
        where:
        - R_Ω: rotation by RAAN (Ω) about Z-axis
        - R_i: rotation by inclination (i) about X-axis
        - R_ω: rotation by argument of perigee (ω) about Z-axis
        
        Returns
        -------
        R : ndarray of shape (3, 3)
            Rotation matrix (orbital → ECI)
        """
        # Convert to radians
        raan_rad = np.radians(self.elements.raan)
        inc_rad = np.radians(self.elements.inclination)
        arg_perigee_rad = np.radians(self.elements.arg_perigee)
        
        # Rotation matrices
        cos_raan, sin_raan = np.cos(raan_rad), np.sin(raan_rad)
        cos_inc, sin_inc = np.cos(inc_rad), np.sin(inc_rad)
        cos_omega, sin_omega = np.cos(arg_perigee_rad), np.sin(arg_perigee_rad)
        
        # R_Ω (rotation about Z by RAAN)
        R_raan = np.array([
            [cos_raan, -sin_raan, 0],
            [sin_raan,  cos_raan, 0],
            [0,         0,        1]
        ])
        
        # R_i (rotation about X by inclination)
        R_inc = np.array([
            [1, 0,        0       ],
            [0, cos_inc, -sin_inc],
            [0, sin_inc,  cos_inc]
        ])
        
        # R_ω (rotation about Z by argument of perigee)
        R_arg_perigee = np.array([
            [cos_omega, -sin_omega, 0],
            [sin_omega,  cos_omega, 0],
            [0,          0,         1]
        ])
        
        # Combined rotation
        R = R_raan @ R_inc @ R_arg_perigee
        
        return R
    
    @staticmethod
    def _print_progress(current: int, total: int, label: str = 'Progress'):
        """
        Print progress bar.
        
        Parameters
        ----------
        current : int
            Current step
        total : int
            Total steps
        label : str
            Label for progress bar
        """
        bar_length = 30
        progress = current / total * 100
        filled_length = int(progress * bar_length / 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r[{label}: {bar}] {progress:.2f}%')
        sys.stdout.flush()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_orbital_elements_from_state(
    position: np.ndarray,
    velocity: np.ndarray,
    mu: float = EARTH_MU
) -> OrbitalElements:
    """
    Compute orbital elements from position and velocity (state vector).
    
    This is the inverse of orbit propagation - given r and v, find the
    Keplerian elements.
    
    Parameters
    ----------
    position : ndarray of shape (3,)
        Position vector in ECI frame [m]
    velocity : ndarray of shape (3,)
        Velocity vector in ECI frame [m/s]
    mu : float, optional
        Gravitational parameter [m³/s²]
        
    Returns
    -------
    OrbitalElements
        Computed orbital elements
        
    Notes
    -----
    This uses standard orbital mechanics formulas from Curtis chapter 4.
    
    References
    ----------
    .. [1] Curtis, H. D. "Orbital Mechanics for Engineering Students", Ch. 4
    
    Examples
    --------
    >>> pos = np.array([7000e3, 0, 0])
    >>> vel = np.array([0, 7500, 0])
    >>> elements = compute_orbital_elements_from_state(pos, vel)
    >>> print(elements.semi_major_axis / 1e6)
    ~6.9 Mm
    """
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)
    
    # Specific angular momentum
    h_vec = np.cross(position, velocity)
    h = np.linalg.norm(h_vec)
    
    # Eccentricity vector
    e_vec = (1/mu) * (
        (v**2 - mu/r) * position - np.dot(position, velocity) * velocity
    )
    e = np.linalg.norm(e_vec)
    
    # Semi-major axis (vis-viva equation)
    a = 1 / (2/r - v**2/mu)
    
    # Inclination
    inc = np.degrees(np.arccos(h_vec[2] / h))
    
    # Node vector
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)
    
    # RAAN
    if n > 1e-10:
        raan = np.degrees(np.arccos(n_vec[0] / n))
        if n_vec[1] < 0:
            raan = 360 - raan
    else:
        raan = 0.0
    
    # Argument of perigee
    if n > 1e-10 and e > 1e-10:
        arg_perigee = np.degrees(np.arccos(np.dot(n_vec, e_vec) / (n * e)))
        if e_vec[2] < 0:
            arg_perigee = 360 - arg_perigee
    else:
        arg_perigee = 0.0
    
    # True anomaly
    if e > 1e-10:
        true_anomaly = np.degrees(np.arccos(np.dot(e_vec, position) / (e * r)))
        if np.dot(position, velocity) < 0:
            true_anomaly = 360 - true_anomaly
    else:
        true_anomaly = 0.0
    
    return OrbitalElements(
        semi_major_axis=a,
        eccentricity=e,
        inclination=inc,
        raan=raan,
        arg_perigee=arg_perigee,
        true_anomaly=true_anomaly
    )
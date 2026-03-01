"""
Simulation scenario configuration.

This module defines the input configuration for a simulation run.
It's like a "recipe card" that specifies all parameters needed.

Classes
-------
SimulationScenario
    Complete configuration for one simulation run
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.domain import SatelliteSpecification

from missile_fly_by_simulation.constants import (
    EARTH_RADIUS_M,
    DEFAULT_FPS,
    DEFAULT_DURATION_S,
    DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS,
    DEFAULT_APOGEE_ALTITUDE_M,
    DEFAULT_PERIGEE_ALTITUDE_M,
    DEFAULT_INCLINATION_DEG,
    DEFAULT_RAAN_DEG,
    DEFAULT_ARG_PERIGEE_DEG,
    DEFAULT_CAMERA_RESOLUTION,
    DEFAULT_CAMERA_FOV_DEG,
    DEFAULT_MISSILE_POSITION_M,
    DEFAULT_MISSILE_VELOCITY,
    EARTH_ESCAPE_VELOCITY_MS,
    PIXEL_NOISE_SIGMA_PX,
)

@dataclass
class SimulationScenario:
    """
    Complete input configuration for a simulation run.
    
    This dataclass contains all parameters needed to execute a simulation.
    Think of it as a "recipe card" - it specifies WHAT to simulate,
    not HOW to simulate it (that's the Simulator's job).
    
    Attributes
    ----------
    satellite_spec : SatelliteSpecification
        Complete satellite design (camera, orbital elements)
    missile_initial_position : ndarray of shape (3,)
        Missile starting position in ECI frame [m]
    missile_velocity : ndarray of shape (3,)
        Missile velocity (constant for ballistic trajectory) [m/s]
    start_time : datetime
        Simulation start time (epoch)
    duration : float
        Total simulation duration [seconds]
    missile_name : str, optional
        Missile identifier, default "Target"
    detection_frame_index : int, optional
        Frame when missile first appears, default 0
    depth_time_offsets : list of float, optional
        Time differences for depth estimation [seconds],
        default [5, 10, 20, 40, 60]
    pixel_noise_sigma : float, optional
        Standard deviation of Gaussian centroid-localisation noise [pixels].
        Models sub-pixel uncertainty from blob centroid estimation in the IR image.
        Default: 0.3 px. Set to 0.0 for noise-free pixel measurements.
    fps : int, optional
        Frames per second (alternative to timestep), default None
    timestep : float, optional
        Time between frames [seconds] (alternative to fps), default None
        
    Notes
    -----
    You must specify EITHER fps OR timestep, not both.
    - If fps is given: timestep = 1 / fps
    - If timestep is given: fps = 1 / timestep
    - If neither: uses satellite camera's fps
    
    The scenario is immutable once created (frozen=False allows
    post_init modifications, but discouraged after creation).
    
    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> 
    >>> # Create satellite spec
    >>> sat_spec = SatelliteSpecification(
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
    >>> 
    >>> # Create scenario
    >>> scenario = SimulationScenario(
    ...     satellite_spec=sat_spec,
    ...     missile_initial_position=np.array([7.0e6, 0, 0]),
    ...     missile_velocity=np.array([100, 50, 0]),
    ...     start_time=datetime(2026, 2, 9, 14, 0, 0),
    ...     duration=1200.0,
    ...     detection_frame_index=15000
    ... )
    >>> 
    >>> print(f"Simulation will have {scenario.num_frames} frames")
    >>> print(f"Timestep: {scenario.timestep} seconds")
    """
    
    # Required fields
    satellite_spec: SatelliteSpecification
    missile_initial_position: npt.NDArray[np.float64]
    missile_velocity: npt.NDArray[np.float64]
    start_time: datetime
    duration: float
    
    # Optional fields
    missile_name: str = "Target"
    detection_frame_index: int = 0
    depth_time_offsets: List[float] = field(default_factory=lambda: list(DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS))
    pixel_noise_sigma: float = PIXEL_NOISE_SIGMA_PX

    # Frame rate (specify either fps or timestep, not both)
    fps: Optional[int] = None
    timestep: Optional[float] = None
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert arrays to numpy
        self.missile_initial_position = np.asarray(
            self.missile_initial_position, dtype=np.float64
        )
        self.missile_velocity = np.asarray(
            self.missile_velocity, dtype=np.float64
        )
        
        # Validate array shapes
        if self.missile_initial_position.shape != (3,):
            raise ValueError(
                f"missile_initial_position must be 3D, "
                f"got shape {self.missile_initial_position.shape}"
            )
        if self.missile_velocity.shape != (3,):
            raise ValueError(
                f"missile_velocity must be 3D, "
                f"got shape {self.missile_velocity.shape}"
            )
        
        # Determine timestep
        if self.fps is not None and self.timestep is not None:
            raise ValueError("Specify either fps OR timestep, not both")
        
        if self.fps is not None:
            # Use provided fps
            self.timestep = 1.0 / self.fps
        elif self.timestep is not None:
            # Use provided timestep
            self.fps = int(1.0 / self.timestep)
        else:
            # Use satellite camera's fps
            self.fps = self.satellite_spec.camera.fps
            self.timestep = 1.0 / self.fps
        
        # Validate other parameters
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive: {self.duration}")
        
        if self.timestep <= 0:
            raise ValueError(f"Timestep must be positive: {self.timestep}")
        
        if self.detection_frame_index < 0:
            raise ValueError(
                f"detection_frame_index must be non-negative: {self.detection_frame_index}"
            )
    
    # =========================================================================
    # DERIVED PROPERTIES
    # =========================================================================
    
    @property
    def num_frames(self) -> int:
        """
        Total number of frames in simulation.
        
        Returns
        -------
        int
            Number of frames (duration / timestep)
        """
        return int(self.duration / self.timestep)
    
    @property
    def end_time(self) -> datetime:
        """
        Simulation end time.
        
        Returns
        -------
        datetime
            Start time + duration
        """
        return self.start_time + timedelta(seconds=self.duration)
    
    @property
    def timestamps(self) -> List[datetime]:
        """
        All simulation timestamps.
        
        Returns
        -------
        list of datetime
            Timestamps for each frame
            
        Notes
        -----
        This can be memory-intensive for long simulations.
        Consider using timestamp_at_index() for large simulations.
        """
        return [
            self.start_time + timedelta(seconds=i * self.timestep)
            for i in range(self.num_frames)
        ]
    
    @property
    def missile_detection_time(self) -> datetime:
        """
        Time when missile first appears.
        
        Returns
        -------
        datetime
            Timestamp of detection_frame_index
        """
        return self.start_time + timedelta(
            seconds=self.detection_frame_index * self.timestep
        )
    
    @property
    def missile_duration(self) -> float:
        """
        Duration of missile trajectory [seconds].
        
        Returns
        -------
        float
            Duration from detection to end of simulation
        """
        return self.duration - (self.detection_frame_index * self.timestep)
    
    @property
    def num_missile_frames(self) -> int:
        """
        Number of frames with missile present.
        
        Returns
        -------
        int
            Total frames minus detection frame index
        """
        return self.num_frames - self.detection_frame_index
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def timestamp_at_index(self, index: int) -> datetime:
        """
        Get timestamp for a specific frame index.
        
        Parameters
        ----------
        index : int
            Frame index
            
        Returns
        -------
        datetime
            Timestamp at that frame
            
        Raises
        ------
        IndexError
            If index is out of range
        """
        if not 0 <= index < self.num_frames:
            raise IndexError(
                f"Index {index} out of range [0, {self.num_frames})"
            )
        
        return self.start_time + timedelta(seconds=index * self.timestep)
    
    def index_at_timestamp(self, timestamp: datetime) -> int:
        """
        Get frame index closest to a timestamp.
        
        Parameters
        ----------
        timestamp : datetime
            Query time
            
        Returns
        -------
        int
            Nearest frame index
        """
        dt = (timestamp - self.start_time).total_seconds()
        index = int(round(dt / self.timestep))
        
        # Clamp to valid range
        return max(0, min(index, self.num_frames - 1))
    
    def validate(self) -> List[str]:
        """
        Validate scenario configuration.
        
        Returns
        -------
        list of str
            List of validation errors (empty if valid)
            
        Examples
        --------
        >>> errors = scenario.validate()
        >>> if errors:
        ...     print("Configuration errors:")
        ...     for error in errors:
        ...         print(f"  - {error}")
        ... else:
        ...     print("Configuration is valid!")
        """
        errors = []
        
        # Check detection happens before end
        if self.detection_frame_index >= self.num_frames:
            errors.append(
                f"Detection frame {self.detection_frame_index} is beyond "
                f"simulation end ({self.num_frames} frames)"
            )
        
        # Check missile has reasonable duration
        if self.missile_duration < 10.0:
            errors.append(
                f"Missile duration ({self.missile_duration:.1f}s) is very short. "
                f"Consider longer simulation or earlier detection."
            )
        
        # Check depth time offsets are reasonable
        for offset in self.depth_time_offsets:
            if offset <= 0:
                errors.append(f"Depth time offset must be positive: {offset}")
            
            if offset > self.missile_duration:
                errors.append(
                    f"Depth time offset {offset}s exceeds missile duration "
                    f"{self.missile_duration:.1f}s"
                )
        
        # Check missile position is reasonable (not inside Earth)
        earth_radius = EARTH_RADIUS_M
        missile_altitude = np.linalg.norm(self.missile_initial_position) - earth_radius
        
        if missile_altitude < 0:
            errors.append(
                f"Missile initial position is inside Earth "
                f"(altitude={missile_altitude/1e3:.1f} km)"
            )
        
        # Check missile speed is reasonable
        missile_speed = np.linalg.norm(self.missile_velocity)
        escape_velocity = EARTH_ESCAPE_VELOCITY_MS  # m/s
        
        if missile_speed > escape_velocity:
            errors.append(
                f"Missile speed ({missile_speed/1e3:.1f} km/s) exceeds "
                f"escape velocity ({escape_velocity/1e3:.1f} km/s)"
            )
        
        if missile_speed < 10:
            errors.append(
                f"Missile speed ({missile_speed:.1f} m/s) is suspiciously low"
            )
        
        return errors
    
    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SimulationScenario("
            f"satellite='{self.satellite_spec.name}', "
            f"duration={self.duration}s, "
            f"fps={self.fps})"
        )
    
    def __str__(self) -> str:
        """Human-readable description."""
        lines = [
            "Simulation Scenario Configuration",
            "=" * 35,
            "",
            f"Satellite: {self.satellite_spec.name}",
            f"  Camera: {self.satellite_spec.camera.resolution[0]}×"
            f"{self.satellite_spec.camera.resolution[1]} @ {self.fps} fps",
            f"  Orbit: {self.satellite_spec.orbital_elements.apogee_altitude/1e3:.1f} km "
            f"(apogee) × {self.satellite_spec.orbital_elements.perigee_altitude/1e3:.1f} km "
            f"(perigee)",
            "",
            f"Missile: {self.missile_name}",
            f"  Initial position: [{self.missile_initial_position[0]/1e6:.3f}, "
            f"{self.missile_initial_position[1]/1e6:.3f}, "
            f"{self.missile_initial_position[2]/1e6:.3f}] Mm",
            f"  Velocity: [{self.missile_velocity[0]:.1f}, "
            f"{self.missile_velocity[1]:.1f}, {self.missile_velocity[2]:.1f}] m/s",
            f"  Speed: {np.linalg.norm(self.missile_velocity):.1f} m/s",
            "",
            f"Simulation Timeline:",
            f"  Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Duration: {self.duration:.1f} seconds ({self.duration/60:.1f} minutes)",
            f"  Frames: {self.num_frames}",
            f"  Timestep: {self.timestep:.4f} seconds",
            f"  Detection at frame: {self.detection_frame_index} "
            f"(t={self.detection_frame_index * self.timestep:.1f}s)",
            "",
            f"Depth Estimation:",
            f"  Time offsets: {self.depth_time_offsets} seconds",
        ]
        
        # Add validation warnings if any
        errors = self.validate()
        if errors:
            lines.extend([
                "",
                "[!] Validation Warnings:",
            ])
            for error in errors:
                lines.append(f"  - {error}")
        
        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTIONS (Convenience constructors)
# =============================================================================

def create_default_scenario(
    satellite_name: str = "ERNST",
    simulation_duration: float = DEFAULT_DURATION_S
) -> SimulationScenario:
    """
    Create a default scenario for quick testing.
    
    Parameters
    ----------
    satellite_name : str, optional
        Satellite name, default "ERNST"
    simulation_duration : float, optional
        Duration [seconds], default 1200 (20 minutes)
        
    Returns
    -------
    SimulationScenario
        Default scenario configuration
        
    Examples
    --------
    >>> scenario = create_default_scenario()
    >>> print(scenario)
    """
    from missile_fly_by_simulation.domain import (
        SatelliteSpecification,
        CameraSpecification,
        OrbitalElements
    )
    
    # Default camera
    camera = CameraSpecification(
        resolution=DEFAULT_CAMERA_RESOLUTION,
        fov_horizontal_deg=DEFAULT_CAMERA_FOV_DEG,
        fps=DEFAULT_FPS
    )
    
    # Default orbit (sun-synchronous LEO)
    orbit = OrbitalElements.from_apogee_perigee(
        apogee_altitude=DEFAULT_APOGEE_ALTITUDE_M,
        perigee_altitude=DEFAULT_PERIGEE_ALTITUDE_M,
        inclination=DEFAULT_INCLINATION_DEG,
        raan=DEFAULT_RAAN_DEG,
        arg_perigee=DEFAULT_ARG_PERIGEE_DEG
    )
    
    # Default satellite
    sat_spec = SatelliteSpecification(
        name=satellite_name,
        camera=camera,
        orbital_elements=orbit
    )
    
    # Create scenario
    scenario = SimulationScenario(
        satellite_spec=sat_spec,
        missile_initial_position=DEFAULT_MISSILE_POSITION_M,
        missile_velocity=DEFAULT_MISSILE_VELOCITY,
        start_time=datetime(2026, 2, 9, 14, 0, 0),
        duration=simulation_duration,
        detection_frame_index=0,  # 500 seconds in
        depth_time_offsets=DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS
    )
    
    return scenario
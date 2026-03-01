"""
Scenario factory for creating simulation scenarios from physical parameters.

This module converts intuitive physical parameters (closest approach distance,
missile speed, crossing angle) into the correct SimulationScenario objects
with properly computed position and velocity vectors.

Classes
-------
ScenarioFactory
    Creates SimulationScenario objects from physical parameters
"""

import math as _math
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.domain import (
    SatelliteSpecification,
    CameraSpecification,
    OrbitalElements,
)
from missile_fly_by_simulation.simulation.scenario import SimulationScenario
from missile_fly_by_simulation.constants import (
    DEFAULT_DURATION_S,
    DEFAULT_DETECTION_FRACTION,
    DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS,
    DEFAULT_CROSSING_ANGLE_DEG,
    DEFAULT_APOGEE_ALTITUDE_M,
    DEFAULT_PERIGEE_ALTITUDE_M,
    DEFAULT_INCLINATION_DEG,
    DEFAULT_CAMERA_RESOLUTION,
    DEFAULT_CAMERA_FOV_DEG,
    DEFAULT_FPS,
)


def fibonacci_hemisphere_points(n: int) -> List[Tuple[float, float]]:
    """
    Generate N quasi-uniformly distributed (azimuth, elevation) pairs
    covering the upper hemisphere with approximately equal solid angle per point.

    Uses the Fibonacci / golden-angle method:
    - sin(elevation) is uniform in (0, 1] → equal solid-angle area per point
    - azimuth grows by the golden angle → avoids radial clustering

    Parameters
    ----------
    n : int
        Number of points to generate.

    Returns
    -------
    list of (azimuth_deg, elevation_deg) tuples
        azimuth in [0, 360), elevation in (0, 90]
    """
    phi = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio
    points = []
    for i in range(n):
        sin_el = (i + 0.5) / n                          # uniform in (0, 1]
        el_deg = _math.degrees(_math.asin(sin_el))      # elevation in (0°, 90°]
        az_deg = (360.0 * i / phi) % 360.0              # azimuth by golden angle
        points.append((az_deg, el_deg))
    return points


class ScenarioFactory:
    """
    Creates SimulationScenario objects from physical parameters.

    Instead of manually computing missile position and velocity vectors,
    this factory takes intuitive physical parameters and computes the
    correct vectors automatically.

    The geometry is defined as:
    - Satellite moves along its orbit
    - Missile trajectory is defined relative to the satellite
    - Closest approach distance: minimum distance between satellite and missile
    - Crossing angle: angle between satellite velocity and missile velocity

    For the parameter study (distance × speed grid), the crossing angle
    is fixed at 90° (perpendicular flyby) which is the most common
    and interesting case.

    Attributes
    ----------
    satellite_spec : SatelliteSpecification
        Satellite design used for all scenarios
    start_time : datetime
        Simulation start time
    simulation_duration : float
        Total simulation duration [seconds]
    detection_fraction : float
        When missile appears as fraction of simulation
        0.3 means missile appears at 30% through simulation

    Examples
    --------
    >>> factory = ScenarioFactory(
    ...     satellite_spec=sat_spec,
    ...     simulation_duration=1200.0
    ... )
    >>>
    >>> # Create one scenario
    >>> scenario = factory.create_flyby_scenario(
    ...     closest_approach_distance=200e3,  # 200 km
    ...     missile_speed=1000.0              # 1000 m/s
    ... )
    >>>
    >>> # Create grid of scenarios for parameter study
    >>> scenarios = factory.create_parameter_grid(
    ...     distances=[100e3, 200e3, 500e3],
    ...     speeds=[500.0, 1000.0, 3000.0]
    ... )
    """

    def __init__(
        self,
        satellite_spec: SatelliteSpecification,
        start_time: Optional[datetime] = None,
        simulation_duration: float = DEFAULT_DURATION_S,
        detection_fraction: float = DEFAULT_DETECTION_FRACTION,
        depth_time_offsets: Optional[list] = None,
        crossing_angle_deg: float = DEFAULT_CROSSING_ANGLE_DEG,
    ):
        """
        Initialize scenario factory.

        Parameters
        ----------
        satellite_spec : SatelliteSpecification
            Satellite design to use for all scenarios
        start_time : datetime, optional
            Simulation start time, default 2026-02-09 14:00:00
        simulation_duration : float, optional
            Total simulation duration [seconds], default 1200
        detection_fraction : float, optional
            When missile appears as fraction of simulation [0-1], default 0.3
            e.g. 0.3 means missile detected at 30% through simulation
        depth_time_offsets : list of float, optional
            Time offsets for depth estimation [seconds]
            default [5, 10, 20, 40, 60]
        crossing_angle_deg : float, optional
            Angle between satellite velocity and missile velocity [degrees]
            default 90.0 (perpendicular flyby - fixed for parameter study)
        """
        self.satellite_spec = satellite_spec
        self.start_time = start_time or datetime.now(timezone.utc)
        self.simulation_duration = simulation_duration
        self.detection_fraction = detection_fraction
        self.depth_time_offsets = depth_time_offsets if depth_time_offsets is not None else DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS
        self.crossing_angle_deg = crossing_angle_deg

    # =========================================================================
    # MAIN FACTORY METHODS
    # =========================================================================

    def create_flyby_scenario(
        self,
        closest_approach_distance: float,
        missile_speed: float,
        crossing_angle_deg: Optional[float] = None,
        elevation_angle_deg: float = 0.0,
    ) -> SimulationScenario:
        """
        Create a scenario from physical flyby parameters.

        Parameters
        ----------
        closest_approach_distance : float
            Minimum distance between satellite and missile [m]
            e.g. 100e3 for 100 km
        missile_speed : float
            Missile speed [m/s]
            e.g. 1000.0 for 1 km/s
        crossing_angle_deg : float, optional
            Override crossing angle for this scenario [degrees]
            If None, uses factory default (90° perpendicular)

        Returns
        -------
        SimulationScenario
            Complete scenario with computed position/velocity vectors

        Notes
        -----
        Geometry setup:

        The satellite is at some position along its orbit at detection time.
        We place the missile such that it will reach closest approach
        at the midpoint of the observation window.

        Coordinate system (at detection moment):
            satellite_pos  = position from orbital mechanics
            satellite_vel  = velocity from orbital mechanics (normalized)

            radial_dir     = satellite_pos / |satellite_pos|  (away from Earth)
            along_track    = satellite_vel / |satellite_vel|  (direction of motion)
            cross_track    = cross(along_track, radial_dir)   (perpendicular)

        Missile placement:
            - Missile starts at closest_approach_distance
              in the cross_track direction
            - Missile velocity is along crossing_angle relative
              to satellite velocity
        """
        angle = crossing_angle_deg if crossing_angle_deg is not None else self.crossing_angle_deg

        # Compute satellite state at detection time
        sat_pos, sat_vel = self._get_satellite_state_at_detection()

        # Build local coordinate frame at satellite position
        radial_dir, along_track_dir, cross_track_dir = self._compute_local_frame(
            sat_pos, sat_vel
        )

        # Compute missile initial position
        # Place missile at closest_approach_distance in cross-track direction
        # Then offset backward so closest approach happens mid-observation
        missile_initial_position = self._compute_missile_initial_position(
            sat_pos=sat_pos,
            sat_vel=sat_vel,
            closest_approach_distance=closest_approach_distance,
            missile_speed=missile_speed,
            crossing_angle_deg=angle,
            along_track_dir=along_track_dir,
            cross_track_dir=cross_track_dir,
            elevation_angle_deg=elevation_angle_deg,
            radial_dir=radial_dir,
        )

        # Compute missile velocity vector
        missile_velocity = self._compute_missile_velocity(
            missile_speed=missile_speed,
            crossing_angle_deg=angle,
            along_track_dir=along_track_dir,
            cross_track_dir=cross_track_dir,
            elevation_angle_deg=elevation_angle_deg,
            radial_dir=radial_dir,
        )

        # Detection frame index
        detection_frame_index = int(
            self.detection_fraction * self.simulation_duration * self.satellite_spec.camera.fps
        )

        # Create scenario
        scenario = SimulationScenario(
            satellite_spec=self.satellite_spec,
            missile_initial_position=missile_initial_position,
            missile_velocity=missile_velocity,
            start_time=self.start_time,
            duration=self.simulation_duration,
            detection_frame_index=detection_frame_index,
            depth_time_offsets=self.depth_time_offsets,
        )

        # Store physical parameters as metadata for later reference
        scenario._closest_approach_distance = closest_approach_distance
        scenario._missile_speed = missile_speed
        scenario._crossing_angle_deg = angle
        scenario._elevation_angle_deg = elevation_angle_deg

        return scenario

    def create_radial_launch_scenario(
        self,
        radial_speed: float = None,
        launch_lead_time_s: float = None,
        pre_launch_observe_s: float = 10.0,
        post_launch_observe_s: float = 500.0,
        pre_detection_buffer_s: float = 30.0,
        crossing_angle_deg: float = 0.0,
        elevation_angle_deg: float = 90.0,
    ) -> SimulationScenario:
        """
        Create a scenario where a rocket launches from Earth's surface.

        Default behaviour (new params at default): purely radial (straight up)
        launch with 30 s satellite warmup — identical to original implementation.

        Parameters
        ----------
        radial_speed : float, optional
            Rocket speed [m/s], default DEFAULT_RADIAL_SPEED_MS (200.0)
        launch_lead_time_s : float, optional
            Seconds before satellite overhead that rocket launches,
            default DEFAULT_LAUNCH_LEAD_TIME_S (20.0)
        pre_launch_observe_s : float, optional
            Seconds before launch to start observing. Default 10.0.
        post_launch_observe_s : float, optional
            Seconds after launch to observe the rocket. Default 500.0.
        pre_detection_buffer_s : float, optional
            Seconds of satellite-only warmup before detection begins. Default 30.0.
            Set to 0.0 for angular study (no warmup, sim starts right before launch).
        crossing_angle_deg : float, optional
            Azimuth in along-track/cross-track plane [deg]. Default 0.0.
            Irrelevant when elevation_angle_deg=90.0 (straight up).
        elevation_angle_deg : float, optional
            Elevation above horizontal [deg]. Default 90.0 (straight up = radial).
            0.0 = horizontal flight.

        Returns
        -------
        SimulationScenario
        """
        from missile_fly_by_simulation.constants import (
            DEFAULT_RADIAL_SPEED_MS, DEFAULT_LAUNCH_LEAD_TIME_S, EARTH_RADIUS_M
        )
        from missile_fly_by_simulation.physics import KeplerianOrbitPropagator
        from datetime import timedelta

        speed  = radial_speed       if radial_speed       is not None else DEFAULT_RADIAL_SPEED_MS
        lead_t = launch_lead_time_s if launch_lead_time_s is not None else DEFAULT_LAUNCH_LEAD_TIME_S

        # --- Simulation timeline ---
        # t=0:                                                     sim start
        # t=pre_detection_buffer_s:                                detection begins
        # t=pre_detection_buffer_s + pre_launch_observe_s:        T_launch
        # t=pre_detection_buffer_s + pre_launch_observe_s + lead_t: T_overhead
        # t=pre_detection_buffer_s + pre_launch_observe_s + post_launch_observe_s: sim end

        detection_time_s = pre_detection_buffer_s
        launch_time_s    = detection_time_s + pre_launch_observe_s
        overhead_time_s  = launch_time_s + lead_t
        sim_duration     = detection_time_s + pre_launch_observe_s + post_launch_observe_s

        overhead_time = self.start_time + timedelta(seconds=overhead_time_s)

        # Satellite position at overhead_time (true curved orbit)
        propagator = KeplerianOrbitPropagator(self.satellite_spec.orbital_elements)
        states = propagator.propagate(
            timestamps=[overhead_time],
            reference_time=self.start_time,
            show_progress=False,
        )
        sat_pos_at_overhead = states[0].position
        sat_vel_at_overhead = states[0].velocity

        # Local coordinate frame
        radial_dir      = sat_pos_at_overhead / np.linalg.norm(sat_pos_at_overhead)
        along_track_dir = sat_vel_at_overhead / np.linalg.norm(sat_vel_at_overhead)
        cross_track_dir = np.cross(along_track_dir, radial_dir)
        cross_track_dir = cross_track_dir / np.linalg.norm(cross_track_dir)

        # Launch site: nadir of satellite, 10 m above Earth surface
        launch_point = (EARTH_RADIUS_M + 10.0) * radial_dir

        # Rocket velocity — general (az, el); el=90° (default) gives pure radial
        rocket_velocity = self._compute_missile_velocity(
            missile_speed=speed,
            crossing_angle_deg=crossing_angle_deg,
            along_track_dir=along_track_dir,
            cross_track_dir=cross_track_dir,
            elevation_angle_deg=elevation_angle_deg,
            radial_dir=radial_dir,
        )

        fps = self.satellite_spec.camera.fps
        detection_frame_index = int(detection_time_s * fps)

        scenario = SimulationScenario(
            satellite_spec=self.satellite_spec,
            missile_initial_position=launch_point,
            missile_velocity=rocket_velocity,
            start_time=self.start_time,
            duration=sim_duration,
            detection_frame_index=detection_frame_index,
            depth_time_offsets=self.depth_time_offsets,
        )

        # Metadata
        scenario._closest_approach_distance = float(np.linalg.norm(sat_pos_at_overhead - launch_point))
        scenario._missile_speed             = speed
        scenario._crossing_angle_deg        = crossing_angle_deg
        scenario._elevation_angle_deg       = elevation_angle_deg
        scenario._launch_lead_time_s        = lead_t

        return scenario

    def create_parameter_grid(
        self,
        distances: list,
        speeds: list,
    ) -> dict:
        """
        Create all scenarios for a distance × speed parameter grid.

        Parameters
        ----------
        distances : list of float
            Closest approach distances [m]
            e.g. [50e3, 100e3, 200e3, 500e3, 1000e3]
        speeds : list of float
            Missile speeds [m/s]
            e.g. [300, 500, 1000, 3000, 7000]

        Returns
        -------
        dict
            Keys: (distance, speed) tuples
            Values: SimulationScenario objects

        Examples
        --------
        >>> scenarios = factory.create_parameter_grid(
        ...     distances=[100e3, 500e3, 1000e3],
        ...     speeds=[500, 1000, 3000]
        ... )
        >>> # Access specific scenario
        >>> scenario = scenarios[(100e3, 1000)]
        >>> print(f"Generated {len(scenarios)} scenarios")
        Generated 9 scenarios
        """
        scenarios = {}

        total = len(distances) * len(speeds)
        count = 0

        print(f"\nCreating {total} scenarios for parameter grid...")

        for distance in distances:
            for speed in speeds:
                count += 1
                scenario = self.create_flyby_scenario(
                    closest_approach_distance=distance,
                    missile_speed=speed,
                )
                scenarios[(distance, speed)] = scenario

                print(
                    f"  [{count:3d}/{total}] "
                    f"distance={distance/1e3:.0f}km, "
                    f"speed={speed:.0f}m/s [OK]"
                )

        print(f"Done! Created {total} scenarios.\n")

        return scenarios

    # =========================================================================
    # GEOMETRY HELPERS
    # =========================================================================

    def _get_satellite_state_at_detection(self):
        """
        Compute satellite position and velocity at detection time.

        Uses orbital mechanics to propagate to detection time.

        Returns
        -------
        sat_pos : ndarray of shape (3,)
            Satellite ECI position at detection [m]
        sat_vel : ndarray of shape (3,)
            Satellite ECI velocity at detection [m/s]
        """
        from missile_fly_by_simulation.physics import KeplerianOrbitPropagator
        from datetime import timedelta

        # Detection time
        detection_time_seconds = self.detection_fraction * self.simulation_duration
        detection_time = self.start_time + timedelta(seconds=detection_time_seconds)

        # Propagate orbit to detection time
        propagator = KeplerianOrbitPropagator(
            self.satellite_spec.orbital_elements
        )
        states = propagator.propagate(
            timestamps=[detection_time],
            reference_time=self.start_time,
            show_progress=False
        )

        sat_pos = states[0].position
        sat_vel = states[0].velocity

        return sat_pos, sat_vel

    def _compute_local_frame(
        self,
        sat_pos: npt.NDArray[np.float64],
        sat_vel: npt.NDArray[np.float64],
    ):
        """
        Compute local coordinate frame at satellite position.

        Returns three orthonormal vectors:
        - radial:      pointing away from Earth center
        - along_track: pointing in direction of satellite motion
        - cross_track: perpendicular to both (completes right-hand system)

        Parameters
        ----------
        sat_pos : ndarray of shape (3,)
            Satellite position [m]
        sat_vel : ndarray of shape (3,)
            Satellite velocity [m/s]

        Returns
        -------
        radial_dir : ndarray of shape (3,)
        along_track_dir : ndarray of shape (3,)
        cross_track_dir : ndarray of shape (3,)
        """
        # Radial direction (away from Earth)
        radial_dir = sat_pos / np.linalg.norm(sat_pos)

        # Along-track direction (direction of motion)
        along_track_dir = sat_vel / np.linalg.norm(sat_vel)

        # Cross-track direction (perpendicular to orbit plane)
        cross_track_dir = np.cross(along_track_dir, radial_dir)
        cross_track_dir = cross_track_dir / np.linalg.norm(cross_track_dir)

        # Re-orthogonalize along-track (ensure perfect orthogonality)
        along_track_dir = np.cross(radial_dir, cross_track_dir)
        along_track_dir = along_track_dir / np.linalg.norm(along_track_dir)

        return radial_dir, along_track_dir, cross_track_dir

    def _compute_missile_velocity(
        self,
        missile_speed: float,
        crossing_angle_deg: float,
        along_track_dir: npt.NDArray[np.float64],
        cross_track_dir: npt.NDArray[np.float64],
        elevation_angle_deg: float = 0.0,
        radial_dir: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute missile velocity vector.

        Two-angle parameterisation of the missile's direction:

        crossing_angle (azimuth): angle in the along-track / cross-track plane
            0°  → missile flies parallel to satellite (along-track)
            90° → missile flies perpendicular (cross-track)

        elevation_angle: tilt out of that horizontal plane toward radial
            0°  → purely horizontal (default, backwards-compatible)
            90° → straight up (radial — equivalent to radial-launch)

        Parameters
        ----------
        missile_speed : float
            Missile speed [m/s]
        crossing_angle_deg : float
            Azimuth angle in along-track / cross-track plane [degrees]
        along_track_dir : ndarray
            Satellite along-track direction
        cross_track_dir : ndarray
            Satellite cross-track direction
        elevation_angle_deg : float, optional
            Elevation above horizontal plane [degrees], default 0.0
        radial_dir : ndarray, optional
            Satellite radial (up) direction — required when elevation_angle_deg != 0

        Returns
        -------
        missile_velocity : ndarray of shape (3,)
            Missile velocity vector in ECI frame [m/s]
        """
        az = np.radians(crossing_angle_deg)
        el = np.radians(elevation_angle_deg)

        # Horizontal component (in along-track / cross-track plane)
        horizontal = (np.cos(az) * along_track_dir +
                      np.sin(az) * cross_track_dir)
        horizontal = horizontal / np.linalg.norm(horizontal)

        if el == 0.0 or radial_dir is None:
            missile_direction = horizontal
        else:
            missile_direction = np.cos(el) * horizontal + np.sin(el) * radial_dir
            missile_direction = missile_direction / np.linalg.norm(missile_direction)

        return missile_speed * missile_direction

    def _compute_missile_initial_position(
        self,
        sat_pos: npt.NDArray[np.float64],
        sat_vel: npt.NDArray[np.float64],
        closest_approach_distance: float,
        missile_speed: float,
        crossing_angle_deg: float,
        along_track_dir: npt.NDArray[np.float64],
        cross_track_dir: npt.NDArray[np.float64],
        elevation_angle_deg: float = 0.0,
        radial_dir: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute missile initial position for desired closest approach.

        Strategy:
        1. Closest approach happens at detection time + half_observation_window
        2. At closest approach: missile is at closest_approach_distance
        from satellite (in cross-track direction for 90° crossing)
        3. Work backward from closest approach to get initial position

        CRITICAL: Must account for satellite ORBITAL CURVATURE, not straight-line motion.
        
        Over a typical 840s observation window, the satellite travels ~3150 km along
        its curved orbit, sweeping through ~28° of arc. Using sat_pos + sat_vel × time
        (straight-line approximation) introduces ~500 km error. This method uses the
        orbital propagator to compute the satellite's true curved position.

        Parameters
        ----------
        sat_pos : ndarray
            Satellite position at detection [m]
        sat_vel : ndarray
            Satellite velocity at detection [m/s]
        closest_approach_distance : float
            Desired closest approach distance [m]
        missile_speed : float
            Missile speed [m/s]
        crossing_angle_deg : float
            Crossing angle [degrees]
        along_track_dir : ndarray
            Along-track unit vector
        cross_track_dir : ndarray
            Cross-track unit vector

        Returns
        -------
        missile_initial_position : ndarray of shape (3,)
            Missile ECI position at detection time [m]
        """
        from missile_fly_by_simulation.physics import KeplerianOrbitPropagator
        from datetime import timedelta
        
        # Time from detection to closest approach (midpoint of observation window)
        observation_duration = self.simulation_duration * (1.0 - self.detection_fraction)
        half_window = observation_duration / 2.0
        
        # Compute missile velocity vector
        missile_velocity = self._compute_missile_velocity(
            missile_speed=missile_speed,
            crossing_angle_deg=crossing_angle_deg,
            along_track_dir=along_track_dir,
            cross_track_dir=cross_track_dir,
            elevation_angle_deg=elevation_angle_deg,
            radial_dir=radial_dir,
        )
        
        # Step 1: WHERE WILL THE SATELLITE BE at closest approach?
        # Propagate orbit to account for curved trajectory (NOT sat_pos + sat_vel × time)
        detection_time_seconds = self.detection_fraction * self.simulation_duration
        detection_time = self.start_time + timedelta(seconds=detection_time_seconds)
        closest_approach_time = detection_time + timedelta(seconds=half_window)
        
        propagator = KeplerianOrbitPropagator(self.satellite_spec.orbital_elements)
        states = propagator.propagate(
            timestamps=[closest_approach_time],
            reference_time=self.start_time,
            show_progress=False
        )
        sat_pos_at_closest_approach = states[0].position
        
        # Step 2: WHERE SHOULD THE MISSILE BE at closest approach?
        # Place missile at closest_approach_distance in cross-track direction
        # from satellite's true position at that time
        closest_approach_offset = closest_approach_distance * cross_track_dir
        missile_pos_at_closest_approach = (
            sat_pos_at_closest_approach + closest_approach_offset
        )
        
        # Step 3: WORK BACKWARD to initial position
        # Missile moves in straight line: position(0) = position(t) - velocity × t
        missile_initial_position = (
            missile_pos_at_closest_approach - missile_velocity * half_window
        )
        
        return missile_initial_position

    # =========================================================================
    # UTILITY
    # =========================================================================

    @staticmethod
    def default_satellite_spec() -> SatelliteSpecification:
        """
        Create default ERNST satellite specification.

        Returns
        -------
        SatelliteSpecification
            Default satellite with 1024×720 camera at 30fps
        """
        camera = CameraSpecification(
            resolution=DEFAULT_CAMERA_RESOLUTION,
            fov_horizontal_deg=DEFAULT_CAMERA_FOV_DEG,
            fps=DEFAULT_FPS
        )

        orbit = OrbitalElements.from_apogee_perigee(
            apogee_altitude=DEFAULT_APOGEE_ALTITUDE_M,
            perigee_altitude=DEFAULT_PERIGEE_ALTITUDE_M,
            inclination=DEFAULT_INCLINATION_DEG,
            raan=0.0,
            arg_perigee=0.0
        )

        return SatelliteSpecification(
            name="ERNST",
            camera=camera,
            orbital_elements=orbit
        )

    def scenario_label(
        self,
        closest_approach_distance: float,
        missile_speed: float
    ) -> str:
        """
        Generate human-readable label for a scenario.

        Used for folder names and plot titles.

        Parameters
        ----------
        closest_approach_distance : float
            Closest approach distance [m]
        missile_speed : float
            Missile speed [m/s]

        Returns
        -------
        str
            Label string

        Examples
        --------
        >>> label = factory.scenario_label(100e3, 1000)
        >>> print(label)
        'dist100km_speed1000ms'
        """
        dist_km = int(closest_approach_distance / 1e3)
        speed_ms = int(missile_speed)
        return f"dist{dist_km}km_speed{speed_ms}ms"

    def __repr__(self) -> str:
        return (
            f"ScenarioFactory("
            f"satellite='{self.satellite_spec.name}', "
            f"duration={self.simulation_duration}s, "
            f"crossing_angle={self.crossing_angle_deg}°)"
        )
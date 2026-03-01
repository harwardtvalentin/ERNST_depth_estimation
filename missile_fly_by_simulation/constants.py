"""
Physical and simulation constants.

Single source of truth for all constants used across the package.
Never hardcode these values elsewhere - always import from here!

Usage
-----
>>> from missile_fly_by_simulation.constants import EARTH_MU, EARTH_RADIUS_M
>>> from missile_fly_by_simulation.constants import MIN_TRIANGULATION_ANGLE_DEG
"""

import datetime
import numpy as np

# =============================================================================
# EARTH PHYSICAL CONSTANTS
# =============================================================================

EARTH_RADIUS_M           = 6.378136e6       # [m]       mean equatorial radius
EARTH_RADIUS_KM          = 6378.136         # [km]      mean equatorial radius
EARTH_MU                 = 3.986004418e14   # [m³/s²]   standard gravitational parameter
EARTH_J2                 = 1.08263e-3       # [-]        oblateness (J2 coefficient)
EARTH_ROTATION_RATE      = 7.2921150e-5     # [rad/s]   sidereal rotation rate
EARTH_ESCAPE_VELOCITY_MS = 11.2e3           # [m/s]     Earth escape velocity

# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================

DEFAULT_FPS                = 30             # [fps]     camera frame rate
DEFAULT_DURATION_S         = 100.0          # [s]       simulation duration
DEFAULT_DETECTION_FRACTION = 0.3            # [-]       when missile appears (30% through)

# Default satellite orbit (ERNST)
DEFAULT_APOGEE_ALTITUDE_M  = 519986.0       # [m]
DEFAULT_PERIGEE_ALTITUDE_M = 514905.0       # [m]
DEFAULT_INCLINATION_DEG    = 97.5           # [deg]     sun-synchronous
DEFAULT_RAAN_DEG           = 0.0            # [deg]     right ascension of ascending node
DEFAULT_ARG_PERIGEE_DEG    = 0.0            # [deg]     argument of perigee

# Default camera
DEFAULT_CAMERA_RESOLUTION    = (1024, 720)  # [px]
DEFAULT_CAMERA_FOV_DEG       = 30.0         # [deg]     horizontal FOV
DEFAULT_NADIR_LOOK_ANGLE_DEG = 30.0         # [deg]     nadir look angle when no target
ATTITUDE_NOISE_DEG           = 0.007        # [deg]     standard deviation of attitude noise (0 = perfect nadir pointing)
PIXEL_NOISE_SIGMA_PX         = 0.2          # [px]      std dev of centroid localisation noise (0 = perfect pixel)

# Default missile
now = datetime.datetime.now()
DEFAULT_START_TIME            = now.replace(microsecond=0)  # [datetime] simulation start time
DEFAULT_DETECTION_FRAME_INDEX = 15000                       # [-]        500s at 30fps
DEFAULT_MISSILE_ALTITUDE_M    = 6.978e6                     # [m]        ~600km altitude
DEFAULT_MISSILE_POSITION_M    = np.array([EARTH_RADIUS_M + DEFAULT_MISSILE_ALTITUDE_M, 0.0, 0.0])  # [m] ECI
DEFAULT_MISSILE_SPEED_MS      = 100.0                       # [m/s]
DEFAULT_MISSILE_VELOCITY      = np.array([0.0, DEFAULT_MISSILE_SPEED_MS, 0.0])  # [m/s] ECI
DEFAULT_CROSSING_ANGLE_DEG    = 90.0                        # [deg]      perpendicular flyby

# =============================================================================
# ESTIMATION CONSTANTS
# =============================================================================

# --- Shared ---
MIN_TRIANGULATION_ANGLE_DEG = 0.01          # [deg]   minimum ray angle → below: reject estimate
MAX_TRIANGULATION_GAP_M     = 10000.0       # [m]     maximum triangulation gap → above: reject estimate
PAIRING_TOLERANCE_S         = 0.5           # [s]     max time difference for observation pairing

# --- Two-Ray Triangulation ---
DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])  # [s]  lookback offsets Δt

# --- Multi-Ray Least Squares ---
DEFAULT_MULTI_RAY_OBSERVATIONS = 100        # [-]     number of evenly-spaced observations per estimate
DEFAULT_MULTI_RAY_TIME_WINDOWS_S = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])  # [s]     default lookback window width
MULTI_RAY_MAX_BATCH_ESTIMATES   = 1000      # [-]     max estimates per batch (performance cap)

# --- Kalman Filter ---
MIN_KALMAN_INIT_OBSERVATIONS    = 3         # [-]     minimum observations to initialise Kalman filter

# =============================================================================
# PARAMETER STUDY DEFAULTS
# =============================================================================

# Full 7×7 grid
STUDY_DISTANCES_M = [
    50e3, 75e3, 100e3, 150e3, 200e3, 500e3, 1000e3     # [m]
]
STUDY_SPEEDS_MS = [
    300.0, 500.0, 750.0, 1000.0, 2000.0, 3000.0, 7000.0  # [m/s]
]

# Quick 3×3 grid (for testing)
STUDY_DISTANCES_QUICK_M = [100e3, 200e3, 500e3]         # [m]
STUDY_SPEEDS_QUICK_MS   = [500.0, 1000.0, 3000.0]       # [m/s]

# =============================================================================
# ANGLE STUDY DEFAULTS (1D crossing-angle sweep)
# =============================================================================

ANGLE_STUDY_DURATION_S  = 400.0                         # [s]    simulation duration
ANGLE_STUDY_DISTANCE_M  = 200e3                         # [m]    fixed closest approach
ANGLE_STUDY_SPEED_MS    = 1000.0                        # [m/s]  fixed missile speed
ANGLE_STUDY_ANGLES_DEG  = list(range(0, 181, 5))        # [deg]  0, 5, 10, ..., 180 (37 values)

# =============================================================================
# ANGULAR STUDY DEFAULTS (2D azimuth × elevation sweep)
# =============================================================================

ANGULAR_STUDY_DISTANCE_M     = 200e3                       # [m]    reference (informational only)
ANGULAR_STUDY_SPEED_MS       = 200.0                       # [m/s]  rocket ascent speed
ANGULAR_STUDY_AZIMUTHS_DEG   = list(range(0, 360, 10))     # [deg]  0,10,...,350 (36 values)
ANGULAR_STUDY_ELEVATIONS_DEG = list(range(0, 91, 10))      # [deg]  0,10,...,90  (10 values)

# Quick 4×4 grid (for testing)
ANGULAR_STUDY_AZIMUTHS_QUICK_DEG   = [0, 90, 180, 270]    # [deg]
ANGULAR_STUDY_ELEVATIONS_QUICK_DEG = [0, 30, 60, 90]      # [deg]

# Ground-launch timing
ANGULAR_STUDY_LEAD_TIME_S       = 20.0   # [s]  rocket launches 20 s before satellite overhead
ANGULAR_STUDY_PRE_OBSERVE_S     = 10.0   # [s]  sim starts 10 s before rocket launch
ANGULAR_STUDY_POST_OVERHEAD_S   = 150.0  # [s]  observe 150 s after satellite overhead
ANGULAR_STUDY_LAUNCH_ALTITUDE_M = 10.0   # [m]  launch height above Earth surface

# Fixed values for 1D sweep comparison plots
ANGULAR_STUDY_AZIMUTH_SWEEP_FIXED_ELEVATION = 0.0          # [deg] el fixed for azimuth sweep
ANGULAR_STUDY_ELEVATION_SWEEP_FIXED_AZIMUTH = 45.0         # [deg] az fixed for elevation sweep

# Fibonacci hemisphere sampling (quasi-uniform solid-angle coverage)
ANGULAR_STUDY_N_FIBONACCI       = 500   # [-]  points for full Fibonacci run
ANGULAR_STUDY_N_FIBONACCI_QUICK = 20    # [-]  points for quick test run

# =============================================================================
# RADIAL LAUNCH SCENARIO DEFAULTS
# =============================================================================

DEFAULT_RADIAL_SPEED_MS    = 200.0   # [m/s]  rocket ascent speed (vertical, straight up)
DEFAULT_LAUNCH_LEAD_TIME_S = 20.0    # [s]    seconds before satellite overhead that rocket launches

# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

KM_TO_M    = 1e3
M_TO_KM    = 1e-3
DEG_TO_RAD = 3.141592653589793 / 180.0
RAD_TO_DEG = 180.0 / 3.141592653589793

# =============================================================================
# REPRODUCIBILITY: SAVE CONSTANTS SNAPSHOT
# =============================================================================

def save_snapshot(output_dir: str) -> None:
    """
    Save a human-readable snapshot of all constants to a run folder.

    Call this once per run right after creating the output directory.
    This ensures every run folder contains an exact record of what
    constants were used, making results fully reproducible.

    Parameters
    ----------
    output_dir : str
        Path to the run folder (e.g. experiments/runs/single_dist200km_...)

    Creates
    -------
    output_dir/constants_snapshot.txt
        Human-readable record of all constants with values and units.
    """
    import os

    filepath = os.path.join(output_dir, 'constants_snapshot.txt')

    lines = []
    lines.append("=" * 60)
    lines.append("CONSTANTS SNAPSHOT")
    lines.append("Saved automatically for reproducibility.")
    lines.append(f"Snapshot saved:        {datetime.datetime.now().isoformat()}")
    lines.append(f"Simulation start time: {DEFAULT_START_TIME.isoformat()}")
    lines.append("=" * 60)

    lines.append("\n--- EARTH PHYSICAL CONSTANTS ---")
    lines.append(f"EARTH_RADIUS_M           = {EARTH_RADIUS_M}  [m]")
    lines.append(f"EARTH_RADIUS_KM          = {EARTH_RADIUS_KM}  [km]")
    lines.append(f"EARTH_MU                 = {EARTH_MU}  [m³/s²]")
    lines.append(f"EARTH_J2                 = {EARTH_J2}  [-]")
    lines.append(f"EARTH_ROTATION_RATE      = {EARTH_ROTATION_RATE}  [rad/s]")
    lines.append(f"EARTH_ESCAPE_VELOCITY_MS = {EARTH_ESCAPE_VELOCITY_MS}  [m/s]")

    lines.append("\n--- SIMULATION DEFAULTS ---")
    lines.append(f"DEFAULT_FPS                    = {DEFAULT_FPS}  [fps]")
    lines.append(f"DEFAULT_DURATION_S             = {DEFAULT_DURATION_S}  [s]")
    lines.append(f"DEFAULT_DETECTION_FRACTION     = {DEFAULT_DETECTION_FRACTION}  [-]")
    lines.append(f"DEFAULT_DETECTION_FRAME_INDEX  = {DEFAULT_DETECTION_FRAME_INDEX}  [-]")
    lines.append(f"DEFAULT_CROSSING_ANGLE_DEG     = {DEFAULT_CROSSING_ANGLE_DEG}  [deg]")

    lines.append("\n--- SATELLITE ORBIT (ERNST) ---")
    lines.append(f"DEFAULT_APOGEE_ALTITUDE_M      = {DEFAULT_APOGEE_ALTITUDE_M}  [m]")
    lines.append(f"DEFAULT_PERIGEE_ALTITUDE_M     = {DEFAULT_PERIGEE_ALTITUDE_M}  [m]")
    lines.append(f"DEFAULT_INCLINATION_DEG        = {DEFAULT_INCLINATION_DEG}  [deg]")
    lines.append(f"DEFAULT_RAAN_DEG               = {DEFAULT_RAAN_DEG}  [deg]")
    lines.append(f"DEFAULT_ARG_PERIGEE_DEG        = {DEFAULT_ARG_PERIGEE_DEG}  [deg]")

    lines.append("\n--- CAMERA ---")
    lines.append(f"DEFAULT_CAMERA_RESOLUTION      = {DEFAULT_CAMERA_RESOLUTION}  [px]")
    lines.append(f"DEFAULT_CAMERA_FOV_DEG         = {DEFAULT_CAMERA_FOV_DEG}  [deg]")
    lines.append(f"DEFAULT_NADIR_LOOK_ANGLE_DEG   = {DEFAULT_NADIR_LOOK_ANGLE_DEG}  [deg]")

    lines.append("\n--- ESTIMATION CONSTANTS ---")
    lines.append(f"MIN_TRIANGULATION_ANGLE_DEG        = {MIN_TRIANGULATION_ANGLE_DEG}  [deg]")
    lines.append(f"MAX_TRIANGULATION_GAP_M            = {MAX_TRIANGULATION_GAP_M}  [m]")
    lines.append(f"PAIRING_TOLERANCE_S                = {PAIRING_TOLERANCE_S}  [s]")
    lines.append(f"DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS = {DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS}  [s]")
    lines.append(f"DEFAULT_MULTI_RAY_OBSERVATIONS     = {DEFAULT_MULTI_RAY_OBSERVATIONS}  [-]")
    lines.append(f"DEFAULT_MULTI_RAY_TIME_WINDOW_S    = {DEFAULT_MULTI_RAY_TIME_WINDOWS_S}  [s]")
    lines.append(f"MULTI_RAY_MAX_BATCH_ESTIMATES      = {MULTI_RAY_MAX_BATCH_ESTIMATES}  [-]")
    lines.append(f"MIN_KALMAN_INIT_OBSERVATIONS       = {MIN_KALMAN_INIT_OBSERVATIONS}  [-]")

    lines.append("\n--- PARAMETER STUDY GRID ---")
    lines.append(f"STUDY_DISTANCES_M       = {STUDY_DISTANCES_M}  [m]")
    lines.append(f"STUDY_SPEEDS_MS         = {STUDY_SPEEDS_MS}  [m/s]")
    lines.append(f"STUDY_DISTANCES_QUICK_M = {STUDY_DISTANCES_QUICK_M}  [m]")
    lines.append(f"STUDY_SPEEDS_QUICK_MS   = {STUDY_SPEEDS_QUICK_MS}  [m/s]")

    lines.append("\n--- UNIT CONVERSIONS ---")
    lines.append(f"KM_TO_M    = {KM_TO_M}")
    lines.append(f"M_TO_KM    = {M_TO_KM}")
    lines.append(f"DEG_TO_RAD = {DEG_TO_RAD}")
    lines.append(f"RAD_TO_DEG = {RAD_TO_DEG}")

    lines.append("\n" + "=" * 60)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
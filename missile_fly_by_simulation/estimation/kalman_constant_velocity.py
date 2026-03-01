"""
Dynamic tracking with Kalman filter.

This module implements depth estimation using a Kalman filter with a
constant-velocity motion model operating directly in 1D depth space.

Unlike a 3D Cartesian formulation, this filter tracks:
    state = [d, d_dot]
where d is the scalar depth (satellite-to-target distance) and d_dot
is the range rate. This makes depth directly observable from a scalar
triangulation measurement, avoiding the fundamental unobservability
problem of angular-only measurements in 3D.

Classes
-------
KalmanDepthTracker
    Track target depth with 1D Kalman filter and constant-velocity model
"""

from typing import List, Optional
from collections import deque
from datetime import timedelta
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.sensing import PinholeCameraModel
from missile_fly_by_simulation.simulation.results import Observation, DepthEstimate
from missile_fly_by_simulation.constants import (
    ATTITUDE_NOISE_DEG,
    DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS,
    DEFAULT_FPS,
)


class KalmanDepthTracker:
    """
    Depth tracking with 1D Kalman filter.

    Tracks target depth using a constant-velocity model in 1D depth space:

    State vector (2D):
        x = [d, d_dot]
        where d      = scalar depth (satellite → target distance) [m]
              d_dot  = range rate [m/s]

    Motion model:
        d(t+dt)     = d(t) + d_dot(t) * dt
        d_dot(t+dt) = d_dot(t)

    Measurement:
        z = triangulated depth from current + past observation (two-ray)
        H = [1, 0]   (depth is directly observed)

    This avoids the unobservability problem of the 3D formulation where
    a single angular measurement cannot constrain depth.

    Examples
    --------
    >>> tracker = KalmanDepthTracker(camera)
    >>> estimates = tracker.estimate_batch(observations)
    """

    def __init__(
        self,
        camera: PinholeCameraModel,
        process_noise_acc: float = 5.0,
        measurement_noise_m: float = 5000.0,
        lookback_time_s: float = None,
        min_init_observations: int = 30,
        fps: float = None,
    ):
        """
        Initialize 1D Kalman filter tracker.

        Parameters
        ----------
        camera : PinholeCameraModel
            Camera model for pixel-to-ray conversion.
        process_noise_acc : float, optional
            Standard deviation of acceleration disturbance [m/s²].
            Drives how fast the filter allows d_dot to change.
            Default 5.0 m/s².
        measurement_noise_m : float, optional
            Standard deviation of the triangulated depth measurement [m].
            Should reflect typical two-ray triangulation error.
            Default 5000.0 m (5 km).
        lookback_time_s : float, optional
            Time offset used to find the past observation for triangulation.
            Defaults to DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS[0] (typically 1s).
        min_init_observations : int, optional
            Number of observations to collect before initialising filter.
            Default 30.
        fps : float, optional
            Camera frame rate [fps]. Used to compute the observation-search
            tolerance window (2 frame intervals). Defaults to DEFAULT_FPS.
        """
        self.camera = camera
        self.process_noise_acc = process_noise_acc
        self.measurement_noise_m = measurement_noise_m
        self.lookback_time_s = (
            lookback_time_s
            if lookback_time_s is not None
            else float(DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS[0])
        )
        self.min_init_observations = min_init_observations
        self._fps = fps if fps is not None else float(DEFAULT_FPS)

        # Measurement noise variance (scalar)
        self.R = measurement_noise_m ** 2

        # Filter state — initialised on first observation
        self.state = None        # [d, d_dot]
        self.covariance = None   # 2×2 matrix
        self.last_time = None

        # Bookkeeping
        self.is_initialized = False
        self.init_observations: List[Observation] = []

        # Rolling buffer of all observations for lookback triangulation
        # Keep enough history to always find an obs ~lookback_time_s ago
        self._obs_buffer: deque = deque(maxlen=10000)

    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================

    def estimate_batch(
        self,
        observations: List[Observation],
    ) -> List[DepthEstimate]:
        """
        Process all observations sequentially through the Kalman filter.

        Parameters
        ----------
        observations : list of Observation
            All observations in chronological order.

        Returns
        -------
        list of DepthEstimate
            One estimate per observation once the filter is initialised.
        """
        self._reset()

        estimates = []
        for obs in observations:
            estimate = self.update(obs)
            if estimate is not None:
                estimates.append(estimate)

        return estimates

    def update(self, observation: Observation) -> Optional[DepthEstimate]:
        """
        Update filter with one new observation.

        Parameters
        ----------
        observation : Observation
            New observation to process.

        Returns
        -------
        DepthEstimate or None
            Depth estimate, or None during initialisation phase.
        """
        # Always buffer the observation for lookback
        self._obs_buffer.append(observation)

        if not self.is_initialized:
            return self._initialize(observation)
        else:
            return self._kalman_update(observation)

    # =========================================================================
    # PRIVATE: INITIALISATION
    # =========================================================================

    def _reset(self):
        """Reset filter to blank state for a fresh batch run."""
        self.state = None
        self.covariance = None
        self.last_time = None
        self.is_initialized = False
        self.init_observations = []
        self._obs_buffer.clear()

    def _initialize(self, observation: Observation) -> Optional[DepthEstimate]:
        """
        Collect initial observations and initialise filter state.

        Waits for min_init_observations, then initialises depth from
        a triangulated measurement and depth_rate from zero.
        """
        self.init_observations.append(observation)

        if len(self.init_observations) < self.min_init_observations:
            return None

        # Get initial depth from triangulation using first and last init obs
        obs_first = self.init_observations[0]
        obs_last  = self.init_observations[-1]

        initial_depth = self._triangulate_two_obs(obs_first, obs_last)

        if initial_depth is None:
            # Triangulation failed — use conservative physical prior
            # (orbital altitude is a reasonable upper bound on depth)
            initial_depth = 300e3  # 300 km conservative prior [m]

        # Initialise state: depth from triangulation, rate = 0 (unknown)
        self.state = np.array([initial_depth, 0.0])

        # Large initial covariance — very uncertain at start
        self.covariance = np.diag([
            (500e3) ** 2,   # ±500 km depth uncertainty
            (2000.0) ** 2,  # ±2000 m/s range rate uncertainty
        ])

        self.last_time = observation.timestamp
        self.is_initialized = True

        return DepthEstimate(
            timestamp=observation.timestamp,
            estimated_depth=self.state[0],
            true_depth=observation.true_depth,
            error=self.state[0] - observation.true_depth,
            num_observations_used=len(self.init_observations),
        )

    # =========================================================================
    # PRIVATE: KALMAN PREDICT + UPDATE
    # =========================================================================

    def _kalman_update(self, observation: Observation) -> DepthEstimate:
        """
        One full Kalman predict + update cycle.

        PREDICT:
            Uses constant-velocity (constant range rate) model.
            d(t+dt)     = d(t) + d_dot(t) * dt
            d_dot(t+dt) = d_dot(t)

        UPDATE:
            Measurement z = triangulated depth from two-ray.
            H = [1, 0]  →  innovation = z - d_predicted
            If triangulation fails, skip update (predict only).
        """
        dt = (observation.timestamp - self.last_time).total_seconds()
        if dt <= 0:
            dt = 1e-6  # guard against duplicate timestamps

        # ── PREDICT ──────────────────────────────────────────────────
        F = np.array([[1.0, dt ],
                      [0.0, 1.0]])

        # Process noise: acceleration disturbance drives d_dot
        # Q = G * G^T * σ_a²  with G = [dt²/2, dt]^T
        G = np.array([0.5 * dt**2, dt])
        Q = np.outer(G, G) * self.process_noise_acc**2

        state_pred = F @ self.state
        cov_pred   = F @ self.covariance @ F.T + Q

        # ── UPDATE ───────────────────────────────────────────────────
        # Measurement: triangulate depth from current obs + past obs
        z = self._get_depth_measurement(observation)

        self.last_time = observation.timestamp

        if z is None:
            # No valid measurement — propagate only (coasted prediction)
            self.state      = state_pred
            self.covariance = cov_pred
        else:
            # H = [1, 0]: depth is directly observed
            H = np.array([[1.0, 0.0]])

            # Innovation
            innovation = z - (H @ state_pred)[0]

            # Innovation covariance
            S = (H @ cov_pred @ H.T)[0, 0] + self.R

            # Kalman gain (2×1)
            K = (cov_pred @ H.T) / S

            # Corrected state and covariance
            self.state      = state_pred + K.flatten() * innovation
            self.covariance = (np.eye(2) - np.outer(K.flatten(), H)) @ cov_pred

        # Depth must be positive
        self.state[0] = max(self.state[0], 0.0)

        # Divergence guard — reset if state has exploded or gone NaN
        if self.state[0] > 1e9 or np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            self._reset()
            return None

        return DepthEstimate(
            timestamp=observation.timestamp,
            estimated_depth=self.state[0],
            true_depth=observation.true_depth,
            error=self.state[0] - observation.true_depth,
            num_observations_used=1,
        )

    # =========================================================================
    # PRIVATE: MEASUREMENT
    # =========================================================================

    def _get_depth_measurement(self, obs_now: Observation) -> Optional[float]:
        """
        Find a past observation ~lookback_time_s ago and triangulate depth.

        Searches the observation buffer for the observation whose timestamp
        is closest to (obs_now.timestamp - lookback_time_s), then calls
        the two-ray triangulator on the pair.

        Returns
        -------
        depth : float or None
            Triangulated depth in metres, or None if no suitable past
            observation exists or triangulation fails.
        """
        target_time = obs_now.timestamp - timedelta(seconds=self.lookback_time_s)

        # Find closest observation in buffer to target_time
        obs_past = None
        best_dt = float('inf')

        for obs in self._obs_buffer:
            if obs is obs_now:
                continue
            dt = abs((obs.timestamp - target_time).total_seconds())
            if dt < best_dt:
                best_dt = dt
                obs_past = obs

        # Reject if best match is more than 2 frames away from desired lookback
        frame_interval = 1.0 / self._fps
        if obs_past is None or best_dt > 2.0 * frame_interval:
            return None

        return self._triangulate_two_obs(obs_past, obs_now)

    def _triangulate_two_obs(
        self,
        obs_a: Observation,
        obs_b: Observation,
    ) -> Optional[float]:
        """
        Triangulate depth from two observations using closest-approach.

        Parameters
        ----------
        obs_a, obs_b : Observation
            Two observations at different satellite positions.

        Returns
        -------
        depth : float or None
            Distance from obs_b satellite position to triangulated point,
            or None if geometry is degenerate.
        """
        # Ray origins (satellite positions)
        o1 = np.asarray(obs_a.satellite_state.position, dtype=np.float64)
        o2 = np.asarray(obs_b.satellite_state.position, dtype=np.float64)

        # Ray directions (world frame)
        ray_cam_a = self.camera.pixel_to_ray(obs_a.pixel)
        ray_cam_b = self.camera.pixel_to_ray(obs_b.pixel)
        d1 = obs_a.satellite_state.attitude.satellite_to_world(ray_cam_a)
        d2 = obs_b.satellite_state.attitude.satellite_to_world(ray_cam_b)

        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)

        # Closest approach between two rays (standard formula)
        w0 = o1 - o2
        a  = np.dot(d1, d1)
        b  = np.dot(d1, d2)
        c  = np.dot(d2, d2)
        d  = np.dot(d1, w0)
        e  = np.dot(d2, w0)

        denom = a * c - b * b

        # Rays nearly parallel — degenerate geometry
        if abs(denom) < 1e-10:
            return None

        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom

        # Both parameters must be positive (target is in front of satellite)
        if t1 < 0 or t2 < 0:
            return None

        p1 = o1 + t1 * d1
        p2 = o2 + t2 * d2

        # Midpoint of closest approach
        midpoint = 0.5 * (p1 + p2)

        # Depth = distance from current satellite to triangulated point
        depth = np.linalg.norm(midpoint - o2)

        return float(depth)

    # =========================================================================
    # DUNDER
    # =========================================================================

    def __repr__(self) -> str:
        if self.is_initialized:
            return (
                f"KalmanDepthTracker(1D, "
                f"d={self.state[0]/1e3:.1f} km, "
                f"d_dot={self.state[1]:.1f} m/s)"
            )
        else:
            n = len(self.init_observations)
            return (
                f"KalmanDepthTracker(1D, "
                f"initializing {n}/{self.min_init_observations})"
            )
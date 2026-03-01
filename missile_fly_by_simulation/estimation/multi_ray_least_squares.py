"""
Multi-ray least-squares depth estimation.

This module implements an improved depth estimation method that uses
multiple observations (N ≥ 2) instead of just two.

By using more observations, this method:
- Averages out measurement noise
- Provides more robust estimates
- Uses sliding time windows for continuous estimation

Classes
-------
MultiRayLeastSquaresEstimator
    Estimate depth from multiple observations using least-squares
"""

from typing import List, Optional
from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.sensing import PinholeCameraModel
from missile_fly_by_simulation.simulation import Observation, DepthEstimate
from missile_fly_by_simulation.constants import (
    DEFAULT_MULTI_RAY_TIME_WINDOWS_S,
    DEFAULT_MULTI_RAY_OBSERVATIONS,
    MULTI_RAY_MAX_BATCH_ESTIMATES,
)


class MultiRayLeastSquaresEstimator:
    """
    Depth estimation from multiple observations (N ≥ 2).

    This estimator finds the 3D point that minimizes the sum of
    squared distances to all observation rays.

    This is more robust than two-ray triangulation because:
    - Uses more data (N observations instead of 2)
    - Averages out noise
    - Can handle inconsistent measurements

    The optimization problem:
        minimize: sum_i (distance from point to ray_i)²

    The N observations are selected evenly across the time window
    to maximize geometric spread along the satellite trajectory,
    which improves the conditioning of the least-squares solve.

    Attributes
    ----------
    camera : PinholeCameraModel
        Camera model for pixel → ray conversion

    Examples
    --------
    >>> estimator = MultiRayLeastSquaresEstimator(camera)
    >>>
    >>> estimate = estimator.estimate_depth_at_time(
    ...     timestamp=some_time,
    ...     observations=all_observations,
    ...     time_window=10.0,
    ...     n_observations=10,
    ... )
    """

    def __init__(self, camera: PinholeCameraModel):
        """
        Initialize multi-ray estimator.

        Parameters
        ----------
        camera : PinholeCameraModel
            Camera model for pixel → ray conversion
        """
        self.camera = camera

    def estimate_depth_at_time(
        self,
        timestamp: datetime,
        observations: List[Observation],
        time_window: float = DEFAULT_MULTI_RAY_TIME_WINDOWS_S,
        n_observations: int = DEFAULT_MULTI_RAY_OBSERVATIONS,
    ) -> Optional[DepthEstimate]:
        """
        Estimate depth using N evenly-spaced observations within time window.

        Collects all observations in [timestamp - time_window, timestamp],
        then selects n_observations evenly distributed across that window.
        Even spacing maximises geometric spread along the satellite trajectory,
        giving the least-squares solver the best possible baseline.

        Parameters
        ----------
        timestamp : datetime
            Reference time for estimation (the "now" point).
        observations : list of Observation
            All available observations (must be sorted by timestamp).
        time_window : float
            Width of the lookback window [seconds].
            Only past observations up to timestamp are used (causal).
        n_observations : int
            Number of evenly-spaced observations to select from the window.
            More observations → better noise averaging but slower solve.
            If fewer than n_observations exist in the window, all are used.

        Returns
        -------
        DepthEstimate or None
            Depth estimate, or None if fewer than 2 observations in window.
        """
        # Collect all observations in the causal window [t - time_window, t]
        window_start = timestamp - timedelta(seconds=time_window)
        window_end = timestamp

        windowed_obs = [
            obs for obs in observations
            if window_start <= obs.timestamp <= window_end
        ]

        # Hard minimum: need at least 2 rays to constrain a 3D point
        if len(windowed_obs) < 2:
            return None

        # ── EVENLY DISTRIBUTED SAMPLING ──────────────────────────────────
        # Pick n_observations indices evenly spaced across the windowed list.
        # This maximises geometric spread along the satellite trajectory,
        # which is what actually improves the least-squares conditioning.
        #
        # np.linspace(0, len-1, n_observations) gives evenly spaced float
        # indices → round to int → deduplicate with dict.fromkeys.
        if len(windowed_obs) <= n_observations:
            # Fewer real observations than requested — use all of them
            selected_obs = windowed_obs
        else:
            indices = np.linspace(0, len(windowed_obs) - 1, n_observations)
            indices = list(dict.fromkeys(round(idx) for idx in indices))
            selected_obs = [windowed_obs[idx] for idx in indices]
        # ─────────────────────────────────────────────────────────────────

        return self._estimate_from_observations(selected_obs, reference_time=timestamp)

    def estimate_batch(
        self,
        observations: List[Observation],
        time_windows: List[float],
        n_observations_list: List[int],
    ) -> List[DepthEstimate]:
        """
        Estimate depths across a grid of time windows and observation counts.

        For each observation, each time window, and each n_observations,
        calls estimate_depth_at_time(). This enables a parameter study
        of both axes simultaneously.

        Parameters
        ----------
        observations : list of Observation
            All observations (must be sorted by timestamp).
        time_windows : list of float
            Time window sizes to try [seconds], e.g. [5.0, 10.0, 20.0].
        n_observations_list : list of int
            Observation counts to try, e.g. [5, 10, 20].

        Returns
        -------
        list of DepthEstimate
            All estimates. Each DepthEstimate carries num_observations_used
            and time_offset (window width) so results can be grouped downstream
            for separate plots per (time_window, n_observations) combination.
        """
        if not observations:
            return []

        estimates = []

        # ── SUBSAMPLING ───────────────────────────────────────────────────
        # Don't estimate at every single frame — stride through observations
        # so total estimates stay within MULTI_RAY_MAX_BATCH_ESTIMATES per
        # (time_window, n_observations) combination.
        step = max(1, len(observations) // MULTI_RAY_MAX_BATCH_ESTIMATES)
        # ─────────────────────────────────────────────────────────────────

        # ── OUTER LOOP: iterate over subsampled observations ──────────────
        # Each obs acts as the "now" reference point for one batch of estimates.
        for i in range(0, len(observations), step):
            obs = observations[i]

            # ── MIDDLE LOOP: try each time window ─────────────────────────
            for window in time_windows:

                # ── INNER LOOP: try each observation count ─────────────────
                # Together with the time_window loop, this sweeps the full
                # (time_window × n_observations) parameter grid for each obs.
                for n_obs in n_observations_list:
                    estimate = self.estimate_depth_at_time(
                        timestamp=obs.timestamp,
                        observations=observations,
                        time_window=window,
                        n_observations=n_obs,
                    )
                    if estimate is not None:
                        estimate.time_offset = window
                        estimates.append(estimate)

        return estimates

    def _estimate_from_observations(
        self,
        observations: List[Observation],
        reference_time: datetime,
    ) -> Optional[DepthEstimate]:
        """
        Estimate depth from a pre-selected set of observations.

        Uses least-squares to find the 3D point minimizing the sum of
        squared distances to all observation rays.

        Parameters
        ----------
        observations : list of Observation
            Pre-selected observations to use (already evenly spaced).
        reference_time : datetime
            Reference timestamp for the estimate (the "now" point).

        Returns
        -------
        DepthEstimate or None
            Estimated depth, or None if the solve fails.
        """
        if len(observations) < 2:
            return None

        try:
            # Build ray origins and directions from each observation
            origins = []
            directions = []

            for obs in observations:
                # Convert pixel → ray in camera frame
                ray_cam = self.camera.pixel_to_ray(obs.pixel)
                # Rotate ray into world frame (ECI)
                ray_world = obs.satellite_state.attitude.satellite_to_world(ray_cam)

                origins.append(obs.satellite_state.position)
                directions.append(ray_world)

            origins = np.array(origins)
            directions = np.array(directions)

            # Find the 3D point that minimizes distance to all rays
            point = self._solve_least_squares(origins, directions)

            # Reference observation = closest to reference_time (i.e. most recent)
            ref_obs = min(
                observations,
                key=lambda obs: abs((obs.timestamp - reference_time).total_seconds())
            )

            # Depth = distance from reference satellite position to estimated point
            depth = np.linalg.norm(point - ref_obs.satellite_state.position)

            # Residual = RMS distance from point to all rays (quality metric)
            residual = self._compute_residual(point, origins, directions)

            # Time span of the selected observations (proxy for effective baseline)
            time_span = (
                max(obs.timestamp for obs in observations) -
                min(obs.timestamp for obs in observations)
            ).total_seconds()

            return DepthEstimate(
                timestamp=reference_time,
                estimated_depth=depth,
                true_depth=ref_obs.true_depth,
                error=depth - ref_obs.true_depth,
                time_offset=time_span,
                triangulation_gap=residual,
                num_observations_used=len(observations),
            )

        except Exception:
            return None

    def _solve_least_squares(
        self,
        origins: npt.NDArray[np.float64],
        directions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Solve least-squares problem to find 3D point.

        Finds point P that minimizes:
            sum_i (distance from P to ray_i)²

        Uses the closed-form normal equations:
            A * P = b
        where A and b are accumulated from the ray geometry via
        the projection matrix (I - d*dᵀ) for each ray direction d.

        Parameters
        ----------
        origins : ndarray of shape (N, 3)
            Ray origins (satellite positions) [m]
        directions : ndarray of shape (N, 3)
            Ray directions (normalized) in world frame

        Returns
        -------
        point : ndarray of shape (3,)
            Estimated 3D position [m]
        """
        A = np.zeros((3, 3))
        b = np.zeros(3)

        for o, d in zip(origins, directions):
            # Projection matrix: I - d*dᵀ (projects onto plane perpendicular to ray)
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ o

        try:
            point = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Singular matrix (rays nearly parallel) — fall back to lstsq
            point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        return point

    def _compute_residual(
        self,
        point: npt.NDArray[np.float64],
        origins: npt.NDArray[np.float64],
        directions: npt.NDArray[np.float64],
    ) -> float:
        """
        Compute RMS distance from estimated point to all rays.

        A low residual means the rays agree well (good geometry).
        A high residual means inconsistent or nearly parallel rays.

        Parameters
        ----------
        point : ndarray of shape (3,)
            Estimated 3D position [m]
        origins : ndarray of shape (N, 3)
            Ray origins [m]
        directions : ndarray of shape (N, 3)
            Ray directions (normalized)

        Returns
        -------
        residual : float
            RMS perpendicular distance from point to all rays [m]
        """
        total = 0.0

        for o, d in zip(origins, directions):
            v = point - o
            dist = np.linalg.norm(v - np.dot(v, d) * d)
            total += dist ** 2

        return np.sqrt(total / len(origins))

    def __repr__(self) -> str:
        return f"MultiRayLeastSquaresEstimator(camera={self.camera})"
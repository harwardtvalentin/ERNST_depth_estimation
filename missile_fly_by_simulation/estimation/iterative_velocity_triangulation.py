"""
Iterative velocity-corrected triangulation depth estimator.

This module implements a novel depth estimation method that corrects for
target motion during the observation window — a systematic error source
that standard two-ray triangulation ignores.

Standard two-ray triangulation assumes the target did not move between
the two observations. For a missile at 1000 m/s observed over 1 second,
this introduces a ~1 km position error into every estimate. This method
removes that assumption by incorporating an iteratively refined velocity
estimate into the ray-matching geometry.

Algorithm
---------
Given four observations at times t, t+dt_short, t+T, t+T+dt_short:

1. Coarse triangulation of two positions (early pair, late pair)
2. Velocity estimate from the two coarse positions
3. Velocity-corrected triangulation: find P such that
       P           lies close to ray(t)
       P + v*dt    lies close to ray(t + dt_short)
4. Recompute velocity from corrected positions
5. Iterate until velocity converges

The corrected positions are physically consistent with the estimated
velocity, eliminating the motion-during-window bias.

Classes
-------
IterativeVelocityTriangulator
    Depth estimator using iterative velocity-corrected triangulation
"""

from typing import List, Optional, Tuple
from datetime import timedelta, datetime
import bisect
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.sensing import PinholeCameraModel
from missile_fly_by_simulation.simulation.results import Observation, DepthEstimate
from missile_fly_by_simulation.constants import (
    DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS,
    DEFAULT_FPS,
)


class IterativeVelocityTriangulator:
    """
    Depth estimation using iterative velocity-corrected triangulation.

    Corrects for target motion during the observation window by jointly
    estimating target position and velocity from two pairs of observations
    separated by a long baseline.

    Parameters
    ----------
    camera : PinholeCameraModel
        Camera model for pixel-to-ray conversion.
    short_window_s : float
        Time separation within each observation pair [s].
        Controls sensitivity to within-pair motion correction.
        Default: 1.0 s (same as two-ray Δt=1s baseline).
    long_window_s : float
        Time separation between the two pairs [s].
        Controls velocity estimation accuracy — larger is better
        but introduces more latency.
        Default: 10.0 s.
    max_iterations : int
        Maximum number of velocity refinement iterations.
        Convergence typically occurs within 3–5 iterations.
        Default: 10.
    convergence_tol_ms : float
        Convergence threshold on velocity magnitude change [m/s].
        Default: 1.0 m/s.

    Notes
    -----
    The method tags each estimate to the earliest of the four
    observations (t_ref). This introduces a latency of
    long_window_s + short_window_s before the first estimate
    can be produced.

    Examples
    --------
    >>> estimator = IterativeVelocityTriangulator(camera)
    >>> estimates = estimator.estimate_batch(observations)
    """

    def __init__(
        self,
        camera: PinholeCameraModel,
        short_window_s: float = None,
        long_window_s: float = 10.0,
        max_iterations: int = 10,
        convergence_tol_ms: float = 1.0,
        fps: float = None,
    ):
        self.camera = camera
        self.short_window_s = (
            short_window_s
            if short_window_s is not None
            else float(DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS[0])
        )
        self.long_window_s = long_window_s
        self.max_iterations = max_iterations
        self.convergence_tol_ms = convergence_tol_ms

        # Maximum allowed mismatch when searching for observations
        # near a target time (2 frame intervals)
        _fps = fps if fps is not None else float(DEFAULT_FPS)
        self._frame_tolerance_s = 2.0 / _fps

    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================

    def estimate_batch(
        self,
        observations: List[Observation],
        max_iterations: int = None,
    ) -> List[DepthEstimate]:
        """
        Process all observations and return velocity-corrected depth estimates.

        Builds a sorted timestamp index once at the start (O(n log n)),
        then uses binary search for all observation lookups (O(log n) each),
        avoiding the O(n²) cost of linear search.

        Parameters
        ----------
        observations : list of Observation
            All observations in chronological order.
        max_iterations : int, optional
            Override self.max_iterations for this batch only.
            Useful for convergence analysis plots.

        Returns
        -------
        list of DepthEstimate
            One estimate per reference time where four valid observations
            are available. First estimates appear after
            long_window_s + short_window_s seconds.
        """
        if len(observations) < 4:
            return []

        # Build timestamp index once — O(n log n)
        # Store as float seconds since first observation for fast comparison
        t0 = observations[0].timestamp
        self._index_t0 = t0
        self._index_obs = observations
        self._index_ts = [
            (obs.timestamp - t0).total_seconds()
            for obs in observations
        ]

        # Temporarily override max_iterations if requested
        original_max_iter = self.max_iterations
        if max_iterations is not None:
            self.max_iterations = max_iterations

        estimates = []
        for obs_ref in observations:
            estimate = self._estimate_at_time(obs_ref)
            if estimate is not None:
                estimates.append(estimate)

        # Restore
        self.max_iterations = original_max_iter

        # Clear index to free memory
        self._index_obs = None
        self._index_ts  = None

        return estimates

    # =========================================================================
    # PRIVATE: MAIN ESTIMATION LOGIC
    # =========================================================================

    def _estimate_at_time(
        self,
        obs_ref: Observation,
    ) -> Optional[DepthEstimate]:
        """
        Produce one velocity-corrected depth estimate at obs_ref time.

        Finds four observations using binary search on the timestamp index:
            A: t_ref                          (early pair, first)
            B: t_ref + short_window_s         (early pair, second)
            C: t_ref + long_window_s          (late pair, first)
            D: t_ref + long_window_s + short  (late pair, second)

        Runs the iterative velocity correction and returns a DepthEstimate
        tagged to t_ref.
        """
        t_ref = obs_ref.timestamp

        obs_A = obs_ref
        obs_B = self._find_obs_near_time(t_ref + timedelta(seconds=self.short_window_s))
        obs_C = self._find_obs_near_time(t_ref + timedelta(seconds=self.long_window_s))
        obs_D = self._find_obs_near_time(
            t_ref + timedelta(seconds=self.long_window_s + self.short_window_s)
        )

        if any(o is None for o in [obs_B, obs_C, obs_D]):
            return None

        # Run iterative velocity-corrected triangulation
        result = self._iterate(obs_A, obs_B, obs_C, obs_D)
        if result is None:
            return None

        P_early, _, v_final, n_iter = result

        # Depth = distance from satellite at t_ref to corrected early position
        sat_pos = np.asarray(obs_A.satellite_state.position, dtype=np.float64)
        depth = float(np.linalg.norm(P_early - sat_pos))

        # Sanity check
        if depth <= 0 or depth > 1e9 or np.isnan(depth):
            return None

        return DepthEstimate(
            timestamp=obs_ref.timestamp,
            estimated_depth=depth,
            true_depth=obs_ref.true_depth,
            error=depth - obs_ref.true_depth,
            num_observations_used=4,
        )

    def _iterate(
        self,
        obs_A: Observation,
        obs_B: Observation,
        obs_C: Observation,
        obs_D: Observation,
    ) -> Optional[Tuple[npt.NDArray, npt.NDArray, npt.NDArray, int]]:
        """
        Iteratively refine velocity and corrected positions.

        Parameters
        ----------
        obs_A, obs_B : Observation
            Early pair (separated by short_window_s)
        obs_C, obs_D : Observation
            Late pair (separated by short_window_s)

        Returns
        -------
        (P_early, P_late, v_final, n_iterations) or None if failed
            P_early : corrected position at time of obs_A [m]
            P_late  : corrected position at time of obs_C [m]
            v_final : converged velocity estimate [m/s]
            n_iter  : number of iterations until convergence
        """
        # Time between the two pairs (for velocity estimation)
        T = (obs_C.timestamp - obs_A.timestamp).total_seconds()
        if T <= 0:
            return None

        # Iteration 0: start with zero velocity (= standard triangulation)
        v_k = np.zeros(3)

        P_early = None
        P_late  = None

        for k in range(self.max_iterations):
            # Velocity-corrected triangulation of early pair
            P_early = self._triangulate_corrected_pair(obs_A, obs_B, v_k)
            if P_early is None:
                return None

            # Velocity-corrected triangulation of late pair
            P_late = self._triangulate_corrected_pair(obs_C, obs_D, v_k)
            if P_late is None:
                return None

            # Updated velocity estimate
            v_new = self._estimate_velocity(P_early, P_late, T)

            # Check convergence
            if np.linalg.norm(v_new - v_k) < self.convergence_tol_ms:
                return P_early, P_late, v_new, k + 1

            v_k = v_new

        # Return best estimate even if not fully converged
        return P_early, P_late, v_k, self.max_iterations

    # =========================================================================
    # PRIVATE: GEOMETRY
    # =========================================================================

    def _triangulate_corrected_pair(
        self,
        obs_a: Observation,
        obs_b: Observation,
        velocity: npt.NDArray[np.float64],
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        Velocity-corrected triangulation from a pair of observations.

        Finds position P at time t_a such that:
            P               lies close to ray(t_a)
            P + v * dt_ab   lies close to ray(t_b)

        The key insight: shift the origin of ray_b back by v*dt so that
        both rays refer to the same reference time t_a. Then apply
        standard closest-approach triangulation on the shifted rays.

        Parameters
        ----------
        obs_a, obs_b : Observation
            Observation pair.
        velocity : ndarray of shape (3,)
            Current velocity estimate [m/s].

        Returns
        -------
        P : ndarray of shape (3,) or None
            Corrected position at time t_a, or None if geometry degenerate.
        """
        dt_ab = (obs_b.timestamp - obs_a.timestamp).total_seconds()

        # Ray origins (satellite positions)
        o1 = np.asarray(obs_a.satellite_state.position, dtype=np.float64)
        o2 = np.asarray(obs_b.satellite_state.position, dtype=np.float64)

        # Ray directions in world frame
        d1 = self._pixel_to_world_ray(obs_a)
        d2 = self._pixel_to_world_ray(obs_b)

        if d1 is None or d2 is None:
            return None

        # Shift ray_b origin back by v*dt so both rays refer to t_a.
        # If target is at P at t_a, it is at P + v*dt_ab at t_b.
        # Equivalently, the ray from o2 toward (P + v*dt_ab) is the same
        # as a ray from (o2 - v*dt_ab) toward P.
        o2_corrected = o2 - velocity * dt_ab

        # Standard closest-approach triangulation on corrected rays
        return self._closest_approach_midpoint(o1, d1, o2_corrected, d2)

    def _closest_approach_midpoint(
        self,
        o1: npt.NDArray[np.float64],
        d1: npt.NDArray[np.float64],
        o2: npt.NDArray[np.float64],
        d2: npt.NDArray[np.float64],
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        Find midpoint of closest approach between two rays.

        Ray 1: o1 + t1 * d1
        Ray 2: o2 + t2 * d2

        Returns midpoint 0.5*(P1 + P2) or None if geometry degenerate.
        """
        w0 = o1 - o2
        a  = np.dot(d1, d1)
        b  = np.dot(d1, d2)
        c  = np.dot(d2, d2)
        d  = np.dot(d1, w0)
        e  = np.dot(d2, w0)

        denom = a * c - b * b

        if abs(denom) < 1e-10:
            return None  # rays nearly parallel

        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom

        if t1 < 0 or t2 < 0:
            return None  # intersection behind satellite

        p1 = o1 + t1 * d1
        p2 = o2 + t2 * d2

        return 0.5 * (p1 + p2)

    def _estimate_velocity(
        self,
        P_early: npt.NDArray[np.float64],
        P_late:  npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """
        Estimate velocity from two corrected positions and time difference.

        Parameters
        ----------
        P_early, P_late : ndarray of shape (3,)
            Corrected positions at the two reference times.
        dt : float
            Time between reference times [s].

        Returns
        -------
        velocity : ndarray of shape (3,)
            Velocity estimate [m/s].
        """
        return (P_late - P_early) / dt

    def _pixel_to_world_ray(
        self,
        obs: Observation,
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        Convert observation pixel to normalised world-frame ray direction.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        ray : ndarray of shape (3,) or None
        """
        try:
            ray_cam = self.camera.pixel_to_ray(obs.pixel)
            ray_world = obs.satellite_state.attitude.satellite_to_world(ray_cam)
            norm = np.linalg.norm(ray_world)
            if norm < 1e-10:
                return None
            return ray_world / norm
        except Exception:
            return None

    # =========================================================================
    # PRIVATE: OBSERVATION LOOKUP
    # =========================================================================

    def _find_obs_near_time(
        self,
        target_time: datetime,
    ) -> Optional[Observation]:
        """
        Find observation closest to target_time using binary search.

        Operates on the timestamp index built in estimate_batch.
        O(log n) per call instead of O(n) linear scan.

        Parameters
        ----------
        target_time : datetime
            Desired timestamp.

        Returns
        -------
        Observation or None
            Closest observation within self._frame_tolerance_s, or None.
        """
        target_s = (target_time - self._index_t0).total_seconds()

        # Binary search for insertion point
        idx = bisect.bisect_left(self._index_ts, target_s)

        # Check the two neighbours around the insertion point
        best_obs = None
        best_dt  = float('inf')

        for i in (idx - 1, idx):
            if 0 <= i < len(self._index_ts):
                dt = abs(self._index_ts[i] - target_s)
                if dt < best_dt:
                    best_dt = dt
                    best_obs = self._index_obs[i]

        if best_obs is None or best_dt > self._frame_tolerance_s:
            return None

        return best_obs

    # =========================================================================
    # DUNDER
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"IterativeVelocityTriangulator("
            f"short={self.short_window_s}s, "
            f"long={self.long_window_s}s, "
            f"max_iter={self.max_iterations})"
        )
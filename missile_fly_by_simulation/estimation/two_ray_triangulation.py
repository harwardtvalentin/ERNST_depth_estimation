"""
Two-ray triangulation for depth estimation.

This module implements the baseline depth estimation method:
uses exactly two observations separated by Δt to triangulate depth.

This corresponds to the classic "depth from motion" approach where
the satellite observes the target at two different times/positions.

CAUSAL DESIGN: All estimates are strictly causal. To produce an estimate
at time t, only observations from times ≤ t are used. Specifically,
obs1 is the reference (current) observation at time t, and obs2 is
looked up at time t - Δt (in the past).

Classes
-------
TwoRayTriangulationEstimator
    Estimate depth from pairs of observations
"""

from typing import Tuple, List, Optional
from datetime import datetime, timedelta
import bisect
import numpy as np
import numpy.typing as npt

from missile_fly_by_simulation.sensing import PinholeCameraModel
from missile_fly_by_simulation.simulation.results import Observation, DepthEstimate
from missile_fly_by_simulation.constants import (
    MIN_TRIANGULATION_ANGLE_DEG,
    MAX_TRIANGULATION_GAP_M,
    PAIRING_TOLERANCE_S,
    DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS,
)


# =============================================================================
# MAIN ESTIMATOR CLASS
# =============================================================================

class TwoRayTriangulationEstimator:
    """
    Depth estimation from two observations.

    This is the baseline method: uses exactly 2 observations
    separated by a time offset Δt to triangulate the target's
    3D position and compute depth.

    CAUSAL: The estimate at time t uses obs1 (at time t) and
    obs2 (at time t - Δt). No future observations are ever used.

    Attributes
    ----------
    camera : PinholeCameraModel
        Camera model for converting pixels to rays
    min_triangulation_angle_deg : float
        Minimum angle between rays for valid triangulation [degrees]
    max_gap_meters : float
        Maximum triangulation gap for valid estimate [m]
    """

    def __init__(
        self,
        camera: PinholeCameraModel,
        min_triangulation_angle_deg: float = MIN_TRIANGULATION_ANGLE_DEG,
        max_gap_meters: float = MAX_TRIANGULATION_GAP_M,
    ):
        self.camera = camera
        self.min_triangulation_angle_deg = min_triangulation_angle_deg
        self.max_gap_meters = max_gap_meters

    def estimate_depth(
        self,
        obs1: Observation,
        obs2: Observation,
        time_offset: Optional[float] = None,
    ) -> Optional[DepthEstimate]:
        """
        Estimate depth from exactly two observations.

        obs1 is the current (later) observation; obs2 is the past
        (earlier) observation. The estimate is attributed to obs1.timestamp.

        Parameters
        ----------
        obs1 : Observation
            Current observation (later in time) — the reference point.
        obs2 : Observation
            Past observation (earlier in time, at t - Δt).
        time_offset : float, optional
            The intended time offset [s] used for pairing (Δt).
            If None, computed from the actual timestamps.

        Returns
        -------
        DepthEstimate or None
            None if triangulation fails or geometry is too poor.
            Timestamp of result = obs1.timestamp (causal).
        """
        try:
            # Step 1: Pixels → rays in camera frame
            ray1_cam = self.camera.pixel_to_ray(obs1.pixel)
            ray2_cam = self.camera.pixel_to_ray(obs2.pixel)

            # Step 2: Rotate rays into world frame (ECI)
            ray1_world = obs1.satellite_state.attitude.satellite_to_world(ray1_cam)
            ray2_world = obs2.satellite_state.attitude.satellite_to_world(ray2_cam)

            # Step 3: Ray origins = satellite positions
            origin1 = obs1.satellite_state.position
            origin2 = obs2.satellite_state.position

            # Step 4: Check angle between rays (geometry quality)
            cos_angle = np.clip(np.dot(ray1_world, ray2_world), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(abs(cos_angle)))

            if angle_deg < self.min_triangulation_angle_deg:
                return None  # Rays too parallel

            # Step 5: Triangulate
            point, depth1, gap = _triangulate_two_rays(
                origin1, ray1_world,
                origin2, ray2_world,
            )

            # Step 6: Reject poor geometry
            if gap > self.max_gap_meters:
                return None

            # Step 7: Package result
            # actual_offset is positive: obs1 is later than obs2
            actual_offset = (obs1.timestamp - obs2.timestamp).total_seconds()

            return DepthEstimate(
                timestamp=obs1.timestamp,       # Causal: attributed to the current time
                estimated_depth=depth1,
                true_depth=obs1.true_depth,
                error=depth1 - obs1.true_depth,
                time_offset=time_offset if time_offset is not None else actual_offset,
                triangulation_gap=gap,
                num_observations_used=2,
            )

        except ValueError:
            return None

    def estimate_batch(
        self,
        observations: List[Observation],
        time_offsets: List[float] = DEFAULT_TWO_RAY_DEPTH_TIME_OFFSETS,
        tolerance_seconds: float = PAIRING_TOLERANCE_S,
    ) -> List[DepthEstimate]:
        """
        Estimate depths for all valid observation pairs (causal).

        For each observation obs1 at time t and each time offset Δt,
        finds the observation closest to (t - Δt) in the past and
        triangulates. The result is attributed to t.

        This is strictly causal: no observation ever uses future data.

        Parameters
        ----------
        observations : list of Observation
            All observations. MUST be sorted by timestamp (they
            always are - simulator generates them in order).
        time_offsets : list of float
            Time differences to look back [seconds].
            e.g. [1.0, 5.0, 10.0, 20.0]
            For each obs at time t, we search for a past obs at t - Δt.
        tolerance_seconds : float, optional
            Max allowed difference between target time and actual
            observation time [seconds], default PAIRING_TOLERANCE_S.

        Returns
        -------
        list of DepthEstimate
            All successful depth estimates in one list, each attributed
            to the timestamp of the current (later) observation "obs1".
        """
        if not observations:
            return []

        estimates = []

        # ── KEY OPTIMISATION: build sorted timestamp list for binary search ───────────
        # Extract all timestamps once as a plain list of floats
        # (seconds since first observation). bisect needs a sorted list.
        t0 = observations[0].timestamp
        timestamps_s = [
            (obs.timestamp - t0).total_seconds()
            for obs in observations
        ]
        # timestamps_s[i] = observations[i].timestamp in seconds from t0
        # Already sorted because simulator generates observations in
        # chronological order.
        # ──────────────────────────────────────────────────────────────────────────────

        # ── OUTER LOOP: iterate over every observation (obs1: present observation) ────
        # Each obs1 at time t is treated as the "now" point.
        # We try to pair it with a past observation to produce one estimate
        # per time offset. The final estimate is attributed to obs1.timestamp.
        for i, obs1 in enumerate(observations):
            t1_s = timestamps_s[i]

            # ── INNER LOOP: try each time offset Δt independently ──
            # For a single obs1 at time t, we attempt up to len(time_offsets)
            # pairings: one for each Δt in [1s, 5s, 10s, 20s, ...].
            # Each successful pairing produces one separate DepthEstimate,
            # so one obs1 can contribute multiple estimates to the output.
            for time_offset in time_offsets:
                # Target time in the PAST (causal: t - Δt)
                target_s = t1_s - time_offset

                # Skip if the target time predates the simulation start.
                # This affects early observations where t < Δt_max.
                if target_s < 0:
                    continue

                # ── BINARY SEARCH: find the observation closest to target_s ──
                # O(log n) jump to the right neighbourhood instead of O(n) scan.
                # Returns the index where target_s would slot into timestamps_s.
                # The closest real observation is at idx or idx-1.
                idx = bisect.bisect_left(timestamps_s, target_s)

                # ── CANDIDATE SELECTION: pick the better of the two neighbours ──
                # bisect gives us the insertion point, so we must check both
                # sides (idx-1 and idx) to find whichever is truly closest.
                best_obs2 = None
                best_diff = float('inf')

                for candidate_idx in (idx - 1, idx):
                    if candidate_idx < 0 or candidate_idx >= len(observations):
                        continue
                    # Causality guard: reject any candidate at or after obs1.
                    # Ensures we never use a future observation.
                    if candidate_idx >= i:
                        continue
                    diff = abs(timestamps_s[candidate_idx] - target_s)
                    # Existence check: only accept if within tolerance window.
                    # If no observation falls within PAIRING_TOLERANCE_S of
                    # target_s, best_obs2 stays None and the pairing is skipped.
                    if diff <= tolerance_seconds and diff < best_diff:
                        best_diff = diff
                        best_obs2 = observations[candidate_idx]

                # No valid past observation found for this Δt — skip pairing
                if best_obs2 is None:
                    continue

                # Valid pair found: obs1 (now) + obs2 (past) → triangulate
                estimate = self.estimate_depth(obs1, best_obs2, time_offset)
                if estimate is not None:
                    estimates.append(estimate)

        return estimates

    def __repr__(self) -> str:
        return (
            f"TwoRayTriangulationEstimator("
            f"min_angle={self.min_triangulation_angle_deg}°, "
            f"max_gap={self.max_gap_meters/1e3:.1f}km)"
        )
    

# =============================================================================
# PRIVATE HELPER FUNCTION (Internal use only)
# =============================================================================

def _triangulate_two_rays(
    origin1: npt.NDArray[np.float64],
    direction1: npt.NDArray[np.float64],
    origin2: npt.NDArray[np.float64],
    direction2: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], float, float]:
    """
    Triangulate 3D point from two rays (internal helper).

    Finds the 3D point that minimizes the sum of squared distances
    to both rays using the closest point of approach method.

    Parameters
    ----------
    origin1 : ndarray of shape (3,)
        First ray origin (satellite position 1) [m]
    direction1 : ndarray of shape (3,)
        First ray direction (normalized)
    origin2 : ndarray of shape (3,)
        Second ray origin (satellite position 2) [m]
    direction2 : ndarray of shape (3,)
        Second ray direction (normalized)

    Returns
    -------
    point : ndarray of shape (3,)
        Estimated 3D position (midpoint of closest approach) [m]
    depth1 : float
        Distance from origin1 to point [m]
    gap : float
        Triangulation gap (closest approach distance) [m]

    Raises
    ------
    ValueError
        If rays are parallel (cannot triangulate)
    """
    # Direction vectors (ensure normalized)
    d1 = direction1 / np.linalg.norm(direction1)
    d2 = direction2 / np.linalg.norm(direction2)

    # Vector between origins
    w0 = origin1 - origin2

    # Coefficients for the linear system
    a = np.dot(d1, d1)  # Always 1 if normalized
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)  # Always 1 if normalized
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    # Check for parallel rays
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        raise ValueError("Rays are parallel - cannot triangulate.")

    # Solve for parameters s and t
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    # Closest points on each ray
    P1 = origin1 + s * d1
    P2 = origin2 + t * d2

    # Gap (uncertainty metric: distance between closest points)
    gap = np.linalg.norm(P1 - P2)

    # Estimated target position (midpoint)
    point = (P1 + P2) / 2

    # Depth from first satellite position to estimated point
    depth1 = np.linalg.norm(point - origin1)

    return point, depth1, gap
"""
Experiment results container for multi-run parameter studies.

This module stores and analyzes results across many simulation runs,
enabling comparison across the parameter grid (distance × speed × angle).

Classes
-------
RunSummary
    Statistics from one simulation run
ExperimentResults
    Complete results from a parameter study (all runs)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import pickle

from missile_fly_by_simulation.simulation.results import SimulationResults


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RunSummary:
    """
    Summary statistics from one simulation run.

    Stores only the key statistics (not full trajectories),
    so all 343 summaries take only ~2 MB total.

    Attributes
    ----------
    closest_approach_distance : float
        Closest approach distance [m]
    missile_speed : float
        Missile speed [m/s]
    crossing_angle_deg : float
        Crossing angle [degrees]
    num_observations : int
        Number of camera observations generated
    runtime_seconds : float
        Simulation runtime [seconds]
    stats : dict
        Per-method statistics
        Keys: method names ('two_ray', 'multi_ray', 'kalman')
        Values: dict with rmse, mean_error, std_error, etc.
    """
    closest_approach_distance: float
    missile_speed: float
    crossing_angle_deg: float
    num_observations: int
    runtime_seconds: float
    stats: Dict[str, dict]
    elevation_angle_deg: float = 0.0

    @property
    def distance_km(self) -> float:
        """Closest approach distance in km."""
        return self.closest_approach_distance / 1e3

    @property
    def label(self) -> str:
        """Human-readable label."""
        return (
            f"dist{int(self.distance_km)}km_"
            f"speed{int(self.missile_speed)}ms_"
            f"angle{int(self.crossing_angle_deg)}deg"
        )

    def rmse(self, method: str) -> float:
        """
        Get RMSE for a specific method.

        Parameters
        ----------
        method : str
            Method name

        Returns
        -------
        float
            RMSE [m], or NaN if method not available
        """
        if method not in self.stats:
            return np.nan
        return self.stats[method].get('rmse', np.nan)

    def __repr__(self) -> str:
        return (
            f"RunSummary("
            f"dist={self.distance_km:.0f}km, "
            f"speed={self.missile_speed:.0f}m/s, "
            f"angle={self.crossing_angle_deg:.0f}°, "
            f"obs={self.num_observations})"
        )


# =============================================================================
# MAIN EXPERIMENT RESULTS CLASS
# =============================================================================

@dataclass
class ExperimentResults:
    """
    Complete results from a DSA parameter study (Distance × Speed × Angle).

    Stores:
    - Summary statistics for ALL runs (small, ~2 MB for 343 runs)
    - Full SimulationResults for SELECTED runs only (large, ~15 MB each)
    - Metadata about the study

    Attributes
    ----------
    summaries : dict
        Keys: (distance, speed, angle) 3-tuples
        Values: RunSummary objects
        Contains ALL runs
    full_results : dict
        Keys: (distance, speed, angle) 3-tuples
        Values: SimulationResults objects
        Contains ONLY selected runs (to save disk space)
    distances : list of float
        All distances in parameter grid [m]
    speeds : list of float
        All speeds in parameter grid [m/s]
    crossing_angles : list of float
        All crossing angles in parameter grid [degrees]
    metadata : dict
        Study metadata (date, runtime, etc.)

    Examples
    --------
    >>> # Load results
    >>> results = ExperimentResults.load('experiments/runs/study_001/experiment_results.pkl')
    >>>
    >>> # Get RMSE matrix for one angle slice
    >>> rmse_matrix = results.get_rmse_matrix('two_ray', crossing_angle=90.0)
    >>>
    >>> # Get best scenario
    >>> best = results.get_best_scenario('two_ray')
    >>> print(f"Best: {best[0]/1e3:.0f}km, {best[1]:.0f}m/s, {best[2]:.0f}°")
    """

    summaries: Dict[Tuple[float, float, float], RunSummary]
    full_results: Dict[Tuple[float, float, float], SimulationResults]
    distances: List[float]
    speeds: List[float]
    crossing_angles: List[float]
    metadata: dict

    # =========================================================================
    # MATRIX EXTRACTION (for heatmaps)
    # =========================================================================

    def get_rmse_matrix(
        self,
        method: str,
        crossing_angle: float,
    ) -> npt.NDArray[np.float64]:
        """
        Extract RMSE values as 2D matrix for heatmap plotting.

        Returns one distance × speed slice at the given crossing angle.

        Parameters
        ----------
        method : str
            Method name ('two_ray', 'multi_ray', 'kalman')
        crossing_angle : float
            Crossing angle slice to extract [degrees]

        Returns
        -------
        matrix : ndarray of shape (num_distances, num_speeds)
            RMSE values [m]
            Rows: distances (sorted ascending)
            Columns: speeds (sorted ascending)

        Notes
        -----
        Row 0 = smallest distance (closest flyby)
        Col 0 = slowest missile

        Examples
        --------
        >>> matrix = results.get_rmse_matrix('two_ray', crossing_angle=90.0)
        >>> print(matrix.shape)
        (7, 7)
        """
        sorted_distances = sorted(self.distances)
        sorted_speeds = sorted(self.speeds)

        matrix = np.full(
            (len(sorted_distances), len(sorted_speeds)),
            fill_value=np.nan
        )

        for i, dist in enumerate(sorted_distances):
            for j, speed in enumerate(sorted_speeds):
                key = (dist, speed, crossing_angle)
                if key in self.summaries:
                    matrix[i, j] = self.summaries[key].rmse(method)

        return matrix

    def get_improvement_matrix(
        self,
        method_improved: str,
        method_baseline: str = 'two_ray',
        crossing_angle: float = 90.0,
    ) -> npt.NDArray[np.float64]:
        """
        Compute improvement matrix: % improvement of one method over baseline.

        Parameters
        ----------
        method_improved : str
            Method to compare against baseline
        method_baseline : str, optional
            Baseline method, default 'two_ray'
        crossing_angle : float, optional
            Crossing angle slice [degrees], default 90.0

        Returns
        -------
        matrix : ndarray of shape (num_distances, num_speeds)
            Improvement values [%]
            Positive = improved over baseline
            Negative = worse than baseline

        Examples
        --------
        >>> improvement = results.get_improvement_matrix('multi_ray', 'two_ray', 90.0)
        >>> # Value of 30 means multi_ray is 30% better than two_ray
        """
        rmse_baseline = self.get_rmse_matrix(method_baseline, crossing_angle)
        rmse_improved = self.get_rmse_matrix(method_improved, crossing_angle)

        # % improvement: positive means improved
        improvement = (rmse_baseline - rmse_improved) / rmse_baseline * 100

        return improvement

    def get_rmse_vs_angle(
        self,
        method: str,
        distance: float,
        speed: float,
    ) -> dict:
        """
        Extract RMSE as a function of crossing angle at fixed distance and speed.

        Used for the 1D angle sweep where distances and speeds each have
        exactly one value.

        Parameters
        ----------
        method : str
            Method name ('two_ray', 'multi_ray', 'kalman', ...)
        distance : float
            Fixed closest approach distance [m]
        speed : float
            Fixed missile speed [m/s]

        Returns
        -------
        dict
            Keys: crossing angle [degrees] (float)
            Values: RMSE [m] (float, or np.nan if missing)
        """
        result = {}
        for angle in sorted(self.crossing_angles):
            key = (distance, speed, angle)
            if key in self.summaries:
                result[angle] = self.summaries[key].rmse(method)
            else:
                result[angle] = np.nan
        return result

    def get_bias_vs_angle(
        self,
        method: str,
        distance: float,
        speed: float,
    ) -> dict:
        """
        Extract mean error (bias) as a function of crossing angle.

        Parameters
        ----------
        method : str
            Method name
        distance : float
            Fixed closest approach distance [m]
        speed : float
            Fixed missile speed [m/s]

        Returns
        -------
        dict
            Keys: crossing angle [degrees] (float)
            Values: mean error [m] (float, or np.nan if missing)
        """
        result = {}
        for angle in sorted(self.crossing_angles):
            key = (distance, speed, angle)
            if key in self.summaries:
                s = self.summaries[key]
                result[angle] = s.stats.get(method, {}).get('mean_error', np.nan)
            else:
                result[angle] = np.nan
        return result

    def get_num_estimates_vs_angle(
        self,
        method: str,
        distance: float,
        speed: float,
    ) -> dict:
        """
        Extract number of valid estimates as a function of crossing angle.

        Parameters
        ----------
        method : str
            Method name
        distance : float
            Fixed closest approach distance [m]
        speed : float
            Fixed missile speed [m/s]

        Returns
        -------
        dict
            Keys: crossing angle [degrees] (float)
            Values: number of estimates (int, or 0 if missing)
        """
        result = {}
        for angle in sorted(self.crossing_angles):
            key = (distance, speed, angle)
            if key in self.summaries:
                s = self.summaries[key]
                result[angle] = s.stats.get(method, {}).get('num_estimates', 0)
            else:
                result[angle] = 0
        return result

    # =========================================================================
    # AXIS LABELS (for plots)
    # =========================================================================

    def distance_labels(self) -> List[str]:
        """
        Human-readable distance labels for plot axes.

        Returns
        -------
        list of str
            e.g. ['50 km', '100 km', '200 km', ...]
        """
        return [f"{d/1e3:.0f} km" for d in sorted(self.distances)]

    def speed_labels(self) -> List[str]:
        """
        Human-readable speed labels for plot axes.

        Returns
        -------
        list of str
            e.g. ['300 m/s', '500 m/s', '1000 m/s', ...]
        """
        return [f"{s:.0f} m/s" for s in sorted(self.speeds)]

    @property
    def crossing_angle_labels(self) -> List[str]:
        """
        Human-readable crossing angle labels for plot axes.

        Returns
        -------
        list of str
            e.g. ['0°', '30°', '60°', '90°']
        """
        return [f"{a:.0f}°" for a in sorted(self.crossing_angles)]

    # =========================================================================
    # BEST / WORST SCENARIOS
    # =========================================================================

    def get_best_scenario(
        self,
        method: str = 'two_ray'
    ) -> Tuple[float, float, float]:
        """
        Get (distance, speed, angle) combination with lowest RMSE.

        Parameters
        ----------
        method : str
            Method to evaluate

        Returns
        -------
        tuple of (distance, speed, angle)
            Parameters of best scenario
        """
        best_key = min(
            self.summaries.keys(),
            key=lambda k: self.summaries[k].rmse(method)
        )
        return best_key

    def get_worst_scenario(
        self,
        method: str = 'two_ray'
    ) -> Tuple[float, float, float]:
        """
        Get (distance, speed, angle) combination with highest RMSE.

        Parameters
        ----------
        method : str
            Method to evaluate

        Returns
        -------
        tuple of (distance, speed, angle)
            Parameters of worst scenario
        """
        worst_key = max(
            self.summaries.keys(),
            key=lambda k: self.summaries[k].rmse(method)
        )
        return worst_key

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================

    def to_dataframe(self):
        """
        Convert all run summaries to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per run, columns for all statistics

        Examples
        --------
        >>> df = results.to_dataframe()
        >>> df.to_csv('summary_table.csv', index=False)
        >>> print(df.describe())
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")

        rows = []

        for (dist, speed, angle), summary in self.summaries.items():
            row = {
                'distance_m': dist,
                'distance_km': dist / 1e3,
                'speed_ms': speed,
                'crossing_angle_deg': angle,
                'num_observations': summary.num_observations,
                'runtime_seconds': summary.runtime_seconds,
            }

            # Add stats for each method
            for method, stats in summary.stats.items():
                for stat_name, value in stats.items():
                    row[f'{method}_{stat_name}'] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by angle, then distance, then speed
        df = df.sort_values(
            ['crossing_angle_deg', 'distance_m', 'speed_ms']
        ).reset_index(drop=True)

        return df

    def summary(self) -> str:
        """
        Human-readable summary of experiment.

        Returns
        -------
        str
            Multi-line summary
        """
        lines = [
            "",
            "=" * 60,
            "Experiment Results Summary (DSA)",
            "=" * 60,
            "",
            f"Parameter Grid:",
            f"  Distances: {[f'{d/1e3:.0f}km' for d in sorted(self.distances)]}",
            f"  Speeds:    {[f'{s:.0f}m/s' for s in sorted(self.speeds)]}",
            f"  Angles:    {[f'{a:.0f}°' for a in sorted(self.crossing_angles)]}",
            f"  Total runs: {len(self.summaries)}",
            "",
            f"Full results stored for {len(self.full_results)} runs",
            "",
        ]

        # Best/worst for each method
        available_methods = set()
        for summary in self.summaries.values():
            available_methods.update(summary.stats.keys())

        for method in sorted(available_methods):
            best = self.get_best_scenario(method)
            worst = self.get_worst_scenario(method)

            best_rmse = self.summaries[best].rmse(method)
            worst_rmse = self.summaries[worst].rmse(method)

            lines.extend([
                f"{method.upper()}:",
                f"  Best:  dist={best[0]/1e3:.0f}km, speed={best[1]:.0f}m/s, "
                f"angle={best[2]:.0f}deg -> RMSE={best_rmse:.1f}m",
                f"  Worst: dist={worst[0]/1e3:.0f}km, speed={worst[1]:.0f}m/s, "
                f"angle={worst[2]:.0f}deg -> RMSE={worst_rmse:.1f}m",
                "",
            ])

        if 'run_date' in self.metadata:
            lines.append(f"Run date: {self.metadata['run_date']}")
        if 'total_runtime_seconds' in self.metadata:
            total = self.metadata['total_runtime_seconds']
            lines.append(f"Total runtime: {total/60:.1f} minutes")

        lines.append("=" * 60)

        return "\n".join(lines)

    # =========================================================================
    # I/O
    # =========================================================================

    def save(self, filepath: str):
        """
        Save experiment results to file.

        Parameters
        ----------
        filepath : str
            Path to save file (.pkl recommended)
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] Saved experiment results to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ExperimentResults':
        """
        Load experiment results from file.

        Parameters
        ----------
        filepath : str
            Path to saved file

        Returns
        -------
        ExperimentResults
            Loaded results
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print(f"[OK] Loaded experiment results from {filepath}")
        return results

    def __repr__(self) -> str:
        return (
            f"ExperimentResults("
            f"{len(self.distances)}×{len(self.speeds)}×{len(self.crossing_angles)} grid, "
            f"{len(self.summaries)} runs, "
            f"{len(self.full_results)} full results)"
        )


# =============================================================================
# ANGULAR STUDY RESULTS (2D: Azimuth × Elevation)
# =============================================================================

@dataclass
class AngularStudyResults:
    """
    Results from a 2D angular parameter study (azimuth × elevation).

    Fixed: distance, speed.
    Variable: azimuth (crossing angle in horizontal plane) ×
              elevation (tilt out of orbital plane toward radial).

    Keys are (azimuth_deg, elevation_deg) 2-tuples.

    Attributes
    ----------
    summaries : dict
        Keys: (azimuth_deg, elevation_deg) 2-tuples
        Values: RunSummary objects
    full_results : dict
        Keys: (azimuth_deg, elevation_deg) 2-tuples
        Values: SimulationResults objects
    azimuths : list of float
        Azimuth angles in grid [degrees]
    elevations : list of float
        Elevation angles in grid [degrees]
    distance : float or None
        Fixed closest approach distance [m], or None for ground-launch scenarios
        where distance is determined by satellite altitude.
    speed : float
        Fixed missile speed [m/s]
    metadata : dict
        Study metadata
    """
    summaries:    Dict[Tuple[float, float], RunSummary]
    full_results: Dict[Tuple[float, float], 'SimulationResults']
    azimuths:     List[float]
    elevations:   List[float]
    speed:        float
    metadata:     dict
    distance:     float = None
    sampling:     str   = 'grid'   # 'grid' or 'fibonacci'

    def get_rmse_matrix(self, method: str) -> npt.NDArray[np.float64]:
        """
        Extract RMSE as 2D matrix for heatmap plotting.

        Returns
        -------
        matrix : ndarray of shape (len(azimuths), len(elevations))
            RMSE [m]. Rows = azimuths (ascending), Cols = elevations (ascending).
        """
        sorted_az = sorted(self.azimuths)
        sorted_el = sorted(self.elevations)
        matrix = np.full((len(sorted_az), len(sorted_el)), np.nan)
        for i, az in enumerate(sorted_az):
            for j, el in enumerate(sorted_el):
                summary = self.summaries.get((az, el))
                if summary is not None:
                    matrix[i, j] = summary.rmse(method)
        return matrix

    def available_methods(self) -> List[str]:
        """Return list of estimation methods present in results."""
        methods: set = set()
        for summary in self.summaries.values():
            methods.update(summary.stats.keys())
        return sorted(methods)

    def save(self, filepath: str):
        """Save to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] Saved angular study results to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'AngularStudyResults':
        """Load from pickle file."""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print(f"[OK] Loaded angular study results from {filepath}")
        return results

    def __repr__(self) -> str:
        return (
            f"AngularStudyResults("
            f"{len(self.azimuths)}×{len(self.elevations)} grid, "
            f"dist={self.distance/1e3:.0f}km, speed={self.speed:.0f}m/s, "
            f"{len(self.summaries)} runs)"
        )

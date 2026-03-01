"""
Simulation results packaging and analysis.

This module defines the output data structures and provides
convenient methods for analyzing simulation results.

Classes
-------
Observation
    Single camera observation (pixel + metadata)
DepthEstimate
    Single depth estimate (from any method)
SimulationResults
    Complete package of all simulation outputs
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt
import pickle

from missile_fly_by_simulation.domain import Satellite, SatelliteState, Missile


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Observation:
    """
    Single camera observation of missile.
    
    Records one detection: when, where (pixel), and ground truth.
    
    Attributes
    ----------
    timestamp : datetime
        Time of observation
    satellite_state : SatelliteState
        Satellite position/velocity/attitude at this time
    pixel : tuple of (float, float)
        Detected pixel coordinates (u, v)
    true_position : ndarray of shape (3,)
        Ground truth missile position [m] (for validation)
    true_depth : float
        Ground truth distance satellite→missile [m] (for validation)
    """
    timestamp: datetime
    satellite_state: SatelliteState
    pixel: Tuple[float, float]
    true_position: npt.NDArray[np.float64]
    true_depth: float


@dataclass
class DepthEstimate:
    """
    Single depth estimate from any estimation method.
    
    Attributes
    ----------
    timestamp : datetime
        Time of first observation used
    estimated_depth : float
        Estimated distance [m]
    true_depth : float
        Ground truth distance [m] (for validation)
    error : float
        Estimation error (estimated - true) [m]
    time_offset : float, optional
        Time between observations used [seconds]
    triangulation_gap : float, optional
        Closest approach distance between rays [m] (uncertainty metric)
    num_observations_used : int, optional
        Number of observations used in estimate
    """
    timestamp: datetime
    estimated_depth: float
    true_depth: float
    error: float
    time_offset: Optional[float] = None
    triangulation_gap: Optional[float] = None
    num_observations_used: Optional[int] = None


# =============================================================================
# MAIN RESULTS CLASS
# =============================================================================

@dataclass
class SimulationResults:
    """
    Complete output package from simulation run.
    
    This bundles all simulation outputs:
    - Trajectories (satellite, missile)
    - Observations (camera detections)
    - Depth estimates (from various methods)
    - Metadata (runtime, configuration)
    
    Attributes
    ----------
    satellite : Satellite
        Satellite entity with complete trajectory
    missile : Missile
        Missile entity with complete trajectory
    observations : list of Observation
        All camera observations
    depth_estimates : dict
        Depth estimates from different methods
        Keys: method name ("two_ray", "multi_ray", etc.)
        Values: list of DepthEstimate objects
    scenario : SimulationScenario
        Input configuration used
    metadata : dict
        Simulation metadata (runtime, versions, etc.)
        
    Examples
    --------
    >>> results = simulator.run()
    >>> print(results.summary())
    >>> df = results.to_dataframe()
    >>> stats = results.get_statistics("two_ray")
    >>> results.save("simulation_001.pkl")
    """
    satellite: Satellite
    missile: Missile
    observations: List[Observation]
    depth_estimates: Dict[str, List[DepthEstimate]]
    scenario: 'SimulationScenario'  # Forward reference
    metadata: dict
    
    # =========================================================================
    # SUMMARY AND OVERVIEW
    # =========================================================================
    
    def summary(self) -> str:
        """
        Human-readable summary of results.
        
        Returns
        -------
        str
            Multi-line summary
        """
        lines = [
            "",
            "=" * 60,
            "Simulation Results Summary",
            "=" * 60,
            "",
            f"Satellite: {self.satellite.spec.name}",
            f"  States: {self.satellite.num_states}",
        ]
        
        if self.satellite.duration:
            lines.append(f"  Duration: {self.satellite.duration:.1f} seconds")
        
        lines.extend([
            "",
            f"Missile: {self.missile.name}",
            f"  States: {self.missile.num_states}",
        ])
        
        if self.missile.duration:
            lines.append(f"  Duration: {self.missile.duration:.1f} seconds")
        
        lines.extend([
            "",
            f"Observations: {len(self.observations)}",
        ])
        
        if self.observations:
            lines.append(f"  First: {self.observations[0].timestamp}")
            lines.append(f"  Last: {self.observations[-1].timestamp}")
        
        lines.extend([
            "",
            "Depth Estimates:",
        ])
        
        for method, estimates in self.depth_estimates.items():
            lines.append(f"  {method}: {len(estimates)} estimates")
        
        if 'runtime_seconds' in self.metadata:
            lines.extend([
                "",
                f"Runtime: {self.metadata['runtime_seconds']:.1f} seconds",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    # =========================================================================
    # DATA CONVERSION
    # =========================================================================
    
    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            One row per observation with all data
            
        Notes
        -----
        Requires pandas. If not installed, raises ImportError.
        
        Columns include:
        - timestamp, frame_index
        - satellite_position_x/y/z, satellite_velocity_x/y/z
        - pixel_u, pixel_v
        - true_depth
        - depth_estimate_{method}, depth_error_{method} for each method
        
        Examples
        --------
        >>> df = results.to_dataframe()
        >>> df.describe()
        >>> df.to_csv('results.csv')
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )
        
        rows = []
        
        for i, obs in enumerate(self.observations):
            row = {
                'frame_index': i,
                'timestamp': obs.timestamp,
                
                # Satellite data
                'satellite_position_x': obs.satellite_state.position[0],
                'satellite_position_y': obs.satellite_state.position[1],
                'satellite_position_z': obs.satellite_state.position[2],
                'satellite_velocity_x': obs.satellite_state.velocity[0],
                'satellite_velocity_y': obs.satellite_state.velocity[1],
                'satellite_velocity_z': obs.satellite_state.velocity[2],
                'satellite_altitude': obs.satellite_state.altitude,
                
                # Observation data
                'pixel_u': obs.pixel[0],
                'pixel_v': obs.pixel[1],
                'true_depth': obs.true_depth,
                
                # True missile position
                'missile_position_x': obs.true_position[0],
                'missile_position_y': obs.true_position[1],
                'missile_position_z': obs.true_position[2],
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add depth estimates (align by timestamp)
        for method, estimates in self.depth_estimates.items():
            # Create temporary DataFrame for this method
            est_rows = []
            for est in estimates:
                est_row = {
                    'timestamp': est.timestamp,
                    f'depth_estimate_{method}': est.estimated_depth,
                    f'depth_error_{method}': est.error,
                }
                
                if est.triangulation_gap is not None:
                    est_row[f'triangulation_gap_{method}'] = est.triangulation_gap
                
                if est.time_offset is not None:
                    est_row[f'time_offset_{method}'] = est.time_offset
                
                est_rows.append(est_row)
            
            est_df = pd.DataFrame(est_rows)
            
            # Merge with main DataFrame on timestamp
            df = df.merge(est_df, on='timestamp', how='left')
        
        return df
    
    # =========================================================================
    # STATISTICS AND ANALYSIS
    # =========================================================================
    
    def get_errors_for_method(self, method: str) -> npt.NDArray[np.float64]:
        """
        Get depth errors for a specific method.
        
        Parameters
        ----------
        method : str
            Method name (e.g., "two_ray", "multi_ray")
            
        Returns
        -------
        errors : ndarray
            Array of errors [m]
            
        Raises
        ------
        ValueError
            If method not found
            
        Examples
        --------
        >>> errors = results.get_errors_for_method("two_ray")
        >>> print(f"RMSE: {np.sqrt(np.mean(errors**2)):.2f} m")
        """
        if method not in self.depth_estimates:
            available = list(self.depth_estimates.keys())
            raise ValueError(
                f"Method '{method}' not found. Available: {available}"
            )
        
        return np.array([est.error for est in self.depth_estimates[method]])
    
    def get_statistics(self, method: str) -> dict:
        """
        Compute statistical summary for a method.
        
        Parameters
        ----------
        method : str
            Estimation method name
            
        Returns
        -------
        stats : dict
            Statistics including RMSE, mean, std, percentiles
            
        Examples
        --------
        >>> stats = results.get_statistics("two_ray")
        >>> print(f"RMSE: {stats['rmse']:.2f} m")
        >>> print(f"95th percentile: {stats['error_95th']:.2f} m")
        """
        errors = self.get_errors_for_method(method)
        
        if len(errors) == 0:
            return {
                'num_estimates': 0,
                'rmse': np.nan,
                'mean_error': np.nan,
                'median_error': np.nan,
                'std_error': np.nan,
                'min_error': np.nan,
                'max_error': np.nan,
            }
        
        return {
            'num_estimates': len(errors),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'error_25th': np.percentile(errors, 25),
            'error_75th': np.percentile(errors, 75),
            'error_95th': np.percentile(errors, 95),
            'mae': np.mean(np.abs(errors)),  # Mean absolute error
        }
    
    def compare_methods(self):
        """
        Compare all estimation methods side-by-side.
        
        Returns
        -------
        pd.DataFrame
            One row per method, columns for each statistic
            
        Examples
        --------
        >>> comparison = results.compare_methods()
        >>> print(comparison)
                      num_estimates   rmse  mean_error  ...
        two_ray              5500   45.23       -2.34  ...
        multi_ray             580   32.15        1.12  ...
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for compare_methods()")
        
        data = {}
        for method in self.depth_estimates.keys():
            data[method] = self.get_statistics(method)
        
        return pd.DataFrame(data).T
    
    # =========================================================================
    # FILTERING
    # =========================================================================
    
    def filter_by_time_range(
        self,
        start: datetime,
        end: datetime
    ) -> 'SimulationResults':
        """
        Create filtered results for a time range.
        
        Parameters
        ----------
        start : datetime
            Start of time range
        end : datetime
            End of time range
            
        Returns
        -------
        SimulationResults
            New results object with filtered data
            
        Examples
        --------
        >>> # Analyze first minute only
        >>> start = results.observations[0].timestamp
        >>> end = start + timedelta(seconds=60)
        >>> results_minute = results.filter_by_time_range(start, end)
        """
        # Filter observations
        filtered_obs = [
            obs for obs in self.observations
            if start <= obs.timestamp <= end
        ]
        
        # Filter depth estimates
        filtered_estimates = {}
        for method, estimates in self.depth_estimates.items():
            filtered_estimates[method] = [
                est for est in estimates
                if start <= est.timestamp <= end
            ]
        
        # Note: Keep full satellite/missile trajectories
        # (could filter these too if needed)
        
        return SimulationResults(
            satellite=self.satellite,
            missile=self.missile,
            observations=filtered_obs,
            depth_estimates=filtered_estimates,
            scenario=self.scenario,
            metadata={**self.metadata, 'filtered': True, 'filter_start': start, 'filter_end': end}
        )
    
    # =========================================================================
    # I/O (SAVE/LOAD)
    # =========================================================================
    
    def save(self, filepath: str):
        """
        Save results to file.
        
        Parameters
        ----------
        filepath : str
            Path to save file (.pkl extension recommended)
            
        Examples
        --------
        >>> results.save('simulation_run_001.pkl')
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] Saved results to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'SimulationResults':
        """
        Load results from file.
        
        Parameters
        ----------
        filepath : str
            Path to saved results file
            
        Returns
        -------
        SimulationResults
            Loaded results object
            
        Examples
        --------
        >>> results = SimulationResults.load('simulation_run_001.pkl')
        >>> print(results.summary())
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print(f"[OK] Loaded results from {filepath}")
        return results
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    @property
    def num_observations(self) -> int:
        """Number of camera observations."""
        return len(self.observations)
    
    @property
    def num_depth_estimates(self) -> int:
        """Total number of depth estimates (all methods)."""
        return sum(len(estimates) for estimates in self.depth_estimates.values())
    
    @property
    def available_methods(self) -> List[str]:
        """List of available estimation methods."""
        return list(self.depth_estimates.keys())
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SimulationResults("
            f"satellite='{self.satellite.spec.name}', "
            f"observations={self.num_observations}, "
            f"methods={self.available_methods})"
        )
"""
Batch runner for parameter study simulations.

This module orchestrates running many simulations systematically,
managing folder structure, saving results, and reporting progress.

Classes
-------
BatchRunner
    Runs the full parameter study and manages all outputs
"""

import os
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from missile_fly_by_simulation.simulation.simulator import Simulator
from missile_fly_by_simulation.experiments.scenario_factory import ScenarioFactory
from missile_fly_by_simulation.experiments.experiment_results import (
    ExperimentResults,
    AngularStudyResults,
    RunSummary,
)
from missile_fly_by_simulation.constants import (
    STUDY_DISTANCES_M,
    STUDY_SPEEDS_MS,
    STUDY_DISTANCES_QUICK_M,
    STUDY_SPEEDS_QUICK_MS,
    DEFAULT_CROSSING_ANGLE_DEG,
    DEFAULT_DURATION_S,
)


# =============================================================================
# HEARTBEAT HELPER
# =============================================================================

_STATUS_WIDTH = 90  # character width for clearing the in-place status line


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}min"


def _heartbeat_printer(
    stop_event: threading.Event,
    completed_ref: list,
    total: int,
    interval_s: float = 10.0,
    print_lock: threading.Lock = None,
) -> None:
    """Background thread: overwrites a single status line in-place every interval_s seconds."""
    start = time.time()
    lock = print_lock if print_lock is not None else threading.Lock()

    while not stop_event.wait(timeout=interval_s):
        elapsed = time.time() - start
        done = completed_ref[0]
        remaining = total - done
        eta_str = (
            _format_duration((elapsed / done) * remaining)
            if done > 0 else "?"
        )
        status = (
            f"  \u27b3 [{done:3d}/{total}]  {remaining} running"
            f"  |  elapsed: {_format_duration(elapsed)}"
            f"  |  ETA: ~{eta_str}"
        )
        with lock:
            print(f"\r{status:<{_STATUS_WIDTH}}", end='', flush=True)


# =============================================================================
# MODULE-LEVEL WORKER
# (must be module-level, NOT a class method, for ProcessPoolExecutor pickling)
# =============================================================================

def _worker(task: dict) -> tuple:
    """
    Run one simulation in a subprocess.

    Must be a module-level function (not a class method) so that
    ProcessPoolExecutor can pickle it for inter-process communication
    on Windows and macOS (spawn start method).

    Parameters
    ----------
    task : dict
        Keys: satellite_spec, duration, detection_fraction,
              depth_time_offsets, angle, distance, speed,
              save_full, study_dir

    Returns
    -------
    tuple of (key, RunSummary, sim_results_or_None)
        key = (distance, speed, angle) 3-tuple
    """
    import time as _time
    import os as _os
    import sys as _sys
    from missile_fly_by_simulation.experiments.scenario_factory import ScenarioFactory
    from missile_fly_by_simulation.experiments.experiment_results import RunSummary
    from missile_fly_by_simulation.simulation.simulator import Simulator

    # Suppress all stdout from worker — parallel workers printing simultaneously
    # produces garbled output. The main process prints the clean [X/Y] summary line.
    _devnull = open(_os.devnull, 'w')
    _old_stdout = _sys.stdout
    _sys.stdout = _devnull

    try:
        run_start = _time.time()

        factory = ScenarioFactory(
            satellite_spec=task['satellite_spec'],
            simulation_duration=task['duration'],
            detection_fraction=task['detection_fraction'],
            depth_time_offsets=task['depth_time_offsets'],
            crossing_angle_deg=task['angle'],
        )

        scenario = factory.create_flyby_scenario(
            closest_approach_distance=task['distance'],
            missile_speed=task['speed'],
        )

        simulator = Simulator(scenario)
        sim_results = simulator.run(show_progress=False)

        runtime = _time.time() - run_start

        stats = {}
        for method in sim_results.available_methods:
            stats[method] = sim_results.get_statistics(method)

        summary = RunSummary(
            closest_approach_distance=task['distance'],
            missile_speed=task['speed'],
            crossing_angle_deg=task['angle'],
            num_observations=sim_results.num_observations,
            runtime_seconds=runtime,
            stats=stats,
        )

        full_result_to_return = None
        if task['save_full']:
            label = (
                f"dist{int(task['distance'] / 1e3)}km_"
                f"speed{int(task['speed'])}ms_"
                f"angle{int(task['angle'])}deg"
            )
            full_results_dir = _os.path.join(task['study_dir'], 'full_results')
            _os.makedirs(full_results_dir, exist_ok=True)
            full_results_path = _os.path.join(full_results_dir, f"{label}.pkl")
            sim_results.save(full_results_path)
            full_result_to_return = sim_results

        key = (task['distance'], task['speed'], task['angle'])
        return key, summary, full_result_to_return

    finally:
        _sys.stdout = _old_stdout
        _devnull.close()


# =============================================================================
# BATCH RUNNER CLASS
# =============================================================================

class BatchRunner:
    """
    Orchestrates the full DSA parameter study (Distance × Speed × Angle).

    Runs all (distance × speed × crossing_angle) combinations in parallel
    using ProcessPoolExecutor, manages the output folder structure, saves
    results selectively, and reports progress throughout.

    Output folder structure created automatically:
        output_dir/
        └── study_XXX/
            ├── experiment_results.pkl    ← All summaries + selected full results
            ├── summary_table.csv         ← Human-readable table
            ├── plots/
            │   ├── parameter_study/      ← Heatmaps (per angle subfolder)
            │   │   ├── angle_0deg/
            │   │   ├── angle_30deg/
            │   │   └── angle_90deg/
            │   └── individual_runs/      ← Per-run plots (selected runs)
            │       ├── dist100km_speed1000ms_angle90deg/
            │       └── dist500km_speed3000ms_angle45deg/
            └── full_results/             ← Full pkl for selected runs
                ├── dist100km_speed1000ms_angle90deg.pkl
                └── dist500km_speed3000ms_angle45deg.pkl

    Attributes
    ----------
    factory : ScenarioFactory
        Creates scenarios from physical parameters
    output_dir : str
        Root directory for all experiment outputs
    save_full_results_for : list of 3-tuples, optional
        Which (distance, speed, angle) combinations to save full results for.
        If None: automatically selects representative runs (3D cube corners + center)

    Examples
    --------
    >>> factory = ScenarioFactory(
    ...     satellite_spec=ScenarioFactory.default_satellite_spec(),
    ...     simulation_duration=1200.0
    ... )
    >>>
    >>> runner = BatchRunner(factory=factory, output_dir='experiments/runs')
    >>>
    >>> experiment_results = runner.run_parameter_study_dsa(
    ...     distances=[100e3, 200e3, 500e3],
    ...     speeds=[500.0, 1000.0, 3000.0],
    ...     crossing_angles=[0.0, 30.0, 60.0, 90.0],
    ... )
    >>>
    >>> print(experiment_results.summary())
    """

    def __init__(
        self,
        factory: ScenarioFactory,
        output_dir: str = 'experiments/runs',
        save_full_results_for: Optional[List[Tuple[float, float, float]]] = None,
    ):
        """
        Initialize batch runner.

        Parameters
        ----------
        factory : ScenarioFactory
            Scenario factory for creating scenarios
        output_dir : str, optional
            Root directory for outputs, default 'experiments/runs'
        save_full_results_for : list of (distance, speed, angle) tuples, optional
            Which runs to save complete SimulationResults for.
            If None: automatically picks representative runs
            (3D cube corners + center of parameter grid)
        """
        self.factory = factory
        self.output_dir = output_dir
        self.save_full_results_for = save_full_results_for

    # =========================================================================
    # MAIN ENTRY POINT — 3D PARALLEL STUDY
    # =========================================================================

    def run_parameter_study_dsa(
        self,
        distances: List[float],
        speeds: List[float],
        crossing_angles: List[float],
        study_name: Optional[str] = None,
        show_progress: bool = True,
    ) -> ExperimentResults:
        """
        Run full parameter study across distance × speed × crossing_angle grid.

        Uses ProcessPoolExecutor to run simulations in parallel (N_cpu - 1 workers).

        Parameters
        ----------
        distances : list of float
            Closest approach distances [m]
            e.g. [50e3, 75e3, 100e3, 150e3, 200e3, 500e3, 1000e3]
        speeds : list of float
            Missile speeds [m/s]
            e.g. [300, 500, 750, 1000, 2000, 3000, 7000]
        crossing_angles : list of float
            Crossing angles [degrees]
            e.g. [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        study_name : str, optional
            Name for this study run, default auto-generated 'study_001'
        show_progress : bool, optional
            Print progress for each completed run, default True

        Returns
        -------
        ExperimentResults
            Complete results with summaries for all runs.
            Keys are (distance, speed, angle) 3-tuples.

        Notes
        -----
        Full SimulationResults are only saved for selected runs
        (3D cube corners + center by default) to save disk space.
        Summary statistics are saved for ALL runs.
        """
        study_start = time.time()

        # Create study folder
        study_dir = self._create_study_directory(study_name)

        if show_progress:
            self._print_study_header(distances, speeds, crossing_angles, study_dir)

        # Determine which runs to save full results for
        save_full_for: Set[Tuple[float, float, float]] = set(
            self.save_full_results_for
            or self._select_representative_runs_dsa(distances, speeds, crossing_angles)
        )

        # Build flat task list (all distance × speed × angle combinations)
        tasks = []
        for distance in sorted(distances):
            for speed in sorted(speeds):
                for angle in sorted(crossing_angles):
                    key = (distance, speed, angle)
                    tasks.append({
                        'satellite_spec':    self.factory.satellite_spec,
                        'duration':          self.factory.simulation_duration,
                        'detection_fraction': self.factory.detection_fraction,
                        'depth_time_offsets': self.factory.depth_time_offsets,
                        'angle':   angle,
                        'distance': distance,
                        'speed':    speed,
                        'save_full': key in save_full_for,
                        'study_dir': study_dir,
                    })

        total_runs = len(tasks)
        summaries: Dict = {}
        full_results: Dict = {}
        failed_runs = []
        completed = 0

        n_workers = max(1, (os.cpu_count() or 1) - 1)
        if show_progress:
            print(
                f"Launching {total_runs} runs on "
                f"{n_workers} parallel worker(s) (CPU count: {os.cpu_count()})\n"
            )

        _completed_ref = [0]
        _stop_event    = threading.Event()
        _print_lock    = threading.Lock()
        _hb = threading.Thread(
            target=_heartbeat_printer,
            args=(_stop_event, _completed_ref, total_runs, 10.0, _print_lock),
            daemon=True,
        )
        _hb.start()

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, task): task for task in tasks}

            for future in as_completed(futures):
                completed += 1
                _completed_ref[0] = completed
                task = futures[future]
                try:
                    key, summary, full = future.result()
                    summaries[key] = summary
                    if full is not None:
                        full_results[key] = full

                    if show_progress:
                        method = list(summary.stats.keys())[0] if summary.stats else None
                        rmse_str = (
                            f"RMSE={summary.rmse(method):.0f}m" if method else ""
                        )
                        with _print_lock:
                            print(f"\r{' ' * _STATUS_WIDTH}\r", end='', flush=True)
                            print(
                                f"  [{completed:3d}/{total_runs}] "
                                f"dist={task['distance']/1e3:6.0f}km  "
                                f"speed={task['speed']:6.0f}m/s  "
                                f"angle={task['angle']:5.1f}°  "
                                f"[OK] ({summary.runtime_seconds:.0f}s)  "
                                f"obs={summary.num_observations:5d}  "
                                f"{rmse_str}"
                            )

                except Exception as exc:
                    failed_runs.append((
                        task['distance'], task['speed'], task['angle'], str(exc)
                    ))
                    if show_progress:
                        with _print_lock:
                            print(f"\r{' ' * _STATUS_WIDTH}\r", end='', flush=True)
                            print(
                                f"  [{completed:3d}/{total_runs}] "
                                f"dist={task['distance']/1e3:.0f}km  "
                                f"speed={task['speed']:.0f}m/s  "
                                f"angle={task['angle']:.0f}°  "
                                f"\u2717 FAILED: {exc}"
                            )

        _stop_event.set()
        _hb.join()
        print(f"\r{' ' * _STATUS_WIDTH}\r", end='', flush=True)

        # Package results
        total_runtime = time.time() - study_start

        experiment_results = ExperimentResults(
            summaries=summaries,
            full_results=full_results,
            distances=distances,
            speeds=speeds,
            crossing_angles=crossing_angles,
            metadata={
                'run_date': datetime.now().isoformat(),
                'total_runtime_seconds': total_runtime,
                'study_dir': study_dir,
                'total_runs': total_runs,
                'failed_runs': failed_runs,
                'distances': distances,
                'speeds': speeds,
                'crossing_angles': crossing_angles,
            }
        )

        # Save everything
        self._save_experiment_results(experiment_results, study_dir)

        # Print final summary
        if show_progress:
            self._print_study_summary_dsa(
                experiment_results, total_runtime, failed_runs, study_dir
            )

        return experiment_results

    # =========================================================================
    # SINGLE RUN (kept for standalone use)
    # =========================================================================

    def _run_single(
        self,
        distance: float,
        speed: float,
        save_full: bool,
        study_dir: str,
    ) -> Tuple[RunSummary, Optional[object]]:
        """
        Run one simulation and return summary + optional full results.

        Parameters
        ----------
        distance : float
            Closest approach distance [m]
        speed : float
            Missile speed [m/s]
        save_full : bool
            Whether to save full SimulationResults
        study_dir : str
            Study output directory

        Returns
        -------
        summary : RunSummary
            Statistics for this run
        sim_results : SimulationResults or None
            Full results if save_full=True, else None
        """
        run_start = time.time()

        # Create scenario
        scenario = self.factory.create_flyby_scenario(
            closest_approach_distance=distance,
            missile_speed=speed,
        )

        # Run simulation (suppress internal progress for batch runs)
        simulator = Simulator(scenario)
        sim_results = simulator.run(show_progress=False)

        runtime = time.time() - run_start

        # Extract summary statistics
        stats = {}
        for method in sim_results.available_methods:
            stats[method] = sim_results.get_statistics(method)

        summary = RunSummary(
            closest_approach_distance=distance,
            missile_speed=speed,
            crossing_angle_deg=self.factory.crossing_angle_deg,
            num_observations=sim_results.num_observations,
            runtime_seconds=runtime,
            stats=stats,
        )

        # Save full results if requested
        if save_full:
            label = self.factory.scenario_label(distance, speed)

            # Create individual run directory for plots
            run_plots_dir = os.path.join(
                study_dir, 'plots', 'individual_runs', label
            )
            os.makedirs(run_plots_dir, exist_ok=True)

            # Save full results pkl
            full_results_dir = os.path.join(study_dir, 'full_results')
            os.makedirs(full_results_dir, exist_ok=True)

            full_results_path = os.path.join(
                full_results_dir, f"{label}.pkl"
            )
            sim_results.save(full_results_path)

            return summary, sim_results

        return summary, None

    # =========================================================================
    # FOLDER STRUCTURE
    # =========================================================================

    def _create_study_directory(self, study_name: Optional[str]) -> str:
        """
        Create study output directory.

        Auto-increments study number if not provided:
        study_001, study_002, etc.

        Parameters
        ----------
        study_name : str or None
            Custom name, or None for auto-naming

        Returns
        -------
        str
            Path to created study directory
        """
        if study_name:
            study_dir = os.path.join(self.output_dir, study_name)
        else:
            # Auto-increment study number
            os.makedirs(self.output_dir, exist_ok=True)
            existing = [
                d for d in os.listdir(self.output_dir)
                if d.startswith('study_')
            ]
            next_num = len(existing) + 1
            study_name = f"study_{next_num:03d}"
            study_dir = os.path.join(self.output_dir, study_name)

        # Create all subdirectories
        os.makedirs(study_dir, exist_ok=True)
        os.makedirs(os.path.join(study_dir, 'plots', 'parameter_study'), exist_ok=True)
        os.makedirs(os.path.join(study_dir, 'plots', 'individual_runs'), exist_ok=True)
        os.makedirs(os.path.join(study_dir, 'full_results'), exist_ok=True)

        return study_dir

    # =========================================================================
    # SAVING
    # =========================================================================

    def _save_experiment_results(
        self,
        experiment_results: ExperimentResults,
        study_dir: str,
    ):
        """Save experiment results and CSV summary."""

        # Save main pkl
        pkl_path = os.path.join(study_dir, 'experiment_results.pkl')
        experiment_results.save(pkl_path)

        # Save CSV summary
        try:
            import pandas as pd
            df = experiment_results.to_dataframe()
            csv_path = os.path.join(study_dir, 'summary_table.csv')
            df.to_csv(csv_path, index=False)
            print(f"[OK] Saved summary table to {csv_path}")
        except ImportError:
            warnings.warn("pandas not installed - skipping CSV export")

    # =========================================================================
    # REPRESENTATIVE RUN SELECTION
    # =========================================================================

    def _select_representative_runs_dsa(
        self,
        distances: List[float],
        speeds: List[float],
        crossing_angles: List[float],
    ) -> List[Tuple[float, float, float]]:
        """
        Automatically select representative runs for full result saving.

        Selects all 8 corners of the 3D (distance × speed × angle) cube
        plus the center point, giving up to 9 representative runs.

        Parameters
        ----------
        distances : list of float
        speeds : list of float
        crossing_angles : list of float

        Returns
        -------
        list of (distance, speed, angle) 3-tuples
        """
        sorted_d = sorted(distances)
        sorted_s = sorted(speeds)
        sorted_a = sorted(crossing_angles)

        # Eight corners of the 3D cube
        d_extremes = [sorted_d[0], sorted_d[-1]]
        s_extremes = [sorted_s[0], sorted_s[-1]]
        a_extremes = [sorted_a[0], sorted_a[-1]]

        selected = set()
        for d in d_extremes:
            for s in s_extremes:
                for a in a_extremes:
                    selected.add((d, s, a))

        # Center of grid
        mid_d = sorted_d[len(sorted_d) // 2]
        mid_s = sorted_s[len(sorted_s) // 2]
        mid_a = sorted_a[len(sorted_a) // 2]
        selected.add((mid_d, mid_s, mid_a))

        return list(selected)

    # =========================================================================
    # PROGRESS PRINTING
    # =========================================================================

    def _print_study_header(
        self,
        distances: List[float],
        speeds: List[float],
        crossing_angles: List[float],
        study_dir: str,
    ):
        """Print study header."""
        total = len(distances) * len(speeds) * len(crossing_angles)
        print("\n" + "=" * 65)
        print("Parameter Study DSA: Distance × Speed × Angle")
        print("=" * 65)
        print(f"Distances:       {[f'{d/1e3:.0f}km' for d in sorted(distances)]}")
        print(f"Speeds:          {[f'{s:.0f}m/s' for s in sorted(speeds)]}")
        print(f"Crossing angles: {[f'{a:.0f}°' for a in sorted(crossing_angles)]}")
        print(f"Total runs:      {total}")
        print(f"Output: {study_dir}")
        print("=" * 65 + "\n")

    def _print_run_header(
        self,
        run_count: int,
        total_runs: int,
        distance: float,
        speed: float,
    ):
        """Print run header (used by serial _run_single only)."""
        print(
            f"[{run_count:3d}/{total_runs}] "
            f"distance={distance/1e3:6.0f}km  "
            f"speed={speed:6.0f}m/s  ...",
            end='',
            flush=True
        )

    def _print_run_result(self, summary: RunSummary, elapsed: float):
        """Print run result on same line (used by serial _run_single only)."""
        method = list(summary.stats.keys())[0] if summary.stats else None

        if method:
            rmse = summary.rmse(method)
            print(
                f"  [OK] ({elapsed:.0f}s)  "
                f"obs={summary.num_observations:5d}  "
                f"RMSE={rmse:.0f}m"
            )
        else:
            print(f"  [OK] ({elapsed:.0f}s)  obs={summary.num_observations}")

    def run_launch_az_el_sweep(
        self,
        azimuths: List[float],
        elevations: List[float],
        speed: float,
        lead_time_s: float = 20.0,
        pre_observe_s: float = 10.0,
        post_overhead_s: float = 150.0,
        launch_altitude_m: float = 10.0,
        study_name: Optional[str] = None,
        show_progress: bool = True,
        points: Optional[List[Tuple[float, float]]] = None,
        sampling: str = 'grid',
    ) -> 'AngularStudyResults':
        """
        Run 2D angular study across azimuth × elevation combinations.

        Rocket launches from Earth's surface directly below the satellite,
        lead_time_s before satellite overhead. Simulation starts pre_observe_s
        before rocket launch. All combinations run in parallel.

        Parameters
        ----------
        azimuths : list of float
            Azimuth angles [degrees]. Used when points=None (grid mode).
        elevations : list of float
            Elevation angles above horizontal [degrees]. Used when points=None (grid mode).
        speed : float
            Rocket speed [m/s]
        lead_time_s : float
            Seconds before satellite overhead that rocket launches. Default 20 s.
        pre_observe_s : float
            Seconds before launch that simulation starts. Default 10 s.
        post_overhead_s : float
            Seconds of observation after satellite overhead. Default 150 s.
        launch_altitude_m : float
            Launch height above Earth surface [m]. Default 10 m.
        study_name : str, optional
            Folder name, auto-generated if None
        show_progress : bool
            Print per-run progress lines
        points : list of (azimuth, elevation) tuples, optional
            If provided, use these exact (az, el) pairs instead of the az×el grid.
            Use fibonacci_hemisphere_points(n) to generate uniform hemisphere coverage.
        sampling : str
            'grid' (default) or 'fibonacci'. Stored in results for plot labelling.

        Returns
        -------
        AngularStudyResults
        """
        study_start = time.time()
        study_dir = self._create_study_directory(study_name or 'launch_az_el_sweep')

        # Determine which (az, el) pairs to run
        if points is not None:
            _pairs = list(points)
        else:
            _pairs = [(az, el) for az in sorted(azimuths) for el in sorted(elevations)]

        total = len(_pairs)
        total_duration = pre_observe_s + lead_time_s + post_overhead_s
        if show_progress:
            print("=" * 65)
            if points is not None:
                print(f"Angular Study ({sampling}): {total} runs")
            else:
                print(f"Angular Study: {len(azimuths)}×{len(elevations)} = {total} runs")
            print(f"  Speed: {speed:.0f} m/s  |  Duration: {total_duration:.0f} s/run")
            print(f"  Timing: launch T-{lead_time_s:.0f}s, sim start T-{pre_observe_s+lead_time_s:.0f}s")
            if points is None:
                print(f"  Azimuths:   {[f'{a:.0f}°' for a in sorted(azimuths)]}")
                print(f"  Elevations: {[f'{e:.0f}°' for e in sorted(elevations)]}")
            print(f"  Output: {study_dir}")
            print("=" * 65)

        tasks = []
        for az, el in _pairs:
            tasks.append({
                'satellite_spec':     self.factory.satellite_spec,
                'depth_time_offsets': self.factory.depth_time_offsets,
                'azimuth':        az,
                'elevation':      el,
                'speed':          speed,
                'lead_time_s':    lead_time_s,
                'pre_observe_s':  pre_observe_s,
                'post_overhead_s': post_overhead_s,
                'launch_altitude_m': launch_altitude_m,
                'save_full': True,
                'study_dir': study_dir,
            })

        summaries: Dict = {}
        full_results: Dict = {}
        failed_runs = []
        completed = 0

        n_workers = max(1, (os.cpu_count() or 1) - 1)
        if show_progress:
            print(
                f"Launching {total} runs on "
                f"{n_workers} parallel worker(s)\n"
            )

        _completed_ref = [0]
        _stop_event    = threading.Event()
        _print_lock    = threading.Lock()
        _hb = threading.Thread(
            target=_heartbeat_printer,
            args=(_stop_event, _completed_ref, total, 10.0, _print_lock),
            daemon=True,
        )
        _hb.start()

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker_angular, task): task for task in tasks}

            for future in as_completed(futures):
                completed += 1
                _completed_ref[0] = completed
                task = futures[future]
                try:
                    key, summary, full = future.result()
                    summaries[key] = summary
                    # Do NOT accumulate full results in RAM — each run is already
                    # saved individually to disk by the worker. Keeping all 360
                    # SimulationResults objects in memory would cause MemoryError.

                    if show_progress:
                        method = list(summary.stats.keys())[0] if summary.stats else None
                        rmse_str = (
                            f"RMSE={summary.rmse(method):.0f}m" if method else ""
                        )
                        with _print_lock:
                            print(f"\r{' ' * _STATUS_WIDTH}\r", end='', flush=True)
                            print(
                                f"  [{completed:3d}/{total}] "
                                f"az={task['azimuth']:5.1f}°  "
                                f"el={task['elevation']:5.1f}°  "
                                f"[OK] ({summary.runtime_seconds:.0f}s)  "
                                f"obs={summary.num_observations:5d}  "
                                f"{rmse_str}"
                            )

                except Exception as exc:
                    failed_runs.append((task['azimuth'], task['elevation'], str(exc)))
                    if show_progress:
                        with _print_lock:
                            print(f"\r{' ' * _STATUS_WIDTH}\r", end='', flush=True)
                            print(
                                f"  [{completed:3d}/{total}] "
                                f"az={task['azimuth']:.0f}deg  "
                                f"el={task['elevation']:.0f}deg  "
                                f"FAILED: {exc}"
                            )

        _stop_event.set()
        _hb.join()
        print(f"\r{' ' * _STATUS_WIDTH}\r", end='', flush=True)

        total_runtime = time.time() - study_start

        results = AngularStudyResults(
            summaries=summaries,
            full_results={},  # individual runs already saved to disk; don't hold all in RAM
            azimuths=azimuths,
            elevations=elevations,
            distance=None,
            speed=speed,
            sampling=sampling,
            metadata={
                'run_date': datetime.now().isoformat(),
                'total_runtime_seconds': total_runtime,
                'study_dir': study_dir,
                'total_runs': total,
                'failed_runs': failed_runs,
                'lead_time_s': lead_time_s,
                'pre_observe_s': pre_observe_s,
                'post_overhead_s': post_overhead_s,
                'launch_altitude_m': launch_altitude_m,
            },
        )

        pkl_path = os.path.join(study_dir, 'launch_az_el_sweep_results.pkl')
        results.save(pkl_path)

        if show_progress:
            print(f"\n{'='*65}")
            print(f"[OK] Angular Study Complete — {total_runtime/60:.1f} min")
            print(f"  Successful: {len(summaries)}  |  Failed: {len(failed_runs)}")
            print(f"  Results saved to: {study_dir}")

        return results

    def _print_study_summary_dsa(
        self,
        experiment_results: ExperimentResults,
        total_runtime: float,
        failed_runs: list,
        study_dir: str,
    ):
        """Print study completion summary."""
        print("\n" + "=" * 65)
        print("[OK] Parameter Study DSA Complete")
        print("=" * 65)
        print(f"Total time: {total_runtime/60:.1f} minutes")
        print(f"Successful runs: {len(experiment_results.summaries)}")

        if failed_runs:
            print(f"Failed runs: {len(failed_runs)}")
            for dist, speed, angle, error in failed_runs:
                print(
                    f"  ✗ dist={dist/1e3:.0f}km, "
                    f"speed={speed:.0f}m/s, "
                    f"angle={angle:.0f}°: {error}"
                )

        print(f"\nResults saved to: {study_dir}")


# =============================================================================
# MODULE-LEVEL WORKER FOR ANGULAR STUDY
# =============================================================================

def _worker_angular(task: dict) -> tuple:
    """
    Run one simulation for the angular study (azimuth × elevation grid).

    Parameters
    ----------
    task : dict
        Keys: satellite_spec, duration, detection_fraction, depth_time_offsets,
              azimuth, elevation, distance, speed, save_full, study_dir

    Returns
    -------
    tuple of ((azimuth, elevation), RunSummary, sim_results_or_None)
    """
    import time as _time
    import os as _os
    import sys as _sys
    from missile_fly_by_simulation.experiments.scenario_factory import ScenarioFactory
    from missile_fly_by_simulation.experiments.experiment_results import RunSummary
    from missile_fly_by_simulation.simulation.simulator import Simulator

    _devnull = open(_os.devnull, 'w')
    _old_stdout = _sys.stdout
    _sys.stdout = _devnull

    try:
        run_start = _time.time()

        factory = ScenarioFactory(
            satellite_spec=task['satellite_spec'],
            depth_time_offsets=task['depth_time_offsets'],
        )

        lead = task.get('lead_time_s', 20.0)
        pre  = task.get('pre_observe_s', 10.0)
        post = task.get('post_overhead_s', 150.0)

        scenario = factory.create_radial_launch_scenario(
            radial_speed=task['speed'],
            launch_lead_time_s=lead,
            pre_launch_observe_s=pre,
            post_launch_observe_s=post + lead,
            pre_detection_buffer_s=0.0,
            crossing_angle_deg=task['azimuth'],
            elevation_angle_deg=task['elevation'],
        )

        simulator = Simulator(scenario)
        sim_results = simulator.run(show_progress=False)

        runtime = _time.time() - run_start

        stats = {}
        for method in sim_results.available_methods:
            stats[method] = sim_results.get_statistics(method)

        summary = RunSummary(
            closest_approach_distance=scenario._closest_approach_distance,
            missile_speed=task['speed'],
            crossing_angle_deg=task['azimuth'],
            elevation_angle_deg=task['elevation'],
            num_observations=sim_results.num_observations,
            runtime_seconds=runtime,
            stats=stats,
        )

        full_result_to_return = None
        if task['save_full']:
            label = (
                f"az{int(task['azimuth'])}deg_"
                f"el{int(task['elevation'])}deg"
            )
            full_results_dir = _os.path.join(task['study_dir'], 'full_results')
            _os.makedirs(full_results_dir, exist_ok=True)
            full_results_path = _os.path.join(full_results_dir, f"{label}.pkl")
            sim_results.save(full_results_path)
            full_result_to_return = sim_results

        key = (task['azimuth'], task['elevation'])
        return key, summary, full_result_to_return

    finally:
        _sys.stdout = _old_stdout
        _devnull.close()

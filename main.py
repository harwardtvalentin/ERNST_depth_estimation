"""
Main entry point for satellite-missile depth estimation simulation.

Three modes:
    single  - Run one simulation and generate all plots
    study   - Run full parameter study (distance × speed grid)
    plot    - Load existing results and regenerate plots

Usage
-----
    # Single run (quick test, default scenario)
    python main.py --mode single

    # Single run with custom parameters
    python main.py --mode single --distance 200 --speed 1000

    # Full parameter study (runs overnight)
    python main.py --mode study

    # Quick study with fewer points (for testing)
    python main.py --mode study --quick

    # Regenerate plots from saved results (no rerunning!)
    python main.py --mode plot --results experiments/runs/study_001/experiment_results.pkl

    # Change plot config
    python main.py --mode single --plot-config thesis
    python main.py --mode single --plot-config presentation
    python main.py --mode single --plot-config preview
"""

import argparse
import os
import sys
import time
from datetime import datetime

from missile_fly_by_simulation.constants import save_snapshot, DEFAULT_DURATION_S

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Satellite-Missile Depth Estimation Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--mode',
        choices=['single', 'study', 'flyby_azimuth_sweep', 'launch_az_el_sweep', 'radial', 'plot'],
        default='single',
        help='Run mode (default: single)',
    )

    # Single run parameters
    parser.add_argument(
        '--distance',
        type=float,
        default=200.0,
        help='Closest approach distance in km (default: 200)',
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1000.0,
        help='Missile speed in m/s (default: 1000)',
    )
    parser.add_argument(
        '--angle',
        type=float,
        default=90.0,
        help='Crossing angle between satellite and missile velocity vectors in degrees (default: 90)',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=DEFAULT_DURATION_S,
        help=f'Simulation duration in seconds (default: {DEFAULT_DURATION_S})',
    )

    parser.add_argument(
        '--lead-time',
        type=float,
        default=None,
        help='Seconds before satellite overhead that rocket launches (--mode radial, default: 20)',
    )

    # Study parameters
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick study with fewer parameter points (for testing): 3x3x4 = 36 runs',
    )
    _sampling = parser.add_mutually_exclusive_group()
    _sampling.add_argument(
        '--grid',
        action='store_true',
        help='(--mode launch_az_el_sweep) Use az x el grid sampling (default)',
    )
    _sampling.add_argument(
        '--fibonacci',
        action='store_true',
        help='(--mode launch_az_el_sweep) Use Fibonacci hemisphere sampling instead of az x el grid',
    )

    # Plot parameters
    parser.add_argument(
        '--results',
        type=str,
        default=None,
        help='Path to saved ExperimentResults .pkl file (for --mode plot)',
    )
    parser.add_argument(
        '--plot-config',
        choices=['preview', 'thesis', 'presentation'],
        default='preview',
        help='Plot quality preset (default: preview)',
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/runs',
        help='Output directory for results (default: experiments/runs)',
    )

    return parser.parse_args()


# =============================================================================
# PLOT CONFIG HELPER
# =============================================================================

def get_plot_config(name: str):
    """Get PlotConfig preset by name."""
    from missile_fly_by_simulation.visualization import (
        thesis_config,
        presentation_config,
        preview_config,
    )
    presets = {
        'thesis': thesis_config,
        'presentation': presentation_config,
        'preview': preview_config,
    }
    return presets[name]()


# =============================================================================
# MODE 1: SINGLE RUN
# =============================================================================

def run_single(args):
    """
    Run one simulation and generate all plots.

    Good for:
    - Quick testing
    - Debugging
    - Visualizing one specific flyby scenario
    """
    from missile_fly_by_simulation.experiments import ScenarioFactory
    from missile_fly_by_simulation.simulation.simulator import Simulator
    from missile_fly_by_simulation.visualization import (
        plot_all_statistical,
        plot_all_trajectory,
    )

    print("\n" + "=" * 60)
    print("Mode: Single Run")
    print("=" * 60)
    print(f"  Distance: {args.distance} km")
    print(f"  Speed:    {args.speed} m/s")
    print(f"  Angle:    {args.angle}°")
    print(f"  Duration: {args.duration} s")
    print(f"  Plots:    {args.plot_config} quality")
    print("=" * 60)

    config = get_plot_config(args.plot_config)

    # Create scenario
    factory = ScenarioFactory(
        satellite_spec=ScenarioFactory.default_satellite_spec(),
        simulation_duration=args.duration,
        crossing_angle_deg=args.angle,
    )

    scenario = factory.create_flyby_scenario(
        closest_approach_distance=args.distance * 1e3,  # km → m
        missile_speed=args.speed,
    )

    print(f"\n{scenario}")

    # Validate
    errors = scenario.validate()
    if errors:
        print("\n[!] Configuration warnings:")
        for e in errors:
            print(f"  - {e}")

    # Run
    print("\nRunning simulation...")
    simulator = Simulator(scenario)
    results = simulator.run(show_progress=True)

    # Create output folder
    label = factory.scenario_label(args.distance * 1e3, args.speed) + f'_angle{int(args.angle)}deg'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        args.output_dir,
        f'single_{label}_{timestamp}'
    )
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_snapshot(run_dir)

    # Save results
    results_path = os.path.join(run_dir, 'simulation_results.pkl')
    results.save(results_path)

    try:
        df = results.to_dataframe()
        df.to_csv(os.path.join(run_dir, 'depth_estimates.csv'), index=False, sep=';')
        print(f"[OK] Saved CSV to {run_dir}/depth_estimates.csv")
    except ImportError:
        pass

    # Print statistics
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(results.summary())

    print("\n" + "=" * 60)
    print("Depth Estimation Statistics")
    print("=" * 60)
    for method in results.available_methods:
        stats = results.get_statistics(method)
        print(f"\n{method.upper()}:")
        print(f"  Estimates:        {stats['num_estimates']}")
        print(f"  RMSE:             {stats['rmse']:.2f} m")
        print(f"  Mean Error:       {stats['mean_error']:.2f} m")
        print(f"  Median Error:     {stats['median_error']:.2f} m")
        print(f"  Std Dev:          {stats['std_error']:.2f} m")
        print(f"  95th Percentile:  {stats['error_95th']:.2f} m")

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    plot_all_statistical(
        results,
        save_path=plots_dir,
        config=config,
    )
    plot_all_trajectory(
        results,
        save_path=plots_dir,
        config=config,
    )

    # Done
    print("\n" + "=" * 60)
    print("[OK] Single Run Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {run_dir}")
    print(f"  simulation_results.pkl")
    print(f"  depth_estimates.csv")
    print(f"  plots/  ({_count_files(plots_dir)} figures)")


# =============================================================================
# MODE 2: PARAMETER STUDY
# =============================================================================

def run_study(args):
    """
    Run full DSA parameter study across distance × speed × crossing_angle grid.

    Quick mode (--quick): 3×3×4 = 36 runs (~1 h on 4 cores)
    Full mode:            7×7×7 = 343 runs (~6 h on 6 cores, run overnight)

    Uses ProcessPoolExecutor for parallel execution (N_cpu - 1 workers).
    """
    from missile_fly_by_simulation.experiments import ScenarioFactory, BatchRunner
    from missile_fly_by_simulation.visualization import (
        plot_all_parameter_study,
        plot_all_statistical,
        plot_all_trajectory,
    )

    print("\n" + "=" * 60)
    print("Mode: Parameter Study DSA (Distance × Speed × Angle)")
    print("=" * 60)

    config = get_plot_config(args.plot_config)

    # Parameter grid (--angle CLI arg is only for --mode single)
    if args.quick:
        # Quick test: 3×3×4 = 36 runs
        distances       = [20e3, 70e3, 150e3]                # 3 distances [m]
        speeds          = [500.0, 1000.0, 3000.0]            # 3 speeds [m/s]
        crossing_angles = [0.0, 30.0, 60.0, 90.0]           # 4 angles [°]
        print("  Mode: QUICK (3×3×4 = 36 runs, ~1 h on 4 cores)")
    else:
        # Full study: 7×7×7 = 343 runs
        distances       = [50e3, 75e3, 100e3, 150e3, 200e3, 500e3, 1000e3]
        speeds          = [300.0, 500.0, 750.0, 1000.0, 2000.0, 3000.0, 7000.0]
        crossing_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        print("  Mode: FULL (7×7×7 = 343 runs, ~6 h on 6 cores)")

    print(f"  Distances:       {[f'{d/1e3:.0f}km' for d in distances]}")
    print(f"  Speeds:          {[f'{s:.0f}m/s' for s in speeds]}")
    print(f"  Crossing angles: {[f'{a:.0f}°' for a in crossing_angles]}")
    print(f"  Plots: {args.plot_config} quality")
    print("=" * 60)

    # Warn about time for full run
    if not args.quick:
        print("\n[!] Full study takes ~6 h on 6 cores (~34 h serial).")
        print("   Consider using --quick first to verify everything works.")
        print("   Press Ctrl+C at any time to stop.\n")
        try:
            input("   Press Enter to continue, or Ctrl+C to cancel...")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Setup (factory crossing_angle_deg is overridden per-run inside the worker)
    factory = ScenarioFactory(
        satellite_spec=ScenarioFactory.default_satellite_spec(),
        simulation_duration=args.duration,
    )

    runner = BatchRunner(
        factory=factory,
        output_dir=args.output_dir,
    )

    # Run (parallel, 3D grid)
    experiment_results = runner.run_parameter_study_dsa(
        distances=distances,
        speeds=speeds,
        crossing_angles=crossing_angles,
        show_progress=True,
    )

    # Study output dir (where batch_runner saved everything)
    study_dir = experiment_results.metadata['study_dir']
    save_snapshot(study_dir)
    param_plots_dir = os.path.join(study_dir, 'plots', 'parameter_study')
    individual_dir = os.path.join(study_dir, 'plots', 'individual_runs')

    # Generate parameter study plots (heatmaps per angle subfolder)
    print("\n" + "=" * 60)
    print("Generating Parameter Study Plots")
    print("=" * 60)

    plot_all_parameter_study(
        experiment_results,
        save_path=param_plots_dir,
        config=config,
    )

    # Generate individual plots for selected runs (3D cube corners + center)
    print("\n" + "=" * 60)
    print("Generating Individual Run Plots (selected runs)")
    print("=" * 60)

    for key, sim_results in experiment_results.full_results.items():
        dist, speed, angle = key
        label = factory.scenario_label(dist, speed) + f'_angle{int(angle)}deg'
        run_plots_dir = os.path.join(individual_dir, label)
        os.makedirs(run_plots_dir, exist_ok=True)

        print(f"\n  -> {label}")
        plot_all_statistical(sim_results, save_path=run_plots_dir, config=config)
        plot_all_trajectory(sim_results, save_path=run_plots_dir, config=config)

    # Done
    print("\n" + "=" * 60)
    print("[OK] Parameter Study DSA Complete!")
    print("=" * 60)
    print(experiment_results.summary())
    print(f"\nAll outputs saved to: {study_dir}")


# =============================================================================
# MODE 3: ANGLE PARAMETER STUDY
# =============================================================================

def run_flyby_azimuth_sweep(args):
    """
    Run 1D crossing-angle parameter study (horizontal fly-by scenario).

    Fixed: distance=200 km, speed=1000 m/s, duration=400 s
    Sweep: crossing angles 0, 5, 10, ..., 180 degrees (37 values)

    All 37 full SimulationResults PKLs are saved.
    Output folder: flyby_azimuth_sweep_d200km_s1000ms_YYYYMMDD_HHMM
    """
    from missile_fly_by_simulation.constants import (
        ANGLE_STUDY_DURATION_S,
        ANGLE_STUDY_DISTANCE_M,
        ANGLE_STUDY_SPEED_MS,
        ANGLE_STUDY_ANGLES_DEG,
    )
    from missile_fly_by_simulation.experiments import ScenarioFactory, BatchRunner
    from missile_fly_by_simulation.visualization import (
        plot_all_flyby_azimuth_sweep,
        plot_all_statistical,
        plot_all_trajectory,
    )

    distances       = [ANGLE_STUDY_DISTANCE_M]
    speeds          = [ANGLE_STUDY_SPEED_MS]
    crossing_angles = [float(a) for a in ANGLE_STUDY_ANGLES_DEG]

    # Save all 37 full PKLs
    save_full_for = [(distances[0], speeds[0], a) for a in crossing_angles]

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M')
    study_name = (
        f"flyby_azimuth_sweep"
        f"_d{int(distances[0]/1e3)}km"
        f"_s{int(speeds[0])}ms"
        f"_{timestamp}"
    )

    print("\n" + "=" * 60)
    print("Mode: 1D Crossing-Angle Parameter Study")
    print("=" * 60)
    print(f"  Distance:  {distances[0]/1e3:.0f} km (fixed)")
    print(f"  Speed:     {speeds[0]:.0f} m/s (fixed)")
    print(f"  Angles:    0 to 180 deg in 5 deg steps ({len(crossing_angles)} runs)")
    print(f"  Duration:  {ANGLE_STUDY_DURATION_S:.0f} s per run")
    print(f"  Full PKLs: all {len(crossing_angles)} runs saved")
    print(f"  Plots:     {args.plot_config} quality")
    print("=" * 60)

    config = get_plot_config(args.plot_config)

    factory = ScenarioFactory(
        satellite_spec=ScenarioFactory.default_satellite_spec(),
        simulation_duration=ANGLE_STUDY_DURATION_S,
    )

    runner = BatchRunner(
        factory=factory,
        output_dir=args.output_dir,
        save_full_results_for=save_full_for,
    )

    experiment_results = runner.run_parameter_study_dsa(
        distances=distances,
        speeds=speeds,
        crossing_angles=crossing_angles,
        study_name=study_name,
        show_progress=True,
    )

    study_dir       = experiment_results.metadata['study_dir']
    param_plots_dir = os.path.join(study_dir, 'plots', 'parameter_study')
    individual_dir  = os.path.join(study_dir, 'plots', 'individual_runs')
    save_snapshot(study_dir)

    # Angle study summary plots (A1-A3, B1-B2)
    print("\n" + "=" * 60)
    print("Generating Angle Study Plots")
    print("=" * 60)
    plot_all_flyby_azimuth_sweep(
        experiment_results,
        save_path=param_plots_dir,
        config=config,
    )

    # Individual run plots for all stored PKLs
    print("\n" + "=" * 60)
    print("Generating Individual Run Plots (all angles)")
    print("=" * 60)
    for key, sim_results in experiment_results.full_results.items():
        dist, speed, angle = key
        label = factory.scenario_label(dist, speed) + f'_angle{int(angle)}deg'
        run_plots_dir = os.path.join(individual_dir, label)
        os.makedirs(run_plots_dir, exist_ok=True)
        print(f"  -> {label}")
        plot_all_statistical(sim_results, save_path=run_plots_dir, config=config)
        plot_all_trajectory(sim_results, save_path=run_plots_dir, config=config)

    print("\n" + "=" * 60)
    print("[OK] Angle Study Complete!")
    print("=" * 60)
    print(experiment_results.summary())
    print(f"\nAll outputs saved to: {study_dir}")


# =============================================================================
# MODE 4: 2D ANGULAR STUDY (Azimuth × Elevation)
# =============================================================================

def run_launch_az_el_sweep(args):
    """
    Run 2D angular parameter study (ground-launch scenario).

    Grid mode (default): azimuth 0-350 deg (every 10 deg) x elevation 0-90 deg (every 10 deg).
    Fibonacci mode (--fibonacci): N quasi-uniformly distributed hemisphere points.
    """
    from missile_fly_by_simulation.constants import (
        ANGULAR_STUDY_SPEED_MS,
        ANGULAR_STUDY_AZIMUTHS_DEG,
        ANGULAR_STUDY_ELEVATIONS_DEG,
        ANGULAR_STUDY_AZIMUTHS_QUICK_DEG,
        ANGULAR_STUDY_ELEVATIONS_QUICK_DEG,
        ANGULAR_STUDY_LEAD_TIME_S,
        ANGULAR_STUDY_PRE_OBSERVE_S,
        ANGULAR_STUDY_POST_OVERHEAD_S,
        ANGULAR_STUDY_LAUNCH_ALTITUDE_M,
        ANGULAR_STUDY_N_FIBONACCI,
        ANGULAR_STUDY_N_FIBONACCI_QUICK,
    )
    from missile_fly_by_simulation.experiments import ScenarioFactory, BatchRunner
    from missile_fly_by_simulation.experiments.scenario_factory import fibonacci_hemisphere_points
    from missile_fly_by_simulation.visualization import plot_all_launch_az_el_sweep

    use_fibonacci = getattr(args, 'fibonacci', False)

    if use_fibonacci:
        n = ANGULAR_STUDY_N_FIBONACCI_QUICK if args.quick else ANGULAR_STUDY_N_FIBONACCI
        points = fibonacci_hemisphere_points(n)
        azimuths   = []
        elevations = []
        sampling   = 'fibonacci'
        mode_str   = f"FIBONACCI ({n} points{'  QUICK' if args.quick else ''})"
    elif args.quick:
        azimuths   = [float(a) for a in ANGULAR_STUDY_AZIMUTHS_QUICK_DEG]
        elevations = [float(e) for e in ANGULAR_STUDY_ELEVATIONS_QUICK_DEG]
        points     = None
        sampling   = 'grid'
        mode_str   = f"QUICK ({len(azimuths)}x{len(elevations)} = {len(azimuths)*len(elevations)} runs)"
    else:
        azimuths   = [float(a) for a in ANGULAR_STUDY_AZIMUTHS_DEG]
        elevations = [float(e) for e in ANGULAR_STUDY_ELEVATIONS_DEG]
        points     = None
        sampling   = 'grid'
        mode_str   = f"FULL ({len(azimuths)}x{len(elevations)} = {len(azimuths)*len(elevations)} runs)"

    total_duration = ANGULAR_STUDY_PRE_OBSERVE_S + ANGULAR_STUDY_LEAD_TIME_S + ANGULAR_STUDY_POST_OVERHEAD_S

    print("\n" + "=" * 60)
    print("Mode: 2D Angular Study (Azimuth x Elevation) — Ground Launch")
    print("=" * 60)
    print(f"  {mode_str}")
    print(f"  Speed:      {ANGULAR_STUDY_SPEED_MS:.0f} m/s")
    print(f"  Timing:     launch T-{ANGULAR_STUDY_LEAD_TIME_S:.0f}s, sim start T-{ANGULAR_STUDY_PRE_OBSERVE_S+ANGULAR_STUDY_LEAD_TIME_S:.0f}s")
    print(f"  Duration:   {total_duration:.0f} s per run")
    if not use_fibonacci:
        print(f"  Azimuths:   {[f'{a:.0f}°' for a in azimuths]}")
        print(f"  Elevations: {[f'{e:.0f}°' for e in elevations]}")
    print(f"  Plots:      {args.plot_config} quality")
    print("=" * 60)

    config = get_plot_config(args.plot_config)

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M')
    suffix     = 'fib' if use_fibonacci else 'grid'
    study_name = f"launch_az_el_sweep_ground_s{int(ANGULAR_STUDY_SPEED_MS)}ms_{suffix}_{timestamp}"

    factory = ScenarioFactory(
        satellite_spec=ScenarioFactory.default_satellite_spec(),
    )

    runner = BatchRunner(
        factory=factory,
        output_dir=args.output_dir,
    )

    angular_results = runner.run_launch_az_el_sweep(
        azimuths=azimuths,
        elevations=elevations,
        speed=ANGULAR_STUDY_SPEED_MS,
        lead_time_s=ANGULAR_STUDY_LEAD_TIME_S,
        pre_observe_s=ANGULAR_STUDY_PRE_OBSERVE_S,
        post_overhead_s=ANGULAR_STUDY_POST_OVERHEAD_S,
        launch_altitude_m=ANGULAR_STUDY_LAUNCH_ALTITUDE_M,
        study_name=study_name,
        show_progress=True,
        points=points,
        sampling=sampling,
    )

    study_dir       = angular_results.metadata['study_dir']
    param_plots_dir = os.path.join(study_dir, 'plots', 'parameter_study')
    os.makedirs(param_plots_dir, exist_ok=True)
    save_snapshot(study_dir)

    print("\n" + "=" * 60)
    print("Generating Launch Sweep Plots")
    print("=" * 60)
    plot_all_launch_az_el_sweep(
        angular_results,
        save_path=param_plots_dir,
        config=config,
    )

    print("\n" + "=" * 60)
    print("[OK] Angular Study Complete!")
    print("=" * 60)
    print(f"  Runs: {len(angular_results.summaries)}")
    print(f"  Output: {study_dir}")


# =============================================================================
# MODE 5: RADIAL LAUNCH
# =============================================================================

def run_radial(args):
    """
    Run a single simulation where a rocket launches vertically from Earth's surface.

    The satellite flies over the launch site; the rocket starts ascending straight up
    (radially) a fixed number of seconds before the satellite passes directly overhead.

    Parameters (CLI):
        --speed      Rocket ascent speed [m/s] (default: 200)
        --lead-time  Seconds before overhead that rocket launches (default: 20)
        --duration   Simulation duration [s]
    """
    from missile_fly_by_simulation.experiments import ScenarioFactory
    from missile_fly_by_simulation.simulation.simulator import Simulator
    from missile_fly_by_simulation.visualization import (
        plot_all_statistical,
        plot_all_trajectory,
    )
    from missile_fly_by_simulation.constants import DEFAULT_LAUNCH_LEAD_TIME_S

    lead_time = args.lead_time if args.lead_time is not None else DEFAULT_LAUNCH_LEAD_TIME_S

    print("\n" + "=" * 60)
    print("Mode: Radial Launch (vertical rocket from Earth surface)")
    print("=" * 60)
    print(f"  Speed:     {args.speed} m/s  (radial, straight up)")
    print(f"  Lead time: {lead_time} s  (before satellite overhead)")
    print(f"  Duration:  {args.duration} s")
    print(f"  Plots:     {args.plot_config} quality")
    print("=" * 60)

    config = get_plot_config(args.plot_config)

    factory = ScenarioFactory(
        satellite_spec=ScenarioFactory.default_satellite_spec(),
        simulation_duration=args.duration,
    )

    scenario = factory.create_radial_launch_scenario(
        radial_speed=args.speed,
        launch_lead_time_s=lead_time,
    )

    print(f"\n{scenario}")

    errors = scenario.validate()
    if errors:
        print("\n[!] Configuration warnings:")
        for e in errors:
            print(f"  - {e}")

    print("\nRunning simulation...")
    simulator = Simulator(scenario)
    results = simulator.run(show_progress=True)

    # Output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        args.output_dir,
        f'single_radial_s{int(args.speed)}ms_lead{int(lead_time)}s_{timestamp}'
    )
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_snapshot(run_dir)

    results_path = os.path.join(run_dir, 'simulation_results.pkl')
    results.save(results_path)

    try:
        df = results.to_dataframe()
        df.to_csv(os.path.join(run_dir, 'depth_estimates.csv'), index=False, sep=';')
        print(f"[OK] Saved CSV to {run_dir}/depth_estimates.csv")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(results.summary())

    print("\n" + "=" * 60)
    print("Depth Estimation Statistics")
    print("=" * 60)
    for method in results.available_methods:
        stats = results.get_statistics(method)
        print(f"\n{method.upper()}:")
        print(f"  Estimates:        {stats['num_estimates']}")
        print(f"  RMSE:             {stats['rmse']:.2f} m")
        print(f"  Mean Error:       {stats['mean_error']:.2f} m")
        print(f"  Median Error:     {stats['median_error']:.2f} m")
        print(f"  Std Dev:          {stats['std_error']:.2f} m")
        print(f"  95th Percentile:  {stats['error_95th']:.2f} m")

    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    plot_all_statistical(results, save_path=plots_dir, config=config)
    plot_all_trajectory(results, save_path=plots_dir, config=config)

    print("\n" + "=" * 60)
    print("[OK] Radial Launch Run Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {run_dir}")
    print(f"  simulation_results.pkl")
    print(f"  plots/  ({_count_files(plots_dir)} figures)")


# =============================================================================
# MODE 5: LOAD AND PLOT
# =============================================================================

def run_plot(args):
    """
    Load existing results and regenerate all plots.

    Useful for:
    - Tweaking plot style without rerunning simulation
    - Changing plot config (preview → thesis)
    - Adding new plot types to existing results
    """
    from missile_fly_by_simulation.experiments import (
        ExperimentResults,
        ScenarioFactory,
    )
    from missile_fly_by_simulation.experiments.experiment_results import AngularStudyResults
    from missile_fly_by_simulation.simulation.results import SimulationResults
    from missile_fly_by_simulation.visualization import (
        plot_all_statistical,
        plot_all_trajectory,
        plot_all_parameter_study,
        plot_all_flyby_azimuth_sweep,
        plot_all_launch_az_el_sweep,
    )

    print("\n" + "=" * 60)
    print("Mode: Load and Plot")
    print("=" * 60)

    if args.results is None:
        print("ERROR: --results path required for --mode plot")
        print("Example: python main.py --mode plot --results experiments/runs/study_001/experiment_results.pkl")
        sys.exit(1)

    if not os.path.exists(args.results):
        print(f"ERROR: File not found: {args.results}")
        sys.exit(1)

    print(f"  Loading: {args.results}")
    print(f"  Plots:   {args.plot_config} quality")
    print("=" * 60)

    config = get_plot_config(args.plot_config)

    # Detect file type: ExperimentResults or SimulationResults
    import pickle
    with open(args.results, 'rb') as f:
        loaded = pickle.load(f)

    results_dir = os.path.dirname(args.results)

    if isinstance(loaded, ExperimentResults):
        # Parameter study results
        print(f"\nDetected: ExperimentResults")
        print(loaded.summary())

        param_plots_dir = os.path.join(results_dir, 'plots', 'parameter_study')
        individual_dir  = os.path.join(results_dir, 'plots', 'individual_runs')

        os.makedirs(param_plots_dir, exist_ok=True)

        # Detect study type: 1D angle sweep vs full DSA grid
        is_flyby_azimuth_sweep = (
            len(loaded.distances) == 1
            and len(loaded.speeds) == 1
            and len(loaded.crossing_angles) > 1
        )

        if is_flyby_azimuth_sweep:
            print("\nDetected 1D flyby sweep — generating flyby sweep plots...")
            plot_all_flyby_azimuth_sweep(loaded, save_path=param_plots_dir, config=config)
        else:
            print("\nGenerating parameter study plots...")
            plot_all_parameter_study(loaded, save_path=param_plots_dir, config=config)

        # Individual run plots (only for stored full results)
        if loaded.full_results:
            print("\nGenerating individual run plots...")
            factory = ScenarioFactory(
                satellite_spec=ScenarioFactory.default_satellite_spec()
            )
            for key, sim_results in loaded.full_results.items():
                dist, speed, angle = key
                label = factory.scenario_label(dist, speed) + f'_angle{int(angle)}deg'
                run_plots_dir = os.path.join(individual_dir, label)
                os.makedirs(run_plots_dir, exist_ok=True)

                print(f"  -> {label}")
                plot_all_statistical(sim_results, save_path=run_plots_dir, config=config)
                plot_all_trajectory(sim_results, save_path=run_plots_dir, config=config)

        print(f"\n[OK] Plots saved to: {results_dir}/plots/")

    elif isinstance(loaded, SimulationResults):
        # Single run results
        print(f"\nDetected: SimulationResults (single run)")
        print(loaded.summary())

        plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        print("\nGenerating plots...")
        plot_all_statistical(loaded, save_path=plots_dir, config=config)
        plot_all_trajectory(loaded, save_path=plots_dir, config=config)

        print(f"\n[OK] Plots saved to: {plots_dir}")

    elif isinstance(loaded, AngularStudyResults):
        # 2D angular study results
        print(f"\nDetected: AngularStudyResults (2D angular study)")

        param_plots_dir = os.path.join(results_dir, 'plots', 'parameter_study')
        os.makedirs(param_plots_dir, exist_ok=True)

        print("\nGenerating launch sweep plots...")
        plot_all_launch_az_el_sweep(loaded, save_path=param_plots_dir, config=config)

        print(f"\n[OK] Plots saved to: {param_plots_dir}")

    else:
        print(f"ERROR: Unrecognized file type: {type(loaded)}")
        sys.exit(1)


# =============================================================================
# UTILITY
# =============================================================================

def _count_files(directory: str) -> int:
    """Count files in a directory."""
    try:
        return len([f for f in os.listdir(directory) if os.path.isfile(
            os.path.join(directory, f)
        )])
    except Exception:
        return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    args = parse_args()

    start = time.time()

    try:
        if args.mode == 'single':
            run_single(args)

        elif args.mode == 'study':
            run_study(args)

        elif args.mode == 'flyby_azimuth_sweep':
            run_flyby_azimuth_sweep(args)

        elif args.mode == 'launch_az_el_sweep':
            run_launch_az_el_sweep(args)

        elif args.mode == 'radial':
            run_radial(args)

        elif args.mode == 'plot':
            run_plot(args)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")


if __name__ == '__main__':
    main()
# ERNSTflyby — Satellite-Based Missile Depth Estimation

Simulation framework for estimating the 3-D depth of a ballistic missile as observed
by the [ERNST CubeSat](https://www.ernst-cubesat.de/) in low Earth orbit.
The satellite carries only a passive IR camera — no radar, no ranging sensor.
Depth is recovered purely from the apparent pixel motion of the target across frames.

---

## Background

When a LEO satellite flies over a ballistic missile, the missile traces a short arc
across the camera's focal plane. The angular direction to the target changes with
every frame, but the raw pixel observations carry no direct range information.

The core idea of **two-ray triangulation** is to treat each pixel observation as a
unit ray in space and intersect two such rays — one from the current frame and one
from a frame Δt seconds earlier — to find the closest-approach midpoint. Because
the satellite has moved by several kilometres between the two frames, the two rays
are no longer parallel, and their intersection gives an estimate of the missile's
3-D position.

This repository implements and compares four estimation methods across a systematic
parameter sweep over approach distance, missile speed, and flyby geometry.

---

## Installation

```bash
git clone https://github.com/valentinharwardt/ERNSTflyby.git
cd ERNSTflyby
pip install -r requirements.txt
pip install -e .
```

Python ≥ 3.9 required.

---

## Quick Start

```bash
# Single run: 200 km approach distance, 1000 m/s missile speed, 90° crossing angle
python main.py --mode single --distance 200 --speed 1000 --angle 90

# Quick 3×3 parameter study (distance × speed grid)
python main.py --mode study --quick

# Reload a saved run and regenerate all plots
python main.py --mode plot --results experiments/runs/<folder>/simulation_results.pkl
```

Results and plots are saved automatically to `experiments/runs/`.

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `single` | `single` / `study` / `flyby_azimuth_sweep` / `launch_az_el_sweep` / `plot` |
| `--distance` | `200` | Closest approach distance [km] |
| `--speed` | `1000` | Missile speed [m/s] |
| `--angle` | `90` | Flyby azimuth angle [°] |
| `--duration` | `100` | Simulation duration [s] |
| `--quick` | — | Use 3×3 grid instead of 7×7 for `--mode study` |
| `--plot-config` | `preview` | Plot quality: `preview` / `thesis` / `presentation` |
| `--results` | — | Path to `.pkl` file for `--mode plot` |

---

## Estimation Methods

All methods are **strictly causal**: estimates at time *t* only use observations
from times ≤ *t*.

| Name | Method | Notes |
|------|--------|-------|
| `two_ray` | Two-ray triangulation with Δt lookback | Baseline |
| `multi_ray` | Least-squares fit over a sliding window of observations | |
| `kalman` | Kalman filter with constant-velocity motion model | |
| `iterative_k1`–`k5` | Velocity-corrected triangulation (k = correction iterations) | Novel method; k=1 ≡ two_ray |

The **iterative velocity-corrected** method shifts the older ray origin by the
estimated missile velocity × Δt so that both rays refer to the same moment in time,
removing the leading-edge bias inherent in plain two-ray triangulation.

---

## Output Structure

```
experiments/runs/
  single_NNN/                      single run (d, v, φ, θ)
    simulation_results.pkl
    depth_estimates.csv
    plots/
  flyby_azimuth_sweep_NNN/         1-D azimuth sweep
    experiment_results.pkl
    plots/
  launch_az_el_sweep_NNN/          2-D azimuth × elevation sweep
    launch_az_el_sweep_results.pkl
    plots/
      sweep_azimuth.png
      hemisphere_heatmap.png
  study_NNN/                       3-D distance × speed parameter study
    experiment_results.pkl
    summary_table.csv
    plots/
      parameter_study/
      individual_runs/
```

`*.pkl` files contain the full `SimulationResults` / `ExperimentResults` objects
and can be reloaded with `pickle.load()` or via `--mode plot`.

---

## Project Structure

```
missile_fly_by_simulation/
  constants.py          single source of truth for all constants
  domain/               pure data classes (Satellite, Missile, Camera, ...)
  physics/              orbital propagation, attitude control
  sensing/              pinhole camera model, FOV geometry
  simulation/           Simulator pipeline (6-step: orbit → observations → estimates)
  estimation/           depth estimation methods (two_ray, multi_ray, kalman, iterative)
  experiments/          batch runner, scenario factory, result aggregation
  visualization/        plots (3 quality presets: preview / thesis / presentation)
main.py                 CLI entry point
diagnose.py             post-run statistics and diagnostics helper
```

---

## Key Parameters

All constants live in [`missile_fly_by_simulation/constants.py`](missile_fly_by_simulation/constants.py).

| Parameter | Value | Description |
|-----------|-------|-------------|
| Orbit altitude | ~517 km | Sun-synchronous LEO (apogee 520 km, perigee 515 km) |
| Inclination | 97.5° | Sun-synchronous |
| Camera resolution | 1024 × 720 px | |
| Horizontal FOV | 30° | → ~250 m/px at nadir from 500 km |
| Frame rate | 30 fps | |
| ADCS noise | 0.007° σ | Gaussian, per-frame attitude uncertainty |
| Pixel noise | 0.2 px σ | Gaussian centroid-localisation noise |
| Triangulation Δt offsets | [1, 5, 10, 20] s | Lookback times for two-ray estimates |

---

## List of Classes

| File | Class | Description |
|------|-------|-------------|
| **Domain** | | |
| domain/missile.py:35 | MissileState | Immutable snapshot of missile position + velocity at one point in time |
| domain/missile.py:122 | Missile | Container holding the missile's full history of MissileState snapshots |
| domain/satellite.py:41 | Attitude | Satellite orientation as 3 orthonormal vectors (right, up, forward) |
| domain/satellite.py:197 | SatelliteState | Immutable snapshot of satellite position, velocity, and attitude at one point in time |
| domain/satellite.py:315 | OrbitalElements | The 6 Keplerian parameters that define an orbit's size, shape, and orientation |
| domain/satellite.py:550 | CameraSpecification | Hardware spec sheet: resolution, FOV, FPS |
| domain/satellite.py:647 | SatelliteSpecification | Blueprint combining name, camera spec, and orbital elements — everything fixed about a satellite |
| domain/satellite.py:701 | Satellite | Container holding the satellite's spec + full history of SatelliteState snapshots |
| **Physics** | | |
| physics/attitude_dynamics.py:28 | NadirPointingController | Computes satellite attitude to point the camera at a target (or nadir as fallback) |
| physics/orbital_mechanics.py:39 | KeplerianOrbitPropagator | Computes satellite position/velocity over time using two-body orbital mechanics |
| **Sensing** | | |
| sensing/camera_model.py:26 | PinholeCameraModel | Projects 3D world points to 2D pixels, and back-projects pixels into 3D rays |
| **Simulation** | | |
| simulation/scenario.py:40 | SimulationScenario | Input config: satellite spec, missile start pos/velocity, duration, noise settings — the "recipe" for one run |
| simulation/simulator.py:55 | Simulator | Orchestrates all modules step-by-step to produce a full simulation run |
| simulation/results.py:32 | Observation | One camera frame: pixel hit + satellite state at that moment (raw sensor input) |
| simulation/results.py:59 | DepthEstimate | One depth result from an estimator: estimated distance, true distance, and error (algorithm output) |
| simulation/results.py:94 | SimulationResults | Full output bundle: trajectories, all observations, all depth estimates from all methods |
| **Estimation** | | |
| estimation/two_ray_triangulation.py:41 | TwoRayTriangulationEstimator | Baseline: triangulates depth from exactly 2 observations separated by Δt |
| estimation/multi_ray_least_squares.py:32 | MultiRayLeastSquaresEstimator | Fits a 3D point to N observation rays via least-squares — more robust than two-ray |
| estimation/kalman_constant_velocity.py:35 | KalmanDepthTracker | Tracks depth over time with a 1D Kalman filter using a constant-velocity motion model |
| estimation/iterative_velocity_triangulation.py:49 | IterativeVelocityTriangulator | Triangulation that iteratively corrects for the missile moving during the observation window |
| **Experiments** | | |
| experiments/scenario_factory.py:69 | ScenarioFactory | Builds SimulationScenario objects from intuitive physical params (distance, speed, angle) |
| experiments/batch_runner.py:184 | BatchRunner | Runs all (distance × speed × angle) combinations in parallel and saves results to disk |
| experiments/experiment_results.py:30 | RunSummary | Lightweight stats from one run (RMSE per method, num observations, runtime) — no full trajectories |
| experiments/experiment_results.py:109 | ExperimentResults | All RunSummary objects from a full DSA parameter study, plus selected full results |
| experiments/experiment_results.py:601 | AngularStudyResults | Like ExperimentResults but for a 2D azimuth × elevation sweep at fixed distance/speed |
| **Visualization** | | |
| visualization/plot_config.py:19 | PlotConfig | Central config for figure sizes, font sizes, and colors — change once, applies to all plots |



---


## License

MIT — see [LICENSE](LICENSE).

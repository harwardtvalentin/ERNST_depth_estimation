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

## License

MIT — see [LICENSE](LICENSE).

"""
Trajectory and geometry plots for simulation results.

This module contains all Category 2 plots - visualizations of
the satellite and missile trajectories and observation geometry.

Functions
---------
plot_3d_orbit
    3D plot of Earth, satellite orbit and missile trajectory
plot_pixel_track
    Missile pixel position over time in camera frame
plot_relative_geometry
    Distance and triangulation angle over time
plot_ground_track
    Satellite ground track on 2D world map (requires cartopy)
plot_depth_comparison  # ← ADD THIS LINE
    Estimated vs true depth over time for all methods
plot_all_trajectory
    Convenience: generate all trajectory plots at once
"""

import os
from typing import Optional, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from missile_fly_by_simulation.simulation.results import SimulationResults
from missile_fly_by_simulation.visualization.plot_config import PlotConfig
from missile_fly_by_simulation.constants import (
    EARTH_RADIUS_KM,
    MIN_TRIANGULATION_ANGLE_DEG,
)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _sort_methods_for_plotting(methods):
    """Iterative (noisy) methods drawn first (background), cleaner methods on top."""
    iterative_order = ['iterative_k5', 'iterative_k4', 'iterative_k3',
                       'iterative_k2', 'iterative_k1']
    front = [m for m in iterative_order if m in methods]
    back  = [m for m in methods if m not in iterative_order]
    return front + back


def _annotate_scenario_params(fig, results, multiline=False):
    """Add scenario parameters as a small note below the figure."""
    dist  = getattr(results.scenario, '_closest_approach_distance', None)
    speed = getattr(results.scenario, '_missile_speed', None)
    angle = getattr(results.scenario, '_crossing_angle_deg', None)
    elev  = getattr(results.scenario, '_elevation_angle_deg', None)
    if elev is None and angle is not None:
        elev = 0.0  # default: purely azimuthal (horizontal) launch

    if dist is None and speed is None and angle is None:
        return

    # Observation duration (time missile was in FOV)
    duration = None
    if hasattr(results, 'observations') and results.observations:
        duration = (
            results.observations[-1].timestamp - results.observations[0].timestamp
        ).total_seconds()

    parts = []
    if dist  is not None: parts.append(f"closest approach={dist/1e3:.0f} km")
    if speed is not None: parts.append(f"missile speed={speed:.0f} m/s")
    if angle is not None: parts.append(f"crossing angle azimuth={angle:.0f} deg")
    if elev  is not None: parts.append(f"crossing angle elevation={elev:.0f} deg")

    if multiline:
        if duration is not None: parts.append(f"duration={duration:.0f}s")
        text = "\n".join(parts)
    else:
        text = "  |  ".join(parts)
        if duration is not None:
            text += f"  [duration={duration:.0f}s]"

    fig.text(
        0.5, -0.02,
        text,
        ha='center', va='top',
        fontsize=16, style='italic', color='#555555',
        transform=fig.transFigure,
    )


# =============================================================================
# INDIVIDUAL PLOT FUNCTIONS
# =============================================================================

def plot_3d_orbit(
    results: SimulationResults,
    save_path: str,
    config: Optional[PlotConfig] = None,
    num_points: int = 500,
):
    """
    3D plot of Earth, satellite orbit, and missile trajectory.

    Shows the full observation geometry in 3D space.
    Color gradient on satellite orbit shows time direction.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    config : PlotConfig, optional
        Visual configuration
    num_points : int, optional
        Number of points to plot (subsampled for performance), default 500
    """
    config = config or PlotConfig()

    # Extract positions (subsampled)
    sat_states = results.satellite.states
    step = max(1, len(sat_states) // num_points)
    sat_positions = np.array([s.position for s in sat_states[::step]])

    # Missile positions
    miss_states = results.missile.states
    step_m = max(1, len(miss_states) // num_points)
    miss_positions = np.array([s.position for s in miss_states[::step_m]])

    # Scale to km for readability
    sat_pos_km = sat_positions / 1e3
    miss_pos_km = miss_positions / 1e3
    earth_radius_km = EARTH_RADIUS_KM

    with plt.style.context(config.style):
        fig = plt.figure(figsize=config.figsize_3d)
        ax = fig.add_subplot(111, projection='3d')

        # Draw Earth as wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_earth = earth_radius_km * np.outer(np.cos(u), np.sin(v))
        y_earth = earth_radius_km * np.outer(np.sin(u), np.sin(v))
        z_earth = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(
            x_earth, y_earth, z_earth,
            color='#4FC3F7',
            alpha=0.15,
            linewidth=0,
        )
        ax.plot_wireframe(
            x_earth, y_earth, z_earth,
            color='#0288D1',
            alpha=0.1,
            linewidth=0.3,
            rstride=3, cstride=3,
        )

        # Satellite orbit (color gradient: blue→red over time)
        n = len(sat_pos_km)
        for i in range(n - 1):
            frac = i / n
            color = plt.cm.coolwarm(frac)
            ax.plot(
                sat_pos_km[i:i+2, 0],
                sat_pos_km[i:i+2, 1],
                sat_pos_km[i:i+2, 2],
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )

        # Missile trajectory
        ax.plot(
            miss_pos_km[:, 0],
            miss_pos_km[:, 1],
            miss_pos_km[:, 2],
            color=config.color_missile,
            linewidth=2.0,
            label='Missile trajectory',
            zorder=5,
        )

        # Mark start and end of satellite orbit
        ax.scatter(
            *sat_pos_km[0],
            color='blue', s=50, zorder=10, label='Satellite start'
        )
        ax.scatter(
            *sat_pos_km[-1],
            color='red', s=50, zorder=10, label='Satellite end'
        )

        # Mark missile start
        ax.scatter(
            *miss_pos_km[0],
            color=config.color_missile,
            s=80, marker='x', zorder=10,
            label='Missile start'
        )

        # Auto-scale axes
        all_pos = np.vstack([sat_pos_km, miss_pos_km])
        center = all_pos.mean(axis=0)
        max_range = np.max(np.abs(all_pos - center)) * 1.1
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_xlabel('X [km]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Y [km]', fontsize=config.fontsize_labels)
        ax.set_zlabel('Z [km]', fontsize=config.fontsize_labels)
        ax.set_title(
            f'3D Orbit: {results.satellite.spec.name}',
            fontsize=config.fontsize_title
        )
        ax.tick_params(labelsize=config.fontsize_ticks - 2)
        ax.legend(fontsize=config.fontsize_legend, loc='upper center')

        # Add colorbar for time
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Time (blue=start, red=end)', fontsize=config.fontsize_ticks)

        filename = config.save_filename('3d_orbit')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)

        # Crop internal whitespace (3D axes leave large empty margins)
        if config.file_format == 'png':
            import matplotlib.image as mpimg
            img = mpimg.imread(filepath)   # float32 H×W×(3|4)
            rgb = img[:, :, :3]
            non_white = np.any(rgb < 0.95, axis=2)
            rows = np.where(np.any(non_white, axis=1))[0]
            cols = np.where(np.any(non_white, axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                pad = 8
                r0 = max(0, rows[0] - pad)
                r1 = min(img.shape[0], rows[-1] + pad)
                c0 = max(0, cols[0] - pad)
                c1 = min(img.shape[1], cols[-1] + pad)
                mpimg.imsave(filepath, img[r0:r1, c0:c1])

        print(f"  [OK] Saved: {filepath}")


def plot_pixel_track(
    results: SimulationResults,
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Plot missile pixel position over time inside camera frame.

    Shows what the satellite camera actually sees - the missile
    moving across the image frame over the observation window.
    Color gradient shows time direction.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()

    if not results.observations:
        print("  [!]No observations for pixel track plot")
        return

    # Extract pixel coordinates
    u_coords = np.array([obs.pixel[0] for obs in results.observations])
    v_coords = np.array([obs.pixel[1] for obs in results.observations])

    # Camera resolution
    width, height = results.scenario.satellite_spec.camera.resolution

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_pixel_track)

        # Camera frame boundary
        rect = plt.Rectangle(
            (0, 0), width, height,
            linewidth=2,
            edgecolor='black',
            facecolor='#F5F5F5',
        )
        ax.add_patch(rect)

        # Pixel track with time color gradient
        n = len(u_coords)
        colors = plt.cm.plasma(np.linspace(0, 1, n))

        # Plot as colored segments
        for i in range(n - 1):
            ax.plot(
                u_coords[i:i+2],
                v_coords[i:i+2],
                color=colors[i],
                linewidth=1.5,
                alpha=0.8,
            )

        # Mark start and end
        ax.scatter(
            u_coords[0], v_coords[0],
            color='blue', s=80, zorder=10,
            label='First detection', marker='o'
        )
        ax.scatter(
            u_coords[-1], v_coords[-1],
            color='red', s=80, zorder=10,
            label='Last detection', marker='s'
        )

        # Crosshair at image center
        ax.axhline(y=height/2, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axvline(x=width/2, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.set_xlim(-width * 0.05, width * 1.05)
        ax.set_ylim(-height * 0.05, height * 1.05)
        ax.set_xlabel('Pixel u (horizontal)', fontsize=config.fontsize_labels)
        ax.set_ylabel('Pixel v (vertical)', fontsize=config.fontsize_labels)
        ax.set_title(
            f'Missile Pixel Track in Camera Frame\n'
            f'Resolution: {width}×{height} px  |  '
            f'{len(results.observations)} detections',
            fontsize=config.fontsize_title
        )
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.legend(fontsize=config.fontsize_legend, loc='upper center')
        ax.set_aspect('equal')

        # Colorbar for time
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Time (early→late)', fontsize=config.fontsize_ticks)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('pixel_track')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_relative_geometry(
    results: SimulationResults,
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Plot relative geometry between satellite and missile over time.

    Two-panel figure:
    - Top: Distance satellite→missile over time
    - Bottom: Triangulation angle between ray pairs over time

    Shows WHY accuracy changes over time - relates geometry
    directly to estimation performance.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()

    if not results.observations:
        print("  [!]No observations for relative geometry plot")
        return

    ref_time = results.scenario.start_time

    # Extract data from observations
    times = [
        (obs.timestamp - ref_time).total_seconds()
        for obs in results.observations
    ]
    distances = [obs.true_depth / 1e3 for obs in results.observations]  # km

    # Compute triangulation angles using the scenario's first time offset (Δt),
    # matching exactly the pairing used by the Two-Ray estimator.
    import bisect as _bisect

    dt_plot = float(results.scenario.depth_time_offsets[0])  # e.g. 1.0 s

    angles = []
    angle_times = []
    angles_true = []   # ground truth: computed from true missile positions

    from missile_fly_by_simulation.sensing import PinholeCameraModel
    camera = PinholeCameraModel(results.scenario.satellite_spec.camera)

    # Build timestamp → missile position lookup for ground truth
    missile_pos_lookup = {
        s.timestamp: s.position for s in results.missile.states
    }

    obs_list = results.observations
    t0_obs = obs_list[0].timestamp
    obs_ts = [(o.timestamp - t0_obs).total_seconds() for o in obs_list]

    for i, obs1 in enumerate(obs_list):
        t1_s = obs_ts[i]
        target_s = t1_s - dt_plot

        if target_s < 0:
            continue

        # Binary search for the observation closest to (t1 - dt_plot)
        idx = _bisect.bisect_left(obs_ts, target_s)
        best_obs2 = None
        best_diff = float('inf')
        for ci in (idx - 1, idx):
            if 0 <= ci < i:  # must be strictly in the past
                diff = abs(obs_ts[ci] - target_s)
                if diff < best_diff:
                    best_diff = diff
                    best_obs2 = obs_list[ci]

        if best_obs2 is None or best_diff > 0.5:  # 0.5 s tolerance
            continue

        # Noisy observed rays (from measured pixels + noisy attitude)
        ray1_cam = camera.pixel_to_ray(obs1.pixel)
        ray2_cam = camera.pixel_to_ray(best_obs2.pixel)

        ray1_world = obs1.satellite_state.attitude.satellite_to_world(ray1_cam)
        ray2_world = best_obs2.satellite_state.attitude.satellite_to_world(ray2_cam)

        dot = np.clip(np.dot(ray1_world, ray2_world), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(abs(dot)))
        angles.append(angle_deg)
        angle_times.append((obs1.timestamp - ref_time).total_seconds())

        # Ground truth rays (from true missile positions, no noise)
        mpos1 = missile_pos_lookup.get(obs1.timestamp)
        mpos2 = missile_pos_lookup.get(best_obs2.timestamp)
        if mpos1 is not None and mpos2 is not None:
            true_ray1 = mpos1 - obs1.satellite_state.position
            true_ray1 /= np.linalg.norm(true_ray1)
            true_ray2 = mpos2 - best_obs2.satellite_state.position
            true_ray2 /= np.linalg.norm(true_ray2)
            dot_true = np.clip(np.dot(true_ray1, true_ray2), -1.0, 1.0)
            angles_true.append(np.degrees(np.arccos(abs(dot_true))))
        else:
            angles_true.append(np.nan)

    with plt.style.context(config.style):
        fig, axes = plt.subplots(2, 1, figsize=config.figsize_tall, sharex=True)

        # Top: Distance
        axes[0].plot(
            times,
            distances,
            color=config.color_satellite,
            linewidth=config.line_width,
        )
        axes[0].set_ylabel('Distance [km]', fontsize=config.fontsize_labels)
        axes[0].set_title(
            'Relative Geometry: Satellite → Missile',
            fontsize=config.fontsize_title
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=config.fontsize_ticks)

        # Mark closest approach
        if distances:
            min_idx = np.argmin(distances)
            axes[0].axvline(
                x=times[min_idx],
                color=config.color_true,
                linestyle='--',
                linewidth=config.line_width_reference,
                alpha=0.7,
                label=f'Closest approach: {distances[min_idx]:.1f} km',
            )
            axes[0].legend(fontsize=config.fontsize_legend)

        # Bottom: Triangulation angle
        if angles:
            axes[1].plot(
                angle_times,
                angles,
                color=config.color_missile,
                linewidth=config.line_width,
                alpha=0.6,
                label='Measured (noisy)',
                zorder=2,
            )
            # Ground truth LOS angle (noiseless)
            if angles_true and not all(np.isnan(angles_true)):
                axes[1].plot(
                    angle_times,
                    angles_true,
                    color='steelblue',
                    linewidth=config.line_width * 1.5,
                    label='Ground truth',
                    zorder=3,
                )
            # Threshold line — pairs below this are rejected by Two-Ray estimator
            axes[1].axhline(
                y=MIN_TRIANGULATION_ANGLE_DEG,
                color='red',
                linestyle='--',
                linewidth=config.line_width_reference,
                alpha=0.8,
                label=f'Rejection threshold ({MIN_TRIANGULATION_ANGLE_DEG}°)',
            )
            axes[1].set_ylabel(
                f'Ray Angle [degrees]  (Δt = {dt_plot:.0f} s)',
                fontsize=config.fontsize_labels,
            )
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(labelsize=config.fontsize_ticks)

            axes[1].legend(fontsize=config.fontsize_legend)

        axes[1].set_xlabel('Time [seconds]', fontsize=config.fontsize_labels)

        if config.tight_layout:
            fig.tight_layout()


        filename = config.save_filename('relative_geometry')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_ground_track(
    results: SimulationResults,
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Plot satellite ground track on 2D world map.

    Requires cartopy. If not installed, saves a simplified
    lat/lon plot without the map background.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()

    # Convert ECI positions to lat/lon
    sat_states = results.satellite.states
    step = max(1, len(sat_states) // 500)
    positions = np.array([s.position for s in sat_states[::step]])

    lats, lons = _eci_to_latlon(positions)

    try:
        _plot_ground_track_cartopy(lats, lons, results, save_path, config)
    except ImportError:
        print("  [!]cartopy not installed - using simple lat/lon plot")
        _plot_ground_track_simple(lats, lons, results, save_path, config)


def plot_depth_comparison(
    results: SimulationResults,
    save_path: str,
    config: Optional[PlotConfig] = None,
    methods: Optional[List[str]] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
):
    """
    Plot estimated vs true depth over time — 2×2 panel layout.

    Four panels (each with independent x-axis):
      (0,0) Two-Ray methods (one line per Δt offset)
      (0,1) Multi-Ray methods (one line per time window)
      (1,0) Iterative Velocity Triangulation (k=1…5)
      (1,1) Kalman Filter

    Every panel also shows the Two-Ray Δt=1s line as a prominent reference
    (bright red) and the ground-truth depth as a dashed line.

    If xlim is provided the plot is saved as depth_comparison_zoomed and
    each panel's y-axis is auto-scaled to the visible data within xlim.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    config : PlotConfig, optional
        Visual configuration
    methods : list of str, optional
        Methods to plot, default all available
    xlim : list of float, optional
        [t_start, t_end] for zoomed version; y-axis auto-scales per panel
    ylim : list of float, optional
        Ignored (kept for API compatibility) — y is auto-scaled from data
    """
    config = config or PlotConfig()
    all_methods = list(methods or results.available_methods)

    if not results.observations:
        print("  [!]No observations for depth comparison plot")
        return

    t0 = results.observations[0].timestamp
    times = np.array([(obs.timestamp - t0).total_seconds()
                      for obs in results.observations])
    true_depths_km = np.array([obs.true_depth for obs in results.observations]) / 1e3

    # Group methods into panels
    two_ray_methods   = [m for m in all_methods if m == 'two_ray']
    multi_ray_methods = [m for m in all_methods if m == 'multi_ray']
    iterative_methods = [m for m in all_methods if m.startswith('iterative')]
    kalman_methods    = [m for m in all_methods if m == 'kalman']

    # Reference: two_ray Δt=1s for overlay in every panel
    REF_COLOR = '#FF1744'
    def _get_ref_line():
        ests = results.depth_estimates.get('two_ray', [])
        sub = [e for e in ests if e.time_offset == 1.0]
        if not sub:
            return None, None
        ttd = {}
        for e in sub:
            t = (e.timestamp - t0).total_seconds()
            ttd.setdefault(t, []).append(e.estimated_depth)
        et = np.array(sorted(ttd.keys()))
        ed = np.array([np.mean(ttd[tt]) / 1e3 for tt in et])
        return et, ed

    ref_t, ref_d = _get_ref_line()

    # 2×2 layout: (0,0) two_ray | (0,1) multi_ray | (1,0) iterative | (1,1) kalman
    panels = [
        ((0, 0), 'Two-Ray (per Δt offset)',                  two_ray_methods,   plt.cm.Blues),
        ((0, 1), 'Multi-Ray (per Time Window)',               multi_ray_methods, plt.cm.Oranges),
        ((1, 0), 'Iterative Velocity Triangulation (k=1…5)', iterative_methods, None),
        ((1, 1), 'Kalman Filter (Const. Velocity)',           kalman_methods,    None),
    ]

    fig_w = config.figsize_wide[0]
    # 16:9 per panel → total height = fig_w * 9/16
    fig_h = fig_w * 9.0 / 16.0

    with plt.style.context(config.style):
        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))

        for (row, col), title, panel_methods, cmap in panels:
            ax = axes[row, col]

            # Reference line (two_ray Δt=1s) — pushed to background
            if ref_t is not None:
                ax.plot(ref_t, ref_d,
                        color=REF_COLOR,
                        linewidth=config.line_width * 0.5,
                        label='Two-Ray Δt=1s (ref)',
                        zorder=1, alpha=0.9)

            # Ground truth
            ax.plot(times, true_depths_km,
                    color=config.color_true,
                    linewidth=config.line_width * 2,
                    label='True Depth',
                    zorder=10, linestyle='--', alpha=0.9)

            for method in panel_methods:
                estimates = results.depth_estimates.get(method, [])
                if not estimates:
                    continue

                time_offsets = sorted({
                    est.time_offset for est in estimates
                    if est.time_offset is not None
                })

                if not time_offsets:
                    # Single-line methods: Kalman and iterative_k*
                    # iterative panel: k=1 (front) → k=5 (back), zorder 8..4
                    if method.startswith('iterative'):
                        try:
                            k = int(method[-1])
                            method_zorder = 9 - k   # k=1→8, k=2→7, …, k=5→4
                        except (ValueError, IndexError):
                            method_zorder = 5
                    else:
                        method_zorder = 5  # Kalman

                    ttd = {}
                    for est in estimates:
                        t = (est.timestamp - t0).total_seconds()
                        ttd.setdefault(t, []).append(est.estimated_depth)
                    est_t = np.array(sorted(ttd.keys()))
                    est_d = np.array([np.mean(ttd[tt]) / 1e3 for tt in est_t])
                    ax.plot(est_t, est_d,
                            color=config.method_color(method),
                            linewidth=config.line_width,
                            label=config.method_label(method),
                            zorder=method_zorder,
                            alpha=0.8)
                else:
                    n_offsets = len(time_offsets)
                    for i, offset in enumerate(time_offsets):
                        # Skip Δt=1s in two_ray panel — already shown as reference
                        if method == 'two_ray' and offset == 1.0:
                            continue
                        frac  = i / max(1, n_offsets - 1)
                        color = cmap(0.35 + 0.65 * frac)

                        off_ests = [e for e in estimates if e.time_offset == offset]
                        ttd = {}
                        for est in off_ests:
                            t = (est.timestamp - t0).total_seconds()
                            ttd.setdefault(t, []).append(est.estimated_depth)
                        est_t = np.array(sorted(ttd.keys()))
                        est_d = np.array([np.mean(ttd[tt]) / 1e3 for tt in est_t])

                        lbl = f'Δt={offset:.0f}s' if method == 'two_ray' else f'W={offset:.0f}s'
                        ax.plot(est_t, est_d, color=color,
                                linewidth=config.line_width, label=lbl, alpha=0.75,
                                zorder=5)

            ax.set_ylabel('Depth [km]', fontsize=config.fontsize_labels)
            ax.set_xlabel('Time [seconds]', fontsize=config.fontsize_labels)
            ax.set_title(title, fontsize=config.fontsize_title)
            ax.legend(fontsize=7, ncol=3, loc='upper center')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=config.fontsize_ticks)

            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])
                # Auto-scale y to visible data within xlim
                y_max = 0.0
                for line in ax.lines:
                    xd = np.array(line.get_xdata(), dtype=float)
                    yd = np.array(line.get_ydata(), dtype=float)
                    mask = (xd >= xlim[0]) & (xd <= xlim[1])
                    if mask.any():
                        valid = yd[mask][np.isfinite(yd[mask])]
                        if len(valid) > 0:
                            y_max = max(y_max, float(np.nanmax(valid)))
                if y_max > 0:
                    ax.set_ylim(0, y_max * 1.1)

        title = ('Depth Estimation Comparison (Zoomed)'
                 if xlim else 'Depth Estimation Comparison')
        fig.suptitle(title, fontsize=config.fontsize_title + 2)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename(
            'depth_comparison_zoomed' if xlim else 'depth_comparison'
        )
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def _plot_ground_track_cartopy(lats, lons, results, save_path, config):
    """Ground track with cartopy world map."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    with plt.style.context(config.style):
        fig = plt.figure(figsize=config.figsize_single)
        ax = fig.add_subplot(
            1, 1, 1, projection=ccrs.PlateCarree()
        )

        # Map features
        ax.add_feature(cfeature.OCEAN, facecolor='#AED6F1', alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor='#A9DFBF', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)

        # Satellite ground track (color by time)
        n = len(lats)
        for i in range(n - 1):
            frac = i / n
            color = plt.cm.coolwarm(frac)
            ax.plot(
                lons[i:i+2], lats[i:i+2],
                color=color, linewidth=2.0,
                transform=ccrs.PlateCarree(),
            )

        # Missile ground position (first observation)
        if results.observations:
            miss_pos = results.observations[0].true_position
            miss_lats, miss_lons = _eci_to_latlon(miss_pos[np.newaxis, :])
            ax.scatter(
                miss_lons[0], miss_lats[0],
                color=config.color_missile, s=100,
                marker='x', zorder=10,
                label='Missile position',
                transform=ccrs.PlateCarree(),
            )

        # Auto-zoom to relevant area with 10% padding
        lat_margin = (max(lats) - min(lats)) * 0.1 + 2
        lon_margin = (max(lons) - min(lons)) * 0.1 + 2
        ax.set_extent([
            min(lons) - lon_margin,
            max(lons) + lon_margin,
            max(min(lats) - lat_margin, -90),
            min(max(lats) + lat_margin, 90),
        ], crs=ccrs.PlateCarree())

        ax.set_title(
            f'Satellite Ground Track: {results.satellite.spec.name}',
            fontsize=config.fontsize_title
        )
        ax.legend(fontsize=config.fontsize_legend, loc='upper center')

        filename = config.save_filename('ground_track')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results, multiline=True)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def _plot_ground_track_simple(lats, lons, results, save_path, config):
    """Simple lat/lon ground track without cartopy."""
    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_single)

        n = len(lats)
        colors = plt.cm.coolwarm(np.linspace(0, 1, n))

        for i in range(n - 1):
            ax.plot(lons[i:i+2], lats[i:i+2], color=colors[i], linewidth=1.5)

        if results.observations:
            miss_pos = results.observations[0].true_position
            miss_lats, miss_lons = _eci_to_latlon(miss_pos[np.newaxis, :])
            ax.scatter(
                miss_lons[0], miss_lats[0],
                color=config.color_missile, s=100, marker='x',
                zorder=10, label='Missile position'
            )

        ax.set_xlabel('Longitude [°]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Latitude [°]', fontsize=config.fontsize_labels)
        ax.set_title(
            f'Satellite Ground Track: {results.satellite.spec.name}\n'
            f'(install cartopy for world map background)',
            fontsize=config.fontsize_title
        )
        ax.legend(fontsize=config.fontsize_legend, loc='upper center')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=config.fontsize_ticks)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('ground_track')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results, multiline=True)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


# =============================================================================
# COORDINATE HELPER
# =============================================================================

def _eci_to_latlon(positions: np.ndarray):
    """
    Convert ECI positions to latitude/longitude.

    Simple conversion assuming J2000 epoch (ignores Earth rotation
    for visualization purposes - good enough for ground track plots).

    Parameters
    ----------
    positions : ndarray of shape (N, 3)
        ECI positions [m]

    Returns
    -------
    lats : ndarray of shape (N,)
        Latitudes [degrees]
    lons : ndarray of shape (N,)
        Longitudes [degrees]
    """
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    r = np.linalg.norm(positions, axis=1)

    lats = np.degrees(np.arcsin(z / r))
    lons = np.degrees(np.arctan2(y, x))

    return lats, lons


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def plot_all_trajectory(
    results: SimulationResults,
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Generate all trajectory and geometry plots at once.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save all figures
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    os.makedirs(save_path, exist_ok=True)

    print(f"\nGenerating trajectory plots -> {save_path}")

    plot_3d_orbit(results, save_path, config)
    plot_pixel_track(results, save_path, config)
    plot_relative_geometry(results, save_path, config)
    plot_ground_track(results, save_path, config)
    plot_depth_comparison(results, save_path, config)

    # Zoomed plot: adaptive window based on actual depth range.
    # Show only the region where true_depth < ZOOM_FACTOR * min_depth.
    # This automatically scales with both closest-approach distance and flyby speed.
    if results.observations:
        obs_times_s = [
            (obs.timestamp - results.observations[0].timestamp).total_seconds()
            for obs in results.observations
        ]
        obs_depths_km = [obs.true_depth / 1e3 for obs in results.observations]
        min_depth_km = min(obs_depths_km)

        ZOOM_FACTOR = 5.0  # show region where depth < 5 × closest approach
        depth_threshold_km = ZOOM_FACTOR * min_depth_km

        # Find time indices where depth is within threshold
        in_range = [d <= depth_threshold_km for d in obs_depths_km]
        if any(in_range):
            t_in = [t for t, m in zip(obs_times_s, in_range) if m]
            t_start, t_end = min(t_in), max(t_in)
            margin = max((t_end - t_start) * 0.15, 1.0)  # at least 1s margin
            xlim_zoom = [t_start - margin, t_end + margin]
            ylim_zoom = [0.0, depth_threshold_km]
        else:
            # Fallback: ±60s around closest approach
            min_idx = int(np.argmin(obs_depths_km))
            t_closest = obs_times_s[min_idx]
            xlim_zoom = [t_closest - 60.0, t_closest + 60.0]
            ylim_zoom = [0.0, min_depth_km * ZOOM_FACTOR]

        plot_depth_comparison(results, save_path, config, xlim=xlim_zoom)

    print(f"  [OK] All trajectory plots saved!\n")
"""
Statistical plots for depth estimation results.

This module contains all Category 1 plots - statistical analysis
of depth estimation performance for a single simulation run.

All functions take SimulationResults and PlotConfig as input
and save figures to disk.

Functions
---------
plot_error_histogram
    Error distribution for all methods (overlaid)
plot_error_vs_time_offset
    RMSE as function of time offset Δt
plot_error_over_time
    Depth error over simulation time
plot_estimated_vs_true_depth
    Scatter: estimated vs true depth
plot_method_comparison_bar
    Bar chart comparing RMSE across methods
plot_triangulation_gap_vs_error
    Scatter: triangulation gap vs error
plot_all_statistical
    Convenience: generate all 6 plots at once
"""

import os
from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (safe for batch runs)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import timedelta

from missile_fly_by_simulation.simulation.results import SimulationResults
from missile_fly_by_simulation.visualization.plot_config import PlotConfig


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _sort_methods_for_plotting(methods):
    """
    Return methods in draw order so iterative (noisy) ones are rendered first
    (background) and cleaner methods render on top.

    Order: iterative_k5 → k4 → k3 → k2 → k1, then all other methods.
    Within the non-iterative group the original order is preserved.
    """
    iterative_order = ['iterative_k5', 'iterative_k4', 'iterative_k3',
                       'iterative_k2', 'iterative_k1']
    front = [m for m in iterative_order if m in methods]
    back  = [m for m in methods if m not in iterative_order]
    return front + back


def _annotate_scenario_params(fig, results):
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

def plot_error_histogram(
    results: SimulationResults,
    save_path: str,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Plot depth error distribution for all methods (overlaid histograms).

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    methods : list of str, optional
        Methods to plot, default all available
    config : PlotConfig, optional
        Visual configuration, default PlotConfig()
    """
    config = config or PlotConfig()
    methods = methods or results.available_methods

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_single)

        def _step(ax, errors, color, label):
            counts, bin_edges = np.histogram(errors, bins=60)
            freqs = counts / counts.sum()
            x = np.append(bin_edges[:-1], bin_edges[-1])
            y = np.append(freqs, 0.0)
            ax.step(x, y, where='post', color=color, label=label, linewidth=1.5)

        for method in _sort_methods_for_plotting(methods):
            if method in ('two_ray', 'multi_ray'):
                estimates = results.depth_estimates.get(method, [])
                if not estimates:
                    continue
                offsets = sorted({e.time_offset for e in estimates
                                  if e.time_offset is not None})
                n    = len(offsets)
                cmap = plt.cm.Blues if method == 'two_ray' else plt.cm.Oranges
                for i, offset in enumerate(offsets):
                    sub    = [e for e in estimates if e.time_offset == offset]
                    errors = np.array([e.error for e in sub])
                    if len(errors) == 0:
                        continue
                    col   = cmap(0.35 + 0.65 * i / max(n - 1, 1))
                    label = (f'Two-Ray Δt={offset:.0f}s' if method == 'two_ray'
                             else f'Multi-Ray W={offset:.0f}s')
                    _step(ax, errors, col, label)
            else:
                errors = results.get_errors_for_method(method)
                if len(errors) == 0:
                    continue
                _step(ax, errors, config.method_color(method),
                      config.method_label(method))

        # Reference line at zero error
        ax.axvline(
            x=0,
            color='black',
            linestyle='--',
            linewidth=config.line_width_reference,
            label='Zero error'
        )

        ax.set_xlabel('Depth Error [m]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Relative Frequency', fontsize=config.fontsize_labels)
        ax.set_title('Depth Estimation Error Distribution', fontsize=config.fontsize_title)
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.legend(fontsize=config.fontsize_legend)
        ax.grid(True, alpha=0.3)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('error_histogram')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_error_vs_time_offset(
    results: SimulationResults,
    save_path: str,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Plot RMSE as function of time offset Δt for all methods.

    This is the KEY result plot - shows how accuracy depends
    on the time between the two observations used.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    methods : list of str, optional
        Methods to plot
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    methods = methods or results.available_methods

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_single)

        for method in methods:
            estimates = results.depth_estimates.get(method, [])
            if not estimates:
                continue

            # Group by time offset
            offset_groups = {}
            for est in estimates:
                if est.time_offset is not None:
                    offset = round(est.time_offset, 1)
                    if offset not in offset_groups:
                        offset_groups[offset] = []
                    offset_groups[offset].append(est.error)

            if not offset_groups:
                continue

            offsets = sorted(offset_groups.keys())
            rmse_values = [
                np.sqrt(np.mean(np.array(offset_groups[o]) ** 2))
                for o in offsets
            ]
            std_values = [
                np.std(offset_groups[o])
                for o in offsets
            ]

            ax.plot(
                offsets,
                rmse_values,
                color=config.method_color(method),
                label=config.method_label(method),
                linewidth=config.line_width,
                marker='o',
                markersize=config.marker_size + 2,
            )

            # Uncertainty band
            ax.fill_between(
                offsets,
                np.array(rmse_values) - np.array(std_values),
                np.array(rmse_values) + np.array(std_values),
                color=config.method_color(method),
                alpha=config.alpha_uncertainty_band,
            )

        ax.set_xlabel('Time Offset Δt [seconds]', fontsize=config.fontsize_labels)
        ax.set_ylabel('RMSE [m]', fontsize=config.fontsize_labels)
        ax.set_title(
            'Depth Estimation Accuracy vs Time Offset',
            fontsize=config.fontsize_title
        )
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.legend(fontsize=config.fontsize_legend)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('error_vs_time_offset')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_error_over_time(
    results: SimulationResults,
    save_path: str,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Plot depth error over simulation time — 2×2 panel layout.

    Four panels, each with independent x-axis:
      (0,0) Two-Ray: one line per Δt offset (legend shows "Δt=Xs")
      (0,1) Multi-Ray: one line per window size (legend shows "W=Xs")
      (1,0) Kalman Filter
      (1,1) Iterative Velocity Triangulation k=1…5

    Each line is the rolling-average error over time.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    methods : list of str, optional
        Ignored (layout is fixed); kept for API compatibility.
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()

    if not results.observations:
        print("  [!]No observations available for error_over_time plot")
        return

    ref_time = results.scenario.start_time

    def _smooth_and_plot(ax, times_list, errors_list, color, label, lw=None):
        """Sort, rolling-average, and plot one error series."""
        lw = lw if lw is not None else config.line_width
        pairs = sorted(zip(times_list, errors_list))
        ts = [p[0] for p in pairs]
        es = [p[1] for p in pairs]
        window = max(1, len(es) // 50)
        es_sm = np.convolve(es, np.ones(window) / window, mode='valid')
        ts_sm = ts[window // 2: window // 2 + len(es_sm)]
        ax.step(ts_sm, es_sm, where='post', color=color, linewidth=lw, label=label)

    def _finish_ax(ax, title):
        ax.axhline(y=0, color='black', linestyle='--',
                   linewidth=config.line_width_reference, alpha=0.5,
                   label='Zero Error (ground truth)')
        ax.set_xlabel('Time [seconds]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Depth Error [m]', fontsize=config.fontsize_labels)
        ax.set_title(title, fontsize=config.fontsize_title)
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.legend(fontsize=config.fontsize_legend - 1, ncol=2)
        ax.grid(True, alpha=0.3)

    panel_h = config.figsize_single[1]
    fig_w   = config.figsize_single[0]

    with plt.style.context(config.style):
        fig, axes = plt.subplots(2, 2, figsize=(fig_w * 2, panel_h * 2))

        # ── Panel (0,0): Two-Ray per Δt ──────────────────────────────────────
        ax = axes[0, 0]
        ests = results.depth_estimates.get('two_ray', [])
        if ests:
            offsets = sorted({e.time_offset for e in ests if e.time_offset is not None})
            n = len(offsets)
            for i, offset in enumerate(offsets):
                sub = [e for e in ests if e.time_offset == offset]
                ts  = [(e.timestamp - ref_time).total_seconds() for e in sub]
                es  = [e.error for e in sub]
                col = plt.cm.Blues(0.35 + 0.65 * i / max(n - 1, 1))
                _smooth_and_plot(ax, ts, es, col, f'Δt={offset:.0f}s')
        _finish_ax(ax, 'Two-Ray Error Over Time (per Δt)')

        # ── Panel (0,1): Multi-Ray per window ────────────────────────────────
        ax = axes[0, 1]
        ests = results.depth_estimates.get('multi_ray', [])
        if ests:
            offsets = sorted({e.time_offset for e in ests if e.time_offset is not None})
            n = len(offsets)
            for i, window_s in enumerate(offsets):
                sub   = [e for e in ests if e.time_offset == window_s]
                n_obs = (sub[0].num_observations_used
                         if sub and sub[0].num_observations_used else '?')
                ts    = [(e.timestamp - ref_time).total_seconds() for e in sub]
                es    = [e.error for e in sub]
                col   = plt.cm.Oranges(0.35 + 0.65 * i / max(n - 1, 1))
                _smooth_and_plot(ax, ts, es, col, f'W={window_s:.0f}s N={n_obs}')
        _finish_ax(ax, 'Multi-Ray Error Over Time (per Window)')

        # ── Panel (1,0): Kalman ───────────────────────────────────────────────
        ax = axes[1, 0]
        ests = results.depth_estimates.get('kalman', [])
        if ests:
            ts = [(e.timestamp - ref_time).total_seconds() for e in ests]
            es = [e.error for e in ests]
            _smooth_and_plot(ax, ts, es, config.color_kalman, 'Kalman Filter')
        _finish_ax(ax, 'Kalman Filter Error Over Time')

        # ── Panel (1,1): Iterative k=1…5 ─────────────────────────────────────
        ax = axes[1, 1]
        for k in range(1, 6):
            method = f'iterative_k{k}'
            ests   = results.depth_estimates.get(method, [])
            if not ests:
                continue
            ts = [(e.timestamp - ref_time).total_seconds() for e in ests]
            es = [e.error for e in ests]
            _smooth_and_plot(ax, ts, es, config.method_color(method),
                             config.method_label(method))
        _finish_ax(ax, 'Iterative Vel. Triang. Error Over Time (k=1…5)')

        fig.suptitle('Depth Estimation Error Over Time',
                     fontsize=config.fontsize_title + 2)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('error_over_time')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_estimated_vs_true_depth(
    results: SimulationResults,
    save_path: str,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Scatter plot of estimated vs true depth — 2×2 panel layout.

    Four panels:
      (0,0) Two-Ray: one scatter per Δt offset  (tab10 colors)
      (0,1) Multi-Ray: one scatter per window   (tab10 colors)
      (1,0) Iterative: k=1…5                   (purple gradient)
      (1,1) Kalman filter                       (single green)

    Each panel includes the y=x perfect-estimation diagonal.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    methods : list of str, optional
        Ignored (layout is fixed); kept for API compatibility.
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()

    def _add_perfect_line(ax, depths):
        """Add y=x diagonal; skip if no data."""
        if not depths:
            return
        mn, mx = min(depths), max(depths)
        ax.plot([mn, mx], [mn, mx],
                color='black', linestyle='--',
                linewidth=config.line_width_reference,
                label='y=x (perfect)', zorder=10)

    panel_size = config.figsize_square[0]  # 8 inches

    with plt.style.context(config.style):
        fig, axes = plt.subplots(2, 2,
                                 figsize=(panel_size * 2, panel_size * 2))

        # ── Panel (0,0): Two-Ray — Blues gradient (same as rmse_by_distance) ──
        ax = axes[0, 0]
        all_depths = []
        estimates = results.depth_estimates.get('two_ray', [])
        if estimates:
            offsets = sorted({e.time_offset for e in estimates
                               if e.time_offset is not None})
            n = len(offsets)
            for i, offset in enumerate(offsets):
                sub = [e for e in estimates if e.time_offset == offset]
                td  = [e.true_depth for e in sub]
                ed  = [e.estimated_depth for e in sub]
                col = plt.cm.Blues(0.35 + 0.65 * i / max(n - 1, 1))
                ax.scatter(td, ed,
                           color=col,
                           alpha=config.alpha_scatter,
                           s=config.marker_size ** 2,
                           edgecolors='none',
                           label=f'Δt={offset:.0f}s')
                all_depths.extend(td)
                all_depths.extend(ed)
        _add_perfect_line(ax, all_depths)
        ax.set_title('Two-Ray (per Δt offset)', fontsize=config.fontsize_title)
        ax.set_xlabel('True Depth [m]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Estimated Depth [m]', fontsize=config.fontsize_labels)
        ax.legend(fontsize=config.fontsize_legend, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=config.fontsize_ticks)

        # ── Panel (0,1): Multi-Ray — Oranges gradient ─────────────────────────
        ax = axes[0, 1]
        all_depths = []
        estimates = results.depth_estimates.get('multi_ray', [])
        if estimates:
            offsets = sorted({e.time_offset for e in estimates
                               if e.time_offset is not None})
            n = len(offsets)
            for i, offset in enumerate(offsets):
                sub   = [e for e in estimates if e.time_offset == offset]
                n_obs = (sub[0].num_observations_used
                         if sub and sub[0].num_observations_used else '?')
                td  = [e.true_depth for e in sub]
                ed  = [e.estimated_depth for e in sub]
                col = plt.cm.Oranges(0.35 + 0.65 * i / max(n - 1, 1))
                ax.scatter(td, ed,
                           color=col,
                           alpha=config.alpha_scatter,
                           s=config.marker_size ** 2,
                           edgecolors='none',
                           label=f'W={offset:.0f}s N={n_obs}')
                all_depths.extend(td)
                all_depths.extend(ed)
        _add_perfect_line(ax, all_depths)
        ax.set_title('Multi-Ray (per Window)', fontsize=config.fontsize_title)
        ax.set_xlabel('True Depth [m]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Estimated Depth [m]', fontsize=config.fontsize_labels)
        ax.legend(fontsize=config.fontsize_legend, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=config.fontsize_ticks)

        # ── Panel (1,0): Iterative — only k=1 (reddish) and k=5 (bluish) ─────
        ax = axes[1, 0]
        all_depths = []
        # k=5 drawn first (background), k=1 drawn last (foreground)
        ITER_COLORS = {
            'iterative_k5': ('#3949AB', 2),  # bluish-purple, behind
            'iterative_k1': ('#D81B60', 3),  # reddish-purple, front
        }
        for method, (col, zo) in ITER_COLORS.items():
            estimates = results.depth_estimates.get(method, [])
            if not estimates:
                continue
            td = [e.true_depth for e in estimates]
            ed = [e.estimated_depth for e in estimates]
            ax.scatter(td, ed,
                       color=col,
                       alpha=config.alpha_scatter,
                       s=config.marker_size ** 2,
                       edgecolors='none',
                       zorder=zo,
                       label=config.method_label(method))
            all_depths.extend(td)
            all_depths.extend(ed)
        _add_perfect_line(ax, all_depths)
        ax.set_title('Iterative Vel. Triang. (k=1 vs k=5)',
                     fontsize=config.fontsize_title)
        ax.set_xlabel('True Depth [m]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Estimated Depth [m]', fontsize=config.fontsize_labels)
        ax.legend(fontsize=config.fontsize_legend)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=config.fontsize_ticks)

        # ── Panel (1,1): Kalman ───────────────────────────────────────────────
        ax = axes[1, 1]
        all_depths = []
        estimates = results.depth_estimates.get('kalman', [])
        if estimates:
            td = [e.true_depth for e in estimates]
            ed = [e.estimated_depth for e in estimates]
            ax.scatter(td, ed,
                       color=config.color_kalman,
                       alpha=config.alpha_scatter,
                       s=config.marker_size ** 2,
                       edgecolors='none',
                       label='Kalman Filter')
            all_depths.extend(td)
            all_depths.extend(ed)
        _add_perfect_line(ax, all_depths)
        ax.set_title('Kalman Filter (Const. Velocity)',
                     fontsize=config.fontsize_title)
        ax.set_xlabel('True Depth [m]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Estimated Depth [m]', fontsize=config.fontsize_labels)
        ax.legend(fontsize=config.fontsize_legend)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=config.fontsize_ticks)

        fig.suptitle('Estimated vs True Depth',
                     fontsize=config.fontsize_title + 2)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('estimated_vs_true_depth')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_method_comparison_bar(
    results: SimulationResults,
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Bar chart comparing RMSE and Mean Absolute Error across all methods.

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
    methods = results.available_methods

    if not methods:
        print("  [!]No methods available for comparison bar chart")
        return

    # Gather statistics — one bar per (method, time_offset) for two_ray/multi_ray
    rmse_values = []
    mae_values  = []
    std_values  = []
    colors      = []
    labels      = []
    two_ray_rmse_val  = None   # tracked for reference line (smallest Δt)
    two_ray_rmse_mae  = None
    two_ray_bar_color = None

    for method in methods:
        if method == 'two_ray':
            ests = results.depth_estimates.get(method, [])
            offsets = sorted({e.time_offset for e in ests if e.time_offset is not None})
            n = len(offsets)
            for i, offset in enumerate(offsets):
                ests_off = [e for e in ests if e.time_offset == offset]
                if not ests_off:
                    continue
                errs = np.array([e.error for e in ests_off])
                rmse = float(np.sqrt(np.mean(errs ** 2)))
                mae  = float(np.mean(np.abs(errs)))
                std  = float(np.std(errs))
                frac = i / max(1, n - 1)
                color = plt.cm.Blues(0.35 + 0.65 * frac)
                if two_ray_rmse_val is None:   # first (smallest) offset = reference
                    two_ray_rmse_val  = rmse
                    two_ray_rmse_mae  = mae
                    two_ray_bar_color = color
                rmse_values.append(rmse)
                mae_values.append(mae)
                std_values.append(std)
                colors.append(color)
                labels.append(f'Two-Ray Δt={offset:.0f}s')

        elif method == 'multi_ray':
            ests = results.depth_estimates.get(method, [])
            offsets = sorted({e.time_offset for e in ests if e.time_offset is not None})
            n = len(offsets)
            for i, offset in enumerate(offsets):
                ests_off = [e for e in ests if e.time_offset == offset]
                if not ests_off:
                    continue
                errs = np.array([e.error for e in ests_off])
                rmse = float(np.sqrt(np.mean(errs ** 2)))
                mae  = float(np.mean(np.abs(errs)))
                std  = float(np.std(errs))
                frac = i / max(1, n - 1)
                color = plt.cm.Oranges(0.35 + 0.65 * frac)
                rmse_values.append(rmse)
                mae_values.append(mae)
                std_values.append(std)
                colors.append(color)
                labels.append(f'Multi-Ray W={offset:.0f}s')

        else:
            stats = results.get_statistics(method)
            if stats['num_estimates'] == 0:
                continue
            rmse  = stats['rmse']
            mae   = stats['mae']
            std   = stats['std_error']
            rmse_values.append(rmse)
            mae_values.append(mae)
            std_values.append(std)
            colors.append(config.method_color(method))
            labels.append(config.method_label(method))

    if not rmse_values:
        return

    with plt.style.context(config.style):
        n_bars = len(labels)
        fig_w  = max(config.figsize_wide[0], n_bars * 0.9)
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, config.figsize_wide[1]))

        x = np.arange(len(labels))
        width = 0.5

        # RMSE bars
        bars = axes[0].bar(
            x, rmse_values,
            width=width,
            color=colors,
            edgecolor='white',
            linewidth=0.5,
        )
        axes[0].errorbar(
            x, rmse_values, yerr=std_values,
            fmt='none', color='black',
            capsize=5, linewidth=1.5,
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha='right',
                                fontsize=config.fontsize_ticks)
        axes[0].set_ylabel('RMSE [m]', fontsize=config.fontsize_labels)
        axes[0].set_title('Root Mean Square Error', fontsize=config.fontsize_title)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(labelsize=config.fontsize_ticks)

        # Annotate bars with values
        for bar, val in zip(bars, rmse_values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'{val:.0f}m',
                ha='left', va='bottom',
                rotation=45,
                fontsize=config.fontsize_annotations,
            )

        # Horizontal reference line at two_ray Δt=1s RMSE
        if two_ray_rmse_val is not None:
            axes[0].axhline(
                y=two_ray_rmse_val,
                color=two_ray_bar_color,
                linestyle='--',
                linewidth=config.line_width,
                alpha=0.85,
                label='Two-Ray Δt=1s (reference)',
                zorder=10,
            )
            axes[0].legend(fontsize=int((config.fontsize_legend - 1) * 1.5))

        # MAE bars
        bars2 = axes[1].bar(
            x, mae_values,
            width=width,
            color=colors,
            edgecolor='white',
            linewidth=0.5,
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha='right',
                                fontsize=config.fontsize_ticks)
        axes[1].set_ylabel('MAE [m]', fontsize=config.fontsize_labels)
        axes[1].set_title('Mean Absolute Error', fontsize=config.fontsize_title)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(labelsize=config.fontsize_ticks)

        for bar, val in zip(bars2, mae_values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'{val:.0f}m',
                ha='left', va='bottom',
                rotation=45,
                fontsize=config.fontsize_annotations,
            )

        # Horizontal reference line at two_ray Δt=1s MAE
        if two_ray_rmse_mae is not None:
            axes[1].axhline(
                y=two_ray_rmse_mae,
                color=two_ray_bar_color,
                linestyle='--',
                linewidth=config.line_width,
                alpha=0.85,
                label='Two-Ray Δt=1s (reference)',
                zorder=10,
            )
            axes[1].legend(fontsize=int((config.fontsize_legend - 1) * 1.5))

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('method_comparison_bar')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_triangulation_gap_vs_error(
    results: SimulationResults,
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Scatter plot of triangulation gap vs absolute error — 1×2 panel layout.

    Left panel:  Two-Ray — one scatter per Δt offset (Blues gradient)
    Right panel: Multi-Ray — one scatter per window size (Oranges gradient)

    Shows whether the triangulation gap is a reliable quality predictor.
    Points are colored by Δt / window to separate overlapping clouds.

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

    has_two_ray   = bool(results.depth_estimates.get('two_ray',   []))
    has_multi_ray = bool(results.depth_estimates.get('multi_ray', []))

    if not has_two_ray and not has_multi_ray:
        print("  [!]No geometric methods available for gap vs error plot")
        return

    with plt.style.context(config.style):
        fig, axes = plt.subplots(1, 2, figsize=config.figsize_wide)

        def _plot_panel(ax, method, cmap, title, offset_fmt):
            estimates = results.depth_estimates.get(method, [])
            ests_gap  = [e for e in estimates if e.triangulation_gap is not None]
            if not ests_gap:
                ax.set_title(title, fontsize=config.fontsize_title)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=config.fontsize_labels, color='grey')
                return

            offsets = sorted({e.time_offset for e in ests_gap
                               if e.time_offset is not None})
            n = len(offsets) if offsets else 1

            if offsets:
                for i, offset in enumerate(offsets):
                    sub = [e for e in ests_gap if e.time_offset == offset]
                    gaps = [e.triangulation_gap for e in sub]
                    errs = [abs(e.error) for e in sub]
                    col  = cmap(0.35 + 0.65 * i / max(n - 1, 1))
                    ax.scatter(gaps, errs,
                               color=col,
                               alpha=config.alpha_scatter,
                               s=config.marker_size ** 2,
                               edgecolors='none',
                               label=offset_fmt(offset))
            else:
                # No offset info — single cloud
                gaps = [e.triangulation_gap for e in ests_gap]
                errs = [abs(e.error) for e in ests_gap]
                ax.scatter(gaps, errs,
                           color=cmap(0.6),
                           alpha=config.alpha_scatter,
                           s=config.marker_size ** 2,
                           edgecolors='none',
                           label=config.method_label(method))

            ax.set_xlabel('Triangulation Gap [m]', fontsize=config.fontsize_labels)
            ax.set_ylabel('|Depth Error| [m]', fontsize=config.fontsize_labels)
            ax.set_title(title, fontsize=config.fontsize_title)
            ax.tick_params(labelsize=config.fontsize_ticks)
            ax.legend(fontsize=config.fontsize_legend - 1, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')

        _plot_panel(axes[0], 'two_ray',   plt.cm.Blues,
                    'Two-Ray: Gap vs Error (per Δt)',
                    lambda o: f'Δt={o:.0f}s')
        _plot_panel(axes[1], 'multi_ray', plt.cm.Oranges,
                    'Multi-Ray: Gap vs Error (per Window)',
                    lambda o: f'W={o:.0f}s')

        fig.suptitle('Triangulation Gap vs Estimation Error\n'
                     '(Gap = quality metric: lower is better)',
                     fontsize=config.fontsize_title + 2)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('triangulation_gap_vs_error')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_rmse_by_distance(
    results: SimulationResults,
    save_path: str,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    RMSE vs true depth — 4×1 panel layout.

    Four stacked panels (sharex, log-log):
      1. Two-Ray (one line per Δt)
      2. Multi-Ray (one line per window)
      3. Kalman Filter
      4. Iterative Velocity Triangulation (k=1…5)

    Every panel also contains a bright-red Two-Ray Δt=1s reference line
    (thick) so all methods can be visually compared to the same baseline.
    Legend is pinned to the lower-right of each panel.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save figure
    methods : list of str, optional
        Ignored (layout is fixed); kept for API compatibility.
    config : PlotConfig, optional
        Visual configuration, default PlotConfig()
    """
    config = config or PlotConfig()

    # ── Depth range from observations ────────────────────────────────────────
    if results.observations:
        min_depth_km = min(obs.true_depth for obs in results.observations) / 1e3
        max_depth_km = max(obs.true_depth for obs in results.observations) / 1e3
    else:
        min_depth_km = 100.0
        max_depth_km = 3000.0

    log_min    = np.log10(min_depth_km)
    log_max    = np.log10(max_depth_km)
    log_margin = (log_max - log_min) * 0.1
    bins_km = np.logspace(log_min - log_margin, log_max + log_margin, 31)
    bins_m  = bins_km * 1e3
    MIN_ESTIMATES = 5

    # ── Bright red reference colour ───────────────────────────────────────────
    REF_COLOR = '#FF1744'   # Material Red A400 — stands out on any background

    # ── Pre-extract two_ray Δt=1s data used in every panel ───────────────────
    two_ray_all = results.depth_estimates.get('two_ray', [])
    ref_sub = [e for e in two_ray_all if e.time_offset == 1.0]
    ref_td  = np.array([e.true_depth for e in ref_sub]) if ref_sub else np.array([])
    ref_err = np.array([e.error      for e in ref_sub]) if ref_sub else np.array([])

    # ── Helper: draw one step-line on a given axes object ────────────────────
    def _draw_steps_on(ax, true_depths, errors, color, label, lw=None):
        lw = lw if lw is not None else config.line_width
        x_seg, y_seg = [], []
        labelled = False
        for i in range(len(bins_m) - 1):
            lo, hi = bins_m[i], bins_m[i + 1]
            mask = (true_depths >= lo) & (true_depths < hi)
            if mask.sum() < MIN_ESTIMATES:
                if x_seg:
                    ax.step(x_seg, y_seg, where='post', color=color,
                            linewidth=lw,
                            label=label if not labelled else '_nolegend_')
                    labelled = True
                    x_seg, y_seg = [], []
                continue
            x_seg.append(bins_km[i])
            y_seg.append(np.sqrt(np.mean(errors[mask] ** 2)) / 1e3)
        if x_seg:
            ax.step(x_seg, y_seg, where='post', color=color,
                    linewidth=lw,
                    label=label if not labelled else '_nolegend_')

    def _add_ref_line(ax):
        """Add Two-Ray Δt=1s as bright-red thick background reference."""
        if len(ref_td) > 0:
            _draw_steps_on(ax, ref_td, ref_err,
                           REF_COLOR, 'Two-Ray Δt=1s (ref)',
                           lw=config.line_width * 2)

    def _finish_ax(ax, title):
        """Apply shared axis formatting."""
        for ref_km in (50, 100, 200, 500, 1000):
            if bins_km[0] <= ref_km <= bins_km[-1]:
                ax.axvline(x=ref_km, color='grey', linestyle='--',
                           linewidth=config.line_width_reference, alpha=0.6,
                           label='_nolegend_')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(bins_km[0], bins_km[-1])
        ax.set_xlabel('True Depth [km]', fontsize=config.fontsize_labels)
        ax.set_ylabel('RMSE [km]', fontsize=config.fontsize_labels)
        ax.set_title(title, fontsize=config.fontsize_title)
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.legend(fontsize=config.fontsize_legend,
                  loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3, which='both')

    # ── Build figure (2×2) ────────────────────────────────────────────────────
    panel_h = config.figsize_single[1]
    fig_w   = config.figsize_single[0]

    with plt.style.context(config.style):
        fig, axes = plt.subplots(2, 2, figsize=(fig_w * 2, panel_h * 2))
        ax_tr = axes[0, 0]
        ax_mr = axes[0, 1]
        ax_kf = axes[1, 0]
        ax_it = axes[1, 1]

        # ── Panel (0,0): Two-Ray ─────────────────────────────────────────────
        _add_ref_line(ax_tr)
        estimates = results.depth_estimates.get('two_ray', [])
        if estimates:
            offsets = sorted({e.time_offset for e in estimates
                               if e.time_offset is not None})
            n    = len(offsets)
            cmap = plt.cm.Blues
            for i, dt in enumerate(offsets):
                if dt == 1.0:
                    continue  # already shown as reference
                sub  = [e for e in estimates if e.time_offset == dt]
                td   = np.array([e.true_depth for e in sub])
                errs = np.array([e.error      for e in sub])
                col  = cmap(0.35 + 0.65 * i / max(n - 1, 1))
                _draw_steps_on(ax_tr, td, errs, col, f'Δt={int(dt)}s')
        _finish_ax(ax_tr, 'Two-Ray Triangulation (per Δt)')

        # ── Panel (0,1): Multi-Ray ───────────────────────────────────────────
        _add_ref_line(ax_mr)
        estimates = results.depth_estimates.get('multi_ray', [])
        if estimates:
            offsets = sorted({e.time_offset for e in estimates
                               if e.time_offset is not None})
            n    = len(offsets)
            cmap = plt.cm.Oranges
            for i, window in enumerate(offsets):
                sub   = [e for e in estimates if e.time_offset == window]
                n_obs = (sub[0].num_observations_used
                         if sub and sub[0].num_observations_used else '?')
                td    = np.array([e.true_depth for e in sub])
                errs  = np.array([e.error      for e in sub])
                col   = cmap(0.35 + 0.65 * i / max(n - 1, 1))
                _draw_steps_on(ax_mr, td, errs, col,
                               f'W={int(window)}s N={n_obs}')
        _finish_ax(ax_mr, 'Multi-Ray Least Squares (per Window)')

        # ── Panel (1,0): Kalman ──────────────────────────────────────────────
        _add_ref_line(ax_kf)
        estimates = results.depth_estimates.get('kalman', [])
        if estimates:
            td   = np.array([e.true_depth for e in estimates])
            errs = np.array([e.error      for e in estimates])
            _draw_steps_on(ax_kf, td, errs,
                           config.color_kalman, 'Kalman Filter')
        _finish_ax(ax_kf, 'Kalman Filter (Const. Velocity)')

        # ── Panel (1,1): Iterative ───────────────────────────────────────────
        _add_ref_line(ax_it)
        for k in range(1, 6):
            method    = f'iterative_k{k}'
            estimates = results.depth_estimates.get(method, [])
            if not estimates:
                continue
            td   = np.array([e.true_depth for e in estimates])
            errs = np.array([e.error      for e in estimates])
            _draw_steps_on(ax_it, td, errs,
                           config.method_color(method),
                           config.method_label(method))
        _finish_ax(ax_it, 'Iterative Velocity Triangulation (k=1…5)')

        fig.suptitle('Depth Estimation RMSE by Distance',
                     fontsize=config.fontsize_title + 2)

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('rmse_by_distance')
        filepath = os.path.join(save_path, filename)
        _annotate_scenario_params(fig, results)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def plot_all_statistical(
    results: SimulationResults,
    save_path: str,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Generate all 6 statistical plots at once.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    save_path : str
        Directory to save all figures
    methods : list of str, optional
        Methods to include
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    os.makedirs(save_path, exist_ok=True)

    print(f"\nGenerating statistical plots -> {save_path}")

    plot_error_histogram(results, save_path, methods, config)
    plot_error_vs_time_offset(results, save_path, methods, config)
    plot_error_over_time(results, save_path, methods, config)
    plot_estimated_vs_true_depth(results, save_path, methods, config)
    plot_method_comparison_bar(results, save_path, config)
    plot_triangulation_gap_vs_error(results, save_path, config)
    plot_rmse_by_distance(results, save_path, methods, config)

    print(f"  [OK] All statistical plots saved!\n")
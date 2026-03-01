"""
Parameter study plots for multi-run experiment results.

This module contains plots that visualize results across the full
parameter grid (distance × speed), showing how estimation accuracy
depends on flyby geometry.

For a DSA (Distance × Speed × Angle) study the plots are generated
separately for each crossing angle and saved in per-angle subfolders.

Functions
---------
plot_rmse_vs_distance
    RMSE as function of closest approach distance
plot_rmse_vs_speed
    RMSE as function of missile speed
plot_heatmap_rmse
    2D heatmap of RMSE across full parameter grid
plot_heatmap_improvement
    2D heatmap of % improvement of one method over baseline
plot_all_parameter_study
    Convenience: generate all parameter study plots at once,
    looping over all crossing angles into per-angle subfolders
"""

import math
import os
from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
# scipy.interpolate imported lazily inside plot_hemisphere_polar to avoid
# slow C-extension loading on Windows when the module is just imported.

# Colormaps for sweep plots:
# - Azimuth is cyclic (0°=360°) → use twilight (cyclic colormap)
# - Elevation is NOT cyclic (0°–90°) → use a linear sequential colormap
_SWEEP_CMAP_CYCLIC  = plt.cm.twilight          # for azimuth sweeps
_SWEEP_CMAP_LINEAR  = plt.cm.plasma            # for elevation sweeps

from missile_fly_by_simulation.experiments.experiment_results import (
    ExperimentResults,
    AngularStudyResults,
)
from missile_fly_by_simulation.visualization.plot_config import PlotConfig


# =============================================================================
# INDIVIDUAL PLOT FUNCTIONS
# =============================================================================

def plot_rmse_vs_distance(
    experiment_results: ExperimentResults,
    save_path: str,
    crossing_angle: float,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Plot RMSE as function of closest approach distance.

    One line per missile speed, one subplot per method.
    Shows how estimation accuracy degrades with range.

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from parameter study
    save_path : str
        Directory to save figure
    crossing_angle : float
        Crossing angle slice to plot [degrees]
    methods : list of str, optional
        Methods to plot, default all available
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    methods = methods or _get_available_methods(experiment_results)

    sorted_distances = sorted(experiment_results.distances)
    sorted_speeds = sorted(experiment_results.speeds)

    # Color map for speeds
    speed_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_speeds)))

    with plt.style.context(config.style):
        fig, axes = plt.subplots(
            1, len(methods),
            figsize=(config.figsize_single[0] * len(methods), config.figsize_single[1]),
            sharey=True,
        )

        # Handle single method case
        if len(methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            for j, speed in enumerate(sorted_speeds):
                rmse_values = []
                for dist in sorted_distances:
                    key = (dist, speed, crossing_angle)
                    if key in experiment_results.summaries:
                        rmse_values.append(
                            experiment_results.summaries[key].rmse(method)
                        )
                    else:
                        rmse_values.append(np.nan)

                ax.plot(
                    [d / 1e3 for d in sorted_distances],
                    rmse_values,
                    color=speed_colors[j],
                    linewidth=config.line_width,
                    marker='o',
                    markersize=config.marker_size + 1,
                    label=f'{speed:.0f} m/s',
                )

            ax.set_xlabel(
                'Closest Approach Distance [km]',
                fontsize=config.fontsize_labels
            )
            ax.set_ylabel('RMSE [m]', fontsize=config.fontsize_labels)
            ax.set_title(
                f'{config.method_label(method)}',
                fontsize=config.fontsize_title
            )
            ax.tick_params(labelsize=config.fontsize_ticks)
            ax.legend(
                title='Missile Speed',
                fontsize=config.fontsize_legend - 1,
                title_fontsize=config.fontsize_legend - 1,
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        fig.suptitle(
            f'Depth Estimation RMSE vs Closest Approach Distance\n'
            f'(Crossing angle: {crossing_angle:.0f}°)',
            fontsize=config.fontsize_title + 1,
            y=1.02,
        )

        if config.tight_layout:
            fig.tight_layout()

        filepath = os.path.join(
            save_path, config.save_filename('rmse_vs_distance')
        )
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_rmse_vs_speed(
    experiment_results: ExperimentResults,
    save_path: str,
    crossing_angle: float,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Plot RMSE as function of missile speed.

    One line per closest approach distance, one subplot per method.
    Shows how estimation accuracy changes with target speed.

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from parameter study
    save_path : str
        Directory to save figure
    crossing_angle : float
        Crossing angle slice to plot [degrees]
    methods : list of str, optional
        Methods to plot, default all available
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    methods = methods or _get_available_methods(experiment_results)

    sorted_distances = sorted(experiment_results.distances)
    sorted_speeds = sorted(experiment_results.speeds)

    # Color map for distances
    dist_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sorted_distances)))

    with plt.style.context(config.style):
        fig, axes = plt.subplots(
            1, len(methods),
            figsize=(config.figsize_single[0] * len(methods), config.figsize_single[1]),
            sharey=True,
        )

        if len(methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            for i, dist in enumerate(sorted_distances):
                rmse_values = []
                for speed in sorted_speeds:
                    key = (dist, speed, crossing_angle)
                    if key in experiment_results.summaries:
                        rmse_values.append(
                            experiment_results.summaries[key].rmse(method)
                        )
                    else:
                        rmse_values.append(np.nan)

                ax.plot(
                    sorted_speeds,
                    rmse_values,
                    color=dist_colors[i],
                    linewidth=config.line_width,
                    marker='o',
                    markersize=config.marker_size + 1,
                    label=f'{dist/1e3:.0f} km',
                )

            ax.set_xlabel(
                'Missile Speed [m/s]',
                fontsize=config.fontsize_labels
            )
            ax.set_ylabel('RMSE [m]', fontsize=config.fontsize_labels)
            ax.set_title(
                f'{config.method_label(method)}',
                fontsize=config.fontsize_title
            )
            ax.tick_params(labelsize=config.fontsize_ticks)
            ax.legend(
                title='Approach Distance',
                fontsize=config.fontsize_legend - 1,
                title_fontsize=config.fontsize_legend - 1,
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        fig.suptitle(
            f'Depth Estimation RMSE vs Missile Speed\n'
            f'(Crossing angle: {crossing_angle:.0f}°)',
            fontsize=config.fontsize_title + 1,
            y=1.02,
        )

        if config.tight_layout:
            fig.tight_layout()

        filepath = os.path.join(
            save_path, config.save_filename('rmse_vs_speed')
        )
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_heatmap_rmse(
    experiment_results: ExperimentResults,
    save_path: str,
    crossing_angle: float,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    2D heatmap of RMSE across full parameter grid.

    One heatmap per method, side by side.
    X axis: missile speed, Y axis: closest approach distance.
    Color: RMSE value.

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from parameter study
    save_path : str
        Directory to save figure
    crossing_angle : float
        Crossing angle slice to plot [degrees]
    methods : list of str, optional
        Methods to plot, default all available
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    methods = methods or _get_available_methods(experiment_results)

    sorted_distances = sorted(experiment_results.distances)
    sorted_speeds = sorted(experiment_results.speeds)

    with plt.style.context(config.style):
        fig, axes = plt.subplots(
            1, len(methods),
            figsize=(config.figsize_square[0] * len(methods), config.figsize_square[1]),
        )

        if len(methods) == 1:
            axes = [axes]

        # Shared color scale across all methods
        all_rmse = []
        rmse_matrices = {}
        for method in methods:
            matrix = experiment_results.get_rmse_matrix(method, crossing_angle)
            rmse_matrices[method] = matrix
            all_rmse.extend(matrix[~np.isnan(matrix)].tolist())

        vmin = 0
        vmax = np.nanpercentile(all_rmse, 95) if all_rmse else 1

        for ax, method in zip(axes, methods):
            matrix = rmse_matrices[method]

            im = ax.imshow(
                matrix,
                aspect='auto',
                origin='lower',
                cmap=config.heatmap_colormap,
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest',
            )

            # Axis labels
            ax.set_xticks(range(len(sorted_speeds)))
            ax.set_xticklabels(
                experiment_results.speed_labels(),
                rotation=45,
                ha='right',
                fontsize=config.fontsize_ticks,
            )
            ax.set_yticks(range(len(sorted_distances)))
            ax.set_yticklabels(
                experiment_results.distance_labels(),
                fontsize=config.fontsize_ticks,
            )

            ax.set_xlabel('Missile Speed', fontsize=config.fontsize_labels)
            ax.set_ylabel('Closest Approach Distance', fontsize=config.fontsize_labels)
            ax.set_title(
                config.method_label(method),
                fontsize=config.fontsize_title,
            )

            # Annotate cells with RMSE values
            if config.heatmap_annotate:
                for i in range(len(sorted_distances)):
                    for j in range(len(sorted_speeds)):
                        val = matrix[i, j]
                        if not np.isnan(val):
                            # White text on dark cells, black on light
                            normalized = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                            text_color = 'white' if normalized > 0.6 else 'black'
                            ax.text(
                                j, i, f'{val:.0f}',
                                ha='center', va='center',
                                fontsize=config.fontsize_annotations,
                                color=text_color,
                                fontweight='bold',
                            )

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('RMSE [m]', fontsize=config.fontsize_ticks)
            cbar.ax.tick_params(labelsize=config.fontsize_ticks - 1)

        fig.suptitle(
            f'Depth Estimation RMSE: Full Parameter Grid\n'
            f'(Crossing angle: {crossing_angle:.0f}°)',
            fontsize=config.fontsize_title + 1,
        )

        if config.tight_layout:
            fig.tight_layout()

        filepath = os.path.join(
            save_path, config.save_filename('heatmap_rmse')
        )
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_heatmap_improvement(
    experiment_results: ExperimentResults,
    save_path: str,
    crossing_angle: float,
    method_improved: str = 'multi_ray',
    method_baseline: str = 'two_ray',
    config: Optional[PlotConfig] = None,
):
    """
    2D heatmap of % improvement of one method over baseline.

    Green = large improvement over baseline.
    Red = method is worse than baseline.

    Shows WHERE in parameter space the improved method
    gives the most benefit.

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from parameter study
    save_path : str
        Directory to save figure
    crossing_angle : float
        Crossing angle slice to plot [degrees]
    method_improved : str, optional
        Method to compare, default 'multi_ray'
    method_baseline : str, optional
        Baseline method, default 'two_ray'
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()

    sorted_distances = sorted(experiment_results.distances)
    sorted_speeds = sorted(experiment_results.speeds)

    # Check both methods exist
    available = _get_available_methods(experiment_results)
    if method_improved not in available or method_baseline not in available:
        print(
            f"  [!] Cannot plot improvement: "
            f"need both '{method_improved}' and '{method_baseline}'"
        )
        return

    improvement_matrix = experiment_results.get_improvement_matrix(
        method_improved, method_baseline, crossing_angle
    )

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_square)

        # Symmetric color scale around 0
        max_abs = np.nanmax(np.abs(improvement_matrix))
        vmin = -max_abs
        vmax = max_abs

        im = ax.imshow(
            improvement_matrix,
            aspect='auto',
            origin='lower',
            cmap=config.heatmap_improvement_colormap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
        )

        # Axis labels
        ax.set_xticks(range(len(sorted_speeds)))
        ax.set_xticklabels(
            experiment_results.speed_labels(),
            rotation=45, ha='right',
            fontsize=config.fontsize_ticks,
        )
        ax.set_yticks(range(len(sorted_distances)))
        ax.set_yticklabels(
            experiment_results.distance_labels(),
            fontsize=config.fontsize_ticks,
        )
        ax.set_xlabel('Missile Speed', fontsize=config.fontsize_labels)
        ax.set_ylabel('Closest Approach Distance', fontsize=config.fontsize_labels)
        ax.set_title(
            f'% RMSE Improvement:\n'
            f'{config.method_label(method_improved)} vs '
            f'{config.method_label(method_baseline)}\n'
            f'(Crossing angle: {crossing_angle:.0f}°)',
            fontsize=config.fontsize_title,
        )

        # Annotate cells
        if config.heatmap_annotate:
            for i in range(len(sorted_distances)):
                for j in range(len(sorted_speeds)):
                    val = improvement_matrix[i, j]
                    if not np.isnan(val):
                        normalized = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        text_color = 'white' if abs(normalized - 0.5) > 0.25 else 'black'
                        sign = '+' if val >= 0 else ''
                        ax.text(
                            j, i, f'{sign}{val:.0f}%',
                            ha='center', va='center',
                            fontsize=config.fontsize_annotations,
                            color=text_color,
                            fontweight='bold',
                        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('% Improvement (positive = better)', fontsize=config.fontsize_ticks)
        cbar.ax.tick_params(labelsize=config.fontsize_ticks - 1)

        if config.tight_layout:
            fig.tight_layout()

        filename = f'heatmap_improvement_{method_improved}_vs_{method_baseline}'
        filepath = os.path.join(save_path, config.save_filename(filename))
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


# =============================================================================
# ANGLE STUDY PLOT FUNCTIONS (1D crossing-angle sweep)
# =============================================================================

def plot_rmse_vs_angle(
    experiment_results: ExperimentResults,
    save_path: str,
    distance: float,
    speed: float,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Plot RMSE as function of crossing angle for all methods.

    Primary output of the 1D angle sweep. X-axis: 0–180 deg,
    Y-axis: RMSE [km] log-scale, one line per method.

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from 1D angle study
    save_path : str
        Directory to save figure
    distance : float
        Fixed closest approach distance [m]
    speed : float
        Fixed missile speed [m/s]
    methods : list of str, optional
        Methods to plot, default all available
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    methods = methods or _get_available_methods(experiment_results)

    sorted_angles = sorted(experiment_results.crossing_angles)

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_single)

        for method in methods:
            rmse_dict = experiment_results.get_rmse_vs_angle(method, distance, speed)
            angles = sorted(rmse_dict.keys())
            rmse_km = [rmse_dict[a] / 1e3 for a in angles]

            ax.plot(
                angles, rmse_km,
                color=config.method_color(method),
                linewidth=config.line_width,
                marker='o', markersize=config.marker_size,
                label=config.method_label(method),
            )

        ax.set_xlabel('Crossing Angle [deg]', fontsize=config.fontsize_labels)
        ax.set_ylabel('RMSE [km]', fontsize=config.fontsize_labels)
        ax.set_title(
            f'Depth Estimation RMSE vs Crossing Angle\n'
            f'dist={distance/1e3:.0f} km  |  speed={speed:.0f} m/s',
            fontsize=config.fontsize_title,
        )
        ax.set_yscale('log')
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 15))
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.legend(fontsize=config.fontsize_legend)
        ax.grid(True, which='both', alpha=0.3)

        if config.tight_layout:
            fig.tight_layout()

        filepath = os.path.join(save_path, config.save_filename('rmse_vs_angle'))
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_bias_vs_angle(
    experiment_results: ExperimentResults,
    save_path: str,
    distance: float,
    speed: float,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Plot mean error (bias) as function of crossing angle.

    Shows systematic over/underestimation as function of angle.
    Positive = overestimate depth (too far), negative = underestimate.

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from 1D angle study
    save_path : str
        Directory to save figure
    distance : float
        Fixed closest approach distance [m]
    speed : float
        Fixed missile speed [m/s]
    methods : list of str, optional
        Methods to plot, default all available
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    methods = methods or _get_available_methods(experiment_results)

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_single)

        for method in methods:
            bias_dict = experiment_results.get_bias_vs_angle(method, distance, speed)
            angles = sorted(bias_dict.keys())
            bias_km = [bias_dict[a] / 1e3 for a in angles]

            ax.plot(
                angles, bias_km,
                color=config.method_color(method),
                linewidth=config.line_width,
                marker='o', markersize=config.marker_size,
                label=config.method_label(method),
            )

        ax.axhline(y=0, color='black', linestyle='--',
                   linewidth=config.line_width_reference, alpha=0.5)
        ax.set_xlabel('Crossing Angle [deg]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Mean Error [km]', fontsize=config.fontsize_labels)
        ax.set_title(
            f'Depth Estimation Bias vs Crossing Angle\n'
            f'dist={distance/1e3:.0f} km  |  speed={speed:.0f} m/s',
            fontsize=config.fontsize_title,
        )
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 15))
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.legend(fontsize=config.fontsize_legend)
        ax.grid(True, alpha=0.3)

        if config.tight_layout:
            fig.tight_layout()

        filepath = os.path.join(save_path, config.save_filename('bias_vs_angle'))
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_valid_estimates_vs_angle(
    experiment_results: ExperimentResults,
    save_path: str,
    distance: float,
    speed: float,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Plot number of valid depth estimates as function of crossing angle.

    Shows at which angles the triangulation geometry is too poor for
    reliable estimation (yields zero or few valid estimates).

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from 1D angle study
    save_path : str
        Directory to save figure
    distance : float
        Fixed closest approach distance [m]
    speed : float
        Fixed missile speed [m/s]
    methods : list of str, optional
        Methods to plot, default all available
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    methods = methods or _get_available_methods(experiment_results)

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_single)

        for method in methods:
            n_dict = experiment_results.get_num_estimates_vs_angle(
                method, distance, speed
            )
            angles = sorted(n_dict.keys())
            counts = [n_dict[a] for a in angles]

            ax.plot(
                angles, counts,
                color=config.method_color(method),
                linewidth=config.line_width,
                marker='o', markersize=config.marker_size,
                label=config.method_label(method),
            )

        ax.set_xlabel('Crossing Angle [deg]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Number of Valid Estimates', fontsize=config.fontsize_labels)
        ax.set_title(
            f'Valid Depth Estimates vs Crossing Angle\n'
            f'dist={distance/1e3:.0f} km  |  speed={speed:.0f} m/s',
            fontsize=config.fontsize_title,
        )
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 15))
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.legend(fontsize=config.fontsize_legend)
        ax.grid(True, alpha=0.3)

        if config.tight_layout:
            fig.tight_layout()

        filepath = os.path.join(save_path, config.save_filename('valid_estimates_vs_angle'))
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_rmse_by_distance_all_angles(
    experiment_results: ExperimentResults,
    save_path: str,
    distance: float,
    speed: float,
    method: str = 'two_ray',
    config: Optional[PlotConfig] = None,
):
    """
    RMSE-by-distance curves for all crossing angles in one plot.

    Each angle is one line, colored by a blue→red gradient (0 deg→180 deg).
    Requires full SimulationResults PKLs in experiment_results.full_results.

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from 1D angle study (must contain full_results PKLs)
    save_path : str
        Directory to save figure
    distance : float
        Fixed closest approach distance [m]
    speed : float
        Fixed missile speed [m/s]
    method : str, optional
        Estimation method to plot, default 'two_ray'
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()

    sorted_angles = sorted(experiment_results.crossing_angles)
    n = len(sorted_angles)
    cmap = plt.cm.coolwarm

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_single)

        for i, angle in enumerate(sorted_angles):
            key = (distance, speed, angle)
            if key not in experiment_results.full_results:
                continue

            results = experiment_results.full_results[key]
            estimates = results.depth_estimates.get(method, [])
            if not estimates:
                continue

            # Use first time-offset for two_ray/multi_ray, all estimates otherwise
            if method in ('two_ray', 'multi_ray'):
                offsets = sorted({e.time_offset for e in estimates
                                   if e.time_offset is not None})
                if not offsets:
                    continue
                # Use longest available time offset for best accuracy
                best_offset = offsets[-1]
                estimates = [e for e in estimates if e.time_offset == best_offset]

            if not estimates:
                continue

            true_depths_m = np.array([e.true_depth for e in estimates])
            errors_m = np.array([e.error for e in estimates])

            # Log-spaced bins over the observed depth range
            min_d = true_depths_m.min()
            max_d = true_depths_m.max()
            if min_d <= 0 or max_d <= 0:
                continue
            bins_m = np.logspace(np.log10(min_d), np.log10(max_d), 26)
            bins_km = bins_m / 1e3

            # Step-RMSE per bin
            x_seg, y_seg = [], []
            for b in range(len(bins_m) - 1):
                mask = (true_depths_m >= bins_m[b]) & (true_depths_m < bins_m[b + 1])
                if mask.sum() < 3:
                    if x_seg:
                        frac = i / max(n - 1, 1)
                        ax.step(x_seg, y_seg, where='post',
                                color=cmap(frac), linewidth=config.line_width,
                                alpha=0.8,
                                label=f'{angle:.0f} deg' if not any(
                                    f'{angle:.0f} deg' in str(l.get_label())
                                    for l in ax.lines
                                ) else '_nolegend_')
                        x_seg, y_seg = [], []
                    continue
                x_seg.append(bins_km[b])
                y_seg.append(np.sqrt(np.mean(errors_m[mask] ** 2)) / 1e3)

            if x_seg:
                frac = i / max(n - 1, 1)
                ax.step(x_seg, y_seg, where='post',
                        color=cmap(frac), linewidth=config.line_width,
                        alpha=0.8, label=f'{angle:.0f} deg')

        ax.set_xlabel('True Depth [km]', fontsize=config.fontsize_labels)
        ax.set_ylabel('RMSE [km]', fontsize=config.fontsize_labels)
        ax.set_title(
            f'RMSE by Distance — All Crossing Angles\n'
            f'Method: {method}  |  dist={distance/1e3:.0f} km  |  speed={speed:.0f} m/s',
            fontsize=config.fontsize_title,
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.grid(True, which='both', alpha=0.3)

        # Colorbar instead of per-line legend (too many lines)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 180))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Crossing Angle [deg]', fontsize=config.fontsize_ticks)
        cbar.set_ticks([0, 30, 60, 90, 120, 150, 180])

        if config.tight_layout:
            fig.tight_layout()

        filepath = os.path.join(save_path,
                                config.save_filename(f'rmse_by_distance_all_angles_{method}'))
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_depth_comparison_all_angles(
    experiment_results: ExperimentResults,
    save_path: str,
    distance: float,
    speed: float,
    method: str = 'two_ray',
    time_offset: Optional[float] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Depth comparison over time for all crossing angles in one plot.

    Each angle is one line, colored by a blue→red gradient (0 deg→180 deg).
    A single black dashed reference line shows the true depth (same for all
    runs since distance and speed are fixed).
    Requires full SimulationResults PKLs.

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from 1D angle study (must contain full_results PKLs)
    save_path : str
        Directory to save figure
    distance : float
        Fixed closest approach distance [m]
    speed : float
        Fixed missile speed [m/s]
    method : str, optional
        Estimation method to plot, default 'two_ray'
    time_offset : float, optional
        Which Dt to use for two_ray/multi_ray; defaults to longest available
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()

    sorted_angles = sorted(experiment_results.crossing_angles)
    n = len(sorted_angles)
    cmap = plt.cm.coolwarm

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_wide)

        true_depth_plotted = False

        for i, angle in enumerate(sorted_angles):
            key = (distance, speed, angle)
            if key not in experiment_results.full_results:
                continue

            results = experiment_results.full_results[key]

            # True depth reference (plot once from first available run)
            if not true_depth_plotted and results.observations:
                ref_t0 = results.observations[0].timestamp
                true_times = np.array([
                    (obs.timestamp - ref_t0).total_seconds()
                    for obs in results.observations
                ])
                true_depths_km = np.array([
                    obs.true_depth / 1e3 for obs in results.observations
                ])
                ax.plot(true_times, true_depths_km,
                        color='black', linewidth=config.line_width * 2,
                        linestyle='--', label='True Depth', zorder=10, alpha=0.8)
                true_depth_plotted = True

            estimates = results.depth_estimates.get(method, [])
            if not estimates:
                continue

            # Filter by time_offset for two_ray/multi_ray
            if method in ('two_ray', 'multi_ray'):
                offsets = sorted({e.time_offset for e in estimates
                                   if e.time_offset is not None})
                if not offsets:
                    continue
                selected_offset = time_offset if time_offset in offsets else offsets[-1]
                estimates = [e for e in estimates if e.time_offset == selected_offset]

            if not estimates or not results.observations:
                continue

            ref_t0 = results.observations[0].timestamp
            time_to_depth = {}
            for est in estimates:
                t = (est.timestamp - ref_t0).total_seconds()
                if t not in time_to_depth:
                    time_to_depth[t] = []
                time_to_depth[t].append(est.estimated_depth)

            est_times = np.array(sorted(time_to_depth.keys()))
            est_depths_km = np.array([
                np.mean(time_to_depth[t]) / 1e3 for t in est_times
            ])

            frac = i / max(n - 1, 1)
            ax.plot(est_times, est_depths_km,
                    color=cmap(frac), linewidth=config.line_width,
                    alpha=0.7, label=f'{angle:.0f} deg')

        ax.set_xlabel('Time [seconds]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Depth [km]', fontsize=config.fontsize_labels)
        ax.set_title(
            f'Depth Comparison — All Crossing Angles\n'
            f'Method: {method}  |  dist={distance/1e3:.0f} km  |  speed={speed:.0f} m/s',
            fontsize=config.fontsize_title,
        )
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.grid(True, alpha=0.3)

        # Colorbar for angles
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 180))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Crossing Angle [deg]', fontsize=config.fontsize_ticks)
        cbar.set_ticks([0, 30, 60, 90, 120, 150, 180])

        # Legend only for the true depth reference line
        handles, labels = ax.get_legend_handles_labels()
        true_depth_handles = [(h, l) for h, l in zip(handles, labels)
                              if l == 'True Depth']
        if true_depth_handles:
            ax.legend(*zip(*true_depth_handles), fontsize=config.fontsize_legend)

        if config.tight_layout:
            fig.tight_layout()

        filepath = os.path.join(save_path,
                                config.save_filename(f'depth_comparison_all_angles_{method}'))
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_all_flyby_azimuth_sweep(
    experiment_results: ExperimentResults,
    save_path: str,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Generate all plots for a 1D crossing-angle parameter study (horizontal fly-by).

    Produces:
    - rmse_vs_angle.png              (A1 — RMSE per method)
    - bias_vs_angle.png              (A2 — mean error per method)
    - valid_estimates_vs_angle.png   (A3 — estimate count per method)
    - rmse_by_distance_all_angles_two_ray.png   (B1, if PKLs available)
    - depth_comparison_all_angles_two_ray.png   (B2, if PKLs available)

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from 1D angle sweep (distances and speeds must each have
        exactly one value)
    save_path : str
        Directory to save all figures
    methods : list of str, optional
        Methods to include in summary plots A1-A3
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    os.makedirs(save_path, exist_ok=True)

    if len(experiment_results.distances) != 1 or len(experiment_results.speeds) != 1:
        raise ValueError(
            "plot_all_flyby_azimuth_sweep expects exactly 1 distance and 1 speed. "
            f"Got {len(experiment_results.distances)} distances and "
            f"{len(experiment_results.speeds)} speeds."
        )

    distance = experiment_results.distances[0]
    speed    = experiment_results.speeds[0]

    print(f"\nGenerating angle study plots -> {save_path}")

    # A1 — RMSE vs angle (all methods, summary only)
    plot_rmse_vs_angle(experiment_results, save_path, distance, speed, methods, config)

    # A2 — Bias vs angle (all methods, summary only)
    plot_bias_vs_angle(experiment_results, save_path, distance, speed, methods, config)

    # A3 — Valid estimate count vs angle (all methods, summary only)
    plot_valid_estimates_vs_angle(experiment_results, save_path, distance, speed,
                                  methods, config)

    # B1 + B2 — need full PKLs
    if experiment_results.full_results:
        plot_rmse_by_distance_all_angles(
            experiment_results, save_path, distance, speed,
            method='two_ray', config=config,
        )
        plot_depth_comparison_all_angles(
            experiment_results, save_path, distance, speed,
            method='two_ray', config=config,
        )
    else:
        print("  (skipping B1/B2 — no full SimulationResults PKLs stored)")

    print(f"  [OK] All angle study plots saved!\n")


# =============================================================================
# HELPER
# =============================================================================

def _get_available_methods(experiment_results: ExperimentResults) -> List[str]:
    """Extract all method names that appear in at least one run."""
    methods = set()
    for summary in experiment_results.summaries.values():
        methods.update(summary.stats.keys())
    return sorted(list(methods))


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def plot_all_parameter_study(
    experiment_results: ExperimentResults,
    save_path: str,
    methods: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None,
):
    """
    Generate all parameter study plots at once.

    For each crossing angle a subfolder ``angle_Ndeg/`` is created and
    all four plot types (rmse_vs_distance, rmse_vs_speed, heatmap_rmse,
    heatmap_improvement) are saved there.

    Output structure::

        save_path/
        ├── angle_0deg/
        │   ├── rmse_vs_distance.png
        │   ├── rmse_vs_speed.png
        │   ├── heatmap_rmse.png
        │   └── heatmap_improvement_multi_ray_vs_two_ray.png
        ├── angle_30deg/
        │   └── ...
        └── angle_90deg/
            └── ...

    Parameters
    ----------
    experiment_results : ExperimentResults
        Results from parameter study
    save_path : str
        Root directory; per-angle subfolders are created automatically
    methods : list of str, optional
        Methods to include
    config : PlotConfig, optional
        Visual configuration
    """
    config = config or PlotConfig()
    os.makedirs(save_path, exist_ok=True)

    available = _get_available_methods(experiment_results)
    sorted_angles = sorted(experiment_results.crossing_angles)

    print(f"\nGenerating parameter study plots -> {save_path}")
    print(f"  Angles: {[f'{a:.0f}°' for a in sorted_angles]}")

    for angle in sorted_angles:
        angle_dir = os.path.join(save_path, f'angle_{int(angle)}deg')
        os.makedirs(angle_dir, exist_ok=True)

        print(f"\n  --- Crossing angle: {angle:.0f}° ---")

        plot_rmse_vs_distance(experiment_results, angle_dir, angle, methods, config)
        plot_rmse_vs_speed(experiment_results, angle_dir, angle, methods, config)
        plot_heatmap_rmse(experiment_results, angle_dir, angle, methods, config)

        # Improvement plots (only if relevant methods available)
        if 'multi_ray' in available and 'two_ray' in available:
            plot_heatmap_improvement(
                experiment_results, angle_dir, angle,
                method_improved='multi_ray',
                method_baseline='two_ray',
                config=config,
            )
        if 'kalman' in available and 'two_ray' in available:
            plot_heatmap_improvement(
                experiment_results, angle_dir, angle,
                method_improved='kalman',
                method_baseline='two_ray',
                config=config,
            )

    print(f"\n  [OK] All parameter study plots saved!\n")


# =============================================================================
# ANGULAR STUDY PLOTS (2D: Azimuth × Elevation)
# =============================================================================

def plot_angular_heatmap(
    angular_results: 'AngularStudyResults',
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    2D heatmap of RMSE for each estimation method across azimuth × elevation.

    One subplot per method. X = elevation angle, Y = azimuth angle, color = RMSE.

    Parameters
    ----------
    angular_results : AngularStudyResults
    save_path : str
        Directory to save figure
    config : PlotConfig, optional
    """
    config = config or PlotConfig()

    methods = angular_results.available_methods()
    if not methods:
        return

    sorted_az = sorted(angular_results.azimuths)
    sorted_el = sorted(angular_results.elevations)

    az_labels = [f'{a:.0f}°' for a in sorted_az]
    el_labels = [f'{e:.0f}°' for e in sorted_el]

    # Shared color scale across all methods
    all_rmse = []
    rmse_matrices = {}
    for method in methods:
        matrix = angular_results.get_rmse_matrix(method)
        rmse_matrices[method] = matrix
        all_rmse.extend(matrix[~np.isnan(matrix)].tolist())

    vmin = 0
    vmax = np.nanpercentile(all_rmse, 95) if all_rmse else 1.0

    n = len(methods)
    with plt.style.context(config.style):
        fig, axes = plt.subplots(
            1, n,
            figsize=(config.figsize_square[0] * n, config.figsize_square[1]),
        )
        if n == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            matrix = rmse_matrices[method]

            im = ax.imshow(
                matrix / 1e3,          # → km
                aspect='auto',
                origin='lower',
                cmap=config.heatmap_colormap,
                vmin=vmin / 1e3,
                vmax=vmax / 1e3,
                interpolation='nearest',
            )

            ax.set_xticks(range(len(sorted_el)))
            ax.set_xticklabels(el_labels, rotation=45, ha='right',
                               fontsize=config.fontsize_ticks)
            ax.set_yticks(range(len(sorted_az)))
            ax.set_yticklabels(az_labels, fontsize=config.fontsize_ticks)

            ax.set_xlabel('Elevation angle [deg]', fontsize=config.fontsize_labels)
            ax.set_ylabel('Azimuth angle [deg]', fontsize=config.fontsize_labels)
            ax.set_title(config.method_label(method), fontsize=config.fontsize_title)

            # Annotate cells with RMSE value
            for i in range(len(sorted_az)):
                for j in range(len(sorted_el)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val/1e3:.0f}',
                                ha='center', va='center',
                                fontsize=config.fontsize_ticks - 1,
                                color='white' if val / vmax > 0.6 else 'black')

            plt.colorbar(im, ax=ax, label='RMSE [km]')

        fig.suptitle(
            f'Depth Estimation RMSE: Azimuth × Elevation\n'
            f'({"ground launch" if angular_results.distance is None else f"dist={angular_results.distance/1e3:.0f} km"}, '
            f'speed={angular_results.speed:.0f} m/s)',
            fontsize=config.fontsize_title + 2,
        )

        if config.tight_layout:
            fig.tight_layout()

        filename = config.save_filename('angular_heatmap_rmse')
        filepath = os.path.join(save_path, filename)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


# =============================================================================
# ANGULAR SWEEP COMPARISON PLOTS (overlay multiple runs with different angles)
# =============================================================================

def _load_sim_results_from_disk(angular_results, az, el):
    """
    Load a single SimulationResults from the per-run PKL saved during the study.

    Returns SimulationResults or None if not found / not loadable.
    """
    import os as _os
    from missile_fly_by_simulation.simulation.results import SimulationResults

    # First check the in-memory cache (may be populated for small runs)
    key = (az, el)
    if key in angular_results.full_results:
        return angular_results.full_results[key]

    study_dir = angular_results.metadata.get('study_dir')
    if not study_dir:
        return None

    full_results_dir = _os.path.join(study_dir, 'full_results')
    label = f"az{int(az)}deg_el{int(el)}deg"
    pkl_path = _os.path.join(full_results_dir, f"{label}.pkl")

    if not _os.path.exists(pkl_path):
        return None

    try:
        return SimulationResults.load(pkl_path)
    except Exception as exc:
        print(f"  [!] Could not load {pkl_path}: {exc}")
        return None


def _sweep_tolerance(angular_results):
    """
    Auto-compute tolerance [deg] for Fibonacci sweep selection.

    Uses the typical nearest-neighbour spacing on a hemisphere with N points:
      spacing ≈ sqrt(2*pi / N)  [rad]
    We use 0.6x this as tolerance so a "slice" around the target catches
    roughly 15-20% of the points regardless of N.
    """
    n = max(len(angular_results.summaries), 1)
    spacing_rad = (2.0 * np.pi / n) ** 0.5
    tol = np.degrees(spacing_rad) * 0.6
    return float(np.clip(tol, 3.0, 45.0))


def _get_sweep_runs(angular_results, sweep):
    """
    Get sorted list of (angle_value, SimulationResults) for one sweep direction.

    For grid data  : loads the exact row/column through the target fixed angle.
    For Fibonacci  : uses a tolerance band around the target fixed angle.
                     Tolerance is auto-computed from point density.

    Returns
    -------
    runs : list of (float, SimulationResults)
    angle_label : str        e.g. 'Azimuth angle [deg]'
    sweep_title : str        e.g. 'Azimuth Sweep'
    fixed_info  : str        e.g. 'el=0°' or 'el≈0° (±18°)'
    """
    from missile_fly_by_simulation.constants import (
        ANGULAR_STUDY_AZIMUTH_SWEEP_FIXED_ELEVATION,
        ANGULAR_STUDY_ELEVATION_SWEEP_FIXED_AZIMUTH,
    )

    is_fibonacci = getattr(angular_results, 'sampling', 'grid') == 'fibonacci'
    all_keys = list(angular_results.summaries.keys())

    if not all_keys:
        return [], 'Azimuth angle [deg]', sweep.capitalize() + ' Sweep', ''

    if sweep == 'azimuth':
        target = ANGULAR_STUDY_AZIMUTH_SWEEP_FIXED_ELEVATION

        if is_fibonacci:
            tol = _sweep_tolerance(angular_results)
            selected = sorted(
                [(az, el) for az, el in all_keys if abs(el - target) <= tol],
                key=lambda x: x[0],
            )
            fixed_info = f'el\u224810\u00b0 (\u00b1{tol:.0f}\u00b0, {len(selected)} runs)'
        else:
            all_el = sorted({el for _, el in all_keys})
            fixed_val = min(all_el, key=lambda x: abs(x - target))
            selected = sorted(
                [(az, el) for az, el in all_keys if el == fixed_val],
                key=lambda x: x[0],
            )
            fixed_info = f'el={fixed_val:.0f}\u00b0'

        angle_label = 'Azimuth angle [deg]'
        sweep_title = 'Azimuth Sweep'
        angle_fn    = lambda az, el: az

    else:  # elevation
        target = ANGULAR_STUDY_ELEVATION_SWEEP_FIXED_AZIMUTH

        if is_fibonacci:
            tol = _sweep_tolerance(angular_results)
            selected = sorted(
                [(az, el) for az, el in all_keys if abs(az - target) <= tol],
                key=lambda x: x[1],
            )
            fixed_info = f'az\u224845\u00b0 (\u00b1{tol:.0f}\u00b0, {len(selected)} runs)'
        else:
            all_az = sorted({az for az, _ in all_keys})
            fixed_val = min(all_az, key=lambda x: abs(x - target))
            selected = sorted(
                [(az, el) for az, el in all_keys if az == fixed_val],
                key=lambda x: x[1],
            )
            fixed_info = f'az={fixed_val:.0f}\u00b0'

        angle_label = 'Elevation angle [deg]'
        sweep_title = 'Elevation Sweep'
        angle_fn    = lambda az, el: el

    runs = []
    for az, el in selected:
        sim_res = _load_sim_results_from_disk(angular_results, az, el)
        if sim_res is not None:
            runs.append((angle_fn(az, el), sim_res))
        else:
            print(f"  [!] SimResults not found for az={az:.1f}° el={el:.1f}° — skipping")

    return runs, angle_label, sweep_title, fixed_info


def _sweep_band(runs, extractor_fn, n_grid=300):
    """
    Aggregate time-series data across all runs into statistical bands.

    Parameters
    ----------
    runs : list of (angle, SimulationResults)
    extractor_fn : callable
        f(sim_res) -> (times_s: ndarray, values: ndarray)
    n_grid : int
        Number of points in the common time grid

    Returns
    -------
    (t_grid, median, p25, p75, pmin, pmax) or None if no data
    """
    series = []
    t_max = 0.0
    for _, sim_res in runs:
        t, v = extractor_fn(sim_res)
        if len(t) > 1:
            series.append((np.asarray(t, float), np.asarray(v, float)))
            t_max = max(t_max, float(t[-1]))
    if not series:
        return None
    t_grid = np.linspace(0.0, t_max, n_grid)
    mat = np.array([
        np.interp(t_grid, t, v, left=np.nan, right=np.nan)
        for t, v in series
    ])
    return (
        t_grid,
        np.nanmedian(mat, axis=0),
        np.nanpercentile(mat, 25, axis=0),
        np.nanpercentile(mat, 75, axis=0),
        np.nanmin(mat, axis=0),
        np.nanmax(mat, axis=0),
    )


def _extract_error_over_time(sim_res):
    """(times_s, |error| [km]) from two_ray nearest-dt estimates."""
    estimates = _get_two_ray_nearest_dt(sim_res)
    if not estimates or not sim_res.observations:
        return np.array([]), np.array([])
    t0 = sim_res.observations[0].timestamp
    t  = np.array([(e.timestamp - t0).total_seconds() for e in estimates])
    v  = np.abs(np.array([e.error for e in estimates])) / 1e3
    return t, v


def _extract_est_depth(sim_res):
    """(times_s, estimated_depth [km]) from two_ray nearest-dt estimates."""
    estimates = _get_two_ray_nearest_dt(sim_res)
    if not estimates or not sim_res.observations:
        return np.array([]), np.array([])
    t0 = sim_res.observations[0].timestamp
    t  = np.array([(e.timestamp - t0).total_seconds() for e in estimates])
    v  = np.array([e.estimated_depth for e in estimates]) / 1e3
    return t, v


def _extract_true_depth(sim_res):
    """(times_s, true_depth [km]) from observations."""
    obs = sim_res.observations
    if not obs:
        return np.array([]), np.array([])
    t0 = obs[0].timestamp
    t  = np.array([(o.timestamp - t0).total_seconds() for o in obs])
    v  = np.array([o.true_depth for o in obs]) / 1e3
    return t, v


_BAND_ALPHA_OUTER = 0.12   # Min–Max band
_BAND_ALPHA_INNER = 0.30   # 25–75th pct band
_BAND_LW_MEDIAN  = 2.2


def _draw_band(ax, band, color, label_prefix=''):
    """Draw fill_between bands + median line onto ax from _sweep_band output."""
    if band is None:
        return
    t, med, p25, p75, pmin, pmax = band
    ax.fill_between(t, pmin, pmax, alpha=_BAND_ALPHA_OUTER, color=color)
    ax.fill_between(t, p25,  p75,  alpha=_BAND_ALPHA_INNER, color=color)
    ax.plot(t, med, lw=_BAND_LW_MEDIAN, color=color,
            label=f'{label_prefix}Median')


def _get_two_ray_nearest_dt(sim_results, target_dt=10.0):
    """Return two_ray estimates with time_offset closest to target_dt [s]."""
    all_est = sim_results.depth_estimates.get('two_ray', [])
    if not all_est:
        return []
    offsets = {e.time_offset for e in all_est if e.time_offset is not None}
    if not offsets:
        return all_est
    best_offset = min(offsets, key=lambda x: abs(x - target_dt))
    return [e for e in all_est if e.time_offset == best_offset]


def _compute_los_angles_from_obs(obs_list):
    """
    Compute true LOS off-nadir angle [deg] for each observation.

    Returns (times_s, angles_deg) as numpy arrays.
    """
    if not obs_list:
        return np.array([]), np.array([])

    t0 = obs_list[0].timestamp
    times  = []
    angles = []

    for obs in obs_list:
        t       = (obs.timestamp - t0).total_seconds()
        sat_pos = obs.satellite_state.position
        mis_pos = obs.true_position
        los     = mis_pos - sat_pos

        los_norm = np.linalg.norm(los)
        sat_norm = np.linalg.norm(sat_pos)
        if los_norm < 1.0 or sat_norm < 1.0:
            continue

        los_unit = los / los_norm
        nadir    = -sat_pos / sat_norm         # points from satellite towards Earth
        cos_a    = np.clip(np.dot(nadir, los_unit), -1.0, 1.0)
        times.append(t)
        angles.append(np.degrees(np.arccos(cos_a)))

    return np.array(times), np.array(angles)


def plot_angle_sweep(
    angular_results: 'AngularStudyResults',
    sweep: str,
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Multi-panel sweep comparison — shows angular dependence of accuracy.

    Azimuth sweep  → 2×2 panels (RMSE bar, signed error, |error| over time, gap)
    Elevation sweep → 2×4 panels (full diagnostic suite, all angles shown)

    Colormap: plasma throughout.

    Parameters
    ----------
    angular_results : AngularStudyResults
    sweep : str
        'azimuth' or 'elevation'
    save_path : str
        Directory to save the figure
    config : PlotConfig, optional
    """
    N_SEL = 4
    cmap  = plt.cm.plasma

    config = config or PlotConfig()
    runs, angle_label, sweep_title, fixed_info = _get_sweep_runs(angular_results, sweep)

    if not runs:
        print(f"  [!] No runs for {sweep} sweep — skipping.")
        return

    n_angles = len(runs)

    # ── Per-run RMSE ──────────────────────────────────────────────────────────
    angles_deg = []
    rmse_vals  = []
    for angle, sim_res in runs:
        estimates = _get_two_ray_nearest_dt(sim_res)
        if estimates:
            errs = np.array([e.error for e in estimates])
            rmse_vals.append(float(np.sqrt(np.mean(errs ** 2))) / 1e3)
        else:
            rmse_vals.append(np.nan)
        angles_deg.append(float(angle))

    angles_deg = np.array(angles_deg)
    rmse_vals  = np.array(rmse_vals)

    sort_idx      = np.argsort(angles_deg)
    angles_sorted = angles_deg[sort_idx]
    rmse_sorted   = rmse_vals[sort_idx]
    runs_sorted   = [runs[i] for i in sort_idx]

    angle_min = float(angles_sorted[0])
    angle_max = float(angles_sorted[-1])
    norm      = plt.Normalize(vmin=angle_min, vmax=angle_max)

    def _ac(a):
        return cmap(norm(float(a)))

    bar_colors = [_ac(a) for a in angles_sorted]
    bar_w = (max(np.diff(angles_sorted).min() * 0.8, 1.0) if n_angles > 1 else 5.0)

    # ── Select N_SEL representative angles by RMSE rank ───────────────────────
    valid_idx = np.where(~np.isnan(rmse_sorted))[0]
    if len(valid_idx) >= N_SEL:
        rmse_rank     = np.argsort(rmse_sorted[valid_idx])
        pick          = np.round(np.linspace(0, len(rmse_rank) - 1, N_SEL)).astype(int)
        sel_in_sorted = [valid_idx[rmse_rank[p]] for p in pick]
    else:
        sel_in_sorted = valid_idx[:N_SEL].tolist()

    sel_runs   = [runs_sorted[i] for i in sel_in_sorted]
    sel_angles = [angles_sorted[i] for i in sel_in_sorted]
    sel_rmses  = [rmse_sorted[i]   for i in sel_in_sorted]
    sel_colors = [_ac(a)           for a in sel_angles]
    sel_labels = ['Best', 'Low', 'High', 'Worst']

    dist_str = ('ground launch' if angular_results.distance is None
                else f'dist={angular_results.distance/1e3:.0f} km')

    def _add_colorbar(fig, left, width, bottom=0.04, height=0.020):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([left, bottom, width, height])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f'{angle_label} [°]', fontsize=config.fontsize_legend)
        cbar.ax.tick_params(labelsize=config.fontsize_ticks - 1)

    def _bar_rmse(ax):
        ax.bar(angles_sorted, rmse_sorted, color=bar_colors,
               width=bar_w, edgecolor='none', alpha=0.85)
        ax.plot(angles_sorted, rmse_sorted, color='#333333',
                lw=1.2, marker='o', markersize=3, zorder=5)
        ax.set_xlabel(f'{angle_label} [°]', fontsize=config.fontsize_labels)
        ax.set_ylabel('RMSE [km]', fontsize=config.fontsize_labels)
        ax.set_title(f'RMSE vs. {angle_label}\n(two_ray, Δt≈10 s)',
                     fontsize=config.fontsize_title)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.grid(True, alpha=0.3, axis='y')

    def _scatter_signed_error(ax):
        for lbl, (sel_angle, sim_res), rmse_v, col in zip(
                sel_labels, sel_runs, sel_rmses, sel_colors):
            ests = _get_two_ray_nearest_dt(sim_res)
            if ests:
                ax.scatter([e.true_depth / 1e3 for e in ests],
                           [e.error / 1e3       for e in ests],
                           color=col, s=8, alpha=0.6, rasterized=True,
                           label=f'{lbl}: {sel_angle:.0f}°  ({rmse_v:.1f} km)')
        ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.6)
        ax.set_xlabel('True Depth [km]', fontsize=config.fontsize_labels)
        ax.set_ylabel('Error  (est. − true) [km]', fontsize=config.fontsize_labels)
        ax.set_title('Signed Error vs. True Depth\n(Best / Low / High / Worst)',
                     fontsize=config.fontsize_title)
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=config.fontsize_legend - 2, loc='upper left')

    def _plot_abs_error_sel(ax):
        for lbl, (sel_angle, sim_res), rmse_v, col in zip(
                sel_labels, sel_runs, sel_rmses, sel_colors):
            t, v = _extract_error_over_time(sim_res)
            if len(t) > 1:
                ax.plot(t, v, lw=2.0, color=col,
                        label=f'{lbl}: {sel_angle:.0f}°  ({rmse_v:.1f} km RMSE)')
        ax.set_xlabel('Time [s]', fontsize=config.fontsize_labels)
        ax.set_ylabel('|Error| [km]', fontsize=config.fontsize_labels)
        ax.set_title('|Error| over Time\n(Best / Low / High / Worst)',
                     fontsize=config.fontsize_title)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=config.fontsize_legend - 2, loc='upper right')

    def _plot_gap_sel(ax):
        has_gap = False
        for lbl, (sel_angle, sim_res), col in zip(sel_labels, sel_runs, sel_colors):
            ests     = _get_two_ray_nearest_dt(sim_res)
            gap_ests = [e for e in ests if e.triangulation_gap is not None]
            if gap_ests and sim_res.observations:
                t0 = sim_res.observations[0].timestamp
                t  = np.array([(e.timestamp - t0).total_seconds() for e in gap_ests])
                v  = np.array([e.triangulation_gap / 1e3 for e in gap_ests])
                ax.plot(t, v, lw=2.0, color=col, label=f'{lbl}: {sel_angle:.0f}°')
                has_gap = True
        if has_gap:
            ax.set_ylabel('Triangulation gap [km]', fontsize=config.fontsize_labels)
            ax.set_title('Triangulation Gap over Time\n(lower = better geometry)',
                         fontsize=config.fontsize_title)
            ax.set_ylim(bottom=0)
        else:
            for lbl, (sel_angle, sim_res), col in zip(sel_labels, sel_runs, sel_colors):
                t, v = _compute_los_angles_from_obs(sim_res.observations)
                if len(t) > 1:
                    ax.plot(t, v, lw=2.0, color=col, label=f'{lbl}: {sel_angle:.0f}°')
            ax.set_ylabel('LOS off-nadir angle [°]', fontsize=config.fontsize_labels)
            ax.set_title('LOS Geometry over Time\n(off-nadir angle)',
                         fontsize=config.fontsize_title)
        ax.set_xlabel('Time [s]', fontsize=config.fontsize_labels)
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=config.fontsize_legend - 2, loc='upper right')

    with plt.style.context(config.style):
        if sweep == 'elevation':
            # ════════════════════════════════════════════════════════════════
            # ELEVATION SWEEP — 2 × 4 panel layout
            # ════════════════════════════════════════════════════════════════
            fig, axes = plt.subplots(2, 4, figsize=(26, 12))
            ax_err_all = axes[0, 0]   # |error| over time — ALL angles
            ax_rmse    = axes[0, 1]   # RMSE vs. angle bar chart
            ax_ed_td   = axes[0, 2]   # estimated vs. true depth [km]
            ax_scat    = axes[0, 3]   # signed error vs. true depth
            ax_depth   = axes[1, 0]   # depth over time (est solid, true dashed)
            ax_err_s   = axes[1, 1]   # |error| over time — 4 selected
            ax_los     = axes[1, 2]   # LOS off-nadir angle
            ax_gap     = axes[1, 3]   # triangulation gap

            # ── (0,0): |Error| over time — all angles ────────────────────
            for angle, sim_res in runs_sorted:
                t, v = _extract_error_over_time(sim_res)
                if len(t) > 1:
                    ax_err_all.plot(t, v, lw=1.4, color=_ac(angle), alpha=0.85)
            ax_err_all.set_xlabel('Time [s]', fontsize=config.fontsize_labels)
            ax_err_all.set_ylabel('|Error| [km]', fontsize=config.fontsize_labels)
            ax_err_all.set_title(
                f'|Error| over Time\n(all {n_angles} elevation angles)',
                fontsize=config.fontsize_title)
            ax_err_all.set_ylim(bottom=0)
            ax_err_all.tick_params(labelsize=config.fontsize_ticks)
            ax_err_all.grid(True, alpha=0.3)

            # ── (0,1): RMSE bar chart ─────────────────────────────────────
            _bar_rmse(ax_rmse)

            # ── (0,2): Estimated vs. True Depth [km] — selected 4 ────────
            all_true_km = []
            for lbl, (sel_angle, sim_res), col in zip(
                    sel_labels, sel_runs, sel_colors):
                ests = _get_two_ray_nearest_dt(sim_res)
                if ests:
                    true_km = [e.true_depth / 1e3 for e in ests]
                    est_km  = [e.estimated_depth / 1e3 for e in ests]
                    ax_ed_td.scatter(true_km, est_km,
                                     color=col, s=8, alpha=0.6, rasterized=True,
                                     label=f'{lbl}: {sel_angle:.0f}°')
                    all_true_km.extend(true_km)
            if all_true_km:
                mn, mx = min(all_true_km), max(all_true_km)
                ax_ed_td.plot([mn, mx], [mn, mx], 'k--', lw=1.2,
                              label='y=x (perfect)', zorder=10)
            ax_ed_td.set_xlabel('True Depth [km]', fontsize=config.fontsize_labels)
            ax_ed_td.set_ylabel('Estimated Depth [km]', fontsize=config.fontsize_labels)
            ax_ed_td.set_title('Estimated vs. True Depth\n(Best / Low / High / Worst)',
                               fontsize=config.fontsize_title)
            ax_ed_td.tick_params(labelsize=config.fontsize_ticks)
            ax_ed_td.grid(True, alpha=0.3)
            ax_ed_td.legend(fontsize=config.fontsize_legend - 2, loc='upper left')

            # ── (0,3): Signed error vs. true depth — selected 4 ──────────
            _scatter_signed_error(ax_scat)

            # ── (1,0): Depth over time — est (solid) + true (dashed) ─────
            for lbl, (sel_angle, sim_res), col in zip(
                    sel_labels, sel_runs, sel_colors):
                t_est,  v_est  = _extract_est_depth(sim_res)
                t_true, v_true = _extract_true_depth(sim_res)
                if len(t_est) > 1:
                    ax_depth.plot(t_est, v_est, lw=1.8, color=col,
                                  label=f'{lbl}: {sel_angle:.0f}°')
                if len(t_true) > 1:
                    ax_depth.plot(t_true, v_true, lw=1.1, color=col,
                                  ls='--', alpha=0.55)
            ax_depth.set_xlabel('Time [s]', fontsize=config.fontsize_labels)
            ax_depth.set_ylabel('Depth [km]', fontsize=config.fontsize_labels)
            ax_depth.set_title('Depth over Time\n(solid = estimated,  dashed = true)',
                               fontsize=config.fontsize_title)
            ax_depth.set_ylim(bottom=0)
            ax_depth.tick_params(labelsize=config.fontsize_ticks)
            ax_depth.grid(True, alpha=0.3)
            ax_depth.legend(fontsize=config.fontsize_legend - 2, loc='upper left')

            # ── (1,1): |Error| over time — selected 4 ────────────────────
            _plot_abs_error_sel(ax_err_s)

            # ── (1,2): LOS off-nadir angle — selected 4 ──────────────────
            for lbl, (sel_angle, sim_res), col in zip(
                    sel_labels, sel_runs, sel_colors):
                t, v = _compute_los_angles_from_obs(sim_res.observations)
                if len(t) > 1:
                    ax_los.plot(t, v, lw=2.0, color=col,
                                label=f'{lbl}: {sel_angle:.0f}°')
            ax_los.set_xlabel('Time [s]', fontsize=config.fontsize_labels)
            ax_los.set_ylabel('LOS off-nadir angle [°]', fontsize=config.fontsize_labels)
            ax_los.set_title('Observation Geometry\n(true LOS off-nadir angle)',
                             fontsize=config.fontsize_title)
            ax_los.tick_params(labelsize=config.fontsize_ticks)
            ax_los.grid(True, alpha=0.3)
            ax_los.legend(fontsize=config.fontsize_legend - 2, loc='upper right')

            # ── (1,3): Triangulation gap — selected 4 ────────────────────
            _plot_gap_sel(ax_gap)

            # ── Shared horizontal colorbar ────────────────────────────────
            _add_colorbar(fig, left=0.12, width=0.76, bottom=0.03, height=0.018)

            fig.suptitle(
                f'Elevation Sweep  |  {fixed_info}\n'
                f'{dist_str},  speed={angular_results.speed:.0f} m/s,'
                f'  n={n_angles} angles',
                fontsize=config.fontsize_title + 2,
            )
            fig.subplots_adjust(hspace=0.52, wspace=0.33, bottom=0.11)

        else:
            # ════════════════════════════════════════════════════════════════
            # AZIMUTH SWEEP — 2 × 2 panel layout (unchanged)
            # ════════════════════════════════════════════════════════════════
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            ax_rmse = axes[0, 0]
            ax_scat = axes[0, 1]
            ax_err  = axes[1, 0]
            ax_gap  = axes[1, 1]

            _bar_rmse(ax_rmse)
            _scatter_signed_error(ax_scat)
            _plot_abs_error_sel(ax_err)
            _plot_gap_sel(ax_gap)

            _add_colorbar(fig, left=0.15, width=0.70, bottom=0.04, height=0.022)

            fig.suptitle(
                f'Angular Sweep — {sweep_title}  |  {fixed_info}\n'
                f'{dist_str},  speed={angular_results.speed:.0f} m/s,'
                f'  n={n_angles} angles',
                fontsize=config.fontsize_title + 2,
            )
            fig.subplots_adjust(hspace=0.48, wspace=0.30, bottom=0.13)

        filename = config.save_filename(f'sweep_{sweep}')
        filepath = os.path.join(save_path, filename)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_sweep_rmse_vs_dt(
    angular_results: 'AngularStudyResults',
    sweep: str,
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    RMSE vs lookback time Δt — Best / Low / High / Worst angle by overall RMSE.

    Shows whether the optimal lookback time differs between launch angles.
    Plots 4 representative angles ranked by their overall RMSE, using the
    same plasma colormap and angle selection as plot_angle_sweep.
    Method: two_ray (all available Δt offsets).

    Parameters
    ----------
    angular_results : AngularStudyResults
    sweep : str
        'azimuth' or 'elevation'
    save_path : str
    config : PlotConfig, optional
    """
    N_SEL = 4
    cmap  = plt.cm.plasma

    config = config or PlotConfig()
    runs, angle_label, sweep_title, fixed_info = _get_sweep_runs(angular_results, sweep)

    if not runs:
        print(f"  [!] No runs for {sweep} sweep — skipping rmse_vs_dt.")
        return

    # ---- Build per-run RMSE(Δt) curves ----
    # run_curves: list of (angle, {dt: rmse_km})
    run_curves = []
    for angle, sim_res in runs:
        all_est = sim_res.depth_estimates.get('two_ray', [])
        by_dt: dict = {}
        for e in all_est:
            if e.time_offset is not None:
                by_dt.setdefault(e.time_offset, []).append(e.error)
        if not by_dt:
            continue
        rmse_by_dt = {dt: float(np.sqrt(np.mean(np.array(errs) ** 2))) / 1e3
                      for dt, errs in by_dt.items()}
        run_curves.append((float(angle), rmse_by_dt))

    if not run_curves:
        print(f"  [!] No two_ray data for {sweep} sweep — skipping rmse_vs_dt.")
        return

    # ---- Rank angles by overall RMSE (mean across all Δt) ----
    overall_rmse = [(angle, float(np.mean(list(d.values()))))
                    for angle, d in run_curves]
    overall_rmse.sort(key=lambda x: x[1])   # ascending = best first

    n_angles = len(overall_rmse)
    pick     = np.round(np.linspace(0, n_angles - 1, N_SEL)).astype(int)
    sel_idx  = [int(p) for p in pick]

    angle_min = min(a for a, _ in overall_rmse)
    angle_max = max(a for a, _ in overall_rmse)
    norm      = plt.Normalize(vmin=angle_min, vmax=angle_max)

    def _ac(a):
        return cmap(norm(float(a)))

    sel_labels = ['Best', 'Low', 'High', 'Worst']

    with plt.style.context(config.style):
        fig, ax = plt.subplots(figsize=config.figsize_single)

        for i, idx in enumerate(sel_idx):
            angle, overall_r = overall_rmse[idx]
            # Find matching run_curve for this angle
            _, rmse_by_dt = next(c for c in run_curves if c[0] == angle)
            dts  = np.array(sorted(rmse_by_dt.keys()))
            vals = np.array([rmse_by_dt[dt] for dt in dts])
            lbl  = f'{sel_labels[i]}: {angle:.0f}°  ({overall_r:.1f} km avg)'
            ax.plot(dts, vals, lw=2.2, color=_ac(angle),
                    marker='o', markersize=config.marker_size + 1,
                    label=lbl)

        ax.set_xlabel('Lookback time Δt [s]', fontsize=config.fontsize_labels)
        ax.set_ylabel('RMSE [km]', fontsize=config.fontsize_labels)
        dist_str = ('ground launch' if angular_results.distance is None
                    else f'dist={angular_results.distance/1e3:.0f} km')
        ax.set_title(
            f'RMSE vs. Lookback Time — {sweep_title}  |  {fixed_info}\n'
            f'(two_ray, {dist_str}, speed={angular_results.speed:.0f} m/s, '
            f'n={n_angles} angles)',
            fontsize=config.fontsize_title,
        )
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=config.fontsize_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=config.fontsize_legend, loc='upper right')

        # Shared horizontal colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(bottom=0.18)
        cbar_ax = fig.add_axes([0.15, 0.06, 0.70, 0.030])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f'{angle_label} [°]', fontsize=config.fontsize_legend)
        cbar.ax.tick_params(labelsize=config.fontsize_ticks - 1)

        filename = config.save_filename(f'sweep_{sweep}_rmse_vs_dt')
        filepath = os.path.join(save_path, filename)
        fig.savefig(filepath, dpi=config.dpi, format=config.file_format,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {filepath}")


def plot_all_angle_sweeps(
    angular_results: 'AngularStudyResults',
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Generate azimuth-sweep and elevation-sweep comparison plots.

    Produces per sweep direction:
    - sweep_{az|el}.png            (2\u00d72 panel: error, scatter, depth, geometry)
    - sweep_{az|el}_rmse_vs_dt.png (RMSE vs lookback time \u0394t)

    Parameters
    ----------
    angular_results : AngularStudyResults
    save_path : str
        Directory to save the figures
    config : PlotConfig, optional
    """
    os.makedirs(save_path, exist_ok=True)
    config = config or PlotConfig()

    print(f"\nGenerating angle sweep comparison plots -> {save_path}")

    for sweep in ('azimuth', 'elevation'):
        plot_angle_sweep(angular_results, sweep, save_path, config)
        plot_sweep_rmse_vs_dt(angular_results, sweep, save_path, config)

    print(f"  [OK] Angle sweep plots saved!\n")


def plot_hemisphere_polar(
    angular_results: 'AngularStudyResults',
    method: str,
    save_path: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Azimuthal polar projection of hemisphere RMSE.

    Displays the upper hemisphere as a top-down view:
    - Center = zenith (elevation 90 deg)
    - Edge   = horizon (elevation 0 deg)
    - Angle  = azimuth (0 deg = along-track = North, clockwise)
    - Color  = RMSE [km]

    Works for both regular grid and Fibonacci sampled data.

    Parameters
    ----------
    angular_results : AngularStudyResults
    method : str
        Estimation method key (e.g. 'two_ray_dt10')
    save_path : str, optional
        Directory to save figure. If None, figure is returned but not saved.
    config : PlotConfig, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    cfg = config or PlotConfig()

    # Collect raw (theta_rad, r, rmse_km) from all summaries for this method
    az_rad_list, r_list, rmse_km_list = [], [], []
    for (az, el), summary in angular_results.summaries.items():
        rmse = summary.rmse(method)
        if not np.isfinite(rmse):
            continue
        az_rad_list.append(math.radians(az))
        r_list.append(1.0 - el / 90.0)   # 0 = zenith, 1 = horizon
        rmse_km_list.append(rmse / 1e3)

    if not rmse_km_list:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_title(f'No data for method: {method}')
        return fig

    az_arr   = np.array(az_rad_list)
    r_arr    = np.array(r_list)
    rmse_arr = np.array(rmse_km_list)

    # Dense polar grid for interpolated background
    theta_g, r_g = np.meshgrid(
        np.linspace(0, 2 * np.pi, 360),
        np.linspace(0, 1, 180),
    )

    from scipy.interpolate import griddata  # lazy import — slow on Windows at module level

    pts = np.column_stack([az_arr, r_arr])
    xi  = np.column_stack([theta_g.ravel(), r_g.ravel()])

    rmse_g_lin = griddata(pts, rmse_arr, xi, method='linear').reshape(theta_g.shape)
    rmse_g_near = griddata(pts, rmse_arr, xi, method='nearest').reshape(theta_g.shape)
    rmse_g = np.where(np.isfinite(rmse_g_lin), rmse_g_lin, rmse_g_near)

    if vmin is None:
        vmin = float(np.nanmin(rmse_arr))
    if vmax is None:
        vmax = float(np.nanmax(rmse_arr))

    fig, ax = plt.subplots(
        figsize=cfg.figsize_square,
        subplot_kw={'projection': 'polar'},
    )

    ax.pcolormesh(theta_g, r_g, rmse_g, cmap='RdYlGn_r',
                  vmin=vmin, vmax=vmax, shading='auto')
    sc = ax.scatter(az_arr, r_arr, c=rmse_arr, cmap='RdYlGn_r',
                    vmin=vmin, vmax=vmax, s=12, zorder=5,
                    linewidths=0, alpha=0.85)

    # 0 deg at top (along-track / North), clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Radial ticks: labels show elevation angle
    ax.set_rticks([0.0, 1/3, 2/3, 1.0])
    ax.set_yticklabels(['90\u00b0', '60\u00b0', '30\u00b0', '0\u00b0'],
                       fontsize=cfg.fontsize_ticks)
    ax.tick_params(axis='x', labelsize=cfg.fontsize_ticks)

    sampling = getattr(angular_results, 'sampling', 'grid')
    n_points = len(rmse_km_list)
    ax.set_title(
        f'Hemisphere RMSE \u2014 {method}\n'
        f'speed = {angular_results.speed:.0f} m/s  |  '
        f'{n_points} points ({sampling})',
        fontsize=cfg.fontsize_title, pad=15,
    )

    cbar = plt.colorbar(sc, ax=ax, label='RMSE [km]',
                        shrink=0.6, pad=0.12)
    cbar.ax.tick_params(labelsize=cfg.fontsize_ticks)
    cbar.set_label('RMSE [km]', fontsize=cfg.fontsize_labels)

    # Cardinal direction labels
    ax.set_thetagrids(
        [0, 90, 180, 270],
        labels=['0\u00b0\n(along-track)', '90\u00b0\n(cross-track)', '180\u00b0', '270\u00b0'],
        fontsize=cfg.fontsize_ticks,
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        safe_method = method.replace(' ', '_').replace('/', '_')
        fname = os.path.join(save_path, f'hemisphere_polar_{safe_method}.png')
        fig.savefig(fname, dpi=cfg.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {fname}")

    return fig


def plot_all_launch_az_el_sweep(
    angular_results: 'AngularStudyResults',
    save_path: str,
    config: Optional[PlotConfig] = None,
):
    """
    Generate all plots for a 2D launch sweep (ground-launch scenario).

    Produces:
    - angular_heatmap_rmse.png  (RMSE heatmap per method)
    - sweep_azimuth.png         (2\u00d72 overlay: azimuth sweep, el=0\u00b0)
    - sweep_elevation.png       (2\u00d72 overlay: elevation sweep, az=45\u00b0)

    Parameters
    ----------
    angular_results : AngularStudyResults
    save_path : str
        Directory to save figures
    config : PlotConfig, optional
    """
    os.makedirs(save_path, exist_ok=True)
    config = config or PlotConfig()

    is_fibonacci = getattr(angular_results, 'sampling', 'grid') == 'fibonacci'

    if not is_fibonacci:
        # Heatmap requires a regular az×el grid
        plot_angular_heatmap(angular_results, save_path, config)

    # Sweep plots work for both grid and Fibonacci (Fibonacci uses tolerance band)
    plot_all_angle_sweeps(angular_results, save_path, config)

    # Hemisphere polar projection — works for both grid and Fibonacci data
    methods = set()
    for summary in angular_results.summaries.values():
        methods.update(summary.stats.keys())

    # Shared color scale across all methods for comparability
    all_rmse_km = []
    for method in methods:
        for summary in angular_results.summaries.values():
            v = summary.rmse(method)
            if np.isfinite(v):
                all_rmse_km.append(v / 1e3)
    global_vmin = float(np.min(all_rmse_km)) if all_rmse_km else 0.0
    global_vmax = float(np.max(all_rmse_km)) if all_rmse_km else 1.0

    for method in sorted(methods):
        plot_hemisphere_polar(angular_results, method, save_path, config,
                              vmin=global_vmin, vmax=global_vmax)

    print(f"\n  [OK] All angular study plots saved!\n")

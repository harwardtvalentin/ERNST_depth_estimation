"""
Visualization module for satellite-missile simulation results.

Provides three groups of plots:

1. Statistical plots (Category 1) - single run analysis
   plot_error_histogram, plot_error_vs_time_offset,
   plot_error_over_time, plot_estimated_vs_true_depth,
   plot_method_comparison_bar, plot_triangulation_gap_vs_error

2. Trajectory plots (Category 2) - geometry visualization
   plot_3d_orbit, plot_pixel_track,
   plot_relative_geometry, plot_ground_track

3. Parameter study plots - multi-run comparison
   plot_rmse_vs_distance, plot_rmse_vs_speed,
   plot_heatmap_rmse, plot_heatmap_improvement

Configuration
-------------
All plot functions accept a PlotConfig object.
Use presets for common use cases:

    from missile_fly_by_simulation.visualization.plot_config import (
        thesis_config,        # 300 DPI, PDF
        presentation_config,  # Large fonts
        preview_config,       # 72 DPI, fast
    )

Examples
--------
>>> from missile_fly_by_simulation.visualization import (
...     plot_all_statistical,
...     plot_all_trajectory,
...     plot_all_parameter_study,
... )
>>> from missile_fly_by_simulation.visualization.plot_config import thesis_config
>>>
>>> config = thesis_config()
>>>
>>> # Single run plots
>>> plot_all_statistical(results, save_path='plots/', config=config)
>>> plot_all_trajectory(results, save_path='plots/', config=config)
>>>
>>> # Multi-run parameter study plots
>>> plot_all_parameter_study(experiment_results, save_path='plots/', config=config)
"""

from missile_fly_by_simulation.visualization.plot_config import (
    PlotConfig,
    thesis_config,
    presentation_config,
    preview_config,
)

from missile_fly_by_simulation.visualization.statistical_plots import (
    plot_error_histogram,
    plot_error_vs_time_offset,
    plot_error_over_time,
    plot_estimated_vs_true_depth,
    plot_method_comparison_bar,
    plot_triangulation_gap_vs_error,
    plot_all_statistical,
)

from missile_fly_by_simulation.visualization.trajectory_plots import (
    plot_3d_orbit,
    plot_pixel_track,
    plot_relative_geometry,
    plot_ground_track,
    plot_all_trajectory,
)

from missile_fly_by_simulation.visualization.parameter_study_plots import (
    plot_rmse_vs_distance,
    plot_rmse_vs_speed,
    plot_heatmap_rmse,
    plot_heatmap_improvement,
    plot_all_parameter_study,
    plot_rmse_vs_angle,
    plot_bias_vs_angle,
    plot_valid_estimates_vs_angle,
    plot_rmse_by_distance_all_angles,
    plot_depth_comparison_all_angles,
    plot_all_flyby_azimuth_sweep,
    plot_angular_heatmap,
    plot_all_launch_az_el_sweep,
    plot_angle_sweep,
    plot_all_angle_sweeps,
    plot_hemisphere_polar,
)

__all__ = [
    # Config
    'PlotConfig',
    'thesis_config',
    'presentation_config',
    'preview_config',

    # Statistical (Category 1)
    'plot_error_histogram',
    'plot_error_vs_time_offset',
    'plot_error_over_time',
    'plot_estimated_vs_true_depth',
    'plot_method_comparison_bar',
    'plot_triangulation_gap_vs_error',
    'plot_all_statistical',

    # Trajectory (Category 2)
    'plot_3d_orbit',
    'plot_pixel_track',
    'plot_relative_geometry',
    'plot_ground_track',
    'plot_all_trajectory',

    # Parameter study (DSA grid)
    'plot_rmse_vs_distance',
    'plot_rmse_vs_speed',
    'plot_heatmap_rmse',
    'plot_heatmap_improvement',
    'plot_all_parameter_study',

    # Flyby sweep (1D crossing-angle sweep, horizontal fly-by)
    'plot_rmse_vs_angle',
    'plot_bias_vs_angle',
    'plot_valid_estimates_vs_angle',
    'plot_rmse_by_distance_all_angles',
    'plot_depth_comparison_all_angles',
    'plot_all_flyby_azimuth_sweep',

    # Launch sweep (2D azimuth × elevation, ground-launch scenario)
    'plot_angular_heatmap',
    'plot_all_launch_az_el_sweep',
    'plot_angle_sweep',
    'plot_all_angle_sweeps',
    'plot_hemisphere_polar',
]
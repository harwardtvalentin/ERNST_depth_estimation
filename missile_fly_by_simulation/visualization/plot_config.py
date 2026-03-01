"""
Plot configuration for all visualizations.

This module defines the PlotConfig dataclass which controls
the visual appearance of all plots. Change settings here
once and they apply consistently across all plots.

Classes
-------
PlotConfig
    Visual configuration for all plots
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class PlotConfig:
    """
    Visual configuration for all plots.

    Change settings here once and they apply
    consistently across ALL plot functions.

    Attributes
    ----------
    Figure Sizes
    figsize_single : tuple
        Standard single plot (width, height) in inches
    figsize_wide : tuple
        Wide comparison plot (width, height) in inches
    figsize_square : tuple
        Square plot for heatmaps (width, height) in inches
    figsize_3d : tuple
        3D orbital plot (width, height) in inches
    figsize_tall : tuple
        Tall plot for multi-panel (width, height) in inches

    Font Sizes
    fontsize_title : int
        Plot title font size
    fontsize_labels : int
        Axis label font size
    fontsize_ticks : int
        Tick label font size
    fontsize_legend : int
        Legend font size

    Output Settings
    dpi : int
        Resolution for saved figures (300 for thesis, 150 for quick preview)
    file_format : str
        Output format: 'png' for Word/PowerPoint, 'pdf' for LaTeX
    style : str
        Matplotlib style sheet

    Method Colors
    color_two_ray : str
        Color for two-ray triangulation method
    color_multi_ray : str
        Color for multi-ray least squares method
    color_kalman : str
        Color for Kalman filter method

    Other
    alpha_histogram : float
        Transparency for overlaid histograms [0-1]
    alpha_uncertainty_band : float
        Transparency for uncertainty bands [0-1]
    marker_size : int
        Marker size for scatter plots

    Examples
    --------
    >>> # Default config (good for quick preview)
    >>> config = PlotConfig()
    >>>
    >>> # Thesis config (high resolution, PDF output)
    >>> thesis_config = PlotConfig(
    ...     dpi=300,
    ...     file_format='pdf',
    ...     style='seaborn-v0_8-paper'
    ... )
    >>>
    >>> # Presentation config (larger fonts, bigger figures)
    >>> presentation_config = PlotConfig(
    ...     figsize_single=(14, 8),
    ...     fontsize_title=20,
    ...     fontsize_labels=16,
    ...     fontsize_ticks=14,
    ...     dpi=150
    ... )
    >>>
    >>> # Quick preview config (low res, fast)
    >>> preview_config = PlotConfig(dpi=72, file_format='png')
    >>>
    >>> # Pass to any plot function
    >>> plot_error_histogram(results, config=thesis_config, save_path='...')
    """

    # =========================================================================
    # FIGURE SIZES (width, height) in inches
    # =========================================================================

    figsize_single: Tuple[float, float] = (10, 6)
    # Standard single plot - most Category 1 plots

    figsize_wide: Tuple[float, float] = (14, 6)
    # Wide comparison plots - method comparison bar chart

    figsize_square: Tuple[float, float] = (8, 8)
    # Square - heatmaps

    figsize_3d: Tuple[float, float] = (10, 10)
    # 3D orbital plot

    figsize_tall: Tuple[float, float] = (10, 12)
    # Tall multi-panel - relative geometry (two stacked subplots)

    figsize_pixel_track: Tuple[float, float] = (8, 6)
    # Pixel track - proportional to camera resolution (1024x720)

    # =========================================================================
    # FONT SIZES
    # =========================================================================

    fontsize_title: int = 14
    fontsize_labels: int = 12
    fontsize_ticks: int = 10
    fontsize_legend: int = 10
    fontsize_annotations: int = 9
    # For text labels on heatmap cells etc.

    # =========================================================================
    # OUTPUT SETTINGS
    # =========================================================================

    dpi: int = 150
    # 72:  screen preview (fast)
    # 150: good quality preview
    # 300: thesis/publication quality (slow to save)

    file_format: str = 'png'
    # 'png': for Word, PowerPoint, general use
    # 'pdf': for LaTeX (vector, infinite resolution)
    # 'svg': for vector graphics editors

    style: str = 'seaborn-v0_8-paper'
    # 'seaborn-v0_8-paper':    clean, publication-ready
    # 'seaborn-v0_8-whitegrid': white background with grid
    # 'default':               standard matplotlib
    # 'ggplot':                ggplot2-inspired

    tight_layout: bool = True
    # Automatically adjust subplot spacing

    # =========================================================================
    # METHOD COLORS (consistent across ALL plots)
    # =========================================================================

    color_two_ray: str = '#2196F3'
    # Blue - baseline method

    color_multi_ray: str = '#FF9800'
    # Orange - improved method

    color_kalman: str = '#4CAF50'
    # Green - advanced method

    color_iterative: str = '#9C27B0'
    # Purple - novel iterative method

    # Iterative method convergence variants (reddish-purple → bluish-purple gradient)
    color_iterative_k1: str = '#D81B60'   # reddish-purple/magenta  — iteration 1
    color_iterative_k2: str = '#AB47BC'   # medium purple           — iteration 2
    color_iterative_k3: str = '#7B1FA2'   # deep purple             — iteration 3
    color_iterative_k4: str = '#5E35B1'   # indigo-purple           — iteration 4
    color_iterative_k5: str = '#3949AB'   # bluish-purple (indigo)  — iteration 5 (= converged)

    color_true: str = "#F43636"
    # Red - ground truth / reference lines

    color_satellite: str = '#3F51B5'
    # Dark blue - satellite trajectory

    color_missile: str = '#E91E63'
    # Pink/red - missile trajectory

    # =========================================================================
    # TRANSPARENCY
    # =========================================================================

    alpha_histogram: float = 0.6
    # Overlaid histograms (semi-transparent so all visible)

    alpha_uncertainty_band: float = 0.2
    # Shaded uncertainty bands around lines

    alpha_scatter: float = 0.4
    # Scatter plot points (many overlapping points)

    # =========================================================================
    # MARKERS AND LINES
    # =========================================================================

    marker_size: int = 4
    # Scatter plot marker size

    line_width: float = 1.5
    # Line plots

    line_width_reference: float = 1.0
    # Reference lines (y=x perfect line, zero error line)

    # =========================================================================
    # HEATMAP SETTINGS
    # =========================================================================

    heatmap_colormap: str = 'YlOrRd'
    # Colormap for RMSE heatmaps
    # 'YlOrRd':  yellow→orange→red (low→high error, intuitive)
    # 'viridis': blue→green→yellow (perceptually uniform)
    # 'RdYlGn_r': red→yellow→green reversed (red=bad, green=good)

    heatmap_improvement_colormap: str = 'RdYlGn'
    # Colormap for improvement heatmap
    # Green = large improvement, Red = small improvement

    heatmap_annotate: bool = True
    # Show numerical values in heatmap cells

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def method_color(self, method: str) -> str:
        """
        Get color for a method by name.

        Parameters
        ----------
        method : str
            Method name: 'two_ray', 'multi_ray', 'kalman', 'iterative', 'iterative_k1' through 'iterative_k5'

        Returns
        -------
        str
            Hex color string

        Examples
        --------
        >>> config = PlotConfig()
        >>> color = config.method_color('two_ray')
        >>> print(color)
        '#2196F3'
        """
        colors = {
            'two_ray':        self.color_two_ray,
            'multi_ray':      self.color_multi_ray,
            'kalman':         self.color_kalman,
            'iterative':      self.color_iterative,
            'iterative_k1':   self.color_iterative_k1,
            'iterative_k2':   self.color_iterative_k2,
            'iterative_k3':   self.color_iterative_k3,
            'iterative_k4':   self.color_iterative_k4,
            'iterative_k5':   self.color_iterative_k5,
        }
        if method not in colors:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available: {list(colors.keys())}"
            )
        return colors[method]

    def method_label(self, method: str) -> str:
        """
        Get human-readable label for a method.

        Parameters
        ----------
        method : str
            Method name

        Returns
        -------
        str
            Display label for legends and titles

        Examples
        --------
        >>> config = PlotConfig()
        >>> label = config.method_label('two_ray')
        >>> print(label)
        'Two-Ray Triangulation'
        """
        labels = {
            'two_ray':        'Two-Ray Triangulation',
            'multi_ray':      'Multi-Ray Least Squares',
            'kalman':         'Kalman Filter (Const. Velocity)',
            'iterative':      'Iterative Velocity Triangulation',
            'iterative_k1':   'Iterative Vel. Triang. (k=1)',
            'iterative_k2':   'Iterative Vel. Triang. (k=2)',
            'iterative_k3':   'Iterative Vel. Triang. (k=3)',
            'iterative_k4':   'Iterative Vel. Triang. (k=4)',
            'iterative_k5':   'Iterative Vel. Triang. (k=5)',
        }
        if method not in labels:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available: {list(labels.keys())}"
            )
        return labels[method]

    def save_filename(self, plot_name: str) -> str:
        """
        Generate save filename with correct extension.

        Parameters
        ----------
        plot_name : str
            Base name without extension

        Returns
        -------
        str
            Filename with extension

        Examples
        --------
        >>> config = PlotConfig(file_format='pdf')
        >>> filename = config.save_filename('error_histogram')
        >>> print(filename)
        'error_histogram.pdf'
        """
        return f"{plot_name}.{self.file_format}"


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def thesis_config() -> PlotConfig:
    """
    High-quality configuration for thesis figures.

    Returns
    -------
    PlotConfig
        300 DPI, PDF format, paper style
    """
    return PlotConfig(
        dpi=300,
        file_format='pdf',
        style='seaborn-v0_8-paper',
        fontsize_title=14,
        fontsize_labels=12,
        fontsize_ticks=10,
    )


def presentation_config() -> PlotConfig:
    """
    Large-font configuration for presentations.

    Returns
    -------
    PlotConfig
        Bigger figures, larger fonts, PNG format
    """
    return PlotConfig(
        dpi=150,
        file_format='png',
        style='seaborn-v0_8-talk',
        figsize_single=(14, 8),
        figsize_wide=(18, 8),
        figsize_square=(10, 10),
        fontsize_title=20,
        fontsize_labels=16,
        fontsize_ticks=14,
        fontsize_legend=11,
        line_width=2.5,
        marker_size=6,
    )


def preview_config() -> PlotConfig:
    """
    Fast low-resolution configuration for quick preview.

    Returns
    -------
    PlotConfig
        72 DPI, PNG format, fast to save
    """
    return PlotConfig(
        dpi=72,
        file_format='png',
        style='seaborn-v0_8-whitegrid',
    )
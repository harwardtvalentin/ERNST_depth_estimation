"""
Depth estimation from camera observations.

This module infers 3D depth information from 2D pixel measurements
using various triangulation and filtering techniques.

The module provides three estimation methods of increasing sophistication:

1. Two-Ray Triangulation (Baseline)
   - Uses exactly 2 observations
   - Simple and fast
   - Sensitive to noise
   - Good baseline for comparison

2. Multi-Ray Least Squares (Improved)
   - Uses N observations (N ≥ 2)
   - More robust (averages noise)
   - Better accuracy
   - Recommended for production

3. Kalman Filter Tracking (Advanced)
   - Uses all observations sequentially
   - Models target motion
   - Smoothest estimates
   - Can predict future positions
   - Best for continuous tracking

Classes
-------
TwoRayTriangulationEstimator
    Simple baseline: depth from pairs of observations
MultiRayLeastSquaresEstimator
    Improved: depth from multiple observations with least-squares
KalmanDepthTracker
    Advanced: smooth tracking with Kalman filter and motion model

Examples
--------
>>> from missile_fly_by_simulation.estimation import TwoRayTriangulationEstimator
>>> from missile_fly_by_simulation.sensing import PinholeCameraModel
>>> 
>>> # Create estimator
>>> camera = PinholeCameraModel(camera_spec)
>>> estimator = TwoRayTriangulationEstimator(camera)
>>> 
>>> # Estimate depth
>>> estimate = estimator.estimate_depth(obs1, obs2)
>>> print(f"Depth: {estimate.estimated_depth:.1f} m")
>>> print(f"Error: {estimate.error:.1f} m")
"""

from missile_fly_by_simulation.estimation.two_ray_triangulation import (
    TwoRayTriangulationEstimator,
)

from missile_fly_by_simulation.estimation.multi_ray_least_squares import (
    MultiRayLeastSquaresEstimator,
)

from missile_fly_by_simulation.estimation.kalman_constant_velocity import (
    KalmanDepthTracker,
)
from missile_fly_by_simulation.estimation.iterative_velocity_triangulation import (
    IterativeVelocityTriangulator,
)

__all__ = [
    # Baseline method
    'TwoRayTriangulationEstimator',
    
    # Improved method
    'MultiRayLeastSquaresEstimator',
    
    # Advanced method
    'KalmanDepthTracker',

    # Novel iterative method
    'IterativeVelocityTriangulator',
]
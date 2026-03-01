"""
Parameter study management for satellite-missile simulation.

This module orchestrates running many simulations systematically across a
DSA parameter grid (Distance × Speed × Angle / crossing angle).

Classes
-------
ScenarioFactory
    Creates SimulationScenario objects from physical parameters
    (closest approach distance, missile speed, crossing angle)
BatchRunner
    Runs all scenarios in parallel using ProcessPoolExecutor,
    manages folder structure and saves results selectively
ExperimentResults
    Stores and analyzes results across all runs.
    Keys are (distance, speed, angle) 3-tuples.
    Provides RMSE matrices for heatmap plotting (per angle slice).
RunSummary
    Summary statistics from one simulation run

Typical usage
-------------
>>> from missile_fly_by_simulation.experiments import (
...     ScenarioFactory,
...     BatchRunner,
...     ExperimentResults,
... )
>>>
>>> # Setup
>>> factory = ScenarioFactory(
...     satellite_spec=ScenarioFactory.default_satellite_spec(),
...     simulation_duration=1200.0,
... )
>>>
>>> runner = BatchRunner(
...     factory=factory,
...     output_dir='experiments/runs'
... )
>>>
>>> # Run DSA parameter study (parallel, 3D grid)
>>> results = runner.run_parameter_study_dsa(
...     distances=[100e3, 200e3, 500e3],
...     speeds=[500, 1000, 3000],
...     crossing_angles=[0.0, 30.0, 60.0, 90.0],
... )
>>>
>>> # Analyze
>>> print(results.summary())
>>> rmse_matrix = results.get_rmse_matrix('two_ray', crossing_angle=90.0)
>>>
>>> # Load later without rerunning
>>> results = ExperimentResults.load('experiments/runs/study_001/experiment_results.pkl')
"""

from missile_fly_by_simulation.experiments.scenario_factory import ScenarioFactory
from missile_fly_by_simulation.experiments.batch_runner import BatchRunner
from missile_fly_by_simulation.experiments.experiment_results import (
    ExperimentResults,
    RunSummary,
)

__all__ = [
    'ScenarioFactory',
    'BatchRunner',
    'ExperimentResults',
    'RunSummary',
]

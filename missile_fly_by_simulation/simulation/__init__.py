"""
Simulation orchestration and configuration.

This module provides the main entry point for running simulations:
- SimulationScenario: Input configuration
- Simulator: Orchestrator that runs the simulation
- SimulationResults: Output package with all results
- Observation, DepthEstimate: Data structures for results
"""

from missile_fly_by_simulation.simulation.scenario import (
    SimulationScenario,
    create_default_scenario,
)
from missile_fly_by_simulation.simulation.simulator import Simulator
from missile_fly_by_simulation.simulation.results import (
    SimulationResults,
    Observation,
    DepthEstimate,
)

__all__ = [
    # Configuration
    'SimulationScenario',
    'create_default_scenario',
    
    # Orchestrator
    'Simulator',
    
    # Results
    'SimulationResults',
    'Observation',
    'DepthEstimate',
]
"""
Simulation orchestrator.

This module coordinates all components to execute a complete simulation.
It's the "conductor" that makes all the modules work together.

Classes
-------
Simulator
    Main simulation orchestrator
"""

from typing import List, Dict
from datetime import datetime, timedelta
import numpy as np
import time

# Domain imports (clean, no submodule specification needed!)
from missile_fly_by_simulation.domain import (
    Satellite,
    SatelliteState,
    Attitude,
    Missile,
    MissileState,
)

# Physics imports
from missile_fly_by_simulation.physics import (
    KeplerianOrbitPropagator,
    NadirPointingController,
)

# Sensing imports
from missile_fly_by_simulation.sensing import (
    PinholeCameraModel,
    check_line_of_sight,
)

# Same package - import directly from siblings to avoid circular imports! ✅  
from missile_fly_by_simulation.simulation.scenario import SimulationScenario
from missile_fly_by_simulation.simulation.results import (
    SimulationResults,
    Observation,
    DepthEstimate,
)

# Constants - single source of truth for all constants used across the package.
from missile_fly_by_simulation.constants import (
    DEFAULT_NADIR_LOOK_ANGLE_DEG,
    DEFAULT_MULTI_RAY_OBSERVATIONS,
    DEFAULT_MULTI_RAY_TIME_WINDOWS_S,
    DEFAULT_FPS,
)

class Simulator:
    """
    Simulation orchestrator - coordinates all modules.
    
    This class is the "conductor" of the simulation. It doesn't
    implement algorithms itself - instead, it coordinates the
    physics, sensing, and estimation modules to produce results.
    
    The simulation pipeline:
    1. Generate satellite orbit (physics/orbital_mechanics)
    2. Generate missile trajectory (simple kinematics)
    3. Compute satellite attitudes (physics/attitude_dynamics)
    4. Generate camera observations (sensing/camera_model)
    5. Estimate depths (basic triangulation)
    6. Package results (simulation/results)
    
    Attributes
    ----------
    scenario : SimulationScenario
        Input configuration
    satellite : Satellite
        Satellite entity (accumulates states)
    missile : Missile
        Missile entity (accumulates states)
    observations : list of Observation
        Camera detections
    depth_estimates : dict
        Depth estimates from different methods
        
    Examples
    --------
    >>> scenario = SimulationScenario(...)
    >>> simulator = Simulator(scenario)
    >>> results = simulator.run(show_progress=True)
    >>> print(results.summary())
    """
    
    def __init__(self, scenario: SimulationScenario):
        """
        Initialize simulator with configuration.
        
        Parameters
        ----------
        scenario : SimulationScenario
            Complete simulation configuration
        """
        self.scenario = scenario
        
        # Create entities (containers for data)
        self.satellite = Satellite(scenario.satellite_spec)
        self.missile = Missile(name=scenario.missile_name)
        
        # Create computation engines
        self.orbit_propagator = KeplerianOrbitPropagator(
            scenario.satellite_spec.orbital_elements
        )
        self.attitude_controller = NadirPointingController()
        self.camera = PinholeCameraModel(scenario.satellite_spec.camera)
        
        # Storage for intermediate results
        self.observations: List[Observation] = []
        self.depth_estimates: Dict[str, List[DepthEstimate]] = {}
        
        # Timing
        self._start_time = None
        self._step_times = {}
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run(self, show_progress: bool = True) -> SimulationResults:
        """
        Execute complete simulation pipeline.
        
        This is the main entry point - runs all simulation steps
        and returns packaged results.
        
        Parameters
        ----------
        show_progress : bool, optional
            If True, print progress messages, default True
            
        Returns
        -------
        SimulationResults
            Complete simulation output
            
        Examples
        --------
        >>> simulator = Simulator(scenario)
        >>> results = simulator.run(show_progress=True)
        [1/6] Generating satellite orbit...
        [2/6] Generating missile trajectory...
        [3/6] Computing satellite attitudes...
        [4/6] Generating camera observations...
        [5/6] Estimating depths...
        [6/6] Packaging results...
        Done! Simulation took 12.3 seconds
        """
        self._start_time = time.time()
        
        if show_progress:
            print("\n" + "="*60)
            print("Starting Simulation")
            print("="*60)
            print(f"Satellite: {self.scenario.satellite_spec.name}")
            print(f"Duration: {self.scenario.duration:.1f}s "
                  f"({self.scenario.num_frames} frames @ {self.scenario.fps} fps)")
            print("="*60 + "\n")
        
        # Step 1: Generate satellite orbit
        self._run_step(
            step_num=1,
            step_name="Generating satellite orbit",
            step_func=self._propagate_satellite_orbit,
            show_progress=show_progress
        )
        
        # Step 2: Generate missile trajectory
        self._run_step(
            step_num=2,
            step_name="Generating missile trajectory",
            step_func=self._propagate_missile_trajectory,
            show_progress=show_progress
        )
        
        # Step 3: Compute satellite attitudes
        self._run_step(
            step_num=3,
            step_name="Computing satellite attitudes",
            step_func=self._compute_satellite_attitudes,
            show_progress=show_progress
        )
        
        # Step 4: Generate camera observations
        self._run_step(
            step_num=4,
            step_name="Generating camera observations",
            step_func=self._generate_observations,
            show_progress=show_progress
        )
        
        # Step 5: Estimate depths
        self._run_step(
            step_num=5,
            step_name="Estimating depths",
            step_func=self._estimate_depths,
            show_progress=show_progress
        )
        
        # Step 6: Package results
        self._run_step(
            step_num=6,
            step_name="Packaging results",
            step_func=lambda: None,  # No-op, just for progress
            show_progress=show_progress
        )
        
        results = self._package_results()
        
        # Print summary
        elapsed = time.time() - self._start_time
        if show_progress:
            print("\n" + "="*60)
            print(f"[OK] Simulation Complete ({elapsed:.1f} seconds)")
            print("="*60)
            print(f"Satellite states: {self.satellite.num_states}")
            print(f"Missile states: {self.missile.num_states}")
            print(f"Observations: {len(self.observations)}")
            print(f"Depth estimates: {sum(len(v) for v in self.depth_estimates.values())}")
            print("="*60 + "\n")
        
        return results
    
    # =========================================================================
    # SIMULATION STEPS
    # =========================================================================
    
    def _propagate_satellite_orbit(self):
        """
        Step 1: Generate satellite orbital trajectory.
        
        Uses physics/orbital_mechanics to compute satellite position
        and velocity at each timestep.
        """
        timestamps = self.scenario.timestamps
        
        # Use orbit propagator (physics module)
        states = self.orbit_propagator.propagate(
            timestamps=timestamps,
            reference_time=self.scenario.start_time,
            show_progress=False  # We handle progress at higher level
        )
        
        # Store in satellite entity
        self.satellite.add_states(states)
    
    def _propagate_missile_trajectory(self):
        """
        Step 2: Generate missile ballistic trajectory.
        
        Simple constant-velocity kinematics:
        position(t) = position_0 + velocity * t
        """
        # Only generate trajectory after detection
        detection_idx = self.scenario.detection_frame_index
        
        states = []
        for i in range(detection_idx, self.scenario.num_frames):
            # Time since detection
            timestamp = self.scenario.timestamp_at_index(i)
            t = (timestamp - self.scenario.missile_detection_time).total_seconds()
            
            # Constant velocity trajectory
            position = self.scenario.missile_initial_position + self.scenario.missile_velocity * t
            
            # Create state
            state = MissileState(
                timestamp=timestamp,
                position=position,
                velocity=self.scenario.missile_velocity
            )
            states.append(state)
        
        # Store in missile entity
        self.missile.add_states(states)
    
    def _compute_satellite_attitudes(self):
        """
        Step 3: Compute satellite orientation at each timestep.
        
        Uses physics/attitude_dynamics to compute how satellite
        points its camera (toward missile if visible, nadir otherwise).
        """
        detection_idx = self.scenario.detection_frame_index
        updated_states = []
        
        for i, sat_state in enumerate(self.satellite.states):
            # Check if missile is active at this time
            if i >= detection_idx:
                # Try to point at missile
                missile_idx = i - detection_idx
                
                if missile_idx < len(self.missile.states):
                    miss_state = self.missile.states[missile_idx]
                    
                    if check_line_of_sight(sat_state.position, miss_state.position):
                        try:
                            attitude = self.attitude_controller.compute_attitude(
                                satellite_position=sat_state.position,
                                target_position=miss_state.position,
                                satellite_velocity=sat_state.velocity
                            )
                        except ValueError:
                            attitude = self.attitude_controller.compute_nadir_pointing_attitude(
                                sat_state.position,
                                sat_state.velocity,
                                look_angle_deg=DEFAULT_NADIR_LOOK_ANGLE_DEG
                            )
                    else:
                        # Line of sight blocked (Earth occlusion or invalid position)
                        attitude = self.attitude_controller.compute_nadir_pointing_attitude(
                            sat_state.position,
                            sat_state.velocity,
                            look_angle_deg=DEFAULT_NADIR_LOOK_ANGLE_DEG
                        )
                else:
                    # Missile not yet available
                    attitude = self.attitude_controller.compute_nadir_pointing_attitude(
                        satellite_position=sat_state.position,
                        satellite_velocity=sat_state.velocity,
                        look_angle_deg=DEFAULT_NADIR_LOOK_ANGLE_DEG
                    )
            else:
                # Before detection: nadir pointing
                attitude = self.attitude_controller.compute_nadir_pointing_attitude(
                    satellite_position=sat_state.position,
                    satellite_velocity=sat_state.velocity,
                    look_angle_deg=DEFAULT_NADIR_LOOK_ANGLE_DEG
                )
            
            # Create new state with attitude
            updated_state = sat_state.with_attitude(attitude)
            updated_states.append(updated_state)
        
        # Replace states with updated versions
        self.satellite.states = updated_states
    
    def _generate_observations(self):
        """
        Step 4: Generate camera observations of missile.
        
        Uses sensing/camera_model to project missile 3D position
        to 2D pixel scoordinates.
        """
        detection_idx = self.scenario.detection_frame_index
        
        observations = []
        
        for i in range(detection_idx, len(self.satellite.states)):
            sat_state = self.satellite.states[i]
            
            # Get corresponding missile state
            missile_idx = i - detection_idx
            if missile_idx >= len(self.missile.states):
                continue
            
            miss_state = self.missile.states[missile_idx]
            
            # Project to camera
            pixel = self.camera.project_to_image(
                point_world=miss_state.position,
                camera_state=sat_state
            )

            # Add centroid-localisation noise (Gaussian, per-frame, independent)
            if pixel is not None and self.scenario.pixel_noise_sigma > 0.0:
                u_n = pixel[0] + np.random.normal(0.0, self.scenario.pixel_noise_sigma)
                v_n = pixel[1] + np.random.normal(0.0, self.scenario.pixel_noise_sigma)
                if 0.0 <= u_n < self.camera.width and 0.0 <= v_n < self.camera.height:
                    pixel = (u_n, v_n)
                else:
                    pixel = None  # noise pushed pixel outside FOV

            if pixel is not None:
                # Compute ground truth depth
                true_depth = np.linalg.norm(
                    miss_state.position - sat_state.position
                )
                
                # Create observation
                obs = Observation(
                    timestamp=sat_state.timestamp,
                    satellite_state=sat_state,
                    pixel=pixel,
                    true_position=miss_state.position,
                    true_depth=true_depth
                )
                observations.append(obs)
        
        self.observations = observations

    def _estimate_depths(self):
        """
        Step 5: Estimate depths using all estimation methods.
        Delegates to the estimation/ module.

        The iterative method is run 5 times with increasing iteration
        caps (k=1 through k=5) so that depth_comparison plots can show
        convergence behaviour across iterations.

        Note: k=1 (max_iterations=1) is mathematically identical to
        standard two-ray triangulation with short_window_s time offset,
        providing a built-in sanity check — iterative_k1 should closely
        match two_ray.
        """
        from missile_fly_by_simulation.estimation import (
            TwoRayTriangulationEstimator,
            MultiRayLeastSquaresEstimator,
            KalmanDepthTracker,
            IterativeVelocityTriangulator,
        )

        two_ray   = TwoRayTriangulationEstimator(self.camera)
        multi_ray = MultiRayLeastSquaresEstimator(self.camera)
        kalman    = KalmanDepthTracker(self.camera, fps=DEFAULT_FPS)
        iterative = IterativeVelocityTriangulator(self.camera, fps=DEFAULT_FPS)

        self.depth_estimates = {
            'two_ray':   two_ray.estimate_batch(
                            self.observations,
                            self.scenario.depth_time_offsets
                        ),
            'multi_ray': multi_ray.estimate_batch(
                            self.observations,
                            time_windows=DEFAULT_MULTI_RAY_TIME_WINDOWS_S,
                            n_observations_list=[DEFAULT_MULTI_RAY_OBSERVATIONS],
                        ),
            'kalman':    kalman.estimate_batch(self.observations),
            # Iterative method at 5 iteration levels for convergence analysis.
            # The timestamp index is built once per call — 5 calls = 5 index builds,
            # each O(n log n), so runtime overhead is small.
            'iterative_k1': iterative.estimate_batch(self.observations, max_iterations=1),
            'iterative_k2': iterative.estimate_batch(self.observations, max_iterations=2),
            'iterative_k3': iterative.estimate_batch(self.observations, max_iterations=3),
            'iterative_k4': iterative.estimate_batch(self.observations, max_iterations=4),
            'iterative_k5': iterative.estimate_batch(self.observations, max_iterations=5),
        }
        
    
    # =========================================================================
    # RESULT PACKAGING
    # =========================================================================
    
    def _estimate_depths(self):
        """
        Step 5: Estimate depths using all estimation methods.
        Delegates to the estimation/ module.

        The iterative method is run 5 times with increasing iteration
        caps (k=1 through k=5) so that depth_comparison plots can show
        convergence behaviour across iterations.

        Note: k=1 (max_iterations=1) is mathematically identical to
        standard two-ray triangulation with short_window_s time offset,
        providing a built-in sanity check — iterative_k1 should closely
        match two_ray.
        """
        from missile_fly_by_simulation.estimation import (
            TwoRayTriangulationEstimator,
            MultiRayLeastSquaresEstimator,
            KalmanDepthTracker,
            IterativeVelocityTriangulator,
        )

        two_ray   = TwoRayTriangulationEstimator(self.camera)
        multi_ray = MultiRayLeastSquaresEstimator(self.camera)
        kalman    = KalmanDepthTracker(self.camera, fps=DEFAULT_FPS)
        iterative = IterativeVelocityTriangulator(self.camera, fps=DEFAULT_FPS)

        def _run_estimator(label, func):
            """Run one estimator with inline progress reporting."""
            print(f"\n      {label}...", end='', flush=True)
            t0 = time.time()
            result = func()
            print(f" [OK] ({time.time() - t0:.1f}s, {len(result)} estimates)", flush=True)
            return result

        self.depth_estimates = {
            'two_ray': _run_estimator(
                'two_ray',
                lambda: two_ray.estimate_batch(
                    self.observations,
                    self.scenario.depth_time_offsets
                )
            ),
            'multi_ray': _run_estimator(
                'multi_ray',
                lambda: multi_ray.estimate_batch(
                    self.observations,
                    time_windows=DEFAULT_MULTI_RAY_TIME_WINDOWS_S,
                    n_observations_list=[DEFAULT_MULTI_RAY_OBSERVATIONS],
                )
            ),
            'kalman': _run_estimator(
                'kalman',
                lambda: kalman.estimate_batch(self.observations)
            ),
            'iterative_k1': _run_estimator(
                'iterative k=1',
                lambda: iterative.estimate_batch(self.observations, max_iterations=1)
            ),
            'iterative_k2': _run_estimator(
                'iterative k=2',
                lambda: iterative.estimate_batch(self.observations, max_iterations=2)
            ),
            'iterative_k3': _run_estimator(
                'iterative k=3',
                lambda: iterative.estimate_batch(self.observations, max_iterations=3)
            ),
            'iterative_k4': _run_estimator(
                'iterative k=4',
                lambda: iterative.estimate_batch(self.observations, max_iterations=4)
            ),
            'iterative_k5': _run_estimator(
                'iterative k=5',
                lambda: iterative.estimate_batch(self.observations, max_iterations=5)
            ),
        }
    
    # =========================================================================
    # RESULT PACKAGING
    # =========================================================================

    def _package_results(self) -> SimulationResults:
        """Package all simulation outputs into a SimulationResults object."""
        elapsed = time.time() - self._start_time
        return SimulationResults(
            satellite=self.satellite,
            missile=self.missile,
            observations=self.observations,
            depth_estimates=self.depth_estimates,
            scenario=self.scenario,
            metadata={
                'runtime_seconds': elapsed,
                'step_times': self._step_times,
            }
        )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _run_step(
        self,
        step_num: int,
        step_name: str,
        step_func,
        show_progress: bool
    ):
        """Execute a simulation step with timing and progress reporting."""
        if show_progress:
            print(f"[{step_num}/6] {step_name}...", end='', flush=True)
        
        start = time.time()
        step_func()
        elapsed = time.time() - start
        
        self._step_times[step_name] = elapsed
        
        if show_progress:
            print(f" [OK] ({elapsed:.1f}s)")
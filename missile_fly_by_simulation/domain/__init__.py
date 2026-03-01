# domain/__init__.py

"""Domain objects."""

# Re-export commonly used classes
from missile_fly_by_simulation.domain.satellite import (
    Satellite,
    SatelliteState,
    SatelliteSpecification,
    OrbitalElements,
    Attitude,
    CameraSpecification
)

from missile_fly_by_simulation.domain.missile import (
    Missile,
    MissileState
)

# Now define what "from domain import *" gives you
__all__ = [
    'Satellite',
    'SatelliteState',
    'SatelliteSpecification',
    'OrbitalElements',
    'Attitude',
    'CameraSpecification',
    'Missile',
    'MissileState'
]
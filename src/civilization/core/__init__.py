"""Core domain exports (config, enums, entities)."""

from .config import (
    CFG,
    SimConfig,
    Sex,
    Profession,
    Governance,
    TerrainType,
    GoalType,
    DisasterType,
)
from .entities import (
    Genome,
    Village,
    Organism,
    Warrior,
    Enemy,
    SpeciesTracker,
    DisasterManager,
    TerrainMap,
)

__all__ = [
    "CFG",
    "SimConfig",
    "Sex",
    "Profession",
    "Governance",
    "TerrainType",
    "GoalType",
    "DisasterType",
    "Genome",
    "Village",
    "Organism",
    "Warrior",
    "Enemy",
    "SpeciesTracker",
    "DisasterManager",
    "TerrainMap",
]


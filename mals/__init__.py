"""
MALS — Multi-Agent Living System

A blackboard-architecture-based multi-agent collaboration framework that replaces
fragile direct communication with a shared, persistent, self-organizing cognitive space.
"""

__version__ = "0.1.0"

from mals.core.blackboard import Blackboard
from mals.core.conductor import Conductor
from mals.core.engine import MALSEngine
from mals.agents.registry import AgentRegistry, specialist

__all__ = [
    "Blackboard",
    "Conductor",
    "MALSEngine",
    "AgentRegistry",
    "specialist",
    "__version__",
]

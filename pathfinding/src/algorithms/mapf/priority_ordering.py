from __future__ import annotations

import random

from pathfinding.src.algorithms.mapf.models import MAPFScenario


def random_priority_order(scenario: MAPFScenario, seed: int) -> MAPFScenario:
    agents = list(scenario.agents)
    rng = random.Random(seed)
    rng.shuffle(agents)
    return MAPFScenario(agents=tuple(agents))

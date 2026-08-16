from __future__ import annotations

from dataclasses import dataclass

from pathfinding.src.algorithms.mapf.cbs import CBSNode
from pathfinding.src.algorithms.mapf.cbs_conflict_classification import (
    ConflictCardinality,
    classify_conflict,
)
from pathfinding.src.algorithms.mapf.models import Conflict, MAPFScenario
from pathfinding.src.core.models import GridMap


@dataclass(frozen=True, slots=True)
class CBSConflictSelection:
    conflict: Conflict
    cardinality: ConflictCardinality
    classified_conflicts: int
    classification_low_level_searches: int


def select_conflict_cardinal_first(
    grid_map: GridMap,
    scenario: MAPFScenario,
    node: CBSNode,
    max_timestep: int,
) -> CBSConflictSelection | None:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    if not node.conflicts:
        return None

    classified_conflicts = 0
    first_semi_cardinal_conflict: Conflict | None = None
    first_semi_cardinal_cardinality: ConflictCardinality | None = None

    for conflict in node.conflicts:
        cardinality = classify_conflict(
            grid_map=grid_map,
            scenario=scenario,
            node=node,
            conflict=conflict,
            max_timestep=max_timestep,
        )
        classified_conflicts += 1

        if cardinality == ConflictCardinality.CARDINAL:
            return CBSConflictSelection(
                conflict=conflict,
                cardinality=cardinality,
                classified_conflicts=classified_conflicts,
                classification_low_level_searches=2 * classified_conflicts,
            )

        if (
            cardinality == ConflictCardinality.SEMI_CARDINAL
            and first_semi_cardinal_conflict is None
        ):
            first_semi_cardinal_conflict = conflict
            first_semi_cardinal_cardinality = cardinality

    if first_semi_cardinal_conflict is not None:
        assert first_semi_cardinal_cardinality is not None
        return CBSConflictSelection(
            conflict=first_semi_cardinal_conflict,
            cardinality=first_semi_cardinal_cardinality,
            classified_conflicts=classified_conflicts,
            classification_low_level_searches=2 * classified_conflicts,
        )

    return CBSConflictSelection(
        conflict=node.conflicts[0],
        cardinality=ConflictCardinality.NON_CARDINAL,
        classified_conflicts=classified_conflicts,
        classification_low_level_searches=2 * classified_conflicts,
    )

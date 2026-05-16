from typing import NamedTuple

GridNode = tuple[int, int]


class RawAlgorithmStep(NamedTuple):
    current: GridNode | None
    open_nodes: frozenset[GridNode]
    closed_nodes: frozenset[GridNode]
    path: tuple[GridNode, ...] = ()
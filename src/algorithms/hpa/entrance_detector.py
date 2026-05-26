from src.core.hpa_models import Cluster, Entrance
from src.core.models import GridMap, Position


class EntranceDetector:
    def __init__(
        self,
        max_entrances_per_cluster_pair: int = 2,
        min_entrance_width: int = 1,
    ) -> None:
        self.max_entrances_per_cluster_pair = max_entrances_per_cluster_pair
        self.min_entrance_width = min_entrance_width

    def detect_entrances(
        self,
        grid_map: GridMap,
        clusters: list[Cluster],
        cluster_lookup: dict[tuple[int, int], Cluster],
    ) -> list[Entrance]:
        entrances: list[Entrance] = []

        for cluster in clusters:
            entrances.extend(
                self._detect_horizontal_neighbor_entrance_groups(
                    grid_map=grid_map,
                    cluster=cluster,
                    cluster_by_grid_position=cluster_lookup,
                )
            )

            entrances.extend(
                self._detect_vertical_neighbor_entrance_groups(
                    grid_map=grid_map,
                    cluster=cluster,
                    cluster_by_grid_position=cluster_lookup,
                )
            )

        return self._prune_entrances(entrances)

    def _detect_horizontal_neighbor_entrance_groups(
        self,
        grid_map: GridMap,
        cluster: Cluster,
        cluster_by_grid_position: dict[tuple[int, int], Cluster],
    ) -> list[Entrance]:
        entrances: list[Entrance] = []

        right_col = cluster.col_end
        left_col = cluster.col_end - 1

        if right_col >= grid_map.width:
            return entrances

        current_group: list[tuple[int, Cluster]] = []

        for row in range(cluster.row_start, cluster.row_end):
            neighbor_cluster = cluster_by_grid_position.get((row, right_col))

            if neighbor_cluster is None:
                self._add_horizontal_group_entrance(
                    entrances=entrances,
                    group=current_group,
                    cluster=cluster,
                    left_col=left_col,
                    right_col=right_col,
                )
                current_group = []
                continue

            position_a = Position(row=row, col=left_col)
            position_b = Position(row=row, col=right_col)

            if grid_map.is_walkable(position_a) and grid_map.is_walkable(position_b):
                current_group.append((row, neighbor_cluster))
            else:
                self._add_horizontal_group_entrance(
                    entrances=entrances,
                    group=current_group,
                    cluster=cluster,
                    left_col=left_col,
                    right_col=right_col,
                )
                current_group = []

        self._add_horizontal_group_entrance(
            entrances=entrances,
            group=current_group,
            cluster=cluster,
            left_col=left_col,
            right_col=right_col,
        )

        return entrances

    def _detect_vertical_neighbor_entrance_groups(
        self,
        grid_map: GridMap,
        cluster: Cluster,
        cluster_by_grid_position: dict[tuple[int, int], Cluster],
    ) -> list[Entrance]:
        entrances: list[Entrance] = []

        bottom_row = cluster.row_end
        top_row = cluster.row_end - 1

        if bottom_row >= grid_map.height:
            return entrances

        current_group: list[tuple[int, Cluster]] = []

        for col in range(cluster.col_start, cluster.col_end):
            neighbor_cluster = cluster_by_grid_position.get((bottom_row, col))

            if neighbor_cluster is None:
                self._add_vertical_group_entrance(
                    entrances=entrances,
                    group=current_group,
                    cluster=cluster,
                    top_row=top_row,
                    bottom_row=bottom_row,
                )
                current_group = []
                continue

            position_a = Position(row=top_row, col=col)
            position_b = Position(row=bottom_row, col=col)

            if grid_map.is_walkable(position_a) and grid_map.is_walkable(position_b):
                current_group.append((col, neighbor_cluster))
            else:
                self._add_vertical_group_entrance(
                    entrances=entrances,
                    group=current_group,
                    cluster=cluster,
                    top_row=top_row,
                    bottom_row=bottom_row,
                )
                current_group = []

        self._add_vertical_group_entrance(
            entrances=entrances,
            group=current_group,
            cluster=cluster,
            top_row=top_row,
            bottom_row=bottom_row,
        )

        return entrances

    def _prune_entrances(
        self,
        entrances: list[Entrance],
    ) -> list[Entrance]:
        grouped: dict[tuple[int, int], list[Entrance]] = {}

        for candidate in entrances:
            if candidate.width < self.min_entrance_width:
                continue

            cluster_ids = (
                candidate.cluster_a_id,
                candidate.cluster_b_id,
            )

            key: tuple[int, int] = (
                min(cluster_ids),
                max(cluster_ids),
            )

            grouped.setdefault(key, []).append(candidate)

        pruned: list[Entrance] = []

        for entrance_group in grouped.values():
            sorted_group = sorted(
                entrance_group,
                key=lambda item: item.width,
                reverse=True,
            )

            pruned.extend(
                sorted_group[: self.max_entrances_per_cluster_pair]
            )

        return pruned

    @staticmethod
    def _add_horizontal_group_entrance(
        entrances: list[Entrance],
        group: list[tuple[int, Cluster]],
        cluster: Cluster,
        left_col: int,
        right_col: int,
    ) -> None:
        if not group:
            return

        middle_index = len(group) // 2
        row, neighbor_cluster = group[middle_index]

        entrances.append(
            Entrance(
                cluster_a_id=cluster.id,
                cluster_b_id=neighbor_cluster.id,
                position_a=Position(row=row, col=left_col),
                position_b=Position(row=row, col=right_col),
                width=len(group),
            )
        )

    @staticmethod
    def _add_vertical_group_entrance(
        entrances: list[Entrance],
        group: list[tuple[int, Cluster]],
        cluster: Cluster,
        top_row: int,
        bottom_row: int,
    ) -> None:
        if not group:
            return

        middle_index = len(group) // 2
        col, neighbor_cluster = group[middle_index]

        entrances.append(
            Entrance(
                cluster_a_id=cluster.id,
                cluster_b_id=neighbor_cluster.id,
                position_a=Position(row=top_row, col=col),
                position_b=Position(row=bottom_row, col=col),
                width=len(group),
            )
        )
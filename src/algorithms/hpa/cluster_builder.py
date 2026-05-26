from src.core.hpa_models import Cluster
from src.core.models import GridMap


def build_clusters(
    grid_map: GridMap,
    cluster_size: int,
) -> list[Cluster]:
    clusters: list[Cluster] = []
    cluster_id = 0

    for row_start in range(0, grid_map.height, cluster_size):
        for col_start in range(0, grid_map.width, cluster_size):
            clusters.append(
                Cluster(
                    id=cluster_id,
                    row_start=row_start,
                    row_end=min(row_start + cluster_size, grid_map.height),
                    col_start=col_start,
                    col_end=min(col_start + cluster_size, grid_map.width),
                )
            )
            cluster_id += 1

    return clusters


def build_cluster_lookup(
    clusters: list[Cluster],
) -> dict[tuple[int, int], Cluster]:
    lookup: dict[tuple[int, int], Cluster] = {}

    for cluster in clusters:
        for row in range(cluster.row_start, cluster.row_end):
            for col in range(cluster.col_start, cluster.col_end):
                lookup[(row, col)] = cluster

    return lookup


def build_allowed_positions_for_cluster(
    cluster: Cluster,
) -> set[tuple[int, int]]:
    return {
        (row, col)
        for row in range(cluster.row_start, cluster.row_end)
        for col in range(cluster.col_start, cluster.col_end)
    }


def build_allowed_positions_by_cluster(
    clusters: list[Cluster],
) -> dict[int, set[tuple[int, int]]]:
    return {
        cluster.id: build_allowed_positions_for_cluster(cluster)
        for cluster in clusters
    }
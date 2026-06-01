from code.src.algorithms.hpa.cluster_builder import build_allowed_positions_for_cluster
from code.src.algorithms.hpa.local_path_cache import LocalPathCache
from code.src.core.hpa_models import AbstractEdge, AbstractGraph, AbstractNode, Cluster, Entrance
from code.src.core.models import GridMap


class AbstractGraphBuilder:
    def __init__(
            self,
            local_path_cache: LocalPathCache,
    ) -> None:
        self.local_path_cache = local_path_cache

    def build_abstract_graph(
            self,
            grid_map: GridMap,
            entrances: list[Entrance],
            clusters: list[Cluster],
            allowed_positions_by_cluster: dict[int, set[tuple[int, int]]],
    ) -> AbstractGraph:
        nodes: list[AbstractNode] = []
        edges: list[AbstractEdge] = []

        node_id = 0

        for entrance in entrances:
            node_a = AbstractNode(
                id=node_id,
                cluster_id=entrance.cluster_a_id,
                position=entrance.position_a,
            )
            node_id += 1

            node_b = AbstractNode(
                id=node_id,
                cluster_id=entrance.cluster_b_id,
                position=entrance.position_b,
            )
            node_id += 1

            nodes.append(node_a)
            nodes.append(node_b)

            edges.append(
                AbstractEdge(
                    from_node_id=node_a.id,
                    to_node_id=node_b.id,
                    cost=1,
                    path=[node_a.position, node_b.position],
                )
            )
            edges.append(
                AbstractEdge(
                    from_node_id=node_b.id,
                    to_node_id=node_a.id,
                    cost=1,
                    path=[node_b.position, node_a.position],
                )
            )

        edges.extend(
            self._build_intra_cluster_edges(
                grid_map=grid_map,
                clusters=clusters,
                nodes=nodes,
                allowed_positions_by_cluster=allowed_positions_by_cluster,
            )
        )

        return AbstractGraph(nodes=nodes, edges=edges)

    def connect_node_to_cluster_nodes(
            self,
            grid_map: GridMap,
            node: AbstractNode,
            cluster: Cluster,
            nodes: list[AbstractNode],
            edges: list[AbstractEdge],
            allowed_positions_by_cluster: dict[int, set[tuple[int, int]]],
    ) -> None:
        allowed_positions = allowed_positions_by_cluster.get(
            cluster.id,
            build_allowed_positions_for_cluster(cluster),
        )

        cluster_nodes = [
            other_node for other_node in nodes
            if other_node.cluster_id == cluster.id
               and other_node.id != node.id
        ]

        for other_node in cluster_nodes:
            path = self.local_path_cache.find_path(
                grid_map=grid_map,
                start=node.position,
                goal=other_node.position,
                cluster_id=cluster.id,
                allowed_positions=allowed_positions,
            )

            if not path:
                continue

            self._add_bidirectional_edge(
                edges=edges,
                from_node=node,
                to_node=other_node,
                path=path,
            )

    def _build_intra_cluster_edges(
            self,
            grid_map: GridMap,
            clusters: list[Cluster],
            nodes: list[AbstractNode],
            allowed_positions_by_cluster: dict[int, set[tuple[int, int]]],
    ) -> list[AbstractEdge]:
        edges: list[AbstractEdge] = []
        nodes_by_cluster: dict[int, list[AbstractNode]] = {}

        for node in nodes:
            nodes_by_cluster.setdefault(node.cluster_id, []).append(node)

        cluster_by_id = {
            cluster.id: cluster
            for cluster in clusters
        }

        for cluster_id, cluster_nodes in nodes_by_cluster.items():
            cluster = cluster_by_id[cluster_id]

            allowed_positions = allowed_positions_by_cluster.get(
                cluster.id,
                build_allowed_positions_for_cluster(cluster),
            )

            for i in range(len(cluster_nodes)):
                for j in range(i + 1, len(cluster_nodes)):
                    node_a = cluster_nodes[i]
                    node_b = cluster_nodes[j]

                    path = self.local_path_cache.find_path(
                        grid_map=grid_map,
                        start=node_a.position,
                        goal=node_b.position,
                        cluster_id=cluster.id,
                        allowed_positions=allowed_positions,
                    )

                    if not path:
                        continue

                    self._add_bidirectional_edge(
                        edges=edges,
                        from_node=node_a,
                        to_node=node_b,
                        path=path,
                    )

        return edges

    @staticmethod
    def _add_bidirectional_edge(
            edges: list[AbstractEdge],
            from_node: AbstractNode,
            to_node: AbstractNode,
            path: list,
    ) -> None:
        cost = float(len(path) - 1)

        edges.append(
            AbstractEdge(
                from_node_id=from_node.id,
                to_node_id=to_node.id,
                cost=cost,
                path=path,
            )
        )

        edges.append(
            AbstractEdge(
                from_node_id=to_node.id,
                to_node_id=from_node.id,
                cost=cost,
                path=list(reversed(path)),
            )
        )

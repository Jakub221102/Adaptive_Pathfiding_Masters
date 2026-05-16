from src.algorithms.hpa_star import HPAStar
from src.core.models import GridMap


class HPADebugExperiment:
    def __init__(
        self,
        grid_map: GridMap,
        cluster_size: int = 32,
    ) -> None:
        self.grid_map = grid_map
        self.cluster_size = cluster_size

    def run(self) -> None:
        hpa = HPAStar(cluster_size=self.cluster_size)

        clusters = hpa.build_clusters(self.grid_map)
        entrances = hpa.detect_entrances(self.grid_map, clusters)

        graph = hpa.build_abstract_graph(
            grid_map=self.grid_map,
            entrances=entrances,
            clusters=clusters,
        )

        print(f"Clusters: {len(clusters)}")
        print(f"Entrances: {len(entrances)}")
        print(f"Abstract nodes: {len(graph.nodes)}")
        print(f"Abstract edges: {len(graph.edges)}")

        if len(graph.nodes) < 2:
            print("Not enough abstract nodes for pathfinding.")
            return

        abstract_path = hpa.find_abstract_path(
            graph=graph,
            start_node_id=graph.nodes[0].id,
            goal_node_id=graph.nodes[-1].id,
        )

        if not abstract_path:
            print("No abstract path found.")
            return

        print(f"Abstract path length: {len(abstract_path)}")

        print("Abstract node IDs:")
        print([node.id for node in abstract_path])

        print("Abstract node positions:")
        print([
            (node.position.row, node.position.col)
            for node in abstract_path
        ])
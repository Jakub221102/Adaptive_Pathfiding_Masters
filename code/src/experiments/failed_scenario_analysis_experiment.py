from code.src.algorithms.astar import AStar
from code.src.algorithms.hpa.hpa_star import HPAStar
from code.src.core.models import FailedScenarioResult
from code.src.experiments.base_experiment import BaseExperiment


class FailedScenarioAnalysisExperiment(BaseExperiment):
    def run(self) -> None:
        grid_map, scenarios = self._load_filtered_scenarios()
        selected_scenarios = scenarios[: self.config.benchmark_scenarios]

        astar = AStar()
        hpa = HPAStar(
            cluster_size=self.config.cluster_size,
            max_entrances_per_cluster_pair=2,
        )

        hpa.preprocess_map(grid_map)

        failed_results: list[FailedScenarioResult] = []

        for scenario_index, scenario in enumerate(selected_scenarios):
            astar_result = astar.find_path(
                grid_map=grid_map,
                start=scenario.start,
                goal=scenario.goal,
            )

            hpa_result = hpa.find_path(
                grid_map=grid_map,
                start=scenario.start,
                goal=scenario.goal,
            )

            if astar_result.found and not hpa_result.found:
                hpa_stats = hpa.get_last_query_stats()

                failed_results.append(
                    FailedScenarioResult(
                        scenario_index=scenario_index,
                        start=scenario.start,
                        goal=scenario.goal,
                        optimal_length=scenario.optimal_length,
                        astar_found=astar_result.found,
                        hpa_found=hpa_result.found,
                        astar_path_length=astar_result.path_length,
                        hpa_path_length=hpa_result.path_length,
                        hpa_abstract_nodes_visited=hpa_stats.abstract_nodes_visited,
                        hpa_abstract_path_edge_count=hpa_stats.abstract_path_edge_count,
                    )
                )

        self._print_failed_results(failed_results)

    @staticmethod
    def _print_failed_results(
            failed_results: list[FailedScenarioResult],
    ) -> None:
        print("\nFailed scenario analysis:")

        if not failed_results:
            print("No failed HPA* scenarios found.")
            return

        print(f"Failed scenarios: {len(failed_results)}")

        for failed in failed_results:
            print(
                f"\nscenario_index={failed.scenario_index}"
                f"\n  start=({failed.start.row}, {failed.start.col})"
                f"\n  goal=({failed.goal.row}, {failed.goal.col})"
                f"\n  optimal_length={failed.optimal_length}"
                f"\n  astar_path_length={failed.astar_path_length}"
                f"\n  hpa_abstract_nodes_visited={failed.hpa_abstract_nodes_visited}"
                f"\n  hpa_abstract_path_edge_count={failed.hpa_abstract_path_edge_count}"
            )

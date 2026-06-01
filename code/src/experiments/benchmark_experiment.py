from code.src.algorithms.hpa.hpa_star import HPAStar
from code.src.core.benchmark_models import BenchmarkResult
from code.src.core.models import PathfindingResult, Scenario
from code.src.experiments.algorithm_registry import create_algorithm
from code.src.experiments.base_experiment import BaseExperiment
from code.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from code.src.utils.csv_exporter import export_benchmark_results


class BenchmarkExperiment(BaseExperiment):
    def __init__(
            self,
            config: ExperimentConfig,
            algorithms: list[AlgorithmName],
    ) -> None:
        super().__init__(config)

        if not algorithms:
            raise ValueError("At least one algorithm is required.")

        self.algorithms = algorithms

    def run(self) -> None:
        grid_map, scenarios = self._load_filtered_scenarios()
        selected_scenarios = scenarios[: self.config.benchmark_scenarios]

        results: list[BenchmarkResult] = []

        for algorithm_name in self.algorithms:
            algorithm = create_algorithm(
                name=algorithm_name,
                cluster_size=self.config.cluster_size,
                max_entrances_per_cluster_pair=self.config.max_entrances_per_cluster_pair,
                min_entrance_width=self.config.min_entrance_width,
            )

            if isinstance(algorithm, HPAStar):
                algorithm.preprocess_map(grid_map)

            for scenario_index, scenario in enumerate(selected_scenarios):
                pathfinding_result = algorithm.find_path(
                    grid_map=grid_map,
                    start=scenario.start,
                    goal=scenario.goal,
                )

                benchmark_result = self._build_benchmark_result(
                    algorithm=algorithm,
                    scenario=scenario,
                    scenario_index=scenario_index,
                    pathfinding_result=pathfinding_result,
                )

                results.append(benchmark_result)

        self._print_summary(results)

        output_dir = self.config.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        export_benchmark_results(
            results=results,
            output_path=output_dir / self.config.benchmark_output_file
        )

    @staticmethod
    def _build_benchmark_result(
            algorithm,
            scenario: Scenario,
            scenario_index: int,
            pathfinding_result: PathfindingResult,
    ) -> BenchmarkResult:
        base_result = BenchmarkResult(
            algorithm=pathfinding_result.algorithm_name,
            scenario_index=scenario_index,
            optimal_length=scenario.optimal_length,
            found=pathfinding_result.found,
            path_length=pathfinding_result.path_length,
            visited_nodes=pathfinding_result.visited_nodes,
            execution_time_ms=pathfinding_result.execution_time_ms,
        )

        if not isinstance(algorithm, HPAStar):
            return base_result

        preprocessing_stats = algorithm.get_preprocessing_stats()
        query_stats = algorithm.get_last_query_stats()

        return BenchmarkResult(
            algorithm=pathfinding_result.algorithm_name,
            scenario_index=scenario_index,
            optimal_length=scenario.optimal_length,
            found=pathfinding_result.found,
            path_length=pathfinding_result.path_length,
            visited_nodes=pathfinding_result.visited_nodes,
            execution_time_ms=pathfinding_result.execution_time_ms,
            preprocessing_time_ms=preprocessing_stats.preprocessing_time_ms,
            abstract_nodes_visited=query_stats.abstract_nodes_visited,
            abstract_path_edge_count=query_stats.abstract_path_edge_count,
            refined_path_length=query_stats.refined_path_length,
            local_path_cache_hits=query_stats.local_path_cache_hits,
            local_path_cache_misses=query_stats.local_path_cache_misses,
            cache_hit_ratio=query_stats.cache_hit_ratio,
            abstract_search_time_ms=query_stats.abstract_search_time_ms,
            refinement_time_ms=query_stats.refinement_time_ms,
            query_time_ms=query_stats.query_time_ms,
        )

    @staticmethod
    def _print_summary(
            results: list[BenchmarkResult],
    ) -> None:
        grouped_results: dict[str, list[BenchmarkResult]] = {}

        for result in results:
            grouped_results.setdefault(result.algorithm, []).append(result)

        print("\nBenchmark summary:")

        for algorithm_name, algorithm_results in grouped_results.items():
            found_results = [
                result for result in algorithm_results
                if result.found
            ]

            if not found_results:
                print(f"\n{algorithm_name}: no paths found")
                continue

            avg_execution_time = sum(
                result.execution_time_ms for result in found_results
            ) / len(found_results)

            avg_path_length = sum(
                result.path_length for result in found_results
            ) / len(found_results)

            avg_visited_nodes = sum(
                result.visited_nodes for result in found_results
            ) / len(found_results)

            print(f"\n{algorithm_name}")
            print(f"  scenarios: {len(algorithm_results)}")
            print(f"  found: {len(found_results)}")
            print(f"  avg_execution_time_ms: {avg_execution_time:.3f}")
            print(f"  avg_path_length: {avg_path_length:.3f}")
            print(f"  avg_visited_nodes: {avg_visited_nodes:.3f}")

            hpa_results = [
                result for result in found_results
                if result.query_time_ms is not None
            ]

            if not hpa_results:
                continue

            avg_query_time = sum(
                result.query_time_ms or 0.0 for result in hpa_results
            ) / len(hpa_results)

            avg_cache_hit_ratio = sum(
                result.cache_hit_ratio or 0.0 for result in hpa_results
            ) / len(hpa_results)

            avg_abstract_nodes_visited = sum(
                result.abstract_nodes_visited or 0 for result in hpa_results
            ) / len(hpa_results)

            preprocessing_time = hpa_results[0].preprocessing_time_ms or 0.0

            print(f"  preprocessing_time_ms: {preprocessing_time:.3f}")
            print(f"  avg_query_time_ms: {avg_query_time:.3f}")
            print(f"  avg_cache_hit_ratio: {avg_cache_hit_ratio:.3f}")
            print(f"  avg_abstract_nodes_visited: {avg_abstract_nodes_visited:.3f}")

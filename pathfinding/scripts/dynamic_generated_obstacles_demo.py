from pathlib import Path

from pathfinding.src.core.moving_obstacle_generator import (
    MovingObstacleGeneratorConfig,
    generate_moving_obstacles,
)
from pathfinding.src.experiments.dynamic_replanning_visualization_experiment import (
    DynamicReplanningVisualizationExperiment,
)
from pathfinding.src.experiments.experiment_config import (
    AlgorithmName,
    ExperimentConfig,
)


def main() -> None:
    map_path = Path("../../Data/bg512-map/AR0204SR.map")
    scen_path = Path("../../Data/bg512-scen/AR0204SR.map.scen")

    config = ExperimentConfig(
        map_path=map_path,
        scen_path=scen_path,
        scenario_index=1,
        min_optimal_length=400,
        algorithm=AlgorithmName.DSTAR_LITE,
        fps=120,
        step_delay_ms=1,
        steps_per_frame=200,
        step_record_interval=40,
        movement_hold_frames=1,
        movement_frame_delay_ms=0,
        movement_steps_per_frame=100,
        obstacle_hold_frames=1,
        obstacle_frame_delay_ms=0,
        path_block_lookahead=40,
        wait_when_no_path=True,
        max_wait_steps=30,
        moving_obstacle_prediction_steps=5,
    )

    scenario_loader = DynamicReplanningVisualizationExperiment(config=config)
    grid_map, scenario = scenario_loader._load_map_and_scenario()

    forbidden_positions = {
        (scenario.start.row, scenario.start.col),
        (scenario.goal.row, scenario.goal.col),
    }

    moving_obstacles = generate_moving_obstacles(
        grid_map=grid_map,
        config=MovingObstacleGeneratorConfig(
            seed=42,
            obstacle_count=15,
            min_width=6,
            max_width=10,
            min_height=6,
            max_height=10,
        ),
        forbidden_positions=forbidden_positions,
    )

    experiment = DynamicReplanningVisualizationExperiment(
        config=config,
        moving_obstacles=moving_obstacles,
        grid_map=grid_map,
        scenario=scenario,
    )

    experiment.run()


if __name__ == "__main__":
    main()

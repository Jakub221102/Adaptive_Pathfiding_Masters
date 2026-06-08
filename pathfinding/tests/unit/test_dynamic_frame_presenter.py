from pathlib import Path
from unittest.mock import MagicMock

from pathfinding.src.core.models import Position
from pathfinding.src.experiments.dynamic_simulation import DynamicSimulationKeyframe
from pathfinding.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from pathfinding.src.visualization.dynamic_frame_presenter import DynamicFramePresenter


def _build_presenter(movement_steps_per_frame: int) -> tuple[DynamicFramePresenter, MagicMock]:
    viewer = MagicMock()
    viewer.show_dynamic_frame.return_value = True

    config = ExperimentConfig(
        map_path=Path("."),
        scen_path=Path("."),
        algorithm=AlgorithmName.DSTAR_LITE,
        movement_steps_per_frame=movement_steps_per_frame,
        movement_hold_frames=1,
        movement_frame_delay_ms=0,
        obstacle_hold_frames=1,
        obstacle_frame_delay_ms=0,
    )

    presenter = DynamicFramePresenter(
        viewer=viewer,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=10),
        config=config,
    )
    return presenter, viewer


def _obstacle_keyframe(row: int) -> DynamicSimulationKeyframe:
    return DynamicSimulationKeyframe(
        frame_type="obstacle",
        agent_position=Position(row=0, col=0),
        planned_path=[],
        travelled_path=[Position(row=0, col=0)],
        dynamic_blocked={(row, 0)},
        status_text=f"obstacle at row {row}",
    )


def test_obstacle_keyframes_are_batched_by_movement_steps_per_frame() -> None:
    presenter, viewer = _build_presenter(movement_steps_per_frame=4)

    for row in range(1, 4):
        presenter.present_keyframe(_obstacle_keyframe(row))

    viewer.show_dynamic_frame.assert_not_called()

    presenter.present_keyframe(_obstacle_keyframe(4))

    assert viewer.show_dynamic_frame.call_count == 1


def test_pending_obstacle_is_flushed_before_search_keyframe() -> None:
    presenter, viewer = _build_presenter(movement_steps_per_frame=4)

    presenter.present_keyframe(_obstacle_keyframe(1))
    presenter.present_keyframe(
        DynamicSimulationKeyframe(
            frame_type="search",
            agent_position=Position(row=0, col=0),
            planned_path=[Position(row=0, col=1)],
            travelled_path=[Position(row=0, col=0)],
            status_text="replanning",
        )
    )

    assert viewer.show_dynamic_frame.call_count >= 1

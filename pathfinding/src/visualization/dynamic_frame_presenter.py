import pygame

from pathfinding.src.core.models import Position
from pathfinding.src.experiments.dynamic_simulation import (
    DynamicPlaybackFrame,
    DynamicSimulationKeyframe,
)
from pathfinding.src.experiments.experiment_config import ExperimentConfig
from pathfinding.src.visualization.pygame_viewer import PygameGridViewer


class DynamicFramePresenter:
    def __init__(
            self,
            viewer: PygameGridViewer,
            start: Position,
            goal: Position,
            config: ExperimentConfig,
    ) -> None:
        self.viewer = viewer
        self.start = start
        self.goal = goal
        self.config = config
        self._running = True
        self._pending_move_keyframe: DynamicSimulationKeyframe | None = None
        self._move_frames_since_present = 0
        self._pending_obstacle_keyframe: DynamicSimulationKeyframe | None = None
        self._obstacle_frames_since_present = 0

    @property
    def running(self) -> bool:
        return self._running

    def finalize(self) -> None:
        self._flush_pending_move()
        self._flush_pending_obstacle()

    def present_keyframe(self, keyframe: DynamicSimulationKeyframe) -> None:
        if not self._running:
            return

        if keyframe.frame_type == "move":
            self._pending_move_keyframe = keyframe
            self._move_frames_since_present += 1

            if self._move_frames_since_present >= self.config.movement_steps_per_frame:
                self._present_move(self._pending_move_keyframe)
                self._pending_move_keyframe = None
                self._move_frames_since_present = 0

            return

        if keyframe.frame_type == "obstacle":
            self._pending_obstacle_keyframe = keyframe
            self._obstacle_frames_since_present += 1

            if self._obstacle_frames_since_present >= self.config.movement_steps_per_frame:
                self._present_obstacle(self._pending_obstacle_keyframe)
                self._pending_obstacle_keyframe = None
                self._obstacle_frames_since_present = 0

            return

        self._flush_pending_move()
        self._flush_pending_obstacle()

        if keyframe.frame_type == "search":
            self._present_search(keyframe)
            return

        if keyframe.frame_type == "path_update":
            self._present_path_update(keyframe)
            return

    def _flush_pending_move(self) -> None:
        if self._pending_move_keyframe is None:
            return

        self._present_move(self._pending_move_keyframe)
        self._pending_move_keyframe = None
        self._move_frames_since_present = 0

    def _flush_pending_obstacle(self) -> None:
        if self._pending_obstacle_keyframe is None:
            return

        self._present_obstacle(self._pending_obstacle_keyframe)
        self._pending_obstacle_keyframe = None
        self._obstacle_frames_since_present = 0

    def _present_search(self, keyframe: DynamicSimulationKeyframe) -> None:
        search_frame = self._to_playback_frame(keyframe).model_copy(
            update={"planned_path": []},
        )
        search_steps = keyframe.search_steps or []

        if not search_steps:
            self._show_revealed_path(keyframe)
            return

        for step_index in range(0, len(search_steps), self.config.steps_per_frame):
            self._show_frame(
                search_frame.model_copy(
                    update={"search_step": search_steps[step_index]},
                ),
                self.config.step_delay_ms,
            )

        self._show_frame(
            search_frame.model_copy(update={"search_step": search_steps[-1]}),
            self.config.step_delay_ms,
        )
        self._show_revealed_path(keyframe)

    def _show_revealed_path(self, keyframe: DynamicSimulationKeyframe) -> None:
        revealed_frame = self._to_playback_frame(keyframe).model_copy(
            update={"search_step": None},
        )
        self._show_frame(revealed_frame, self.config.step_delay_ms)

    def _present_obstacle(self, keyframe: DynamicSimulationKeyframe) -> None:
        base_frame = self._to_playback_frame(keyframe)

        for _ in range(self.config.obstacle_hold_frames):
            self._show_frame(base_frame, self.config.obstacle_frame_delay_ms)

    def _present_path_update(self, keyframe: DynamicSimulationKeyframe) -> None:
        self._show_frame(
            self._to_playback_frame(keyframe),
            self.config.obstacle_frame_delay_ms,
        )

    def _present_move(self, keyframe: DynamicSimulationKeyframe) -> None:
        base_frame = self._to_playback_frame(keyframe)

        for _ in range(self.config.movement_hold_frames):
            self._show_frame(base_frame, self.config.movement_frame_delay_ms)

    def _show_frame(
            self,
            frame: DynamicPlaybackFrame,
            frame_delay_ms: int,
    ) -> None:
        if not self._running:
            return

        self._running = self.viewer.show_dynamic_frame(
            start=self.start,
            goal=self.goal,
            frame=frame,
        )

        if not self._running or frame_delay_ms <= 0:
            return

        pygame.time.wait(frame_delay_ms)

    @staticmethod
    def _to_playback_frame(
            keyframe: DynamicSimulationKeyframe,
    ) -> DynamicPlaybackFrame:
        return DynamicPlaybackFrame(
            frame_kind=keyframe.frame_type,
            agent_position=keyframe.agent_position,
            planned_path=keyframe.planned_path,
            travelled_path=keyframe.travelled_path,
            dynamic_blocked=set(keyframe.dynamic_blocked),
            new_obstacle_positions=list(keyframe.new_obstacle_positions),
            status_text=keyframe.status_text,
        )

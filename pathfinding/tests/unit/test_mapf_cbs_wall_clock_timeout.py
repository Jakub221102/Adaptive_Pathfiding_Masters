from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.core.models import Position
from pathfinding.src.experiments.mapf_benchmark_execution import (
    TERMINATION_ERROR,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_SUCCESS,
    TERMINATION_TIME_LIMIT,
    BenchmarkCheckpointStore,
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkRunKey,
    MAPFCBSSearchMetrics,
    create_time_limit_benchmark_record,
    load_checkpoint_records,
    run_cbs_with_wall_clock_limit,
    validate_benchmark_run_record,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFInteractionLevel,
)
from pathfinding.src.experiments.mapf_cbs_process_runner import (
    CBSWorkerInput,
    CBSWorkerOutput,
    run_cbs_worker_subprocess,
    terminate_process,
)
from pathfinding.tests.helpers import build_grid_map

_FAKE_METRICS = MAPFCBSSearchMetrics(
    expanded_ct_nodes=3,
    generated_ct_nodes=5,
    low_level_replans=4,
    max_open_size=2,
    unique_constraint_signatures=0,
    duplicate_constraint_signatures=0,
    unique_path_signatures=0,
    duplicate_path_signatures=0,
    classified_conflicts=0,
    classification_low_level_searches=0,
    selected_cardinal_conflicts=0,
    selected_semi_cardinal_conflicts=0,
    selected_non_cardinal_conflicts=0,
    combined_low_level_searches=4,
)


def _worker_input() -> CBSWorkerInput:
    grid_map = build_grid_map([[0, 0, 0]], name="bench.map")
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(
                agent_id=0,
                start=Position(row=0, col=0),
                goal=Position(row=0, col=2),
            ),
        )
    )
    return CBSWorkerInput(
        grid_map=grid_map,
        scenario=scenario,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC.value,
        max_timestep=8,
        max_expanded_nodes=250,
    )


def _benchmark_instance() -> MAPFBenchmarkInstance:
    return MAPFBenchmarkInstance(
        instance_id="bench_n02_low_000",
        agent_count=2,
        interaction_level=MAPFInteractionLevel.LOW,
        scenario_indices=(0, 1),
        independent_conflict_count=0,
        conflicting_agent_pair_count=0,
        independent_soc=4,
        independent_makespan=2,
        vertex_conflict_count=0,
        edge_conflict_count=0,
    )


def fake_success_worker(
    worker_input: CBSWorkerInput,
    output_queue: multiprocessing.Queue,
) -> None:
    _ = worker_input
    output_queue.put(
        CBSWorkerOutput(
            termination_reason=TERMINATION_SUCCESS,
            success=True,
            soc=4,
            makespan=2,
            cbs_search_metrics=_FAKE_METRICS,
        )
    )


def fake_expansion_limit_worker(
    worker_input: CBSWorkerInput,
    output_queue: multiprocessing.Queue,
) -> None:
    _ = worker_input
    output_queue.put(
        CBSWorkerOutput(
            termination_reason=TERMINATION_EXPANSION_LIMIT,
            success=False,
            soc=None,
            makespan=None,
            cbs_search_metrics=_FAKE_METRICS,
        )
    )


def fake_sleep_worker(
    worker_input: CBSWorkerInput,
    output_queue: multiprocessing.Queue,
) -> None:
    _ = worker_input
    time.sleep(5.0)
    output_queue.put(
        CBSWorkerOutput(
            termination_reason=TERMINATION_SUCCESS,
            success=True,
            soc=4,
            makespan=2,
            cbs_search_metrics=_FAKE_METRICS,
        )
    )


def fake_error_worker(
    worker_input: CBSWorkerInput,
    output_queue: multiprocessing.Queue,
) -> None:
    _ = worker_input
    try:
        raise RuntimeError("synthetic worker failure")
    except Exception as error:
        import traceback

        output_queue.put(
            CBSWorkerOutput(
                termination_reason=TERMINATION_ERROR,
                success=False,
                soc=None,
                makespan=None,
                cbs_search_metrics=None,
                error_message=f"{type(error).__name__}: {error}",
                error_traceback=traceback.format_exc(),
            )
        )


def fake_empty_exit_worker(
    worker_input: CBSWorkerInput,
    output_queue: multiprocessing.Queue,
) -> None:
    _ = worker_input
    _ = output_queue


def test_subprocess_fast_success_returns_success_and_cleans_up() -> None:
    worker_input = _worker_input()
    output, timed_out, execution_time_ms = run_cbs_worker_subprocess(
        worker_input,
        max_runtime_seconds=2.0,
        worker_target=fake_success_worker,
    )

    assert timed_out is False
    assert output is not None
    assert output.termination_reason == TERMINATION_SUCCESS
    assert output.success is True
    assert execution_time_ms >= 0.0


def test_subprocess_expansion_limit_before_timeout() -> None:
    worker_input = _worker_input()
    output, timed_out, execution_time_ms = run_cbs_worker_subprocess(
        worker_input,
        max_runtime_seconds=2.0,
        worker_target=fake_expansion_limit_worker,
    )

    assert timed_out is False
    assert output is not None
    assert output.termination_reason == TERMINATION_EXPANSION_LIMIT
    assert output.termination_reason != TERMINATION_TIME_LIMIT
    assert execution_time_ms >= 0.0


def test_subprocess_wall_clock_timeout_terminates_worker() -> None:
    worker_input = _worker_input()
    timeout_seconds = 0.15
    active_worker: dict[str, object] = {"process": None}

    output, timed_out, execution_time_ms = run_cbs_worker_subprocess(
        worker_input,
        max_runtime_seconds=timeout_seconds,
        worker_target=fake_sleep_worker,
        active_worker=active_worker,
    )

    assert timed_out is True
    assert output is None
    assert execution_time_ms >= timeout_seconds * 1000.0 * 0.8
    assert active_worker["process"] is None


def test_subprocess_child_error_returns_error() -> None:
    worker_input = _worker_input()
    output, timed_out, _ = run_cbs_worker_subprocess(
        worker_input,
        max_runtime_seconds=2.0,
        worker_target=fake_error_worker,
    )

    assert timed_out is False
    assert output is not None
    assert output.termination_reason == TERMINATION_ERROR
    assert output.error_message is not None
    assert "synthetic worker failure" in output.error_message


def test_subprocess_exit_without_payload_is_error() -> None:
    worker_input = _worker_input()
    output, timed_out, _ = run_cbs_worker_subprocess(
        worker_input,
        max_runtime_seconds=2.0,
        worker_target=fake_empty_exit_worker,
    )

    assert timed_out is False
    assert output is None


def test_run_cbs_with_wall_clock_limit_success() -> None:
    worker_input = _worker_input()
    instance = _benchmark_instance()
    record = run_cbs_with_wall_clock_limit(
        grid_map=worker_input.grid_map,
        scenario=worker_input.scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        max_timestep=worker_input.max_timestep,
        max_expanded_nodes=worker_input.max_expanded_nodes,
        max_runtime_seconds=2.0,
        worker_target=fake_success_worker,
    )

    assert record.success is True
    assert record.termination_reason == TERMINATION_SUCCESS
    assert record.cbs_search_metrics is not None
    validate_benchmark_run_record(record)


def test_run_cbs_with_wall_clock_limit_time_limit() -> None:
    worker_input = _worker_input()
    instance = _benchmark_instance()
    record = run_cbs_with_wall_clock_limit(
        grid_map=worker_input.grid_map,
        scenario=worker_input.scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        max_timestep=worker_input.max_timestep,
        max_expanded_nodes=worker_input.max_expanded_nodes,
        max_runtime_seconds=0.15,
        worker_target=fake_sleep_worker,
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_TIME_LIMIT
    assert record.soc is None
    assert record.makespan is None
    assert record.cbs_search_metrics is None
    validate_benchmark_run_record(record)


def test_run_cbs_with_wall_clock_limit_expansion_limit() -> None:
    worker_input = _worker_input()
    instance = _benchmark_instance()
    record = run_cbs_with_wall_clock_limit(
        grid_map=worker_input.grid_map,
        scenario=worker_input.scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        max_timestep=worker_input.max_timestep,
        max_expanded_nodes=worker_input.max_expanded_nodes,
        max_runtime_seconds=2.0,
        worker_target=fake_expansion_limit_worker,
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_EXPANSION_LIMIT
    assert record.cbs_search_metrics is not None
    validate_benchmark_run_record(record)


def test_run_cbs_with_wall_clock_limit_child_error() -> None:
    worker_input = _worker_input()
    instance = _benchmark_instance()
    record = run_cbs_with_wall_clock_limit(
        grid_map=worker_input.grid_map,
        scenario=worker_input.scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        max_timestep=worker_input.max_timestep,
        max_expanded_nodes=worker_input.max_expanded_nodes,
        max_runtime_seconds=2.0,
        worker_target=fake_error_worker,
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_ERROR
    assert record.error_message is not None
    validate_benchmark_run_record(record)


def test_run_cbs_with_wall_clock_limit_missing_payload() -> None:
    worker_input = _worker_input()
    instance = _benchmark_instance()
    record = run_cbs_with_wall_clock_limit(
        grid_map=worker_input.grid_map,
        scenario=worker_input.scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        max_timestep=worker_input.max_timestep,
        max_expanded_nodes=worker_input.max_expanded_nodes,
        max_runtime_seconds=2.0,
        worker_target=fake_empty_exit_worker,
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_ERROR
    assert record.error_message == "CBS worker exited without returning a result"
    validate_benchmark_run_record(record)


def test_time_limit_checkpoint_resume_skips_completed_run(tmp_path: Path) -> None:
    instance = _benchmark_instance()
    record = create_time_limit_benchmark_record(
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        execution_time_ms=180_000.0,
    )
    validate_benchmark_run_record(record)

    store = BenchmarkCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(record)

    loaded = load_checkpoint_records(tmp_path / "results_details.jsonl")
    key = MAPFBenchmarkRunKey(instance.instance_id, MAPFBenchmarkAlgorithm.CBS_BASIC)
    assert key in loaded
    assert loaded[key].termination_reason == TERMINATION_TIME_LIMIT


def test_terminate_process_reaps_sleeping_worker() -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=fake_sleep_worker,
        args=(_worker_input(), context.Queue()),
        daemon=False,
    )
    process.start()
    assert process.is_alive()

    terminate_process(process)
    assert process.is_alive() is False

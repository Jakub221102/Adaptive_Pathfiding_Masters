from __future__ import annotations

import multiprocessing
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from queue import Empty

from pathfinding.src.algorithms.mapf.cbs import (
    solve_cbs_cardinal_first_with_stats,
    solve_cbs_with_stats,
)
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import MAPFScenario
from pathfinding.src.core.models import GridMap
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAPFBenchmarkAlgorithm,
    MAPFCBSSearchMetrics,
    TERMINATION_ERROR,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
    _cbs_search_metrics_from_stats,
)


@dataclass(frozen=True, slots=True)
class CBSWorkerInput:
    grid_map: GridMap
    scenario: MAPFScenario
    algorithm: str
    max_timestep: int
    max_expanded_nodes: int | None


@dataclass(frozen=True, slots=True)
class CBSWorkerOutput:
    termination_reason: str
    success: bool
    soc: int | None
    makespan: int | None
    cbs_search_metrics: MAPFCBSSearchMetrics | None
    error_message: str | None = None
    error_traceback: str | None = None


def cbs_worker_main(
    worker_input: CBSWorkerInput,
    output_queue: multiprocessing.Queue,
) -> None:
    """Top-level CBS worker entry point (spawn-safe on Windows)."""
    try:
        grid_map = worker_input.grid_map
        scenario = worker_input.scenario
        algorithm = MAPFBenchmarkAlgorithm(worker_input.algorithm)

        if algorithm == MAPFBenchmarkAlgorithm.CBS_BASIC:
            run = solve_cbs_with_stats(
                grid_map=grid_map,
                scenario=scenario,
                max_timestep=worker_input.max_timestep,
                max_expanded_nodes=worker_input.max_expanded_nodes,
            )
        elif algorithm == MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST:
            run = solve_cbs_cardinal_first_with_stats(
                grid_map=grid_map,
                scenario=scenario,
                max_timestep=worker_input.max_timestep,
                max_expanded_nodes=worker_input.max_expanded_nodes,
            )
        else:
            raise ValueError(f"unsupported CBS worker algorithm: {algorithm}")

        result = run.result
        metrics = _cbs_search_metrics_from_stats(run.stats)

        if result is not None and result.success:
            output_queue.put(
                CBSWorkerOutput(
                    termination_reason=TERMINATION_SUCCESS,
                    success=True,
                    soc=sum_of_costs(result.paths),
                    makespan=makespan(result.paths),
                    cbs_search_metrics=metrics,
                )
            )
            return

        termination_reason = run.termination_reason
        if termination_reason not in {
            TERMINATION_FAILURE,
            TERMINATION_EXPANSION_LIMIT,
        }:
            termination_reason = TERMINATION_FAILURE

        output_queue.put(
            CBSWorkerOutput(
                termination_reason=termination_reason,
                success=False,
                soc=None,
                makespan=None,
                cbs_search_metrics=metrics,
            )
        )
    except Exception as error:
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


def terminate_process(process: Process, *, join_timeout: float = 5.0) -> None:
    """Terminate a child process and ensure it is reaped."""
    if not process.is_alive():
        process.join(timeout=join_timeout)
        return

    process.terminate()
    process.join(timeout=join_timeout)

    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=join_timeout)


def run_cbs_worker_subprocess(
    worker_input: CBSWorkerInput,
    *,
    max_runtime_seconds: float,
    worker_target=cbs_worker_main,
    join_grace_seconds: float = 1.0,
    active_worker: dict[str, object] | None = None,
) -> tuple[CBSWorkerOutput | None, bool, float]:
    """Run CBS in an isolated child process.

    Returns (worker_output, timed_out, execution_time_ms).
    """
    if max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")

    context = multiprocessing.get_context("spawn")
    output_queue: multiprocessing.Queue = context.Queue()
    process = context.Process(
        target=worker_target,
        args=(worker_input, output_queue),
        daemon=False,
    )

    start = time.perf_counter()
    if active_worker is not None:
        active_worker["process"] = process

    try:
        process.start()
        process.join(timeout=max_runtime_seconds)

        timed_out = process.is_alive()
        execution_time_ms = (time.perf_counter() - start) * 1000.0

        if timed_out:
            terminate_process(process, join_timeout=join_grace_seconds)
        else:
            process.join(timeout=join_grace_seconds)

        worker_output: CBSWorkerOutput | None = None
        try:
            worker_output = output_queue.get_nowait()
        except Empty:
            worker_output = None
        finally:
            output_queue.close()
            output_queue.join_thread()

        return worker_output, timed_out, execution_time_ms
    finally:
        if active_worker is not None:
            active_worker["process"] = None
        if process.is_alive():
            terminate_process(process, join_timeout=join_grace_seconds)

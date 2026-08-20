from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from pathfinding.src.experiments.mapf_benchmark_analysis import (
    build_pp_vs_basic_quality_summary,
    count_directional_soc_better,
    within_common_budget as analysis_within_budget,
)
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkRunRecord,
    MAPFPPSearchMetrics,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_SUCCESS,
    TERMINATION_TIME_LIMIT,
)
from pathfinding.src.experiments.mapf_benchmark_plots import (
    CBS_BUDGET_S,
    PlotInputs,
    build_thesis_tables_markdown,
    load_cardinal_overhead_points,
    load_high_outcome_cells,
    load_paired_soc_gaps,
    load_runtime_outcome_points,
    runtime_success_group_has_observations,
)


def _csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_zero_success_runtime_group_is_not_converted_to_zero() -> None:
    rows = _csv_rows(
        "algorithm,agent_count,interaction_level,metric,n,mean_ms,median_ms,min_ms,max_ms,std_ms\n"
        "cbs_basic,10,high,solve_time,0,,,,,\n"
    )
    assert runtime_success_group_has_observations(
        rows,
        algorithm="cbs_basic",
        agent_count=10,
        interaction_level="high",
    ) is False


def test_missing_runtime_success_group_is_none_not_zero() -> None:
    rows = _csv_rows(
        "algorithm,agent_count,interaction_level,metric,n,mean_ms\n"
        "cbs_basic,5,low,solve_time,3,1000.0\n"
    )
    assert runtime_success_group_has_observations(
        rows,
        algorithm="cbs_basic",
        agent_count=10,
        interaction_level="high",
    ) is None


def test_timeout_remains_distinct_outcome_in_runtime_plot_data() -> None:
    rows = _csv_rows(
        "instance_id,agent_count,interaction_level,independent_conflict_count,independent_soc,independent_makespan,"
        "fixed_priority_pp_termination_reason,fixed_priority_pp_success,fixed_priority_pp_execution_time_ms,"
        "fixed_priority_pp_soc,fixed_priority_pp_makespan,fixed_priority_pp_within_180s,"
        "cbs_basic_termination_reason,cbs_basic_success,cbs_basic_execution_time_ms,cbs_basic_soc,cbs_basic_makespan,"
        "cbs_basic_within_180s,cbs_basic_expanded_ct_nodes,cbs_basic_combined_low_level_searches,"
        "cbs_cardinal_first_termination_reason,cbs_cardinal_first_success,cbs_cardinal_first_execution_time_ms,"
        "cbs_cardinal_first_soc,cbs_cardinal_first_makespan,cbs_cardinal_first_within_180s,"
        "cbs_cardinal_first_expanded_ct_nodes,cbs_cardinal_first_combined_low_level_searches\n"
        "AR0204SR_n10_high_000,10,high,10,2483,511,success,True,18812.0,2509,511,True,"
        "time_limit,False,180169.0,,,False,,,"
        "time_limit,False,180179.0,,,False,,\n"
    )
    points = load_runtime_outcome_points(rows)
    basic = next(point for point in points if point.algorithm == "cbs_basic")
    assert basic.termination_reason == TERMINATION_TIME_LIMIT
    assert basic.execution_time_s == 180.169


def test_pp_slow_success_remains_success_in_runtime_plot_data() -> None:
    rows = _csv_rows(
        "instance_id,agent_count,interaction_level,independent_conflict_count,independent_soc,independent_makespan,"
        "fixed_priority_pp_termination_reason,fixed_priority_pp_success,fixed_priority_pp_execution_time_ms,"
        "fixed_priority_pp_soc,fixed_priority_pp_makespan,fixed_priority_pp_within_180s,"
        "cbs_basic_termination_reason,cbs_basic_success,cbs_basic_execution_time_ms,cbs_basic_soc,cbs_basic_makespan,"
        "cbs_basic_within_180s,cbs_basic_expanded_ct_nodes,cbs_basic_combined_low_level_searches,"
        "cbs_cardinal_first_termination_reason,cbs_cardinal_first_success,cbs_cardinal_first_execution_time_ms,"
        "cbs_cardinal_first_soc,cbs_cardinal_first_makespan,cbs_cardinal_first_within_180s,"
        "cbs_cardinal_first_expanded_ct_nodes,cbs_cardinal_first_combined_low_level_searches\n"
        "AR0204SR_n20_high_001,20,high,10,4742,490,success,True,190051.0,5461,490,False,"
        "time_limit,False,180174.0,,,False,,,"
        "time_limit,False,180179.0,,,False,,\n"
    )
    points = load_runtime_outcome_points(rows)
    pp = next(point for point in points if point.algorithm == "fixed_priority_pp")
    assert pp.termination_reason == TERMINATION_SUCCESS
    assert pp.is_pp_slow_success is True
    assert pp.execution_time_s > CBS_BUDGET_S


def test_paired_soc_plot_contains_only_common_success_rows() -> None:
    rows = _csv_rows(
        "comparison,instance_id,agent_count,interaction_level,independent_conflict_count,left_algorithm,right_algorithm,"
        "left_soc,right_soc,left_makespan,right_makespan,pp_minus_basic_soc,pp_minus_basic_makespan,"
        "pp_minus_cardinal_soc,pp_minus_cardinal_makespan,basic_minus_cardinal_soc,basic_minus_cardinal_makespan\n"
        "pp_minus_basic,inst_ok,5,low,0,fixed_priority_pp,cbs_basic,110,100,20,20,10,0,,,,\n"
        "pp_minus_basic,inst_missing,10,high,3,fixed_priority_pp,cbs_basic,,,,,,,,,\n"
    )
    gaps = load_paired_soc_gaps(rows)
    assert len(gaps) == 1
    assert gaps[0].instance_id == "inst_ok"
    assert gaps[0].soc_gap == 10


def test_cardinal_aggregate_row_excluded_from_scatter_data() -> None:
    rows = _csv_rows(
        "instance_id,agent_count,interaction_level,independent_conflict_count,basic_runtime_ms,cardinal_runtime_ms,"
        "runtime_ratio_cardinal_over_basic,basic_expanded_ct_nodes,cardinal_expanded_ct_nodes,"
        "ct_expansion_reduction_basic_minus_cardinal,basic_combined_low_level_searches,"
        "cardinal_low_level_replans,cardinal_classification_low_level_searches,"
        "cardinal_combined_low_level_searches,additional_low_level_searches_cardinal_minus_basic,"
        "cardinal_classified_conflicts,cardinal_selected_cardinal_conflicts,"
        "cardinal_selected_semi_cardinal_conflicts,cardinal_selected_non_cardinal_conflicts,"
        "summary_level,paired_instance_count,mean_runtime_ratio,mean_ct_reduction,"
        "mean_additional_low_level_searches,median_runtime_ratio\n"
        "inst_a,5,low,0,1000,1200,1.2,2,1,1,4,3,1,4,0,0,0,0,0,,,,,,\n"
        ",,,,,,,,,,,,,,,,,,,common_success_instances,1,1.2,1.0,0.0,1.2\n"
    )
    points = load_cardinal_overhead_points(rows)
    assert len(points) == 1
    assert points[0].instance_id == "inst_a"


def test_cardinal_runtime_ratio_reference_semantics() -> None:
    rows = _csv_rows(
        "instance_id,agent_count,interaction_level,independent_conflict_count,basic_runtime_ms,cardinal_runtime_ms,"
        "runtime_ratio_cardinal_over_basic,basic_expanded_ct_nodes,cardinal_expanded_ct_nodes,"
        "ct_expansion_reduction_basic_minus_cardinal,basic_combined_low_level_searches,"
        "cardinal_low_level_replans,cardinal_classification_low_level_searches,"
        "cardinal_combined_low_level_searches,additional_low_level_searches_cardinal_minus_basic\n"
        "fast_cardinal,5,low,0,2000,1500,0.75,4,4,0,8,8,0,8,0\n"
        "slow_cardinal,5,low,0,1000,1500,1.5,4,2,2,8,8,4,12,4\n"
    )
    points = load_cardinal_overhead_points(rows)
    fast = next(point for point in points if point.instance_id == "fast_cardinal")
    slow = next(point for point in points if point.instance_id == "slow_cardinal")
    assert fast.runtime_ratio < 1.0
    assert slow.runtime_ratio > 1.0
    assert slow.ct_reduction > 0


def test_high_outcome_matrix_has_nine_instances_times_three_algorithms() -> None:
    rows = _csv_rows(
        "instance_id,agent_count,interaction_level,independent_conflict_count,independent_soc,independent_makespan,"
        "algorithm,termination_reason,success,execution_time_ms,soc,makespan\n"
    )
    for agent_count in (5, 10, 20):
        for index in range(3):
            instance_id = f"AR0204SR_n{agent_count:02d}_high_{index:03d}"
            for algorithm in (
                "fixed_priority_pp",
                "cbs_basic",
                "cbs_cardinal_first",
            ):
                termination = TERMINATION_SUCCESS
                execution = "1000.0"
                if agent_count >= 10 and algorithm != "fixed_priority_pp" and index == 0:
                    termination = TERMINATION_TIME_LIMIT
                    execution = "180000.0"
                if agent_count == 20 and algorithm == "cbs_basic" and index == 0:
                    termination = TERMINATION_EXPANSION_LIMIT
                rows.append(
                    {
                        "instance_id": instance_id,
                        "agent_count": str(agent_count),
                        "interaction_level": "high",
                        "independent_conflict_count": str(index + 1),
                        "independent_soc": "100",
                        "independent_makespan": "20",
                        "algorithm": algorithm,
                        "termination_reason": termination,
                        "success": str(termination == TERMINATION_SUCCESS),
                        "execution_time_ms": execution,
                        "soc": "100",
                        "makespan": "20",
                    }
                )

    cells = load_high_outcome_cells(rows)
    assert len(cells) == 27
    assert len({cell.instance_id for cell in cells}) == 9
    assert len({cell.algorithm for cell in cells}) == 3


def test_pp_post_hoc_budget_helpers() -> None:
    record = MAPFBenchmarkRunRecord(
        instance_id="slow_pp",
        algorithm=MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        agent_count=20,
        interaction_level="high",
        success=True,
        termination_reason=TERMINATION_SUCCESS,
        execution_time_ms=181_000.0,
        soc=100,
        makespan=20,
        remaining_conflicts=0,
        independent_soc=100,
        independent_makespan=20,
        independent_conflict_count=3,
        pp_search_metrics=MAPFPPSearchMetrics(low_level_searches=20, agents_planned=20),
    )
    assert record.success
    assert analysis_within_budget(record, 180_000.0) is False


def test_count_directional_soc_better_semantics() -> None:
    assert count_directional_soc_better([5]).right_better == 1
    assert count_directional_soc_better([-5]).left_better == 1
    assert count_directional_soc_better([0]).equal == 1


def test_thesis_table_b_recomputes_directional_counts_from_paired_rows(
    tmp_path: Path,
) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    (analysis_dir / "termination_summary.csv").write_text(
        "algorithm,termination_reason,count\n"
        "fixed_priority_pp,success,1\n",
        encoding="utf-8",
    )
    (analysis_dir / "common_180s_budget_summary.csv").write_text(
        "algorithm,total_runs,observed_success_count,observed_success_rate,"
        "within_common_budget_success_count,within_common_budget_success_rate,common_budget_ms\n"
        "fixed_priority_pp,1,1,1.0,1,1.0,180000\n"
        "cbs_basic,1,1,1.0,1,1.0,180000\n"
        "cbs_cardinal_first,1,1,1.0,1,1.0,180000\n",
        encoding="utf-8",
    )
    (analysis_dir / "cardinal_overhead_summary.csv").write_text(
        "instance_id,agent_count,interaction_level,independent_conflict_count,"
        "basic_runtime_ms,cardinal_runtime_ms,runtime_ratio_cardinal_over_basic,"
        "basic_expanded_ct_nodes,cardinal_expanded_ct_nodes,"
        "ct_expansion_reduction_basic_minus_cardinal,basic_combined_low_level_searches,"
        "cardinal_low_level_replans,cardinal_classification_low_level_searches,"
        "cardinal_combined_low_level_searches,additional_low_level_searches_cardinal_minus_basic\n"
        "inst,5,low,0,1000,1200,1.2,2,1,1,4,3,1,4,0\n",
        encoding="utf-8",
    )
    quality_header = (
        "comparison,paired_instance_count,mean_soc_diff,median_soc_diff,std_soc_diff,"
        "mean_makespan_diff,median_makespan_diff,std_makespan_diff,"
        "left_better_soc_count,right_better_soc_count,equal_soc_count\n"
    )
    (analysis_dir / "pp_vs_basic_quality_summary.csv").write_text(
        quality_header + "pp_minus_basic,2,5.0,5.0,0,0,0,0,99,0,0\n",
        encoding="utf-8",
    )
    (analysis_dir / "pp_vs_cardinal_quality_summary.csv").write_text(
        quality_header + "pp_minus_cardinal,1,0,0,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    (analysis_dir / "basic_vs_cardinal_quality_summary.csv").write_text(
        quality_header + "basic_minus_cardinal,1,0,0,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    (analysis_dir / "paired_quality_comparison.csv").write_text(
        "comparison,instance_id,agent_count,interaction_level,independent_conflict_count,"
        "left_algorithm,right_algorithm,left_soc,right_soc,left_makespan,right_makespan,"
        "pp_minus_basic_soc,pp_minus_basic_makespan,pp_minus_cardinal_soc,"
        "pp_minus_cardinal_makespan,basic_minus_cardinal_soc,basic_minus_cardinal_makespan\n"
        "pp_minus_basic,a,5,low,0,fixed_priority_pp,cbs_basic,110,100,20,20,10,0,,,\n"
        "pp_minus_basic,b,5,low,0,fixed_priority_pp,cbs_basic,100,105,20,20,-5,0,,,\n"
        "pp_minus_cardinal,a,5,low,0,fixed_priority_pp,cbs_cardinal_first,100,100,20,20,,,0,0,,\n"
        "basic_minus_cardinal,a,5,low,0,cbs_basic,cbs_cardinal_first,100,100,20,20,,,,0,0\n",
        encoding="utf-8",
    )

    markdown = build_thesis_tables_markdown(PlotInputs(analysis_dir=analysis_dir))
    assert "| PP vs Basic CBS | 2 | 5.00 | 5.00 | 10 | 1 | 0 | 1 |" in markdown
    assert "| PP vs Cardinal-First CBS | 1 | 0.00 | 0.00 | 0 | 0 | 1 | 0 |" in markdown

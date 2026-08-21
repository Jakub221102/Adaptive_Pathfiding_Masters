from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_HELD_OUT,
    DEFAULT_HELD_OUT_SEED,
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    collect_scenario_set_signatures_by_agent_count,
    generate_benchmark_manifest,
    generate_held_out_benchmark_manifest,
    load_benchmark_manifest,
    save_benchmark_manifest,
    validate_held_out_catalogue,
)
from pathfinding.tests.unit.test_mapf_benchmark_instances import (
    _build_rich_scenario_pool,
)


def _primary_manifest(grid_map, scenarios, *, seed: int = 2026) -> MAPFBenchmarkManifest:
    return generate_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name="bench.map.scen",
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=seed,
        min_reference_length=20.0,
        max_attempts=5000,
    ).manifest


def test_held_out_generation_excludes_primary_scenario_sets() -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    primary = _primary_manifest(grid_map, scenarios, seed=2026)

    held_out = generate_held_out_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name="bench.map.scen",
        primary_manifest=primary,
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=DEFAULT_HELD_OUT_SEED,
        min_reference_length=20.0,
        max_attempts=5000,
    ).manifest

    primary_sigs = collect_scenario_set_signatures_by_agent_count(primary.instances)
    held_out_sigs = collect_scenario_set_signatures_by_agent_count(held_out.instances)
    assert primary_sigs[3].isdisjoint(held_out_sigs[3])

    validate_held_out_catalogue(
        held_out,
        primary,
        scenarios=scenarios,
        grid_map=grid_map,
        agent_counts=(3,),
        instances_per_level=1,
    )


def test_held_out_instance_ids_use_ho_prefix() -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    primary = _primary_manifest(grid_map, scenarios)
    held_out = generate_held_out_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name="bench.map.scen",
        primary_manifest=primary,
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=DEFAULT_HELD_OUT_SEED,
        min_reference_length=20.0,
        max_attempts=5000,
    ).manifest

    for instance in held_out.instances:
        assert "_HO_" in instance.instance_id
        assert instance.instance_id.startswith("bench_HO_")


def test_held_out_seed_must_differ_from_primary() -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    primary = _primary_manifest(grid_map, scenarios, seed=2026)

    with pytest.raises(ValueError, match="Held-out seed must differ"):
        generate_held_out_benchmark_manifest(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_name="bench.map.scen",
            primary_manifest=primary,
            agent_counts=(3,),
            instances_per_level=1,
            max_timestep=64,
            seed=2026,
            min_reference_length=20.0,
            max_attempts=5000,
        )


def test_held_out_deterministic_regeneration() -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    primary = _primary_manifest(grid_map, scenarios)

    first = generate_held_out_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name="bench.map.scen",
        primary_manifest=primary,
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=DEFAULT_HELD_OUT_SEED,
        min_reference_length=20.0,
        max_attempts=5000,
    )
    second = generate_held_out_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name="bench.map.scen",
        primary_manifest=primary,
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=DEFAULT_HELD_OUT_SEED,
        min_reference_length=20.0,
        max_attempts=5000,
        source_pool=first.source_pool,
    )
    assert first.manifest.instances == second.manifest.instances


def test_held_out_manifest_json_round_trip(tmp_path: Path) -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    primary = _primary_manifest(grid_map, scenarios)
    held_out = generate_held_out_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name="bench.map.scen",
        primary_manifest=primary,
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=DEFAULT_HELD_OUT_SEED,
        min_reference_length=20.0,
        max_attempts=5000,
        primary_manifest_reference="AR0204SR_manifest.json",
    ).manifest

    output_path = tmp_path / "heldout_manifest.json"
    save_benchmark_manifest(held_out, output_path)
    loaded = load_benchmark_manifest(output_path)

    assert loaded == held_out
    assert loaded.catalogue_role == CATALOGUE_ROLE_HELD_OUT
    assert loaded.seed == DEFAULT_HELD_OUT_SEED

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["catalogue_role"] == CATALOGUE_ROLE_HELD_OUT
    assert payload["generation_version"] == "MAPF-7.6"


def test_validate_held_out_rejects_primary_duplicate_signature() -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    primary = _primary_manifest(grid_map, scenarios)
    duplicate = primary.instances[0]

    held_out_full = generate_held_out_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name="bench.map.scen",
        primary_manifest=primary,
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=DEFAULT_HELD_OUT_SEED,
        min_reference_length=20.0,
        max_attempts=5000,
    ).manifest

    bad_instances = list(held_out_full.instances)
    bad_instances[0] = MAPFBenchmarkInstance(
        instance_id=bad_instances[0].instance_id,
        agent_count=duplicate.agent_count,
        interaction_level=duplicate.interaction_level,
        scenario_indices=duplicate.scenario_indices,
        independent_conflict_count=duplicate.independent_conflict_count,
        conflicting_agent_pair_count=duplicate.conflicting_agent_pair_count,
        independent_soc=duplicate.independent_soc,
        independent_makespan=duplicate.independent_makespan,
        vertex_conflict_count=duplicate.vertex_conflict_count,
        edge_conflict_count=duplicate.edge_conflict_count,
    )
    bad_manifest = MAPFBenchmarkManifest(
        map_name=held_out_full.map_name,
        scenario_name=held_out_full.scenario_name,
        seed=held_out_full.seed,
        max_timestep=held_out_full.max_timestep,
        min_reference_length=held_out_full.min_reference_length,
        instances=tuple(bad_instances),
        catalogue_role=CATALOGUE_ROLE_HELD_OUT,
    )

    with pytest.raises(ValueError, match="duplicates primary"):
        validate_held_out_catalogue(
            bad_manifest,
            primary,
            scenarios=scenarios,
            grid_map=grid_map,
            agent_counts=(3,),
            instances_per_level=1,
        )


def test_frozen_ar0204sr_heldout_manifest_structure() -> None:
    repo_root = Path("pathfinding/results/mapf_benchmarks")
    primary_path = repo_root / "AR0204SR_manifest.json"
    held_out_path = repo_root / "AR0204SR_heldout_manifest.json"
    if not primary_path.is_file() or not held_out_path.is_file():
        pytest.skip("Frozen AR0204SR held-out manifest not generated yet")

    primary = load_benchmark_manifest(primary_path)
    held_out = load_benchmark_manifest(held_out_path)

    assert held_out.catalogue_role == CATALOGUE_ROLE_HELD_OUT
    assert held_out.seed == DEFAULT_HELD_OUT_SEED
    assert held_out.seed != primary.seed
    assert len(held_out.instances) == 27

    validate_held_out_catalogue(
        held_out,
        primary,
        agent_counts=(5, 10, 20),
        instances_per_level=3,
    )

    primary_sigs = collect_scenario_set_signatures_by_agent_count(primary.instances)
    held_out_sigs = collect_scenario_set_signatures_by_agent_count(held_out.instances)
    for agent_count in (5, 10, 20):
        assert primary_sigs[agent_count].isdisjoint(held_out_sigs[agent_count])

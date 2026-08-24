from __future__ import annotations

from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    DEFAULT_AGENT_COUNTS,
    DEFAULT_INSTANCES_PER_LEVEL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TIMESTEP,
    DEFAULT_MIN_REFERENCE_LENGTH,
    MAPF8_CROSS_MAP_SEEDS,
    MAPF8_CROSS_MAP_TARGETS,
    MAPFBenchmarkCatalogueConfig,
    PROTECTED_PRODUCTION_MANIFEST_NAMES,
    assert_safe_manifest_output,
    mapf8_cross_map_catalogue_config,
    resolve_mapf8_cross_map_target,
    run_benchmark_catalogue_generation,
    validate_manifest_content,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFInteractionLevel,
    generate_benchmark_manifest,
)
from pathfinding.tests.unit.test_mapf_benchmark_instances import (
    _build_rich_scenario_pool,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_mapf8_cross_map_targets_and_seeds() -> None:
    assert MAPF8_CROSS_MAP_TARGETS == frozenset({"AR0400SR", "AR0307SR"})
    assert MAPF8_CROSS_MAP_SEEDS["AR0400SR"] == 2028
    assert MAPF8_CROSS_MAP_SEEDS["AR0307SR"] == 2029


def test_ar0404sr_is_rejected_as_cross_map_target() -> None:
    with pytest.raises(ValueError, match="AR0404SR is not a valid MAPF-8"):
        resolve_mapf8_cross_map_target("AR0404SR")


def test_unknown_cross_map_target_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Unsupported MAPF-8 cross-map target"):
        resolve_mapf8_cross_map_target("AR0204SR")


def test_frozen_cross_map_configuration_values() -> None:
    repo_root = _repo_root()
    config_400 = mapf8_cross_map_catalogue_config("AR0400SR", repo_root)
    config_307 = mapf8_cross_map_catalogue_config("AR0307SR", repo_root)

    assert config_400.map_stem == "AR0400SR"
    assert config_400.seed == 2028
    assert config_400.map_path == repo_root / "Data" / "bg512-map" / "AR0400SR.map"
    assert config_400.scen_path == repo_root / "Data" / "bg512-scen" / "AR0400SR.map.scen"
    assert config_400.output_manifest_path == (
        repo_root / "pathfinding" / "results" / "mapf_benchmarks" / "AR0400SR_manifest.json"
    )

    assert config_307.map_stem == "AR0307SR"
    assert config_307.seed == 2029
    assert config_307.output_manifest_path.name == "AR0307SR_manifest.json"

    for config in (config_400, config_307):
        assert config.max_timestep == DEFAULT_MAX_TIMESTEP == 512
        assert config.min_reference_length == DEFAULT_MIN_REFERENCE_LENGTH == 20.0
        assert config.max_attempts == DEFAULT_MAX_ATTEMPTS == 5000
        assert config.agent_counts == DEFAULT_AGENT_COUNTS == (5, 10, 20)
        assert config.instances_per_level == DEFAULT_INSTANCES_PER_LEVEL == 3


def test_protected_manifest_paths_include_cross_map_and_historical() -> None:
    assert "AR0204SR_manifest.json" in PROTECTED_PRODUCTION_MANIFEST_NAMES
    assert "AR0204SR_heldout_manifest.json" in PROTECTED_PRODUCTION_MANIFEST_NAMES
    assert "AR0400SR_manifest.json" in PROTECTED_PRODUCTION_MANIFEST_NAMES
    assert "AR0307SR_manifest.json" in PROTECTED_PRODUCTION_MANIFEST_NAMES


def test_protected_manifest_write_blocked_by_default(tmp_path: Path) -> None:
    output_path = tmp_path / "AR0400SR_manifest.json"
    with pytest.raises(RuntimeError, match="Refusing to write protected production manifest"):
        assert_safe_manifest_output(
            output_path,
            allow_production_manifest_write=False,
        )


def test_protected_manifest_cannot_be_silently_overwritten(tmp_path: Path) -> None:
    output_path = tmp_path / "AR0307SR_manifest.json"
    output_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite existing protected manifest"):
        assert_safe_manifest_output(
            output_path,
            allow_production_manifest_write=True,
        )


def test_map_identity_propagates_to_instance_ids(tmp_path: Path) -> None:
    grid_map_a, scenarios_a = _build_rich_scenario_pool(size=9)
    grid_map_a = grid_map_a.model_copy(update={"name": "AR0400SR.map"})
    grid_map_b, scenarios_b = _build_rich_scenario_pool(size=9)
    grid_map_b = grid_map_b.model_copy(update={"name": "AR0307SR.map"})

    manifest_a = generate_benchmark_manifest(
        grid_map=grid_map_a,
        scenarios=scenarios_a,
        scenario_name="AR0400SR.map.scen",
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=2028,
        min_reference_length=20.0,
        max_attempts=5000,
    ).manifest
    manifest_b = generate_benchmark_manifest(
        grid_map=grid_map_b,
        scenarios=scenarios_b,
        scenario_name="AR0307SR.map.scen",
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=2029,
        min_reference_length=20.0,
        max_attempts=5000,
    ).manifest

    for instance in manifest_a.instances:
        assert instance.instance_id.startswith("AR0400SR_")
        assert "AR0307SR" not in instance.instance_id
        assert "_HO_" not in instance.instance_id

    for instance in manifest_b.instances:
        assert instance.instance_id.startswith("AR0307SR_")
        assert "AR0400SR" not in instance.instance_id
        assert "_HO_" not in instance.instance_id


def test_deterministic_generation_with_same_configuration(tmp_path: Path) -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    grid_map = grid_map.model_copy(update={"name": "AR0400SR.map"})

    kwargs = {
        "grid_map": grid_map,
        "scenarios": scenarios,
        "scenario_name": "AR0400SR.map.scen",
        "agent_counts": (3,),
        "instances_per_level": 1,
        "max_timestep": 64,
        "seed": 2028,
        "min_reference_length": 20.0,
        "max_attempts": 5000,
    }
    first = generate_benchmark_manifest(**kwargs).manifest
    second = generate_benchmark_manifest(**kwargs).manifest

    assert first.instances == second.instances
    assert first.seed == 2028
    assert {instance.instance_id for instance in first.instances} == {
        instance.instance_id for instance in second.instances
    }


def test_different_seeds_produce_independent_catalogues(tmp_path: Path) -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    grid_map = grid_map.model_copy(update={"name": "cross.map"})

    common = {
        "grid_map": grid_map,
        "scenarios": scenarios,
        "scenario_name": "cross.map.scen",
        "agent_counts": (3,),
        "instances_per_level": 1,
        "max_timestep": 64,
        "min_reference_length": 20.0,
        "max_attempts": 5000,
    }
    manifest_2028 = generate_benchmark_manifest(seed=2028, **common).manifest
    manifest_2029 = generate_benchmark_manifest(seed=2029, **common).manifest

    assert manifest_2028.instances != manifest_2029.instances


def test_smoke_generation_does_not_write_production_manifest(tmp_path: Path) -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    grid_map = grid_map.model_copy(update={"name": "AR0400SR.map"})
    output_path = tmp_path / "AR0400SR_manifest.json"
    log_path = tmp_path / "smoke.log"

    config = MAPFBenchmarkCatalogueConfig(
        map_stem="AR0400SR",
        map_path=tmp_path / "unused.map",
        scen_path=tmp_path / "unused.scen",
        output_manifest_path=output_path,
        log_file_path=log_path,
        seed=2028,
        max_timestep=64,
        min_reference_length=20.0,
        max_attempts=5000,
        agent_counts=(3,),
        instances_per_level=1,
        skip_validation=True,
        allow_production_manifest_write=False,
    )

    # Bind synthetic map/scenario data without touching Data/ paths.
    import pathfinding.src.experiments.mapf_benchmark_catalogue_generation as generation_module

    original_load_map = generation_module.load_moving_ai_map
    original_load_scen = generation_module.load_moving_ai_scenarios
    generation_module.load_moving_ai_map = lambda _path: grid_map
    generation_module.load_moving_ai_scenarios = lambda _path: scenarios
    try:
        result = run_benchmark_catalogue_generation(config)
    finally:
        generation_module.load_moving_ai_map = original_load_map
        generation_module.load_moving_ai_scenarios = original_load_scen

    assert result.manifest_written is False
    assert not output_path.exists()
    assert len(result.manifest.instances) == len(MAPFInteractionLevel)
    validate_manifest_content(
        result.manifest,
        grid_map=grid_map,
        scenarios=scenarios,
        source_pool=result.source_pool,
        agent_counts=(3,),
        instances_per_level=1,
        expected_map_stem="AR0400SR",
    )


def test_explicit_seed_preserved_in_manifest(tmp_path: Path) -> None:
    grid_map, scenarios = _build_rich_scenario_pool(size=9)
    manifest = generate_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name="bench.map.scen",
        agent_counts=(3,),
        instances_per_level=1,
        max_timestep=64,
        seed=2029,
        min_reference_length=20.0,
        max_attempts=5000,
    ).manifest
    assert manifest.seed == 2029

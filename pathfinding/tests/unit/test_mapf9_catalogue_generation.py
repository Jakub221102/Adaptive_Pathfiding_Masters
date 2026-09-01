"""Tests for MAPF-9 catalogue generation and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    DEFAULT_AGENT_COUNTS,
    DEFAULT_INSTANCES_PER_LEVEL,
    MAPF8_CROSS_MAP_SEEDS,
    PROTECTED_PRODUCTION_MANIFEST_NAMES,
    assert_mapf9_manifest_output,
    mapf9_cross_map_catalogue_config,
    mapf9_manifest_path,
    resolve_mapf9_cross_map_target,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_MAPF9,
    MAPF9_CATALOGUE_SEEDS,
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    benchmark_scenario_set_signature,
    collect_all_scenario_set_signatures,
    load_benchmark_manifest,
    mapf8_mapf9_signature_overlap,
    save_benchmark_manifest,
    validate_mapf8_mapf9_signature_disjointness,
)
from pathfinding.src.experiments.mapf9_catalogue_validation import (
    MAPF9CatalogueValidationConfig,
    manifest_file_sha256,
    mapf9_catalogue_validation_config,
    validate_mapf9_manifest_identity,
    validate_mapf9_production_manifest,
    write_mapf9_freeze_report,
)
from pathfinding.tests.unit.test_mapf_cross_map_catalogue_validation import (
    _synthetic_27_instance_manifest,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mapf9_manifest_from_mapf8_base(
    mapf8_manifest: MAPFBenchmarkManifest,
    *,
    seed: int,
    signature_offset: int = 10000,
) -> MAPFBenchmarkManifest:
    instances: list[MAPFBenchmarkInstance] = []
    for instance in mapf8_manifest.instances:
        shifted_indices = tuple(index + signature_offset for index in instance.scenario_indices)
        instances.append(
            replace(
                instance,
                scenario_indices=shifted_indices,
                independent_soc=instance.independent_soc + 1,
            )
        )
    return replace(
        mapf8_manifest,
        seed=seed,
        instances=tuple(instances),
        catalogue_role=CATALOGUE_ROLE_MAPF9,
        generation_version="MAPF-9.3",
        primary_manifest_reference="AR0400SR_manifest.json",
    )


def test_mapf9_seeds_mapped_to_correct_maps() -> None:
    assert MAPF9_CATALOGUE_SEEDS["AR0400SR"] == 2030
    assert MAPF9_CATALOGUE_SEEDS["AR0307SR"] == 2031
    assert MAPF8_CROSS_MAP_SEEDS["AR0400SR"] == 2028
    assert MAPF8_CROSS_MAP_SEEDS["AR0307SR"] == 2029


def test_mapf9_config_uses_isolated_manifest_paths() -> None:
    repo_root = _repo_root()
    config_400 = mapf9_cross_map_catalogue_config("AR0400SR", repo_root)
    config_307 = mapf9_cross_map_catalogue_config("AR0307SR", repo_root)

    assert config_400.seed == 2030
    assert config_307.seed == 2031
    assert config_400.output_manifest_path.name == "AR0400SR_mapf9_manifest.json"
    assert config_307.output_manifest_path.name == "AR0307SR_mapf9_manifest.json"
    assert config_400.output_manifest_path.name not in PROTECTED_PRODUCTION_MANIFEST_NAMES
    assert config_400.mapf8_reference_manifest_path.name == "AR0400SR_manifest.json"
    assert config_400.catalogue_mode == "mapf9"


def test_mapf9_manifest_output_rejects_mapf8_path(tmp_path: Path) -> None:
    path = tmp_path / "AR0400SR_manifest.json"
    with pytest.raises(RuntimeError, match="Refusing to write MAPF-9 catalogue"):
        assert_mapf9_manifest_output(path, allow_production_manifest_write=True)


def test_mapf9_manifest_output_requires_mapf9_suffix(tmp_path: Path) -> None:
    path = tmp_path / "AR0400SR_wrong_name.json"
    with pytest.raises(RuntimeError, match="_mapf9_manifest.json"):
        assert_mapf9_manifest_output(path, allow_production_manifest_write=True)


def test_validation_requires_exactly_27_instances(tmp_path: Path) -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    mapf9 = _mapf9_manifest_from_mapf8_base(mapf8, seed=2030)
    truncated = replace(mapf9, instances=mapf9.instances[:26])
    config = mapf9_catalogue_validation_config("AR0400SR", tmp_path)
    save_benchmark_manifest(mapf8, config.mapf8_reference_manifest_path)
    save_benchmark_manifest(truncated, config.manifest_path)

    result = validate_mapf9_production_manifest(
        replace(config, recompute_source_pool=False),
        manifest=truncated,
    )
    assert result.passed is False


def test_validation_requires_exactly_three_per_cell(tmp_path: Path) -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    mapf9 = _mapf9_manifest_from_mapf8_base(mapf8, seed=2030)
    medium_n5_index = next(
        index
        for index, instance in enumerate(mapf9.instances)
        if instance.agent_count == 5
        and instance.interaction_level == MAPFInteractionLevel.MEDIUM
    )
    broken_instances = list(mapf9.instances)
    broken_instances[medium_n5_index] = replace(
        broken_instances[medium_n5_index],
        interaction_level=MAPFInteractionLevel.LOW,
        independent_conflict_count=0,
    )
    broken = replace(mapf9, instances=tuple(broken_instances))
    config = mapf9_catalogue_validation_config("AR0400SR", tmp_path)
    save_benchmark_manifest(mapf8, config.mapf8_reference_manifest_path)
    save_benchmark_manifest(broken, config.manifest_path)

    result = validate_mapf9_production_manifest(
        replace(config, recompute_source_pool=False),
        manifest=broken,
    )
    assert result.passed is False


def test_duplicate_scenario_set_signature_rejected(tmp_path: Path) -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    mapf9 = _mapf9_manifest_from_mapf8_base(mapf8, seed=2030)
    duplicate = replace(
        mapf9.instances[1],
        scenario_indices=mapf9.instances[0].scenario_indices,
        agent_count=mapf9.instances[0].agent_count,
    )
    broken = replace(
        mapf9,
        instances=(mapf9.instances[0], duplicate, *mapf9.instances[2:]),
    )
    config = mapf9_catalogue_validation_config("AR0400SR", tmp_path)
    save_benchmark_manifest(mapf8, config.mapf8_reference_manifest_path)

    result = validate_mapf9_production_manifest(
        replace(config, recompute_source_pool=False),
        manifest=broken,
    )
    assert result.passed is False


def test_mapf8_cross_catalogue_overlap_rejected() -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    mapf9 = replace(
        mapf8,
        seed=2030,
        catalogue_role=CATALOGUE_ROLE_MAPF9,
    )
    with pytest.raises(ValueError, match="shares .* scenario-set signature"):
        validate_mapf8_mapf9_signature_disjointness(mapf9, mapf8)


def test_zero_overlap_catalogue_accepted() -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    mapf9 = _mapf9_manifest_from_mapf8_base(mapf8, seed=2030)
    validate_mapf8_mapf9_signature_disjointness(mapf9, mapf8)
    mapf9_sigs, mapf8_sigs, overlap = mapf8_mapf9_signature_overlap(mapf9, mapf8)
    assert len(mapf9_sigs) == 27
    assert len(mapf8_sigs) == 27
    assert len(overlap) == 0


def test_overlap_uses_complete_scenario_set_signature() -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    shared_signature = benchmark_scenario_set_signature(mapf8.instances[0].scenario_indices)
    different_id = replace(
        mapf8.instances[0],
        instance_id="AR0400SR_n05_low_999",
    )
    mapf9 = replace(
        mapf8,
        seed=2030,
        instances=(different_id, *mapf8.instances[1:]),
        catalogue_role=CATALOGUE_ROLE_MAPF9,
    )
    assert (
        benchmark_scenario_set_signature(mapf9.instances[0].scenario_indices)
        == shared_signature
    )
    with pytest.raises(ValueError, match="scenario-set signature"):
        validate_mapf8_mapf9_signature_disjointness(mapf9, mapf8)


def test_different_instance_ids_same_scenario_set_still_rejected() -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    renamed = replace(
        mapf8.instances[0],
        instance_id="AR0400SR_n05_low_999",
    )
    mapf9 = replace(
        mapf8,
        seed=2030,
        instances=(renamed, *mapf8.instances[1:]),
        catalogue_role=CATALOGUE_ROLE_MAPF9,
    )
    with pytest.raises(ValueError, match="scenario-set signature"):
        validate_mapf8_mapf9_signature_disjointness(mapf9, mapf8)


def test_manifest_hash_deterministic(tmp_path: Path) -> None:
    manifest = _mapf9_manifest_from_mapf8_base(
        _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028),
        seed=2030,
    )
    path = tmp_path / "AR0400SR_mapf9_manifest.json"
    save_benchmark_manifest(manifest, path)
    first = manifest_file_sha256(path)
    second = manifest_file_sha256(path)
    assert first == second
    assert len(first) == 64


def test_wrong_seed_rejected(tmp_path: Path) -> None:
    manifest = _mapf9_manifest_from_mapf8_base(
        _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028),
        seed=9999,
    )
    config = mapf9_catalogue_validation_config("AR0400SR", tmp_path)
    with pytest.raises(ValueError, match="Expected MAPF-9 seed 2030"):
        validate_mapf9_manifest_identity(manifest, config)


def test_wrong_map_rejected(tmp_path: Path) -> None:
    manifest = _mapf9_manifest_from_mapf8_base(
        _synthetic_27_instance_manifest(map_stem="AR0307SR", seed=2029),
        seed=2031,
    )
    config = mapf9_catalogue_validation_config("AR0400SR", tmp_path)
    with pytest.raises(ValueError, match="Expected map stem 'AR0400SR'"):
        validate_mapf9_manifest_identity(manifest, config)


def test_mapf9_validation_accepts_disjoint_synthetic_catalogue(tmp_path: Path) -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    mapf9 = _mapf9_manifest_from_mapf8_base(mapf8, seed=2030)
    config = MAPF9CatalogueValidationConfig(
        map_stem="AR0400SR",
        manifest_path=tmp_path / "AR0400SR_mapf9_manifest.json",
        mapf8_reference_manifest_path=tmp_path / "AR0400SR_manifest.json",
        map_path=tmp_path / "AR0400SR.map",
        scen_path=tmp_path / "AR0400SR.map.scen",
        report_path=tmp_path / "AR0400SR_mapf9_validation.md",
        log_file_path=tmp_path / "validation.log",
        expected_seed=2030,
        expected_mapf8_seed=2028,
        recompute_source_pool=False,
    )
    save_benchmark_manifest(mapf8, config.mapf8_reference_manifest_path)
    save_benchmark_manifest(mapf9, config.manifest_path)

    result = validate_mapf9_production_manifest(config, manifest=mapf9)
    assert result.passed
    assert result.instance_count == 27
    assert result.signature_overlap_count == 0
    report_path = write_mapf9_freeze_report(config, result)
    text = report_path.read_text(encoding="utf-8")
    assert "signature intersection count: **0**" in text
    assert result.sha256 in text


def test_mapf9_manifest_path_helper(tmp_path: Path) -> None:
    path = mapf9_manifest_path(tmp_path, "AR0400SR")
    assert path.name == "AR0400SR_mapf9_manifest.json"


def test_ar0404sr_rejected_for_mapf9() -> None:
    with pytest.raises(ValueError, match="AR0404SR is not a valid MAPF-9"):
        resolve_mapf9_cross_map_target("AR0404SR")


def test_historical_mapf8_manifest_bytes_unchanged() -> None:
    repo_root = _repo_root()
    mapf8_path = repo_root / "pathfinding/results/mapf_benchmarks/AR0400SR_manifest.json"
    if not mapf8_path.is_file():
        pytest.skip("Frozen MAPF-8 manifest not present")
    expected = "ab652d77ad99c251d7827684b7b2978a3bd5151609c5b557840eec8522a54259"
    actual = hashlib.sha256(mapf8_path.read_bytes()).hexdigest()
    assert actual == expected
    payload = json.loads(mapf8_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 2028
    assert len(payload["instances"]) == 27

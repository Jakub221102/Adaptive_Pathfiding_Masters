from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    assert_production_generation_ready,
    assert_safe_manifest_output,
    mapf8_cross_map_catalogue_config,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    save_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
    CrossMapCatalogueValidationConfig,
    manifest_file_sha256,
    mapf8_cross_map_validation_config,
    validate_cross_map_production_manifest,
    validate_instance_id_prefixes,
    validate_manifest_identity,
    validate_manifest_structure,
    write_freeze_report,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _synthetic_27_instance_manifest(
    *,
    map_stem: str = "AR0400SR",
    seed: int = 2028,
) -> MAPFBenchmarkManifest:
    instances: list[MAPFBenchmarkInstance] = []
    conflict_by_level = {
        MAPFInteractionLevel.LOW: 0,
        MAPFInteractionLevel.MEDIUM: 1,
        MAPFInteractionLevel.HIGH: 3,
    }
    offset = 0
    for agent_count in (5, 10, 20):
        for level in MAPFInteractionLevel:
            for seq in range(3):
                conflict_count = conflict_by_level[level]
                scenario_indices = tuple(range(offset, offset + agent_count))
                offset += agent_count
                instances.append(
                    MAPFBenchmarkInstance(
                        instance_id=f"{map_stem}_n{agent_count:02d}_{level.value}_{seq:03d}",
                        agent_count=agent_count,
                        interaction_level=level,
                        scenario_indices=scenario_indices,
                        independent_conflict_count=conflict_count,
                        conflicting_agent_pair_count=min(conflict_count, 1),
                        independent_soc=100 + agent_count + seq,
                        independent_makespan=50 + agent_count,
                        vertex_conflict_count=conflict_count,
                        edge_conflict_count=0,
                    )
                )
    return MAPFBenchmarkManifest(
        map_name=f"{map_stem}.map",
        scenario_name=f"{map_stem}.map.scen",
        seed=seed,
        max_timestep=512,
        min_reference_length=20.0,
        instances=tuple(instances),
    )


def _validation_config(tmp_path: Path, map_stem: str = "AR0400SR") -> CrossMapCatalogueValidationConfig:
    seed = 2028 if map_stem == "AR0400SR" else 2029
    return CrossMapCatalogueValidationConfig(
        map_stem=map_stem,
        manifest_path=tmp_path / f"{map_stem}_manifest.json",
        map_path=tmp_path / f"{map_stem}.map",
        scen_path=tmp_path / f"{map_stem}.map.scen",
        report_path=tmp_path / f"{map_stem}_validation.md",
        log_file_path=tmp_path / f"{map_stem}_validation.log",
        expected_seed=seed,
        recompute_source_pool=False,
    )


def test_validation_accepts_correct_synthetic_27_instance_catalogue(tmp_path: Path) -> None:
    manifest = _synthetic_27_instance_manifest()
    config = _validation_config(tmp_path, "AR0400SR")
    save_benchmark_manifest(manifest, config.manifest_path)

    result = validate_cross_map_production_manifest(config, manifest=manifest)
    assert result.passed
    assert result.instance_count == 27
    assert result.counts_by_interaction_level == {"low": 9, "medium": 9, "high": 9}


def test_validation_rejects_wrong_seed(tmp_path: Path) -> None:
    manifest = replace(_synthetic_27_instance_manifest(), seed=9999)
    config = _validation_config(tmp_path, "AR0400SR")

    with pytest.raises(ValueError, match="Expected seed 2028"):
        validate_manifest_identity(manifest, config)


def test_validation_rejects_wrong_map_identity(tmp_path: Path) -> None:
    manifest = _synthetic_27_instance_manifest(map_stem="AR0307SR", seed=2029)
    config = _validation_config(tmp_path, "AR0400SR")

    with pytest.raises(ValueError, match="Expected map stem 'AR0400SR'"):
        validate_manifest_identity(manifest, config)


def test_validation_rejects_wrong_instance_count(tmp_path: Path) -> None:
    manifest = _synthetic_27_instance_manifest()
    truncated = replace(manifest, instances=manifest.instances[:26])

    with pytest.raises(ValueError, match="Expected 27 instances"):
        validate_manifest_structure(
            truncated,
            agent_counts=(5, 10, 20),
            instances_per_level=3,
        )


def test_validation_rejects_missing_3x3_cell() -> None:
    manifest = _synthetic_27_instance_manifest()
    medium_n5_index = next(
        index
        for index, instance in enumerate(manifest.instances)
        if instance.agent_count == 5
        and instance.interaction_level == MAPFInteractionLevel.MEDIUM
    )
    instances = list(manifest.instances)
    instances[medium_n5_index] = replace(
        instances[medium_n5_index],
        interaction_level=MAPFInteractionLevel.LOW,
        independent_conflict_count=0,
    )
    broken = replace(manifest, instances=tuple(instances))

    with pytest.raises(ValueError, match="Expected 3 instances for agent_count=5"):
        validate_manifest_structure(
            broken,
            agent_counts=(5, 10, 20),
            instances_per_level=3,
        )


def test_validation_rejects_wrong_interaction_classification(tmp_path: Path) -> None:
    manifest = _synthetic_27_instance_manifest()
    bad_instance = replace(
        manifest.instances[0],
        interaction_level=MAPFInteractionLevel.HIGH,
        independent_conflict_count=0,
    )
    broken = replace(
        manifest,
        instances=(bad_instance, *manifest.instances[1:]),
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_manifest_structure(
            broken,
            agent_counts=(5, 10, 20),
            instances_per_level=3,
        )


def test_validation_rejects_duplicate_scenario_signature(tmp_path: Path) -> None:
    manifest = _synthetic_27_instance_manifest()
    duplicate = replace(
        manifest.instances[1],
        scenario_indices=manifest.instances[0].scenario_indices,
        agent_count=manifest.instances[0].agent_count,
    )
    broken = replace(
        manifest,
        instances=(manifest.instances[0], duplicate, *manifest.instances[2:]),
    )

    with pytest.raises(ValueError, match="Duplicate scenario-set signatures"):
        validate_manifest_structure(
            broken,
            agent_counts=(5, 10, 20),
            instances_per_level=3,
        )


def test_validation_rejects_ar0404sr_instance_id_leakage() -> None:
    manifest = _synthetic_27_instance_manifest()
    bad = replace(
        manifest.instances[0],
        instance_id="AR0404SR_n05_low_000",
    )
    broken = replace(manifest, instances=(bad, *manifest.instances[1:]))

    with pytest.raises(ValueError, match="does not begin with 'AR0400SR_'"):
        validate_instance_id_prefixes(broken, map_stem="AR0400SR")


def test_manifest_sha256_is_deterministic(tmp_path: Path) -> None:
    manifest = _synthetic_27_instance_manifest()
    path = tmp_path / "manifest.json"
    save_benchmark_manifest(manifest, path)
    assert manifest_file_sha256(path) == manifest_file_sha256(path)
    assert len(manifest_file_sha256(path)) == 64


def test_freeze_report_uses_correct_map_identity(tmp_path: Path) -> None:
    manifest = _synthetic_27_instance_manifest(map_stem="AR0307SR", seed=2029)
    config = _validation_config(tmp_path, "AR0307SR")
    save_benchmark_manifest(manifest, config.manifest_path)
    result = validate_cross_map_production_manifest(config, manifest=manifest)
    report_path = write_freeze_report(config, result)
    text = report_path.read_text(encoding="utf-8")
    assert "AR0307SR" in text
    assert "2029" in text
    assert result.sha256 in text
    assert "AR0404SR" not in text


def test_production_manifest_overwrite_blocked(tmp_path: Path) -> None:
    path = tmp_path / "AR0400SR_manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite existing protected manifest"):
        assert_safe_manifest_output(path, allow_production_manifest_write=True)


def test_historical_ar0204sr_manifest_write_blocked(tmp_path: Path) -> None:
    path = tmp_path / "AR0204SR_manifest.json"
    with pytest.raises(RuntimeError, match="Refusing to write protected production manifest"):
        assert_safe_manifest_output(path, allow_production_manifest_write=False)


def test_production_ready_fails_when_map_files_missing(tmp_path: Path) -> None:
    config = mapf8_cross_map_catalogue_config("AR0400SR", _repo_root())
    config = replace(
        config,
        map_path=tmp_path / "missing.map",
        scen_path=tmp_path / "missing.scen",
        allow_production_manifest_write=True,
    )
    with pytest.raises(FileNotFoundError):
        assert_production_generation_ready(config)


def test_mapf8_validation_config_seeds() -> None:
    cfg400 = mapf8_cross_map_validation_config("AR0400SR", _repo_root())
    cfg307 = mapf8_cross_map_validation_config("AR0307SR", _repo_root())
    assert cfg400.expected_seed == 2028
    assert cfg307.expected_seed == 2029

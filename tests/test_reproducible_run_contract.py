from pathlib import Path

import pytest

from src.trustworthy_credit.reproducible_run import (
    ReproducibleRunError,
    ReproducibleRunPaths,
    timestamp_run_id,
)


def test_reproducible_run_paths_stay_under_results_runs(tmp_path):
    paths = ReproducibleRunPaths(project_root=tmp_path, run_id="smoke-run")

    assert paths.run_dir == tmp_path / "results" / "runs" / "smoke-run"
    assert paths.tables_dir == paths.run_dir / "tables"
    assert paths.figures_dir == paths.run_dir / "figures"
    assert paths.models_dir == paths.run_dir / "models"
    assert paths.tuner_dir == paths.run_dir / "kt_dir"


def test_artifact_configs_do_not_point_to_canonical_results(tmp_path):
    paths = ReproducibleRunPaths(project_root=tmp_path, run_id="isolated")

    tuning = paths.tuning_artifacts()
    uncertainty = paths.uncertainty_artifacts()

    assert tuning.tables_directory == paths.tables_dir
    assert tuning.models_directory == paths.models_dir
    assert tuning.tuner_directory == paths.tuner_dir
    assert uncertainty.tables_directory == paths.tables_dir
    assert uncertainty.models_directory == paths.models_dir

    assert tuning.tables_directory != tmp_path / "results" / "tables"
    assert tuning.models_directory != tmp_path / "results" / "models"
    assert tuning.tuner_directory != tmp_path / "kt_dir"


def test_run_id_is_sanitized_and_non_empty(tmp_path):
    paths = ReproducibleRunPaths(project_root=tmp_path, run_id=" run 01:demo ")

    assert paths.run_id == "run-01-demo"
    assert paths.run_dir == tmp_path / "results" / "runs" / "run-01-demo"


def test_timestamp_run_id_is_stable_for_injected_datetime():
    from datetime import datetime

    assert timestamp_run_id(datetime(2026, 6, 24, 21, 5, 7)) == "20260624_210507"


def test_empty_run_id_is_rejected(tmp_path):
    with pytest.raises(ReproducibleRunError):
        ReproducibleRunPaths(project_root=tmp_path, run_id="   ")

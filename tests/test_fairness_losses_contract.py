import pytest

pytest.importorskip("tensorflow")

from src.trustworthy_credit.fairness_losses import (
    FairnessLossExperimentError,
    SquaredFairnessComparisonRunner,
    SquaredFairnessResultInterpreter,
    SquaredFairnessTrainingConfig,
)
from src.trustworthy_credit.gbm_experiments import GBMArtifactPaths


def test_squared_fairness_config_requires_alpha_zero():
    with pytest.raises(FairnessLossExperimentError):
        SquaredFairnessTrainingConfig(alpha_values=(0.01, 0.1))


def test_squared_fairness_interpreter_marks_collapsed_result():
    config = SquaredFairnessTrainingConfig(alpha_values=(0.0, 0.1))
    status = SquaredFairnessResultInterpreter().classify(
        alpha=0.1,
        test_auc=0.5,
        threshold=1.0,
        prediction_std=0.0,
        config=config,
    )

    assert status == "colapsado"


def test_squared_fairness_paths_stay_inside_extras_run(tmp_path):
    paths = GBMArtifactPaths(project_root=tmp_path, run_id="fairness smoke")
    runner = SquaredFairnessComparisonRunner(
        config=SquaredFairnessTrainingConfig(alpha_values=(0.0, 0.1)),
        artifacts=paths,
    )

    experiment_paths = runner.experiment_paths()

    assert paths.run_dir == tmp_path / "results" / "extras" / "fairness-smoke"
    for path in (
        experiment_paths.sweep_csv,
        experiment_paths.comparison_csv,
        experiment_paths.split_report_csv,
        experiment_paths.histories_dir,
        experiment_paths.models_dir,
        experiment_paths.predictions_dir,
    ):
        assert path.resolve().is_relative_to(paths.run_dir.resolve())

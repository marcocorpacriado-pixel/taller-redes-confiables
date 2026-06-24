"""Structural checks for the unified main MVP notebook."""

import ast
from pathlib import Path
import unicodedata

import nbformat


NOTEBOOK_PATH = Path("notebooks/01_main_mvp.ipynb")


def _normalize_text(value: str) -> str:
    """Normalize notebook text so section checks are accent-insensitive."""

    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    ).lower()


def test_main_mvp_notebook_exists() -> None:
    """The unified notebook should exist without replacing the Dani notebook."""

    assert NOTEBOOK_PATH.exists()
    assert Path("notebooks/01_mvp_dani_professional.ipynb").exists()


def test_main_mvp_notebook_contains_expected_sections() -> None:
    """The notebook should preserve the full MVP execution narrative."""

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    markdown = _normalize_text(
        "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        )
    )

    expected_sections = [
        "# sistema neuronal confiable para concesion de credito",
        "## mapa del notebook",
        "## 0. configuracion global",
        "## 1. problema, datos y contrato",
        "## 2. carga de datos principales",
        "## 3. eda: desbalance de clase y variable sensible",
        "## 4. eda: valores ausentes y calidad de fuentes externas",
        "## 5. eda: variables financieras y fuentes externas",
        "## 6. preprocesamiento sin leakage",
        "## 7. auditoria del preprocesamiento",
        "## 8. arquitectura customizada",
        "## 9. progresion experimental m0-m6",
        "## 10. fair loss y objetivo de justicia",
        "## 11. automl con keras tuner",
        "## 12. barrido de `lambda_fair`",
        "## 13. pareto principal: rendimiento vs dependencia fair",
        "## 14. robustez de pareto: contraste multi-semilla",
        "## 15. seleccion de modelos finales",
        "## 16. curvas de convergencia",
        "## 17. evaluacion final en test",
        "## 18. comparativa visual en test",
        "## 19. incertidumbre principal: modelo m2",
        "## 20. contraste de incertidumbre: mc dropout",
        "## 21. distribucion de incertidumbre por clase predicha",
        "## 22. calidad de datos: `ext_null_count` e incertidumbre",
        "## 23. ablacion informacional: 12 vs 42 features",
        "## 24. conclusiones y limitaciones",
    ]

    for section in expected_sections:
        assert section in markdown


def test_main_mvp_notebook_markdown_is_final_facing() -> None:
    """The final notebook narrative should not describe personal integration history."""

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    markdown = _normalize_text(
        "\n".join(
            "".join(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "markdown"
        )
    )

    forbidden_terms = [
        "dani",
        "marco",
        "javi",
        "aportaciones",
        "notebook profesional",
        "pegotes",
    ]

    for term in forbidden_terms:
        assert term not in markdown


def test_main_mvp_notebook_has_real_execution_cells() -> None:
    """The notebook should orchestrate the real MVP, not only describe it."""

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook.cells
        if cell.get("cell_type") == "code"
    )

    required_snippets = [
        "HomeCreditMVPPreprocessingPipeline",
        "FairKerasTunerRunner",
        "FairLambdaSweepTrainer",
        "ProbabilityMetricCalculator",
        "UncertaintyMVPTrainer",
        "MCDropoutUncertaintyEstimator",
        "MultiSeedParetoArtifactReporter",
        "SquaredDemographicParityLoss",
        "ExtendedFeaturePreprocessingPipeline",
        "uncertainty_test.csv",
        "pareto_auc_vs_fairness.png",
        "training_curves_base_vs_fair.png",
        "test_results_base_vs_fair.csv",
    ]

    for snippet in required_snippets:
        assert snippet in code


def test_main_mvp_notebook_uses_unified_package_imports() -> None:
    """The notebook should import from trustworthy_credit, not Dani directly."""

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook.cells
        if cell.get("cell_type") == "code"
    )

    assert "src.trustworthy_credit" in code
    assert "src.dani_credit" not in code
    assert "MCDropoutUncertaintyEstimator" in code
    assert "MultiSeedParetoArtifactReporter" in code


def test_main_mvp_notebook_code_cells_are_valid_python() -> None:
    """All code cells should parse before the notebook is executed end to end."""

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ""))
        ast.parse(source, filename=f"{NOTEBOOK_PATH}::cell-{index}")


def test_main_mvp_notebook_uses_local_save_figure_contract() -> None:
    """Notebook cells should call save_figure with the local one-argument API."""

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook.cells
        if cell.get("cell_type") == "code"
    )

    assert "save_figure(fig," not in code

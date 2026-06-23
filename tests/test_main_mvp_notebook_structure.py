"""Structural checks for the unified main MVP notebook."""

from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path("notebooks/01_main_mvp.ipynb")


def test_main_mvp_notebook_exists() -> None:
    """The unified notebook should exist without replacing the Dani notebook."""

    assert NOTEBOOK_PATH.exists()
    assert Path("notebooks/01_mvp_dani_professional.ipynb").exists()


def test_main_mvp_notebook_contains_expected_sections() -> None:
    """The notebook should expose the agreed narrative structure."""

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    markdown = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook.cells
        if cell.get("cell_type") == "markdown"
    )

    expected_sections = [
        "## 0. Configuracion global",
        "## 1. Contrato de datos e inventario",
        "## 2. EDA inicial",
        "## 3. Preprocesamiento y split",
        "## 4. Arquitectura custom",
        "## 5. FAIR loss",
        "## 6. AutoML",
        "## 7. Pareto lambda_fair",
        "## 8. Evaluacion test base vs FAIR",
        "## 9. Incertidumbre M2",
        "## 10. Figuras obligatorias",
        "## 11. Conclusiones y limitaciones",
    ]

    for section in expected_sections:
        assert section in markdown


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

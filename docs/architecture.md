# Arquitectura unificada del MVP

## Objetivo

Este documento define la arquitectura objetivo del proyecto B4-T1 Redes
Neuronales Confiables. La rama de integracion no reemplaza el MVP validado de
Dani de forma inmediata: construye una capa canonica nueva y profesional donde
se iran incorporando, de forma controlada, las mejores aportaciones de Dani,
Marco y Javi.

El tag estable del MVP actual es:

```text
v1.0-mvp-defendible
```

Ese tag conserva una version defendible del proyecto antes de iniciar la
refactorizacion.

## Principio rector

La arquitectura unificada debe cumplir tres reglas:

1. No romper resultados ya validados.
2. No borrar codigo existente hasta tener un sustituto probado.
3. Convertir aportaciones personales en modulos comunes, testeables y
   reutilizables.

## Estructura objetivo

```text
taller-redes-confiables/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_mvp_dani_professional.ipynb
│   └── 01_main_mvp.ipynb
├── src/
│   ├── dani_credit/
│   └── trustworthy_credit/
│       ├── __init__.py
│       ├── config.py
│       ├── data_contract.py
│       ├── preprocessing.py
│       ├── splitting.py
│       ├── features.py
│       ├── layers.py
│       ├── models.py
│       ├── metrics.py
│       ├── tuning.py
│       ├── uncertainty.py
│       ├── uncertainty_mc.py
│       ├── evaluation.py
│       ├── reporting.py
│       ├── plots.py
│       └── artifacts.py
├── tests/
├── docs/
│   ├── architecture.md
│   ├── migration_plan.md
│   ├── audit_mvp.md
│   └── defense_notes.md
└── reports/
    └── presentation/
```

Durante la migracion, `src/dani_credit/` permanece intacto. El paquete
`src/trustworthy_credit/` se llenara gradualmente hasta que pueda convertirse
en la interfaz comun del proyecto.

## Responsabilidades por modulo

### `config.py`

Centralizara rutas, semillas, hiperparametros de entrenamiento, configuracion
FAIR, configuracion de AutoML e incertidumbre. Debe evitar que el notebook
contenga constantes dispersas.

### `data_contract.py`

Contendra el contrato del dataset Home Credit: columnas obligatorias, variable
objetivo, variable sensible, columnas financieras, columnas `EXT_SOURCE` y
validaciones de integridad.

Base canonica: `src/dani_credit/data_contract.py`.

### `preprocessing.py`

Contendra carga, limpieza determinista, imputacion, escalado y preservacion de
features semanticas crudas para auditoria, especialmente `EXT_NULL_COUNT`.

Base canonica: `src/dani_credit/preprocessing.py`.

### `splitting.py`

Gestionara el split train/validation/test con estratificacion y reporte de
tasas de `TARGET` y `CODE_GENDER`.

Base canonica: `src/dani_credit/splitting.py`.

### `features.py`

Contendra ingenieria de variables tabulares y, en una fase posterior, features
extendidas o relacionales. No debe introducir leakage temporal ni usar test para
seleccion de features.

Fuentes futuras: ideas de features extendidas de Marco y enriquecimiento
relacional del dataset Home Credit.

### `layers.py`

Agrupara capas custom de Keras: ratios financieros, transformaciones
entrenables y penalizacion FAIR.

Base canonica: `src/dani_credit/layers.py`.

Posibles aportaciones de Marco: adaptar `DebtRatioLayer` y `ExtSourceLayer`
solo si aportan interpretabilidad adicional sin sustituir la arquitectura
validada.

### `models.py`

Contendra builders POO para modelo base, modelo custom y modelo FAIR. El
notebook no debe definir arquitecturas largas en celdas.

Base canonica: `src/dani_credit/models.py`.

### `metrics.py`

Agrupara metricas predictivas y de justicia: ROC-AUC, PR-AUC, F1, recall,
threshold, correlacion absoluta con genero, DPD y EOD.

Base canonica: `src/dani_credit/metrics.py`.

### `tuning.py`

Gestionara Keras Tuner, busqueda de arquitectura y sweep de `lambda_fair`.

Base canonica: `src/dani_credit/tuning.py`.

### `uncertainty.py`

Contendra la incertidumbre principal del MVP: M1 -> M2, donde M2 aprende el
error absoluto esperado del clasificador. Esta es la via mas alineada con el
enunciado y con las indicaciones del profesor.

Base canonica: `src/dani_credit/uncertainty.py`.

### `uncertainty_mc.py`

Modulo futuro para MC Dropout como incertidumbre complementaria. No sustituye
M2; sirve para comparar enfoques y enriquecer la defensa.

Fuentes: `src/uncertainty.py` de Marco y el bloque de incertidumbre del
notebook de Javi.

### `evaluation.py`

Preparara evaluaciones finales base vs FAIR, validaciones defensivas y tablas
de resultados para notebook y presentacion.

### `reporting.py` y `plots.py`

Contendran visualizaciones y tablas. Deben convertir graficas utiles de Marco y
Javi en componentes comunes, sin recalcular ni reemplazar resultados validados.

Fuentes: `src/visualization.py` de Marco y narrativa visual del notebook de
Javi.

### `artifacts.py`

Centralizara rutas y escritura de artefactos: tablas, figuras, modelos y
metadatos. Debe reducir la posibilidad de mezclar resultados antiguos con
resultados actuales.

## Notebook maestro

El futuro `notebooks/01_main_mvp.ipynb` debe ser narrativo. Su papel sera
orquestar clases del paquete unificado, no definir funciones largas.

Orden objetivo:

1. Configuracion y reproducibilidad.
2. Contrato de datos e inventario.
3. EDA minima.
4. Preprocesamiento y split.
5. Arquitectura custom.
6. FAIR loss.
7. AutoML.
8. Pareto de `lambda_fair`.
9. Evaluacion test base vs FAIR.
10. Incertidumbre M2.
11. Figuras obligatorias.
12. Conclusiones y limitaciones.

## Politica de limpieza

No se eliminara codigo antiguo hasta que:

1. Exista sustituto en `src/trustworthy_credit/`.
2. Los tests pasen.
3. El notebook maestro ejecute sin errores.
4. Las metricas obligatorias sigan siendo defendibles.
5. El equipo apruebe la migracion.

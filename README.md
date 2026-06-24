# Taller B4-T1 - Redes Neuronales Confiables

Repositorio del proyecto de clasificacion de riesgo de impago sobre
Home Credit Default Risk. El entregable implementa una red neuronal
confiable que combina arquitectura custom, aprendizaje justo, busqueda
AutoML e incertidumbre sobre las predicciones.

## Entregable Oficial

El flujo final del proyecto se ejecuta desde:

- Notebook principal: `notebooks/01_final_mvp.ipynb`
- Paquete Python principal: `src/trustworthy_credit/`
- Tests de regresion: `tests/test_uncertainty_regressions.py`
- Material historico archivado: `legacy/`

El notebook principal es el unico cuaderno necesario para reproducir el MVP
validado. Los notebooks, scripts y documentos antiguos se conservan en
`legacy/` solo por trazabilidad.

## Objetivo

El proyecto aborda tres requisitos del taller:

1. Precision predictiva para detectar solicitudes con riesgo de impago.
2. Justicia estadistica, reduciendo la dependencia entre prediccion y genero.
3. Honestidad del modelo, estimando incertidumbre para identificar casos que
   deberian revisarse con mas cautela.

La variable objetivo es `TARGET` y la variable sensible principal es
`CODE_GENDER`.

## Estructura

```text
taller-redes-confiables/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_final_mvp.ipynb
├── src/
│   └── trustworthy_credit/
│       ├── data_contract.py
│       ├── preprocessing.py
│       ├── splitting.py
│       ├── layers.py
│       ├── models.py
│       ├── metrics.py
│       ├── tuning.py
│       └── uncertainty.py
├── tests/
│   └── test_uncertainty_regressions.py
├── legacy/
│   ├── development/
│   └── original_experiments/
└── data/
    └── raw/
```

## Flujo Del Notebook

`notebooks/01_final_mvp.ipynb` ejecuta el MVP completo:

1. Configuracion de rutas, semillas y parametros.
2. Carga del dataset `application_train.csv`.
3. EDA breve y motivacion tecnica.
4. Split honesto train/validation/test.
5. Preprocesamiento sin leakage.
6. Entrenamiento de modelo base.
7. Arquitectura custom con capas financieras.
8. Busqueda AutoML y barrido de `lambda_fair`.
9. Curva de Pareto AUC vs dependencia con genero.
10. Evaluacion test: AUC, PR-AUC, F1, recall, rho, DPD y EOD.
11. Incertidumbre M2 basada en error esperado del clasificador.
12. Analisis de incertidumbre por clase real y por `EXT_NULL_COUNT`.

## Resultados Validados Del MVP

La ejecucion validada del MVP produjo:

- AUC base test: 0.7436.
- AUC FAIR test: 0.7380.
- `|rho|` base test: 0.0971.
- `|rho|` FAIR test: 0.0088.
- Reduccion aproximada de dependencia lineal con genero: 91%.
- Incertidumbre no constante en test.
- `EXT_NULL_COUNT` conservado con valores semanticos `{0, 1, 2, 3}`.

La pequena perdida de AUC se acepta como trade-off para obtener una reduccion
fuerte de dependencia con la variable sensible.

## Instalacion

Se recomienda usar un entorno virtual de Python. Desde la raiz del repositorio:

```bash
python -m pip install -r requirements.txt
```

Para ejecutar tests:

```bash
python -m pytest -q
```

En la maquina de desarrollo usada para el proyecto tambien se valido con:

```powershell
C:\venvs\homecredit311\Scripts\python.exe -m pytest -q
```

## Datos

El repositorio no incluye datasets de Kaggle. Para ejecutar el notebook, coloca
el archivo principal en:

```text
data/raw/application_train.csv
```

No deben subirse a GitHub:

- `data/`
- `results/`
- `results_dani/`
- `.venv/`
- modelos `.keras`
- artefactos temporales de tuning

## Paquete Principal

El codigo reutilizable vive en `src/trustworthy_credit/`:

- `data_contract.py`: contratos de columnas y validaciones de entrada.
- `splitting.py`: particion estratificada y estructuras de splits.
- `preprocessing.py`: imputacion, escalado, codificacion y preservacion de
  variables semanticas de auditoria.
- `layers.py`: capas customizadas de Keras.
- `models.py`: construccion de modelos neuronales.
- `metrics.py`: metricas predictivas y de fairness.
- `tuning.py`: AutoML, barrido de arquitectura y barrido de fairness.
- `uncertainty.py`: modelo M2 de incertidumbre y artefactos asociados.

## Legacy

La carpeta `legacy/` contiene material historico que ya no forma parte del
camino oficial de ejecucion:

- `legacy/original_experiments/`: scripts, notebooks, checkpoints, reportes y
  figuras de experimentos previos.
- `legacy/development/validated_mvp_build_notes/`: notas tecnicas y notebook de
  desarrollo conservados para trazabilidad.

Este material no debe usarse como punto de entrada del proyecto final.

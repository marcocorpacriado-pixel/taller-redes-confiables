# Taller B4-T1 - Redes Neuronales Confiables

Repositorio del proyecto de clasificación de riesgo de impago sobre Home Credit
Default Risk. El entregable final combina una red neuronal confiable, aprendizaje
justo, búsqueda AutoML, estimación de incertidumbre y un notebook de extras para
estudiar el techo predictivo con tablas relacionales.

## Entregable Oficial

El flujo final del proyecto se ejecuta desde:

- Notebook principal: `notebooks/01_final_mvp.ipynb`
- Notebook de extras: `notebooks/02_extras_grade10.ipynb`
- Paquete Python principal: `src/trustworthy_credit/`
- Tests automatizados: `tests/`
- Material histórico archivado: `legacy/`

El notebook principal es el único cuaderno necesario para reproducir el MVP
validado. El notebook de extras es complementario: no modifica el MVP y concentra
los experimentos relacionales con LightGBM/XGBoost, la comparación de 12 frente a
42 features y la variante FAIR cuadrática.

En este repositorio, **MVP** significa **Producto Mínimo Viable**: el flujo
obligatorio, reproducible y defendible del taller. No significa que el modelo sea
básico o incompleto; significa que concentra los requisitos centrales sin mezclar
experimentos secundarios que puedan dificultar la evaluación.

## Objetivo

El proyecto aborda tres requisitos del taller:

1. Precisión predictiva para detectar solicitudes con riesgo de impago.
2. Justicia estadística, reduciendo la dependencia entre predicción y género.
3. Honestidad del modelo, estimando incertidumbre para identificar casos que
   deberían revisarse con más cautela.

La variable objetivo es `TARGET` y la variable sensible principal es
`CODE_GENDER`.

## Estructura

```text
taller-redes-confiables/
|-- README.md
|-- requirements.txt
|-- notebooks/
|   |-- 01_final_mvp.ipynb
|   `-- 02_extras_grade10.ipynb
|-- src/
|   `-- trustworthy_credit/
|       |-- data_contract.py
|       |-- preprocessing.py
|       |-- splitting.py
|       |-- layers.py
|       |-- models.py
|       |-- metrics.py
|       |-- tuning.py
|       |-- uncertainty.py
|       |-- reproducible_run.py
|       |-- relational_features.py
|       |-- gbm_experiments.py
|       `-- fairness_losses.py
|-- tests/
|   |-- test_reproducible_run_contract.py
|   |-- test_uncertainty_regressions.py
|   |-- test_relational_extras_contract.py
|   `-- test_fairness_losses_contract.py
|-- legacy/
|   |-- development/
|   `-- original_experiments/
`-- data/
    `-- raw/
```

## Flujo Del Notebook Principal

`notebooks/01_final_mvp.ipynb` ejecuta el MVP completo:

1. Configuración de rutas, semillas y parámetros.
2. Carga del dataset `application_train.csv`.
3. EDA breve y motivación técnica.
4. Split honesto train/validation/test.
5. Preprocesamiento sin leakage.
6. Entrenamiento de modelo base.
7. Arquitectura custom con capas financieras.
8. Búsqueda AutoML y barrido de `lambda_fair`.
9. Curva de Pareto AUC frente a dependencia con género.
10. Evaluación final en test: AUC, PR-AUC, F1, recall, rho, DPD y EOD.
11. Incertidumbre M2 basada en error esperado del clasificador.
12. Análisis de incertidumbre por clase real y por `EXT_NULL_COUNT`.

Los artefactos del MVP se guardan en `results/runs/<run_id>/`. Esa carpeta está
ignorada por Git para no versionar modelos, predicciones ni resultados pesados.

## Flujo Del Notebook De Extras

`notebooks/02_extras_grade10.ipynb` ejecuta los extras:

1. Verificación de las ocho tablas CSV del dataset.
2. Comparación intermedia entre 12 y 42 features de `application_train`.
3. Feature engineering relacional en POO.
4. Agregación a una fila por `SK_ID_CURR` desde bureau, previous applications,
   installments, POS cash y credit card.
5. LightGBM OOF sobre el dataset relacional enriquecido.
6. XGBoost OOF como contraste independiente de boosting.
7. Sweep de penalización FAIR cuadrática sobre la arquitectura neuronal.
8. Comparación global contra el MVP neuronal y resumen de artefactos.

Los artefactos de extras se guardan en `results/extras/<run_id>/`, separados de
`results/runs/<run_id>/` y de cualquier resultado histórico suelto en
`results/tables/`.

## Runs Canónicos

Los resultados publicados en esta sección corresponden a las últimas ejecuciones
validadas localmente.

Las carpetas `results/` son artefactos locales ignorados por Git para evitar
versionar modelos, predicciones y salidas pesadas. En GitHub, la evidencia
revisable está en los notebooks ejecutados y en las tablas resumidas de este
README.

### MVP

Ejecución canónica local, reflejada en los outputs ejecutados del notebook:

```text
results/runs/20260624_214719
```

| Métrica | Base | FAIR |
| --- | ---: | ---: |
| AUC test | 0.743811 | 0.738011 |
| PR-AUC test | 0.223342 | 0.218635 |
| Accuracy test | 0.666139 | 0.673033 |
| Recall test | 0.696294 | 0.678303 |
| F1 test | 0.251919 | 0.250919 |
| abs_rho test | 0.100951 | 0.002211 |
| DPD test | 0.090622 | 0.022133 |
| EOD test | 0.081288 | 0.011903 |

Lectura principal:

- La pérdida de AUC al pasar de Base a FAIR es de `0.005800`.
- La dependencia lineal absoluta con género (`abs_rho`) baja de `0.100951` a
  `0.002211`.
- La reducción relativa de `abs_rho` es aproximadamente `97.8%`.
- La incertidumbre M2 genera `46,052` valores únicos en test, por lo que no está
  colapsada.
- `EXT_NULL_COUNT` conserva los valores semánticos `{0, 1, 2, 3}`.

### Extras

Ejecución canónica local de extras, reflejada en los outputs ejecutados del
notebook:

```text
results/extras/20260625_113552
```

| Experimento | Métrica | Valor |
| --- | --- | ---: |
| Red neuronal 42 features | AUC test auditado | 0.755500 |
| LightGBM relacional | OOF AUC | 0.796264 |
| XGBoost relacional | OOF AUC | 0.795414 |
| Cuadrática FAIR alpha=0.5 | AUC test | 0.740491 |
| Cuadrática FAIR alpha=0.5 | abs_rho test | 0.039893 |
| Cuadrática FAIR alpha=0.5 | DPD test | 0.046588 |
| Cuadrática FAIR alpha=0.5 | EOD test | 0.036503 |

Lectura principal:

- LightGBM y XGBoost elevan el techo predictivo al usar información relacional
  completa.
- La red de 42 features muestra una mejora intermedia sin usar tablas
  relacionales.
- La variante cuadrática es un análisis de sensibilidad de fairness; no sustituye
  al FAIR principal del MVP.

## Instalación

Se recomienda usar Python 3.11 y un entorno virtual. Desde la raíz del
repositorio:

```bash
python -m pip install -r requirements.txt
```

Para ejecutar tests:

```bash
python -m pytest -q
```

En la máquina de desarrollo usada para el proyecto también se validó con:

```powershell
C:\venvs\homecredit311\Scripts\python.exe -m pytest -q
```

## Datos

El repositorio no incluye datasets de Kaggle. Para ejecutar el notebook
principal, coloca el archivo principal en:

```text
data/raw/application_train.csv
```

Para ejecutar el notebook de extras coloca también:

```text
data/raw/application_test.csv
data/raw/bureau.csv
data/raw/bureau_balance.csv
data/raw/previous_application.csv
data/raw/installments_payments.csv
data/raw/POS_CASH_balance.csv
data/raw/credit_card_balance.csv
```

No deben subirse a GitHub:

- `data/`
- `results/`
- `results_dani/`
- `results_dani_old_broken_do_not_use/`
- `.venv/`
- `kt_dir/`
- modelos `.keras`
- artefactos temporales de entrenamiento o tuning

## Paquete Principal

El código reutilizable vive en `src/trustworthy_credit/`:

- `data_contract.py`: contratos de columnas y validaciones de entrada.
- `splitting.py`: partición estratificada y estructuras de splits.
- `preprocessing.py`: imputación, escalado, codificación y preservación de
  variables semánticas de auditoría.
- `layers.py`: capas customizadas de Keras.
- `models.py`: construcción de modelos neuronales.
- `metrics.py`: métricas predictivas y de fairness.
- `tuning.py`: AutoML, barrido de arquitectura y barrido de fairness.
- `uncertainty.py`: modelo M2 de incertidumbre y artefactos asociados.
- `reproducible_run.py`: orquestación reproducible del MVP en
  `results/runs/<run_id>/`.
- `relational_features.py`: feature engineering relacional para los extras.
- `gbm_experiments.py`: runners OOF de LightGBM/XGBoost y artefactos de extras.
- `fairness_losses.py`: sweep neuronal de penalización FAIR cuadrática.

## Tests

La suite cubre los contratos más importantes del entregable:

- aislamiento de artefactos del MVP en `results/runs/<run_id>/`;
- preservación de metadatos de incertidumbre;
- contratos del pipeline relacional de extras;
- aislamiento de artefactos de la variante FAIR cuadrática.

## Legacy

La carpeta `legacy/` contiene material histórico que ya no forma parte del camino
oficial de ejecución:

- `legacy/original_experiments/`: scripts, notebooks, checkpoints, reportes y
  figuras de experimentos previos.
- `legacy/development/validated_mvp_build_notes/`: notas técnicas y notebook de
  desarrollo conservados para trazabilidad.

Este material no debe usarse como punto de entrada del proyecto final.

# Bloque 0 - Setup del proyecto y reproducibilidad

## Objetivo del bloque

El objetivo de este bloque es preparar el entorno minimo de trabajo antes de tocar el modelado. La idea es que cualquier persona del equipo pueda entender donde van los datos, donde va el codigo, donde se guardan los resultados y que elementos no deben subirse a GitHub.

Este bloque no entrena ningun modelo todavia. Es una fase de orden, trazabilidad y prevencion de errores.

## Que se ha hecho

Se ha creado una estructura inicial de carpetas:

```text
data/
  raw/
  processed/
docs/
notebooks/
src/
results/
  figures/
  tables/
  models/
```

Tambien se ha creado un archivo `.gitignore` para evitar que se suban accidentalmente datos pesados, modelos entrenados, caches de Python, caches de notebooks o artefactos temporales.

Ademas, se han creado archivos `.gitkeep` en carpetas que normalmente estarian vacias. Esto permite que, cuando se inicialice Git, la estructura de carpetas pueda conservarse aunque todavia no haya datos o resultados finales dentro.

## Como se ha hecho

La estructura se ha creado directamente en la carpeta actual del taller:

```text
C:\Users\dgall\OneDrive\Desktop\Master_MIAX_14\Modulo_4_IA_Avanzada\Taller TransferLearning AML
```

No se han modificado los archivos originales:

```text
home-credit-default-risk-v6.ipynb
Lectura_datos_Taller_B4_T1.ipynb
Taller TL.txt
Taller_B4_T1.pdf
```

Estos archivos siguen siendo materiales de referencia. El proyecto reproducible debe crecer alrededor de ellos, no encima de ellos, para no perder informacion original.

## Por que se ha hecho asi

La practica exige entregar un repositorio de GitHub con codigo de entrenamiento, optimizacion y evaluacion. Para que ese repositorio sea defendible, debe separar claramente:

- datos originales
- datos procesados
- codigo fuente
- notebooks de exploracion
- figuras y tablas finales
- modelos entrenados
- documentacion tecnica

Esta separacion evita tres problemas frecuentes:

1. Mezclar notebooks, datos y resultados en una unica carpeta.
2. Subir a GitHub archivos muy grandes o datos que deben descargarse aparte.
3. No poder reproducir una figura porque no se sabe que script, notebook o configuracion la genero.

## Papel de cada carpeta

### `data/raw/`

Aqui deben ir los CSV originales descargados de Kaggle.

Ejemplos:

```text
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
POS_CASH_balance.csv
credit_card_balance.csv
HomeCredit_columns_description.csv
sample_submission.csv
```

Estos archivos no se deben subir a GitHub. Son grandes y se pueden descargar desde Kaggle.

### `data/processed/`

Aqui iran datasets intermedios generados por nuestro pipeline.

Ejemplos:

```text
mvp_train_features.parquet
mvp_val_features.parquet
mvp_test_features.parquet
feature_names_mvp.json
```

Por ahora tambien se ignora en Git, porque son artefactos regenerables.

### `docs/`

Aqui va la documentacion por bloques. Esta carpeta es importante para explicar al equipo que se ha hecho, por que se ha hecho y que decisiones tecnicas se han tomado.

Los documentos creados en este primer paso son:

```text
docs/bloque_00_setup.md
docs/bloque_01_data_contract_mvp.md
```

### `notebooks/`

Aqui iran notebooks nuevos y limpios del proyecto.

Recomendacion:

```text
00_eda.ipynb
01_mvp_pipeline.ipynb
02_fair_tuning.ipynb
03_uncertainty_analysis.ipynb
04_advanced_features.ipynb
```

Los notebooks originales se dejan en la raiz como material de partida. Mas adelante se pueden mover o copiar a `notebooks/references/`, pero por ahora no se han tocado para preservar el estado inicial.

### `src/`

Aqui ira el codigo reutilizable:

```text
config.py
preprocessing.py
layers.py
losses.py
metrics.py
models.py
tuning.py
uncertainty.py
plots.py
```

La idea es que el notebook no contenga toda la logica. El notebook deberia orquestar, visualizar y explicar; el codigo repetible deberia estar en `src/`.

### `results/figures/`

Aqui se guardaran las graficas obligatorias:

```text
pareto_auc_vs_fairness.png
uncertainty_distribution_target.png
loss_curves_base_fair.png
```

Tambien se pueden guardar graficas extra:

```text
ext_null_count_vs_uncertainty.png
lambda_sweep_metrics.png
```

### `results/tables/`

Aqui se guardaran tablas finales:

```text
test_results_base_vs_fair.csv
pareto_results.csv
bootstrap_confidence_intervals.csv
```

Estas tablas son utiles porque el PDF no deberia ser el unico sitio donde vivan los resultados.

### `results/models/`

Aqui iran checkpoints o modelos entrenados.

No se suben a GitHub por defecto porque pueden pesar mucho y porque, idealmente, deben poder regenerarse.

## Que se ha anadido al `.gitignore`

Se ignoran:

- caches de Python
- entornos virtuales
- checkpoints de Jupyter
- datos originales
- datos procesados
- modelos entrenados
- directorios de Keras Tuner
- archivos de sistema operativo y editor

Fragmentos relevantes:

```gitignore
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
kt_dir/
*.keras
*.h5
*.pkl
*.joblib
results/models/*
!results/models/.gitkeep
```

La excepcion `!data/raw/.gitkeep` permite mantener la carpeta en el repositorio sin subir sus contenidos.

## Estado actual de Git

Se ha comprobado que la carpeta actual todavia no es un repositorio Git:

```text
fatal: not a git repository
```

Esto no bloquea el trabajo. Simplemente significa que, cuando el equipo decida publicar o versionar, habra que ejecutar:

```bash
git init
git add .
git commit -m "Initial project structure"
```

Y despues conectar con GitHub.

## `requirements.txt` recomendado

Para la entrega no conviene generar `requirements.txt` con `pip freeze` a ciegas.

Motivo:

```text
pip freeze captura todo el entorno local, incluidos paquetes que no forman parte real del proyecto.
```

Eso puede hacer que otra persona no consiga instalar el entorno o que se mezclen dependencias innecesarias.

Decision recomendada:

Crear un `requirements.txt` manual y pequeno, con las dependencias que realmente usa la practica.

Version inicial propuesta:

```text
tensorflow==2.15.0
keras-tuner==1.4.6
scikit-learn==1.3.2
fairlearn==0.10.0
pandas==2.1.4
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.13.0
```

Ademas, documentar en el README la version de Python recomendada:

```text
Python >=3.10,<3.12
```

Nota:

La opcion `tf.config.experimental.enable_op_determinism()` requiere una version moderna de TensorFlow. TensorFlow 2.15 es suficiente para este proyecto.

## Asunciones de este bloque

1. Los datos de Kaggle no estan todavia descargados en `data/raw/`.
2. El MVP se hara usando `application_train.csv`.
3. `application_test.csv` no se usara para evaluar metricas, porque no tiene `TARGET`.
4. Los materiales originales adjuntos se mantienen como referencias y no como pipeline final.
5. El codigo definitivo se ira moviendo a `src/` para que sea reproducible.

## Preocupaciones y riesgos

### Riesgo 1: subir datos grandes a GitHub

Los CSV del reto completo pesan mucho. Subirlos al repo puede generar problemas con GitHub y con la entrega.

Mitigacion:

- `data/raw/*` esta ignorado.
- Documentar en README como descargar los datos.

### Riesgo 2: mezclar notebooks exploratorios con resultados finales

Un notebook exploratorio puede tener celdas ejecutadas fuera de orden. Eso hace que los resultados sean dificiles de reproducir.

Mitigacion:

- Guardar la logica final en `src/`.
- Dejar los notebooks para analisis y visualizacion.

### Riesgo 3: no poder regenerar las graficas del PDF

Si una grafica se hace manualmente o copiando valores a mano, luego no se puede auditar.

Mitigacion:

- Guardar CSV de resultados en `results/tables/`.
- Guardar PNG de graficas en `results/figures/`.
- Crear funciones de graficado en `src/plots.py`.

### Riesgo 4: diferencias entre ejecuciones

TensorFlow puede no ser completamente determinista si no se fijan semillas y opciones de determinismo.

Mitigacion futura:

En el primer bloque de codigo del pipeline se debe incluir:

```python
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
```

Esto no garantiza reproducibilidad perfecta en todos los equipos, pero reduce mucho la variabilidad.

## Que obtenemos al finalizar el bloque

Al terminar el Bloque 0 tenemos:

- una estructura ordenada de proyecto
- un `.gitignore` seguro
- carpetas preparadas para datos, codigo, resultados y documentacion
- un documento explicativo del setup
- una base clara para pasar al Bloque 1 sin mezclar responsabilidades

## Criterio de terminado

Este bloque se considera terminado cuando:

- existe la estructura de carpetas indicada
- existe `.gitignore`
- los datos tienen un sitio definido: `data/raw/`
- los resultados tienen un sitio definido: `results/`
- el equipo entiende que los CSV no se suben a GitHub
- existe esta documentacion en `docs/bloque_00_setup.md`

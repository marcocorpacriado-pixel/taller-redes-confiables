# Bloque 7 - Keras Tuner y barrido de lambda

## Objetivo del bloque

El objetivo de este bloque es cumplir la Tarea 3 del enunciado: usar Keras Tuner para buscar una arquitectura razonable de red y, despues, generar una familia controlada de modelos con distintos valores de `lambda_fair`.

El resultado importante no es solo encontrar "el mejor modelo", sino producir evidencia del trade-off:

```text
mayor rendimiento predictivo  <->  menor dependencia con genero
```

Ese trade-off se guardara en:

```text
results/tables/pareto_results.csv
```

y se convertira en la figura obligatoria de Pareto en el Bloque 12.

## Archivos implementados o modificados

El codigo del bloque vive en:

```text
src/tuning.py
```

Tambien se han actualizado:

```text
src/__init__.py
tests/test_tuning.py
docs/bloque_07_keras_tuner_y_barrido_lambda.md
```

El modulo `src/tuning.py` no define nuevas arquitecturas neuronales. Su responsabilidad es orquestar busqueda, entrenamiento, evaluacion de validation y escritura de artefactos. La arquitectura sigue viniendo de:

```text
CustomMLPModelBuilder
FairCustomModelBuilder
FairnessPenalty
```

## Decision metodologica

No se usa optimizacion multiobjetivo en Keras Tuner.

El flujo se divide en dos fases:

```text
Fase A - Keras Tuner
    busca la arquitectura con lambda_fair fijo = 0.5
    objetivo principal = val_auc

Fase B - Barrido manual
    fija la arquitectura ganadora
    entrena varios lambdas
    genera pareto_results.csv
```

Esta decision tiene sentido por tres razones:

```text
1. Keras Tuner es mas estable optimizando un objetivo claro.
2. lambda=0.5 mete presion FAIR moderada durante la busqueda.
3. El barrido manual produce una curva de Pareto reproducible y facil de explicar.
```

## Contrato de artefactos

La clase `TuningArtifactPaths` centraliza las rutas:

```python
TuningArtifactPaths(
    project_root=PROJECT_ROOT,
    tuner_dir_name="kt_dir",
    tables_dir_name="results/tables",
    models_dir_name="results/models",
    pareto_filename="pareto_results.csv",
)
```

Todas las escrituras usan rutas absolutas ancladas a `project_root`. Esto evita el error tipico de ejecutar un notebook desde `notebooks/` y terminar guardando en `notebooks/results/`.

Convencion de nombres:

```text
results/models/fair_lambda_{slug}.keras
results/tables/history_fair_lambda_{slug}.csv
results/tables/pareto_results.csv
```

Ejemplos:

```text
results/models/fair_lambda_0_0.keras
results/models/fair_lambda_0_5.keras
results/tables/history_fair_lambda_0_0.csv
results/tables/history_fair_lambda_0_5.csv
```

El slug se genera con:

```python
lambda_slug(0.5) -> "0_5"
lambda_slug(1.0) -> "1_0"
```

Los CSV guardan las rutas como strings relativos al proyecto para que sean portables entre ordenadores. Internamente, el guardado usa rutas absolutas.

## `TuningConfig`

`TuningConfig` contiene todos los parametros del bloque:

```python
TuningConfig(
    tuning_lambda_fair=0.5,
    lambda_values=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    max_trials=30,
    executions_per_trial=1,
    tuner_objective="val_auc",
    tuner_direction="max",
    min_layers=1,
    max_layers=4,
    units_choices=(64, 128, 256),
    activation_choices=("relu", "elu"),
    dropout_min=0.0,
    dropout_max=0.5,
    dropout_step=0.1,
    learning_rate_min=1e-4,
    learning_rate_max=1e-2,
    batch_size=1024,
    epochs=100,
)
```

Esta configuracion se usa tanto en el tuner como en el barrido para que el entrenamiento sea consistente.

## Fase A - Busqueda con Keras Tuner

La factory del tuner se implementa con:

```text
FairTunerBuildFunctionFactory
```

Su metodo principal es:

```python
model = factory.build(hp)
```

Ese metodo:

```text
1. Lee hiperparametros de Keras Tuner.
2. Construye un CustomMLPConfig.
3. Crea un CustomMLPModelBuilder.
4. Crea un FairCustomModelBuilder con lambda_fair=0.5.
5. Devuelve un modelo dual-input compilado.
```

No llama directamente a `FinancialRatiosLayer`, `TrainableGammaLayer`, `Dense` ni `Dropout`. Eso ya lo hace `CustomMLPModelBuilder.build_probability_graph()` desde los bloques anteriores.

### Espacio de busqueda

El espacio implementado es:

```text
n_layers:       1 a 4
units_i:        64, 128, 256
activation:     relu, elu
dropout:        0.0 a 0.5
learning_rate:  1e-4 a 1e-2 en escala log
```

Para evitar un espacio condicional fragil en `BayesianOptimization`, la factory define siempre:

```text
units_0
units_1
units_2
units_3
```

y solo usa los primeros `n_layers`. Asi todos los trials tienen el mismo conjunto de hiperparametros, aunque no todos esten activos en cada arquitectura.

## `FairKerasTunerRunner`

`FairKerasTunerRunner` ejecuta la Fase A de principio a fin:

```python
runner = FairKerasTunerRunner(config=config, artifacts=artifacts)
search_result = runner.search(processed, verbose=1)
```

Internamente hace:

```text
1. Aplica reproducibilidad.
2. Valida arrays procesados.
3. Calcula class_weight solo con y_train.
4. Resuelve FinancialRatioIndices desde feature_names.
5. Construye el tuner BayesianOptimization.
6. Entrena trials con inputs duales.
7. Extrae best_hyperparameters.
8. Convierte best_hyperparameters a CustomMLPConfig.
```

El resultado es:

```python
TunerSearchResult(
    tuner=...,
    best_hyperparameters=...,
    best_config=...,
    ratio_indices=...,
    class_weight=...,
)
```

El campo `best_config` es la arquitectura fija que se usara en la Fase B.

## Callback comun de entrenamiento

La clase:

```text
FairTuningCallbackFactory
```

crea callbacks frescos para cada `fit`:

```text
EarlyStopping(monitor="val_auc", mode="max", restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", mode="min")
FairnessLogger(include_sensitive_input=True)
```

`include_sensitive_input=True` es obligatorio porque todos los modelos de Bloque 7 son dual-input:

```text
features
sensitive
```

Si ese flag se olvidara, `FairnessLogger` llamaria a `model.predict(X_val)` sin el input `sensitive`, y Keras fallaria.

## Formato de entrada dual

La clase:

```text
DualInputFormatter
```

convierte arrays normales en el diccionario que espera Keras:

```python
{
    "features": X,
    "sensitive": s.reshape(-1, 1),
}
```

Tambien valida que `X` y `s` tengan el mismo numero de filas. Esto es importante porque la FAIR loss se calcula por batch y requiere que cada prediccion este alineada con su genero.

## Fase B - Barrido manual de lambda

El barrido se implementa con:

```text
FairLambdaSweepTrainer
```

Uso esperado:

```python
sweep = FairLambdaSweepTrainer(config=config, artifacts=artifacts)

result = sweep.run(
    data=processed,
    custom_config=search_result.best_config,
    ratio_indices=search_result.ratio_indices,
    class_weight=search_result.class_weight,
    save_models=True,
    verbose=1,
)
```

Para cada lambda:

```text
1. Construye un modelo nuevo desde cero.
2. Reutiliza la misma arquitectura fija.
3. Cambia solo lambda_fair.
4. Entrena con los mismos callbacks y class_weight.
5. Guarda history_fair_lambda_{slug}.csv.
6. Guarda fair_lambda_{slug}.keras si save_models=True.
7. Evalua validation.
8. Crea una fila Pareto.
```

La regla cientifica de esta fase es:

```text
solo cambia lambda_fair
```

Todo lo demas queda fijo: arquitectura, optimizador, callbacks, `class_weight`, batch size y split.

## Evaluacion en validation

La evaluacion se implementa en:

```text
ValidationParetoEvaluator
```

Calcula metricas de probabilidad:

```text
val_auc
val_pr_auc
val_abs_rho
```

Despues calcula un threshold en validation con:

```text
ValidationThresholdSelector.choose_youden()
```

El threshold se obtiene con Youden's J:

```text
J = TPR - FPR
threshold = argmax(J)
```

El valor se clipea a `[0, 1]`, porque `sklearn.metrics.roc_curve` puede devolver un primer threshold artificial mayor que 1.

Con ese threshold se calculan:

```text
val_accuracy
val_precision
val_recall
val_f1
```

Este orden es importante:

```text
primero probabilidades
luego threshold
luego metricas binarias
```

Calcular F1 antes de decidir threshold seria metodologicamente incorrecto.

## `pareto_results.csv`

El CSV final se escribe con:

```text
TrainingArtifactWriter.save_pareto()
```

Columnas:

```text
lambda_fair
val_auc
val_pr_auc
val_abs_rho
val_threshold
val_accuracy
val_precision
val_recall
val_f1
epochs_trained
model_path
history_path
selected_for_test
```

`selected_for_test` marca los modelos que pasaran al Bloque 11.

La seleccion la hace:

```text
ParetoModelSelector
```

Regla implementada:

```text
1. lambda=0.0 siempre se selecciona como base final.
2. Entre lambdas > 0, se busca el menor val_abs_rho.
3. Ese candidato debe no perder mas de fair_selection_max_auc_drop de AUC.
4. Si ninguno cumple el suelo de AUC, se elige el FAIR con mejor AUC.
```

Por defecto:

```text
fair_selection_max_auc_drop = 0.02
```

El test final sigue sin tocarse. Todo esto ocurre solo con train y validation.

## Ejecucion completa recomendada

```python
from src.tuning import (
    FairKerasTunerRunner,
    FairLambdaSweepTrainer,
    TuningArtifactPaths,
    TuningConfig,
)

config = TuningConfig()
artifacts = TuningArtifactPaths()

tuner_runner = FairKerasTunerRunner(
    config=config,
    artifacts=artifacts,
)

search_result = tuner_runner.search(
    processed,
    verbose=1,
)

sweep_trainer = FairLambdaSweepTrainer(
    config=config,
    artifacts=artifacts,
)

sweep_result = sweep_trainer.run(
    data=processed,
    custom_config=search_result.best_config,
    ratio_indices=search_result.ratio_indices,
    class_weight=search_result.class_weight,
    save_models=True,
    verbose=1,
)
```

Salida esperada:

```text
kt_dir/fair_credit_mvp/...
results/tables/pareto_results.csv
results/tables/history_fair_lambda_0_0.csv
results/tables/history_fair_lambda_0_5.csv
...
results/models/fair_lambda_0_0.keras
results/models/fair_lambda_0_5.keras
...
```

## Tests implementados

El archivo nuevo es:

```text
tests/test_tuning.py
```

Cubre:

```text
1. DualInputFormatter genera {"features", "sensitive"} con shapes correctas.
2. FairTunerBuildFunctionFactory crea modelos dual-input.
3. BestHyperparameterExtractor reconstruye CustomMLPConfig.
4. ValidationThresholdSelector devuelve thresholds seguros en [0, 1].
5. FairLambdaSweepTrainer produce histories y pareto_results.csv en un sweep mini.
```

Los tests no lanzan un tuning real de 30 trials porque seria lento e innecesario para CI local. Prueban el contrato de construccion y un sweep pequeno de 2 lambdas y 1 epoch con datos sinteticos.

## Verificacion ejecutada

Tras implementar el bloque:

```text
py_compile sobre src/ y tests/ -> OK
pytest -q -> 26 tests passed
```

Los warnings habituales de TensorFlow no indican fallo del proyecto.

## Que obtenemos al finalizar

Al cerrar este bloque tenemos:

```text
infraestructura POO para Keras Tuner
factory de modelos FAIR sin duplicar arquitectura
callbacks dual-input correctos
barrido controlado de lambda
seleccion validation-only de base y FAIR final
artefactos con nombres estables
pareto_results.csv listo para Bloques 11 y 12
tests del contrato principal
```

## Riesgos y mitigaciones

### Riesgo 1 - El tuner ignora fairness

Mitigacion:

```text
tuning_lambda_fair = 0.5
```

La arquitectura se busca bajo presion FAIR moderada.

### Riesgo 2 - Duplicar arquitectura

Mitigacion:

```text
FairTunerBuildFunctionFactory -> FairCustomModelBuilder -> build_probability_graph()
```

No se reconstruye el modelo a mano.

### Riesgo 3 - Comparacion no controlada

Mitigacion:

```text
FairLambdaSweepTrainer fija custom_config y solo cambia lambda_fair.
```

### Riesgo 4 - Callback mal alimentado

Mitigacion:

```text
FairTuningCallbackFactory hardcodea FairnessLogger(include_sensitive_input=True).
```

### Riesgo 5 - Artefactos perdidos por paths relativos

Mitigacion:

```text
TuningArtifactPaths usa project_root y rutas absolutas para escribir.
```

## Criterio de terminado

El Bloque 7 se considera terminado porque:

```text
existe src/tuning.py
Keras Tuner puede construir modelos con lambda_fair=0.5
se puede extraer CustomMLPConfig desde best_hp
se puede entrenar un barrido de lambda_values
se guardan histories con history_fair_lambda_{slug}.csv
se guarda pareto_results.csv
los paths de modelos siguen fair_lambda_{slug}.keras
lambda=0 queda marcado como base final
un modelo FAIR queda marcado para test segun validation
los tests pasan
```

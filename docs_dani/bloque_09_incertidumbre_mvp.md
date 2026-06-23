# Bloque 9 - Incertidumbre MVP con modelo dual

## Objetivo del bloque

El objetivo de este bloque es cumplir la Tarea 4 del enunciado: el sistema debe devolver una prediccion y una estimacion de incertidumbre.

La estrategia implementada es la explicada en clase:

```text
M1 -> modelo clasificador FAIR seleccionado en validation
M2 -> modelo regresor que predice el error absoluto esperado de M1
```

La incertidumbre se interpreta como:

```text
incertidumbre(x) ~= E[ |M1(x) - y| ]
```

No es una garantia formal de cobertura. Es un proxy aprendido de error esperado.

## Archivos implementados o modificados

El codigo principal vive en:

```text
src/dani_credit/uncertainty.py
```

Tambien se han actualizado:

```text
src/dani_credit/__init__.py
tests/test_dani_uncertainty_regressions.py
docs_dani/bloque_09_incertidumbre_mvp.md
```

El modulo se ha disenado con POO para que el Bloque 10 pueda reutilizar piezas, especialmente el builder de M2, la construccion de `[X, y_proba]`, la escritura de artefactos y la prediccion dual-input de M1.

## Decision MVP

Para el MVP:

```text
M1 = mejor modelo FAIR seleccionado por Bloque 7
M2 = red pequena entrenada con errores de validation
target de M2 = abs(y_val_proba - y_val)
loss de M2 = MAE
salida de M2 = softplus
clipping final = [0, 1]
normalizacion interna de Z dentro de M2
```

No se usa MSE como loss principal.

Motivo:

```text
los errores absolutos suelen concentrarse cerca de cero
MAE es mas robusta que MSE ante unos pocos errores grandes
```

La salida de M2 usa:

```python
tf.keras.layers.Dense(1, activation="softplus")
```

No `sigmoid`.

Motivo:

```text
el error absoluto no puede ser negativo
softplus evita el colapso duro a cero que puede producir una salida ReLU
sigmoid comprime todo a (0, 1) y puede dificultar aprender errores muy pequenos
```

Despues de predecir, la incertidumbre se recorta con `np.clip(uncertainty, 0,
1)`, porque M2 estima el error absoluto de una probabilidad.

## Flujo implementado

El flujo real es:

```text
1. Recibir M1 FAIR ya entrenado.
2. Predecir y_val_proba con M1 en validation.
3. Calcular error_val = abs(y_val_proba - y_val).
4. Construir Z_val = [X_val, y_val_proba].
5. Dividir Z_val/error_val en train/validation internos para M2.
6. Adaptar la normalizacion interna de M2 con el train interno de M2.
7. Entrenar M2 con MAE y EarlyStopping.
8. Predecir y_test_proba con M1 en test.
9. Construir Z_test = [X_test, y_test_proba].
10. Predecir uncertainty_test con M2.
11. Recortar uncertainty a [0, 1] y validar que no sea constante.
12. Aplicar el threshold seleccionado en validation para crear y_pred_label.
13. Recuperar EXT_NULL_COUNT crudo desde ProcessedSplitDataset.
14. Guardar uncertainty_test.csv, summary, history y opcionalmente M2.
```

## Por que M2 no aprende de errores in-sample

Incorrecto:

```text
M1 entrena en train
M1 predice en train
M2 aprende error_train
```

Problema:

```text
M1 ya vio train
sus errores en train son demasiado optimistas
M2 aprenderia una incertidumbre artificialmente baja
```

Correcto para MVP:

```text
M1 entrena en train
M1 predice en validation
M2 aprende error_val
```

Limitacion:

```text
validation ya se uso para arquitectura, lambda y threshold
```

Por eso el Bloque 9 es defendible como MVP, pero el Bloque 10 propone OOF como version mas fuerte.

## `UncertaintyModelConfig`

La configuracion de M2 vive en:

```python
UncertaintyModelConfig(
    hidden_units=(32,),
    activation="relu",
    dropout=0.3,
    output_activation="softplus",
    normalize_inputs=True,
    learning_rate=1e-3,
    gradient_clipnorm=1.0,
    loss="mae",
    batch_size=1024,
    epochs=100,
    internal_validation_size=0.2,
    random_state=42,
    early_stopping_patience=10,
)
```

Puntos importantes:

```text
hidden_units pequeno -> evita sobreajuste de M2
dropout=0.3 -> regulariza el regresor
loss="mae" -> robusta para errores absolutos
output_activation="softplus" -> incertidumbre no negativa sin colapso ReLU
normalize_inputs=True -> estabiliza M2 ante importes financieros sin escalar
internal_validation_size=0.2 -> split explicito para M2
```

## `UncertaintyArtifactPaths`

Los artefactos se centralizan en:

```python
UncertaintyArtifactPaths()
```

Rutas:

```text
results/tables/uncertainty_test.csv
results/tables/uncertainty_summary_by_target.csv
results/tables/history_uncertainty_m2.csv
results/models/uncertainty_m2.keras
```

Todas se anclan a `project_root`, igual que en bloques anteriores, para evitar que un notebook guarde en una carpeta equivocada.

## `DualInputModelPredictor`

M1 es un modelo dual-input:

```text
features
sensitive
```

Por tanto se predice con:

```python
{
    "features": X,
    "sensitive": s.reshape(-1, 1),
}
```

La clase `DualInputModelPredictor` encapsula este formato y devuelve un vector plano de probabilidades.

## `UncertaintyFeatureBuilder`

Esta clase construye la entrada aumentada de M2:

### Construir `Z`

```python
Z = UncertaintyFeatureBuilder().build_augmented(X, y_proba)
```

Produce:

```text
Z = [X || y_proba]
```

M2 aprende no solo a partir de las features originales, sino tambien a partir de la probabilidad que emitio M1.

`EXT_NULL_COUNT` ya no se extrae desde `X_test` procesado para el CSV final,
porque esa matriz puede contener una version escalada. El valor usado para
reporting sale de `ProcessedSplitDataset.ext_null_count_test`, conservado en
crudo por el Bloque 2.

## `UncertaintyTrainingDataBuilder`

Construye el dataset de M2 desde validation:

```python
training_data = UncertaintyTrainingDataBuilder().build(
    m1_model=m1_model,
    data=processed,
)
```

Devuelve:

```python
UncertaintyTrainingData(
    Z=...,
    error=...,
    y_proba=...,
)
```

Donde:

```text
Z     = [X_val, y_val_proba]
error = abs(y_val_proba - y_val)
```

## `UncertaintyInternalSplitter`

No se usa `validation_split` opaco de Keras.

Se usa un split explicito con `train_test_split`:

```text
test_size = internal_validation_size
random_state = 42
shuffle = True
```

Esto deja trazabilidad y reproducibilidad del entrenamiento de M2.

## `UncertaintyM2ModelBuilder`

Construye el modelo M2:

```text
Input(Z_dim)
Normalization(adaptado con Z_train interno)
Dense(32, relu)
Dropout(0.3)
Dense(1, softplus)
```

Compilacion:

```text
optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
loss = mae
```

El nombre del modelo es:

```text
uncertainty_m2_mvp
```

## `UncertaintyPredictionBuilder`

Construye las predicciones finales de test:

```text
y_test_proba = M1(X_test, s_test)
Z_test = [X_test, y_test_proba]
uncertainty = clip(M2(Z_test), 0, 1)
y_pred_label = apply_threshold(y_test_proba, selected_threshold)
EXT_NULL_COUNT = ext_null_count_test crudo de ProcessedSplitDataset
```

La salida principal es un DataFrame con columnas:

```text
SK_ID_CURR
y_true
y_proba
y_pred_label
sensitive
threshold
uncertainty
EXT_NULL_COUNT
```

Esta incertidumbre corresponde al M1 FAIR seleccionado. No debe asumirse que aplica al modelo base salvo que se entrene otro M2 especifico para el base.

Antes de guardar artefactos se validan dos invariantes:

```text
uncertainty debe ser finita, estar en [0, 1] y no ser constante
EXT_NULL_COUNT debe contener solo valores crudos 0, 1, 2 o 3
```

## `UncertaintySummaryBuilder`

Genera:

```text
uncertainty_summary_by_target.csv
```

Columnas:

```text
y_true
count
uncertainty_mean
uncertainty_median
uncertainty_q1
uncertainty_q3
uncertainty_iqr
```

La Figura 2 obligatoria del Bloque 12 comparara:

```text
TARGET=0 -> pago a tiempo real
TARGET=1 -> dificultad real
```

No se agrupa por `y_pred_label` si la figura se describe como buen/mal pagador real.

## `UncertaintyArtifactWriter`

Guarda:

```text
history_uncertainty_m2.csv
uncertainty_test.csv
uncertainty_summary_by_target.csv
uncertainty_m2.keras si save_model=True
```

`save_model` puede desactivarse en tests para no escribir modelos innecesarios.

## `FairModelLoader`

El modulo incluye:

```python
FairModelLoader().load(model_path)
```

Carga un M1 guardado con los `custom_model_objects()` del proyecto.

Esto sera util cuando Bloque 9 se ejecute desde los artefactos del Bloque 7:

```text
pareto_results.csv -> model_path del FAIR seleccionado -> FairModelLoader
```

## `UncertaintyMVPTrainer`

Es el orquestador principal:

```python
trainer = UncertaintyMVPTrainer(
    config=UncertaintyModelConfig(),
    artifacts=UncertaintyArtifactPaths(),
)

result = trainer.run(
    m1_model=fair_model,
    data=processed,
    selected_threshold=threshold_from_validation,
    save_artifacts=True,
    save_model=True,
    verbose=1,
)
```

Devuelve:

```python
UncertaintyMVPResult(
    m2_model=...,
    history=...,
    training_data=...,
    prediction_result=...,
    artifacts=...,
)
```

## Relacion con Bloques 7 y 8

El Bloque 7 produce:

```text
pareto_results.csv
modelos fair_lambda_*.keras
val_threshold por lambda
selected_for_test
```

El Bloque 8 produce:

```text
apply_threshold
```

El Bloque 9 necesita:

```text
M1 FAIR seleccionado
selected_threshold de validation
processed dataset
```

Con eso genera:

```text
uncertainty_test.csv
```

que sera usado por Bloques 11 y 12.

## Ejecucion recomendada

```python
from src.dani_credit.uncertainty import (
    FairModelLoader,
    UncertaintyArtifactPaths,
    UncertaintyModelConfig,
    UncertaintyMVPTrainer,
)

m1 = FairModelLoader().load("results/models/fair_lambda_0_5.keras")

trainer = UncertaintyMVPTrainer(
    config=UncertaintyModelConfig(),
    artifacts=UncertaintyArtifactPaths(),
)

result = trainer.run(
    m1_model=m1,
    data=processed,
    selected_threshold=0.37,
    save_artifacts=True,
    save_model=True,
    verbose=1,
)
```

En una ejecucion real, `selected_threshold` se lee de la fila seleccionada en `pareto_results.csv`.

## Tests implementados

El archivo nuevo es:

```text
tests/test_dani_uncertainty_regressions.py
```

Cubre:

```text
1. UncertaintyM2ModelBuilder usa salida softplus y produce incertidumbre no constante en un caso sintetico.
2. UncertaintyPredictionBuilder rechaza incertidumbre constante.
3. ProcessedSplitDataset conserva EXT_NULL_COUNT crudo en train, validation y test.
4. UncertaintyPredictionBuilder rechaza valores escalados de EXT_NULL_COUNT, como -1.0.
```

Los tests usan un M1 sintetico dual-input para verificar el flujo sin entrenar un Keras Tuner real.

## Verificacion requerida

Tras corregir el bloque deben pasar al menos:

```text
python -m py_compile sobre src/dani_credit/ y tests/
python -m pytest tests/test_dani_uncertainty_regressions.py
```

Ademas, el notebook debe verificar antes de graficar:

```text
unc["uncertainty"].nunique() > 1
set(unc["EXT_NULL_COUNT"].unique()) <= {0, 1, 2, 3}
```

## Que obtenemos al finalizar

Al cerrar este bloque tenemos:

```text
src/dani_credit/uncertainty.py
M2 MVP con MAE y salida softplus
normalizacion interna de features en M2
clipping de incertidumbre a [0, 1]
validacion anti-colapso de incertidumbre constante
split interno reproducible para M2
predicciones de incertidumbre en test
uncertainty_test.csv
uncertainty_summary_by_target.csv
history_uncertainty_m2.csv
uncertainty_m2.keras opcional
EXT_NULL_COUNT incluido para analisis
tests especificos del bloque
```

## Riesgos y mitigaciones

### Riesgo 1 - M2 colapsa a error bajo

Mitigacion:

```text
M2 pequeno
MAE
Dropout
salida softplus
normalizacion interna de Z
validacion anti-colapso antes de guardar
summary por target
histograma en Bloque 12
```

### Riesgo 2 - Entrenar M2 con errores in-sample

Mitigacion:

```text
UncertaintyTrainingDataBuilder usa X_val, y_val y s_val
```

### Riesgo 3 - Sobreinterpretar incertidumbre

Mitigacion:

```text
documentar que es error absoluto esperado aprendido
no venderlo como intervalo con cobertura garantizada
```

### Riesgo 4 - Perder EXT_NULL_COUNT

Mitigacion:

```text
ProcessedSplitDataset conserva ext_null_count_train/val/test en crudo
UncertaintyPredictionBuilder rechaza valores fuera de {0, 1, 2, 3}
```

## Criterio de terminado

El Bloque 9 se considera terminado porque:

```text
M1 FAIR puede predecir validation y test
M2 entrena con MAE
M2 usa salida softplus
M2 normaliza internamente Z
M2 usa split interno reproducible
uncertainty_test.csv se genera
uncertainty_test.csv no contiene incertidumbre constante
uncertainty_summary_by_target.csv se genera
history_uncertainty_m2.csv se genera
uncertainty_m2.keras se guarda si save_model=True
EXT_NULL_COUNT esta en el CSV final con valores crudos 0, 1, 2 o 3
los tests pasan
```

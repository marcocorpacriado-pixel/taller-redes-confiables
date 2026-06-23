# Bloque 10 - Incertidumbre extra con OOF

## Objetivo del bloque

El objetivo de este bloque es mejorar la estimacion de incertidumbre del Bloque 9 usando predicciones out-of-fold.

Este bloque es camino al 10. No es obligatorio para el MVP, pero es metodologicamente mucho mas fuerte.

## Que significa OOF

OOF significa:

```text
out-of-fold
```

Cada fila recibe una prediccion generada por un modelo que no fue entrenado con esa fila.

Esto produce errores mas honestos que usar predicciones in-sample.

## Construccion explicita de train+validation

El Bloque 2 produce arrays separados:

```text
processed.X_train
processed.X_val
processed.X_test
```

Para OOF se construye:

```python
X_trainval = np.vstack([processed.X_train, processed.X_val])
y_trainval = np.concatenate([processed.y_train, processed.y_val])
s_trainval = np.concatenate([processed.s_train, processed.s_val])
ids_trainval = np.array(processed.train_ids + processed.val_ids)
```

Esto es valido porque:

```text
el preprocesador fue fitteado solo en X_train
X_val fue transformado con parametros aprendidos en train
no se usa test
```

No se debe refittear el preprocesador con train+val para este MVP avanzado, porque eso cambiaria las features respecto a los modelos ya seleccionados.

## Flujo recomendado

1. Construir `X_trainval`, `y_trainval`, `s_trainval`.
2. Crear 5 folds estratificados por `TARGET + SENSITIVE`.
3. Para cada fold:
   - construir M1 con mejor arquitectura y mejor lambda.
   - entrenar en 4 folds.
   - predecir en el fold restante.
   - guardar prediccion OOF.
4. Calcular `err_oof = abs(oof_pred - y_trainval)`.
5. Entrenar M2 con `[X_trainval, oof_pred] -> err_oof`.
6. Entrenar M1 final con train+validation.
7. Evaluar M1 final y M2 en test.

## Como obtener hiperparametros

Del Bloque 7:

```python
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

best_config = CustomMLPConfig(
    hidden_units=...,
    activation=best_hp.get("activation"),
    dropout=best_hp.get("dropout"),
    learning_rate=best_hp.get("learning_rate"),
)
```

El `best_lambda` no viene del tuner.

Viene de:

```text
results/tables/pareto_results.csv
```

con la fila seleccionada en validation:

```text
selected_for_test == True
lambda_fair != 0.0
```

## Estratificacion OOF

```python
from sklearn.model_selection import StratifiedKFold

strata = (
    y_trainval.astype(int).astype(str)
    + "_"
    + s_trainval.astype(int).astype(str)
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Esto mantiene representados los grupos TARGET x SENSITIVE en cada fold.

## Callbacks por fold

Los folds OOF usan callbacks de entrenamiento, pero no necesitan guardar modelos temporales.

Callbacks recomendados:

```text
EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6)
```

No es necesario `FairnessLogger` en cada fold para entrenar M2. Puede omitirse para ahorrar tiempo.

## Codigo conceptual del loop OOF

```python
oof_pred = np.zeros(len(y_trainval), dtype="float32")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, strata)):
    class_weight_fold = compute_class_weight_for_binary(y_trainval[tr_idx])

    builder = CustomMLPModelBuilder(best_config)
    model = build_fair_custom_model(
        builder=builder,
        ratio_indices=ratio_indices,
        input_dim=X_trainval.shape[1],
        lambda_fair=best_lambda,
    )

    fold_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        {
            "features": X_trainval[tr_idx],
            "sensitive": s_trainval[tr_idx].reshape(-1, 1),
        },
        y_trainval[tr_idx],
        validation_data=(
            {
                "features": X_trainval[va_idx],
                "sensitive": s_trainval[va_idx].reshape(-1, 1),
            },
            y_trainval[va_idx],
        ),
        class_weight=class_weight_fold,
        epochs=100,
        batch_size=1024,
        callbacks=fold_callbacks,
        verbose=0,
    )

    oof_pred[va_idx] = model.predict(
        {
            "features": X_trainval[va_idx],
            "sensitive": s_trainval[va_idx].reshape(-1, 1),
        },
        batch_size=1024,
        verbose=0,
    ).ravel()
```

Luego:

```python
err_oof = np.abs(oof_pred - y_trainval)
Z_oof = np.column_stack([X_trainval, oof_pred])
```

M2 se entrena igual que en Bloque 9:

```text
loss = MAE
salida = softplus
normalizacion interna de Z
clipping final de incertidumbre a [0, 1]
```

## Entrenar M1 final

Despues de crear OOF, se entrena un M1 final con train+validation.

Opciones:

```text
usar validation_split interno reproducible para early stopping
usar epochs aproximados desde los folds OOF
```

No se usa test para parar entrenamiento.

## Artefactos a guardar

```text
results/tables/oof_predictions.csv
results/tables/uncertainty_oof_test.csv
results/tables/history_uncertainty_oof_m2.csv
results/models/uncertainty_oof_m2.keras
results/models/fair_oof_final.keras
```

Columnas de `oof_predictions.csv`:

```text
SK_ID_CURR
y_true
sensitive
oof_proba
oof_abs_error
fold
```

## Version practica vs estricta

Version practica:

```text
usar X_trainval ya procesado con el preprocesador fitteado en train
```

Version estricta:

```text
reajustar preprocesamiento dentro de cada fold
```

La version estricta es mas pura, pero mucho mas costosa y puede complicar la comparacion con modelos de Bloque 7.

Para esta practica, la version practica es razonable si se declara como limitacion.

## Ventaja sobre Bloque 9

Bloque 9 entrena M2 con validation, aproximadamente 15% del dataset.

Bloque 10 entrena M2 con train+validation completos, usando predicciones honestas OOF.

Ventajas:

```text
mas datos para M2
mas variedad de errores
incertidumbre mas estable
mejor histograma
mejor defensa metodologica
```

## Riesgos y mitigaciones

### Riesgo 1 - Coste computacional

Mitigacion:

```text
arquitectura fija
early stopping
no guardar modelos de cada fold
```

### Riesgo 2 - Los modelos OOF no son M1 final

Mitigacion:

```text
esto es normal en OOF; M2 aprende patrones generales de error
```

### Riesgo 3 - Confundir trainval con test

Mitigacion:

```text
test sigue aislado hasta la evaluacion final
```

## Criterio de terminado

El Bloque 10 se considera terminado cuando:

```text
X_trainval esta construido explicitamente con vstack
existen predicciones OOF para train+validation
M2 se entrena con err_oof
M1 final se entrena sin usar test
uncertainty_oof_test.csv existe
se compara contra Bloque 9
```

# Bloque 11 - Evaluacion final

## Objetivo del bloque

El objetivo de este bloque es producir la tabla final de resultados en test.

La tabla debe comparar como minimo:

```text
modelo base final: lambda_fair = 0.0
mejor modelo FAIR: lambda_fair seleccionado en validation
```

Ambos modelos deben ser de la arquitectura final dual-input de Bloque 6/7.

El modelo del Bloque 4 es un baseline historico auxiliar. No es el "base final" de la tabla principal salvo que se cree una seccion separada.

## Politica de test

El test se usa solo al final.

No se usa para:

```text
elegir arquitectura
elegir lambda
elegir threshold
elegir hiperparametros
elegir modelo de incertidumbre
```

Todas esas decisiones vienen de validation y de `pareto_results.csv`.

## Artefactos de entrada

Del Bloque 7:

```text
results/tables/pareto_results.csv
results/models/fair_lambda_0_0.keras
results/models/fair_lambda_{best}.keras
```

Del Bloque 8:

```text
threshold por modelo en pareto_results.csv
```

Del Bloque 9 o 10:

```text
results/tables/uncertainty_test.csv
```

La incertidumbre aplica al modelo FAIR seleccionado, salvo que se haya entrenado un M2 separado para el base.

## Metricas finales

Metricas sobre probabilidades:

```text
ROC-AUC
PR-AUC
abs_rho(y_proba, sensitive)
```

Metricas sobre etiquetas binarias:

```text
Accuracy
Precision
Recall
F1
DPD
EOD
```

El threshold se eligio en validation por modelo.

## Modulo de metricas

Este bloque debe usar funciones centralizadas en:

```text
src/metrics.py
```

Si todavia no existen, deben implementarse antes de programar este bloque.

Funciones esperadas:

```text
absolute_pearson_correlation
apply_threshold
classification_metrics
fairness_metrics
bootstrap_metric
```

Alternativa minima para `abs_rho`:

```python
from src.base_model import AbsolutePearsonCorrelation

abs_rho = AbsolutePearsonCorrelation().compute(y_proba, s_test)
```

No usar nombres de funciones auxiliares que no existan en el proyecto.

## Evaluacion conceptual

```python
def predict_dual_input(model, X, s, batch_size=1024):
    return model.predict(
        {
            "features": X,
            "sensitive": s.reshape(-1, 1),
        },
        batch_size=batch_size,
        verbose=0,
    ).ravel()
```

```python
def evaluate_dual_input_model(
    *,
    model,
    X_test,
    y_test,
    s_test,
    threshold,
):
    proba = predict_dual_input_model(model, X_test, s_test)
    label = apply_threshold(proba, threshold)

    return {
        **probability_metrics(y_test, proba, s_test),
        **classification_metrics(y_test, label),
        **fairness_metrics(y_test, label, s_test),
    }
```

Este flujo asume modelos dual-input. Si se evalua el baseline historico de Bloque 4, se necesita otra funcion que pase solo `X_test`.

## Tabla final

Archivo:

```text
results/tables/test_results.csv
```

Columnas recomendadas:

```text
model_name
lambda_fair
threshold
auc
auc_ci_low
auc_ci_high
pr_auc
pr_auc_ci_low
pr_auc_ci_high
accuracy
precision
recall
f1
f1_ci_low
f1_ci_high
abs_rho
abs_rho_ci_low
abs_rho_ci_high
dpd
dpd_ci_low
dpd_ci_high
eod
eod_ci_low
eod_ci_high
test_n
model_path
```

Si no hay tiempo para bootstrap, se pueden omitir columnas `_ci_*`, pero es recomendable incluirlas al menos para AUC, F1, abs_rho, DPD y EOD.

## Bootstrap correcto

La funcion bootstrap debe aceptar una metrica con firma homogenea:

```text
metric_fn(y, proba, s) -> float
```

No pasar siempre `s` como tercer argumento a metricas sklearn directamente, porque por ejemplo `roc_auc_score(y, proba, s)` interpreta `s` como `sample_weight`.

Usar wrappers:

```python
def auc_fn(y, proba, s):
    return roc_auc_score(y, proba)

def f1_fn(y, proba, s, threshold):
    label = apply_threshold(proba, threshold)
    return f1_score(y, label, zero_division=0)

def dpd_fn(y, proba, s, threshold):
    label = apply_threshold(proba, threshold)
    return demographic_parity_difference(
        y_true=y,
        y_pred=label,
        sensitive_features=s,
    )
```

Bootstrap conceptual:

```python
def bootstrap_metric(y, proba, s, metric_fn, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            values.append(metric_fn(y[idx], proba[idx], s[idx]))
        except ValueError:
            continue

    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }
```

## Predicciones a guardar

Archivos:

```text
results/tables/test_predictions_base.csv
results/tables/test_predictions_fair.csv
```

Columnas comunes:

```text
SK_ID_CURR
y_true
y_proba
y_pred_label
sensitive
threshold
EXT_NULL_COUNT
```

Para `test_predictions_fair.csv`, si Bloque 9/10 genero incertidumbre:

```text
uncertainty
```

No incluir `uncertainty` en `test_predictions_base.csv` salvo que se entrene un M2 especifico para el modelo base.

## Relacion con `uncertainty_test.csv`

`uncertainty_test.csv` del Bloque 9 normalmente corresponde al modelo FAIR seleccionado.

El Bloque 11 puede unirlo con predicciones FAIR por `SK_ID_CURR` para crear:

```text
test_predictions_fair.csv
```

## Riesgos y mitigaciones

### Riesgo 1 - Comparar modelos con configuraciones distintas

Mitigacion:

```text
base final y FAIR final son ambos dual-input; solo cambia lambda_fair
```

### Riesgo 2 - Usar test para elegir mejor FAIR

Mitigacion:

```text
usar selected_for_test de pareto_results.csv
```

### Riesgo 3 - Bootstrap con firma incorrecta

Mitigacion:

```text
wrappers metric_fn(y, proba, s)
```

### Riesgo 4 - Incertidumbre ambigua

Mitigacion:

```text
uncertainty solo en predicciones FAIR salvo M2 separado para base
```

## Criterio de terminado

El Bloque 11 se considera terminado cuando:

```text
test_results.csv existe
test_predictions_base.csv existe
test_predictions_fair.csv existe
la tabla incluye threshold por modelo
DPD y EOD estan calculados
abs_rho usa funcion real del proyecto
bootstrap no pasa argumentos erroneos a sklearn
test no se uso para seleccionar modelo
```

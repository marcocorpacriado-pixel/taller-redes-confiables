# Bloque 8 - Threshold de decision y metricas reutilizables

## Objetivo del bloque

El objetivo de este bloque es convertir probabilidades en decisiones binarias de forma metodologicamente correcta y reutilizable.

El modelo devuelve:

```text
y_pred_proba = P(TARGET=1 | X)
```

Pero metricas como F1, precision, recall, DPD y EOD necesitan:

```text
y_pred_label in {0, 1}
```

Por tanto necesitamos elegir un threshold. La regla del proyecto queda fijada asi:

```text
el threshold se elige por modelo
el threshold se elige usando validation
el threshold se aplica despues fijo en test
```

No se usa `0.5` por defecto.

## Archivos implementados o modificados

El codigo principal se ha implementado en:

```text
src/metrics.py
```

Tambien se han actualizado:

```text
src/tuning.py
src/__init__.py
tests/test_metrics.py
docs/bloque_08_threshold_decision.md
```

`src/tuning.py` ya no mantiene una implementacion propia de Youden y metricas de validation. Ahora delega en `src.metrics`, que queda como fuente comun para Bloques 7, 8, 9, 11 y 12.

## Por que no usar threshold 0.5

Home Credit es un problema desbalanceado. La clase positiva `TARGET=1` suele ser minoritaria.

Si se usa `0.5`, muchos modelos pueden clasificar casi todo como 0. Eso puede producir:

```text
recall bajo
F1 bajo
DPD artificialmente bajo
falsa impresion de fairness
```

Ejemplo:

```text
si todos son clasificados como 0, DPD puede ser 0,
pero el modelo no sirve porque no detecta malos pagadores.
```

Por eso el threshold se selecciona en validation para cada modelo.

## Separacion conceptual de metricas

El bloque separa tres familias:

```text
metricas de probabilidad
    ROC-AUC
    PR-AUC
    abs_rho(y_pred_proba, sensitive)

metricas binarias
    accuracy
    precision
    recall
    F1

metricas de fairness binarias
    DPD
    EOD
```

El orden correcto en cualquier evaluacion es:

```text
1. predecir probabilidades
2. calcular metricas de probabilidad
3. elegir threshold en validation
4. aplicar threshold para crear etiquetas binarias
5. calcular metricas binarias y fairness binaria
```

## `MetricInputValidator`

La clase `MetricInputValidator` centraliza validaciones:

```text
y_true debe ser binario
y_proba debe ser finito
sensitive debe ser binario
las longitudes deben coincidir
algunas metricas requieren que y_true tenga ambas clases
```

Esto evita errores silenciosos. Por ejemplo, si `sensitive` se desalineara con las predicciones, la correlacion y las metricas FAIR serian numericamente posibles pero semanticamente falsas.

## `ThresholdSelector`

La clase principal para elegir thresholds es:

```python
ThresholdSelector
```

Metodo principal del MVP:

```python
result = ThresholdSelector().choose_youden(y_val, y_val_proba)
threshold = result.threshold
```

Devuelve:

```python
ThresholdSelectionResult(
    threshold=...,
    criterion="youden_j",
    score=...,
)
```

## Threshold por Youden's J

El criterio principal es:

```text
J = TPR - FPR
threshold = argmax(J)
```

Donde:

```text
TPR = true positive rate
FPR = false positive rate
```

El codigo real usa `sklearn.metrics.roc_curve` y despues clipea el threshold:

```text
threshold = clip(threshold, 0, 1)
```

Este clip es importante porque `roc_curve` puede devolver un primer threshold artificial mayor que 1 para representar el punto donde ningun ejemplo se clasifica como positivo.

## Threshold alternativo por F1

Tambien existe:

```python
ThresholdSelector().choose_f1(y_val, y_val_proba)
```

Este metodo busca el threshold que maximiza F1 en validation. No es la regla principal del MVP, pero queda disponible para analisis de sensibilidad si el equipo quiere comparar:

```text
Youden's J vs max F1
```

## `ThresholdApplier`

La clase que convierte probabilidades en labels es:

```python
ThresholdApplier
```

Uso:

```python
y_label = ThresholdApplier().apply(y_proba, threshold)
```

Regla:

```text
y_label = 1 si y_proba >= threshold
y_label = 0 si y_proba < threshold
```

El threshold debe estar en `[0, 1]`. Si no, se lanza `MetricsError`.

## `ProbabilityMetricCalculator`

Calcula metricas que no dependen de threshold:

```python
prob_metrics = ProbabilityMetricCalculator().calculate(
    y_true=y_true,
    y_proba=y_proba,
    sensitive=sensitive,
)
```

Devuelve:

```python
ProbabilityMetrics(
    roc_auc=...,
    pr_auc=...,
    abs_rho=...,
)
```

`abs_rho` es:

```text
|corr(y_pred_proba, sensitive)|
```

Esta es la metrica de fairness continua que alimenta la curva Pareto del Bloque 7.

## `AbsolutePearsonCorrelation`

El calculo de correlacion vive tambien en `src.metrics.py`:

```python
AbsolutePearsonCorrelation().compute(y_proba, sensitive)
```

Regla numerica:

```text
si el denominador es demasiado pequeno -> 0.0
si no -> |numerator / denominator|
```

No se suma `eps` despues de validar el denominador. Esto evita sesgar ligeramente correlaciones perfectamente definidas.

## `BinaryClassificationMetricCalculator`

Calcula metricas binarias:

```python
binary_metrics = BinaryClassificationMetricCalculator().calculate(
    y_true=y_true,
    y_proba=y_proba,
    threshold=threshold,
)
```

Devuelve:

```python
BinaryClassificationMetrics(
    accuracy=...,
    precision=...,
    recall=...,
    f1=...,
    threshold=...,
)
```

Estas metricas se deben reportar siempre junto al threshold usado.

## DPD

Demographic Parity Difference:

```text
DPD = | P(y_hat=1 | S=0) - P(y_hat=1 | S=1) |
```

Mide si la tasa de decisiones positivas cambia entre grupos sensibles.

Un DPD bajo no basta para decir que el modelo es bueno. Si el modelo predice todo como 0, DPD puede ser 0, pero recall sera malo. Por eso el proyecto reporta DPD junto con recall y F1.

## EOD

Equalized Odds Difference:

```text
EOD = max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)
```

Donde:

```text
TPR_g = P(y_hat=1 | Y=1, S=g)
FPR_g = P(y_hat=1 | Y=0, S=g)
```

Esta definicion es mas clara que escribir solo `max_y`, porque explica que EOD compara tanto verdaderos positivos como falsos positivos entre grupos.

## `FairnessMetricCalculator`

La clase que calcula DPD y EOD es:

```python
fair_metrics = FairnessMetricCalculator().calculate(
    y_true=y_true,
    y_proba=y_proba,
    sensitive=sensitive,
    threshold=threshold,
)
```

Internamente:

```text
1. aplica el threshold
2. calcula labels binarias
3. llama a fairlearn.metrics.demographic_parity_difference
4. llama a fairlearn.metrics.equalized_odds_difference
```

Devuelve:

```python
FairnessMetrics(
    demographic_parity_difference=...,
    equalized_odds_difference=...,
)
```

## Bootstrap

El modulo incluye:

```python
BootstrapMetricCalculator
bootstrap_metric(...)
```

La firma segura para funciones bootstrap es:

```python
metric_fn(y_true, y_proba, sensitive) -> float
```

Esto evita un bug comun: pasar `sensitive` como tercer argumento posicional a una metrica de sklearn como `roc_auc_score`, que lo interpretaria como `sample_weight`.

Ejemplo correcto:

```python
def auc_fn(y, proba, sensitive):
    del sensitive
    return roc_auc_score(y, proba)

interval = bootstrap_metric(
    y_true,
    y_proba,
    sensitive,
    auc_fn,
    n_bootstrap=500,
)
```

El resultado es:

```python
BootstrapInterval(
    mean=...,
    lower=...,
    upper=...,
    n_bootstrap=...
)
```

Esto se usara especialmente en Bloque 11 para intervalos de confianza.

## Funciones de conveniencia

Aunque el modulo esta disenado con POO, tambien expone funciones cortas para notebooks:

```python
choose_threshold_youden(y_true, y_proba)
apply_threshold(y_proba, threshold)
absolute_pearson_correlation(y_proba, sensitive)
classification_metrics(y_true, y_proba, threshold)
fairness_metrics(y_true, y_proba, sensitive, threshold)
bootstrap_metric(y_true, y_proba, sensitive, metric_fn)
```

Todas delegan en las clases principales. No hay logica duplicada.

## Integracion con Bloque 7

`src/tuning.py` se ha actualizado para delegar en `src.metrics`.

En concreto:

```text
ValidationThresholdSelector -> usa ThresholdSelector().choose_youden(...)
ValidationParetoEvaluator  -> usa ProbabilityMetricCalculator
ValidationParetoEvaluator  -> usa BinaryClassificationMetricCalculator
```

Asi, el `pareto_results.csv` del Bloque 7 y la evaluacion final del Bloque 11 usaran la misma logica de threshold y metricas.

## Uso esperado en Bloque 11

Ejemplo de evaluacion final:

```python
threshold = row_from_pareto["val_threshold"]

probability = ProbabilityMetricCalculator().calculate(
    y_true=y_test,
    y_proba=y_test_proba,
    sensitive=s_test,
)

binary = BinaryClassificationMetricCalculator().calculate(
    y_true=y_test,
    y_proba=y_test_proba,
    threshold=threshold,
)

fairness = FairnessMetricCalculator().calculate(
    y_true=y_test,
    y_proba=y_test_proba,
    sensitive=s_test,
    threshold=threshold,
)
```

El threshold viene de validation. No se recalcula en test.

## Tests implementados

El archivo nuevo es:

```text
tests/test_metrics.py
```

Cubre:

```text
1. Youden devuelve thresholds seguros en [0, 1].
2. apply_threshold usa la regla proba >= threshold.
3. metricas de probabilidad y binarias estan separadas.
4. DPD y EOD se calculan con fairlearn.
5. absolute Pearson devuelve 1.0 en correlacion perfecta.
6. bootstrap_metric usa firma (y, proba, sensitive).
7. thresholds fuera de [0, 1] fallan con MetricsError.
```

Ademas, los tests del Bloque 7 siguen pasando tras cambiar `src/tuning.py` para usar `src.metrics`.

## Verificacion ejecutada

Tras implementar el bloque:

```text
py_compile sobre src/ y tests/ -> OK
pytest -q -> 33 tests passed
imports publicos del Bloque 8 -> OK
```

Los warnings habituales de TensorFlow no indican fallo del proyecto.

## Que obtenemos al finalizar

Al cerrar este bloque tenemos:

```text
src/metrics.py como fuente unica de metricas reutilizables
threshold por modelo elegido en validation
Youden's J implementado con clip a [0, 1]
threshold alternativo max F1 disponible
apply_threshold validado
ROC-AUC, PR-AUC y abs_rho separados de metricas binarias
accuracy, precision, recall y F1 calculados tras threshold
DPD y EOD implementados con fairlearn
bootstrap seguro para intervalos de confianza
Bloque 7 alineado con las metricas comunes
tests especificos del bloque
```

## Riesgos y mitigaciones

### Riesgo 1 - Usar threshold de test

Mitigacion:

```text
el modulo solo selecciona thresholds; la politica del proyecto exige pasarle validation
Bloque 11 debe recibir el threshold desde pareto_results.csv
```

### Riesgo 2 - Reutilizar threshold del modelo base para FAIR

Mitigacion:

```text
pareto_results.csv guarda val_threshold por lambda
```

Cada modelo tiene su threshold.

### Riesgo 3 - Fairness trivial

Si el threshold es demasiado alto, casi nadie sera positivo y DPD puede parecer bajo.

Mitigacion:

```text
reportar recall y F1 junto con DPD/EOD
```

### Riesgo 4 - Bootstrap con firmas incompatibles

Mitigacion:

```text
BootstrapMetricCalculator exige metric_fn(y, proba, sensitive)
```

Asi evitamos pasar `sensitive` accidentalmente como `sample_weight` a sklearn.

## Criterio de terminado

El Bloque 8 se considera terminado porque:

```text
existe src/metrics.py
choose_threshold_youden usa np.clip
apply_threshold valida threshold en [0, 1]
metricas de probabilidad y binarias estan separadas
DPD y EOD estan implementadas con fairlearn
bootstrap_metric tiene firma segura
src/tuning.py consume las metricas comunes
tests/test_metrics.py existe
la suite completa pasa
```

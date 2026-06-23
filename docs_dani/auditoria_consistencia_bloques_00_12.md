# Auditoria de consistencia - Bloques 00 a 12

## Objetivo

Este documento deja constancia de la revision de consistencia realizada sobre la documentacion activa del proyecto.

La revision comprueba que los bloques ya creados estan alineados con las criticas tecnicas incorporadas y que no quedan instrucciones activas de versiones antiguas que puedan desviar la implementacion.

## Alcance revisado

Se ha revisado como fuente activa:

```text
docs/bloque_00_setup.md
docs/bloque_01_data_contract_mvp.md
docs/bloque_02_preprocesamiento_sin_leakage.md
docs/bloque_03_split_honesto.md
docs/bloque_04_modelo_base.md
docs/bloque_05_capas_custom.md
docs/bloque_06_fair_loss_profesional.md
docs/bloque_07_keras_tuner_y_barrido_lambda.md
docs/bloque_08_threshold_decision.md
docs/bloque_09_incertidumbre_mvp.md
docs/bloque_10_incertidumbre_oof_extra.md
docs/bloque_11_evaluacion_final.md
docs/bloque_12_figuras_obligatorias.md
requirements.txt
.gitignore
```

Estado importante:

```text
src/ esta vacio por ahora.
```

Por tanto, todavia no existe codigo Python ejecutable del pipeline final. Lo que existe es la especificacion tecnica por bloques y la estructura de proyecto. Cuando se implementen los Bloques 6-15, el codigo de `src/` debe seguir exactamente estas decisiones.

## Materiales no considerados fuente de verdad

Los archivos originales de la practica quedan como referencia:

```text
home-credit-default-risk-v6.ipynb
Lectura_datos_Taller_B4_T1.ipynb
Taller TL.txt
Taller_B4_T1.pdf
```

Pueden contener codigo didactico, aproximaciones del profesor o estrategias exploratorias. No son el pipeline final.

La fuente de verdad del MVP es:

```text
docs/bloque_00_setup.md a docs/bloque_12_figuras_obligatorias.md
```

## Decisiones verificadas

### 1. Dataset del MVP

Decision activa:

```text
MVP solo con application_train.csv
```

Motivo:

`application_train.csv` contiene `TARGET`, `CODE_GENDER`, variables financieras y `EXT_SOURCE_1/2/3`, suficientes para cumplir arquitectura custom, FAIR loss, AutoML e incertidumbre.

`application_test.csv` no se usa para evaluar porque no tiene `TARGET`.

Las tablas relacionales quedan para fase extra:

```text
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
POS_CASH_balance.csv
credit_card_balance.csv
```

### 2. Variables del MVP

Decision activa:

```text
TARGET -> objetivo
CODE_GENDER -> sensible, transformada a SENSITIVE
CODE_GENDER no entra en X
SK_ID_CURR se conserva como indice/trazabilidad, no como feature
```

Variables de entrada activas:

```text
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
DAYS_BIRTH -> AGE_YEARS
DAYS_EMPLOYED -> EMPLOYED_YEARS + DAYS_EMPLOYED_ANOM
EXT_SOURCE_1
EXT_SOURCE_2
EXT_SOURCE_3
NAME_EDUCATION_TYPE
NAME_FAMILY_STATUS
NAME_INCOME_TYPE
REGION_RATING_CLIENT_W_CITY
FLAG_OWN_CAR
CNT_CHILDREN
```

Variables derivadas activas:

```text
EXT_SOURCE_1_WAS_MISSING
EXT_SOURCE_2_WAS_MISSING
EXT_SOURCE_3_WAS_MISSING
EXT_NULL_COUNT
```

### 3. Preprocesamiento financiero

Decision activa:

```text
No aplicar log1p global a los importes financieros.
No escalar los importes financieros con RobustScaler.
```

Los importes:

```text
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
```

quedan imputados con mediana aprendida solo en train y en escala original.

Motivo:

La capa custom necesita calcular ratios financieros interpretables. Si los importes estuvieran log-transformados o robust-scaled, `CREDIT/INCOME` dejaria de tener sentido financiero directo.

### 4. Ruta C de capas custom

Decision activa:

```text
Importes originales imputados
-> FinancialRatiosLayer
-> TrainableGammaLayer sobre los ratios
-> BatchNormalization
-> capas densas
```

Ratios:

```text
CREDIT / INCOME
ANNUITY / INCOME
CREDIT / GOODS
ANNUITY / CREDIT
```

La capa gamma se aplica a los ratios generados, no a importes log-transformados ni a variables robust-scaled.

La gamma se parametriza como:

```text
gamma = 0.1 + 1.4 * sigmoid(theta)
```

con inicializacion cercana a `gamma = 1`.

### 5. Desbalance de clases

Decision activa para MVP:

```text
Binary Crossentropy + class_weight
```

Focal Loss queda como experimento extra, no como configuracion principal del MVP.

Regla activa:

```text
No combinar focal loss con class_weight.
```

### 6. Comparacion controlada Base vs FAIR

Decision activa:

```text
Base final = arquitectura final + lambda_fair = 0
FAIR final = misma arquitectura + lambda_fair > 0
```

Solo cambia:

```text
lambda_fair
```

Debe mantenerse igual:

```text
preprocesamiento
split
arquitectura
optimizador
learning rate
callbacks
batch size
estrategia de desbalance
```

Esto evita atribuir a la FAIR loss cambios que en realidad provengan de otra configuracion.

### 7. FAIR loss profesional

Decision activa:

```text
Functional API con dos inputs:
features
sensitive
```

No se usa `y_true_aug` como via principal.

La penalizacion se anade con:

```text
self.add_loss(lambda_fair * Pearson(y_pred, sensitive)^2)
```

Ventaja:

`y_true` sigue siendo `TARGET`, por lo que `AUC`, `accuracy`, `class_weight` y callbacks funcionan de forma normal.

### 8. Keras Tuner y barrido lambda

Decision activa:

```text
Fase A: Keras Tuner con lambda_fair = 0.5 fijo y objetivo val_auc
Fase B: fijar arquitectura ganadora y barrer lambda_fair manualmente
```

Lambdas activos:

```text
[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
```

El tuner no usa test.

La curva Pareto se construye con validation:

```text
X = val_abs_rho
Y = val_auc
```

### 9. Threshold por modelo

Decision activa:

```text
threshold elegido por modelo en validation
threshold aplicado fijo en test
```

No se usa `0.5` por defecto.

Criterio MVP:

```text
Youden's J = TPR - FPR
```

La tabla final debe incluir columna:

```text
threshold
```

### 10. Incertidumbre MVP

Decision activa:

```text
M1 = mejor modelo FAIR seleccionado por validation
M2 = modelo simple que predice error absoluto esperado
```

M2 se entrena con:

```text
loss = MAE
```

No se usa MSE como loss principal de M2.

MVP:

```text
M1 entrena en train
M1 predice en validation
M2 aprende error_val = abs(pred_val - y_val)
M2 estima incertidumbre en test
```

Limitacion declarada:

validation tambien se usa para seleccion de arquitectura, lambda y threshold. Por eso el Bloque 10 propone OOF como version fuerte.

### 11. Incertidumbre extra OOF

Decision activa:

```text
5-fold OOF sobre train + validation
```

Cada punto recibe una prediccion de un modelo que no lo entreno.

Luego:

```text
M2 aprende err_oof
M1 final se entrena en train + validation
test se usa solo para evaluacion final
```

### 12. Evaluacion final

Decision activa:

El test se usa solo al final para:

```text
Modelo base final
Mejor modelo FAIR seleccionado en validation
```

Metricas de probabilidad:

```text
AUC ROC
PR-AUC
abs_rho(y_proba, sensitive)
```

Metricas binarias:

```text
Accuracy
Precision
Recall
F1
DPD
EOD
```

DPD y EOD se calculan con labels generadas usando el threshold de validation de cada modelo.

### 13. Figuras obligatorias

Decision activa:

```text
Pareto en validation: val_abs_rho vs val_auc
Incertidumbre en test agrupada por TARGET real
Curvas de loss base vs FAIR
```

Figura extra recomendada:

```text
EXT_NULL_COUNT vs incertidumbre
```

Esta figura responde directamente a la pregunta de si el modelo duda mas cuando las fuentes externas tienen mala calidad o valores ausentes.

## Busqueda de restos antiguos

Se buscaron referencias a decisiones antiguas o peligrosas:

```text
y_true_aug como implementacion principal
lambda=0 como configuracion del tuner
MSE como loss principal de M2
log1p global activo
ratios sobre logs
RobustScaler sobre todas las variables
focal loss como MVP principal
batch_size 512
LayerNormalization como decision activa
TrainableGammaFeaturesLayer
application_test.csv como test evaluable
```

Resultado:

```text
No quedan instrucciones activas de esas rutas antiguas.
```

Las menciones que permanecen son advertencias o explicaciones de por que no se usan.

Ejemplos:

```text
No usamos MSE como loss principal para M2.
Las columnas financieras no estan log-transformadas.
application_test.csv no se usara para evaluar metricas.
No usamos y_true_aug como via principal.
```

## Estado de preparacion para Bloques 6 a 12

Los Bloques 6 a 12 ya estan actualizados con la estrategia final:

```text
Bloque 6 -> add_loss con dos inputs, no y_true_aug
Bloque 7 -> tuner lambda=0.5 + barrido controlado de lambda
Bloque 8 -> threshold por modelo en validation
Bloque 9 -> M2 con MAE y arquitectura simple
Bloque 10 -> OOF como mejora fuerte
Bloque 11 -> test solo para tabla final
Bloque 12 -> figuras desde CSV reproducibles
```

## Lo que falta implementar en codigo

Cuando se pase de documentacion a codigo, crear:

```text
src/config.py
src/preprocessing.py
src/layers.py
src/metrics.py
src/models.py
src/tuning.py
src/uncertainty.py
src/plots.py
```

Y asegurar que cada modulo respeta esta auditoria.

## Criterio para seguir al Bloque 13

Antes de empezar con tablas relacionales o mejoras avanzadas, debe existir:

```text
pipeline MVP ejecutable
pareto_results.csv
test_results.csv
uncertainty_test.csv
las tres figuras obligatorias
```

Solo despues tiene sentido ampliar hacia:

```text
bureau
previous_application
installments
POS_CASH
credit_card_balance
Spearman/dCor
MC-Dropout
calibracion
ensemble
```


# Bloque 3 - Split honesto train / validation / test

## Objetivo del bloque

El Bloque 3 implementa el split interno del MVP:

```text
70% train
15% validation
15% test
```

El codigo real esta en:

```text
src/splitting.py
```

Tambien se ha actualizado:

```text
src/__init__.py
```

para exportar las clases del bloque.

Este bloque recibe el `DeterministicDataset` creado por el Bloque 2 y devuelve un `RawSplitDataset`, que despues vuelve al Bloque 2 para aplicar imputacion, escalado y one-hot encoding solo con train.

## Por que este bloque existe separado del preprocesamiento

Separar split y preprocesamiento evita el error clasico:

```text
imputar/escalar todo el dataset
despues dividir
```

Ese orden introduce leakage porque validation y test influyen en medianas, escalas y categorias.

El flujo correcto es:

```text
Bloque 2 deterministico
-> Bloque 3 split honesto
-> Bloque 2 estadistico fitted solo en train
```

## Codigo creado

Modulo:

```text
src/splitting.py
```

Clases principales:

```text
SplitConfig
HomeCreditTrainValTestSplitter
StratificationKeyBuilder
SplitReportBuilder
SplitIndexExporter
```

Validadores:

```text
SplitConfigValidator
DatasetAlignmentValidator
```

Contenedores:

```text
SplitArtifacts
SplitReportRow
```

Excepcion:

```text
SplitError
```

## `SplitConfig`

Configuracion inmutable del split.

Valores por defecto:

```text
test_size = 0.15
validation_size = 0.15
random_state = 42
shuffle = True
```

Propiedades:

```python
train_size
validation_size_relative_to_trainval
```

`validation_size_relative_to_trainval` calcula:

```text
0.15 / (1 - 0.15) = 0.17647
```

Esto es necesario porque primero se separa test, y despues validation sale del subconjunto train+validation.

## `SplitConfigValidator`

Responsabilidad:

```text
validar que las proporciones del split tienen sentido
```

Comprueba:

```text
test_size > 0
validation_size > 0
test_size + validation_size < 1
```

Si algo falla, lanza:

```text
SplitError
```

## `DatasetAlignmentValidator`

Responsabilidad:

```text
asegurar que X, y, s estan alineados antes de dividir
```

Comprueba:

```text
len(features) == len(target) == len(sensitive)
features.index == target.index
features.index == sensitive.index
TARGET binario
SENSITIVE binario
```

Motivo:

Si los indices no estan alineados, podriamos asignar a una fila el target o genero de otra. Eso invalidaria todo el proyecto.

## `StratificationKeyBuilder`

Responsabilidad:

```text
crear la variable auxiliar TARGET + SENSITIVE
```

Metodo principal:

```python
build(target, sensitive) -> pd.Series
```

Ejemplos de strata:

```text
0_0
0_1
1_0
1_1
```

Interpretacion:

```text
TARGET=0, SENSITIVE=0
TARGET=0, SENSITIVE=1
TARGET=1, SENSITIVE=0
TARGET=1, SENSITIVE=1
```

Tambien implementa:

```python
value_counts(target, sensitive)
```

para revisar cuantos ejemplos hay en cada grupo combinado.

## `HomeCreditTrainValTestSplitter`

Clase central del Bloque 3.

Responsabilidad:

```text
crear el split 70/15/15 estratificado por TARGET + SENSITIVE
```

Metodo principal:

```python
split(dataset: DeterministicDataset) -> SplitArtifacts
```

Flujo interno:

```text
1. Validar configuracion
2. Validar alineacion X/y/s
3. Crear strata TARGET_SENSITIVE
4. Separar test desde el dataset completo
5. Separar validation desde trainval
6. Crear RawSplitDataset
7. Crear reporte de control
8. Devolver SplitArtifacts
```

## Orden del split

El split se hace en dos pasos:

```text
dataset completo
-> trainval + test
-> train + validation
```

Motivo:

El test queda aislado desde el principio y no participa en:

```text
tuning
seleccion de lambda
seleccion de threshold
early stopping
incertidumbre MVP
```

## Estratificacion conjunta

El splitter usa `train_test_split` de sklearn con:

```python
stratify = TARGET + "_" + SENSITIVE
```

No basta estratificar solo por `TARGET`.

Motivo:

La practica evalua tambien fairness por genero. Si la proporcion de genero cambia mucho entre train, validation y test, las metricas de fairness serian inestables.

## `SplitArtifacts`

Salida completa del bloque:

```text
raw_splits
report
config
```

### `raw_splits`

Es un `RawSplitDataset` del Bloque 2.

Contiene:

```text
X_train
X_val
X_test
y_train
y_val
y_test
s_train
s_val
s_test
```

Este objeto se pasa despues a:

```python
HomeCreditMVPPreprocessingPipeline.fit_transform_splits(raw_splits)
```

### `report`

Es un `pd.DataFrame` con:

```text
split
n
target_rate
sensitive_rate
target_0_sensitive_0
target_0_sensitive_1
target_1_sensitive_0
target_1_sensitive_1
```

Sirve para comprobar que train, validation y test mantienen proporciones similares.

### `config`

Guarda la configuracion usada:

```text
test_size
validation_size
random_state
shuffle
```

## `SplitReportBuilder`

Responsabilidad:

```text
crear una tabla de diagnostico de los splits
```

Metodo:

```python
build(raw_splits) -> pd.DataFrame
```

Por cada split calcula:

```text
n
mean(TARGET)
mean(SENSITIVE)
conteos de los cuatro grupos TARGET+SENSITIVE
```

Esta tabla deberia guardarse o al menos imprimirse en el notebook del MVP.

## `SplitIndexExporter`

Responsabilidad:

```text
guardar los SK_ID_CURR de train, validation y test
```

Metodos:

```python
to_dict(raw_splits)
save_json(raw_splits, path)
```

Ejemplo de ruta:

```text
data/processed/mvp_split_indices.json
```

Motivo:

Permite reproducir exactamente que clientes estaban en cada split, incluso si mas adelante cambia algun detalle del pipeline.

## Uso real esperado

Ejemplo:

```python
from src.preprocessing import HomeCreditMVPPreprocessingPipeline
from src.splitting import HomeCreditTrainValTestSplitter, SplitIndexExporter

preprocessing = HomeCreditMVPPreprocessingPipeline()

raw_df = preprocessing.load_raw()
deterministic = preprocessing.apply_deterministic_transforms(raw_df)

splitter = HomeCreditTrainValTestSplitter()
artifacts = splitter.split(deterministic)

print(artifacts.report)

SplitIndexExporter().save_json(
    artifacts.raw_splits,
    "data/processed/mvp_split_indices.json",
)

processed = preprocessing.fit_transform_splits(artifacts.raw_splits)
```

## Politica de uso de cada split

### Train

Se usa para:

```text
fit de imputadores, scaler y one-hot encoder
entrenar pesos del modelo
calcular class_weight
```

### Validation

Se usa para:

```text
Keras Tuner
early stopping
seleccion de lambda_fair
seleccion de threshold
entrenar M2 en incertidumbre MVP si no se usa OOF
```

### Test

Se usa solo para:

```text
tabla final
figuras finales en holdout
bootstrap final
comparacion base vs FAIR
```

No se usa para elegir arquitectura, lambda, threshold ni hiperparametros.

## Errores evitados

### Error 1 - Tuning sobre test

El codigo aisla test en el primer split.

### Error 2 - Elegir threshold con test

Validation queda separado precisamente para elegir thresholds sin tocar test.

### Error 3 - Ajustar scaler antes del split

El Bloque 3 devuelve `RawSplitDataset`. Solo despues el Bloque 2 ajusta el preprocesador con `X_train`.

### Error 4 - No estratificar por genero

La estratificacion es conjunta:

```text
TARGET + SENSITIVE
```

### Error 5 - Perder indices

Los indices son `SK_ID_CURR` y se conservan dentro de cada `X_*`.

## Validaciones implementadas

El codigo valida:

```text
proporciones de split validas
indices alineados entre X, y, s
TARGET binario
SENSITIVE binario
posibilidad de stratify por grupos combinados
```

Si sklearn no puede estratificar porque algun grupo es demasiado pequeno, se lanza:

```text
SplitError
```

con un mensaje especifico.

## Relacion con Bloque 2

Entrada:

```text
DeterministicDataset
```

Salida:

```text
RawSplitDataset
```

Despues:

```text
RawSplitDataset -> HomeCreditFeaturePreprocessor -> ProcessedSplitDataset
```

## Relacion con Bloque 4

El modelo base entrenara con:

```text
processed.X_train
processed.y_train
```

y monitorizara:

```text
processed.X_val
processed.y_val
```

El test queda reservado.

## Que obtenemos al finalizar el Bloque 3

Codigo:

```text
src/splitting.py
src/__init__.py actualizado
```

Capacidades:

```text
split 70/15/15 reproducible
estratificacion TARGET+SENSITIVE
reporte de composicion
exportacion opcional de indices
handoff limpio hacia el preprocesador estadistico
```

## Criterio de terminado

Bloque 3 queda terminado cuando:

```text
src/splitting.py existe
HomeCreditTrainValTestSplitter crea RawSplitDataset
SplitReportBuilder genera reporte
SplitIndexExporter puede guardar indices
el split usa random_state=42
el split es estratificado por TARGET+SENSITIVE
el modulo compila
hay smoke test con datos sinteticos
este .md describe el codigo real
```


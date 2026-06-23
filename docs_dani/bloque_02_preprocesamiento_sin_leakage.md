# Bloque 2 - Preprocesamiento sin leakage

## Objetivo del bloque

El Bloque 2 implementa el preprocesamiento del MVP sin introducir leakage.

El codigo real esta en:

```text
src/preprocessing.py
```

Tambien se ha actualizado:

```text
src/__init__.py
```

para exportar las clases principales.

Este bloque no crea el split train/validation/test. Esa responsabilidad queda para el Bloque 3. La separacion es intencionada: si el preprocesador hiciera el split internamente, seria mas facil acabar imputando o escalando antes de dividir.

## Regla central anti-leakage

Hay dos tipos de transformaciones:

```text
deterministas
estadisticas
```

Las transformaciones deterministas no aprenden estadisticas del dataset y pueden hacerse antes del split.

Ejemplos:

```text
filtrar CODE_GENDER a F/M
crear SENSITIVE
crear flags de missing de EXT_SOURCE
crear EXT_NULL_COUNT
tratar DAYS_EMPLOYED == 365243
convertir dias a anos
mapear FLAG_OWN_CAR
```

Las transformaciones estadisticas si aprenden parametros y deben ajustarse solo con train.

Ejemplos:

```text
medianas de imputacion
modas de imputacion
RobustScaler
categorias del OneHotEncoder
```

Por eso el flujo real queda asi:

```text
1. Bloque 1 declara contrato de datos
2. Bloque 2 carga CSV y aplica transformaciones deterministas
3. Bloque 3 crea split train/validation/test
4. Bloque 2 ajusta imputadores/scaler/encoder solo con train
5. Bloque 2 transforma train/validation/test con el mismo preprocesador
```

## Codigo creado

Modulo:

```text
src/preprocessing.py
```

Clases principales:

```text
HomeCreditRawDataLoader
HomeCreditDeterministicTransformer
HomeCreditPreprocessingColumnSpecFactory
HomeCreditFeaturePreprocessor
HomeCreditMVPPreprocessingPipeline
```

Contenedores de datos:

```text
DeterministicDataset
RawSplitDataset
PreprocessingColumnSpec
ProcessedSplitDataset
```

Excepcion propia:

```text
PreprocessingError
```

## `HomeCreditRawDataLoader`

Responsabilidad:

```text
cargar application_train.csv usando solo las columnas del contrato
```

Metodo principal:

```python
load(path=None) -> pd.DataFrame
```

Que hace:

1. Usa `contract.training_file_path()` si no se pasa ruta.
2. Comprueba que el archivo existe.
3. Lee solo `contract.required_raw_columns()`.
4. Valida que las columnas leidas cumplen el contrato.

Por que existe:

Para que ningun notebook lea columnas distintas por accidente.

## `HomeCreditDeterministicTransformer`

Responsabilidad:

```text
aplicar transformaciones deterministas antes del split
```

Metodo principal:

```python
transform(raw_df) -> DeterministicDataset
```

Transformaciones aplicadas:

```text
validar columnas obligatorias
filtrar CODE_GENDER en F/M
crear SENSITIVE
set_index(SK_ID_CURR)
crear EXT_SOURCE_i_WAS_MISSING
crear EXT_NULL_COUNT
crear DAYS_EMPLOYED_ANOM
reemplazar DAYS_EMPLOYED == 365243 por NaN
crear AGE_YEARS
crear EMPLOYED_YEARS
mapear FLAG_OWN_CAR: N -> 0, Y -> 1
validar importes financieros no negativos
eliminar DAYS_BIRTH y DAYS_EMPLOYED
separar X, y, s
```

Salida:

```python
DeterministicDataset(
    features=...,
    target=...,
    sensitive=...,
)
```

Importante:

`CODE_GENDER` y `SENSITIVE` no quedan dentro de `features`.

## `DeterministicDataset`

Contenedor inmutable con:

```text
features
target
sensitive
```

`features` sigue siendo un `pd.DataFrame` porque todavia necesitamos indices, nombres de columnas y trazabilidad.

`target` y `sensitive` son `pd.Series` alineadas con `features`.

## Transformaciones deterministas en detalle

### Filtrado de genero

Codigo real:

```text
CODE_GENDER solo F/M
```

Motivo:

La practica trabaja con variable sensible binaria. Las filas `XNA` se eliminan antes de crear `SENSITIVE`.

### Creacion de `SENSITIVE`

Codificacion:

```text
F -> 0.0
M -> 1.0
```

`SENSITIVE` se usara despues para:

```text
fairness loss
abs_rho
DPD
EOD
estratificacion TARGET + SENSITIVE
```

No se usa como feature normal.

### Indice `SK_ID_CURR`

`SK_ID_CURR` se convierte en indice.

Motivo:

Permite saber que clientes estan en cada split y guardar predicciones trazables.

No entra en `X`.

### Missingness de `EXT_SOURCE`

Columnas creadas:

```text
EXT_SOURCE_1_WAS_MISSING
EXT_SOURCE_2_WAS_MISSING
EXT_SOURCE_3_WAS_MISSING
EXT_NULL_COUNT
```

Motivo:

Los valores ausentes de `EXT_SOURCE` son una senal importante para la parte de incertidumbre.

La regla es:

```text
crear flags antes de imputar
```

### `DAYS_EMPLOYED`

Valor anomalo:

```text
365243
```

Tratamiento:

```text
DAYS_EMPLOYED_ANOM = 1 si DAYS_EMPLOYED == 365243
DAYS_EMPLOYED pasa a NaN en esos casos
```

Motivo:

Ese valor no representa dias reales de empleo.

### Dias a anos

Columnas creadas:

```text
AGE_YEARS = -DAYS_BIRTH / 365.25
EMPLOYED_YEARS = -DAYS_EMPLOYED / 365.25
```

Despues se eliminan:

```text
DAYS_BIRTH
DAYS_EMPLOYED
```

### `FLAG_OWN_CAR`

Mapeo:

```text
N -> 0.0
Y -> 1.0
```

Si aparece otro valor, se lanza `PreprocessingError`.

### Importes financieros

Se valida que sean no negativos:

```text
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
```

Decision activa:

```text
no log1p global
no RobustScaler
```

Los importes se mantienen en escala original para que el Bloque 5 pueda calcular ratios financieros interpretables.

## `PreprocessingColumnSpec`

Define las columnas de la fase estadistica:

```text
financial_cols
continuous_scaled_cols
binary_cols
categorical_cols
```

Esta clase evita que el `ColumnTransformer` tenga listas magicas dispersas.

## `HomeCreditPreprocessingColumnSpecFactory`

Responsabilidad:

```text
crear PreprocessingColumnSpec a partir del contrato del Bloque 1
```

Metodo:

```python
build() -> PreprocessingColumnSpec
```

Grupos creados:

### `financial_cols`

```text
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
```

Tratamiento:

```text
SimpleImputer(strategy="median")
sin escalado
```

### `continuous_scaled_cols`

```text
AGE_YEARS
EMPLOYED_YEARS
EXT_SOURCE_1
EXT_SOURCE_2
EXT_SOURCE_3
EXT_NULL_COUNT
REGION_RATING_CLIENT_W_CITY
CNT_CHILDREN
```

Tratamiento:

```text
SimpleImputer(strategy="median")
RobustScaler()
```

### `binary_cols`

```text
EXT_SOURCE_1_WAS_MISSING
EXT_SOURCE_2_WAS_MISSING
EXT_SOURCE_3_WAS_MISSING
DAYS_EMPLOYED_ANOM
FLAG_OWN_CAR
```

Tratamiento:

```text
SimpleImputer(strategy="most_frequent")
sin escalado
```

### `categorical_cols`

```text
NAME_EDUCATION_TYPE
NAME_FAMILY_STATUS
NAME_INCOME_TYPE
```

Tratamiento:

```text
SimpleImputer(strategy="constant", fill_value="MISSING")
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

## `HomeCreditFeaturePreprocessor`

Responsabilidad:

```text
ajustar y aplicar el ColumnTransformer sin leakage
```

Metodos principales:

```python
fit(X_train)
transform(X)
fit_transform_splits(raw_splits)
financial_feature_indices()
```

Punto critico:

```python
self._preprocessor.fit(X_train)
```

solo se ejecuta con train.

Validation y test se transforman con:

```python
self._preprocessor.transform(...)
```

Esto evita que medianas, escalas o categorias se aprendan usando validation/test.

## `RawSplitDataset`

Contenedor que recibira el output del Bloque 3:

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

El Bloque 2 no crea este split. Solo sabe procesarlo cuando ya existe.

## `ProcessedSplitDataset`

Contenedor final del Bloque 2:

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
train_ids
val_ids
test_ids
feature_names
preprocessor
```

Las matrices `X_*` son `np.ndarray` en `float32`.

`feature_names` conserva el orden exacto tras el `ColumnTransformer`.

## Feature names

El metodo interno `_build_feature_names()` genera el orden:

```text
financial_cols
continuous_scaled_cols
binary_cols
one-hot categorical names
```

Esto es fundamental para el Bloque 5.

La capa custom necesitara indices de:

```text
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
```

Para eso existe:

```python
financial_feature_indices()
```

que devuelve:

```python
{
    "AMT_INCOME_TOTAL": indice,
    "AMT_CREDIT": indice,
    "AMT_ANNUITY": indice,
    "AMT_GOODS_PRICE": indice,
}
```

## `HomeCreditMVPPreprocessingPipeline`

Facade de alto nivel.

Responsabilidad:

```text
conectar contrato, loader, transformador determinista y preprocesador estadistico
```

Metodos:

```python
load_raw(path=None)
apply_deterministic_transforms(raw_df)
build_feature_preprocessor()
fit_transform_splits(raw_splits)
```

Uso esperado:

```python
from src.preprocessing import HomeCreditMVPPreprocessingPipeline

pipeline = HomeCreditMVPPreprocessingPipeline()

raw_df = pipeline.load_raw()
deterministic = pipeline.apply_deterministic_transforms(raw_df)

# Bloque 3 hara el split:
# raw_splits = split_train_val_test(
#     deterministic.features,
#     deterministic.target,
#     deterministic.sensitive,
# )

processed = pipeline.fit_transform_splits(raw_splits)
```

## Por que el split no esta aqui

Porque el Bloque 3 debe ser auditable por separado.

El riesgo que evitamos:

```text
preprocesar todo el dataset
despues dividir
```

Eso introduciria leakage.

La arquitectura obliga a:

```text
deterministico -> split -> fit train -> transform val/test
```

## Validaciones implementadas

El codigo valida:

```text
columnas obligatorias
genero F/M
SENSITIVE binario
TARGET binario
SK_ID_CURR no duplicado
FLAG_OWN_CAR solo Y/N
importes financieros no negativos
columnas esperadas por ColumnTransformer
ausencia de NaN en matrices finales
```

## Que obtenemos al finalizar el Bloque 2

Codigo:

```text
src/preprocessing.py
src/__init__.py actualizado
```

Capacidades:

```text
cargar application_train.csv con columnas MVP
crear SENSITIVE
preservar SK_ID_CURR como indice
crear flags de EXT_SOURCE
crear EXT_NULL_COUNT
tratar DAYS_EMPLOYED anomalo
crear AGE_YEARS y EMPLOYED_YEARS
mapear FLAG_OWN_CAR
separar X, y, s
crear ColumnTransformer sin leakage
generar matrices float32
conservar feature_names
obtener indices financieros para Bloque 5
```

## Que NO hace este bloque

No hace:

```text
split train/validation/test
class_weight
modelo base
capas custom
FAIR loss
Keras Tuner
incertidumbre
```

## Relacion con Bloque 1

Bloque 2 depende de:

```text
HomeCreditMVPDataContract
DataContractValidator
```

Esto evita duplicar columnas.

## Relacion con Bloque 3

Bloque 2 produce:

```text
DeterministicDataset
```

Bloque 3 debe convertirlo en:

```text
RawSplitDataset
```

Despues Bloque 2 recibe `RawSplitDataset` y devuelve:

```text
ProcessedSplitDataset
```

## Relacion con Bloque 5

Bloque 5 usara:

```text
processed.feature_names
preprocessor.financial_feature_indices()
```

para localizar las columnas financieras dentro de la matriz final.

## Criterio de terminado

Bloque 2 queda terminado cuando:

```text
src/preprocessing.py existe
HomeCreditRawDataLoader carga columnas MVP
HomeCreditDeterministicTransformer crea X/y/s
HomeCreditFeaturePreprocessor ajusta solo con train
ProcessedSplitDataset contiene arrays float32
feature_names se conserva
el modulo compila
hay smoke test con datos sinteticos
este .md describe el codigo real
```


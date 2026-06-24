# Bloque 1 - Contrato de datos del MVP

## Objetivo del bloque

El objetivo del Bloque 1 es convertir las decisiones de datos del MVP en codigo Python reutilizable.

Antes, el bloque solo describia que archivo y columnas ibamos a usar. Ahora esa decision esta implementada en:

```text
src/data_contract.py
src/__init__.py
```

El Bloque 1 no preprocesa, no imputa, no divide train/validation/test y no entrena modelos. Su unica responsabilidad es declarar y validar el contrato de datos del MVP.

En terminos practicos, este bloque responde a:

```text
Que CSV es obligatorio?
Que columnas son obligatorias?
Cual es TARGET?
Cual es la variable sensible?
Que columnas no pueden entrar como features normales?
Que tablas quedan fuera del MVP?
Como validamos que el input cumple el contrato?
```

## Codigo creado

Se ha creado el modulo:

```text
src/data_contract.py
```

Tambien se ha creado:

```text
src/__init__.py
```

para exponer las clases principales del paquete `src`.

## Clases implementadas

### `DatasetFileSpec`

Clase inmutable que describe un archivo CSV del dataset.

Contiene:

```text
file_name
role
required_for_mvp
has_target
notes
```

Ejemplo conceptual:

```python
DatasetFileSpec(
    file_name="application_train.csv",
    role="mvp_train_and_internal_evaluation",
    required_for_mvp=True,
    has_target=True,
)
```

Por que existe:

Evita que la informacion sobre los archivos quede dispersa en strings sueltos por notebooks o scripts.

### `ColumnGroup`

Clase inmutable que agrupa columnas por responsabilidad.

Ejemplos de grupos:

```text
financial
temporal
external_scores
categorical_low_cardinality
simple_numeric_and_binary
```

Por que existe:

Permite hablar de conjuntos logicos de columnas sin duplicar listas en varios sitios.

### `DataContractValidationResult`

Clase de resultado para validaciones no destructivas.

Contiene:

```text
is_valid
missing_required_columns
unexpected_columns
checked_columns
message
```

Por que existe:

Permite validar columnas sin lanzar excepcion inmediatamente. Esto es util en notebooks, donde queremos imprimir un diagnostico antes de decidir si parar.

### `DataContractError`

Excepcion propia del proyecto para errores de contrato.

Se lanza cuando:

```text
falta application_train.csv
faltan columnas obligatorias
se pide un grupo de columnas inexistente
```

Por que existe:

Un error propio es mas claro que dejar que falle pandas, sklearn o TensorFlow mas adelante con mensajes menos conectados al proyecto.

### `HomeCreditMVPDataContract`

Clase central del Bloque 1.

Declara:

```text
directorio raw por defecto: data/raw
archivo MVP obligatorio: application_train.csv
archivo test oficial: application_test.csv
tablas relacionales fuera del MVP
columna identificadora: SK_ID_CURR
columna objetivo: TARGET
columna sensible raw: CODE_GENDER
columna sensible derivada: SENSITIVE
grupos de features del MVP
columnas derivadas que creara el Bloque 2
```

Esta clase es inmutable porque el contrato del MVP no deberia cambiar accidentalmente durante una ejecucion.

### `DataContractValidator`

Clase responsable de validar el contrato.

Metodos principales:

```text
validate_training_file_exists()
assert_training_file_exists()
validate_columns(...)
assert_columns(...)
validate_mapping_keys(...)
```

Por que se separa del contrato:

Aplicamos responsabilidad unica. El contrato declara las reglas; el validador comprueba si algo las cumple.

### `DataContractReporter`

Clase responsable de generar resumenes legibles del contrato.

Metodos:

```text
summary_lines()
as_text()
```

Por que existe:

Sirve para imprimir en notebooks un resumen corto de que columnas y archivos esta usando el MVP.

### Factories

Se han anadido dos funciones de conveniencia:

```python
build_default_home_credit_contract(...)
build_default_contract_validator(...)
```

Por que existen:

Permiten crear el contrato o su validador desde notebooks o scripts sin repetir detalles internos.

## Decision principal del MVP

El MVP usa solo:

```text
data/raw/application_train.csv
```

Motivo:

Es el unico archivo principal que contiene:

```text
TARGET
CODE_GENDER
variables financieras
EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
```

Con eso se pueden cumplir las cuatro tareas obligatorias:

```text
arquitectura custom
FAIR loss
Keras Tuner
incertidumbre
```

## Por que `application_test.csv` no evalua el MVP

`application_test.csv` esta registrado en el contrato, pero con:

```text
required_for_mvp = False
has_target = False
role = official_kaggle_inference_only
```

Esto significa:

```text
puede usarse para inferencia opcional
no puede usarse para calcular metricas finales
```

Sin `TARGET` no podemos calcular:

```text
AUC
Precision
Recall
F1
error real
Equalized Odds
incertidumbre comparada con error observado
```

Por tanto, el test evaluable saldra de un holdout interno de `application_train.csv`, que se implementara en Bloque 3.

## Tablas relacionales fuera del MVP

El contrato registra como fase avanzada:

```text
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
POS_CASH_balance.csv
credit_card_balance.csv
```

Todas tienen:

```text
required_for_mvp = False
has_target = False
role = advanced_relational_enrichment
```

Por que se registran aunque no se usen:

Para dejar claro que no estan olvidadas. Quedan conscientemente fuera del MVP y se reservan para el camino al 10.

## Columnas obligatorias

El metodo:

```python
HomeCreditMVPDataContract.required_raw_columns()
```

devuelve:

```text
SK_ID_CURR
TARGET
CODE_GENDER
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
DAYS_BIRTH
DAYS_EMPLOYED
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

Estas son las columnas que deberia leer el MVP desde `application_train.csv`.

## Columna objetivo

La columna objetivo esta definida como:

```python
target_column = "TARGET"
```

Interpretacion:

```text
TARGET = 1 -> cliente con dificultades de pago
TARGET = 0 -> cliente que pago a tiempo
```

No entra como feature. Es la `y` que la red intenta predecir.

## Variable sensible

La variable sensible raw es:

```python
sensitive_column = "CODE_GENDER"
```

En Bloque 2 se transformara en:

```python
engineered_sensitive_column = "SENSITIVE"
```

con la decision:

```text
F -> 0
M -> 1
XNA -> se filtra
```

`CODE_GENDER` y `SENSITIVE` no deben entrar como features normales.

Se usaran para:

```text
penalizacion FAIR
abs_rho
DPD
EOD
estratificacion TARGET + SENSITIVE
```

## Columnas excluidas del modelo

El metodo:

```python
excluded_from_model_columns()
```

devuelve:

```text
SK_ID_CURR
TARGET
CODE_GENDER
SENSITIVE
```

Motivo:

```text
SK_ID_CURR -> identificador
TARGET -> etiqueta
CODE_GENDER -> sensible raw
SENSITIVE -> sensible numerica
```

Estas columnas no deben concatenarse a `X` como variables predictoras normales.

## Grupos de features del MVP

### Grupo `financial`

Columnas:

```text
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
```

Uso futuro:

Estas columnas se mantendran en escala original imputada para que la capa custom del Bloque 5 pueda calcular ratios financieros interpretables.

Importante:

```text
No se aplicara log1p global.
No se aplicara RobustScaler a estas columnas financieras.
```

### Grupo `temporal`

Columnas:

```text
DAYS_BIRTH
DAYS_EMPLOYED
```

Uso futuro:

En Bloque 2 se transformaran a:

```text
AGE_YEARS
EMPLOYED_YEARS
DAYS_EMPLOYED_ANOM
```

### Grupo `external_scores`

Columnas:

```text
EXT_SOURCE_1
EXT_SOURCE_2
EXT_SOURCE_3
```

Uso futuro:

Son variables muy predictivas y ademas son centrales para la parte de incertidumbre, porque su missingness se analizara despues.

### Grupo `categorical_low_cardinality`

Columnas:

```text
NAME_EDUCATION_TYPE
NAME_FAMILY_STATUS
NAME_INCOME_TYPE
```

Uso futuro:

Se codificaran con one-hot en Bloque 2.

Nota de fairness:

`NAME_FAMILY_STATUS` puede actuar como proxy parcial de genero. Se mantiene porque permite demostrar que quitar `CODE_GENDER` no basta y justifica la penalizacion FAIR.

### Grupo `simple_numeric_and_binary`

Columnas:

```text
REGION_RATING_CLIENT_W_CITY
FLAG_OWN_CAR
CNT_CHILDREN
```

Uso futuro:

Refuerzan el MVP sin abrir todavia todo el dataset de Kaggle.

## Columnas derivadas documentadas

El contrato declara las columnas que el Bloque 2 debera crear:

```text
AGE_YEARS
EMPLOYED_YEARS
DAYS_EMPLOYED_ANOM
EXT_SOURCE_1_WAS_MISSING
EXT_SOURCE_2_WAS_MISSING
EXT_SOURCE_3_WAS_MISSING
EXT_NULL_COUNT
```

Estas columnas no se esperan en el CSV raw. Estan en el contrato para que el flujo completo sea explicito desde el principio.

## Uso basico desde Python

Ejemplo:

```python
from src.data_contract import (
    DataContractReporter,
    build_default_contract_validator,
    build_default_home_credit_contract,
)

contract = build_default_home_credit_contract()
validator = build_default_contract_validator()

print(DataContractReporter(contract).as_text())
```

## Validar que existe `application_train.csv`

Ejemplo:

```python
validator = build_default_contract_validator()
result = validator.validate_training_file_exists()
print(result.message)
```

Si queremos que el pipeline falle directamente:

```python
validator.assert_training_file_exists()
```

Si el archivo no existe, lanza:

```python
DataContractError
```

## Validar columnas leidas de un CSV

Ejemplo:

```python
import pandas as pd

contract = build_default_home_credit_contract()
validator = build_default_contract_validator()

df = pd.read_csv(
    contract.training_file_path(),
    usecols=contract.required_raw_columns(),
)

validator.assert_columns(df.columns)
```

Modo no estricto:

```python
validator.validate_columns(df.columns, strict=False)
```

En modo no estricto, no pasa nada si el DataFrame tiene columnas extra. Solo fallan las columnas obligatorias ausentes.

Modo estricto:

```python
validator.validate_columns(df.columns, strict=True)
```

En modo estricto, tambien se reportan columnas fuera del contrato.

## Validar sin pandas

Para tests unitarios o pruebas pequenas:

```python
contract = build_default_home_credit_contract()
validator = build_default_contract_validator()

fake_columns = {column: None for column in contract.required_raw_columns()}
result = validator.validate_mapping_keys(fake_columns)

assert result.is_valid
```

## Principios de arquitectura aplicados

### Responsabilidad unica

Cada clase tiene una funcion:

```text
HomeCreditMVPDataContract -> declara reglas
DataContractValidator -> valida reglas
DataContractReporter -> resume reglas
DatasetFileSpec -> describe archivos
ColumnGroup -> agrupa columnas
DataContractValidationResult -> transporta resultados
```

### Inmutabilidad

Las clases de contrato usan:

```python
@dataclass(frozen=True)
```

Esto evita que otro modulo cambie accidentalmente columnas o archivos durante una ejecucion.

### Bajo acoplamiento

El Bloque 1 no importa TensorFlow, sklearn ni Keras.

Motivo:

El contrato de datos debe poder cargarse rapido y no depender de librerias pesadas de modelado.

### Fallar pronto

El validador permite parar el pipeline si falta:

```text
application_train.csv
alguna columna obligatoria
```

Esto evita errores mas tarde en preprocesamiento o entrenamiento.

## Que se obtiene al finalizar el Bloque 1

Artefactos creados:

```text
src/data_contract.py
src/__init__.py
docs/bloque_01_data_contract_mvp.md
```

Capacidades nuevas:

```text
declarar contrato MVP
consultar columnas obligatorias
consultar columnas excluidas del modelo
consultar grupos de features
validar existencia del CSV principal
validar columnas disponibles
generar resumen legible del contrato
```

## Que NO hace este bloque

Este bloque no:

```text
filtra CODE_GENDER == XNA
crea SENSITIVE
crea EXT_NULL_COUNT
imputa nulos
escala variables
hace split train/validation/test
entrena modelos
calcula fairness
```

Todo eso empieza en Bloque 2 y posteriores.

## Riesgos evitados

### Riesgo 1 - Usar el CSV equivocado

Mitigacion:

`training_file_path()` centraliza la ruta esperada de `application_train.csv`.

### Riesgo 2 - Evaluar con `application_test.csv`

Mitigacion:

El contrato marca `application_test.csv` como `has_target=False`.

### Riesgo 3 - Meter `CODE_GENDER` en X

Mitigacion:

`excluded_from_model_columns()` declara que `CODE_GENDER` y `SENSITIVE` no son features normales.

### Riesgo 4 - Olvidar columnas clave para incertidumbre

Mitigacion:

`external_score_columns()` y `derived_columns` documentan desde el principio la familia `EXT_SOURCE`.

### Riesgo 5 - Duplicar listas de columnas

Mitigacion:

Las columnas se declaran una vez en `HomeCreditMVPDataContract` y se consultan mediante metodos.

## Criterio de terminado

El Bloque 1 queda terminado cuando:

```text
src/data_contract.py existe
src/__init__.py exporta las clases principales
HomeCreditMVPDataContract declara archivos y columnas
DataContractValidator valida archivo y columnas
DataContractReporter genera resumen
el modulo compila sin errores de sintaxis
este .md describe el codigo real
```

## Siguiente bloque

El siguiente paso sera Bloque 2:

```text
Preprocesamiento sin leakage
```

Ese bloque usara el contrato del Bloque 1 para leer columnas correctas y empezar a construir:

```text
X
y
s
flags de missing
variables temporales transformadas
pipeline de imputacion/escalado/codificacion
```


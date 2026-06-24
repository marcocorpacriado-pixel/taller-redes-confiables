# Bloque 5 - Capas customizadas de Keras

## Estado del bloque

Este bloque ya esta implementado en codigo.

Archivos creados:

```text
src/layers.py
src/models.py
```

Archivo actualizado:

```text
src/__init__.py
```

El objetivo de este bloque es cumplir la Tarea 1 del enunciado: introducir una arquitectura customizada en Keras con conocimiento financiero explicable.

La decision tecnica definitiva del MVP es la siguiente:

```text
Features procesadas del Bloque 2
-> ratios financieros sobre importes originales imputados
-> gamma entrenable sobre los ratios
-> BatchNormalization
-> capas densas
-> probabilidad TARGET=1
```

Esta ruta es la version corregida y coherente de la arquitectura custom. No mezcla transformaciones contradictorias: los ratios se calculan donde tienen interpretacion financiera, y la gamma se aplica despues para saturar suavemente ratios extremos.

## Por que este bloque existe

Una MLP normal podria aprender relaciones entre variables financieras, pero tendria que descubrir por si sola relaciones como:

```text
credito / ingresos
anualidad / ingresos
credito / precio del bien
anualidad / credito
```

Estas relaciones son conocidas en scoring financiero. Por eso tiene sentido incorporarlas como una capa customizada:

1. La red recibe conocimiento de dominio desde el inicio.
2. Las nuevas variables son interpretables para el PDF y la defensa.
3. Se cumple el requisito de crear una capa customizada en Keras.
4. La arquitectura queda preparada para que el Bloque 6 anada la penalizacion FAIR sin cambiar el resto del modelo.

## Decision matematica: Ruta C

La ruta implementada es:

```text
importes originales imputados -> ratios -> gamma -> BatchNorm
```

Las cuatro variables monetarias usadas por la capa son:

```text
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY
AMT_GOODS_PRICE
```

Estas columnas salen del Bloque 2:

```text
imputadas con mediana aprendida solo en train
en escala monetaria original
sin RobustScaler
sin log1p
```

Esto es importante porque un ratio financiero solo conserva su significado si se calcula sobre magnitudes comparables en su escala original. Por ejemplo:

```text
AMT_CREDIT / AMT_INCOME_TOTAL
```

se puede interpretar como deuda solicitada relativa a ingresos. Si ambas columnas estuvieran robust-scaled, el cociente ya no tendria esa lectura.

## Archivo `src/layers.py`

Este archivo contiene las capas y utilidades de indices del Bloque 5.

### `CustomLayerError`

Excepcion especifica del bloque.

Se lanza cuando una capa o resolver detecta una configuracion insegura:

```text
indices negativos
indices duplicados
columnas financieras ausentes
input_dim desconocido
gamma fuera de rango
eps no positivo
clip_value no positivo
```

Usar una excepcion propia ayuda a diferenciar errores de arquitectura de errores genericos de TensorFlow.

### `FinancialRatioIndices`

Dataclass inmutable que guarda los cuatro indices financieros:

```python
FinancialRatioIndices(
    idx_credit=...,
    idx_annuity=...,
    idx_income=...,
    idx_goods=...,
)
```

Tiene dos metodos utiles:

```python
as_layer_kwargs()
as_tuple()
```

`as_layer_kwargs()` devuelve los nombres exactos que espera `FinancialRatiosLayer`, evitando repetir mapeos manuales.

`as_tuple()` facilita validaciones de rango y duplicados.

### `FinancialRatioIndexResolver`

Clase responsable de convertir `feature_names` en indices numericos seguros.

La clase usa este mapeo por defecto:

```python
{
    "credit": "AMT_CREDIT",
    "annuity": "AMT_ANNUITY",
    "income": "AMT_INCOME_TOTAL",
    "goods": "AMT_GOODS_PRICE",
}
```

Uso esperado:

```python
resolver = FinancialRatioIndexResolver()
indices = resolver.resolve(processed.feature_names)
```

La validacion comprueba:

1. `feature_names` no esta vacio.
2. No hay nombres de columnas duplicados.
3. Las cuatro columnas financieras existen.
4. Los indices son distintos.
5. Cada indice apunta al nombre esperado.

Este resolver es una pieza critica porque una capa Keras recibe tensores, no nombres de columnas. Si los indices se calculan mal, el modelo podria dividir columnas incorrectas sin que TensorFlow lo detectara. Por eso el fallo se fuerza antes de entrenar.

### `FinancialRatiosLayer`

Capa Keras registrada como serializable:

```python
@tf.keras.utils.register_keras_serializable(package="HomeCredit")
class FinancialRatiosLayer(tf.keras.layers.Layer):
    ...
```

Entrada:

```text
(batch, n_features)
```

Salida:

```text
(batch, n_features + 4)
```

La capa conserva todas las features originales y concatena cuatro ratios:

```text
CREDIT / INCOME
ANNUITY / INCOME
CREDIT / GOODS
ANNUITY / CREDIT
```

Interpretacion:

```text
CREDIT / INCOME      -> deuda solicitada relativa a ingresos
ANNUITY / INCOME     -> esfuerzo periodico de pago
CREDIT / GOODS       -> proporcion financiada del bien
ANNUITY / CREDIT     -> intensidad de pago respecto al principal
```

#### Estabilidad numerica

La capa aplica tres protecciones:

```text
tf.maximum(x, 0.0)
denominador + eps
tf.clip_by_value(ratio, 0.0, clip_value)
```

Valores por defecto:

```text
eps = 1.0
clip_value = 10.0
```

`eps=1.0` tiene sentido porque las columnas monetarias estan en unidades originales. Sumar 1 al denominador evita divisiones por cero sin modificar de forma relevante importes normales.

`clip_value=10.0` evita que outliers extremos monopolicen el gradiente. La senal de que el ratio es extremo se conserva, pero queda acotada.

#### Serializacion

La capa implementa:

```python
get_config()
compute_output_shape()
```

Esto permite guardar modelos `.keras` y cargarlos despues con `custom_objects` si fuera necesario.

### `TrainableGammaLayer`

Capa Keras registrada como serializable:

```python
@tf.keras.utils.register_keras_serializable(package="HomeCredit")
class TrainableGammaLayer(tf.keras.layers.Layer):
    ...
```

Entrada:

```text
(batch, n_features + 4)
```

Salida:

```text
(batch, n_features + 8)
```

La capa selecciona columnas no negativas y concatena una version transformada:

```text
z = (x + epsilon) ^ gamma
```

En nuestro flujo, las columnas seleccionadas son solo los cuatro ratios creados por `FinancialRatiosLayer`.

Si la matriz original tiene `N` features, los ratios se encuentran en:

```python
ratio_feature_indices = tuple(range(N, N + 4))
```

Por tanto:

```text
entrada original               -> N
despues de FinancialRatiosLayer -> N + 4
despues de TrainableGammaLayer  -> N + 8
```

#### Parametrizacion de gamma

La gamma no se entrena directamente. Se entrena un parametro libre `theta`, y gamma se calcula como:

```text
gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(theta)
```

Valores por defecto:

```text
gamma_min = 0.1
gamma_max = 1.5
theta_init = 0.588
l2_reg = 1e-4
epsilon = 1e-6
```

Esta parametrizacion garantiza:

```text
0.1 < gamma < 1.5
```

Interpretacion:

```text
gamma < 1 -> comprime ratios altos
gamma = 1 -> transformacion casi neutra
gamma > 1 -> expande diferencias
```

`theta_init=0.588` inicia gamma aproximadamente en 1. En el smoke test se obtuvo:

```text
gamma_values = [1.0001, 1.0001, 1.0001, 1.0001]
```

Eso significa que el modelo empieza casi neutro y solo aprende saturacion si los datos lo justifican.

### `custom_layer_objects()`

Funcion auxiliar para cargar modelos guardados:

```python
tf.keras.models.load_model(
    "results/models/custom_financial_mlp.keras",
    custom_objects=custom_layer_objects(),
)
```

Aunque las capas estan registradas con `register_keras_serializable`, esta funcion hace la carga mas explicita en notebooks y scripts.

## Archivo `src/models.py`

Este archivo contiene el constructor POO del modelo custom completo.

### `CustomModelError`

Excepcion especifica de construccion de modelos custom.

Se lanza cuando:

```text
hidden_units esta vacio
dropout esta fuera de [0, 1)
learning_rate no es positivo
gradient_clipnorm no es positivo
input_dim no es positivo
ratio_indices exceden input_dim
```

### `CustomMLPConfig`

Dataclass inmutable con toda la configuracion del modelo custom.

Valores por defecto:

```python
CustomMLPConfig(
    hidden_units=(128, 64),
    activation="elu",
    dropout=0.2,
    learning_rate=1e-3,
    gradient_clipnorm=1.0,
    loss="binary_crossentropy",
    ratio_eps=1.0,
    ratio_clip_value=10.0,
    gamma_min=0.1,
    gamma_max=1.5,
    theta_init=0.588,
    gamma_l2_reg=1e-4,
    gamma_epsilon=1e-6,
)
```

La configuracion mantiene los mismos defaults controlados del Bloque 4:

```text
BCE
Adam
clipnorm=1.0
AUC
PR-AUC
BinaryAccuracy
Precision
Recall
```

Esto es deliberado: el modelo base custom y el modelo FAIR posterior deben diferir solo en la penalizacion de fairness, no en metricas, optimizador o arquitectura base.

### `CustomModelBuildResult`

Dataclass que devuelve el builder.

Campos:

```python
model
ratio_indices
ratio_feature_indices
output_feature_count_after_custom_block
```

Esto permite auditar que el modelo se ha construido con los indices correctos.

### `CustomProbabilityGraph`

Dataclass introducida en el Bloque 5.5 para preparar el Bloque 6 sin duplicar arquitectura.

Campos:

```python
features_input
probability_output
ratio_indices
ratio_feature_indices
output_feature_count_after_custom_block
```

Representa el backbone comun:

```text
features -> ratios -> gamma -> BatchNorm -> densas -> prob
```

El modelo FAIR del Bloque 6 podra envolver `probability_output` con una capa de penalizacion FAIR usando un segundo input `sensitive`, sin copiar las capas financieras ni la MLP.

### `CustomMLPModelBuilder`

Clase principal del Bloque 5.

Responsabilidades:

1. Validar `CustomMLPConfig`.
2. Resolver indices financieros desde `feature_names`.
3. Construir la arquitectura custom.
4. Compilar el modelo Keras.
5. Devolver trazabilidad de indices y shapes.
6. Exponer el backbone comun mediante `build_probability_graph`.

Uso recomendado:

```python
from src.models import CustomMLPModelBuilder

builder = CustomMLPModelBuilder()
build_result = builder.build_from_feature_names(processed.feature_names)
model = build_result.model
```

Arquitectura creada:

```text
Input(name="features")
-> FinancialRatiosLayer(name="financial_ratios")
-> TrainableGammaLayer(name="ratio_gamma")
-> BatchNormalization(name="post_custom_batchnorm")
-> Dense(name="dense_0")
-> Dropout(name="dropout_0")
-> Dense(name="dense_1")
-> Dropout(name="dropout_1")
-> Dense(1, sigmoid, name="prob")
```

El modelo se llama:

```text
custom_financial_mlp
```

### `build_from_feature_names(feature_names)`

Metodo de alto nivel.

Hace:

1. Comprueba que `feature_names` no esta vacio.
2. Usa `FinancialRatioIndexResolver`.
3. Construye el modelo con `input_dim=len(feature_names)`.
4. Devuelve `CustomModelBuildResult`.

Este es el metodo que deberia usarse en el pipeline normal.

### `build(input_dim, ratio_indices)`

Metodo de bajo nivel.

Sirve para tests o casos donde los indices ya se conocen.

Valida que:

```text
input_dim > 0
indices >= 0
max(indices) < input_dim
```

Despues construye el mismo grafo Keras.

### `build_probability_graph(features_input, input_dim, ratio_indices)`

Metodo publico para construir la arquitectura comun hasta la probabilidad sigmoid.

Uso previsto para Bloque 6:

```python
features_in = tf.keras.Input(shape=(input_dim,), name="features")
graph = builder.build_probability_graph(
    features_input=features_in,
    input_dim=input_dim,
    ratio_indices=ratio_indices,
)
```

Despues el modelo FAIR podra crear:

```python
sensitive_in = tf.keras.Input(shape=(1,), name="sensitive")
fair_output = FairnessPenalty(lambda_fair)([graph.probability_output, sensitive_in])
```

Esto garantiza que modelo base y modelo FAIR comparten exactamente la misma arquitectura predictiva.

### `compile_model(model)`

Metodo publico que compila un modelo con los mismos settings controlados:

```text
Adam
clipnorm=1.0
BCE
AUC
PR-AUC
BinaryAccuracy
Precision
Recall
```

El Bloque 6 debe usar este metodo para que el FAIR model no tenga una configuracion distinta por accidente.

### `custom_model_objects()`

Funcion auxiliar para carga de modelos guardados.

Internamente devuelve las clases custom de `src.layers`.

## Archivo `src/__init__.py`

Se actualizo para exportar las clases de Bloque 5 si TensorFlow esta instalado.

Exports nuevos:

```text
CustomLayerError
FinancialRatioIndexResolver
FinancialRatioIndices
FinancialRatiosLayer
TrainableGammaLayer
custom_layer_objects
CustomMLPConfig
CustomMLPModelBuilder
CustomModelBuildResult
CustomModelError
CustomProbabilityGraph
custom_model_objects
```

La importacion sigue siendo opcional para objetos TensorFlow. Esto mantiene usable el paquete para tareas de datos aunque TensorFlow no este instalado.

## Flujo real de uso

Ejemplo minimo:

```python
from src.models import CustomMLPModelBuilder

builder = CustomMLPModelBuilder()
build_result = builder.build_from_feature_names(processed.feature_names)

model = build_result.model

history = model.fit(
    processed.X_train,
    processed.y_train,
    validation_data=(processed.X_val, processed.y_val),
    epochs=10,
    batch_size=1024,
)
```

Para auditar indices:

```python
print(build_result.ratio_indices)
print(build_result.ratio_feature_indices)
print(build_result.output_feature_count_after_custom_block)
```

## Shape esperado

Si el Bloque 2 produce `N` features:

```text
Input                         -> N
FinancialRatiosLayer          -> N + 4
TrainableGammaLayer           -> N + 8
BatchNormalization            -> N + 8
Dense stack                   -> segun config
Output                        -> 1
```

En el smoke test ejecutado con datos sinteticos:

```text
feature_count = 26
ratio_shape = (4, 30)
gamma_shape = (4, 34)
model_pred_shape = (3, 1)
ratio_feature_indices = (26, 27, 28, 29)
custom_block_output_dim = 34
```

Esto confirma que:

```text
26 + 4 = 30
30 + 4 = 34
```

## Verificacion ejecutada

Se ejecuto compilacion de sintaxis con Python 3.11 del entorno:

```powershell
& "C:\venvs\homecredit311\Scripts\python.exe" -m py_compile `
    src\data_contract.py `
    src\preprocessing.py `
    src\splitting.py `
    src\base_model.py `
    src\layers.py `
    src\models.py `
    src\__init__.py
```

Resultado:

```text
sin errores
```

Tambien se ejecuto un smoke test con datos sinteticos que comprobo:

1. Transformaciones deterministas del Bloque 2.
2. Split estratificado del Bloque 3.
3. Preprocesado estadistico del Bloque 2.
4. Resolucion de indices financieros.
5. Forward pass de `FinancialRatiosLayer`.
6. Forward pass de `TrainableGammaLayer`.
7. Construccion de `CustomMLPModelBuilder`.
8. Prediccion del modelo.
9. Entrenamiento minimo de una epoca.

Resultado:

```text
sin errores
```

TensorFlow mostro warnings informativos/deprecations de su propia libreria. No son errores del proyecto.

## Explicacion matematica para el PDF

Texto defendible:

```text
La arquitectura incorpora una capa customizada que calcula ratios financieros clasicos sobre importes monetarios en escala original: credito/ingresos, anualidad/ingresos, credito/precio del bien y anualidad/credito. Estos ratios aproximan endeudamiento relativo, esfuerzo periodico de pago, proporcion financiada del bien e intensidad del pago. Para asegurar estabilidad numerica se anade un epsilon al denominador y se saturan ratios extremos mediante clipping.
```

Para la gamma:

```text
Sobre los ratios generados se aplica una segunda capa custom con exponentes entrenables. La transformacion es z = (x + epsilon)^gamma. La gamma se parametriza como gamma = 0.1 + 1.4 sigmoid(theta), garantizando que cada exponente permanezca entre 0.1 y 1.5. La inicializacion se elige cerca de gamma = 1 para empezar desde una transformacion casi neutra. Valores menores que 1 comprimen ratios extremos; valores mayores que 1 amplifican diferencias.
```

## Relacion con el Bloque 4

El Bloque 4 construia una MLP base sin conocimiento financiero explicito.

El Bloque 5 construye una MLP customizada:

```text
misma idea de clasificacion
misma BCE
mismas metricas
mismo Adam con clipnorm
pero con capas financieras al inicio
```

Esto permite comparar una arquitectura simple contra una arquitectura con conocimiento de dominio.

## Relacion con el Bloque 6

El Bloque 6 debe usar esta arquitectura como base.

La comparacion controlada sera:

```text
custom base: lambda_fair = 0
custom FAIR: lambda_fair > 0
```

La arquitectura, optimizador, metricas, callbacks y estrategia de desbalance deben permanecer iguales. Solo debe cambiar el termino de penalizacion FAIR.

## Relacion con Keras Tuner

En bloques posteriores, Keras Tuner podra variar:

```text
numero de capas densas
unidades por capa
activacion
dropout
learning rate
```

Las capas custom no deben desaparecer del modelo porque son requisito central de la practica.

## Asunciones del bloque

1. `processed.feature_names` conserva el orden exacto de las columnas de `processed.X_*`.
2. Las columnas `AMT_*` financieras existen en `feature_names`.
3. Las columnas financieras estan imputadas pero no escaladas.
4. La variable sensible `CODE_GENDER` no entra como feature normal.
5. Los ratios generados son no negativos tras las defensas numericas.
6. Gamma se aplica solo a ratios, no a todas las features.
7. La salida del modelo es una probabilidad sigmoid para `TARGET=1`.

## Riesgos y mitigaciones

### Riesgo 1 - Indices financieros mal alineados

Si los indices apuntan a columnas incorrectas, los ratios pierden sentido.

Mitigacion implementada:

```text
FinancialRatioIndexResolver valida nombres, indices y duplicados antes de construir el modelo.
```

### Riesgo 2 - Denominadores pequenos

Un denominador cero o casi cero podria disparar ratios.

Mitigacion implementada:

```text
denominador + eps
clip_by_value
```

### Riesgo 3 - Ratios extremos

Outliers financieros podrian dominar gradientes.

Mitigacion implementada:

```text
clip_value=10.0
BatchNormalization
Adam(clipnorm=1.0)
```

### Riesgo 4 - Gamma inestable

Entrenar gamma directamente podria producir exponentes negativos o demasiado grandes.

Mitigacion implementada:

```text
gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(theta)
```

### Riesgo 5 - Modelo guardado no cargable

Las capas custom pueden fallar al cargar si Keras no conoce sus clases.

Mitigacion implementada:

```text
register_keras_serializable
get_config()
custom_layer_objects()
custom_model_objects()
```

## Que obtenemos al terminar el Bloque 5

Al finalizar este bloque tenemos:

1. `FinancialRatiosLayer`.
2. `TrainableGammaLayer`.
3. Resolver seguro de indices financieros.
4. Builder POO del modelo custom.
5. Configuracion inmutable del modelo custom.
6. Serializacion preparada para modelos `.keras`.
7. Smoke test de forward pass y entrenamiento minimo superado.
8. Base lista para el Bloque 6 FAIR Loss.

## Como explicarlo al equipo

Resumen corto:

```text
En el Bloque 5 hemos metido conocimiento financiero dentro de la red. La primera capa custom calcula ratios clasicos como credito/ingresos y anualidad/ingresos usando importes originales imputados. La segunda capa aprende una gamma sobre esos ratios para comprimir o expandir valores extremos de forma controlada. Despues normalizamos y pasamos a la MLP. Esto cumple el requisito de capa custom y deja una explicacion matematica limpia para la defensa.
```

## Criterio de terminado

El Bloque 5 se considera terminado porque:

```text
existe src/layers.py
existe src/models.py
FinancialRatiosLayer esta implementada
TrainableGammaLayer esta implementada
ambas capas tienen get_config
las capas estan registradas como serializables
CustomMLPModelBuilder construye y compila el modelo
los indices financieros se validan desde feature_names
la sintaxis compila
el smoke test entrena una epoca sin errores
```

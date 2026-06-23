# Bloque 6 - FAIR loss profesional con `add_loss`

## Objetivo del bloque

El objetivo de este bloque es implementar la Tarea 2 del taller: entrenar una red neuronal que mantenga una buena capacidad predictiva, pero que penalice la dependencia estadistica entre la prediccion del modelo y la variable sensible `CODE_GENDER`.

La decision final del proyecto es usar la Functional API de Keras con dos entradas:

```text
features  -> matriz procesada del Bloque 2
sensitive -> genero codificado como 0/1, separado del modelo predictivo
output    -> probabilidad P(TARGET=1)
```

La penalizacion de fairness se introduce con `self.add_loss()` dentro de una capa Keras propia. Esto evita el truco de apilar `TARGET` y `SENSITIVE` dentro de `y_true`, mantiene `y_true` como un vector normal de etiquetas binarias y permite que `class_weight`, `AUC`, `Precision`, `Recall` y los callbacks de Keras sigan funcionando de forma estandar.

## Archivos implementados o modificados

Este bloque ha anadido codigo real en cuatro zonas del proyecto:

```text
src/layers.py
    FairnessPenalty

src/models.py
    FairModelConfig
    FairModelBuildResult
    FairCustomModelBuilder
    build_fair_custom_model
    lambda_slug

src/__init__.py
    exportaciones publicas del Bloque 6

tests/test_fair_model.py
    tests unitarios y smoke tests especificos de FAIR loss
```

No se ha creado un segundo modelo paralelo ni una segunda copia de la arquitectura custom. Esto era un punto critico: el Bloque 5.5 preparo `CustomMLPModelBuilder.build_probability_graph()` precisamente para que el Bloque 6 pudiera envolver la arquitectura existente sin duplicarla.

## Idea matematica

La funcion objetivo final es:

```text
loss_total = BCE(y, y_hat) + lambda_fair * rho(y_hat, s)^2 + regularizadores
```

Donde:

```text
BCE          = binary crossentropy de clasificacion
y_hat        = probabilidad predicha por la red
s            = variable sensible, 0/1
rho          = correlacion de Pearson calculada dentro de cada batch
lambda_fair  = peso de la penalizacion FAIR
```

El cuadrado de la correlacion es importante. El objetivo no es que la correlacion sea negativa, sino que se acerque a cero. Si se minimizara `rho` directamente, la red podria intentar ir hacia `rho = -1`, que sigue siendo dependencia perfecta, solo que en sentido contrario.

En terminos de fairness, esta penalizacion aproxima Demographic Parity sobre la probabilidad predicha:

```text
rho(y_hat, S) cerca de 0  ->  menor dependencia lineal entre score y genero
```

Esto no garantiza justicia perfecta. Es una restriccion concreta y auditable sobre la relacion entre score y genero. Por eso los bloques posteriores tambien reportaran DPD y EOD.

## `FairnessPenalty`

La clase principal del bloque vive en `src/layers.py`:

```python
@tf.keras.utils.register_keras_serializable(package="HomeCredit")
class FairnessPenalty(tf.keras.layers.Layer):
    ...
```

Su contrato es deliberadamente simple:

```text
entrada: [y_pred, sensitive]
salida:  y_pred sin modificar
efecto:  anade lambda_fair * rho^2 a model.losses si lambda_fair > 0
```

Es una capa identidad. No cambia las probabilidades, no cambia el threshold y no altera la forma de salida del modelo. Su unica responsabilidad es inyectar una perdida adicional durante el entrenamiento.

La formula interna calcula la correlacion de Pearson de forma diferenciable:

```text
y_centered = y_pred - mean(y_pred)
s_centered = sensitive - mean(sensitive)

rho = mean(y_centered * s_centered)
      / sqrt(mean(y_centered^2) * mean(s_centered^2) + eps)
```

### Por que `eps` va dentro de la raiz

En entrenamiento se usa:

```text
sqrt(var_pred * var_sensitive + eps)
```

Esto es intencionado. La loss necesita ser diferenciable y no debe producir `NaN` si un batch tiene poca varianza en predicciones o en genero.

La metrica de evaluacion, en cambio, usa otra estrategia en `FairnessLogger` y `AbsolutePearsonCorrelation`: si el denominador es demasiado pequeno devuelve `0.0`; si no, divide sin anadir `eps`. Esta diferencia es correcta:

```text
training    -> prioriza estabilidad numerica y gradientes finitos
evaluacion  -> prioriza una metrica fiel y auditable
```

### Serializacion

`FairnessPenalty` implementa `get_config()` y esta registrada con:

```python
@tf.keras.utils.register_keras_serializable(package="HomeCredit")
```

Esto permite guardar y cargar modelos `.keras` que contienen la capa FAIR. La funcion `custom_layer_objects()` tambien incluye `FairnessPenalty`, por lo que los bloques de guardado y carga tienen acceso a todos los objetos custom.

## `FairModelConfig`

`FairModelConfig` vive en `src/models.py` y centraliza la configuracion propia del wrapper FAIR:

```python
FairModelConfig(
    lambda_fair=0.0,
    fairness_eps=1e-8,
    model_name_prefix="fair_custom_lambda",
    fairness_layer_name="fair_penalty",
)
```

Sus responsabilidades son:

```text
lambda_fair          -> peso de la penalizacion
fairness_eps         -> estabilizador de FairnessPenalty
model_name_prefix    -> prefijo estable para modelos guardados
fairness_layer_name  -> nombre estable de la capa FAIR
```

La configuracion del modelo predictivo, como capas densas, dropout, learning rate, clipnorm o activacion, sigue perteneciendo a `CustomMLPConfig`. Esto separa dos decisiones diferentes:

```text
CustomMLPConfig -> como predice la red
FairModelConfig -> como se penaliza la dependencia con genero
```

## `FairCustomModelBuilder`

La clase `FairCustomModelBuilder` es el orquestador POO del Bloque 6.

Recibe un `CustomMLPModelBuilder` ya existente y lo reutiliza:

```python
custom_builder = CustomMLPModelBuilder(custom_config)
fair_builder = FairCustomModelBuilder(
    custom_builder=custom_builder,
    fair_config=FairModelConfig(lambda_fair=0.5),
)
```

El punto clave es que el builder FAIR no reconstruye esta arquitectura:

```text
FinancialRatiosLayer
TrainableGammaLayer
BatchNormalization
Dense/Dropout stack
sigmoid
```

En su lugar llama a:

```python
graph = custom_builder.build_probability_graph(...)
```

Ese metodo devuelve el grafo comun hasta la probabilidad `y_hat`. Despues el Bloque 6 solo anade:

```text
sensitive input
FairnessPenalty([probability_output, sensitive])
```

El flujo real de `FairCustomModelBuilder.build()` es:

```text
1. Resolver lambda_fair.
2. Crear Input(name="features").
3. Crear Input(name="sensitive").
4. Construir el grafo predictivo comun con build_probability_graph().
5. Envolver la salida con FairnessPenalty.
6. Crear tf.keras.Model con inputs {"features", "sensitive"}.
7. Compilar el modelo con custom_builder.compile_model().
8. Devolver FairModelBuildResult con modelo y metadatos.
```

Esto garantiza una comparacion controlada: el modelo base final y el modelo FAIR final comparten exactamente la misma arquitectura, optimizador, loss base y metricas. Lo unico que cambia entre ellos es `lambda_fair`.

## `FairModelBuildResult`

`FairCustomModelBuilder` no devuelve solo el modelo. Devuelve un contenedor con trazabilidad:

```python
FairModelBuildResult(
    model=...,
    ratio_indices=...,
    ratio_feature_indices=...,
    output_feature_count_after_custom_block=...,
    lambda_fair=...,
    model_name=...,
)
```

Esto importa porque los siguientes bloques necesitaran:

```text
model_name      -> guardar artefactos con nombres consistentes
lambda_fair     -> construir pareto_results.csv
ratio_indices   -> auditar que las variables financieras usadas son las correctas
```

## Funcion `build_fair_custom_model`

Tambien existe una funcion wrapper:

```python
model = build_fair_custom_model(
    builder=custom_builder,
    ratio_indices=ratio_indices,
    input_dim=input_dim,
    lambda_fair=1.0,
)
```

Esta funcion existe para notebooks y ejemplos cortos. Internamente no implementa otra arquitectura; delega en `FairCustomModelBuilder`.

La regla del proyecto es:

```text
codigo productivo y extensible -> FairCustomModelBuilder
notebooks o snippets breves    -> build_fair_custom_model
```

## Funcion `lambda_slug`

El Bloque 6 tambien define:

```python
lambda_slug(0.5) -> "0_5"
lambda_slug(1.0) -> "1_0"
```

Esto se usara en Bloque 7 para nombres estables de modelos, historiales y filas de resultados:

```text
fair_custom_lambda_0_5
history_fair_lambda_0_5.csv
fair_lambda_0_5.keras
```

## Modelo base historico vs modelo base final

Hay que distinguir dos conceptos:

```text
Bloque 4:
    baseline historico de una entrada.
    Sirve como sanity-check inicial.

Bloque 6/7:
    familia final dual-input.
    Incluye lambda_fair=0 como base final controlada.
```

Para la tabla final del Bloque 11, la comparacion justa sera:

```text
base final -> modelo dual-input con lambda_fair = 0
FAIR final -> modelo dual-input con lambda_fair seleccionado
```

El modelo del Bloque 4 no se elimina, pero no debe mezclarse como comparador principal en la tabla final si el FAIR usa una API distinta. La comparacion cientifica correcta requiere que ambos modelos finales compartan la misma familia arquitectonica.

## Como construir un modelo FAIR desde `feature_names`

Uso recomendado:

```python
from src.models import (
    CustomMLPConfig,
    CustomMLPModelBuilder,
    FairCustomModelBuilder,
    FairModelConfig,
)

custom_builder = CustomMLPModelBuilder(
    CustomMLPConfig(
        hidden_units=(128, 64),
        dropout=0.2,
        learning_rate=1e-3,
    )
)

fair_builder = FairCustomModelBuilder(
    custom_builder=custom_builder,
    fair_config=FairModelConfig(lambda_fair=0.5),
)

result = fair_builder.build_from_feature_names(processed.feature_names)
model = result.model
```

`build_from_feature_names()` resuelve internamente los indices financieros mediante el `FinancialRatioIndexResolver` del builder custom. Por tanto, no se usa un diccionario manual tipo `ratio_idx["AMT_CREDIT"]`. La fuente de verdad es `FinancialRatioIndices`.

## Como construir un modelo FAIR desde indices ya resueltos

Uso recomendado para Bloque 7, donde se repetiran builds en el tuner:

```python
ratio_indices = custom_builder.index_resolver.resolve(processed.feature_names)

result = fair_builder.build(
    input_dim=processed.X_train.shape[1],
    ratio_indices=ratio_indices,
    lambda_fair=1.0,
)
```

El parametro `lambda_fair` del metodo permite barrer lambdas sin recrear el builder. Esto es util para:

```text
lambda = 0.0
lambda = 0.05
lambda = 0.1
lambda = 0.25
lambda = 0.5
...
```

## Como entrenar

El entrenamiento usa `y_true` limpio:

```python
history = model.fit(
    {
        "features": processed.X_train,
        "sensitive": processed.s_train.reshape(-1, 1),
    },
    processed.y_train,
    validation_data=(
        {
            "features": processed.X_val,
            "sensitive": processed.s_val.reshape(-1, 1),
        },
        processed.y_val,
    ),
    class_weight=class_weight,
    epochs=config.epochs,
    batch_size=config.batch_size,
    callbacks=callbacks,
    verbose=1,
)
```

`class_weight` funciona porque Keras sigue recibiendo `processed.y_train` como vector de etiquetas binarias. No hay `y_true_aug`.

## Callback de fairness

El callback ya existe en `src/callbacks.py`:

```python
from src.callbacks import FairnessLogger

fairness_cb = FairnessLogger(
    X_val=processed.X_val,
    s_val=processed.s_val,
    include_sensitive_input=True,
)
```

Para modelos dual-input, `include_sensitive_input=True` es obligatorio. Si se deja en `False`, el callback intentara predecir solo con `X_val`, y Keras fallara porque el modelo tambien necesita la entrada `sensitive`.

Este callback registra:

```text
val_abs_rho
```

por epoca dentro de `history.history`. Esa serie se usara en:

```text
Bloque 7  -> analizar cada trial y cada lambda
Bloque 12 -> graficar convergencia de fairness
```

## Asunciones del bloque

1. `SENSITIVE` ya llega como vector binario 0/1 desde el Bloque 2.
2. `SENSITIVE` no forma parte de `features`.
3. La rama predictiva solo ve `features`.
4. La variable sensible solo se usa para calcular la penalizacion FAIR y las metricas de auditoria.
5. La estrategia de desbalance sigue siendo BCE + `class_weight`.
6. El batch size debe ser suficientemente grande para que haya variacion de genero dentro de los batches.
7. `lambda_fair=0` representa el modelo base final dentro de la misma familia dual-input.

## Riesgos controlados

### Riesgo 1 - Duplicar arquitectura

Mitigacion implementada:

```text
FairCustomModelBuilder llama a CustomMLPModelBuilder.build_probability_graph().
```

No hay una segunda copia manual de `FinancialRatiosLayer`, `TrainableGammaLayer`, BatchNorm, Dense ni Dropout.

### Riesgo 2 - Meter genero como feature

Mitigacion implementada:

```text
features y sensitive son inputs separados.
sensitive solo entra en FairnessPenalty.
```

El modelo puede seguir usando proxies de genero presentes en otras variables, como `NAME_FAMILY_STATUS` o `NAME_EDUCATION_TYPE`. Precisamente por eso existe la penalizacion estadistica: no basta con quitar `CODE_GENDER` de las features.

### Riesgo 3 - Batches con poca varianza

Mitigacion implementada:

```text
FairnessPenalty usa eps dentro del denominador diferenciable.
FairnessLogger usa una metrica robusta para evaluacion.
```

Ademas, el entrenamiento debe mantener `shuffle=True` y un batch size razonable.

### Riesgo 4 - Lambda alto inestable

Mitigacion heredada del Bloque 5:

```text
CustomMLPModelBuilder.compile_model() usa Adam con clipnorm.
```

Esto reduce el riesgo de gradientes explosivos cuando `lambda_fair` sea grande durante el barrido del Bloque 7.

## Tests implementados

El archivo nuevo es:

```text
tests/test_fair_model.py
```

Cubre cinco escenarios:

```text
1. FairnessPenalty devuelve y_pred sin modificar y anade loss cuando lambda > 0.
2. FairnessPenalty con lambda=0 no anade perdida extra.
3. FairCustomModelBuilder crea modelos dual-input para lambda=0 y lambda>0.
4. Un modelo FAIR entrena una epoca con y_true limpio y FairnessLogger dual-input.
5. build_fair_custom_model y lambda_slug funcionan como API publica documentada.
```

Ademas, los tests previos de `FairnessLogger` ya cubren la ruta `include_sensitive_input=True`, por lo que el callback esta preparado para el modelo de este bloque.

## Verificacion ejecutada

Tras implementar el bloque se ha verificado:

```text
python -m py_compile sobre src/ y tests/ -> OK
pytest -q -> 22 tests passed
imports publicos de Bloque 6 -> OK
```

Los warnings de TensorFlow sobre oneDNN, TensorRT o deprecaciones internas no indican fallo del proyecto.

## Que obtenemos al finalizar

Al cerrar este bloque el proyecto tiene:

```text
FairnessPenalty serializable
builder FAIR POO sin duplicacion de arquitectura
modelo base final dual-input con lambda_fair=0
modelo FAIR dual-input con lambda_fair>0
y_true limpio compatible con Keras
class_weight compatible
val_abs_rho por epoca mediante FairnessLogger
lambda_slug para nombres estables de artefactos
tests especificos del bloque
```

Esto deja preparado el Bloque 7, donde se usara esta familia de modelos para:

```text
1. buscar arquitectura con Keras Tuner
2. fijar la arquitectura ganadora
3. barrer lambda_fair
4. construir la curva de Pareto AUC vs fairness
```

## Criterio de terminado

El Bloque 6 se considera terminado porque:

```text
FairnessPenalty existe, es serializable y tiene get_config()
FairCustomModelBuilder reutiliza build_probability_graph()
build_fair_custom_model delega en el builder POO
lambda_fair=0 produce la base final dual-input
lambda_fair>0 anade penalizacion FAIR
FairnessLogger funciona con include_sensitive_input=True
el modelo compila con BCE y metricas normales
los tests pasan
```

# Bloque 4 - Modelo base

## Objetivo del bloque

El Bloque 4 implementa el primer modelo neuronal base del MVP.

El codigo real esta en:

```text
src/base_model.py
```

Tambien se ha actualizado:

```text
src/__init__.py
```

Nota practica:

`src/base_model.py` depende de TensorFlow. En este entorno local actual TensorFlow no esta instalado, asi que `src/__init__.py` carga los exports del Bloque 4 de forma opcional. Esto permite seguir importando Bloques 1-3 aunque falte TensorFlow. Para entrenar el Bloque 4 hay que instalar las dependencias de `requirements.txt`.

Este bloque entrena una MLP sencilla sin penalizacion FAIR. Su objetivo principal es comprobar que el pipeline de datos funciona, que la red entrena, que las metricas se calculan y que podemos auditar una primera dependencia entre prediccion y genero.

## Alcance exacto

Este bloque SI hace:

```text
crear MLP base
compilar con binary_crossentropy
usar class_weight
crear callbacks
entrenar en train
monitorizar en validation
calcular metricas iniciales
calcular abs_rho(prediccion, sensitive)
guardar history y predicciones de validation
```

Este bloque NO hace:

```text
capas custom
FAIR loss
Keras Tuner
barrido lambda
threshold optimo
evaluacion final en test
```

La decision importante sigue siendo:

```text
modelo base final de la tabla = arquitectura final + lambda_fair = 0
```

Ese modelo final se construira despues de Bloques 5 y 6. El Bloque 4 es el baseline funcional inicial y la base de utilidades de entrenamiento.

## Codigo creado

Modulo:

```text
src/base_model.py
```

Clases principales:

```text
BaseModelConfig
BaseMLPModelBuilder
BaseModelTrainer
ClassWeightCalculator
TrainingCallbackFactory
BaseValidationEvaluator
AbsolutePearsonCorrelation
BaseModelArtifactSaver
ReproducibilityManager
TrainingArrayValidator
```

Contenedores:

```text
BaseValidationMetrics
BaseTrainingArtifacts
BaseTrainingResult
ReproducibilityConfig
```

Excepcion:

```text
BaseModelError
```

## `BaseModelConfig`

Configuracion inmutable de la MLP base.

Valores principales:

```text
hidden_units = (128, 64)
activation = "elu"
dropout = 0.2
learning_rate = 1e-3
gradient_clipnorm = 1.0
loss = "binary_crossentropy"
batch_size = 1024
epochs = 100
early_stopping_monitor = "val_auc"
early_stopping_patience = 10
reduce_lr_monitor = "val_loss"
reduce_lr_patience = 5
provisional_threshold = 0.5
```

Nota:

`provisional_threshold=0.5` solo se usa para diagnostico inicial del Bloque 4. El threshold real se elegira en Bloque 8 usando validation.

## `ReproducibilityManager`

Responsabilidad:

```text
fijar semillas antes de construir y entrenar el modelo
```

Metodo:

```python
apply()
```

Aplica:

```text
PYTHONHASHSEED
TF_DETERMINISTIC_OPS
random.seed
np.random.seed
tf.random.set_seed
tf.config.experimental.enable_op_determinism si esta disponible
```

Motivo:

Queremos que curvas y metricas sean lo mas reproducibles posible entre ejecuciones.

## `TrainingArrayValidator`

Responsabilidad:

```text
validar ProcessedSplitDataset antes de entrenar
```

Comprueba:

```text
X/y/s alineados por longitud
X_train, X_val, X_test sin NaN
y_train, y_val, y_test binarios
s_train, s_val, s_test binarios
```

Si algo falla, lanza:

```text
BaseModelError
```

## `ClassWeightCalculator`

Responsabilidad:

```text
calcular class_weight solo con y_train
```

Metodo:

```python
compute(y_train) -> dict[int, float]
```

Usa:

```python
sklearn.utils.class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train.astype(int),
)
```

Motivo:

Home Credit esta desbalanceado. Sin pesos de clase, la red puede aprender a predecir casi siempre `TARGET=0`.

Regla:

```text
class_weight se calcula solo con train
```

No se calcula con validation ni test.

## `BaseMLPModelBuilder`

Responsabilidad:

```text
construir y compilar la MLP base
```

Metodo:

```python
build(input_dim: int) -> tf.keras.Model
```

Arquitectura por defecto:

```text
Input(shape=input_dim)
Dense(128, activation="elu")
Dropout(0.2)
Dense(64, activation="elu")
Dropout(0.2)
Dense(1, activation="sigmoid")
```

Optimizador:

```text
Adam(learning_rate=1e-3, clipnorm=1.0)
```

Loss:

```text
binary_crossentropy
```

Metricas Keras:

```text
AUC(name="auc")
AUC(name="pr_auc", curve="PR")
BinaryAccuracy(name="binary_accuracy")
Precision(name="precision")
Recall(name="recall")
```

## Por que sigmoid

La salida:

```text
Dense(1, activation="sigmoid")
```

devuelve:

```text
P(TARGET=1 | X)
```

Interpretacion:

```text
cercano a 1 -> mayor riesgo de dificultades de pago
cercano a 0 -> menor riesgo
```

## `TrainingCallbackFactory`

Responsabilidad:

```text
crear callbacks nuevos para cada entrenamiento
```

Metodo:

```python
build() -> list[tf.keras.callbacks.Callback]
```

Callbacks:

```text
EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=configurable)
ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-6)
```

Motivo:

`EarlyStopping` evita sobreajuste y recupera los mejores pesos.

`ReduceLROnPlateau` reduce learning rate cuando la validacion se estanca.

## `AbsolutePearsonCorrelation`

Responsabilidad:

```text
calcular abs(corr(y_proba, sensitive))
```

Metodo:

```python
compute(predictions, sensitive) -> float
```

Motivo:

Aunque el modelo base no usa `SENSITIVE` como feature, puede discriminar por proxies. Por eso medimos si sus probabilidades estan correlacionadas con genero.

Interpretacion:

```text
abs_rho cercano a 0 -> poca dependencia lineal con genero
abs_rho alto -> mayor dependencia con genero
```

## `BaseValidationEvaluator`

Responsabilidad:

```text
evaluar el baseline en validation
```

Metodos:

```python
predict_probabilities(model, X)
evaluate(model, X_val, y_val, s_val, threshold)
```

Metricas calculadas:

```text
ROC-AUC
PR-AUC
Accuracy
Precision
Recall
F1
abs_rho
threshold usado
```

Las metricas binarias usan el threshold provisional de `BaseModelConfig`.

Importante:

Este threshold no es el definitivo. Bloque 8 elegira threshold por modelo usando validation.

## `BaseTrainingArtifacts`

Define rutas absolutas de artefactos ancladas a la raiz del proyecto:

```text
<PROJECT_ROOT>/results/tables/base_training_history.csv
<PROJECT_ROOT>/results/tables/base_val_predictions.csv
<PROJECT_ROOT>/results/models/base_mlp.keras
```

Esto evita que un notebook ejecutado desde `notebooks/` guarde artefactos en una carpeta incorrecta.

El modelo se guarda solo si se solicita con:

```python
save_model=True
```

Por defecto, el trainer guarda history y predicciones, pero no guarda el modelo.

## `BaseModelArtifactSaver`

Responsabilidad:

```text
guardar artefactos del baseline
```

Metodos:

```python
save(result, save_model=False)
save_history(history)
save_validation_predictions(predictions)
save_model(model)
```

Guarda:

```text
history.history -> CSV
predicciones validation -> CSV
modelo Keras -> opcional
```

## `BaseModelTrainer`

Clase orquestadora del bloque.

Metodo principal:

```python
train(data: ProcessedSplitDataset, save_artifacts=True, save_model=False, verbose=1)
    -> BaseTrainingResult
```

Flujo interno:

```text
1. aplicar reproducibilidad
2. validar arrays procesados
3. calcular class_weight con y_train
4. construir modelo con input_dim = X_train.shape[1]
5. crear callbacks
6. entrenar con train y validation
7. evaluar validation
8. empaquetar BaseTrainingResult
9. guardar artefactos si se solicita
```

## `BaseTrainingResult`

Contenedor final del bloque:

```text
model
history
class_weight
validation_metrics
validation_predictions
```

`validation_predictions` contiene:

```text
y_true
y_proba
y_pred_label
sensitive
threshold
```

## Uso esperado

Ejemplo:

```python
from src.base_model import BaseModelTrainer

trainer = BaseModelTrainer()

result = trainer.train(
    processed,
    save_artifacts=True,
    save_model=False,
    verbose=1,
)

print(result.class_weight)
print(result.validation_metrics.to_dict())
```

Donde `processed` es un `ProcessedSplitDataset` generado por Bloques 2 y 3.

## Relacion con Bloques 2 y 3

Entrada del Bloque 4:

```text
ProcessedSplitDataset
```

Ese objeto contiene:

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
feature_names
```

Bloque 4 usa:

```text
X_train, y_train -> entrenamiento
X_val, y_val -> monitorizacion
s_val -> auditoria abs_rho
```

No usa test para tomar decisiones.

## Sobre el test

El Bloque 4 no evalua test.

Motivo:

El test debe reservarse para la evaluacion final. Mirarlo ahora podria convertirlo en un validation set encubierto.

## Sobre Focal Loss

Focal Loss queda como experimento avanzado, no como configuracion del MVP.

Decision activa del Bloque 4:

```text
binary_crossentropy + class_weight
```

Regla:

```text
No combinar focal loss con class_weight.
```

## Sobre el modelo base final

Este baseline es un sanity check.

Cuando existan:

```text
Bloque 5 -> capas custom
Bloque 6 -> FAIR loss con lambda
```

el modelo base final para la tabla debera ser:

```text
misma arquitectura final
lambda_fair = 0
```

Asi la comparacion Base vs FAIR sera controlada.

## Artefactos generados

Por defecto:

```text
results/tables/base_training_history.csv
results/tables/base_val_predictions.csv
```

Opcional:

```text
results/models/base_mlp.keras
```

`results/models/` esta ignorado por Git.

## Dependencia de TensorFlow

Este bloque requiere:

```text
tensorflow==2.15.0
```

Si TensorFlow no esta instalado, se puede seguir usando:

```python
from src.preprocessing import ...
from src.splitting import ...
```

pero no se podra ejecutar:

```python
from src.base_model import BaseModelTrainer
```

hasta instalar TensorFlow.

## Errores evitados

### Accuracy enganosa

Se monitorizan AUC y PR-AUC, no solo accuracy.

### Ignorar clase minoritaria

Se usa `class_weight` calculado solo con train.

### Sobreajuste

Se usan `EarlyStopping` y `Dropout`.

### Gradientes inestables

Se usa `Adam(clipnorm=1.0)`.

### Usar genero como feature

El baseline recibe solo `X`. `s_val` se usa para auditar `abs_rho`, no para entrenar.

### Usar test demasiado pronto

El trainer evalua validation. Test queda para Bloque 11.

## Que obtenemos al finalizar el Bloque 4

Codigo:

```text
src/base_model.py
src/__init__.py actualizado
```

Capacidades:

```text
construir MLP base
entrenar con class_weight
usar callbacks
guardar history
guardar predicciones validation
calcular AUC, PR-AUC, precision, recall, F1
calcular abs_rho
registrar val_abs_rho por epoch con FairnessLogger
validar arrays antes de entrenar
fijar semillas
```

## Criterio de terminado

Bloque 4 queda terminado cuando:

```text
src/base_model.py existe
BaseMLPModelBuilder construye modelo Keras
BaseModelTrainer entrena con ProcessedSplitDataset
ClassWeightCalculator calcula pesos de train
BaseValidationEvaluator calcula metricas de validation
AbsolutePearsonCorrelation calcula abs_rho
FairnessLogger registra val_abs_rho en history
BaseModelArtifactSaver guarda CSVs
el modulo compila
hay tests automatizados con datos sinteticos
este .md describe el codigo real
```

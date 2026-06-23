# Bloque 5.5 - Endurecimiento previo a FAIR Loss

## Objetivo del bloque

Este bloque no corresponde a una tarea nueva del enunciado. Es un bloque de integridad tecnica antes de empezar el Bloque 6.

El objetivo es corregir deuda tecnica detectada en la auditoria externa y dejar el proyecto preparado para crecer hacia:

```text
Bloque 6  -> FAIR loss con sensitive como input separado
Bloque 7  -> Keras Tuner + barrido de lambda
Bloque 8  -> threshold optimo por modelo
Bloque 9  -> incertidumbre MVP
Bloque 10 -> incertidumbre OOF
Bloque 11 -> evaluacion final
Bloque 12 -> figuras obligatorias
```

La idea central es sencilla:

```text
Antes de anadir complejidad, protegemos lo que ya funciona.
```

## Por que habia que hacer este bloque

Hasta el Bloque 5 el proyecto ya tenia:

1. Contrato de datos.
2. Preprocesamiento sin leakage.
3. Split honesto TARGET + SENSITIVE.
4. Modelo base MLP.
5. Capas custom financieras.

Pero habia riesgos importantes:

```text
los smoke tests no estaban guardados
las rutas de artefactos dependian del current working directory
val_abs_rho solo se calculaba despues de entrenar
el builder custom no exponia un backbone reutilizable para FAIR
requirements.txt no fijaba scipy
AbsolutePearsonCorrelation aplicaba eps dos veces
```

Si se seguia directamente al Bloque 6, estos problemas habrian creado duplicacion de codigo y riesgo de roturas silenciosas.

## Cambios realizados

## 1. Tests repetibles

Se creo la carpeta:

```text
tests/
```

Archivos creados:

```text
tests/conftest.py
tests/test_data_contract.py
tests/test_preprocessing.py
tests/test_splitting.py
tests/test_layers.py
tests/test_base_model_hardening.py
tests/test_callbacks.py
```

### Por que era necesario

Antes, los smoke tests se habian ejecutado de forma interactiva. Eso sirve para comprobar en el momento, pero no protege el proyecto en el futuro.

Con tests guardados, cualquier cambio posterior puede validarse con:

```powershell
& "C:\venvs\homecredit311\Scripts\python.exe" -m pytest -q
```

Esto es especialmente importante porque los bloques siguientes tocaran:

```text
arquitectura
loss
inputs del modelo
callbacks
tuning
metricas
figuras
```

Sin tests, un cambio en el preprocesado podria romper los indices financieros del Bloque 5 sin aviso.

### Dataset sintetico de tests

`tests/conftest.py` define un dataset sintetico Home Credit-like con:

```text
320 filas
TARGET binario con aproximadamente 8% positivos
CODE_GENDER con F/M
los cuatro grupos TARGET x SENSITIVE representados
variables financieras positivas
missing values en EXT_SOURCE_1/2/3
sentinel 365243 en DAYS_EMPLOYED
categoricas MVP
```

El fixture contiene solo columnas del contrato MVP. Se elimino una columna extra (`DAYS_REGISTRATION`) que no formaba parte del contrato y que era descartada por `remainder="drop"`. No rompia el pipeline, pero podia confundir al leer los tests porque parecia sugerir que el MVP la usaba.

Esto permite probar el pipeline sin depender de los CSV reales de Kaggle.

### Tests de contrato

`tests/test_data_contract.py` valida:

```text
el contrato acepta las columnas sinteticas MVP
el contrato rechaza columnas requeridas ausentes
CODE_GENDER, SENSITIVE, TARGET y SK_ID_CURR estan excluidas como features normales
```

### Tests de preprocesamiento

`tests/test_preprocessing.py` valida:

```text
TARGET, CODE_GENDER y SENSITIVE no aparecen en X
SENSITIVE queda binaria
no quedan NaNs en X, y ni s
las columnas financieras quedan al inicio de feature_names
el pipeline financiero solo tiene SimpleImputer, no scaler
```

Este test protege directamente la decision critica del Bloque 5:

```text
FinancialRatiosLayer debe recibir importes originales imputados, no escalados.
```

### Tests de split

`tests/test_splitting.py` valida:

```text
proporcion 70/15/15
los cuatro grupos TARGET x SENSITIVE aparecen en train, validation y test
el report de split coincide con los tamanos reales
```

La proporcion 70/15/15 se comprueba con `pytest.approx(..., abs=0.01)`, no con igualdad exacta. Esto evita que el test sea fragil si en el futuro cambia el numero de filas del fixture y sklearn debe redondear algun split.

Esto protege la evaluacion de fairness. Si un split perdiera algun grupo, las metricas por genero podrian ser inestables o directamente invalidas.

### Tests de capas y builder custom

`tests/test_layers.py` valida:

```text
FinancialRatiosLayer expande N -> N+4
TrainableGammaLayer expande N+4 -> N+8
no se generan NaNs
get_config permite recrear capas
CustomMLPModelBuilder genera predicciones
el backbone compartido puede conectarse a un modelo de dos inputs
```

La ultima comprobacion es importante para Bloque 6:

```text
features + sensitive podran convivir en un modelo FAIR sin duplicar arquitectura.
```

### Tests de endurecimiento

`tests/test_base_model_hardening.py` valida:

```text
AbsolutePearsonCorrelation devuelve 1.0 en correlacion perfecta
las rutas de artefactos son absolutas
```

`tests/test_callbacks.py` valida:

```text
FairnessLogger escribe val_abs_rho en history.history
FairnessLogger funciona con modelos de una entrada
FairnessLogger funciona con modelos dual-input {"features", "sensitive"}
```

El segundo caso es especialmente importante para el Bloque 6. El test declara un input `sensitive` aunque todavia no lo conecta a una penalizacion FAIR, porque `FairnessPenalty` aun no existe. Lo que se valida en Bloque 5.5 es el plumbing: el callback sabe alimentar un modelo que recibe un diccionario con `features` y `sensitive`.

## 2. Nuevo callback `FairnessLogger`

Archivo creado:

```text
src/callbacks.py
```

Clases nuevas:

```text
CallbackError
FairnessLogger
```

### Que hace `FairnessLogger`

Calcula al final de cada epoch:

```text
val_abs_rho = |corr(predicciones_validacion, sensitive_validacion)|
```

Y lo inserta en los logs de Keras:

```python
logs["val_abs_rho"] = ...
```

Por tanto, despues de entrenar:

```python
history.history["val_abs_rho"]
```

contiene una serie por epoch.

### Por que se hizo asi

Antes, `abs_rho` se calculaba solo una vez despues del entrenamiento. Eso bastaba para una auditoria puntual, pero no para:

```text
comparar trials de Keras Tuner
ver evolucion fairness durante training
guardar curvas para figuras
detectar si lambda esta afectando al modelo
```

El callback deja el proyecto preparado para el Bloque 7 y el Bloque 12.

### Compatibilidad con modelo base y FAIR

`FairnessLogger` acepta:

```python
include_sensitive_input=False
```

para modelos de una entrada:

```text
X -> pred
```

y:

```python
include_sensitive_input=True
```

para modelos de dos entradas:

```text
{"features": X, "sensitive": s} -> pred
```

Esto evita duplicar callbacks en el Bloque 6.

## 3. `BaseModelTrainer` ahora guarda `val_abs_rho` por epoch

Archivo modificado:

```text
src/base_model.py
```

Cambio:

```python
callbacks.append(
    FairnessLogger(
        X_val=data.X_val,
        s_val=data.s_val,
        include_sensitive_input=False,
    )
)
```

### Por que importa

El modelo base del Bloque 4 no usa `SENSITIVE` como input, pero si debe auditar su dependencia con genero.

Con este cambio, su history incluye:

```text
loss
auc
pr_auc
binary_accuracy
precision
recall
val_loss
val_auc
val_pr_auc
val_binary_accuracy
val_precision
val_recall
val_abs_rho
```

Esto mejora la trazabilidad de fairness desde el primer modelo.

## 4. Rutas absolutas para artefactos

Archivo modificado:

```text
src/base_model.py
```

Se anadio:

```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

Y `BaseTrainingArtifacts` paso de rutas relativas:

```python
Path("results/tables/base_training_history.csv")
```

a rutas absolutas ancladas al repo:

```python
_PROJECT_ROOT / "results/tables/base_training_history.csv"
```

### Por que importa

Si se ejecuta un notebook desde:

```text
notebooks/
```

una ruta relativa podria guardar en:

```text
notebooks/results/
```

en vez de:

```text
results/
```

Esto romperia las figuras y tablas de bloques posteriores.

Con rutas absolutas, los artefactos se guardan siempre en el mismo sitio, independientemente del directorio desde el que se lance Python.

## 5. `restore_best_weights` configurable

Archivo modificado:

```text
src/base_model.py
```

Se anadio a `BaseModelConfig`:

```python
early_stopping_restore_best_weights: bool = True
```

Y `TrainingCallbackFactory` lo usa:

```python
restore_best_weights=self._config.early_stopping_restore_best_weights
```

### Por que importa

El valor por defecto sigue siendo el mismo:

```text
True
```

pero ahora queda controlado desde configuracion. Esto sera util si Keras Tuner o algun experimento posterior necesita variar callbacks de forma explicita.

## 6. Correccion de `AbsolutePearsonCorrelation`

Archivo modificado:

```text
src/base_model.py
```

Antes:

```python
if denominator <= eps:
    return 0.0
return abs(numerator / (denominator + eps))
```

Ahora:

```python
if denominator <= eps:
    return 0.0
return abs(numerator / denominator)
```

### Por que importa

El epsilon ya se usa para decidir si el denominador es demasiado pequeno. Si el denominador es valido, volver a sumarlo altera ligeramente la correlacion.

El cambio es pequeno numericamente, pero correcto matematicamente.

El test:

```text
test_absolute_pearson_returns_exact_unit_correlation
```

protege que una correlacion perfecta devuelva exactamente:

```text
1.0
```

## 7. Refactor de `CustomMLPModelBuilder`

Archivo modificado:

```text
src/models.py
```

Clase nueva:

```text
CustomProbabilityGraph
```

Metodos nuevos:

```python
build_probability_graph(...)
compile_model(...)
_validate_input_dim_and_indices(...)
```

### Problema que se queria evitar

El Bloque 6 necesita un modelo con dos inputs:

```text
features
sensitive
```

Si se construyera una funcion nueva `build_fair_model()` copiando la arquitectura del Bloque 5, tendriamos dos sitios con la misma logica:

```text
FinancialRatiosLayer
TrainableGammaLayer
BatchNormalization
Dense stack
Dropout
Dense sigmoid
```

Eso seria peligroso porque cualquier cambio futuro podria aplicarse al modelo base pero no al FAIR, o al reves.

### Solucion aplicada

`CustomMLPModelBuilder` ahora separa:

```text
construir el grafo probabilistico
compilar el modelo
envolver el grafo en un Model de Keras
```

El modelo base sigue funcionando igual:

```python
build_result = builder.build_from_feature_names(processed.feature_names)
model = build_result.model
```

Pero el Bloque 6 podra hacer:

```python
features_in = tf.keras.Input(shape=(input_dim,), name="features")
graph = builder.build_probability_graph(
    features_input=features_in,
    input_dim=input_dim,
    ratio_indices=ratio_indices,
)

sensitive_in = tf.keras.Input(shape=(1,), name="sensitive")
fair_output = FairnessPenalty(lambda_fair)([graph.probability_output, sensitive_in])

model = tf.keras.Model(
    inputs={"features": graph.features_input, "sensitive": sensitive_in},
    outputs=fair_output,
)

builder.compile_model(model)
```

Asi, base y FAIR comparten exactamente:

```text
capas custom
BatchNorm
dense stack
dropout
sigmoid
optimizer
loss base
metricas
```

La unica diferencia sera:

```text
lambda_fair = 0       -> modelo base
lambda_fair > 0       -> modelo FAIR
```

Esa es la comparacion controlada que necesitamos para la tabla final.

## 8. `requirements.txt` actualizado

Archivo modificado:

```text
requirements.txt
```

Se anadio:

```text
scipy>=1.11.0,<1.13.0
pytest==7.4.4
```

### Por que `scipy`

`fairlearn` y `scikit-learn` pueden traer `scipy` de forma transitiva, pero si no se fija version, pip puede instalar una version demasiado antigua o demasiado nueva.

En el entorno local se detecto que habia:

```text
scipy 1.16.3
```

y se actualizo a:

```text
scipy 1.12.0
```

compatible con:

```text
numpy 1.26.0
scikit-learn 1.3.2
```

### Por que `pytest`

Los tests ahora son parte real del proyecto. Por tanto, `pytest` debe estar en requirements para que cualquier companero pueda ejecutar:

```powershell
python -m pytest -q
```

## 9. Exports actualizados

Archivo modificado:

```text
src/__init__.py
```

Nuevos exports opcionales dependientes de TensorFlow:

```text
CallbackError
FairnessLogger
CustomProbabilityGraph
```

Esto permite importar desde el paquete principal:

```python
from src import FairnessLogger, CustomProbabilityGraph
```

## Verificacion ejecutada

### Sintaxis

Comando:

```powershell
$files = @(Get-ChildItem src -Filter *.py | ForEach-Object { $_.FullName }) + `
         @(Get-ChildItem tests -Filter *.py | ForEach-Object { $_.FullName })

& "C:\venvs\homecredit311\Scripts\python.exe" -m py_compile @files
```

Resultado:

```text
sin errores
```

### Tests

Comando:

```powershell
& "C:\venvs\homecredit311\Scripts\python.exe" -m pytest -q
```

Resultado:

```text
17 passed
```

## Que NO se ha cambiado en este bloque

No se ha implementado FAIR loss todavia.

No se ha anadido `FairnessPenalty`.

No se ha cambiado el comportamiento matematico de las capas custom.

No se ha eliminado el nivel alto de comentarios/docstrings, porque fue una decision pedagogica solicitada para que el equipo entienda el proyecto. Se podra limpiar antes de la entrega si se quiere un estilo mas profesional y menos tutorial.

No se ha movido `RawSplitDataset` a un modulo comun. Es una mejora posible, pero no bloquea el Bloque 6 y meteria ruido de imports ahora mismo.

## Por que este bloque protege los bloques futuros

### Protege Bloque 6

El builder custom ya expone un backbone compartido. El modelo FAIR no tendra que copiar arquitectura.

Esto evita que base y FAIR diverjan accidentalmente.

### Protege Bloque 7

`val_abs_rho` ya existe por epoch y `CustomMLPModelBuilder` acepta configuracion inyectable.

Keras Tuner podra reutilizar el mismo builder y registrar fairness de forma consistente.

### Protege Bloque 8

Los tests garantizan que `SENSITIVE` y `TARGET` siguen alineados con las predicciones.

Esto sera clave para thresholds, F1, DPD y EOD.

### Protege Bloques 9 y 10

Los tests validan que `EXT_SOURCE` missingness se conserva y que no quedan NaNs.

Esto protege los experimentos de incertidumbre basados en mala calidad de `EXT_SOURCE`.

### Protege Bloques 11 y 12

Las rutas absolutas aseguran que histories, predicciones, tablas y modelos se guardan en la carpeta correcta.

`val_abs_rho` por epoch ayuda a construir figuras y auditorias sin recalcular todo manualmente.

## Estado final

El proyecto queda listo para empezar el Bloque 6.

Checklist completada:

```text
[x] tests repetibles
[x] dataset sintetico de pruebas
[x] fixture sintetico limitado al contrato MVP
[x] test de split robusto con tolerancia
[x] FairnessLogger probado en single-input y dual-input
[x] rutas absolutas de artefactos base
[x] scipy fijado en requirements
[x] pytest anadido a requirements
[x] AbsolutePearsonCorrelation corregida
[x] FairnessLogger implementado
[x] BaseModelTrainer registra val_abs_rho por epoch
[x] CustomMLPModelBuilder expone backbone reutilizable
[x] __init__.py exporta nuevas utilidades
[x] py_compile correcto
[x] pytest completo correcto
```

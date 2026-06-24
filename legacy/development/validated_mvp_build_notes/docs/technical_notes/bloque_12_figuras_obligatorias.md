# Bloque 12 - Figuras obligatorias

## Objetivo del bloque

El objetivo de este bloque es generar las figuras exigidas para la entrega y la defensa.

Figuras obligatorias:

```text
1. Curva Pareto: rendimiento vs fairness
2. Distribucion de incertidumbre: TARGET=0 vs TARGET=1
3. Curvas de loss: base vs FAIR
```

Figura extra recomendada:

```text
EXT_NULL_COUNT vs incertidumbre
```

## Regla de rutas

No usar paths relativos sueltos como:

```python
pd.read_csv("results/tables/pareto_results.csv")
```

Todas las figuras deben anclarse a `PROJECT_ROOT`.

En scripts `.py` dentro del repo:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

En notebooks dentro de `notebooks/`:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
```

Uso:

```python
tables_dir = PROJECT_ROOT / "results" / "tables"
figures_dir = PROJECT_ROOT / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
```

## Figura 1 - Pareto AUC vs fairness

Archivo:

```text
results/figures/pareto_auc_vs_fairness.png
```

Fuente:

```text
results/tables/pareto_results.csv
```

Ejes:

```text
X = val_abs_rho
Y = val_auc
```

Por que validation:

```text
la curva Pareto se usa para elegir modelo FAIR
test no debe intervenir en seleccion
```

Elementos:

```text
puntos de todos los lambdas
color por lambda_fair
lambda=0 resaltado
FAIR seleccionado resaltado
frontera Pareto si da tiempo
```

Codigo conceptual:

```python
pareto = pd.read_csv(tables_dir / "pareto_results.csv")

plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    pareto["val_abs_rho"],
    pareto["val_auc"],
    c=pareto["lambda_fair"],
)
plt.xlabel("|corr(prediccion, genero)| en validation")
plt.ylabel("AUC en validation")
plt.title("Trade-off rendimiento vs fairness")
plt.colorbar(scatter, label="lambda_fair")
plt.savefig(figures_dir / "pareto_auc_vs_fairness.png", dpi=200)
```

Interpretacion:

```text
queremos puntos arriba y a la izquierda
```

## Figura 2 - Distribucion de incertidumbre

Archivo:

```text
results/figures/uncertainty_distribution_by_target.png
```

Fuente:

```text
results/tables/uncertainty_test.csv
```

Grupos:

```text
TARGET=0 -> pago a tiempo real
TARGET=1 -> dificultades reales
```

No agrupar por `y_pred_label` si el titulo habla de buen/mal pagador real.

Para evitar que seaborn trate `y_true` como continuo, crear una columna string:

```python
unc = pd.read_csv(tables_dir / "uncertainty_test.csv")

assert unc["uncertainty"].nunique() > 1
assert set(unc["EXT_NULL_COUNT"].unique()).issubset({0, 1, 2, 3})

unc["target_group"] = unc["y_true"].map(
    {
        0: "Pago puntual (TARGET=0)",
        1: "Dificultad (TARGET=1)",
    }
)

sns.kdeplot(
    data=unc,
    x="uncertainty",
    hue="target_group",
    common_norm=False,
)
plt.xlabel("Incertidumbre estimada")
plt.title("Distribucion de incertidumbre por TARGET real")
plt.savefig(figures_dir / "uncertainty_distribution_by_target.png", dpi=200)
```

Si la KDE queda poco informativa por concentracion de valores, usar
`sns.histplot(..., stat="density", common_norm=False)` o un boxplot por grupo.
Lo importante es que la figura compare distribuciones reales y que la
incertidumbre no sea constante.

Metricas a anadir:

```text
mediana por grupo
IQR por grupo
```

## Figura 3 - Curvas de loss y convergencia

Archivo:

```text
results/figures/loss_curves_base_fair.png
```

Fuentes segun convencion del Bloque 7:

```text
results/tables/history_fair_lambda_0_0.csv
results/tables/history_fair_lambda_{best}.csv
```

No usar nombres de historial que no hayan sido generados por el Bloque 7.

## Problema de comparabilidad de loss

La loss del modelo FAIR incluye:

```text
BCE + lambda * rho^2 + regularizadores
```

La loss del modelo lambda=0 no incluye penalizacion FAIR efectiva.

Por tanto, comparar valores absolutos punto a punto puede ser enganoso.

Decision para el MVP:

```text
graficar loss normalizada por su valor en epoch 1
```

Esto permite comparar convergencia relativa sin afirmar que las escalas absolutas sean identicas.

Codigo conceptual:

```python
def normalized(series):
    return series / series.iloc[0]

base_hist = pd.read_csv(tables_dir / "history_fair_lambda_0_0.csv")
fair_hist = pd.read_csv(tables_dir / f"history_fair_lambda_{best_slug}.csv")

plt.figure(figsize=(8, 5))
plt.plot(normalized(base_hist["loss"]), label="base train loss norm")
plt.plot(normalized(base_hist["val_loss"]), label="base val loss norm")
plt.plot(normalized(fair_hist["loss"]), label="FAIR train loss norm")
plt.plot(normalized(fair_hist["val_loss"]), label="FAIR val loss norm")
```

## Eje secundario para AUC y fairness

Gracias al Bloque 5.5, los histories incluyen:

```text
val_abs_rho
```

La Figura 3 debe incluir en eje secundario:

```text
val_auc
val_abs_rho
```

Esto muestra no solo convergencia de loss, sino tambien evolucion de rendimiento y fairness.

Codigo conceptual:

```python
ax1 = plt.gca()
ax2 = ax1.twinx()

ax2.plot(base_hist["val_auc"], "--", label="base val_auc")
ax2.plot(fair_hist["val_auc"], "--", label="FAIR val_auc")
ax2.plot(base_hist["val_abs_rho"], ":", label="base val_abs_rho")
ax2.plot(fair_hist["val_abs_rho"], ":", label="FAIR val_abs_rho")
```

## Figura extra - EXT_NULL_COUNT vs incertidumbre

Archivo:

```text
results/figures/ext_null_count_vs_uncertainty.png
```

Fuente:

```text
results/tables/uncertainty_test.csv
```

Grafico recomendado:

```text
boxplot de uncertainty por EXT_NULL_COUNT
```

o:

```text
barplot de mediana de uncertainty por EXT_NULL_COUNT
```

Por que es importante:

```text
responde directamente a si el modelo duda mas en perfiles con mala calidad de EXT_SOURCE
```

## Estilo comun

Todas las figuras deben:

```text
tener titulo claro
tener ejes claros
indicar validation o test
guardar PNG en results/figures/
poder regenerarse desde CSV
no depender de objetos en memoria del notebook
```

## Artefactos esperados

Entradas:

```text
results/tables/pareto_results.csv
results/tables/uncertainty_test.csv
results/tables/history_fair_lambda_0_0.csv
results/tables/history_fair_lambda_{best}.csv
```

Salidas:

```text
results/figures/pareto_auc_vs_fairness.png
results/figures/uncertainty_distribution_by_target.png
results/figures/loss_curves_base_fair.png
results/figures/ext_null_count_vs_uncertainty.png
```

## Riesgos y mitigaciones

### Riesgo 1 - Pareto en test

Mitigacion:

```text
Pareto de seleccion usa validation
```

### Riesgo 2 - Paths relativos

Mitigacion:

```text
PROJECT_ROOT / "results" / ...
```

### Riesgo 3 - Etiquetas ambiguas

Mitigacion:

```text
crear target_group con strings legibles
```

### Riesgo 4 - Loss no comparable

Mitigacion:

```text
usar loss normalizada por epoch 1
```

### Riesgo 5 - Ignorar fairness por epoch

Mitigacion:

```text
incluir val_abs_rho en Figura 3
```

## Criterio de terminado

El Bloque 12 se considera terminado cuando:

```text
existe figura Pareto
existe figura de incertidumbre
existe figura de loss/convergencia
existe figura extra EXT_NULL_COUNT si da tiempo
todas las figuras usan PROJECT_ROOT
todas las figuras salen de CSV guardados
las referencias de nombres coinciden con Bloque 7 y 9
```

# Taller B4-T1 — Diseño de Redes Neuronales Confiables

> **Justicia e Incertidumbre en la Concesión de Crédito**  
> Instituto BME · Junio de 2026

## Descripción

Diseño, entrenamiento y auditoría de un modelo de clasificación neuronal
para la concesión de créditos, garantizando que sus decisiones sean:

- ✅ **Precisas** — AUC-ROC de 0.755 con 42 features (0.745 con 12 features del enunciado)
- ⚖️ **Justas** — FAIR Loss que reduce la discriminación por género un 90.5%
- 🔍 **Honestas** — Incertidumbre calibrada mediante MC Dropout

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)  
**307.507 solicitudes de crédito** · Variable objetivo: dificultades de pago (8.1%)

---

##  Estructura del proyecto
taller-redes-confiables/

├── Taller_B4_T1.ipynb              # Notebook principal (12 features del enunciado)

├── EDA_features_extendido.ipynb    # Notebook análisis extendido (42 features)

├── src/

│   ├── preprocessing.py            # Pipeline de datos

│   ├── custom_layers.py            # DebtRatioLayer, ExtSourceLayer

│   ├── fair_loss.py                # FairBCELoss, FairAUC

│   ├── models.py                   # Construcción de modelos M0-M6

│   ├── train.py                    # Entrenamiento y callbacks

│   ├── uncertainty.py              # MC Dropout

│   ├── visualization.py            # Gráficos y curvas

│   └── checkpoints.py              # Guardado y carga de modelos

├── pareto_experiment.py            # Experimento Pareto multi-semilla

├── long_training_experiment.py     # Entrenamiento extendido M2*/M3/M6

├── uncertainty_experiment.py       # Análisis MC Dropout FAIR λ=1.0

├── main.py                         # Verificación de módulos

├── report/

│   ├── main.tex                    # Report en LaTeX

│   └── REDES_CONFIABLES.pdf        # Report compilado

├── report_plots/                   # Todas las figuras del report

├── checkpoints/                    # Pesos de los modelos entrenados

└── README.md

---

---

## Resultados

### Progresión de modelos (12 features del enunciado)

| Modelo | Arquitectura | AUC test | Justificación |
|--------|-------------|----------|---------------|
| M0 | Regresión Logística | 0.7335 | Baseline lineal |
| M1 | MLP Dense 64 | 0.7449 | Interacciones no lineales |
| M2 | MLP Dense 128→64 | 0.7447 | Joroba en AMT_CREDIT |
| M3 | M2 + Dropout(0.3, 0.2) | 0.7452 | Regularización |
| M4 | M3 + DebtRatioLayer | 0.7449 | Restricción regulatoria |
| **M6** | **Dual Custom + Dropout** | **0.7451** | **ExtSourceLayer auditable** |
| M3 largo | M3 + ReduceLR + 500 épocas | **0.7457** | Convergencia garantizada |

### Features extendidas (42 features)

| Modelo | 12 features | 42 features | ΔAUC |
|--------|------------|------------|------|
| M0 — Reg. Logística | 0.7335 | 0.7501 | +0.017 |
| M3 — MLP + Dropout  | 0.7457 | **0.7555** | +0.010 |
| M6 — Dual Custom    | 0.7451 | 0.7550 | +0.010 |

### FAIR Loss — Curva de Pareto (multi-semilla, 3 semillas por λ)

| λ | AUC test | ±std | DP Gap | ±std | Reducción DP |
|---|----------|------|--------|------|-------------|
| 0.0 | 0.7465 | 0.0002 | 0.0295 | 0.0004 | — |
| 0.5 | 0.7425 | 0.0005 | 0.0051 | 0.0019 | 82.9% |
| **1.0** | **0.7417** | **0.0006** | **0.0028** | **0.0020** | **90.5% ✅** |
| 5.0 | 0.7351 | 0.0003 | 0.0014 | 0.0008 | 95.1% |

> **Modelo recomendado: λ=1.0** — reduce la discriminación un 90.5% sacrificando solo 0.005 de AUC.

### Incertidumbre (MC Dropout, N=100 pasadas)

| Métrica | M6 base | FAIR λ=1.0 |
|---------|---------|-----------|
| Var. mayor en impagos | ❌ −8% | ✅ +145% |
| Var. crece con datos imputados | ❌ plano | ✅ monotóno (+79%) |
| AUC MC-mean vs puntual | +0.0004 | +0.0004 |

---

## Componentes técnicos clave

### 1. Capas custom de Keras

**`DebtRatioLayer`** — Restricción regulatoria sobre el ratio cuota/ingresos:
```python
f(x) = sigmoid(α · (x − θ))   # α≥0, θ inicializado en 0.35 (35%)
```
Tras entrenamiento: θ convergió a 0.16 (umbral real < referencia regulatoria).

**`ExtSourceLayer`** — Índice ponderado aprendible de fuentes externas:
```python
g(s) = sigmoid(w₁s₁ + w₂s₂ + w₃s₃ + b)   # wᵢ≥0 (monotonía garantizada)
```
Pesos aprendidos: w₁=0.44, w₂=w₃=0.61 (menos peso a EXT_SOURCE_1, con 56% nulos).

### 2. FAIR Loss customizada

```python
L = BCE(y, ŷ) + λ · |E[ŷ|G=M] − E[ŷ|G=F]|
```

Penaliza la **Paridad Demográfica**: diferencia en la probabilidad predicha
de impago entre hombres y mujeres. Evaluada con 3 semillas por λ para
cuantificar la varianza de inicialización.

### 3. Incertidumbre — MC Dropout

```python
# Dropout activo en inferencia → N predicciones estocásticas
mean, variance = mc_dropout_predict(model, X, n_passes=100)
```

---

## Instalación y uso

```bash
# Clonar
git clone https://github.com/marcocorpacriado-pixel/taller-redes-confiables.git
cd taller-redes-confiables

# Dependencias
pip install keras tensorflow scikit-learn pandas numpy matplotlib seaborn kagglehub

# Dataset: se descarga automáticamente desde Kaggle al ejecutar cualquier script
# O manualmente: descargar application_train.csv y colocarlo en la raíz

# Verificar instalación
python main.py

# Ejecutar notebooks
jupyter notebook Taller_B4_T1.ipynb

# Experimentos adicionales (scripts independientes)
python pareto_experiment.py          # Curva de Pareto multi-semilla
python long_training_experiment.py   # Entrenamiento extendido
python uncertainty_experiment.py     # Análisis de incertidumbre FAIR λ=1.0
```

---

## Pipeline POO Dani

La implementacion POO de Dani vive aislada en `src/dani_credit/`, sin
sobrescribir los modulos originales de Marco en `src/`.

- Notebook principal: `notebooks/01_mvp_dani_professional.ipynb`
- Documentacion tecnica: `docs_dani/`
- Tests de regresion: `tests/test_dani_uncertainty_regressions.py`

La incertidumbre MVP usa el esquema `M1 -> M2`: M2 aprende el error absoluto de
M1 sobre validation. La version corregida usa salida `softplus`, normalizacion
interna de las features de M2, clipping final a `[0, 1]` y validaciones para no
guardar una incertidumbre constante. `EXT_NULL_COUNT` se conserva crudo en
`ProcessedSplitDataset` y se reporta con valores semanticos `0`, `1`, `2` o `3`.

---

## Entregables

- 📓 **Notebook principal** — `Taller_B4_T1.ipynb` (12 features del enunciado)
- 📓 **Notebook extendido** — `EDA_features_extendido.ipynb` (42 features)
- 📄 **Report** — `report/REDES_CONFIABLES.pdf`
- 📦 **Módulos** — `src/` (código modular reutilizable)
- 🔬 **Scripts** — experimentos reproducibles en la raíz

---

## Autores

| Nombre |
|--------|
| Javier Fernández Guerra |
| Marco Corpa Criado |
| Daniel Gallego Sánchez |

---

## Referencias

- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML.
- Home Credit Group. *Home Credit Default Risk*. Kaggle, 2018.
- Directiva Europea de IA (2024) — Requisitos de no discriminación en sistemas crediticios.

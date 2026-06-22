# Taller B4-T1 — Diseño de Redes Neuronales Confiables

> **Justicia e Incertidumbre en la Concesión de Crédito**  
> Instituto BME Junio de 2026

## Descripción

Diseño, entrenamiento y auditoría de un modelo de clasificación neuronal
para la concesión de créditos, garantizando que sus decisiones sean:

- ✅ **Precisas** — AUC-ROC de 0.745 con arquitectura MLP customizada
- ⚖️ **Justas** — FAIR Loss que reduce la discriminación por género un 89%
- 🔍 **Honestas** — Incertidumbre calibrada mediante MC Dropout

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)  
**307.507 solicitudes de crédito** · Variable objetivo: dificultades de pago (8.1%)

---

##  Estructura del proyecto
taller-redes-confiables/

├── Taller_B4_T1.ipynb          # Notebook principal (código completo)

├── report/

│   ├── main.tex                # Report en LaTeX

│   └── report.pdf              # Report compilado

├── report_plots/               # Todas las figuras del report

├── checkpoints/                # Pesos de los modelos entrenados

└── README.md


---

## 📊 Resultados

### Progresión de modelos

| Modelo | Arquitectura | AUC test | Justificación del añadido |
|--------|-------------|----------|--------------------------|
| M0 | Regresión Logística | 0.7335 | Baseline lineal |
| M1 | MLP (Dense 64) | 0.7449 | Interacciones no lineales |
| M2 | MLP (Dense 128→64) | 0.7447 | Joroba en AMT_CREDIT |
| M3 | M2 + Dropout | 0.7452 | Regularización |
| M4 | M3 + DebtRatioLayer | 0.7449 | Restricción regulatoria |
| **M6** | **Dual Custom + Dropout** | **0.7451** | **ExtSourceLayer auditable** |

### FAIR Loss — Curva de Pareto

| λ | AUC test | DP Gap | Reducción sesgo |
|---|----------|--------|-----------------|
| 0.0 | 0.7463 | 0.0300 | — |
| **0.5** | **0.7429** | **0.0034** | **−89%** ✅ |
| 5.0 | 0.7352 | 0.0019 | −94% (coste alto) |

> **Modelo recomendado: λ=0.5** — reduce la discriminación un 89% sacrificando solo 0.003 de AUC.

### Incertidumbre (MC Dropout, N=100 pasadas)

| Métrica | M6 base | FAIR λ=0.5 |
|---------|---------|------------|
| Var. mayor en impagos | ❌ −8% | ✅ +148% |
| Var. crece con datos imputados | ❌ plano | ✅ monotóno (+73%) |
| AUC MC-mean vs puntual | +0.0004 | +0.0004 |

---

##  Componentes técnicos clave

### 1. Capas custom de Keras

**`DebtRatioLayer`** — Restricción regulatoria sobre el ratio cuota/ingresos:
```python
f(x) = sigmoid(α · (x − θ))   # α≥0, θ inicializado en 0.35 (35%)
```

**`ExtSourceLayer`** — Índice ponderado aprendible de fuentes externas:
```python
g(s) = sigmoid(w₁s₁ + w₂s₂ + w₃s₃ + b)   # wᵢ≥0 (monotonía garantizada)
```

### 2. FAIR Loss customizada

```python
L = BCE(y, ŷ) + λ · |E[ŷ|G=M] − E[ŷ|G=F]|
```

Penaliza la **Paridad Demográfica**: diferencia en la probabilidad predicha
de impago entre hombres y mujeres.

### 3. Incertidumbre — MC Dropout

```python
# Dropout activo en inferencia → N predicciones estocásticas
mean, variance = mc_dropout_predict(model, X, n_passes=100)
```

---

## ⚙️ Instalación y uso

```bash
# Clonar
git clone https://github.com/marcocorpacriado-pixel/taller-redes-confiables.git
cd taller-redes-confiables

# Dependencias
pip install keras tensorflow scikit-learn pandas numpy matplotlib seaborn

# Dataset: descargar application_train.csv desde Kaggle y colocarlo en la raíz

# Ejecutar
jupyter notebook Taller_B4_T1.ipynb
```

---

## 📁 Entregables

- 📓 **Notebook** — `Taller_B4_T1.ipynb` (código completo y reproducible)
- 📄 **Report** — `report/report.pdf` (análisis detallado con justificaciones)
- 📦 **Checkpoints** — pesos de todos los modelos entrenados

---

## 👥 Autores

| Nombre | 
|--------|
| Javier Fernández Guerra |
| Marco Corpa Criado |
| Daniel Gallego Sánchez |

---

## 📚 Referencias

- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML.
- Home Credit Group. *Home Credit Default Risk*. Kaggle, 2018.
- Directiva Europea de IA (2024) — Requisitos de no discriminación en sistemas crediticios.



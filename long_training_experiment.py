"""
long_training_experiment.py
────────────────────────────
Entrena M2*, M3 y M6 con más épocas para garantizar convergencia.

Modelos:
  M2* — MLP 2 capas + ReduceLROnPlateau (sin Dropout)
         Objetivo: ver cuánto mejora M2 con scheduler de LR
  M3  — MLP 2 capas + Dropout(0.3, 0.2) + ReduceLROnPlateau
         Objetivo: mejor modelo sin capas custom con convergencia garantizada
  M6  — Dual Custom + Dropout + ReduceLROnPlateau
         Objetivo: mejor modelo completo con convergencia garantizada

Estrategia de entrenamiento:
  · max_epochs=500 con EarlyStopping(patience=50)
    → Para solo si 50 épocas consecutivas sin mejora en val_auc
    → restore_best_weights=True garantiza que guardamos el mejor siempre
  · ReduceLROnPlateau(patience=10, factor=0.3)
    → Reduce LR cuando val_auc se estanca 10 épocas
    → Permite convergencia fina sin saltos de gradiente grandes
  · min_lr=1e-7 → suelo muy bajo para agotar la capacidad del modelo

Ejecutar desde la raíz del proyecto:
    python long_training_experiment.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')   # Cambiar a 'Agg' si no hay display
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

# ── Imports del proyecto ──────────────────────────────────────────────────
from src.preprocessing import full_pipeline, FEATURE_COLS
from src.models        import build_model_m2, build_model_m3, build_model_m6
from src.train         import get_class_weights
from src.checkpoints   import save_checkpoint, load_checkpoint
from src.visualization import (plot_curves_with_lr, print_model_table, savefig)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════
CSV_PATH   = 'application_train.csv'
SEED       = 42
MAX_EPOCHS = 500
PATIENCE_ES  = 75    # EarlyStopping: para si 50 épocas sin mejora
PATIENCE_LR  = 10    # ReduceLR: reduce LR si 10 épocas sin mejora
FACTOR_LR    = 0.3   # LR × 0.3 en cada reducción
MIN_LR       = 1e-7  # suelo de LR
BATCH_SIZE   = 512
LR_INICIAL   = 1e-3

os.makedirs('checkpoints',  exist_ok=True)
os.makedirs('report_plots', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. PIPELINE DE DATOS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("ENTRENAMIENTO LARGO — M2*, M3, M6 (hasta 500 épocas)")
print("═"*60)

if not os.path.exists(CSV_PATH):
    print(f"  '{CSV_PATH}' no encontrado — descargando desde Kaggle...")
    try:
        import kagglehub, glob, shutil
        dataset_path = kagglehub.dataset_download(
            "megancrenshaw/home-credit-default-risk")
        matches = glob.glob(
            os.path.join(dataset_path, "**", "application_train.csv"),
            recursive=True)
        if not matches:
            raise FileNotFoundError("No se encontró application_train.csv")
        shutil.copy(matches[0], CSV_PATH)
        print(f"  Copiado a '{CSV_PATH}'")
    except Exception as e:
        raise RuntimeError(
            f"Descarga fallida: {e}\n"
            "Pon application_train.csv en la raíz del proyecto.")

print("\n── 1. Cargando datos ────────────────────────────────────────")
(X_tr,  y_tr,  s_tr), \
(X_val, y_val, s_val), \
(X_te,  y_te,  s_te), \
FEATURE_COLS, scaler, medians = full_pipeline(CSV_PATH)

N_FEATURES     = X_tr.shape[1]
DEBT_RATIO_IDX = FEATURE_COLS.index('DEBT_RATIO')
EXT_IDXS       = [FEATURE_COLS.index(c) for c in
                  ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]

CLASS_WEIGHT = get_class_weights(y_tr)


# ══════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE ENTRENAMIENTO LARGO
# ══════════════════════════════════════════════════════════════════════════
def train_long(model, nombre: str):
    """
    Entrena con EarlyStopping(patience=50) + ReduceLROnPlateau.
    Devuelve history y AUC en test.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_INICIAL),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=PATIENCE_ES,          # 50 épocas de paciencia
            restore_best_weights=True,     # recupera el mejor siempre
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=FACTOR_LR,
            patience=PATIENCE_LR,
            min_lr=MIN_LR,
            verbose=1
        )
    ]

    print(f"\n  Entrenando {nombre} (max {MAX_EPOCHS} épocas, "
          f"patience={PATIENCE_ES})...")

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=CLASS_WEIGHT,
        callbacks=callbacks,
        verbose=0
    )

    # Métricas finales
    best_val  = max(history.history['val_auc'])
    n_epochs  = len(history.history['loss'])
    y_pred    = model.predict(X_te, verbose=0).ravel()
    test_auc  = roc_auc_score(y_te, y_pred)
    lrs_used  = history.history.get('learning_rate', [])
    min_lr_reached = min(lrs_used) if lrs_used else None

    print(f"\n  ── Resumen {nombre} ──────────────────────────────")
    print(f"  Épocas totales:     {n_epochs}")
    print(f"  Mejor AUC val:      {best_val:.4f}")
    print(f"  AUC test:           {test_auc:.4f}")
    if min_lr_reached:
        print(f"  LR mínimo alcanzado: {min_lr_reached:.2e}")

    return history, test_auc, best_val, n_epochs


# ══════════════════════════════════════════════════════════════════════════
# 2. ENTRENAR M2* (M2 + ReduceLR, sin Dropout)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("── M2* — MLP 2 capas + ReduceLR (sin Dropout) ─────────────")
print("─"*60)
print("Justificación: ¿cuánto mejora M2 si le damos más tiempo")
print("y un scheduler de LR? Referencia sin regularización.")

keras.utils.set_random_seed(SEED)
model_m2star = build_model_m2(N_FEATURES)
hist_m2star, auc_m2star_te, auc_m2star_val, ep_m2star = \
    train_long(model_m2star, 'M2*')

save_checkpoint(model_m2star, hist_m2star, 'M2star_ReduceLR')
plot_curves_with_lr(hist_m2star, 'M2* — MLP 2 capas + ReduceLR',
                    zoom_n=40, save=True)

# ══════════════════════════════════════════════════════════════════════════
# 3. ENTRENAR M3 (M2 + Dropout + ReduceLR)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("── M3 — MLP 2 capas + Dropout + ReduceLR ───────────────────")
print("─"*60)
print("Justificación: Dropout regulariza el overfitting de M2*;")
print("¿converge a mejor AUC con más épocas?")

keras.utils.set_random_seed(SEED)
model_m3 = build_model_m3(N_FEATURES)
hist_m3, auc_m3_te, auc_m3_val, ep_m3 = \
    train_long(model_m3, 'M3')

save_checkpoint(model_m3, hist_m3, 'M3_largo')
plot_curves_with_lr(hist_m3, 'M3 — MLP 2 capas + Dropout + ReduceLR',
                    zoom_n=40, save=True)

# ══════════════════════════════════════════════════════════════════════════
# 4. ENTRENAR M6 (Dual Custom + Dropout + ReduceLR)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("── M6 — Dual Custom + Dropout + ReduceLR ───────────────────")
print("─"*60)
print("Justificación: modelo completo con capas custom;")
print("¿la convergencia larga mejora el AUC sobre el M6 original?")

keras.utils.set_random_seed(SEED)
model_m6 = build_model_m6(N_FEATURES, DEBT_RATIO_IDX, EXT_IDXS)
hist_m6, auc_m6_te, auc_m6_val, ep_m6 = \
    train_long(model_m6, 'M6')

save_checkpoint(model_m6, hist_m6, 'M6_largo')
plot_curves_with_lr(hist_m6, 'M6 — Dual Custom + Dropout + ReduceLR',
                    zoom_n=40, save=True)

# Pesos aprendidos por ExtSourceLayer
ext_layer  = model_m6.get_layer('ext_source_index')
debt_layer = model_m6.get_layer('debt_saturation')
print(f"\n  Pesos aprendidos ExtSourceLayer:")
for name, w in zip(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'],
                    ext_layer.w.numpy()):
    print(f"    {name}: {w:.4f}")
print(f"  DebtRatioLayer: α={debt_layer.slope.numpy()[0]:.4f}, "
      f"θ={debt_layer.threshold.numpy()[0]:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. TABLA COMPARATIVA FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("TABLA COMPARATIVA — Entrenamiento largo vs original")
print("═"*60)

resultados = [
    {'nombre': 'M2  (original, ~20 épocas)',
     'val_auc': 0.7411, 'test_auc': 0.7447, 'n_params': 9985,  'n_epochs': 18},
    {'nombre': 'M2* (ReduceLR, largo)',
     'val_auc': auc_m2star_val, 'test_auc': auc_m2star_te,
     'n_params': model_m2star.count_params(), 'n_epochs': ep_m2star},
    {'nombre': 'M3  (original, ~45 épocas)',
     'val_auc': 0.7424, 'test_auc': 0.7452, 'n_params': 9985,  'n_epochs': 47},
    {'nombre': 'M3  (Dropout+ReduceLR, largo)',
     'val_auc': auc_m3_val, 'test_auc': auc_m3_te,
     'n_params': model_m3.count_params(), 'n_epochs': ep_m3},
    {'nombre': 'M6  (original, ~50 épocas)',
     'val_auc': 0.7421, 'test_auc': 0.7451, 'n_params': 9993,  'n_epochs': 52},
    {'nombre': 'M6  (DualCustom+ReduceLR, largo)',
     'val_auc': auc_m6_val, 'test_auc': auc_m6_te,
     'n_params': model_m6.count_params(), 'n_epochs': ep_m6},
]
print_model_table(resultados)

# ── Gráfico comparativo final ─────────────────────────────────────────────
nombres   = [r['nombre'] for r in resultados]
aucs_test = [r['test_auc'] for r in resultados]
colores   = ['#AED6F1','#2980B9','#A9DFBF','#27AE60','#F9E79F','#F39C12']

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(nombres, aucs_test, color=colores, alpha=0.9, edgecolor='gray')
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=10)
ax.axvline(0.7447, color='gray', ls='--', lw=1, alpha=0.5,
           label='M2 original (0.7447)')
ax.set_xlabel('AUC-ROC (test)', fontsize=11)
ax.set_title('Impacto del entrenamiento largo\n'
             '(mismo modelo, más épocas + ReduceLR)',
             fontsize=12, fontweight='bold')
ax.set_xlim(min(aucs_test) - 0.003, max(aucs_test) + 0.005)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
savefig('11_long_training_comparison.pdf')
plt.show()

print("\n" + "═"*60)
print("✅ ENTRENAMIENTO LARGO COMPLETADO")
print("   Checkpoints: M2star_ReduceLR, M3_largo, M6_largo")
print("   Gráficos:    report_plots/")
print("═"*60)
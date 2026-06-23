"""
uncertainty_experiment.py
─────────────────────────
Entrena FAIR(λ=1.0) — el modelo recomendado — y regenera todas las
figuras de incertidumbre con ese modelo en vez de λ=0.5.

Figuras que se regeneran:
  · 09_uncertainty_by_class.pdf
  · 10_uncertainty_vs_missing.pdf

Ejecutar desde la raíz del proyecto:
    python uncertainty_experiment.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')   # Cambiar a 'Agg' si no hay display
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

from src.preprocessing import full_pipeline, FEATURE_COLS
from src.models        import build_model_m6
from src.fair_loss     import FairBCELoss, FairAUC, fairness_metrics
from src.train         import get_class_weights
from src.uncertainty   import mc_dropout_predict, uncertainty_by_class, uncertainty_by_missing
from src.checkpoints   import save_checkpoint, load_checkpoint
from src.visualization import savefig

sns.set_theme(style='whitegrid')
os.makedirs('checkpoints',  exist_ok=True)
os.makedirs('report_plots', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. DATOS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("INCERTIDUMBRE — FAIR(λ=1.0) como modelo recomendado")
print("="*60)

CSV_PATH = 'application_train.csv'
if not os.path.exists(CSV_PATH):
    try:
        import kagglehub, glob, shutil
        dataset_path = kagglehub.dataset_download(
            "megancrenshaw/home-credit-default-risk")
        matches = glob.glob(
            os.path.join(dataset_path, "**", "application_train.csv"),
            recursive=True)
        shutil.copy(matches[0], CSV_PATH)
        print(f"  Dataset descargado en '{CSV_PATH}'")
    except Exception as e:
        raise RuntimeError(f"Pon application_train.csv en la raíz. Error: {e}")

print("\n── 1. Cargando datos ─────────────────────────────────────────")
(X_tr,  y_tr,  s_tr), \
(X_val, y_val, s_val), \
(X_te,  y_te,  s_te), \
FEATURE_COLS, scaler, medians = full_pipeline(CSV_PATH)

N_FEATURES     = X_tr.shape[1]
DEBT_RATIO_IDX = FEATURE_COLS.index('DEBT_RATIO')
EXT_IDXS       = [FEATURE_COLS.index(c) for c in
                  ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]

CLASS_WEIGHT = get_class_weights(y_tr)

# Máscaras de ausencia para el análisis de incertidumbre
MISSING_IDXS = [FEATURE_COLS.index(f'EXT_SOURCE_{i}_MISSING')
                for i in [1, 2, 3]]
missing_matrix = X_te[:, MISSING_IDXS]
n_missing      = missing_matrix.sum(axis=1).astype(int)

print(f"\nDistribución de nulos en test:")
for k, v in zip(*np.unique(n_missing, return_counts=True)):
    print(f"  {k} fuentes ausentes: {v:6,} ({v/len(n_missing)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# 2. FAIR(λ=1.0) — cargar checkpoint o entrenar si no existe
# ══════════════════════════════════════════════════════════════════════════
print("\n── 2. Modelo FAIR(λ=1.0) ─────────────────────────────────────")

# Usamos seed=42 (mejor AUC: 0.7424 con DP gap=0.0018)
LAMBDA_REC = 1.0
SEED_REC   = 42
CKPT_NAME  = f'FAIR_lam{LAMBDA_REC}_seed{SEED_REC}'

# Preparar datos FAIR
y_tr_fair  = np.stack([y_tr,  s_tr],  axis=1)
y_val_fair = np.stack([y_val, s_val], axis=1)
sw_tr = np.where(y_tr == 1, CLASS_WEIGHT[1], CLASS_WEIGHT[0])

keras.utils.set_random_seed(SEED_REC)
model_fair = build_model_m6(N_FEATURES, DEBT_RATIO_IDX, EXT_IDXS)
model_fair.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=FairBCELoss(lambda_fair=LAMBDA_REC),
    metrics=[FairAUC(name='auc')]
)

ckpt_path = os.path.join('checkpoints', f'{CKPT_NAME}.weights.h5')
if os.path.exists(ckpt_path):
    print(f"  Cargando checkpoint: {ckpt_path}")
    model_fair.load_weights(ckpt_path)
    print("  ✅ Modelo cargado desde checkpoint")
else:
    print(f"  Checkpoint no encontrado — entrenando FAIR(λ={LAMBDA_REC}, seed={SEED_REC})...")
    model_fair.fit(
        X_tr, y_tr_fair,
        validation_data=(X_val, y_val_fair),
        sample_weight=sw_tr,
        epochs=100, batch_size=512,
        callbacks=[
            EarlyStopping(monitor='val_auc', mode='max',
                          patience=15, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', mode='max',
                factor=0.3, patience=5, min_lr=1e-6, verbose=0)
        ],
        verbose=0
    )
    model_fair.save_weights(ckpt_path)
    print(f"  ✅ Entrenado y guardado en {ckpt_path}")

# Verificar métricas
auc_fair, dp_fair, mF, mM = fairness_metrics(model_fair, X_te, y_te, s_te)
print(f"\n  FAIR(λ=1.0, seed=42):")
print(f"  AUC test = {auc_fair:.4f}  |  DP gap = {dp_fair:.4f}")
print(f"  P̂(M) = {mM:.4f}  |  P̂(F) = {mF:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. M6 BASE — cargar checkpoint o entrenar
# ══════════════════════════════════════════════════════════════════════════
print("\n── 3. Modelo M6 base ─────────────────────────────────────────")

keras.utils.set_random_seed(42)
model_m6 = build_model_m6(N_FEATURES, DEBT_RATIO_IDX, EXT_IDXS)
model_m6.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=[keras.metrics.AUC(name='auc')]
)

ckpt_m6 = os.path.join('checkpoints', 'M6_DualCustom.weights.h5')
if os.path.exists(ckpt_m6):
    print(f"  Cargando checkpoint: {ckpt_m6}")
    model_m6.load_weights(ckpt_m6)
    print("  ✅ M6 cargado desde checkpoint")
else:
    print("  Entrenando M6 base...")
    model_m6.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=512,
        class_weight=CLASS_WEIGHT,
        callbacks=[
            EarlyStopping(monitor='val_auc', mode='max',
                          patience=15, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', mode='max',
                factor=0.3, patience=5, min_lr=1e-6, verbose=0)
        ],
        verbose=0
    )
    model_m6.save_weights(ckpt_m6)
    print(f"  ✅ M6 base entrenado y guardado")

auc_m6 = roc_auc_score(y_te, model_m6.predict(X_te, verbose=0).ravel())
print(f"\n  M6 base: AUC test = {auc_m6:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. MC DROPOUT — 100 pasadas para ambos modelos
# ══════════════════════════════════════════════════════════════════════════
print("\n── 4. MC Dropout (N=100 pasadas) ─────────────────────────────")
N_PASSES = 100

print("\n  M6 base:")
mean_m6, var_m6, _ = mc_dropout_predict(model_m6, X_te, N_PASSES)

print("\n  FAIR(λ=1.0):")
mean_fair, var_fair, _ = mc_dropout_predict(model_fair, X_te, N_PASSES)

# AUC con media MC
auc_m6_mc   = roc_auc_score(y_te, mean_m6)
auc_fair_mc = roc_auc_score(y_te, mean_fair)

# Guardar arrays para no repetir
np.save('checkpoints/mc_m6_mean.npy',       mean_m6)
np.save('checkpoints/mc_m6_var.npy',        var_m6)
np.save('checkpoints/mc_fair10_mean.npy',   mean_fair)
np.save('checkpoints/mc_fair10_var.npy',    var_fair)
print("\n  Arrays MC guardados en checkpoints/")

# ══════════════════════════════════════════════════════════════════════════
# 5. TABLA RESUMEN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESUMEN — MC Dropout (N=100 pasadas)")
print("="*60)
print(f"\n{'Métrica':<40} {'M6 base':>12} {'FAIR λ=1.0':>12}")
print("─"*66)

rows = [
    ('AUC predicción puntual',   auc_m6,                         auc_fair),
    ('AUC media MC Dropout',     auc_m6_mc,                      auc_fair_mc),
    ('Var. media global',        var_m6.mean(),                  var_fair.mean()),
    ('Var. media TARGET=0',      var_m6[y_te==0].mean(),         var_fair[y_te==0].mean()),
    ('Var. media TARGET=1',      var_m6[y_te==1].mean(),         var_fair[y_te==1].mean()),
    ('Ratio var(T=1)/var(T=0)',  var_m6[y_te==1].mean()/var_m6[y_te==0].mean(),
                                 var_fair[y_te==1].mean()/var_fair[y_te==0].mean()),
    ('Var. 0 EXT ausentes',      var_m6[n_missing==0].mean(),    var_fair[n_missing==0].mean()),
    ('Var. 1 EXT ausente',       var_m6[n_missing==1].mean(),    var_fair[n_missing==1].mean()),
    ('Var. 2 EXT ausentes',      var_m6[n_missing==2].mean(),    var_fair[n_missing==2].mean()),
    ('Var. 3 EXT ausentes',      var_m6[n_missing==3].mean(),    var_fair[n_missing==3].mean()),
]
for label, v_m6, v_fair in rows:
    print(f"  {label:<38} {v_m6:>12.6f} {v_fair:>12.6f}")

# ══════════════════════════════════════════════════════════════════════════
# 6. FIGURA 09 — Distribución de incertidumbre por clase
# ══════════════════════════════════════════════════════════════════════════
print("\n── 5. Generando figura 09 ────────────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Distribución de Incertidumbre (Varianza MC Dropout)\n'
             'por clase TARGET', fontsize=14, fontweight='bold')

for ax, (var, mname) in zip(axes, [
    (var_m6,   'M6 — Base (sin FAIR)'),
    (var_fair, 'M6 — FAIR (λ=1.0)')
]):
    sns.kdeplot(var[y_te == 0], ax=ax, fill=True, alpha=0.5,
                color='steelblue', label='TARGET=0 (pagó)')
    sns.kdeplot(var[y_te == 1], ax=ax, fill=True, alpha=0.5,
                color='tomato',    label='TARGET=1 (impago)')
    for t, c in [(0, 'steelblue'), (1, 'tomato')]:
        m = np.median(var[y_te == t])
        ax.axvline(m, color=c, ls='--', lw=1.5,
                   label=f'Mediana T={t}: {m:.5f}')
    ax.set_xlim(0, np.percentile(var, 99))
    ax.set(xlabel='Varianza (incertidumbre)', ylabel='Densidad', title=mname)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
savefig('09_uncertainty_by_class.pdf')
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 7. FIGURA 10 — Incertidumbre vs calidad de EXT_SOURCE
# ══════════════════════════════════════════════════════════════════════════
print("\n── 6. Generando figura 10 ────────────────────────────────────")

colors_mis = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
fig, axes  = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Incertidumbre vs Calidad de EXT_SOURCE\n'
             '(0 = info completa, 3 = todo imputado)',
             fontsize=13, fontweight='bold')

for ax, (var, mname) in zip(axes, [
    (var_m6,   'M6 Base'),
    (var_fair, 'FAIR λ=1.0')
]):
    medias = [var[n_missing == k].mean() for k in range(4)]
    stds   = [var[n_missing == k].std()  for k in range(4)]
    counts = [(n_missing == k).sum()      for k in range(4)]
    errors = [1.96 * s / np.sqrt(c) if c > 0 else 0
              for s, c in zip(stds, counts)]

    ax.bar(range(4), medias, yerr=errors, capsize=5,
           color=colors_mis, alpha=0.85,
           error_kw=dict(elinewidth=1.5))
    for i, (m, c) in enumerate(zip(medias, counts)):
        ax.text(i, m + errors[i] * 1.1, f'n={c:,}',
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['0 aus.', '1 aus.', '2 aus.', '3 aus.'])
    ax.set(ylabel='Varianza media (± IC 95%)', title=mname)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
savefig('10_uncertainty_vs_missing.pdf')
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 8. FIGURA EXTRA — Comparativa directa M6 vs FAIR(λ=1.0)
# ══════════════════════════════════════════════════════════════════════════
print("\n── 7. Generando figura comparativa extra ─────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparación M6 vs FAIR(λ=1.0) — Análisis de Incertidumbre',
             fontsize=14, fontweight='bold')

# Scatter varianza por muestra
ax = axes[0]
idx_s = np.random.choice(len(var_m6), size=3000, replace=False)
sc = ax.scatter(var_m6[idx_s], var_fair[idx_s],
                c=y_te[idx_s], cmap='RdBu_r',
                alpha=0.4, s=8, vmin=0, vmax=1)
lim = max(var_m6[idx_s].max(), var_fair[idx_s].max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='y=x')
ax.set(xlabel='Varianza M6 base', ylabel='Varianza FAIR λ=1.0',
       title='Varianza por muestra\n(azul=pagó, rojo=impago)')
plt.colorbar(sc, ax=ax, label='TARGET')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# AUC comparativa
ax2 = axes[1]
resultados_auc = {
    'M6\n(punt.)':    auc_m6,
    'M6\n(MC-mean)':  auc_m6_mc,
    'FAIR\n(punt.)':  auc_fair,
    'FAIR\n(MC-mean)':auc_fair_mc,
}
colors_auc = ['steelblue', 'dodgerblue', 'tomato', 'firebrick']
bars = ax2.bar(resultados_auc.keys(), resultados_auc.values(),
               color=colors_auc, alpha=0.8)
for bar, val in zip(bars, resultados_auc.values()):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.0003,
             f'{val:.4f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
ax2.set(ylabel='AUC-ROC (test)',
        title='AUC: predicción puntual\nvs media MC Dropout')
ax2.set_ylim(min(resultados_auc.values()) - 0.003,
             max(resultados_auc.values()) + 0.003)
ax2.grid(alpha=0.3, axis='y')

# Varianza por ausentes: M6 vs FAIR
ax3 = axes[2]
x = np.arange(4); w = 0.35
vm6_m   = [var_m6[n_missing == k].mean()   for k in range(4)]
vfair_m = [var_fair[n_missing == k].mean() for k in range(4)]
ax3.bar(x - w/2, vm6_m,   width=w, color='steelblue', alpha=0.8, label='M6 base')
ax3.bar(x + w/2, vfair_m, width=w, color='tomato',    alpha=0.8, label='FAIR λ=1.0')
ax3.set_xticks(x)
ax3.set_xticklabels(['0 aus.', '1 aus.', '2 aus.', '3 aus.'])
ax3.set(ylabel='Varianza media',
        title='Varianza por nº EXT_SOURCE\nausentes: M6 vs FAIR')
ax3.legend(fontsize=9); ax3.grid(alpha=0.3, axis='y')

plt.tight_layout()
savefig('09b_uncertainty_comparison.pdf')
plt.show()

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("✅ EXPERIMENTO DE INCERTIDUMBRE COMPLETADO")
print("   Modelo FAIR usado: λ=1.0 (recomendado), seed=42")
print("   Figuras generadas:")
print("     · report_plots/09_uncertainty_by_class.pdf")
print("     · report_plots/10_uncertainty_vs_missing.pdf")
print("     · report_plots/09b_uncertainty_comparison.pdf")
print("="*60)
print("\n⚠️  Actualiza la tabla de incertidumbre del main.tex con:")

# Imprimir los valores clave para el report
v0_fair = var_fair[y_te==0].mean()
v1_fair = var_fair[y_te==1].mean()
v0_miss = var_fair[n_missing==0].mean()
v3_miss = var_fair[n_missing==3].mean()
print(f"""
   Var. media TARGET=0:      {v0_fair:.6f}
   Var. media TARGET=1:      {v1_fair:.6f}
   Ratio var(T=1)/var(T=0):  {v1_fair/v0_fair:.2f}
   Var. 0 EXT ausentes:      {v0_miss:.6f}
   Var. 3 EXT ausentes:      {v3_miss:.6f}
   Incremento 0→3 ausentes:  {(v3_miss/v0_miss - 1)*100:.0f}%
""")
"""
pareto_experiment.py
────────────────────
Experimento Pareto mejorado: entrena FAIR Loss con múltiples semillas
por cada valor de lambda para obtener una curva de Pareto más estable.

Ejecutar desde la raíz del proyecto:
    python pareto_experiment.py

Requisito: application_train.csv en la raíz del proyecto.
Resultados guardados en: report_plots/08_pareto_fair.pdf
                         checkpoints/pareto_v2_results.npy
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')   # Cambia a 'Agg' si no tienes display (servidor)
import matplotlib.pyplot as plt
import keras

# ── Imports del proyecto ──────────────────────────────────────────────────
from src.preprocessing  import full_pipeline, FEATURE_COLS
from src.models         import build_model_m6
from src.fair_loss      import FairBCELoss, FairAUC, fairness_metrics
from src.train          import get_class_weights
from src.visualization  import plot_pareto, print_pareto_table, savefig
from src.checkpoints    import save_checkpoint

from keras.callbacks import EarlyStopping

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH   = 'application_train.csv'
LAMBDAS    = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS      = [42, 123, 7]
MAX_EPOCHS = 100
PATIENCE   = 15
BATCH_SIZE = 512

os.makedirs('checkpoints',  exist_ok=True)
os.makedirs('report_plots', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("EXPERIMENTO PARETO — FAIR Loss (multi-semilla)")
print("═"*60)

print("\n── 1. Cargando y preprocesando datos ────────────────────────")

if not os.path.exists(CSV_PATH):
    print(f"  '{CSV_PATH}' no encontrado — descargando desde Kaggle...")
    try:
        import kagglehub
        import pandas as pd
        import glob

        # dataset_download descarga los archivos a una carpeta local en cache
        dataset_path = kagglehub.dataset_download(
            "megancrenshaw/home-credit-default-risk"
        )
        print(f"  Dataset descargado en: {dataset_path}")

        # Buscar el archivo application_train.csv en la carpeta descargada
        matches = glob.glob(
            os.path.join(dataset_path, "**", "application_train.csv"),
            recursive=True
        )
        if not matches:
            # Listar todos los CSV disponibles para depurar
            all_csv = glob.glob(os.path.join(dataset_path, "**", "*.csv"),
                                recursive=True)
            raise FileNotFoundError(
                f"No se encontro application_train.csv.\n"
                f"CSVs disponibles: {all_csv}"
            )

        import shutil
        shutil.copy(matches[0], CSV_PATH)
        n = len(pd.read_csv(CSV_PATH, nrows=1))  # solo comprueba que abre
        print(f"  Copiado a '{CSV_PATH}'")

    except Exception as e:
        raise RuntimeError(
            f"No se pudo descargar el dataset: {e}\n"
            "Opciones:\n"
            "  1. pip install kagglehub y configura ~/.kaggle/kaggle.json\n"
            "  2. Descarga manualmente application_train.csv "
            "y ponlo en la raiz del proyecto."
        )
(X_tr,  y_tr,  s_tr), \
(X_val, y_val, s_val), \
(X_te,  y_te,  s_te), \
FEATURE_COLS, scaler, medians = full_pipeline(CSV_PATH)

N_FEATURES     = X_tr.shape[1]
DEBT_RATIO_IDX = FEATURE_COLS.index('DEBT_RATIO')
EXT_IDXS       = [FEATURE_COLS.index(c) for c in
                  ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]

print(f"\n  Features: {N_FEATURES}  |  DEBT_RATIO idx: {DEBT_RATIO_IDX}")
print(f"  EXT_SOURCE idxs: {EXT_IDXS}")

# ── Class weights y datos FAIR ────────────────────────────────────────────
print("\n── 2. Preparando pesos y datos FAIR ─────────────────────────")
CLASS_WEIGHT = get_class_weights(y_tr)

# y_combined: (N, 2) → col 0 = TARGET, col 1 = género
y_tr_fair  = np.stack([y_tr,  s_tr],  axis=1)
y_val_fair = np.stack([y_val, s_val], axis=1)

# sample_weight: equivalente a class_weight para y de forma (N, 2)
sw_tr = np.where(y_tr == 1, CLASS_WEIGHT[1], CLASS_WEIGHT[0])

dp_base_real = abs(y_tr[s_tr == 0].mean() - y_tr[s_tr == 1].mean())
print(f"\n  DP gap real en train: {dp_base_real:.4f}")
print(f"  (H={y_tr[s_tr==0].mean():.4f}, F={y_tr[s_tr==1].mean():.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPERIMENTO PARETO MULTI-SEMILLA
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. Entrenando modelos FAIR ────────────────────────────────")
print(f"   λ values:  {LAMBDAS}")
print(f"   Seeds:     {SEEDS}")
print(f"   Total entrenamientos: {len(LAMBDAS) * len(SEEDS)}\n")

pareto = []

for lam in LAMBDAS:
    print(f"\n{'─'*55}")
    print(f"  λ = {lam}")
    print(f"{'─'*55}")
    resultados_lam = []

    for seed in SEEDS:
        keras.utils.set_random_seed(seed)

        m = build_model_m6(N_FEATURES, DEBT_RATIO_IDX, EXT_IDXS)
        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=FairBCELoss(lambda_fair=lam),
            metrics=[FairAUC(name='auc')]
        )

        m.fit(
            X_tr, y_tr_fair,
            validation_data=(X_val, y_val_fair),
            sample_weight=sw_tr,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                EarlyStopping(
                    monitor='val_auc', mode='max',
                    patience=PATIENCE,
                    restore_best_weights=True,
                    verbose=0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_auc', mode='max',
                    factor=0.3, patience=5,
                    min_lr=1e-6, verbose=0
                )
            ],
            verbose=0
        )

        auc, dp, mF, mM = fairness_metrics(m, X_te, y_te, s_te)
        resultados_lam.append({
            'auc': auc, 'dp': dp, 'mean_F': mF, 'mean_M': mM
        })
        print(f"  seed={seed:>3}: AUC={auc:.4f}  DP gap={dp:.4f}  "
              f"(M:{mM:.4f}, F:{mF:.4f})")

        # Guardar checkpoint de cada modelo
        save_checkpoint(
            m,
            type('H', (), {'history': {}})(),
            f'FAIR_lam{lam}_seed{seed}'
        )

    # Agregar resultados de las 3 semillas
    pareto.append({
        'lambda':   lam,
        'auc':      np.mean([r['auc']    for r in resultados_lam]),
        'auc_std':  np.std( [r['auc']    for r in resultados_lam]),
        'dp':       np.mean([r['dp']     for r in resultados_lam]),
        'dp_std':   np.std( [r['dp']     for r in resultados_lam]),
        'mean_F':   np.mean([r['mean_F'] for r in resultados_lam]),
        'mean_M':   np.mean([r['mean_M'] for r in resultados_lam]),
    })
    p = pareto[-1]
    print(f"\n  ► Media λ={lam}: "
          f"AUC={p['auc']:.4f}±{p['auc_std']:.4f}  "
          f"DP={p['dp']:.4f}±{p['dp_std']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RESULTADOS Y VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════
df_pareto = pd.DataFrame(pareto)

# Guardar resultados numéricos
df_pareto.to_csv('checkpoints/pareto_v2_results.csv', index=False)
print("\n✅ Resultados guardados en checkpoints/pareto_v2_results.csv")

print("\n── 4. Tabla de resultados ────────────────────────────────────")
print_pareto_table(df_pareto)

print("\n── 5. Visualización ──────────────────────────────────────────")
plot_pareto(df_pareto, save=True)

# ── Comparativa con experimento anterior (si existe) ─────────────────────
prev_path = 'checkpoints/pareto_v1_results.csv'
if os.path.exists(prev_path):
    df_v1 = pd.read_csv(prev_path)
    print("\n── 6. Comparativa v1 (1 semilla) vs v2 (3 semillas) ─────────")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_v1['dp'],     df_v1['auc'],
            'o--', color='gray', lw=1.5, ms=8, label='v1 (1 semilla)')
    ax.errorbar(df_pareto['dp'], df_pareto['auc'],
                xerr=df_pareto['dp_std'], yerr=df_pareto['auc_std'],
                fmt='o-', color='steelblue', ecolor='lightblue',
                elinewidth=2, capsize=4, ms=10, lw=2,
                label='v2 (media 3 semillas ± std)')

    for _, row in df_pareto.iterrows():
        ax.annotate(f"λ={row['lambda']:.1f}",
                    xy=(row['dp'], row['auc']),
                    xytext=(6, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('DP Gap  |E[ŷ|G=M] − E[ŷ|G=F]|', fontsize=11)
    ax.set_ylabel('AUC-ROC (test)', fontsize=11)
    ax.set_title('Pareto: 1 semilla vs media de 3 semillas\n'
                 'Las barras de error muestran la varianza de inicialización',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('08b_pareto_comparativa.pdf')
    plt.show()

print("\n" + "═"*60)
print("✅ EXPERIMENTO PARETO COMPLETADO")
print(f"   Gráficos guardados en: report_plots/")
print(f"   Resultados en:         checkpoints/pareto_v2_results.csv")
print("═"*60)
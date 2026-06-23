"""
main.py
───────
Script de verificación del proyecto.
Comprueba que todos los módulos de src/ se importan correctamente
y que las clases y funciones principales funcionan sin datos reales.

Ejecutar desde la raíz del proyecto:
    python main.py
"""

import sys
import numpy as np

print("=" * 60)
print("VERIFICACIÓN DEL PROYECTO — Taller B4-T1")
print("=" * 60)

# ── 1. Imports de módulos propios ─────────────────────────────────────────
print("\n── 1. Importando módulos src/ ───────────────────────────────")
errors = []

try:
    from src.preprocessing import (
        full_pipeline, load_and_clean, engineer_features,
        split_data, apply_transformations, FEATURE_COLS
    )
    print("  ✅ src.preprocessing")
except Exception as e:
    print(f"  ❌ src.preprocessing: {e}")
    errors.append(e)

try:
    from src.custom_layers import DebtRatioLayer, ExtSourceLayer
    print("  ✅ src.custom_layers")
except Exception as e:
    print(f"  ❌ src.custom_layers: {e}")
    errors.append(e)

try:
    from src.fair_loss import FairBCELoss, FairAUC, fairness_metrics
    print("  ✅ src.fair_loss")
except Exception as e:
    print(f"  ❌ src.fair_loss: {e}")
    errors.append(e)

try:
    from src.models import (
        build_model_m0, build_model_m1, build_model_m2,
        build_model_m3, build_model_m4, build_model_m6
    )
    print("  ✅ src.models")
except Exception as e:
    print(f"  ❌ src.models: {e}")
    errors.append(e)

try:
    from src.train import get_class_weights, compile_and_train, compile_and_train_fair
    print("  ✅ src.train")
except Exception as e:
    print(f"  ❌ src.train: {e}")
    errors.append(e)

try:
    from src.uncertainty import (
        mc_dropout_predict, uncertainty_by_class, uncertainty_by_missing
    )
    print("  ✅ src.uncertainty")
except Exception as e:
    print(f"  ❌ src.uncertainty: {e}")
    errors.append(e)

if errors:
    print(f"\n❌ {len(errors)} error(es) en los imports. Revisar arriba.")
    sys.exit(1)

# ── 2. Verificar capas custom con datos sintéticos ────────────────────────
print("\n── 2. Verificando capas custom ──────────────────────────────")

try:
    layer_debt = DebtRatioLayer()
    layer_debt.build((None, 1))
    test_ratios = np.array([[0.10],[0.35],[0.80],[1.50]], dtype=np.float32)
    out_debt    = layer_debt(test_ratios).numpy().ravel()
    print(f"  DebtRatioLayer (α=10, θ=0.35):")
    for r, o in zip(test_ratios.ravel(), out_debt):
        print(f"    ratio={r:.2f} → salida={o:.4f}")

    # Verificar monotonía: salida debe crecer con el ratio
    assert all(out_debt[i] <= out_debt[i+1] for i in range(len(out_debt)-1)), \
        "DebtRatioLayer NO es monótona creciente"
    print("  ✅ DebtRatioLayer — monotonía verificada")
except Exception as e:
    print(f"  ❌ DebtRatioLayer: {e}")
    errors.append(e)

try:
    layer_ext = ExtSourceLayer()
    layer_ext.build((None, 3))
    test_ext = np.array([[0.1, 0.1, 0.1],
                          [0.5, 0.5, 0.5],
                          [0.9, 0.9, 0.9]], dtype=np.float32)
    out_ext  = layer_ext(test_ext).numpy().ravel()
    print(f"\n  ExtSourceLayer (pesos=1/3, b=0):")
    for src, o in zip(['bajos (0.1)', 'medios (0.5)', 'altos (0.9)'], out_ext):
        print(f"    EXT={src} → salida={o:.4f}")

    assert out_ext[0] < out_ext[1] < out_ext[2], \
        "ExtSourceLayer NO es monótona creciente"
    print("  ✅ ExtSourceLayer — monotonía verificada")
except Exception as e:
    print(f"  ❌ ExtSourceLayer: {e}")
    errors.append(e)

# ── 3. Verificar FairBCELoss ──────────────────────────────────────────────
print("\n── 3. Verificando FairBCELoss ────────────────────────────────")
try:
    loss_fn = FairBCELoss(lambda_fair=0.5)
    y_comb  = np.array([[1,0],[0,1],[1,1],[0,0]], dtype=np.float32)
    y_pred  = np.array([[0.8],[0.2],[0.7],[0.3]], dtype=np.float32)
    val     = loss_fn(y_comb, y_pred).numpy()
    print(f"  FairBCELoss(λ=0.5) sobre datos de prueba: {val:.4f}")
    assert val > 0, "La pérdida debe ser positiva"

    # λ=0 debe ser igual a BCE puro
    loss_bce  = FairBCELoss(lambda_fair=0.0)(y_comb, y_pred).numpy()
    loss_fair = FairBCELoss(lambda_fair=1.0)(y_comb, y_pred).numpy()
    assert loss_fair >= loss_bce, \
        "FairLoss con λ>0 debe ser >= BCE puro"
    print(f"  BCE puro (λ=0): {loss_bce:.4f}  |  FAIR (λ=1): {loss_fair:.4f}")
    print("  ✅ FairBCELoss — comportamiento verificado")
except Exception as e:
    print(f"  ❌ FairBCELoss: {e}")
    errors.append(e)

# ── 4. Verificar construcción de modelos ──────────────────────────────────
print("\n── 4. Verificando construcción de modelos ────────────────────")

N_FEATURES      = len(FEATURE_COLS)
DEBT_RATIO_IDX  = FEATURE_COLS.index('DEBT_RATIO')
EXT_IDXS        = [FEATURE_COLS.index(c) for c in
                   ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]

modelos = [
    ('M0', build_model_m0(N_FEATURES)),
    ('M1', build_model_m1(N_FEATURES)),
    ('M2', build_model_m2(N_FEATURES)),
    ('M3', build_model_m3(N_FEATURES)),
    ('M4', build_model_m4(N_FEATURES, DEBT_RATIO_IDX)),
    ('M6', build_model_m6(N_FEATURES, DEBT_RATIO_IDX, EXT_IDXS)),
]

try:
    print(f"\n  {'Modelo':<8} {'Parámetros':>12} {'Salida':>12}")
    print(f"  {'─'*34}")
    for nombre, modelo in modelos:
        n_params  = modelo.count_params()
        dummy_in  = np.zeros((4, N_FEATURES), dtype=np.float32)
        dummy_out = modelo(dummy_in, training=False).numpy()
        assert dummy_out.shape == (4, 1), f"Salida inesperada: {dummy_out.shape}"
        assert dummy_out.min() >= 0 and dummy_out.max() <= 1, \
            "Salida fuera de [0,1]"
        print(f"  {nombre:<8} {n_params:>12,} {str(dummy_out.shape):>12}")
    print("  ✅ Todos los modelos construidos y con salida en [0,1]")
except Exception as e:
    print(f"  ❌ Construcción de modelos: {e}")
    errors.append(e)

# ── 5. Verificar MC Dropout ───────────────────────────────────────────────
print("\n── 5. Verificando MC Dropout ─────────────────────────────────")
try:
    import keras
    keras.utils.set_random_seed(42)
    modelo_test = build_model_m3(N_FEATURES)
    X_dummy = np.random.randn(50, N_FEATURES).astype(np.float32)

    mean_p, var_p, all_p = mc_dropout_predict(modelo_test, X_dummy, n_passes=10)

    assert mean_p.shape == (50,), f"Shape inesperado: {mean_p.shape}"
    assert var_p.shape  == (50,), f"Shape inesperado: {var_p.shape}"
    assert all_p.shape  == (50, 10)
    assert var_p.mean() > 0, "Varianza cero: Dropout no está activo"

    print(f"  Varianza media (10 pasadas, modelo sin entrenar): {var_p.mean():.6f}")
    print("  ✅ MC Dropout — varianza > 0 (Dropout activo en inferencia)")
except Exception as e:
    print(f"  ❌ MC Dropout: {e}")
    errors.append(e)

# ── Resumen final ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not errors:
    print("✅ TODOS LOS MÓDULOS VERIFICADOS CORRECTAMENTE")
    print(f"   {len(FEATURE_COLS)} features | {len(modelos)} modelos | "
          f"FairLoss ✓ | MC Dropout ✓")
else:
    print(f"❌ {len(errors)} ERROR(ES) ENCONTRADO(S) — revisar arriba")
print("=" * 60)
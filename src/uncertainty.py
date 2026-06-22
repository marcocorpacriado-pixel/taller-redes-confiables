"""
uncertainty.py
──────────────
Cuantificación de incertidumbre mediante Monte Carlo Dropout.

Método: MC Dropout (Gal & Ghahramani, 2016)
  - Mantiene las capas Dropout ACTIVAS durante la inferencia
  - Realiza N pasadas estocásticas hacia adelante por muestra
  - La varianza entre pasadas = incertidumbre epistémica del modelo

Justificación frente a alternativas:
  · Redes Bayesianas: requieren rediseñar toda la arquitectura
  · Deep Ensembles: N modelos × 6 lambdas = entrenamientos inviables
  · MC Dropout: M6 ya tiene Dropout(0.30, 0.20) → coste de rediseño = 0
    Y su fundamento teórico como aproximación variacional bayesiana
    está establecido (Gal & Ghahramani, 2016, ICML).

Conexión con el EDA:
  Las máscaras EXT_SOURCE_*_MISSING generadas en preprocessing
  permiten cruzar la incertidumbre con la calidad de los datos de entrada.
"""

import numpy as np


def mc_dropout_predict(model, X: np.ndarray,
                       n_passes: int = 100,
                       batch_size: int = 512) -> tuple:
    """
    Inferencia con MC Dropout.

    Parámetros:
      model    : modelo Keras con capas Dropout
      X        : array de entrada (N, n_features)
      n_passes : número de pasadas estocásticas
                 100 es el estándar empírico (Gal & Ghahramani, 2016)
      batch_size: tamaño de batch (para no saturar memoria)

    Devuelve:
      mean_pred : (N,)         predicción media sobre las N pasadas
      var_pred  : (N,)         varianza → INCERTIDUMBRE EPISTÉMICA
      all_preds : (N, n_passes) todas las predicciones individuales
    """
    all_preds = np.zeros((X.shape[0], n_passes), dtype=np.float32)

    for i in range(n_passes):
        # training=True → Dropout permanece activo en esta pasada
        preds = model(X, training=True)
        all_preds[:, i] = np.array(preds).ravel()
        if (i + 1) % 20 == 0:
            print(f"  Pasada {i+1}/{n_passes}", end='\r')

    print(f"  Completadas {n_passes} pasadas.          ")

    mean_pred = all_preds.mean(axis=1)
    var_pred  = all_preds.var(axis=1)

    return mean_pred, var_pred, all_preds


def uncertainty_by_class(var_pred: np.ndarray,
                          y_true: np.ndarray) -> dict:
    """
    Calcula estadísticas de incertidumbre por clase TARGET.

    Indicador clave: ratio var(TARGET=1) / var(TARGET=0)
      > 1 → el modelo es más incierto en impagos (comportamiento deseable)
      < 1 → el modelo es igual o más confiado en impagos (problema)
    """
    var_0 = var_pred[y_true == 0]
    var_1 = var_pred[y_true == 1]

    stats = {
        'mean_var_class0': float(var_0.mean()),
        'mean_var_class1': float(var_1.mean()),
        'median_var_class0': float(np.median(var_0)),
        'median_var_class1': float(np.median(var_1)),
        'ratio_T1_T0': float(var_1.mean() / var_0.mean())
    }

    print(f"  Varianza media TARGET=0: {stats['mean_var_class0']:.6f}")
    print(f"  Varianza media TARGET=1: {stats['mean_var_class1']:.6f}")
    print(f"  Ratio T1/T0: {stats['ratio_T1_T0']:.3f} "
          f"({'✅ más incierto en impagos' if stats['ratio_T1_T0'] > 1 else '❌ no distingue'})")

    return stats


def uncertainty_by_missing(var_pred: np.ndarray,
                            n_missing: np.ndarray) -> dict:
    """
    Calcula varianza media según cuántas EXT_SOURCE estaban ausentes.

    n_missing: array con valores 0, 1, 2 o 3
      0 → información completa (las tres EXT_SOURCE disponibles)
      1 → una fuente imputada con mediana
      2 → dos fuentes imputadas
      3 → todo imputado (caso extremo, solo 29 muestras en test)

    Indicador: ¿la varianza crece monotónamente con n_missing?
      Sí → el modelo reconoce que datos imputados = más incertidumbre
      No → el modelo no distingue entre información real e imputada
    """
    stats = {}
    print(f"\n  {'N ausentes':>10} {'n muestras':>12} {'Var media':>12}")
    print(f"  {'─'*36}")

    for k in range(4):
        mask = n_missing == k
        n    = mask.sum()
        if n > 0:
            v = var_pred[mask].mean()
            stats[k] = {'n': int(n), 'mean_var': float(v)}
            print(f"  {k:>10} {n:>12,} {v:>12.6f}")
        else:
            stats[k] = {'n': 0, 'mean_var': None}

    return stats
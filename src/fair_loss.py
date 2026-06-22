"""
fair_loss.py
────────────
Función de pérdida customizada para Aprendizaje Justo (Fair Learning).

  FairBCELoss — BCE + penalización por Paridad Demográfica (DP)
  FairAUC     — AUC compatible con y_combined de forma (batch, 2)

Formulación:
    L = BCE(y, ŷ) + λ · |E[ŷ|G=M] − E[ŷ|G=F]|

Justificación de la métrica de equidad elegida (Demographic Parity):
  - Diferenciable analíticamente → compatible con backpropagation
  - Solo requiere ŷ y G (no y_true) → grafo computacional limpio
  - Es el requisito regulatorio explícito en la Directiva Europea de IA
    y la ECOA (Equal Credit Opportunity Act): la tasa de aprobación
    predicha debe ser similar entre grupos demográficos.

Alternativas descartadas:
  - Equalized Odds: requiere y_true en el cálculo del gap → complica el grafo
  - Mutual Information: no diferenciable directamente
  - Correlación |Corr(ŷ,G)|: solo captura dependencia lineal
"""

import keras
import keras.backend as K


class FairBCELoss(keras.losses.Loss):
    """
    Función de coste que combina clasificación y equidad demográfica.

    L = BCE(y, ŷ) + λ · |E[ŷ|G=M] − E[ŷ|G=F]|

    Parámetros:
      lambda_fair : float
          Peso de la penalización de fairness.
          λ=0   → BCE puro (sin restricción de equidad)
          λ=0.5 → recomendado: −89% DP gap con solo −0.003 AUC
          λ=5   → máxima equidad, pero coste alto en AUC (−0.011)

    y_combined: tensor de forma (batch, 2)
      · col 0 → y_true  (etiqueta real: 0=pagó, 1=impago)
      · col 1 → G       (género: 0=Hombre, 1=Mujer)

    y_pred: tensor de forma (batch, 1)
      · probabilidad predicha de impago
    """

    def __init__(self, lambda_fair: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_fair = float(lambda_fair)

    def call(self, y_combined, y_pred):
        y_true = y_combined[:, 0:1]
        gender = y_combined[:, 1:2]

        # ── 1. Binary Cross-Entropy ───────────────────────────────────────
        bce = K.mean(keras.losses.binary_crossentropy(y_true, y_pred))

        # ── 2. Demographic Parity gap ─────────────────────────────────────
        eps   = K.epsilon()
        y_flat = K.flatten(y_pred)
        g_flat = K.flatten(gender)      # 1=Mujer, 0=Hombre

        n_F  = K.sum(g_flat) + eps
        n_M  = K.sum(1.0 - g_flat) + eps

        mean_F = K.sum(y_flat * g_flat)          / n_F
        mean_M = K.sum(y_flat * (1.0 - g_flat))  / n_M

        dp = K.abs(mean_F - mean_M)

        return bce + self.lambda_fair * dp

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'lambda_fair': self.lambda_fair})
        return cfg


class FairAUC(keras.metrics.AUC):
    """
    AUC compatible con y_combined de forma (batch, 2).

    Extrae y_true de la columna 0 e ignora la columna de género,
    de modo que la métrica durante el entrenamiento sea el AUC real
    de clasificación (no de género).
    """

    def update_state(self, y_combined, y_pred, sample_weight=None):
        y_true = y_combined[:, 0:1]
        return super().update_state(y_true, y_pred, sample_weight)


def fairness_metrics(model, X, y_true, gender):
    """
    Calcula AUC y DP gap sobre un conjunto de datos.

    Parámetros:
      model  : modelo Keras entrenado
      X      : array numpy de features
      y_true : array numpy de etiquetas reales
      gender : array numpy de géneros (0=H, 1=M)

    Devuelve:
      auc    : float — AUC-ROC en el conjunto dado
      dp     : float — |E[ŷ|H] − E[ŷ|M]| (DP gap)
      mean_F : float — predicción media para mujeres
      mean_M : float — predicción media para hombres
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np

    y_pred = model.predict(X, verbose=0).ravel()
    auc    = roc_auc_score(y_true, y_pred)
    mean_F = y_pred[gender == 1].mean()
    mean_M = y_pred[gender == 0].mean()
    dp     = abs(mean_F - mean_M)

    return auc, dp, mean_F, mean_M
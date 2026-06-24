"""
models.py
─────────
Construcción de los modelos M0–M6 con justificación de cada decisión.

Progresión de arquitecturas:
  M0 — Regresión Logística (baseline lineal)
  M1 — MLP 1 capa oculta (interacciones no lineales)
  M2 — MLP 2 capas ocultas (joroba en AMT_CREDIT)
  M3 — MLP 2 capas + Dropout (regularización)
  M4 — M3 + DebtRatioLayer (restricción regulatoria)
  M5 — M4 + ReduceLROnPlateau + más Dropout (afinar convergencia)
  M6 — Dual Custom: M4 + DebtRatioLayer + ExtSourceLayer (auditabilidad)

Decisiones comunes a todos los modelos:
  · Tipo de red: MLP denso (datos tabulares sin estructura espacial/temporal)
  · Activación ocultas: ReLU (evita vanishing gradient, estándar tabular)
  · Activación salida: sigmoid (salida en [0,1] = P(impago))
  · Métrca: AUC-ROC (insensible al umbral, correcta con 8% de positivos)
"""

import keras
from keras import layers, Model, Input
from src.custom_layers import DebtRatioLayer, ExtSourceLayer


def build_model_m0(n_features: int) -> Model:
    """
    M0 — Regresión Logística (0 capas ocultas).

    Justificación: baseline lineal. Si modelos más complejos no mejoran,
    no añaden valor. La joroba en AMT_CREDIT (deciles EDA) anticipa que SÍ
    harán falta capas ocultas.
    """
    inputs = Input(shape=(n_features,), name='input')
    output = layers.Dense(1, activation='sigmoid', name='output')(inputs)
    return Model(inputs, output, name='M0_RegresionLogistica')


def build_model_m1(n_features: int) -> Model:
    """
    M1 — MLP 1 capa oculta (64 neuronas, ReLU).

    Justificación: 1 capa oculta + ReLU permite aprender interacciones entre
    features. El hexbin mostró que EXT_SOURCE_2 bajo + edad joven = riesgo
    conjunto. Eso no es capturable sin al menos una capa no lineal.
    64 neuronas ≈ 5× el número de features (regla empírica para tabular).
    """
    inputs = Input(shape=(n_features,), name='input')
    x      = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    return Model(inputs, output, name='M1_MLP_1capa')


def build_model_m2(n_features: int) -> Model:
    """
    M2 — MLP 2 capas ocultas (128→64, ReLU).

    Justificación: AMT_CREDIT y AMT_ANNUITY muestran una JOROBA en los
    deciles (D1 bajo → D5 alto → D10 bajo). Una sola capa ReLU (lineal por
    partes con una sola quiebra) NO puede capturar ese máximo interior.
    Con 2 capas el modelo compone dos linealizaciones.
    Arquitectura en embudo (128→64): más capacidad → compresión de representación.
    """
    inputs = Input(shape=(n_features,), name='input')
    x      = layers.Dense(128, activation='relu', name='dense_1')(inputs)
    x      = layers.Dense(64,  activation='relu', name='dense_2')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    return Model(inputs, output, name='M2_MLP_2capas')


def build_model_m3(n_features: int) -> Model:
    """
    M3 — MLP 2 capas + Dropout(0.30, 0.20).

    Justificación: 128 neuronas sobre 12 features → riesgo de overfitting.
    Dropout(0.30) desactiva el 30% de neuronas aleatoriamente en cada batch:
      · El modelo no puede depender de ninguna neurona individual
      · Aprende representaciones distribuidas y más robustas
      · Actúa como ensemble de subredes implícitas
    """
    inputs = Input(shape=(n_features,), name='input')
    x      = layers.Dense(128, activation='relu', name='dense_1')(inputs)
    x      = layers.Dropout(0.30,                 name='dropout_1')(x)
    x      = layers.Dense(64,  activation='relu', name='dense_2')(x)
    x      = layers.Dropout(0.20,                 name='dropout_2')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    return Model(inputs, output, name='M3_MLP_Dropout')


def build_model_m4(n_features: int, debt_ratio_idx: int) -> Model:
    """
    M4 — M3 + DebtRatioLayer (rama custom paralela).

    Justificación de la capa custom:
      Rama paralela que satura el DEBT_RATIO con sigmoide aprendible.
      La justificación NO es predictiva sino regulatoria: un ratio de 0.8
      y uno de 1.5 son ambos 'imposible pagar'. La sigmoide garantiza que
      el modelo los trate de forma similar, incrustando la restricción
      de dominio financiero en la arquitectura.
    """
    inputs   = Input(shape=(n_features,), name='input')

    # Rama custom: DebtRatioLayer
    debt_col  = layers.Lambda(
        lambda t: t[:, debt_ratio_idx:debt_ratio_idx + 1],
        output_shape=(1,), name='extract_debt_ratio'
    )(inputs)
    saturated = DebtRatioLayer(name='debt_saturation')(debt_col)

    # Rama densa principal
    x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
    x = layers.Dropout(0.30,                 name='dropout_1')(x)
    x = layers.Dense(64,  activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.20,                 name='dropout_2')(x)

    combined = layers.Concatenate(name='concat')([x, saturated])
    output   = layers.Dense(1, activation='sigmoid', name='output')(combined)
    return Model(inputs, output, name='M4_Custom_Dropout')


def build_model_m6(n_features: int, debt_ratio_idx: int,
                   ext_source_idxs: list) -> Model:
    """
    M6 — Dual Custom: DebtRatioLayer + ExtSourceLayer + Dropout.

    Arquitectura de tres ramas paralelas:
      1. DebtRatioLayer  → señal de alarma de endeudamiento (restricción regulatoria)
      2. ExtSourceLayer  → índice ponderado aprendible de creditworthiness externa
      3. Dense(128→64)   → interacciones generales entre las 12 features

    Justificación de ExtSourceLayer:
      Las tres EXT_SOURCE miden lo mismo (creditworthiness) pero con distinta
      intensidad (visible en los deciles del EDA). La capa aprende los pesos
      relativos, haciendo el modelo AUDITABLE: podemos ver qué fuente considera
      más importante.
      Restricción wᵢ ≥ 0: garantiza la monotonía negativa con el impago
      (más score externo = menos riesgo), confirmada por los deciles del EDA.

    La Concatenate fusiona las tres ramas antes de la neurona de salida.
    """
    inputs = Input(shape=(n_features,), name='input')

    # Rama 1: DebtRatioLayer
    debt_col  = layers.Lambda(
        lambda t: t[:, debt_ratio_idx:debt_ratio_idx + 1],
        output_shape=(1,), name='extract_debt_ratio'
    )(inputs)
    debt_sat  = DebtRatioLayer(name='debt_saturation')(debt_col)

    # Rama 2: ExtSourceLayer
    ext_start = ext_source_idxs[0]
    ext_end   = ext_source_idxs[-1] + 1
    ext_cols  = layers.Lambda(
        lambda t: t[:, ext_start:ext_end],
        output_shape=(3,), name='extract_ext_sources'
    )(inputs)
    ext_index = ExtSourceLayer(name='ext_source_index')(ext_cols)

    # Rama 3: MLP principal
    x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
    x = layers.Dropout(0.30,                 name='dropout_1')(x)
    x = layers.Dense(64,  activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.20,                 name='dropout_2')(x)

    # Fusión
    combined = layers.Concatenate(name='concat')([x, debt_sat, ext_index])
    output   = layers.Dense(1, activation='sigmoid', name='output')(combined)
    return Model(inputs, output, name='M6_DualCustom_Dropout')
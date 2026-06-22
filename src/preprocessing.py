"""
preprocessing.py
────────────────
Pipeline completo de carga, limpieza, feature engineering,
split estratificado y transformaciones para Home Credit Default Risk.

Decisiones de diseño documentadas:
  - Eliminar XNA (4 filas) de CODE_GENDER: artefacto de calidad
  - log1p en AMT_*: colas asimétricas extremas
  - DEBT_RATIO calculado ANTES del log: significado financiero
  - Máscaras _MISSING: señal adicional + base para análisis de incertidumbre
  - Imputación mediana ajustada SOLO en train: evita data leakage
  - Split 60/20/20 estratificado por TARGET: mantiene ratio de impago 8.07%
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Columnas del dataset ───────────────────────────────────────────────────
COLS = [
    'TARGET', 'CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
    'AMT_ANNUITY', 'DAYS_BIRTH', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]

EXT_COLS   = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
MONEY_COLS = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']

FEATURE_COLS = (
    ['CODE_GENDER'] +
    MONEY_COLS +
    ['AGE_YEARS', 'DEBT_RATIO'] +
    EXT_COLS +
    [f'{c}_MISSING' for c in EXT_COLS]
)


def load_and_clean(file_path: str) -> pd.DataFrame:
    """
    Carga el CSV y aplica la limpieza básica:
      - Elimina filas con CODE_GENDER='XNA' (4 filas, tasa impago=0, artefacto)
      - Codifica género: M→0, F→1
      - Crea AGE_YEARS desde DAYS_BIRTH (negativo → positivo en años)
    """
    df = pd.read_csv(file_path, usecols=COLS)

    print(f"  Filas antes de limpiar XNA: {len(df)}")
    df = df[df['CODE_GENDER'] != 'XNA'].copy()
    print(f"  Filas después de limpiar XNA: {len(df)}")

    df['CODE_GENDER'] = df['CODE_GENDER'].map({'M': 0, 'F': 1})
    df['AGE_YEARS']   = np.abs(df['DAYS_BIRTH']) / 365
    df.drop(columns=['DAYS_BIRTH'], inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
      - DEBT_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL (antes de log, clip a 3.0)
        Justificación: el ratio en escala original tiene significado financiero;
        se clipea en 3.0 para eliminar outliers extremos sin perder información.
      - *_MISSING: máscaras binarias de ausencia en EXT_SOURCE
        Justificación: la ausencia del dato es señal predictiva en sí misma
        y es la base del análisis de incertidumbre (Parte 4).
    """
    df = df.copy()

    # Ratio de endeudamiento (cuota / ingresos)
    df['DEBT_RATIO'] = (
        df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median()) /
        df['AMT_INCOME_TOTAL']
    ).clip(upper=3.0)

    # Máscaras de ausencia
    for c in EXT_COLS:
        df[f'{c}_MISSING'] = df[c].isna().astype(np.float32)

    return df


def split_data(df: pd.DataFrame, val_size: float = 0.20,
               test_size: float = 0.20, seed: int = 42):
    """
    Split estratificado 60/20/20 arrastrando la variable sensible (género).

    Justificación del split estratificado:
      - TARGET tiene solo 8.07% de positivos → sin estratificar los conjuntos
        pueden tener proporciones muy distintas y las métricas serían inestables.
      - Arrastramos s (CODE_GENDER) para que la FAIR Loss y la auditoría
        de fairness tengan el grupo de cada muestra en cada conjunto.
    """
    X = df[FEATURE_COLS].copy()
    y = df['TARGET']
    s = df['CODE_GENDER']

    X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp = train_test_split(
        X, y, s,
        test_size=val_size + test_size,
        stratify=y,
        random_state=seed
    )
    rel = test_size / (val_size + test_size)
    X_val, X_te, y_val, y_te, s_val, s_te = train_test_split(
        X_tmp, y_tmp, s_tmp,
        test_size=rel,
        stratify=y_tmp,
        random_state=seed
    )

    print(f"\nShapes:")
    print(f"  Train:      X={X_tr.shape},  y={y_tr.shape}")
    print(f"  Val:        X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:       X={X_te.shape},  y={y_te.shape}")
    print(f"\nTasa de impago (TARGET=1):")
    for name, yy in [('Train', y_tr), ('Val', y_val), ('Test', y_te)]:
        print(f"  {name}: {yy.mean():.4f}")

    return (X_tr, y_tr, s_tr), (X_val, y_val, s_val), (X_te, y_te, s_te)


def apply_transformations(X_tr, X_val, X_te):
    """
    Transformaciones sin data leakage:
      1. Imputación por mediana (ajustada SOLO en train)
         Justificación: mediana es robusta a outliers; la mediana de val/test
         no debe conocerse en train.
      2. log1p en variables monetarias
         Justificación: colas largas extremas (AMT_INCOME_TOTAL max ~1.17e8);
         log1p comprime sin perder información y linealiza relaciones financieras.
      3. StandardScaler ajustado SOLO en train
         Justificación: redes con gradiente descendente convergen mejor con
         entradas en escala similar. No escalamos binarias (CODE_GENDER, *_MISSING)
         ni DEBT_RATIO (ya acotado en [0, 3]).

    Devuelve los tres splits transformados + el scaler y las medianas
    (necesarios para transformar nuevos datos en producción).
    """
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()

    # 1. Imputación
    medians_train = X_tr[EXT_COLS + ['AMT_ANNUITY']].median()
    for split in (X_tr, X_val, X_te):
        split[EXT_COLS + ['AMT_ANNUITY']] = (
            split[EXT_COLS + ['AMT_ANNUITY']].fillna(medians_train)
        )

    # 2. log1p en monetarias
    for split in (X_tr, X_val, X_te):
        split[MONEY_COLS] = np.log1p(split[MONEY_COLS])

    # 3. StandardScaler
    SCALE_COLS = MONEY_COLS + ['AGE_YEARS'] + EXT_COLS
    scaler = StandardScaler()
    scaler.fit(X_tr[SCALE_COLS])
    for split in (X_tr, X_val, X_te):
        split[SCALE_COLS] = scaler.transform(split[SCALE_COLS])

    print("\nTransformaciones aplicadas:")
    print(f"  Medias tras escalar (train, deben ser ~0): "
          f"{X_tr[SCALE_COLS].mean().round(3).values}")
    print(f"  Std tras escalar (train, deben ser ~1):    "
          f"{X_tr[SCALE_COLS].std().round(3).values}")

    return X_tr, X_val, X_te, scaler, medians_train


def full_pipeline(file_path: str, seed: int = 42):
    """
    Ejecuta el pipeline completo:
      load_and_clean → engineer_features → split → transformations

    Devuelve:
      (X_tr, y_tr, s_tr), (X_val, y_val, s_val), (X_te, y_te, s_te)
      en formato numpy float32 listo para Keras.
    """
    print("── 1. Carga y limpieza ──────────────────────────────────────")
    df = load_and_clean(file_path)

    print("\n── 2. Feature engineering ───────────────────────────────────")
    df = engineer_features(df)

    print("\n── 3. Split estratificado 60/20/20 ──────────────────────────")
    (X_tr, y_tr, s_tr), (X_val, y_val, s_val), (X_te, y_te, s_te) = \
        split_data(df, seed=seed)

    print("\n── 4. Transformaciones (sin data leakage) ───────────────────")
    X_tr, X_val, X_te, scaler, medians = apply_transformations(X_tr, X_val, X_te)

    # Convertir a numpy float32
    to_np = lambda arr: arr.values.astype(np.float32)
    return (
        (to_np(X_tr),  y_tr.values.astype(np.float32),  s_tr.values.astype(np.float32)),
        (to_np(X_val), y_val.values.astype(np.float32), s_val.values.astype(np.float32)),
        (to_np(X_te),  y_te.values.astype(np.float32),  s_te.values.astype(np.float32)),
        FEATURE_COLS, scaler, medians
    )
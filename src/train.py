"""
train.py
────────
Funciones de compilación y entrenamiento comunes a todos los modelos.

Decisiones de entrenamiento:
  · Optimizador: Adam(lr=1e-3) — estándar empírico, converge rápido
  · Loss: binary_crossentropy — pérdida natural para Bernoulli
  · Métrica: AUC-ROC — insensible al umbral, correcta con 8% de positivos
  · class_weight: compensa el desbalanceo 8%/92% sin alterar las predicciones
  · EarlyStopping(val_auc, patience): evita overfitting, restaura mejor epoch
  · ReduceLROnPlateau: reduce LR cuando val_auc se estanca → convergencia fina
"""

import numpy as np
import keras
from keras.metrics import AUC
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Calcula class_weights balanceados para compensar el desbalanceo 8%/92%.

    Fórmula 'balanced': n_samples / (n_classes * count_class)
      Clase 0 (pagó):   ~0.54
      Clase 1 (impago): ~6.19  → ratio ≈ 11.4×

    Justificación: el gradiente de cada ejemplo de impago vale ~11× más
    que el de un pago. Sin este peso, el modelo tendería a predecir siempre
    clase 0 y conseguir 92% de accuracy sin utilidad alguna.
    """
    weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f"Class weights: {class_weight}")
    print(f"Ratio clase 1/clase 0: {class_weight[1]/class_weight[0]:.1f}×")
    return class_weight


def compile_and_train(model, X_tr, y_tr, X_val, y_val,
                      class_weight: dict,
                      learning_rate: float = 1e-3,
                      batch_size: int = 512,
                      max_epochs: int = 100,
                      patience: int = 10,
                      use_reduce_lr: bool = False):
    """
    Compila el modelo y lo entrena con early stopping.

    Parámetros:
      model          : modelo Keras ya construido
      X_tr, y_tr     : datos de entrenamiento (numpy)
      X_val, y_val   : datos de validación (numpy)
      class_weight   : dict con pesos por clase {0: w0, 1: w1}
      learning_rate  : LR inicial para Adam
      batch_size     : tamaño de batch (512 es buen equilibrio velocidad/gradiente)
      max_epochs     : máximo de épocas (el early stopping para antes)
      patience       : épocas sin mejora en val_auc antes de parar
      use_reduce_lr  : si True, añade ReduceLROnPlateau
                       (útil cuando el modelo llega al techo)

    Devuelve:
      history : objeto History de Keras con las métricas por época
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[AUC(name='auc')]
    )

    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    if use_reduce_lr:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                mode='max',
                factor=0.3,       # LR × 0.3 cuando no mejora
                patience=5,       # espera 5 épocas antes de reducir
                min_lr=1e-6,
                verbose=1
            )
        )

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0
    )

    best_auc = max(history.history['val_auc'])
    n_epochs = len(history.history['loss'])
    print(f"  Entrenamiento completado: {n_epochs} épocas | "
          f"Mejor AUC val: {best_auc:.4f}")

    return history


def compile_and_train_fair(model, X_tr, y_tr_fair, X_val, y_val_fair,
                            sample_weight_tr: np.ndarray,
                            learning_rate: float = 1e-3,
                            batch_size: int = 512,
                            max_epochs: int = 100,
                            patience: int = 15):
    """
    Compilación y entrenamiento para modelos con FairBCELoss.

    Diferencias respecto a compile_and_train:
      · y_combined tiene forma (N, 2): [y_true, género]
      · Usa sample_weight en lugar de class_weight (incompatibles con y de forma (N,2))
      · Paciencia mayor (15) porque ReduceLROnPlateau puede retrasar la mejora
    """
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.3,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]

    history = model.fit(
        X_tr, y_tr_fair,
        validation_data=(X_val, y_val_fair),
        sample_weight=sample_weight_tr,
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    best_auc = max(history.history['val_auc'])
    n_epochs = len(history.history['loss'])
    print(f"  λ completado: {n_epochs} épocas | Mejor AUC val: {best_auc:.4f}")

    return history
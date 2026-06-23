"""
checkpoints.py
──────────────
Guardado y carga de modelos Keras y sus historiales de entrenamiento.
"""

import os
import json

CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)


def save_checkpoint(model, history, name: str):
    """Guarda pesos del modelo y el historial de entrenamiento."""
    path_w = os.path.join(CKPT_DIR, f'{name}.weights.h5')
    path_h = os.path.join(CKPT_DIR, f'{name}_history.json')

    model.save_weights(path_w)

    hist_serial = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    with open(path_h, 'w') as f:
        json.dump(hist_serial, f, indent=2)

    print(f"  💾 {path_w}")


def load_checkpoint(model, name: str):
    """Carga pesos en un modelo ya construido y devuelve el historial."""
    path_w = os.path.join(CKPT_DIR, f'{name}.weights.h5')
    path_h = os.path.join(CKPT_DIR, f'{name}_history.json')

    if not os.path.exists(path_w):
        print(f"  ⚠️  No existe '{name}' — entrena primero.")
        return None

    model.load_weights(path_w)

    history_dict = {}
    if os.path.exists(path_h):
        with open(path_h) as f:
            history_dict = json.load(f)

    class _FakeHistory:
        def __init__(self, d):
            self.history = d

    print(f"  ✅ Cargado: {path_w}")
    return _FakeHistory(history_dict)
import os
import kagglehub

# Fíjate en el prefijo 'src.' añadido a todos tus módulos
from src.preprocessing import full_pipeline
from src.models import build_model_m6
from src.train import get_class_weights, compile_and_train
from src.uncertainty import mc_dropout_predict, uncertainty_by_class

# 1. Descargar o apuntar al dataset
print("Descargando dataset...")
path = kagglehub.dataset_download("megancrenshaw/home-credit-default-risk")
csv_path = os.path.join(path, 'home-credit-default-risk', 'application_train.csv')

# 2. Pipeline de Preprocesamiento
print("\nIniciando Preprocesamiento...")
(X_tr, y_tr, s_tr), (X_val, y_val, s_val), (X_te, y_te, s_te), feature_cols, scaler, medians = full_pipeline(csv_path)

# Encontrar índices para M6
debt_idx = feature_cols.index('DEBT_RATIO')
ext_idxs = [feature_cols.index(c) for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
n_features = len(feature_cols)

# 3. Construir Modelo (Ejemplo con M6)
print("\nConstruyendo Modelo M6...")
model = build_model_m6(n_features=n_features, debt_ratio_idx=debt_idx, ext_source_idxs=ext_idxs)

# 4. Entrenar
print("\nCalculando pesos de clase y entrenando...")
class_weights = get_class_weights(y_tr)
history = compile_and_train(
    model, X_tr, y_tr, X_val, y_val, 
    class_weight=class_weights, 
    max_epochs=5, 
    use_reduce_lr=True
)

# 5. Incertidumbre (MC Dropout) en Test
print("\nEjecutando MC Dropout...")
# Reducido a 50 para una prueba rápida, puedes subirlo a 100 luego
mean_pred, var_pred, all_preds = mc_dropout_predict(model, X_te, n_passes=50) 
stats = uncertainty_by_class(var_pred, y_te)
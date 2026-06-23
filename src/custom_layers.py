"""
custom_layers.py
────────────────
Capas Keras customizadas con restricciones de dominio financiero.

  DebtRatioLayer  — saturación sigmoide sobre el ratio cuota/ingresos
  ExtSourceLayer  — combinación ponderada aprendible de EXT_SOURCE_1/2/3
"""

import keras


class DebtRatioLayer(keras.layers.Layer):
    """
    Saturación sigmoide aprendible sobre el Ratio de Endeudamiento.

    Fórmula:
        f(x) = sigmoid( α · (x − θ) )

    Parámetros aprendibles (2 en total):
      · α (slope):     pendiente de la curva.
                       Inicializado en 10 (sigmoide pronunciada).
                       Restricción: α ≥ 0 (la función siempre crece con el ratio).
      · θ (threshold): umbral de inflexión.
                       Inicializado en 0.35 (referencia regulatoria del 35%).
                       Sin restricción → el modelo lo ajusta a los datos reales.

    Justificación (NO predictiva, SÍ regulatoria):
      El ratio cuota/ingresos tiene un techo efectivo en regulación crediticia.
      Un ratio de 0.8 y uno de 1.5 son ambos "imposible pagar". Sin saturación,
      el modelo trataría estos valores como muy distintos.
      La sigmoide garantiza que a partir del umbral θ, el modelo no distingue
      entre ratios: todos llegan a ~1 (señal de alarma máxima).

    Salida: (batch, 1) — señal de riesgo de endeudamiento saturada.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.slope = self.add_weight(
            name='slope',
            shape=(1,),
            initializer=keras.initializers.Constant(10.0),
            trainable=True,
            constraint=keras.constraints.NonNeg()
        )
        self.threshold = self.add_weight(
            name='threshold',
            shape=(1,),
            initializer=keras.initializers.Constant(0.35),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return keras.activations.sigmoid(self.slope * (x - self.threshold))

    def get_config(self):
        return super().get_config()


class ExtSourceLayer(keras.layers.Layer):
    """
    Combinación ponderada aprendible de EXT_SOURCE_1, _2, _3.

    Fórmula:
        g(s) = sigmoid( w₁·s₁ + w₂·s₂ + w₃·s₃ + b )

    Parámetros aprendibles (4 en total):
      · w₁, w₂, w₃: pesos de cada fuente.
                     Inicializados en 1/3 (igual peso de partida).
                     Restricción: wᵢ ≥ 0 (la relación con el impago es
                     monotóna NEGATIVA: más score = menos riesgo;
                     la restricción de no-negatividad garantiza esta monotonía).
      · b: sesgo.
           Inicializado en 0. Sin restricción.

    Justificación:
      Las tres fuentes miden creditworthiness externa (misma semántica,
      distintas fuentes de datos). Los deciles del EDA muestran que las
      tres tienen relación monotóna decreciente con el impago, pero con
      distinta intensidad. En lugar de promediarlas (1/3, 1/3, 1/3),
      la capa aprende la importancia relativa de cada fuente.
      Esto hace el modelo AUDITABLE: los pesos aprendidos revelan qué
      fuente considera más relevante el modelo.

    Entrada: (batch, 3) — las tres EXT_SOURCE ya imputadas y escaladas.
    Salida:  (batch, 1) — índice de confianza externa consolidado.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name='source_weights',
            shape=(3,),
            initializer=keras.initializers.Constant(1 / 3),
            trainable=True,
            constraint=keras.constraints.NonNeg()
        )
        self.b = self.add_weight(
            name='bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, 3) → salida: (batch, 1)
        return keras.activations.sigmoid(
            keras.ops.sum(x * self.w, axis=-1, keepdims=True) + self.b
        )

    def get_config(self):
        return super().get_config()
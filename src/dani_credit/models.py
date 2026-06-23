"""Custom model builders for the Home Credit trustworthy neural network.

This module implements the model-level part of Block 5. It wires the custom
financial layers into a compiled Keras MLP while keeping the same controlled
training defaults used by the MVP:

    BCE loss, Adam, gradient clipping, AUC/PR-AUC/accuracy/precision/recall

Block 6 extends this module with a FAIR model builder. The FAIR builder reuses
the Block 5 probability graph instead of duplicating custom financial layers or
dense architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import tensorflow as tf

from .layers import (
    FairnessPenalty,
    FinancialRatioIndexResolver,
    FinancialRatioIndices,
    FinancialRatiosLayer,
    TrainableGammaLayer,
    custom_layer_objects,
)


class CustomModelError(ValueError):
    """Raised when a custom model cannot be built safely.

    Args:
        message: Human-readable explanation of the model-building failure.

    Returns:
        None.

    Raises:
        This exception is raised by custom model builders when configuration or
        input dimensions violate Block 5 assumptions.
    """


class FairModelError(ValueError):
    """Raised when a FAIR model cannot be built safely.

    Args:
        message: Human-readable explanation of the FAIR model-building failure.

    Returns:
        None.

    Raises:
        This exception is raised by Block 6 builders when fairness-specific
        configuration values are invalid.
    """


@dataclass(frozen=True)
class CustomMLPConfig:
    """Configuration for the Block 5 custom MLP.

    Args:
        hidden_units: Width of each hidden Dense layer.
        activation: Activation function used by hidden Dense layers.
        dropout: Dropout rate after each hidden Dense layer.
        learning_rate: Adam optimizer learning rate.
        gradient_clipnorm: Gradient norm clipping value for Adam.
        loss: Keras loss name used for the base custom model.
        ratio_eps: Denominator stabilizer used by `FinancialRatiosLayer`.
        ratio_clip_value: Upper clip value for financial ratios.
        gamma_min: Lower bound for trainable gamma.
        gamma_max: Upper bound for trainable gamma.
        theta_init: Initial unconstrained theta value.
        gamma_l2_reg: L2 regularization for theta in `TrainableGammaLayer`.
        gamma_epsilon: Stabilizer used before the power operation.

    Returns:
        Immutable configuration object.

    Raises:
        None.
    """

    # Match Block 4 so comparisons remain controlled.
    hidden_units: tuple[int, ...] = (128, 64)

    # ELU is the documented default for the tabular MVP network.
    activation: str = "elu"

    # Dropout remains part of the base architecture and later supports
    # MC-Dropout experiments if the team goes for extra uncertainty analysis.
    dropout: float = 0.2

    # Same starting learning rate as the controlled baseline.
    learning_rate: float = 1e-3

    # Gradient clipping is important because ratios can increase gradients.
    gradient_clipnorm: float = 1.0

    # The base custom model still optimizes binary cross-entropy.
    loss: str = "binary_crossentropy"

    # Epsilon is in monetary units because financial columns are not scaled.
    ratio_eps: float = 1.0

    # Saturates extreme financial ratios without deleting the signal.
    ratio_clip_value: float = 10.0

    # Gamma bounds mirror the mathematical explanation in Block 5.
    gamma_min: float = 0.1
    gamma_max: float = 1.5

    # Chosen so gamma starts close to 1, i.e. nearly neutral.
    theta_init: float = 0.588

    # Small regularization discourages unnecessary extreme gamma values.
    gamma_l2_reg: float = 1e-4

    # Stabilizer before pow for zero ratios.
    gamma_epsilon: float = 1e-6


@dataclass(frozen=True)
class CustomModelBuildResult:
    """Container returned after building a Block 5 model.

    Args:
        model: Compiled Keras model.
        ratio_indices: Original financial column indices used by the ratio
            layer.
        ratio_feature_indices: Positions of the four ratio columns after they
            are appended by `FinancialRatiosLayer`.
        output_feature_count_after_custom_block: Feature count after ratios and
            gamma transforms, before dense layers.

    Returns:
        Immutable build result with model and traceability metadata.

    Raises:
        None.
    """

    # The compiled Keras model ready for training.
    model: tf.keras.Model

    # The original financial indices are stored so experiments can audit them.
    ratio_indices: FinancialRatioIndices

    # These indices are the columns selected by TrainableGammaLayer.
    ratio_feature_indices: tuple[int, ...]

    # Expected dense input dimension after custom feature expansion.
    output_feature_count_after_custom_block: int


@dataclass(frozen=True)
class CustomProbabilityGraph:
    """Reusable probability graph built from the Block 5 custom architecture.

    Args:
        features_input: Keras input tensor named `features`.
        probability_output: Sigmoid probability tensor produced by the model
            backbone.
        ratio_indices: Original financial column indices used by the ratio
            layer.
        ratio_feature_indices: Positions of the four ratio columns after they
            are appended by `FinancialRatiosLayer`.
        output_feature_count_after_custom_block: Feature count after custom
            financial expansion and before dense layers.

    Returns:
        Immutable graph metadata used by base and future FAIR model builders.

    Raises:
        None.
    """

    # Keras symbolic input for processed features.
    features_input: Any

    # Keras symbolic output with P(TARGET=1 | X).
    probability_output: Any

    # Original financial indices used for ratios.
    ratio_indices: FinancialRatioIndices

    # Appended ratio indices selected by gamma.
    ratio_feature_indices: tuple[int, ...]

    # Dense input dimension after ratios and gamma transforms.
    output_feature_count_after_custom_block: int


@dataclass(frozen=True)
class FairModelConfig:
    """Configuration for the Block 6 FAIR wrapper.

    Args:
        lambda_fair: Non-negative fairness penalty strength.
        fairness_eps: Positive stabilizer used by `FairnessPenalty`.
        model_name_prefix: Prefix used to name generated Keras models.
        fairness_layer_name: Stable Keras layer name for the penalty layer.

    Returns:
        Immutable FAIR-model configuration.

    Raises:
        None.
    """

    # lambda=0 is the controlled base-final model in the same dual-input family.
    lambda_fair: float = 0.0

    # eps is specific to the differentiable training penalty.
    fairness_eps: float = 1e-8

    # Prefix keeps saved models and summaries easy to identify by lambda.
    model_name_prefix: str = "fair_custom_lambda"

    # Stable layer name makes model summaries and saved graphs predictable.
    fairness_layer_name: str = "fair_penalty"


@dataclass(frozen=True)
class FairModelBuildResult:
    """Container returned after building a Block 6 FAIR model.

    Args:
        model: Compiled dual-input Keras model.
        ratio_indices: Original financial column indices used by the shared
            Block 5 graph.
        ratio_feature_indices: Positions of ratio features appended by the
            shared graph.
        output_feature_count_after_custom_block: Feature count after custom
            financial expansion and before dense layers.
        lambda_fair: Fairness penalty strength used by the model.
        model_name: Keras model name.

    Returns:
        Immutable build result with model and traceability metadata.

    Raises:
        None.
    """

    # The compiled Keras model with inputs {"features", "sensitive"}.
    model: tf.keras.Model

    # Traceability metadata inherited from the shared custom probability graph.
    ratio_indices: FinancialRatioIndices
    ratio_feature_indices: tuple[int, ...]
    output_feature_count_after_custom_block: int

    # FAIR-specific metadata.
    lambda_fair: float
    model_name: str


class CustomMLPModelBuilder:
    """Build compiled MLPs with the Block 5 custom financial layers.

    Args:
        config: Optional `CustomMLPConfig`.
        index_resolver: Optional resolver for financial column names.

    Returns:
        Builder object able to create compiled Keras models.

    Raises:
        CustomModelError: If the configuration is invalid.
    """

    def __init__(
        self,
        config: CustomMLPConfig | None = None,
        index_resolver: FinancialRatioIndexResolver | None = None,
    ) -> None:
        """Initialize the custom MLP builder.

        Args:
            config: Optional custom model configuration.
            index_resolver: Optional financial index resolver.

        Returns:
            None.

        Raises:
            CustomModelError: If config values are invalid.
        """

        # Config injection makes the builder reusable in tuner and tests.
        self._config = config or CustomMLPConfig()

        # Resolver injection lets later blocks adapt the same builder if feature
        # names change after relational feature engineering.
        self._index_resolver = index_resolver or FinancialRatioIndexResolver()

        # Validate once at construction time so errors are caught early.
        self._validate_config()

    @property
    def config(self) -> CustomMLPConfig:
        """Return the custom model configuration.

        Args:
            None.

        Returns:
            CustomMLPConfig used by this builder.

        Raises:
            None.
        """

        # Dataclass is frozen, so returning it directly is safe.
        return self._config

    @property
    def index_resolver(self) -> FinancialRatioIndexResolver:
        """Return the financial index resolver.

        Args:
            None.

        Returns:
            Resolver used to map feature names to financial indices.

        Raises:
            None.
        """

        # The resolver has no mutating public methods, so exposing it is safe.
        return self._index_resolver

    def _validate_config(self) -> None:
        """Validate the model configuration.

        Args:
            None.

        Returns:
            None.

        Raises:
            CustomModelError: If any configuration value is invalid.
        """

        # A model without hidden layers would not match the documented MVP MLP.
        if not self._config.hidden_units:
            raise CustomModelError("hidden_units cannot be empty.")

        # Every Dense layer width must be positive.
        if any(units <= 0 for units in self._config.hidden_units):
            raise CustomModelError("hidden_units must contain positive values.")

        # Dropout must be in Keras' valid interval [0, 1).
        if not 0.0 <= self._config.dropout < 1.0:
            raise CustomModelError("dropout must be in [0, 1).")

        # Learning rate and gradient clipping must be positive for Adam.
        if self._config.learning_rate <= 0:
            raise CustomModelError("learning_rate must be positive.")

        if self._config.gradient_clipnorm <= 0:
            raise CustomModelError("gradient_clipnorm must be positive.")

        # Ratio and gamma settings mirror validations in the layer classes, but
        # checking here gives clearer model-level errors.
        if self._config.ratio_eps <= 0:
            raise CustomModelError("ratio_eps must be positive.")

        if self._config.ratio_clip_value <= 0:
            raise CustomModelError("ratio_clip_value must be positive.")

        if self._config.gamma_min <= 0:
            raise CustomModelError("gamma_min must be positive.")

        if self._config.gamma_max <= self._config.gamma_min:
            raise CustomModelError("gamma_max must be greater than gamma_min.")

        if self._config.gamma_l2_reg < 0:
            raise CustomModelError("gamma_l2_reg must be non-negative.")

        if self._config.gamma_epsilon <= 0:
            raise CustomModelError("gamma_epsilon must be positive.")

    def build_from_feature_names(
        self,
        feature_names: Sequence[str],
    ) -> CustomModelBuildResult:
        """Resolve financial indices and build the custom model.

        Args:
            feature_names: Ordered feature names emitted by Block 2.

        Returns:
            CustomModelBuildResult with compiled model and index metadata.

        Raises:
            CustomModelError: If `feature_names` is empty.
            CustomLayerError: Propagated if required financial names are absent.
        """

        # Convert to tuple so the length and order cannot change mid-build.
        names = tuple(feature_names)

        # Building a model without features is invalid.
        if not names:
            raise CustomModelError("feature_names cannot be empty.")

        # Resolve and validate the four financial indices.
        ratio_indices = self._index_resolver.resolve(names)

        # The model input dimension is exactly the number of processed columns.
        return self.build(input_dim=len(names), ratio_indices=ratio_indices)

    def build(
        self,
        input_dim: int,
        ratio_indices: FinancialRatioIndices,
    ) -> CustomModelBuildResult:
        """Build and compile the custom MLP.

        Args:
            input_dim: Number of processed input features.
            ratio_indices: Positions of the four financial amount columns.

        Returns:
            CustomModelBuildResult with compiled model and metadata.

        Raises:
            CustomModelError: If input dimension or indices are invalid.
        """

        # The Functional API input represents the processed numeric feature
        # matrix produced by Block 2.
        features_input = tf.keras.Input(shape=(input_dim,), name="features")

        # Build the shared probability graph. Block 6 will reuse the same method
        # and only add the sensitive input plus fairness penalty on top.
        graph = self.build_probability_graph(
            features_input=features_input,
            input_dim=input_dim,
            ratio_indices=ratio_indices,
        )

        # Name makes saved models and summaries easy to recognize.
        model = tf.keras.Model(
            inputs=graph.features_input,
            outputs=graph.probability_output,
            name="custom_financial_mlp",
        )

        # Compile through a shared method so FAIR builders can reuse identical
        # optimizer/loss/metrics settings.
        self.compile_model(model)

        return CustomModelBuildResult(
            model=model,
            ratio_indices=graph.ratio_indices,
            ratio_feature_indices=graph.ratio_feature_indices,
            output_feature_count_after_custom_block=(
                graph.output_feature_count_after_custom_block
            ),
        )

    def build_probability_graph(
        self,
        *,
        features_input: Any,
        input_dim: int,
        ratio_indices: FinancialRatioIndices,
    ) -> CustomProbabilityGraph:
        """Build the reusable custom architecture up to the probability output.

        Args:
            features_input: Keras input tensor for processed features.
            input_dim: Number of processed input features.
            ratio_indices: Positions of the four financial amount columns.

        Returns:
            CustomProbabilityGraph with tensors and traceability metadata.

        Raises:
            CustomModelError: If input dimension or indices are invalid.
        """

        self._validate_input_dim_and_indices(
            input_dim=input_dim,
            ratio_indices=ratio_indices,
        )

        # First custom layer: append interpretable ratios over monetary amounts.
        x = FinancialRatiosLayer(
            **ratio_indices.as_layer_kwargs(),
            eps=self._config.ratio_eps,
            clip_value=self._config.ratio_clip_value,
            name="financial_ratios",
        )(features_input)

        # FinancialRatiosLayer appends exactly four columns at positions
        # input_dim, input_dim+1, input_dim+2 and input_dim+3.
        ratio_feature_indices = tuple(range(input_dim, input_dim + 4))

        # Second custom layer: append trainable saturation transforms for the
        # newly-created ratio columns only.
        x = TrainableGammaLayer(
            selected_idx=ratio_feature_indices,
            gamma_min=self._config.gamma_min,
            gamma_max=self._config.gamma_max,
            theta_init=self._config.theta_init,
            l2_reg=self._config.gamma_l2_reg,
            epsilon=self._config.gamma_epsilon,
            name="ratio_gamma",
        )(x)

        # After ratios and gamma, the dense stack sees input_dim + 8 features.
        custom_block_output_dim = input_dim + 8

        # Batch normalization stabilizes the mixed-scale feature vector before
        # Dense layers. This is especially useful because financial amounts stay
        # in original scale for interpretability.
        x = tf.keras.layers.BatchNormalization(name="post_custom_batchnorm")(x)

        # Build the configurable dense stack.
        for layer_index, units in enumerate(self._config.hidden_units):
            # Dense layer learns interactions between raw processed features,
            # ratios and gamma-saturated ratios.
            x = tf.keras.layers.Dense(
                units,
                activation=self._config.activation,
                name=f"dense_{layer_index}",
            )(x)

            # Dropout regularizes the tabular neural network.
            x = tf.keras.layers.Dropout(
                self._config.dropout,
                name=f"dropout_{layer_index}",
            )(x)

        # Sigmoid returns P(TARGET=1 | X), i.e. probability of payment
        # difficulty.
        probability_output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="prob",
        )(x)

        return CustomProbabilityGraph(
            features_input=features_input,
            probability_output=probability_output,
            ratio_indices=ratio_indices,
            ratio_feature_indices=ratio_feature_indices,
            output_feature_count_after_custom_block=custom_block_output_dim,
        )

    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Compile a custom architecture with controlled MVP settings.

        Args:
            model: Uncompiled Keras model.

        Returns:
            The same model instance, compiled in place.

        Raises:
            None.
        """

        # Adam is kept consistent with Block 4, and clipnorm protects the
        # training loop when ratios or later fairness penalties create spikes.
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.learning_rate,
            clipnorm=self._config.gradient_clipnorm,
        )

        # Compile with the same metrics as the baseline so model comparison is
        # controlled and not polluted by different instrumentation.
        model.compile(
            optimizer=optimizer,
            loss=self._config.loss,
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        return model

    def _validate_input_dim_and_indices(
        self,
        *,
        input_dim: int,
        ratio_indices: FinancialRatioIndices,
    ) -> None:
        """Validate input dimension and financial indices.

        Args:
            input_dim: Number of processed input features.
            ratio_indices: Candidate financial indices.

        Returns:
            None.

        Raises:
            CustomModelError: If input dimension or indices are invalid.
        """

        # Input dimension must be strictly positive.
        if input_dim <= 0:
            raise CustomModelError("input_dim must be positive.")

        # Validate indices against the numeric dimension even when the caller
        # bypasses `build_from_feature_names`.
        if min(ratio_indices.as_tuple()) < 0:
            raise CustomModelError("ratio_indices must be non-negative.")

        if max(ratio_indices.as_tuple()) >= input_dim:
            raise CustomModelError("ratio_indices exceed input_dim.")


class FairCustomModelBuilder:
    """Build dual-input FAIR models without duplicating the custom backbone.

    Args:
        custom_builder: Optional `CustomMLPModelBuilder` used to build the
            shared probability graph and compile the model.
        fair_config: Optional `FairModelConfig` controlling lambda and penalty
            settings.

    Returns:
        Builder object able to create compiled FAIR Keras models.

    Raises:
        FairModelError: If fairness-specific configuration is invalid.
    """

    def __init__(
        self,
        custom_builder: CustomMLPModelBuilder | None = None,
        fair_config: FairModelConfig | None = None,
    ) -> None:
        """Initialize the FAIR model builder.

        Args:
            custom_builder: Optional base custom builder. If omitted, the
                default Block 5 builder is used.
            fair_config: Optional fairness configuration.

        Returns:
            None.

        Raises:
            FairModelError: If the FAIR configuration is invalid.
        """

        # The custom builder owns all predictive architecture decisions.
        self._custom_builder = custom_builder or CustomMLPModelBuilder()

        # FAIR config owns only lambda, eps and naming choices.
        self._fair_config = fair_config or FairModelConfig()

        # Validate once so all build methods can trust the config.
        self._validate_fair_config(self._fair_config)

    @property
    def custom_builder(self) -> CustomMLPModelBuilder:
        """Return the wrapped custom MLP builder.

        Args:
            None.

        Returns:
            CustomMLPModelBuilder used to build the shared probability graph.

        Raises:
            None.
        """

        return self._custom_builder

    @property
    def fair_config(self) -> FairModelConfig:
        """Return the fairness configuration.

        Args:
            None.

        Returns:
            FairModelConfig used by this builder.

        Raises:
            None.
        """

        return self._fair_config

    def build_from_feature_names(
        self,
        feature_names: Sequence[str],
        *,
        lambda_fair: float | None = None,
    ) -> FairModelBuildResult:
        """Resolve financial indices and build a compiled FAIR model.

        Args:
            feature_names: Ordered feature names emitted by Block 2.
            lambda_fair: Optional override for the configured lambda.

        Returns:
            FairModelBuildResult with compiled model and traceability metadata.

        Raises:
            FairModelError: If feature names are empty.
            CustomLayerError: Propagated if financial features are missing.
        """

        # Convert to tuple to freeze order while resolving indices.
        names = tuple(feature_names)

        # Empty features indicate an upstream preprocessing failure.
        if not names:
            raise FairModelError("feature_names cannot be empty.")

        # Reuse the exact resolver owned by the custom builder.
        ratio_indices = self._custom_builder.index_resolver.resolve(names)

        # Build with the resolved input dimension and financial indices.
        return self.build(
            input_dim=len(names),
            ratio_indices=ratio_indices,
            lambda_fair=lambda_fair,
        )

    def build(
        self,
        *,
        input_dim: int,
        ratio_indices: FinancialRatioIndices,
        lambda_fair: float | None = None,
    ) -> FairModelBuildResult:
        """Build and compile a dual-input FAIR model.

        Args:
            input_dim: Number of processed input features.
            ratio_indices: Positions of the four financial amount columns.
            lambda_fair: Optional override for the configured lambda.

        Returns:
            FairModelBuildResult with compiled model and metadata.

        Raises:
            FairModelError: If `lambda_fair` is invalid.
            CustomModelError: If input dimension or ratio indices are invalid.
        """

        # Resolve the lambda used for this concrete model.
        lambda_value = self._resolve_lambda(lambda_fair)

        # The features input is the same processed matrix used in Block 5.
        features_input = tf.keras.Input(shape=(input_dim,), name="features")

        # The sensitive input is separate and must never be concatenated to the
        # dense feature branch.
        sensitive_input = tf.keras.Input(shape=(1,), name="sensitive")

        # Build the shared probability graph from Block 5. This is the key
        # anti-duplication point of Block 6.
        graph = self._custom_builder.build_probability_graph(
            features_input=features_input,
            input_dim=input_dim,
            ratio_indices=ratio_indices,
        )

        # Add the FAIR regularizer as an identity layer on top of probability.
        fair_output = FairnessPenalty(
            lambda_fair=lambda_value,
            eps=self._fair_config.fairness_eps,
            name=self._fair_config.fairness_layer_name,
        )([graph.probability_output, sensitive_input])

        # Give each model a stable name that encodes lambda.
        model_name = self._model_name(lambda_value)

        # The model has two named inputs, but a single normal sigmoid output.
        model = tf.keras.Model(
            inputs={"features": graph.features_input, "sensitive": sensitive_input},
            outputs=fair_output,
            name=model_name,
        )

        # Compile through the shared builder so BCE/Adam/metrics stay identical
        # to the base custom architecture.
        self._custom_builder.compile_model(model)

        return FairModelBuildResult(
            model=model,
            ratio_indices=graph.ratio_indices,
            ratio_feature_indices=graph.ratio_feature_indices,
            output_feature_count_after_custom_block=(
                graph.output_feature_count_after_custom_block
            ),
            lambda_fair=lambda_value,
            model_name=model_name,
        )

    def _resolve_lambda(self, lambda_fair: float | None) -> float:
        """Resolve and validate the lambda value used by one model.

        Args:
            lambda_fair: Optional method-level override.

        Returns:
            Non-negative lambda value as float.

        Raises:
            FairModelError: If lambda is negative.
        """

        # Method-level lambda lets Block 7 sweep lambdas without rebuilding the
        # builder object itself.
        lambda_value = (
            self._fair_config.lambda_fair
            if lambda_fair is None
            else float(lambda_fair)
        )

        # Negative lambda would reward dependence instead of penalizing it.
        if lambda_value < 0.0:
            raise FairModelError("lambda_fair must be non-negative.")

        return float(lambda_value)

    def _model_name(self, lambda_fair: float) -> str:
        """Build a stable Keras model name from lambda.

        Args:
            lambda_fair: Lambda value used by the model.

        Returns:
            Model name safe for Keras summaries and saved artifacts.

        Raises:
            None.
        """

        # Replace decimal point because filenames and model names are easier to
        # read with underscores.
        return f"{self._fair_config.model_name_prefix}_{lambda_slug(lambda_fair)}"

    def _validate_fair_config(self, config: FairModelConfig) -> None:
        """Validate fairness-specific configuration.

        Args:
            config: FairModelConfig to validate.

        Returns:
            None.

        Raises:
            FairModelError: If the config is unsafe.
        """

        # Reuse lambda validation for the default config value.
        if config.lambda_fair < 0.0:
            raise FairModelError("lambda_fair must be non-negative.")

        # eps must be positive for the differentiable denominator.
        if config.fairness_eps <= 0.0:
            raise FairModelError("fairness_eps must be positive.")

        # Empty names make model summaries and saved artifacts confusing.
        if not config.model_name_prefix:
            raise FairModelError("model_name_prefix cannot be empty.")

        if not config.fairness_layer_name:
            raise FairModelError("fairness_layer_name cannot be empty.")


def lambda_slug(lambda_fair: float) -> str:
    """Convert a lambda value into a stable string slug.

    Args:
        lambda_fair: Lambda value to encode.

    Returns:
        String with decimal points replaced by underscores.

    Raises:
        None.
    """

    # Normalize through float so int-like values become `0_0`, `1_0`, etc.
    return str(float(lambda_fair)).replace(".", "_")


def build_fair_custom_model(
    *,
    builder: CustomMLPModelBuilder,
    ratio_indices: FinancialRatioIndices,
    input_dim: int,
    lambda_fair: float,
) -> tf.keras.Model:
    """Build a compiled dual-input FAIR model.

    Args:
        builder: CustomMLPModelBuilder that owns the predictive architecture.
        ratio_indices: Positions of the four financial amount columns.
        input_dim: Number of processed input features.
        lambda_fair: Non-negative fairness penalty strength.

    Returns:
        Compiled Keras model with inputs `features` and `sensitive`.

    Raises:
        FairModelError: If `lambda_fair` is invalid.
        CustomModelError: If input dimension or ratio indices are invalid.
    """

    # This functional wrapper keeps compatibility with the documentation while
    # still delegating all real work to the POO builder.
    fair_builder = FairCustomModelBuilder(
        custom_builder=builder,
        fair_config=FairModelConfig(lambda_fair=lambda_fair),
    )

    return fair_builder.build(
        input_dim=input_dim,
        ratio_indices=ratio_indices,
    ).model


def custom_model_objects() -> dict[str, type[tf.keras.layers.Layer]]:
    """Return custom objects required to load saved custom models.

    Args:
        None.

    Returns:
        Dictionary containing Block 5 custom layer classes.

    Raises:
        None.
    """

    # Model loading only needs layer classes because the saved Keras model
    # stores the graph itself, not the Python builder.
    return custom_layer_objects()


__all__ = [
    "build_fair_custom_model",
    "CustomMLPConfig",
    "CustomMLPModelBuilder",
    "CustomModelBuildResult",
    "CustomModelError",
    "CustomProbabilityGraph",
    "FairCustomModelBuilder",
    "FairModelBuildResult",
    "FairModelConfig",
    "FairModelError",
    "lambda_slug",
    "custom_model_objects",
]

"""Custom Keras layers for the Home Credit trustworthy model.

This module implements Block 5 of the project. The design follows the official
MVP decision called "Ruta C":

    original monetary amounts -> financial ratios -> trainable gamma -> model

The key point is that financial ratios are computed from monetary columns that
Block 2 deliberately leaves in original scale after train-only median
imputation. We do not compute ratios on robust-scaled or log-transformed
columns because that would destroy their financial interpretation.

Block 6 extends this module with `FairnessPenalty`, an identity layer that adds
the FAIR regularization term through `self.add_loss()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import tensorflow as tf


class CustomLayerError(ValueError):
    """Raised when a custom layer cannot be configured safely.

    Args:
        message: Human-readable explanation of the layer configuration error.

    Returns:
        None.

    Raises:
        This exception is raised by helper classes in this module when feature
        names, indices or layer hyperparameters violate Block 5 assumptions.
    """


@dataclass(frozen=True)
class FinancialRatioIndices:
    """Column positions required by `FinancialRatiosLayer`.

    Args:
        idx_credit: Position of `AMT_CREDIT` in the processed feature matrix.
        idx_annuity: Position of `AMT_ANNUITY` in the processed feature matrix.
        idx_income: Position of `AMT_INCOME_TOTAL` in the processed matrix.
        idx_goods: Position of `AMT_GOODS_PRICE` in the processed matrix.

    Returns:
        Immutable index object passed to custom model builders.

    Raises:
        None.
    """

    # Credit requested by the applicant.
    idx_credit: int

    # Periodic annuity/payment amount.
    idx_annuity: int

    # Applicant total income.
    idx_income: int

    # Price of the financed goods.
    idx_goods: int

    def as_layer_kwargs(self) -> dict[str, int]:
        """Return indices using the argument names expected by the layer.

        Args:
            None.

        Returns:
            Dictionary compatible with `FinancialRatiosLayer(**kwargs)`.

        Raises:
            None.
        """

        # Keeping this conversion here avoids duplicating argument-name mapping
        # in every caller that needs to instantiate the layer.
        return {
            "idx_credit": self.idx_credit,
            "idx_annuity": self.idx_annuity,
            "idx_income": self.idx_income,
            "idx_goods": self.idx_goods,
        }

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the four indices as a tuple.

        Args:
            None.

        Returns:
            Tuple ordered as credit, annuity, income and goods.

        Raises:
            None.
        """

        # A tuple is convenient for validation loops and tests.
        return (
            self.idx_credit,
            self.idx_annuity,
            self.idx_income,
            self.idx_goods,
        )


class FinancialRatioIndexResolver:
    """Resolve financial feature names into matrix indices.

    Args:
        role_to_feature_name: Optional mapping from financial roles to concrete
            feature names. Leave as None for the Home Credit MVP defaults.

    Returns:
        Resolver object able to validate and create `FinancialRatioIndices`.

    Raises:
        CustomLayerError: If the role mapping is incomplete.
    """

    # These are the exact feature names emitted by Block 2 before scaling and
    # encoding. They are intentionally kept as a single source of truth here.
    DEFAULT_ROLE_TO_FEATURE_NAME: Mapping[str, str] = {
        "credit": "AMT_CREDIT",
        "annuity": "AMT_ANNUITY",
        "income": "AMT_INCOME_TOTAL",
        "goods": "AMT_GOODS_PRICE",
    }

    def __init__(self, role_to_feature_name: Mapping[str, str] | None = None) -> None:
        """Initialize the resolver.

        Args:
            role_to_feature_name: Optional custom role mapping.

        Returns:
            None.

        Raises:
            CustomLayerError: If any required role is missing.
        """

        # Copy the mapping so external mutation cannot silently affect this
        # resolver after construction.
        self._role_to_feature_name = dict(
            role_to_feature_name or self.DEFAULT_ROLE_TO_FEATURE_NAME
        )

        # All four roles are mandatory because the ratios are fixed by the
        # mathematical definition used in the report.
        missing_roles = set(self.DEFAULT_ROLE_TO_FEATURE_NAME) - set(
            self._role_to_feature_name
        )

        # Fail during initialization instead of producing a partial model later.
        if missing_roles:
            raise CustomLayerError(
                "Missing financial role mappings: "
                + ", ".join(sorted(missing_roles))
            )

    @property
    def role_to_feature_name(self) -> Mapping[str, str]:
        """Return the role-to-feature-name mapping.

        Args:
            None.

        Returns:
            Mapping from semantic role to processed feature name.

        Raises:
            None.
        """

        # Return a defensive copy to keep the resolver immutable from outside.
        return dict(self._role_to_feature_name)

    def resolve(self, feature_names: Sequence[str]) -> FinancialRatioIndices:
        """Resolve financial feature names into numeric positions.

        Args:
            feature_names: Ordered feature names emitted by the Block 2
                preprocessor.

        Returns:
            FinancialRatioIndices with validated positions.

        Raises:
            CustomLayerError: If a required feature is absent or duplicated.
        """

        # Convert to tuple so indexing is deterministic and cheap in repeated
        # validation calls.
        names = tuple(feature_names)

        # Empty features indicate an upstream preprocessing failure.
        if not names:
            raise CustomLayerError("feature_names cannot be empty.")

        # Build a name -> index lookup. This assumes unique feature names; we
        # validate that explicitly just below.
        feature_to_idx = {name: index for index, name in enumerate(names)}

        # Duplicate names would make index lookup ambiguous.
        if len(feature_to_idx) != len(names):
            raise CustomLayerError("feature_names contains duplicated columns.")

        # Collect missing raw financial features before raising so the error is
        # useful to diagnose preprocessing changes.
        missing_features = [
            feature_name
            for feature_name in self._role_to_feature_name.values()
            if feature_name not in feature_to_idx
        ]

        # Without these four columns the custom ratio layer is mathematically
        # undefined, so we stop immediately.
        if missing_features:
            raise CustomLayerError(
                "Missing financial features required by Block 5: "
                + ", ".join(missing_features)
            )

        # Map each semantic role to the exact integer position expected by the
        # TensorFlow layer.
        indices = FinancialRatioIndices(
            idx_credit=feature_to_idx[self._role_to_feature_name["credit"]],
            idx_annuity=feature_to_idx[self._role_to_feature_name["annuity"]],
            idx_income=feature_to_idx[self._role_to_feature_name["income"]],
            idx_goods=feature_to_idx[self._role_to_feature_name["goods"]],
        )

        # Validate once here so the caller can trust the returned object.
        self.validate_indices(feature_names=names, indices=indices)

        return indices

    def validate_indices(
        self,
        feature_names: Sequence[str],
        indices: FinancialRatioIndices,
    ) -> None:
        """Validate that indices point to the expected feature names.

        Args:
            feature_names: Ordered feature names emitted by Block 2.
            indices: Candidate financial indices.

        Returns:
            None.

        Raises:
            CustomLayerError: If an index is out of bounds, duplicated or points
                to the wrong feature.
        """

        # Convert to tuple so bounds checks and name lookups are stable.
        names = tuple(feature_names)

        # All indices must be different; otherwise a ratio could divide one
        # financial concept by itself without anyone noticing.
        if len(set(indices.as_tuple())) != 4:
            raise CustomLayerError("Financial ratio indices must be distinct.")

        # Pair each expected role with the resolved index.
        expected_pairs = {
            "credit": indices.idx_credit,
            "annuity": indices.idx_annuity,
            "income": indices.idx_income,
            "goods": indices.idx_goods,
        }

        # Validate every index against its expected feature name.
        for role, index in expected_pairs.items():
            # Negative indices are legal in Python but dangerous here because
            # they would silently point to columns from the end of the matrix.
            if index < 0:
                raise CustomLayerError(f"Index for role '{role}' is negative.")

            # Out-of-range indices would break the TensorFlow slice at runtime.
            if index >= len(names):
                raise CustomLayerError(
                    f"Index for role '{role}' is outside the feature matrix."
                )

            # This is the most important safety check: it guarantees that the
            # mathematical formula uses the intended financial variables.
            expected_name = self._role_to_feature_name[role]
            actual_name = names[index]

            if actual_name != expected_name:
                raise CustomLayerError(
                    f"Index for role '{role}' points to '{actual_name}', "
                    f"but expected '{expected_name}'."
                )


@tf.keras.utils.register_keras_serializable(package="HomeCredit")
class FinancialRatiosLayer(tf.keras.layers.Layer):
    """Append stable, interpretable financial ratios to the feature matrix.

    Args:
        idx_credit: Position of `AMT_CREDIT`.
        idx_annuity: Position of `AMT_ANNUITY`.
        idx_income: Position of `AMT_INCOME_TOTAL`.
        idx_goods: Position of `AMT_GOODS_PRICE`.
        eps: Positive value added to denominators to avoid division by zero.
        clip_value: Upper bound used to saturate extreme ratios.
        **kwargs: Standard Keras layer keyword arguments.

    Returns:
        A Keras layer. Calling it returns the original input concatenated with
        four ratio columns.

    Raises:
        CustomLayerError: If indices or numeric parameters are invalid.
    """

    def __init__(
        self,
        idx_credit: int,
        idx_annuity: int,
        idx_income: int,
        idx_goods: int,
        eps: float = 1.0,
        clip_value: float = 10.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the financial ratio layer.

        Args:
            idx_credit: Position of `AMT_CREDIT`.
            idx_annuity: Position of `AMT_ANNUITY`.
            idx_income: Position of `AMT_INCOME_TOTAL`.
            idx_goods: Position of `AMT_GOODS_PRICE`.
            eps: Positive denominator stabilizer.
            clip_value: Positive ratio clipping value.
            **kwargs: Standard Keras layer keyword arguments.

        Returns:
            None.

        Raises:
            CustomLayerError: If the layer configuration is unsafe.
        """

        # Initialize the Keras base class first so name/dtype/trainable
        # machinery is available.
        super().__init__(**kwargs)

        # Store indices as ints because they will be used for TensorFlow slices.
        self.idx_credit = int(idx_credit)
        self.idx_annuity = int(idx_annuity)
        self.idx_income = int(idx_income)
        self.idx_goods = int(idx_goods)

        # `eps` and `clip_value` are floats to avoid integer division surprises.
        self.eps = float(eps)
        self.clip_value = float(clip_value)

        # Validate values that can be checked before the input shape is known.
        self._validate_static_config()

    def _validate_static_config(self) -> None:
        """Validate layer settings that do not depend on input shape.

        Args:
            None.

        Returns:
            None.

        Raises:
            CustomLayerError: If indices, eps or clip value are invalid.
        """

        # Negative Python indices are intentionally disallowed because they can
        # silently point to unintended columns.
        if min(self._indices_tuple()) < 0:
            raise CustomLayerError("Financial ratio indices must be non-negative.")

        # All four indices must be distinct for the ratios to mean what their
        # labels say they mean.
        if len(set(self._indices_tuple())) != 4:
            raise CustomLayerError("Financial ratio indices must be distinct.")

        # A non-positive epsilon would not protect against zero denominators.
        if self.eps <= 0:
            raise CustomLayerError("eps must be positive.")

        # A non-positive clip value would erase all ratio information.
        if self.clip_value <= 0:
            raise CustomLayerError("clip_value must be positive.")

    def _indices_tuple(self) -> tuple[int, int, int, int]:
        """Return all configured indices.

        Args:
            None.

        Returns:
            Tuple with credit, annuity, income and goods indices.

        Raises:
            None.
        """

        # Centralizing this avoids small ordering mistakes in validation.
        return (
            self.idx_credit,
            self.idx_annuity,
            self.idx_income,
            self.idx_goods,
        )

    def build(self, input_shape: tf.TensorShape | tuple[Any, ...]) -> None:
        """Validate the layer against the incoming tensor shape.

        Args:
            input_shape: Shape of the incoming Keras tensor.

        Returns:
            None.

        Raises:
            CustomLayerError: If any configured index is out of bounds.
        """

        # TensorShape normalizes tuples, lists and symbolic Keras shapes.
        shape = tf.TensorShape(input_shape)

        # The last dimension must be known because this layer slices by column.
        input_dim = shape[-1]

        if input_dim is None:
            raise CustomLayerError("FinancialRatiosLayer requires known input_dim.")

        # Convert TensorShape dimension to int for normal Python comparisons.
        input_dim_int = int(input_dim)

        # Any index beyond the last column indicates a mismatch between feature
        # names and model input shape.
        if max(self._indices_tuple()) >= input_dim_int:
            raise CustomLayerError(
                "Financial ratio index outside input dimension "
                f"({input_dim_int})."
            )

        # This layer has no trainable weights, but calling super().build marks
        # it as built inside Keras.
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Append four clipped financial ratios to `inputs`.

        Args:
            inputs: Tensor with shape `(batch, n_features)`.

        Returns:
            Tensor with shape `(batch, n_features + 4)`.

        Raises:
            None during graph execution; invalid configuration is checked in
            `__init__` and `build`.
        """

        # Keras may provide float64 arrays from numpy. Casting to the layer's
        # compute dtype keeps operations consistent with the model.
        x = tf.cast(inputs, self.compute_dtype)

        # Slice each financial variable as a 2-D column tensor. Keeping
        # `k:k+1` preserves shape `(batch, 1)`, which makes concatenation clear.
        credit = x[:, self.idx_credit : self.idx_credit + 1]
        annuity = x[:, self.idx_annuity : self.idx_annuity + 1]
        income = x[:, self.idx_income : self.idx_income + 1]
        goods = x[:, self.idx_goods : self.idx_goods + 1]

        # The monetary columns should be non-negative. The max operation is a
        # final defensive guard against dirty or synthetic test data.
        credit_safe = tf.maximum(credit, 0.0)
        annuity_safe = tf.maximum(annuity, 0.0)
        income_safe = tf.maximum(income, 0.0)
        goods_safe = tf.maximum(goods, 0.0)

        # Compute the four ratios required by the Block 5 mathematical
        # explanation. Epsilon is added only to denominators.
        ratios = tf.concat(
            [
                credit_safe / (income_safe + self.eps),
                annuity_safe / (income_safe + self.eps),
                credit_safe / (goods_safe + self.eps),
                annuity_safe / (credit_safe + self.eps),
            ],
            axis=1,
        )

        # Clip ratios to keep extreme cases informative but numerically bounded.
        ratios = tf.clip_by_value(ratios, 0.0, self.clip_value)

        # The layer appends features instead of replacing anything so later
        # components can still use the original processed matrix.
        return tf.concat([x, ratios], axis=1)

    def compute_output_shape(
        self,
        input_shape: tf.TensorShape | tuple[Any, ...],
    ) -> tf.TensorShape:
        """Return the output shape after appending four ratios.

        Args:
            input_shape: Shape of the incoming tensor.

        Returns:
            TensorShape with last dimension increased by four.

        Raises:
            CustomLayerError: If the input feature dimension is unknown.
        """

        # Normalize the shape so the method works in eager and graph contexts.
        shape = tf.TensorShape(input_shape).as_list()

        # The last dimension is required to compute the new feature count.
        if shape[-1] is None:
            raise CustomLayerError("Cannot infer output shape without input_dim.")

        # Append exactly four ratio features.
        shape[-1] += 4

        return tf.TensorShape(shape)

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable layer configuration.

        Args:
            None.

        Returns:
            Dictionary used by Keras when saving/loading the model.

        Raises:
            None.
        """

        # Start from the Keras base config so name, dtype and trainable are
        # preserved during serialization.
        config = super().get_config()

        # Add every constructor argument needed to recreate this exact layer.
        config.update(
            {
                "idx_credit": self.idx_credit,
                "idx_annuity": self.idx_annuity,
                "idx_income": self.idx_income,
                "idx_goods": self.idx_goods,
                "eps": self.eps,
                "clip_value": self.clip_value,
            }
        )

        return config


@tf.keras.utils.register_keras_serializable(package="HomeCredit")
class TrainableGammaLayer(tf.keras.layers.Layer):
    """Append trainable power transformations for selected non-negative columns.

    Args:
        selected_idx: Column indices to transform with `x ** gamma`.
        gamma_min: Lower bound for each learned gamma.
        gamma_max: Upper bound for each learned gamma.
        theta_init: Initial value for the unconstrained theta parameters.
        l2_reg: L2 regularization applied to theta.
        epsilon: Small positive value added before the power operation.
        **kwargs: Standard Keras layer keyword arguments.

    Returns:
        A Keras layer. Calling it returns the original input concatenated with
        transformed selected columns.

    Raises:
        CustomLayerError: If the configuration is invalid.
    """

    def __init__(
        self,
        selected_idx: Sequence[int],
        gamma_min: float = 0.1,
        gamma_max: float = 1.5,
        theta_init: float = 0.588,
        l2_reg: float = 1e-4,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainable gamma layer.

        Args:
            selected_idx: Column indices to transform.
            gamma_min: Minimum gamma value after sigmoid parametrization.
            gamma_max: Maximum gamma value after sigmoid parametrization.
            theta_init: Initial unconstrained parameter value.
            l2_reg: L2 regularization strength for theta.
            epsilon: Positive stabilizer before `pow`.
            **kwargs: Standard Keras layer keyword arguments.

        Returns:
            None.

        Raises:
            CustomLayerError: If selected indices or numeric parameters are
            invalid.
        """

        # Initialize Keras internals first.
        super().__init__(**kwargs)

        # Store selected indices as a tuple for immutability and serialization.
        self.selected_idx = tuple(int(index) for index in selected_idx)

        # Store numeric hyperparameters as floats because TensorFlow operations
        # below expect floating point values.
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.theta_init = float(theta_init)
        self.l2_reg = float(l2_reg)
        self.epsilon = float(epsilon)

        # The trainable variable is created in build because its shape depends
        # on the selected index count.
        self.theta: tf.Variable | None = None

        # Validate values that can be checked immediately.
        self._validate_static_config()

    def _validate_static_config(self) -> None:
        """Validate configuration independent of input shape.

        Args:
            None.

        Returns:
            None.

        Raises:
            CustomLayerError: If the configuration is unsafe.
        """

        # At least one column must be transformed; otherwise the layer is a
        # confusing no-op that still adds complexity.
        if not self.selected_idx:
            raise CustomLayerError("selected_idx cannot be empty.")

        # Negative indices are forbidden for the same reason as in the ratio
        # layer: they silently refer to columns from the end.
        if min(self.selected_idx) < 0:
            raise CustomLayerError("selected_idx values must be non-negative.")

        # Duplicates would append the same transformed column twice.
        if len(set(self.selected_idx)) != len(self.selected_idx):
            raise CustomLayerError("selected_idx values must be distinct.")

        # Gamma range must be ordered and strictly positive for power transforms.
        if self.gamma_min <= 0 or self.gamma_max <= self.gamma_min:
            raise CustomLayerError("gamma range must satisfy 0 < min < max.")

        # Negative regularization would make no mathematical sense.
        if self.l2_reg < 0:
            raise CustomLayerError("l2_reg must be non-negative.")

        # Epsilon must be positive so zero-valued ratios remain safe for pow.
        if self.epsilon <= 0:
            raise CustomLayerError("epsilon must be positive.")

    def build(self, input_shape: tf.TensorShape | tuple[Any, ...]) -> None:
        """Create the trainable theta parameters.

        Args:
            input_shape: Shape of the incoming Keras tensor.

        Returns:
            None.

        Raises:
            CustomLayerError: If a selected index is out of bounds.
        """

        # Normalize input shape for robust bounds checks.
        shape = tf.TensorShape(input_shape)

        # The last dimension must be known because `tf.gather` selects columns.
        input_dim = shape[-1]

        if input_dim is None:
            raise CustomLayerError("TrainableGammaLayer requires known input_dim.")

        # Convert TensorShape dimension to int for normal comparisons.
        input_dim_int = int(input_dim)

        # Selected indices must all exist in the input matrix.
        if max(self.selected_idx) >= input_dim_int:
            raise CustomLayerError(
                "selected_idx contains an index outside input dimension "
                f"({input_dim_int})."
            )

        # L2 is optional; Keras expects None when no regularizer is desired.
        regularizer = (
            tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        )

        # Theta is unconstrained. Gamma is bounded later through the sigmoid
        # parametrization gamma = min + (max-min)*sigmoid(theta).
        self.theta = self.add_weight(
            name="theta",
            shape=(len(self.selected_idx),),
            initializer=tf.keras.initializers.Constant(self.theta_init),
            regularizer=regularizer,
            trainable=True,
        )

        # Mark layer as built after weights are created.
        super().build(input_shape)

    def current_gamma(self) -> tf.Tensor:
        """Return the current bounded gamma values.

        Args:
            None.

        Returns:
            Tensor with one gamma per selected column.

        Raises:
            CustomLayerError: If the layer has not been built yet.
        """

        # Accessing gamma before build would hide a lifecycle error in tests.
        if self.theta is None:
            raise CustomLayerError("TrainableGammaLayer must be built first.")

        # Sigmoid maps theta to (0, 1), then we stretch it to the configured
        # gamma interval. This guarantees every gamma remains in range.
        return self.gamma_min + (self.gamma_max - self.gamma_min) * tf.sigmoid(
            self.theta
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Append `x ** gamma` transforms for selected columns.

        Args:
            inputs: Tensor with shape `(batch, n_features)`.

        Returns:
            Tensor with shape `(batch, n_features + len(selected_idx))`.

        Raises:
            None during graph execution; configuration is validated earlier.
        """

        # Keep dtype consistent with the rest of the model.
        x = tf.cast(inputs, self.compute_dtype)

        # Gather selected columns as a dense `(batch, n_selected)` tensor.
        selected = tf.gather(x, self.selected_idx, axis=1)

        # The Block 5 route applies gamma to clipped ratios, so selected values
        # should already be non-negative. The max is a defensive guard.
        selected_non_negative = tf.maximum(selected, 0.0)

        # Get one bounded gamma per selected column.
        gamma = self.current_gamma()

        # Add epsilon before pow so zero ratios have a defined gradient path.
        transformed = tf.pow(selected_non_negative + self.epsilon, gamma)

        # Append transformed features instead of replacing the original ratios.
        # This lets the dense network use both raw ratio and saturated ratio.
        return tf.concat([x, transformed], axis=1)

    def compute_output_shape(
        self,
        input_shape: tf.TensorShape | tuple[Any, ...],
    ) -> tf.TensorShape:
        """Return the output shape after appending gamma features.

        Args:
            input_shape: Shape of the incoming tensor.

        Returns:
            TensorShape with last dimension increased by `len(selected_idx)`.

        Raises:
            CustomLayerError: If the input feature dimension is unknown.
        """

        # Normalize to a mutable list.
        shape = tf.TensorShape(input_shape).as_list()

        # We need a known feature dimension to add selected feature count.
        if shape[-1] is None:
            raise CustomLayerError("Cannot infer output shape without input_dim.")

        # Append one transformed feature for each selected column.
        shape[-1] += len(self.selected_idx)

        return tf.TensorShape(shape)

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable layer configuration.

        Args:
            None.

        Returns:
            Dictionary used by Keras when saving/loading the model.

        Raises:
            None.
        """

        # Preserve Keras-managed settings from the base class.
        config = super().get_config()

        # Add all constructor parameters needed to recreate the layer.
        config.update(
            {
                "selected_idx": list(self.selected_idx),
                "gamma_min": self.gamma_min,
                "gamma_max": self.gamma_max,
                "theta_init": self.theta_init,
                "l2_reg": self.l2_reg,
                "epsilon": self.epsilon,
            }
        )

        return config


@tf.keras.utils.register_keras_serializable(package="HomeCredit")
class FairnessPenalty(tf.keras.layers.Layer):
    """Identity layer that adds a differentiable FAIR penalty to model loss.

    Args:
        lambda_fair: Non-negative weight multiplying the squared batch Pearson
            correlation between predictions and sensitive values.
        eps: Positive stabilizer added inside the square root denominator.
        **kwargs: Standard Keras layer keyword arguments.

    Returns:
        A Keras layer. Calling it returns `y_pred` unchanged while adding
        `lambda_fair * rho^2` to the model loss when `lambda_fair > 0`.

    Raises:
        CustomLayerError: If `lambda_fair` is negative or `eps` is not positive.
    """

    def __init__(
        self,
        lambda_fair: float = 0.0,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        """Initialize the FAIR penalty layer.

        Args:
            lambda_fair: Non-negative fairness regularization strength.
            eps: Positive numerical stabilizer for batch correlation.
            **kwargs: Standard Keras layer keyword arguments.

        Returns:
            None.

        Raises:
            CustomLayerError: If the configuration is unsafe.
        """

        # Initialize Keras layer internals first.
        super().__init__(**kwargs)

        # Store lambda as a plain float because it controls a Python-level
        # decision and is serialized in get_config.
        self.lambda_fair = float(lambda_fair)

        # eps stays as a float; TensorFlow will cast it during graph execution.
        self.eps = float(eps)

        # Validate immediately so invalid FAIR settings fail before training.
        self._validate_static_config()

    def _validate_static_config(self) -> None:
        """Validate FAIR penalty configuration.

        Args:
            None.

        Returns:
            None.

        Raises:
            CustomLayerError: If lambda or epsilon are invalid.
        """

        # Negative lambda would reward correlation instead of penalizing it.
        if self.lambda_fair < 0.0:
            raise CustomLayerError("lambda_fair must be non-negative.")

        # eps must be positive to prevent sqrt(0) while keeping gradients finite.
        if self.eps <= 0.0:
            raise CustomLayerError("eps must be positive.")

    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        """Return predictions unchanged while adding the fairness loss.

        Args:
            inputs: Sequence `[y_pred, sensitive]`, both aligned by batch row.

        Returns:
            `y_pred` tensor unchanged, preserving the model's normal sigmoid
            probability output.

        Raises:
            CustomLayerError: If the layer does not receive exactly two inputs.
        """

        # The layer is defined over exactly two tensors. Failing here catches
        # wiring mistakes early when a future model is built.
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise CustomLayerError("FairnessPenalty expects [y_pred, sensitive].")

        # Unpack with explicit names because the formula is easier to audit.
        y_pred, sensitive = inputs

        # Cast both tensors to the layer compute dtype so mixed numpy/tensor
        # inputs behave consistently.
        y_pred = tf.cast(y_pred, self.compute_dtype)
        sensitive = tf.cast(sensitive, self.compute_dtype)

        # Flatten to one column per row. This makes the layer robust to `(B,)`
        # or `(B, 1)` tensors while keeping batch alignment.
        y_pred = tf.reshape(y_pred, (-1, 1))
        sensitive = tf.reshape(sensitive, (-1, 1))

        # Compute differentiable batch Pearson correlation.
        rho = self._batch_pearson(y_pred=y_pred, sensitive=sensitive)

        # lambda=0 is the base final model. In that case we skip add_loss so
        # the model has exactly the normal BCE and regularization losses.
        if self.lambda_fair > 0.0:
            self.add_loss(self.lambda_fair * tf.square(rho))

        # Return the prediction unchanged. This keeps metrics and BCE operating
        # on the standard sigmoid output.
        return y_pred

    def _batch_pearson(self, y_pred: tf.Tensor, sensitive: tf.Tensor) -> tf.Tensor:
        """Compute differentiable Pearson correlation inside one batch.

        Args:
            y_pred: Predicted probabilities with shape `(batch, 1)`.
            sensitive: Sensitive values with shape `(batch, 1)`.

        Returns:
            Scalar Tensor containing batch Pearson correlation.

        Raises:
            None.
        """

        # Centering removes the batch mean from both variables.
        y_centered = y_pred - tf.reduce_mean(y_pred)
        s_centered = sensitive - tf.reduce_mean(sensitive)

        # Numerator is the empirical covariance term.
        numerator = tf.reduce_mean(y_centered * s_centered)

        # Denominator multiplies standard deviations. eps is added inside sqrt
        # for training stability, as documented in Block 6.
        denominator = tf.sqrt(
            tf.reduce_mean(tf.square(y_centered))
            * tf.reduce_mean(tf.square(s_centered))
            + self.eps
        )

        return numerator / denominator

    def compute_output_shape(
        self,
        input_shape: Sequence[tf.TensorShape | tuple[Any, ...]],
    ) -> tf.TensorShape:
        """Return the output shape, equal to the prediction input shape.

        Args:
            input_shape: Shape sequence for `[y_pred, sensitive]`.

        Returns:
            TensorShape of `y_pred`.

        Raises:
            CustomLayerError: If shape information does not contain two inputs.
        """

        # Keep the shape contract explicit for model summaries and serialization
        # utilities.
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise CustomLayerError("FairnessPenalty expects two input shapes.")

        return tf.TensorShape(input_shape[0])

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable layer configuration.

        Args:
            None.

        Returns:
            Dictionary used by Keras when saving/loading the model.

        Raises:
            None.
        """

        # Preserve Keras-managed fields such as name, dtype and trainable.
        config = super().get_config()

        # Add constructor arguments needed to recreate the same FAIR penalty.
        config.update(
            {
                "lambda_fair": self.lambda_fair,
                "eps": self.eps,
            }
        )

        return config


def custom_layer_objects() -> dict[str, type[tf.keras.layers.Layer]]:
    """Return custom objects required to load saved custom models.

    Args:
        None.

    Returns:
        Dictionary mapping Keras serialization names to custom layer classes.

    Raises:
        None.
    """

    # Keras can often load registered serializable classes automatically, but an
    # explicit dictionary is clearer for notebooks and defensive loading code.
    return {
        "FairnessPenalty": FairnessPenalty,
        "FinancialRatiosLayer": FinancialRatiosLayer,
        "TrainableGammaLayer": TrainableGammaLayer,
    }


__all__ = [
    "CustomLayerError",
    "FinancialRatioIndexResolver",
    "FinancialRatioIndices",
    "FairnessPenalty",
    "FinancialRatiosLayer",
    "TrainableGammaLayer",
    "custom_layer_objects",
]

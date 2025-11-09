"""
Model definitions including CNN architecture and custom metrics.
"""
from typing import Dict, Tuple, Optional
import tensorflow as tf


class RoundingMAE(tf.keras.metrics.Metric):
    """Custom metric that rounds predictions before computing MAE."""
    
    def __init__(self, name: str = "mae_rounded", **kwargs):
        super().__init__(name=name, **kwargs)
        self._mae = tf.keras.metrics.MeanAbsoluteError()

    def update_state(self, y_true, y_pred, sample_weight: Optional[tf.Tensor] = None):
        y_pred_round = tf.round(tf.cast(y_pred, tf.float32))
        y_true = tf.cast(y_true, tf.float32)
        return self._mae.update_state(y_true, y_pred_round, sample_weight)

    def result(self):
        return self._mae.result()

    def reset_state(self):
        self._mae.reset_state()


def _get_initializer(name: str, gaussian_stddev: float):
    """Get weight initializer based on name."""
    n = (name or "").lower()
    if n == "xavier" or n == "glorot":
        return tf.keras.initializers.GlorotUniform()
    if n == "gaussian":
        return tf.keras.initializers.RandomNormal(stddev=float(gaussian_stddev))
    return tf.keras.initializers.GlorotUniform()


def build_custom_cnn(input_shape: Tuple[int, int, int], cfg: Dict):
    """
    Build a custom CNN for regression.
    
    Args:
        input_shape: (height, width, channels) of input images
        cfg: Model configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    initializer = _get_initializer(cfg.get("initializer", "xavier"), cfg.get("gaussian_stddev", 0.02))
    use_bn = bool(cfg.get("use_batch_norm", True))
    l2_w = float(cfg.get("l2_weight", 0.0))
    dropout_rate = float(cfg.get("dropout_rate", 0.0))

    arch = cfg.get("architecture", {})
    activation = arch.get("activation", "relu")
    conv_blocks = arch.get("conv_blocks", [])
    dense_units = arch.get("dense_units", [128])

    reg = tf.keras.regularizers.l2(l2_w) if l2_w > 0 else None

    x_in = tf.keras.Input(shape=input_shape)
    x = x_in

    # Convolutional blocks
    for block in conv_blocks:
        filters = int(block.get("filters", 32))
        kernel_size = int(block.get("kernel_size", 3))
        pool = int(block.get("pool", 2))

        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=reg,
            use_bias=not use_bn,
        )(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=pool)(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Flatten()(x)

    # Dense layers
    for units in dense_units:
        x = tf.keras.layers.Dense(
            int(units), kernel_initializer=initializer, kernel_regularizer=reg, use_bias=True
        )(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Regressor head: single linear output
    out = tf.keras.layers.Dense(1, kernel_initializer=initializer, kernel_regularizer=reg)(x)

    return tf.keras.Model(inputs=x_in, outputs=out, name="custom_cnn_regressor")


def get_optimizer(cfg_opt: Dict):
    """Create optimizer from config."""
    name = (cfg_opt.get("name", "adam") or "adam").lower()
    lr = float(cfg_opt.get("learning_rate", 1e-3))
    if name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=float(cfg_opt.get("beta_1", 0.9)),
            beta_2=float(cfg_opt.get("beta_2", 0.999)),
            epsilon=float(cfg_opt.get("epsilon", 1e-7)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def get_loss(name: str):
    """Get loss function by name."""
    n = (name or "mse").lower()
    if n == "mse":
        return tf.keras.losses.MeanSquaredError()
    if n == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    if n == "huber":
        return tf.keras.losses.Huber()
    raise ValueError(f"Unsupported loss: {name}")

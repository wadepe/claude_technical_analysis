"""
cnn_model.py

Shared model architecture imported by both train_cnn.py and evaluate_cnn.py.
Keeping the definition in one place guarantees that weight files are always
compatible between the two scripts.

Architecture — 1D CNN binary classifier
-----------------------------------------
Three Conv1D blocks (64 -> 128 -> 256 filters) with BatchNorm + MaxPooling
extract local and medium-range price structure from the 250-bar sequence.
GlobalAveragePooling collapses the full sequence into a single vector before
the Dense classification head.

Why this architecture for rising wedge detection:
  - Conv filters detect local slope / channel-width changes naturally
  - Stacked blocks combine those into progressively longer-range patterns
  - GlobalAveragePooling makes the model position-agnostic: it finds the wedge
    wherever it sits in the 250-bar window
  - Dropout + BatchNorm prevent overfitting on the 400K training samples
"""

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Features fed to the model — trendline / segment / label columns are excluded
# to prevent label leakage (trendlines are NaN for all noise datasets)
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
N_BARS       = int(os.environ.get('WEDGE_TOTAL_BARS', '250'))
N_FEATURES   = len(FEATURE_COLS)


def build_model(print_summary: bool = False) -> keras.Model:
    """
    Build and return the 1D CNN wedge classifier (uncompiled).

    Parameters
    ----------
    print_summary : bool
        Print a Keras model summary to stdout if True.

    Returns
    -------
    keras.Model
        Input  : (batch, 250, 5)  normalised OHLCV
        Output : (batch, 1)       P(rising wedge)  in [0, 1]
    """
    inputs = keras.Input(shape=(N_BARS, N_FEATURES), name='ohlcv')

    # ── Block 1  (7-bar receptive field — candle-level structure) ────────────
    x = layers.Conv1D(64, 7, padding='same', use_bias=False, name='conv1a')(inputs)
    x = layers.BatchNormalization(name='bn1a')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 7, padding='same', use_bias=False, name='conv1b')(x)
    x = layers.BatchNormalization(name='bn1b')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2, name='pool1')(x)          # (125, 64)
    x = layers.Dropout(0.20, name='drop1')(x)

    # ── Block 2  (5-bar — swing-level patterns) ───────────────────────────────
    x = layers.Conv1D(128, 5, padding='same', use_bias=False, name='conv2a')(x)
    x = layers.BatchNormalization(name='bn2a')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 5, padding='same', use_bias=False, name='conv2b')(x)
    x = layers.BatchNormalization(name='bn2b')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2, name='pool2')(x)          # (62, 128)
    x = layers.Dropout(0.20, name='drop2')(x)

    # ── Block 3  (3-bar — channel convergence / trend-level features) ─────────
    x = layers.Conv1D(256, 3, padding='same', use_bias=False, name='conv3a')(x)
    x = layers.BatchNormalization(name='bn3a')(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 3, padding='same', use_bias=False, name='conv3b')(x)
    x = layers.BatchNormalization(name='bn3b')(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)     # (256,)
    x = layers.Dropout(0.30, name='drop3')(x)

    # ── Classification head ───────────────────────────────────────────────────
    x       = layers.Dense(128, activation='relu', name='dense1')(x)
    x       = layers.Dropout(0.30, name='drop4')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs, outputs, name='wedge_1d_cnn')

    if print_summary:
        model.summary()

    return model

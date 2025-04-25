# ==============================================================
#  model.py   –  stacked‑LSTM with Masking (April 2025)
# ==============================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, Dense


def build_lstm_model(input_shape, num_classes):
    """
    Parameters
    ----------
    input_shape : (timesteps, feature_dim)
        For our project: (30, 26)   ← 26 engineered features / frame
    num_classes : int
        Number of behaviour categories (incl. “none”).

    Returns
    -------
    keras.Model
    """
    model = Sequential([
        # Skip any time‑steps that are all‑zeros (padding frames)
        Masking(mask_value=0.0, input_shape=input_shape),

        # Temporal feature extraction
        LSTM(64, return_sequences=True),
        Dropout(0.30),
        LSTM(32, return_sequences=False),
        Dropout(0.30),

        # Sequence‑level classification
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

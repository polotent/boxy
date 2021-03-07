from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


def get_model():
    model = Sequential([
        Flatten(input_shape=(400, 13, 1), name="flatten"),
        Dense(256, kernel_regularizer=regularizers.l2(1e-5), activation="relu", name="dense_1"),
        Dense(128, kernel_regularizer=regularizers.l2(1e-5), activation="relu", name="dense_2"),
        Dense(128, kernel_regularizer=regularizers.l2(1e-5), activation="relu", name="dense_3"),
        Dense(64, kernel_regularizer=regularizers.l2(1e-5), activation="relu", name="dense_4"),
        Dense(11, activation="softmax", name="dense_5"),
    ])
    return model

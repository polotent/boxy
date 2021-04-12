from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import numpy as np


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

def from_categorical(prediction):
    return np.argmax(prediction)

def predict_with_threshold(prediction, num_classes, threshold=0.5):
    class_num = np.argmax(prediction)
    return class_num if prediction[class_num] > threshold else num_classes    

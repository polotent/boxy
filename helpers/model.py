from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.math import confusion_matrix
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

def get_class_by_threshold(prediction, num_classes, threshold=0.5):
    class_num = np.argmax(prediction)
    return class_num if prediction[class_num] > threshold else num_classes    

def get_confusion_matrix(labels, predictions, num_classes, threshold):
    labels = [from_categorical(label) for label in labels]
    predictions = [get_class_by_threshold(prediction, num_classes, threshold=threshold) for prediction in predictions]
    return confusion_matrix(labels, predictions, num_classes+1)

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.math import confusion_matrix
import numpy as np
import pandas as pd
from helpers import get_commands_list_with_silence


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

def normalize_matrix(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)

def get_confusion_matrix(labels, predictions, nums, threshold):
    labels = [from_categorical(label) for label in labels]
    predictions = [get_class_by_threshold(prediction, len(nums), threshold=threshold) for prediction in predictions]
    conf_matrix = confusion_matrix(labels, predictions, len(nums)+1).numpy()
    conf_matrix = normalize_matrix(conf_matrix)
    df = pd.DataFrame(data=conf_matrix, columns=get_commands_list_with_silence(nums), index=get_commands_list_with_silence(nums))
    df.drop(df.tail(1).index,inplace=True)
    return df

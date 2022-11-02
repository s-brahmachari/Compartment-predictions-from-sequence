import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def FFN_block(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)

class CompPred(keras.Model):

    def __init__(self):
        super().__init__()

        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(128, activation="relu", bias_regularizer=tf.keras.regularizers.L2(0.001))
        self.dense3 = layers.Dense(64, activation="relu", bias_regularizer=tf.keras.regularizers.L2(0.001))
        self.dropout = layers.Dropout(0.5)
        self.softmax = layers.Dense(5, activation="sigmoid")

    def call(self, inputs,):    
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout(x)
        return self.softmax(x)



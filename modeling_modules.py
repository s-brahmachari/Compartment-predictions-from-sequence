import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
 
def FFN_block(hyper_params, name=None):
    fnn_layers = []
    for units in hyper_params["block_sizes"]:
        
        if hyper_params["useBatchNorm"]:
        
            fnn_layers.append(layers.BatchNormalization())
        
        if hyper_params["useDropout"]:
            
            assert "dropout_rate" in hyper_params, "Droput selected without specifying dropout rate"

            fnn_layers.append(layers.Dropout(hyper_params["dropout_rate"]))
        
        fnn_layers.append(layers.Dense(units, activation=hyper_params["activation"], 
                            kernel_initializer=hyper_params["initializer"]))

    return keras.Sequential(fnn_layers, name=name)

def baseFNNmodel(input_shape, output_shape, hyper_params):
    
    assert ("useDropout" in hyper_params and
            "block_sizes" in hyper_params and 
            "num_blocks" in hyper_params and
            "activation" in hyper_params), "Error! make sure hyper_params has dropout_rate, block_sizes, and num_blocks"
    
    inputs = layers.Input(shape=(input_shape,), name="input_features")
    
    x = FFN_block(hyper_params, name=f"ffn_block1")(inputs)

    for block_ndx in range(hyper_params["num_blocks"]):
    
        x1 = FFN_block(hyper_params, name=f"ffn_block{block_ndx+3}")(x)
    
        x = layers.Add(name=f"skip_connection{block_ndx+1}")([x, x1])

    # Compute logits.
    logits = layers.Dense(output_shape, name="logits")(x)
    # Create the model.
    return keras.Model(inputs=inputs, outputs=logits, name="baseFFN")


def run_experiment(model, x_train, y_train, x_test, y_test, num_epochs, hyper_params):
    # Compile the model.
    assert ("learning_rate" in hyper_params and
            "batch_size" in hyper_params and 
            "validation_split" in hyper_params), "Error! make sure hyper_params has learning_rate, batch_size, and validation_split"
    
    model.compile(
        optimizer=keras.optimizers.Adam(hyper_params["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=30, restore_best_weights=True, min_delta=0.02
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=hyper_params["batch_size"],
        validation_split=hyper_params["validation_split"],
        callbacks=[early_stopping],
    )

    eval_test = model.evaluate(x_test, y_test,)

    test_scores={
        "loss": eval_test[0],
        "accuracy": eval_test[1],
        "roc": roc_auc_score(y_test, np.argmax(model.predict(x_test), axis=1))
    }

    return history, test_scores


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



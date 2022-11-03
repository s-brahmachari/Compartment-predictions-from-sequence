import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import preprocessing_modules as pm
import modeling_modules as mm
import pickle

enc = pm.ENCODE_data(cell_line='GM12878', assembly='hg19', res=100000, histones=True, total_rna=False, tf=False)

if not os.path.exists(enc.cell_line_path): enc.download()

hyper_params={
    "learning_rate":0.001,
    "dropout_rate":0.2,
    "batch_size":32,
    "block_sizes":[64,64],
    "num_blocks":2,
    "validation_split":0.2,
    "useDropout": True,
    "useBatchNorm": True,
    "activation":"gelu",
    "initializer":"he_normal",
 }

save_dict = {"hyper_params":hyper_params}

for nn in range(5):

    x_df, labels = enc.get_training_data(n_neighbor=nn)

    for feature in x_df.columns:
        if "RNA" in feature:
            x_df.drop([feature], axis=1, inplace=True)

    X_train, X_test, Y_train, Y_test = train_test_split(x_df.to_numpy(), labels, test_size=0.25, random_state=324)

    model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)

    #print(model.summary())

    hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=10, hyper_params=hyper_params)

    save_dict[f"n{nn}"] = {"train_loss":hist.history["loss"], 
                            "val_loss":hist.history["val_loss"],
                            "train_accu":hist.history["accuracy"],
                            "val_accu":hist.history["val_accuracy"],
                            "test_accu":test_scores["accuracy"],
                            "test_loss":test_scores["loss"],
                            "roc":test_scores["roc"]
    }


with open('analysis_data/vary_neighbor_num.pickle', 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


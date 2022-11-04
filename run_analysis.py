import numpy as np
import itertools
import copy
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import preprocessing_modules as pm
import modeling_modules as mm
import pickle

def get_summary_dict(hist, test_scores):
    return {"train_loss":hist.history["loss"],
            "val_loss":hist.history["val_loss"],
            "train_accu":hist.history["accuracy"],
            "val_accu":hist.history["val_accuracy"],
            "test_accu":test_scores["accuracy"],
            "test_loss":test_scores["loss"],
            "roc":test_scores["roc"],}
    
def neighbor_importance(hyper_params, num_epochs=50):
    enc = pm.ENCODE_data(cell_line='GM12878', assembly='hg19', res=100000, histones=True, total_rna=False, tf=False)
    if not os.path.exists(enc.cell_line_path): enc.download()

    save_dict = {"hyper_params":hyper_params}

    for nn in range(5):
        x_df, labels = enc.get_training_data(n_neighbor=nn)

        for feature in x_df.columns:
            if "RNA" in feature:
                x_df.drop([feature], axis=1, inplace=True)

        X_train, X_test, Y_train, Y_test = train_test_split(x_df.to_numpy(), labels, test_size=0.25, random_state=324)

        model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)

        hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=num_epochs, hyper_params=hyper_params)

        save_dict[f"n{nn}"] = get_summary_dict(hist, test_scores)

    with open('analysis_data/vary_neighbor_num.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Saved analysis at analysis_data/vary_neighbor_num.pickle")


def feature_importance(hyper_params, num_epochs=50, remove_num_features=1, n_neighbor=1):

    enc = pm.ENCODE_data(cell_line='GM12878', assembly='hg19', res=100000, histones=True, total_rna=False, tf=False)
    if not os.path.exists(enc.cell_line_path): enc.download()

    save_dict = {"hyper_params":hyper_params}
    x_df, labels = enc.get_training_data(n_neighbor=n_neighbor)
    
    features = []
    for ftr in x_df.columns:
        features.append(ftr.split('_')[0])
    features = np.unique(features)
    features = features[features!='RNA']

    print(features)

    X_train, X_test, Y_train, Y_test = train_test_split(x_df.to_numpy(), labels, test_size=0.25, random_state=324)

    model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)
    hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=num_epochs, hyper_params=hyper_params)

    save_dict["all"] = get_summary_dict(hist, test_scores)

    for ftr_tuple in itertools.combinations(features, remove_num_features):
        x_df_filtered = copy.deepcopy(x_df)
        for col, ftr in itertools.product(x_df.columns, ftr_tuple):
            if ftr in col:
                x_df_filtered.drop([col], axis=1, inplace=True)
                    
        print(f"\nRemoved features: {ftr_tuple}")
        model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)
        hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=num_epochs, hyper_params=hyper_params)
             
        save_dict["-".join(ftr_tuple)] = get_summary_dict(hist, test_scores)

    with open(f"analysis_data/feature_importance_{remove_num_features}features_removed.pickle", 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved analysis at analysis_data/")
        

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

#neighbor_importance(hyper_params, num_epochs=50)
feature_importance(hyper_params, num_epochs=50, remove_num_features=10)

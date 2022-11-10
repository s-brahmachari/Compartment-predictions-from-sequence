import numpy as np
import itertools
import copy
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import preprocessing_modules as pm
import modeling_modules as mm
import pickle
import multiprocessing

def get_summary_dict(hist, test_scores):
    return {"train_loss":hist.history["loss"],
            "val_loss":hist.history["val_loss"],
            "train_accu":hist.history["accuracy"],
            "val_accu":hist.history["val_accuracy"],
            "test_accu":test_scores["accuracy"],
            "test_loss":test_scores["loss"],
            "roc":test_scores["roc"],}
    
def neighbor_importance(hyper_params, num_epochs=50, use_features=None,rand=None, ):
    enc = pm.ENCODE_data(cell_line='GM12878', assembly='hg19', res=100000, avg_type="mean",
                        save_dest="/home/sb95/Compartment-predictions-from-sequence/ENCODE_data", 
                        histones=True, total_rna=False, tf=False)
    if not os.path.exists(enc.cell_line_path): enc.download()

    save_dict = {}
    if rand==None: rand = np.random.randint(1,1000)
    for nn in range(8):
        x_df, labels = enc.get_training_data(n_neighbor=nn)
        features = []
        if use_features!=None:
            for ftr in x_df.columns:
                if ftr.split("_")[0] not in use_features:
                    x_df.drop([ftr], axis=1, inplace=True)
                else:
                    features.append(ftr.split('_')[0])
        features = np.unique(features)
        print(x_df.columns)
        # for feature in x_df.columns:
        #     if "RNA" in feature:
        #         x_df.drop([feature], axis=1, inplace=True)

        X_train, X_test, Y_train, Y_test = train_test_split(x_df.to_numpy(), labels, test_size=0.25, random_state=rand)

        model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)

        hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=num_epochs, hyper_params=hyper_params)

        save_dict[f"n{nn}"] = get_summary_dict(hist, test_scores)

    return save_dict


def create_and_run_model(X_in):        
    model = mm.baseFNNmodel(input_shape=X_in[0].shape[1], output_shape=len(np.unique(X_in[1])), hyper_params=hyper_params)
    print(model.summary())
    hist, test_scores = mm.run_experiment(model, X_in[0], X_in[1], X_in[2], X_in[3], num_epochs=X_in[4], hyper_params=X_in[5])
    return np.array([hist, test_scores])

def feature_importance(hyper_params, num_epochs=50, use_num_features=1, n_neighbor=1, rand=None, use_features=None ):
    
    enc = pm.ENCODE_data(cell_line='GM12878', assembly='hg19', res=100000, avg_type="mean",
                        save_dest="/home/sb95/Compartment-predictions-from-sequence/ENCODE_data", 
                        histones=True, total_rna=False, tf=False)
    if not os.path.exists(enc.cell_line_path): enc.download()

    save_dict = {}#"hyper_params":hyper_params}
    x_df, labels = enc.get_training_data(n_neighbor=n_neighbor)
    
    if use_features!=None:
        features = []
        for ftr in x_df.columns:
            if ftr.split("_")[0] not in use_features:
                x_df.drop([ftr], axis=1, inplace=True)
            else:
                features.append(ftr.split('_')[0])
    else:
        features = [xx.split('_')[0] for xx in x_df.columns]

    features = np.unique(features)

    print(features)
    if rand==None: rand = np.random.randint(1,1000)
    X_train, X_test, Y_train, Y_test = train_test_split(x_df.to_numpy(), labels, test_size=0.25, random_state=rand)

    model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)
    hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=num_epochs, hyper_params=hyper_params)
    save_dict["-".join(features)] = get_summary_dict(hist, test_scores)
    
    for ftr_tuple in itertools.combinations(features, use_num_features):
        # print(ftr_tuple)
        x_df_filtered = copy.deepcopy(x_df)
        for col in x_df.columns:
            if col.split('_')[0] not in ftr_tuple:
                # print('drop')
                x_df_filtered.drop([col], axis=1, inplace=True)
        
        print(x_df_filtered.shape)
        # features_used.append("-".join(ftr_tuple))

        print(f"\nUsed features: {ftr_tuple}")
        X_train, X_test, Y_train, Y_test = train_test_split(x_df_filtered.to_numpy(), labels, test_size=0.25, random_state=rand)
        # inputs_red_features.append(np.array([X_train, Y_train, X_test, Y_test, num_epochs, hyper_params]))
        model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)
        hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=num_epochs, hyper_params=hyper_params)
    
        save_dict["-".join(ftr_tuple)] = get_summary_dict(hist, test_scores)
    
    return save_dict

def search_hyper_params(hyper_params, search_param, search_range, num_epochs=50, n_neighbor=1, rand=None, use_features=None ):
    
    enc = pm.ENCODE_data(cell_line='GM12878', assembly='hg19', res=100000, avg_type="mean",
                        save_dest="/home/sb95/Compartment-predictions-from-sequence/ENCODE_data", 
                        histones=True, total_rna=False, tf=False)
    
    if not os.path.exists(enc.cell_line_path): enc.download()

    save_dict = {}#{"hyper_params":hyper_params}
    x_df, labels = enc.get_training_data(n_neighbor=n_neighbor, 
                                          pcut_high=hyper_params["pcut_high"],
                                          pcut_low=hyper_params["pcut_low"])
    
    features=[]
    if use_features!=None:
        for ftr in x_df.columns:
            if ftr.split("_")[0] not in use_features:
                x_df.drop([ftr], axis=1, inplace=True)
            else:
                features.append(ftr.split('_')[0])
    features = np.unique(features)
    # features = features[features!='RNA']

    print(features)
    if rand==None: rand = np.random.randint(1,1000)
    X_train, X_test, Y_train, Y_test = train_test_split(x_df.to_numpy(), labels, test_size=0.25, random_state=rand)

    for param in search_range:
        hyper_params[search_param] = param
        print(f"\n{search_param} = {param}")
        model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)
        print(model.summary())
        hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=num_epochs, hyper_params=hyper_params)
        
        save_dict[f"{param}"] = get_summary_dict(hist, test_scores)
        print(f"ROC: {test_scores['roc']}\n")

    # with open(f"analysis_data/hyper_param_search_{search_param}_rep{replica}.pickle", 'wb') as handle:
        # pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print("Saved analysis at analysis_data/")
    return save_dict

def main():
    hyper_params={
        "pcut_high":95,
        "pcut_low":5,
        "learning_rate":0.005,
        "dropout_rate":0.05,
        "batch_size":64,
        "units_per_layer":32,
        "block_size":3,
        "num_blocks":2,
        "validation_split":0.2,
        "useDropout": True,
        "useBatchNorm": True,
        "useSkip":True,
        "activation":"gelu",
        "initializer":"he_normal",
    }
    save_dict = {"hyper_params": hyper_params}

    param_to_optimize = "useBatchNorm"
    optimize_list = [True, False]
    num_features=3

    for rep in range(10):
        print(f"\nReplica: {rep+1}\n")
        save_dict[f"replica{rep+1}"] = feature_importance(hyper_params, num_epochs=30, use_num_features=num_features,
                                                        use_features=["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"])

        # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, param_to_optimize, optimize_list, num_epochs=30, 
        #                                                 use_features=["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"])

        # save_dict[f"replica{rep+1}"] = neighbor_importance(hyper_params, num_epochs=30,
        #                                                 use_features=["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"])
        
    # with open(f"/home/sb95/Compartment-predictions-from-sequence/analysis_data/hyper_param_search_{param_to_optimize}.pickle", 'wb') as handle:
    with open(f"/home/sb95/Compartment-predictions-from-sequence/analysis_data/feature_importance_use{num_features}feature.pickle", 'wb') as handle:
    # with open(f"/home/sb95/Compartment-predictions-from-sequence/analysis_data/neighbor_importance.pickle", 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
# for rep in range(5):
#     
#     #neighbor_importance(hyper_params, num_epochs=50)
#     save_dict[f"replica{rep+1}"] = feature_importance(hyper_params, num_epochs=50, remove_num_features=10)
    
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "units_per_layer", [4, 8, 16, 32, 64], num_epochs=60, rand=rep)
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "block_size", [1, 2, 3, 4, 5], num_epochs=60)
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "dropout_rate", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], num_epochs=60)
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "learning_rate", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05], num_epochs=60)
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "num_blocks", [0, 1, 2, 3, 4, 5], num_epochs=60)
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "activation", ["gelu", "relu", "sigmoid", "linear", "selu"], num_epochs=60)
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "batch_size", [32, 64, 128, 256, 512], num_epochs=60)
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "initializer", ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform", "ones", "RandomUniform", "RandomNormal"], num_epochs=60)
#     # save_dict[f"replica{rep+1}"] = search_hyper_params(hyper_params, "useSkip", [True, False], num_epochs=60)

# # with open("/home/sb95/Compartment-predictions-from-sequence/analysis_data/hyper_param_search_useSkip.pickle", 'wb') as handle:
# with open("/home/sb95/Compartment-predictions-from-sequence/analysis_data/feature_importance_remove10features.pickle", 'wb') as handle:
#     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import preprocessing_modules as pm
import modeling_modules as mm
import pickle
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sn
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-rand', dest='rand', type=int, default=465)
parser.add_argument('-n_epochs', dest='num_epochs', type=int, default=20)

args = parser.parse_args()

hyper_params={
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

def get_data():
    enc = pm.ENCODE_data(cell_line='GM12878', assembly='hg19', res=100000, 
                        histones=True, total_rna=False, tf=False, avg_type="mean")

    if not os.path.exists(enc.cell_line_path): enc.download()
    x_df, labels = enc.get_training_data(n_neighbor=3, pcut_high=95, pcut_low=5)
    return enc, x_df, labels


# save_dict = {"hyper_params":hyper_params}
def main():
    
    enc, x_df_0, labels = get_data()

    use_features=["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]

    # features_all = np.unique([xx.split('_')[0] for xx in x_df.columns])
    
    for feature in x_df_0.columns:
        if feature.split('_')[0] not in use_features:
            x_df_0.drop([feature], axis=1, inplace=True)
    
    _, df_cm, df_cm_AB = create_confusion_matrix(x_df_0, labels, enc)
    baseline = np.append(np.diag(df_cm.to_numpy()), np.diag(df_cm_AB.to_numpy()))
    ftr_drop_df = []
    
    for ftr_drop in use_features:
        x_df = copy.deepcopy(x_df_0)
        for feature in x_df.columns:
            if feature.split('_')[0]==ftr_drop:
                x_df.drop([feature], axis=1, inplace=True)

        print(x_df.columns)
        fig, df_cm, df_cm_AB = create_confusion_matrix(x_df, labels, enc)
        diag_vals = np.append(np.diag(df_cm.to_numpy()), np.diag(df_cm_AB.to_numpy()))
        fig.savefig(f'analysis_data/plots/confusion-{ftr_drop}.png', dpi=300, bbox_inches="tight")

        ftr_drop_df.append(diag_vals-baseline)

    ftr_drop_df = pd.DataFrame(np.array(ftr_drop_df).reshape(len(use_features),-1),
                                index = use_features,
                                columns = ['A1','A2','B1','B2','B3', 'A', 'B'])
    
    fig2 = plt.figure(figsize = (8,5), dpi=300)
    v0 = max(abs(ftr_drop_df.to_numpy().max()),abs(ftr_drop_df.to_numpy().min()))
    g1 = sn.heatmap(ftr_drop_df, vmin=-v0, vmax=v0,annot=True,cmap="RdBu_r", annot_kws={'fontsize':12}, fmt='.2f', cbar=False)
    # g1.set(xlabel='Predictions (DNN)', ylabel='Actual (Hi-C)')
    sn.set(font_scale=1.2)
    fig2.savefig(f'analysis_data/plots/confusion-ftr_drop.png', dpi=300, bbox_inches="tight")


def create_confusion_matrix(x_df, labels, enc):

    X_train, X_test, Y_train, Y_test = train_test_split(x_df.to_numpy(), labels, test_size=0.5, random_state=args.rand)

    model = mm.baseFNNmodel(input_shape=X_train.shape[1], output_shape=len(np.unique(Y_train)), hyper_params=hyper_params)
    # keras.utils.plot_model(model, to_file='analysis_data/plots/model.png', show_shapes=False, dpi=300, show_layer_names=False, rankdir='TB')
    print(model.summary())

    hist, test_scores = mm.run_experiment(model, X_train, Y_train, X_test, Y_test, num_epochs=args.num_epochs, hyper_params=hyper_params)

    Y_pred = model.predict(X_test)
    Y_pred = Y_pred/np.sum(Y_pred, axis=1, keepdims=True)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_pred_AB = list(map(enc.int_subtype_to_int_AB.get, Y_pred))
    Y_pred_AB = tf.convert_to_tensor(Y_pred_AB,dtype=tf.int32)
    Y_pred = tf.convert_to_tensor(Y_pred,dtype=tf.int32)

    Y_test_AB = list(map(enc.int_subtype_to_int_AB.get, Y_test))
    Y_test_AB=tf.convert_to_tensor(Y_test_AB, dtype=tf.int32)
    Y_test=tf.convert_to_tensor(Y_test, dtype=tf.int32)

    conf_mat=tf.math.confusion_matrix(Y_test, Y_pred,  dtype=tf.dtypes.float32)
    conf_mat=normalize(conf_mat, norm='l1', axis=0)

    conf_mat_AB=tf.math.confusion_matrix(Y_test_AB, Y_pred_AB,  dtype=tf.dtypes.float32)
    conf_mat_AB=normalize(conf_mat_AB, norm='l1', axis=0)

    df_cm = pd.DataFrame(conf_mat, index = ['A1','A2','B1','B2','B3'],
                columns = ['A1','A2','B1','B2','B3'])

    df_cm_AB = pd.DataFrame(conf_mat_AB, index = ['A','B'],
                columns = ['A','B'])

    fig = plt.figure(figsize = (8,5), dpi=300)

    ax1 = fig.add_axes([0.1,0.1,0.5,0.8])
    ax2 = fig.add_axes([0.7,0.1,0.25,0.4])

    g1 = sn.heatmap(df_cm, annot=True,cmap="OrRd", annot_kws={'fontsize':12}, fmt='.2f', cbar=False, ax=ax1)
    g1.set(xlabel='Predictions (DNN)', ylabel='Actual (Hi-C)')

    g2 = sn.heatmap(df_cm_AB, annot=True,cmap="OrRd", annot_kws={'fontsize':12}, fmt='.2f', cbar=False, ax=ax2)
    g2.set(xlabel='Predictions (DNN)', ylabel='Actual (Hi-C)')

    sn.set(font_scale=1.2)
    
    return fig, df_cm, df_cm_AB

    

    # save_dict["result"] = {"train_loss":hist.history["loss"], 
    #                             "val_loss":hist.history["val_loss"],
    #                             "train_accu":hist.history["accuracy"],
    #                             "val_accu":hist.history["val_accuracy"],
    #                             "test_accu":test_scores["accuracy"],
    #                             "test_loss":test_scores["loss"],
    #                             "roc":test_scores["roc"]
    # }

    # print(test_scores)
#with open('analysis_data/vary_neighbor_num.pickle', 'wb') as handle:
#    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__': main()

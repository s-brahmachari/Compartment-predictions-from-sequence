import os
import numpy as np
import copy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class preprocess():
    def __init__(self,data_path):
        #set the data path
        self.path=data_path

        #list of histone marks
        self.tracks=[]
        
        for val in sorted(os.listdir(self.path)):
            h=val.split('-')
            if len(h)>1:
                self.tracks.append(h[0])

        self.tracks=np.unique(self.tracks)
        #combine experiment replicas
        self.avg_data={f'{xx}':{} for xx in self.tracks}

        self.type_to_int={'A1':0, 'A2':1, 'B1':2, 'B2':3, 'B3':4, 'B4':5, 'NA':6}
        self.int_to_type={0:'A1', 1:'A2', 2:'B1', 3:'B2', 4:'B3', 5:'B4', 6:'NA'}
        self.type_to_int_AB={'A1':0, 'A2':0, 'B1':1, 'B2':1, 'B3':1, 'B4':1, 'NA':2}
        self.sub_to_comp={0:0,1:0,2:1,3:1,4:1,5:1,6:2}
        
        # histone_marks=['RNA-seq']
        # print(self.tracks)
        print(f'Loading data from: {data_path}')
        self.load_data()
        print(f'Using {len(self.tracks)} tracks:', self.tracks)
        print(f'Number of chromosomes:{self.Nchr}')
        print('\nNormalizing data...')
        self.normalize_data()

        print('done!')


    def load_data(self,):
        
        for track in self.tracks:
            data=[xx for xx in os.listdir(self.path) if track in xx]
            # print(track)
            for val in data:
                # print(val)
                for fname in os.listdir(self.path+val):
                    if '.track' not in fname or 'chrX' in fname: continue
                    # if 'chrX' in ff: continue
                    chrm=fname.replace('.track','')
                    # print(chrkey, chip[mark].keys())
                    if chrm in self.avg_data[track].keys():
                        # print('yellow')
                        self.avg_data[track][chrm]=np.vstack((self.avg_data[track][chrm], np.loadtxt(self.path+val+'/'+fname, skiprows=3,usecols=2)))
                    elif chrm not in self.avg_data[track].keys():
                        # print('red')
                        self.avg_data[track][chrm]=np.loadtxt(self.path+val+'/'+fname, skiprows=3,usecols=2)
        self.Nchr=len(self.avg_data[track].keys())
        
        # self.Nchr=len(self.avg_data[track].keys())
            # for key in chip[mark].keys():
            #     print(key, chip[mark][key].shape)

    def normalize_data(self,p_cut=98):
        #average the chip seq signals then normalize them to be between 0 and 1
        self.norm_data=copy.deepcopy(self.avg_data)
        for track in self.avg_data.keys():
            # print(track)
            for chrm in self.avg_data[track].keys():
                self.norm_data[track][chrm] = np.mean(self.avg_data[track][chrm], axis=0)
                self.norm_data[track][chrm] -= self.norm_data[track][chrm].min()

                per=np.percentile(self.norm_data[track][chrm],p_cut)
                self.norm_data[track][chrm] /= per #norm_chip[key][k2].max()
                self.norm_data[track][chrm][self.norm_data[track][chrm]>1]=1.0
                # print(self.norm_data[track][chrm].min(), self.norm_data[track][chrm].max())
                # print(k2, chip[key][k2].shape, norm_chip[key][k2].shape)

    def gen_Xtrain(self, n_neighbor):
        #state (input) vector
        xdata=[]
        for track in self.avg_data.keys():
            # print(track)
            row=self.norm_data[track]['chr1']
            # rowlen=row.shape[0]
            # Nchr=len(self.norm_data[track].keys())
            for jj in range(2,self.Nchr):
                chrm='chr{}'.format(jj)
                row = np.concatenate((row,self.norm_data[track][chrm]))
                # rowlen+=norm_chip[mark][chrm].shape[0]
            # row=np.array(row)
            xdata.append(row)
            for n in range(1,n_neighbor+1):
                xdata.append(np.pad(row[n:],(0,n)))
                xdata.append(np.pad(row[:-n],(n,0)))
            # print(row.shape,)
        
        xdata=np.array(xdata).reshape(len(self.tracks)*(2*n_neighbor+1),-1)

        return xdata

    def gen_Ytrain(self,):
        typepath=self.path+'/types/'
        ydata=np.loadtxt(typepath+'chr1_beads.txt.original', dtype=str, usecols=1)
        for chrm in range(2,self.Nchr):
            ydata=np.concatenate((ydata, np.loadtxt(typepath+'chr{}_beads.txt.original'.format(chrm), dtype=str, usecols=1)))
        return ydata

    def gen_input_vec(self,n_neighbor):
        xtrain=self.gen_Xtrain(n_neighbor)
        ytrain=self.gen_Ytrain()
        vec=np.vstack((ytrain, xtrain))

        #remove 'NA' and 'B4' from predictions
        indices_NA=np.where(vec[0]=='NA')[0]
        indices_B4=np.where(vec[0]=='B4')[0]
        indices_to_remove=np.concatenate((indices_NA, indices_B4))
        vec_reduced = np.delete(vec, indices_to_remove, axis=1)
        
        xtrain=np.array(vec_reduced[1:].T, dtype=float)
        ytrain=np.array(list(map(self.type_to_int.get, vec_reduced[0])), dtype=float)
        return [xtrain, ytrain]


class CompPred(keras.Model):

    def __init__(self):
        super().__init__()

        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(128, activation="relu", bias_regularizer=tf.keras.regularizers.L2(0.001))
        self.dense3 = layers.Dense(64, activation="relu", bias_regularizer=tf.keras.regularizers.L2(0.001))
        self.dropout = layers.Dropout(0.5)
        self.softmax = layers.Dense(5, activation="relu")

    def call(self, inputs,):    
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout(x)
        return self.softmax(x)


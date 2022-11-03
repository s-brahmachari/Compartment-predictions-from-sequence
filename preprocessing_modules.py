import numpy as np
import copy
import pandas as pd
import pyBigWig
import sys, os, time, glob, random, requests, shutil
from tqdm import tqdm
from joblib import Parallel, delayed


class ENCODE_data:
    def __init__(self, cell_line='GM12878', assembly='hg19',
                 signal_type='signal p-value',
                 save_dest='ENCODE_data',
                 res=100000,
                 histones=True,tf=False,atac=False,small_rna=False,total_rna=False):

        self.cell_line=cell_line
        self.res = res
        self.assembly=assembly
        self.signal_type=signal_type
        self.savedest=save_dest
        self.cell_line_path=os.path.join(save_dest,cell_line+'_'+assembly+'_'+str(int(res/1000))+'k')
        self.ref_cell_line='GM12878'
        self.ref_assembly='hg19'
        # self.ref_cell_line_path=ref_cell_line_path
        # self.types_path=types_path
        self.hist=histones
        self.tf=tf
        self.atac=atac
        self.small_rna=small_rna
        self.total_rna=total_rna

        self.type_to_int={'A1':0, 'A2':0, 'B1':1, 'B2':1, 'B3':1, 'B4':1, 'NA':0}
        self.int_to_type={0:'A1', 1:'A2', 2:'B1', 3:'B2', 4:'B3', 5:'B4', 6:'NA'}
        self.set_chrm_size(res=self.res)
        
        print('Selected cell line to predict: '+self.cell_line)
        print('Selected assembly: '+self.assembly)
        print('Selected signal type: '+self.signal_type)
        print('Selected resolution: ', int(self.res/1000),'kb')
        
    def _download_replicas(self,line,cell_line_path,chrm_size):
        text=line.split()[0]
        exp=line.split()[1]
        count=line.split()[2]

        #Experiment directory 
        exp_path=cell_line_path+'/'+exp+'_'+str(count)

        if exp in self.experiments_unique:
            try:
                os.mkdir(exp_path)
            except:
                print('Directory ',exp_path,' already exist')

            with open(exp_path+'/exp_name.txt', 'w') as f:
                f.write(text+' '+exp+'\n')
                
            #Load data from server
            try:
                bw = pyBigWig.open("https://www.encodeproject.org/files/"+text+"/@@download/"+text+".bigWig")
                for chr in range(1,23):
                    
                    signal = bw.stats("chr"+str(chr), type="mean", nBins=chrm_size[chr-1])
                    signal=np.array(signal)
                    signal[signal==None]=0.0
                    signal[signal<0.0]=0.0
                    
                    #Save data
                    with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:

                        f.write("#chromosome file number of beads\n"+str(chrm_size[chr-1]))
                        f.write("#\n")
                        f.write("#bead, signal, discrete signal\n")
                        for i in range(len(signal)):
                            f.write(str(i)+" "+str(signal[i])+" "+str(signal[i])+"\n")
                chr='X'
                signal = bw.stats("chr"+chr, type="mean", nBins=chrm_size[-1])

                #Process signal and binning
                signal=np.array(signal)
                signal[signal==None]=0.0
                signal[signal<0.0]=0.0
                
                #Save data
                with open(exp_path+'/chr'+chr+'.track', 'w') as f:

                    f.write("#chromosome file number of beads\n"+str(chrm_size[-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i])+"\n")
                return exp

            except:
                print('This experiment was incomplete:',text,'\nit will not be used.')

    def set_chrm_size(self, res):
        self.res = res

        if self.assembly=='GRCh38':
            url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_assembly_report.txt"
        else:
            url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.13_GRCh37/GCF_000001405.13_GRCh37_assembly_report.txt"

        ref=pd.read_table(url, comment='#', sep='\t',
                    names=['Sequence-Name', 'Sequence-Role', 'Assigned-Molecule','Assigned-Molecule-Location/Type',	'GenBank-Accn','Relationship', 'RefSeq-Accn', 'Assembly-Unit', 'Sequence-Length', 'UCSC-style-name'])

        self.chrm_size=list((ref[ (ref['Sequence-Role']=='assembled-molecule') & (ref['Sequence-Name']!='Y') ]['Sequence-Length']/self.res).apply(np.ceil).astype(int))
        
    def get_chrm_size(self):
        return self.chrm_size

    def download(self, nproc=4,):

        assert hasattr(self, "chrm_size"), "Resolution not set. Use set_chrm_size(res)"

        url='https://www.encodeproject.org/metadata/?type=Experiment&'
        if self.hist==True:
            url=url+'assay_title=Histone+ChIP-seq'
        if self.tf==True:
            url=url+'&assay_title=TF+ChIP-seq'
        if self.atac==True:
            url=url+'&assay_title=ATAC-seq'
        if self.small_rna==True:
            url=url+'&assay_title=small+RNA-seq'
        if self.total_rna==True:
            url=url+'&assay_title=total+RNA-seq'
        self.url_ref=url+'&biosample_ontology.term_name='+self.ref_cell_line+'&files.file_type=bigWig'

        r = requests.get(self.url_ref)
        content=str(r.content)
        experiments=[]
        for k in content.split('\\n')[:-1]:
            l=k.split('\\t')
            if l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                experiments.append(l[22])
            elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                experiments.append(l[7])
            elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                experiments.append(l[22])
            elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                experiments.append('RNA-seq-plus-small')
            elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                experiments.append('RNA-seq-plus-total')          
            elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                experiments.append('RNA-seq-minus-small')
            elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                experiments.append('RNA-seq-minus-total')          

        self.experiments_unique=np.unique(experiments)   

        
        try:
            if self.savedest not in os.listdir():
                os.mkdir(self.savedest)
            os.mkdir(self.cell_line_path)
        except (FileExistsError, ):
            print('Directory ',self.cell_line_path,' already exists!')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.cell_line_path)
            os.mkdir(self.cell_line_path)
        
        url='https://www.encodeproject.org/metadata/?type=Experiment&'
        if self.hist==True:
            url=url+'assay_title=Histone+ChIP-seq'
        if self.tf==True:
            url=url+'&assay_title=TF+ChIP-seq'
        if self.atac==True:
            url=url+'&assay_title=ATAC-seq'
        if self.small_rna==True:
            url=url+'&assay_title=small+RNA-seq'
        if self.total_rna==True:
            url=url+'&assay_title=total+RNA-seq'
        self.url_cell_line=url+'&biosample_ontology.term_name='+self.cell_line+'&files.file_type=bigWig'

        r = requests.get(self.url_cell_line)
        content=str(r.content)

        with open(self.cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                    f.write(l[0]+' '+l[7]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'RNA-seq-plus-small'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'RNA-seq-plus-total'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'RNA-seq-minus-small'+' '+l[5]+' '+l[4]+'\n')
                elif l[5]==self.assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'RNA-seq-minus-total'+' '+l[5]+' '+l[4]+'\n')
       
        count=0
        self.exp_found={}
        exp_name=''
        list_names=[]

        with open(self.cell_line_path+'/meta.txt') as fp:
            Lines = fp.readlines()
            for line in Lines:
                count += 1
                text=line.split()[0]
                exp=line.split()[1]

                #Register if experiment is new
                if exp!=exp_name:
                    try:
                        count=self.exp_found[exp]+1
                    except:
                        count=1
                    exp_name=exp
                self.exp_found[exp]=count
                list_names.append(text+' '+exp+' '+str(count))

        print('Number of replicas:', len(list_names))
        self.successful_exp = Parallel(n_jobs=nproc)(delayed(self._download_replicas)(list_names[i],self.cell_line_path,self.chrm_size) 
                                      for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        self.successful_exp= [i for i in self.successful_exp if i]
        self.successful_unique_exp=np.unique(self.successful_exp)

        print('Experiments found in ENCODE for the selected cell line:')
        self.unique=[]

        with open(self.cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.experiments_unique:
                if e in self.successful_unique_exp:
                    f.write(e.split('-hum')[0]+'\n')
                    print(e.split('-hum')[0])
                    self.unique.append(e)
        
        print('Downoaded data stored at: ',self.cell_line_path)

    def _load_data(self,):
        # load each track from data folder
        for track in self.tracks:
            track_replicas=[xx for xx in os.listdir(self.cell_line_path) if track in xx]
            
            # average replicas
            for replica in track_replicas:
                for fname in os.listdir(os.path.join(self.cell_line_path,replica)):

                    # ignore chr X
                    if '.track' not in fname or 'chrX' in fname: continue
                    chrm=fname.replace('.track','')
                    
                    # if avg_data has the track and chromosome key store the value 
                    # else create the keys
                    if chrm in self.avg_data[track].keys():
                        self.avg_data[track][chrm]=np.vstack((self.avg_data[track][chrm], np.loadtxt(os.path.join(self.cell_line_path,replica,fname), skiprows=3,usecols=2)))
                    elif chrm not in self.avg_data[track].keys():
                        self.avg_data[track][chrm]=np.loadtxt(os.path.join(self.cell_line_path,replica,fname), skiprows=3,usecols=2)

        self.Nchr=len(self.avg_data[track].keys())
        
    def _normalize_data(self, p_cut=90):
        # average the replica tracks then normalize them to be between 0 and 1
        self.norm_data=copy.deepcopy(self.avg_data)
        for track in self.avg_data.keys():
            
            for chrm in self.avg_data[track].keys():
                
                # average when more than one replica
                if len(self.norm_data[track][chrm].shape)>1:
                    self.norm_data[track][chrm] = np.mean(self.avg_data[track][chrm], axis=0)
                
                # shift the baseline to the minumum value
                self.norm_data[track][chrm] -= self.norm_data[track][chrm].min()

                # linear transform the data by dividing by p_cut percentile
                percentile_cutoff = np.percentile(self.norm_data[track][chrm], p_cut)
                self.norm_data[track][chrm] /= percentile_cutoff
                
                # saturate the high values at p_cut percentile
                self.norm_data[track][chrm][self.norm_data[track][chrm]>1]=1.0

    def preprocess(self, n_neighbor=2, save=False):
        
        #list of experiment tracks
        #combine plus/minus RNA seq signals
        self.tracks=np.unique([xx.split('-')[0] for xx in 
                        np.loadtxt(os.path.join(self.cell_line_path, 'unique_exp.txt'), dtype=str)])

        # create dict of dict to store data for tracks and chromosomes
        self.avg_data={f'{xx}':{} for xx in self.tracks}

        # self.type_to_int_AB={'A1':0, 'A2':0, 'B1':1, 'B2':1, 'B3':1, 'B4':1, 'NA':2}
        # self.sub_to_comp={0:0,1:0,2:1,3:1,4:1,5:1,6:2}
        
        print(f'Loading data from: {self.cell_line_path}')
        self._load_data()
        print(f'There are {len(self.tracks)} tracks:', self.tracks)
        print(f'Number of chromosomes:{self.Nchr}')
        print('\nNormalizing data...')
        self._normalize_data()
        print('done!')

        x_pre, y_pre, features = self.get_training_data(n_neighbor=n_neighbor, features=self.get_features(), typespath=None)
        df = pd.DataFrame(np.concatenate((x_pre,y_pre.reshape(-1,1)), axis=1), 
                    columns=np.concatenate((features,['labels'])))
        
        if save==True:
            print('Saving training data ...')
            df.to_csv(os.path.join(self.savedest, 'processed_data',self.cell_line+'_'+self.assembly,'.csv.zip' ), compression='zip', index=False)
        print('done!')
        
        return df

    def get_features(self, ):
        return list(self.norm_data.keys())

    def get_training_labels(self,):
        df=pd.read_table('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_subcompartments.bed.gz', 
                            names=['chr', 'x1', 'x2', 'type','label'], 
                            usecols=range(5),keep_default_na=False)
    
        labels=[]
        for ii in list(range(1,self.Nchr+1)):
            for xx in df[df['chr']==f'chr{ii}'].to_numpy():
                n=np.ceil((xx[2]-xx[1])/self.res)
                for _ in range(int(n)):
                    # print(int(xx[1]/res)+nn, xx[3])
                    labels.append(xx[3])
        return np.array(labels, dtype=str)

    def get_input_vec(self, n_neighbor=2, features=None):
        
        unique_exps= os.path.join(self.cell_line_path, 'unique_exp.txt')
        assert os.path.exists(unique_exps), 'Data path does not exist! Run download() first!'

        #list of experiment tracks
        #combine plus/minus RNA seq signals
        self.tracks=np.unique([xx.split('-')[0] for xx in 
                        np.loadtxt(unique_exps, dtype=str)])

        # create dict of dict to store data for tracks and chromosomes
        self.avg_data={f'{xx}':{} for xx in self.tracks}

        # self.type_to_int_AB={'A1':0, 'A2':0, 'B1':1, 'B2':1, 'B3':1, 'B4':1, 'NA':2}
        # self.sub_to_comp={0:0,1:0,2:1,3:1,4:1,5:1,6:2}
        
        print(f'Loading data from: {self.cell_line_path}')
        self._load_data()
        print(f'There are {len(self.tracks)} tracks:', self.tracks)
        print(f'Number of chromosomes:{self.Nchr}')
        print('\nNormalizing data...')
        self._normalize_data()
        print('done!')

        #state (input) vector
        xdata = []
        fts_vec = []

        if features is None: features = self.norm_data.keys()
        
        for chrm in range(1,self.Nchr+1):
            chrm_data = []
            for track in self.tracks:
                if track not in features: continue
                row=self.norm_data[track][f'chr{chrm}']
                chrm_data.append(row)
                if chrm==1: fts_vec.append(f'{track}_0')
                for n in range(1,n_neighbor+1):
                    chrm_data.append(np.pad(row[n:],(0,n)))
                    chrm_data.append(np.pad(row[:-n],(n,0)))
                    if chrm==1:
                        fts_vec.append(f'{track}_R{n}')
                        fts_vec.append(f'{track}_L{n}')
            
            chrm_data=np.array(chrm_data).reshape(len(features)*(2*n_neighbor+1),-1)
            xdata.append(chrm_data)
            
        xdata = np.concatenate(xdata,axis=1)
    
        return pd.DataFrame(xdata.T, columns=fts_vec)
        
        # for track in self.tracks:
        #     if track not in features: continue
        #     # print(track)
        #     row=self.norm_data[track]['chr1']
        #     # rowlen=row.shape[0]
        #     # Nchr=len(self.norm_data[track].keys())
        #     for jj in range(2,self.Nchr+1):
        #         chrm='chr{}'.format(jj)
        #         row = np.concatenate((row,self.norm_data[track][chrm]))
        #         # rowlen+=norm_chip[mark][chrm].shape[0]
        #     # row=np.array(row)
        #     xdata.append(row)
        #     fts_vec.append(f'{track}_0')
        #     for n in range(1,n_neighbor+1):
        #         xdata.append(np.pad(row[n:],(0,n)))
        #         fts_vec.append(f'{track}_R{n}')

        #         xdata.append(np.pad(row[:-n],(n,0)))
        #         fts_vec.append(f'{track}_L{n}')
        
        # xdata=np.array(xdata).reshape(len(features)*(2*n_neighbor+1),-1)

        # return [xdata, np.array(fts_vec)]

    # def get_Ydata(self, typespath):
    #     # typepath='../data/GM12878_hg19_chip_seq/types/'
    #     ydata=np.loadtxt(os.path.join(typespath,'chr1_beads.txt.original'), dtype=str, usecols=1)
    #     for chrm in range(2,self.Nchr+1):
    #         ydata=np.concatenate((ydata, np.loadtxt(os.path.join(typespath,f'chr{chrm}_beads.txt.original'), dtype=str, usecols=1)))
    #     return ydata

    def get_training_data(self, n_neighbor=2, features=None, ):
        assert self.cell_line=='GM12878', "Training is only done on GM12878."
        
        df_train = self.get_input_vec(n_neighbor, features)
        labels = self.get_training_labels()
        
        assert df_train.shape[0]==labels.shape[0], 'labels and input data of different dimensions!!'

        df_train_filtered = df_train[(labels!='NA') & (labels!='B4')]
        labels_filtered = labels[(labels!='NA') & (labels!='B4')]
        labels_filtered_num = np.array(list(map(self.type_to_int.get, labels_filtered)), dtype=float)
        # df_train_filtered = df_train[(df_train['type']!='NA') | (df_train['type']!='B4')]
        # df_train_filtered['labels'] = np.array(list(map(self.type_to_int.get, df_train_filtered['type'])), dtype=float)
        # df_train_filtered.drop(['type'], axis=1)

        return df_train_filtered, labels_filtered_num

        # try:
        #     ytrain=self.get_training_labels()
        #     xtrain, fts_vec = self.get_input_vec(n_neighbor, features)
        #     ret_vec=np.vstack((ytrain, xtrain))

        #     #remove 'NA' and 'B4' from predictions
        #     indices_NA=np.where(ret_vec[0]=='NA')[0]
        #     indices_B4=np.where(ret_vec[0]=='B4')[0]
        #     indices_to_remove=np.concatenate((indices_NA, indices_B4))
        #     ret_vec = np.delete(ret_vec, indices_to_remove, axis=1)
            
        #     xtrain=np.array(ret_vec[1:].T, dtype=float)
        #     ytrain=np.array(list(map(self.type_to_int.get, ret_vec[0])), dtype=float)
        
        # except (OSError,):
        #     print('No labels found! only generating Xdata.')
        #     xtrain, fts_vec = self.get_Xdata(n_neighbor, features)
        #     xtrain=xtrain.T
        #     ytrain=np.array([None for _ in range(xtrain.shape[0])])

        # return [xtrain, ytrain, fts_vec]


# class preprocess():

#     def __init__(self,data_path):
#         #set the path to the data for analysis
#         self.path=data_path
#         if len(data_path.split('/')[-1])>0: name=data_path.split('/')[-1]
#         else: name=data_path.split('/')[-2]
#         self.name=name

#         #list of experiment tracks
#         #combine plus/minus RNA seq signals
#         self.tracks=np.unique([xx.split('-')[0] for xx in 
#                         np.loadtxt(os.path.join(self.path, 'unique_exp.txt'), dtype=str)])

#         # create dict of dict to store data for tracks and chromosomes
#         self.avg_data={f'{xx}':{} for xx in self.tracks}

#         self.type_to_int={'A1':0, 'A2':1, 'B1':2, 'B2':3, 'B3':4, 'B4':5, 'NA':6}
#         self.int_to_type={0:'A1', 1:'A2', 2:'B1', 3:'B2', 4:'B3', 5:'B4', 6:'NA'}
#         # self.type_to_int_AB={'A1':0, 'A2':0, 'B1':1, 'B2':1, 'B3':1, 'B4':1, 'NA':2}
#         # self.sub_to_comp={0:0,1:0,2:1,3:1,4:1,5:1,6:2}
        
#         print(f'Loading data from: {data_path}')
#         self._load_data()
#         print(f'There are {len(self.tracks)} tracks:', self.tracks)
#         print(f'Number of chromosomes:{self.Nchr}')
#         print('\nNormalizing data...')
#         self._normalize_data()

#         print('done!')

#     def _load_data(self,):
#         # load each track from data folder
#         for track in self.tracks:
#             track_replicas=[xx for xx in os.listdir(self.path) if track in xx]
            
#             # average replicas
#             for replica in track_replicas:
#                 for fname in os.listdir(os.path.join(self.path,replica)):

#                     # ignore chr X
#                     if '.track' not in fname or 'chrX' in fname: continue
#                     chrm=fname.replace('.track','')
                    
#                     # if avg_data has the track and chromosome key store the value 
#                     # else create the keys
#                     if chrm in self.avg_data[track].keys():
#                         self.avg_data[track][chrm]=np.vstack((self.avg_data[track][chrm], np.loadtxt(os.path.join(self.path,replica,fname), skiprows=3,usecols=2)))
#                     elif chrm not in self.avg_data[track].keys():
#                         self.avg_data[track][chrm]=np.loadtxt(os.path.join(self.path,replica,fname), skiprows=3,usecols=2)

#         self.Nchr=len(self.avg_data[track].keys())
        
#     def _normalize_data(self, p_cut=90):
#         # average the replica tracks then normalize them to be between 0 and 1
#         self.norm_data=copy.deepcopy(self.avg_data)
#         for track in self.avg_data.keys():
            
#             for chrm in self.avg_data[track].keys():
                
#                 # average when more than one replica
#                 if len(self.norm_data[track][chrm].shape)>1:
#                     self.norm_data[track][chrm] = np.mean(self.avg_data[track][chrm], axis=0)
                
#                 # shift the baseline to the minumum value
#                 self.norm_data[track][chrm] -= self.norm_data[track][chrm].min()

#                 # linear transform the data by dividing by p_cut percentile
#                 percentile_cutoff = np.percentile(self.norm_data[track][chrm], p_cut)
#                 self.norm_data[track][chrm] /= percentile_cutoff
                
#                 # saturate the high values at p_cut percentile
#                 self.norm_data[track][chrm][self.norm_data[track][chrm]>1]=1.0

#     def get_features(self, ):
#         return list(self.norm_data.keys())

#     def get_Xdata(self, n_neighbor=2, features=None):
#         #state (input) vector
#         xdata = []
#         fts_vec = []
#         if features is None: features = self.norm_data.keys()

#         for chrm in range(1,self.Nchr+1):
#             chrm_data = []
#             for track in self.tracks:
#                 if track not in features: continue
#                 row=self.norm_data[track][f'chr{chrm}']
#                 chrm_data.append(row)
#                 if chrm==1: fts_vec.append(f'{track}_0')
#                 for n in range(1,n_neighbor+1):
#                     chrm_data.append(np.pad(row[n:],(0,n)))
#                     chrm_data.append(np.pad(row[:-n],(n,0)))
#                     if chrm==1:
#                         fts_vec.append(f'{track}_R{n}')
#                         fts_vec.append(f'{track}_L{n}')
            
#             chrm_data=np.array(chrm_data).reshape(len(features)*(2*n_neighbor+1),-1)
#             xdata.append(chrm_data)
#             # rowlen+=len(row)
#         xdata = np.concatenate(xdata,axis=1)
        
#         # for track in self.tracks:
#         #     if track not in features: continue
#         #     # print(track)
#         #     row=self.norm_data[track]['chr1']
#         #     # rowlen=row.shape[0]
#         #     # Nchr=len(self.norm_data[track].keys())
#         #     for jj in range(2,self.Nchr+1):
#         #         chrm='chr{}'.format(jj)
#         #         row = np.concatenate((row,self.norm_data[track][chrm]))
#         #         # rowlen+=norm_chip[mark][chrm].shape[0]
#         #     # row=np.array(row)
#         #     xdata.append(row)
#         #     fts_vec.append(f'{track}_0')
#         #     for n in range(1,n_neighbor+1):
#         #         xdata.append(np.pad(row[n:],(0,n)))
#         #         fts_vec.append(f'{track}_R{n}')

#         #         xdata.append(np.pad(row[:-n],(n,0)))
#         #         fts_vec.append(f'{track}_L{n}')
        
#         # xdata=np.array(xdata).reshape(len(features)*(2*n_neighbor+1),-1)

#         return [xdata, np.array(fts_vec)]

#     def get_Ydata(self, typespath):
#         # typepath='../data/GM12878_hg19_chip_seq/types/'
#         ydata=np.loadtxt(os.path.join(typespath,'chr1_beads.txt.original'), dtype=str, usecols=1)
#         for chrm in range(2,self.Nchr+1):
#             ydata=np.concatenate((ydata, np.loadtxt(os.path.join(typespath,f'chr{chrm}_beads.txt.original'), dtype=str, usecols=1)))
#         return ydata

#     def get_training_data(self, n_neighbor=2, features=None, typespath=None):
#         if typespath is None: typespath = os.path.join('ENCODE_data/types',self.name)
#         xtrain, fts_vec = self.get_Xdata(n_neighbor, features)
#         ytrain=self.get_Ydata(typespath)
#         ret_vec=np.vstack((ytrain, xtrain))

#         #remove 'NA' and 'B4' from predictions
#         indices_NA=np.where(ret_vec[0]=='NA')[0]
#         indices_B4=np.where(ret_vec[0]=='B4')[0]
#         indices_to_remove=np.concatenate((indices_NA, indices_B4))
#         ret_vec = np.delete(ret_vec, indices_to_remove, axis=1)
        
#         xtrain=np.array(ret_vec[1:].T, dtype=float)
#         ytrain=np.array(list(map(self.type_to_int.get, ret_vec[0])), dtype=float)
#         return [xtrain, ytrain, fts_vec]

# # def run_preprocessing(self, path=self.path, n_neighbor=2):
# #     pre = preprocess(self.path)
# #     print('Get training data ...')
# #     x_pre, y_pre, features = pre.get_training_data(n_neighbor=n_neighbor, features=pre.get_features(), typespath=None)

# #     df = pd.DataFrame(np.concatenate((x_pre,y_pre.reshape(-1,1)), axis=1), 
# #                     columns=np.concatenate((features,['labels'])))
# #     print('Save training data ...')
# #     df.to_csv('ENCODE_data/processed_data/GM12878_hg19.csv.zip', compression='zip', index=False)
# #     print('done!')

# #     return df




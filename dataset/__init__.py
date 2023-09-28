from __future__ import print_function, division
import os
import torch

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils


from os.path import join 
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage.interpolation import zoom

#tqdm 'wraps' around an iterable and shows a visible progress bar
from tqdm import tqdm
#ipdb is an interactive python debugger
from ipdb import set_trace as bp

'''Dataset loader for EEG_Encoding_SHHS2 dataset'''
class EEG_Encoding_SHHS2_Dataset(Dataset):

    '''
    Initializes the dataset
    Inputs: self, args, encoding_path, include_shhs1   
    Output: N/A
    '''
    def __init__(self, args, 
    encoding_path="/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/shhs2_new/abdominal_c4_m1", 
    incude_shhs1=True):
        
        self.no_attention = args.no_attention
        self.args = args
        self.file = "/data/netmit/wifall/ADetect/data/csv/shhs2-dataset-augmented.csv" # label file
        self.file_shhs1 = "/data/netmit/wifall/ADetect/data/csv/shhs1-dataset-augmented.csv"
        self.label = args.label
        self.control = args.control
        self.tca = args.tca
        self.ntca = args.ntca
        if args.PCA_embedding:
            encoding_path="/data/scratch-oc40/lth/mage-br-eeg-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug_ali_pca_32/shhs2_new/"
        elif args.stage_input:
            encoding_path ="/data/netmit/wifall/ADetect/data/shhs2_new/stage/"
        self.encoding_path = encoding_path
        self.encoding_path_shhs1 = "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/shhs1_new/abdominal_c4_m1"
        self.num_classes = args.num_classes
        self.all_shhs2_encodings = os.listdir(self.encoding_path)
        self.all_shhs1_encodings = os.listdir(self.encoding_path_shhs1)

        if self.label == "antidep" or self.label == "nsrrid": #FIXME: could add second label for nsrrid w/ cli, or nah
            self.data_dict = self.parse_shhs2_antidep()
            if incude_shhs1:
                self.data_dict.update(self.parse_shhs1_antidep())

        elif self.label == "dep":
            self.data_dict = self.parse_shhs2_sf36()
        elif self.label == "benzo":
            self.data_dict = self.parse_shhs2_benzos()
            if incude_shhs1:
                self.data_dict.update(self.parse_shhs1_benzos())
        elif self.label == "betablocker":
            self.data_dict = self.parse_shhs2_beta_blockers()
        elif self.label == "ace":
            self.data_dict = self.parse_shhs2_ace()
        elif self.label == "thyroid":
            self.data_dict = self.parse_shhs2_thyroid()
        elif self.label == "calciumblocker":
            self.data_dict = self.parse_shhs2_calcium_blockers()
        self.all_valid_files = list(self.data_dict.keys())    


    '''
    Parsing functions for EEG_Encoding_SHHS2_Dataset
    '''
    def parse_shhs2_antidep(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['nsrrid', 'tca2', 'ntca2']]
        df = df.dropna()
        output = {f"shhs2-{k}.npz": [v1, v2] for k, v1, v2 in zip(df['nsrrid'], df['tca2'], df['ntca2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings}
        return output

    def parse_shhs1_antidep(self):
        df = pd.read_csv(self.file_shhs1, encoding='mac_roman')
        df = df[['nsrrid', 'tca1', 'ntca1']]
        df = df.dropna()
        output = {f"shhs1-{k}.npz": [v1, v2] for k, v1, v2 in zip(df['nsrrid'], df['tca1'], df['ntca1']) if f"shhs1-{k}.npz" in self.all_shhs1_encodings}
        output = {key:output[key] for key in output if sum(output[key]) == 1}
        return output
    def parse_shhs1_benzos(self):
        df = pd.read_csv(self.file_shhs1, encoding='mac_roman')
        df = df[['nsrrid', 'benzod1']]
        df = df.dropna()
        output = {f"shhs1-{k}.npz": v for k, v in zip(df['nsrrid'], df['benzod1']) if f"shhs1-{k}.npz" in self.all_shhs1_encodings}
        output = {key:output[key] for key in output if output[key] == 1}
        return output
    
    def parse_shhs2_benzos(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['nsrrid', 'benzod2']]
        df = df.dropna()
        output = {f"shhs2-{k}.npz": v for k, v in zip(df['nsrrid'], df['benzod2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings}
        return output
    
    def parse_shhs2_beta_blockers(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['nsrrid', 'beta2']] # w/out diuretics
        df = df.dropna()
        output = {f"shhs2-{k}.npz": v for k, v in zip(df['nsrrid'], df['beta2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings}
        return output
    
    def parse_shhs2_ace(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['nsrrid', 'ace2']] # w/out diuretics
        df = df.dropna()
        output = {f"shhs2-{k}.npz": v for k, v in zip(df['nsrrid'], df['ace2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings}
        return output

    def parse_shhs2_thyroid(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['nsrrid', 'thry2']]
        df =  df.dropna()
        output = {f"shhs2-{k}.npz": v for k, v in zip(df['nsrrid'], df['thry2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings}
        return output
    
    def parse_shhs2_calcium_blockers(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['nsrrid', 'ccb2']]
        df =  df.dropna()
        output = {f"shhs2-{k}.npz": v for k, v in zip(df['nsrrid'], df['ccb2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings}
        return output
     
    def parse_shhs2_sf36(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['nsrrid', 'ql209a', 'ql209b', 'ql209c', 'ql209d', 'ql209e', 'ql209f', 'ql209g', 'ql209h', 'ql209i', 'tca2', 'ntca2']] # all sf-36 questions and total raw score
        df = df.dropna()

        df['label'] = 3*self.threshold_func(df['ql209c'], 4, 'less') + 2*self.threshold_func(df['ql209g'], 3, 'less') \
                    + self.threshold_func(df['ql209i'], 3, 'less') + self.threshold_func(df['ql209e'], 5, 'greater') \
                    + self.threshold_func(df['ql209f'], 4, 'less') + self.threshold_func(df['ql209d'], 5, 'greater') \
                    + self.threshold_func(df['ql209h'], 4, 'greater')
    
        if(self.control):
            output = {f"shhs2-{k}.npz": v for k, v, on_tca, on_ntca in zip(df['nsrrid'], df['label'], df['tca2'], df['ntca2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings and not (on_tca or on_ntca)}
            #bp()
        elif(self.tca):
            output = {f"shhs2-{k}.npz": v for k, v, on_tca in zip(df['nsrrid'], df['label'], df['tca2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings and on_tca}
        elif(self.ntca):
            output = {f"shhs2-{k}.npz": v for k, v, on_ntca in zip(df['nsrrid'], df['label'], df['ntca2']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings and on_ntca}
        else:
            output = {f"shhs2-{k}.npz": v for k, v in zip(df['nsrrid'], df['label']) if f"shhs2-{k}.npz" in self.all_shhs2_encodings}
        return output

    '''
    Returns the length of the dataset
    '''
    def __len__(self):
        return len(self.data_dict)

    '''
    Returns label 
    '''
    def get_label(self, filename):
        if(self.label == "antidep" and self.num_classes <= 2):
            return torch.tensor((int(self.data_dict[filename][0]) or int(self.data_dict[filename][1])), dtype=torch.int64) # could modify parseantidep to be like parsesf36 and avoid 'or'
        elif(self.label == "antidep"):
            pass #FIXME:::
        elif(self.label == "nsrrid"):
            return filename
        else:
            return torch.tensor(int(self.data_dict[filename]), dtype=torch.int64)
        
    '''
    Returns item at index idx
    '''
    def __getitem__(self, idx):
        filename = self.all_valid_files[idx]
        if 'shhs2' in filename:
            filepath = os.path.join(self.encoding_path, filename)
        else:
            filepath = os.path.join(self.encoding_path_shhs1, filename)
        x = np.load(filepath)
        x = dict(x)
        
        if self.args.stage_input:
            self.input_size = 2 * 60 * 10
            feature = x['data']
            stages_fs = x['fs']
            factor = round(stages_fs * 30)
            feature = feature[::factor]
            feature = process_stages(feature)
            
            if len(feature) > self.input_size:
                feature = feature[:self.input_size]
            else:
                feature = np.concatenate((feature, np.zeros((self.input_size-len(feature)),dtype=int)), axis=0)
        else:
            if not self.no_attention:
                feature = x['decoder_eeg_latent'].squeeze(0)
                if feature.shape[0] >= 150:
                    feature = feature[:150, :]
                else:
                    feature = np.concatenate((feature, np.zeros((150-feature.shape[0],feature.shape[-1]),dtype=np.float32)), axis=0)
            else:
                feature = x['decoder_eeg_latent'].mean(1).squeeze(0) # it gives you 768 dim feature for each night

        label = self.get_label(filename)

        feature = torch.from_numpy(feature)

        return feature, label

    '''
    Applies thresholding to a column of data
    Inputs: self, column, threshold, comparison
    Outputs
    '''
    def threshold_func(self, column, threshold, comparison):
        if comparison=="less":
            return column.apply(lambda x: threshold-x+1 if x <= threshold else 0)
        elif comparison=="greater":
            return column.apply(lambda x: x-threshold+1 if x >= threshold else 0)

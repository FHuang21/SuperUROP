from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import argparse
from os.path import join
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm 
from ipdb import set_trace as bp 
import datetime 
from torch.utils.data import random_split

class DatasetCombiner(Dataset):
    def __init__(self, datasets, phase):
        self.datasets = []
        self.datasets_len = []
        for dataset in datasets:
            generator1 = torch.Generator().manual_seed(42)
            trainset, valset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))], generator=generator1)
            
            if phase == 'train':
                self.datasets.append(trainset)
                self.datasets_len.append(len(trainset))
            else:
                self.datasets.append(valset)
                self.datasets_len.append(len(valset))
            
    def __getitem__(self, idx):
        tempidx = idx
        i = 0
        while True:
            if tempidx - self.datasets_len[i] < 0:
                break 
            else:
                tempidx = tempidx - self.datasets_len[i]
                i += 1
        return self.datasets[i][tempidx]
    def __len__(self):
        return sum(self.datasets_len)
            
class EEG_MGH_Dataset(Dataset):
    def __init__(self, args, transform=None):
        data_path = '/data/netmit/wifall/ADetect/data/'
        self.mgh_label_path = os.path.join(data_path, "csv/mgh-dataset-augmented.csv")
        if args.data_source == 'eeg':
            self.root_dir = os.path.join(data_path, 'mgh_new/c4_m1') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz
        elif args.data_source == 'bb':
            self.root_dir = os.path.join(data_path, 'mgh_new/thorax') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 5 #hours to 5hz
        elif args.data_source == 'stage':
            self.root_dir = os.path.join(data_path, 'mgh_new/stage') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = int(12 * 60 * 60 * (1/30)) #hours to 5hz
        self.data_source = args.data_source
        BAD_AGE = [141, 264, 266, 270, 433, 458, 544, 549, 551, 581, 655, 667, 676, 677, 768, 773, 197]
        
        df = pd.read_csv(self.mgh_label_path)
        # age labels
        all_ages = list(zip(df['SubjectId'],df['age']))
        self.ages = {}
        for data in all_ages:
            if np.isnan(data[1]):
                continue 
            
            self.ages[int(data[0])] = data[1]
        # antidep labels
        all_antideps = list(zip(df['SubjectId'],df['dx_elix_depre']))
        self.antideps = {}
        for data in all_antideps:
            if np.isnan(data[1]):
                continue 
            
            self.antideps[int(data[0])] = data[1]
        
        all_valid_files = []
        for file in os.listdir(self.root_dir):
            subj_id = int(file.split('-')[-1].split('.')[0])
            if subj_id not in BAD_AGE:
                all_valid_files.append(file)
        self.all_valid_files = all_valid_files
        self.total_length = len(self.all_valid_files)
    def get_label(self, filename):
        subj_id = int(filename.split('-')[-1].split('.')[0])
        # return self.ages[subj_id]
        return self.antideps[subj_id]
    def __getitem__(self, idx):
        filename = self.all_valid_files[idx]
        # data = np.load(os.path.join(self.root_dir, filename)).get('data')
        datafile = np.load(os.path.join(self.root_dir, filename))
        data = datafile.get('data')
        fs = datafile.get('fs').item()
        if self.data_source == 'stage' and fs != 1/30:
            data = data[::int(fs * 30)]
        sample = np.zeros(self.INPUT_SIZE)
        
        if len(data) < self.INPUT_SIZE:
            sample[:len(data)] = data
        else:
            sample[:] = data[:self.INPUT_SIZE]
        
        output = torch.tensor(sample, dtype=torch.float)
        output = output.unsqueeze(0)
        label = torch.tensor(self.get_label(filename))
        
        if False:
            sample_stages = np.zeros(self.INPUT_SIZE//(64 * 30))
            file = np.load(os.path.join(self.stage_dir, filename))
            stages_raw = file.get('data')
            stages_fs = file['fs']
            factor = round(stages_fs * 30)
            stages_raw = stages_raw[::factor]
            
            stages_raw = self.process_stages(stages_raw)
            if len(stages_raw) < self.INPUT_SIZE / (64 * 30):
                sample_stages[:len(stages_raw)] = stages_raw
            else:
                sample_stages[:] = stages_raw[:(self.INPUT_SIZE//(64 * 30))]
                
            stages = torch.tensor(sample_stages).type(torch.LongTensor)
            return output, stages, label
        else:
            return output, label.float()
    def __len__(self):
        return self.total_length

class EEG_SOF_Dataset(Dataset):
    def __init__(self, args, transform=None, include_labels=['pd','ad','nm']):
        data_path = '/data/netmit/wifall/ADetect/data/'
        
        self.label_path = '/data/netmit/RadarFS/SleepProject/Dataset/Raw/sof/raw/datasets/sof-visit-8-dataset-0.5.0.csv'

        if args.data_source == 'eeg':
            self.root_dir = os.path.join(data_path, 'sof/c4_m1') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz
        elif args.data_source == 'bb':
            self.root_dir = os.path.join(data_path, 'sof/thorax') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 5 #hours to 5hz
        elif args.data_source == 'stage':
            self.root_dir = os.path.join(data_path, 'sof/stage') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = int(12 * 60 * 60 * (1/30)) #hours to 5hz
        self.data_source = args.data_source
        PARKINSONS_SOF = {405, 2648, 3589, 5674, 5782, 6897, 8810, 5194, 5841}
        
        ALZHEIMER_SOF = {405, 445, 1342, 1404, 1525, 1546, 1632, 1715, 1960, 2027, 2077, 3439, 3660, 3788, 4428, 4442, 4905, 5015, 5449, 5674, 5741, 6595, 7354, 7441, 7459, 8055, 8250, 8515, 8699}
        
        # all_files = os.listdir(self.root_dir)
        self.ages = {}
        df = pd.read_csv(self.label_path)
        all_ages = list(zip(df['sofid'],df['V8AGE']))
        for label in all_ages:
            sofid = label[0]
            sofid = "00000" + str(sofid)
            sofid = sofid[-5:]
            age = label[1]
            if np.isnan(age):
                continue
            self.ages['sof-visit-8-{0}.npz'.format(sofid)] = age

        self.is_pd = ['sof-visit-8-{0}.npz'.format(("00000"+str(item))[-5:]) for item in PARKINSONS_SOF]
        self.is_ad = ['sof-visit-8-{0}.npz'.format(("00000"+str(item))[-5:]) for item in ALZHEIMER_SOF]

        self.include_labels = include_labels
        self.labels = {}
        for item in os.listdir(self.root_dir):
            if item in self.is_ad and item in self.ages:
                self.labels[item] = 'ad'
            elif item in self.is_pd and item in self.ages:
                self.labels[item] = 'pd'
            elif item in self.ages:
                self.labels[item] = 'nm'
        
        self.all_valid_files = list(self.labels.keys())
        self.total_length = len(self.all_valid_files)

    def __getitem__(self, idx):
        filename = self.all_valid_files[idx]
        
        # data = np.load(os.path.join(self.root_dir, filename)).get('data')
        datafile = np.load(os.path.join(self.root_dir, filename))
        data = datafile.get('data')
        fs = datafile.get('fs').item()
        if self.data_source == 'stage' and fs != 1/30:
            data = data[::int(fs * 30)]
        # else:
        #     data = np.load(os.path.join(self.shhs1_root_dir, filename)).get('data')
        
        sample = np.zeros(self.INPUT_SIZE)
        
        if len(data) < self.INPUT_SIZE:
            sample[:len(data)] = data
        else:
            sample[:] = data[:self.INPUT_SIZE]
        
        output = torch.tensor(sample, dtype=torch.float)
        output = output.unsqueeze(0)
        
        label = torch.tensor(self.ages[filename])
        
        if False:
            sample_stages = np.zeros(self.INPUT_SIZE//(64 * 30))
            file = np.load(os.path.join(self.stage_dir, filename))
            stages_raw = file.get('data')
            stages_fs = file['fs']
            factor = round(stages_fs * 30)
            stages_raw = stages_raw[::factor]
            
            stages_raw = self.process_stages(stages_raw)
            if len(stages_raw) < self.INPUT_SIZE / (64 * 30):
                sample_stages[:len(stages_raw)] = stages_raw
            else:
                sample_stages[:] = stages_raw[:(self.INPUT_SIZE//(64 * 30))]
                
            stages = torch.tensor(sample_stages).type(torch.LongTensor)
            return output, stages, label
        else:
            return output, label.float()
    def __len__(self):
        return self.total_length
    

class EEG_MrOS_Dataset(Dataset):
    def __init__(self, args, transform=None, include_labels=['pd','ad','nm']):
        data_path = '/data/netmit/wifall/ADetect/data/'
        
        self.label_path = os.path.join(data_path, "csv/mros1-dataset-augmented.csv")

        if args.data_source == 'eeg':
            self.root_dir = os.path.join(data_path, 'mros1_new/c4_m1') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz
        elif args.data_source == 'bb':
            self.root_dir = os.path.join(data_path, 'mros1_new/thorax') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 5 #hours to 5hz
        elif args.data_source == 'stage':
            self.root_dir = os.path.join(data_path, 'mros1_new/stage') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = int(12 * 60 * 60 * (1/30)) #hours to 5hz
        self.data_source = args.data_source
        df = pd.read_csv(self.label_path)
        
        
        ages = list(zip(df['nsrrid'],df['age']))
        self.ages = {}
        for item in ages:
            k,v = item 
            self.ages['mros-visit1-{0}.npz'.format(k.lower())] = v 
        
        ALZHEIMER_MROS1 = {
            '10153', '10169', '10251', '10498', '10557', '10598', '10647', '10860', '11105', '11217', '11439', '11688', '12005',
            '12576', '12686', '12697', '12762', '12765', '12914', '12956', '12983', '13099', '13196', '13370', '13410', '13477',
            '13483', '13600', '13656', '13695', '13726', '13823', '13975', '14016', '14022', '14242', '14294', '14329', '14711',
            '14776', '15460', '15535', '15565', '15574', '15610', '15653', '15749', '15809', '15891', '15907', '15926'
        }
        # ALZHEIMER_MROS2 = {
        #     '20085', '20334', '20574', '20633', '20948', '20980', '21177', '21354', '21403', '21482', '21528', '21567', '21682',
        #     '21747', '21801', '22167', '22237', '22407', '22706', '22743', '22826', '23363', '23372', '23500', '23975', '24329',
        #     '24637', '24687', '25228', '25240', '25345', '25411', '25544', '25926', '25981'
        # }
        PARKINSON_MROS1 = {
            '10169', '10275', '10286', '10451', '10707', '10948', '10959', '10982', '11128', '11263', '11383', '11492',
            '11725', '11944', '12268', '12686', '12697', '12718', '12738', '12983', '13170', '13254', '13290', '13438',
            '13530', '13965', '14464', '14572', '15101', '15241', '15345', '15548', '15669', '15842', '15843', '15955'
        }
        # PARKINSON_MROS2 = {
        #     '20216', '20718', '20948', '21486', '22035', '22167', '23254', '23397', '23438', '23661', '24587', '24723',
        #     '25345', '25393', '25411', '25797'
        # }
        
        self.is_pd = ['mros-visit1-aa{0}.npz'.format(item[1:]) for item in PARKINSON_MROS1]
        self.is_ad = ['mros-visit1-aa{0}.npz'.format(item[1:]) for item in ALZHEIMER_MROS1]
        
        self.include_labels = include_labels
        self.labels = {}
        for item in os.listdir(self.root_dir):
            if 'ad' in self.include_labels and item in self.is_ad:
                self.labels[item] = 'ad'
            elif 'pd' in self.include_labels and item in self.is_pd:
                self.labels[item] = 'pd'
            elif 'nm' in self.include_labels:
                self.labels[item] = 'nm'
        self.all_valid_files = list(self.labels.keys())
        self.total_length = len(self.all_valid_files)

    def __getitem__(self, idx):
        filename = self.all_valid_files[idx]
        
        datafile = np.load(os.path.join(self.root_dir, filename))
        data = datafile.get('data')
        fs = datafile.get('fs').item()
        if self.data_source == 'stage' and fs != 1/30:
            data = data[::int(fs * 30)]
        # else:
        #     data = np.load(os.path.join(self.shhs1_root_dir, filename)).get('data')
        
        sample = np.zeros(self.INPUT_SIZE)
        
        if len(data) < self.INPUT_SIZE:
            sample[:len(data)] = data
        else:
            sample[:] = data[:self.INPUT_SIZE]
        
        output = torch.tensor(sample, dtype=torch.float)
        output = output.unsqueeze(0)
        label = torch.tensor(self.ages[filename])
        
        if False:
            sample_stages = np.zeros(self.INPUT_SIZE//(64 * 30))
            file = np.load(os.path.join(self.stage_dir, filename))
            stages_raw = file.get('data')
            stages_fs = file['fs']
            factor = round(stages_fs * 30)
            stages_raw = stages_raw[::factor]
            
            stages_raw = self.process_stages(stages_raw)
            if len(stages_raw) < self.INPUT_SIZE / (64 * 30):
                sample_stages[:len(stages_raw)] = stages_raw
            else:
                sample_stages[:] = stages_raw[:(self.INPUT_SIZE//(64 * 30))]
                
            stages = torch.tensor(sample_stages).type(torch.LongTensor)
            return output, stages, label
        else:
            return output, label.float()
    def __len__(self):
        return self.total_length
    

class EEG_SHHS_Dataset(Dataset):

    def __init__(self, args, target_label="antidep", transform=None, include_stages=False):
        data_path = '/data/netmit/wifall/ADetect/data/'
        self.shhs_label_path = os.path.join(data_path, "shhs2_new/csv") # using for shhs1, do not confuse
        
        if args.data_source == 'eeg':
            self.shhs1_root_dir = os.path.join(data_path, 'shhs1_new/c4_m1') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz
        elif args.data_source == 'bb':
            self.shhs1_root_dir = os.path.join(data_path, 'shhs1_new/thorax') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 5 #hours to 5hz
        elif args.data_source == 'stage':
            self.shhs1_root_dir = os.path.join(data_path, 'shhs1_new/stage') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = int(12 * 60 * 60 * (1/30)) #hours to 5hz
        self.data_source = args.data_source
        
        self.target_label = target_label

        self.include_stages = include_stages
        
        self.dict_hy = {}
        
        ## adding shhs1 here
        self.all_shhs1_files = os.listdir(self.shhs1_root_dir)
        shhs1_dict = self.parse_shhs1_antidep()
        self.dict_hy.update(shhs1_dict)
        self.all_valid_files = list(self.dict_hy.keys())
        
    def parse_shhs1_ages(self, file='/data/netmit/wifall/ADetect/data/shhs2_new/csv/shhs1-dataset-0.14.0.csv'):
        df = pd.read_csv(file, encoding='mac_roman')
        output = {f"shhs1-{k}.npz": float(v) for k, v in zip(df['nsrrid'], df['age_s1'])}
        return output
    def parse_shhs1_antidep(self, file='/data/netmit/wifall/ADetect/data/shhs2_new/csv/shhs1-dataset-0.14.0.csv'):
        # need to parse only the people w/ valid files and who answered antidepressant questions
        df = pd.read_csv(file, encoding='mac_roman')
        #df = df.drop(df[df['shhs1_psg'] == 0].index)
        df = df[['nsrrid', 'TCA1', 'NTCA1']]
        df = df.dropna()
        output = {f"shhs1-{k}.npz": [float(v1), float(v2)] for k, v1, v2 in zip(df['nsrrid'], df['TCA1'], df['NTCA1']) if f"shhs1-{k}.npz" in self.all_shhs1_files}
        return output
    def get_age_from_pid(self, pid_tensor):
        return self.dict_hy['shhs1-{0}.npz'.format(int(pid_tensor.item()))]
    def get_idx_from_pid(self, pid_tensor):
        return self.all_valid_files.index('shhs1-{0}.npz'.format(int(pid_tensor.item())))
    def get_antidep_status_from_pid(self, pid_tensor):
        return self.dict_hy['shhs1-{0}.npz'.format(int(pid_tensor.item()))]
    def get_hy_score(self, filename):
        if filename not in self.dict_hy:
            return -1
        return torch.tensor(self.dict_hy[filename])
        
    def process_stages(self, stages):
        stages[stages < 0] = 0
        stages[stages > 5] = 0
        stages = stages.astype(int)
        mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
        return mapping[stages]
    
    def get_label(self, filename):
        if self.target_label == 'HY':
            return self.get_hy_score(filename)
        elif self.target_label == "PD":
            is_pd = "pd" in filename
            output =  torch.tensor(is_pd)
            return output.type(torch.LongTensor)
        elif self.target_label == "AGE":
            return torch.tensor(self.dict_hy[filename])
        elif self.target_label == "pid":
            return torch.tensor(int(filename.split('-')[1].split('.')[0]))
        elif self.target_label == "antidep":
            return torch.tensor(self.dict_hy[filename][0] or self.dict_hy[filename][1], dtype=torch.int64)#.type(torch.long) # needs to be long type for some thing
        
    def __len__(self):
        return len(self.dict_hy)

    def __getitem__(self, idx):
         
        filename = self.all_valid_files[idx]
        sample = np.zeros(self.INPUT_SIZE)

        datafile = np.load(os.path.join(self.shhs1_root_dir, filename))
        data = datafile.get('data')
        fs = datafile.get('fs').item()

        if self.data_source == 'stage' and fs != 1/30:
            data = data[::int(fs * 30)]


        if len(data) < self.INPUT_SIZE:
            sample[:len(data)] = data
        else:
            sample[:] = data[:self.INPUT_SIZE]
        
        output = torch.tensor(sample, dtype=torch.float32)
        output = output.unsqueeze(0)
        label = self.get_label(filename)
        return output, label

class EEG_SHHS2_Dataset(Dataset):

    def __init__(self, args, target_label="antidep", transform=None, include_stages=False):
        data_path = '/data/netmit/wifall/ADetect/data/'
        self.shhs_label_path = os.path.join(data_path, "shhs2_new/csv") # using for shhs1, do not confuse
        
        if args.data_source == 'eeg':
            self.shhs2_root_dir = os.path.join(data_path, 'shhs2_new/c4_m1') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz
        elif args.data_source == 'bb':
            self.shhs2_root_dir = os.path.join(data_path, 'shhs2_new/thorax') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 5 #hours to 5hz
        elif args.data_source == 'stage':
            self.shhs2_root_dir = os.path.join(data_path, 'shhs2_new/stage') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = int(12 * 60 * 60 * (1/30)) #hours to 5hz
        self.data_source = args.data_source
        
        self.target_label = target_label

        self.include_stages = include_stages
        
        self.dict_hy = {}
        
        
        ## adding shhs2 here
        self.all_shhs2_files = os.listdir(self.shhs2_root_dir)
        shhs2_dict = self.parse_shhs2_antidep()
        self.dict_hy.update(shhs2_dict)
        self.all_valid_files = list(self.dict_hy.keys())
        
    def parse_shhs2_ages(self, file='/data/netmit/wifall/ADetect/data/shhs2_new/csv/shhs2-dataset-0.14.0.csv'):
        df = pd.read_csv(file, encoding='mac_roman')
        output = {f"shhs2-{k}.npz": float(v) for k, v in zip(df['nsrrid'], df['age_s2'])}
        return output
    

    def parse_shhs2_antidep(self, file='/data/netmit/wifall/ADetect/data/shhs2_new/csv/shhs2-dataset-0.14.0.csv'):
        # need to parse only the people w/ valid files and who answered antidepressant questions
        df = pd.read_csv(file, encoding='mac_roman')
        #df = df.drop(df[df['shhs2_psg'] == 0].index)
        df = df[['nsrrid', 'TCA2', 'NTCA2']]
        df = df.dropna(how='any')
        output = {f"shhs2-{k}.npz": [float(v1), float(v2)] for k, v1, v2 in zip(df['nsrrid'], df['TCA2'], df['NTCA2']) if f"shhs2-{k}.npz" in self.all_shhs2_files}
        return output
    def get_age_from_pid(self, pid_tensor):
        return self.dict_hy['shhs2-{0}.npz'.format(int(pid_tensor.item()))]
    def get_idx_from_pid(self, pid_tensor):
        return self.all_valid_files.index('shhs2-{0}.npz'.format(int(pid_tensor.item())))
    def get_antidep_status_from_pid(self, pid_tensor):
        return self.dict_hy['shhs2-{0}.npz'.format(int(pid_tensor.item()))]
    def get_hy_score(self, filename):
        if filename not in self.dict_hy:
            return -1
        return torch.tensor(self.dict_hy[filename])
        
    def process_stages(self, stages):
        stages[stages < 0] = 0
        stages[stages > 5] = 0
        stages = stages.astype(int)
        mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
        return mapping[stages]
    
    def get_label(self, filename):
        if self.target_label == 'HY':
            return self.get_hy_score(filename)
        elif self.target_label == "PD":
            is_pd = "pd" in filename
            output =  torch.tensor(is_pd)
            return output.type(torch.LongTensor)
        elif self.target_label == "AGE":
            return torch.tensor(self.dict_hy[filename])
        elif self.target_label == "pid":
            return torch.tensor(int(filename.split('-')[1].split('.')[0]))
        elif self.target_label == "antidep":
            return torch.tensor(self.dict_hy[filename][0] or self.dict_hy[filename][1], dtype=torch.int64)
        
    def __len__(self):
        return len(self.dict_hy)

    def __getitem__(self, idx):
         
        filename = self.all_valid_files[idx]
        sample = np.zeros(self.INPUT_SIZE)

        datafile = np.load(os.path.join(self.shhs2_root_dir, filename))
        data = datafile.get('data')
        fs = datafile.get('fs').item()

        if self.data_source == 'stage' and fs != 1/30:
            data = data[::int(fs * 30)]


        if len(data) < self.INPUT_SIZE:
            sample[:len(data)] = data
        else:
            sample[:] = data[:self.INPUT_SIZE]
        
        output = torch.tensor(sample, dtype=torch.float32)
        output = output.unsqueeze(0)
        label = self.get_label(filename)
        return output, label#.float()


class EEG_MAYO_Dataset(Dataset):

    def __init__(self, args, include_labels=['nm','pd','ad'], transform=None):
        data_path = '/data/netmit/wifall/ADetect/data/'
        self.mayo_label_path = os.path.join(data_path, "mayo_new/csv")
        # self.root_dir = os.path.join(data_path, 'mayo_new/c4_m1') ## c3_m2, c3_m1_spec,   
        # self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz
        
        if args.data_source == 'eeg':
            self.root_dir = os.path.join(data_path, 'mayo_new/c4_m1') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz
        elif args.data_source == 'bb':
            self.root_dir = os.path.join(data_path, 'mayo_new/thorax') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = 12 * 60 * 60 * 5 #hours to 5hz
        elif args.data_source == 'stage':
            self.root_dir = os.path.join(data_path, 'mayo_new/stage') ## c3_m2, c3_m1_spec,   
            self.INPUT_SIZE = int(12 * 60 * 60 * (1/30)) #hours to 5hz
        self.data_source = args.data_source
        self.stage_dir = os.path.join(data_path, 'mayo_new/stage')
        self.include_labels = include_labels
        self.target_label = 'AGE'
        
        self.MAYO_BAD = {    # 21
            '198', '246', '257', '439', '542', '377', '345', '510', '368', '284', '293', '416', '327', '364', '423', '316',
            '291', '455', '200', '189', '478'
        }
        self.MAYO_SHORT = {  # 42
            '251', '199', '495', '165', '272', '174', '254', '406', '344', '170', '154', '522', '525', '338', '260', '292',
            '276', '341', '310', '473', '408', '441', '178', '366', '329', '428', '446', '434', '171', '523', '311', '322',
            '528', '281', '156', '141', '422', '279', '313', '546', '318', '351'
        }
        
        self.transform = transform
        all_files = os.listdir(self.root_dir)
        
        
        
        # if "target_label=="PD"":
        #     # self.all_valid_files = [item for item in all_files if ("nm" in item or "pd" in item)]
        #     include_mayo_labels = ['pd','nm']
        #     self.total_nm = len([item for item in all_files if "nm" in item])
        #     self.total_pd = len([item for item in all_files if "pd" in item])
        # elif target_label=="HY":
        #     self.dict_hy = self.parse_mayo_hy_scores()
        #     # self.all_valid_files = list(self.dict_hy.keys())
        # elif target_label=="AGE":
        self.dict_hy = {}
        for mayo_label in include_labels:
            self.dict_hy.update(self.parse_mayo_ages(target=mayo_label))
        
        #self.all_valid_files = [item for item in all_files if "pd" in item]
        self.all_valid_files = list(self.dict_hy.keys())
        
        # self.all_valid_files = all_files
        self.mayo_size = len(self.all_valid_files)
        
    def __len__(self):
        return len(self.all_valid_files)
    def __getitem__(self, idx):
        filename = self.all_valid_files[idx]
        # data = np.load(os.path.join(self.root_dir, filename)).get('data')
        datafile = np.load(os.path.join(self.root_dir, filename))
        data = datafile.get('data')
        fs = datafile.get('fs').item()
        if self.data_source == 'stage' and fs != 1/30:
            data = data[::int(fs * 30)]
            
        sample = np.zeros(self.INPUT_SIZE)
        
        if len(data) < self.INPUT_SIZE:
            sample[:len(data)] = data
        else:
            sample[:] = data[:self.INPUT_SIZE]
        
        output = torch.tensor(sample, dtype=torch.long)
        output = output.unsqueeze(0)
        label = self.get_label(filename)
        
        return output, label.float()
        # if self.include_stages:
        #     sample_stages = np.zeros(self.INPUT_SIZE//(64 * 30))
        #     file = np.load(os.path.join(self.stage_dir, filename))
        #     stages_raw = file.get('data')
        #     stages_fs = file['fs']
        #     factor = round(stages_fs * 30)
        #     stages_raw = stages_raw[::factor]
            
        #     stages_raw = self.process_stages(stages_raw)
        #     if len(stages_raw) < self.INPUT_SIZE / (64 * 30):
        #         sample_stages[:len(stages_raw)] = stages_raw
        #     else:
        #         sample_stages[:] = stages_raw[:(self.INPUT_SIZE//(64 * 30))]
                
        #     stages = torch.tensor(sample_stages).type(torch.LongTensor)
        #     return output, stages, label
        # else:
        #     return output, label


    def parse_mayo_ages(self, file='/data/netmit/wifall/ADetect/data/mayo_new/csv/AD-PD De-identified updated SLG 06-25-20.xlsx', show=False, exist=True, exclude=True, target='pd'):
        df = pd.DataFrame(pd.read_excel(file, header=0, sheet_name='PD'))
        dict_hy = {f"mayo{target}-{k[-4:]}.npz": float(v) for k, v in zip(df['Study ID'], df['Age at PSG'])
                if isinstance(v, (int, float))}
        dict_hy = {k: dict_hy[k] for k in dict_hy if not np.isnan(dict_hy[k])}
        # filenames = [name.split('.')[0] for name in os.listdir('/home/yuzhe/remote2/data/learning_full/prknsn_mayo/thorax')]
        if exclude:
            dict_hy = {k: dict_hy[k] for k in dict_hy if k.split('-')[-1] not in self.MAYO_BAD.union(self.MAYO_SHORT)}
        
        print(len(dict_hy))
        dict_hy = {k: dict_hy[k] for k in dict_hy if k in os.listdir(self.root_dir)}
        print(len(dict_hy))
        return dict_hy

        
    def parse_mayo_hy_scores(self, file='/data/netmit/wifall/ADetect/data/mayo_new/csv/AD-PD De-identified updated SLG 06-25-20.xlsx', show=False, exist=True, exclude=True):
        df = pd.DataFrame(pd.read_excel(file, header=0, sheet_name='PD'))
        dict_hy = {f"mayopd-{k[-4:]}.npz": float(v) for k, v in zip(df['Study ID'], df['H/Y Score'])
                if isinstance(v, (int, float))}
        dict_hy = {k: dict_hy[k] for k in dict_hy if not np.isnan(dict_hy[k])}
        # filenames = [name.split('.')[0] for name in os.listdir('/home/yuzhe/remote2/data/learning_full/prknsn_mayo/thorax')]
        if exclude:
            dict_hy = {k: dict_hy[k] for k in dict_hy if k.split('-')[-1] not in self.MAYO_BAD.union(self.MAYO_SHORT)}
        
        print(len(dict_hy))
        dict_hy = {k: dict_hy[k] for k in dict_hy if k in os.listdir(self.root_dir)}
        print(len(dict_hy))

        self.dict_hy = dict_hy
        
    def get_label(self, filename):
        if self.target_label == 'HY':
            return self.get_hy_score(filename)
        elif self.target_label == "PD":
            is_pd = "pd" in filename
            # if is_pd:
                # return torch.tensor([0,1],dtype=torch.float)
            # else:
                # return torch.tensor([1,0],dtype=torch.float)
            output =  torch.tensor(is_pd)
            return output.type(torch.LongTensor)
        elif self.target_label == "AGE":
            return torch.tensor(self.dict_hy[filename])
        
        
if __name__ == '__main__':
    class Object(object):
        pass
    args = Object
    args.data_source = 'eeg'
    dataset = EEG_SHHS2_Dataset(args)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    aa = iter(dataloader)
    for i in range(10):
        t = datetime.datetime.now()
        data, label = next(aa)
        bp()
        print(data.shape, label)
        print(datetime.datetime.now() - t)
    bp()  

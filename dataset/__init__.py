from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#import argparse
from os.path import join
from scipy.stats import pearsonr, spearmanr
#import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm 
from ipdb import set_trace as bp 
#import datetime 
from torch.utils.data import random_split

def process_stages(stages):
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    mapping = np.array([0, 2, 2, 3, 3, 1, 0, 0, 0, 0], int)
    # mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
    return mapping[stages]

class DatasetCombiner(Dataset):
    def __init__(self, datasets, phase):
        self.datasets = []
        self.datasets_len = []
        for dataset in datasets:
            generator1 = torch.Generator().manual_seed(20)
            #trainset, valset = random_split(dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))], generator=generator1)
            trainset, valset = random_split(dataset, [0.7, 0.3], generator=generator1)
            
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

        self.debug = args.debug

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
        df = df[['nsrrid', 'TCA1', 'NTCA1']]
        df = df.dropna()
        if(self.debug == True):
            output = {f"shhs1-{k}.npz": [float(v1), float(v2)] for k, v1, v2 in zip(df['nsrrid'], df['TCA1'], df['NTCA1']) if f"shhs1-{k}.npz" in self.all_shhs1_files[:10]}
            return output
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
            return torch.tensor(self.dict_hy[filename][0] or self.dict_hy[filename][1], dtype=torch.int64)
        
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

        self.debug = args.debug

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
        if(self.debug == True):
            #output = {f"shhs2-{k}.npz": [float(v1), float(v2)] for k, v1, v2 in zip(df['nsrrid'], df['TCA2'], df['NTCA2']) if f"shhs2-{k}.npz" in self.all_shhs2_files[:10]}
            ids = [200077,200078,200079,200080, 200088, 200123, 2000139, 200187, 200093,200154]
            output = {f"shhs2-{k}.npz": [float(v1), float(v2)] for k, v1, v2 in zip(df['nsrrid'], df['TCA2'], df['NTCA2']) if k in ids}
            print(output, '\n\n\n\n')
            return output
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

# class EEG_WSC_Dataset(Dataset):
#     def __init__(self, args, file="/data/netmit/wifall/ADetect/data/csv/shhs2-dataset-augmented.csv",)

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

class EEG_Encoding_SHHS2_Dataset(Dataset):
    def __init__(self, args, 
                 encoding_path="/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/shhs2_new/abdominal_c4_m1", incude_shhs1=True):
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
        #print("output:")
        #print(output)
        return output
    
    def threshold_func(self, column, threshold, comparison):
        if comparison=="less":
            return column.apply(lambda x: threshold-x+1 if x <= threshold else 0)
        elif comparison=="greater":
            return column.apply(lambda x: x-threshold+1 if x >= threshold else 0)

    def get_label(self, filename):
        if(self.label == "antidep" and self.num_classes <= 2):
            return torch.tensor((int(self.data_dict[filename][0]) or int(self.data_dict[filename][1])), dtype=torch.int64) # could modify parseantidep to be like parsesf36 and avoid 'or'
        elif(self.label == "antidep"):
            pass #FIXME:::
        elif(self.label == "nsrrid"):
            return filename
        else:
            return torch.tensor(int(self.data_dict[filename]), dtype=torch.int64)
        
    def get_label_from_filename(self, filename): # FIXME
        # on_tca = self.data_dict[filename][0]
        # on_ntca = self.data_dict[filename][1]
        # # if(on_ntca):
        # #     return 2
        # # elif(on_tca):
        # #     return 1
        # # else:
        # #     return 0
        # return (1 if (on_tca or on_ntca) else 0) #FIXME:::

        ##temp
        if type(self.data_dict[filename])==list: # antidep
            return (1 if (self.data_dict[filename][0]==1 or self.data_dict[filename][1]==1) else 0)
        else:
            return self.data_dict[filename]

    def threshold_values(self): # hella inefficient
        #th = 4
        med_names = ['tca', 'ntca', 'control']
        data = {'pos':[0,0,0], 'neg':[0,0,0]}
        df = pd.DataFrame(data, index=med_names)

        pos_tca = 0
        pos_ntca = 0
        pos_control = 0
        neg_tca = 0
        neg_ntca = 0
        neg_control = 0
        for pid in self.data_dict_hs.keys():
            if pid in self.data_dict.keys():
                is_tca = self.data_dict[pid][0]
                is_ntca = self.data_dict[pid][1]
                is_control = 1 if not (is_tca or is_ntca) else 0
                if self.data_dict_hs[pid] == 1:
                    df.loc['tca','pos'] += is_tca
                    df.loc['ntca','pos'] += is_ntca
                    df.loc['control','pos'] += is_control
                    pos_tca += is_tca
                    pos_ntca += is_ntca
                    pos_control += is_control
                else:
                    df.loc['tca','neg'] += is_tca
                    df.loc['ntca','neg'] += is_ntca
                    df.loc['control','neg'] += is_control
                    neg_tca += is_tca
                    neg_ntca += is_ntca
                    neg_control += is_control

        new_row_values = {
            'pos': (1 - (pos_tca+pos_ntca)/(pos_tca+pos_ntca+pos_control)),
            'neg': (1 - (neg_tca+neg_ntca)/(neg_tca+neg_ntca+neg_control))
        }
        new_row_df = pd.DataFrame([new_row_values])
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        df['% pos'] = df['pos']/(df['pos']+df['neg'])

        return df
        
    def __len__(self):
        return len(self.data_dict)
    
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

class EEG_Encoding_WSC_Dataset(Dataset):
    def __init__(self, args, file="/data/netmit/wifall/ADetect/data/csv/wsc-dataset-augmented.csv", 
                 encoding_path="/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/wsc_new/abdominal_c4_m1"):
        self.no_attention = args.no_attention
        self.file = file # label file ['NIHND126MXDGP', 'NIHBE740TFYAH', 'NIHPT334YGJLK', 'NIHFW795KLATW', 'NIHDW178UFZHB', 'NIHHD991PGRJC']
        self.label = args.label
        self.control = args.control
        self.tca = args.tca
        self.ssri = args.ssri
        self.other = args.other
        self.num_classes = args.num_classes
        if args.PCA_embedding:
            encoding_path="/data/scratch-oc40/lth/mage-br-eeg-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug_ali_pca_32/wsc_new/"
        
        self.encoding_path = encoding_path
        self.all_wsc_encodings = os.listdir(self.encoding_path)

        if self.label == "antidep" or self.label == "nsrrid": # FIXME:::
            self.data_dict = self.parse_wsc_antidep()
        elif self.label == "dep":
            self.data_dict = self.parse_wsc_zung()
        elif self.label == "betablocker":
            self.data_dict = self.parse_wsc_beta_blockers()
        elif self.label == "ace":
            self.data_dict = self.parse_wsc_ace()
        elif self.label == "thyroid":
            self.data_dict = self.parse_wsc_thyroid()
        self.all_valid_files = list(self.data_dict.keys())

        # self.data_dict = self.parse_wsc_antidep() #if (label=='antidep' or label=='nsrrid') else self.parse_wsc_zung() # get dictionary of encodings with associated labels
        # self.data_dict_hs = self.parse_wsc_zung()

        # if(self.label == "dep"):
        #     self.all_valid_files = list(self.data_dict_hs.keys())
        # elif(self.label == "antidep"):
        #     self.all_valid_files = list(self.data_dict.keys())
        # # elif(self.label == "benzo"):
        # #     self.all_valid_files = list(self.data_dict_benzos.keys())
        # elif(self.label == "nsrrid"):
        #     self.all_valid_files = list(self.data_dict.keys())

    def parse_wsc_antidep(self): # label: depression_med
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'depression_med','dep_tca_med','dep_ssri_med']]
        df = df.dropna()
        output = {f"wsc-visit{vst}-{id}-nsrr.npz": [v1,v2,v3] for id, vst, v1, v2, v3 in zip(df['wsc_id'], df['wsc_vst'], df['depression_med'],df['dep_tca_med'], df['dep_ssri_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output
    
    def parse_wsc_zung(self): # label: zung_score (> __ => depressed)
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'zung_score', 'depression_med','dep_ssri_med', 'dep_tca_med']]
        df = df.dropna()
        #df['zung_score'] = df['zung_score'].apply(lambda x: 1 if x >= 40 else 0)
        if self.control: # hella inefficient
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v, on_antidep in zip(df['wsc_id'], df['wsc_vst'], df['zung_score'], df['depression_med']) if (f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings) and (not on_antidep)}
        elif self.ssri:
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v, on_ssri in zip(df['wsc_id'], df['wsc_vst'], df['zung_score'], df['dep_ssri_med']) if (f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings) and on_ssri}
        elif self.tca:
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v, on_tca in zip(df['wsc_id'], df['wsc_vst'], df['zung_score'], df['dep_tca_med']) if (f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings) and on_tca}
        elif self.other:
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v, on_antidep, on_ssri, on_tca in zip(df['wsc_id'], df['wsc_vst'], df['zung_score'], df['depression_med'], df['dep_ssri_med'], df['dep_tca_med']) if (f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings) and (not on_tca) and (not on_ssri) and on_antidep}
        else:
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['zung_score']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output
    
    def parse_wsc_beta_blockers(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'htn_beta_med']]
        df =  df.dropna()
        output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['htn_beta_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output
    
    def parse_wsc_ace(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'htn_acei_med']]
        df =  df.dropna()
        output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['htn_acei_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output

    def parse_wsc_thyroid(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'thyroid_med']]
        df =  df.dropna()
        output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['thyroid_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output

    def get_label(self, filename):
        if(self.label == "nsrrid"):
            return filename
        elif(self.label == "dep" and self.num_classes == 2):
            # return torch.tensor(1 if self.data_dict_hs[filename]>=36 else 0, dtype=torch.int64)
            return torch.tensor(self.data_dict_hs[filename], dtype=torch.int64) # returns raw score, threshold before calculating loss
        elif(self.label == "dep"): # regression w/raw zung score
            return torch.tensor(self.data_dict_hs[filename], dtype=torch.float32)
        elif(self.label == "antidep" and self.num_classes <= 2):
            return torch.tensor(self.data_dict[filename][0], dtype=torch.int64)
        elif(self.label == "antidep"): # can add thing based on args.num_classes
            #return torch.tensor(self.data_dict[filename][0], dtype=torch.int64) #binary classification
            if(self.data_dict[filename][2]): #ssri
                return torch.tensor(2)
            elif(self.data_dict[filename][1]): #tca
                return torch.tensor(1)
            elif(self.data_dict[filename][0]): #other
                return torch.tensor(3)
            else:
                return torch.tensor(0) #control

    def threshold_values(self):
        th = 36
        med_names = ['tca', 'ssri', 'other', 'control']
        data = {'>=th':[0,0,0,0], '<th':[0,0,0,0]}
        df = pd.DataFrame(data, index=med_names)
        #bp()
        # so inefficient, yikes
        for pid in self.data_dict_hs.keys():
            if pid in self.data_dict.keys():
                is_tca = self.data_dict[pid][1]
                is_ssri = self.data_dict[pid][2]
                is_other = self.data_dict[pid][0] if not is_tca and not is_ssri else 0
                is_control = 1 if not (is_tca or is_ssri or is_other) else 0
                if self.data_dict_hs[pid] >= th:
                    df.loc['tca','>=th'] += is_tca
                    df.loc['ssri','>=th'] += is_ssri
                    df.loc['other','>=th'] += is_other
                    df.loc['control','>=th'] += is_control
                else:
                    df.loc['tca','<th'] += is_tca
                    df.loc['ssri','<th'] += is_ssri
                    df.loc['other','<th'] += is_other
                    df.loc['control','<th'] += is_control
        
        return df

    
    def get_label_from_filename(self, filename): # FIXME
        # on_antidep = self.data_dict[filename][0]
        # on_tca = self.data_dict[filename][1]
        # on_ssri = self.data_dict[filename][2]
        # if(on_tca):
        #     return 1
        # elif(on_ssri):
        #     return 2
        # elif(on_antidep):
        #     return 3
        # else:
        #     return 0
        ##return self.data_dict[filename][0] #FIXME::: later

        ##temp
        if type(self.data_dict[filename])==list: #antidep
            return (1 if (self.data_dict[filename][0]==1 or self.data_dict[filename][1]==1) else 0)
        else:
            return self.data_dict[filename]
    
    def get_happysad_from_filename(self, filename):
        return self.data_dict_hs[filename]
    
    def get_happysadbinary_from_filename(self, filename):
        return 1 if self.data_dict_hs[filename]>=36 else 0
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        #bp()
        filename = self.all_valid_files[idx]
        filepath = os.path.join(self.encoding_path, filename)
        x = np.load(filepath)
        x = dict(x)
        #bp()
        #print(x['br_latent'].shape) # (565, 768) for ex.
        if not self.no_attention:
            feature = x['decoder_eeg_latent'].squeeze(0)
            if feature.shape[0] >= 150:
                feature = feature[:150, :]
            else:
                feature = np.concatenate((feature, np.zeros((150-feature.shape[0],feature.shape[-1]),dtype=np.float32)), axis=0)
        else:
            feature = x['decoder_eeg_latent'].mean(1).squeeze(0) # it gives you 768 dim feature for each night
        # print("feature shape: ", feature.shape)
        # print(type(feature))
        label = self.get_label(filename)
        # print("label: ", label)
        # print(type(label))

        feature = torch.from_numpy(feature)

        # should return feature and label, both tensors
        return feature, label
    
class EEG_Encoding_MrOS1_Dataset(Dataset):
    def __init__(self, args, file="/data/netmit/wifall/ADetect/data/csv/mros1-dataset-augmented.csv", encoding_path="/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/mros1_new/abdominal_c4_m1"):
        self.file = file
        self.encoding_path = encoding_path
        self.all_mros1_encodings = os.listdir(encoding_path)
        self.no_attention = args.no_attention
        
        if args.label == "benzo":
            self.data_dict = self.parse_mros1_benzos()
        else:
            raise Exception()
        
        self.all_valid_files = list(self.data_dict.keys())
    
    def parse_mros1_benzos(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['nsrrid', 'm1benzo']]
        df = df.dropna()
        #bp()
        output = {f'mros-visit1-aa{k[2:]}.npz': v for k, v in zip(df['nsrrid'], df['m1benzo']) if f'mros-visit1-aa{k[2:]}.npz' in self.all_mros1_encodings}
        return output
    
    def get_label(self, filename):
        return torch.tensor(self.data_dict[filename], dtype=torch.int64)
    
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        filename = self.all_valid_files[idx]
        filepath = os.path.join(self.encoding_path, filename)
        x = np.load(filepath)
        x = dict(x)
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

        # should return feature and label, both tensors
        return feature, label

class EEG_Encoding_UDALL_Dataset(Dataset): #FIXME
    def __init__(self, args, file="/data/netmit/wifall/ADetect/data/csv/wsc-dataset-augmented.csv", 
                 encoding_path="/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/udall/abdominal_c4_m1"):
        self.no_attention = args.no_attention
        self.args = args 
        self.file = file # label file ['NIHND126MXDGP', 'NIHBE740TFYAH', 'NIHPT334YGJLK', 'NIHFW795KLATW', 'NIHDW178UFZHB', 'NIHHD991PGRJC']
        self.label = args.label
        self.control = args.control
        self.tca = args.tca
        self.ssri = args.ssri
        self.other = args.other
        self.num_classes = args.num_classes
        if args.stage_input:
            encoding_path ="/data/netmit/wifall/ADetect/data/udall/stage/"
        self.encoding_path = encoding_path
        self.all_wsc_encodings = os.listdir(self.encoding_path)

        if self.label == "antidep" or self.label == "nsrrid": # FIXME:::
            self.data_dict = self.parse_wsc_antidep()
        elif self.label == "dep":
            self.data_dict = self.parse_wsc_zung()
        elif self.label == "betablocker":
            self.data_dict = self.parse_wsc_beta_blockers()
        elif self.label == "ace":
            self.data_dict = self.parse_wsc_ace()
        elif self.label == "thyroid":
            self.data_dict = self.parse_wsc_thyroid()
        self.all_valid_files = list(self.data_dict.keys())

        # self.data_dict = self.parse_wsc_antidep() #if (label=='antidep' or label=='nsrrid') else self.parse_wsc_zung() # get dictionary of encodings with associated labels
        # self.data_dict_hs = self.parse_wsc_zung()

        # if(self.label == "dep"):
        #     self.all_valid_files = list(self.data_dict_hs.keys())
        # elif(self.label == "antidep"):
        #     self.all_valid_files = list(self.data_dict.keys())
        # # elif(self.label == "benzo"):
        # #     self.all_valid_files = list(self.data_dict_benzos.keys())
        # elif(self.label == "nsrrid"):
        #     self.all_valid_files = list(self.data_dict.keys())

    def parse_wsc_antidep(self): # label: depression_med
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'depression_med','dep_tca_med','dep_ssri_med']]
        df = df.dropna()
        output = {f"wsc-visit{vst}-{id}-nsrr.npz": [v1,v2,v3] for id, vst, v1, v2, v3 in zip(df['wsc_id'], df['wsc_vst'], df['depression_med'],df['dep_tca_med'], df['dep_ssri_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output
    
    def parse_wsc_zung(self): # label: zung_score (> __ => depressed)
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'zung_score', 'depression_med','dep_ssri_med', 'dep_tca_med']]
        df = df.dropna()
        #df['zung_score'] = df['zung_score'].apply(lambda x: 1 if x >= 40 else 0)
        if self.control: # hella inefficient
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v, on_antidep in zip(df['wsc_id'], df['wsc_vst'], df['zung_score'], df['depression_med']) if (f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings) and (not on_antidep)}
        elif self.ssri:
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v, on_ssri in zip(df['wsc_id'], df['wsc_vst'], df['zung_score'], df['dep_ssri_med']) if (f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings) and on_ssri}
        elif self.tca:
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v, on_tca in zip(df['wsc_id'], df['wsc_vst'], df['zung_score'], df['dep_tca_med']) if (f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings) and on_tca}
        elif self.other:
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v, on_antidep, on_ssri, on_tca in zip(df['wsc_id'], df['wsc_vst'], df['zung_score'], df['depression_med'], df['dep_ssri_med'], df['dep_tca_med']) if (f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings) and (not on_tca) and (not on_ssri) and on_antidep}
        else:
            output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['zung_score']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output
    
    def parse_wsc_beta_blockers(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'htn_beta_med']]
        df =  df.dropna()
        output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['htn_beta_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output
    
    def parse_wsc_ace(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'htn_acei_med']]
        df =  df.dropna()
        output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['htn_acei_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output

    def parse_wsc_thyroid(self):
        df = pd.read_csv(self.file, encoding='mac_roman')
        df = df[['wsc_id', 'wsc_vst', 'thyroid_med']]
        df =  df.dropna()
        output = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['thyroid_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in self.all_wsc_encodings}
        return output

    def get_label(self, filename):
        if(self.label == "nsrrid"):
            return filename
        elif(self.label == "dep" and self.num_classes == 2):
            # return torch.tensor(1 if self.data_dict_hs[filename]>=36 else 0, dtype=torch.int64)
            return torch.tensor(self.data_dict_hs[filename], dtype=torch.int64) # returns raw score, threshold before calculating loss
        elif(self.label == "dep"): # regression w/raw zung score
            return torch.tensor(self.data_dict_hs[filename], dtype=torch.float32)
        elif(self.label == "antidep" and self.num_classes <= 2):
            return torch.tensor(self.data_dict[filename][0], dtype=torch.int64)
        elif(self.label == "antidep"): # can add thing based on args.num_classes
            #return torch.tensor(self.data_dict[filename][0], dtype=torch.int64) #binary classification
            if(self.data_dict[filename][2]): #ssri
                return torch.tensor(2)
            elif(self.data_dict[filename][1]): #tca
                return torch.tensor(1)
            elif(self.data_dict[filename][0]): #other
                return torch.tensor(3)
            else:
                return torch.tensor(0) #control

    def threshold_values(self):
        th = 36
        med_names = ['tca', 'ssri', 'other', 'control']
        data = {'>=th':[0,0,0,0], '<th':[0,0,0,0]}
        df = pd.DataFrame(data, index=med_names)
        #bp()
        # so inefficient, yikes
        for pid in self.data_dict_hs.keys():
            if pid in self.data_dict.keys():
                is_tca = self.data_dict[pid][1]
                is_ssri = self.data_dict[pid][2]
                is_other = self.data_dict[pid][0] if not is_tca and not is_ssri else 0
                is_control = 1 if not (is_tca or is_ssri or is_other) else 0
                if self.data_dict_hs[pid] >= th:
                    df.loc['tca','>=th'] += is_tca
                    df.loc['ssri','>=th'] += is_ssri
                    df.loc['other','>=th'] += is_other
                    df.loc['control','>=th'] += is_control
                else:
                    df.loc['tca','<th'] += is_tca
                    df.loc['ssri','<th'] += is_ssri
                    df.loc['other','<th'] += is_other
                    df.loc['control','<th'] += is_control
        
        return df

    
    def get_label_from_filename(self, filename): # FIXME
        # on_antidep = self.data_dict[filename][0]
        # on_tca = self.data_dict[filename][1]
        # on_ssri = self.data_dict[filename][2]
        # if(on_tca):
        #     return 1
        # elif(on_ssri):
        #     return 2
        # elif(on_antidep):
        #     return 3
        # else:
        #     return 0
        ##return self.data_dict[filename][0] #FIXME::: later

        ##temp
        if type(self.data_dict[filename])==list: #antidep
            return (1 if (self.data_dict[filename][0]==1 or self.data_dict[filename][1]==1) else 0)
        else:
            return self.data_dict[filename]
    
    def get_happysad_from_filename(self, filename):
        return self.data_dict_hs[filename]
    
    def get_happysadbinary_from_filename(self, filename):
        return 1 if self.data_dict_hs[filename]>=36 else 0
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        #bp()
        filename = self.all_valid_files[idx]
        filepath = os.path.join(self.encoding_path, filename)
        x = np.load(filepath)
        x = dict(x)
        #bp()
        #print(x['br_latent'].shape) # (565, 768) for ex.
        
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
        # print("feature shape: ", feature.shape)
        # print(type(feature))
        label = self.get_label(filename)
        # print("label: ", label)
        # print(type(label))

        feature = torch.from_numpy(feature)

        # should return feature and label, both tensors
        return feature, label

# class BR_Encoding_UDALL_Dataset(Dataset):
#     def __init__(self, patient_id, label='antidep'):
#         self.label = label
#         self.id = patient_id
    
#     def pa
        
# if __name__ == '__main__':
#     class Object(object):
#         pass
#     args = Object
#     args.data_source = 'eeg'
#     dataset = EEG_SHHS2_Dataset(args)
    
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#     aa = iter(dataloader)
#     for i in range(10):
#         t = datetime.datetime.now()
#         data, label = next(aa)
#         bp()
#         print(data.shape, label)
#         print(datetime.datetime.now() - t)
#     bp()  

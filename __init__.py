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

class EEG_SHHS_Dataset(Dataset):

    def __init__(self, target_label="PD", transform=None):
        data_path = '/data/netmit/wifall/ADetect/data/'
        self.mayo_label_path = os.path.join(data_path, "mayo_new/csv")
        self.root_dir = os.path.join(data_path, 'mayo_new/c4_m1') ## c3_m2, c3_m1_spec,         
        
        self.transform = transform
        all_files = os.listdir(self.root_dir)
        self.all_valid_files = [item for item in all_files if ("nm" in item or "pd" in item)]
        self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz
        self.target_label = target_label
        
        self.total_nm = len([item for item in all_files if "nm" in item])
        self.total_pd = len([item for item in all_files if "pd" in item])


        self.MAYO_BAD = {    # 21
            '198', '246', '257', '439', '542', '377', '345', '510', '368', '284', '293', '416', '327', '364', '423', '316',
            '291', '455', '200', '189', '478'
        }
        self.MAYO_SHORT = {  # 42
            '251', '199', '495', '165', '272', '174', '254', '406', '344', '170', '154', '522', '525', '338', '260', '292',
            '276', '341', '310', '473', '408', '441', '178', '366', '329', '428', '446', '434', '171', '523', '311', '322',
            '528', '281', '156', '141', '422', '279', '313', '546', '318', '351'
        }

        # shhs1_label_path = data_path + 'shhs2/csv/shhs1-dataset-0.14.0.csv'
        # shhs2_label_path = data_path + 'shhs2/csv/shhs2-dataset-0.14.0.csv'
        
        # shhs1_df = pd.read_csv(shhs1_label_path, encoding='mac_roman')
        # shhs2_df = pd.read_csv(shhs2_label_path, encoding='mac_roman')

        # antidep_df = shhs2_df[['nsrrid', 'TCA2', 'NTCA2']].copy()
        # antidep_df.insert(1, 'TCA1', 0)
        # antidep_df.insert(2, 'NTCA1', 0)

        # for id in shhs1_df['nsrrid']:
        #     antidep_df.at[id, 'TCA1'] = shhs1_df[shhs1_df['nsrrid'] == id]['TCA1']
        #     antidep_df.at[id, 'NTCA1'] = shhs1_df[shhs1_df['nsrrid'] == id]['NTCA1']

    def get_label(self, filename):
        is_pd = "pd" in filename
        # if is_pd:
            # return torch.tensor([0,1],dtype=torch.float)
        # else:
            # return torch.tensor([1,0],dtype=torch.float)
        output =  torch.tensor(is_pd)
        return output.type(torch.LongTensor)
    def __len__(self):
        return len(self.all_valid_files)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        filename = self.all_valid_files[idx]
        data = np.load(os.path.join(self.root_dir, filename)).get('data')
        sample = np.zeros(self.INPUT_SIZE)
        
        if len(data) < self.INPUT_SIZE:
            sample[:len(data)] = data
        else:
            sample[:] = data[:self.INPUT_SIZE]
        
        output = torch.tensor(sample, dtype=torch.float)
        output = output.unsqueeze(0)
        
        label = self.get_label(filename)
        return output, label

if __name__ == '__main__':
    dataset = EEG_SHHS_Dataset()
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    aa = iter(dataloader)
    for i in range(10):
        t = datetime.datetime.now()
        data, label = next(aa)
        print(data.shape, label)
        print(datetime.datetime.now() - t)
    bp()

    # # /home/yuzhe/Downloads/pd_files/mayo/mayo_hy.xlsx
    # def parse_mayo_hy_scores(self, file='/data/netmit/wifall/ADetect/data/mayo_new/csv/AD-PD De-identified updated SLG 06-25-20.xlsx', show=True, exist=True, exclude=True):
    #     df = pd.DataFrame(pd.read_excel(file, header=0))
    #     dict_hy = {f"mayopd-{k[-4:] if k[-4] != '0' else k[-3:]}": float(v) for k, v in zip(df['Study ID'], df['H/Y Score'])
    #             if isinstance(v, (int, float))}
    #     dict_hy = {k: dict_hy[k] for k in dict_hy if not np.isnan(dict_hy[k])}
    #     filenames = [name.split('.')[0] for name in os.listdir('/home/yuzhe/remote2/data/learning_full/prknsn_mayo/thorax')]
    #     if exclude:
    #         dict_hy = {k: dict_hy[k] for k in dict_hy if k.split('-')[-1] not in self.MAYO_BAD.union(self.MAYO_SHORT)}
    #     if show:
    #         print(dict_hy)
    #         lst = [v for _, v in dict_hy.items()] if not exist else [v for k, v in dict_hy.items() if k in filenames]
    #         bin_width = 0.5
    #         _, ax = plt.subplots(figsize=(6, 4))
    #         arr = ax.hist(lst, bins=np.arange(-1, 6 + bin_width, bin_width), width=bin_width - .1)
    #         for i in range(2, len(np.arange(-1, 6 + bin_width, bin_width)) - 1):
    #             if int(arr[0][i]):
    #                 plt.text(arr[1][i] + bin_width / 4, arr[0][i] + .2, str(int(arr[0][i])))
    #         ax.set_ylabel('# of subjects')
    #         ax.set_xlabel('H&Y score')
    #         ax.set_xlim([-.2, 6.2])
    #         plt.tight_layout()
    #         plt.show()
    #     return dict_hy
    
    # #'/home/yuzhe/remote2/data/learning_full/mayo'
    # def visualize_extracted_eeg(folder='/data/netmit/wifall/ADetect/data/mayo_new/', processed=True, file_id=None):
    #     variable_names = ['stage', 'eeg_c4_m1']
    #     for filename in tqdm(sorted(os.listdir(join(folder, variable_names[0])))):
    #         if file_id is not None and file_id not in filename:
    #             continue
    #         stage = np.load(join(folder, variable_names[0], filename))['data']
    #         stage = stage[::10]
    #         subfigs = 6 if processed else 2
    #         _, ax = plt.subplots(subfigs, 1, figsize=(16, 8), sharex='all')
    #         plt.suptitle(f"Mayo {filename.split('.')[0]}", fontsize=15)
    #         ax[0].plot(stage, 'b')
    #         eeg = np.load(join(folder, variable_names[1], filename))['data']
    #         eeg = zoom(eeg, 1. / 256)
    #         ax[1].plot(eeg, 'r', alpha=.5)
    #         if processed:
    #             mask = np.load(join(folder, 'eeglab_mask', filename))['data']
    #             ax[1].fill_between(np.arange(0, mask.shape[-1]), 0., (1 - mask) * max(eeg), facecolor='black')
    #             band_name = ['Delta', 'Theta', 'Alpha', 'Beta']
    #             eeg_spec = np.load(join(folder, 'eeg_wspec_20_1_psd_4', filename))['data'][0]
    #             eeg_spec /= np.sum(eeg_spec, axis=0)
    #             for i in range(len(band_name)):
    #                 ax[2 + i].plot(eeg_spec[i], 'g')
    #                 ax[2 + i].set_title(band_name[i])
    #         for i in range(len(variable_names)):
    #             ax[i].set_title(variable_names[i])
    #         plt.show()
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

def id_from_file(filename):
    return int(filename[6:-4])

class EEG_SHHS_Dataset(Dataset):

    def __init__(self, transform=None):
        data_path = '/data/netmit/wifall/ADetect/data/'
        self.shhs_label_path = os.path.join(data_path, "shhs2/csv")
        self.root_dir = os.path.join(data_path, 'shhs2/c4_m1') ## c3_m2, c3_m1_spec,

        shhs2_data_path = data_path + 'shhs2/csv/shhs2-dataset-0.14.0.csv'
        shhs2_df = pd.read_csv(shhs2_data_path, encoding='mac_roman', index_col=0)
        antidep_df = shhs2_df[['TCA2', 'NTCA2']].copy()

        for id in tqdm(shhs2_df.index):
            if (not shhs2_df.loc[id, 'shhs2_psg']):
                antidep_df.drop(index=id)
                continue
        antidep_df = antidep_df.dropna(how='any') # drop any patients with nan values
        self.antidep_df = antidep_df

        def has_psg(filename):
            id = id_from_file(filename)
            psg = shhs2_df.loc[id, 'shhs2_psg']
            return psg
        def has_med_info(filename):
            id = id_from_file(filename)
            med_info = not (pd.isna(shhs2_df.loc[id, 'TCA2']) or pd.isna(shhs2_df.loc[id, 'NTCA2']))
            return med_info
        
        self.transform = transform
        all_files = os.listdir(self.root_dir)
        self.all_valid_files = [item for item in all_files if (not '._' in item and has_psg(item) and has_med_info(item))]
        self.INPUT_SIZE = 12 * 60 * 60 * 64 #hours to 64hz

    def get_label(self, filename):
        nsrrid = id_from_file(filename)
        on_med = (self.antidep_df.loc[nsrrid, 'TCA2'] or self.antidep_df.loc[nsrrid, 'NTCA2'])
        # is_pd = "pd" in filename
        # if is_pd:
            # return torch.tensor([0,1],dtype=torch.float)
        # else:
            # return torch.tensor([1,0],dtype=torch.float)
        
        output =  torch.tensor(on_med)
        return output.type(torch.LongTensor)
    def __len__(self):
        return len(self.all_valid_files)

    def __getitem__(self, idx):

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
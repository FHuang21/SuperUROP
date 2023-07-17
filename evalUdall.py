import torch
import argparse
import numpy as np
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, random_split, default_collate
from model import SimplePredictor
import torch.nn as nn
import os
from tqdm import tqdm
from ipdb import set_trace as bp

def get_date_from_filename(filename):
    # Extract the date part from the filename
    date_part = filename.split('.')[0].split('-')
    return ''.join(date_part)

def get_ordered_nights_list(nights_dir):
    unordered_nights_list = os.listdir(nights_dir)
    return sorted(unordered_nights_list, key=get_date_from_filename, reverse=False)

def process_nights(nights_list, patient_dir):
    features = []
    for night in nights_list:
        x = np.load(os.path.join(patient_dir, night))
        x = dict(x)
        x = x['decoder_eeg_latent'].mean(1).squeeze(0)
        feature = torch.from_numpy(x)
        features.append(feature)
    return features


model = SimplePredictor(output_dim=2)
state_dict = torch.load("/data/scratch/scadavid/projects/data/models/encoding/shhs2_wsc/class_2/eeg/lr_0.0002_w_1.0,14.0_f1macro_0.79_antidep-binary_.pt")
model.load_state_dict(state_dict)

data_path = "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/udall/abdominal_c4_m1/"
patient1_dir = os.path.join(data_path, 'Hao_data_NIHXB175YAGF7')
patient2_dir = os.path.join(data_path, 'Hao_data_NIHNT823CHAC3')

patient1_nights = get_ordered_nights_list(patient1_dir)
patient1_processed_nights = process_nights(patient1_nights, patient1_dir) # to get the 768 dim features
patient2_nights = get_ordered_nights_list(patient2_dir)
patient2_processed_nights = process_nights(patient2_nights, patient2_dir)
with torch.no_grad():
    patient1_preds = [torch.argmax(model(proc_night)) for proc_night in patient1_processed_nights] # 658 nights (162 pos)
    patient2_preds = [torch.argmax(model(proc_night)) for proc_night in patient2_processed_nights] # 272 nights (265 pos)

bp()
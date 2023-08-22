import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, f1_score

#import os
from ipdb import set_trace as bp

# initialize args and model
class Object(object):
    pass

args = Object()
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.5; args.no_attention = False
args.label = "nsrrid"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False

dataset = EEG_Encoding_WSC_Dataset(args)
data_dict = dataset.data_dict
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# args.tca = True
# tca_dataset = EEG_Encoding_WSC_Dataset(args)
# args.tca = False
# args.ssri = True
# ssri_dataset = EEG_Encoding_WSC_Dataset(args)
# args.ssri = False
# args.other = True
# other_dataset = EEG_Encoding_WSC_Dataset(args)
# args.other = False

# using best antidep model (tuned relu BCE)
model_path =  "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4bce_tuned_relu_081123_final/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_relu_081123_final_epochs2_fold0.pt"
model = SimonModel(args)
fc_end = nn.Linear(2, 1)
model = nn.Sequential(model, nn.ReLU(), fc_end)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

control_preds = []; control_true = []; control_zungs = []
tca_preds = []; tca_true = []; tca_zungs = []
ssri_preds = []; ssri_true = []; ssri_zungs = []
other_preds = []; other_true = []; other_zungs = []

# get happy sad data dict
df = pd.read_csv("/data/netmit/wifall/ADetect/data/csv/wsc-dataset-augmented.csv", encoding='mac_roman')
df = df[['wsc_id', 'wsc_vst', 'zung_score']]
df = df.dropna()
# dep lists:
tp_zung = []
fp_zung = []
tn_zung = []
fn_zung = []
data_dict_zung = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['zung_score']) if f"wsc-visit{vst}-{id}-nsrr.npz" in data_dict.keys()}
with torch.no_grad():
    for X, pid in dataloader:
        #bp()
        pid = pid[0]
        if pid not in data_dict_zung.keys():
            continue
        
        pred = model(X)
        pred = torch.sigmoid(pred)
        pred = pred.item()
        pred_class = 1 if pred >= 0.2 else 0 # predicted antidepressant label
        y = dataset.get_label_from_filename(pid) # actual antidepressant label
        zung = data_dict_zung[pid] # zung score

        if pred_class==1 and y==1:
            tp_zung.append(zung)
        elif pred_class==0 and y==0:
            tn_zung.append(zung)
        elif pred_class==1 and y==0:
            fp_zung.append(zung)
        elif pred_class==0 and y==1:
            fn_zung.append(zung)

        # if data_dict[pid][1]==1: # tca
        #     tca_preds.append(pred_class)
        #     tca_true.append(y)
        #     tca_zungs.append(data_dict_zung[pid])
        # elif data_dict[pid][2]==1: # ssri
        #     ssri_preds.append(pred_class)
        #     ssri_true.append(y)
        #     ssri_zungs.append(data_dict_zung[pid])
        # elif data_dict[pid][0]==1: # other
        #     other_preds.append(pred_class)
        #     other_true.append(y)
        #     other_zungs.append(data_dict_zung[pid])
        # else: # control
        #     control_preds.append(pred_class)
        #     control_true.append(y)
        #     control_zungs.append(data_dict_zung[pid])

#conf_mat = confusion_matrix(y_true, y_pred)
#bp()

# print("precision, recall, and f1 for each group")

# print("tca:")
# print(precision_score(tca_true, tca_preds), " ", recall_score(tca_true, tca_preds), " ", f1_score(tca_true, tca_preds))
# print(sum(tca_zungs) / len(tca_zungs))

# print("ssri:")
# print(precision_score(ssri_true, ssri_preds), " ", recall_score(ssri_true, ssri_preds), " ", f1_score(ssri_true, ssri_preds))
# print(sum(ssri_zungs) / len(ssri_zungs))

# print("other:")
# print(precision_score(other_true, other_preds), " ", recall_score(other_true, other_preds), " ", f1_score(other_true, other_preds))
# print(sum(other_zungs) / len(other_zungs))

## zung stuff ##
print("tp zung avg: ", sum(tp_zung) / len(tp_zung))
print("tn zung avg: ", sum(tn_zung) / len(tn_zung))
print("fp zung avg: ", sum(fp_zung) / len(fp_zung))
print("fn zung avg: ", sum(fn_zung) / len(fn_zung))

bp()
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset, EEG_Encoding_MrOS1_Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from ipdb import set_trace as bp

# def get_pos_f1(y_pred_classes, y_true):
#     pass
# def get_neg_f1(y_pred_classes, y_true):
#     pass
# def get_macro_f1(y_pred_classes, y_true):
#     pass

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=4e-4, help='learning rate')
args = parser.parse_args()
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.5
args.no_attention = False; args.label = "benzo"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False

## model stuff
#model_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/checkpoint_simon_model_w14.0/lr_0.0004_w_1.0,14.0_bs_16_f1macro_-1.0_256,64,16_bns_0,0,0_heads4_0.5_att_ctrl_simonmodelweight2_fold0_epoch34.pt"
#old ali best:
#model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/ali_best/lr_0.0002_w_1.0,14.0_bs_16_f1macro_0.72_256,64,16_bns_0,0,0_heads4_0.5_att_alibest_fold0_epoch29.pt"
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4tuning_081023/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_atttuning_081023_epochs2_fold0.pt"
# ^ new tuned model
# new model:
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0004_w_1.0,1.0_ds_eeg_bs_16_epochs_20_dpt_0.5_fold0_256,64,16_heads4balanced_optimization081023_final/lr_0.0004_w_1.0,1.0_bs_16_heads4_0.5_attbalanced_optimization081023_final_epochs20_fold0.pt"
# new new model (added layer):
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0002_w_1.0,2.5_ds_eeg_bs_16_epochs_5_dpt_0.0_fold0_256,64,16_heads4bce_tuned_final/lr_0.0002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_final_epochs5_fold0.pt"
# bce (tuned) model:
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0002_w_1.0,2.5_ds_eeg_bs_16_epochs_3_dpt_0.0_fold0_256,64,16_heads4bce_tuned_final/lr_0.0002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_final_epochs3_fold0.pt"
# bce w/relu model:
model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4bce_tuned_relu_081123_final/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_relu_081123_final_epochs2_fold0.pt"
# benzo1 model:
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0003_w_1.0,1.0_ds_eeg_bs_16_epochs_20_dpt_0.35_fold0_256,64,16_heads4BENZO_balanced_optimization081023/lr_0.0003_w_1.0,1.0_bs_16_heads4_0.35_attBENZO_balanced_optimization081023_epochs20_fold0.pt"


model = SimonModel(args)
# following two lines are for added layer model
#fc_end = nn.Linear(2,1)
#model = nn.Sequential(model, nn.ReLU(), fc_end)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

### wsc stuff ###
# dataset = EEG_Encoding_WSC_Dataset(args)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
####

### mros1 stuff ###
dataset = EEG_Encoding_MrOS1_Dataset(args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

## get preds and pids
y_pred = []
y_true = [] 
with torch.no_grad():
    for X, y in dataloader:
        #bp()
        pred = model(X)
        #pred = torch.sigmoid(pred)
        pred = torch.softmax(pred, dim=1)[0][1].item()
        y_pred.append(pred)

        y = y[0].item()
        y_true.append(y)

antidep_data_dict = dataset.data_dict

# ok, need to get positive f1, negative f1, and macro f1 for classification thresholds from .05 to .95
thresholds = np.arange(0.0, 1.01, .01)
thresholds = thresholds.tolist()
pos_f1s = []
neg_f1s = []
macro_f1s = []
for j, thresh in enumerate(thresholds):
    #bp()
    y_pred_classes = [1 if pred>=thresh else 0 for pred in y_pred]
    # pos_f1s.append(get_pos_f1(y_pred_classes, y_true))
    # neg_f1s.append(get_neg_f1(y_pred_classes, y_true))
    # macro_f1s.append(get_macro_f1(y_pred_classes, y_true))
    class_report = classification_report(y_pred_classes, y_true, output_dict=True)
    pos_f1s.append(class_report['0']['f1-score'])
    neg_f1s.append(class_report['1']['f1-score'])
    macro_f1s.append(class_report['macro avg']['f1-score'])

plt.plot(thresholds, pos_f1s, marker='o', linestyle='-', color='red', label='pos_f1')
plt.plot(thresholds, neg_f1s, marker='o', linestyle='-', color='blue', label='neg_f1')
plt.plot(thresholds, macro_f1s, marker='o', linestyle='-', color='green', label='macro_f1')
plt.legend()
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.title('Benzo Model 1 Evaluated on MrOS1')
plt.savefig('/data/scratch/scadavid/projects/data/figures/f1_stuff/benzo1_thresh_metrics_mros1.pdf')
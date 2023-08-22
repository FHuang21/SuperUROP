import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset, EEG_Encoding_MrOS1_Dataset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from sklearn.model_selection import KFold
#from sklearn.metrics import precision_score, recall_score

import os
from ipdb import set_trace as bp

def plot_auroc(a, x, y, pos_thr=np.log(3)):
    from sklearn.metrics import roc_curve, auc
    #bp()
    # x = np.squeeze(x)
    # idx = np.logical_not(np.isnan(x))
    # idx = idx & (x > -2)
    # x, y = x[idx], y[idx]
    #y_true = (x > pos_thr).astype(int)
    y_true = x
    y_scores = y

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    a.plot(fpr, tpr, color='darkorange', lw=2, )
    a.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    a.set_xlim([0.0, 1.0])
    a.set_ylim([0.0, 1.05])
    a.set_xlabel('False Positive Rate')
    a.set_ylabel('True Positive Rate')
    a.set_title('AUROC: %0.2f' % roc_auc)

def plot_auprc(a, x, y, pos_thr=np.log(3)):
    from sklearn.metrics import precision_recall_curve, auc

    # idx = np.logical_not(np.isnan(x))
    # idx = idx & (x > -2)
    # x, y = x[idx], y[idx]
    # y_true = (x > pos_thr).astype(int)
    # y_scores = y
    y_true = x
    y_scores = y

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    a.step(recall, precision, color='darkorange', alpha=0.2, where='post')
    a.fill_between(recall, precision, step='post', alpha=0.2, color='navy')
    a.set_xlabel('Recall')
    a.set_ylabel('Precision')
    a.set_ylim([0.0, 1.05])
    a.set_xlim([0.0, 1.0])
    a.set_title('AUPRC: %0.2f' % pr_auc)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=4e-4, help='learning rate')
args = parser.parse_args()
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.35

## models:
#model_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/checkpoint_simon_model_w14.0/lr_0.0004_w_1.0,14.0_bs_16_f1macro_-1.0_256,64,16_bns_0,0,0_heads4_0.5_att_ctrl_simonmodelweight2_fold0_epoch34.pt"
# ^for dep
# for antidep:
#model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/ali_best/lr_0.0002_w_1.0,14.0_bs_16_f1macro_0.72_256,64,16_bns_0,0,0_heads4_0.5_att_alibest_fold0_epoch29.pt"
# new best untuned antidep model:
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0004_w_1.0,1.0_ds_eeg_bs_16_epochs_20_dpt_0.5_fold0_256,64,16_heads4balanced_optimization081023_final/lr_0.0004_w_1.0,1.0_bs_16_heads4_0.5_attbalanced_optimization081023_final_epochs20_fold0.pt"
# new best tuned antidep model
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4tuning_081023/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_atttuning_081023_epochs2_fold0.pt"
# benzo 1:
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0003_w_1.0,1.0_ds_eeg_bs_16_epochs_20_dpt_0.35_fold0_256,64,16_heads4BENZO_balanced_optimization081023/lr_0.0003_w_1.0,1.0_bs_16_heads4_0.35_attBENZO_balanced_optimization081023_epochs20_fold0.pt"
# benzo 2:
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0004_w_1.0,1.0_ds_eeg_bs_16_epochs_12_dpt_0.35_fold0_256,64,16_heads4BENZO_balanced_optimization081023/lr_0.0004_w_1.0,1.0_bs_16_heads4_0.35_attBENZO_balanced_optimization081023_epochs12_fold0.pt"
# bce tuned model:
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0002_w_1.0,2.5_ds_eeg_bs_16_epochs_3_dpt_0.0_fold0_256,64,16_heads4bce_tuned_final/lr_0.0002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_final_epochs3_fold0.pt"
# bce tuned relu model:
model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4bce_tuned_relu_081123_final/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_relu_081123_final_epochs2_fold0.pt"

model = SimonModel(args)
# following two for bonus model:
# fc_end = nn.Linear(2,1)
# model = nn.Sequential(model, nn.ReLU(), fc_end)
state_dict = torch.load(model_path)
# model_state_dict = model.state_dict()
# model_state_dict['2'] = model_state_dict['1']
# # model_state_dict['2.bias'] = model_state_dict['1']
# model_state_dict.pop('1.weight')
# model_state_dict.pop('1.bias')
model.load_state_dict(state_dict)
model.eval()
args.no_attention = False; args.label = "benzo"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False

### wsc stuff ###
# dataset = EEG_Encoding_WSC_Dataset(args)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
####

### shhs2 stuff ###
# dataset = EEG_Encoding_SHHS2_Dataset(args)
# kfold = KFold(n_splits=5, shuffle=True, random_state=20)
# train_ids, test_ids = [(train_id_set, test_id_set) for (train_id_set, test_id_set) in kfold.split(dataset)][0]
# trainset = Subset(dataset, train_ids)
# valset = Subset(dataset, test_ids)
# trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
# valloader = DataLoader(valset, batch_size=1, shuffle=False)
####

### mros1 stuff ###
dataset = EEG_Encoding_MrOS1_Dataset(args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
####

y_pred = []
y_true = []
#bp()
softmax = nn.Softmax(dim=1)
num_pos = 0
with torch.no_grad():
    for X, y in dataloader:
        #bp()
        pred = model(X)#.detach().numpy()
        pred = softmax(pred)[0][1]
        #pred = torch.sigmoid(pred)
        pred = pred.item()
        #pred = np.exp(pred[0][0])/sum(np.exp(pred[0][0]))
        y_pred.append(pred)
        #num_pos += (1 if pred==1 else 0)
        y = y.item()
        y_true.append(y)

fig, ax = plt.subplots()
plot_auprc(ax, y_true, y_pred) # change to roc or prc
figure_savepath = "/data/scratch/scadavid/projects/data/figures/roc_prc" 
plt.savefig(os.path.join(figure_savepath,"benzo1_PRC_mros1.pdf"))
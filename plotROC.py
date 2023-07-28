import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score

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

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=4e-4, help='learning rate')
args = parser.parse_args()
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2

#model_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/checkpoint_simon_model_w14.0/lr_0.0004_w_1.0,14.0_bs_16_f1macro_-1.0_256,64,16_bns_0,0,0_heads4_0.5_att_ctrl_simonmodelweight2_fold0_epoch34.pt"
# ^for dep
# for antidep:
model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/ali_best/lr_0.0002_w_1.0,14.0_bs_16_f1macro_0.72_256,64,16_bns_0,0,0_heads4_0.5_att_alibest_fold0_epoch29.pt"

model = SimonModel(args)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

args.no_attention = False; args.label = "antidep"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False

#dataset = EEG_Encoding_WSC_Dataset(args)
dataset = EEG_Encoding_WSC_Dataset(args)
# NOTE: UNCOMMENT THE FOLLOWING IF DOING FOLDS
# kfold = KFold(n_splits=5, shuffle=True, random_state=20)
# train_ids, test_ids = [(train_id_set, test_id_set) for (train_id_set, test_id_set) in kfold.split(dataset)][0]
# trainset = Subset(dataset, train_ids)
# valset = Subset(dataset, test_ids)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

y_pred = []
y_true = []
#bp()
softmax = nn.Softmax(dim=1)
num_pos = 0
with torch.no_grad():
    for X, y in dataloader:
       # bp()
        pred = model(X)#.detach().numpy()
        pred = softmax(pred)[0][1] # NOTE: CHANGE SECOND SUBSCRIPT TO 1 IF WANT THE OTHER CLASS'S ROC CURVE
        pred = pred.item()
        #pred = np.exp(pred[0][0])/sum(np.exp(pred[0][0]))
        y_pred.append(pred)
        #num_pos += (1 if pred==1 else 0)
        y = y.item()
        y_true.append(y)



fig, ax = plt.subplots()
plot_auroc(ax, y_true, y_pred)
figure_savepath = "/data/scratch/scadavid/projects/data/figures" 
plt.savefig(os.path.join(figure_savepath,"ali_antidep_model_ROC_wsc.pdf"))
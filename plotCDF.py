import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from ipdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=4e-4, help='learning rate')
args = parser.parse_args()
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.0

#model_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/checkpoint_simon_model_w14.0/lr_0.0004_w_1.0,14.0_bs_16_f1macro_-1.0_256,64,16_bns_0,0,0_heads4_0.5_att_ctrl_simonmodelweight2_fold0_epoch34.pt"
#model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/ali_best/lr_0.0002_w_1.0,14.0_bs_16_f1macro_0.72_256,64,16_bns_0,0,0_heads4_0.5_att_alibest_fold0_epoch29.pt"
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4tuning_081023/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_atttuning_081023_epochs2_fold0.pt"
# ^ new tuned model
# new model:
#model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0004_w_1.0,1.0_ds_eeg_bs_16_epochs_20_dpt_0.5_fold0_256,64,16_heads4balanced_optimization081023_final/lr_0.0004_w_1.0,1.0_bs_16_heads4_0.5_attbalanced_optimization081023_final_epochs20_fold0.pt"
# new new model (tuned more correctly):
model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0002_w_1.0,2.5_ds_eeg_bs_16_epochs_5_dpt_0.0_fold0_256,64,16_heads4bce_tuned_final/lr_0.0002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_final_epochs5_fold0.pt"

model = SimonModel(args)
# following two lines are for new model
fc_end = nn.Linear(2,1)
model = nn.Sequential(model, fc_end)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

# print("parameters:")
# print(len([parameter for parameter in model.parameters()]))

#bp()

## NOTE: CHANGE THE ANTIDEP SUBSET ARGS APPROPRIATELY
args.no_attention = False; args.label = "nsrrid"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False

### shhs2 stuff ###
# dataset = EEG_Encoding_SHHS2_Dataset(args)
# kfold = KFold(n_splits=5, shuffle=True, random_state=20)
# train_ids, test_ids = [(train_id_set, test_id_set) for (train_id_set, test_id_set) in kfold.split(dataset)][0]
# trainset = Subset(dataset, train_ids)
# valset = Subset(dataset, test_ids)
# trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
# testloader = DataLoader(valset, batch_size=1, shuffle=False)
####

### wsc stuff ###
dataset = EEG_Encoding_WSC_Dataset(args)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
####

# print("len trainset: ", len(trainset))
# print("len valset: ", len(valset))
print("len dataset: ", len(dataset))
#bp()

y_pred = []
y_label = [] # pids
with torch.no_grad():
    for X, y in dataloader:
        #bp()
        pred = model(X)
        pred = torch.sigmoid(pred)
        # pred = torch.softmax(pred, dim=1)[0][1]
        pred = pred.item()
        y_pred.append(pred)
        #bp()

        y = y[0] # weird thing where label is put into first element of tuple... not due to my dataset implementation so must be dataloader thing??X.shape
        y_label.append(y)
        #num_actual_pos += (1 if y==1 else 0)

## dina plot generation stuff
antidep_data_dict = dataset.data_dict

# # need to get list of the y_pred/true indices for the guys on and not on antidep
on_antidep_ids = []
control_ids = []
for i, label in enumerate(y_label):
    #bp()
    if (antidep_data_dict[label][0] == 1): # on antidepressant
        on_antidep_ids.append(i)
    else:
        control_ids.append(i)
antidep_preds = [y_pred[i] for i in on_antidep_ids]
control_preds = [y_pred[i] for i in control_ids]
##

### boxplot stuff ###
# plt.boxplot([control_preds, antidep_preds]) 
# plt.xticks([1, 2], ['control', 'taking antidepressant'])

# plt.title("probabilities for binary antidepressant classification")
# plt.ylabel("logit")
#####

### cdf stuff ###
# Calculate the CDFs using numpy
#bp()
data1_sorted = np.sort(control_preds)
data2_sorted = np.sort(antidep_preds)
cdf1 = np.arange(1, len(data1_sorted) + 1) / len(data1_sorted)
cdf2 = np.arange(1, len(data2_sorted) + 1) / len(data2_sorted)

# # Create the CDF plot for data1 (blue color)
plt.plot(data1_sorted, cdf1, marker='o', linestyle='-', color='blue', label='control')

# # Create the CDF plot for data2 (red color)
plt.plot(data2_sorted, cdf2, marker='o', linestyle='-', color='red', label='taking antidepressant')

# # Add labels and title for better readability
plt.xlabel('Softmax Probability')
plt.ylabel('Cumulative Probability')
plt.title('CDFs for Control and Antidepressant Groups for WSC')

plt.xlim(0, 1.0)

# # Add legend to distinguish between the two datasets
plt.legend()
#####

plt.savefig("/data/scratch/scadavid/projects/data/figures/cdf/new_new_cdfs_wsc_binary.pdf")
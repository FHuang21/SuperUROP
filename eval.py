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
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

#import os
from ipdb import set_trace as bp

# def plot_auroc(a, x, y, pos_thr=np.log(3)):
#     from sklearn.metrics import roc_curve, auc
#     #bp()
#     x = np.squeeze(x)
#     idx = np.logical_not(np.isnan(x))
#     idx = idx & (x > -2)
#     x, y = x[idx], y[idx]
#     #y_true = (x > pos_thr).astype(int)
#     y_true = x
#     y_scores = y

#     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     a.plot(fpr, tpr, color='darkorange', lw=2, )
#     a.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     a.set_xlim([0.0, 1.0])
#     a.set_ylim([0.0, 1.05])
#     a.set_xlabel('False Positive Rate')
#     a.set_ylabel('True Positive Rate')
#     a.set_title('AUROC: %0.2f' % roc_auc)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=4e-4, help='learning rate')
args = parser.parse_args()
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.5

#model_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/checkpoint_simon_model_w14.0/lr_0.0004_w_1.0,14.0_bs_16_f1macro_-1.0_256,64,16_bns_0,0,0_heads4_0.5_att_ctrl_simonmodelweight2_fold0_epoch34.pt"
model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/ali_best/lr_0.0002_w_1.0,14.0_bs_16_f1macro_0.72_256,64,16_bns_0,0,0_heads4_0.5_att_alibest_fold0_epoch29.pt"

model = SimonModel(args)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

# print("parameters:")
# print(len([parameter for parameter in model.parameters()]))

#bp()

## NOTE: CHANGE THE ANTIDEP SUBSET ARGS APPROPRIATELY
args.no_attention = False; args.label = "antidep"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False

### shhs2 stuff ###
dataset = EEG_Encoding_SHHS2_Dataset(args)
kfold = KFold(n_splits=5, shuffle=True, random_state=20)
train_ids, test_ids = [(train_id_set, test_id_set) for (train_id_set, test_id_set) in kfold.split(dataset)][0]
trainset = Subset(dataset, train_ids)
valset = Subset(dataset, test_ids)
trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
testloader = DataLoader(valset, batch_size=1, shuffle=False)
####

### wsc stuff ###
# dataset = EEG_Encoding_WSC_Dataset(args)
# # kfold = KFold(n_splits=5, shuffle=True, random_state=20)
# # train_ids, test_ids = [(train_id_set, test_id_set) for (train_id_set, test_id_set) in kfold.split(dataset)][0]
# # trainset = Subset(dataset, train_ids)
# # valset = Subset(dataset, test_ids)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
####

print("len trainset: ", len(trainset))
print("len valset: ", len(valset))
print("len dataset: ", len(dataset))
bp()

y_pred = []
y_true = []
#y_label = [] # pids
# #bp()
# #softmax = nn.Softmax(dim=1)
# num_pos = 0
#num_actual_pos = 0
with torch.no_grad():
    for X, y in trainloader:
        #bp()
        pred = model(X).detach().numpy()
        pred = np.argmax(pred, axis=1)
        pred = pred.item()
        #pred = pred[0][1]
        # pred = torch.softmax(pred, dim=1)[0][1]
        # pred = pred.item()
        y_pred.append(pred)
        #bp()
        y = y.detach().numpy()
        y_true.append(y)

        #y = y[0] # weird thing where label is put into first element of tuple... not due to my dataset implementation so must be dataloader thing??
        #y_label.append(y)
        #num_actual_pos += (1 if y==1 else 0)

## dina plot generation stuff
# antidep_data_dict = dataset.data_dict

# # need to get list of the y_pred/true indices for the guys on and not on antidep
# on_antidep_ids = []
# control_ids = []
# for i, label in enumerate(y_label):
#     #bp()
#     if (antidep_data_dict[label][0] == 1): # on antidepressant
#         on_antidep_ids.append(i)
#     else:
#         control_ids.append(i)
# antidep_preds = [y_pred[i] for i in on_antidep_ids]
# control_preds = [y_pred[i] for i in control_ids]
##

### boxplot stuff ###
# plt.boxplot([control_preds, antidep_preds]) 
# plt.xticks([1, 2], ['control', 'taking antidepressant'])

# plt.title("probabilities for binary antidepressant classification")
# plt.ylabel("logit")
#####

### cdf stuff ###
# Calculate the CDFs using numpy
# data1_sorted = np.sort(control_preds)
# data2_sorted = np.sort(antidep_preds)
# cdf1 = np.arange(1, len(data1_sorted) + 1) / len(data1_sorted)
# cdf2 = np.arange(1, len(data2_sorted) + 1) / len(data2_sorted)

# # Create the CDF plot for data1 (blue color)
# plt.plot(data1_sorted, cdf1, marker='o', linestyle='-', color='blue', label='control')

# # Create the CDF plot for data2 (red color)
# plt.plot(data2_sorted, cdf2, marker='o', linestyle='-', color='red', label='taking antidepressant')

# # Add labels and title for better readability
# plt.xlabel('softmax probability')
# plt.ylabel('cumulative probability')
# plt.title('CDFs for control and on_antidep groups for wsc')

# # Add legend to distinguish between the two datasets
# plt.legend()
#####

# plt.show()

# plt.savefig("/data/scratch/scadavid/projects/data/figures/cdfs_wsc_binary.pdf")

#print("num actual pos in valset: ", num_actual_pos)
# percent_pos = num_pos / len(y_pred)
# print("num pos: ", num_pos)
# print("total: ", len(y_pred))
# print("% positive: ", percent_pos)

# # Calculate class-wise precision
# precision_classwise = precision_score(y_true, y_pred, average=None)

# # Calculate class-wise recall
# recall_classwise = recall_score(y_true, y_pred, average=None)

# print("Class-wise Precision:", precision_classwise)
# print("Class-wise Recall:", recall_classwise)

###
# y_pred = []
# y_true = []
# #softmax = nn.Softmax(dim=1)
# num_pos = 0
# with torch.no_grad():
#     for X, y in dataloader:
#         #bp()
#         pred = model(X).detach().numpy()
#         pred = np.argmax(pred, axis=1)
#         y_pred.append(pred)
#         #num_pos += (1 if pred==1 else 0)
#         y = y.detach().numpy()
#         #if y >= 36:
#         y_true.append(y)

# #avg = sum(y_true_pos) / len(y_true_pos)
# #print("avg pos dep score in wsc valset is ", avg)
# # percent_pos = num_pos / len(y_pred)
# # print("num pos: ", num_pos)

conf_mat = confusion_matrix(y_true, y_pred)
# report = classification_report(y_true, y_pred)

# print(report)
# ###

# # # Suppose you have class-wise F1 scores
# F1_class0 = 0.91
# F1_class1 = 0.51



# precision_class0 = (2 * F1_class0) / (1 + F1_class0)
# recall_class0 = F1_class0 / (1 + F1_class0)

# # Calculate precision and recall for each class
# precision_class1 = (2 * F1_class1) / (1 + F1_class1)
# recall_class1 = F1_class1 / (1 + F1_class1)


# # # Print the results
# # print("Class 1 - Precision:", precision_class1, "Recall:", recall_class1)
# # print("Class 0 - Precision:", precision_class0, "Recall:", recall_class0)

# print("confusion matrix:")
print(conf_mat)

bp()
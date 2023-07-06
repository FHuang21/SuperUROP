import torch
import torch.nn as nn
#from torchvision.models import resnet50, ResNet50_Weights
from model import EEG_Encoder, BranchVarEncoder, BranchVarPredictor
#from dataLoader2 import MultiClassDataset
from dataset import EEG_SHHS2_Dataset
from torch.utils.data import DataLoader#, random_split, default_collate
#from metrics import Metrics
from tqdm import tqdm
from ipdb import set_trace as bp
import torchmetrics
import os
import argparse
#import sklearn.metrics

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def state_dict_rm_module(state_dict, encoder=True, stagepred=False):
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        if k.startswith('encoder.'):
            if encoder:
                k = k[len('encoder.'):]
            else:
                continue
        if k.startswith('stagepreder'):
            if stagepred:
                k = k[len('stagepreder.'):]
            else:
                continue
        new_dict[k] = v
    return new_dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='shhs2', help='which dataset to train on')
parser.add_argument('--datatype', type=str, default='ts', help='ts or spec')
parser.add_argument('--data_source', type=str, default='eeg', help='eeg, bb, or stage')
args = parser.parse_args()
dataset_name = args.dataset
datatype = args.datatype
data_source = args.data_source
args.task = 'multiclass'
args.num_channel = 256

# this file is to verify that the model loads properly, and in fact it does
data_path = '/data/scratch/scadavid/projects/data'
model_name = 'lr_0.0001_w1_1.0_w2_9.0_f1_0.92.pt'
model_path = os.path.join(data_path, 'models', 'ts', 'shhs1', 'eeg', model_name)

# initialize DataLoader
dataset = EEG_SHHS2_Dataset(args)
batch_size = 16
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# set up model skeleton
eeg_encoder = EEG_Encoder()
branch_var_encoder = BranchVarEncoder(args)
branch_hy_predictor = BranchVarPredictor(args)
model = nn.Sequential(eeg_encoder, branch_var_encoder, branch_hy_predictor).to(device="cuda:0")
model = nn.DataParallel(nn.Sequential(eeg_encoder, branch_var_encoder, branch_hy_predictor), range(0, torch.cuda.device_count()))


# load and test model (trained on SHHS1, going to test on SHHS2)
state_dict = torch.load(model_path)
#model.load_state_dict(state_dict_rm_module(state_dict)) # if not wrapping w/ DataParallel
model.load_state_dict(state_dict)
f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=2).cuda()
# bcm = torchmetrics.classification.BinaryConfusionMatrix
# TP = 0
# TN = 0
# FP = 0
# FN = 0
with torch.no_grad(): # don't store gradients since just evaluating not training
    y_preds = []
    y_true = []
    for X_batch, y_batch in tqdm(test_loader):

        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        y_pred = model(X_batch)
        y_pred_classes = torch.argmax(y_pred, dim=1)

        f1_metric.update(y_pred_classes, y_batch)

        y_preds.append(y_pred_classes)
        y_true.append(y_batch)


y_preds = torch.cat(y_preds)
y_true = torch.cat(y_true)

conf_vals = perf_measure(y_preds, y_true)

f1_score = f1_metric.compute()

print("f1 score: ", f1_score)
print("TP/FP/TN/FN ", conf_vals)

#2651
bp()
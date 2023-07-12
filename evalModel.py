import torch
import argparse
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, random_split, default_collate
from model import SimplePredictor
import torch.nn as nn
import os
from tqdm import tqdm
from ipdb import set_trace as bp

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

parser = argparse.ArgumentParser(description='evaluate specified model')
parser.add_argument('--model_path', type=str, required=True, help='model filename')
parser.add_argument('--dataset', type=str, default='shhs2') # eval set vs model dataset?
parser.add_argument('--datatype', type=str, default='encoding')
parser.add_argument('--data_source', type=str, default='eeg') #should rename 'data_source' to 'input', makes more sense
parser.add_argument('--task', type=str, default='antidep')


args = parser.parse_args()
model_path = args.model_path
dataset_name = args.dataset
datatype = args.datatype
datasource = args.data_source
task = args.task

batch_size = 16

if (datatype == 'encoding'):
    model = SimplePredictor()

if(dataset_name == 'shhs2'):
    dataset = EEG_Encoding_SHHS2_Dataset(label='nsrrid')
else:
    dataset = EEG_Encoding_WSC_Dataset(label='nsrrid')

#model_root = '/data/scratch/scadavid/projects/data/models'
#model_path = os.path.join(model_root, datatype, dataset_name, datasource, model_name)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

gen = torch.Generator().manual_seed(20)
trainset, testset = random_split(dataset, [0.7, 0.3], generator=gen)
#bp()
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
print("len(trainset): ", len(trainset))
print("len(testset): ", len(testset))

#bp()

y_pred = []
y_true = [] # 0s, 1s, and 2s (control/tca/ntca) (if in both then go ntca) ## 0,1,2,3 (control/tca/ssri/other)
with torch.no_grad():
    for X_batch, id_batch in tqdm(testloader):
        for feature in X_batch:
            y_pred.append(model(feature))
        for id in id_batch: # id is really filename
            try:
                current_label = dataset.get_happysadbinary_from_filename(id)
                y_true.append(current_label)
            except:
                y_pred.pop()
                continue


y_pred_classes = []
for tensor in y_pred:
    current_pred = torch.argmax(tensor, dim=0)
    y_pred_classes.append(current_pred.item())
#bp()
#y_pred = torch.argmax(y_pred, dim=1)
y_pred_classes = torch.IntTensor(y_pred_classes)
#y_pred = torch.argmax(y_pred, dim=1)
y_true = torch.IntTensor(y_true)

if(dataset_name == 'shhs2'):
    num_tca = 0
    num_ntca = 0
    tp_tca = 0
    tp_ntca = 0
    fn_tca = 0
    fn_ntca = 0
    for i in range(0, len(y_pred_classes)):
        if(y_pred_classes[i] == 1 and y_true[i] == 1):
            tp_tca += 1
            num_tca += 1
        elif(y_pred_classes[i] == 1 and y_true[i] == 2):
            tp_ntca += 1
            num_ntca += 1
        elif(y_pred_classes[i] == 0 and y_true[i] == 1):
            fn_tca += 1
            num_tca += 1
        elif(y_pred_classes[i] == 0 and y_true[i] == 2):
            fn_ntca += 1
            num_ntca += 1

    print("num tca: ", num_tca)
    print("num ntca: ", num_ntca)
    print("TP (tca/ntca+tca): ", (tp_tca)/(tp_tca+tp_ntca))
    print("TP (ntca/ntca+tca): ", (tp_ntca)/(tp_tca+tp_ntca))
    print("FN (tca/ntca+tca): ", (fn_tca)/(fn_tca+fn_ntca))
    print("FN (ntca/ntca+tca): ", (fn_ntca)/(fn_tca+fn_ntca))
    print("% tca pred correctly: ", (tp_tca)/(tp_tca+fn_tca))
    print("% ntca pred correctly: ", (tp_ntca)/(tp_ntca+fn_ntca))
elif(dataset_name == 'wsc' and task == 'antidep'):
    num_tca = 0
    tp_tca = 0
    fn_tca = 0
    num_ssri = 0
    tp_ssri = 0
    fn_ssri = 0
    num_other = 0
    tp_other = 0
    fn_other = 0
    for i in range(0, len(y_pred_classes)):
        if(y_pred_classes[i] == 1 and y_true[i] == 1):
            tp_tca += 1
            num_tca += 1
        elif(y_pred_classes[i] == 1 and y_true[i] == 2):
            tp_ssri += 1
            num_ssri += 1
        elif(y_pred_classes[i] == 1 and y_true[i] == 3):
            tp_other += 1
            num_other += 1
        elif(y_pred_classes[i] == 0 and y_true[i] == 1):
            fn_tca += 1
            num_tca += 1
        elif(y_pred_classes[i] == 0 and y_true[i] == 2):
            fn_ssri += 1
            num_ssri += 1
        elif(y_pred_classes[i] == 0 and y_true[i] == 3):
            fn_other += 1
            num_other += 1
    
    print("num on antidep: ", )
    print("num tca: ", num_tca)
    print("num ssri: ", num_ssri)
    print("num other: ", num_other)
    print("TP (tca/tca+ssri+other): ", (tp_tca)/(tp_tca+tp_ssri+tp_other))
    print("TP (ssri/tca+ssri+other): ", (tp_ssri)/(tp_tca+tp_ssri+tp_other))
    print("TP (other/tca+ssri+other): ", (tp_other)/(tp_tca+tp_ssri+tp_other))
    print("FN (tca/tca+ssri+other): ", (fn_tca)/(fn_tca+fn_ssri+fn_other))
    print("FN (ssri/tca+ssri+other): ", (fn_ssri)/(fn_tca+fn_ssri+fn_other))
    print("FN (other/tca+ssri+other): ", (fn_other)/(fn_tca+fn_ssri+fn_other))
    print("percent tca pred correctly: ", (tp_tca)/(tp_tca+fn_tca))
    print("percent ssri pred correctly: ", (tp_ssri)/(tp_ssri+fn_ssri))
    print("percent other pred correctly: ", (tp_other)/(tp_other+fn_other))
elif (dataset_name == 'wsc' and task == 'dep'):
    tp, fp, tn, fn = perf_measure(y_pred_classes, y_true)
    print("tp: ", tp)
    print("tn: ", tn)
    print("fp: ", fp)
    print("fn: ", fn)

# best shhs2 antidep predictor path: /data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/lr_0.001_w1_1.0_w2_14.0_posf1_0.6.pt
# best wsc antidep predictor path: /data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/lr_0.001_w1_1.0_w2_14.0_posf1_0.68.pt

# python evalModel.py --model_path /data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/lr_0.001_w1_1.0_w2_14.0_posf1_0.6.pt
import torch
import argparse
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, random_split, default_collate, Subset
from model import SimpleAttentionPredictor
import sklearn.metrics
from sklearn.model_selection import KFold
import torch.nn as nn
#import os
from tqdm import tqdm
from ipdb import set_trace as bp

torch.manual_seed(20)

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
#parser.add_argument('--model_arch', type=str, required=True, help='model architecture (e.g. SimpleAttentionPredictor)')
parser.add_argument('--model_path', type=str, required=True, help='model filepath')
parser.add_argument('--dataset', type=str, default='shhs2') # eval set vs model dataset?
parser.add_argument('--datatype', type=str, default='encoding')
parser.add_argument('--data_source', type=str, default='eeg') #should rename 'data_source' to 'input', makes more sense
parser.add_argument('--task', type=str, default='dep')

args = parser.parse_args()
args.num_classes = 2
args.dropout = 0.5
args.label = 'dep'
args.num_folds = 5
args.no_attention = False
args.control = False  #########
args.tca = False
args.ntca = True
args.batch_norms = [0,0,0]
args.layer_dims = [256,64,16]
args.num_heads = 3
model_path = args.model_path
dataset_name = args.dataset
datatype = args.datatype
datasource = args.data_source
task = args.task

batch_size = 16

if (datatype == 'encoding'):
    model = SimpleAttentionPredictor(args)

if(dataset_name == 'shhs2'):
    dataset = EEG_Encoding_SHHS2_Dataset(args)
else:
    dataset = EEG_Encoding_WSC_Dataset(args)

#model_root = '/data/scratch/scadavid/projects/data/models'
#model_path = os.path.join(model_root, datatype, dataset_name, datasource, model_name)

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

#gen = torch.Generator().manual_seed(20)
# trainset, testset = random_split(dataset, [0.7, 0.3])
# kfold = KFold(n_splits=5, shuffle=True, random_state=20)

# train_test_tuples = [(train_ids, test_ids) for (train_ids, test_ids) in kfold.split(dataset)]
# train_ids, test_ids = train_test_tuples[4]
# trainset = Subset(dataset, train_ids)
# testset = Subset(dataset, test_ids)

#bp()
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
# print("len(trainset): ", len(trainset))
# print("len(testset): ", len(testset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print("len(dataset): ", len(dataset))


y_pred = []
y_true = [] # 0s, 1s, and 2s (control/tca/ntca) (if in both then go ntca) ## 0,1,2,3 (control/tca/ssri/other)
with torch.no_grad():
    for X_batch, y_batch in dataloader:
        for feature in X_batch:
            y_pred.append(model(feature.unsqueeze(0)))
        for label in y_batch: # id is really filename
            y_true.append(label)
            # try:
            #     current_label = dataset.(id)
            #     y_true.append(current_label)
            # except:
            #     y_pred.pop()
            #     continue


y_pred_classes = []
for tensor in y_pred:
    current_pred = torch.argmax(tensor, dim=1)
    y_pred_classes.append(current_pred.item())

#y_pred = torch.argmax(y_pred, dim=1)
y_pred_classes = torch.IntTensor(y_pred_classes)
#y_pred = torch.argmax(y_pred, dim=1)
y_true = torch.IntTensor(y_true)

print(sklearn.metrics.classification_report(y_true, y_pred_classes))

#bp()

# if(dataset_name == 'shhs2'):
#     num_tca = 0
#     num_ntca = 0
#     tp_tca = 0
#     tp_ntca = 0
#     fn_tca = 0
#     fn_ntca = 0
#     for i in range(0, len(y_pred_classes)):
#         if(y_pred_classes[i] == 1 and y_true[i] == 1):
#             tp_tca += 1
#             num_tca += 1
#         elif(y_pred_classes[i] == 1 and y_true[i] == 2):
#             tp_ntca += 1
#             num_ntca += 1
#         elif(y_pred_classes[i] == 0 and y_true[i] == 1):
#             fn_tca += 1
#             num_tca += 1
#         elif(y_pred_classes[i] == 0 and y_true[i] == 2):
#             fn_ntca += 1
#             num_ntca += 1

#     print("num tca: ", num_tca)
#     print("num ntca: ", num_ntca)
#     print("TP (tca/ntca+tca): ", (tp_tca)/(tp_tca+tp_ntca))
#     print("TP (ntca/ntca+tca): ", (tp_ntca)/(tp_tca+tp_ntca))
#     print("FN (tca/ntca+tca): ", (fn_tca)/(fn_tca+fn_ntca))
#     print("FN (ntca/ntca+tca): ", (fn_ntca)/(fn_tca+fn_ntca))
#     print("% tca pred correctly: ", (tp_tca)/(tp_tca+fn_tca))
#     print("% ntca pred correctly: ", (tp_ntca)/(tp_ntca+fn_ntca))
# elif(dataset_name == 'wsc' and task == 'antidep'):
#     num_tca = 0
#     tp_tca = 0
#     fn_tca = 0
#     num_ssri = 0
#     tp_ssri = 0
#     fn_ssri = 0
#     num_other = 0
#     tp_other = 0
#     fn_other = 0
#     for i in range(0, len(y_pred_classes)):
#         if(y_pred_classes[i] == 1 and y_true[i] == 1):
#             tp_tca += 1
#             num_tca += 1
#         elif(y_pred_classes[i] == 1 and y_true[i] == 2):
#             tp_ssri += 1
#             num_ssri += 1
#         elif(y_pred_classes[i] == 1 and y_true[i] == 3):
#             tp_other += 1
#             num_other += 1
#         elif(y_pred_classes[i] == 0 and y_true[i] == 1):
#             fn_tca += 1
#             num_tca += 1
#         elif(y_pred_classes[i] == 0 and y_true[i] == 2):
#             fn_ssri += 1
#             num_ssri += 1
#         elif(y_pred_classes[i] == 0 and y_true[i] == 3):
#             fn_other += 1
#             num_other += 1
    
#     print("num on antidep: ", )
#     print("num tca: ", num_tca)
#     print("num ssri: ", num_ssri)
#     print("num other: ", num_other)
#     print("TP (tca/tca+ssri+other): ", (tp_tca)/(tp_tca+tp_ssri+tp_other))
#     print("TP (ssri/tca+ssri+other): ", (tp_ssri)/(tp_tca+tp_ssri+tp_other))
#     print("TP (other/tca+ssri+other): ", (tp_other)/(tp_tca+tp_ssri+tp_other))
#     print("FN (tca/tca+ssri+other): ", (fn_tca)/(fn_tca+fn_ssri+fn_other))
#     print("FN (ssri/tca+ssri+other): ", (fn_ssri)/(fn_tca+fn_ssri+fn_other))
#     print("FN (other/tca+ssri+other): ", (fn_other)/(fn_tca+fn_ssri+fn_other))
#     print("percent tca pred correctly: ", (tp_tca)/(tp_tca+fn_tca))
#     print("percent ssri pred correctly: ", (tp_ssri)/(tp_ssri+fn_ssri))
#     print("percent other pred correctly: ", (tp_other)/(tp_other+fn_other))
# elif (dataset_name == 'wsc' and task == 'dep'):
#     tp, fp, tn, fn = perf_measure(y_pred_classes, y_true)
#     print("tp: ", tp)
#     print("tn: ", tn)
#     print("fp: ", fp)
#     print("fn: ", fn)

# best shhs2 antidep predictor path: /data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/lr_0.001_w1_1.0_w2_14.0_posf1_0.6.pt
# best wsc antidep predictor path: /data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/lr_0.001_w1_1.0_w2_14.0_posf1_0.68.pt

# python evalModel.py --model_path /data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/lr_0.0004_w_1.0,10.0_bs_16_f1macro_0.57_256,64,16_bns_0,0,0_heads3_0.5_att_ctrl_fold4.pt
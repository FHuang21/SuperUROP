from dataset import EEG_Encoding_SHHS2_Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace as bp

class Object(object):
    pass
args = Object()

args.no_attention = False; args.label = "thyroid"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False
args.num_classes = 2
kfold = KFold(n_splits=5, shuffle=True, random_state=20)
dataset = EEG_Encoding_SHHS2_Dataset(args)
fold_splits = [(train_id_set, val_id_set) for (train_id_set, val_id_set) in kfold.split(dataset)]
#train_ids, val_ids = fold_splits[0]


for fold, (train_ids, val_ids) in enumerate(fold_splits):
    trainset = Subset(dataset, train_ids)
    valset = Subset(dataset, val_ids)

    print(f"--- FOLD {fold} ---")
    print("len trainset: ", len(trainset))
    print("len valset: ", len(valset))
    num_pos_train = 0
    for X, y in trainset:
        bp()
        if y.item()==1:
            num_pos_train += 1

    num_pos_val = 0
    for X, y in valset:
        if y.item()==1:
            num_pos_val += 1

    
    print("# train pos: ", num_pos_train)
    print("# val pos ", num_pos_val)

bp()

## for benzos:
# --- FOLD 0 ---
# len trainset:  2114
# len valset:  529
# # train pos:  107
# # val pos  33
# --- FOLD 1 ---
# len trainset:  2114
# len valset:  529
# # train pos:  111
# # val pos  29
# --- FOLD 2 ---
# len trainset:  2114
# len valset:  529
# # train pos:  126
# # val pos  14
# --- FOLD 3 ---
# len trainset:  2115
# len valset:  528
# # train pos:  111
# # val pos  29
# --- FOLD 4 ---
# len trainset:  2115
# len valset:  528
# # train pos:  105
# # val pos  35

## for thyroid:
# --- FOLD 0 ---
# len trainset:  2114
# len valset:  529
# # train pos:  254
# # val pos  74
# --- FOLD 1 ---
# len trainset:  2114
# len valset:  529
# # train pos:  258
# # val pos  70
# --- FOLD 2 ---
# len trainset:  2114
# len valset:  529
# # train pos:  257
# # val pos  71
# --- FOLD 3 ---
# len trainset:  2115
# len valset:  528
# # train pos:  276
# # val pos  52
# --- FOLD 4 ---
# len trainset:  2115
# len valset:  528
# # train pos:  267
# # val pos  61
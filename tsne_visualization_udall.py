import matplotlib.pyplot as plt 
# from label_util import Labeler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
#import path 
import torch 
import numpy as np 
# from dataset import UdallBreathingStagesDataset, DataLoader
from tqdm import tqdm 
import math 
from ipdb import set_trace as bp
import pandas as pd 
from torch.utils.data import random_split, DataLoader, Subset
# from model import BreathStageNet
import umap 
from torch import nn
from model import BranchHYPredictor, EEG_Encoder, BranchVarEncoder, BranchVarPredictor, StagePredictorModel, SimpleAttentionPredictor, SimonModel
from sklearn.model_selection import KFold
from dataset import *
import argparse
from interactive_tsne import main
import pickle
parser = argparse.ArgumentParser()

ROOT_DIR = "/data/scratch/scadavid/projects/data"
EXPERIMENT = "checkpoints/age_pred/shhs1_mayoall"
MODEL = "model_newest.pt"

MODEL_PATH = os.path.join(ROOT_DIR,EXPERIMENT,MODEL)
modeldir = os.path.join(ROOT_DIR,EXPERIMENT)

MAX_NUMBER_PATIENTS = 10000 
BATCH_SIZE = 16
LATENT_SPACE_SIZE = 32

embedding_path = "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/udall/abdominal_c4_m1"
# NOTE


def get_foldername(pid):
    all_foldernames = os.listdir(embedding_path)
    for foldername in all_foldernames:
        if pid in foldername:
            return foldername

def get_embedding(np_file_path):
    file = np.load(np_file_path)
    feature = file['decoder_eeg_latent'].squeeze(0)
    if feature.shape[0] >= 150:
        feature = feature[:150, :]
    else:
        feature = np.concatenate((feature, np.zeros((150-feature.shape[0],feature.shape[-1]),dtype=np.float32)), axis=0)
    feature = torch.from_numpy(feature)
    feature = torch.unsqueeze(feature, 0)
    feature = feature.to(device)
    return feature

args = parser.parse_args()
args.data_source = 'eeg'

args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.0
args.no_attention = False; args.label = "nsrrid"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#wsc_happysadmodel_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/lr_0.0004_w_1.0,10.0_bs_16_f1macro_0.57_256,64,16_bns_0,0,0_heads3_0.5_att_ctrl_fold4.pt"
#wsc_happysadmodel_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/checkpoint_simon_model_w14.0/lr_0.0004_w_1.0,14.0_bs_16_f1macro_-1.0_256,64,16_bns_0,0,0_heads4_0.5_att_ctrl_simonmodelweight2_fold0_epoch34.pt"
#ali_best_antidep_model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/ali_best/lr_0.0002_w_1.0,14.0_bs_16_f1macro_0.72_256,64,16_bns_0,0,0_heads4_0.5_att_alibest_fold0_epoch29.pt"
#tuned_antidep_model_path = model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4tuning_081023/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_atttuning_081023_epochs2_fold0.pt"
#benzo1_model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0003_w_1.0,1.0_ds_eeg_bs_16_epochs_20_dpt_0.35_fold0_256,64,16_heads4BENZO_balanced_optimization081023/lr_0.0003_w_1.0,1.0_bs_16_heads4_0.35_attBENZO_balanced_optimization081023_epochs20_fold0.pt"
#best_bce_tuned_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.0002_w_1.0,2.5_ds_eeg_bs_16_epochs_3_dpt_0.0_fold0_256,64,16_heads4bce_tuned_final/lr_0.0002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_final_epochs3_fold0.pt"
bce_tuned_relu_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4bce_tuned_relu_081123_final/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_relu_081123_final_epochs2_fold0.pt"

model = SimonModel(args)#.to(device)
fc_end = nn.Linear(2, 1)
model = nn.Sequential(model, nn.ReLU(), fc_end).to(device)
state_dict = torch.load(bce_tuned_relu_path)
model.load_state_dict(state_dict)
model.eval()


# patient_dict = {}

# for group in groups:
#     i = 0
#     patient_i = 0
    
#     for data, labels in tqdm(dataloaders[group]):
#         data = data.to(device)
#         labels = np.array(labels)
#         batch_labels = labels
        
#         if labels.shape[0] < BATCH_SIZE:
#             continue 
        
#         if patient_i >= MAX_NUMBER_PATIENTS:
#             break 

#         #bp()
#         for j in range(BATCH_SIZE):
#             patient_i += 1
            
#             eeg = data[j:j+1].to(device)
#             labels = batch_labels[j]
            
#             if labels not in patient_dict:
#                 patient_dict[labels] = True
            
#             with torch.set_grad_enabled(False):
#                 #bp()
#                 # eeg_output = model.attention(eeg)
#                 # age_output1 = model.fc2(eeg_output) # output is 64 dims
#                 age_output1 = model[0].fc2(model[0].relu(model[0].fc1(model[0].encoder(eeg))))
#                 # bp()
#                 # features, code, pred_class = model(breathing, stages)
#                 raw_outputs[group][i] = age_output1.cpu().numpy().flatten() # projection into lower dimensional space
#                 #pred_classes[group][i] = torch.argmax(model(eeg), dim=1)
#                 pred_classes[group][i] = 1 if model(eeg).item() >= .2 else 0
#                 raw_labels[group][i] = labels
#                 i += 1
#     raw_outputs[group] = raw_outputs[group][:i]
#     pred_classes[group] = pred_classes[group][:i]
#     raw_labels[group] = raw_labels[group][:i]
#     print(len(raw_outputs[group]))

antidep_dict = {'NIHBL760KMGXL': 'escitalopram, trazodone',
'NIHNT823CHAC3': 'escitalopram',
'NIHXN782DBBP7': 'prozac',
'NIHYM875FLXFF': 'fluoxetine',
'NIHCX409ZDTJU': 'escitalopram',
'NIHPX213JXJZC': 'zoloft',
'NIHVA109LWXMF': 'sertraline',
'NIHPV178MDAUT': 'venlafaxine',
'NIHCJ555VCWZY': 'venlafaxine',
'NIHXN551LBFMK': 'paroxetine',
'NIHEP519MZAEZ': 'fluoxetine',
'NIHHD991PGRJC': 'mirtazapine',
'NIHFW795KLATW': 'paroxetine, sertraline', 
'NIHPT334YGJLK': 'paroxetine, desvenlafaxine',
'NIHAV871KZCVE': 'paroxetine',
'NIHMR963TPLWF': 'paroxetine, zoloft',
'NIHJW557ZEUZV': 'control',
'NIHMF399WYNH5': 'control',
'NIHAV025ZCBGB': 'control',
'NIHBY076JZFYN': 'control',
'NIHEB701YGBEC': 'control',
'NIHFT628PHTAY': 'control',
'NIHHG558EJJMM': 'control',
'NIHRY949ZYWHQ': 'control',
'NIHXB175YAGF7': 'control',
'NIHYW557MLDFE': 'control',
'NIHZT156UUPLX': 'control',
'NIHGK080AGLJH': 'control',
'NIHND126MXDGP': 'control',
'NIHBE740TFYAH': 'control',
'NIHNX715KUVY8': 'control',
'NIHFX695VBHFM': 'control',
'NIHDW178UFZHB': 'control',
'NIHTK278VZHYL': 'control',
'NIHGA312KVEC2': 'control',
'NIHWR605ZHTE7': 'control'}
# FIXME: missing a couple patients in above dict?

pids = antidep_dict.keys()
# print("# of pids:")
# print(len(pids))

raw_outputs = {}
#raw_labels = {}
pred_classes = {}
#ratios = {} # FIXME :::
for pid in pids:
    patient_df = pd.read_csv(f'/data/scratch/scadavid/projects/data/udall/{pid}.csv', encoding='mac_roman')
    night_filenames = patient_df['night'].to_numpy(dtype=object)
    preds = patient_df['preds'].to_numpy(dtype=int)
    raw_outputs[pid] = np.zeros((len(patient_df), LATENT_SPACE_SIZE))
    pred_classes[pid] = preds

    for j, night_file in enumerate(night_filenames):
        pid_foldername = get_foldername(pid) # e.g. these have the PD_Hao_data_ or Hao_data
        night_path = os.path.join(embedding_path, pid_foldername, night_file)
        embedding = get_embedding(night_path)
        with torch.no_grad():
            output = model[0].fc2(model[0].relu(model[0].fc1(model[0].encoder(embedding)))) # embedding of embedding
        output = output.cpu().numpy().flatten()
        raw_outputs[pid][j] = output
        

pca_reduce = 10

HOW_MANY_PATIENTS = 10000

perplexities = [64]


fig, axs = plt.subplots(6,6, sharex=True, sharey=True, figsize=(19,12)) # FIXME: currently 36 pids in the antidep_dict, but should be two more...

i=0 # 64 dim

with open('/data/scratch/scadavid/projects/data/bce_relu_umap_fit.pkl', 'rb') as file:
    umap_transform_all = pickle.load(file)


df = pd.DataFrame()

for j, pid in enumerate(pids): 
    
    # if j==1:
    #     break

    tsne_x_group = raw_outputs[pid] # get all night predictions for a patient

    group_embedding = umap_transform_all.transform(tsne_x_group)

    df["comp-1"] = group_embedding[:,0]
    df["comp-2"] = group_embedding[:,1]

    Hues = [str(pred) for pred in pred_classes[pid]]
    Palette = {'0':'red', '1':'blue'}

    # # now, save comp-1/2, colors, and hues_pid in csv
    # data = {'tsne_x1': group_embedding[:,0],'tsne_x2': group_embedding[:,1], 'colors': colors, 'pids': hues_pid}
    # wsc_umap_df = pd.DataFrame(data)
    # #bp()
    # wsc_umap_df.to_csv(f'/data/scratch/scadavid/projects/data/csv/bce_relu_antidep_{group}_umap_check_df.csv', index=False)

    patient_meds = antidep_dict[pid]
    sns.scatterplot(ax=axs.flatten()[j], x="comp-1", y="comp-2", legend='full',
                    data=df, linewidth=0, s=8, hue=Hues, palette=Palette)
    axs.flatten()[j].set_title(f'{pid}; {patient_meds}', fontsize=5)
    #axs.flatten()[j].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs.flatten()[j].get_xaxis().set_visible(False)
    axs.flatten()[j].get_yaxis().set_visible(False)
    axs.flatten()[j].get_legend().remove()

    df.drop(df.index, inplace=True) # otherwise df dim mismatch when assigning values to it again

plt.savefig(os.path.join(ROOT_DIR, 'figures', 'umap', "udall_patient_umaps.pdf"))

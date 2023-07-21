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
from model import BranchHYPredictor, EEG_Encoder, BranchVarEncoder, BranchVarPredictor, StagePredictorModel, SimpleAttentionPredictor
from sklearn.model_selection import KFold
from dataset import *
import argparse
parser = argparse.ArgumentParser()

ROOT_DIR = "/data/scratch/scadavid/projects/data"
EXPERIMENT = "checkpoints/age_pred/shhs1_mayoall"
MODEL = "model_newest.pt"

MODEL_PATH = os.path.join(ROOT_DIR,EXPERIMENT,MODEL)
modeldir = os.path.join(ROOT_DIR,EXPERIMENT)

MAX_NUMBER_PATIENTS = 10000 
BATCH_SIZE = 4
LATENT_SPACE_SIZE = 64



dataloaders = {}

# dataset = UdallBreathingStagesDataset(root_dir='/data/scratch/alimirz/DATA/UDALL_BREATHING_STAGES/')

# labeler = Labeler(dataset = dataset)
def get_val_set(dataset):
    generator1 = torch.Generator().manual_seed(42)
    _, valset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))], generator=generator1)
    return valset
def get_both_set(dataset):
    generator1 = torch.Generator().manual_seed(42)
    trainset, valset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))], generator=generator1)
    return trainset, valset

# if args.multivalset:
#     phases = ['ctrl_train','val_shhs','val_mgh','val_mros','val_sof','val_mayo',]
# else:
phases = ['ctrl_train','ctrl_val']
args = parser.parse_args()
args.data_source = 'eeg'
args.num_channel = 256

# if args.target in ['age', 'hy', 'HY', 'AGE']:
#     args.task = 'regression'
# else:
#     args.task = 'multiclass'
    
# dataset1 = EEG_SHHS_Dataset(target_label='AGE',args=args)
# dataset2 = EEG_MGH_Dataset(args=args)
# dataset3 = EEG_MrOS_Dataset(include_labels=['nm'],args=args)
# dataset4 = EEG_SOF_Dataset(include_labels=['nm'],args=args)
# dataset5 = EEG_MAYO_Dataset(include_labels=['nm'],args=args)

# dataset6 = EEG_SHHS2_Dataset(target_label='AGE',args=args)

# # trainset = DatasetCombiner([dataset1, dataset2, dataset3, dataset4, dataset5],phase='ctrl_train')
# trainset, valset = get_both_set(dataset6)
# dataloaders['ctrl_train'] = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
# dataloaders['ctrl_val'] = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

args.control = True; args.tca = False; args.ssri = False; args.other = False; args.no_attention = False; args.num_classes = 2; args.batch_norms = [False,False,False]
args.layer_dims = [256,64,16]; args.num_heads = 3; args.dropout = 0.5
args.label = 'nsrrid'
dataset = EEG_Encoding_WSC_Dataset(args)
kfold = KFold(n_splits=5, shuffle=True, random_state=20)
#bp()
train_ids, test_ids = [(train_id_set, test_id_set) for (train_id_set, test_id_set) in kfold.split(dataset)][4]
trainset = Subset(dataset, train_ids)
valset = Subset(dataset, test_ids)
dataloaders['ctrl_train'] = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
dataloaders['ctrl_val'] = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
#dataset2 = EEG_Encoding_WSC_Dataset(label='nsrrid')
#trainset, valset = get_both_set(dataset)

##FIXME:: SHHS2 NSRRID LABEL

args.control = False
args.tca = True
wsc_tca_dataset = EEG_Encoding_WSC_Dataset(args)
dataloaders['wsc_tca_val'] = DataLoader(wsc_tca_dataset, batch_size=BATCH_SIZE, shuffle=False)
args.tca = False
args.ssri = True
wsc_ssri_dataset = EEG_Encoding_WSC_Dataset(args)
dataloaders['wsc_ssri_val'] = DataLoader(wsc_ssri_dataset, batch_size=BATCH_SIZE, shuffle=False)
args.ssri = False
args.other = True
wsc_other_dataset = EEG_Encoding_WSC_Dataset(args)
dataloaders['wsc_other_val'] = DataLoader(wsc_other_dataset, batch_size=BATCH_SIZE, shuffle=False)
args.other = False

args.control = True
shhs2_ctrl_dataset = EEG_Encoding_SHHS2_Dataset(args)
dataloaders['shhs2_ctrl_val'] = DataLoader(shhs2_ctrl_dataset, batch_size=BATCH_SIZE, shuffle=False)
args.control = False
args.tca = True
shhs2_tca_dataset = EEG_Encoding_SHHS2_Dataset(args)
dataloaders['shhs2_tca_val'] = DataLoader(shhs2_tca_dataset, batch_size=BATCH_SIZE, shuffle=False)
args.tca = False
args.ntca = True
shhs2_ntca_dataset = EEG_Encoding_SHHS2_Dataset(args)
dataloaders['shhs2_ntca_val'] = DataLoader(shhs2_ntca_dataset, batch_size=BATCH_SIZE, shuffle=False)
args.ntca = False



# if args.multivalset:
#     dataloaders['val_shhs'] = DataLoader(get_val_set(dataset1), batch_size=BATCH_SIZE, shuffle=False)
#     dataloaders['val_mgh'] = DataLoader(get_val_set(dataset2), batch_size=BATCH_SIZE, shuffle=False)
#     dataloaders['val_mros'] = DataLoader(get_val_set(dataset3), batch_size=BATCH_SIZE, shuffle=False)
#     dataloaders['val_sof'] = DataLoader(get_val_set(dataset4), batch_size=BATCH_SIZE, shuffle=False)
#     dataloaders['val_mayo'] = DataLoader(get_val_set(dataset5), batch_size=BATCH_SIZE, shuffle=False)
# else:
#     valset = DatasetCombiner([dataset1, dataset2, dataset3, dataset4, dataset5],phase='ctrl_val')
#     dataloaders['ctrl_val'] = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
###    

#groups = dataloaders.keys()

wsc_groups = ['ctrl_train', 'ctrl_val', 'wsc_tca_val', 'wsc_ssri_val', 'wsc_other_val']
shhs2_groups = ['shhs2_ctrl_val', 'shhs2_tca_val', 'shhs2_ntca_val']
groups = wsc_groups + shhs2_groups

raw_outputs = {}
raw_labels = {}

for group in groups:
    raw_outputs[group] = np.zeros((len(dataloaders[group])*BATCH_SIZE, LATENT_SPACE_SIZE))
    raw_labels[group] = np.zeros(len(dataloaders[group])*BATCH_SIZE, dtype=object)

# raw_outputs['ctrl_train'] = np.zeros((len(dataloaders['ctrl_train'])*BATCH_SIZE, LATENT_SPACE_SIZE))
# raw_outputs['ctrl_val'] = np.zeros((len(dataloaders['ctrl_val'])*BATCH_SIZE, LATENT_SPACE_SIZE))

# raw_labels['ctrl_train'] = np.zeros(len(dataloaders['ctrl_train'])*BATCH_SIZE, dtype=object)
# raw_labels['ctrl_val'] = np.zeros(len(dataloaders['ctrl_val'])*BATCH_SIZE, dtype=object)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def state_dict_rm_module(state_dict, encoder=True, stagepred=False):
#     new_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith('module.'):
#             k = k[len('module.'):]
#         if k.startswith('encoder.'):
#             if encoder:
#                 k = k[len('encoder.'):]
#             else:
#                 continue
#         if k.startswith('stagepreder'):
#             if stagepred:
#                 k = k[len('stagepreder.'):]
#             else:
#                 continue
#         new_dict[k] = v
#     return new_dict

# eeg_encoder = EEG_Encoder().to(device)
# # stage_pred_model = StagePredictorModel().to(device)
# age_encoder = BranchVarEncoder(args).to(device)

# age_predictor = BranchHYPredictor(args).to(device)

# age_pred_model = nn.Sequential(age_encoder, age_predictor) 

# instantiate model and load weights
#shhs2_model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/lr_0.001_w1_1.0_w2_14.0_posf1_0.6.pt"
#wsc_model_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/lr_0.001_w1_1.0_w2_14.0_posf1_0.68.pt"

##FIXME::: to best model once it's done training
wsc_happysadmodel_path = "/data/scratch/scadavid/projects/data/models/encoding/wsc/eeg/dep/class_2/lr_0.0004_w_1.0,10.0_bs_16_f1macro_0.57_256,64,16_bns_0,0,0_heads3_0.5_att_ctrl_fold4.pt"
model = SimpleAttentionPredictor(args).to(device)
state_dict = torch.load(wsc_happysadmodel_path)
model.load_state_dict(state_dict)


# modeldir = '/data/scratch/alimirz/2023/EEG_TIMESERIES_CLASSIFICATION/checkpoints/age_pred/lr1e-4_bs4_SHHS1_MGH_MROSn_SOFn_MAYOn'

# ckpt = torch.load(os.path.join(modeldir,'eeg_encoder_model_newest.pt'))
# eeg_encoder.load_state_dict(state_dict_rm_module(ckpt['model'], encoder=True, stagepred=False))
# ckpt = torch.load(os.path.join(modeldir,'age_pred_model_model_newest.pt'))
# age_pred_model.load_state_dict(state_dict_rm_module(ckpt['model'], encoder=False, stagepred=True))
# for param in eeg_encoder.parameters():
#     param.requires_grad = False

# eeg_encoder = nn.DataParallel(eeg_encoder)
# # stage_pred_model = nn.DataParallel(stage_pred_model)
# age_pred_model = nn.DataParallel(age_pred_model)
# models = {'eeg_encoder':eeg_encoder, 'age_pred_model': age_pred_model}




patient_dict = {}

for group in groups:
    i = 0
    patient_i = 0
    
    for data, labels in tqdm(dataloaders[group]):
        data = data.to(device)
        labels = np.array(labels)
        batch_labels = labels
        
        if labels.shape[0] < BATCH_SIZE:
            continue 
        
        if patient_i >= MAX_NUMBER_PATIENTS:
            break 

        #bp()
        for j in range(BATCH_SIZE):
            patient_i += 1
            
            eeg = data[j:j+1].to(device)
            labels = batch_labels[j]
            
            if labels not in patient_dict:
                patient_dict[labels] = True
            
            with torch.set_grad_enabled(False):
                #bp()
                eeg_output = model.attention(eeg)
                age_output1 = model.fc2(eeg_output) # output is 64 dims
                
                # features, code, pred_class = model(breathing, stages)
                raw_outputs[group][i] = age_output1.cpu().numpy().flatten()
                raw_labels[group][i] = labels
                i += 1
    raw_outputs[group] = raw_outputs[group][:i]
    raw_labels[group] = raw_labels[group][:i]
    print(len(raw_outputs[group]))


# patient_dict = {}

# for phase in ['ctrl_train', 'ctrl_val', ]:
#     i = 0
#     patient_i = 0
    
#     for data, labels in tqdm(dataloaders[phase]):
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
#                 eeg_output = model.attention(eeg)
#                 age_output1 = model.fc2(eeg_output) # output is 64 dims
                
#                 # features, code, pred_class = model(breathing, stages)
#                 raw_outputs[phase][i] = age_output1.cpu().numpy().flatten()
#                 raw_labels[phase][i] = labels
#                 i += 1
#     raw_outputs[phase] = raw_outputs[phase][:i]
#     raw_labels[phase] = raw_labels[phase][:i]
#     print(len(raw_outputs[phase]))


# assert 0 not in raw_labels['ctrl_train']
# assert 0 not in raw_labels['ctrl_val']
pca_reduce = 10
#bp()

# tsne_y = df['label'] #['label'] #df['label_pd_dep']

#for the labels, not to be changed. just tossing missing labels
# tsne_x = tsne_x[tsne_y >= 0]
# tsne_y = tsne_y[tsne_y >= 0]


#from sklearn.decomposition import PCA

#pca = PCA(n_components=pca_reduce)
#tsne_x_train = pca.fit_transform(raw_outputs['ctrl_train'])
#tsne_x_val = pca.transform(raw_outputs['ctrl_val'])




HOW_MANY_PATIENTS = 10000

# # idx_selected_patients = raw_labels['ctrl_train'] < HOW_MANY_PATIENTS
# raw_outputs['ctrl_train'] = raw_outputs['ctrl_train']
# raw_labels['ctrl_train'] = raw_labels['ctrl_train']
# # idx_selected_patients = raw_labels['ctrl_val'] < HOW_MANY_PATIENTS
# raw_outputs['ctrl_val'] = raw_outputs['ctrl_val']
# raw_labels['ctrl_val'] = raw_labels['ctrl_val']


# tsne_x_train = raw_outputs['ctrl_train']
# tsne_x_val = raw_outputs['ctrl_val']

#perplexities = [len(tsne_x_val) // 40, len(tsne_x_val) // 8, len(tsne_x_val) // 2]
perplexities = [64]
#bp()


# all_plots = []
# for i in range(len(all_lbls)):
#     all_plots.append( plt.subplots(2,2, figsize=(12,10)))
fig, axs = plt.subplots(2,4, figsize=(18,10))

if True:
    i=0 # 64 dim
    umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean')
    umap_embeddings_train = umap_model.fit_transform(tsne_x_train)
    umap_embeddings_val = umap_model.transform(tsne_x_val)
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    
    # df2["y"] = tsne_y
    df_train["comp-1"] = umap_embeddings_train[:,0]
    df_train["comp-2"] = umap_embeddings_train[:,1]
    df_val["comp-1"] = umap_embeddings_val[:,0]
    
    df_val["comp-2"] = umap_embeddings_val[:,1]
    
    for j, phase in enumerate(['ctrl_train','ctrl_val']):
        # ax = all_plots[j][1][i//2,i%2]
        df = df_train if phase == 'ctrl_train' else df_val
        hues_pid = np.array(raw_labels[phase].tolist())
        #bp()
        hues_gender = [str(dataset.get_label_from_filename(item)) for item in hues_pid]
        hues_sadbinary = []
        for item in hues_pid:
            try:
                hues_sadbinary.append(dataset.get_happysad_from_filename(item))
            except:
                hues_sadbinary.append(-1)
                print('problem')
        # hues_age = [str(int(labeler.get_age(int(item)))) for item in hues_pid]
        # hues_pd = [str(labeler.get_pd(item)) for item in hues_pid]
        # bp()
        # hues_pidd = [str(int(item)) for item in hues_pid]
        #sns.set_palette("viridis")
        Palette = {'0':'lightgrey', '1':'red'}
        # sns.scatterplot(ax=axs[j,i], x="comp-1", y="comp-2", hue=hues_gender, palette=Palette, legend='full',
        #                 data=df, linewidth=0, s=8).set(title=phase+" UMAP projection, Perplexity: "+str(perplexities[i]))
        # axs[j,i].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        # axs[j,i].get_xaxis().set_visible(False)
        # axs[j,i].get_yaxis().set_visible(False)
        # axs[j,i].get_legend().remove()
        sns.scatterplot(ax=axs[j], x="comp-1", y="comp-2", hue=hues_gender, palette=Palette, legend='full',
                        data=df, linewidth=0, s=8).set(title=phase+" UMAP projection, Perplexity: "+str(perplexities[i]))
        axs[j].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        axs[j].get_xaxis().set_visible(False)
        axs[j].get_yaxis().set_visible(False)
        axs[j].get_legend().remove()
        #bp()
else:
    fig, axs = plt.subplots(2,4, figsize=(18,10))

    def split_array(A, B):
        groups = []
        indices = [0] + [i + 1 for i in range(len(B) - 1) if B[i] != B[i + 1]] + [len(A)]
        groups = [A[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]
        return groups

    for i in range(len(perplexities)):
        
        
        tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=perplexities[i])
        z_train = tsne.fit_transform(tsne_x_train)
        z_val = tsne.fit_transform(tsne_x_val)
        
        df_train = pd.DataFrame()
        df_val = pd.DataFrame()
        
        # df2["y"] = tsne_y
        df_train["comp-1"] = z_train[:,0]
        df_train["comp-2"] = z_train[:,1]
        df_val["comp-1"] = z_val[:,0]
        df_val["comp-2"] = z_val[:,1]
        
        for j, phase in enumerate(['ctrl_train','ctrl_val']):
            # ax = all_plots[j][1][i//2,i%2]
            df = df_train if phase == 'ctrl_train' else df_val
            hues_pid = np.array(raw_labels[phase].tolist())
            # hues_gender = [labeler.get_gender(int(item)) for item in hues_pid]
            # hues_age = [str(int(labeler.get_age(int(item)))) for item in hues_pid]
            # hues_pidd = [int(item) for item in hues_pid]
            # hues_gender = [dataset.get_label_from_filename(item) for item in hues_pid]
            
            hues_sadbinary = []
            for item in hues_pid:
                try:
                    hues_sadbinary.append(dataset.get_happysad_from_filename(item))
                except:
                    hues_sadbinary.append(-1)
                    print('problem')

            # hues_sadscore = [dataset.get_happysad_from_filename(item) for item in hues_pid]

            sns.set_palette("viridis")
            sns.scatterplot(ax=axs[j,i], x="comp-1", y="comp-2", hue=hues_sadbinary, legend='full',
                            data=df, linewidth=0, s=8).set(title=phase+" T-SNE projection, Perplexity: "+str(perplexities[i]))
            # comp_1_subgroups = split_array(df["comp-1"],hues_pidd)
            # comp_2_subgroups = split_array(df["comp-2"],hues_pidd)
            # for k in range(len(comp_1_subgroups)):
            #     axs[j,i].plot(comp_1_subgroups[k],comp_2_subgroups[k],label=hues_pidd[k])
            
            axs[j,i].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            axs[j,i].get_xaxis().set_visible(False)
            axs[j,i].get_yaxis().set_visible(False)
            axs[j,i].get_legend().remove()
        

        key_label = "PPXTY_" + str(perplexities[i])
plt.savefig(os.path.join(ROOT_DIR, 'figures', "umap_control_train_wsc.pdf"))
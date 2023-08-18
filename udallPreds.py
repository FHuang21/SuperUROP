import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model import SimonModel
import os
from ipdb import set_trace as bp

def get_date_from_filename(filename):
    # Extract the date part from the filename
    date_part = filename.split('.')[0].split('-')
    return ''.join(date_part)

def get_ordered_nights_list(nights_dir):
    unordered_nights_list = os.listdir(nights_dir)
    return sorted(unordered_nights_list, key=get_date_from_filename, reverse=False)

def get_predicted_prob(model, np_file_path):
    file = np.load(np_file_path)
    feature = file['decoder_eeg_latent'].squeeze(0)
    if feature.shape[0] >= 150:
        feature = feature[:150, :]
    else:
        feature = np.concatenate((feature, np.zeros((150-feature.shape[0],feature.shape[-1]),dtype=np.float32)), axis=0)
    feature = torch.from_numpy(feature)
    #bp()
    feature = torch.unsqueeze(feature, 0)
    with torch.no_grad():
        output = torch.sigmoid(model(feature))
    output = output.detach().numpy()
    return output[0][0]

class Object(object):
    pass
args = Object()

encoding_path = "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/udall/abdominal_c4_m1"
udall_filenames = os.listdir(encoding_path)
pids = [filename.split('data_')[1] for filename in udall_filenames]

# best antidep model (currently)
model_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/exp_lr_0.002_w_1.0,2.5_ds_eeg_bs_16_epochs_2_dpt_0.0_fold0_256,64,16_heads4bce_tuned_relu_081123_final/lr_0.002_w_1.0,2.5_bs_16_heads4_0.0_attbce_tuned_relu_081123_final_epochs2_fold0.pt"

args.label = "antidep"; args.num_classes = 2
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.0 # do these even matter?
#args.no_attention = False; args.label = "antidep"; args.tca = False; args.ntca = False; args.ssri = False; args.other = False; args.control = False
model = SimonModel(args)
fc_end = nn.Linear(2,1)
model = nn.Sequential(model, nn.ReLU(), fc_end)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()
threshold = 0.2

path = f'/data/scratch/scadavid/projects/data/udall/'
if not os.path.exists(path):
    os.makedirs(path)

new_files = ['PD_Hao_data_NIHND126MXDGP', 'PD_Hao_data_NIHBE740TFYAH', 'PD_Hao_data_NIHPT334YGJLK', 'PD_Hao_data_NIHDW178UFZHB', 'PD_Hao_data_NIHHD991PGRJC', 'PD_Hao_data_NIHFW795KLATW']

#for filename in udall_filenames:
for filename in new_files: # filenames are really folder names
    night_dir = os.path.join(encoding_path, filename)
    ordered_night_files = get_ordered_nights_list(night_dir)
    
    # for night_file in ordered_night_files:
    #     prob = get_predicted_prob(model, night_file)
    probs = [get_predicted_prob(model, os.path.join(encoding_path, filename, night_file)) for night_file in ordered_night_files]
    binary_preds = [1 if get_predicted_prob(model, os.path.join(encoding_path, filename, night_file))>=.2 else 0 for night_file in ordered_night_files]

    patient_dict = {'night': ordered_night_files, 'prob': probs, 'preds': binary_preds}
    current_df = pd.DataFrame(patient_dict)
    current_df.to_csv(os.path.join(path, f'{filename}.csv')) # filename is really udall patient id



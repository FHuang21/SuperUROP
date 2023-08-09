import torch
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import numpy as np
from ipdb import set_trace as bp


def process_stages(stages):
        stages[stages < 0] = 0
        stages[stages > 5] = 0
        stages = stages.astype(int)
        mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
        return mapping[stages]

class Object(object):
    pass
args = Object()

args.no_attention = False; args.label = "nsrrid"; args.tca = False; args.ntca = False; args.other = False; args.control = False; args.ssri = False
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.5

model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/ali_best/lr_0.0002_w_1.0,14.0_bs_16_f1macro_0.72_256,64,16_bns_0,0,0_heads4_0.5_att_alibest_fold0_epoch29.pt"
state_dict = torch.load(model_path)
model = SimonModel(args)
model.load_state_dict(state_dict) # trained on shhs2 fold0
model.eval()

# wsc dataset
wsc_dataset = EEG_Encoding_WSC_Dataset(args)
wsc_dataloader = DataLoader(wsc_dataset, batch_size=1, shuffle=False)

wsc_stage_path = "/data/netmit/wifall/ADetect/data/wsc_new/stage"

# get attentions and sleep stages
attention_avgs = []
#labels = []
stages = []
### get 20 most confident true positive predictions. first get all true positives
tp_pids = []
y_pos_preds = []
softmax = torch.nn.Softmax(dim=1)
with torch.no_grad():
    for idx, (X, y) in enumerate(wsc_dataloader): # can change y to nsrrid to plot along with it
        label = y[0]

        y_pred = torch.argmax(model(X), dim=1).item()
        y_true = wsc_dataset.get_label_from_filename(label)
        if y_pred==1 and y_true==1:
            tp_pids.append(label)
            y_pos_preds.append(softmax(model(X))[0][1].item())
        else:
            continue

        #labels.append(label)
        stage = np.load(os.path.join(wsc_stage_path, label))['data']
        stage = process_stages(stage)

        attentions = [model.encoder.softmax(model.encoder.query_layer[i](X)) for i in range(model.num_heads)]
        attention_avg = (attentions[0] + attentions[1] + attentions[2] + attentions[3]) / 4
        attention_avg = attention_avg.squeeze().detach().numpy()
        
        #bp()
        t=len(attention_avg)-1
        while(attention_avg[t]==attention_avg[t-1]):
            attention_avg[t] = 0
            #attention_avg = attention_avg[:t]
            t -= 1
        
        #bp()
        attention_avg = np.repeat(attention_avg, 8)
        length_diff = len(attention_avg) - len(stage)
        if length_diff < 0:
            stage = stage[:1200] #FIXME later
        else:
            stage = np.pad(stage, (0, length_diff), mode='constant', constant_values=0)
        stages.append(stage)
        attention_avgs.append(attention_avg)

bp()
sorted_pairs = sorted(zip(y_pos_preds, tp_pids, stages, attention_avgs), reverse=True)
sorted_y_pos_preds, sorted_tp_pids, sorted_stages, sorted_attentions = zip(*sorted_pairs)
#most_confident_pids = sorted_tp_pids[:20]

fig, axs = plt.subplots(10, 4, figsize=(24,16))

for i in range(0, 10, 2):
    for j in range(0, 4):
        idx = i*4 + j

        #pid = most_confident_pids[idx]
        axs[i,j].set_title(sorted_tp_pids[idx])
        axs[i,j].plot(sorted_stages[idx])
        axs[i,j].get_xaxis().set_visible(False)
        axs[i,j].get_yaxis().set_visible(False)
        axs[i+1,j].plot(sorted_attentions[idx])
        axs[i+1,j].get_xaxis().set_visible(False)
        axs[i+1,j].get_yaxis().set_visible(False)

plt.suptitle('20 Most Confident True Positive Examples')
plt.savefig('/data/scratch/scadavid/projects/data/figures/wsc_20_tp_stage_attentions_most_confidentv3.pdf')
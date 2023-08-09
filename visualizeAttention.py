import torch
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, Subset
#from utils_plotter import MageAttentionPlotter
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from ipdb import set_trace as bp
import numpy as np

def min_max_normalization(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

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

kfold = KFold(n_splits=5, shuffle=True, random_state=20)
shhs2_dataset = EEG_Encoding_SHHS2_Dataset(args)
train_ids, test_ids = [(train, test) for (train, test) in kfold.split(shhs2_dataset)][0]
shhs2_valset = Subset(shhs2_dataset, test_ids)
shhs2_dataloader = DataLoader(shhs2_dataset, batch_size=1, shuffle=False)
wsc_dataset = EEG_Encoding_WSC_Dataset(args)
wsc_dataloader = DataLoader(wsc_dataset, batch_size=1, shuffle=False)

### altogether ###
features = []
attention_avgs = []
with torch.no_grad():
    for idx, (X, y) in enumerate(wsc_dataloader): # can change y to nsrrid to plot along with it
        feature = X.squeeze().detach().numpy()
        #feature = min_max_normalization(feature) * 255
        features.append(feature)
        attentions = [model.encoder.softmax(model.encoder.query_layer[i](X)) for i in range(model.num_heads)]
        #attention_avg = (attentions[0] + attentions[1] + attentions[2] + attentions[3]) / 4
        attention_avg = attentions[0] # Hao says don't average them, just treat individually
        #attention_avg = np.array(attention_avg)
        # t=len(attention_avg-1)
        # while(attention_avg[t]==attention_avg[t-1]):
        #     attention_avg = attention_avg[0:t-1]
        #     t -= 1
        attention_avg = attention_avg.squeeze().detach().numpy()
        #attention_avg = min_max_normalization(attention_avg) # ali says normalize each individual
        attention_avgs.append(attention_avg)

#         # pred = model.encoder(X)#.detach().numpy()
#         # y_att_outputs.append(pred)
#         #num_pos += (1 if pred==1 else 0)
#         #y = y.detach().numpy()
#         #y_true.append(y)
#         if (idx == 15): #FIXME
#             break

### separate control / medicated ###

# wsc_data_dict = wsc_dataset.data_dict
# features = []
# ctrl_attention_avgs = []
# med_attention_avgs = []
# with torch.no_grad():
#     for idx, (X, y) in enumerate(wsc_dataloader): # can change y to nsrrid to plot along with it
#         feature = X.squeeze().detach().numpy()
#         #feature = min_max_normalization(feature) * 255
#         features.append(feature)
#         attentions = [model.encoder.softmax(model.encoder.query_layer[i](X)) for i in range(model.num_heads)]
#         attention_avg = (attentions[0] + attentions[1] + attentions[2] + attentions[3]) / 4
#         attention_avg = attention_avg.squeeze().detach().numpy()
#         #attention_avg = min_max_normalization(attention_avg) # ali says normalize each individual

#         #print(attention_avg[20:])
#         t=len(attention_avg)-1
#         while(attention_avg[t]==attention_avg[t-1]):
#             #bp()
#             #attention_avg = attention_avg[0:t]
#             attention_avg[t] = 0
#             t -= 1
#         #bp()
#         attention_avg = np.exp(attention_avg)/sum(np.exp(attention_avg))
#         label = y[0]
#         if (wsc_data_dict[label][0] == 1):
#             med_attention_avgs.append(attention_avg)
#         else:
#             ctrl_attention_avgs.append(attention_avg)

#         # pred = model.encoder(X)#.detach().numpy()
#         # y_att_outputs.append(pred)
#         #num_pos += (1 if pred==1 else 0)
#         #y = y.detach().numpy()
#         #y_true.append(y)
#         # if (idx == 15): #FIXME
#         #     break

# ### plot average across two groups stuff ###
#bp()
# ctrl_attention_avgs = np.array(ctrl_attention_avgs)
# ctrl_attention_avg_all = np.mean(ctrl_attention_avgs, axis=0)
# med_attention_avgs = np.array(med_attention_avgs)
# med_attention_avg_all = np.mean(med_attention_avgs, axis=0)

# plt.plot(ctrl_attention_avg_all, color='blue', label='control')
# plt.plot(med_attention_avg_all, color='red', label='taking antidepressant')
# plt.xlabel('Time')
# plt.ylabel('Attention Weight')
# plt.title('Averaged Attention Weight across all Examples in Control and Med Groups')

# # Add a legend to distinguish between the two arrays
# plt.legend()

# plt.savefig('/data/scratch/scadavid/projects/data/figures/avg_attention_head0_wsc_all_noatt_to_padd.pdf')


### ssri stuff ###
# # Create a 16x8 grid of subplots
# fig, axs = plt.subplots(4, 8, figsize=(16, 8), gridspec_kw={'wspace': 0.3, 'hspace': 0.3}) #FIXME

# # Loop over each subplot and populate it with the two plots
# for i in range(0, 4, 2): #FIXME
#     for j in range(0, 8):
#         idx = i*4 + j #FIXME
#         #bp()
#         # Create a two-dimensional random image for the first plot
#         image = features[idx] # need to normalize these

#         # Create a one-dimensional random signal for the second plot
#         att = attention_avgs[idx]

#         # Plot the image in the top subplot
#         axs[i, j].imshow(image, cmap='gray', vmin=0, vmax=255)
#         #axs[i, j].set_title(f'Feature {i}x{j}') # FIXME :: NSRRID

#         # Plot the signal in the bottom subplot
#         axs[i+1, j].plot(att)
#         #axs[i+1, j].set_xlabel('Time')
#         #axs[i+1, j].set_ylabel('Attention Weight')

# # Add a title for the entire plot
# plt.suptitle('Mage Embedding and Attention Head 0 Weight for 16 Control Patients', fontsize=16)

# # Adjust the layout and spacing between subplots
# #plt.tight_layout()

# # Show the plot
# plt.show()

# # Save the plot
# plt.savefig("/data/scratch/scadavid/projects/data/figures/ssri_attention_test_nonorm_head0_noatt_to_padd.pdf")
###


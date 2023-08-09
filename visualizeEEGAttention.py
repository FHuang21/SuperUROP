import torch
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import numpy as np
from ipdb import set_trace as bp

# def get_spec(eeg_file):  # adapt from Haoqi
#     import mne
#     # eeg.shape = (#channel, #point)
#     Fs = eeg_file['fs']  # 200  # [Hz]
#     # preprocessing by filtering
#     eeg = eeg_file['data'].astype(np.float64)

#     # s = eeg_file['data'].std()
#     # st.write(f'std {s}')
#     if eeg_file['data'].std() < 1:
#         eeg *= 1e6
#     if eeg_file['data'].std() > 50:
#         eeg /= 5  # FIXME: hack for mgh
#     eeg = eeg - eeg.mean()  # remove baseline
#     # eeg = mne.filter.notch_filter(eeg, Fs, 60)  # remove line noise, US is 60Hz
#     # eeg = mne.filter.filter_data(eeg, Fs, 0.3, 32)  # bandpass, for sleep EEG it's 0.3-35Hz

#     # first segment into 30-second epochs
#     epoch_time = 30  # [second]
#     step_time = 30
#     epoch_size = int(round(epoch_time * Fs))
#     step_size = int(round(step_time * Fs))

#     # epoch_start_ids = np.arange(0, len(eeg) - epoch_size, epoch_size)
#     # ts = epoch_start_ids / Fs
#     # epochs = eeg.reshape(-1, epoch_size)  # #epoch, size(epoch)

    # def get_seg(eeg, l, r):
    #     if l >= 0 and r <= len(eeg):
    #         return eeg[l:r]
    #     else:
    #         if r > len(eeg):
    #             tmp = eeg[l:]
    #             tmp = np.concatenate([tmp, np.zeros((r - l - len(tmp)))])
    #             return tmp
    #         else:
    #             tmp = eeg[:r]
    #             tmp = np.concatenate([np.zeros((-l)), tmp])
    #             return tmp

    # epochs = np.array([
    #     get_seg(eeg, i + step_size // 2 - epoch_size // 2, i + step_size // 2 + epoch_size // 2)
    #     for i in range(0, len(eeg), step_size)
    # ])

    # spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.0, fmax=32, bandwidth=0.5,
    #                                                      normalization='full', verbose=False,
    #                                                      n_jobs=12)  # spec.shape = (#epoch, #freq)
    # # freq == np.linspace(0,32,961)
    # spec_db = 10 * np.log10(spec)
    # return spec_db.T

def plot_eeg_spec(a, c4m1_path):
    #if c4m1_path.is_file():
    #bp()
    c4m1_file = np.load(c4m1_path)
    eeg_spec = c4m1_file['signal']
    tmp = eeg_spec
    a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30,
                    np.linspace(0, 32, tmp.shape[0]),
                    tmp,
                    vmin=-10, vmax=15,
                    antialiased=True,
                    shading='auto',
                    cmap='jet')
    # else:
    #     tmp = 0
    #     c4m1_file = {'data': 0}

    eeg_y_max = 32
    a.set_title(f'Multitaper EEG {c4m1_path}')
    a.set_ylabel('Hz')
    a.set_ylim([0, eeg_y_max])
    a.set_yticks(list(np.arange(0, eeg_y_max, 4)), list(np.arange(0, eeg_y_max, 4)))
    a.grid()
    return tmp

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

eeg_mt_path = "/data/netmit/wifall/ADetect/data/wsc_new/c4_m1_multitaper"
eeg_spec_path = "/data/netmit/wifall/ADetect/data/wsc_new/c4_m1_spec"
all_eeg_mts = os.listdir(eeg_mt_path)

features = []
#eegs = []
attention_avgs = []
labels = []
with torch.no_grad():
    for idx, (X, y) in enumerate(wsc_dataloader): # can change y to nsrrid to plot along with it
        #bp()
        label = y[0]
        labels.append(label)
        # eeg_mt_file = np.load(os.path.join(eeg_mt_path, label)) # y is pid
        # eeg_mt = eeg_mt_file['data']

        # eegs.append(eeg_mt)

        # feature = X.squeeze().detach().numpy()
        # features.append(feature)
        attentions = [model.encoder.softmax(model.encoder.query_layer[i](X)) for i in range(model.num_heads)]
        #attention_avg = (attentions[0] + attentions[1] + attentions[2] + attentions[3]) / 4
        attention_avg = attentions[0] # Hao says don't average them, just treat individually
        attention_avg = attention_avg.squeeze().detach().numpy()
        
        #bp()
        t=len(attention_avg)-1
        while(attention_avg[t]==attention_avg[t-1]):
            attention_avg[t] = 0
            t -= 1
        attention_avgs.append(attention_avg)

# Create a 16x8 grid of subplots
fig, axs = plt.subplots(4, 4) #FIXME

# Loop over each subplot and populate it with the two plots
flag = False
for i in range(0, 4): #FIXME
    if flag == True:
        break
    for j in range(0, 4):
        #idx = i*4 + j #FIXME
        idx = i*4 + j
        if (idx == 1):
            flag = True
            break
        #bp()
        # Create a two-dimensional random image for the first plot
        #image = features[idx] # need to normalize these
        #current_eeg_mt = eegs[idx]
        filepath = os.path.join(eeg_spec_path, labels[idx]) # labels[idx] is current pid
        plot_eeg_spec(axs[i,j], filepath) 

        # Create a one-dimensional random signal for the second plot
        att = attention_avgs[idx]
        att = np.repeat(att, 8) # upsample
        att = np.expand_dims(att, axis=0)
        att = np.repeat(att, 256, axis=0)

        #bp()
        # att is already on padded embedding, so need to pad eeg_mt to match that
        # length_diff = att.shape[1] - current_eeg_mt.shape[1]
        # if length_diff > 0:
        #     current_eeg_mt = np.pad(current_eeg_mt, [(0, 0), (0, length_diff)], mode='constant', constant_values=0)

        # ok, now both att and current_eeg_mt are same dimensions.
        
        #bp()
        # softmax eeg columns
        # FIXME: currently not softmaxing along every column???
        #current_eeg_mt_softmaxed = np.exp(current_eeg_mt - np.max(current_eeg_mt, axis=0, keepdims=True))
        # current_eeg_mt_softmaxed = np.exp(current_eeg_mt - np.repeat(np.max(current_eeg_mt, axis=0, keepdims=True),256,axis=0)) # try this?
        # current_eeg_mt_softmaxed /= np.sum(current_eeg_mt_softmaxed, axis=0, keepdims=True)
        #bp()
        #eeg_masked = np.multiply(current_eeg_mt, att)

        # eeg_masked /= np.max(eeg_masked)
        # eeg_masked *= 255

        #bp()

        # Plot the image in the top subplot
        # axs[i, j].imshow(eeg_masked, cmap='gray', vmin=0, vmax=255) # FIXME : vmin / vmax ???
        # axs[i, j].pcolormesh(np.arange(current_eeg_mt.shape[1]) / 30 * 30,
        #              np.linspace(0, 32, current_eeg_mt.shape[0]),
        #              current_eeg_mt,
        #              antialiased=True,
        #              shading='auto',
        #              cmap='jet')
        # axs[i, j].imshow(current_eeg_mt) # FIXME : vmin / vmax ???
        #axs[i, j].set_title(f'Feature {i}x{j}') # FIXME :: NSRRID

        # Plot the signal in the bottom subplot
        #axs[i+1, j].plot(att)
        #axs[i+1, j].set_xlabel('Time')
        #axs[i+1, j].set_ylabel('Attention Weight')

# Add a title for the entire plot
plt.suptitle('Some Attention Softmasked EEG Multitaper Specs', fontsize=10)

# Adjust the layout and spacing between subplots
#plt.tight_layout()

# Show the plot
plt.show()

# Save the plot
plt.savefig("/data/scratch/scadavid/projects/data/figures/some_att_softmasked_eegs_test.pdf")
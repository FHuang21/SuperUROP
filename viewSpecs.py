import torch
from torch.utils.data import DataLoader
from dataset import EEG_Encoding_WSC_Dataset
from model import SimonModel
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from utils_plotter import MageAttentionPlotter
from ipdb import set_trace as bp

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

# # now, need list of 20 true positive pids and 20 false negative pids
# tp_pids = []
# fn_pids = []
# for X, y in wsc_dataloader:
#     #bp()
#     pid = y[0]
#     y_pred = torch.argmax(model(X), dim=1).item()
#     y_true = wsc_dataset.get_label_from_filename(pid)
#     if y_pred==1 and y_true==1:
#         tp_pids.append(pid)
#     elif y_pred==0 and y_true==1:
#         fn_pids.append(pid)

#     if len(tp_pids)==20 and len(fn_pids)==20:
#         break

### get 20 most confident true negative predictions
tn_pids = []
y_pos_preds = []
softmax = torch.nn.Softmax(dim=1)
for X, y in wsc_dataloader:
    #bp()
    pid = y[0]
    y_pred = torch.argmax(model(X), dim=1).item()
    y_true = wsc_dataset.get_label_from_filename(pid)
    if y_pred==0 and y_true==0:
        tn_pids.append(pid)
        #bp()
        y_pos_preds.append(softmax(model(X))[0][0].item())

    # if len(tp_pids)==20 and len(fn_pids)==20:
    #     break

#bp()
sorted_pairs = sorted(zip(y_pos_preds, tn_pids), reverse=True)
sorted_y_neg_preds, sorted_tn_pids = zip(*sorted_pairs)
most_confident_pids = sorted_tn_pids[:20]

###

eeg_mt_img_path = "/data/netmit/wifall/ADetect/data/wsc_new/c4_m1_multitaper_img"
#eeg_mt_imgs = os.listdir(eeg_mt_img_path)

fig, axs = plt.subplots(4, 5, figsize=(24,16))
for j, pid in enumerate(most_confident_pids):
    img_filename = pid.replace('.npz', '.jpg')
    mt = mpimg.imread(os.path.join(eeg_mt_img_path, img_filename))
    bp()
    axs.flatten()[j].imshow(mt)
    #bp()
    axs.flatten()[j].get_xaxis().set_visible(False)
    axs.flatten()[j].title.set_text(pid.split('.')[0])

    if j==19:
        break

fig.align_ylabels

plt.suptitle('20 Most Confident True Negative Patient Spectrograms', fontsize=40)

plt.tight_layout()

plt.savefig('/data/scratch/scadavid/projects/data/figures/wsc_20_tn_antidep_specs_most_confident_attended.pdf', dpi=400)
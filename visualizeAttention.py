import torch
from model import SimonModel
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from ipdb import set_trace as bp

class Object(object):
    pass
args = Object()

args.no_attention = False; args.label = "antidep"; args.tca = False; args.ntca = False; args.ssri = True; args.other = False; args.control = False
args.num_heads = 3; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2

model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/simonmodelantidep/lr_0.0001_w_1.0,14.0_bs_16_f1macro_-1.0_256,64,16_bns_0,0,0_heads3_0.5_attsimonmodelv2_fold0_epoch24.pt"
state_dict = torch.load(model_path)
model = SimonModel(args)
model.load_state_dict(state_dict) # trained on shhs2 fold0

kfold = KFold(n_splits=5, shuffle=True, random_state=20)
shhs2_dataset = EEG_Encoding_SHHS2_Dataset(args)
train_ids, test_ids = [(train, test) for (train, test) in kfold.split(shhs2_dataset)][0]
shhs2_valset = Subset(shhs2_dataset, test_ids)
shhs2_dataloader = DataLoader(shhs2_dataset, batch_size=1, shuffle=False)
wsc_dataset = EEG_Encoding_WSC_Dataset(args)

features = []
attention_avgs = []
with torch.no_grad():
    for idx, (X, y) in enumerate(shhs2_dataloader): # can change y to nsrrid to plot along with it
        #bp()
        features.append(X.squeeze().detach().numpy())
        attentions = [model.encoder.softmax(model.encoder.query_layer[i](X)) for i in range(model.num_heads)]
        attention_avg = (attentions[0] + attentions[1] + attentions[2]) / 3
        attention_avg = attention_avg.squeeze().detach().numpy()
        attention_avgs.append(attention_avg)

        # pred = model.encoder(X)#.detach().numpy()
        # y_att_outputs.append(pred)
        #num_pos += (1 if pred==1 else 0)
        #y = y.detach().numpy()
        #y_true.append(y)
        if (idx == 67):
            break

#bp()

# Create an 8x8 grid of subplots
fig, axs = plt.subplots(16, 8, figsize=(12, 12))

# Loop over each subplot and populate it with the two plots
for i in range(8):
    for j in range(8):
        idx = i*8 + j

        # Create a two-dimensional random image for the first plot
        image = features[idx]

        # Create a one-dimensional random signal for the second plot
        att = attention_avgs[idx]

        # Plot the image in the top subplot
        axs[i, j].imshow(image, cmap='gray')
        axs[i, j].set_title(f'Feature {i}x{j}') # FIXME :: NSRRID

        # Plot the signal in the bottom subplot
        axs[i, j].plot(att)
        axs[i, j].set_xlabel('Time')
        axs[i, j].set_ylabel('Attention Weight')

# Add a title for the entire plot
plt.suptitle('Mage Embedding and Avg. Attention Weights for 64 SSRI Patients', fontsize=16)

# Adjust the layout and spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Save the plot
plt.savefig("/data/scratch/scadavid/projects/data/figures/ssri_attention_64.pdf")
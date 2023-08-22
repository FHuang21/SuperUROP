import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
# from utils_plotter import *
from model import SimonModel
from pathlib import Path
import os
from sklearn.decomposition import PCA

#local_mnt = '/Users/dina/mnt3/' # FIXME:::::
#local_mnt = '/data'
mage_path = '/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/wsc_new/abdominal_c4_m1/'
multitaper_path = '/data/netmit/wifall/ADetect/data/wsc_new/c4_m1_multitaper/'
stage_path = '/data/netmit/wifall/ADetect/data/wsc_new/stage/'
idx, ax_red, fig, z, ax_eeg, fig_eeg, uid_list = None, None, None, None, None, None, None

# model for attentions:
class Object(object):
    pass
args = Object()
args.no_attention = False; args.label = "antidep"; args.tca = False; args.ntca = False; args.other = False; args.control = False; args.ssri = False
args.num_heads = 4; args.hidden_size = 8; args.fc2_size = 32; args.num_classes = 2; args.dropout = 0.5
model_path = "/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/ali_best/lr_0.0002_w_1.0,14.0_bs_16_f1macro_0.72_256,64,16_bns_0,0,0_heads4_0.5_att_alibest_fold0_epoch29.pt"
state_dict = torch.load(model_path)
model = SimonModel(args)
model.load_state_dict(state_dict) # trained on shhs2 fold0
model.eval()

## code to create the pca transform
hsv_cmap = plt.get_cmap('hsv')
aa = np.load('/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/shhs2_new/abdominal_c4_m1/shhs2-205688.npz')
aa = aa['decoder_eeg_latent']
n_components = 3
pca = PCA(n_components=n_components)
pca.fit(aa[0])
reduced_data = pca.transform(aa[0])
kinda_min = np.min(reduced_data) - 2
kinda_max = np.max(reduced_data) + 5

def plot_eeg(tmp, a, uid, name='EEG'):
    a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30,
                 np.linspace(0, 32, tmp.shape[0]),
                 tmp,
                 vmin=-10, vmax=15,
                 antialiased=True,
                 shading='auto',
                 cmap='jet')
    a.set_title(f'{name} {uid}')
    a.set_ylabel('Hz')
    a.set_ylim([0, 32])
    a.set_yticks(list(np.arange(0, 32, 4)), list(np.arange(0, 32, 4)))
    a.grid()

def plot_stage(stage, a, uid):
    #stage = stage[::30]
    stage = stage.astype(int)
    idx_invalid = (stage < 0) | (stage > 5)
    stage_remap = np.array([0, 2, 2, 3, 3, 1] + [np.nan] * 10)
    stage = stage_remap[stage].astype(float)
    stage[idx_invalid] = np.nan

    a.plot(np.arange(len(stage)), stage)
    a.plot(stage)
    a.set_title('stage')
    a.set_xlim(0, len(stage))
    a.set_yticks([0, 1, 2, 3], ['A', 'R', 'L', 'D'])
    #a.set_xticks(np.arange(0, len(stage), 60), np.arange(0, len(stage), 60) // 2) # ?????
    #a.set_xticks(np.arange(0, len(stage), 30))
    a.grid()

def plot_mage(mage, a, uid):
    reduced_data = pca.transform(mage[0])
    reduced_data = reduced_data - kinda_min
    reduced_data = reduced_data / kinda_max
    reduced_data = np.clip(reduced_data, 0, 1)
    for i in range(reduced_data.shape[0]):
        a.add_patch(plt.Rectangle((i, 0), 1, 1, color=reduced_data[i]))

    a.set_xlim(0, reduced_data.shape[0])
    a.set_ylim(0, 1)
    a.axis('off')

def plot_attention(attentions, a, uid):
    for j, attention in enumerate(attentions):
        attention = attention.cpu().detach().numpy()
        offset_attention = attention + j*0.1 # may need to scale down j
        a.plot(offset_attention)
    
    a.set_xlim(0, attentions[0].shape[0])
    a.axis('off')
    

def plot_main(ax_eeg, uid, ssim=0):
    import datetime
    aa = datetime.datetime.now()
    stage = np.load(os.path.join(stage_path, uid))['data']
    plot_stage(stage, ax_eeg[0], uid)
    ab = datetime.datetime.now()
    print('Stage loading and plotting: ', ab - aa)
    eeg_spec = np.load(os.path.join(multitaper_path, uid))['data']
    # eeg_spec = np.load(f'./files/c4_m1_multitaper/{uid}.npz')['data']
    # eeg_mage = np.load(f'./files/c4_m1_mage_256/{uid}.npz')['pred']
    min_len = eeg_spec.shape[1]
    eeg_spec = np.clip(eeg_spec, -10, 15)
    # eeg_mage = np.clip(eeg_mage, -10, 15)
    # eeg_error = np.abs(eeg_spec[:, :min_len] - eeg_mage[:, :min_len])
    ac = datetime.datetime.now()
    print('Multitaper loading: ', ac - ab)
    plot_eeg(eeg_spec, ax_eeg[2], uid, f'{ssim} GT')
    ad = datetime.datetime.now()
    print('EEG plotting: ', ad - ac)

    mage_spec = np.load(os.path.join(mage_path, uid))['decoder_eeg_latent']
    plot_mage(mage_spec, ax_eeg[1], uid)
    # plot_eeg(eeg_mage, ax_eeg[2], uid, 'Mage')
    # plot_eeg(eeg_error, ax_eeg[3], uid, 'Error')
    # eeg_vqgan = np.load(f'./files/c4_m1_vqgan_256/{uid}.npz')['pred']
    # plot_eeg(eeg_vqgan, ax_eeg[3], uid, 'Vqgan')

    mage_spec_T = torch.from_numpy(mage_spec)
    attentions = [(model.encoder.softmax(model.encoder.query_layer[i](mage_spec_T))).squeeze() for i in range(model.num_heads)]
    plot_attention(attentions, ax_eeg[3], uid)


def on_keypress(event):
    global idx, ax_red, fig, z, ax_eeg, fig_eeg, uid_list
    if event.key == "up":
        x = np.array([z[idx, 0], z[idx, 1]])
        d = ((z - x[None, :]) ** 2).sum(1)
        is_above = z[:,1] - x[1] <= 0
        d = np.array(d)
        d[is_above] += 9999
        idx = np.argmin(d).item()
        ax_red.set_data([z[idx, 0]], [z[idx, 1]])
        fig.canvas.draw()
        for a in ax_eeg:
            a.clear()
        uid = uid_list[idx]
        if os.path.exists(os.path.join(multitaper_path, uid)):
            plot_main(ax_eeg, uid)
        else:
            pass
        fig_eeg.canvas.draw()
    elif event.key == "down":
        x = np.array([z[idx, 0], z[idx, 1]])
        d = ((z - x[None, :]) ** 2).sum(1)
        is_above = z[:,1] - x[1] >= 0
        d = np.array(d)
        d[is_above] += 9999
        idx = np.argmin(d).item()
        ax_red.set_data([z[idx, 0]], [z[idx, 1]])
        fig.canvas.draw()
        for a in ax_eeg:
            a.clear()
        uid = uid_list[idx]
        if os.path.exists(os.path.join(multitaper_path, uid)):
            plot_main(ax_eeg, uid)
        else:
            pass
        fig_eeg.canvas.draw()
    elif event.key == "right":
        x = np.array([z[idx, 0], z[idx, 1]])
        d = ((z - x[None, :]) ** 2).sum(1)
        is_above = z[:,0] - x[0] <= 0
        d = np.array(d)
        d[is_above] += 9999
        idx = np.argmin(d).item()
        ax_red.set_data([z[idx, 0]], [z[idx, 1]])
        fig.canvas.draw()
        for a in ax_eeg:
            a.clear()
        uid = uid_list[idx]
        if os.path.exists(os.path.join(multitaper_path, uid)):
            plot_main(ax_eeg, uid)
        else:
            pass
        fig_eeg.canvas.draw()
    elif event.key == "left":
        x = np.array([z[idx, 0], z[idx, 1]])
        d = ((z - x[None, :]) ** 2).sum(1)
        is_above = z[:,0] - x[0] >= 0
        d = np.array(d)
        d[is_above] += 9999
        idx = np.argmin(d).item()
        ax_red.set_data([z[idx, 0]], [z[idx, 1]])
        fig.canvas.draw()
        for a in ax_eeg:
            a.clear()
        uid = uid_list[idx]
        if os.path.exists(os.path.join(multitaper_path, uid)):
            plot_main(ax_eeg, uid)
        else:
            pass
        fig_eeg.canvas.draw()

def on_click(event):
    global idx, ax_red, fig, z, ax_eeg, fig_eeg, uid_list
    if event.button == 1:  # Check for left mouse button click
        x = event.xdata
        y = event.ydata
        x = np.array([x, y])
        d = ((z - x[None, :]) ** 2).sum(1)
        idx = np.argmin(d).item()
        ax_red.set_data([z[idx, 0]], [z[idx, 1]])
        fig.canvas.draw()

        for a in ax_eeg:
            a.clear()
        uid = uid_list[idx]
        if os.path.exists(os.path.join(multitaper_path, uid)):
            plot_main(ax_eeg, uid)
        else:
            pass
        fig_eeg.canvas.draw()

def main(tsne_x1, tsne_x2, tsne_color, tsne_id):
    global idx, ax_red, fig, z, ax_eeg, fig_eeg, uid_list, pca

    uid_list = tsne_id #reduced['uid']
    # z = reduced['tsne']
    # z = reduced['umap']
    z = np.stack([tsne_x1, tsne_x2],1)

    fig = plt.figure(figsize=(8, 6))

    plt.scatter(
        z[:, 0],
        z[:, 1],
        alpha=0.5,
        c=tsne_color,
        s=8,
    )


    plt.gca().set_aspect('equal')
    plt.colorbar()

    idx = 0
    ax_red, = plt.plot(z[idx, 0], z[idx, 1], 'k.')
    # ax_red, = plt.plot(tsne_x1, tsne_x2, 'r.')

    nrows = 4
    fig_eeg, ax_eeg = plt.subplots(nrows, 1, figsize=(10, 2 * nrows), sharex=False)
    uid = uid_list[idx]
    plot_main(ax_eeg, uid)


    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    # from ipdb import set_trace as bp 
    # bp()
    plt.show()

if __name__ == "__main__":
    wsc_umap_df = pd.read_csv('/data/scratch/scadavid/projects/data/new_tuned_wsc_umap_df_2color.csv', encoding='mac_roman') #FIXME
    main(wsc_umap_df['tsne_x1'],wsc_umap_df['tsne_x2'],wsc_umap_df['colors'],wsc_umap_df['pids'])
    #main(np.random.randn(100),np.random.randn(100),np.random.randn(100),os.listdir(os.path.join(local_mnt,multitaper_path))[:100])
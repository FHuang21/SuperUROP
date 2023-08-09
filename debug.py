from BrEEG.utils_eeg import eeg_spectrogram_multitaper
import numpy as np
import pandas as pd
import os
from ipdb import set_trace as bp

# NOTE: all for wsc
df = pd.read_csv("/data/netmit/wifall/ADetect/data/csv/wsc-dataset-augmented.csv", encoding='mac_roman')
df = df[['wsc_id', 'wsc_vst', 'depression_med']]
embedding_path = "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/wsc_new/abdominal_c4_m1"
eeg_path = "/data/netmit/wifall/ADetect/data/wsc_new/c4_m1"
eeg_mt_path = "/data/netmit/wifall/ADetect/data/wsc_new/c4_m1_multitaper"
stage_path = "/data/netmit/wifall/ADetect/data/wsc_new/stage"

all_embeddings = os.listdir(embedding_path)
all_eegs = os.listdir(eeg_path) # only 11?
all_eeg_mts = os.listdir(eeg_mt_path)
all_stages = os.listdir(stage_path)

embedding_examples_dict = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['depression_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in all_embeddings}
eeg_examples_dict = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['depression_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in all_eegs}
eeg_mt_examples_dict = {f"wsc-visit{vst}-{id}-nsrr.npz": v for id, vst, v in zip(df['wsc_id'], df['wsc_vst'], df['depression_med']) if f"wsc-visit{vst}-{id}-nsrr.npz" in all_eeg_mts}

common_examples = []
for pid in embedding_examples_dict.keys():
    if pid in eeg_mt_examples_dict.keys():
        common_examples.append(pid)

#bp()
problem_count = 0
for pid in common_examples:
    #bp()
    x = np.load(os.path.join(embedding_path, pid))
    x = dict(x)
    embedding = x['decoder_eeg_latent'].squeeze(0)
    # if embedding.shape[0] >= 150:
    #     embedding = embedding[:150, :]
    # else:
    #     embedding = np.concatenate((embedding, np.zeros((150-embedding.shape[0], embedding.shape[-1]),dtype=np.float32)), axis=0)
    #eeg = np.load(os.path.join(eeg_path, pid))
    eeg_mt_file = np.load(os.path.join(eeg_mt_path, pid))
    eeg_mt = eeg_mt_file['data']
    stage_file = np.load(os.path.join(stage_path, pid))
    stage = stage_file['data']
    is_sleep = (stage > 0) & (stage <= 5)
    idx_is_sleep = np.where(is_sleep)[0]
    if not len(idx_is_sleep) > 0:
        continue
    sleep_end = idx_is_sleep[-1]
    try:
        assert abs(embedding.shape[0] - sleep_end/8) < 1
    except:
        problem_count += 1
        print("problem pid: ", pid)
        #bp()
print("problem count: ", problem_count)
bp()
import pandas as pd
import numpy as np
import os
from ipdb import set_trace as bp

def process_stages(stages):
        stages[stages < 0] = 0
        stages[stages > 5] = 0
        stages = stages.astype(int)
        mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
        return mapping[stages]

shhs2_csv_path = "/data/netmit/wifall/ADetect/data/csv/shhs2-dataset-augmented.csv"
df = pd.read_csv(shhs2_csv_path, encoding='mac_roman')
df = df[['nsrrid', 'slplatp']]
df.rename(columns={'slplatp':'sleep_onset'}, inplace=True)
df = df.dropna()
df['sleep_onset'] = df['sleep_onset'] * 2

nsrr_sleep_onset_mean = np.mean(df['sleep_onset'])
nsrr_sleep_onset_std = np.std(df['sleep_onset'])
root = '/data/netmit/wifall/ADetect/data'
all_stage_filenames = os.listdir(os.path.join(root, 'shhs2_new', 'stage'))
calculated_sleep_onsets = []
for filename in all_stage_filenames:
    file = np.load(os.path.join(root, 'shhs2_new', 'stage', filename))
    stage = file['data']
    fs = file['fs']
    if fs == 1:
        stage = stage[::30]
    stage = process_stages(stage)
    idx = np.where(stage > 0)[0][0]
    calculated_sleep_onsets.append(idx)

calculated_sleep_onset_mean = np.mean(np.array(calculated_sleep_onsets, dtype=int))
calculated_sleep_onset_std = np.std(np.array(calculated_sleep_onsets, dtype=int))

print('nsrr mean std')
print(nsrr_sleep_onset_mean)
print(nsrr_sleep_onset_std)
print('calculated mean std')
print(calculated_sleep_onset_mean)
print(calculated_sleep_onset_std)

bp()
import pandas as pd
import numpy as np
from tqdm import tqdm
from ipdb import set_trace as bp

import streamlit as st
from BrEEG.task_spec import SpecMultitaperFolder, SpecFolder
import matplotlib.pyplot as plt
from pathlib import Path

data_path = '/data/netmit/wifall/ADetect/data/'

shhs1data = data_path + 'shhs2/csv/shhs1-dataset-0.14.0.csv'
shhs2data = data_path + 'shhs2/csv/shhs2-dataset-0.14.0.csv'

shhs1_df = pd.read_csv(shhs1data, encoding='mac_roman', index_col=0)
shhs2_df = pd.read_csv(shhs2data, encoding='mac_roman', index_col=0)

# shhs1_patients = set(shhs1_df['nsrrid'])
# shhs2_patients = set(shhs2_df['nsrrid'])
# print(shhs1_patients.intersection(shhs2_patients) == shhs2_patients) # True

# patients who did both studies will be those in shhs2 labels csv. then add in columns for tca1 and ntca1
antidep_df = shhs2_df[['TCA2', 'NTCA2']].copy()
antidep_df.insert(0, 'TCA1', 0)
antidep_df.insert(1, 'NTCA1', 0)

# now need to fill in values for tca1/ntca1 from shhs1 csv
# if either shhs1/2 psg missing, remove patient from dataframe
for id in tqdm(shhs2_df.index):
    if (not shhs1_df.loc[id, 'shhs1_psg']) or (not shhs2_df.loc[id, 'shhs2_psg']):
        antidep_df.drop(index=id)
        continue
    antidep_df.loc[id, 'TCA1'] = shhs1_df.loc[id, 'TCA1']
    antidep_df.loc[id, 'NTCA1'] = shhs1_df.loc[id, 'NTCA1']

antidep_df = antidep_df.dropna(how='any') # drop any patients with nan values
antidep_df = antidep_df[(antidep_df['TCA1'] != antidep_df['TCA2']) | (antidep_df['NTCA1'] != antidep_df['NTCA2'])]

def get_spec(eeg_file):  # adapt from Haoqi
    import mne
    # eeg.shape = (#channel, #point)
    Fs = eeg_file['fs']  # 200  # [Hz]
    # preprocessing by filtering
    eeg = eeg_file['data'].astype(np.float64)

    # s = eeg_file['data'].std()
    # st.write(f'std {s}')
    if eeg_file['data'].std() < 1:
        eeg *= 1e6
    if eeg_file['data'].std() > 50:
        eeg /= 5  # FIXME: hack for mgh
    eeg = eeg - eeg.mean()  # remove baseline
    # eeg = mne.filter.notch_filter(eeg, Fs, 60)  # remove line noise, US is 60Hz
    # eeg = mne.filter.filter_data(eeg, Fs, 0.3, 32)  # bandpass, for sleep EEG it's 0.3-35Hz

    # first segment into 30-second epochs
    epoch_time = 30  # [second]
    step_time = 30
    epoch_size = int(round(epoch_time * Fs))
    step_size = int(round(step_time * Fs))

    # epoch_start_ids = np.arange(0, len(eeg) - epoch_size, epoch_size)
    # ts = epoch_start_ids / Fs
    # epochs = eeg.reshape(-1, epoch_size)  # #epoch, size(epoch)

    def get_seg(eeg, l, r):
        if l >= 0 and r <= len(eeg):
            return eeg[l:r]
        else:
            if r > len(eeg):
                tmp = eeg[l:]
                tmp = np.concatenate([tmp, np.zeros((r - l - len(tmp)))])
                return tmp
            else:
                tmp = eeg[:r]
                tmp = np.concatenate([np.zeros((-l)), tmp])
                return tmp

    epochs = np.array([
        get_seg(eeg, i + step_size // 2 - epoch_size // 2, i + step_size // 2 + epoch_size // 2)
        for i in range(0, len(eeg), step_size)
    ])

    spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.0, fmax=32, bandwidth=0.5,
                                                         normalization='full', verbose=False,
                                                         n_jobs=12)  # spec.shape = (#epoch, #freq)
    # freq == np.linspace(0,32,961)
    spec_db = 10 * np.log10(spec)
    return spec_db.T

# new spec
def plot_eeg_spec(a, c4m1_path):
    if c4m1_path.is_file():
        c4m1_file = np.load(c4m1_path)
        eeg_spec = get_spec(c4m1_file)
        tmp = eeg_spec
        a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30,
                     np.linspace(0, 32, tmp.shape[0]),
                     tmp,
                     vmin=-10, vmax=15,
                     antialiased=True,
                     shading='auto',
                     cmap='jet')
    else:
        tmp = 0
        c4m1_file = {'data': 0}

    eeg_y_max = 32
    a.set_title(f'Multitaper EEG {c4m1_path}')
    a.set_ylabel('Hz')
    a.set_ylim([0, eeg_y_max])
    a.set_yticks(list(np.arange(0, eeg_y_max, 4)), list(np.arange(0, eeg_y_max, 4)))
    a.grid()
    return tmp


def plot_stage(a, stage):
    a.plot(np.arange(len(stage)), stage)
    a.set_title('stage')
    a.set_yticks([0, 1, 2, 3], ['A', 'R', 'L', 'D'])
    a.grid()


def plot_visit(data, nsrrid):
    
    fig, ax = plt.subplots(2,1, sharex=True)

    uid = data + '-' + str(nsrrid)
    # plot stage
    stage_path = Path('/data/netmit/wifall/ADetect/data/') / data / 'stage' / (uid + '.npz')

    try:
        file = np.load(stage_path)
        stage, fs = file['data'], file['fs']
        if fs == 1:
            stage = stage[::30]
        stage = stage.astype(int)
        stage_raw = np.copy(stage)
        idx_invalid = (stage < 0) | (stage > 5)
        stage_remap = np.array([0, 2, 2, 3, 3, 1] + [np.nan] * 10)
        stage = stage_remap[stage].astype(float)
        stage[idx_invalid] = np.nan
        plot_stage(ax[0], stage)
        ax[0].set_xticks(np.arange(0, len(stage), 60), np.arange(0, len(stage), 60))

        # plot eeg spec
        c4m1_path = Path('/data/netmit/wifall/ADetect/data/') / data / 'c4_m1' / (uid + '.npz')
        plot_eeg_spec(ax[1], c4m1_path)
        fig.tight_layout()
        st.pyplot(fig)
    except FileNotFoundError:
        pass

idx = st.slider('idx', value=0,min_value=0,max_value=len(antidep_df)-1)
nsrrid = antidep_df.index[idx]

patient_meds = antidep_df[antidep_df.index == nsrrid]
st.write(patient_meds)
plot_visit('shhs1', nsrrid)
plot_visit('shhs2', nsrrid)

# could also include age + other relevant info on plots
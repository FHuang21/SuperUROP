import pandas as pd
import torch
from PIL import Image
import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils_plotter import *


@st.cache_data
def load_df(dataset):
    return pd.read_csv(f'/data/netmit/wifall/ADetect/data/csv/{dataset}-dataset-augmented.csv', low_memory=False)


# @st.cache_data
def load_df_error(dataset):
    return pd.read_csv(f'./{dataset}_error.csv', low_memory=False)


# config dataset
dataset = st.sidebar.selectbox('dataset', ['shhs2', 'p18c', 'nchsdb',
                                           'mgh'
                                           # 'shhs1', 'ccshs'
                                           # 'mros1', 'mros2',
                                           # 'mayo', 'ccshs', 'wsc',
                                           # 'mgh',
                                           # 'stages'
                                           ])
df = load_df(dataset)
df.columns = df.columns.str.lower()
df_err = load_df_error(dataset)
sort_key = st.selectbox('sort by', df_err.columns)
df_sort = df_err.sort_values(by=sort_key)
uid_list = df_sort.uid

# [get model list]
lth_root = Path('/data/scratch-oc40/lth/mage-br-eeg-inference')

# model_list = []
# for x in lth_root.iterdir():
#     for y in x.iterdir():
#         model_list += [x.name + '/' + y.name]
# model_list.sort(reverse=True)
# model_name = st.selectbox('model', model_list)
# error_df_path = P ath(f'./files/{model_name}/error_full_{dataset}.csv')

# idx = st.slider('idx', value=0, max_value=len(uid_list) - 1, step=1)
idx = st.number_input(f'idx {0} to {len(uid_list) - 1}', min_value=0, max_value=len(uid_list) - 1)
uid = uid_list.iloc[idx]  # FIXME: cannot use uid_list[idx]
row = df[df.uid == uid].iloc[0]

genderMap = {1: 'Male', 2: 'Female'}
gender = row.mit_gender
try:
    gender = genderMap[gender]
except:
    pass

st.text(f'Uid {uid} Age {row.age} Gender {gender}')
st.write(df_sort.iloc[idx])

mage_root = lth_root / '20230615-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-30kdata-neweeg-ema-br1d-1layerbbenc-multi-channel-2/iter1-temp0.0-mr1.0'
mage_128_root = lth_root / '20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps128x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0'
mage_256_root = lth_root / '20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0'
vqgan_root = lth_root / '20230615-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-30kdata-neweeg-ema-br1d-1layerbbenc-multi-channel-2/iter1-temp0.0-mr0.0'
vqgan_128_root = lth_root / '20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps128x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr0.0'
vqgan_256_root = lth_root / '20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr0.0'

# [control bars]
eeg_y_max = st.sidebar.slider('EEG freq max', min_value=1, max_value=32, step=1, value=32)
br_y_min = st.sidebar.slider('Breathing freq min', min_value=1, max_value=20, step=1, value=8)
br_y_max = st.sidebar.slider('Breathing freq max', min_value=br_y_min + 1, max_value=32, step=1, value=24)

plotter = Plotter()

plotter_list = [
    StagePlotter(),
    BreathPlotter('thorax'),
    BreathPlotter('abdominal'),
    EEGPlotter('c3_m2'),
    EEGPlotter('c4_m1'),
    MagePlotter(mage_root, 'abdominal_c4_m1', 'Mage 32x8'),
    MageProbPlotter(mage_root, 'abdominal_c4_m1', 'Mage 32x8 Token', (32,8)),
    MagePlotter(mage_128_root, 'abdominal_c4_m1', 'Mage 128x8'),
    MageProbPlotter(mage_128_root, 'abdominal_c4_m1', 'Mage 128x8 Token', (128,8)),
    MagePlotter(mage_256_root, 'abdominal_c4_m1', 'Mage 256x8'),
    MageProbPlotter(mage_256_root, 'abdominal_c4_m1', 'Mage 256x8 Token', (256,8)),
    # MageRescalePlotter(mage_root, 'c4_m1', 'Mage'),
    MagePlotter(vqgan_root, 'c4_m1', 'Vqgan 32x8'),
    MagePlotter(vqgan_128_root, 'abdominal_c4_m1', 'Vqgan 128x8'),
    MagePlotter(vqgan_256_root, 'abdominal_c4_m1', 'Vqgan 256x8'),
    EEGPlotter('f3_m2'),
    MagePlotter(mage_root, 'f3_m2', 'Mage 32x8'),
    MagePlotter(mage_128_root, 'f3_m2', 'Mage 128x8'),
    MagePlotter(vqgan_root, 'f3_m2', 'Vqgan 32x8'),
    EEGPlotter('o1_m2'),
    MagePlotter(mage_root, 'o1_m2', 'Mage 32x8'),
    MagePlotter(mage_128_root, 'o1_m2', 'Mage 128x8'),
    MagePlotter(vqgan_root, 'o1_m2', 'Vqgan 32x8'),
]

if dataset == 'mgh':
    plotter_list = [
        StagePlotter(),
        BreathPlotter('thorax'),
        BreathPlotter('abdominal'),
        BreathPlotter('rf'),
        EEGPlotter('c3_m2'),
        EEGPlotter('c4_m1'),
        MagePlotter(mage_root, 'thorax_c4_m1', 'Mage thorax'),
        MagePlotter(mage_root, 'abdominal_c4_m1', 'Mage abdominal'),
        MagePlotter(mage_root, 'rf_c4_m1', 'Mage rf'),
    ]

plotter.add_plotter(plotter_list)
plotter.load(dataset, uid).plot(eeg_y_max, br_y_min, br_y_max)
st.pyplot(plotter.fig)

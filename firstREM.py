import numpy as np
import pandas as pd
import os
from ipdb import set_trace as bp

def process_stages(stages):
        stages[stages < 0] = 0
        stages[stages > 5] = 0
        stages = stages.astype(int)
        mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
        return mapping[stages]

def get_first_rem_duration(filename, dataset):
    my_root = '/data/scratch/scadavid/projects/data'
    cycle_idx = np.load(os.path.join(my_root, dataset, filename))
    root = '/data/netmit/wifall/ADetect/data/'
    filename = filename.replace('.npy', '.npz')
    file = np.load(os.path.join(root, f'{dataset}_new', 'stage', filename))
    stage = file['data']
    fs = file['fs']
    if fs == 1:
          stage = stage[::30]
    stage = process_stages(stage)
    if(len(cycle_idx)>=2):
        first_cycle = stage[cycle_idx[0]:cycle_idx[1]]
        output = sum(first_cycle==4)
    else:
        print('problem: ', filename)
        output = np.nan
    return output

def get_initial_rem_duration(filename, dataset):
    root = '/data/netmit/wifall/ADetect/data/'
    file = np.load(os.path.join(root, f'{dataset}_new', 'stage', filename))
    stage = file['data']
    fs = file['fs']
    if fs == 1:
          stage = stage[::30]
    stage = process_stages(stage)
    initial_stage = stage[0:480]
    return sum(initial_stage==4)

def get_window_rem_duration(filename, dataset):
    root = '/data/netmit/wifall/ADetect/data/'
    file = np.load(os.path.join(root, f'{dataset}_new', 'stage', filename))
    stage = file['data']
    fs = file['fs']
    if fs == 1:
          stage = stage[::30]
    stage = process_stages(stage)
    idx = np.where(stage>0)[0][0]
    stage_window = stage[(idx+160):(idx+340)]
    return sum(stage_window==4)

def get_rem_latency(filename, dataset):
    root = '/data/netmit/wifall/ADetect/data/'
    file = np.load(os.path.join(root, f'{dataset}_new', 'stage', filename))
    stage = file['data']
    fs = file['fs']
    if fs == 1:
          stage = stage[::30]
    stage = process_stages(stage)
    sleep_latency = np.where(stage>0)[0][0]
    try:
        first_rem = np.where(stage == 4)[0][0]
        rem_latency = first_rem - sleep_latency
    except: # no REM!
        rem_latency = 0.0 # if no rem they just set it to 0 in the shhs2 dataset
    return rem_latency

root = '/data/netmit/wifall/ADetect/data'
data_root = '/data/scratch/scadavid/projects/data'

#wsc
# cycle_filenames = os.listdir(os.path.join(data_root, 'wsc'))

# first_rem_dict = {'wsc_id': [filename.split('-')[2] for filename in cycle_filenames if not filename.startswith('._')], 
#                   'wsc_vst': [filename.split('visit')[1][0] for filename in cycle_filenames if not filename.startswith('._')], 
#                   'time_first_rem': [get_first_rem_duration(filename, 'wsc') for filename in cycle_filenames if not filename.startswith('._')]}

#shhs2
#cycle_filenames = os.listdir(os.path.join(data_root, 'shhs2'))
stage_filenames = os.listdir(os.path.join(root, 'shhs2_new', 'stage'))

first_rem_dict = {'nsrrid': [filename.split('.')[0].split('-')[1] for filename in stage_filenames if not filename.startswith('._')], 
                  'my_rem_latency': [get_rem_latency(filename, 'shhs2') for filename in stage_filenames if not filename.startswith('._')]}


first_rem_df = pd.DataFrame(first_rem_dict)
csv_path = '/data/scratch/scadavid/projects/data/csv/rem_latency_shhs2.csv'
first_rem_df.to_csv(csv_path, index=False)  # Set index=False to omit row numbers
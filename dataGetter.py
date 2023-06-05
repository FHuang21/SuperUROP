import pandas as pd
import numpy as np
from tqdm import tqdm

data_path = '/data/netmit/wifall/ADetect/data/'

shhs1data = data_path + 'shhs2/csv/shhs1-dataset-0.14.0.csv'
shhs2data = data_path + 'shhs2/csv/shhs2-dataset-0.14.0.csv'

shhs1_df = pd.read_csv(shhs1data, encoding='mac_roman')
shhs2_df = pd.read_csv(shhs2data, encoding='mac_roman')

# shhs1_patients = set(shhs1_df['nsrrid'])
# shhs2_patients = set(shhs2_df['nsrrid'])
# print(shhs1_patients.intersection(shhs2_patients) == shhs2_patients) # True!

antidep_df = shhs2_df[['nsrrid', 'TCA2', 'NTCA2']].copy()
antidep_df.insert(1, 'TCA1', 0)
antidep_df.insert(2, 'NTCA1', 0)

for id in tqdm(shhs1_df['nsrrid']):
    antidep_df.at[id, 'TCA1'] = shhs1_df[shhs1_df['nsrrid'] == id]['TCA1']
    antidep_df.at[id, 'NTCA1'] = shhs1_df[shhs1_df['nsrrid'] == id]['NTCA1']

# for id in tqdm(antidep_df['nsrrid']):
#     if 
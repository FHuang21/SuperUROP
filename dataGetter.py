import pandas as pd
import numpy as np
from tqdm import tqdm
from ipdb import set_trace as bp

data_path = '/data/netmit/wifall/ADetect/data/'

shhs1data = data_path + 'shhs2/csv/shhs1-dataset-0.14.0.csv'
shhs2data = data_path + 'shhs2/csv/shhs2-dataset-0.14.0.csv'

shhs1_df = pd.read_csv(shhs1data, encoding='mac_roman', index_col=0)
shhs2_df = pd.read_csv(shhs2data, encoding='mac_roman', index_col=0)

# shhs1_patients = set(shhs1_df['nsrrid'])
# shhs2_patients = set(shhs2_df['nsrrid'])
# print(shhs1_patients.intersection(shhs2_patients) == shhs2_patients) # True!

# patients who did both studies will be those in shhs2 labels csv. then add in columns for tca1 and ntca1
antidep_df = shhs2_df[['TCA2', 'NTCA2']].copy()
antidep_df.insert(0, 'TCA1', 0)
antidep_df.insert(1, 'NTCA1', 0)
print(antidep_df)

# now need to fill in values for tca1/ntca1 from shhs1 csv
for id in tqdm(shhs2_df.index):
    antidep_df.loc[id, 'TCA1'] = shhs1_df.loc[id, 'TCA1']
    antidep_df.loc[id, 'NTCA1'] = shhs1_df.loc[id, 'NTCA1']
antidep_df = antidep_df.dropna(how='any') # drop any patients with nan values
antidep_df = antidep_df[(antidep_df['TCA1'] != antidep_df['TCA2']) & (antidep_df['NTCA1'] != antidep_df['NTCA2'])]
print(antidep_df)
for id in tqdm(antidep_df.index):
    assert antidep_df.loc[id, 'TCA1'] == shhs1_df.loc[id, 'TCA1']
    assert antidep_df.loc[id, 'NTCA1'] == shhs1_df.loc[id, 'NTCA1']
    assert antidep_df.loc[id, 'TCA2'] == shhs2_df.loc[id, 'TCA2']
    assert antidep_df.loc[id, 'NTCA2'] == shhs2_df.loc[id, 'NTCA2']

# think we did it
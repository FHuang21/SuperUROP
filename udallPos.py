import pandas as pd
import os
from ipdb import set_trace as bp

my_root = '/data/scratch/scadavid/projects/data/udall'

patient_csv_files = os.listdir(my_root)

results_dict = {'pid': [], 'num_pos_nights': [], 'num_neg_nights': [], 'ratio_pos_to_total': []}

for csv_name in patient_csv_files:
    if csv_name.startswith('._') or csv_name.startswith('results') or csv_name=='figs':
        continue

    pid = csv_name.split('.')[0].split('data_')[1]

    patient_df = pd.read_csv(os.path.join(my_root, csv_name), encoding='mac_roman')
    #num_pos_nights = sum(patient_df['preds'])#NOTE
    num_pos_nights = len(patient_df[patient_df['prob']>=0.5]) # NOTE
    num_neg_nights = len(patient_df) - num_pos_nights
    ratio = num_pos_nights / len(patient_df)

    results_dict['pid'].append(pid)
    results_dict['num_pos_nights'].append(num_pos_nights)
    results_dict['num_neg_nights'].append(num_neg_nights)
    results_dict['ratio_pos_to_total'].append(ratio)

results_df = pd.DataFrame(results_dict)
results_df.to_csv(f'{my_root}/results_th50_new.csv')



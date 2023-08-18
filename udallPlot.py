import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from ipdb import set_trace as bp

def get_average_prob(df):
    return sum(df['prob']) / len(df['prob'])

my_udall_root = '/data/scratch/scadavid/projects/data/udall'

antidep_dict = {'NIHBL760KMGXL': 'escitalopram, trazodone',
'NIHNT823CHAC3': 'escitalopram',
'NIHXN782DBBP7': 'prozac',
'NIHYM875FLXFF': 'fluoxetine',
'NIHCX409ZDTJU': 'escitalopram',
'NIHPX213JXJZC': 'zoloft',
'NIHVA109LWXMF': 'sertraline',
'NIHPV178MDAUT': '	venlafaxine',
'NIHCJ555VCWZY': '	venlafaxine',
'NIHXN551LBFMK': 'paroxetine',
'NIHEP519MZAEZ': 'fluoxetine',
'NIHHD991PGRJC': '	mirtazapine',
'NIHFW795KLATW': 'paroxetine, sertraline', 
'NIHPT334YGJLK': 'paroxetine, desvenlafaxine',
'NIHAV871KZCVE': 'paroxetine',
'NIHMR963TPLWF': 'paroxetine, zoloft',
'NIHJW557ZEUZV': 'control',
'NIHMF399WYNH5': 'control',
'NIHAV025ZCBGB': 'control',
'NIHBY076JZFYN': 'control',
'NIHEB701YGBEC': 'control',
'NIHFT628PHTAY': 'control',
'NIHHG558EJJMM': 'control',
'NIHRY949ZYWHQ': 'control',
'NIHXB175YAGF7': 'control',
'NIHYW557MLDFE': 'control',
'NIHZT156UUPLX': 'control',
'NIHGK080AGLJH': 'control',
'NIHND126MXDGP': 'control',
'NIHBE740TFYAH': 'control',
'NIHNX715KUVY8': 'control',
'NIHFX695VBHFM': 'control',
'NIHDW178UFZHB': 'control',
'NIHTK278VZHYL': 'control',
'NIHGA312KVEC2': 'control',
'NIHWR605ZHTE7': 'control'}

#filename = 'PD_Hao_data_NIHPV178MDAUT.csv'
patient_csvs = [csv for csv in os.listdir(my_udall_root) if csv.endswith('.csv') and not csv.startswith('._')]

for j, csv_name in enumerate(patient_csvs):
    # if j==1: # NOTE: temp
    #     break

    pid = csv_name.split('.')[0].split('data_')[1]
    try:
        meds = antidep_dict[pid]
    except:
        print(pid)
        continue
    patient_df = pd.read_csv(os.path.join(my_udall_root, csv_name), encoding='mac_roman')
    all_preds = patient_df['prob'].to_numpy(dtype=float)

    avg_prob = get_average_prob(patient_df)

    # Calculate moving average
    window_size = 3
    smoothed_preds = np.convolve(all_preds, np.ones(window_size)/window_size, mode='valid') # window

    

    plt.figure()
    plt.plot(all_preds)
    plt.plot(smoothed_preds)
    plt.ylim(0.0, 1.0)
    plt.axhline(y=0.5,ls='--',alpha=0.5,color='gray')
    plt.axhline(y=0.2,ls='--',alpha=0.5,color='gray')
    plt.title(f'PID: {pid}, Avg Prob: {round(avg_prob,5)}, Meds: {meds}')
    plt.savefig(f'/data/scratch/scadavid/projects/data/udall/figs/{pid}.png')
    plt.clf()
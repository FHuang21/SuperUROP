import pandas as pd

qol_data_path = '/data/scratch/scadavid/projects/data/v6qol.sas7bdat'
med_data_path = '/data/scratch/scadavid/projects/data/v6meds.sas7bdat'

sof_qol_df = pd.read_sas(qol_data_path, index='ID')
sof_med_df = pd.read_sas(med_data_path, index='ID')

sof_depressed_df = sof_qol_df[sof_qol_df['V6GDS15'] > 4.0] # >4.0 indicates at least mild depression
num_depressed = len(sof_depressed_df)
sof_antidep_df = sof_med_df[sof_med_df['V6ADEPR'] == 1.0]
num_antidep = len(sof_antidep_df)
sof_ssri_df = sof_med_df[sof_med_df['V6SSRI'] == 1.0]
num_ssri = len(sof_ssri_df)
sof_tca_df = sof_med_df[sof_med_df['V6TAD'] == 1.0]
num_tca = len(sof_tca_df)
sof_traz_df = sof_med_df[sof_med_df['V6TRAZ'] == 1.0]
num_traz = len(sof_traz_df)
print("Number depressed: ", num_depressed)
print("Number taking antidepressant: ", num_antidep)
print("Number taking SSRI: ", num_ssri)
print("Number taking trazodone: ", num_traz)
print("Number taking TCA: ", num_tca)
print("Total: ", len(sof_qol_df))
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import streamlit as st
#import numpy as np

data_path = '/data/netmit/wifall/ADetect/data/'

shhs1data = data_path + 'shhs2/csv/shhs1-dataset-0.14.0.csv'
shhs2data = data_path + 'shhs2/csv/shhs2-dataset-0.14.0.csv'

shhs1_df = pd.read_csv(shhs1data, encoding='mac_roman', index_col=0)
shhs2_df = pd.read_csv(shhs2data, encoding='mac_roman', index_col=0)

# patients who did both studies will be those in shhs2 labels csv. then add in columns for tca1 and ntca1
antidep_df = shhs2_df[['TCA2', 'NTCA2', 'timerem', 'remlaip']].copy()
antidep_df = antidep_df.rename(columns={'timerem':'timerem2', 'remlaip':'remlaip2'}) #
antidep_df.insert(0, 'TCA1', 0)
antidep_df.insert(1, 'NTCA1', 0)
antidep_df.insert(2, 'timerem1', 0) #
antidep_df.insert(3, 'remlaip1', 0) #

# now need to fill in values for tca1/ntca1 from shhs1 csv
# if either shhs1/2 psg missing, remove patient from dataframe
for id in tqdm(shhs2_df.index):
    if (not shhs1_df.loc[id, 'shhs1_psg']) or (not shhs2_df.loc[id, 'shhs2_psg']):
        antidep_df.drop(index=id)
        continue
    antidep_df.loc[id, 'TCA1'] = shhs1_df.loc[id, 'TCA1']
    antidep_df.loc[id, 'NTCA1'] = shhs1_df.loc[id, 'NTCA1']
    antidep_df.loc[id, 'timerem1'] = shhs1_df.loc[id, 'timerem'] #
    antidep_df.loc[id, 'remlaip1'] = shhs1_df.loc[id, 'RemLaIP'] / 60.0 # shhs1 csv's rem latencies are in seconds for whatever reason
    #print(shhs1_df.loc[id, 'RemLaIP'])

antidep_df = antidep_df.dropna(how='any') # drop any patients with nan values
antidep_changed_df = antidep_df[(antidep_df['TCA1'] != antidep_df['TCA2']) | (antidep_df['NTCA1'] != antidep_df['NTCA2'])] # changed med pattern between
antidep_started_df = antidep_df[~((antidep_df['TCA1'] == 1.0) | (antidep_df['NTCA1'] == 1.0)) & ((antidep_df['TCA2'] == 1.0) | (antidep_df['NTCA2'] == 1.0))] # started meds between
antidep_stopped_df = antidep_df[((antidep_df['TCA1'] == 1.0) | (antidep_df['NTCA1'] == 1.0)) & ~((antidep_df['TCA2'] == 1.0) | (antidep_df['NTCA2'] == 1.0))] # stopped meds between


# combine all these boxplots in one plot
fig, axs = plt.subplots(2,2, sharex=True, sharey='row')
# get boxplots
antidep_started_timerem_boxplot = antidep_started_df.boxplot(column=['timerem1', 'timerem2'], ax=axs[0,0])
antidep_stopped_timerem_boxplot = antidep_stopped_df.boxplot(column=['timerem1', 'timerem2'], ax=axs[0,1])
antidep_started_remlaip_boxplot = antidep_started_df.boxplot(column=['remlaip1', 'remlaip2'], ax=axs[1,0])
antidep_stopped_remlaip_boxplot = antidep_stopped_df.boxplot(column=['remlaip1', 'remlaip2'], ax=axs[1,1])
# label subplots
axs[0,0].set_title('Started Antidepressants')
axs[0,1].set_title('Taken Off Antidepressants')
common_ticks = [1, 2]  # Example positions on the x-axis
common_labels = ['shhs1', 'shhs2']
axs[1,0].set_xticks(common_ticks)
axs[1,0].set_xticklabels(common_labels)
axs[0,0].set_ylabel('REM Duration (min)')
axs[1,0].set_ylabel('REM Latency (min)')

##### now delta plots

# get delta dataframes
antidep_started_timerem_delta_df = (antidep_started_df['timerem2'] - antidep_started_df['timerem1']).to_frame()
antidep_stopped_timerem_delta_df = (antidep_stopped_df['timerem2'] - antidep_stopped_df['timerem1']).to_frame()
antidep_started_remlaip_delta_df = (antidep_started_df['remlaip2'] - antidep_started_df['remlaip1']).to_frame()
antidep_stopped_remlaip_delta_df = (antidep_stopped_df['remlaip2'] - antidep_stopped_df['remlaip1']).to_frame()
antidep_started_timerem_delta_df.columns = ['delta_timerem']
antidep_stopped_timerem_delta_df.columns = ['delta_timerem']
antidep_started_remlaip_delta_df.columns = ['delta_remlaip']
antidep_stopped_remlaip_delta_df.columns = ['delta_remlaip']
#print(antidep_started_timerem_delta_df)

fig2, axs2 = plt.subplots(2,2, sharex=True, sharey='row')
# get boxplots
antidep_started_timerem_delta_boxplot = antidep_started_timerem_delta_df.boxplot(column='delta_timerem', ax=axs2[0,0])
antidep_stopped_timerem_delta_boxplot = antidep_stopped_timerem_delta_df.boxplot(column='delta_timerem', ax=axs2[0,1])
antidep_started_remlaip_delta_boxplot = antidep_started_remlaip_delta_df.boxplot(column='delta_remlaip', ax=axs2[1,0])
antidep_stopped_remlaip_delta_boxplot = antidep_stopped_remlaip_delta_df.boxplot(column='delta_remlaip', ax=axs2[1,1])
# label subplots
axs2[0,0].set_title('Started Antidepressants')
axs2[0,1].set_title('Taken Off Antidepressants')
# common_ticks = [1, 2]  # Example positions on the x-axis
# common_labels = ['shhs1', 'shhs2']
# axs2[1,0].set_xticks(common_ticks)
# axs2[1,0].set_xticklabels(common_labels)
axs2[0,0].set_ylabel('Change REM Duration (min)')
axs2[1,0].set_ylabel('Change REM Latency (min)')
axs2[1,0].set_xticklabels('')
axs2[1,1].set_xticklabels('')

##### cohort analysis (control / on antidep meds)
# kinda inefficient, but idc just need the plots rn

# need df with two columns, one for all control patients' REM latencies, and one for all medicated patients' REM latencies
control_df1 = antidep_df[~((antidep_df['TCA1'] == 1.0) | (antidep_df['NTCA1'] == 1.0))]
control_df2 = antidep_df[~((antidep_df['TCA2'] == 1.0) | (antidep_df['NTCA2'] == 1.0))]
med_df1     = antidep_df[(antidep_df['TCA1'] == 1.0) | (antidep_df['NTCA1'] == 1.0)]
med_df2     = antidep_df[(antidep_df['TCA2'] == 1.0) | (antidep_df['NTCA2'] == 1.0)]

cohort_remlaip_df1 = pd.concat([control_df1['remlaip1'], med_df1['remlaip1']], axis=1)
cohort_remlaip_df1.columns = ['control1','med1']
cohort_remlaip_df2 = pd.concat([control_df2['remlaip2'], med_df2['remlaip2']], axis=1)
cohort_remlaip_df2.columns = ['control2','med2']

cohort_timerem_df1 = pd.concat([control_df1['timerem1'], med_df1['timerem1']], axis=1)
cohort_timerem_df1.columns = ['control1','med1']
cohort_timerem_df2 = pd.concat([control_df2['timerem2'], med_df2['timerem2']], axis=1)
cohort_timerem_df2.columns = ['control2','med2']

fig3, axs3 = plt.subplots(2, 2, sharey='row')
cohort_remlaip_boxplot1 = cohort_remlaip_df1.boxplot(column=['control1', 'med1'], ax=axs3[0,0])
cohort_remlaip_boxplot2 = cohort_remlaip_df2.boxplot(column=['control2', 'med2'], ax=axs3[0,1])
cohort_timerem_boxplot1 = cohort_timerem_df1.boxplot(column=['control1', 'med1'], ax=axs3[1,0])
cohort_timerem_boxplot2 = cohort_timerem_df2.boxplot(column=['control2', 'med2'], ax=axs3[1,1])
axs3[0,0].set_ylabel('REM Latency (min)')
axs3[1,0].set_ylabel('REM Duration (min)')

#### 2nd cohort analysis (control / tca / ntca) ((doesn't include ppl taking both))
# control_df1, control_df2 the same
tca_df1 = antidep_df[(antidep_df['TCA1'] == 1.0) & (antidep_df['NTCA1'] == 0.0)]
tca_df2 = antidep_df[(antidep_df['TCA2'] == 1.0) & (antidep_df['NTCA2'] == 0.0)]
ntca_df1 = antidep_df[(antidep_df['TCA1'] == 0.0) & (antidep_df['NTCA1'] == 1.0)]
ntca_df2 = antidep_df[(antidep_df['TCA2'] == 0.0) & (antidep_df['NTCA2'] == 1.0)]

# REM latency dfs
cohort2_remlaip_df1 = pd.concat([control_df1['remlaip1'], tca_df1['remlaip1'], ntca_df1['remlaip1']], axis=1)
cohort2_remlaip_df1.columns = ['control1','tca1','ntca1']
cohort2_remlaip_df2 = pd.concat([control_df2['remlaip2'], tca_df2['remlaip2'], ntca_df2['remlaip2']], axis=1)
cohort2_remlaip_df2.columns = ['control2','tca2', 'ntca2']

# REM duration dfs
cohort2_timerem_df1 = pd.concat([control_df1['timerem1'], tca_df1['timerem1'], ntca_df1['timerem1']], axis=1)
cohort2_timerem_df1.columns = ['control1','tca1','ntca1']
cohort2_timerem_df2 = pd.concat([control_df2['timerem2'], tca_df2['timerem2'], ntca_df2['timerem2']], axis=1)
cohort2_timerem_df2.columns = ['control2','tca2', 'ntca2']

fig4, axs4 = plt.subplots(2, 2, sharey='row')
cohort2_remlaip_boxplot1 = cohort2_remlaip_df1.boxplot(column=['control1', 'tca1', 'ntca1'], ax=axs4[0,0])
cohort2_remlaip_boxplot2 = cohort2_remlaip_df2.boxplot(column=['control2', 'tca2', 'ntca2'], ax=axs4[0,1])
cohort2_timerem_boxplot1 = cohort2_timerem_df1.boxplot(column=['control1', 'tca1', 'ntca1'], ax=axs4[1,0])
cohort2_timerem_boxplot2 = cohort2_timerem_df2.boxplot(column=['control2', 'tca2', 'ntca2'], ax=axs4[1,1])
axs4[0,0].set_ylabel('REM Latency (min)')
axs4[1,0].set_ylabel('REM Duration (min)')

# slider for each figure
figs = [fig, fig2, fig3, fig4]
idx = st.slider('idx', value=0,min_value=0,max_value=3)
st.pyplot(figs[idx])

save_path = '/data/scratch/scadavid/projects/data/'
filename = 'antidep_changed_patients.csv'
antidep_changed_df.to_csv(save_path+filename)
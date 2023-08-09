import pandas as pd
import os
from ipdb import set_trace as bp

wsc_data_path = "/data/netmit/wifall/ADetect/data/csv/wsc-dataset-augmented.csv"
#wsc_label_path = os.path.join(data_path, "/csv/wsc-dataset-augmented.csv")

df = pd.read_csv(wsc_data_path, encoding='mac_roman')
print(df.columns)
#print(df['zung_score'])
print(df['wsc_id'])

bp()
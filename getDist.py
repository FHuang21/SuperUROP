import pandas as pd
from ipdb import set_trace as bp

def threshold_func(column, threshold, comparison):
        if comparison=="less":
            return column.apply(lambda x: threshold-x+1 if x <= threshold else 0)
        elif comparison=="greater":
            return column.apply(lambda x: x-threshold+1 if x >= threshold else 0)

df = pd.read_csv("/data/netmit/wifall/ADetect/data/csv/shhs2-dataset-augmented.csv", encoding='mac_roman')
df = df[['nsrrid', 'ql209a', 'ql209b', 'ql209c', 'ql209d', 'ql209e', 'ql209f', 'ql209g', 'ql209h', 'ql209i']]
df = df.dropna()

df['label'] = 3*threshold_func(df['ql209c'], 4, 'less') + 2*threshold_func(df['ql209g'], 3, 'less') \
            + threshold_func(df['ql209i'], 3, 'less') + threshold_func(df['ql209e'], 5, 'greater') \
            + threshold_func(df['ql209f'], 4, 'less') + threshold_func(df['ql209d'], 5, 'greater') \
            + threshold_func(df['ql209h'], 4, 'greater')

# c is down in the dumps, h is been happy

print(df['label'].value_counts())

#bp()
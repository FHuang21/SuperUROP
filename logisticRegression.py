import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import matplotlib.pyplot as plt
from ipdb import set_trace as bp
#import matplotlib.pyplot as plt

### wsc stuff ###
# wsc_csv_path = "/data/netmit/wifall/ADetect/data/csv/wsc-dataset-augmented.csv"
# df = pd.read_csv(wsc_csv_path, encoding='mac_roman')
# hand_features = ['rem_latency', 'tst_rem', 'ahi', 'sleep_latency', 'waso', 'tst_nrem', 'tst']
# # hand_features = ['rem_latency', 'tst_rem']
# df = df[hand_features + ['depression_med']]
# df = df.dropna()
# features = df.drop(columns=['depression_med'])
# labels = df['depression_med']

### shhs2 stuff ###
#bp()
shhs2_csv_path = "/data/netmit/wifall/ADetect/data/csv/shhs2-dataset-augmented.csv"
df = pd.read_csv(shhs2_csv_path, encoding='mac_roman')
#NOTE
#shhs2_first_rem_df = pd.read_csv('/data/scratch/scadavid/projects/data/csv/first_rem_shhs2.csv')
shhs2_window_rem_df = pd.read_csv('/data/scratch/scadavid/projects/data/csv/window_rem_shhs2.csv')
shhs2_my_rem_latency_df = pd.read_csv('/data/scratch/scadavid/projects/data/csv/rem_latency_shhs2.csv')
#bp()
df = df.merge(shhs2_window_rem_df.merge(shhs2_my_rem_latency_df, on='nsrrid', how='inner'), on='nsrrid', how='inner')

#hand_features = ['remlaip', 'timerem', 'waso', 'ai_all', 'nremepop', 'nremepbp', 'slpprdp', 'time_window_rem'] #'remlaiip', 'timeremp', 'time_first_rem'
hand_features = ['timerem', 'waso', 'my_rem_latency']
#hand_features = ['my_rem_latency']
df['depression_med'] = (df['tca2'] + df['ntca2']) > 0
#df['depression_med'] = df['ntca2']
df = pd.concat([df[hand_features], df['depression_med']], axis=1)
df = df.dropna()
if 'remlaip' in hand_features:
    df['remlaip'] = df['remlaip'] / 60
if 'time_window_rem' in hand_features: # log this column
    df['time_window_rem'] = df['time_window_rem'] + .001
    df['time_window_rem'] = np.log(df['time_window_rem'])
    #df['time_window_rem'].replace([np.inf, -np.inf], np.nan, inplace=True)

# for feat in hand_features:
#     df[feat] = (df[feat] - np.mean(df[feat])) / np.std(df[feat])
    # print(feat, "mean and std")
    # print(np.mean(df[feat]))
    # print(np.std(df[feat]))
###
# class0 = np.log(df[df['depression_med'] == False]['time_window_rem'])
# class0.replace([np.inf, -np.inf], np.nan, inplace=True)
# class1 = np.log(df[df['depression_med'] == True]['time_window_rem'])
# class1.replace([np.inf, -np.inf], np.nan, inplace=True)
# plt.hist([class0, class1])
# plt.savefig('/data/scratch/scadavid/projects/data/figures/window_rem_hists_log.pdf')
# bp()
###
# plt.boxplot([df[df['depression_med']==False]['time_window_rem'], df[df['depression_med']==True]['time_window_rem']], labels=['class 0', 'class 1'])
# plt.savefig('/data/scratch/scadavid/projects/data/figures/window_rem_boxplots_updated_nolog.pdf')
# bp()
###
features = df[hand_features]
labels = df['depression_med'].astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=20)

# Combine features and labels for undersampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority_class = train_data[train_data['depression_med'] == 0]
minority_class = train_data[train_data['depression_med'] == 1]

# Undersample majority class
majority_downsampled = resample(majority_class,
                                replace=False,
                                n_samples=len(minority_class),
                                random_state=20)

# Combine minority class with downsampled majority class
downsampled_data = pd.concat([majority_downsampled, minority_class])


# Split the downsampled data back into features and labels
X_train_downsampled = downsampled_data.drop('depression_med', axis=1)
y_train_downsampled = downsampled_data['depression_med']

### grid search best hyperparams ###
# Define the parameter grid
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200, 300]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_downsampled, y_train_downsampled)
print(grid_search.best_params_)
print(grid_search.best_estimator_.coef_)

y_pred = grid_search.predict_proba(X_test)

# ## print metrics
# auroc = roc_auc_score(y_test, y_pred)
# print(f'auroc: {auroc:.5f}')

# auprc = average_precision_score(y_test, y_pred)
# print(f'auprc: {auprc:.5f}')

thresholds = np.arange(0.01, 1.0, 0.01)  # You can adjust the range and granularity of thresholds

best_f1 = 0  # To keep track of the best F1 score
best_threshold = 0  # To keep track of the threshold that results in the best F1 score

for threshold in thresholds:
    predicted_labels = [1 if prob[1] >= threshold else 0 for prob in y_pred]
    f1 = f1_score(y_test, predicted_labels)
    # print("threshold: ", threshold)
    # print("f1: ", f1)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("Best F1 score:", best_f1)
print("Best threshold:", best_threshold)


# # Calculate accuracy
# precision = precision_score(y_test, y_pred)
# print(f'precision: {precision:.5f}')

# recall = recall_score(y_test, y_pred)
# print(f'recall: {recall:.5f}')

# f1 = f1_score(y_test, y_pred)
# print(f'f1: {f1:.5f}')

# then can validate on testset (wsc) as well
bp()
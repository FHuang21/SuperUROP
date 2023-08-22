import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import os
from ipdb import set_trace as bp
import matplotlib.pyplot as plt

### wsc stuff ###
# wsc_csv_path = "/data/netmit/wifall/ADetect/data/csv/wsc-dataset-augmented.csv"
# df = pd.read_csv(wsc_csv_path, encoding='mac_roman')
# #NOTE
# wsc_first_rem_df = pd.read_csv('/data/scratch/scadavid/projects/data/csv/first_rem_wsc.csv')
# #bp()
# df = df.merge(wsc_first_rem_df, on=['wsc_id', 'wsc_vst'], how='inner')

# hand_features = ['rem_latency', 'tst_rem', 'ahi', 'sleep_latency', 'waso', 'tst_nrem', 'tst', 'time_first_rem']
# # hand_features = ['rem_latency', 'tst_rem']
# df = df[hand_features + ['depression_med']]
# df = df.dropna()
# features = df.drop(columns=['depression_med'])
# labels = df['depression_med']

### shhs2 stuff ###

shhs2_csv_path = "/data/netmit/wifall/ADetect/data/csv/shhs2-dataset-augmented.csv"
df = pd.read_csv(shhs2_csv_path, encoding='mac_roman')
#NOTE
shhs2_first_rem_df = pd.read_csv('/data/scratch/scadavid/projects/data/csv/first_rem_shhs2.csv')
#bp()
df = df.merge(shhs2_first_rem_df, on='nsrrid', how='inner')

hand_features = ['time_first_rem'] # 'time_first_rem'
#hand_features = ['remlaip', 'timerem']
df['depression_med'] = (df['tca2'] + df['ntca2']) > 0
df = pd.concat([df[hand_features], df['depression_med']], axis=1)
df = df.dropna()
#df['remlaip'] = df['remlaip'] / 60
#NOTE
# normalizing our features (make this a loop)
for feat in hand_features:
    df[feat] = (df[feat] - np.mean(df[feat]))
    print(feat, " mean and std")
    print(np.mean(df[feat]))
    print(np.std(df[feat]))


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

# # Plot boxplots for the resampled classes
# plt.boxplot([X_train_downsampled[downsampled_data['depression_med'] == 0]['remlaip'],
#              X_train_downsampled[downsampled_data['depression_med'] == 1]['remlaip']],
#             labels=['Class 0', 'Class 1'])
# plt.title('Resampled Classes Boxplots')
# plt.ylabel('Feature Value')
# plt.xlabel('Class')
# plt.savefig('/data/scratch/scadavid/projects/data/figures')

## boxplot for each feature
# train_data['remlaip'] = (train_data['remlaip'] - np.mean(train_data['remlaip'])) / np.std(train_data['remlaip'])
# train_data['timerem'] = (train_data['timerem'] - np.mean(train_data['timerem'])) / np.std(train_data['timerem'])
# train_data['time_first_rem'] = (train_data['time_first_rem'] - np.mean(train_data['time_first_rem'])) / np.std(train_data['time_first_rem'])
# train_data['log(time_first_rem)'] = np.log(df['time_first_rem'])
# train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
# train_data.dropna(inplace=True)
# train_data.drop(columns=['depression_med']).hist()
# plt.savefig('/data/scratch/scadavid/projects/data/figures/features_hist_std.pdf')
# bp()

### grid search best hyperparams ###
# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100, 1000, 10000],  # Regularization parameter
    'kernel': ['rbf'],  # Kernel type
    'degree': [2, 3, 4],  # Degree of the polynomial kernel
    'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient for 'rbf' and 'poly'
    'class_weight': [None],  # Class weights
}

# param_grid = {
#     'C': [1100, 1250, 1350],
#     'kernel': ['rbf'],
#     'gamma': ['scale']  # Only for rbf kernel
# }

# Create the GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_downsampled, y_train_downsampled)

# Create an SVM model
#svm_model = SVC()
svm_model = grid_search.best_estimator_
print(grid_search.best_params_)

# Train the model on the training data
#svm_model.fit(X_train_downsampled, y_train_downsampled)

# Predict on the testing data
y_pred = svm_model.predict(X_test)



# Calculate accuracy
precision = precision_score(y_test, y_pred)
print(f'precision: {precision:.5f}')

recall = recall_score(y_test, y_pred)
print(f'recall: {recall:.5f}')

f1 = f1_score(y_test, y_pred)
print(f'f1: {f1:.5f}')

# then can validate on testset (wsc) as well
bp()
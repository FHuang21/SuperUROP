import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
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
hand_features = ['remlaip', 'timerem', 'waso', 'ai_all', 'nremepop', 'nremepbp', 'slpprdp'] #'remlaiip', 'timeremp'
#label_col = (df['tca1'] + df['ntca1']) > 0
df['depression_med'] = (df['tca2'] + df['ntca2']) > 0
df = pd.concat([df[hand_features], df['depression_med']], axis=1)
df = df.dropna()
df['remlaip'] = df['remlaip'] / 60
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
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create the GridSearchCV object
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_downsampled, y_train_downsampled)
print(grid_search.best_params_)

y_pred = grid_search.predict(X_test)

# Calculate accuracy
precision = precision_score(y_test, y_pred)
print(f'precision: {precision:.5f}')

recall = recall_score(y_test, y_pred)
print(f'recall: {recall:.5f}')

f1 = f1_score(y_test, y_pred)
print(f'f1: {f1:.5f}')

# then can validate on testset (wsc) as well
bp()
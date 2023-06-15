import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from dataLoader2 import MultiClassDataset
from torch.utils.data import DataLoader, random_split, default_collate
import sklearn.metrics

# this file is to verify that the model loads properly, and in fact it does

spec_data_path = '/data/scratch/scadavid/projects/data/eeg_mt_spec/'

# initialize DataLoader
dataset = MultiClassDataset(root=spec_data_path)
trainset, testset = random_split(dataset, [0.7, 0.3])
batch_size = 16
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) # don't shuffle cause not training on the test set

X_test, y_test = default_collate(testset)

# model params
resnet_layer = resnet50(weights=ResNet50_Weights.DEFAULT)
num_features = resnet_layer.fc.in_features  # Get the number of input features for the last layer
num_classes = 2
resnet_layer.fc = nn.Linear(num_features, num_classes)  # Replace the last layer with a new fully connected layer
conv_layer = nn.Conv2d(3, 3, kernel_size=(1,7), stride=(1,3), padding=(0,3))
pool_layer = nn.MaxPool2d((1,2))

# # to load model:
model = nn.Sequential(conv_layer, pool_layer, resnet_layer) # to reset model and see if loading new weights gives same performance as end of training loop
model = model.cpu()
X_test = X_test.cpu()
model.load_state_dict(torch.load(spec_data_path + 'model.pt'))
y_pred = model(X_test)
y_pred_classes = torch.argmax(y_pred, dim=1)
print(sklearn.metrics.classification_report(y_test, y_pred_classes))
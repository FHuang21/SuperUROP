import torch
import sklearn.metrics
import argparse
from dataLoader2 import MultiClassDataset
from torch.utils.data import random_split, default_collate
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

parser = argparse.ArgumentParser(description='evaluate specified model')
parser.add_argument('-f', '--filename', required=True, help='model filename')
args = parser.parse_args()
filename = args.filename

# initialize model that is modification of resnet50 architecture
resnet_layer = resnet50(weights=ResNet50_Weights.DEFAULT)
num_features = resnet_layer.fc.in_features  # Get the number of input features for the last layer
num_classes = 2
resnet_layer.fc = nn.Linear(num_features, num_classes)  # Replace the last layer with a new fully connected layer
conv_layer = nn.Conv2d(3, 3, kernel_size=(1,7), stride=(1,3), padding=(0,3))
pool_layer = nn.MaxPool2d((1,2))
model = nn.Sequential(conv_layer, pool_layer, resnet_layer).cuda()

spec_data_path = '/data/scratch/scadavid/projects/data/eeg_mt_spec/'
models_path = spec_data_path + 'models/'
model_path = models_path + filename
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model = model.cpu()
model.eval()

dataset = MultiClassDataset(root=spec_data_path)
trainset, testset = random_split(dataset, [0.7, 0.3]) # CHANGE THIS
X_test, y_test = default_collate(testset)

X_test = X_test.cpu()
y_pred = model(X_test)
y_pred_classes = torch.argmax(y_pred, dim=1)

print(sklearn.metrics.classification_report(y_test, y_pred_classes))

generator1 = torch.Generator().manual_seed(42)
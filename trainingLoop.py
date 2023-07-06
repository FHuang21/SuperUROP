import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from metrics import Metrics
from dataLoader2 import MultiClassDataset
from tqdm import tqdm
from ipdb import set_trace as bp
import numpy as np
import argparse
import os

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():
        writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)

    writer.flush()

parser = argparse.ArgumentParser(description='trainingLoop w/specified hyperparams')
parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-w1', '--w1', type=float, default=1.0, help='weight for control class')
parser.add_argument('-w2', '--w2', type=float, default=9.0, help='weight for antidep med class')
parser.add_argument('--task', type=str, default='multiclass', help='multiclass or regression')
parser.add_argument('-bs', type=int, default=4, help='batch size')
parser.add_argument('--dataset', type=str, default='shhs2', help='which dataset to train on')
parser.add_argument('--datatype', type=str, default='spec', help='ts or spec')
parser.add_argument('--data_source', type=str, default='eeg', help='eeg, bb, or stage')
parser.add_argument('--target', type=str, default='antidep', help='target label')
args = parser.parse_args()
lr = args.lr
w1 = args.w1
w2 = args.w2
task = args.task
batch_size = args.bs
dataset_name = args.dataset
datatype = args.datatype
data_source = args.data_source

available_devices = range(0, torch.cuda.device_count())

data_path = '/data/scratch/scadavid/projects/data'

writer = SummaryWriter(log_dir='tensorboard_log')

# initialize DataLoader
gen = torch.Generator()
gen.manual_seed(20)
dataset = MultiClassDataset(root=f'{data_path}/eeg_mt_spec')
trainset, testset = random_split(dataset, [0.7, 0.3])
#batch_size = 16
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) 

# initialize model that is modification of resnet50 architecture
resnet_layer = resnet50(weights=ResNet50_Weights.DEFAULT).cuda()
num_features = resnet_layer.fc.in_features  # Get the number of input features for the last layer
num_classes = 2
resnet_layer.fc = nn.Linear(num_features, num_classes)  # Replace the last layer with a new fully connected layer
conv_layer = nn.Conv2d(3, 3, kernel_size=(1,7), stride=(1,3), padding=(0,3)).cuda()
pool_layer = nn.MaxPool2d((1,2)).cuda()
model = nn.DataParallel((nn.Sequential(conv_layer, pool_layer, resnet_layer)).to(available_devices[0]), available_devices)

binary_class_weights = torch.tensor([w1, w2], dtype=torch.float32).to(available_devices[0])
loss_fn = nn.CrossEntropyLoss(weight=binary_class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)
num_epochs = 100
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

metrics = Metrics(args)

max_f1 = 0.0
for epoch in tqdm(range(num_epochs)):

    running_loss = 0.0
    model.train()
    for X_batch, y_batch in tqdm(train_loader):

        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.fill_metrics(y_pred, y_batch)

    epoch_loss = running_loss / len(train_loader)
    computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
    logger(writer, computed_metrics, 'train', epoch)

    metrics.clear_metrics()
    
    scheduler.step()

    model.eval()
    with torch.no_grad():

        running_loss = 0.0
        for X_test_batch, y_test_batch in test_loader:

            X_test_batch = X_test_batch.cuda()
            y_test_batch = y_test_batch.cuda()
            y_pred = model(X_test_batch)
            loss = loss_fn(y_pred, y_test_batch)

            running_loss += loss.item()

            metrics.fill_metrics(y_pred, y_test_batch)
        
        epoch_loss = running_loss / len(test_loader)
        computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
        logger(writer, computed_metrics, 'val', epoch)
        metrics.clear_metrics()

        new_f1 = computed_metrics["f1"].item() # if tensor is single item can get float from it using .item()
        model_path = os.path.join(data_path, 'models', datatype, dataset_name, data_source)
        if new_f1 > max_f1:
            max_f1 = new_f1
            if 'model_name' in globals():
                os.remove(os.path.join(model_path, model_name)) # delete older, worse model
            model_name = f"lr_{lr}_w1_{w1}_w2_{w2}_f1_{round(max_f1, 2)}.pt"
            model_save_path = os.path.join(model_path, model_name)
            torch.save(model.state_dict(), model_save_path)

    torch.cuda.empty_cache()

writer.close()

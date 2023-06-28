import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split, default_collate
from torch.optim import lr_scheduler
import sklearn.metrics
from torch.utils.tensorboard import SummaryWriter
from metrics import Metrics
from dataLoader2 import MultiClassDataset
from tqdm import tqdm
from ipdb import set_trace as bp
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='trainingLoop w/specified hyperparams')
parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-w1', '--w1', type=float, default=1.0, help='weight for control class')
parser.add_argument('-w2', '--w2', type=float, default=9.0, help='weight for antidep med class')
parser.add_argument('--task', type=str, default='multiclass', help='multiclass or regression')
parser.add_argument('--device', type=str, default='cuda', help='cuda')
args = parser.parse_args()
lr = args.lr
w1 = args.w1
w2 = args.w2
task = args.task
device = args.device
# can also add batch_size, more in the future

# can also add batch_size, more in the future

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

torch.cuda.empty_cache() # in case early stopping and then restarting

spec_data_path = '/data/scratch/scadavid/projects/data/eeg_mt_spec/'

def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():
        writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)

    writer.flush()

writer = SummaryWriter(log_dir='tensorboard_log')


# initialize DataLoader
dataset = MultiClassDataset(root=spec_data_path)
trainset, testset = random_split(dataset, [0.7, 0.3])
batch_size = 16
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) # don't shuffle cause not training on the test set

X_test, y_test = default_collate(testset)

# initialize model that is modification of resnet50 architecture
resnet_layer = resnet50(weights=ResNet50_Weights.DEFAULT)
num_features = resnet_layer.fc.in_features  # Get the number of input features for the last layer
num_classes = 2
resnet_layer.fc = nn.Linear(num_features, num_classes)  # Replace the last layer with a new fully connected layer
conv_layer = nn.Conv2d(3, 3, kernel_size=(1,7), stride=(1,3), padding=(0,3))
pool_layer = nn.MaxPool2d((1,2))
model = nn.Sequential(conv_layer, pool_layer, resnet_layer).cuda()

# lr = 1e-3
# w1, w2 = 1.0, 9.0
binary_class_weights = torch.tensor([w1, w2]).cuda()
loss_fn = nn.CrossEntropyLoss(weight=binary_class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)
num_epochs = 100
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

metrics = Metrics(args)

# running_loss = 0.0
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

        y_pred_classes = torch.argmax(y_pred, dim=1)

        metrics.fill_metrics(y_pred_classes, y_batch)

    epoch_loss = running_loss
    computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
    logger(writer, computed_metrics, 'train', epoch)

    metrics.clear_metrics()
    
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]

    model.eval()
    with torch.no_grad():

        running_loss = 0.0
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.cuda()
            y_test_batch = y_test_batch.cuda()
            y_pred = model(X_test_batch)
            loss = loss_fn(y_pred, y_test_batch)

            running_loss += loss.item()

            y_pred_classes = torch.argmax(y_pred, dim=1)
            # print('y_pred_classes shape: ', y_pred_classes.shape)
            # print('y_test_batch shape: ', y_test_batch.shape)
            metrics.fill_metrics(y_pred_classes, y_test_batch)
        
        epoch_loss = running_loss
        computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
        logger(writer, computed_metrics, 'val', epoch)
        metrics.clear_metrics()

        new_f1 = computed_metrics["f1"].item() # if tensor is single item can get float from it using .item()
        if new_f1 > max_f1:
            max_f1 = new_f1
            if 'model_name' in globals():
                os.remove(spec_data_path + 'models/' + model_name) # delete older, worse model
            model_name = str(lr)+'_'+str(w1)+'_'+str(w2)+'_'+str(round(max_f1, 2))+'.pt'
            torch.save(model.state_dict(), spec_data_path + 'models/' + model_name)
        # # Compute the average test loss
        # test_loss = test_loss / total_samples
    # Print and store loss
    
    # print("TEST METRICS:")
    # print("Epoch {}: Average Loss: {:.4f}".format(epoch + 1, test_loss))
    #print(sklearn.metrics.classification_report(y_test, y_pred_classes))
    #del X_test, y_test
    torch.cuda.empty_cache()

writer.close()

# model = model.cpu()
# model.eval()
# X_test = X_test.cpu()
# y_pred = model(X_test)
# y_pred_classes = torch.argmax(y_pred, dim=1)

# print(sklearn.metrics.classification_report(y_test, y_pred_classes))

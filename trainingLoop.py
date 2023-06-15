import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split, default_collate
from torch.optim import lr_scheduler
import sklearn.metrics
from dataLoader2 import MultiClassDataset
from tqdm import tqdm
from ipdb import set_trace as bp
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='trainingLoop w/specified hyperparams')
parser.add_argument('-lr', '--lr', type=float, required=True, help='learning rate')
parser.add_argument('-w1', '--w1', type=float, required=True, help='weight for control class')
parser.add_argument('-w2', '--w2', type=float, required=True, help='weight for antidep med class')
args = parser.parse_args()
lr = args.lr
w1 = args.w1
w2 = args.w2
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

# running_loss = 0.0
max_f1 = 0.0
for epoch in tqdm(range(num_epochs)):

    model.train()

    # y_preds = []
    # y_truths = []
    for X_batch, y_batch in tqdm(train_loader):

        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # y_pred = y_pred.detach().cpu()
        # y_batch = y_batch.cpu()

        # y_preds.append(y_pred.argmax(dim=1))
        # y_truths.append(y_batch)

        #del X_batch, y_batch, y_pred
        torch.cuda.empty_cache()

        # running_loss += loss.item()
    
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    
    # y_preds = torch.cat(y_preds)
    # y_truths = torch.cat(y_truths)

    # print("num y_preds: ", y_preds.sum())
    # print("num y_truths: ", y_truths.sum())
    
    # #y_preds = torch.argmax(y_preds, dim=1)
    
    # y_preds = y_preds.detach().cpu().numpy()
    # y_truths = y_truths.detach().cpu().numpy()
    # accuracy = sklearn.metrics.accuracy_score(y_truths, y_preds)
    # precision = sklearn.metrics.precision_score(y_truths, y_preds, average='macro')
    # recall = sklearn.metrics.recall_score(y_truths, y_preds, average='macro')
    # f1 = sklearn.metrics.f1_score(y_truths, y_preds, average='macro')
    
    # # now calculate loss
    # # perhaps define a function... print_metrics()?
    # average_loss = running_loss / len(train_loader)
    # running_loss = 0.0

    # # print("y_truths:")
    # # print(y_truths)
    # # print("y_preds")
    # # print(y_preds)

    # print("TRAINING METRICS:")
    # print("Accuracy: ", accuracy)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("Average Loss: ", average_loss)
    # print("F1: ", f1)
    # print("TP/TN/FP/FN:", perf_measure(y_truths, y_preds))

    model.eval()
    with torch.no_grad():
        # test_loss = 0.0
        # total_samples = 0

        y_pred_classes = []
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.cuda()
            y_test_batch = y_test_batch.cuda()

            # Compute test loss for the current batch
            y_pred = model(X_test_batch)
            y_pred_classes_batch = y_pred.argmax(dim=1)
            y_pred_classes.append(y_pred_classes_batch)

            # test_loss += loss_fn(y_pred, y_test_batch).item() * X_test_batch.size(0)
            # total_samples += X_test_batch.size(0)
        y_pred_classes = torch.cat(y_pred_classes)
        y_pred_classes = y_pred_classes.detach().cpu().numpy()
        #print(len(y_test), len(y_pred_classes)) # y_test: 788, y_pred_classes: 50 (?????)
        new_f1 = sklearn.metrics.f1_score(y_test, y_pred_classes, average='macro')
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

model = model.cpu()
model.eval()
X_test = X_test.cpu()
y_pred = model(X_test)
y_pred_classes = torch.argmax(y_pred, dim=1)

print(sklearn.metrics.classification_report(y_test, y_pred_classes))

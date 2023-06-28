import torch
import torch.nn as nn
import torch.optim as optim
from model import EEG_Encoder, BranchVarEncoder, BranchVarPredictor
from torch.utils.data import DataLoader, random_split, default_collate
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import sklearn.metrics
from __init__ import EEG_SHHS_Dataset
from tqdm import tqdm
from ipdb import set_trace as bp
import argparse
import os
from metrics import Metrics

# Charlie tensorboard stuff:
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir = tensorboard_log)
# writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)

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
dataset = EEG_SHHS_Dataset()
gen = torch.Generator()
gen.manual_seed(20)
trainset, testset = random_split(dataset, [0.7, 0.3], generator=gen)
batch_size = 8
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) # don't shuffle cause not training on the test set

X_test, y_test = default_collate(testset)

# initialize Ali's model
eeg_encoder = EEG_Encoder().cuda()
pd_encoder = BranchVarEncoder().cuda()
pd_predictor = BranchVarPredictor().cuda()
model = nn.Sequential(eeg_encoder, pd_encoder, pd_predictor)
model = nn.DataParallel(model)

binary_class_weights = torch.tensor([w1, w2]).cuda()
loss_fn = nn.CrossEntropyLoss(weight=binary_class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)
num_epochs = 100
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

metrics = Metrics(args)

max_f1 = -1.0
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

        #torch.cuda.empty_cache()
    
    epoch_loss = running_loss
    computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
    logger(writer, computed_metrics, 'train', epoch)

    metrics.clear_metrics()

    scheduler.step()

    model.eval()
    with torch.no_grad():

        running_loss = 0.0
        #y_pred_classes = []
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.cuda()
            y_test_batch = y_test_batch.cuda()

            # Compute test loss for the current batch
            y_pred = model(X_test_batch)
            # y_pred_classes_batch = y_pred.argmax(dim=1)
            # y_pred_classes.append(y_pred_classes_batch)
            y_pred_classes = torch.argmax(y_pred, dim=1)
            metrics.fill_metrics(y_pred_classes, y_test_batch)
            loss = loss_fn(y_pred_classes, y_test_batch.float()) # y_test_batch is 'Long' type (why tho?)
            running_loss += loss.item()

        epoch_loss = running_loss
        # y_pred_classes = torch.cat(y_pred_classes)
        # y_pred_classes = y_pred_classes.detach().cpu().numpy()
        computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
        logger(writer, computed_metrics, 'val', epoch)
        metrics.clear_metrics()

        # new_f1 = sklearn.metrics.f1_score(y_test, y_pred_classes, average='macro')
        new_f1 = computed_metrics["f1"]
        if new_f1 > max_f1:
            max_f1 = new_f1
            if 'model_name' in globals():
                os.remove(spec_data_path + 'models/' + model_name) # delete older, worse model
            model_name = str(lr)+'_'+str(w1)+'_'+str(w2)+'_'+str(round(max_f1, 2))+'.pt'
            torch.save(model.state_dict(), spec_data_path + 'models/' + model_name)

    torch.cuda.empty_cache()

writer.close()

# model = model.cpu()
# model.eval()
# X_test = X_test.cpu()
# y_pred = model(X_test)
# y_pred_classes = torch.argmax(y_pred, dim=1)

# print(sklearn.metrics.classification_report(y_test, y_pred_classes))
import torch
import torch.nn as nn
import torch.optim as optim
from model import EEG_Encoder, BranchVarEncoder, BranchVarPredictor, BBEncoder, SimplePredictor
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import EEG_SHHS_Dataset, EEG_SHHS2_Dataset, EEG_MGH_Dataset, EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset
from BrEEG.task_spec import SpecDataset, DeepClassifier, get_df
from tqdm import tqdm
from ipdb import set_trace as bp
import numpy as np
import argparse
import os
from metrics import Metrics

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

def set_transform(args):
    crop_size = (256, 2048 // args.downsample_time)
    valid_crop_op = transforms.CenterCrop(crop_size)
    train_crop_op = transforms.RandomCrop(crop_size)
    train_transform = transforms.Compose([
        train_crop_op,
    ])
    valid_transform = transforms.Compose([
        valid_crop_op,
    ])
    return train_transform, valid_transform

def css_to_list(css):
    return [int(i) for i in css.split(",")]

parser = argparse.ArgumentParser(description='trainingLoop w/specified hyperparams')
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-w1', type=float, default=1.0, help='weight for control class')
parser.add_argument('-w2', type=float, default=9.0, help='weight for antidep med class')
parser.add_argument('--task', type=str, default='multiclass', help='multiclass or regression')
parser.add_argument('--num_classes', type=int, default=2, help='for multiclass')
parser.add_argument('--dataset', type=str, default='wsc', help='which dataset to train on')
parser.add_argument('--datatype', type=str, default='encoding', help='ts, spec, or encoding')
parser.add_argument('--data_source', type=str, default='eeg', help='eeg, bb, or stage')
parser.add_argument('--input', type=str, default='eeg', help='eeg, br, others (for Hao datset)')
parser.add_argument('--num_channel', type=int, default=256, help='number of channels')
parser.add_argument('--target', type=str, default='', help='idk')
parser.add_argument('-bs', type=int, default=16, help='batch size')
parser.add_argument('--arch', type=str, default='res18', help='model architecture')
parser.add_argument('--downsample_time', type=int, default=2)
parser.add_argument('--ratio', type=float, default=4)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--label', type=str, default='antidep')
parser.add_argument('--pretrained', action="store_true", default=False)
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--layers', type=str, default="")
#parser.add_argument('--model_mage', type=str, default='20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.5')
args = parser.parse_args()
lr = args.lr
w1 = args.w1
w2 = args.w2
task = args.task
dataset_name = args.dataset
datatype = args.datatype
data_source = args.data_source
label = args.label
batch_size = args.bs
arch = args.arch
debug = args.debug
add_name = args.add_name
pretrained = 'pretrained' if args.pretrained else ''
model_path = args.model_path
layers = css_to_list(args.layers)

data_path = '/data/scratch/scadavid/projects/data'

available_devices = range(0, torch.cuda.device_count())

exp_name = f"exp_lr_{lr}_w1_{w1}_w2_{w2}_ds_{data_source}_bs_{batch_size}_label_{label}_{pretrained}_{add_name}"
exp_event_path = os.path.join('tensorboard_log', datatype, dataset_name, exp_name)
writer = SummaryWriter(log_dir=exp_event_path)

# initialize DataLoader w/ appropriate dataset (EEG/BB, corresponding dataset)
is_hao = False

if (datatype == 'spec'):
    train_transform, val_transform = set_transform(args)
    dataset = SpecDataset(dataset_name, 0, 'all', df=get_df(dataset_name, args), transform=train_transform, args=args) # hao says cv parameter doesn't matter
    is_hao = True
elif (dataset_name == 'shhs1'):
    dataset = EEG_SHHS_Dataset(args)
elif (dataset_name == 'shhs2' and (datatype == 'eeg' or datatype == 'bb')):
    dataset = EEG_SHHS2_Dataset(args)
elif (dataset_name == 'shhs2' and datatype == 'encoding'):
    dataset = EEG_Encoding_SHHS2_Dataset(label=label)
elif (dataset_name == 'wsc'):
    dataset = EEG_Encoding_WSC_Dataset(label=label)
elif (dataset_name == 'mgh'):
    dataset = EEG_MGH_Dataset(args)
# decide which model to use
if (data_source == 'eeg' and datatype == 'ts'):
    model = nn.DataParallel(nn.Sequential(EEG_Encoder(), BranchVarEncoder(args), BranchVarPredictor(args)).to(available_devices[0]), available_devices)
elif (datatype == 'encoding'):
    
    model = SimplePredictor().to(available_devices[0]) if layers=="" else SimplePredictor(layer_dims=layers).to(available_devices[0])
    print("simple predictor model")
    if (args.pretrained):
        ## FIXME: this doesn't work if you want to load a model from different dataset/different input data
        model_path = args.model_path
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
elif (data_source == 'bb' and datatype == 'ts'):
    model = nn.DataParallel(nn.Sequential(BBEncoder(), BranchVarEncoder(args), BranchVarPredictor(args)).to(available_devices[0]), available_devices)
else: # DeepClassifier can be used for both EEG and BR spectrograms
    model = nn.DataParallel(DeepClassifier(args).to(available_devices[0]), available_devices)

gen = torch.Generator()
gen.manual_seed(20)
trainset, testset = random_split(dataset, [0.7, 0.3], generator=gen)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) # don't shuffle cause not training on the test set

## get # pos instances
num_pos_train = 0
for X_batch, y_batch in trainset:
    num_pos_train += y_batch.sum().item()
num_pos_val = 0
for X_batch, y_batch in testset:
    num_pos_val += y_batch.sum().item()
print("length trainset: ", len(trainset))
print("num pos in train: ", num_pos_train)
print("length testset: ", len(testset))
print("num pos in val: ", num_pos_val)

#bp()

binary_class_weights = torch.tensor([w1, w2], dtype=torch.float32).to(available_devices[0])
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
        # if (batch_size != len(y_batch)): # DataParallel issue...
        #     continue
        #print("X_batch shape: ", X_batch.shape)
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        #bp()
        y_pred = model(X_batch) if not is_hao else model(X_batch)[0] # Hao's model returns tuple (y_pred, embedding)

        # print("y_pred shape: ", y_pred.shape)
        # print("y_batch shape: ", y_batch.shape)

        loss = loss_fn(y_pred, y_batch)

        # print("y_pred: ", y_pred)
        # print("y_batch: ", y_batch)
        # print("loss: ", loss)

        running_loss += loss.item()
        #print("running loss: ", running_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.fill_metrics(y_pred, y_batch)

    epoch_loss = running_loss / len(train_loader)
    print("epoch_loss: ", epoch_loss)
    computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
    logger(writer, computed_metrics, 'train', epoch)
    metrics.clear_metrics()

    scheduler.step()

    model.eval()
    with torch.no_grad():

        running_loss = 0.0
        for X_batch, y_batch in test_loader:
            # if (batch_size != len(y_batch)):
            #     continue
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
            y_pred = model(X_batch) if not is_hao else model(X_batch)[0]
            # print("y_pred shape: ", y_pred.shape)
            # print("y_batch shape: ", y_batch.shape)
            loss = loss_fn(y_pred, y_batch)

            running_loss += loss.item()

            # print("y_pred: ", y_pred)
            # print("y_batch: ", y_batch)
            # print("loss: ", loss)

            #print("filling eval metrics...")
            metrics.fill_metrics(y_pred, y_batch)

        epoch_loss = running_loss / len(test_loader)
        computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
        logger(writer, computed_metrics, 'val', epoch)
        metrics.clear_metrics()

        new_f1 = computed_metrics["1_f1"].item() # mainly care about the positive class f1 score

        model_path = os.path.join(data_path, 'models', datatype, dataset_name, data_source)
        if new_f1 > max_f1:
            max_f1 = new_f1
            if 'model_name' in globals():
                os.remove(os.path.join(model_path, model_name)) # delete older, worse model (is this necessary?)
            model_name = f"lr_{lr}_w1_{w1}_w2_{w2}_posf1_{round(max_f1, 2)}_{label}_{args.layers}.pt"
            model_save_path = os.path.join(model_path, model_name)
            torch.save(model.state_dict(), model_save_path)

    torch.cuda.empty_cache()

writer.close()
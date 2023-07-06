import torch
import torch.nn as nn
import torch.optim as optim
from model import EEG_Encoder, BranchVarEncoder, BranchVarPredictor, BBEncoder
from torch.utils.data import DataLoader, random_split, default_collate
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
# import torchmetrics
# import sklearn.metrics
from dataset import EEG_SHHS_Dataset, EEG_SHHS2_Dataset, EEG_MGH_Dataset
from BrEEG.task_spec import SpecDataset, DeepClassifier # EEG_SPEC dataset, model from Hao
from tqdm import tqdm
from ipdb import set_trace as bp
import argparse
import os
from metrics import Metrics

def get_df(data, args):
    info = SpecDataset.info

    df = getattr(info, f'df_{data}')

    if args.label == 'ad':
        if data == 'shhs2':
            is_pos = df.alzh2 == 1
        elif data == 'mros2':
            is_pos = (df.m1alzh == 1) | (df.mhalzh == 1) | (df.mhalzh == '1')
        elif data == 'mros1':
            is_pos = df.m1alzh == 1
        else:
            assert False

    elif args.label == 'pd':

        if data == 'shhs2':
            is_pos = df['prknsn2'] == 1
        elif data in ['mros1', 'mros2']:
            is_pos = (df['mhpark'] == '1') | (df['mhparkt'] == '1')
        else:
            assert False
    
    elif args.label == 'antidep':
        if data == 'shhs1': #FIXME: 
            is_pos = (df['TCA1'] == 1) | (df['NTCA1'] == 1)
        elif data == 'shhs2':
            is_pos = (df['TCA2'] == 1) | (df['NTCA2'] == 1)
        elif data == 'mgh':
            is_pos = df['dx_elix_depre']
        else:
            assert False

    else:
        assert False

    df_pos = df[is_pos]
    df_neg = df[~is_pos]

    num_pos = len(df_pos)
    num_neg = int(num_pos * args.ratio)
    df_neg = df_neg[:num_neg]
    print(f'#Pos {len(df_pos)} #Neg {len(df_neg)}')

    df_pos = df_pos.assign(label=1)
    df_neg = df_neg.assign(label=0)

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    return df

parser = argparse.ArgumentParser(description='trainingLoop w/specified hyperparams')
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-w1', type=float, default=1.0, help='weight for control class')
parser.add_argument('-w2', type=float, default=9.0, help='weight for antidep med class')
parser.add_argument('--task', type=str, default='multiclass', help='multiclass or regression')
#parser.add_argument('--device', type=int, default=0, help='cuda device #')
parser.add_argument('--dataset', type=str, default='shhs1', help='which dataset to train on')
parser.add_argument('--datatype', type=str, default='ts', help='ts or spec')
parser.add_argument('--data_source', type=str, default='eeg', help='eeg, bb, or stage')
parser.add_argument('--input', type=str, default='br', help='eeg, br, others (for Hao datset)')
parser.add_argument('--num_channel', type=int, default=256, help='number of channels')
parser.add_argument('--target', type=str, default='', help='idk')
parser.add_argument('-bs', type=int, default=4, help='batch size')
parser.add_argument('--arch', type=str, default='res50', help='model architecture')
parser.add_argument('--downsample_time', type=int, default=2)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--label', type=str, default='antidep')
parser.add_argument('--model_mage', type=str, default='20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.5')
args = parser.parse_args()
lr = args.lr
w1 = args.w1
w2 = args.w2
task = args.task
dataset_name = args.dataset
datatype = args.datatype
data_source = args.data_source
target = args.target
batch_size = args.bs
arch = args.arch
# can also add batch_size, more in the future

data_path = '/data/scratch/scadavid/projects/data'

available_devices = range(0, torch.cuda.device_count())
# print(available_devices)
# print(torch.cuda.is_available())

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


exp_name = f"exp_lr_{lr}_w1_{w1}_w2_{w2}_ds_{data_source}"
exp_event_path = os.path.join('tensorboard_log', datatype, dataset_name, exp_name)
writer = SummaryWriter(log_dir=exp_event_path)

# eeg_encoder = EEG_Encoder()
# branch_var_encoder = BranchVarEncoder(args)
# branch_var_predictor = BranchVarPredictor(args)
# bb_encoder = BBEncoder()
# spec_classifier = DeepClassifier(args)

# initialize DataLoader w/ appropriate dataset (EEG/BB, corresponding dataset)
if (data_source == 'eeg' and datatype == 'spec'):
    dataset = SpecDataset(dataset_name, 0, 'all', df=get_df(dataset_name, args), args=args) # hao says cv parameter doesn't matter
elif (dataset_name == 'shhs1'):
    dataset = EEG_SHHS_Dataset(args)
elif (dataset_name == 'shhs2'):
    dataset = EEG_SHHS2_Dataset(args)
elif (dataset_name == 'mgh'):
    dataset = EEG_MGH_Dataset(args)
# decide which model to use
if (data_source == 'eeg' and datatype == 'ts'):
    model = nn.DataParallel(nn.Sequential(EEG_Encoder(), BranchVarEncoder(args), BranchVarPredictor(args)).to(available_devices[0]), available_devices)
elif (data_source == 'bb' and datatype == 'ts'):
    model = nn.DataParallel(nn.Sequential(BBEncoder(), BranchVarEncoder(args), BranchVarPredictor(args)).to(available_devices[0]), available_devices)
else: # DeepClassifier can be used for both EEG and BR spectrograms
    model = nn.DataParallel(DeepClassifier(args).to(available_devices[0]), available_devices)

# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 338.00 MiB (GPU 0; 11.91 GiB total capacity; 
# 3.76 GiB already allocated; 214.12 MiB free; 4.01 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory 
# try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

gen = torch.Generator()
gen.manual_seed(20)
trainset, testset = random_split(dataset, [0.7, 0.3], generator=gen)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) # don't shuffle cause not training on the test set

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

        #print("X_batch shape: ", X_batch.shape)
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        #bp()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        print("y_pred: ", y_pred)
        print("y_batch: ", y_batch)
        print("loss: ", loss)

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #y_pred_classes = torch.argmax(y_pred, dim=1)

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
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.cuda()
            y_test_batch = y_test_batch.cuda()
            y_pred = model(X_test_batch)
            loss = loss_fn(y_pred, y_test_batch)

            running_loss += loss.item()

            #y_pred_classes = torch.argmax(y_pred, dim=1)
            # print('y_pred_classes shape: ', y_pred_classes.shape)
            # print('y_test_batch shape: ', y_test_batch.shape)
            metrics.fill_metrics(y_pred, y_test_batch)

        epoch_loss = running_loss / len(test_loader)
        computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
        logger(writer, computed_metrics, 'val', epoch)
        metrics.clear_metrics()

        new_f1 = computed_metrics["f1"].item()

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
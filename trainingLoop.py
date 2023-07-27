import torch
import torch.nn as nn
import torch.optim as optim
from model import EEG_Encoder, BranchVarEncoder, BranchVarPredictor, BBEncoder, SimplePredictor, SimpleAttentionPredictor, SimonModel
from torch.utils.data import DataLoader, Subset #, random_split
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import EEG_SHHS_Dataset, EEG_SHHS2_Dataset, EEG_MGH_Dataset, EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset, DatasetCombiner
from BrEEG.task_spec import SpecDataset, DeepClassifier, get_df
from sklearn.model_selection import KFold
from copy import deepcopy
from tqdm import tqdm
from ipdb import set_trace as bp
#import numpy as np
import argparse
import os
from metrics import Metrics

torch.manual_seed(20)

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

def css_to_float_list(css):
    return [float(i) for i in css.split(",")]

def css_to_int_list(css):
    return [int(i) for i in css.split(",")]

def css_to_bool_list(css):
    bool_list = []
    for i in css.split(","):
        i = int(i)
        if i==1:
            bool_list.append(True)
        elif i==0:
            bool_list.append(False)

    return bool_list


parser = argparse.ArgumentParser(description='trainingLoop w/specified hyperparams')
parser.add_argument('-lr', type=float, default=4e-4, help='learning rate')
parser.add_argument('-w', type=str, default='1.0,10.0', help='respective class weights (comma-separated)')
parser.add_argument('--task', type=str, default='multiclass', help='multiclass or regression')
parser.add_argument('--num_classes', type=int, default=2, help='for multiclass')
parser.add_argument('--dataset', type=str, default='wsc', help='which dataset to train on')
parser.add_argument('--datatype', type=str, default='encoding', help='ts, spec, or encoding')
parser.add_argument('--data_source', type=str, default='eeg', help='eeg, bb, or stage')
parser.add_argument('--input', type=str, default='eeg', help='eeg, br, others (for Hao datset)') # having just input (eeg/br) and format (enc/ts/spec) makes sense
parser.add_argument('--num_channel', type=int, default=256, help='number of channels')
parser.add_argument('--target', type=str, default='', help='idk')
parser.add_argument('-bs', type=int, default=16, help='batch size')
parser.add_argument('--arch', type=str, default='res18', help='model architecture') # Hao
parser.add_argument('--downsample_time', type=int, default=2) # Hao
parser.add_argument('--ratio', type=float, default=4) # Hao
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--label', type=str, default='antidep', help="dep, antidep, or benzo")
parser.add_argument('--pretrained', action="store_true", default=False)
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--num_folds', type=int, default=5, help="for cross-validation")
parser.add_argument('--num_heads', type=int, default=3, help="for attention condensation")
parser.add_argument('--add_name', type=str, default="", help="adds argument to the experiment name")
parser.add_argument('--layer_dims', type=str, default="256,64,16", help="for NN predictor") # note to Ali: this doesn't matter for SimonModel
parser.add_argument('--batch_norms', type=str, default="0,0,0", help="for each layer")
parser.add_argument('--dropout', type=float, default=0.5, help="for all layers")
parser.add_argument('--no_attention', action='store_true', default=False, help="use simple, no attention predictor")
parser.add_argument('--control', action='store_true', default=False, help="just train on control")
parser.add_argument('--tca', action='store_true', default=False, help="just train on tca")
parser.add_argument('--ntca', action='store_true', default=False, help="just train on ntca") # only shhs2
parser.add_argument('--ssri', action='store_true', default=False, help="just train on ssri") # only wsc
parser.add_argument('--other', action='store_true', default=False, help="just train on other") # only wsc
parser.add_argument('--simon_model', action='store_true', default=False, help="use simon model")
parser.add_argument('--hidden_size', type=int, default=8, help="for SimonModel")
parser.add_argument('--fc2_size', type=int, default=32, help="for SimonModel")
#parser.add_argument('--model_mage', type=str, default='20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.5')
args = parser.parse_args()
lr = args.lr
weights = css_to_float_list(args.w)
task = args.task
dataset_name = args.dataset
datatype = args.datatype
data_source = args.data_source
label = args.label
num_classes = args.num_classes
num_class_name = f"class_{num_classes}"
batch_size = args.bs
arch = args.arch
debug = args.debug
num_epochs = args.num_epochs
#num_folds = args.num_folds
add_name = args.add_name
pretrained = '_pretrained' if args.pretrained else ''
model_path = args.model_path
#with_attention = args.attention
att = "" if args.no_attention else "_att"
ctrl = "" if not args.control else "_ctrl"
layer_dims_str = f"_{args.layer_dims}"
batch_norms_str = f"_{args.batch_norms}"
args.layer_dims = css_to_int_list(args.layer_dims)
args.batch_norms = css_to_bool_list(args.batch_norms)
dpt = args.dropout
dpt_str = f"_{dpt}"
print("Label: ", label)

#data_path = '/data/scratch/scadavid/projects/data'

#available_devices = range(0, torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

# initialize DataLoader w/ appropriate dataset (EEG/BB, corresponding dataset)
is_hao = False
if (datatype == 'spec'):
    train_transform, val_transform = set_transform(args)
    dataset = SpecDataset(dataset_name, 0, 'all', df=get_df(dataset_name, args), transform=train_transform, args=args) # hao says cv parameter doesn't matter
    is_hao = True
elif (dataset_name == 'shhs2_wsc'):
    trainset = DatasetCombiner(datasets=[EEG_Encoding_SHHS2_Dataset(args), EEG_Encoding_WSC_Dataset(args)], phase='train')
    testset = DatasetCombiner(datasets=[EEG_Encoding_SHHS2_Dataset(args), EEG_Encoding_WSC_Dataset(args)], phase='val')
elif (dataset_name == 'shhs1'):
    dataset = EEG_SHHS_Dataset(args)
elif (dataset_name == 'shhs2' and (datatype == 'eeg' or datatype == 'bb')):
    dataset = EEG_SHHS2_Dataset(args)
elif (dataset_name == 'shhs2' and datatype == 'encoding'):
    dataset = EEG_Encoding_SHHS2_Dataset(args)
elif (dataset_name == 'wsc'):
    dataset = EEG_Encoding_WSC_Dataset(args)
elif (dataset_name == 'mgh'):
    dataset = EEG_MGH_Dataset(args)

# decide which model to use
if (args.simon_model):
    model = SimonModel(args).to(device)
    print("Simon model!")
elif (data_source == 'eeg' and datatype == 'ts'):
    model = nn.Sequential(EEG_Encoder(), BranchVarEncoder(args), BranchVarPredictor(args)).to(device)
elif (datatype == 'encoding' and not args.no_attention):
    model = SimpleAttentionPredictor(args).to(device)
    print("simple attention predictor")
elif (datatype == 'encoding'):
    model = SimplePredictor(output_dim=num_classes).to(device)
    print("simple predictor model")
    if (args.pretrained):
        ## FIXME
        model_path = args.model_path
        state_dict = torch.load(model_path)
        # del state_dict['fc_final.weight'] # works in bp() environment but not here????
        # del state_dict['fc_final.bias']
        modules = [child for child in model.children()]
        modules = modules[:-1]
        model = nn.Sequential(*modules, nn.Linear(modules[-1][0].out_features, num_classes)).to(device)
        # bp()
        # model.load_state_dict(state_dict)
elif (data_source == 'bb' and datatype == 'ts'):
    model = nn.Sequential(BBEncoder(), BranchVarEncoder(args), BranchVarPredictor(args)).to(device)
else: # DeepClassifier can be used for both EEG and BR spectrograms
    model = DeepClassifier(args).to(device)

# gen = torch.Generator()
# gen.manual_seed(20)
# if 'trainset' not in globals() or 'testset' not in globals(): # i.e. haven't combined datasets
#     trainset, testset = random_split(dataset, [0.7, 0.3], generator=gen)
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) # don't shuffle cause not training on the test set

#torch.manual_seed(20) # for when torch shuffles the train loader

kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=20)

# just going to use fold 0
# fold, (train_ids, test_ids) = next(enumerate(kfold.split(dataset)))
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print("----FOLD ", fold, "----")

    n_model = deepcopy(model).to(device) # need to reset model w/ untrained params each fold so no overfitting
    #n_model = model

    exp_name = f"exp_lr_{lr}_w_{args.w}_ds_{data_source}_bs_{batch_size}_epochs_{num_epochs}_fold{fold}{pretrained}{layer_dims_str}_heads{args.num_heads}{ctrl}{add_name}"
    folder_path = "/data/scratch/scadavid/projects/code/tensorboard_log/test" #FIXME::: change to what you want
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder path '{folder_path}' created successfully.")
    exp_event_path = os.path.join(folder_path, exp_name)
    writer = SummaryWriter(log_dir=exp_event_path)

    trainset = Subset(dataset, train_ids)
    testset = Subset(dataset, test_ids)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights) if task=='multiclass' else nn.MSELoss()
    optimizer = optim.Adam(n_model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    metrics = Metrics(args)

    max_f1 = -1.0
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        n_model.train()

        for X_batch, y_batch in tqdm(train_loader):

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = n_model(X_batch) if not is_hao else n_model(X_batch)[0] # Hao's model returns tuple (y_pred, embedding)
            #bp()
            # threshold zung score here
            y_batch_classes = (y_batch >= 36).int()
            loss = loss_fn(y_pred, y_batch_classes.long())
            running_loss += loss.item()

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

        n_model.eval()
        with torch.no_grad():

            running_loss = 0.0
            for X_batch, y_batch in test_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = n_model(X_batch) if not is_hao else n_model(X_batch)[0]

                y_batch_classes = (y_batch >= 36).int()
                loss = loss_fn(y_pred, y_batch_classes.long())
                running_loss += loss.item()

                metrics.fill_metrics(y_pred, y_batch) # feed the raw scores, not thresh'd

            epoch_loss = running_loss / len(test_loader)
            computed_metrics = metrics.compute_and_log_metrics(epoch_loss)
            logger(writer, computed_metrics, 'val', epoch)
            metrics.clear_metrics()

            new_f1 = computed_metrics["f1_macro"].item() # shoudl be f1_macro if multiple positive labels, 1_f1 if binary

            # ## temp changes
            # #model_path = os.path.join(data_path, 'models', datatype, dataset_name, data_source, label, num_class_name)
            model_path = f"/data/scratch/scadavid/projects/data/models/encoding/shhs2/eeg/antidep/class_2/simonmodelantidep"

            # ## TEMP
            # if((epoch+1) % 5 == 0):
            #     model_name = f"lr_{lr}_w_{args.w}_bs_{batch_size}_f1macro_{round(max_f1, 2)}{layer_dims_str}_bns{batch_norms_str}_heads{args.num_heads}{dpt_str}{pretrained}{att}{ctrl}{add_name}_fold{fold}_epoch{epoch}.pt"
            #     model_save_path = os.path.join(model_path, model_name)
            #     if not os.path.exists(model_path):
            #         # Create the folder if it does not exist
            #         os.makedirs(model_path)
            #         #print(f"Folder '{model_save_path}' created successfully.")
            #     torch.save(n_model.state_dict(), model_save_path)
            # if new_f1 > max_f1:
            #     max_f1 = new_f1
            #     if 'model_name' in globals():
            #         try:
            #             os.remove(os.path.join(model_path, model_name)) # FIXED
            #             print("model removed.")
            #         except:
            #             print("model not removed.")
            #     model_name = f"lr_{lr}_w_{args.w}_bs_{batch_size}_f1macro_{round(max_f1, 2)}{layer_dims_str}_bns{batch_norms_str}_heads{args.num_heads}{dpt_str}{pretrained}{att}{ctrl}{add_name}_fold{fold}.pt"
            #     model_save_path = os.path.join(model_path, model_name)
            #     torch.save(n_model.state_dict(), model_save_path)

        torch.cuda.empty_cache()

    # model_name = "" # otherwise it overwrites the best model from the previous fold

    writer.close()
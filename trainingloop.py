import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset #, random_split
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import EEG_SHHS_Dataset, EEG_SHHS2_Dataset, EEG_MGH_Dataset, EEG_Encoding_SHHS2_Dataset, EEG_Encoding_WSC_Dataset, DatasetCombiner
from sklearn import KFold
from copy import deepcopy
from tqdm import tqdm
from ipdb import set_trace as bp
#import numpy as np
import argparse
import os
from metrics import Metrics
import numpy as np 
import pandas as pd

'''
Performance measurement
Returns True Positive, False Positive, True Negative, False Negative Count
'''
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
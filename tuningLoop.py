import torch
import torch.nn as nn
import torch.optim as optim
from model import EEG_Encoder, BranchVarEncoder, BranchVarPredictor, BBEncoder, SimplePredictor, SimpleAttentionPredictor, SimonModel, SimonModel_Antidep
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
folder_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD"


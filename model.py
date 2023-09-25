import torch 
import torch.nn as nn
from torch.nn import functional as F
from ipdb import set_trace as bp
# from BrEEG.task_spec import ResidualBlock1D
from dataset import EEG_Encoding_SHHS2_Dataset, EEG_SHHS_Dataset
from torch.utils.data import  DataLoader
import math 
import argparse
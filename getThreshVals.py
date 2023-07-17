import argparse
import torch
#from torch.utils.data import Dataset
from dataset import EEG_Encoding_WSC_Dataset, EEG_Encoding_SHHS2_Dataset
from ipdb import set_trace as bp 

parser = argparse.ArgumentParser(description="bruh")
parser.add_argument('--attention', action='store_true', default=True)
args = parser.parse_args()

wsc_dataset = EEG_Encoding_WSC_Dataset(args)
print("wsc dataset")
wsc_df = wsc_dataset.threshold_values()
print(wsc_df)

shhs2_dataset = EEG_Encoding_SHHS2_Dataset(args)
print("\nshhs2 dataset")
print(shhs2_dataset.threshold_values())

bp()
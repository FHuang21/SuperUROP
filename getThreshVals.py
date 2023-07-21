import argparse
#import torch
#from torch.utils.data import Dataset
from dataset import EEG_Encoding_WSC_Dataset, EEG_Encoding_SHHS2_Dataset
from ipdb import set_trace as bp 

parser = argparse.ArgumentParser(description="bruh")
parser.add_argument('--no_attention', action='store_true', default=False)
parser.add_argument('--label', type=str, default='dep')
parser.add_argument('--control', action='store_true', default=False)
parser.add_argument('--num_classes', type=int, default=2)
args = parser.parse_args()

# wsc_dataset = EEG_Encoding_WSC_Dataset(args)
# print("wsc dataset")
# wsc_df = wsc_dataset.threshold_values()
# print(wsc_df)

shhs2_dataset = EEG_Encoding_SHHS2_Dataset(args)
print("\nshhs2 dataset")
shhs2_df = shhs2_dataset.threshold_values()
print(shhs2_df)
print("threshold: ", shhs2_dataset.th)

#bp()
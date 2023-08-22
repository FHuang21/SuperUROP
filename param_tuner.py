import os 
import random 

BALANCED_OR_WEIGHTED 
EQUI_OVER_SAMPLED
NO_FIXED_LEARNED_ENCODING
BINARY_MULTICLASS

## Model Params 
hidden_size = []
fc1_size = []
fc2_size = []
num_heads = []

## Tuning Params
dropout = []
lr = []
epochs = []
bs = []


os.system(f"CUDA_VISIBLE_DEVICES=3 python trainingLoop.py -lr 2e-4 -w 1.0,1.0 -bs 16 --num_classes 2 --num_heads 4 --dataset shhs2 --label antidep --num_epochs 30 --simon_model --add_name BINARY_no_pe_081723_balanced3 --hidden_size 8 --fc2_size 32 --dropout 0.4 --task binary &")

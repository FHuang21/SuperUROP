import os 
import random 
import numpy as np
from ipdb import set_trace as bp
# BALANCED_OR_WEIGHTED 
# EQUI_OVER_SAMPLED
# NO_FIXED_LEARNED_ENCODING
# BINARY_MULTICLASS
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# ## Model Params 
# hidden_size = []
# fc1_size = []
# fc2_size = []
# num_heads = []
folder_path = "/data/scratch/alimirz/2023/SIMON/TENSORBOARD/AUTOMATIC_TUNING/"

good_seeds = ['exp_lr_0.0006_w_1.0,1.0_ds_eeg_bs_16_epochs_15_dpt_0.35_fold0_256,64,16_heads4_PARAM_TUNER_08-24-23_PEamp_1',
              'exp_lr_0.0003_w_1.0,1.0_ds_eeg_bs_16_epochs_20_dpt_0.3_fold0_256,64,16_heads4_PARAM_TUNER_08-24-23_PEamp_1',
            #   'exp_lr_0.001_w_1.0,1.0_ds_eeg_bs_4_epochs_30_dpt_0.35_fold0_256,64,16_heads4_PARAM_TUNER_08-24-23_PEamp_1', needs to just run longer
              'exp_lr_0.0003_w_1.0,1.0_ds_eeg_bs_16_epochs_19_dpt_0.35_fold0_256,64,16_heads4_PARAM_TUNER_08-24-23_PEamp_1',
              'exp_lr_0.0006_w_1.0,1.0_ds_eeg_bs_16_epochs_16_dpt_0.3_fold0_256,64,16_heads4_PARAM_TUNER_08-24-23_PEamp_1',
              'exp_lr_0.002_w_1.0,1.0_ds_eeg_bs_10_epochs_19_dpt_0.25_fold0_256,64,16_heads4_PARAM_TUNER_08-24-23_PEamp_1',
              
              'exp_lr_0.0004_w_1.0,1.0_ds_eeg_bs_16_epochs_20_dpt_0.5_fold0_256,64,16_heads4balanced_optimization081023_final',
              'exp_lr_0.0002_w_1.0,1.0_ds_eeg_bs_16_epochs_30_dpt_0.5_fold0_256,64,16_heads4_08-22-23_fixedpe_PEamp_1',
              ]
tried_experiments = []
GOOD_SEEDS = True
if GOOD_SEEDS:
    STARTS = []
    for item in good_seeds:
        splitted = item.split('_')
        STARTS.append([splitted[2], splitted[8], splitted[10], splitted[12]])
    dropout = np.array([-0.05,0,0,0,0.05])
    lr = np.array([-2e-4,-1e-4,0,0,0,1e-4,2e-4])
    epochs = [-5,-3,0,0,0,3,5]
    bs = [-2,0,0,0,2]
else:
    ## Tuning Params
    dropout = np.arange(0.3,0.6,0.05)
    lr = np.arange(1e-4,2e-3,1e-4)
    epochs = np.arange(15,45,5)
    bs = np.arange(4,24,4)

print("Total Possible Experiments: ", len(dropout) * len(lr) * len(epochs) * len(bs))

NUM_ITERATIONS = 30

for round in range(NUM_ITERATIONS):
    round_dropout = float(random.choice(dropout).round(3))
    round_lr = float(random.choice(lr).round(3))
    round_epochs = int(random.choice(epochs))
    round_bs = int(random.choice(bs))
    if GOOD_SEEDS:
        which_seed = random.choice(STARTS)
        round_dropout += float(which_seed[3])
        round_lr += float(which_seed[0])
        round_epochs += int(which_seed[2])
        round_bs += int(which_seed[1])
    
    SAVE_NAME = f"exp_lr_{round_lr}_w_1.0,1.0_ds_eeg_bs_{round_bs}_epochs_{round_epochs}_dpt_{round_dropout}_fold0_256,64,16_heads4_PARAM_TUNER_08-24-23_PEamp_1"
    if SAVE_NAME in os.listdir(folder_path):
        print('run already found! skipping')
        continue
    if str([round_dropout, round_lr, round_epochs, round_bs]) in tried_experiments:
        print("Skipping repeat")
        continue

    COMMAND = f"CUDA_VISIBLE_DEVICES={round % 4} python trainingLoop.py -lr {round_lr} -w 1.0,1.0 -bs {round_bs} --num_classes 2 --num_heads 4 --dataset shhs2 --label antidep --num_epochs {round_epochs} --simon_model --add_name PARAM_TUNER_08-24-23 --hidden_size 8 --fc2_size 32 --dropout {round_dropout} --task multiclass --pe_fixed &"
    if round % 4 == 3:
        COMMAND = COMMAND[:-1]
    os.system(COMMAND)
    print('done with round ', round)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ipdb import set_trace as bp 
from torch.utils.tensorboard import SummaryWriter
import numpy as np

log_dir = './neweventout'

import os 

# filename = 'exp_lr_0.0004_w_1.0,10.0_ds_eeg_bs_16_epochs_100_fold0_256,128,16_heads3_ctrl_new2'
# folders_to_combine = [filename.split('fold')[0] + item + filename.split('fold')[1][1:] for item in range(5)]

folders_to_combine = ['exp_lr_0.0004_w_1.0,10.0_ds_eeg_bs_16_epochs_100_fold{0}_256,64,16_heads3_ctrl_bn11_nofc0'.format(item) for item in range(5)]
#exp_lr_0.0004_w_1.0,10.0_ds_eeg_bs_16_epochs_100_fold{0}_256,64,16_heads3_ctrl_dropout0.5_nofc0
#exp_lr_0.0004_w_1.0,10.0_ds_eeg_bs_16_epochs_100_fold{0}_256,64,16_heads3_ctrl_dropout0_nofc0
#exp_lr_0.0004_w_1.0,10.0_ds_eeg_bs_16_epochs_100_fold{0}_256,64,16_heads3_ctrl_bn00_nofc0
#exp_lr_0.0004_w_1.0,10.0_ds_eeg_bs_16_epochs_100_fold{0}_256,64,16_heads3_ctrl_bn01_nofc0
#exp_lr_0.0004_w_1.0,10.0_ds_eeg_bs_16_epochs_100_fold{0}_256,64,16_heads3_ctrl_bn10_nofc0
#exp_lr_0.0004_w_1.0,10.0_ds_eeg_bs_16_epochs_100_fold{0}_256,64,16_heads3_ctrl_bn11_nofc0

def parse_tensorboard_log(log_file_path):
    event_acc = EventAccumulator(log_file_path)
    event_acc.Reload()
    # Get all tags from the log file
    tags = event_acc.Tags()
    # Check if there are any scalars available

    # Retrieve the loss at each epoch
    outputs = {}

    for scalar_name in tags['scalars']:
        outputs[scalar_name] = []

        for scalar in event_acc.Scalars(scalar_name):
            outputs[scalar_name].append((scalar.step, scalar.value))

    return outputs

# Specify the path to your TensorBoard log file


# Parse the TensorBoard log file and get the loss at each epoch

all_results = {}
for folder in folders_to_combine:
    folder_path = os.path.join('/data/scratch/scadavid/projects/code/tensorboard_log/encoding/wsc/dep/class_2', folder)
    all_files = os.listdir(folder_path)
    if len(all_files) != 1:
        raise Exception()
    all_results[folder] = parse_tensorboard_log(os.path.join(folder_path, all_files[0]))

combined_output = {}
for key in all_results[folder]:
    step_output = [item[0] for item in all_results[folder][key]]
    key_output = np.array([[item[1] for item in all_results[folder][key]] for folder in folders_to_combine])
    combined_output[key] = zip(step_output,list(np.average(key_output, axis=0)))
# Create a summary writer
writer = SummaryWriter(log_dir)

# Log the train loss values
for key in combined_output:
    for epoch, loss in list(combined_output[key]):
        writer.add_scalar(key, loss, epoch)

# Close the summary writer
writer.close()
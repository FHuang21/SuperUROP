#import pandas as pd
import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
#from torchvision import transforms
#from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
#from BrEEG.task_spec import SpecMultitaperFolder, SpecFolder
from PIL import Image
from tqdm import tqdm
import pickle
#from ipdb import set_trace as bp

spec_data_path = '/data/scratch/scadavid/projects/data/eeg_mt_spec/'

# returns list of all unique image names in the given folder
def get_unique_image_names(folder_path):
    unique_names = set()  # Set to store unique image names

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has an image extension
            if file.lower().endswith(('.jpg', '.jpeg')) and not file.startswith('._'):
                # Get the image name w/out extension
                image_name = os.path.splitext(file)[0]
                # Add the image name to the set
                unique_names.add(image_name)

    return list(unique_names)

# checks whether the given image is in the given folder
def is_image_in_folder(image_filename, folder_path):
    for root, dirs, files in os.walk(folder_path):
        if image_filename in files:
            return True  # image file found in the folder
    return False  # image file not found in the folder

# given image names and the root directory where they're organized into label folders,
# returns two lists: one of the path for each image, and one of their corresponding labels
def get_image_paths_and_labels(image_names, root):
    image_paths = [] # need image paths to get the arrays from the saved jpgs, which we return as part of the getitem method
    labels = []
    for image_name in tqdm(image_names): # seems inefficient to do this loop every time. is there a way to avoid this???
            # idea: use pickle to store list of image_paths and corresponding labels
            current_path = root + 'control/' + image_name + '.jpg'
            current_label = 0
            if is_image_in_folder(image_name + '.jpg', root + '/tca') and is_image_in_folder(image_name + '.jpg', root + '/ntca'):
                continue # Hao/Charlie say I should ignore the (~20) examples that are both tca/ntca
            if is_image_in_folder(image_name + '.jpg', root + '/tca'):
                current_path = root + 'tca/' + image_name + '.jpg'
                current_label = 1
            if is_image_in_folder(image_name + '.jpg', root + '/ntca'):
                current_path = root + 'ntca/' + image_name + '.jpg'
                #current_label = 2
                current_label = 1 # switching to binary classification for now (either nothing, or tca/ntca)
            image_paths.append(current_path)
            labels.append(current_label)
    return image_paths, labels

class MultiClassDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root

        # idea: loop through every unique image (images can be in multiple folders).
        # if image in control, then label is 0. if image only in tca, then 1. if image in ntca or both tca/ntca, then 2.
        self.image_names = get_unique_image_names(root)
        with open(spec_data_path + "image_paths.pkl", "rb") as f:
            self.image_paths = pickle.load(f)
        with open(spec_data_path + "image_labels.pkl", "rb") as f:
            self.labels = pickle.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image_arr = np.array(image)

        # get sleep stage info to cut non-sleep at end of array (could make this its own function...)

        stage_path = '/data/netmit/wifall/ADetect/data/shhs2_new/stage/'
        stage_file = np.load(stage_path + self.image_names[index] + '.npz')
        stage_data = stage_file['data']
        if stage_file['fs'] == 1:
            stage_data = stage_data[::30]

        is_sleep = (stage_data > 0) & (stage_data <= 5)
        idx_is_sleep = np.where(is_sleep)[0]
        if len(idx_is_sleep) > 0:# and self.data != 'wsc':
            sleep_start, sleep_end = idx_is_sleep[0], idx_is_sleep[-1] + 1
        else:
            sleep_start, sleep_end = 0, len(stage_data)
        
        # cut non-sleep tail
        image_arr = image_arr[:, :sleep_end]
        assert not np.isnan(image_arr).any()

        # pad end with zeros so uniform (should i do this with transforms instead?) (nah)
        if image_arr.shape[-1] < 1536: # double checked that none greater than 1536
            image_arr = np.concatenate([image_arr, np.zeros((image_arr.shape[0], 1536 - image_arr.shape[-1]), dtype=np.float32)], 1)

        # convert image array and label to tensor
        image_tensor = torch.from_numpy(image_arr)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = image_tensor.repeat(3,1,1) # why 3,1,1?
        label_tensor = torch.tensor(self.labels[index])

        return image_tensor, label_tensor

# # have to do this every time an image is moved, added, or removed in the spectrogram folders. saves the hassle of
# # going through the directory every time the training loop is run
# image_names = get_unique_image_names(spec_data_path)
# image_paths, image_labels = get_image_paths_and_labels(image_names, spec_data_path)
# with open(spec_data_path + "image_paths.pkl", "wb") as f:
#     pickle.dump(image_paths, f)
# with open(spec_data_path + "image_labels.pkl", "wb") as f:
#     pickle.dump(image_labels, f)
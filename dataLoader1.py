import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, dataloader
import os
from BrEEG.task_spec import SpecMultitaperFolder, SpecFolder
from PIL import Image
from tqdm import tqdm

spec_data_path = '/data/scratch/scadavid/projects/data/eeg_mt_spec/'

# now, assuming images are all less than 1400 wide (which is true for what i've seen), want to add padding to the right side to make them all 1400
desired_size = (1400, 256)

def calculate_padding(image_size):
    width, height = image_size
    max_dim = max(width, height)
    pad_width = max_dim - width
    pad_height = max_dim - height
    return pad_width, pad_height

# Define the transform to add padding to the right side
padding = calculate_padding(desired_size)
transform = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.Pad((0,0,padding[0],0), fill=0, padding_mode='constant'),
])

#dataset = ImageFolder(spec_data_path, transform=transform)
# can't use ImageFolder since it assumes each image belongs to a single class

def get_unique_image_names(folder_path):
    unique_names = set()  # Set to store unique image names

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has an image extension
            if file.lower().endswith(('.jpg', '.jpeg')):
                # Get the image name w/out extension
                image_name = os.path.splitext(file)[0]
                # Add the image name to the set
                unique_names.add(image_name)

    return list(unique_names)

def is_image_in_folder(image_filename, folder_path):
    for root, dirs, files in os.walk(folder_path):
        if image_filename in files:
            return True  # Image file found in the folder
    return False  # Image file not found in the folder


class MultiLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root

        # idea: loop through every unique image (images can be in multiple folders).
        # if image in control, then label is [0, 0]. if image in tca, then flip first bit. if image in ntca, flip second bit.
        self.image_names = get_unique_image_names(root)
        self.image_paths = []
        self.labels = []
        for image_name in tqdm(self.image_names):
            current_path = self.root + '/control/' + image_name + '.jpg'
            current_label = [0, 0]
            if is_image_in_folder(image_name + '.jpg', self.root + '/tca'):
                current_path = self.root + '/tca/' + image_name + '.jpg'
                current_label[0] = 1
            if is_image_in_folder(image_name + '.jpg', self.root + '/ntca'):
                current_path = self.root + '/ntca/' + image_name + '.jpg'
                current_label[1] = 1
            self.image_paths.append(current_path)
            self.labels.append(current_label)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

dataset = MultiLabelDataset(root=spec_data_path)

batch_size = 16
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)






# STUFF FOR LATER:

# data_test = np.load(os.path.expanduser('~/mnt2/stage/shhs2-200077.npz'))
# # note: 
# print(list(data_test.keys()))
# print(data_test['fs'])
# print(data_test['data'].shape[0])#/data_test['fs'])

# # 337,800 data points for abdominal data during shh2 for patient 200077
# # 342,000 data points for abdominal data during shh2 for patient 203359 

# # idea: take abdominal tensor for patient X, sleep stage tensor for patient X, concatenate them. then create tensor, 
# # with each dimension being one of those concatenated tensors, each corresponding to a different patient.

# def load_into_tensor(folder_path):
#     data = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.npz'):
#             file_path = os.path.join(folder_path, filename)
#             loaded_data = np.load(file_path)
#             data.append(loaded_data['data'])
#     tensor_data = torch.stack(data)
#     return torch.DataFrame(tensor_data)
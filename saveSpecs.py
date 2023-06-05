import pandas as pd
import numpy as np
from BrEEG.task_spec import SpecMultitaperFolder, SpecFolder
from PIL import Image
from tqdm import tqdm
# for testing image quality:
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
#

# get label (i.e. medication data)

# note: starting w/ shhs2 for now

data_path = '/data/netmit/wifall/ADetect/data/'

shhs1path = data_path + 'shhs2/csv/shhs1-dataset-0.14.0.csv'
shhs2path = data_path + 'shhs2/csv/shhs2-dataset-0.14.0.csv'

shhs1_df = pd.read_csv(shhs1path, encoding='mac_roman')
shhs2_df = pd.read_csv(shhs2path, encoding='mac_roman')

#tca1_data = shhs1_df["TCA1"]
#ntca1_data = shhs1_df["NTCA1"]
tca2_data = shhs2_df["TCA2"]
ntca2_data = shhs2_df["NTCA2"]

# LOAD FEATURES

spec_mt_folder = SpecMultitaperFolder(data='shhs2')

spec_data_path = '/data/scratch/scadavid/projects/data/eeg_mt_spec/'

num_patients = len(tca2_data)

for patient_index in tqdm(range(0, num_patients)):
    uid = 'shhs2-' + str(shhs2_df["nsrrid"][patient_index])
    on_tca = shhs2_df["TCA2"][patient_index]
    on_ntca = shhs2_df["NTCA2"][patient_index]
    try:
        patient_eeg_spec_arr = spec_mt_folder.load(uid)
    except FileNotFoundError:
        continue # some patients decided not to have EEGs done
    patient_eeg_spec_arr = ((patient_eeg_spec_arr * 127.5) + 127.5).astype(np.uint8) # (-1,1) -> (0, 255), and convert to uint8
    patient_eeg_spec_img = Image.fromarray(patient_eeg_spec_arr)
    patient_eeg_spec_img.save(spec_data_path + uid + '.jpg')
    if(on_tca):
        patient_eeg_spec_img.save(spec_data_path + 'tca/' + uid + '.jpg')
    if(on_ntca):
        patient_eeg_spec_img.save(spec_data_path + 'ntca/' + uid + '.jpg')
    if((not on_tca) and (not on_ntca)):
        patient_eeg_spec_img.save(spec_data_path + 'control/' + uid + '.jpg')
    # need to do three separate saves since could end up in both tca and ntca folders





# STUFF FOR LATER:

#abdominal_data_path = os.path.expanduser('~/mnt2/abdominal')
# breathing_data = []

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
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
#from BrEEG.task_spec import SpecMultitaperFolder
from BrEEG.utils_eeg import eeg_spectrogram_multitaper

# get label (i.e. medication data)

# note: starting w/ shhs2 for now


shhs1path = '~/mnt2/csv/shhs1-dataset-0.14.0.csv'
shhs2path = '~/mnt2/csv/shhs2-dataset-0.14.0.csv'

shhs1_df = pd.read_csv(shhs1path, encoding='mac_roman')
shhs2_df = pd.read_csv(shhs2path, encoding='mac_roman')

#tca1_data = shhs1_df["TCA1"]
#ntca1_data = shhs1_df["NTCA1"]
tca2_data = shhs2_df["TCA2"]
ntca2_data = shhs2_df["NTCA2"]

# LOAD FEATURES

#spec_mt_folder = SpecMultitaperFolder(data='shhs2')

spec_data_path = '~/mnt/projects/data/eeg_mt_spec/'

num_patients = len(tca2_data)

# thing to check if images look good visualized w/ matplotlib. then that confirms the grainy looking ones are okay for the future model
for patient_index in tqdm(range(2, 3)):
    uid = 'shhs2-' + str(shhs2_df["nsrrid"][patient_index])
    on_tca = shhs2_df["TCA2"][patient_index]
    on_ntca = shhs2_df["NTCA2"][patient_index]
    # try:
    #     patient_eeg_spec_arr = spec_mt_folder.load(uid)
    # except FileNotFoundError:
    #     continue # some patients decided not to have EEGs done
    eeg_file = np.load(f'/Users/dina/mnt2/c4_m1/{uid}.npz')
    eeg = eeg_file['data']
    image = eeg_spectrogram_multitaper(eeg)
    image = np.clip(image, -10, 15)
    #image = (image + 10) / 25 * 2 - 1  # -10,15 -> -1,1
    image = image.astype(np.float32)
    #patient_eeg_spec_arr = ((patient_eeg_spec_arr * 127.5) + 127.5).astype(np.uint8) # (-1,1) -> (0, 255), and convert to uint8
    #patient_eeg_spec_img = Image.fromarray(patient_eeg_spec_arr)
    #tmp = patient_eeg_spec_arr
    #image = (image * 127.5) + 127.5
    tmp = image

    fig, a = plt.subplots()
    a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30, np.linspace(0, 32, tmp.shape[0]), tmp, vmin=-10, vmax=15, antialiased=True, shading='auto', cmap='jet')

    plt.imshow(tmp, cmap='hot', aspect='auto', origin='lower')
    a = plt.gca()
    a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30, np.linspace(0, 32, tmp.shape[0]), tmp, 
              vmin=-10, vmax=15, antialiased=True, shading='auto', cmap='jet')
    
    a.set_title(f'Multitaper EEG ' + str(uid))
    a.set_ylabel('Hz')
    eeg_y_max = 32 # should y max just be highest frequency? if so, then it's 32 Hz
    a.set_ylim([0, eeg_y_max])
    a.set_yticks(list(np.arange(0, eeg_y_max, 4)), list(np.arange(0, eeg_y_max, 4)))
    #a.set_xticks(np.arange(0, len(stage), 60), np.arange(0, len(stage), 60))
    a.grid()
    plt.show()
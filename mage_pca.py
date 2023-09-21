import os 
from sklearn.decomposition import PCA
import numpy as np 
from ipdb import set_trace as bp 
from tqdm import tqdm 

encoding_path = "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/shhs2_new/abdominal_c4_m1"
save_path = "/data/scratch-oc40/lth/mage-br-eeg-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug_ali_pca_32/"

all_shhs2_encodings = os.listdir(encoding_path)
all_outputs = []
for filename in tqdm(all_shhs2_encodings):
    filepath = os.path.join(encoding_path, filename)
    x = np.load(filepath)
    x = dict(x)

    feature = x['decoder_eeg_latent'].squeeze(0)
    if feature.shape[0] >= 150:
        feature = feature[:150, :]
    else:
        feature = np.concatenate((feature, np.zeros((150-feature.shape[0],feature.shape[-1]),dtype=np.float32)), axis=0)
    all_outputs.append(feature)

all_outputs_2 = np.concatenate(all_outputs, axis=0)

pca = PCA(n_components=32)

output = pca.fit(all_outputs_2)
for enc_path in [ "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/udall/abdominal_c4_m1"]:#["/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/shhs2_new/abdominal_c4_m1", "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/wsc_new/abdominal_c4_m1", "/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0/udall/abdominal_c4_m1"]:
    all_encodings = os.listdir(enc_path)
    study = "udall" if "udall" in enc_path else "shhs2_new" if "shhs2_new" in enc_path else "wsc_new" 
    if study != "udall":
        
        for filename in tqdm(all_encodings):
            filepath = os.path.join(enc_path, filename)
            x = np.load(filepath)
            x = dict(x)

            feature = x['decoder_eeg_latent'].squeeze(0)
            if feature.shape[0] >= 150:
                feature = feature[:150, :]
            else:
                feature = np.concatenate((feature, np.zeros((150-feature.shape[0],feature.shape[-1]),dtype=np.float32)), axis=0)
            new_feature = pca.transform(feature)
            output = {'decoder_eeg_latent':np.expand_dims(new_feature,0)}
            np.savez(os.path.join(save_path, study, filename),**output)
    else:
        for foldername in tqdm(all_encodings):
            if not os.path.exists(os.path.join(save_path, study, foldername)):
                os.mkdir(os.path.join(save_path, study, foldername))
            for filename in os.listdir(os.path.join(enc_path, foldername)):
                filepath = os.path.join(enc_path, foldername, filename)
                x = np.load(filepath)
                x = dict(x)

                feature = x['decoder_eeg_latent'].squeeze(0)
                if feature.shape[0] >= 150:
                    feature = feature[:150, :]
                else:
                    feature = np.concatenate((feature, np.zeros((150-feature.shape[0],feature.shape[-1]),dtype=np.float32)), axis=0)
                new_feature = pca.transform(feature)
                output = {'decoder_eeg_latent':np.expand_dims(new_feature,0)}
                np.savez(os.path.join(save_path, study, foldername, filename),**output)

print('hi')
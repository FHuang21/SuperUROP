import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score, AveragePrecision, AUROC
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, R2Score
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from tqdm import tqdm

import PIL
from pathlib import Path
from BrEEG.utils_eeg import eeg_spectrogram, post_process_eeg, eeg_spectrogram_multitaper

torch.backends.cudnn.benchmark = True

scratch_root = Path('/data/netmit/wifall')
scratch2_root = Path('/data/scratch-oc40/hehaodele')

# added by S
mnt2_root = Path('~/mnt2')

class NsrrInfo:
    """From Raw Dataset"""
    # df_mros1 = pd.read_csv(scratch_root / 'ADetect/data/mros2_new/csv/mros-visit1-dataset-0.3.0.csv')
    # df_mros2 = pd.read_csv(scratch_root / 'ADetect/data/mros2_new/csv/mros-visit2-dataset-0.3.0.csv')
    # df_shhs2 = pd.read_csv(scratch_root / 'ADetect/data/shhs2_new/csv/shhs2-dataset-0.14.0.csv',
    #                        encoding='cp1252', low_memory=False)
    # df_shhs1 = pd.read_csv(scratch_root / 'ADetect/data/shhs2_new/csv/shhs1-dataset-0.14.0.csv',
    #                        encoding='cp1252', low_memory=False)
    # df_wsc = pd.read_csv(scratch_root / 'ADetect/data/wsc_new/csv/wsc-dataset-0.5.0.csv',
    #                      encoding='cp1252', low_memory=False)

    try: # added by S
        df_shhs2_vardict = pd.read_csv(
            scratch_root / 'ADetect/data/shhs2_new/csv/shhs-data-dictionary-0.14.0-variables.csv',
            encoding='cp1252', low_memory=False)
        df_mayoad = pd.read_excel(scratch_root / 'ADetect/data/mayo_new/csv/AD-PD De-identified updated SLG 06-25-20.xlsx',
                                sheet_name='AD')

        """From Processed"""
        # df_mros1 = pd.read_csv(scratch_root / 'ADetect/data/csv/mros1-dataset-augmented.csv', low_memory=False)
        # df_mros2 = pd.read_csv(scratch_root / 'ADetect/data/csv/mros2-dataset-augmented.csv', low_memory=False)
        df_shhs1 = pd.read_csv(scratch_root / 'ADetect/data/csv/shhs1-dataset-augmented.csv', low_memory=False)
        df_shhs2 = pd.read_csv(scratch_root / 'ADetect/data/csv/shhs2-dataset-augmented.csv', low_memory=False)
        df_mayoad = pd.read_csv(scratch_root / 'ADetect/data/csv/mayoad-dataset-augmented.csv', low_memory=False)
        df_mgh = pd.read_csv(scratch_root / 'ADetect/data/csv/mgh-dataset-augmented.csv', low_memory=False)
        df_wsc = pd.read_csv(scratch_root / 'ADetect/data/csv/wsc-dataset-augmented.csv', low_memory=False)
    except FileNotFoundError:
        # df_shhs2_vardict = pd.read_csv(
        #     scratch_root / 'ADetect/data/shhs2_new/csv/shhs-data-dictionary-0.14.0-variables.csv',
        #     encoding='cp1252', low_memory=False)
        # df_mayoad = pd.read_excel(scratch_root / 'ADetect/data/mayo_new/csv/AD-PD De-identified updated SLG 06-25-20.xlsx',
        #                         sheet_name='AD')

        """From Processed"""
        # df_mros1 = pd.read_csv(scratch_root / 'ADetect/data/csv/mros1-dataset-augmented.csv', low_memory=False)
        # df_mros2 = pd.read_csv(scratch_root / 'ADetect/data/csv/mros2-dataset-augmented.csv', low_memory=False)
        

        df_shhs1 = pd.read_csv(mnt2_root / 'csv/shhs1-dataset-augmented.csv', low_memory=False)
        df_shhs2 = pd.read_csv(mnt2_root / 'csv/shhs2-dataset-augmented.csv', low_memory=False)
        df_mayoad = pd.read_csv(mnt2_root / 'csv/mayoad-dataset-augmented.csv', low_memory=False)
        df_mgh = pd.read_csv(mnt2_root / 'csv/mgh-dataset-augmented.csv', low_memory=False)
        df_wsc = pd.read_csv(mnt2_root / 'csv/wsc-dataset-augmented.csv', low_memory=False)

    """
    [MrOS] Alzheimers related variables
        m1alzh: Alzheimers disease medication use (0: No, 1: Yes)
        mhalzh: Has a doctor or other health care provider ever told you that you had dementia or Alzheimer's disease? (0: No, 1: Yes)
        mhalzht: Are you currently being treated for dementia or Alzheimer's disease by a doctor? (0: No, 1: Yes)

    [SHHS] Alzheimers related variables
        alzh2:Acetylcholine Esterase Inhibitors For Alzheimers (Sleep Heart Health Study Visit Two (SHHS2))
    """

    # DatasetNames = ['shhs1', 'shhs2', 'mros1', 'mros2', 'mayoad', 'mgh', 'wsc']
    DatasetNames = ['shhs1', 'shhs2', 'mayoad', 'mgh', 'wsc']

    def __init__(self, require_psg=True, min_len_psg=0):
        """For Raw dataset"""
        # self.df_shhs2 = self.df_shhs2.rename(columns=str.lower)  # make column names lowercase
        # self.df_shhs1 = self.df_shhs1.rename(columns=str.lower)  # make column names lowercase
        # self.df_shhs2 = self.df_shhs2.assign(uid=[*map(lambda x: f'shhs2-{x}', self.df_shhs2.nsrrid)])
        # self.df_shhs1 = self.df_shhs1.assign(uid=[*map(lambda x: f'shhs1-{x}', self.df_shhs1.nsrrid)])
        # self.df_mros1 = self.df_mros1.assign(uid=[*map(lambda x: f'mros-visit1-{x.lower()}', self.df_mros1.nsrrid)])
        # self.df_mros2 = self.df_mros2.assign(uid=[*map(lambda x: f'mros-visit2-{x.lower()}', self.df_mros2.nsrrid)])

        """Now We use processed datasets"""
        if require_psg:
            for name in self.DatasetNames:
                df = getattr(self, f'df_{name}')
                df = df[df.has_psg & (df.len_psg >= min_len_psg)]
                setattr(self, f'df_{name}', df)

    def debug_idx_shhs2(self):
        df = self.df_shhs2

        df_info = self.df_shhs2_vardict

        for x in sorted(df.columns):
            print(x)

        # folder = 'Medical History'
        folder = 'Medications'
        df_sub = df_info[df_info.folder == folder]
        for var, display in zip(df_sub.id, df_sub.display_name):
            if var in df.columns:
                print(var, display, sum(df[var] == 0), sum(df[var] == 1), df[var].corr(df['alzh2']))

    def debug_idx(self):
        df_mros1, df_mros2 = self.df_mros1, self.df_mros2
        idx_mros1_m1alzh_1 = df_mros1.m1alzh == 1
        idx_mros2_m1alzh_1 = df_mros2.m1alzh == 1
        idx_mros2_mhalzh_1 = (df_mros2.mhalzh == 1) | (df_mros2.mhalzh == '1')
        idx_mros2_mhalzht_1 = (df_mros2.mhalzht == 1) | (df_mros2.mhalzht == '1')
        idx_mros2_mhalzht_0 = (df_mros2.mhalzht == 0) | (df_mros2.mhalzht == '0')
        print(idx_mros1_m1alzh_1.sum())
        print(idx_mros2_m1alzh_1.sum())
        print(idx_mros2_mhalzh_1.sum(), (idx_mros2_mhalzh_1 & idx_mros2_m1alzh_1).sum())
        print(idx_mros2_mhalzht_1.sum(), (idx_mros2_mhalzh_1 & idx_mros2_mhalzht_1).sum(),
              (idx_mros2_m1alzh_1 & idx_mros2_mhalzht_1).sum())
        print(idx_mros2_mhalzht_0.sum(), (idx_mros2_mhalzh_1 & idx_mros2_mhalzht_0).sum(),
              (idx_mros2_m1alzh_1 & idx_mros2_mhalzht_0).sum())

    @property
    def df_shhs2nm_match(self):
        df = self.df_shhs2notad
        idx = df.idx_match_ad
        return df[idx]

    @property
    def df_shhs2nm_notmatch(self):
        df = self.df_shhs2notad
        idx = df.idx_match_ad
        return df[~idx]

    @property
    def df_shhs1nm_match(self):
        df = self.df_shhs1
        idx = df.idx_match_ad
        return df[idx]

    @property
    def df_shhs1nm_notmatch(self):
        df = self.df_shhs1
        idx = df.idx_match_ad
        return df[~idx]

    @property
    def df_mros1nm_match(self):
        df = self.df_mros1notad
        idx = df.idx_match_ad
        return df[idx]

    @property
    def df_mros1nm_notmatch(self):
        df = self.df_mros1notad
        idx = df.idx_match_ad
        return df[~idx]

    @property
    def df_mros2nm_match(self):
        df = self.df_mros2notad
        idx = df.idx_match_ad
        return df[idx]

    @property
    def df_mros2nm_notmatch(self):
        df = self.df_mros2notad
        idx = df.idx_match_ad
        return df[~idx]

    @property
    def idx_shhs2ad(self):
        return self.df_shhs2.alzh2 == 1

    @property
    def idx_mros2ad(self):
        df = self.df_mros2
        return (df.m1alzh == 1) | (df.mhalzh == 1) | (df.mhalzh == '1')

    @property
    def idx_mros1ad(self):
        return self.df_mros1.m1alzh == 1

    @property
    def df_mros1ad(self):
        df_mros1 = self.df_mros1
        idx = df_mros1.m1alzh == 1
        return df_mros1[idx]

    @property
    def df_mghad(self):
        df = self.df_mgh
        idx = df.dx_cci_dementia > 0
        return df[idx]

    @property
    def df_mghnotad(self):
        df = self.df_mgh
        idx = df.dx_cci_dementia > 0
        return df[~idx]

    @property
    def df_shhs2ad(self):
        df = self.df_shhs2
        idx = df.alzh2 == 1
        return df[idx]

    @property
    def df_shhs2notad(self):
        df = self.df_shhs2
        idx = df.alzh2 == 1
        return df[~idx]

    @property
    def df_shhs2notad_eeg(self):
        df = self.df_shhs2notad
        fn_list = np.loadtxt('./splits/shhs2.txt', dtype=str)
        idx = [f'shhs2-{nsrrid}' in fn_list for nsrrid in df.nsrrid]
        idx = np.array(idx)
        return df[idx]

    @property
    def df_shhs2health(self):
        """
        hi201a 110
        hi201b 177
        hi201c 62
        hi201d 371
        hi201e 256
        hi216 75
        htnderv_s2 2033
        a2a2 176
        a2ad2 36
        ace2 697
        aced2 45
        adpi2 56
        agdi2 7
        alpha2 178
        alphad2 0
        alzh2 33
        amlod2 210
        anar1a2 10
        anar1b2 19
        anar1c2 12
        anar32 33
        asa2 1613
        basq2 14
        benzod2 218
        beta2 743
        betad2 19
        bgnd2 159
        ccb2 519
        ccbir2 47
        ccbsr2 480
        ccbt2 0
        cox22 267
        dig2 148
        dihir2 0
        dihsr2 57
        diur2 804
        dltir2 22
        dltsr2 102
        edd2 34
        estrgn2 565
        fibr2 97
        h2b2 242
        hctz2 312
        hctzk2 183
        hprns2 0
        htnmed2 1678
        insuln2 59
        iprtr2 55
        istrd2 130
        kblkr2 2
        kcl2 241
        kspr2 91
        lipid2 1108
        loop2 255
        mlpd2 0
        niac2 51
        nifir2 9
        nifsr2 57
        nsaid2 660
        ntca2 325
        ntg2 107
        oaia2 39
        ohga2 313
        ostrd2 101
        otch2b2 25
        pdei2 65
        ppi2 406
        premar2 423
        prknsn2 33
        prob2 0
        progst2 226
        pvdl2 2
        slf12 8
        slf22 209
        sttn2 1003
        sympth2 189
        tca2 101
        thry2 459
        thzd2 83
        urcos2 4
        vaso2 235
        vasod2 2
        verir2 16
        versr2 55
        warf2 150
        wtls2 1
        xoi2 76
        """
        df = self.df_shhs2
        df_info = self.df_shhs2_vardict

        idx_has_disease = False
        for folder in ['Medical History', 'Medications']:
            df_sub = df_info[df_info.folder == folder]
            for var in df_sub.id:  # disease variable
                if var in df.columns:
                    idx_has_disease |= df[var] == 1
                    # print(var, sum(df[var] == 1))
        return df[~idx_has_disease]

    @property
    def df_shhs2notad_gt70(self):
        df = self.df_shhs2notad
        return df[df.age_s2 > 70]

    @property
    def df_shhs2health_gt70(self):
        df = self.df_shhs2health
        return df[df.age_s2 > 70]

    @property
    def df_mros2ad(self):
        df = self.df_mros2
        idx = (df.m1alzh == 1) | (df.mhalzh == 1) | (df.mhalzh == '1')
        return df[idx]

    @property
    def df_mros2notad(self):
        df = self.df_mros2
        return df[~self.mros_is_ad]

    @property
    def mros_is_ad(self):
        idx1 = self.df_mros1.m1alzh == 1
        df = self.df_mros2
        idx2 = (df.m1alzh == 1) | (df.mhalzh == 1) | (df.mhalzh == '1')
        return idx1 | idx2

    @property
    def df_mros1ad(self):
        df_mros1 = self.df_mros1
        idx = df_mros1.m1alzh == 1
        return df_mros1[idx]

    @property
    def df_mros1notad(self):
        df_mros1 = self.df_mros1
        return df_mros1[~self.mros_is_ad]

    @property
    def df_mros1health(self):
        """
        mhdiab 387
        mhhthy 40
        mhlthy 260
        mhosteo 212
        mhoa 701
        mhprost 871
        mhpark 36
        mhliver 60
        mhrenal 30
        mhcobpd 151
        mhbronc 146
        mhasthm 224
        mhaller 793
        mhglau 323
        mhmi 508
        mhangin 440
        mhchf 174
        mhstrk 111
        mhbp 1453 # hypertension
        """
        df_mros1 = self.df_mros1
        idx_health = True
        for key in df_mros1.columns:
            if key.startswith('mh') and (key + 't') in df_mros1.columns:
                idx_disease = df_mros1[key] == '1'
                # print(key, sum(idx_disease))
                idx_health &= df_mros1[key] == '0'
        return df_mros1[idx_health]


class SpecMageFolder:
    lth_root = Path('/data/scratch-oc40/lth/mage-br-eeg-inference')
    def __init__(self, data, model='20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.0'):
        self.data_root = self.lth_root / model / f'{data}_new'

    def load(self, uid):
        eeg_file = np.load(self.data_root / f'{uid}.npz')
        eeg = eeg_file['pred'] # 256, T
        image = np.clip(eeg, -10, 15)
        image = (image + 10) / 25 * 2 - 1  # -10,15 -> -1,1
        return image.astype(np.float32)


class SpecMultitaperFolder:
    def __init__(self, data):
        self.data = data

    def load(self, uid):
        #slightly modified cause i (Simon) want to sometimes run this locally w/ the mnts
        try:
            eeg_file = np.load(scratch_root / f'ADetect/data/{self.data}/c4_m1/{uid}.npz')
        except FileNotFoundError:
            eeg_file = np.load(f'~/mnt2/c4_m1/{uid}.npz')
        eeg = eeg_file['data']
        image = eeg_spectrogram_multitaper(eeg)
        image = np.clip(image, -10, 15)
        image = (image + 10) / 25 * 2 - 1  # -10,15 -> -1,1
        return image.astype(np.float32)


class SpecFlyFolder:  # generate spectrum on-the-fly
    def __init__(self, data, clip=(-3, 5)):
        self.data = data
        self.clip = clip

    def load(self, uid):
        eeg_file = np.load(scratch_root / f'ADetect/data/{self.data}/c4_m1/{uid}.npz')
        eeg = eeg_file['data']
        image = eeg_spectrogram(eeg, 64, spec_win_sec=30, spec_step_sec=30, npad=(32 / 30))
        image = image[::4, :]  # 256 x like 1024
        image = post_process_eeg(image)  # range [-3,5]
        image = np.clip(image, -3, 5)
        return image.astype(np.float32)


class SpecJpgFolder:
    """
    By default, Jpg is 1 / 30 Hz
    EEG clip: -3,5
    Br clip: -4,4
    """

    def __init__(self, data, spec='eeg_256_30_cuttail_jpg', clip=(-3, 5)):
        self.data = data
        self.spec = spec
        self.clip = clip

    def load(self, uid):
        spec_file = scratch_root / f'ADetect/data/{self.data}/spec/{self.spec}/{uid}.jpg'
        x = np.asarray(PIL.Image.open(spec_file))
        x = x.astype(np.float32) / 255 * (self.clip[1] - self.clip[0]) + self.clip[0]
        return x


class SpecRecJpgFolder:
    def __init__(self, data, spec='eeg_256_30_cuttail_jpg_vqgan_dec', clip=(-3, 5)):
        self.data = data
        self.spec = spec
        self.clip = clip

    def load(self, uid):
        eeg_spec_file = scratch_root / f'ADetect/data/{self.data}/spec/eeg_256_30_cuttail_jpg/{uid}.jpg'
        x_eeg = np.asarray(PIL.Image.open(eeg_spec_file))
        spec_file = scratch_root / f'ADetect/data/{self.data}/spec/{self.spec}/{uid}.jpg'
        x = np.asarray(PIL.Image.open(spec_file))
        x = x[:, :x_eeg.shape[-1]]  # x maybe padded, remove it
        x = x.astype(np.float32) / 255 * (self.clip[1] - self.clip[0]) + self.clip[0]
        return x


class SpecPredJpgFolder:
    def __init__(self, data, clip=(-3, 5)):
        self.data = data
        self.clip = clip

    def load(self, uid):
        eeg_spec_file = scratch_root / f'ADetect/data/{self.data}/spec/eeg_256_30_cuttail_jpg/{uid}.jpg'
        x_eeg = np.asarray(PIL.Image.open(eeg_spec_file))
        spec_file = Path(
            f'/data/scratch-oc40/kzha/mage-br-eeg/output_dir/20230420-mage-br-eeg-cond-rawbrps8x32-8192x32-recon-iter1-new-stage-0.1/eeg_pred/{self.data}/{uid}.jpg')
        x = np.asarray(PIL.Image.open(spec_file))
        x = x[:, :x_eeg.shape[-1]]  # x maybe padded, remove it
        x = x.astype(np.float32) / 255 * (self.clip[1] - self.clip[0]) + self.clip[0]
        return x


class SpecFolder:
    """
    By default, pix2pix model predict spec with 1 / 15 Hz
    """

    def __init__(self, data, spec, downsample_time=2, remove_tail=True):
        self.data = data
        self.spec = spec
        self.downsample_time = downsample_time
        self.remove_tail = remove_tail

    def load(self, uid):
        #spec_file = np.load(scratch_root / f'ADetect/data/{self.data}/spec/{self.spec}/{uid}.npz')
        #my add ins:
        spec_data_path = '/data/scratch/scadavid/projects/data/eeg_mt_spec/'
        spec_file = np.load(spec_data_path)
        #
        x = spec_file['signal']
        stage_file = np.load(scratch_root / f'ADetect/data/{self.data}/stage/{uid}.npz')
        stage = stage_file['data']
        if stage_file['fs'] == 1:
            stage = stage[::30]

        is_sleep = (stage > 0) & (stage <= 5)
        idx_is_sleep = np.where(is_sleep)[0]
        if len(idx_is_sleep) > 0 and self.data != 'wsc':
            sleep_start, sleep_end = idx_is_sleep[0], idx_is_sleep[-1] + 1
        else:
            sleep_start, sleep_end = 0, len(stage)

        # [cut signal after sleep] #
        x = x[:, :sleep_end * 2]
        assert not np.isnan(x).any()
        # nan_idx = np.isnan(x)
        # x[nan_idx] = 0

        if x.shape[-1] < 2048:
            x = np.concatenate([x, np.zeros((x.shape[0], 2048 - x.shape[-1]), dtype=np.float32)], 1)

        # [down-sample] #
        factor = self.downsample_time
        x = x[:, :x.shape[-1] // factor * factor].reshape(x.shape[0], -1, factor)
        # x = np.log(np.exp(x).mean(-1))
        x = x[:, :, 0]
        return x


class SpecDataset(Dataset):
    info = NsrrInfo(True, 2)  # have at least 2 hours PSG
    NumCv = 4
    AgeStat = (65, 10)
    AhiKeys = {
        'shhs2': 'ahi_a0h4',
        'shhs1': 'ahi_a0h4',
        'wsc': 'ahi',
    }

    def __init__(self, data, cv, phase, df=None, transform=None, args=None):
        super(SpecDataset, self).__init__()
        self.data = data
        self.phase = phase
        self.transform = transform
        self.args = args

        if df is None:
            df = getattr(self.info, f'df_{data}')

            idx_valid = np.zeros(len(df), dtype=bool)
            for i in range(len(df)):
                idx_valid[i] = (i % self.NumCv == cv)

            df_valid, df_train = df[idx_valid], df[~idx_valid]
            if phase == 'all':
                self.df = df
            elif phase == 'train':
                self.df = df_train
            else:
                self.df = df_valid
        else:
            self.df = df

        self.load_data()

    def load_data(self):
        eeg_spec_folder = SpecJpgFolder(data=self.data, spec='eeg_256_30_cuttail_jpg', clip=(-3, 5))
        # eeg_spec_folder = SpecFolder(data=self.data, spec='eeg_256_15', downsample_time=self.args.downsample_time)
        eeg_vqgan_spec_folder = SpecRecJpgFolder(data=self.data, clip=(-3, 5))
        eeg_pred_spec_folder = SpecPredJpgFolder(data=self.data, clip=(-3, 5))
        eeg_fly_spec_folder = SpecFlyFolder(data=self.data, clip=(-3, 5))
        eeg_multitaper_spec_folder = SpecMultitaperFolder(data=self.data)
        br_spec_folder = SpecFolder(data=self.data, spec='br_256_15', downsample_time=self.args.downsample_time)
        eeg_mage_spec_folder = SpecMageFolder(data=self.data, model=self.args.model_mage)

        input = self.args.input
        if input == 'eeg':
            spec_folders = [eeg_spec_folder]
        elif input == 'eeg-mage':
            spec_folders = [eeg_mage_spec_folder]
        elif input == 'eeg-vqgan':
            spec_folders = [eeg_vqgan_spec_folder]
        elif input == 'eeg-pred':
            spec_folders = [eeg_pred_spec_folder]
        elif input == 'eeg-fly':
            spec_folders = [eeg_fly_spec_folder]
        elif input == 'eeg-mt':
            spec_folders = [eeg_multitaper_spec_folder]
        elif input == 'br':
            spec_folders = [br_spec_folder]
        elif input == 'eeg-br':
            spec_folders = [eeg_spec_folder, br_spec_folder]
        else:
            assert False

        self.x_list = []
        for i in tqdm(range(len(self))):
            uid = self.df.uid.iloc[i]
            spec = [fd.load(uid) for fd in spec_folders]
            while len(spec) < 3:  # make 3 channels
                spec.append(np.zeros_like(spec[-1]))
            # # [check length]
            # spec_len = [x.shape[-1] for x in spec]
            # min_spec_len = np.min(spec_len)
            # spec = np.array([x[:, :min_spec_len] for x in spec])
            # print(spec.shape, spec.min(), spec.max())
            spec = np.array(spec)
            self.x_list.append(spec)

    def __len__(self):
        if self.args.debug:
            return min(len(self.df), 32)
        return len(self.df)

    def load_label(self, i):
        if self.args.task == 'age':
            y = self.df.age.iloc[i]
            y_mean, y_std = self.AgeStat
            y = (y - y_mean) / y_std
        elif self.args.task == 'ahi':
            k = self.AhiKeys[self.data]
            y = self.df[k].iloc[i]
            y = np.clip(np.log(y + 1e-10), -np.log(100), np.log(100)) / np.log(5)
            assert not np.isnan(y) and not np.isinf(y)
        else:
            y = self.df.label.iloc[i]
        return np.array([y], dtype=np.float32)

    def __getitem__(self, i):
        x = self.x_list[i]
        # print(x.shape)
        lenx = x.shape[2]
        if lenx < 1024:
            x = np.concatenate([x, np.zeros((3, 256, 1024 - lenx))], 2).astype(np.float32)
        # elif lenx > 1024:
        #     x = x[:, :, :1024]
        x = torch.from_numpy(x)
        x = self.transform(x)
        y = self.load_label(i)
        return x, y


class SpecStageDataset(SpecDataset):
    def __init__(self, data, cv, phase, df=None, transform=None, args=None):
        super().__init__(data, cv, phase, df, transform, args)

        if args.train_size and transform == 'train_transform':
            print(f'Reset Train Size from {len(self.df)} to {args.train_size}')
            self.df = self.df[:args.train_size]

    def load_label(self, i):
        uid = self.df.uid.iloc[i]
        stage_file = np.load(scratch_root / f'ADetect/data/{self.data}/stage/{uid}.npz')
        stage = stage_file['data']
        if stage_file['fs'] == 1:
            stage = stage[::30]

        assert self.args.downsample_time == 2
        stage_remap = np.array([0, 2, 2, 3, 3, 1, 0, 0, 0, 0, 0])
        y = stage_remap[stage.astype(int)]
        y_mask = (stage >= 0) & (stage <= 5)

        assert not np.isnan(y).any()
        # nan_idx = np.isnan(y)
        # y[nan_idx] = 0
        return y, y_mask

    def __getitem__(self, i):
        x = self.x_list[i]
        # x = torch.from_numpy(x)
        y, y_mask = self.load_label(i)
        x = x[0]  # assume there is only one spectrum
        lenx = x.shape[1]
        if len(y) >= lenx:
            y, y_mask = y[:lenx], y_mask[:lenx]
        else:
            y = np.concatenate([y, np.zeros(lenx - len(y))]).astype(int)
            y_mask = np.concatenate([y_mask, np.zeros(lenx - len(y_mask))]).astype(bool)

        # lenx might < 1024
        if lenx < 1024:
            x = np.concatenate([x, np.zeros((256, 1024 - lenx))], 1).astype(np.float32)
            y = np.concatenate([y, np.zeros(1024 - len(y))]).astype(int)
            y_mask = np.concatenate([y_mask, np.zeros(1024 - len(y_mask))]).astype(bool)
            lenx = 1024

        if self.transform == 'train_transform':
            i = np.random.randint(lenx - 1024 + 1)
            x, y, y_mask = x[:, i:i + 1024], y[i:i + 1024], y_mask[i:i + 1024]
        elif self.transform == 'valid_transform':
            x, y, y_mask = x[:, :1024], y[:1024], y_mask[:1024]
        else:
            assert False, f'{self.transform}'

        return x.astype(np.float32), y.astype(np.int64), y_mask.astype(np.int64)


class DeepClassifier(pl.LightningModule):
    ArchDict = {
        'res18': resnet18,
        'res50': resnet50,
    }

    def __init__(self, args):
        super().__init__()
        self.encoder = self.ArchDict[args.arch]()
        self.encoder.fc = nn.Identity()
        self.fc_var = nn.Linear(512, 1)
        self.args = args

        # setup metrics
        self.metrics = {
            'train': {'acc': Accuracy(task='binary'), 'auroc': AUROC(task='binary'),
                      'ap': AveragePrecision(task='binary'), 'f1': F1Score(task='binary')},
            'valid': {'acc': Accuracy(task='binary'), 'auroc': AUROC(task='binary'),
                      'ap': AveragePrecision(task='binary'), 'f1': F1Score(task='binary')},
        }
        for k, v in self.metrics.items():
            for k2, v2 in v.items():
                v2 = v2.cuda()
                v[k2] = v2
                setattr(self, f'metric_{k}_{k2}', v2)

    def forward(self, x):
        embedding = self.encoder(x)
        y_pred = self.fc_var(embedding)
        return y_pred, embedding

    def configure_optimizers(self):
        lr = self.args.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch, eta_min=lr * 1e-3)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.encoder(x)
        y_hat = self.fc_var(z)
        y_hat = torch.sigmoid(y_hat)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def training_step_end(self, outputs):
        self.log_metric(outputs, 'train')

    def validation_step_end(self, outputs):
        self.log_metric(outputs, 'valid')

    def log_metric(self, step_outputs, phase):
        y_true = step_outputs['y'][:, 0]
        y_pred = step_outputs['y_hat'][:, 0].detach()
        metrics = self.metrics[phase]
        for k, v in metrics.items():
            print(y_pred.shape)
            v(target=y_true.long(), preds=y_pred)
            self.log(f'{phase}_metric_{k}', v, on_step=False, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.encoder(x)
        y_hat = self.fc_var(z)
        y_hat = torch.sigmoid(y_hat)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        z = self.encoder(x)
        y_hat = self.fc_var(z)
        y_hat = torch.sigmoid(y_hat)
        return {'y': y, 'y_hat': y_hat}


class DeepRegressor(pl.LightningModule):
    ArchDict = {
        'res18': resnet18,
        'res50': resnet50,
    }

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = self.ArchDict[args.arch]()
        self.encoder.fc = nn.Identity()
        self.fc_age = nn.Linear(512, 1)

        self.metrics = {
            'train': {'mae': MeanAbsoluteError(), 'mse': MeanSquaredError(), 'r2': R2Score(), 'cor': PearsonCorrCoef()},
            'valid': {'mae': MeanAbsoluteError(), 'mse': MeanSquaredError(), 'r2': R2Score(), 'cor': PearsonCorrCoef()},
        }
        for k, v in self.metrics.items():
            for k2, v2 in v.items():
                v2 = v2.cuda()
                v[k2] = v2
                setattr(self, f'metric_{k}_{k2}', v2)

    def forward(self, x):
        embedding = self.encoder(x)
        y_pred = self.fc_age(embedding)
        return y_pred, embedding

    def configure_optimizers(self):
        lr = self.args.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch, eta_min=lr * 1e-3)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.encoder(x)
        y_hat = self.fc_age(z)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def training_step_end(self, outputs):
        self.log_metric(outputs, 'train')

    def validation_step_end(self, outputs):
        self.log_metric(outputs, 'valid')

    def log_metric(self, step_outputs, phase):
        y_true = step_outputs['y'].squeeze()
        y_pred = step_outputs['y_hat'].squeeze().detach()

        if self.args.task == 'age':
            y_true = y_true * 10
            y_pred = y_pred * 10
        elif self.args.task == 'ahi':
            y_true = torch.clamp_(torch.exp(y_true * np.log(5)), 0, 100)
            y_pred = torch.clamp_(torch.exp(y_pred * np.log(5)), 0, 100)
        else:
            assert False
        metrics = self.metrics[phase]
        for k, v in metrics.items():
            v(y_true, y_pred)
            self.log(f'{phase}_metric_{k}', v, on_step=False, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.encoder(x)
        y_hat = self.fc_age(z)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        z = self.encoder(x)
        y_hat = self.fc_age(z)
        # print(x.min(), x.max(), 'x')
        # print(y.min(), y.max(), 'y')
        # print(y_hat.min(), y_hat.max(), 'y_hat')
        return y_hat, y


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super(ResNet1D, self).__init__()
        self.inplanes = 256
        self.layer0 = self._make_layer(block, 256, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 256, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[3], stride=1)
        self.fc = nn.Conv1d(256, num_classes, kernel_size=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x


class DeepStagePredictor(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = ResNet1D(ResidualBlock1D, [2, 2, 2, 2])
        # print(self.model)
        self.args = args

        # setup metrics
        self.metrics = {
            'train': {'acc': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      'f1-macro': F1Score(task='multiclass', num_classes=4, average='macro', top_k=1),
                      'acc-0': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      'acc-1': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      'acc-2': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      'acc-3': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      },
            'valid': {'acc': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      'f1-macro': F1Score(task='multiclass', num_classes=4, average='macro', top_k=1),
                      'acc-0': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      'acc-1': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      'acc-2': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      'acc-3': Accuracy(task='multiclass', num_classes=4, top_k=1),
                      },
        }
        for k, v in self.metrics.items():
            for k2, v2 in v.items():
                v2 = v2.cuda()
                v[k2] = v2
                setattr(self, f'metric_{k}_{k2}', v2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        lr = self.args.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch, eta_min=lr * 1e-3)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y, _ = train_batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y, weight=torch.Tensor([1, 1, 1, self.args.weight_deep]).to(y.device))
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def training_step_end(self, outputs):
        self.log_metric(outputs, 'train')

    def validation_step_end(self, outputs):
        self.log_metric(outputs, 'valid')

    def log_metric(self, step_outputs, phase):
        print(f'Log Metric {phase}')
        y_true = step_outputs['y'].flatten().long()
        y_pred = torch.argmax(step_outputs['y_hat'].detach(), 1).flatten().long()

        metrics = self.metrics[phase]
        for k, v in metrics.items():
            if k in ['acc-0', 'acc-1', 'acc-2', 'acc-3']:
                idx = (y_true == int(k[-1]))
                v(target=y_true[idx], preds=y_pred[idx])
            else:
                v(target=y_true, preds=y_pred)
            self.log(f'{phase}_metric_{k}', v, on_step=False, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        x, y, _ = val_batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y, weight=torch.Tensor([1, 1, 1, 4]).to(y.device))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        y_hat = self.model(x)
        return {'y': y, 'y_hat': y_hat}


def set_args():
    import argparse

    parser = argparse.ArgumentParser(description='Parkinson')

    parser.add_argument('--arch', type=str, default='res18')
    parser.add_argument('--task', type=str, default='age')

    parser.add_argument('--input', type=str, default='eeg')
    parser.add_argument('--input_size', type=str, default=None, help='deprecated')
    parser.add_argument('--cv', type=int, default=0)

    parser.add_argument('--train', type=str, default='mros1,mros2')
    parser.add_argument('--valid', type=str, default='shhs2')
    parser.add_argument('--ratio', type=float, default=4)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--weight_deep', type=float, default=1.0)
    parser.add_argument('--train_size', type=int, default=None)

    parser.add_argument('--downsample_time', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model_mage', type=str,
                        default='20230507-mage-br-eeg-cond-rawbrps8x32-8192x32-ce-iter1-alldata-neweeg/iter1-temp0.0-minmr0.5')

    args = parser.parse_args()

    return args


def set_transform(args):
    crop_size = (256, 2048 // args.downsample_time)
    valid_crop_op = transforms.CenterCrop(crop_size)
    train_crop_op = transforms.RandomCrop(crop_size)
    train_transform = transforms.Compose([
        train_crop_op,
    ])
    valid_transform = transforms.Compose([
        valid_crop_op,
    ])
    return train_transform, valid_transform


def get_df(data, args):
    info = SpecDataset.info

    df = getattr(info, f'df_{data}')

    if args.task == 'ad':
        if data == 'shhs2':
            is_pos = df.alzh2 == 1
        elif data == 'mros2':
            is_pos = (df.m1alzh == 1) | (df.mhalzh == 1) | (df.mhalzh == '1')
        elif data == 'mros1':
            is_pos = df.m1alzh == 1
        else:
            assert False

    elif args.task == 'pd':

        if data == 'shhs2':
            is_pos = df['prknsn2'] == 1
        elif data in ['mros1', 'mros2']:
            is_pos = (df['mhpark'] == '1') | (df['mhparkt'] == '1')
        else:
            assert False
    else:
        assert False

    df_pos = df[is_pos]
    df_neg = df[~is_pos]

    num_pos = len(df_pos)
    num_neg = int(num_pos * args.ratio)
    df_neg = df_neg[:num_neg]
    print(f'#Pos {len(df_pos)} #Neg {len(df_neg)}')

    df_pos = df_pos.assign(label=1)
    df_neg = df_neg.assign(label=0)

    df = pd.concat([df_pos, df_neg], ignore_index=True)
    return df


if __name__ == '__main__':
    args = set_args()
    args.name = f'{args.task}-{args.input}_Bs-{args.bs}_Lr-{args.lr}_Epoch-{args.epoch}_DT-{args.downsample_time}'

    if args.task in ['age', 'ahi']:
        args.name += f'_Trn-{args.train}_Cv-{args.cv}'
    else:
        args.name += f'_R-{args.ratio}_Trn-{args.train}_Val-{args.valid}'

    if args.task == 'stage':
        args.name += f'_WeightDeep-{args.weight_deep}'
        if args.train_size:
            args.name += f'_Trn-sz-{args.train_size}'

    if args.exp is not None:
        args.name = f'{args.exp}/{args.name}'

    if args.input == 'eeg-mage':
        args.name = f'{args.name}/{args.model_mage}'

    print(args.name)

    train_transform, valid_transform = set_transform(args)

    if args.task in ['age', 'ahi']:
        dataset_list = [args.train]
        train_dataset = torch.utils.data.ConcatDataset([
            SpecDataset(k, 0, 'train', None, train_transform, args)
            for k in dataset_list
        ])
        valid_dataset = torch.utils.data.ConcatDataset([
            SpecDataset(k, 0, 'valid', None, valid_transform, args)
            for k in dataset_list
        ])
        train_loader = DataLoader(train_dataset, batch_size=args.bs, num_workers=12, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=args.bs, num_workers=12)

        model = DeepRegressor(args)

    elif args.task == 'stage':
        dataset_list = args.train.split(',')

        train_dataset = torch.utils.data.ConcatDataset([
            SpecStageDataset(k, 0, 'all', None, 'train_transform', args) for k in dataset_list
        ])
        dataset_list = args.valid.split(',')
        valid_dataset = torch.utils.data.ConcatDataset([
            SpecStageDataset(k, 0, 'all', None, 'valid_transform', args) for k in dataset_list
        ])
        train_loader = DataLoader(train_dataset, batch_size=args.bs, num_workers=12, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=args.bs, num_workers=12)
        model = DeepStagePredictor(args)

    else:
        dataset_list = args.train.split(',')
        train_dataset = torch.utils.data.ConcatDataset([
            SpecDataset(k, 0, 'all', get_df(k, args), train_transform, args)
            for k in dataset_list
        ])

        dataset_list = args.valid.split(',')
        valid_dataset = torch.utils.data.ConcatDataset([
            SpecDataset(k, 0, 'all', get_df(k, args), valid_transform, args)
            for k in dataset_list
        ])
        train_loader = DataLoader(train_dataset, batch_size=args.bs, num_workers=6, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=args.bs, num_workers=6)

        model = DeepClassifier(args)

    # training
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.epoch,
                         default_root_dir=str(scratch2_root / 'RespLearn' / args.name))
    trainer.fit(model, train_loader, val_loader)

from utils_spec import compute_multitaper_spec, load_file
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
#import streamlit as st
from ipdb import set_trace as bp


class AbsPlotter:
    def __init__(self):
        self.ax = None

    def get_spec(self):  # dumb function for some plotter
        return self


class StagePlotter(AbsPlotter):
    def __init__(self):
        super().__init__()
        self.stage = None

    @property
    def main_file(self):
        return self.stage

    def load(self, dataset, uid):
        file = load_file(dataset, 'stage', uid)

        if file is None:
            self.stage = None
            return self

        stage, fs = file['data'], file['fs']
        if fs == 1:
            stage = stage[::30]

        if dataset == 'udall':  # [4 class] -> [raw]
            stage_inverse_remap = np.array([0, 5, 2, 3])
            stage = stage_inverse_remap[stage.astype(int)].astype(float)

        self.stage = stage
        return self

    def plot(self, a, use_five_stage=False):
        stage = self.stage.astype(int)
        idx_invalid = (stage < 0) | (stage > 5)
        if not use_five_stage:
            stage_remap = np.array([0, 2, 2, 3, 3, 1] + [np.nan] * 10)
        else:
            stage_remap = np.array([0, 2, 3, 4, 4, 1] + [np.nan] * 10)
        stage = stage_remap[stage].astype(float)
        stage[idx_invalid] = np.nan

        a.plot(np.arange(len(stage)), stage)
        a.set_title('stage')
        if not use_five_stage:
            a.set_yticks([0, 1, 2, 3], ['A', 'R', 'L', 'D'])
        else:
            a.set_yticks([0, 1, 2, 3, 4], ['A', 'R', 'N1', 'N2', 'N3'])
        a.set_xticks(np.arange(0, len(stage), 60), np.arange(0, len(stage), 60) // 2)
        a.grid()

        self.ax = a
        return self


class BreathPlotter(AbsPlotter):
    def __init__(self, chn):
        super().__init__()
        self.chn = chn
        self.br_file = None
        self.br_spec = None
        self.br_rate = None

    @property
    def main_file(self):
        return self.br_file

    def load(self, dataset, uid):
        self.br_file = load_file(dataset, self.chn, uid)
        return self

    def get_spec(self):
        from utils_spec import spectrogram
        from scipy.signal import savgol_filter

        if self.br_file is None:
            return self

        br_file = self.br_file
        br = spectrogram(br_file['data'], fs=br_file['fs'],
                         spec_win_sec=30, cutoff_bpm=35, col_norm=True)
        self.br_spec = br

        br_rate = np.argmax(br, axis=0) / 2
        smooth_window, smooth_order = 25, 2
        smoothed_rate = savgol_filter(br_rate, smooth_window, smooth_order)
        self.br_rate = smoothed_rate

        return self

    def plot(self, a, br_y_min=8, br_y_max=30):
        br, rate = self.br_spec, self.br_rate
        a.plot(np.arange(br.shape[1]) / 6, rate, 'k-')
        a.pcolormesh(np.arange(br.shape[1]) / 6, np.arange(br.shape[0]) / 2, br,
                     antialiased=True,
                     shading='auto',
                     cmap='jet')
        a.set_title(f'Breathing ({self.chn})')
        a.set_ylabel('BPM')
        a.set_ylim([br_y_min, br_y_max])
        a.set_yticks(list(np.arange(br_y_min, br_y_max, 3)),
                     list(np.arange(br_y_min, br_y_max, 3)))
        a.grid()
        self.ax = a
        return self


class EEGPlotter(AbsPlotter):
    def __init__(self, chn):
        super().__init__()
        self.chn = chn
        self.file = None
        self.eeg_spec = None

    @property
    def main_file(self):
        return self.file

    def load(self, dataset, uid):
        self.file = load_file(dataset, self.chn, uid)
        return self

    def get_spec(self):
        if self.file is not None:
            self.eeg_spec = compute_multitaper_spec(self.file)
        else:
            self.eeg_spec = None
        return self

    def plot(self, a, eeg_y_max=20):
        assert self.eeg_spec is not None
        tmp = self.eeg_spec
        a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30,
                     np.linspace(0, 32, tmp.shape[0]),
                     tmp,
                     vmin=-10, vmax=15,
                     antialiased=True,
                     shading='auto',
                     cmap='jet')
        a.set_title(f'Multitaper EEG ({self.chn})')
        a.set_ylabel('Hz')
        a.set_ylim([0, eeg_y_max])
        a.set_yticks(list(np.arange(0, eeg_y_max, 4)), list(np.arange(0, eeg_y_max, 4)))
        a.grid()
        self.ax = a
        return self


class MagePlotter(AbsPlotter):
    lth_root = Path('/data/scratch-oc40/lth/mage-br-eeg-inference')

    def __init__(self, model_name, chn, short_name):
        super().__init__()
        self.model_name = model_name
        self.short_name = short_name
        self.chn = chn
        self.eeg_spec = None
        self.file = None

    @property
    def main_file(self):
        return self.eeg_spec

    def load(self, dataset, uid):
        file = self.lth_root / self.model_name / f'{dataset}_new' / self.chn / f'{uid}.npz'
        if not file.is_file():
            if self.chn == 'c4_m1':
                file = self.lth_root / self.model_name / f'{dataset}_new' / f'{uid}.npz'
                # st.write(str(file))
                if not file.is_file():
                    self.eeg_spec = None
                    return self
            else:
                self.eeg_spec = None
                return self
        self.eeg_spec = np.load(file)['pred']
        return self

    def plot(self, a, eeg_y_max=20):
        # print(self.short_name, self.eeg_spec.shape)
        tmp = self.eeg_spec
        a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30,
                     np.linspace(0, 32, tmp.shape[0]),
                     tmp,
                     vmin=-10, vmax=15,
                     antialiased=True,
                     shading='auto',
                     cmap='jet')
        a.set_title(f'{self.short_name} EEG ({self.chn})')
        a.set_ylabel('Hz')
        a.set_ylim([0, eeg_y_max])
        a.set_yticks(list(np.arange(0, eeg_y_max, 4)), list(np.arange(0, eeg_y_max, 4)))
        a.grid()
        self.ax = a
        return self


class MageProbPlotter(AbsPlotter):
    lth_root = Path('/data/scratch-oc40/lth/mage-br-eeg-inference')

    def __init__(self, model_name, chn, short_name, token_size):
        super().__init__()
        self.model_name = model_name
        self.short_name = short_name
        self.token_size = token_size
        self.chn = chn
        self.eeg_spec = None
        self.file = None

    @property
    def main_file(self):
        return self.eeg_spec

    def load(self, dataset, uid):
        file = self.lth_root / self.model_name / f'{dataset}_new' / self.chn / f'{uid}.npz'
        if not file.is_file():
            self.eeg_spec = None
            return self
        tmp = np.load(file)['selected_probs']
        tmp = np.repeat(tmp, self.token_size[0], axis=0)
        tmp = np.repeat(tmp, self.token_size[1], axis=1)
        self.eeg_spec = tmp
        return self

    def plot(self, a, eeg_y_max=20):
        # print(self.short_name, self.eeg_spec.shape)
        tmp = self.eeg_spec
        tmp = - np.log(tmp)
        print(tmp.min(), tmp.max(), np.median(tmp))
        a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30,
                     np.linspace(0, 32, tmp.shape[0]),
                     tmp,
                     vmin=1.5, vmax=6,
                     antialiased=True,
                     shading='auto',
                     cmap='RdYlBu')
        a.set_title(f'{self.short_name} EEG ({self.chn})')
        a.set_ylabel('Hz')
        a.set_ylim([0, eeg_y_max])
        a.set_yticks(list(np.arange(0, eeg_y_max, 4)), list(np.arange(0, eeg_y_max, 4)))
        a.grid()
        self.ax = a
        return self

class MageAttentionPlotter(AbsPlotter): # FIXME ::::
    #lth_root = Path('/data/scratch-oc40/lth/mage-br-eeg-inference')
    my_root = Path('/data/netmit/wifall/ADetect/mage-inference/20230626-mage-br-eeg-cond-8192x32-ce-iter1-alldata-eegps256x8-br1d-1layerbbenc-maskpad-thoraxaug/iter1-temp0.0-mr1.0')

    def __init__(self, chn, short_name, token_size):
        super().__init__()
        #self.model_name = model_name
        self.short_name = short_name
        self.token_size = token_size
        self.chn = chn
        self.eeg_spec = None
        self.file = None

    @property
    def main_file(self):
        return self.eeg_spec

    def load(self, dataset, uid):
        #file = self.lth_root / self.model_name / f'{dataset}_new' / self.chn / f'{uid}.npz'
        file = self.my_root / f'{dataset}_new' / self.chn /  f'{uid}'
        if not file.is_file():
            self.eeg_spec = None
            return self
        #bp()
        tmp = np.load(file)['decoder_eeg_latent']
        tmp = np.repeat(tmp, self.token_size[0], axis=0)
        tmp = np.repeat(tmp, self.token_size[1], axis=1)
        self.eeg_spec = tmp
        return self

    def plot(self, a, eeg_y_max=20):
        # print(self.short_name, self.eeg_spec.shape)
        tmp = self.eeg_spec
        bp()
        #tmp = - np.log(tmp)
        print(tmp.min(), tmp.max(), np.median(tmp))
        a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30,
                     np.linspace(0, 32, tmp.shape[0]),
                     tmp,
                     vmin=1.5, vmax=6,
                     antialiased=True,
                     shading='auto',
                     cmap='RdYlBu')
        a.set_title(f'{self.short_name} EEG ({self.chn})')
        a.set_ylabel('Hz')
        a.set_ylim([0, eeg_y_max])
        a.set_yticks(list(np.arange(0, eeg_y_max, 4)), list(np.arange(0, eeg_y_max, 4)))
        a.grid()
        self.ax = a
        return self

def rescale_pred(eeg_spec, eeg_pred):
    if eeg_pred is None:
        return None
    min_len = min(eeg_pred.shape[1], eeg_spec.shape[1])
    eeg_pred_norm_by_col = ((eeg_pred - eeg_pred[:, :min_len].mean(0, keepdims=True)) / eeg_pred[:, :min_len].std(0,
                                                                                                                  keepdims=True)) * eeg_spec[
                                                                                                                                    :,
                                                                                                                                    :min_len].std(
        0, keepdims=True) + eeg_spec[:, :min_len].mean(0, keepdims=True)
    return eeg_pred_norm_by_col


class MageRescalePlotter(AbsPlotter):
    lth_root = Path('/data/scratch-oc40/lth/mage-br-eeg-inference')

    def __init__(self, model_name, chn, short_name):
        super().__init__()
        self.model_name = model_name
        self.short_name = short_name
        self.chn = chn
        self.eeg_spec = None
        self.file = None

    @property
    def main_file(self):
        return self.eeg_spec

    def load(self, dataset, uid):
        self.eeg_spec = None

        file = self.lth_root / self.model_name / f'{dataset}_new' / self.chn / f'{uid}.npz'
        if not file.is_file():
            return self
        self.eeg_spec = np.load(file)['pred']

        eeg_file = load_file(dataset, self.chn, uid)
        if eeg_file is None:
            return self
        eeg_gt = compute_multitaper_spec(eeg_file)
        self.eeg_spec = rescale_pred(eeg_gt, self.eeg_spec)

        return self

    def plot(self, a, eeg_y_max=20):
        # print(self.short_name, self.eeg_spec.shape)
        tmp = self.eeg_spec
        a.pcolormesh(np.arange(tmp.shape[1]) / 30 * 30,
                     np.linspace(0, 32, tmp.shape[0]),
                     tmp,
                     vmin=-10, vmax=15,
                     antialiased=True,
                     shading='auto',
                     cmap='jet')
        a.set_title(f'{self.short_name} EEG Rescaled ({self.chn})')
        a.set_ylabel('Hz')
        a.set_ylim([0, eeg_y_max])
        a.set_yticks(list(np.arange(0, eeg_y_max, 4)), list(np.arange(0, eeg_y_max, 4)))
        a.grid()
        self.ax = a
        return self


class Plotter:
    def __init__(self):
        self.plotters = []
        self.fig = None
        self.ax = None

    def add_plotter(self, plotter):
        if isinstance(plotter, list):
            self.plotters.extend(plotter)
        else:
            self.plotters.append(plotter)
        return self

    def load(self, dataset, uid):
        for plotter in self.plotters:
            plotter.load(dataset, uid).get_spec()
        return self

    def plot(self,
             eeg_y_max=32,
             br_y_min=8,
             br_y_max=30,
             use_five_stage=False):

        valid_plotters = []
        for p in self.plotters:
            if p.main_file is not None:
                valid_plotters.append(p)
        nrows = len(valid_plotters)
        # print(nrows, valid_plotters)
        fig, ax = plt.subplots(nrows, 1, sharex='all', figsize=(12, nrows * 2))
        if nrows == 1:
            ax = [ax]
        for a, plotter in zip(ax, valid_plotters):
            if isinstance(plotter, BreathPlotter):
                plotter.plot(a, br_y_min, br_y_max)
            elif isinstance(plotter, EEGPlotter) or isinstance(plotter, MagePlotter) \
                or isinstance(plotter, MageRescalePlotter) or isinstance(plotter, MageProbPlotter):
                plotter.plot(a, eeg_y_max)
            elif isinstance(plotter, StagePlotter):
                plotter.plot(a, use_five_stage)
            else:
                plotter.plot(a)

        fig.tight_layout()
        self.fig = fig
        self.ax = ax
        return self

if __name__ == '__main__':
    fig, ax = plt.subplots()
    mage_att_plotter = MageAttentionPlotter(chn='abdominal_c4_m1', short_name='short_name', token_size=(256,8))
    uid = 'wsc-visit1-58328-nsrr.npz'
    mage_att_plotter.load('wsc', uid)
    mage_att_plotter.plot(ax)

    plt.savefig('/data/scratch/scadavid/projects/data/figures/test_mage_plot.pdf')

    bp()
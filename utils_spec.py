from matplotlib.ticker import Locator
from matplotlib import transforms as mtransforms

import numpy as np
import matplotlib.pyplot as plt

StandardFps = 5

# Range of valid Breathing per minute.
# Signals with too slow or too fast BPS will be ignored
BPM_RANGE = np.array([8., 30.])


def label_to_interval(label: np.array, val=0):
    # finds intervals where entries in label array are equal to val

    # e.g., suppose label = np.array([0, 0, 0, 1, 2, 0, 0, 2, 0])
    # and val = 0

    # results should be [[0 3], [5 8], [8 9]]

    # make a copy of the array (label == val), which is boolean
    # cast to type int (True == 1, False == 0)
    # e.g., [1, 1, 1, 0, 0, 1, 1, 0, 1]

    hit = (label == val).astype(int)

    # add a zero to each end of the array
    # e.g., [0., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0.]

    a = np.concatenate([np.zeros((1,)), hit.flatten(), np.zeros((1,))], axis=0)

    # compute successive differences (note length decreases by 1)
    # e.g., [1., 0., 0., -1., 0., 1., 0., -1., 1., -1.]

    a = np.diff(a, axis=0)

    # find indices where a 1 appears (start of an interval)
    # where can be used to values where condition
    # evalutes to either true or false.  We only want true,
    # so pick element [0] of the result

    left = np.where(a == 1)[0]

    # find indices where a -1 appears (end of an interval)

    right = np.where(a == -1)[0]

    # create [start of interval, end of interval] pairs,
    # converting from floating point to int
    # zip function returns an iterator
    # *zip lists all instances

    return np.array([*zip(left, right)], dtype=np.int32)


def butter_highpass(fs, order=5):
    from scipy.signal import butter
    target_cutoff = 5 / 60  # 5 bpm
    nyq = 0.5 * fs
    normal_cutoff = target_cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)


DefaultHighpassParameters = butter_highpass(StandardFps)


def breathing_highpass(signal, fs=StandardFps, axis=-1):
    from scipy.signal import filtfilt
    if fs == StandardFps:
        params = DefaultHighpassParameters
    else:
        params = butter_highpass(fs)
    y = filtfilt(params[0], params[1], signal, axis=axis)
    return y


def butter_lowpass(fs, order=30):
    from scipy.signal import butter
    target_cutoff = 30 / 60  # 30 bpm
    nyq = 0.5 * fs
    normal_cutoff = target_cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)


def breathing_lowpass(signal, fs, axis=-1):
    from scipy.signal import filtfilt
    params = butter_lowpass(fs)
    y = filtfilt(params[0], params[1], signal, axis=axis)
    return y


def highpass_parameter(freq, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = freq / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)


def eeg_highpass(signal, fs, cutoff, axis=-1):
    from scipy.signal import filtfilt
    params = highpass_parameter(cutoff, fs)
    y = filtfilt(params[0], params[1], signal, axis=axis)
    return y


def bandpass(signal, fs, low, high, axis=-1):
    from scipy.signal import sosfiltfilt
    from scipy.signal import butter
    nyq = 0.5 * fs
    params = butter(5, [low / nyq, high / nyq], btype='bandpass', output='sos', analog=False)
    y = sosfiltfilt(params, signal, axis=axis)
    return y


def signal_std(signal):
    if len(signal) < 10:
        return 1
    else:
        cut = int(len(signal) * 0.1)
        std = np.std(np.sort(signal)[cut:-cut])
    std = 1 if std == 0 else std
    return std


def signal_normalize(signal, axis=-1):
    # signal -= np.mean(signal)
    # return signal / signal_std(signal)
    signal -= np.mean(signal, axis=axis, keepdims=True)
    signal = signal / (np.std(signal, axis=axis, keepdims=True) + 1e-5)
    return signal


def signal_crop_motion(signal, window=10, fs=StandardFps, threshold=5, show=False):
    from scipy.ndimage.filters import minimum_filter1d
    signal_norm = signal_normalize(signal)
    threshold = max(np.max(np.abs(signal_norm)) * 0.5, threshold)
    normal_part = np.abs(signal_norm) < threshold
    normal_part = minimum_filter1d(normal_part, int(window * fs))
    indices = np.where(normal_part == 1)[0]
    signal_crop = signal_norm[indices]
    if show:
        fig, ax = plt.subplots(5, 1, sharex='all')
        ax[0].plot(signal)
        ax[1].plot(signal_norm)
        ax[2].plot(normal_part)
        ax[3].plot(indices, signal_crop)
        ax[4].plot(signal_crop)
        plt.show()
    return signal_crop, indices


def signal_crop(signal, clip_limit=5):
    signal = np.clip(signal, -clip_limit, clip_limit)
    return signal


def get_breathing_rate_raw(signal, fs=StandardFps, n_pad=9, spec_win_sec=60):
    from scipy.signal import spectrogram
    from scipy.ndimage.filters import gaussian_filter1d
    spec_win = spec_win_sec * fs
    spec_step = 5 * fs

    signal, _ = signal_crop_motion(signal, window=10)

    length = len(signal)
    duration = length / fs
    if duration <= spec_win_sec:
        signal_pad = np.pad(signal, [[0, spec_win * n_pad - length]], 'constant', constant_values=0)
        spec = np.repeat(np.abs(np.fft.fft(signal_pad)[:, np.newaxis]), 2, axis=1)
    else:
        _, _, spec = spectrogram(
            signal, window='hamming', nperseg=spec_win, noverlap=(spec_win - spec_step),
            nfft=n_pad * spec_win, mode='magnitude'
        )
    spec = spec[:n_pad * spec_win_sec]
    spec_gaussian = gaussian_filter1d(spec, 10, axis=0)

    ratio = int(n_pad * spec_win_sec / 60)
    signal = np.argmax(spec_gaussian[8 * ratio: 30 * ratio], axis=0) / ratio + 8

    return spec_gaussian, signal


def get_breathing_rate(signal, fs=StandardFps, n_pad=9, show=False, spec_win_sec=60):
    spec_gaussian, breathing_rate = get_breathing_rate_raw(signal, fs, n_pad, spec_win_sec)

    if show:
        fig, ax = plt.subplots(4, 1, sharex='all')
        ax[0].plot(np.linspace(0, len(breathing_rate), len(signal)), signal)
        spec_col = spec / np.max(spec, axis=0, keepdims=True)
        ax[1].imshow(spec_col, cmap='jet', aspect='auto', extent=[0, len(breathing_rate), 60, 0])
        ax[1].invert_yaxis()
        spec_gaussian_col = spec_gaussian / np.max(spec_gaussian, axis=0, keepdims=True)
        ax[2].imshow(spec_gaussian_col, cmap='jet', aspect='auto', extent=[0, len(breathing_rate), 60, 0])
        ax[2].invert_yaxis()
        ax[3].plot(breathing_rate)
        plt.show()

    return breathing_rate.mean(), breathing_rate.std()


def signal_snr(signal, fs=StandardFps, with_std=True, crop_motion=True, show=False):
    from scipy.signal import spectrogram

    if crop_motion:
        signal, _ = signal_crop_motion(signal)
    if len(signal) < 30 * fs:
        return 0
    _, _, spec = spectrogram(signal, fs, ('tukey', .25), 15 * fs, 10 * fs, 30 * fs, 'constant', True, mode='magnitude')
    energy_sum = np.sum(spec, axis=0)
    peak_index = np.argmax(spec, axis=0)
    peak_index[np.where(peak_index * 2 < BPM_RANGE[0])] = 0
    peak_index[np.where(peak_index * 2 > BPM_RANGE[1])] = 0
    col_score = np.zeros(spec.shape[1])
    for i in range(spec.shape[1]):
        col_score[i] = np.sum(spec[peak_index[i] - 1:peak_index[i] + 2, i])
        col_score[i] += np.sum(spec[peak_index[i] * 2 - 1:peak_index[i] * 2 + 2, i])
    col_score /= energy_sum + 1e-5
    snr = np.median(col_score)
    if with_std:
        snr = snr - 0.01 * np.std(peak_index)

    if show:
        fig, ax = plt.subplots(4, 1, sharex='all')
        ax[0].plot(np.linspace(0, len(col_score), len(signal)), signal)
        spec_col = spec / np.max(spec, axis=0, keepdims=True)
        ax[1].imshow(spec_col, cmap='jet', aspect='auto', extent=[0, len(col_score), 150, 0])
        ax[1].invert_yaxis()
        ax[2].plot(col_score)
        ax[3].plot(peak_index)
        plt.suptitle(f'{np.median(col_score)}, {np.std(peak_index)}')
        plt.show()
    return snr


def is_breathing(signal, snr_threshold=0.3, fs=StandardFps, show=False):
    breathing_rate_std_limit = 4
    br_avg, br_std = get_breathing_rate(signal, fs)
    snr = signal_snr(signal, fs=fs)
    if show:
        print(f"{br_avg:.1f}, {br_std:.1f}, {snr:.3f}")
    if snr > snr_threshold and BPM_RANGE[0] <= br_avg < BPM_RANGE[1]:
        if snr > 0.4:
            return True
        elif br_std < breathing_rate_std_limit:
            return True
        else:
            return False
    else:
        return False


def detect_motion_iterative(signal, fs=StandardFps, level=3):
    signal = signal.copy()
    motion = np.ones(len(signal), dtype=np.int)
    right_most_ratio = 1
    if level == 0 or len(signal) < 30 * fs:
        std = signal_std(signal)
        signal = signal / std
        right_most_ratio = 1 / std
        motion *= 0
    else:
        signal_crop, indices = signal_crop_motion(signal, window=10, threshold=10)
        if level == 3 and len(signal_crop) == len(signal):
            signal_crop, indices = signal_crop_motion(signal, window=10, threshold=6)
        motion[indices] = 0
        stable_periods = label_to_interval(motion, 0)
        for i, (p0, p1) in enumerate(stable_periods):
            signal_norm, right_r, motion_seg = detect_motion_iterative(signal[p0: p1], level=level - 1)
            signal[p0: p1] = signal_norm
            motion[p0: p1] = motion_seg
            if i != len(stable_periods) - 1:
                signal[p1:stable_periods[i + 1][0]] *= right_r
            else:
                right_most_ratio = right_r
    signal = np.clip(signal, -8, 8)
    return signal, right_most_ratio, motion


def detect_sleep_from_stage(stage, extend=0, threshold=400):
    stage_mask = stage.copy() > 0
    for (start, end) in label_to_interval(stage_mask, 0):
        # awake period smaller than 20 epochs -> 10 minutes
        if end - start < 20:
            stage_mask[start:end] = 1
    final_mask = stage_mask.copy() * 0
    for (start, end) in label_to_interval(stage_mask, 1):
        if end - start > threshold:
            final_mask[max(start - extend, 0):min(end + extend, len(stage_mask))] = 1
    return final_mask


def stage_mapping(stage):
    # [0: 'Awake', 1: 'REM', 2:'N1', 3: 'N2', 4: 'N3', 5:'Unknown'])
    map_function = np.array([0, 2, 3, 4, 5, 1, 5, 5, 5, 5])
    # map_function = np.array([0, 2, 2, 3, 1])
    return map_function[np.array(stage, dtype=np.int)]


def notch_filter(signal, fs, freqs=np.arange(60, 241, 60)):
    """
    https://en.wikipedia.org/w/index.php?title=Notch_filter
    Commonly used to filter powerline artifacts out of EEG signals.
    Default ``freqs`` correspond to US AC frequency and harmonics.
    """
    import mne
    return mne.filter.notch_filter(signal.astype(np.float64), fs, freqs, verbose=40)


def get_stats_exclude_outlier(signal):
    signal_sort = np.sort(signal)
    l = len(signal_sort)
    p75 = signal_sort[int(l * 0.75)]
    p25 = signal_sort[int(l * 0.25)]
    iqr = p75 - p25
    cell = p75 + 1.5 * iqr
    floor = p25 - 1.5 * iqr
    valid_index = np.where((signal_sort >= floor) & (signal_sort <= cell))
    std = np.std(signal_sort[valid_index])
    mean = np.mean(signal_sort[valid_index])
    return floor, cell, std, mean


def calculate_breathing_rate(signal, fs):
    FACTOR = 4
    THRESHOLD = 0.1 / FACTOR
    STEP = 30
    WIN = 60
    BPM_RANGE = np.array([8., 30.])

    from scipy.signal import spectrogram
    from scipy.ndimage.filters import gaussian_filter1d

    signal_pad = np.pad(signal, [[WIN * fs // 2, WIN * fs // 2 - 1]], mode="reflect")
    f, t, spec = spectrogram(signal_pad, fs=fs, nperseg=int(WIN * fs),
                             noverlap=int((WIN - STEP) * fs), nfft=int(WIN * fs * FACTOR),
                             mode='magnitude')

    spec = spec[:50 * FACTOR]
    f = f[:50 * FACTOR] * 60
    spec /= np.max(spec, axis=0)
    mask = (np.argmax(spec, axis=0) >= BPM_RANGE[0] * FACTOR) * (np.argmax(spec, axis=0) <= BPM_RANGE[1] * FACTOR)
    raw_score = np.max(spec[int(BPM_RANGE[0] * FACTOR): int(BPM_RANGE[1] * FACTOR)],
                       axis=0) / np.sum(spec, axis=0) * mask
    score = gaussian_filter1d(raw_score, 11)
    sleep_array = (score > THRESHOLD).astype(np.int)
    zero_intervals_raw = label_to_interval(sleep_array, 0)
    zero_intervals_after = []
    for (start, end) in zero_intervals_raw:
        if end - start > 3600 * 2 / STEP or end > len(sleep_array) - 10 or start < 10:
            zero_intervals_after.append((start, end))
    sleep_array = sleep_array * 0 + 1
    for (start, end) in zero_intervals_after:
        sleep_array[start:end] = 0

    one_intervals_raw = label_to_interval(sleep_array, 1)
    one_intervals_after = []
    for (start, end) in one_intervals_raw:
        if end - start > 3600 * 2 / STEP:
            one_intervals_after.append((start, end))
    sleep_array *= 0
    for (start, end) in one_intervals_after:
        sleep_array[start:end] = 1

    high_quality_mask = raw_score > THRESHOLD
    sleep_array_masked = sleep_array * high_quality_mask

    res = f[1] - f[0]
    spec_masked = spec * sleep_array_masked
    breathing_rate = np.argmax(spec_masked, axis=0)[np.where(sleep_array_masked)] * res
    b_distribution = np.sum(spec_masked, axis=1)
    b_distribution /= np.max(b_distribution)
    major_rate = f[np.argmax(b_distribution)]
    averaged_rate = np.average(breathing_rate)
    std_rate = np.std(breathing_rate)

    return major_rate, averaged_rate, std_rate


class TimeLocator(Locator):
    StepSizes = np.array([1e-3, 0.1, 1, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600, 7200, 14400, 1e10])

    def __init__(self, multiples=8):
        super().__init__()
        self.multiples = multiples

    def _raw_ticks(self, vmin, vmax):
        duration = vmax - vmin
        step_select = np.argmax(self.StepSizes * self.multiples >= duration) - 1
        step = self.StepSizes[step_select]
        begin_ts = (vmin // step) * step
        # if int(begin_ts) % int(step) != 0:
        #     begin_ts += step
        end_ts = (vmax // step + 1) * step
        ticks = np.arange(begin_ts, end_ts, step)
        return ticks

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=1e-13, tiny=1e-14)
        locs = self._raw_ticks(vmin, vmax)
        return self.raise_if_exceeds(locs)


def fake_start_ts():
    import datetime
    return datetime.datetime(2021, 1, 1, 0, 0, 0).timestamp()


def plot_stage(stage_ax, true_stage, pred_stage=None, stage_mask=None):
    import numpy as np
    from sklearn.metrics import accuracy_score
    stage_x = fake_start_ts() + np.arange(len(true_stage)) * 30
    stage_ax.step(stage_x, true_stage, where='post', label='Manual')
    if stage_mask is not None:
        stage_ax.step(stage_x, stage_mask, where='post', label=f'Mask', color='green')
    if pred_stage is not None:
        if stage_mask is None:
            stage_mask = np.ones_like(true_stage)
        pred_stage *= stage_mask
        acc_str = f"{accuracy_score(true_stage, pred_stage, sample_weight=stage_mask):.3f}"
        stage_ax.step(stage_x, pred_stage, where='post', label=f'Pred ({acc_str})', color='orange')
    stage_ax.set_yticks([0, 1, 2, 3, 4, 5])
    stage_ax.set_yticklabels(['Awake', 'REM', 'N1', 'N2', 'N3', 'U'])
    stage_ax.grid()
    stage_ax.legend(loc='upper right')
    set_xaxis_time_auto(stage_ax)


def plot_breathing_signal_and_spec(resp_ax, resp_spec_ax, sig, fps=None):
    fps = StandardFps if fps is None else fps
    resp_x = fake_start_ts() + np.arange(len(sig)) / fps
    duration = len(sig) / fps
    snr = signal_snr(sig, fps, crop_motion=False)
    resp_ax.plot(resp_x, sig, label=f'SNR: {snr:.3f}')
    resp_ax.set_ylim([-5, 5])
    resp_ax.legend(loc='upper right')
    resp_spec = spectrogram(sig, fs=5, spec_win_sec=30, cutoff_bpm=40, col_norm=True)
    plot_spec(resp_spec_ax, resp_spec, fake_start_ts(), fake_start_ts() + duration, 40)


def plot_eeg_signal_and_spec(sig_ax, sig, fps=64):
    import numpy as np
    resp_x = fake_start_ts() + np.arange(len(sig)) / fps
    sig_ax.plot(resp_x, sig, label=f'eeg')
    # sig_ax.set_ylim([-5, 5])
    sig_ax.legend(loc='upper right')


def standard_breathing_plot(signal, stage, unified_id=None, pred_stage=None, stage_mask=None):
    import matplotlib.pyplot as plt
    set_matplotlib_rcparams()
    fig, (stage_ax, resp_ax, spec_ax) = plt.subplots(3, 1, sharex='all')
    plot_stage(stage_ax, stage, pred_stage, stage_mask)
    plot_breathing_signal_and_spec(resp_ax, spec_ax, signal)
    plt.suptitle(unified_id)
    return fig, (stage_ax, resp_ax, spec_ax)


def eeg_stage_plot(signal, stage, uid=None, pred_stage=None, stage_mask=None):
    import matplotlib.pyplot as plt
    set_matplotlib_rcparams()
    fig, (stage_ax, sig_ax) = plt.subplots(2, 1, sharex='all')
    plot_stage(stage_ax, stage, pred_stage, stage_mask)
    plot_eeg_signal_and_spec(sig_ax, signal)
    plt.suptitle(uid)
    return fig, (stage_ax, sig_ax)


def set_xaxis_time_auto(ax, show_ticks=True, multiples=8):
    import datetime
    from matplotlib.ticker import FuncFormatter

    def format_hour(x, _=None):
        t = datetime.datetime.fromtimestamp(x)
        # return '%02d/%02d %02d:%02d:%02d' % (t.month, t.day, t.hour, t.minute, t.second)
        return f'{t.hour:02}:{t.minute:02}:{t.second:02}'

    if show_ticks:
        ax.xaxis.set_major_locator(TimeLocator(multiples))
    else:
        ax.axes.get_xaxis().set_visible(False)

    ax.xaxis.set_major_formatter(FuncFormatter(format_hour))


def plot_spec(ax, spec, start_ts, end_ts, y_top=None, cmap='jet'):
    if y_top is None:
        y_top = spec.shape[0]
    image = ax.imshow(spec, cmap=cmap, aspect='auto', extent=[start_ts, end_ts, y_top, 0])
    ax.invert_yaxis()
    return image


def set_matplotlib_rcparams():
    import matplotlib.pyplot as plt
    for key in plt.rcParams.keys():
        if key.startswith('keymap'):
            plt.rcParams[key] = ''
    plt.rcParams['keymap.pan'] = 'a'
    plt.rcParams['keymap.quit'] = 'ctrl+w, cmd+w'
    plt.rcParams['keymap.home'] = 'ctrl+h, cmd+h'
    plt.rcParams['keymap.fullscreen'] = 'ctrl+f, cmd+f'


def ctrl_decorator(key):
    return [f'ctrl+{key}', f'cmd+{key}']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# [classic breathing EEG]
def spectrogram(signal, fs=5, spec_win_sec=15, spec_step_sec=5, npad=4, cutoff_bpm=64, col_norm=False,
                return_sig=False):
    """
    :param signal: breathing signal from SHHS with a sampling frequency of 10, and its dtype should be float
    :param fs: sampling frequency of the signal, should only be 5 or 10
    :param spec_win_sec: spectrogram window size in seconds
    :param spec_step_sec: spectrogram step size in seconds
    :param return_sig: whether return processed breathing signal or not
    :return: spec: spectrogram of the breathing signal
    """
    from scipy.signal import spectrogram
    from scipy.ndimage.interpolation import zoom

    spec_win = StandardFps * spec_win_sec
    spec_step = StandardFps * spec_step_sec

    # Change signal sampling rate to 5
    if fs == 10:
        signal = signal[::2]
    elif fs != StandardFps:
        signal = zoom(signal, StandardFps / fs)

    signal = signal_crop(signal)
    cutoff = int(spec_win_sec * npad / 60 * cutoff_bpm)

    signal_pad = np.pad(signal, [[spec_win // 2, spec_win // 2 - 1]], mode="reflect")
    _, _, spec = spectrogram(
        signal_pad, window='hamming', nperseg=spec_win, noverlap=(spec_win - spec_step),
        nfft=npad * spec_win, mode='magnitude'
    )
    # assert spec.shape[1] * spec_step == signal.shape[0], f'{spec.shape[1] * spec_step}, {signal.shape[0]}'
    spec = spec[:cutoff]
    if col_norm:
        spec /= (np.max(spec, axis=0, keepdims=True) + 1e-5)
    else:
        spec /= np.std(spec)  # Normalizing spectrogram based on its standard deviation
    spec = spec.astype(np.float32)
    if not return_sig:
        return spec
    else:
        return spec, signal


# ================================================================================ #
# [for EEG]
def eeg_spectrogram(signal, fs=5, spec_win_sec=15, spec_step_sec=5,
                    npad=4, cutoff_hz=32, col_norm=False, return_sig=False,
                    mode='magnitude'):  # [old version]
    """
    :param signal: breathing signal from SHHS with a sampling frequency of 10, and its dtype should be float
    :param fs: sampling frequency of the signal, should only be 5 or 10
    :param spec_win_sec: spectrogram window size in seconds
    :param spec_step_sec: spectrogram step size in seconds
    :param return_sig: whether return processed breathing signal or not
    :return: spec: spectrogram of the breathing signal
    """
    from scipy.signal import spectrogram
    from scipy.ndimage.interpolation import zoom

    StandardFps = 64
    spec_win = StandardFps * spec_win_sec
    spec_step = StandardFps * spec_step_sec

    # Change signal sampling rate to 5
    if fs == 10:
        signal = signal[::2]
    elif fs != StandardFps:
        signal = zoom(signal, StandardFps / fs)

    # signal = signal_crop(signal)
    cutoff = int(spec_win_sec * npad * cutoff_hz)

    signal_pad = np.pad(signal, [[spec_win // 2, spec_win // 2 - 1]], mode="reflect")
    _, _, spec = spectrogram(
        signal_pad, window='hamming', nperseg=spec_win, noverlap=(spec_win - spec_step),
        nfft=int(npad * spec_win), mode=mode,
    )
    # assert spec.shape[1] * spec_step == signal.shape[0], f'{spec.shape[1] * spec_step}, {signal.shape[0]}'
    spec = spec[:cutoff]
    if col_norm:
        spec /= (np.max(spec, axis=0, keepdims=True) + 1e-5)
    else:
        spec /= np.std(spec)  # Normalizing spectrogram based on its standard deviation
    spec = spec.astype(np.float32)
    if not return_sig:
        return spec
    else:
        return spec, signal


def compute_multitaper_spec(eeg_file):
    import mne
    # [eeg.shape = (#channel, #point)]
    Fs = eeg_file['fs']  # 200  # [Hz]
    eeg = eeg_file['data'].astype(np.float64)
    if eeg_file['data'].std() < 1:
        eeg *= 1e6
    if eeg_file['data'].std() > 50:
        eeg /= 5  # FIXME: hack for mgh
    eeg = eeg - eeg.mean()  # remove baseline

    # [segment into 30-second epochs]
    epoch_time = 30  # [second]
    step_time = 30
    epoch_size = int(round(epoch_time * Fs))
    step_size = int(round(step_time * Fs))

    epochs = eeg.reshape(-1, epoch_size)  # #epoch, size(epoch)
    epochs = np.concatenate([epochs, np.zeros((epochs.shape[0],
                                               int(32 * Fs) - epoch_size))], 1)

    spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.0, fmax=32, bandwidth=0.5,
                                                         normalization='full', verbose=False,
                                                         n_jobs=12)  # spec.shape = (#epoch, #freq)
    spec_db = 10 * np.log10(spec)
    spec_db = spec_db[:, 1::4]
    return spec_db.T


# ================================================================================ #
# [for loading]

def load_file(dataset, chn, uid):
    from pathlib import Path
    root = Path('/data/netmit/wifall/ADetect/data')
    file_path = root / f'{dataset}' / chn / f'{uid}.npz'
    if file_path.is_file():
        return np.load(file_path, allow_pickle=True)
    else:
        return None

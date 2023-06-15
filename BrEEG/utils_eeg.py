import numpy as np


def eeg_spectrogram_multitaper(eeg, Fs=64, fmin=0, fmax=32):
    import mne
    # preprocessing by filtering
    if eeg.std() < 1:  # detect whether eeg
        eeg = eeg * 1e6
    eeg = eeg - eeg.mean()  # remove baseline

    # first segment into 30-second epochs
    epoch_time = 30  # [second]
    epoch_size = int(round(epoch_time * Fs))

    epochs = eeg.reshape(-1, epoch_size)  # #epoch, size(epoch)
    epochs = np.concatenate([epochs, np.zeros((epochs.shape[0],
                                               int(32 * Fs) - epoch_size))], 1)
    spec, freq = mne.time_frequency.psd_array_multitaper(
        epochs, Fs, fmin=fmin, fmax=fmax, bandwidth=0.5, normalization='full', verbose=False, n_jobs=12
    )  # spec.shape = (#epoch, #freq)
    spec_db = 10 * np.log10(spec)
    spec_db = spec_db[:, 1::4]
    freq = freq[1::4]
    return spec_db.T


def eeg_spectrogram(signal, fs=5, spec_win_sec=15, spec_step_sec=5,
                    npad=4, cutoff_hz=32, col_norm=False, return_sig=False, mode='magnitude'):
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
        nfft=npad * spec_win, mode=mode,
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


def post_process_eeg(eeg):
    eeg = (eeg - np.median(eeg)) / (np.percentile(eeg, 75) - np.percentile(eeg, 25)) * 1.35
    return eeg

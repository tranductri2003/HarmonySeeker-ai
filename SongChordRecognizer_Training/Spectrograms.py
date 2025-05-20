import librosa
import numpy as np
import sys


def log_mel_spectrogram(
    waveform, sample_rate: int = 44100, nfft: int = 2**14, hop_length: int = 4410
):
    """
    Generate spectrogram from audio.
    The logarithmized mel spectrogram is used with 128 filter banks.

    Parameters
    ----------
    waveform : list of floats
        data of audio waveform
    sample_rate : int
        audio sample rate
    nfft : int
        length of FFT, power of 2
    hop_length : int
        number of target rate, sample_rate/hop_length = interval between two spectrograms in miliseconds
    Returns
    -------
    log_spectrogram : float array
        logarithmized mel spectrogram over all audio
    """
    n_mels = 128

    mel_spectrogram = librosa.feature.melspectrogram(
        waveform, sr=sample_rate, n_fft=nfft, hop_length=hop_length, n_mels=n_mels
    )
    log_spectrogram = librosa.amplitude_to_db(mel_spectrogram)

    return log_spectrogram


def cqt_spectrogram(
    waveform, sample_rate: int = 44100, nfft: int = 2**14, hop_length: int = 4410
):
    """
    Generate spectrogram from audio.
    The cqt spectrogram is used with 252 bins started at C1 and with 36 filter banks for each octaves.

    Parameters
    ----------
    waveform : list of floats
        data of audio waveform
    sample_rate : int
        audio sample rate
    nfft : int
        length of FFT, power of 2
    hop_length : int
        number of target rate, sample_rate/hop_length = interval between two spectrograms in miliseconds
    Returns
    -------
    cqt_spectrogram : float array
        cqt spectrogram over all audio
    """
    n_bins = 252
    bins_per_octave = 36

    cqt_spectrogram = np.abs(
        librosa.cqt(
            waveform,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=librosa.note_to_hz("C1"),
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )

    return cqt_spectrogram


def stft_chromagram(
    waveform, sample_rate: int = 44100, nfft: int = 2**14, hop_length: int = 4410
):
    """
    Generate chromagram from audio.
    The stft spectrogram is used.

    Parameters
    ----------
    waveform : list of floats
        data of audio waveform
    sample_rate : int
        audio sample rate
    nfft : int
        length of FFT, power of 2
    hop_length : int
        number of target rate, sample_rate/hop_length = interval between two spectrograms in miliseconds
    Returns
    -------
    stft_chroma : float array
        stft chromagram
    """
    stft_chroma = librosa.feature.chroma_stft(
        y=waveform, sr=sample_rate, n_chroma=12, hop_length=hop_length, n_fft=nfft
    )

    return stft_chroma


def cqt_chromagram(
    waveform, sample_rate: int = 44100, nfft: int = 2**14, hop_length: int = 4410
):
    """
    Generate chromagram from audio.
    The cqt chromagram is used with 252 bins started at C1 and with 36 filter banks for each octaves.

    Parameters
    ----------
    waveform : list of floats
        data of audio waveform
    sample_rate : int
        audio sample rate
    nfft : int
        length of FFT, power of 2
    hop_length : int
        number of target rate, sample_rate/hop_length = interval between two spectrograms in miliseconds
    Returns
    -------
    cqt_chroma : float array
        cqt chromagram
    """
    cqt_chroma = librosa.feature.chroma_cqt(
        y=waveform, sr=sample_rate, hop_length=hop_length
    )

    return cqt_chroma

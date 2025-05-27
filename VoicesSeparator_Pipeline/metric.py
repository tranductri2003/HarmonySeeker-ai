from keras import saving, ops, layers
import keras
from VoicesSeparator_Pipeline.audio_process import inverse_stft


# Configuration parameters (phải giống với training)
CHUNK_SIZE = 65024  # ~4 seconds at 16kHz
STFT_N_FFT = 2048
STFT_HOP_LENGTH = 512
N_SUBBANDS = 4
N_INSTRUMENTS = 1  # vocals only
TARGET_SAMPLE_RATE = 16000


@saving.register_keras_serializable()
def spectral_loss(y_true, y_pred):
    """Mean absolute error in the STFT domain."""

    def target_to_stft(y):
        y = ops.reshape(y, (-1, CHUNK_SIZE))
        y_real, y_imag = ops.stft(y, STFT_N_FFT, STFT_HOP_LENGTH, STFT_N_FFT)
        y_real, y_imag = y_real[..., :-1], y_imag[..., :-1]
        y = ops.stack([y_real, y_imag], axis=-1)
        return ops.reshape(y, (-1, N_INSTRUMENTS, y.shape[1], y.shape[2] * 2))

    y_true = target_to_stft(y_true)
    return ops.mean(ops.absolute(y_true - y_pred))


@saving.register_keras_serializable()
def sdr(y_true, y_pred):
    """Signal-to-Distortion Ratio metric."""

    def prediction_to_wave(x, n_instruments=N_INSTRUMENTS):
        x = ops.reshape(x, (-1, x.shape[2], x.shape[3] // 2, 2))
        x = inverse_stft(x)
        return ops.reshape(x, (-1, n_instruments, x.shape[1]))

    y_pred = prediction_to_wave(y_pred)
    num = ops.sum(ops.square(y_true), axis=-1) + 1e-8
    den = ops.sum(ops.square(y_true - y_pred), axis=-1) + 1e-8
    return 10 * ops.log10(num / den)

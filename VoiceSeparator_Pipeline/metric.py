from keras import saving, ops, layers
import keras
import os
from dotenv import load_dotenv
from VoiceSeparator_Pipeline.audio_process import inverse_stft

# Load environment variables
load_dotenv()

# Configuration parameters from environment
CHUNK_SIZE = int(os.getenv("VOICE_CHUNK_SIZE"))
STFT_N_FFT = int(os.getenv("VOICE_STFT_N_FFT"))
STFT_HOP_LENGTH = int(os.getenv("VOICE_STFT_HOP_LENGTH"))
N_SUBBANDS = int(os.getenv("VOICE_N_SUBBANDS"))
N_INSTRUMENTS = int(os.getenv("VOICE_N_INSTRUMENTS"))
TARGET_SAMPLE_RATE = int(os.getenv("VOICE_TARGET_SAMPLE_RATE"))


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

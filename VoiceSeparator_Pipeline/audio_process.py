import os
import numpy as np
import soundfile as sf
from keras import saving, ops, layers
import keras
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration parameters from environment
CHUNK_SIZE = int(os.getenv("VOICE_CHUNK_SIZE"))
STFT_N_FFT = int(os.getenv("VOICE_STFT_N_FFT"))
STFT_HOP_LENGTH = int(os.getenv("VOICE_STFT_HOP_LENGTH"))
N_SUBBANDS = int(os.getenv("VOICE_N_SUBBANDS"))
N_INSTRUMENTS = int(os.getenv("VOICE_N_INSTRUMENTS"))
TARGET_SAMPLE_RATE = int(os.getenv("VOICE_TARGET_SAMPLE_RATE"))


def load_audio(audio, sr, target_sr=TARGET_SAMPLE_RATE):
    """Load audio file and convert to mono at target sample rate."""
    # file_bytes = await file.read()
    # audio, sr = sf.read(io.BytesIO(file_bytes), dtype='float32')
    # audio, sr = sf.read(file_path, dtype='float32')

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Simple resampling if needed
    if sr != target_sr:
        print(f"Warning: Resampling from {sr}Hz to {target_sr}Hz")
        try:
            from scipy import signal

            audio = signal.resample(audio, int(len(audio) * target_sr / sr))
        except ImportError:
            print("scipy not available, keeping original sample rate")

    return audio.astype(np.float32)


def preprocess_audio(audio, chunk_size=CHUNK_SIZE):
    """Preprocess audio into chunks for model input."""
    chunks = []

    # Pad audio if shorter than chunk_size
    if len(audio) < chunk_size:
        audio = np.pad(audio, (0, chunk_size - len(audio)))

    # Split audio into chunks
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)

    return np.array(chunks)


def inverse_stft(inputs, fft_size=STFT_N_FFT, sequence_stride=STFT_HOP_LENGTH):
    """Compute inverse STFT."""
    x = inputs

    if keras.backend.backend() == "torch":
        x = ops.pad(x, ((0, 0), (0, 0), (0, 1), (0, 0)))

    real_x, imag_x = ops.split(x, 2, axis=-1)
    real_x = ops.squeeze(real_x, axis=-1)
    imag_x = ops.squeeze(imag_x, axis=-1)

    return ops.istft((real_x, imag_x), fft_size, sequence_stride, fft_size)


def prediction_to_wave(x, n_instruments=N_INSTRUMENTS):
    """Convert STFT predictions back to waveform."""
    x = ops.reshape(x, (-1, x.shape[2], x.shape[3] // 2, 2))
    x = inverse_stft(x)
    return ops.reshape(x, (-1, n_instruments, x.shape[1]))


def postprocess_audio(chunks, original_length):
    """Reconstruct full audio from chunks."""
    if len(chunks.shape) == 3:  # (batch, instruments, samples)
        chunks = chunks[:, 0, :]  # Take first instrument (vocals)

    # Concatenate chunks
    full_audio = chunks.flatten()

    # Trim to original length
    if original_length < len(full_audio):
        full_audio = full_audio[:original_length]

    return full_audio

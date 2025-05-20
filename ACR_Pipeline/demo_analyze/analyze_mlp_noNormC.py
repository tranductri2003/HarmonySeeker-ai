import os

os.environ["TF_DISABLE_GPU"] = "1"  # Disable GPU
os.environ["DISABLE_METAL_PLUGIN"] = "1"  # Disable Metal plugin

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import os
import sys
import joblib

# Fix the import path
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Check if ACR_Training module exists
try:
    import ACR_Training

    print("ACR_Training module found!")
except ImportError as e:
    print(f"Error importing ACR_Training: {e}")
    sys.exit(1)

# Import required modules
from ACR_Training.Models import MLP_scalered
from ACR_Training.Spectrograms import log_mel_spectrogram
from ACR_Pipeline.KeyRecognizer import KeyRecognizer
from ACR_Pipeline.DataPreprocessor import DataPreprocessor


def analyze_song(audio_path):
    print(f"Analyzing {audio_path}...")

    try:
        print("Step 1: Loading audio")
        y, sr = librosa.load(audio_path, sr=44100)

        print("Step 2: Loading MLP model")
        model = joblib.load(
            r"/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/ACR_Pipeline/models/obsolete/original_mlp.model"
        )

        print("Step 3: Preprocessing audio")
        x = DataPreprocessor.flatten_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=1024,
            window_size=5,
            spectrogram_generator=log_mel_spectrogram,
            norm_to_C=False,
            skip_coef=22,
        )

        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        print("Step 4: Predicting chords")
        predictions = model.predict(x)
        chords, counts = np.unique(predictions, return_counts=True)
        chord_counts = dict(zip(chords, counts))

        print("Step 5: Estimating key")
        key = KeyRecognizer.estimate_key(chord_counts)

        print("Step 6: Converting chord indices to notations")
        chord_sequence = DataPreprocessor.chord_indices_to_notations(predictions)

        print(f"\nResults for {os.path.basename(audio_path)}:")
        print(f"Key: {key}")
        # print(f"Chord progression: {chord_sequence}")

        return key, chord_sequence

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    audio_path = "/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/ACR_Pipeline/demo_song/muahong-lequyen.wav"

    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} not found")
        sys.exit(1)

    analyze_song(audio_path)

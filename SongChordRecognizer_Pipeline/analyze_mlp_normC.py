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

# Check if SongChordRecognizer_Training module exists
import SongChordRecognizer_Training

print("SongChordRecognizer_Training module found!")
from SongChordRecognizer_Training.Models import MLP_scalered
from SongChordRecognizer_Training.Spectrograms import log_mel_spectrogram
from SongChordRecognizer_Pipeline.KeyRecognizer import KeyRecognizer
from SongChordRecognizer_Pipeline.DataPreprocessor import DataPreprocessor


def analyze_song(audio_path):
    print(f"Analyzing {audio_path}...")

    try:
        print("Step 1: Loading audio")
        y, sr = librosa.load(audio_path, sr=44100)

        print("Step 2: Loading MLP model")
        # Load the model that was trained with norm_to_C=True
        model = joblib.load(
            r"/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/SongChordRecognizer_Pipeline/my_models/mlp22_isophonics_206songs_normC.model"
        )

        print("Step 3: Getting initial key for transposition")
        # First get non-normalized spectrograms to detect key
        x_for_key = DataPreprocessor.flatten_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=1024,
            window_size=5,
            spectrogram_generator=log_mel_spectrogram,
            norm_to_C=False,
            skip_coef=22,
        )

        if len(x_for_key.shape) > 2:
            x_for_key = x_for_key.reshape(x_for_key.shape[0], -1)

        initial_predictions = model.predict(x_for_key)
        chords, counts = np.unique(initial_predictions, return_counts=True)
        chord_counts = dict(zip(chords, counts))
        original_key = KeyRecognizer.estimate_key(chord_counts)

        print(f"Original key detected: {original_key}")

        print("Step 4: Preprocessing audio with normalization to C")
        x = DataPreprocessor.flatten_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=1024,
            window_size=5,
            spectrogram_generator=log_mel_spectrogram,
            norm_to_C=True,
            key=original_key,
            skip_coef=22,
        )

        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        print("Step 5: Predicting chords")
        predictions = model.predict(x)

        print("Step 6: Transposing predictions back to original key")
        original_predictions = DataPreprocessor.transpose(
            chord_sequence=predictions, from_key="C", to_key=original_key
        )

        print("Step 7: Converting chord indices to notations")
        chord_sequence = DataPreprocessor.chord_indices_to_notations(
            original_predictions
        )

        print(f"\nResults for {os.path.basename(audio_path)}:")
        print(f"Key: {original_key}")
        # print(f"Chord progression: {chord_sequence}")

        return original_key, chord_sequence

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    audio_path = "/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/SongChordRecognizer_Pipeline/demo_song/muahong-lehieu.wav"

    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} not found")
        sys.exit(1)

    analyze_song(audio_path)

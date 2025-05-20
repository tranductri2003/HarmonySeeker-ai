import os
import sys
import time
import traceback
import numpy as np
import librosa
import sounddevice as sd

# Environment setup
os.environ["DISABLE_METAL_PLUGIN"] = (
    "1"  # Disable Metal plugin (for macOS GPU compatibility issues)
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Suppress most TensorFlow logs

# Fix import path for ACR_Training module
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Check if ACR_Training module exists
try:
    import ACR_Training
    from ACR_Training.Models import CRNN_basic_WithStandardScaler
    from ACR_Training.Spectrograms import cqt_spectrogram
    from ACR_Pipeline.KeyRecognizer import KeyRecognizer
    from ACR_Pipeline.DataPreprocessor import DataPreprocessor

    print("ACR_Training module found!")
except ImportError as e:
    print(f"Error importing ACR_Training: {e}")
    sys.exit(1)


def analyze_song(audio_path):
    print(f"Analyzing {audio_path}...")

    try:
        # Step 1: Load audio
        print("Step 1: Loading audio")
        y, sr = librosa.load(audio_path, sr=22050)

        # Step 2: Load pre-trained model
        print("Step 2: Loading model")
        basic_crnn = CRNN_basic_WithStandardScaler()
        basic_crnn.load(
            "/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/ACR_Pipeline/models/original_crnn.h5",
            "/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/ACR_Pipeline/models/original_preprocessor.bin",
        )

        # Step 3: Preprocess audio into spectrogram
        print("Step 3: Preprocessing audio")
        x = DataPreprocessor.sequence_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=512,
            n_frames=1000,
            spectrogram_generator=cqt_spectrogram,
            norm_to_C=False,
        )

        # Step 4: Predict chord sequence
        print("Step 4: Predicting chords")
        predictions = basic_crnn.predict(x)
        chord_prediction = predictions.argmax(axis=2).flatten()
        chords, counts = np.unique(chord_prediction, return_counts=True)
        chord_counts = dict(zip(chords, counts))

        # Step 5: Estimate musical key
        print("Step 5: Estimating key")
        key = KeyRecognizer.estimate_key(chord_counts)

        # Step 6: Convert indices to chord labels
        print("Step 6: Converting chord indices to notations")
        chord_sequence = DataPreprocessor.chord_indices_to_notations(chord_prediction)

        print(f"\nResults for {os.path.basename(audio_path)}:")
        print(f"Chord progression: {chord_sequence}")
        print(f"Key: {key}")

        # --- Play audio and print corresponding chords in sync ---
        if sd is None:
            print(
                "\n[ERROR] 'sounddevice' is not installed. Please run: pip install sounddevice\nThen re-run the script."
            )
            return key, chord_sequence

        duration = librosa.get_duration(y=y, sr=sr)
        num_chords = len(chord_sequence)
        interval = duration / num_chords if num_chords > 0 else duration

        print(f"\nPlaying audio and displaying corresponding chords:")
        print(
            f"Total duration: {duration:.2f}s | Number of chords: {num_chords} | Each chord duration: {interval:.2f}s"
        )

        # Normalize audio to [-1, 1] for sounddevice
        y_norm = y / np.max(np.abs(y))

        # Start background audio playback
        sd.play(y_norm, sr)
        start_time = time.time()
        for i, chord in enumerate(chord_sequence):
            now = time.time()
            elapsed = now - start_time
            target = i * interval
            if elapsed < target:
                time.sleep(target - elapsed)
            print(f"[Time {target:.2f}s] Chord: {chord}")
        sd.wait()
        print("\nAudio playback finished.")
        # --- End ---

        return key, chord_sequence

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    audio_path = "/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/ACR_Pipeline/demo_song/muahong-lehieu.wav"

    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        sys.exit(1)

    analyze_song(audio_path)

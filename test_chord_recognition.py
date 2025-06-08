import os
import sys
import numpy as np
import librosa
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Disable TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0 = all, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
)
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# Import necessary modules
from SongChordRecognizer_Pipeline.DataPreprocessor import DataPreprocessor
from SongChordRecognizer_Training.Models import CRNN_basic_WithStandardScaler
from SongChordRecognizer_Training.Spectrograms import cqt_spectrogram
from SongChordRecognizer_Pipeline.KeyRecognizer import KeyRecognizer

# Import Essentia for scale detection
from essentia.standard import KeyExtractor

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Load environment variables
load_dotenv()

# Constants
DATASET_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/Datasets/test"
CRNN_MODEL_PATH = os.getenv("CRNN_MODEL_PATH")
CRNN_SCALER_PATH = os.getenv("CRNN_SCALER_PATH")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "22050"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "512"))
N_FRAMES = int(os.getenv("N_FRAMES", "1000"))


def estimate_scale_with_essentia(y, sr):
    """
    Estimate the scale (major/minor) using Essentia KeyExtractor.
    Only the scale is returned and used to assist CRNN prediction.
    """
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100
    y = y.astype(np.float32)

    key_extractor = KeyExtractor()
    key, scale, _ = key_extractor(y)

    return scale


def predict_song_key(model, audio_path):
    """Predict key for a song with scale detection"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Get scale from Essentia
        scale = estimate_scale_with_essentia(y, sr)

        # Preprocess audio
        x = DataPreprocessor.sequence_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=HOP_LENGTH,
            n_frames=N_FRAMES,
            spectrogram_generator=cqt_spectrogram,
            norm_to_C=False,
        )

        # Predict chords
        predictions = model.predict(x)
        chord_indices = predictions.argmax(axis=2).flatten()

        # Get key with scale consideration
        chords, counts = np.unique(chord_indices, return_counts=True)
        chord_counts = dict(zip(chords, counts))
        key = KeyRecognizer.estimate_key(
            chord_counts, use_relative_mode=True, target_scale=scale
        )

        return key
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def analyze_dataset():
    """Analyze the entire dataset"""
    print("🔄 Loading model...")
    model = CRNN_basic_WithStandardScaler()
    model.load(CRNN_MODEL_PATH, CRNN_SCALER_PATH)

    results = []

    print("\n📂 Analyzing dataset...")
    # Iterate through each subfolder (true chord = folder name)
    for chord_folder in tqdm(os.listdir(DATASET_PATH), desc="Processing chord folders"):
        chord_folder_path = os.path.join(DATASET_PATH, chord_folder)
        if not os.path.isdir(chord_folder_path):
            continue

        true_chord = chord_folder  # The folder name is the true chord

        # Iterate through each audio file in the folder
        for audio_file in os.listdir(chord_folder_path):
            if not audio_file.endswith(".wav"):
                continue

            audio_path = os.path.join(chord_folder_path, audio_file)
            predicted_key = predict_song_key(model, audio_path)

            if predicted_key:
                results.append(
                    {
                        "file": audio_file,
                        "true_chord": true_chord,
                        "predicted_key": predicted_key,
                        "is_correct": predicted_key == true_chord,
                    }
                )

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print("\n📊 Analysis Results:")
    print(f"Total files analyzed: {len(results)}")
    print(f"Correct predictions: {df['is_correct'].sum()}")
    print(f"Accuracy: {df['is_correct'].mean():.2%}")

    # Print classification report
    print("\n📈 Key Classification Report:")
    print(classification_report(df["true_chord"], df["predicted_key"]))

    # Plot confusion matrix
    print("\n🎯 Plotting Key Confusion Matrix...")
    labels = sorted(list(set(df["true_chord"].tolist() + df["predicted_key"].tolist())))
    cm = confusion_matrix(df["true_chord"], df["predicted_key"], labels=labels)

    plt.figure(figsize=(15, 15))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Key Confusion Matrix")
    plt.xlabel("Predicted Key")
    plt.ylabel("True Chord")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig("key_confusion_matrix.png")

    # Save detailed results
    print("\n💾 Saving results...")
    df.to_csv("key_recognition_results.csv", index=False)

    # Print examples
    print("\n📝 Sample Predictions:")
    print(df[["file", "true_chord", "predicted_key", "is_correct"]].head(10))


if __name__ == "__main__":
    analyze_dataset()

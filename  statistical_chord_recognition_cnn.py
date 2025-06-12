from email.quoprimime import body_length
import os
import sys
import warnings
from matplotlib.mlab import window_none
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from essentia.standard import KeyExtractor
from joblib import load
import tensorflow as tf


# Suppress warnings and TF logging
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Custom imports
from SongChordRecognizer_Pipeline.DataPreprocessor import DataPreprocessor
from SongChordRecognizer_Training.Models import CNN
from SongChordRecognizer_Training.Spectrograms import cqt_spectrogram
from SongChordRecognizer_Pipeline.KeyRecognizer import KeyRecognizer

# Load environment variables
load_dotenv()

# Constants
DATASET_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/Datasets/test_vocal_only"
CNN_MODEL_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.keras"
SCALER_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.bin"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FRAMES = 17
TESTING_REPORT_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_testing_report"


# Chord definitions
MAJOR_CHORDS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MINOR_CHORDS = [chord + ":min" for chord in MAJOR_CHORDS]

CHORD_CLASSES = []
for major, minor in zip(MAJOR_CHORDS, MINOR_CHORDS):
    CHORD_CLASSES.append(major)
    CHORD_CLASSES.append(minor)


def shift_chord(chord, semitone_shift):
    if chord in MAJOR_CHORDS:
        idx = MAJOR_CHORDS.index(chord)
        shifted_idx = (idx + semitone_shift) % len(MAJOR_CHORDS)
        return MAJOR_CHORDS[shifted_idx]
    elif chord in MINOR_CHORDS:
        idx = MINOR_CHORDS.index(chord)
        shifted_idx = (idx + semitone_shift) % len(MINOR_CHORDS)
        return MINOR_CHORDS[shifted_idx]
    else:
        return chord


def is_match(predicted, true, tolerance=0, use_scale=False):
    if predicted == true:
        return True
    if tolerance == 0:
        return False

    semitone_range = int(tolerance / 0.5)
    for shift in range(-semitone_range, semitone_range + 1):
        shifted = shift_chord(true, shift)
        if use_scale:
            if shifted == predicted:
                return True
        else:
            pred_root = predicted.split(":")[0]
            true_root = shifted.split(":")[0]
            if pred_root == true_root:
                return True
    return False


def estimate_scale_with_essentia(y, sr):
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100
    y = y.astype(np.float32)
    key_extractor = KeyExtractor()
    _, scale, _ = key_extractor(y)
    return scale


def predict_song_key(model, audio_path, use_scale=False):
    """Predict the musical key of a song file using CNN model."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        scale = estimate_scale_with_essentia(y, sr)

        x = DataPreprocessor.sequence_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=HOP_LENGTH,
            n_frames=N_FRAMES,
            spectrogram_generator=cqt_spectrogram,
            norm_to_C=False,
        )

        # Normalize
        mean, std = load(SCALER_PATH)
        x = (x - mean) / std
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))

        # Predict chords
        predictions = model.predict(x)
        chord_indices = predictions.argmax(axis=1)
        chords, counts = np.unique(chord_indices, return_counts=True)
        chord_counts = dict(zip(chords, counts))

        # Determine key
        key = KeyRecognizer.estimate_key(
            chord_counts, use_relative_mode=use_scale, target_scale=scale
        )
        return key
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def analyze_dataset(tolerance=0.0, use_scale=False):
    """Run prediction analysis on dataset and produce reports/metrics using CNN."""
    print("🔄 Loading model...")
    model = CNN(input_shape=(N_FRAMES, 252, 1))
    model.load(CNN_MODEL_PATH)

    results = []
    print("\n📂 Analyzing dataset...")

    for chord_folder in tqdm(os.listdir(DATASET_PATH), desc="Processing chord folders"):
        chord_folder_path = os.path.join(DATASET_PATH, chord_folder)
        if not os.path.isdir(chord_folder_path):
            continue

        true_chord = chord_folder

        for audio_file in os.listdir(chord_folder_path):
            if not audio_file.endswith(".wav"):
                continue

            audio_path = os.path.join(chord_folder_path, audio_file)
            predicted_key = predict_song_key(model, audio_path, use_scale=use_scale)

            if predicted_key:
                match = is_match(
                    predicted_key, true_chord, tolerance=tolerance, use_scale=use_scale
                )
                results.append(
                    {
                        "file": audio_file,
                        "true_chord": true_chord,
                        "predicted_key": predicted_key,
                        "is_correct": match,
                    }
                )

    df = pd.DataFrame(results)

    # Print results summary
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

    os.makedirs(TESTING_REPORT_PATH, exist_ok=True)

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
    plt.savefig(
        os.path.join(
            TESTING_REPORT_PATH,
            "[cnn_separated_semi_strict_with_major_minor]_confusion_matrix.png",
        )
    )

    # Save results
    print("\n💾 Saving results...")
    df.to_csv(
        os.path.join(
            TESTING_REPORT_PATH,
            "[cnn_separated_semi_strict_with_major_minor]_key_recognition_results.csv",
        ),
        index=False,
    )

    # Display sample results
    print(f"\n✅ Saved confusion matrix and CSV to {TESTING_REPORT_PATH}")
    print("\n📝 Sample Predictions:")
    print(df[["file", "true_chord", "predicted_key", "is_correct"]].head(10))


if __name__ == "__main__":
    analyze_dataset(tolerance=0.5, use_scale=True)

"""
1. [cnn_original_strict_no_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 21
Accuracy: 16.80%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.25      0.36      0.29        14
       A:min       0.00      0.00      0.00         9
           B       0.00      0.00      0.00         4
       B:min       0.00      0.00      0.00         5
           C       0.22      0.13      0.17        15
          C#       0.00      0.00      0.00         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.14      0.19      0.16        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       0.38      0.27      0.32        11
       E:min       0.00      0.00      0.00         7
           F       0.00      0.00      0.00        12
          F#       0.00      0.00      0.00         1
      F#:min       0.00      0.00      0.00         4
       F:min       0.00      0.00      0.00         1
           G       0.13      0.89      0.22         9
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.17       125
   macro avg       0.06      0.09      0.06       125
weighted avg       0.12      0.17      0.12       125





2. [cnn_original_strict_with_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 28
Accuracy: 22.40%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.33      0.36      0.34        14
       A:min       0.00      0.00      0.00         9
           B       0.00      0.00      0.00         4
       B:min       0.00      0.00      0.00         5
           C       0.29      0.13      0.18        15
          C#       0.00      0.00      0.00         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.21      0.19      0.20        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       0.50      0.27      0.35        11
       E:min       0.26      0.86      0.40         7
           F       0.00      0.00      0.00        12
          F#       0.00      0.00      0.00         1
      F#:min       0.20      0.25      0.22         4
       F:min       0.00      0.00      0.00         1
           G       0.20      0.89      0.33         9
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.22       125
   macro avg       0.10      0.15      0.10       125
weighted avg       0.18      0.22      0.17       125





3. [cnn_original_semi_strict_no_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 27
Accuracy: 21.60%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.30      0.43      0.35        14
       A:min       0.00      0.00      0.00         9
           B       0.00      0.00      0.00         4
       B:min       0.00      0.00      0.00         5
           C       0.18      0.13      0.15        15
          C#       0.00      0.00      0.00         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.24      0.31      0.27        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       0.50      0.27      0.35        11
       E:min       0.00      0.00      0.00         7
           F       0.00      0.00      0.00        12
          F#       0.00      0.00      0.00         1
      F#:min       0.00      0.00      0.00         4
       F:min       0.00      0.00      0.00         1
           G       0.13      0.89      0.22         9
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.19       125
   macro avg       0.07      0.10      0.07       125
weighted avg       0.14      0.19      0.14       125





4. [cnn_original_semi_strict_with_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 36
Accuracy: 28.80%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.40      0.43      0.41        14
       A:min       0.00      0.00      0.00         9
           B       0.00      0.00      0.00         4
       B:min       0.17      0.20      0.18         5
           C       0.22      0.13      0.17        15
          C#       0.00      0.00      0.00         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.33      0.31      0.32        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       0.60      0.27      0.38        11
       E:min       0.29      1.00      0.45         7
           F       0.00      0.00      0.00        12
          F#       0.00      0.00      0.00         1
      F#:min       0.40      0.50      0.44         4
       F:min       0.00      0.00      0.00         1
           G       0.21      0.89      0.33         9
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.27       125
   macro avg       0.13      0.19      0.13       125
weighted avg       0.22      0.27      0.21       125





5. [cnn_separated_strict_no_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 25
Accuracy: 20.00%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.00      0.00      0.00        14
          A#       0.00      0.00      0.00         0
       A:min       0.00      0.00      0.00         9
           B       0.50      0.25      0.33         4
       B:min       0.00      0.00      0.00         5
           C       0.47      0.47      0.47        15
          C#       0.00      0.00      0.00         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.17      0.62      0.27        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       1.00      0.18      0.31        11
       E:min       0.00      0.00      0.00         7
           F       0.29      0.17      0.21        12
          F#       0.00      0.00      0.00         1
      F#:min       0.00      0.00      0.00         4
       F:min       0.00      0.00      0.00         1
           G       0.10      0.33      0.15         9
          G#       0.00      0.00      0.00         0
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.20       125
   macro avg       0.11      0.09      0.08       125
weighted avg       0.22      0.20      0.16       125





6. [cnn_separated_strict_with_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 34
Accuracy: 27.20%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.00      0.00      0.00        14
          A#       0.00      0.00      0.00         0
       A:min       0.33      0.11      0.17         9
           B       0.50      0.25      0.33         4
       B:min       0.12      0.60      0.19         5
           C       0.58      0.47      0.52        15
          C#       0.00      0.00      0.00         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.33      0.20      0.25         5
           D       0.30      0.62      0.41        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       1.00      0.18      0.31        11
       E:min       0.25      0.43      0.32         7
           F       0.40      0.17      0.24        12
          F#       0.00      0.00      0.00         1
      F#:min       0.50      0.25      0.33         4
       F:min       0.00      0.00      0.00         1
           G       0.17      0.33      0.22         9
          G#       0.00      0.00      0.00         0
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.27       125
   macro avg       0.20      0.16      0.15       125
weighted avg       0.34      0.27      0.25       125





7. [cnn_separated_semi_strict_no_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 34
Accuracy: 27.20%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.00      0.00      0.00        14
          A#       0.00      0.00      0.00         0
       A:min       0.00      0.00      0.00         9
           B       0.50      0.25      0.33         4
       B:min       0.00      0.00      0.00         5
           C       0.47      0.47      0.47        15
          C#       0.00      0.00      0.00         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.17      0.62      0.27        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       1.00      0.18      0.31        11
       E:min       0.00      0.00      0.00         7
           F       0.29      0.17      0.21        12
          F#       0.00      0.00      0.00         1
      F#:min       0.00      0.00      0.00         4
       F:min       0.00      0.00      0.00         1
           G       0.10      0.33      0.15         9
          G#       0.00      0.00      0.00         0
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.20       125
   macro avg       0.11      0.09      0.08       125
weighted avg       0.22      0.20      0.16       125





8. [cnn_separated_semi_strict_with_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 37
Accuracy: 29.60%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.00      0.00      0.00        14
          A#       0.00      0.00      0.00         0
       A:min       0.33      0.11      0.17         9
           B       0.50      0.25      0.33         4
       B:min       0.12      0.60      0.19         5
           C       0.58      0.47      0.52        15
          C#       0.00      0.00      0.00         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.33      0.20      0.25         5
           D       0.30      0.62      0.41        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       1.00      0.18      0.31        11
       E:min       0.25      0.43      0.32         7
           F       0.40      0.17      0.24        12
          F#       0.00      0.00      0.00         1
      F#:min       0.50      0.25      0.33         4
       F:min       0.00      0.00      0.00         1
           G       0.17      0.33      0.22         9
          G#       0.00      0.00      0.00         0
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.27       125
   macro avg       0.20      0.16      0.15       125
weighted avg       0.34      0.27      0.25       125
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from essentia.standard import KeyExtractor
import tensorflow as tf

# Suppress warnings and TF logging
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Custom imports
from SongChordRecognizer_Pipeline.DataPreprocessor import DataPreprocessor
from SongChordRecognizer_Training.Models import CRNN_basic_WithStandardScaler
from SongChordRecognizer_Training.Spectrograms import cqt_spectrogram
from SongChordRecognizer_Pipeline.KeyRecognizer import KeyRecognizer

# Load environment variables
load_dotenv()

# Constants
DATASET_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/Datasets/test_vocal_only"
CRRN_MODEL_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/models/original_crnn.keras"
CRRN_SCALER_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/models/original_preprocessor.bin"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FRAMES = 1000
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
    """Predict the musical key of a song file using CRNN model."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        scale = estimate_scale_with_essentia(y, sr) if use_scale else None
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
    """Run prediction analysis on dataset and produce reports/metrics using CRNN."""
    print("🔄 Loading model...")
    model = CRNN_basic_WithStandardScaler()
    model.load(CRRN_MODEL_PATH, CRRN_SCALER_PATH)
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
            "[crnn_separated_semi_strict_with_major_minor]_confusion_matrix.png",
        )
    )

    # Save results
    print("\n💾 Saving results...")
    df.to_csv(
        os.path.join(
            TESTING_REPORT_PATH,
            "[crnn_separated_semi_strict_with_major_minor]_key_recognition_results.csv",
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
1. [crnn_original_strict_no_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 73
Accuracy: 58.40%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.60      0.86      0.71        14
          A#       0.00      0.00      0.00         0
       A:min       0.00      0.00      0.00         9
           B       0.67      0.50      0.57         4
       B:min       0.00      0.00      0.00         5
           C       0.57      0.87      0.68        15
          C#       0.75      0.75      0.75         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.68      0.94      0.79        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       0.67      0.73      0.70        11
       E:min       0.00      0.00      0.00         7
           F       0.67      0.83      0.74        12
          F#       1.00      1.00      1.00         1
      F#:min       0.00      0.00      0.00         4
       F:min       0.00      0.00      0.00         1
           G       0.50      1.00      0.67         9
          G#       0.00      0.00      0.00         0
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.58       125
   macro avg       0.28      0.34      0.30       125
weighted avg       0.43      0.58      0.49       125





2. [crnn_original_strict_with_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 99
Accuracy: 79.20%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.86      0.86      0.86        14
          A#       0.00      0.00      0.00         0
      A#:min       0.00      0.00      0.00         0
       A:min       0.60      0.67      0.63         9
           B       0.67      0.50      0.57         4
       B:min       1.00      0.60      0.75         5
           C       1.00      0.87      0.93        15
          C#       0.67      0.50      0.57         4
      C#:min       1.00      1.00      1.00         2
       C:min       1.00      0.60      0.75         5
           D       0.79      0.94      0.86        16
          D#       0.00      0.00      0.00         1
       D:min       0.50      1.00      0.67         3
           E       0.80      0.73      0.76        11
       E:min       0.75      0.86      0.80         7
           F       1.00      0.75      0.86        12
          F#       1.00      1.00      1.00         1
      F#:min       0.67      1.00      0.80         4
       F:min       0.50      1.00      0.67         1
           G       0.90      1.00      0.95         9
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.79       125
   macro avg       0.62      0.63      0.61       125
weighted avg       0.82      0.79      0.79       125





3. [crnn_original_semi_strict_no_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 81
Accuracy: 64.80%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.60      0.86      0.71        14
          A#       0.00      0.00      0.00         0
       A:min       0.00      0.00      0.00         9
           B       0.67      0.50      0.57         4
       B:min       0.00      0.00      0.00         5
           C       0.57      0.87      0.68        15
          C#       0.75      0.75      0.75         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.68      0.94      0.79        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       0.67      0.73      0.70        11
       E:min       0.00      0.00      0.00         7
           F       0.67      0.83      0.74        12
          F#       1.00      1.00      1.00         1
      F#:min       0.00      0.00      0.00         4
       F:min       0.00      0.00      0.00         1
           G       0.50      1.00      0.67         9
          G#       0.00      0.00      0.00         0
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.58       125
   macro avg       0.28      0.34      0.30       125
weighted avg       0.43      0.58      0.49       125





4. [crnn_original_semi_strict_with_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 105
Accuracy: 84.00%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.86      0.86      0.86        14
          A#       0.00      0.00      0.00         0
      A#:min       0.00      0.00      0.00         0
       A:min       0.60      0.67      0.63         9
           B       0.67      0.50      0.57         4
       B:min       1.00      0.60      0.75         5
           C       1.00      0.87      0.93        15
          C#       0.67      0.50      0.57         4
      C#:min       1.00      1.00      1.00         2
       C:min       1.00      0.60      0.75         5
           D       0.79      0.94      0.86        16
          D#       0.00      0.00      0.00         1
       D:min       0.50      1.00      0.67         3
           E       0.80      0.73      0.76        11
       E:min       0.75      0.86      0.80         7
           F       1.00      0.75      0.86        12
          F#       1.00      1.00      1.00         1
      F#:min       0.67      1.00      0.80         4
       F:min       0.50      1.00      0.67         1
           G       0.90      1.00      0.95         9
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.79       125
   macro avg       0.62      0.63      0.61       125
weighted avg       0.82      0.79      0.79       125





5. [crnn_separated_strict_no_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 48
Accuracy: 38.40%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.67      0.71      0.69        14
          A#       0.00      0.00      0.00         0
       A:min       0.00      0.00      0.00         9
           B       0.00      0.00      0.00         4
       B:min       0.00      0.00      0.00         5
           C       0.33      0.87      0.48        15
          C#       0.50      0.75      0.60         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.57      0.81      0.67        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       0.80      0.36      0.50        11
       E:min       0.00      0.00      0.00         7
           F       0.50      0.25      0.33        12
          F#       0.00      0.00      0.00         1
      F#:min       0.00      0.00      0.00         4
       F:min       0.00      0.00      0.00         1
           G       0.15      0.22      0.18         9
          G#       0.00      0.00      0.00         0
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.38       125
   macro avg       0.16      0.18      0.16       125
weighted avg       0.33      0.38      0.33       125




6. [crnn_separated_strict_with_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 55
Accuracy: 44.00%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.73      0.57      0.64        14
          A#       0.00      0.00      0.00         0
      A#:min       0.00      0.00      0.00         0
       A:min       0.40      0.44      0.42         9
           B       0.00      0.00      0.00         4
       B:min       0.12      0.20      0.15         5
           C       0.45      0.87      0.59        15
          C#       0.25      0.25      0.25         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.80      0.80      0.80         5
           D       0.80      0.75      0.77        16
          D#       0.00      0.00      0.00         1
      D#:min       0.00      0.00      0.00         0
       D:min       0.00      0.00      0.00         3
           E       0.75      0.27      0.40        11
       E:min       0.22      0.29      0.25         7
           F       0.75      0.25      0.38        12
          F#       0.00      0.00      0.00         1
      F#:min       0.50      0.50      0.50         4
       F:min       0.00      0.00      0.00         1
           G       0.50      0.22      0.31         9
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.44       125
   macro avg       0.27      0.24      0.24       125
weighted avg       0.51      0.44      0.44       125





7. [crnn_separated_semi_strict_no_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 59
Accuracy: 47.20%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.67      0.71      0.69        14
          A#       0.00      0.00      0.00         0
       A:min       0.00      0.00      0.00         9
           B       0.00      0.00      0.00         4
       B:min       0.00      0.00      0.00         5
           C       0.33      0.87      0.48        15
          C#       0.50      0.75      0.60         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.00      0.00      0.00         5
           D       0.57      0.81      0.67        16
          D#       0.00      0.00      0.00         1
       D:min       0.00      0.00      0.00         3
           E       0.80      0.36      0.50        11
       E:min       0.00      0.00      0.00         7
           F       0.50      0.25      0.33        12
          F#       0.00      0.00      0.00         1
      F#:min       0.00      0.00      0.00         4
       F:min       0.00      0.00      0.00         1
           G       0.15      0.22      0.18         9
          G#       0.00      0.00      0.00         0
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.38       125
   macro avg       0.16      0.18      0.16       125
weighted avg       0.33      0.38      0.33       125





8. [crnn_separated_semi_strict_with_major_minor]
📊 Analysis Results:
Total files analyzed: 125
Correct predictions: 61
Accuracy: 48.80%

📈 Key Classification Report:
              precision    recall  f1-score   support

           A       0.73      0.57      0.64        14
          A#       0.00      0.00      0.00         0
      A#:min       0.00      0.00      0.00         0
       A:min       0.40      0.44      0.42         9
           B       0.00      0.00      0.00         4
       B:min       0.12      0.20      0.15         5
           C       0.45      0.87      0.59        15
          C#       0.25      0.25      0.25         4
      C#:min       0.00      0.00      0.00         2
       C:min       0.80      0.80      0.80         5
           D       0.80      0.75      0.77        16
          D#       0.00      0.00      0.00         1
      D#:min       0.00      0.00      0.00         0
       D:min       0.00      0.00      0.00         3
           E       0.75      0.27      0.40        11
       E:min       0.22      0.29      0.25         7
           F       0.75      0.25      0.38        12
          F#       0.00      0.00      0.00         1
      F#:min       0.50      0.50      0.50         4
       F:min       0.00      0.00      0.00         1
           G       0.50      0.22      0.31         9
      G#:min       0.00      0.00      0.00         1
       G:min       0.00      0.00      0.00         1

    accuracy                           0.44       125
   macro avg       0.27      0.24      0.24       125
weighted avg       0.51      0.44      0.44       125
"""

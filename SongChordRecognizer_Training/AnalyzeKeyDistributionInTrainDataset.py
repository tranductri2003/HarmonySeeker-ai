import os
import numpy as np
from collections import Counter

# List of chord labels: grouped for better readability
chord_labels = [
    # Silence / Unknown
    "N",
    # Major chords
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
    # Minor chords
    "C:min",
    "C#:min",
    "D:min",
    "D#:min",
    "E:min",
    "F:min",
    "F#:min",
    "G:min",
    "G#:min",
    "A:min",
    "A#:min",
    "B:min",
]


def count_dataset_files():
    print("\n📊 Count of audio files in 'dataset_chords' directory:")
    print("--------------------------------------------------------")

    dataset_dir = "dataset_chords"
    chord_counts = {chord: 0 for chord in chord_labels}
    total_files = 0

    if os.path.exists(dataset_dir):
        for chord in chord_labels:
            chord_dir = os.path.join(dataset_dir, chord)
            if os.path.exists(chord_dir):
                files = [
                    f
                    for f in os.listdir(chord_dir)
                    if os.path.isfile(os.path.join(chord_dir, f))
                    and not f.startswith(".")
                ]
                chord_counts[chord] = len(files)
                total_files += len(files)

    for chord in chord_labels:
        count = chord_counts[chord]
        if total_files > 0:
            percentage = (count / total_files) * 100
            print(f"{chord:8}: {count:3} files ({percentage:5.2f}%)")
        else:
            print(f"{chord:8}: {count:3} files")

    print(f"\n🎯 Total audio files: {total_files}")

    # Highlight problematic chords
    print("\n🚨 Specifically mentioned problematic chords:")
    print("--------------------------------------------------------")
    for chord in ["E:min", "D:min", "F#:min"]:
        count = chord_counts[chord]
        percentage = (count / total_files) * 100 if total_files else 0
        print(f"{chord:8}: {count:3} files ({percentage:5.2f}%)")


if __name__ == "__main__":
    count_dataset_files()

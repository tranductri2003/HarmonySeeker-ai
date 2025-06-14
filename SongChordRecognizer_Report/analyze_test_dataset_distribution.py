import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sys

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Define the chord labels
chord_labels = [
    "C",
    "C:min",
    "C#",
    "C#:min",
    "D",
    "D:min",
    "D#",
    "D#:min",
    "E",
    "E:min",
    "F",
    "F:min",
    "F#",
    "F#:min",
    "G",
    "G:min",
    "G#",
    "G#:min",
    "A",
    "A:min",
    "A#",
    "A#:min",
    "B",
    "B:min",
]


def analyze_test_dataset_distribution(test_dir):
    """
    Analyze the chord distribution in the test dataset by counting audio files in each chord directory.

    Args:
        test_dir (str): Path to the test dataset directory.

    Returns:
        dict: Dictionary mapping chord labels to their file counts.
    """
    print(f"Analyzing test dataset in {test_dir}...")

    chord_counts = {chord: 0 for chord in chord_labels}
    total_files = 0

    try:
        # Count files in each chord directory
        for chord in chord_labels:
            chord_dir = os.path.join(test_dir, chord)
            if os.path.exists(chord_dir):
                files = [
                    f
                    for f in os.listdir(chord_dir)
                    if os.path.isfile(os.path.join(chord_dir, f)) and f.endswith(".wav")
                ]
                chord_counts[chord] = len(files)
                total_files += len(files)

        # Calculate percentages
        chord_percentages = {
            chord: (count / total_files * 100) if total_files > 0 else 0
            for chord, count in chord_counts.items()
        }

        # Print statistics
        print("\n📊 Chord Distribution in Test Dataset:")
        print("-" * 50)
        print(f"{'Chord':<10} {'Count':<10} {'Percentage':<10}")
        print("-" * 50)

        for chord in chord_labels:
            count = chord_counts.get(chord, 0)
            percentage = chord_percentages.get(chord, 0)
            print(f"{chord:<10} {count:<10} {percentage:.2f}%")

        print("-" * 50)
        print(f"Total: {total_files} audio files")

        return chord_counts, chord_percentages

    except Exception as e:
        print(f"Error analyzing test dataset: {str(e)}")
        return None, None


def plot_test_distribution(chord_counts, output_path=None):
    """
    Plot the chord distribution as a bar chart.

    Args:
        chord_counts (dict): Dictionary mapping chord labels to their file counts.
        output_path (str, optional): Path to save the plot. If None, plot is displayed only.
    """
    if not chord_counts:
        print("No chord counts to plot.")
        return

    # Create figure
    plt.figure(figsize=(14, 8))

    # Split into major and minor
    major_chords = [c for c in chord_labels if not ":min" in c]
    minor_chords = [c for c in chord_labels if ":min" in c]

    major_counts = [chord_counts.get(c, 0) for c in major_chords]
    minor_counts = [chord_counts.get(c, 0) for c in minor_chords]

    # Positions for bars
    x1 = np.arange(len(major_chords))
    x2 = np.arange(len(minor_chords))

    # Create subplots
    plt.subplot(2, 1, 1)
    plt.bar(x1, major_counts, color="blue", alpha=0.7)
    plt.xticks(x1, major_chords, rotation=45)
    plt.title("Major Chord Distribution in Test Dataset")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.bar(x2, minor_counts, color="red", alpha=0.7)
    plt.xticks(x2, minor_chords, rotation=45)
    plt.title("Minor Chord Distribution in Test Dataset")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    # Test dataset paths
    test_datasets = [
        "SongChordRecognizer_Training/Datasets/test",
        "SongChordRecognizer_Training/Datasets/test_vocal_only",
    ]

    for test_dir in test_datasets:
        if os.path.exists(test_dir):
            print(f"\n\n===== Analyzing {os.path.basename(test_dir)} dataset =====")
            chord_counts, chord_percentages = analyze_test_dataset_distribution(
                test_dir
            )

            if chord_counts:
                output_path = (
                    f"chord_distribution_{os.path.basename(test_dir)}_test.png"
                )
                plot_test_distribution(chord_counts, output_path)
        else:
            print(f"Test directory {test_dir} not found")

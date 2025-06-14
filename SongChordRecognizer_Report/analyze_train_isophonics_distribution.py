import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sys

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Custom imports
from SongChordRecognizer_Training.Datasets import IsophonicsDataset

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


def analyze_chord_distribution(dataset_path):
    """
    Analyze the chord distribution in a preprocessed dataset.

    Args:
        dataset_path (str): Path to the preprocessed dataset file.

    Returns:
        dict: Dictionary mapping chord labels to their frequency counts.
    """
    print(f"Loading dataset from {dataset_path}...")
    try:
        # Load the preprocessed dataset
        x, y = IsophonicsDataset.load_preprocessed_dataset(dataset_path)

        print(f"Dataset loaded. Shape of x: {x.shape}, Shape of y: {y.shape}")

        # Flatten the labels array to count chord occurrences
        flat_y = y.flatten()

        # Count the occurrences of each chord class
        counts = Counter(flat_y)

        # Convert numeric indices to chord labels
        chord_counts = {
            chord_labels[i]: counts[i] for i in counts if i < len(chord_labels)
        }

        # Calculate total count
        total_count = sum(chord_counts.values())

        # Calculate percentages
        chord_percentages = {
            chord: (count / total_count) * 100 for chord, count in chord_counts.items()
        }

        # Print statistics
        print("\n📊 Chord Distribution in Dataset:")
        print("-" * 50)
        print(f"{'Chord':<10} {'Count':<10} {'Percentage':<10}")
        print("-" * 50)

        for chord in chord_labels:
            count = chord_counts.get(chord, 0)
            percentage = chord_percentages.get(chord, 0)
            print(f"{chord:<10} {count:<10} {percentage:.2f}%")

        print("-" * 50)
        print(f"Total: {total_count} chord frames")

        return chord_counts, chord_percentages

    except Exception as e:
        print(f"Error loading or analyzing dataset: {str(e)}")
        return None, None


def plot_chord_distribution(chord_counts, output_path=None):
    """
    Plot the chord distribution as a bar chart.

    Args:
        chord_counts (dict): Dictionary mapping chord labels to their frequency counts.
        output_path (str, optional): Path to save the plot. If None, plot is displayed only.
    """
    if not chord_counts:
        print("No chord counts to plot.")
        return

    # Prepare data for plotting
    chords = list(chord_counts.keys())
    counts = list(chord_counts.values())

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
    plt.title("Major Chord Distribution")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.bar(x2, minor_counts, color="red", alpha=0.7)
    plt.xticks(x2, minor_chords, rotation=45)
    plt.title("Minor Chord Distribution")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    # List of datasets to analyze
    datasets = [
        "SongChordRecognizer_Training/PreprocessedDatasets/isophonics_cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.ds"
    ]

    for dataset_path in datasets:
        print(f"\n\n===== Analyzing {os.path.basename(dataset_path)} =====")
        chord_counts, chord_percentages = analyze_chord_distribution(dataset_path)

        if chord_counts:
            output_path = (
                f"chord_distribution_{os.path.basename(dataset_path).split('.')[0]}.png"
            )
            plot_chord_distribution(chord_counts, output_path)

#!/usr/bin/env python3
import argparse
import pickle
import lzma
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from collections import Counter
import sys

# Add parent directory to path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from SongChordRecognizer_Training.annotation_maps import chords_map


def reverse_chord_map():
    """Create a mapping from chord indices to chord names."""
    return {v: k for k, v in chords_map.items()}


def load_preprocessed_dataset(ds_file_path):
    """Load a preprocessed dataset from a .ds file."""
    print(f"Loading dataset from {ds_file_path}...")
    with lzma.open(ds_file_path, "rb") as dataset_file:
        dataset = pickle.load(dataset_file)

    print("Dataset loaded successfully.")
    return dataset


def analyze_chord_distribution(targets):
    """Analyze the distribution of chords in the dataset."""
    # Flatten the targets if they are multi-dimensional
    if isinstance(targets, np.ndarray) and targets.ndim > 1:
        targets = targets.flatten()

    # Count occurrences of each chord
    chord_counts = Counter(targets)

    # Get the reverse chord map
    rev_chord_map = reverse_chord_map()

    # Convert chord indices to chord names
    chord_distribution = {
        rev_chord_map.get(idx, f"Unknown-{idx}"): count
        for idx, count in chord_counts.items()
    }

    return chord_distribution


def compare_chord_distributions(datasets, dataset_names, save_path=None):
    """Compare chord distributions between multiple datasets."""
    # Get chord distributions for each dataset
    chord_distributions = []
    for dataset in datasets:
        # Extract targets
        if isinstance(dataset, tuple) and len(dataset) == 2:
            _, targets = dataset
        elif isinstance(dataset, tuple) and len(dataset) == 1:
            _, targets = dataset[0]
        else:
            print(f"Unexpected dataset format: {type(dataset)}")
            continue

        chord_distributions.append(analyze_chord_distribution(targets))

    # Create figure for comparison
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Chord Distribution Comparison", fontsize=18, fontweight="bold")

    # 1. Major vs Minor comparison
    ax1 = plt.subplot(2, 2, 1)

    major_percentages = []
    minor_percentages = []
    n_percentages = []

    for dist in chord_distributions:
        major_chords = {k: v for k, v in dist.items() if ":min" not in k and k != "N"}
        minor_chords = {k: v for k, v in dist.items() if ":min" in k}
        n_chord = dist.get("N", 0)

        total = sum(dist.values())
        major_percentages.append(sum(major_chords.values()) / total * 100)
        minor_percentages.append(sum(minor_chords.values()) / total * 100)
        n_percentages.append(n_chord / total * 100)

    x = np.arange(len(dataset_names))
    width = 0.25

    ax1.bar(x - width, major_percentages, width, label="Major Chords", color="#3498db")
    ax1.bar(x, minor_percentages, width, label="Minor Chords", color="#e74c3c")
    ax1.bar(x + width, n_percentages, width, label="N (No Chord)", color="#95a5a6")

    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Major vs Minor Distribution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names)
    ax1.legend()

    # 2. Top 5 chords comparison
    ax2 = plt.subplot(2, 2, 2)

    # Get top 5 chords across all datasets
    all_chords = {}
    for dist in chord_distributions:
        for chord, count in dist.items():
            if chord in all_chords:
                all_chords[chord] += count
            else:
                all_chords[chord] = count

    top_chords = [
        chord
        for chord, _ in sorted(all_chords.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Prepare data for grouped bar chart
    top_chord_percentages = []
    for dist in chord_distributions:
        total = sum(dist.values())
        top_chord_percentages.append(
            [dist.get(chord, 0) / total * 100 for chord in top_chords]
        )

    # Create grouped bar chart
    x = np.arange(len(top_chords))
    width = 0.8 / len(dataset_names)

    for i, (percentages, name) in enumerate(zip(top_chord_percentages, dataset_names)):
        ax2.bar(x + i * width - 0.4 + width / 2, percentages, width, label=name)

    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Top 5 Chords Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_chords)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.legend()

    # 3. Heatmap comparison - Major chords
    ax3 = plt.subplot(2, 2, 3)

    # Notes in order
    notes = [
        "C",
        "C#/Db",
        "D",
        "D#/Eb",
        "E",
        "F",
        "F#/Gb",
        "G",
        "G#/Ab",
        "A",
        "A#/Bb",
        "B",
    ]

    # Create matrix for major chords
    major_matrix = np.zeros((len(dataset_names), len(notes)))

    for i, dist in enumerate(chord_distributions):
        total = sum(dist.values())

        for chord, count in dist.items():
            if chord == "N" or ":min" in chord:
                continue

            root = chord.split(":")[0] if ":" in chord else chord

            # Handle enharmonic equivalents
            if "/" in root:
                root = root.split("/")[0]

            # Find the note index
            for j, note in enumerate(notes):
                if note.startswith(root) or ("/" + root) in note:
                    major_matrix[i, j] = count / total * 100
                    break

    # Plot heatmap
    sns.heatmap(
        major_matrix,
        ax=ax3,
        cmap="Blues",
        xticklabels=notes,
        yticklabels=dataset_names,
        annot=True,
        fmt=".1f",
        cbar=True,
    )
    ax3.set_title("Major Chord Distribution (%)")

    # 4. Heatmap comparison - Minor chords
    ax4 = plt.subplot(2, 2, 4)

    # Create matrix for minor chords
    minor_matrix = np.zeros((len(dataset_names), len(notes)))

    for i, dist in enumerate(chord_distributions):
        total = sum(dist.values())

        for chord, count in dist.items():
            if chord == "N" or ":min" not in chord:
                continue

            root = chord.split(":")[0]

            # Handle enharmonic equivalents
            if "/" in root:
                root = root.split("/")[0]

            # Find the note index
            for j, note in enumerate(notes):
                if note.startswith(root) or ("/" + root) in note:
                    minor_matrix[i, j] = count / total * 100
                    break

    # Plot heatmap
    sns.heatmap(
        minor_matrix,
        ax=ax4,
        cmap="Reds",
        xticklabels=notes,
        yticklabels=dataset_names,
        annot=True,
        fmt=".1f",
        cbar=True,
    )
    ax4.set_title("Minor Chord Distribution (%)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_path:
        plt.savefig(f"{save_path}_comparison.png", dpi=300, bbox_inches="tight")
        print(f"Comparison visualization saved to {save_path}_comparison.png")
    else:
        plt.show()

    # Create CSV with detailed comparison
    if save_path:
        with open(f"{save_path}_comparison.csv", "w") as f:
            # Header
            f.write("Chord," + ",".join(dataset_names) + "\n")

            # Get all unique chords
            all_unique_chords = set()
            for dist in chord_distributions:
                all_unique_chords.update(dist.keys())

            # Sort chords (N first, then major, then minor)
            sorted_chords = (
                ["N"]
                + sorted([c for c in all_unique_chords if c != "N" and ":min" not in c])
                + sorted([c for c in all_unique_chords if ":min" in c])
            )

            # Write data
            for chord in sorted_chords:
                if chord in all_unique_chords:
                    row = [chord]
                    for dist in chord_distributions:
                        total = sum(dist.values())
                        percentage = dist.get(chord, 0) / total * 100
                        row.append(f"{percentage:.2f}%")
                    f.write(",".join(row) + "\n")

        print(f"Comparison data saved to {save_path}_comparison.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple preprocessed datasets (.ds files)"
    )
    parser.add_argument(
        "--ds_files", required=True, nargs="+", help="Paths to the .ds files"
    )
    parser.add_argument(
        "--names",
        required=True,
        nargs="+",
        help="Names for each dataset (must match number of files)",
    )
    parser.add_argument(
        "--output", default="dataset_comparison", help="Base name for output files"
    )
    args = parser.parse_args()

    if len(args.ds_files) != len(args.names):
        print("Error: Number of dataset files and names must match")
        return

    # Load datasets
    datasets = []
    for ds_file in args.ds_files:
        datasets.append(load_preprocessed_dataset(ds_file))

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compare chord distributions
    compare_chord_distributions(datasets, args.names, save_path=args.output)


if __name__ == "__main__":
    main()

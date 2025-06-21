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


def visualize_spectrograms(data, targets, num_samples=5, save_path=None):
    """Visualize sample spectrograms from the dataset with their chord labels."""
    # Get the reverse chord map
    rev_chord_map = reverse_chord_map()

    # Determine the shape of the data
    if data.ndim == 3:  # (n_samples, n_frames, n_features)
        n_samples, n_frames, n_features = data.shape
        sample_indices = np.random.choice(
            n_samples, min(num_samples, n_samples), replace=False
        )

        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, idx in enumerate(sample_indices):
            # Reshape if needed and visualize
            spectrogram = data[idx]

            # Get the most common chord in this sample
            if targets.ndim == 2:
                sample_targets = targets[idx]
                chord_idx = Counter(sample_targets).most_common(1)[0][0]
            else:
                chord_idx = targets[idx]

            chord_name = rev_chord_map.get(chord_idx, f"Unknown-{chord_idx}")

            # Plot spectrogram
            im = axes[i].imshow(
                spectrogram.T,
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
            )
            axes[i].set_title(f"Sample {idx}, Chord: {chord_name}")
            axes[i].set_ylabel("Feature")
            if i == num_samples - 1:
                axes[i].set_xlabel("Frame")

        # First apply tight layout to position the subplots properly
        plt.tight_layout()

        # Then make room for and add the colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Intensity")

        if save_path:
            plt.savefig(f"{save_path}_spectrograms.png", dpi=300, bbox_inches="tight")
            print(f"Spectrogram visualization saved to {save_path}_spectrograms.png")
        else:
            plt.show()

    elif data.ndim == 2:  # (n_samples, n_features) - flattened or single frame
        n_samples, n_features = data.shape

        # Try to determine if this is a flattened window
        # Assume square-ish spectrogram if it's flattened
        feature_dim = int(np.sqrt(n_features))
        if feature_dim**2 == n_features:
            # It might be a flattened square spectrogram
            reshape_dim = (feature_dim, feature_dim)
        else:
            # Try to find reasonable dimensions
            for i in range(1, min(100, n_features)):
                if n_features % i == 0:
                    reshape_dim = (i, n_features // i)
                    break
            else:
                reshape_dim = (1, n_features)

        sample_indices = np.random.choice(
            n_samples, min(num_samples, n_samples), replace=False
        )

        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, idx in enumerate(sample_indices):
            # Reshape and visualize
            try:
                spectrogram = data[idx].reshape(reshape_dim)

                # Get chord name
                chord_idx = targets[idx]
                chord_name = rev_chord_map.get(chord_idx, f"Unknown-{chord_idx}")

                # Plot spectrogram
                im = axes[i].imshow(
                    spectrogram.T,
                    aspect="auto",
                    origin="lower",
                    interpolation="none",
                    cmap="viridis",
                )
                axes[i].set_title(f"Sample {idx}, Chord: {chord_name}")
                axes[i].set_ylabel("Feature")
                if i == num_samples - 1:
                    axes[i].set_xlabel("Frame")
            except:
                # If reshaping fails, show as a 1D signal
                axes[i].plot(data[idx])
                chord_idx = targets[idx]
                chord_name = rev_chord_map.get(chord_idx, f"Unknown-{chord_idx}")
                axes[i].set_title(
                    f"Sample {idx}, Chord: {chord_name} (1D representation)"
                )

        # First apply tight layout to position the subplots properly
        plt.tight_layout()

        # Then make room for and add the colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Intensity")

        if save_path:
            plt.savefig(f"{save_path}_spectrograms.png", dpi=300, bbox_inches="tight")
            print(f"Spectrogram visualization saved to {save_path}_spectrograms.png")
        else:
            plt.show()


def visualize_chord_distribution(chord_distribution, save_path=None):
    """Visualize the distribution of chords in the dataset."""
    # Separate major and minor chords
    major_chords = {
        k: v for k, v in chord_distribution.items() if ":min" not in k and k != "N"
    }
    minor_chords = {k: v for k, v in chord_distribution.items() if ":min" in k}
    n_chord = chord_distribution.get("N", 0)

    # Calculate percentages
    total_chords = sum(chord_distribution.values())
    major_percent = sum(major_chords.values()) / total_chords * 100
    minor_percent = sum(minor_chords.values()) / total_chords * 100
    n_percent = n_chord / total_chords * 100

    # Create figure with multiple plots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Chord Distribution Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Major vs Minor distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.pie(
        [major_percent, minor_percent, n_percent],
        labels=["Major Chords", "Minor Chords", "N (No Chord)"],
        autopct="%1.2f%%",
        startangle=90,
        colors=["#3498db", "#e74c3c", "#95a5a6"],
    )
    ax1.set_title("Major vs Minor Chord Distribution")

    # Plot 2: Top 10 most common chords
    ax2 = plt.subplot(2, 3, 2)
    top_chords = dict(
        sorted(chord_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    )
    chord_names = list(top_chords.keys())
    chord_counts = list(top_chords.values())

    # Create colors based on chord type
    colors = ["#e74c3c" if ":min" in chord else "#3498db" for chord in chord_names]

    ax2.bar(chord_names, chord_counts, color=colors)
    ax2.set_title("Top 10 Most Common Chords")
    ax2.set_ylabel("Count")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # Plot 3: Major chords distribution
    ax3 = plt.subplot(2, 3, 3)
    sorted_major = dict(sorted(major_chords.items(), key=lambda x: x[1], reverse=True))
    ax3.bar(sorted_major.keys(), sorted_major.values(), color="#3498db")
    ax3.set_title("Major Chords Distribution")
    ax3.set_ylabel("Count")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

    # Plot 4: Minor chords distribution
    ax4 = plt.subplot(2, 3, 4)
    sorted_minor = dict(sorted(minor_chords.items(), key=lambda x: x[1], reverse=True))
    ax4.bar(sorted_minor.keys(), sorted_minor.values(), color="#e74c3c")
    ax4.set_title("Minor Chords Distribution")
    ax4.set_ylabel("Count")
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")

    # Plot 5: Chord frequency heatmap for all 24 chords
    ax5 = plt.subplot(2, 3, (5, 6))

    # Create a 12x2 grid (12 notes, major and minor)
    chord_matrix = np.zeros((2, 12))

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

    # Fill the matrix
    for chord, count in chord_distribution.items():
        if chord == "N":
            continue

        is_minor = ":min" in chord
        root = chord.split(":")[0]

        # Handle enharmonic equivalents
        if "/" in root:
            root = root.split("/")[0]

        # Find the note index
        for i, note in enumerate(notes):
            if note.startswith(root) or ("/" + root) in note:
                chord_matrix[1 if is_minor else 0, i] = count
                break

    # Normalize for better visualization
    chord_matrix = chord_matrix / chord_matrix.max()

    # Create custom colormap from white to blue for major and white to red for minor
    major_cmap = LinearSegmentedColormap.from_list("major", ["#ffffff", "#3498db"])
    minor_cmap = LinearSegmentedColormap.from_list("minor", ["#ffffff", "#e74c3c"])

    # Plot heatmap
    sns.heatmap(
        chord_matrix,
        ax=ax5,
        cmap="YlOrRd",
        xticklabels=notes,
        yticklabels=["Major", "Minor"],
        annot=True,
        fmt=".2f",
        cbar=True,
    )
    ax5.set_title("Normalized Chord Frequency by Root Note and Type")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_path:
        plt.savefig(f"{save_path}_distribution.png", dpi=300, bbox_inches="tight")
        print(f"Chord distribution visualization saved to {save_path}_distribution.png")
    else:
        plt.show()


def visualize_dataset_info(data, targets, save_path=None):
    """Visualize general information about the dataset."""
    # Get dataset shape information
    data_shape = data.shape
    targets_shape = targets.shape

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove axes
    ax.axis("off")

    # Create a table with dataset information
    table_data = [
        ["Dataset Information", "Value"],
        ["Number of samples", str(data_shape[0])],
        ["Data shape", str(data_shape)],
        ["Targets shape", str(targets_shape)],
    ]

    # Add data type information
    table_data.append(["Data type", str(data.dtype)])
    table_data.append(["Targets type", str(targets.dtype)])

    # Add min/max values
    table_data.append(["Data min value", f"{data.min():.4f}"])
    table_data.append(["Data max value", f"{data.max():.4f}"])

    # Add unique chords count
    unique_chords = np.unique(targets)
    table_data.append(["Unique chord labels", str(len(unique_chords))])

    # Create the table
    table = ax.table(
        cellText=table_data, loc="center", cellLoc="left", colWidths=[0.4, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Set title
    plt.title("Dataset Information", fontsize=16, pad=20)

    if save_path:
        plt.savefig(f"{save_path}_info.png", dpi=300, bbox_inches="tight")
        print(f"Dataset information visualization saved to {save_path}_info.png")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed dataset (.ds) files"
    )
    parser.add_argument("--ds_file", required=True, help="Path to the .ds file")
    parser.add_argument(
        "--output", default="visualization", help="Base name for output files"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of spectrogram samples to visualize",
    )
    args = parser.parse_args()

    # Load the dataset
    dataset = load_preprocessed_dataset(args.ds_file)

    # Extract data and targets
    if isinstance(dataset, tuple) and len(dataset) == 2:
        data, targets = dataset
    elif isinstance(dataset, tuple) and len(dataset) == 1:
        # Some datasets might be packed in an extra tuple
        data, targets = dataset[0]
    else:
        print(f"Unexpected dataset format: {type(dataset)}")
        if isinstance(dataset, tuple):
            print(f"Tuple length: {len(dataset)}")
        return

    print(f"Data shape: {data.shape}")
    print(f"Targets shape: {targets.shape}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Analyze chord distribution
    chord_distribution = analyze_chord_distribution(targets)

    # Visualize dataset information
    visualize_dataset_info(data, targets, save_path=args.output)

    # Visualize chord distribution
    visualize_chord_distribution(chord_distribution, save_path=args.output)

    # Visualize sample spectrograms
    visualize_spectrograms(
        data, targets, num_samples=args.samples, save_path=args.output
    )

    # Save chord distribution to CSV
    with open(f"{args.output}_chord_distribution.csv", "w") as f:
        f.write("Chord,Count\n")
        for chord, count in sorted(
            chord_distribution.items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"{chord},{count}\n")

    print(f"Chord distribution saved to {args.output}_chord_distribution.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd

# Add parent directory to path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from SongChordRecognizer_Report.visualize_preprocessed_dataset import (
    load_preprocessed_dataset,
    analyze_chord_distribution,
)


def create_chord_distribution_report(distributions, dataset_names, save_path=None):
    """Create a comprehensive report on chord distributions."""
    # Create a DataFrame to store all distributions
    all_chords = set()
    for dist in distributions:
        all_chords.update(dist.keys())

    # Sort chords (N first, then major, then minor)
    sorted_chords = (
        ["N"]
        + sorted([c for c in all_chords if c != "N" and ":min" not in c])
        + sorted([c for c in all_chords if ":min" in c])
    )

    # Create DataFrame
    df = pd.DataFrame(index=sorted_chords)

    # Add each distribution as a column
    for name, dist in zip(dataset_names, distributions):
        total = sum(dist.values())
        df[name] = pd.Series(
            {chord: dist.get(chord, 0) / total * 100 for chord in sorted_chords}
        )

    # Add major/minor summary
    summary_data = []
    for name, dist in zip(dataset_names, distributions):
        total = sum(dist.values())
        major_chords = {k: v for k, v in dist.items() if ":min" not in k and k != "N"}
        minor_chords = {k: v for k, v in dist.items() if ":min" in k}
        n_chord = dist.get("N", 0)

        major_percent = sum(major_chords.values()) / total * 100
        minor_percent = sum(minor_chords.values()) / total * 100
        n_percent = n_chord / total * 100

        summary_data.append(
            {
                "Dataset": name,
                "Major Chords (%)": major_percent,
                "Minor Chords (%)": minor_percent,
                "N (No Chord) (%)": n_percent,
                "Total Samples": total,
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Create a detailed report
    if save_path:
        # Save the full distribution
        df.to_csv(f"{save_path}_full_distribution.csv")
        print(f"Full chord distribution saved to {save_path}_full_distribution.csv")

        # Save the summary
        summary_df.to_csv(f"{save_path}_summary.csv", index=False)
        print(f"Distribution summary saved to {save_path}_summary.csv")

        # Create a markdown report
        with open(f"{save_path}_report.md", "w") as f:
            f.write("# Chord Distribution Analysis Report\n\n")

            f.write("## Summary\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")

            f.write("## Top 10 Most Common Chords by Dataset\n\n")
            for name, dist in zip(dataset_names, distributions):
                f.write(f"### {name}\n\n")
                top_chords = dict(
                    sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10]
                )
                total = sum(dist.values())
                top_df = pd.DataFrame(
                    {
                        "Chord": top_chords.keys(),
                        "Count": top_chords.values(),
                        "Percentage (%)": [
                            count / total * 100 for count in top_chords.values()
                        ],
                    }
                )
                f.write(top_df.to_markdown(index=False))
                f.write("\n\n")

            f.write("## Observations\n\n")
            f.write("### Major vs Minor Distribution\n\n")

            # Compare major/minor distribution
            if len(dataset_names) > 1:
                major_diff = abs(
                    summary_df["Major Chords (%)"].iloc[0]
                    - summary_df["Major Chords (%)"].iloc[1]
                )
                minor_diff = abs(
                    summary_df["Minor Chords (%)"].iloc[0]
                    - summary_df["Minor Chords (%)"].iloc[1]
                )

                if major_diff > 10 or minor_diff > 10:
                    f.write(
                        f"There is a significant difference in the major/minor chord distribution between datasets. "
                    )
                    f.write(
                        f"The difference in major chords is {major_diff:.2f}% and in minor chords is {minor_diff:.2f}%.\n\n"
                    )

                    # Determine which dataset has more major/minor chords
                    if (
                        summary_df["Major Chords (%)"].iloc[0]
                        > summary_df["Major Chords (%)"].iloc[1]
                    ):
                        f.write(
                            f"The '{dataset_names[0]}' dataset has more major chords ({summary_df['Major Chords (%)'].iloc[0]:.2f}%) "
                        )
                        f.write(
                            f"compared to '{dataset_names[1]}' ({summary_df['Major Chords (%)'].iloc[1]:.2f}%).\n\n"
                        )
                    else:
                        f.write(
                            f"The '{dataset_names[1]}' dataset has more major chords ({summary_df['Major Chords (%)'].iloc[1]:.2f}%) "
                        )
                        f.write(
                            f"compared to '{dataset_names[0]}' ({summary_df['Major Chords (%)'].iloc[0]:.2f}%).\n\n"
                        )

                    if (
                        summary_df["Minor Chords (%)"].iloc[0]
                        > summary_df["Minor Chords (%)"].iloc[1]
                    ):
                        f.write(
                            f"The '{dataset_names[0]}' dataset has more minor chords ({summary_df['Minor Chords (%)'].iloc[0]:.2f}%) "
                        )
                        f.write(
                            f"compared to '{dataset_names[1]}' ({summary_df['Minor Chords (%)'].iloc[1]:.2f}%).\n\n"
                        )
                    else:
                        f.write(
                            f"The '{dataset_names[1]}' dataset has more minor chords ({summary_df['Minor Chords (%)'].iloc[1]:.2f}%) "
                        )
                        f.write(
                            f"compared to '{dataset_names[0]}' ({summary_df['Minor Chords (%)'].iloc[0]:.2f}%).\n\n"
                        )
                else:
                    f.write(
                        "The major/minor chord distribution is relatively similar between datasets.\n\n"
                    )

            f.write("### Top Chords Comparison\n\n")

            # Compare top chords
            if len(dataset_names) > 1:
                top_chords_1 = [
                    chord
                    for chord, _ in sorted(
                        distributions[0].items(), key=lambda x: x[1], reverse=True
                    )[:5]
                ]
                top_chords_2 = [
                    chord
                    for chord, _ in sorted(
                        distributions[1].items(), key=lambda x: x[1], reverse=True
                    )[:5]
                ]

                common_top = set(top_chords_1) & set(top_chords_2)

                if len(common_top) >= 3:
                    f.write(
                        f"The datasets share {len(common_top)} common chords among their top 5: {', '.join(common_top)}.\n\n"
                    )
                else:
                    f.write(
                        f"The datasets have different top chords. Only {len(common_top)} chords appear in both top 5 lists.\n\n"
                    )
                    f.write(
                        f"Top 5 chords in '{dataset_names[0]}': {', '.join(top_chords_1)}\n\n"
                    )
                    f.write(
                        f"Top 5 chords in '{dataset_names[1]}': {', '.join(top_chords_2)}\n\n"
                    )

            f.write("## Implications for Model Training and Evaluation\n\n")

            if len(dataset_names) > 1:
                if major_diff > 20 or minor_diff > 20:
                    f.write(
                        "The significant difference in chord distribution between datasets may impact model performance during evaluation. "
                    )
                    f.write(
                        "The model might be biased towards the distribution it was trained on, potentially resulting in lower accuracy on the test set.\n\n"
                    )

                    f.write("Possible solutions to consider:\n\n")
                    f.write(
                        "1. **Data Augmentation**: Balance the training data to better represent the test distribution\n"
                    )
                    f.write(
                        "2. **Weighted Loss Function**: Apply class weights during training to account for imbalanced classes\n"
                    )
                    f.write(
                        "3. **Stratified Sampling**: Ensure validation splits maintain the same distribution as the training data\n"
                    )
                    f.write(
                        "4. **Separate Evaluation Metrics**: Report performance separately for major and minor chords\n\n"
                    )
                else:
                    f.write(
                        "The chord distributions are reasonably similar between datasets, which should allow for fair model training and evaluation.\n\n"
                    )

            print(f"Analysis report saved to {save_path}_report.md")


def visualize_chord_distribution_comparison(
    distributions, dataset_names, save_path=None
):
    """Create visualizations comparing chord distributions between datasets."""
    # Create figure for comparison
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Chord Distribution Comparison", fontsize=18, fontweight="bold")

    # 1. Major vs Minor comparison
    ax1 = plt.subplot(2, 2, 1)

    major_percentages = []
    minor_percentages = []
    n_percentages = []

    for dist in distributions:
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

    # 2. Top 10 chords comparison
    ax2 = plt.subplot(2, 2, 2)

    # Get top 10 chords across all datasets
    all_chords = {}
    for dist in distributions:
        for chord, count in dist.items():
            if chord in all_chords:
                all_chords[chord] += count
            else:
                all_chords[chord] = count

    top_chords = [
        chord
        for chord, _ in sorted(all_chords.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
    ]

    # Prepare data for grouped bar chart
    top_chord_percentages = []
    for dist in distributions:
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
    ax2.set_title("Top 10 Chords Comparison")
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

    for i, dist in enumerate(distributions):
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

    for i, dist in enumerate(distributions):
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

    # Apply tight_layout first
    plt.tight_layout()
    # Then adjust top margin
    plt.subplots_adjust(top=0.92)

    if save_path:
        plt.savefig(
            f"{save_path}_distribution_comparison.png", dpi=300, bbox_inches="tight"
        )
        print(
            f"Distribution comparison visualization saved to {save_path}_distribution_comparison.png"
        )
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare chord distributions in datasets"
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
        "--output",
        default="chord_distribution_analysis",
        help="Base name for output files",
    )
    args = parser.parse_args()

    if len(args.ds_files) != len(args.names):
        print("Error: Number of dataset files and names must match")
        return

    # Load datasets and analyze chord distributions
    distributions = []
    for ds_file in args.ds_files:
        dataset = load_preprocessed_dataset(ds_file)

        # Extract targets
        if isinstance(dataset, tuple) and len(dataset) == 2:
            _, targets = dataset
        elif isinstance(dataset, tuple) and len(dataset) == 1:
            _, targets = dataset[0]
        else:
            print(f"Unexpected dataset format: {type(dataset)}")
            continue

        distributions.append(analyze_chord_distribution(targets))

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create visualizations
    visualize_chord_distribution_comparison(
        distributions, args.names, save_path=args.output
    )

    # Create comprehensive report
    create_chord_distribution_report(distributions, args.names, save_path=args.output)


if __name__ == "__main__":
    main()

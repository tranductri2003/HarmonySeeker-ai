import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
plt.rcParams.update({"font.size": 12})


# Define the data from analysis
def visualize_chord_distributions():
    # Data from the analysis
    # Training dataset (Isophonics Raw Annotations)
    training_major = 68.31
    training_minor = 24.26
    training_other = 7.43

    # Test dataset
    test_major = 69.60
    test_minor = 30.40
    test_other = 0.0

    # Top 5 chords in training dataset
    training_top_chords = ["A", "G", "D", "E", "C"]
    training_top_percentages = [14.29, 11.54, 11.36, 11.11, 9.26]

    # Top 5 chords in test dataset
    test_top_chords = ["D", "C", "A", "F", "E"]
    test_top_percentages = [12.80, 12.00, 11.20, 9.60, 8.80]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Chord Distribution Analysis - HarmonySeeker AI", fontsize=16, fontweight="bold"
    )

    # 1. Pie charts for major vs minor distribution
    ax1 = plt.subplot(2, 3, 1)
    training_distribution = [training_major, training_minor, training_other]
    training_labels = ["Major", "Minor", "Other"]
    training_colors = ["#3498db", "#e74c3c", "#95a5a6"]
    ax1.pie(
        training_distribution,
        labels=training_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=training_colors,
        explode=(0.05, 0, 0),
    )
    ax1.set_title("Training Dataset Distribution")

    ax2 = plt.subplot(2, 3, 2)
    test_distribution = [test_major, test_minor, test_other]
    test_labels = ["Major", "Minor", "Other"]
    ax2.pie(
        test_distribution,
        labels=test_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=training_colors,
        explode=(0.05, 0, 0),
    )
    ax2.set_title("Test Dataset Distribution")

    # 2. Bar chart comparing major/minor across datasets
    ax3 = plt.subplot(2, 3, 3)
    categories = ["Major", "Minor", "Other"]
    training_values = [training_major, training_minor, training_other]
    test_values = [test_major, test_minor, test_other]

    x = np.arange(len(categories))
    width = 0.35

    ax3.bar(
        x - width / 2,
        training_values,
        width,
        label="Training",
        color="#3498db",
        alpha=0.8,
    )
    ax3.bar(x + width / 2, test_values, width, label="Test", color="#2ecc71", alpha=0.8)

    ax3.set_title("Major vs Minor Distribution Comparison")
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel("Percentage (%)")
    ax3.legend()

    # 3. Bar chart for top 5 chords in training dataset
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(training_top_chords, training_top_percentages, color="#3498db", alpha=0.8)
    ax4.set_title("Top 5 Chords in Training Dataset")
    ax4.set_ylabel("Percentage (%)")
    ax4.set_ylim(0, 16)

    # 4. Bar chart for top 5 chords in test dataset
    ax5 = plt.subplot(2, 3, 5)
    ax5.bar(test_top_chords, test_top_percentages, color="#2ecc71", alpha=0.8)
    ax5.set_title("Top 5 Chords in Test Dataset")
    ax5.set_ylabel("Percentage (%)")
    ax5.set_ylim(0, 16)

    # 5. Radar chart comparing common chords
    ax6 = plt.subplot(2, 3, 6, polar=True)

    # Common chords between datasets
    common_chords = ["A", "C", "D", "E"]

    # Find percentages for common chords
    training_common = []
    test_common = []

    for chord in common_chords:
        if chord in training_top_chords:
            idx = training_top_chords.index(chord)
            training_common.append(training_top_percentages[idx])
        else:
            training_common.append(0)

        if chord in test_top_chords:
            idx = test_top_chords.index(chord)
            test_common.append(test_top_percentages[idx])
        else:
            test_common.append(0)

    # Repeat first value to close the polygon
    common_chords.append(common_chords[0])
    training_common.append(training_common[0])
    test_common.append(test_common[0])

    # Compute angles for each chord
    angles = np.linspace(0, 2 * np.pi, len(common_chords), endpoint=True)

    # Plot radar chart
    ax6.plot(
        angles, training_common, "o-", linewidth=2, label="Training", color="#3498db"
    )
    ax6.fill(angles, training_common, alpha=0.25, color="#3498db")
    ax6.plot(angles, test_common, "o-", linewidth=2, label="Test", color="#2ecc71")
    ax6.fill(angles, test_common, alpha=0.25, color="#2ecc71")

    # Set labels and title
    ax6.set_thetagrids(angles[:-1] * 180 / np.pi, common_chords[:-1])
    ax6.set_title("Common Chord Comparison")
    ax6.legend(loc="upper right")

    # Adjust layout and save figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("chord_distribution_visualization.png", dpi=300, bbox_inches="tight")

    print("Visualization saved as 'chord_distribution_visualization.png'")

    # Create a second figure for detailed chord distribution
    fig2 = plt.figure(figsize=(14, 8))
    fig2.suptitle("Detailed Chord Distribution", fontsize=16, fontweight="bold")

    # Create dataframe for detailed chord analysis
    detailed_chords = ["A", "G", "D", "E", "C", "A:min", "B", "E:min", "F", "D:min"]
    detailed_percentages = [
        14.29,
        11.54,
        11.36,
        11.11,
        9.26,
        4.81,
        4.02,
        3.71,
        3.67,
        3.09,
    ]

    # Create a color map that distinguishes major and minor chords
    chord_types = [
        "Major" if ":min" not in chord else "Minor" for chord in detailed_chords
    ]
    colors = [
        "#3498db" if chord_type == "Major" else "#e74c3c" for chord_type in chord_types
    ]

    # Create horizontal bar chart
    ax = plt.subplot(1, 1, 1)
    bars = ax.barh(
        detailed_chords[::-1], detailed_percentages[::-1], color=colors[::-1]
    )

    # Add chord type as text on the bars
    for i, bar in enumerate(bars):
        chord_type = chord_types[::-1][i]
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            chord_type,
            ha="left",
            va="center",
        )

    ax.set_title("Top 10 Chords in Training Dataset")
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("detailed_chord_distribution.png", dpi=300, bbox_inches="tight")

    print("Detailed visualization saved as 'detailed_chord_distribution.png'")

    # Create a third figure for heatmap comparison
    fig3, ax = plt.subplots(figsize=(10, 8))
    fig3.suptitle("Chord Presence Comparison", fontsize=16, fontweight="bold")

    # Create a combined list of all chords from both datasets
    all_chords = list(set(training_top_chords + test_top_chords))
    all_chords.sort()

    # Create a matrix for the heatmap
    data = []
    for chord in all_chords:
        row = []
        # Training dataset
        if chord in training_top_chords:
            idx = training_top_chords.index(chord)
            row.append(training_top_percentages[idx])
        else:
            row.append(0)

        # Test dataset
        if chord in test_top_chords:
            idx = test_top_chords.index(chord)
            row.append(test_top_percentages[idx])
        else:
            row.append(0)

        data.append(row)

    # Convert to numpy array
    data = np.array(data)

    # Create custom colormap
    colors = ["#ffffff", "#3498db", "#2980b9"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

    # Create heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        xticklabels=["Training", "Test"],
        yticklabels=all_chords,
        ax=ax,
    )

    ax.set_title("Chord Percentage Comparison Between Datasets")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("chord_comparison_heatmap.png", dpi=300, bbox_inches="tight")

    print("Comparison heatmap saved as 'chord_comparison_heatmap.png'")


if __name__ == "__main__":
    visualize_chord_distributions()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input values
models = ["CNN", "CRNN"]
data_types = ["original_audio", "separated_audio"]
tolerance_modes = [
    ("strict", "no_major_minor"),
    ("strict", "with_major_minor"),
    ("semi-strict", "no_major_minor"),
    ("semi-strict", "with_major_minor"),
]

# Step 1: Hard-coded results (default = 0.5)
results = {
    "CNN": {
        "original_audio": {
            ("strict", "no_major_minor"): 0.168,
            ("strict", "with_major_minor"): 0.224,
            ("semi-strict", "no_major_minor"): 0.216,
            ("semi-strict", "with_major_minor"): 0.288,
        },
        "separated_audio": {
            ("strict", "no_major_minor"): 0.2,
            ("strict", "with_major_minor"): 0.272,
            ("semi-strict", "no_major_minor"): 0.272,
            ("semi-strict", "with_major_minor"): 0.296,
        },
    },
    "CRNN": {
        "original_audio": {
            ("strict", "no_major_minor"): 0.584,
            ("strict", "with_major_minor"): 0.792,
            ("semi-strict", "no_major_minor"): 0.648,
            ("semi-strict", "with_major_minor"): 0.84,
        },
        "separated_audio": {
            ("strict", "no_major_minor"): 0.384,
            ("strict", "with_major_minor"): 0.44,
            ("semi-strict", "no_major_minor"): 0.472,
            ("semi-strict", "with_major_minor"): 0.488,
        },
    },
}

# Step 2: Convert to DataFrame
data = []
for model in models:
    for data_type in data_types:
        for tol, mode in tolerance_modes:
            acc = results[model][data_type][(tol, mode)]
            label = f"{model}-{data_type}"
            metric_label = f"{tol}_{mode}"  # Ensures consistency
            data.append([label, metric_label, acc])

results_df = pd.DataFrame(data, columns=["Model-Data", "Metric", "Accuracy"])

# Fix column order
ordered_columns = [
    "strict_no_major_minor",
    "semi-strict_no_major_minor",
    "strict_with_major_minor",
    "semi-strict_with_major_minor",
]


# Step 3: Visualization
def visualize_results(df):
    pivot = df.pivot(index="Model-Data", columns="Metric", values="Accuracy")

    # Ensure consistent column order
    pivot = pivot[ordered_columns]

    # Set style
    sns.set(style="whitegrid", font_scale=1.2)

    # Plot
    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="crest",
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Accuracy"},
        annot_kws={"size": 11},
    )

    # Titles and labels with bold fonts
    plt.title(
        "Chord Recognition Accuracy by Model, Data, and Evaluation Mode",
        fontsize=16,
        weight="bold",
        pad=20,
    )
    plt.xticks(rotation=15, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.xlabel("Tolerance & Mode", fontsize=13, fontweight="bold")
    plt.ylabel("Model & Data Type", fontsize=13, fontweight="bold")

    # Bold colorbar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_weight("bold")

    # Save the figure
    TESTING_REPORT_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_testing_report"
    save_path = f"{TESTING_REPORT_PATH}/chord_recognition_accuracy_heatmap.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to: {save_path}")

    plt.show()


# Call the visualization
visualize_results(results_df)

"""
CNN:
1	cnn_original_strict_no_major_minor
2	cnn_original_strict_with_major_minor
3	cnn_original_semi_strict_no_major_minor
4	cnn_original_semi_strict_with_major_minor
5	cnn_separated_strict_no_major_minor
6	cnn_separated_strict_with_major_minor
7	cnn_separated_semi_strict_no_major_minor
8	cnn_separated_semi_strict_with_major_minor

CRNN:
1. crnn_original_strict_no_major_minor
2. crnn_original_strict_with_major_minor
3. crnn_original_semi_strict_no_major_minor
4. crnn_original_semi_strict_with_major_minor
5. crnn_separated_strict_no_major_minor
6. crnn_separated_strict_with_major_minor
7. crnn_separated_semi_strict_no_major_minor
8. crnn_separated_semi_strict_with_major_minor
"""

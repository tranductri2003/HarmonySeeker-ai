import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.offline import plot


def create_interactive_visualizations():
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

    # Create figure with subplots
    fig = sp.make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "domain"}, {"type": "domain"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "polar"}],
        ],
        subplot_titles=(
            "Training Dataset Distribution",
            "Test Dataset Distribution",
            "Major vs Minor Comparison",
            "Top 5 Chords in Training Dataset",
            "Top 5 Chords in Test Dataset",
            "Common Chord Comparison",
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    # 1. Pie chart for training dataset
    fig.add_trace(
        go.Pie(
            labels=["Major", "Minor", "Other"],
            values=[training_major, training_minor, training_other],
            marker=dict(colors=["#3498db", "#e74c3c", "#95a5a6"]),
            textinfo="label+percent",
            insidetextorientation="radial",
            pull=[0.05, 0, 0],
            hoverinfo="label+percent",
            name="Training Dataset",
        ),
        row=1,
        col=1,
    )

    # 2. Pie chart for test dataset
    fig.add_trace(
        go.Pie(
            labels=["Major", "Minor", "Other"],
            values=[test_major, test_minor, test_other],
            marker=dict(colors=["#3498db", "#e74c3c", "#95a5a6"]),
            textinfo="label+percent",
            insidetextorientation="radial",
            pull=[0.05, 0, 0],
            hoverinfo="label+percent",
            name="Test Dataset",
        ),
        row=1,
        col=2,
    )

    # 3. Bar chart comparing major/minor across datasets
    categories = ["Major", "Minor", "Other"]
    training_values = [training_major, training_minor, training_other]
    test_values = [test_major, test_minor, test_other]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=training_values,
            name="Training",
            marker_color="#3498db",
            opacity=0.8,
            text=training_values,
            texttemplate="%{text:.1f}%",
            hovertemplate="%{x}: %{y:.1f}%<extra>Training</extra>",
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Bar(
            x=categories,
            y=test_values,
            name="Test",
            marker_color="#2ecc71",
            opacity=0.8,
            text=test_values,
            texttemplate="%{text:.1f}%",
            hovertemplate="%{x}: %{y:.1f}%<extra>Test</extra>",
        ),
        row=1,
        col=3,
    )

    # 4. Bar chart for top 5 chords in training dataset
    fig.add_trace(
        go.Bar(
            x=training_top_chords,
            y=training_top_percentages,
            marker_color="#3498db",
            opacity=0.8,
            text=training_top_percentages,
            texttemplate="%{text:.1f}%",
            hovertemplate="%{x}: %{y:.1f}%<extra>Training</extra>",
        ),
        row=2,
        col=1,
    )

    # 5. Bar chart for top 5 chords in test dataset
    fig.add_trace(
        go.Bar(
            x=test_top_chords,
            y=test_top_percentages,
            marker_color="#2ecc71",
            opacity=0.8,
            text=test_top_percentages,
            texttemplate="%{text:.1f}%",
            hovertemplate="%{x}: %{y:.1f}%<extra>Test</extra>",
        ),
        row=2,
        col=2,
    )

    # 6. Radar chart comparing common chords
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

    # Add the first value again to close the loop
    common_chords_closed = common_chords + [common_chords[0]]
    training_common_closed = training_common + [training_common[0]]
    test_common_closed = test_common + [test_common[0]]

    fig.add_trace(
        go.Scatterpolar(
            r=training_common_closed,
            theta=common_chords_closed,
            fill="toself",
            name="Training",
            line_color="#3498db",
            fillcolor="rgba(52, 152, 219, 0.2)",
        ),
        row=2,
        col=3,
    )

    fig.add_trace(
        go.Scatterpolar(
            r=test_common_closed,
            theta=common_chords_closed,
            fill="toself",
            name="Test",
            line_color="#2ecc71",
            fillcolor="rgba(46, 204, 113, 0.2)",
        ),
        row=2,
        col=3,
    )

    # Update layout
    fig.update_layout(
        title_text="Interactive Chord Distribution Analysis - HarmonySeeker AI",
        title_font_size=20,
        barmode="group",
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update y-axes for bar charts
    fig.update_yaxes(title_text="Percentage (%)", range=[0, 16], row=2, col=1)
    fig.update_yaxes(title_text="Percentage (%)", range=[0, 16], row=2, col=2)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=3)

    # Save as HTML
    plot(fig, filename="interactive_chord_distribution.html", auto_open=False)
    print("Interactive visualization saved as 'interactive_chord_distribution.html'")

    # Create a detailed horizontal bar chart
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
    chord_types = [
        "Major" if ":min" not in chord else "Minor" for chord in detailed_chords
    ]

    # Create a DataFrame for the detailed chart
    df = pd.DataFrame(
        {
            "Chord": detailed_chords,
            "Percentage": detailed_percentages,
            "Type": chord_types,
        }
    )

    # Sort by percentage
    df = df.sort_values("Percentage", ascending=True)

    # Create horizontal bar chart with color coding
    fig2 = px.bar(
        df,
        x="Percentage",
        y="Chord",
        color="Type",
        color_discrete_map={"Major": "#3498db", "Minor": "#e74c3c"},
        orientation="h",
        text="Percentage",
        title="Top 10 Chords in Training Dataset (Detailed View)",
        labels={"Percentage": "Percentage (%)", "Chord": "Chord", "Type": "Chord Type"},
        height=600,
        width=900,
    )

    fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    fig2.update_layout(
        xaxis_range=[0, 16],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Save as HTML
    plot(fig2, filename="interactive_detailed_distribution.html", auto_open=False)
    print(
        "Interactive detailed visualization saved as 'interactive_detailed_distribution.html'"
    )

    # Create a heatmap for comparison
    all_chords = list(set(training_top_chords + test_top_chords))
    all_chords.sort()

    # Create a matrix for the heatmap
    heatmap_data = []
    for chord in all_chords:
        row = {}
        row["Chord"] = chord

        # Training dataset
        if chord in training_top_chords:
            idx = training_top_chords.index(chord)
            row["Training"] = training_top_percentages[idx]
        else:
            row["Training"] = 0

        # Test dataset
        if chord in test_top_chords:
            idx = test_top_chords.index(chord)
            row["Test"] = test_top_percentages[idx]
        else:
            row["Test"] = 0

        # Calculate difference
        row["Difference"] = row["Test"] - row["Training"]

        heatmap_data.append(row)

    # Convert to DataFrame
    df_heatmap = pd.DataFrame(heatmap_data)

    # Melt the DataFrame for the heatmap
    df_melted = pd.melt(
        df_heatmap,
        id_vars=["Chord"],
        value_vars=["Training", "Test"],
        var_name="Dataset",
        value_name="Percentage",
    )

    # Create heatmap
    fig3 = px.density_heatmap(
        df_melted,
        x="Dataset",
        y="Chord",
        z="Percentage",
        color_continuous_scale=px.colors.sequential.Blues,
        text_auto=True,
        title="Chord Percentage Comparison Between Datasets",
        labels={"Percentage": "Percentage (%)"},
        height=600,
        width=800,
    )

    # Save as HTML
    plot(fig3, filename="interactive_comparison_heatmap.html", auto_open=False)
    print(
        "Interactive comparison heatmap saved as 'interactive_comparison_heatmap.html'"
    )


if __name__ == "__main__":
    create_interactive_visualizations()

import pandas as pd
from pathlib import Path
import webbrowser

from pyecharts.charts import Line, Tab
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig

# -----------------------------
# Fix rendering
# -----------------------------
CurrentConfig.ONLINE_HOST = "https://assets.pyecharts.org/assets/"

# -----------------------------
# Load data
# -----------------------------
BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment")
df = pd.read_csv(BASE_DIR / "combined_results.csv")

# -----------------------------
# Create overall F
# -----------------------------
df["F"] = (
    df["F1_DIR"] +
    df["F1_DM"] +
    df["F1_INTJ"] +
    df["F1_AS"]
) / 4

# -----------------------------
# Create overall AUC (NEW ⭐)
# -----------------------------
df["AUC"] = (
    df["AUC_DIR"] +
    df["AUC_DM"] +
    df["AUC_INTJ"] +
    df["AUC_AS"]
) / 4

# -----------------------------
# Condition order
# -----------------------------
conditions = [
    "REAL_ONLY",
    "REAL_PLUS_DISTANCE_BALANCED",
    "REAL_PLUS_FAR",
    "REAL_PLUS_MIDDLE",
    "REAL_PLUS_NEAR",
    "REAL_PLUS_RANDOM"
]

# -----------------------------
# Metrics to plot
# -----------------------------
metrics = {
    "F": "F (overall functions) across conditions",
    "accuracy": "Accuracy across conditions",
    "AUC": "AUC (overall) across conditions"
}

# -----------------------------
# Function to create plot
# -----------------------------
def create_plot(metric, title):
    line = Line()
    line.add_xaxis(conditions)

    for split in sorted(df["split"].unique()):
        subset = df[df["split"] == split].copy()

        subset["condition"] = pd.Categorical(
            subset["condition"],
            categories=conditions,
            ordered=True
        )
        subset = subset.sort_values("condition")

        line.add_yaxis(
            series_name=f"Split {split}",
            y_axis=subset[metric].round(4).tolist(),
            is_smooth=True,
            symbol="circle",
            symbol_size=7,
            linestyle_opts=opts.LineStyleOpts(width=3),
            label_opts=opts.LabelOpts(is_show=False),
        )

    line.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        xaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(rotate=30)
        ),
        yaxis_opts=opts.AxisOpts(name=metric),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="5%"),
    )

    return line

# -----------------------------
# Create tab view
# -----------------------------
tab = Tab()

for metric, title in metrics.items():
    chart = create_plot(metric, title)
    tab.add(chart, metric)

# -----------------------------
# Render + open
# -----------------------------
output_path = BASE_DIR / "interactive_split_analysis_final.html"
tab.render(str(output_path))

webbrowser.open(f"file://{output_path}")

print("✅ Final interactive plots created:", output_path)
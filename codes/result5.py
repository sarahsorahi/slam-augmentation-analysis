import pandas as pd
from pathlib import Path
import webbrowser

from pyecharts.charts import Radar
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig

# -----------------------------
# Fix rendering (important)
# -----------------------------
CurrentConfig.ONLINE_HOST = "https://assets.pyecharts.org/assets/"

# -----------------------------
# Path
# -----------------------------
BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/experiment")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(BASE_DIR / "combined_results.csv")

# -----------------------------
# Compute mean per condition
# -----------------------------
summary = df.groupby("condition").mean().reset_index()

# -----------------------------
# CONDITIONS as axes
# -----------------------------
conditions = summary["condition"].tolist()

schema = [
    opts.RadarIndicatorItem(name=cond, max_=1.0)
    for cond in conditions
]

# -----------------------------
# Metrics (LINES)
# -----------------------------
metrics = [
    "macro_f1",
    "accuracy",
    "macro_auc"
]

# -----------------------------
# Color mapping
# -----------------------------
color_map = {
    "macro_f1": "#1f77b4",   # blue
    "accuracy": "#2ca02c",   # green
    "macro_auc": "#ff7f0e",  # orange
}

# -----------------------------
# Create radar
# -----------------------------
radar = Radar()

radar.add_schema(
    schema=schema,
    shape="circle"
)

# -----------------------------
# Add each metric as a line
# -----------------------------
for metric in metrics:
    values = summary[metric].round(4).tolist()

    radar.add(
        series_name=metric,
        data=[values],
        linestyle_opts=opts.LineStyleOpts(
            width=3,
            color=color_map[metric]
        ),
        areastyle_opts=opts.AreaStyleOpts(
            opacity=0.15,
            color=color_map[metric]
        ),
    )

# -----------------------------
# Styling
# -----------------------------
radar.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Spider Plot: Metrics across Conditions"
    ),
    legend_opts=opts.LegendOpts(
        pos_top="5%",
        orient="horizontal"
    ),
)

# Optional: remove labels clutter
radar.set_series_opts(
    label_opts=opts.LabelOpts(is_show=False)
)

# -----------------------------
# Render + open
# -----------------------------
output_path = BASE_DIR / "radar_conditions_as_axes.html"
radar.render(str(output_path))

webbrowser.open(f"file://{output_path}")

print("✅ Spider plot generated and opened")
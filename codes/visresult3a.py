import pandas as pd
from pathlib import Path
import webbrowser

from pyecharts.charts import Radar, Page
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
# Compute mean per condition
# -----------------------------
summary = df.groupby("condition").mean().reset_index()

# -----------------------------
# Radar axes = functions
# -----------------------------
functions = ["DIR", "DM", "INTJ", "AS"]

schema = [
    opts.RadarIndicatorItem(name=f, max_=1.0)
    for f in functions
]

# -----------------------------
# Create page (one radar per condition)
# -----------------------------
page = Page()

# -----------------------------
# Loop over conditions
# -----------------------------
for _, row in summary.iterrows():

    condition = row["condition"]

    radar = Radar()
    radar.add_schema(schema=schema, shape="circle")

    # -----------------------------
    # F1
    # -----------------------------
    radar.add(
        series_name="F1",
        data=[[
            row["F1_DIR"],
            row["F1_DM"],
            row["F1_INTJ"],
            row["F1_AS"]
        ]],
        linestyle_opts=opts.LineStyleOpts(width=2),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.2),
    )

    # -----------------------------
    # AUC
    # -----------------------------
    radar.add(
        series_name="AUC",
        data=[[
            row["AUC_DIR"],
            row["AUC_DM"],
            row["AUC_INTJ"],
            row["AUC_AS"]
        ]],
        linestyle_opts=opts.LineStyleOpts(width=2),
    )

    # -----------------------------
    # Accuracy (flat line)
    # -----------------------------
    radar.add(
        series_name="Accuracy",
        data=[[row["accuracy"]] * 4],
        linestyle_opts=opts.LineStyleOpts(width=2),
    )

    # -----------------------------
    # Styling
    # -----------------------------
    radar.set_global_opts(
        title_opts=opts.TitleOpts(title=f"{condition}"),
        legend_opts=opts.LegendOpts(pos_top="5%"),
    )

    page.add(radar)

# -----------------------------
# Render
# -----------------------------
output_path = BASE_DIR / "radar_metrics_within_condition.html"
page.render(str(output_path))

webbrowser.open(f"file://{output_path}")

print("✅ Radar plots per condition created:", output_path)
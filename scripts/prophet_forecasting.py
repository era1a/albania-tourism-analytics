"""
Albania Tourism - Prophet Forecasting
Diploma Thesis: AI-Assisted Data Analytics Platform for Predictive Tourism Insights
------------------------------------------------------------------------------------
This script performs:
1. Time-series forecasting per tourism type using Facebook Prophet
2. 12-month ahead predictions
3. Trend and seasonality decomposition
4. Forecast visualization with confidence intervals

HOW TO RUN:
    pip install prophet
    Make sure master_worldwide.csv exists in Desktop/thesis/data/
    Run this script from PyCharm
    Charts will be saved in: Desktop/thesis/output/forecasting/
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# Suppress Prophet's verbose output
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from prophet import Prophet

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "master_worldwide.csv")
OUT_DIR   = os.path.join(BASE_DIR, "output", "forecasting")
os.makedirs(OUT_DIR, exist_ok=True)

TYPE_COLORS = {
    "Cultural & Heritage":  "#8B4513",
    "Sun & Sea":            "#1E90FF",
    "Nature & Adventure":   "#228B22",
    "Rural & Agrotourism":  "#DAA520",
    "Urban & Business":     "#6A0DAD",
}

FORECAST_MONTHS = 12

# ── Load Data ──────────────────────────────────────────────────────────────────

print("\nAlbania Tourism - Prophet Forecasting")
print("=" * 60)

df      = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_dest = df[df["is_anchor"] == False].copy()

# Normalize each keyword to its own max
df_dest["interest"] = df_dest.groupby("keyword")["interest"].transform(
    lambda x: (x / x.max()) * 100 if x.max() > 0 else x
)

# Monthly average per tourism type
type_monthly = (
    df_dest
    .groupby(["date", "tourism_type"])["interest"]
    .mean()
    .reset_index()
)
type_monthly.columns = ["ds", "tourism_type", "y"]
type_monthly["ds"] = pd.to_datetime(type_monthly["ds"])

print(f"Data loaded: {len(type_monthly)} monthly observations")
print(f"Tourism types: {type_monthly['tourism_type'].nunique()}")
print(f"Forecasting {FORECAST_MONTHS} months ahead\n")

# ── 1. INDIVIDUAL FORECASTS PER TOURISM TYPE ───────────────────────────────────

print("Chart 1: Individual forecasts per tourism type...")

all_forecasts = {}

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
axes = axes.flatten()

for i, (ttype, color) in enumerate(TYPE_COLORS.items()):
    ax = axes[i]

    # Prepare data for Prophet
    data = type_monthly[type_monthly["tourism_type"] == ttype][["ds", "y"]].copy()
    data = data.sort_values("ds").reset_index(drop=True)

    # Train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.fit(data)

    # Make future dataframe
    future   = model.make_future_dataframe(periods=FORECAST_MONTHS, freq="MS")
    forecast = model.predict(future)
    all_forecasts[ttype] = forecast

    # Split historical and future
    historical = forecast[forecast["ds"] <= data["ds"].max()]
    future_fc  = forecast[forecast["ds"] > data["ds"].max()]

    # Plot
    ax.plot(data["ds"], data["y"], color=color, linewidth=2,
            label="Historical", alpha=0.9)
    ax.plot(forecast["ds"], forecast["yhat"], color=color,
            linewidth=2, linestyle="--", label="Forecast")
    ax.fill_between(forecast["ds"],
                    forecast["yhat_lower"],
                    forecast["yhat_upper"],
                    alpha=0.15, color=color, label="Confidence Interval")

    # Shade forecast period
    ax.axvspan(data["ds"].max(), forecast["ds"].max(),
               alpha=0.05, color="gray")
    ax.axvline(x=data["ds"].max(), color="gray",
               linestyle=":", linewidth=1.5, label="Forecast Start")

    # Peak prediction
    peak_month = future_fc.loc[future_fc["yhat"].idxmax(), "ds"]
    peak_val   = future_fc["yhat"].max()
    ax.annotate(
        f"Peak: {peak_month.strftime('%b %Y')}\n({peak_val:.1f})",
        xy=(peak_month, peak_val),
        xytext=(10, 10), textcoords="offset points",
        fontsize=8, color=color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=color, alpha=0.8)
    )

    ax.set_title(ttype, fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("")
    ax.set_ylabel("Interest (0-100)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, None)

    print(f"   {ttype}: Peak forecast {peak_month.strftime('%B %Y')} at {peak_val:.1f}")

axes[5].set_visible(False)

fig.suptitle(f"Tourism Demand Forecast by Type\nAlbania — Next {FORECAST_MONTHS} Months",
             fontsize=16, fontweight="bold")
plt.tight_layout()
path1 = os.path.join(OUT_DIR, "1_individual_forecasts.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path1}")

# ── 2. COMBINED FORECAST COMPARISON ───────────────────────────────────────────

print("\nChart 2: Combined forecast comparison...")

fig, ax = plt.subplots(figsize=(16, 8))

for ttype, color in TYPE_COLORS.items():
    forecast = all_forecasts[ttype]
    data     = type_monthly[type_monthly["tourism_type"] == ttype]
    cutoff   = data["ds"].max()
    future_fc = forecast[forecast["ds"] > cutoff]

    ax.plot(future_fc["ds"], future_fc["yhat"],
            color=color, linewidth=2.5,
            marker="o", markersize=4, label=ttype)
    ax.fill_between(future_fc["ds"],
                    future_fc["yhat_lower"],
                    future_fc["yhat_upper"],
                    alpha=0.1, color=color)

ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Predicted Interest (0-100)", fontsize=11)
ax.set_title(f"12-Month Tourism Demand Forecast by Type\nAlbania — All Tourism Types Compared",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path2 = os.path.join(OUT_DIR, "2_combined_forecast.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path2}")

# ── 3. SEASONALITY DECOMPOSITION ──────────────────────────────────────────────

print("Chart 3: Seasonality decomposition...")

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(14, 7))

for ttype, color in TYPE_COLORS.items():
    forecast = all_forecasts[ttype]
    seasonal = forecast[["ds", "yearly"]].copy()
    seasonal["month"] = seasonal["ds"].dt.month
    monthly_seasonal  = seasonal.groupby("month")["yearly"].mean()
    ax.plot(range(1, 13), monthly_seasonal.values,
            color=color, linewidth=2.5,
            marker="o", markersize=6, label=ttype)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_NAMES, fontsize=10)
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Seasonal Component", fontsize=11)
ax.set_title("Prophet Seasonal Decomposition by Tourism Type\nAlbania — Monthly Patterns",
             fontsize=14, fontweight="bold")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path3 = os.path.join(OUT_DIR, "3_seasonality_decomposition.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path3}")

# ── 4. FORECAST SUMMARY TABLE ──────────────────────────────────────────────────

print("Chart 4: Forecast summary table...")

summary_rows = []
for ttype, color in TYPE_COLORS.items():
    forecast  = all_forecasts[ttype]
    data      = type_monthly[type_monthly["tourism_type"] == ttype]
    cutoff    = data["ds"].max()
    future_fc = forecast[forecast["ds"] > cutoff].copy()

    peak_month   = future_fc.loc[future_fc["yhat"].idxmax(), "ds"]
    peak_val     = future_fc["yhat"].max()
    low_month    = future_fc.loc[future_fc["yhat"].idxmin(), "ds"]
    low_val      = future_fc["yhat"].min()
    avg_forecast = future_fc["yhat"].mean()

    # Historical average for comparison
    hist_avg = data["y"].mean()
    change   = ((avg_forecast - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0

    summary_rows.append({
        "Tourism Type":        ttype,
        "Peak Month":          peak_month.strftime("%B %Y"),
        "Peak Value":          f"{peak_val:.1f}",
        "Low Month":           low_month.strftime("%B %Y"),
        "Avg Forecast":        f"{avg_forecast:.1f}",
        "vs Historical Avg":   f"{change:+.1f}%",
    })

summary_df = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for j in range(len(summary_df.columns)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Style rows alternating
for i in range(1, len(summary_df) + 1):
    color = list(TYPE_COLORS.values())[i-1]
    for j in range(len(summary_df.columns)):
        table[i, j].set_facecolor(color + "22")

ax.set_title(f"12-Month Forecast Summary by Tourism Type\nAlbania",
             fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
path4 = os.path.join(OUT_DIR, "4_forecast_summary_table.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path4}")

# ── KEY FINDINGS ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("KEY FINDINGS - PROPHET FORECASTING")
print("=" * 60)
print(f"\n12-Month Forecast Summary:")
print(summary_df.to_string(index=False))

print("\n" + "=" * 60)
print("Prophet forecasting complete!")
print(f"Charts saved in: {OUT_DIR}")
print("=" * 60)
print("\nNext step: Build the Streamlit dashboard!")
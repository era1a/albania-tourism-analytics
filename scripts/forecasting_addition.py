"""
Albania Tourism - Forecasting Additions
Accuracy Metrics + Promotion Windows
------------------------------------------------------------------------------------
Run this AFTER prophet_forecasting.py
Adds:
1. Model accuracy metrics (MAE, RMSE, MAPE) per tourism type
2. Promotion window recommendations chart

HOW TO RUN:
    Run this script from PyCharm after prophet_forecasting.py
    Charts saved in: Desktop/thesis/output/forecasting/
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

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

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── Load Data ──────────────────────────────────────────────────────────────────

print("\nAlbania Tourism - Forecasting Additions")
print("=" * 60)

df      = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_dest = df[df["is_anchor"] == False].copy()

df_dest["interest"] = df_dest.groupby("keyword")["interest"].transform(
    lambda x: (x / x.max()) * 100 if x.max() > 0 else x
)

type_monthly = (
    df_dest
    .groupby(["date", "tourism_type"])["interest"]
    .mean()
    .reset_index()
)
type_monthly.columns = ["ds", "tourism_type", "y"]
type_monthly["ds"]   = pd.to_datetime(type_monthly["ds"])

print(f"Data loaded: {len(type_monthly)} monthly observations\n")

# ── Train models and collect metrics ──────────────────────────────────────────

all_metrics   = {}
all_forecasts = {}
all_data      = {}

print("Training models and computing accuracy metrics...")

for ttype in TYPE_COLORS.keys():
    data = type_monthly[type_monthly["tourism_type"] == ttype][["ds", "y"]].copy()
    data = data.sort_values("ds").reset_index(drop=True)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.fit(data)

    # Cross validation
    try:
        cv_results = cross_validation(
            model,
            initial="730 days",
            period="180 days",
            horizon="180 days",
            disable_tqdm=True,
        )
        metrics = performance_metrics(cv_results)

        mae  = metrics["mae"].mean()
        rmse = metrics["rmse"].mean()
        mape = metrics["mape"].mean() * 100

        all_metrics[ttype] = {
            "MAE":  round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
        }
        print(f"   {ttype}: MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.1f}%")

    except Exception as e:
        print(f"   {ttype}: Cross-validation skipped ({e})")
        all_metrics[ttype] = {"MAE": None, "RMSE": None, "MAPE": None}

    # Forecast for promotion windows
    future   = model.make_future_dataframe(periods=12, freq="MS")
    forecast = model.predict(future)
    all_forecasts[ttype] = forecast
    all_data[ttype]      = data

# ── 5. ACCURACY METRICS CHART ──────────────────────────────────────────────────

print("\nChart 5: Model accuracy metrics...")

metrics_df = pd.DataFrame(all_metrics).T.reset_index()
metrics_df.columns = ["tourism_type", "MAE", "RMSE", "MAPE"]
metrics_df = metrics_df.dropna()

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

metrics_list = ["MAE", "RMSE", "MAPE"]
metric_labels = ["MAE\n(Lower is better)", "RMSE\n(Lower is better)", "MAPE %\n(Lower is better)"]

for i, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
    ax     = axes[i]
    types  = metrics_df["tourism_type"].tolist()
    values = metrics_df[metric].tolist()
    colors = [TYPE_COLORS.get(t, "#888") for t in types]
    x      = np.arange(len(types))

    bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor="white", width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" & ", "\n& ") for t in types], fontsize=8)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Prophet Model Accuracy Metrics by Tourism Type\nAlbania — Cross-Validation Results",
             fontsize=14, fontweight="bold")
plt.tight_layout()
path5 = os.path.join(OUT_DIR, "5_accuracy_metrics.png")
plt.savefig(path5, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path5}")

# ── 6. PROMOTION WINDOWS CHART ────────────────────────────────────────────────

print("Chart 6: Promotion windows...")

fig, ax = plt.subplots(figsize=(16, 8))

y_pos     = 0
y_ticks   = []
y_labels  = []
bar_height = 0.6

for ttype, color in TYPE_COLORS.items():
    forecast = all_forecasts[ttype]
    data     = all_data[ttype]
    cutoff   = data["ds"].max()
    future_fc = forecast[forecast["ds"] > cutoff].copy()
    future_fc["month"] = future_fc["ds"].dt.month

    # Monthly average forecast
    monthly_avg = future_fc.groupby("month")["yhat"].mean()

    # Peak month
    peak_month = monthly_avg.idxmax()

    # Promotion window = 2 months before peak
    promo_months = [(peak_month - 2) % 12 or 12, (peak_month - 1) % 12 or 12]
    peak_action  = [peak_month]
    post_months  = [(peak_month + 1) % 12 or 12, (peak_month + 2) % 12 or 12]

    # Draw bars for each month
    for month in range(1, 13):
        if month in promo_months:
            bar_color = "#F39C12"
            alpha     = 0.9
        elif month in peak_action:
            bar_color = "#E74C3C"
            alpha     = 0.9
        elif month in post_months:
            bar_color = "#3498DB"
            alpha     = 0.6
        else:
            bar_color = "#ECF0F1"
            alpha     = 0.8

        ax.barh(y_pos, 1, left=month - 1, height=bar_height,
                color=bar_color, alpha=alpha, edgecolor="white", linewidth=0.5)

        # Month label inside bar
        if bar_color != "#ECF0F1":
            ax.text(month - 0.5, y_pos, MONTH_NAMES[month-1],
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")
        else:
            ax.text(month - 0.5, y_pos, MONTH_NAMES[month-1],
                    ha="center", va="center", fontsize=7, color="#AAA")

    y_ticks.append(y_pos)
    y_labels.append(ttype)
    y_pos += 1

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=11)
ax.set_xticks([])
ax.set_xlim(0, 12)
ax.set_title("Recommended Promotion Windows by Tourism Type\nAlbania 2026",
             fontsize=14, fontweight="bold")

# Legend
legend_elements = [
    mpatches.Patch(color="#F39C12", label="Promotion Window (2 months before peak)"),
    mpatches.Patch(color="#E74C3C", label="Peak Month (maximum demand)"),
    mpatches.Patch(color="#3498DB", label="Post-Peak (follow-up campaigns)"),
    mpatches.Patch(color="#ECF0F1", label="Off-Season"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
          framealpha=0.9, title="Calendar Key")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

plt.tight_layout()
path6 = os.path.join(OUT_DIR, "6_promotion_windows.png")
plt.savefig(path6, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path6}")

# ── KEY FINDINGS ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("KEY FINDINGS - ACCURACY & PROMOTION WINDOWS")
print("=" * 60)

print("\nModel Accuracy (Cross-Validation):")
for ttype, m in all_metrics.items():
    if m["MAPE"] is not None:
        print(f"   {ttype}: MAPE={m['MAPE']:.1f}% | MAE={m['MAE']:.2f} | RMSE={m['RMSE']:.2f}")

print("\nPromotion Windows:")
for ttype, color in TYPE_COLORS.items():
    forecast  = all_forecasts[ttype]
    data      = all_data[ttype]
    cutoff    = data["ds"].max()
    future_fc = forecast[forecast["ds"] > cutoff].copy()
    future_fc["month"] = future_fc["ds"].dt.month
    monthly_avg  = future_fc.groupby("month")["yhat"].mean()
    peak_month   = monthly_avg.idxmax()
    promo1       = (peak_month - 2) % 12 or 12
    promo2       = (peak_month - 1) % 12 or 12
    print(f"   {ttype}:")
    print(f"      Peak: {MONTH_NAMES[peak_month-1]} | Promote in: {MONTH_NAMES[promo1-1]} & {MONTH_NAMES[promo2-1]}")

print("\n" + "=" * 60)
print("Forecasting additions complete!")
print(f"Charts saved in: {OUT_DIR}")
print("=" * 60)
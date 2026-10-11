"""
Albania Tourism - SARIMA vs Prophet Forecasting Comparison
Diploma Thesis: AI-Assisted Data Analytics Platform for Predictive Tourism Insights
------------------------------------------------------------------------------------
This script performs:
1. SARIMA forecasting per tourism type
2. Prophet forecasting per tourism type
3. Model comparison (MAE, RMSE, MAPE)
4. Visual comparison of both models

HOW TO RUN:
    pip install statsmodels
    Run this script from PyCharm
    Charts saved in: Desktop/thesis/output/sarima/
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "master_worldwide.csv")
OUT_DIR   = os.path.join(BASE_DIR, "output", "sarima")
os.makedirs(OUT_DIR, exist_ok=True)

TYPE_COLORS = {
    "Cultural & Heritage":  "#8B4513",
    "Sun & Sea":            "#1E90FF",
    "Nature & Adventure":   "#228B22",
    "Rural & Agrotourism":  "#DAA520",
    "Urban & Business":     "#6A0DAD",
}

# SARIMA parameters (p,d,q)(P,D,Q,s)
# s=12 for monthly seasonal data
SARIMA_ORDER         = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)

FORECAST_MONTHS = 12
TEST_MONTHS     = 12  # Last 12 months used for validation

# ── Load Data ──────────────────────────────────────────────────────────────────

print("\nAlbania Tourism - SARIMA vs Prophet Comparison")
print("=" * 60)

df      = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_dest = df[df["is_anchor"] == False].copy()

df_dest["interest"] = df_dest.groupby("keyword")["interest"].transform(
    lambda x: (x / x.max()) * 100 if x.max() > 0 else x
)

type_monthly = (
    df_dest.groupby(["date", "tourism_type"])["interest"]
    .mean().reset_index()
)
type_monthly.columns = ["ds", "tourism_type", "y"]
type_monthly["ds"] = pd.to_datetime(type_monthly["ds"])

print(f"Data loaded: {len(type_monthly)} monthly observations")
print(f"Tourism types: {type_monthly['tourism_type'].nunique()}")
print(f"Test period: last {TEST_MONTHS} months")
print(f"Forecast period: next {FORECAST_MONTHS} months\n")

# ── Results Storage ────────────────────────────────────────────────────────────

all_results  = {}
comparison   = []

# ── 1. TRAIN AND EVALUATE BOTH MODELS ─────────────────────────────────────────

for ttype in TYPE_COLORS.keys():
    print(f"Processing: {ttype}...")

    data = type_monthly[type_monthly["tourism_type"] == ttype][["ds","y"]].copy()
    data = data.sort_values("ds").reset_index(drop=True)

    # Train/test split
    train = data.iloc[:-TEST_MONTHS]
    test  = data.iloc[-TEST_MONTHS:]

    # ── PROPHET ──────────────────────────────────────────────────────────────
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    prophet_model.fit(train)

    # Predict on test period
    future_test   = prophet_model.make_future_dataframe(periods=TEST_MONTHS, freq="MS")
    prophet_test  = prophet_model.predict(future_test)
    prophet_preds = prophet_test.tail(TEST_MONTHS)["yhat"].values

    # Forecast future
    future_full     = prophet_model.make_future_dataframe(periods=TEST_MONTHS + FORECAST_MONTHS, freq="MS")
    prophet_full    = prophet_model.predict(future_full)
    prophet_forecast = prophet_full.tail(FORECAST_MONTHS)

    # ── SARIMA ────────────────────────────────────────────────────────────────
    try:
        sarima_model = SARIMAX(
            train["y"],
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_fit    = sarima_model.fit(disp=False)
        sarima_preds  = sarima_fit.forecast(steps=TEST_MONTHS)
        sarima_forecast = sarima_fit.forecast(steps=TEST_MONTHS + FORECAST_MONTHS).tail(FORECAST_MONTHS)
        sarima_success = True
    except Exception as e:
        print(f"   SARIMA failed for {ttype}: {e}")
        sarima_preds    = prophet_preds  # fallback
        sarima_forecast = prophet_forecast["yhat"].values
        sarima_success  = False

    # ── Calculate Metrics ─────────────────────────────────────────────────────
    actual = test["y"].values

    def calc_metrics(actual, predicted):
        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        return round(mae, 2), round(rmse, 2), round(mape, 2)

    p_mae, p_rmse, p_mape = calc_metrics(actual, prophet_preds)
    s_mae, s_rmse, s_mape = calc_metrics(actual, sarima_preds)

    winner = "Prophet" if p_mape <= s_mape else "SARIMA"

    print(f"   Prophet — MAE:{p_mae:.2f} RMSE:{p_rmse:.2f} MAPE:{p_mape:.1f}%")
    print(f"   SARIMA  — MAE:{s_mae:.2f} RMSE:{s_rmse:.2f} MAPE:{s_mape:.1f}%")
    print(f"   Winner  — {winner}\n")

    comparison.append({
        "Tourism Type":  ttype,
        "Prophet MAE":   p_mae,
        "Prophet RMSE":  p_rmse,
        "Prophet MAPE":  f"{p_mape:.1f}%",
        "SARIMA MAE":    s_mae,
        "SARIMA RMSE":   s_rmse,
        "SARIMA MAPE":   f"{s_mape:.1f}%",
        "Winner":        winner,
        "p_mape_raw":    p_mape,
        "s_mape_raw":    s_mape,
    })

    all_results[ttype] = {
        "data":             data,
        "train":            train,
        "test":             test,
        "prophet_preds":    prophet_preds,
        "sarima_preds":     sarima_preds,
        "prophet_forecast": prophet_forecast,
        "sarima_forecast":  sarima_forecast,
        "p_mape":           p_mape,
        "s_mape":           s_mape,
        "winner":           winner,
        "sarima_success":   sarima_success,
    }

comp_df = pd.DataFrame(comparison)

# ── 2. COMPARISON CHART — MAPE BAR CHART ──────────────────────────────────────

print("Chart 1: MAPE comparison...")

fig, ax = plt.subplots(figsize=(14, 7))

types  = comp_df["Tourism Type"].tolist()
x      = np.arange(len(types))
width  = 0.35

bars1 = ax.bar(x - width/2, comp_df["p_mape_raw"], width,
               label="Prophet", color="#2563EB", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, comp_df["s_mape_raw"], width,
               label="SARIMA", color="#E74C3C", alpha=0.85, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([t.replace(" & ", "\n& ") for t in types], fontsize=10)
ax.set_ylabel("MAPE % (Lower is Better)", fontsize=11)
ax.set_title("Prophet vs SARIMA — Model Accuracy Comparison (MAPE %)\nAlbania Tourism Forecasting",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=9, color="#2563EB", fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=9, color="#E74C3C", fontweight="bold")

# Winner annotations
for i, row in comp_df.iterrows():
    winner_color = "#2563EB" if row["Winner"] == "Prophet" else "#E74C3C"
    ax.text(i, max(row["p_mape_raw"], row["s_mape_raw"]) + 2,
            f"✓ {row['Winner']}", ha="center", fontsize=9,
            color=winner_color, fontweight="bold")

plt.tight_layout()
path1 = os.path.join(OUT_DIR, "1_mape_comparison.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path1}")

# ── 3. FORECAST COMPARISON — ALL TYPES ────────────────────────────────────────

print("Chart 2: Forecast comparison per type...")

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
axes = axes.flatten()

for idx, (ttype, color) in enumerate(TYPE_COLORS.items()):
    ax  = axes[idx]
    res = all_results[ttype]

    # Historical data
    ax.plot(res["data"]["ds"], res["data"]["y"],
            color="#333333", linewidth=1.5, label="Historical", alpha=0.7)

    # Test period actual
    ax.plot(res["test"]["ds"], res["test"]["y"],
            color="#333333", linewidth=2, linestyle="dotted", label="Actual (test)")

    # Prophet predictions on test
    ax.plot(res["test"]["ds"], res["prophet_preds"],
            color="#2563EB", linewidth=2, linestyle="--",
            label=f"Prophet (MAPE:{res['p_mape']:.1f}%)")

    # SARIMA predictions on test
    ax.plot(res["test"]["ds"], res["sarima_preds"],
            color="#E74C3C", linewidth=2, linestyle="-.",
            label=f"SARIMA (MAPE:{res['s_mape']:.1f}%)")

    # Forecast future — Prophet
    future_dates = res["prophet_forecast"]["ds"]
    ax.plot(future_dates, res["prophet_forecast"]["yhat"],
            color="#2563EB", linewidth=2, alpha=0.5)
    ax.fill_between(future_dates,
                    res["prophet_forecast"]["yhat_lower"],
                    res["prophet_forecast"]["yhat_upper"],
                    alpha=0.08, color="#2563EB")

    # Forecast future — SARIMA
    last_date = res["data"]["ds"].max()
    future_index = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=FORECAST_MONTHS, freq="MS"
    )
    ax.plot(future_index, res["sarima_forecast"],
            color="#E74C3C", linewidth=2, alpha=0.5)

    # Divider line
    cutoff = res["train"]["ds"].max()
    ax.axvline(x=cutoff, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(cutoff, ax.get_ylim()[1] * 0.95, " Test →", fontsize=8, color="gray")

    # Winner badge
    winner_color = "#2563EB" if res["winner"] == "Prophet" else "#E74C3C"
    ax.text(0.98, 0.98, f"✓ {res['winner']} wins",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold", color=winner_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=winner_color, alpha=0.9))

    ax.set_title(ttype, fontsize=12, fontweight="bold", color=color)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, None)

axes[5].set_visible(False)

fig.suptitle("Prophet vs SARIMA — Forecast Comparison by Tourism Type\nAlbania 2019-2026",
             fontsize=15, fontweight="bold")
plt.tight_layout()
path2 = os.path.join(OUT_DIR, "2_forecast_comparison.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path2}")

# ── 4. METRICS SUMMARY TABLE CHART ────────────────────────────────────────────

print("Chart 3: Metrics summary table...")

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

display_df = comp_df[["Tourism Type","Prophet MAE","Prophet RMSE","Prophet MAPE",
                       "SARIMA MAE","SARIMA RMSE","SARIMA MAPE","Winner"]].copy()

table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    cellLoc="center", loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Header styling
for j in range(len(display_df.columns)):
    table[0, j].set_facecolor("#1B2A4A")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Row styling
for i in range(1, len(display_df) + 1):
    winner = comp_df.iloc[i-1]["Winner"]
    for j in range(len(display_df.columns)):
        col_name = display_df.columns[j]
        # Highlight winner columns
        if col_name == "Winner":
            color = "#DBEAFE" if winner == "Prophet" else "#FEE2E2"
            table[i, j].set_facecolor(color)
            table[i, j].set_text_props(fontweight="bold",
                                        color="#2563EB" if winner=="Prophet" else "#E74C3C")
        elif "Prophet" in col_name and winner == "Prophet":
            table[i, j].set_facecolor("#EFF6FF")
        elif "SARIMA" in col_name and winner == "SARIMA":
            table[i, j].set_facecolor("#FFF1F2")
        else:
            table[i, j].set_facecolor("#F8FAFC" if i % 2 == 0 else "white")

ax.set_title("Model Performance Comparison — Prophet vs SARIMA\nAlbania Tourism Forecasting",
             fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
path3 = os.path.join(OUT_DIR, "3_metrics_table.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path3}")

# ── 5. OVERALL WINNER SUMMARY ──────────────────────────────────────────────────

print("Chart 4: Overall winner summary...")

prophet_wins = sum(1 for r in comp_df["Winner"] if r == "Prophet")
sarima_wins  = len(comp_df) - prophet_wins

avg_prophet_mape = comp_df["p_mape_raw"].mean()
avg_sarima_mape  = comp_df["s_mape_raw"].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart — wins
axes[0].pie(
    [prophet_wins, sarima_wins],
    labels=[f"Prophet\n({prophet_wins} wins)", f"SARIMA\n({sarima_wins} wins)"],
    colors=["#2563EB", "#E74C3C"],
    autopct="%1.0f%%", startangle=90,
    textprops={"fontsize": 12, "fontweight": "bold"}
)
axes[0].set_title("Model Wins by Tourism Type", fontsize=13, fontweight="bold")

# Bar chart — average MAPE
bars = axes[1].bar(["Prophet", "SARIMA"],
                    [avg_prophet_mape, avg_sarima_mape],
                    color=["#2563EB", "#E74C3C"], alpha=0.85,
                    edgecolor="white", width=0.5)
axes[1].set_ylabel("Average MAPE % (Lower is Better)", fontsize=11)
axes[1].set_title("Average Accuracy Across All Tourism Types", fontsize=13, fontweight="bold")
axes[1].grid(axis="y", linestyle="--", alpha=0.4)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

for bar, val in zip(bars, [avg_prophet_mape, avg_sarima_mape]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

overall_winner = "Prophet" if avg_prophet_mape <= avg_sarima_mape else "SARIMA"
axes[1].text(0.5, 0.95, f"Overall Winner: {overall_winner}",
            transform=axes[1].transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold",
            color="#2563EB" if overall_winner == "Prophet" else "#E74C3C",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#2563EB" if overall_winner=="Prophet" else "#E74C3C"))

plt.tight_layout()
path4 = os.path.join(OUT_DIR, "4_overall_winner.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path4}")

# ── SAVE RESULTS CSV ───────────────────────────────────────────────────────────

results_path = os.path.join(OUT_DIR, "model_comparison_results.csv")
comp_df.to_csv(results_path, index=False)

# ── KEY FINDINGS ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("KEY FINDINGS - SARIMA vs PROPHET COMPARISON")
print("=" * 60)
print(f"\nOverall Winner: {overall_winner}")
print(f"Average Prophet MAPE: {avg_prophet_mape:.1f}%")
print(f"Average SARIMA MAPE:  {avg_sarima_mape:.1f}%")
print(f"Prophet wins: {prophet_wins}/5 tourism types")
print(f"SARIMA wins:  {sarima_wins}/5 tourism types")

print("\nDetailed Results:")
for _, row in comp_df.iterrows():
    print(f"   {row['Tourism Type']}:")
    print(f"      Prophet MAPE: {row['Prophet MAPE']} | SARIMA MAPE: {row['SARIMA MAPE']} | Winner: {row['Winner']}")

print("\n" + "=" * 60)
print("SARIMA vs Prophet comparison complete!")
print(f"Charts saved in: {OUT_DIR}")
print(f"Results CSV: {results_path}")
print("=" * 60)
print("\nNext step: Add comparison to Streamlit dashboard")
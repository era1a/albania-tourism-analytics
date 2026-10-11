"""
Albania Tourism - INSTAT Validation Analysis
Diploma Thesis: AI-Assisted Data Analytics Platform for Predictive Tourism Insights
------------------------------------------------------------------------------------
This script performs:
1. Google Trends vs INSTAT arrivals validation (correlation proof)

HOW TO RUN:
    Make sure master_worldwide.csv, tab-1.xlsx and tab-4.xlsx exist
    Copy tab-1.xlsx and tab-4.xlsx to Desktop/thesis/data/instat/
    Run this script from PyCharm
    Charts will be saved in: Desktop/thesis/output/validation/
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "master_worldwide.csv")
INSTAT_DIR  = os.path.join(BASE_DIR, "data", "instat")
TAB1_PATH   = os.path.join(INSTAT_DIR, "tab-1.xlsx")
TAB4_PATH   = os.path.join(INSTAT_DIR, "tab-4.xlsx")
OUT_DIR     = os.path.join(BASE_DIR, "output", "validation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Google Trends Data ────────────────────────────────────────────────────

print("\nAlbania Tourism - INSTAT Validation Analysis")
print("=" * 60)

df      = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_dest = df[df["is_anchor"] == False].copy()

# Monthly average interest across all destinations
trends_monthly = (
    df_dest
    .groupby("date")["interest"]
    .mean()
    .reset_index()
)
trends_monthly.columns = ["date", "avg_interest"]
trends_monthly["date"] = pd.to_datetime(trends_monthly["date"])

print(f"Google Trends data loaded: {len(trends_monthly)} months")

# ── Load INSTAT tab-4 — Monthly Arrivals ──────────────────────────────────────

tab4_raw = pd.read_excel(TAB4_PATH, header=None)

# Row 4 = headers (months), Row 5 = total arrivals
headers = tab4_raw.iloc[4, 1:].tolist()
total_row = tab4_raw.iloc[5, 1:].tolist()

instat_monthly = pd.DataFrame({
    "date":     pd.to_datetime(headers, format="%Y-%m", errors="coerce"),
    "arrivals": pd.to_numeric(total_row, errors="coerce")
}).dropna()

print(f"INSTAT monthly data loaded: {len(instat_monthly)} months")
print(f"   Date range: {instat_monthly['date'].min().date()} to {instat_monthly['date'].max().date()}")

# ── Load INSTAT tab-1 — Arrivals by Region ────────────────────────────────────

tab1_raw = pd.read_excel(TAB1_PATH, header=None)

# Extract region data rows 6-17, columns 0 (description) and 1-6 (2020-2025)
regions = {
    "Africa":              tab1_raw.iloc[6,  1:7].tolist(),
    "Americas":            tab1_raw.iloc[7,  1:7].tolist(),
    "East Asia & Pacific": tab1_raw.iloc[8,  1:7].tolist(),
    "Middle East":         tab1_raw.iloc[9,  1:7].tolist(),
    "South Asia":          tab1_raw.iloc[10, 1:7].tolist(),
    "Europe":              tab1_raw.iloc[11, 1:7].tolist(),
}

years = [2020, 2021, 2022, 2023, 2024, 2025]

# Europe sub-regions
europe_sub = {
    "Central/Eastern Europe": tab1_raw.iloc[12, 1:7].tolist(),
    "Northern Europe":        tab1_raw.iloc[13, 1:7].tolist(),
    "Southern Europe":        tab1_raw.iloc[14, 1:7].tolist(),
    "Western Europe":         tab1_raw.iloc[15, 1:7].tolist(),
    "East/Mediterranean":     tab1_raw.iloc[16, 1:7].tolist(),
}

# Convert to numeric
for key in regions:
    regions[key] = [pd.to_numeric(v, errors="coerce") for v in regions[key]]
for key in europe_sub:
    europe_sub[key] = [pd.to_numeric(v, errors="coerce") for v in europe_sub[key]]

print(f"INSTAT regional data loaded: {len(regions)} world regions")

# ── 1. GOOGLE TRENDS vs INSTAT VALIDATION ─────────────────────────────────────

print("\nChart 1: Google Trends vs INSTAT validation...")

# Merge on date — only keep overlapping months
merged = pd.merge(
    trends_monthly,
    instat_monthly,
    on="date",
    how="inner"
)

# Normalize both to 0-100 for visual comparison
merged["trends_norm"]   = (merged["avg_interest"] / merged["avg_interest"].max()) * 100
merged["arrivals_norm"] = (merged["arrivals"] / merged["arrivals"].max()) * 100

# Calculate correlation
corr = merged["trends_norm"].corr(merged["arrivals_norm"])
print(f"   Correlation between Google Trends and INSTAT arrivals: {corr:.3f}")

fig, ax1 = plt.subplots(figsize=(16, 7))

# Google Trends line
color1 = "#E74C3C"
ax1.plot(merged["date"], merged["trends_norm"],
         color=color1, linewidth=2.5, label="Google Trends Interest (Normalized)",
         marker="o", markersize=3)
ax1.fill_between(merged["date"], merged["trends_norm"],
                 alpha=0.15, color=color1)
ax1.set_xlabel("Date", fontsize=11)
ax1.set_ylabel("Google Trends Interest (Normalized 0-100)", color=color1, fontsize=11)
ax1.tick_params(axis="y", labelcolor=color1)

# INSTAT arrivals line on second axis
ax2 = ax1.twinx()
color2 = "#2980B9"
ax2.plot(merged["date"], merged["arrivals_norm"],
         color=color2, linewidth=2.5, label="INSTAT Actual Arrivals (Normalized)",
         linestyle="--", marker="s", markersize=3)
ax2.fill_between(merged["date"], merged["arrivals_norm"],
                 alpha=0.1, color=color2)
ax2.set_ylabel("Actual Tourist Arrivals (Normalized 0-100)", color=color2, fontsize=11)
ax2.tick_params(axis="y", labelcolor=color2)

# COVID marker
ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"),
            alpha=0.1, color="gray", label="COVID-19 Period")

# Correlation annotation
ax1.text(0.02, 0.95,
         f"Pearson Correlation: r = {corr:.3f}",
         transform=ax1.transAxes,
         fontsize=13, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="#2C3E50", alpha=0.9))

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

ax1.set_title("Google Trends Search Interest vs Actual Tourist Arrivals\nAlbania — INSTAT Validation",
              fontsize=15, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
path1 = os.path.join(OUT_DIR, "1_trends_vs_instat_validation.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path1}")


# ── KEY FINDINGS ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("KEY FINDINGS - INSTAT VALIDATION")
print("=" * 60)

print(f"\nValidation Result:")
print(f"   Pearson correlation (Google Trends vs INSTAT): {corr:.3f}")
if corr >= 0.7:
    print(f"   Result: STRONG validation - Google Trends is a reliable predictor")
elif corr >= 0.5:
    print(f"   Result: MODERATE validation - Google Trends is a reasonable predictor")
else:
    print(f"   Result: WEAK validation - results should be interpreted with caution")

total_2025 = tab1_raw.iloc[5, 6]
europe_2025 = tab1_raw.iloc[11, 6]
europe_pct = pd.to_numeric(europe_2025, errors="coerce") / pd.to_numeric(total_2025, errors="coerce") * 100
southern_2025 = tab1_raw.iloc[14, 6]
southern_pct = pd.to_numeric(southern_2025, errors="coerce") / pd.to_numeric(total_2025, errors="coerce") * 100

print(f"\nSource Market 2025:")
print(f"   Total arrivals       : {pd.to_numeric(total_2025, errors='coerce'):,.0f}")
print(f"   European tourists    : {europe_pct:.1f}% of total")
print(f"   Southern European    : {southern_pct:.1f}% of total")

print(f"\nGrowth Story:")
total_2020 = pd.to_numeric(tab1_raw.iloc[5, 1], errors="coerce")
total_2025_num = pd.to_numeric(total_2025, errors="coerce")
growth = (total_2025_num - total_2020) / total_2020 * 100
print(f"   Arrivals 2020: {total_2020:,.0f}")
print(f"   Arrivals 2025: {total_2025_num:,.0f}")
print(f"   Total growth : {growth:.1f}% since 2020")

print("\n" + "=" * 60)
print("INSTAT Validation complete!")
print(f"Charts saved in: {OUT_DIR}")
print("=" * 60)
print("\nNext step: Run weather_data.py")
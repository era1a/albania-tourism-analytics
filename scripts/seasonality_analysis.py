"""
Albania Tourism - Seasonality & Trend Analysis
Diploma Thesis: AI-Assisted Data Analytics Platform for Predictive Tourism Insights
------------------------------------------------------------------------------------
This script performs:
1. Seasonal pattern analysis per tourism type
2. Overall trend analysis (growing/declining)
3. Monthly average interest heatmap
4. COVID impact visualization

HOW TO RUN:
    Make sure master_worldwide.csv exists in Desktop/thesis/data/
    Run this script from PyCharm
    Charts will be saved in: Desktop/thesis/output/seasonality/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "master_worldwide.csv")
OUT_DIR    = os.path.join(BASE_DIR, "output", "seasonality")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors per Tourism Type ────────────────────────────────────────────────────

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

print("\n🇦🇱 Albania Tourism - Seasonality & Trend Analysis")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Remove anchor keyword (Tirana) from destination analysis
df_dest = df[df["is_anchor"] == False].copy()
# Normalize each keyword to its own max so all are on 0-100 scale
df_dest["interest"] = df_dest.groupby("keyword")["interest"].transform(
    lambda x: (x / x.max()) * 100 if x.max() > 0 else x
)
# Add time columns
df_dest["year"]  = df_dest["date"].dt.year
df_dest["month"] = df_dest["date"].dt.month

print(f"✅ Data loaded: {len(df_dest):,} rows")
print(f"   Date range : {df_dest['date'].min().date()} → {df_dest['date'].max().date()}")
print(f"   Types      : {df_dest['tourism_type'].unique().tolist()}\n")

# ── 1. SEASONAL PATTERN — Average Interest by Month per Tourism Type ───────────

print("📊 Chart 1: Seasonal patterns per tourism type...")

monthly_avg = (
    df_dest
    .groupby(["tourism_type", "month"])["interest"]
    .mean()
    .reset_index()
)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for i, (ttype, color) in enumerate(TYPE_COLORS.items()):
    ax = axes[i]
    data = monthly_avg[monthly_avg["tourism_type"] == ttype]

    ax.fill_between(data["month"], data["interest"], alpha=0.3, color=color)
    ax.plot(data["month"], data["interest"], color=color, linewidth=2.5, marker="o", markersize=5)

    ax.set_title(ttype, fontsize=13, fontweight="bold", color=color)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax.set_ylabel("Avg Interest (0-100)", fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Mark peak month
    peak_month = data.loc[data["interest"].idxmax(), "month"]
    peak_val   = data["interest"].max()
    ax.annotate(
        f"Peak: {MONTH_NAMES[peak_month-1]}",
        xy=(peak_month, peak_val),
        xytext=(peak_month + 0.5, peak_val + 0.3),
        fontsize=8, color=color, fontweight="bold"
    )

# Hide the 6th empty subplot
axes[5].set_visible(False)

fig.suptitle("Seasonal Tourism Interest Patterns by Type\nAlbania 2019–2025 (Worldwide Search Interest)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
path1 = os.path.join(OUT_DIR, "1_seasonal_patterns.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {path1}")

# ── 2. TREND ANALYSIS — Annual Average Interest per Tourism Type ───────────────

print("📊 Chart 2: Annual trend per tourism type...")

annual_avg = (
    df_dest
    .groupby(["tourism_type", "year"])["interest"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 7))

for ttype, color in TYPE_COLORS.items():
    data = annual_avg[annual_avg["tourism_type"] == ttype]
    ax.plot(data["year"], data["interest"], color=color, linewidth=2.5,
            marker="o", markersize=7, label=ttype)

# COVID marker
ax.axvspan(2020, 2021, alpha=0.1, color="red", label="COVID-19 Period")
ax.axvline(x=2020, color="red", linestyle="--", alpha=0.5, linewidth=1)

ax.set_title("Annual Tourism Interest Trends by Type\nAlbania 2019–2025",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Average Interest (0-100)", fontsize=11)
ax.set_xticks(range(2019, 2026))
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path2 = os.path.join(OUT_DIR, "2_annual_trends.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {path2}")

# ── 3. HEATMAP — Monthly Interest by Tourism Type ─────────────────────────────

print("📊 Chart 3: Heatmap of monthly interest...")

heatmap_data = (
    df_dest
    .groupby(["tourism_type", "month"])["interest"]
    .mean()
    .unstack(level="month")
)
heatmap_data.columns = MONTH_NAMES

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Average Interest (0-100)"}
)
ax.set_title("Monthly Interest Heatmap by Tourism Type\nAlbania 2019–2025 (Worldwide)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Tourism Type", fontsize=11)
plt.tight_layout()
path3 = os.path.join(OUT_DIR, "3_monthly_heatmap.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {path3}")

# ── 4. COVID IMPACT — Interest Before/During/After COVID ──────────────────────

print("📊 Chart 4: COVID impact analysis...")

def covid_period(year):
    if year <= 2019:
        return "Pre-COVID\n(2019)"
    elif year == 2020:
        return "COVID\n(2020)"
    elif year == 2021:
        return "Recovery\n(2021)"
    else:
        return "Post-COVID\n(2022-2025)"

df_dest["period"] = df_dest["year"].apply(covid_period)

period_order = ["Pre-COVID\n(2019)", "COVID\n(2020)", "Recovery\n(2021)", "Post-COVID\n(2022-2025)"]

covid_avg = (
    df_dest
    .groupby(["tourism_type", "period"])["interest"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 7))

x = range(len(period_order))
width = 0.15
offsets = [-0.3, -0.15, 0, 0.15, 0.3]

for i, (ttype, color) in enumerate(TYPE_COLORS.items()):
    data = covid_avg[covid_avg["tourism_type"] == ttype]
    data = data.set_index("period").reindex(period_order).reset_index()
    bars = ax.bar(
        [xi + offsets[i] for xi in x],
        data["interest"],
        width=width,
        color=color,
        label=ttype,
        alpha=0.85
    )

ax.set_title("COVID-19 Impact on Tourism Interest by Type\nAlbania",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Period", fontsize=11)
ax.set_ylabel("Average Interest (0-100)", fontsize=11)
ax.set_xticks(list(x))
ax.set_xticklabels(period_order, fontsize=10)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path4 = os.path.join(OUT_DIR, "4_covid_impact.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {path4}")

# ── 5. PRINT KEY FINDINGS ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("🔍 KEY FINDINGS")
print("=" * 60)

for ttype in TYPE_COLORS.keys():
    data = monthly_avg[monthly_avg["tourism_type"] == ttype]
    peak_month  = data.loc[data["interest"].idxmax(), "month"]
    low_month   = data.loc[data["interest"].idxmin(), "month"]
    peak_val    = data["interest"].max()
    low_val     = data["interest"].min()

    annual = annual_avg[annual_avg["tourism_type"] == ttype]
    trend  = "📈 GROWING" if annual.iloc[-1]["interest"] > annual.iloc[0]["interest"] else "📉 DECLINING"

    print(f"\n{ttype}:")
    print(f"   Peak month  : {MONTH_NAMES[peak_month-1]} (avg interest: {peak_val:.1f})")
    print(f"   Lowest month: {MONTH_NAMES[low_month-1]} (avg interest: {low_val:.1f})")
    print(f"   Trend       : {trend}")

print("\n" + "=" * 60)
print("✅ Seasonality analysis complete!")
print(f"📁 Charts saved in: {OUT_DIR}")
print("=" * 60)
print("\nNext step: Run correlation_analysis.py")
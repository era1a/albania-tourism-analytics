"""
Albania Tourism - Correlation Analysis
Diploma Thesis: AI-Assisted Data Analytics Platform for Predictive Tourism Insights
------------------------------------------------------------------------------------
This script performs:
1. Correlation matrix between tourism types
2. Destination pair correlation (which places are searched together)
3. Tourism corridor identification
4. Top correlated destination pairs

HOW TO RUN:
    Make sure master_worldwide.csv exists in Desktop/thesis/data/
    Run this script from PyCharm
    Charts will be saved in: Desktop/thesis/output/correlation/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "master_worldwide.csv")
OUT_DIR   = os.path.join(BASE_DIR, "output", "correlation")
os.makedirs(OUT_DIR, exist_ok=True)

TYPE_COLORS = {
    "Cultural & Heritage":  "#8B4513",
    "Sun & Sea":            "#1E90FF",
    "Nature & Adventure":   "#228B22",
    "Rural & Agrotourism":  "#DAA520",
    "Urban & Business":     "#6A0DAD",
}

# ── Load Data ──────────────────────────────────────────────────────────────────

print("\n🇦🇱 Albania Tourism - Correlation Analysis")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_dest = df[df["is_anchor"] == False].copy()

# Normalize each keyword to its own max
df_dest["interest"] = df_dest.groupby("keyword")["interest"].transform(
    lambda x: (x / x.max()) * 100 if x.max() > 0 else x
)

print(f"✅ Data loaded: {len(df_dest):,} rows")
print(f"   Unique keywords: {df_dest['keyword'].nunique()}")

# ── 1. CORRELATION MATRIX — Tourism Type Level ─────────────────────────────────

print("\n📊 Chart 1: Correlation matrix between tourism types...")

# Pivot: rows = date, columns = tourism_type, values = mean interest
type_pivot = df_dest.groupby(["date", "tourism_type"])["interest"].mean().unstack("tourism_type")
type_corr  = type_pivot.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(type_corr, dtype=bool), k=1)

sns.heatmap(
    type_corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax,
    square=True,
    cbar_kws={"label": "Correlation Coefficient"}
)

ax.set_title("Correlation Matrix Between Tourism Types\nAlbania 2019–2025 (Worldwide)",
             fontsize=14, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
path1 = os.path.join(OUT_DIR, "1_type_correlation_matrix.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {path1}")

# ── 2. TOP CORRELATED DESTINATION PAIRS ────────────────────────────────────────

print("📊 Chart 2: Top correlated destination pairs...")

# Pivot: rows = date, columns = keyword
dest_pivot = df_dest.pivot_table(index="date", columns="keyword", values="interest", aggfunc="mean")
dest_corr  = dest_pivot.corr()

# Extract upper triangle pairs
pairs = []
keywords = dest_corr.columns.tolist()
for i in range(len(keywords)):
    for j in range(i+1, len(keywords)):
        corr_val = dest_corr.iloc[i, j]
        if not np.isnan(corr_val):
            # Get tourism type for each keyword
            type_i = df_dest[df_dest["keyword"] == keywords[i]]["tourism_type"].iloc[0] if len(df_dest[df_dest["keyword"] == keywords[i]]) > 0 else "Unknown"
            type_j = df_dest[df_dest["keyword"] == keywords[j]]["tourism_type"].iloc[0] if len(df_dest[df_dest["keyword"] == keywords[j]]) > 0 else "Unknown"
            pairs.append({
                "dest_1":   keywords[i],
                "dest_2":   keywords[j],
                "type_1":   type_i,
                "type_2":   type_j,
                "correlation": round(corr_val, 3),
                "same_type": type_i == type_j
            })

pairs_df = pd.DataFrame(pairs).sort_values("correlation", ascending=False)

# Top 20 positive correlations
top20 = pairs_df.head(20)

fig, ax = plt.subplots(figsize=(14, 9))

colors = [TYPE_COLORS.get(row["type_1"], "#888888") for _, row in top20.iterrows()]
bars   = ax.barh(
    [f"{row['dest_1']} ↔ {row['dest_2']}" for _, row in top20.iterrows()],
    top20["correlation"],
    color=colors,
    alpha=0.85,
    edgecolor="white"
)

ax.set_xlabel("Correlation Coefficient", fontsize=11)
ax.set_title("Top 20 Most Correlated Destination Pairs\nAlbania 2019–2025 (Worldwide)",
             fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.1)
ax.axvline(x=0.7, color="red", linestyle="--", alpha=0.5, label="Strong correlation (0.7)")
ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="Moderate correlation (0.5)")
ax.legend(fontsize=9)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add value labels
for bar, val in zip(bars, top20["correlation"]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=8)

plt.tight_layout()
path2 = os.path.join(OUT_DIR, "2_top_correlated_pairs.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {path2}")

# ── 3. TOURISM CORRIDORS ───────────────────────────────────────────────────────

print("📊 Chart 3: Tourism corridors...")

# Strong correlations = potential corridors (correlation > 0.6)
corridors = pairs_df[pairs_df["correlation"] >= 0.6].copy()

# Group corridors by type combination
corridors["corridor_type"] = corridors.apply(
    lambda r: f"{r['type_1']} ↔ {r['type_2']}" if r['type_1'] != r['type_2']
    else r['type_1'], axis=1
)

corridor_counts = corridors["corridor_type"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(12, 6))
colors_corr = [TYPE_COLORS.get(ct.split(" ↔ ")[0].strip(), "#888888")
               for ct in corridor_counts.index]

bars = ax.bar(
    range(len(corridor_counts)),
    corridor_counts.values,
    color=colors_corr,
    alpha=0.85,
    edgecolor="white",
    width=0.6
)

ax.set_xticks(range(len(corridor_counts)))
ax.set_xticklabels(corridor_counts.index, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Number of Strong Destination Pairs", fontsize=11)
ax.set_title("Tourism Corridors by Type\n(Destination pairs with correlation ≥ 0.6)",
             fontsize=14, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(int(bar.get_height())), ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
path3 = os.path.join(OUT_DIR, "3_tourism_corridors.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {path3}")

# ── 4. CROSS TYPE CORRELATION ──────────────────────────────────────────────────

print("📊 Chart 4: Cross-type correlation heatmap...")

# Average correlation between each pair of tourism types
cross_type = pairs_df.groupby(["type_1", "type_2"])["correlation"].mean().reset_index()

# Build symmetric matrix
types = list(TYPE_COLORS.keys())
cross_matrix = pd.DataFrame(index=types, columns=types, dtype=float)

for _, row in cross_type.iterrows():
    if row["type_1"] in types and row["type_2"] in types:
        cross_matrix.loc[row["type_1"], row["type_2"]] = row["correlation"]
        cross_matrix.loc[row["type_2"], row["type_1"]] = row["correlation"]

# Fill diagonal with 1
for t in types:
    cross_matrix.loc[t, t] = 1.0

cross_matrix = cross_matrix.astype(float)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cross_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax,
    square=True,
    cbar_kws={"label": "Average Correlation"}
)

ax.set_title("Average Cross-Type Correlation\nAlbania Tourism 2019–2025",
             fontsize=14, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
path4 = os.path.join(OUT_DIR, "4_cross_type_correlation.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: {path4}")

# ── PRINT KEY FINDINGS ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("🔍 KEY FINDINGS — CORRELATION ANALYSIS")
print("=" * 60)

print("\n📌 Tourism Type Correlations:")
for i, row in type_corr.iterrows():
    for j, val in row.items():
        if i < j and abs(val) > 0.5:
            strength = "Strong" if abs(val) > 0.7 else "Moderate"
            direction = "positive" if val > 0 else "negative"
            print(f"   {i} ↔ {j}: {val:.2f} ({strength} {direction})")

print("\n📌 Top 5 Most Correlated Destination Pairs:")
for _, row in pairs_df.head(5).iterrows():
    print(f"   {row['dest_1']} ↔ {row['dest_2']}: {row['correlation']:.2f} ({row['type_1']} / {row['type_2']})")

print(f"\n📌 Strong corridors found (correlation ≥ 0.6): {len(corridors)} destination pairs")

print("\n" + "=" * 60)
print("✅ Correlation analysis complete!")
print(f"📁 Charts saved in: {OUT_DIR}")
print("=" * 60)
print("\nNext step: Run overcrowding_detection.py")
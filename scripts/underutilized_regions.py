"""
Albania Tourism - Underutilized Regions Analysis (Simplified)
Diploma Thesis: AI-Assisted Data Analytics Platform for Predictive Tourism Insights
------------------------------------------------------------------------------------
This script performs:
1. Destination tier classification (Established / Emerging / Undiscovered)
2. Priority investment targets (Emerging + High Potential)
3. Potential vs Actual gap by tourism type
4. Top recommendations for institutions

HOW TO RUN:
    Make sure master_albania.csv and potential_scores.xlsx exist in Desktop/thesis/data/
    Run this script from PyCharm
    Charts will be saved in: Desktop/thesis/output/underutilized/
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
DATA_PATH   = os.path.join(BASE_DIR, "data", "master_albania.csv")
SCORES_PATH = os.path.join(BASE_DIR, "data", "potential_scores.xlsx")
OUT_DIR     = os.path.join(BASE_DIR, "output", "underutilized")
os.makedirs(OUT_DIR, exist_ok=True)

TYPE_COLORS = {
    "Cultural & Heritage":  "#8B4513",
    "Sun & Sea":            "#1E90FF",
    "Nature & Adventure":   "#228B22",
    "Rural & Agrotourism":  "#DAA520",
    "Urban & Business":     "#6A0DAD",
}

TIER_COLORS = {
    "Established":   "#E74C3C",
    "Emerging":      "#F39C12",
    "Undiscovered":  "#95A5A6",
}

# ── Load Data ──────────────────────────────────────────────────────────────────

print("\nAlbania Tourism - Underutilized Regions Analysis")
print("=" * 60)

df      = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_dest = df[df["is_anchor"] == False].copy()

# ── Calculate Actual Interest Per Destination ──────────────────────────────────

actual = (
    df_dest
    .groupby(["keyword", "tourism_type"])["interest"]
    .mean()
    .reset_index()
)
actual.columns = ["destination", "tourism_type", "avg_interest"]

# ── Tier Classification ────────────────────────────────────────────────────────

def classify_tier(val):
    if val >= 3.0:
        return "Established"
    elif val > 0.0:
        return "Emerging"
    else:
        return "Undiscovered"

actual["tier"] = actual["avg_interest"].apply(classify_tier)

print(f"Destinations loaded: {len(actual)}")
print(f"\nTier Distribution:")
print(actual["tier"].value_counts().to_string())

# ── Load Potential Scores ──────────────────────────────────────────────────────

scores_raw = pd.read_excel(SCORES_PATH, header=1)
scores_raw.columns = scores_raw.iloc[0]
scores_raw = scores_raw.iloc[1:].reset_index(drop=True)
scores_raw.columns = ["destination", "tourism_type", "heritage", "natural",
                      "events", "infrastructure", "accessibility", "total"]
scores_raw = scores_raw.dropna(subset=["destination"])
scores_raw["total"] = pd.to_numeric(scores_raw["total"], errors="coerce")
scores_raw["destination"] = scores_raw["destination"].str.strip()

NAME_MAP = {
    "Berat":               "Berat",
    "Gjirokastra":         "Gjirokastër",
    "Kruja":               "Krujë",
    "Butrint":             "Butrint National Archaeological Park",
    "Apollonia":           "Apollonia",
    "Antigonea":           "Archaeological Park of Antigona",
    "Byllis":              "Byllis Archaeological Park",
    "Amantia":             "Amantia",
    "Ksamil":              "Ksamil",
    "Dhermi":              "Dhërmi",
    "Himara":              "Himarë",
    "Saranda":             "Sarandë",
    "Vlora":               "Vlorë",
    "Porto Palermo":       "Palermo Beach",
    "Gjipe Beach":         "Gjipe Beach",
    "Borsh":               "Borsh",
    "Karaburun":           "Karaburun Peninsula",
    "Livadhi Beach":       "Livadi Beach",
    "Palasa":              "Palasë",
    "Sazan Island":        "Sazan Island",
    "Jale Beach":          "Jala",
    "Qeparo":              "Qeparo Fushë",
    "Lukova":              "Lukovë",
    "Shengjin":            "Shëngjin Beach",
    "Theth":               "Theth",
    "Valbona":             "Valbonë",
    "Osumi Canyons":       "Bogovë Waterfall and Osumi Canyon",
    "Vjosa River":         "Vjosa",
    "Permet":              "Përmet",
    "Bogova Waterfall":    "Bogovë Waterfall and Osumi Canyon",
    "Koman Lake":          "Komani Lake",
    "Llogara":             "Llogara National Park",
    "Blue Eye":            "The Blue Eye",
    "Shala River":         "Shala",
    "Accursed Mountains":  "Albanian Alps",
    "Fierza Lake":         "Fierza Reservoir",
    "Vermosh":             "Vermosh",
    "Roshnik":             "Roshnik festival",
    "Pogradec":            "Pogradec",
    "Voskopoja":           "Voskopojë",
    "Korça":               "Korçë",
    "Drilon":              "Drilon",
    "Permet thermal":      "permet thermal",
    "Lin Peninsula":       "Lin Peninsula, Pogradec",
    "Leskovik":            "Leskovik",
    "Permet wine":         "permet wine",
    "Berat wine":          "berat wine",
    "Durres":              "Durrës",
    "Shkodra":             "Shkodër",
    "Elbasan":             "Elbasan",
    "Fier":                "Fier",
    "Shkodra lake":        "Shkodra Lake",
    "Tirana nightlife":    "Albania Business",
}

scores_raw["destination"] = scores_raw["destination"].map(NAME_MAP).fillna(scores_raw["destination"])

# ── Merge ──────────────────────────────────────────────────────────────────────

merged = pd.merge(
    actual[["destination", "tourism_type", "avg_interest", "tier"]],
    scores_raw[["destination", "total"]],
    on="destination",
    how="left"
)
merged["total"] = merged["total"].fillna(merged["total"].median())
merged = merged.sort_values("avg_interest", ascending=False)

# ── Priority Targets = Emerging + High Potential ───────────────────────────────

priority = merged[
    (merged["tier"] == "Emerging") &
    (merged["total"] >= 6)
].sort_values("total", ascending=False)

print(f"\nPriority Investment Targets (Emerging + High Potential): {len(priority)}")

# ── 1. DESTINATION TIERS CHART ────────────────────────────────────────────────

print("\nChart 1: Destination tiers...")

# Only show destinations with interest > 0 for clarity
visible = merged[merged["avg_interest"] > 0].sort_values("avg_interest", ascending=True)

fig, ax = plt.subplots(figsize=(14, 12))

colors = [TIER_COLORS.get(t, "#888") for t in visible["tier"]]
bars   = ax.barh(visible["destination"], visible["avg_interest"],
                 color=colors, alpha=0.85, edgecolor="white")

# Tier threshold lines
ax.axvline(x=3.0, color=TIER_COLORS["Established"], linestyle="--",
           linewidth=1.5, label="Established threshold (3.0)")
ax.axvline(x=0.5, color=TIER_COLORS["Emerging"], linestyle="--",
           linewidth=1.5, label="Emerging threshold (0.5)")

ax.set_xlabel("Average Search Interest (Albania, 2019-2025)", fontsize=11)
ax.set_title("Albanian Tourism Destinations by Interest Tier\n(Established / Emerging / Undiscovered)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

tier_patches = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items()]
ax.legend(handles=tier_patches, loc="lower right", fontsize=9, title="Tier")

for bar, val in zip(bars, visible["avg_interest"]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=8)

plt.tight_layout()
path1 = os.path.join(OUT_DIR, "1_destination_tiers.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path1}")

# ── 2. PRIORITY INVESTMENT TARGETS ────────────────────────────────────────────

print("Chart 2: Priority investment targets...")

fig, ax = plt.subplots(figsize=(14, 8))

colors = [TYPE_COLORS.get(t, "#888") for t in priority["tourism_type"]]
x      = np.arange(len(priority))
width  = 0.35

bars1 = ax.bar(x - width/2, priority["total"], width,
               label="Potential Score (0-10)",
               color=colors, alpha=0.9, edgecolor="white")
bars2 = ax.bar(x + width/2, priority["avg_interest"], width,
               label="Actual Interest",
               color=colors, alpha=0.35, edgecolor="white", hatch="//")

ax.set_xticks(x)
ax.set_xticklabels(priority["destination"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Score / Interest", fontsize=11)
ax.set_title("Priority Investment Targets\nEmerging Destinations with High Potential - Albania",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
ax.legend(handles=legend_patches, loc="upper right", fontsize=8, title="Tourism Type")

plt.tight_layout()
path2 = os.path.join(OUT_DIR, "2_priority_targets.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path2}")

# ── 3. TIER DISTRIBUTION BY TOURISM TYPE ──────────────────────────────────────

print("Chart 3: Tier distribution by tourism type...")

tier_type = merged.groupby(["tourism_type", "tier"]).size().unstack(fill_value=0)
tier_order = ["Established", "Emerging", "Undiscovered"]
for t in tier_order:
    if t not in tier_type.columns:
        tier_type[t] = 0
tier_type = tier_type[tier_order]

fig, ax = plt.subplots(figsize=(12, 6))
x     = np.arange(len(tier_type))
width = 0.25

for i, (tier, color) in enumerate(TIER_COLORS.items()):
    ax.bar(x + i*width, tier_type[tier], width,
           label=tier, color=color, alpha=0.85, edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels(tier_type.index, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Number of Destinations", fontsize=11)
ax.set_title("Destination Tier Distribution by Tourism Type\nAlbania 2019-2025",
             fontsize=14, fontweight="bold")
ax.legend(title="Tier", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path3 = os.path.join(OUT_DIR, "3_tier_by_type.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path3}")

# ── 4. POTENTIAL VS ACTUAL — ESTABLISHED ONLY ─────────────────────────────────

print("Chart 4: Established destinations - Potential vs Actual...")

established = merged[merged["tier"] == "Established"].sort_values("avg_interest", ascending=False)

fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(established))
width = 0.35

colors = [TYPE_COLORS.get(t, "#888") for t in established["tourism_type"]]

ax.bar(x - width/2, established["total"], width,
       label="Potential Score (0-10)", color=colors, alpha=0.9, edgecolor="white")
ax.bar(x + width/2, established["avg_interest"], width,
       label="Actual Interest", color=colors, alpha=0.35,
       edgecolor="white", hatch="//")

ax.set_xticks(x)
ax.set_xticklabels(established["destination"], rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Score / Interest", fontsize=11)
ax.set_title("Established Destinations - Potential vs Actual Interest\nAlbania 2019-2025",
             fontsize=14, fontweight="bold")

legend_patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
fig.legend(handles=legend_patches, loc="upper right", fontsize=8,
           title="Tourism Type", bbox_to_anchor=(0.98, 0.95))

ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path4 = os.path.join(OUT_DIR, "4_established_potential_vs_actual.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path4}")

# ── KEY FINDINGS ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("KEY FINDINGS - UNDERUTILIZED REGIONS")
print("=" * 60)

est   = merged[merged["tier"] == "Established"]
emer  = merged[merged["tier"] == "Emerging"]
undis = merged[merged["tier"] == "Undiscovered"]

print(f"\nEstablished Destinations ({len(est)}):")
for _, r in est.sort_values("avg_interest", ascending=False).iterrows():
    print(f"   {r['destination']} ({r['tourism_type']}): interest={r['avg_interest']:.2f}, potential={r['total']:.0f}/10")

print(f"\nTop Priority Emerging Destinations ({len(priority)}):")
for _, r in priority.iterrows():
    print(f"   {r['destination']} ({r['tourism_type']}): interest={r['avg_interest']:.2f}, potential={r['total']:.0f}/10")

print(f"\nUndiscovered Destinations: {len(undis)} — zero international visibility")

print("\n" + "=" * 60)
print("Underutilized regions analysis complete!")
print(f"Charts saved in: {OUT_DIR}")
print("=" * 60)
print("\nNext step: Run social_network_analysis.py")
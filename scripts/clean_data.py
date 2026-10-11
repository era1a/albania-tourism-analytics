"""
Albania Tourism - Data Cleaning & Merging Script
Diploma Thesis: AI-Assisted Data Analytics Platform for Predictive Tourism Insights
------------------------------------------------------------------------------------
This script:
1. Loads all 36 CSV files (18 AL + 18 WW)
2. Labels each file with tourism type and geo
3. Merges everything into two master datasets (AL and WW)
4. Saves clean master CSVs ready for analysis

HOW TO RUN:
    Make sure your files are in: Desktop/thesis/data/raw/
    Run this script from PyCharm
    Output will be saved in: Desktop/thesis/data/
"""

import os
import pandas as pd

# ── File Mapping ───────────────────────────────────────────────────────────────
# Maps each file number to its tourism type and keywords

FILE_MAP = {
    1:  {"type": "Cultural & Heritage",   "keywords": ["Tirana", "Berat", "Gjirokastra", "Kruja", "Butrint"]},
    2:  {"type": "Cultural & Heritage",   "keywords": ["Tirana", "Berat Castle", "Gjirokastra Castle", "Rozafa Castle", "Apollonia Albania"]},
    3:  {"type": "Cultural & Heritage",   "keywords": ["Tirana", "Onufri Museum", "Antigonea", "Byllis", "Amantia"]},
    4:  {"type": "Sun & Sea",             "keywords": ["Tirana", "Ksamil", "Dhermi", "Himara", "Saranda"]},
    5:  {"type": "Sun & Sea",             "keywords": ["Tirana", "Vlora", "Porto Palermo", "Gjipe Beach", "Borsh"]},
    6:  {"type": "Sun & Sea",             "keywords": ["Tirana", "Karaburun", "Livadhi Beach", "Palasa", "Sazan Island"]},
    7:  {"type": "Sun & Sea",             "keywords": ["Tirana", "Jale Beach", "Qeparo", "Lukova", "Shengjin"]},
    8:  {"type": "Nature & Adventure",    "keywords": ["Tirana", "Theth", "Valbona", "Osumi Canyons", "Vjosa River"]},
    9:  {"type": "Nature & Adventure",    "keywords": ["Tirana", "Permet", "Bogova Waterfall", "Koman Lake", "Llogara"]},
    10: {"type": "Nature & Adventure",    "keywords": ["Tirana", "Theth National Park", "Valbona Valley", "Fierza Lake", "Vermosh"]},
    11: {"type": "Nature & Adventure",    "keywords": ["Tirana", "Blue Eye Albania", "Shala River", "Grunas Waterfall", "Accursed Mountains"]},
    12: {"type": "Rural & Agrotourism",   "keywords": ["Tirana", "Roshnik", "Pogradec", "Voskopoja", "Korca"]},
    13: {"type": "Rural & Agrotourism",   "keywords": ["Tirana", "Korca wine", "Albanian wine", "Drilon", "Permet thermal"]},
    14: {"type": "Rural & Agrotourism",   "keywords": ["Tirana", "Agrotourism Albania", "Roshnik festival", "Lin Peninsula", "Shelcan"]},
    15: {"type": "Rural & Agrotourism",   "keywords": ["Tirana", "Leskovik", "Permet wine", "Berat wine", "Albanian villages"]},
    16: {"type": "Urban & Business",      "keywords": ["Tirana", "Durres", "Shkodra", "Elbasan", "Fier"]},
    17: {"type": "Urban & Business",      "keywords": ["Tirana", "Tirana nightlife", "Durres beach", "Shkodra lake", "Albania business"]},
    18: {"type": "Urban & Business",      "keywords": ["Tirana", "Albania conference", "Tirana hotels", "Albania airport", "Visit Tirana"]},
}

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & Clean Function ──────────────────────────────────────────────────────

def load_csv(filepath, file_num, geo):
    """Load one CSV file and add metadata columns."""
    try:
        df = pd.read_csv(filepath)

        # Rename first column to 'date'
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

        # Parse date
        df["date"] = pd.to_datetime(df["date"])

        # Replace <1 with 0.5
        df = df.replace("<1", 0.5)

        # Convert all value columns to numeric
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add metadata
        df["file_num"]     = file_num
        df["geo"]          = geo
        df["tourism_type"] = FILE_MAP[file_num]["type"]

        # Melt from wide to long format
        # This turns each keyword column into a row
        id_vars = ["date", "file_num", "geo", "tourism_type"]
        value_vars = [c for c in df.columns if c not in id_vars]

        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="keyword",
            value_name="interest"
        )

        # Remove Tirana rows - it was only anchor, not a destination to analyze
        # We keep it only for reference if needed
        df_long ["is_anchor"] = df_long ["keyword"].str.strip().str.lower().str.contains("tiran", na=False)
        df_long = df_long [df_long ["is_anchor"] == False]
        print(f"  ✅ Loaded: {os.path.basename(filepath)} → {len(df_long)} rows")
        return df_long

    except Exception as e:
        print(f"  ❌ Error loading {filepath}: {e}")
        return None

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n🇦🇱 Albania Tourism - Data Cleaning & Merging")
    print("=" * 60)
    print(f"Looking for files in: {RAW_DIR}\n")

    all_frames = []
    missing_files = []

    for num in range(1, 19):
        for geo in ["AL", "WW"]:
            filename = f"{geo}.{num}.csv"
            filepath = os.path.join(RAW_DIR, filename)

            if not os.path.exists(filepath):
                print(f"  ⚠️  Missing: {filename}")
                missing_files.append(filename)
                continue

            df = load_csv(filepath, num, geo)
            if df is not None:
                all_frames.append(df)

    if not all_frames:
        print("\n❌ No files loaded. Check your RAW_DIR path.")
        return

    # Combine all into master dataframe
    master = pd.concat(all_frames, ignore_index=True)

    # Clean up
    master["date"]         = pd.to_datetime(master["date"])
    master["interest"]     = pd.to_numeric(master["interest"], errors="coerce")
    master["keyword"]      = master["keyword"].str.strip()
    master["tourism_type"] = master["tourism_type"].str.strip()

    # Filter to 2019 onwards (ignore the Dec 2018 row some files have)
    master = master[master["date"] >= "2019-01-01"]

    # Sort
    master = master.sort_values(["tourism_type", "keyword", "geo", "date"])
    master = master.reset_index(drop=True)

    # ── Save Master Datasets ───────────────────────────────────────────────────

    # Full master (both AL and WW)
    master_path = os.path.join(OUT_DIR, "master_dataset.csv")
    master.to_csv(master_path, index=False)
    print(f"\n✅ Master dataset saved: {master_path}")

    # Worldwide only
    ww = master[master["geo"] == "WW"]
    ww_path = os.path.join(OUT_DIR, "master_worldwide.csv")
    ww.to_csv(ww_path, index=False)
    print(f"✅ Worldwide dataset saved: {ww_path}")

    # Albania only
    al = master[master["geo"] == "AL"]
    al_path = os.path.join(OUT_DIR, "master_albania.csv")
    al.to_csv(al_path, index=False)
    print(f"✅ Albania dataset saved: {al_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows        : {len(master):,}")
    print(f"Date range        : {master['date'].min().date()} → {master['date'].max().date()}")
    print(f"Tourism types     : {master['tourism_type'].nunique()}")
    print(f"Unique keywords   : {master['keyword'].nunique()}")
    print(f"Geo versions      : {master['geo'].unique().tolist()}")

    print("\n📂 Rows per Tourism Type:")
    summary = master[master["is_anchor"] == False].groupby(
        ["tourism_type", "geo"]
    )["keyword"].nunique().unstack()
    print(summary.to_string())

    if missing_files:
        print(f"\n⚠️  Missing files ({len(missing_files)}):")
        for f in missing_files:
            print(f"   - {f}")

    print("\n" + "=" * 60)
    print("✅ Cleaning complete! Next step: run analysis scripts.")
    print("=" * 60)

if __name__ == "__main__":
    main()
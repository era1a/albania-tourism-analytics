"""
Albania Tourism - AI Features (Groq Version)
Diploma Thesis: AI-Assisted Data Analytics Platform for Predictive Tourism Insights
------------------------------------------------------------------------------------
This script contains:
1. AI Report Generator
2. Off-Season Strategy Generator
3. Chatbot context builder

pip install groq
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from groq import Groq

# ── PASTE YOUR GROQ API KEY HERE ───────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = "openai/gpt-oss-120b"

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "master_worldwide.csv")

# ── Off-Season Strategy Table ──────────────────────────────────────────────────

OFF_SEASON_STRATEGY = {
    1:  {
        "month_name":   "January",
        "best_type":    "Urban & Business",
        "attractions":  ["Tirana city breaks", "New Year events", "conferences", "Tirana nightlife"],
        "destinations": ["Tirana", "Durrës", "Shkodra"],
        "rationale":    "Urban destinations maintain stable interest year-round. January is ideal for city breaks and business tourism when beach destinations are closed."
    },
    2:  {
        "month_name":   "February",
        "best_type":    "Rural & Agrotourism",
        "attractions":  ["Korça Carnival", "snow tourism Korça", "wine culture", "village life"],
        "destinations": ["Korça", "Voskopoja", "Roshnik"],
        "rationale":    "Korça Carnival is one of Albania's most celebrated winter events. Snow-covered Korça and Voskopoja offer unique rural winter experiences."
    },
    3:  {
        "month_name":   "March",
        "best_type":    "Cultural & Heritage",
        "attractions":  ["Easter heritage trails", "Gjirokastra", "Berat", "Apollonia", "spring festivals"],
        "destinations": ["Gjirokastra", "Berat", "Kruja", "Apollonia"],
        "rationale":    "Spring brings ideal weather for heritage exploration. Easter period drives cultural tourism. UNESCO sites are uncrowded and accessible."
    },
    4:  {
        "month_name":   "April",
        "best_type":    "Nature & Adventure",
        "attractions":  ["hiking season opening", "Vjosa rafting", "wildflower season", "Llogara Pass"],
        "destinations": ["Permet", "Vjosa River", "Llogara", "Osumi Canyons"],
        "rationale":    "April marks the opening of hiking and rafting season. Wildflowers bloom across Albanian mountains."
    },
    5:  {
        "month_name":   "May",
        "best_type":    "Nature & Adventure",
        "attractions":  ["Theth hiking", "Valbona Valley", "Permet thermal baths", "Shala River"],
        "destinations": ["Theth", "Valbona", "Permet", "Shala River"],
        "rationale":    "May offers the best conditions for northern Albania trekking. Theth and Valbona are snow-free and uncrowded."
    },
    9:  {
        "month_name":   "September",
        "best_type":    "Cultural & Heritage",
        "attractions":  ["post-summer heritage", "Berat", "Gjirokastra", "cooler weather touring", "wine harvest"],
        "destinations": ["Berat", "Gjirokastra", "Butrint", "Kruja"],
        "rationale":    "September offers ideal temperatures for heritage exploration. Summer crowds have left. Wine harvest begins."
    },
    10: {
        "month_name":   "October",
        "best_type":    "Rural & Agrotourism",
        "attractions":  ["wine harvest Korça", "Roshnik festival", "olive harvest", "autumn foliage"],
        "destinations": ["Korça", "Roshnik", "Pogradec", "Voskopoja"],
        "rationale":    "October is wine and harvest season across Albanian rural regions. Autumn foliage in Korça rivals European destinations."
    },
    11: {
        "month_name":   "November",
        "best_type":    "Urban & Business",
        "attractions":  ["conference season", "Tirana cultural events", "city tourism", "Independence Day"],
        "destinations": ["Tirana", "Durrës", "Shkodra"],
        "rationale":    "November hosts major conferences in Tirana. Albanian Independence Day (Nov 28) creates a celebration period."
    },
    12: {
        "month_name":   "December",
        "best_type":    "Urban & Business",
        "attractions":  ["Korça Christmas Market", "snow tourism Theth & Valbona", "Tirana Christmas events", "Vermosh winter"],
        "destinations": ["Korça", "Theth", "Valbona", "Tirana", "Vermosh"],
        "rationale":    "Korça Christmas Market is Albania's premier winter event. Northern Albania offers untapped snow tourism."
    },
}

PEAK_MONTHS = [6, 7, 8]

# ── Load Data ──────────────────────────────────────────────────────────────────

def load_analysis_summary():
    df      = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df_dest = df[df["is_anchor"] == False].copy()

    df_dest["interest"] = df_dest.groupby("keyword")["interest"].transform(
        lambda x: (x / x.max()) * 100 if x.max() > 0 else x
    )

    df_dest["month"] = df_dest["date"].dt.month
    df_dest["year"]  = df_dest["date"].dt.year

    seasonal = (
        df_dest.groupby(["tourism_type", "month"])["interest"]
        .mean().round(1).reset_index()
    )
    annual = (
        df_dest.groupby(["tourism_type", "year"])["interest"]
        .mean().round(1).reset_index()
    )
    peaks = seasonal.loc[seasonal.groupby("tourism_type")["interest"].idxmax()]

    return {
        "seasonal": seasonal,
        "annual":   annual,
        "peaks":    peaks,
        "df":       df_dest,
    }

# ── Helper: call Groq ──────────────────────────────────────────────────────────

def call_groq(system_prompt, user_prompt, max_tokens=800):
    try:
        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# ── 1. AI REPORT GENERATOR ────────────────────────────────────────────────────

def generate_institutional_report(analysis_data, selected_type=None):
    seasonal = analysis_data["seasonal"]
    annual   = analysis_data["annual"]
    peaks    = analysis_data["peaks"]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    types_to_report = [selected_type] if selected_type else seasonal["tourism_type"].unique().tolist()

    data_summary = []
    for ttype in types_to_report:
        s_data   = seasonal[seasonal["tourism_type"] == ttype]
        a_data   = annual[annual["tourism_type"] == ttype]
        peak_row = peaks[peaks["tourism_type"] == ttype]

        peak_month = int(peak_row["month"].values[0]) if len(peak_row) > 0 else 8
        peak_val   = float(peak_row["interest"].values[0]) if len(peak_row) > 0 else 0
        avg_2019   = float(a_data[a_data["year"] == 2019]["interest"].values[0]) if len(a_data[a_data["year"] == 2019]) > 0 else 0
        avg_2025   = float(a_data[a_data["year"] == 2025]["interest"].values[0]) if len(a_data[a_data["year"] == 2025]) > 0 else 0
        growth     = ((avg_2025 - avg_2019) / avg_2019 * 100) if avg_2019 > 0 else 0

        data_summary.append(
            f"{ttype}: Peak={month_names[peak_month-1]} ({peak_val:.1f}/100), "
            f"Growth 2019-2025={growth:+.1f}%, 2025 avg={avg_2025:.1f}/100"
        )

    system = "You are an expert tourism analyst advising the Albanian Ministry of Tourism. Write professional, concise institutional reports."

    user = f"""Based on this Albanian tourism data (2019-2025):

{chr(10).join(data_summary)}

Context: Albania received 12.4M tourists in 2025 (+369% since 2020). 95.5% from Europe. All types peak in August.

Write a professional institutional report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (3-4 bullet points with numbers)
3. Strategic Recommendations (3-4 actionable recommendations)
4. Priority Actions for Next 6 Months

Keep under 400 words. Be specific and data-driven."""

    return call_groq(system, user)

# ── 2. OFF-SEASON STRATEGY GENERATOR ─────────────────────────────────────────

def generate_off_season_strategy(month_number, analysis_data):
    if month_number in PEAK_MONTHS:
        return "This is already a peak season month (June-August). No off-season strategy needed."

    strategy = OFF_SEASON_STRATEGY.get(month_number)
    if not strategy:
        return "No strategy defined for this month."

    seasonal     = analysis_data["seasonal"]
    month_data   = seasonal[seasonal["month"] == month_number]
    month_summary = []
    for _, row in month_data.iterrows():
        month_summary.append(f"  {row['tourism_type']}: {row['interest']:.1f}/100")

    system = "You are an expert tourism strategist for Albania specializing in off-season tourism development."

    user = f"""Month: {strategy['month_name']}
Current interest levels:
{chr(10).join(month_summary)}

Best tourism type for this month: {strategy['best_type']}
Key attractions: {', '.join(strategy['attractions'])}
Target destinations: {', '.join(strategy['destinations'])}
Rationale: {strategy['rationale']}

Generate a specific off-season campaign strategy including:
1. Campaign Name (3-5 words)
2. Target Markets (which European countries)
3. Key Messages (2-3 marketing messages)
4. Recommended Activities (3-4 specific experiences)
5. Expected Impact
6. Budget Priority (High/Medium/Low)

Keep under 300 words. Be specific to Albania and {strategy['month_name']}."""

    return call_groq(system, user)

# ── 3. CHATBOT ────────────────────────────────────────────────────────────────

def build_chatbot_context(analysis_data):
    seasonal    = analysis_data["seasonal"]
    annual      = analysis_data["annual"]
    peaks       = analysis_data["peaks"]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    lines = [
        "You are an AI assistant for Albania's Tourism Analytics Platform.",
        "Help institutional users understand tourism data and make strategic decisions.",
        "Always reference actual numbers. Keep answers to 3-5 sentences unless a detailed report is requested.",
        "",
        "KEY DATA:",
        "- Google Trends 2019-2025, validated with INSTAT (r=0.943)",
        "- 12.4M arrivals in 2025, +369% since 2020",
        "- 95.5% European tourists, 71.5% Southern European",
        "- All tourism types peak in August",
        "",
        "TOURISM TYPES:",
    ]

    for ttype in seasonal["tourism_type"].unique():
        a_data   = annual[annual["tourism_type"] == ttype]
        peak_row = peaks[peaks["tourism_type"] == ttype]
        peak_month = int(peak_row["month"].values[0]) if len(peak_row) > 0 else 8
        peak_val   = float(peak_row["interest"].values[0]) if len(peak_row) > 0 else 0
        avg_2025   = float(a_data[a_data["year"] == 2025]["interest"].values[0]) if len(a_data[a_data["year"] == 2025]) > 0 else 0
        avg_2019   = float(a_data[a_data["year"] == 2019]["interest"].values[0]) if len(a_data[a_data["year"] == 2019]) > 0 else 0
        growth     = ((avg_2025 - avg_2019) / avg_2019 * 100) if avg_2019 > 0 else 0
        lines.append(f"- {ttype}: peak {month_names[peak_month-1]} ({peak_val:.1f}/100), growth {growth:+.1f}%, 2025 avg {avg_2025:.1f}/100")

    lines += [
        "",
        "KEY FINDINGS:",
        "- 9 established destinations, 14 high-potential emerging destinations",
        "- Top underutilized: Apollonia, Llogara, Korça, Pogradec, Permet",
        "- Strongest corridor: Southern Riviera (Saranda-Ksamil-Dhermi-Vlora)",
        "- Rural & Agrotourism most stable year-round, most accurate model (15% MAPE)",
        "- Nature & Adventure fastest growing (+81.3% forecast)",
        "- Off-season opportunities: snow tourism (Theth/Korça Dec-Feb), wine harvest (Oct), Korça Carnival (Feb)",
    ]

    return "\n".join(lines)

def chatbot_response(user_message, analysis_data, chat_history=None):
    context  = build_chatbot_context(analysis_data)
    messages = []

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=800,
            messages=[{"role": "system", "content": context}] + messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# ── TEST RUN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nAlbania Tourism - AI Features Test")
    print("=" * 60)

    print("Loading analysis data...")
    data = load_analysis_summary()
    print("Data loaded!\n")

    print("=" * 60)
    print("TEST 1: Institutional Report...")
    print("=" * 60)
    print(generate_institutional_report(data))

    print("\n" + "=" * 60)
    print("TEST 2: Off-Season Strategy — February (Korça Carnival + Snow)...")
    print("=" * 60)
    print(generate_off_season_strategy(2, data))

    print("\n" + "=" * 60)
    print("TEST 3: Off-Season Strategy — December (Snow Tourism)...")
    print("=" * 60)
    print(generate_off_season_strategy(12, data))

    print("\n" + "=" * 60)
    print("TEST 4: Chatbot Question...")
    print("=" * 60)
    question = "Which destinations should we prioritize for investment in 2026?"
    print(f"Q: {question}\n")
    print(chatbot_response(question, data))

    print("\n" + "=" * 60)
    print("AI Features complete!")
    print("=" * 60)
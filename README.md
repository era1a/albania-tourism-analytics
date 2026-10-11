# Albania Tourism Analytics Platform

> AI-assisted data analytics platform for predictive tourism insights. Bachelor's thesis project.

## Live demo : https://albania-tourism-analytics.streamlit.app/

## The problem

Tourism in Albania is growing, but the people who plan for it (municipalities, tourism boards, local businesses) often lack data-driven tools:

- Demand is **highly seasonal**, so campaigns and capacity are planned late or by guesswork.
- Interest is **concentrated in a few well-known destinations**, while many places have almost no international visibility.
- Official arrival statistics are published with a delay, so they are hard to use for forward planning.

## The solution

This platform turns freely available search-interest data into forecasts and concrete recommendations, and checks them against official statistics.

| Problem | How the platform addresses it |
|---|---|
| Seasonality is hard to plan for | 12-month demand forecasts per tourism type, with peak months and confidence intervals |
| Campaigns start too late | **Promotion windows**: the best months to run promotion before each type's peak |
| A few destinations get all the attention | Destinations classified as *Established / Emerging / Undiscovered*, combined with a potential score to find **priority investment targets** |
| Hard to know which places work together | Correlation analysis to find destinations that are searched together (**tourism corridors**) |
| Can search data be trusted? | **Validation against INSTAT** tourist arrivals, plus a Prophet vs SARIMA model comparison |
| Results are hard to interpret | **AI insights**: each chart can be explained in plain language by an LLM |

## Features

- Seasonal patterns, annual trends and COVID-19 impact per tourism type
- Correlation between tourism types and between destination pairs
- Prophet forecasting with accuracy metrics (MAE, RMSE, MAPE) and SARIMA comparison
- Underutilized regions analysis and priority targets
- Google Trends vs INSTAT arrivals validation
- AI-generated chart explanations, off-season strategies and reports (Groq API)
- Interactive Streamlit dashboard

Five tourism types are analysed: Cultural & Heritage, Sun & Sea, Nature & Adventure, Rural & Agrotourism, Urban & Business (2019 to 2025).

## Tech stack

Python, Pandas, NumPy, Prophet, statsmodels (SARIMA), scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, Groq API.

## Run it locally

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set your Groq API key as an environment variable :

```bash
# Windows (PowerShell): $env:GROQ_API_KEY="your-key"
# Mac/Linux:            export GROQ_API_KEY="your-key"
```

Start the dashboard:

```bash
streamlit run scripts/dashboard.py
```

The first load can take 1 to 2 minutes while the forecasting models are trained. The AI features need the API key; the rest of the dashboard works without it.

## Project structure

```
data/        master datasets, INSTAT arrivals, destination potential scores
output/      generated charts (only the SARIMA comparison is kept in the repo)
scripts/
  clean_data.py             merge and clean the raw Google Trends files
  seasonality_analysis.py   seasonality, trends, COVID impact
  correlation_analysis.py   correlations and tourism corridors
  prophet_forecasting.py    12-month forecasts per tourism type
  forecasting_addition.py   accuracy metrics and promotion windows
  sarima_comparison.py      Prophet vs SARIMA evaluation
  instat_validation.py      validation against INSTAT arrivals
  underutilized_regions.py  destination tiers and priority targets
  ai_features.py            LLM explanations and strategies (Groq)
  dashboard.py              Streamlit dashboard
```

## Data sources

- **Google Trends:** search interest for Albanian destinations, 2019 to 2025
- **INSTAT (Albania):** official tourist arrival statistics
- **Destination potential scores:** manual scoring of heritage, nature, events, infrastructure and accessibility

## Author

**Era Zeneli** |GitHub:era1a|
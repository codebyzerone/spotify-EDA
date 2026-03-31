# Spotify 2024 Multi-Platform EDA Dashboard

Build an interactive, high-performance Streamlit dashboard using Python (Pandas/Plotly) for multi-platform EDA on the 2024 Kaggle "Most Streamed Spotify Songs" dataset. The dashboard correlates viral TikTok/YouTube metrics with core audio features through vectorized data cleaning and modular "Song DNA" radar charts.

## User Review Required

> [!IMPORTANT]
> **Dataset required**: You will need to download the CSV from Kaggle and place it at `d:\spotify.py\data\spotify_2024.csv`. The app will show instructions if the file is missing. The expected dataset is the **"Most Streamed Spotify Songs 2024"** that includes both platform metrics (Spotify/YouTube/TikTok) **and** audio features (danceability, energy, valence, etc.). If your CSV only has streaming metrics, the radar charts will be disabled gracefully.

## Proposed Changes

### Data Layer

#### [NEW] [data_loader.py](file:///d:/spotify.py/data_loader.py)

Modular data loading & vectorized cleaning:
- `load_data(path)` — reads CSV with encoding fallback, caches with `@st.cache_data`
- `clean_data(df)` — vectorized pipeline: strip whitespace, coerce numeric columns (streams/views/likes) with `pd.to_numeric(..., errors='coerce')`, parse dates, drop all-null rows, fill missing numeric with median
- `get_audio_features(df)` — returns subset of audio feature columns that exist in the dataset
- `get_platform_metrics(df)` — returns subset of platform metric columns
- Constants: `AUDIO_FEATURES`, `PLATFORM_METRICS`, `NUMERIC_COLS`

---

### Dashboard App

#### [NEW] [app.py](file:///d:/spotify.py/app.py)

Single-file Streamlit app with 5 sections:

1. **Sidebar Filters** — artist multi-select, release year range slider, platform toggle
2. **KPI Row** — total tracks, total streams, avg popularity, top artist (metric cards via `st.columns`)
3. **Cross-Platform Correlation** — Plotly heatmap of viral metrics (TikTok Views/Likes/Posts, YouTube Views/Likes) vs audio features (danceability, energy, valence, acousticness, liveness); interactive scatter plot with dropdown axis selectors
4. **Song DNA Radar Charts** — searchable track selector → Plotly `Scatterpolar` radar chart showing normalized audio features for that track; side-by-side comparison of up to 3 tracks
5. **Distribution Explorer** — histogram/violin plots for any selected numeric column, top-N bar charts by platform

All charts use Plotly for interactivity, consistent dark-themed color palette, and responsive layout.

---

### Assets

#### [NEW] [requirements.txt](file:///d:/spotify.py/requirements.txt)

```
pandas
numpy
plotly
streamlit
```

#### [NEW] [data/](file:///d:/spotify.py/data/)

Empty directory with a `.gitkeep`; user places their Kaggle CSV here.

## Verification Plan

### Automated Tests
- Run `streamlit run app.py` and verify the server starts without errors
- Browser test: open `http://localhost:8501`, verify the page renders with a "missing data" message (if no CSV), or the full dashboard (if CSV present)

### Manual Verification
- Place a Kaggle CSV in `d:\spotify.py\data\spotify_2024.csv`
- Run `streamlit run app.py`
- Confirm sidebar filters work and charts update
- Select a track for the Song DNA radar chart and verify it renders
- Compare 2–3 tracks side-by-side on the radar chart
- Hover over the heatmap and scatter plots to check tooltips

"""
app.py — Spotify 2024 Multi-Platform EDA Dashboard
Interactive Streamlit dashboard with Plotly charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from data_loader import (
    load_data, load_uploaded_file, clean_data,
    get_audio_features, get_platform_metrics,
    AUDIO_FEATURES, ALL_PLATFORM_COLS, PLATFORM_METRICS,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spotify 2024 · Multi-Platform EDA",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* KPI metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(30, 215, 96, 0.15);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
div[data-testid="stMetric"] label {
    color: #b3b3b3 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1ed760 !important;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(30, 215, 96, 0.1);
}

/* Section dividers */
hr {
    border-color: rgba(30, 215, 96, 0.15) !important;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    margin=dict(l=40, r=40, t=50, b=40),
)

SPOTIFY_COLORS = [
    "#1ed760", "#1db954", "#17a74a", "#14943f",
    "#b3b3b3", "#ff6b6b", "#ffd93d", "#6bcbff",
    "#c084fc", "#fb923c", "#f472b6",
]

# ─── Data Loading ─────────────────────────────────────────────────────────────

DATA_PATH = Path("data/spotify_2024.csv")
DATA_ZIP_PATH = Path("data/spotify_2024.zip")

def render_missing_data_page():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ## 📁 Dataset Not Found

        Place your **Most Streamed Spotify Songs 2024** CSV at:

        ```
        data/spotify_2024.csv
        ```

        ### How to get the dataset
        1. Visit [Kaggle – Most Streamed Spotify Songs 2024](https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024)
        2. Download the CSV file
        3. Rename it to `spotify_2024.csv`
        4. Place it in the `data/` folder of this project
        5. Refresh this page ↻

        > 💡 For the full experience (Song DNA radar charts), use a dataset
        > version that includes audio features like *Danceability*, *Energy*,
        > *Valence*, etc.
        """)
    st.stop()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame):
    """Render sidebar filters and return the filtered DataFrame."""
    with st.sidebar:
        st.markdown("## 🎧 Filters")
        st.markdown("---")

        # Artist filter
        artists = sorted(df["Artist"].dropna().unique().tolist())
        selected_artists = st.multiselect(
            "🎤 Artists",
            options=artists,
            default=[],
            help="Leave empty to include all artists",
        )

        # Year range
        if "Release Year" in df.columns and df["Release Year"].notna().any():
            min_yr = int(df["Release Year"].min())
            max_yr = int(df["Release Year"].max())
            if min_yr < max_yr:
                year_range = st.slider(
                    "📅 Release Year",
                    min_value=min_yr,
                    max_value=max_yr,
                    value=(min_yr, max_yr),
                )
            else:
                year_range = (min_yr, max_yr)
        else:
            year_range = None

        # Platform toggle
        st.markdown("---")
        st.markdown("### 📊 Platforms")
        platforms = {}
        for p in ("spotify", "youtube", "tiktok"):
            emoji = {"spotify": "🟢", "youtube": "🔴", "tiktok": "🎵"}[p]
            platforms[p] = st.checkbox(f"{emoji} {p.title()}", value=True)

        st.markdown("---")
        st.caption(f"**{len(df):,}** tracks loaded")

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if selected_artists:
        mask &= df["Artist"].isin(selected_artists)
    if year_range is not None:
        mask &= df["Release Year"].between(*year_range)

    return df[mask], platforms


# ─── Section 1: KPI Row ──────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame):
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Tracks", f"{len(df):,}")
    with cols[1]:
        if "Spotify Streams" in df.columns:
            total = df["Spotify Streams"].sum()
            st.metric("Total Streams", _fmt_big(total))
        else:
            st.metric("Total Streams", "N/A")
    with cols[2]:
        if "Spotify Popularity" in df.columns:
            avg_pop = df["Spotify Popularity"].mean()
            st.metric("Avg Popularity", f"{avg_pop:.1f}")
        else:
            st.metric("Avg Popularity", "N/A")
    with cols[3]:
        if "Artist" in df.columns:
            top = df["Artist"].value_counts().idxmax()
            st.metric("Top Artist", top)
        else:
            st.metric("Top Artist", "N/A")


def _fmt_big(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"




# ─── Section 3: Song DNA Radar Chart ─────────────────────────────────────────

def render_song_dna(df: pd.DataFrame):
    st.markdown("## 🧬 Song DNA · Radar Chart")
    st.caption("Compare normalised audio feature fingerprints of up to 3 tracks")

    audio_cols = get_audio_features(df)
    if not audio_cols:
        st.info(
            "Your dataset doesn't include audio feature columns "
            "(Danceability, Energy, Valence, …). "
            "Use a dataset version with Spotify audio features for this chart."
        )
        return

    # Build track label
    if "Track" in df.columns:
        label_col = "Track"
    elif "Track Name" in df.columns:
        label_col = "Track Name"
    else:
        st.warning("No track name column found.")
        return

    track_labels = df[label_col].dropna().unique().tolist()
    selected = st.multiselect(
        "🔎 Select tracks to compare (up to 3)",
        options=sorted(track_labels),
        max_selections=3,
        key="dna_tracks",
    )

    if not selected:
        st.info("Select one or more tracks above to render their Song DNA.")
        return

    # Normalise audio features to 0–100 for radar
    norm_df = df.copy()
    for c in audio_cols:
        cmin, cmax = norm_df[c].min(), norm_df[c].max()
        if cmax > cmin:
            norm_df[c] = (norm_df[c] - cmin) / (cmax - cmin) * 100
        else:
            norm_df[c] = 50

    fig = go.Figure()
    colors = SPOTIFY_COLORS[:3]

    for i, track in enumerate(selected):
        row = norm_df[norm_df[label_col] == track].iloc[0]
        values = [row[c] for c in audio_cols]
        values.append(values[0])  # close the polygon
        theta = audio_cols + [audio_cols[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill="toself",
            name=track[:40],
            line=dict(color=colors[i], width=2),
            fillcolor=colors[i].replace(")", ", 0.12)").replace("rgb", "rgba")
                      if "rgb" in colors[i] else colors[i] + "20",
            opacity=0.85,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        ),
        title="Song DNA Comparison",
        height=520,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(30,215,96,0.2)",
            borderwidth=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Section 4: Distribution Explorer ────────────────────────────────────────

def render_distributions(df: pd.DataFrame, platforms: dict):
    st.markdown("## 📊 Distribution Explorer")

    tab_dist, tab_topn = st.tabs(["📈 Distributions", "🏆 Top-N Rankings"])

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != "Release Year"]

    with tab_dist:
        if not numeric_cols:
            st.info("No numeric columns available.")
            return
        col_pick = st.selectbox("Select a metric", numeric_cols, key="dist_col")

        plot_type = st.radio(
            "Plot type", ["Histogram", "Violin"], horizontal=True, key="dist_type"
        )

        series = df[col_pick].dropna()
        if plot_type == "Histogram":
            fig = px.histogram(
                series, nbins=50,
                color_discrete_sequence=["#1ed760"],
                labels={"value": col_pick},
            )
        else:
            fig = px.violin(
                y=series,
                color_discrete_sequence=["#1ed760"],
                box=True, points="outliers",
                labels={"y": col_pick},
            )

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Distribution of {col_pick}",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_topn:
        c1, c2 = st.columns([2, 1])

        # Determine label column
        label_col = "Track" if "Track" in df.columns else (
            "Track Name" if "Track Name" in df.columns else None
        )
        if label_col is None:
            st.warning("No track name column found for rankings.")
            return

        with c1:
            rank_metric = st.selectbox(
                "Rank by", numeric_cols, key="rank_metric"
            )
        with c2:
            top_n = st.slider("Top N", 5, 30, 15, key="top_n")

        top = df.nlargest(top_n, rank_metric)[[label_col, "Artist", rank_metric]].copy()
        top["label"] = top[label_col].str[:30] + " · " + top["Artist"].str[:20]
        top = top.sort_values(rank_metric, ascending=True)

        fig = px.bar(
            top, x=rank_metric, y="label",
            orientation="h",
            color=rank_metric,
            color_continuous_scale=["#16213e", "#1ed760"],
            labels={rank_metric: rank_metric, "label": ""},
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Top {top_n} Tracks by {rank_metric}",
            height=max(350, top_n * 28),
            coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 0 0;'>
        <h1 style='font-size:2.4rem; font-weight:700;
            background: linear-gradient(90deg, #1ed760, #1db954, #6bcbff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0;'>
            🎧 Spotify 2024 · Multi-Platform EDA
        </h1>
        <p style='color:#b3b3b3; font-size:0.95rem; margin-top:4px;'>
            Exploring how viral TikTok & YouTube metrics correlate with core audio features
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──
    data_path = None
    if DATA_PATH.exists():
        data_path = DATA_PATH
    elif DATA_ZIP_PATH.exists():
        data_path = DATA_ZIP_PATH

    if data_path is None:
        # Also allow uploading
        st.markdown("---")
        uploaded = st.file_uploader(
            "📤 Upload your Spotify 2024 dataset", type=["csv", "zip"],
        )
        if uploaded is not None:
            raw = load_uploaded_file(uploaded)
        else:
            render_missing_data_page()
            return
    else:
        raw = load_data(data_path)

    df = clean_data(raw)

    # ── Sidebar ──
    filtered, platforms = render_sidebar(df)

    if filtered.empty:
        st.warning("No tracks match the current filters. Adjust the sidebar.")
        st.stop()

    # ── KPIs ──
    render_kpis(filtered)

    st.markdown("---")

    # ── Song DNA ──
    render_song_dna(filtered)

    st.markdown("---")

    # ── Distribution Explorer ──
    render_distributions(filtered, platforms)

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.8rem; padding:10px;'>"
        "Built with Streamlit & Plotly · Dataset: "
        "<a href='https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024' "
        "style='color:#1db954;'>Kaggle</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

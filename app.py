"""
app.py — Spotify 2024 Multi-Platform EDA Dashboard
Interactive Streamlit dashboard with Plotly charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
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

# ── Matplotlib / Seaborn dark-theme helper ────────────────────────────────────

def _apply_mpl_dark_theme():
    """Set a dark Spotify-flavoured theme for matplotlib/seaborn figures."""
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#b3b3b3",
        "ytick.color": "#b3b3b3",
        "grid.color": "#0f0f0f",
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.size": 11,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#333333",
    })
    sns.set_style("darkgrid", {
        "axes.facecolor": "#0e1117",
        "figure.facecolor": "#0e1117",
        "grid.color": "#0f0f0f",
    })

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

        > 💡 For the full experience, use a dataset version that includes
        > audio features like *Danceability*, *Energy*, *Valence*, etc.
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
        for p in ("spotify", "youtube"):
            emoji = {"spotify": "🟢", "youtube": "🔴"}[p]
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





# ─── Section 4: Correlation Heatmap (Seaborn) ────────────────────────────────

def render_correlation_heatmap(df: pd.DataFrame):
    """Render a seaborn correlation heatmap for numeric features."""
    st.markdown("## 🔥 Feature Correlation Heatmap")
    st.caption("Pearson correlations across streaming metrics & audio features — powered by **Seaborn**")

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != "Release Year"]
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns to compute correlations.")
        return

    # Let user pick which columns to include
    default_cols = numeric_cols[:12]  # sensible default
    selected = st.multiselect(
        "Select features for the heatmap",
        options=numeric_cols,
        default=default_cols,
        key="heatmap_cols",
    )
    if len(selected) < 2:
        st.warning("Select at least 2 features.")
        return

    corr = df[selected].corr()

    _apply_mpl_dark_theme()
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.75), max(6, len(selected) * 0.6)))

    # Custom diverging palette: teal → black → Spotify green
    cmap = sns.diverging_palette(190, 140, s=80, l=55, as_cmap=True)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="#1a1a2e",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold",
                 color="#1ed760", pad=14)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─── Section 5: Audio Feature KDE / Box Plots (Matplotlib + Seaborn) ────────

def render_audio_feature_plots(df: pd.DataFrame):
    """Render seaborn KDE and box plots for audio features."""
    st.markdown("## 🎶 Audio Feature Landscape")
    st.caption("Density curves & boxplots for audio features — powered by **Matplotlib + Seaborn**")

    audio_cols = get_audio_features(df)
    if not audio_cols:
        st.info(
            "Your dataset doesn't include audio feature columns "
            "(Danceability, Energy, Valence, …). "
            "Upload a version with audio features to see this section."
        )
        return

    plot_style = st.radio(
        "Visualisation style",
        ["KDE Density", "Box + Strip", "Violin"],
        horizontal=True,
        key="audio_plot_style",
    )

    _apply_mpl_dark_theme()
    palette = ["#1ed760", "#6bcbff", "#c084fc", "#ff6b6b",
               "#ffd93d", "#fb923c", "#f472b6"]

    if plot_style == "KDE Density":
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, col in enumerate(audio_cols):
            sns.kdeplot(
                data=df, x=col, ax=ax,
                fill=True, alpha=0.25, linewidth=1.8,
                color=palette[i % len(palette)],
                label=col,
            )
        ax.set_xlabel("Feature Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Audio Feature Density Curves", fontsize=14,
                     fontweight="bold", color="#1ed760", pad=12)
        ax.legend(framealpha=0.6, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif plot_style == "Box + Strip":
        melted = df[audio_cols].melt(var_name="Feature", value_name="Value")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(
            data=melted, x="Feature", y="Value",
            palette=palette[:len(audio_cols)],
            ax=ax, linewidth=1.2, fliersize=2,
            boxprops=dict(alpha=0.7),
        )
        sns.stripplot(
            data=melted, x="Feature", y="Value",
            color="#ffffff", alpha=0.08, size=2, ax=ax, jitter=True,
        )
        ax.set_title("Audio Features — Box + Strip", fontsize=14,
                     fontweight="bold", color="#1ed760", pad=12)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:  # Violin
        melted = df[audio_cols].melt(var_name="Feature", value_name="Value")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(
            data=melted, x="Feature", y="Value",
            palette=palette[:len(audio_cols)],
            ax=ax, inner="quartile", linewidth=1,
        )
        ax.set_title("Audio Features — Violin Plots", fontsize=14,
                     fontweight="bold", color="#1ed760", pad=12)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ─── Section 6: Distribution Explorer ────────────────────────────────────────

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
            Exploring how YouTube metrics correlate with core audio features
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

    # ── Correlation Heatmap (Seaborn) ──
    render_correlation_heatmap(filtered)

    st.markdown("---")

    # ── Audio Feature Landscape (Matplotlib + Seaborn) ──
    render_audio_feature_plots(filtered)

    st.markdown("---")

    # ── Distribution Explorer ──
    render_distributions(filtered, platforms)

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.8rem; padding:10px;'>"
        "Built with Streamlit, Plotly, Matplotlib &amp; Seaborn · Dataset: "
        "<a href='https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024' "
        "style='color:#1db954;'>Kaggle</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

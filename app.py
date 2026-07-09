"""
app.py — Spotify 2024 Multi-Platform Analytics Dashboard
Interactive Streamlit dashboard with Plotly charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from data_loader import (
    load_data, load_uploaded_file, clean_data,
    ALL_PLATFORM_COLS, PLATFORM_METRICS,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spotify 2024 · Multi-Platform Analytics",
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

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(30, 215, 96, 0.1);
    border-radius: 8px;
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
        """)
    st.stop()


# ─── Shared Helpers ───────────────────────────────────────────────────────────

def _fmt_big(n: float) -> str:
    """Format large numbers with T/B/M/K suffixes."""
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def _get_track_col(df: pd.DataFrame) -> str | None:
    """Return the track-name column ('Track' or 'Track Name'), or None."""
    for name in ("Track", "Track Name"):
        if name in df.columns:
            return name
    return None


def _horizontal_bar(data: pd.DataFrame, x: str, y: str,
                     title: str, height: int, **extra_layout):
    """Create a styled horizontal Plotly bar chart and return the figure."""
    fig = px.bar(
        data, x=x, y=y,
        orientation="h",
        color=x,
        color_continuous_scale=["#16213e", "#1ed760"],
        labels={x: x, y: ""},
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=height,
        coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=11)),
        **extra_layout,
    )
    return fig


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

        st.markdown("---")
        st.caption(f"**{len(df):,}** tracks loaded")

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if selected_artists:
        mask &= df["Artist"].isin(selected_artists)
    if year_range is not None:
        mask &= df["Release Year"].between(*year_range)

    return df[mask]


# ─── Section 1: KPI Row ──────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame):
    """Render the top-level KPI metric cards."""
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Tracks", f"{len(df):,}")
    with cols[1]:
        if "Spotify Streams" in df.columns:
            st.metric("Total Streams", _fmt_big(df["Spotify Streams"].sum()))
        else:
            st.metric("Total Streams", "N/A")
    with cols[2]:
        if "Spotify Popularity" in df.columns:
            st.metric("Avg Popularity", f"{df['Spotify Popularity'].mean():.1f}")
        else:
            st.metric("Avg Popularity", "N/A")
    with cols[3]:
        if "Artist" in df.columns:
            st.metric("Top Artist", df["Artist"].value_counts().idxmax())
        else:
            st.metric("Top Artist", "N/A")
    with cols[4]:
        if "Artist" in df.columns:
            st.metric("Unique Artists", f"{df['Artist'].nunique():,}")
        else:
            st.metric("Unique Artists", "N/A")


# ─── Section 2: Artist Analytics ─────────────────────────────────────────────

# Declarative spec: (source_column, agg_func, display_label)
_AGG_SPEC = [
    ("Spotify Streams",        "sum",  "Total Streams"),
    ("Track Score",            "mean", "Avg Track Score"),
    ("Spotify Popularity",     "mean", "Avg Popularity"),
    ("Spotify Playlist Reach", "mean", "Avg Playlist Reach"),
]


def _build_artist_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by Artist and compute aggregate metrics.
    Only includes columns that actually exist in the dataframe.
    Returns a sorted DataFrame (by Total Streams desc, then Tracks desc).
    """
    track_col = _get_track_col(df)

    # Start with track count — always available
    agg_dict = {track_col: "count"} if track_col else {}
    rename_map = {track_col: "Tracks"} if track_col else {}

    # Add optional metrics from the spec
    for src, func, label in _AGG_SPEC:
        if src in df.columns:
            agg_dict[src] = func
            rename_map[src] = label

    grouped = df.groupby("Artist", as_index=False).agg(agg_dict)
    grouped = grouped.rename(columns=rename_map)

    sort_col = "Total Streams" if "Total Streams" in grouped.columns else "Tracks"
    return grouped.sort_values(sort_col, ascending=False).reset_index(drop=True)


def _available_metrics(artist_df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Return [(display_label, column_name), …] for every metric column
    present in the aggregated artist DataFrame.
    """
    candidates = [
        ("Spotify Streams",    "Total Streams"),
        ("Track Score",        "Avg Track Score"),
        ("Spotify Popularity", "Avg Popularity"),
        ("Playlist Reach",     "Avg Playlist Reach"),
        ("Playlist Count",     "Total Playlist Count"),
        ("Number of Tracks",   "Tracks"),
    ]
    return [(label, col) for label, col in candidates if col in artist_df.columns]


def render_artist_analytics(df: pd.DataFrame):
    """
    🎤 Artist Analytics — master section.
    Renders: performance table, top-N chart, artist comparison, auto-insights.
    """
    st.markdown("## 🎤 Artist Analytics")
    st.caption("Who dominates the charts — and by how much?")

    if "Artist" not in df.columns:
        return

    # Pre-compute the aggregated artist table (shared by all sub-features)
    artist_df = _build_artist_agg(df)

    # Attach Playlist Count if the source column exists
    if "Spotify Playlist Count" in df.columns:
        artist_df["Total Playlist Count"] = (
            df.groupby("Artist")["Spotify Playlist Count"]
            .sum()
            .reindex(artist_df["Artist"])
            .values
        )

    # Metric list reused across Features 2 / 3 / 4
    metrics = _available_metrics(artist_df)
    metric_cols = [col for _, col in metrics]

    # ── Performance Table ─────────────────────────────────────────────────────

    st.markdown("### 📋 Artist Performance Table")

    display_df = artist_df.copy()
    # Format large-number columns with K/M/B/T suffixes
    for col in ("Total Streams", "Avg Playlist Reach"):
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda v: _fmt_big(v) if pd.notna(v) else "N/A"
            )
    # Round decimal columns for readability
    for col in ("Avg Track Score", "Avg Popularity"):
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1)

    st.dataframe(display_df.head(50), use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── Top Artists Chart ─────────────────────────────────────────────────────

    st.markdown("### 🏆 Top Artists")

    c1, c2 = st.columns([2, 1])
    with c1:
        selected_label = st.selectbox(
            "Rank artists by",
            [label for label, _ in metrics],
            key="artist_rank_metric",
        )
    with c2:
        top_n = st.slider("Top N", 5, 20, 10, key="artist_top_n")

    rank_col = dict(metrics)[selected_label]
    top_artists = artist_df.nlargest(top_n, rank_col).sort_values(rank_col, ascending=True)

    fig = _horizontal_bar(
        top_artists, x=rank_col, y="Artist",
        title=f"Top {top_n} Artists by {selected_label}",
        height=max(380, top_n * 34),
        xaxis_title=selected_label,
    )
    fig.update_traces(text=top_artists[rank_col], texttemplate="%{text:.3s}",
                      textposition="outside", textfont_size=11)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # ── Artist Comparison ─────────────────────────────────────────────────────

    st.markdown("### 🔀 Artist Comparison")

    all_artists = artist_df["Artist"].tolist()
    default_picks = all_artists[:min(3, len(all_artists))]

    selected_artists = st.multiselect(
        "Select up to 3 artists to compare",
        options=all_artists,
        default=default_picks,
        max_selections=3,
        key="artist_compare",
    )

    if len(selected_artists) >= 2:
        compare_df = artist_df[artist_df["Artist"].isin(selected_artists)]
        compare_cols = [c for c in metric_cols if c in compare_df.columns
                        and c != "Total Playlist Count"]

        # Normalise to 0-100 so bars are visually comparable across metrics
        melted = compare_df.melt(
            id_vars="Artist", value_vars=compare_cols,
            var_name="Metric", value_name="Value",
        )
        melted["Raw"] = melted["Value"].copy()
        for m in compare_cols:
            mask = melted["Metric"] == m
            max_val = melted.loc[mask, "Value"].max()
            if max_val > 0:
                melted.loc[mask, "Value"] = melted.loc[mask, "Value"] / max_val * 100

        fig_cmp = px.bar(
            melted, x="Metric", y="Value", color="Artist",
            barmode="group",
            color_discrete_sequence=SPOTIFY_COLORS,
            custom_data=["Raw"],
            labels={"Value": "Relative %", "Metric": ""},
        )
        fig_cmp.update_traces(
            hovertemplate="%{x}: %{customdata[0]:,.0f}<extra>%{fullData.name}</extra>",
        )
        fig_cmp.update_layout(
            **PLOTLY_LAYOUT,
            title="Artist Head-to-Head",
            height=420,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            yaxis_title="Relative Scale (0–100)",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Select at least **2** artists to see a head-to-head comparison.")

    st.markdown("---")

    # ── Artist Insights ───────────────────────────────────────────────────────

    st.markdown("### 💡 Artist Insights")

    insights: list[str] = []

    # Highest-streamed artist (always first row since artist_df is sorted)
    if "Total Streams" in artist_df.columns and len(artist_df) > 0:
        top_row = artist_df.iloc[0]
        insights.append(
            f"**Highest-streamed artist:** {top_row['Artist']} "
            f"with **{_fmt_big(top_row['Total Streams'])}** total streams"
        )

    # Average tracks per artist
    insights.append(f"**Average tracks per artist:** {artist_df['Tracks'].mean():.1f}")

    # Data-driven "best of" insights
    best_of = [
        ("Highest avg track score", "Avg Track Score", ".1f"),
        ("Highest avg popularity",  "Avg Popularity",  ".1f"),
    ]
    for label, col, fmt in best_of:
        if col in artist_df.columns:
            row = artist_df.loc[artist_df[col].idxmax()]
            insights.append(f"**{label}:** {row['Artist']} ({row[col]:{fmt}})")

    # Gap between #1 and #2 by streams
    if "Total Streams" in artist_df.columns and len(artist_df) >= 2:
        first, second = artist_df.iloc[0], artist_df.iloc[1]
        gap = first["Total Streams"] - second["Total Streams"]
        insights.append(
            f"**Gap between #1 and #2:** {_fmt_big(gap)} streams "
            f"({first['Artist']} leads {second['Artist']})"
        )

    for insight in insights:
        st.markdown(f"• {insight}")


# ─── Section 3: Correlation Heatmap ──────────────────────────────────────────

def render_correlation_heatmap(df: pd.DataFrame):
    """Render a Plotly correlation heatmap for numeric features."""
    st.markdown("## 🔥 Feature Correlation Heatmap")
    st.caption("Pearson correlations across streaming & platform metrics")

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != "Release Year"]
    if len(numeric_cols) < 2:
        return

    # Let user pick which columns to include
    default_cols = numeric_cols[:12]
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

    # Lower-triangle mask — set upper triangle to NaN
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)

    fig = px.imshow(
        corr_masked,
        text_auto=".2f",
        color_continuous_scale=["#16213e", "#0e1117", "#1ed760"],
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Feature Correlation Matrix",
        height=max(500, len(selected) * 45),
        coloraxis_colorbar=dict(title="Pearson r"),
    )
    fig.update_traces(
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Section 4: Distribution Explorer ────────────────────────────────────────

def render_distributions(df: pd.DataFrame):
    """Render distribution histograms/violins and Top-N track rankings."""
    st.markdown("## 📊 Distribution Explorer")

    tab_dist, tab_topn = st.tabs(["📈 Distributions", "🏆 Top-N Rankings"])

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != "Release Year"]

    with tab_dist:
        if not numeric_cols:
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

        label_col = _get_track_col(df)
        if label_col is None:
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

        fig = _horizontal_bar(
            top, x=rank_metric, y="label",
            title=f"Top {top_n} Tracks by {rank_metric}",
            height=max(350, top_n * 28),
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
            🎧 Spotify 2024 · Multi-Platform Analytics
        </h1>
        <p style='color:#b3b3b3; font-size:0.95rem; margin-top:4px;'>
            Exploring streaming patterns across Spotify, YouTube, TikTok & more
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
    filtered = render_sidebar(df)

    if filtered.empty:
        st.warning("No tracks match the current filters. Adjust the sidebar.")
        st.stop()

    # ── KPIs ──
    render_kpis(filtered)

    st.markdown("---")

    # ── Artist Analytics ──
    render_artist_analytics(filtered)

    st.markdown("---")

    # ── Correlation Heatmap ──
    render_correlation_heatmap(filtered)

    st.markdown("---")

    # ── Distribution Explorer ──
    render_distributions(filtered)

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

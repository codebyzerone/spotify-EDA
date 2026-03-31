"""
data_loader.py — Vectorized data loading & cleaning for Spotify 2024 dataset.
"""

import pandas as pd
import numpy as np
import streamlit as st
import zipfile
import io
from pathlib import Path

# ─── Column Constants ─────────────────────────────────────────────────────────

AUDIO_FEATURES = [
    "Danceability", "Energy", "Valence", "Acousticness",
    "Instrumentalness", "Liveness", "Speechiness",
]

# Alternate column names (some Kaggle CSVs use % suffix)
AUDIO_FEATURES_PCT = [f"{f} %" for f in AUDIO_FEATURES]
AUDIO_FEATURES_UNDERSCORE = [f"{f.lower()}_%"  for f in AUDIO_FEATURES]

PLATFORM_METRICS = {
    "spotify": [
        "Spotify Streams", "Spotify Playlist Count",
        "Spotify Playlist Reach", "Spotify Popularity",
    ],
    "youtube": [
        "YouTube Views", "YouTube Likes", "YouTube Playlist Reach",
    ],
    "tiktok": [
        "TikTok Posts", "TikTok Likes", "TikTok Views",
    ],
}

ALL_PLATFORM_COLS = [c for cols in PLATFORM_METRICS.values() for c in cols]

NUMERIC_COLS = ALL_PLATFORM_COLS + AUDIO_FEATURES

ID_COLS = ["Track", "Album Name", "Artist", "Release Date", "ISRC"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalise_audio_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map alternate audio-feature column names to canonical names."""
    for canon, pct, uscore in zip(
        AUDIO_FEATURES, AUDIO_FEATURES_PCT, AUDIO_FEATURES_UNDERSCORE
    ):
        if canon not in df.columns:
            for alt in (pct, uscore):
                if alt in df.columns:
                    df = df.rename(columns={alt: canon})
                    break
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Vectorised coercion of object columns to numeric."""
    present = [c for c in cols if c in df.columns]
    for c in present:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset …")
def load_data(path: str | Path) -> pd.DataFrame:
    """Read CSV (or ZIP containing a CSV) with encoding fallback."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        return _read_csv_from_zip(path)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with any known encoding.")


def _read_csv_from_zip(path: Path) -> pd.DataFrame:
    """Open a ZIP, find the first CSV inside, and read it."""
    with zipfile.ZipFile(path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found inside {path}")
        with zf.open(csv_names[0]) as f:
            raw = f.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode CSV inside {path} with any known encoding.")


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded file (CSV or ZIP) from Streamlit's file_uploader."""
    name = uploaded_file.name.lower()
    if name.endswith(".zip"):
        raw_bytes = uploaded_file.read()
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV file found inside the uploaded ZIP.")
            with zf.open(csv_names[0]) as f:
                csv_bytes = f.read()
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(io.BytesIO(csv_bytes), encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV inside the uploaded ZIP.")
    else:
        return pd.read_csv(uploaded_file)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised cleaning pipeline:
    1. Strip whitespace from string columns
    2. Normalise audio-feature column names
    3. Coerce numeric columns
    4. Parse release dates
    5. Drop rows that are entirely null
    6. Fill remaining NaN numerics with column median
    """
    df = df.copy()

    # 1. Strip whitespace
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # 2. Normalise audio feature column names
    df = _normalise_audio_cols(df)

    # 3. Coerce numeric
    df = _coerce_numeric(df, NUMERIC_COLS)

    # 4. Parse dates
    if "Release Date" in df.columns:
        df["Release Date"] = pd.to_datetime(
            df["Release Date"], errors="coerce", infer_datetime_format=True
        )
        df["Release Year"] = df["Release Date"].dt.year.astype("Int64")

    # 5. Drop all-null rows
    df = df.dropna(how="all")

    # 6. Fill missing numerics with median
    num_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[num_present] = df[num_present].fillna(df[num_present].median())

    return df


def get_audio_features(df: pd.DataFrame) -> list[str]:
    """Return the audio-feature columns that actually exist in *df*."""
    return [c for c in AUDIO_FEATURES if c in df.columns]


def get_platform_metrics(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {platform: [cols…]} for columns present in *df*."""
    return {
        plat: [c for c in cols if c in df.columns]
        for plat, cols in PLATFORM_METRICS.items()
    }

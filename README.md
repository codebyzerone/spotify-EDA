# 🎧 Spotify-EDA — What Makes a Song Blow Up?

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/codebyzerone/spotify-EDA/main/app.py)

I've always wondered why certain songs dominate every playlist while others disappear.
This project is my attempt to answer that using data — diving into Spotify's most
streamed songs to find patterns in what actually makes music stick.

## ✨ Live Demo

> **[Launch the Dashboard →](https://share.streamlit.io/codebyzerone/spotify-EDA/main/app.py)**
>
> Upload the Kaggle dataset when prompted and explore the interactive charts.

## What I Explored
- Which artists consistently top the charts (and by how much)
- How audio features like energy, danceability, and valence relate to streams
- Music trends across years — is any era actually louder?
- How Spotify popularity compares across TikTok, YouTube, and Apple Music

## Project Structure
```
spotify-eda/
├── app.py                 # Streamlit dashboard entry point
├── data_loader.py         # Dataset loading + vectorised cleanup
├── data/                  # Place your dataset here (or upload via the app)
├── requirements.txt       # Python dependencies
└── .streamlit/
    └── config.toml        # Dark theme + upload config
```

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack
Python · Pandas · Plotly · Matplotlib · Seaborn · Streamlit

## What's Next
- Pull live data using the Spotify Web API
- Maybe throw in a "predict a song's popularity" model if the features hold up

---
Dataset: [Most Streamed Spotify Songs 2024](https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024) via Kaggle

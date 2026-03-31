# 🎧 Spotify-EDA - What Makes a Song Blow Up?

I've always wondered why certain songs dominate every playlist while others disappear.
This project is my attempt to answer that using data — diving into Spotify's most
streamed songs to find patterns in what actually makes music stick.

## What I Explored
- Which artists consistently top the charts (and by how much)
- How audio features like energy, danceability, and valence relate to streams
- Music trends across years — is any era actually louder?
- How Spotify popularity compares across TikTok, YouTube, and Apple Music

## Project Structure
```
spotify-eda/
├── app.py              # entry point
├── data_loader.py      # handles dataset loading + cleanup
├── data/               # raw dataset lives here
└── requirements.txt
```

## Getting Started
```bash
pip install -r requirements.txt
python app.py
```

That's it. No config files, no setup headaches.

## Stack
Python · Pandas · Seaborn · Plotly · (Streamlit dashboard — coming soon)

## What's Next
- Wrap the whole thing into an interactive Streamlit dashboard
- Pull live data using the Spotify Web API
- Maybe throw in a "predict a song's popularity" model if the features hold up

---
Dataset: [Most Streamed Spotify Songs 2024](https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024) via Kaggle

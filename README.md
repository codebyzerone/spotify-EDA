# 🎧 Spotify EDA — What Makes a Song Blow Up?

> Diving into 100,000+ Spotify streams to find the patterns behind viral music.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/)

---

## 📌 The Question

Why do some songs dominate every playlist while others disappear after a week?

I've always been curious about this. So I stopped guessing and started analyzing — diving into Spotify's most streamed songs to find what actually makes music stick.

---

## 🔍 What I Found

- **Energy alone doesn't drive streams** — high-energy songs don't consistently outperform calm ones
- **Danceability matters more than loudness** — the most streamed songs cluster around 0.6-0.8 danceability
- **2019-2022 was the loudness peak** — music has measurably gotten louder then slightly retreated
- **Cross-platform popularity gaps are real** — TikTok virality doesn't always translate to Spotify streams
- **A handful of artists dominate disproportionately** — the top 1% of artists account for ~40% of total streams

---

## 📊 What I Explored

- Which artists consistently top the charts — and by how much
- How audio features (energy, danceability, valence, loudness) relate to stream counts
- Music trends across years — is any era actually louder?
- How Spotify popularity compares across TikTok, YouTube, and Apple Music
- Correlation analysis between audio features and commercial success

---

## 🖥️ Interactive Dashboard

Built with Streamlit + Plotly for fully interactive exploration:

```
[  SCREENSHOT PLACEHOLDER — paste your dashboard screenshot here  ]
```

**Dashboard features:**
- Filter by year, artist, genre
- Interactive scatter plots (audio features vs streams)
- Top artists bar chart (animated)
- Cross-platform popularity comparison
- Correlation heatmap

---

## 🏗️ Project Structure

```
spotify-eda/
│
├── app.py                  # Streamlit dashboard entry point
├── data_loader.py          # Dataset loading + cleaning pipeline
├── eda_notebook.ipynb      # Full exploratory analysis notebook
├── data/
│   └── spotify_2024.csv    # Raw dataset (from Kaggle)
├── screenshots/            # Dashboard screenshots
└── requirements.txt        # All dependencies
```

---

## 🚀 Quick Start

### 1 — Clone The Repo

```bash
git clone https://github.com/codebyzerone/spotify-EDA.git
cd spotify-EDA
```

### 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 3 — Run The Dashboard

```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

That's it. No config files, no setup headaches.

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Pandas | Data manipulation + cleaning |
| Plotly | Interactive visualizations |
| Streamlit | Web dashboard framework |
| Seaborn | Statistical plots |
| Matplotlib | Supporting charts |

---

## 📈 Key Visualizations

### 1. Audio Features vs Stream Count
Scatter plot showing how danceability, energy, and valence correlate with total streams. Colored by year of release.

### 2. Top Artists By Total Streams
Animated horizontal bar chart ranking artists by cumulative stream count across all their songs.

### 3. Cross-Platform Popularity
Side-by-side comparison of Spotify popularity scores vs TikTok views, YouTube views, and Apple Music rankings.

### 4. Feature Correlation Heatmap
Full correlation matrix of all audio features — revealing which combinations predict commercial success.

### 5. Music Loudness Over Time
Line chart showing average loudness (dB) per year from 2010-2024 — the loudness war visualized.

---

## 🧹 Data Cleaning Highlights

The raw dataset required significant preprocessing:

```python
# Key cleaning steps
- Removed duplicate entries (same song, multiple regions)
- Handled missing stream counts (< 2% of dataset)
- Normalized audio features to 0-1 scale
- Parsed release dates from multiple formats
- Merged cross-platform data on track + artist name
```

---

## 💡 Insights Summary

```
Most Streamed Audio Profile:
├── Danceability:  0.68 avg  (sweet spot: 0.6-0.8)
├── Energy:        0.64 avg  (moderate, not extreme)
├── Valence:       0.51 avg  (neither happy nor sad)
├── Loudness:     -5.2 dB avg
└── Tempo:        122 BPM avg
```

---

## 🗺️ Roadmap

- [x] Exploratory data analysis
- [x] Interactive Streamlit dashboard
- [x] Cross-platform comparison
- [x] Audio feature correlation analysis
- [ ] Live data via Spotify Web API
- [ ] Song popularity prediction model (ML)
- [ ] Genre-level breakdown
- [ ] Playlist inclusion analysis

---

## 📦 Dataset

**Most Streamed Spotify Songs 2024** via Kaggle

Dataset includes:
- Track name, artist, release date
- Stream counts (Spotify, YouTube, TikTok, Apple Music)
- Audio features (energy, danceability, valence, loudness, tempo, key)
- Playlist inclusion counts across platforms

---

## 👨‍💻 Built By

**Manodip Bhattacharjee**
B.Tech Computer Science & Engineering
Guru Nanak Institute of Technology, Kolkata

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Spotify EDA** · Data tells the story music feels

*What makes a song blow up? The data knows.*

</div>

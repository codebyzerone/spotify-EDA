# 🎵 Spotify Multi-Platform Analytics Dashboard

An interactive analytics dashboard built with **Python**, **Streamlit**, and **Plotly** to explore music performance across multiple streaming platforms using the **Spotify Most Streamed Songs 2024** dataset.

Instead of displaying static charts, the dashboard allows users to upload the dataset, filter artists, compare performance metrics, analyze correlations, and discover insights through interactive visualizations.

---

## 📸 Dashboard Preview

> *(Add screenshots here after deployment)*

### Dashboard Overview
![Dashboard](images/dashboard.png)

### Artist Analytics
![Artist Analytics](images/artist_analytics.png)

### Correlation Heatmap
![Correlation](images/correlation_heatmap.png)

### Distribution Explorer
![Distribution](images/distribution.png)

---

## ✨ Features

### 📂 Dataset Upload
- Upload the **Spotify Most Streamed Songs 2024** dataset (CSV or ZIP).
- Automatically loads and preprocesses the data.

### 🎤 Artist Analytics
- Analyze individual artist performance.
- View:
  - Total Tracks
  - Total Streams
  - Average Track Score
  - Average Popularity
  - Playlist Reach
- Compare artists using interactive tables and charts.

### 📊 Correlation Heatmap
Explore relationships between numerical features such as:
- Track Score
- Spotify Streams
- Playlist Reach
- Playlist Count
- Popularity

This helps identify which metrics are strongly related.

### 📈 Distribution Explorer
Visualize how different ranking metrics are distributed across the dataset.

Supports:
- Histograms
- Top-N Rankings

### 🏆 Ranking Analysis
Rank tracks using different metrics such as:
- Spotify Streams
- Track Score
- Spotify Popularity
- YouTube Views
- TikTok Posts
- Apple Music Playlist Count
- Deezer Playlist Count
- Amazon Playlist Count

### 💡 Artist Insights
Automatically summarizes key observations, including:
- Highest streamed artist
- Average track performance
- Popularity trends
- Playlist reach statistics

### 🎛 Interactive Filters
Filter the dashboard by:
- Artist
- Release Year

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data cleaning & analysis |
| Plotly Express | Interactive visualizations |
| Matplotlib | Statistical plotting |
| Seaborn | Correlation heatmaps |
| NumPy | Numerical operations |

---

# 📂 Project Structure

```text
spotify-multi-platform-analytics/

├── app.py                  # Streamlit application
├── data_loader.py          # Dataset loading & preprocessing
├── data/                   # Dataset location
├── requirements.txt
├── README.md
├── LICENSE
└── .streamlit/
    └── config.toml
```

---

# 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/codebyzerone/spotify-multi-platform-analytics.git
```

```bash
cd spotify-multi-platform-analytics
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 📊 Dataset

This project uses the **Spotify Most Streamed Songs 2024** dataset available on Kaggle.

Download the dataset and upload it through the dashboard.

---

# 📚 What I Learned

While building this project I gained practical experience with:

- Data preprocessing using Pandas
- Interactive dashboard development with Streamlit
- Data visualization using Plotly
- Statistical analysis using correlation matrices
- Aggregation and ranking techniques
- Building reusable and modular Python code
- Designing user-friendly analytical dashboards

---

# 🔮 Future Improvements

Some ideas that could further extend this project:

- Export charts as images or PDFs
- Advanced filtering options
- Search by song title
- Additional statistical insights

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Manodip Bhattacharjee**

Computer Science Engineering Student

GitHub: https://github.com/codebyzerone

LinkedIn: www.linkedin.com/in/manodip-bhattacharjee-651b09360
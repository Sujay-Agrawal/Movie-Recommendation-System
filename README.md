# 🎬 CineMatch - Movie Recommendation System

A content-based movie recommender with a clean Streamlit interface. Pick a movie you like, or just describe what you feel like watching, and CineMatch suggests similar movies with posters, ratings and genres.

Built on the **TMDB 5000** dataset using **TF-IDF + cosine similarity**.

---

## ✨ Features

- **Search by movie** - choose any title from a searchable dropdown and get similar movies with a match percentage.
- **Describe what you want** - type a plot, mood or theme (e.g. *"space crew fights alien on a ship"*) and get matching movies.
- **Top rated** - browse the highest-rated movies in the dataset.
- **Movie posters** - fetched from the TMDB API, loaded in parallel and cached.
- **Sidebar filters** - number of results, minimum rating, genre filter, sort by similarity / rating / popularity, and an optional rating boost.
- **Responsive card UI** - poster, rating, popularity, genre chips, tagline and overview on every card.

---

## 🧠 How It Works

1. Each movie's **overview, genres and tagline** are combined into one text field (`tags`).
2. The text is lowercased, stop words are removed and words are lemmatized (NLTK).
3. A **TF-IDF vectorizer** (up to 50,000 features, unigrams + bigrams) turns every movie into a vector.
4. To recommend, the app compares the chosen movie's vector with all others using **cosine similarity** (`linear_kernel`) and returns the closest matches.
5. For free-text search, your query goes through the same saved vectorizer and is compared against every movie.

Only one row is compared at a time, so no large similarity matrix is stored in memory.

---

## 📁 Project Structure

```
movie-app/
├── app.py                  # Streamlit application
├── movie_list.pkl          # Processed DataFrame (title, genres, overview, tagline, vote_average, popularity, tags)
├── tfidf_vectorizer.pkl    # Fitted TfidfVectorizer
├── tfidf_matrix.pkl        # TF-IDF matrix for all movies
├── indices.pkl             # Title -> row index mapping
├── movie_recommendation.ipynb  # Notebook used to build the pickle files
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── secrets.toml        # (optional) your TMDB API key - do NOT commit this
```

---

## 🚀 Getting Started

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd movie-app
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** the pickle files must be loaded with a compatible scikit-learn version. If you see version warnings or errors, install the version you used when creating the files (in your notebook run `import sklearn; print(sklearn.__version__)`), e.g. `pip install scikit-learn==1.6.1`.

### 4. Get a free TMDB API key (for posters)

1. Create an account at [themoviedb.org](https://www.themoviedb.org/).
2. Go to **Settings → API** and request a developer key (free for personal use).
3. Copy the **API Key** (the short 32-character one, *not* the long Read Access Token).

### 5. Add the key to the app

Choose **one** of these:

**Option A - Sidebar box (quickest):** run the app and paste the key into the *TMDB API key* field in the sidebar. You'll need to re-enter it after every restart.

**Option B - Secrets file (recommended):** create `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_api_key_here"
```

**Option C - Environment variable:**

```bash
# Windows (PowerShell)
$env:TMDB_API_KEY="your_api_key_here"

# Mac / Linux
export TMDB_API_KEY="your_api_key_here"
```

### 6. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🔄 Rebuilding the Pickle Files

The pickle files are produced by `movie_recommendation.ipynb`, which:

1. Loads `tmdb_5000_merged.csv` (TMDB 5000 movies + credits merged).
2. Keeps `title, genres, overview, tagline, vote_average, popularity`.
3. Cleans missing values and converts the genres JSON into plain words.
4. Builds and preprocesses the `tags` text.
5. Fits the TF-IDF vectorizer and saves the four `.pkl` files.

Re-run the notebook whenever you change the data or preprocessing, then replace the `.pkl` files next to `app.py`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and the four `.pkl` files to a GitHub repository (GitHub's file limit is 100 MB per file, so use [Git LFS](https://git-lfs.com/) if `tfidf_matrix.pkl` is larger).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a new app from your repo, with `app.py` as the main file.
3. In **Advanced settings → Secrets**, add:
   ```toml
   TMDB_API_KEY = "your_api_key_here"
   ```
4. Deploy.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Missing file: ...pkl` | Keep all four `.pkl` files in the same folder as `app.py` and run `streamlit run app.py` from that folder. |
| Pickle / scikit-learn version error | Install the same scikit-learn version used to create the files. |
| Posters show a 🎬 placeholder | Check the TMDB key is entered, is the **API Key** (not the Read Access Token), and has had a few minutes to activate. |
| Wrong poster for a movie | Posters are matched by title through TMDB search. Remakes or movies sharing a name can be mismatched; adding the TMDB `id` to your DataFrame would make it exact. |
| Card layout looks broken | Make sure you are using the latest `app.py`. |

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) - UI
- [scikit-learn](https://scikit-learn.org/) - TF-IDF and cosine similarity
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) - data handling
- [NLTK](https://www.nltk.org/) - text preprocessing (in the notebook)
- [TMDB API](https://developer.themoviedb.org/) - movie posters

---

## 🔮 Ideas for Improvement

- Add spaces between overview, genres and tagline when building `tags` (they are currently joined without a separator, which glues some words together).
- Include cast, crew and keywords in the tags for better recommendations.
- Store the TMDB `id` or `poster_path` in the DataFrame for exact posters.
- Add "add to watchlist" and a movie details page.

---

## 📜 Credits & Attribution

- Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
- *This product uses the TMDB API but is not endorsed or certified by TMDB.*

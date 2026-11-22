import html
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel

# --------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #1f1c2c 0%, #3a2a5e 50%, #8e3a59 100%);
        color: #fff;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2.4rem; }
    .hero p  { margin: .4rem 0 0 0; opacity: .85; font-size: 1.05rem; }

    .selected-card {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        border: 1px solid rgba(150,150,150,.3);
        background: rgba(120,90,200,.08);
        margin-bottom: 1.2rem;
    }
    .movie-card {
        height: 100%;
        padding: 1rem 1.1rem;
        border-radius: 14px;
        border: 1px solid rgba(150,150,150,.25);
        background: rgba(150,150,150,.06);
        transition: transform .15s ease, box-shadow .15s ease;
        margin-bottom: 1rem;
    }
    .movie-card:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(0,0,0,.18); }
    .movie-title { font-size: 1.1rem; font-weight: 700; margin-bottom: .2rem; }
    .tagline { font-style: italic; opacity: .7; font-size: .88rem; margin-bottom: .5rem; }
    .overview { font-size: .88rem; opacity: .9; line-height: 1.4; margin-top: .5rem; }
    .chip {
        display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0;
        border-radius: 999px; font-size: .75rem;
        background: rgba(142,58,89,.18); border: 1px solid rgba(142,58,89,.45);
    }
    .badge {
        display: inline-block; padding: 2px 10px; margin-right: 6px;
        border-radius: 8px; font-size: .8rem; font-weight: 600;
    }
    .rating  { background: #f5c518; color: #111; }
    .match   { background: #2e7d32; color: #fff; }
    .pop     { background: rgba(120,120,120,.25); }
    .card-row { display: flex; gap: .9rem; }
    .card-body { flex: 1; min-width: 0; }
    .poster, .poster-ph {
        width: 92px; min-width: 92px; height: 138px; border-radius: 8px; object-fit: cover;
        box-shadow: 0 4px 12px rgba(0,0,0,.3);
    }
    .poster-ph {
        display: flex; align-items: center; justify-content: center; font-size: 2rem;
        background: linear-gradient(135deg, #3a2a5e, #8e3a59);
    }
    .poster-lg, .poster-lg-ph { width: 150px; min-width: 150px; height: 225px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _load(path: Path):
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource(show_spinner="Loading movie data...")
def load_artifacts(folder: str = "."):
    base = Path(folder)
    movies = _load(base / "movie_list.pkl")
    vectorizer = _load(base / "tfidf_vectorizer.pkl")
    matrix = _load(base / "tfidf_matrix.pkl")
    indices = _load(base / "indices.pkl")

    if not isinstance(movies, pd.DataFrame):
        movies = pd.DataFrame(movies)
    movies = movies.reset_index(drop=True)

    for col in ["title", "genres", "overview", "tagline"]:
        if col not in movies.columns:
            movies[col] = ""
        movies[col] = movies[col].fillna("")
    for col in ["vote_average", "popularity"]:
        if col not in movies.columns:
            movies[col] = 0.0
        movies[col] = pd.to_numeric(movies[col], errors="coerce").fillna(0.0)

    # Title -> row position lookup
    if isinstance(indices, pd.Series):
        title_to_idx = indices[~indices.index.duplicated(keep="first")]
    else:
        title_to_idx = pd.Series(range(len(movies)), index=movies["title"])
        title_to_idx = title_to_idx[~title_to_idx.index.duplicated(keep="first")]

    return movies, vectorizer, matrix, title_to_idx


def genre_list(value) -> list:
    """genres may be a list, a 'Action Adventure' string, or 'Action, Adventure'."""
    if isinstance(value, (list, tuple, set)):
        items = []
        for g in value:
            items.append(g.get("name", "") if isinstance(g, dict) else str(g))
        return [g for g in items if g]
    text = str(value).replace("|", ",")
    parts = [p.strip() for p in text.split(",")] if "," in text else text.split()
    return [p for p in parts if p]


try:
    movies, vectorizer, tfidf_matrix, title_to_idx = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Keep the 4 .pkl files in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the pickle files: {e}")
    st.stop()

if "genre_sets" not in st.session_state:
    st.session_state["genre_sets"] = movies["genres"].apply(lambda v: set(genre_list(v)))
genre_sets = st.session_state["genre_sets"]
all_genres = sorted(set().union(*genre_sets.tolist())) if len(genre_sets) else []


# --------------------------------------------------------------------------
# Recommendation logic
# --------------------------------------------------------------------------
def rank(sim_scores: np.ndarray, exclude_idx, n, min_rating, genres, rating_weight, sort_by):
    scores = sim_scores.copy()

    if rating_weight > 0:
        r = movies["vote_average"].to_numpy()
        r_norm = r / (r.max() if r.max() > 0 else 1)
        scores = (1 - rating_weight) * scores + rating_weight * r_norm * (scores > 0)

    mask = np.ones(len(movies), dtype=bool)
    mask &= movies["vote_average"].to_numpy() >= min_rating
    if genres:
        mask &= genre_sets.apply(lambda s: bool(s & set(genres))).to_numpy()
    if exclude_idx is not None:
        mask[exclude_idx] = False

    candidate = np.where(mask & (sim_scores > 0))[0]
    if len(candidate) == 0:
        return pd.DataFrame()

    if sort_by == "Rating":
        order = candidate[np.argsort(-movies["vote_average"].to_numpy()[candidate])]
    elif sort_by == "Popularity":
        order = candidate[np.argsort(-movies["popularity"].to_numpy()[candidate])]
    else:
        order = candidate[np.argsort(-scores[candidate])]
    order = order[:n]

    out = movies.iloc[order].copy()
    out["match"] = sim_scores[order]
    return out


def similar_to_movie(title, **kw):
    idx = int(title_to_idx[title])
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).ravel()
    return idx, rank(sims, exclude_idx=idx, **kw)


def similar_to_text(text, **kw):
    vec = vectorizer.transform([text])
    sims = linear_kernel(vec, tfidf_matrix).ravel()
    return rank(sims, exclude_idx=None, **kw)


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x))


def truncate(text, n=220):
    text = str(text)
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def chips(value):
    return "".join(f'<span class="chip">{esc(g)}</span>' for g in genre_list(value))


@st.cache_resource
def _poster_store():
    return {}


def _tmdb_lookup(title, key):
    """Returns poster URL, '' if TMDB has no poster, None on error (so it is retried later)."""
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": key, "query": title},
            timeout=6,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w342" + results[0]["poster_path"]
        return ""
    except Exception:
        return None


def get_posters(titles):
    """Fetch posters for many titles in parallel; cached across reruns."""
    titles = list(titles)
    if not tmdb_key:
        return {t: "" for t in titles}
    store = _poster_store()
    missing = [t for t in set(titles) if t not in store]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for t, url in zip(missing, ex.map(lambda t: _tmdb_lookup(t, tmdb_key), missing)):
                if url is not None:
                    store[t] = url
    return {t: store.get(t, "") for t in titles}


def flat(s: str) -> str:
    """Collapse HTML to one line: blank/indented lines would be parsed as Markdown code blocks."""
    return "".join(line.strip() for line in s.splitlines())


def poster_html(url, large=False):
    cls = "poster poster-lg" if large else "poster"
    ph = "poster-ph poster-lg-ph" if large else "poster-ph"
    if url:
        return f'<img class="{cls}" src="{esc(url)}" loading="lazy">'
    return f'<div class="{ph}">🎬</div>'


def movie_card(row, show_match=True, poster=""):
    match_badge = f'<span class="badge match">{row["match"]*100:.0f}% match</span>' if show_match and "match" in row else ""
    tagline = f'<div class="tagline">{esc(row["tagline"])}</div>' if row["tagline"] else ""
    return flat(f"""
    <div class="movie-card">
      <div class="card-row">
        {poster_html(poster)}
        <div class="card-body">
            <div class="movie-title">{esc(row['title'])}</div>
            {tagline}
            <div>{match_badge}<span class="badge rating">⭐ {row['vote_average']:.1f}</span>
            <span class="badge pop">🔥 {row['popularity']:.0f}</span></div>
            <div style="margin-top:.4rem">{chips(row['genres'])}</div>
        </div>
      </div>
      <div class="overview">{esc(truncate(row['overview']))}</div>
    </div>
    """)


def render_grid(df, cols_per_row=3, show_match=True):
    if df.empty:
        st.info("No movies matched. Try lowering the minimum rating or clearing the genre filter.")
        return
    posters = get_posters(df["title"].tolist())
    rows = list(df.itertuples(index=False))
    for i in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, r in zip(cols, rows[i : i + cols_per_row]):
            with col:
                row = pd.Series(r._asdict())
                st.markdown(
                    movie_card(row, show_match, posters.get(row["title"], "")),
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Filters")
    n_recs = st.slider("Number of recommendations", 3, 30, 9, step=3)
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, step=0.5)
    sel_genres = st.multiselect("Limit to genres", all_genres)
    sort_by = st.radio("Sort results by", ["Similarity", "Rating", "Popularity"], horizontal=True)
    rating_weight = st.slider(
        "Boost highly-rated movies", 0.0, 0.5, 0.0, step=0.05,
        help="Blends the movie's rating into the similarity score (only used when sorting by Similarity).",
    )
    st.divider()
    st.subheader("🖼️ Posters")
    _default_key = os.environ.get("TMDB_API_KEY", "")
    try:
        _default_key = st.secrets.get("TMDB_API_KEY", _default_key)
    except Exception:
        pass
    tmdb_key = st.text_input(
        "TMDB API key",
        value=_default_key,
        type="password",
        help="Free key from themoviedb.org/settings/api. Or set TMDB_API_KEY as an env variable / in .streamlit/secrets.toml.",
    ).strip()
    if not tmdb_key:
        st.caption("Add a TMDB key to show posters.")
    st.divider()
    st.caption(f"📚 {len(movies):,} movies in the database")

opts = dict(
    n=n_recs, min_rating=min_rating, genres=sel_genres,
    rating_weight=rating_weight if sort_by == "Similarity" else 0.0, sort_by=sort_by,
)

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 CineMatch</h1>
        <p>Pick a movie you love, or describe what you feel like watching, and get recommendations instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_movie, tab_text, tab_top = st.tabs(["🔎 Search by movie", "✍️ Describe what you want", "🏆 Top rated"])

# ---- Tab 1: by movie ------------------------------------------------------
with tab_movie:
    titles = sorted(title_to_idx.index.astype(str).unique())
    choice = st.selectbox(
        "Search for a movie (type to filter)",
        options=titles,
        index=None,
        placeholder="e.g. The Dark Knight, Toy Story, Inception...",
    )

    if choice:
        idx, recs = similar_to_movie(choice, **opts)
        sel = movies.iloc[idx]

        sel_poster = get_posters([sel["title"]]).get(sel["title"], "")
        st.markdown(
            flat(f"""
            <div class="selected-card">
              <div class="card-row">
                {poster_html(sel_poster, large=True)}
                <div class="card-body">
                    <div style="font-size:.8rem;opacity:.7">YOU SELECTED</div>
                    <div style="font-size:1.6rem;font-weight:800">{esc(sel['title'])}</div>
                    <div class="tagline">{esc(sel['tagline'])}</div>
                    <div><span class="badge rating">⭐ {sel['vote_average']:.1f}</span>
                    <span class="badge pop">🔥 {sel['popularity']:.0f}</span> {chips(sel['genres'])}</div>
                    <div class="overview" style="margin-top:.6rem">{esc(sel['overview'])}</div>
                </div>
              </div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        st.subheader(f"Because you liked *{choice}*")
        render_grid(recs)
    else:
        st.info("👆 Choose a movie to see similar recommendations.")

# ---- Tab 2: by free text --------------------------------------------------
with tab_text:
    query = st.text_input(
        "Describe a plot, mood, or theme",
        placeholder="e.g. space crew fights alien on a ship, heist with a twist, feel-good animated family adventure",
    )
    if query.strip():
        recs = similar_to_text(query, **opts)
        st.subheader("Movies matching your description")
        render_grid(recs)
    else:
        st.info("Type a few keywords above to find matching movies.")

# ---- Tab 3: top rated -----------------------------------------------------
with tab_top:
    st.caption("Highest rated movies in the dataset (filters from the sidebar apply).")
    pool = movies.copy()
    pool = pool[pool["vote_average"] >= min_rating]
    if sel_genres:
        pool = pool[genre_sets.loc[pool.index].apply(lambda s: bool(s & set(sel_genres)))]
    pool = pool.sort_values(["vote_average", "popularity"], ascending=False).head(n_recs)
    render_grid(pool, show_match=False)

st.markdown("---")
st.caption("Built with Streamlit · TF-IDF + cosine similarity on genres, overview and tagline")
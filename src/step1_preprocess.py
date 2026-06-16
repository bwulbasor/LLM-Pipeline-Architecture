"""Step 1: Download MovieLens 10M, filter, engineer features, and cluster archetypes."""
import json
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (
    APPLY_KNOWLEDGE_CUTOFF,
    AUTO_SELECT_K,
    CLUSTER_K_CANDIDATES,
    DATA_DIR,
    FIGURES_DIR,
    MIN_ITEM_RATINGS,
    MIN_USER_RATINGS,
    MODEL_KNOWLEDGE_CUTOFF_YEAR,
    MOVIELENS_DIR,
    MOVIELENS_URL,
    N_ARCHETYPES,
    RANDOM_SEED,
    RESULTS_DIR,
    SEED_USERS_PER_ARCHETYPE,
)

np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid")

ARCHETYPE_LABELS = [
    "Blockbuster Follower",
    "Niche Explorer",
    "Genre Specialist",
    "Casual Positive Rater",
    "Critical Analyst",
]


def download_movielens():
    """Download and extract MovieLens 10M if not present."""
    zip_path = DATA_DIR / "ml-10m.zip"
    if MOVIELENS_DIR.exists():
        print("MovieLens 10M already exists, skipping download.")
        return

    print("Downloading MovieLens 10M (~63MB)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(MOVIELENS_URL, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            handle.write(chunk)

    print("Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(DATA_DIR)
    zip_path.unlink()
    print("Done.")


def load_ratings():
    ratings_file = MOVIELENS_DIR / "ratings.dat"
    print(f"Loading ratings from {ratings_file}...")
    ratings = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    print(
        f"  Loaded {len(ratings):,} ratings from "
        f"{ratings['userId'].nunique():,} users on {ratings['movieId'].nunique():,} movies"
    )
    return ratings


def extract_release_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else np.nan


def load_movies():
    movies_file = MOVIELENS_DIR / "movies.dat"
    movies = pd.read_csv(
        movies_file,
        sep="::",
        header=None,
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    movies["release_year"] = movies["title"].apply(extract_release_year)
    return movies


def load_tags():
    tags_file = MOVIELENS_DIR / "tags.dat"
    tags = pd.read_csv(
        tags_file,
        sep="::",
        header=None,
        names=["userId", "movieId", "tag", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    return tags


def apply_movie_cutoff(ratings_df, movies_df):
    """Filter out movies beyond the configured model knowledge cutoff year."""
    if not APPLY_KNOWLEDGE_CUTOFF:
        return ratings_df, movies_df

    eligible_movies = movies_df[
        movies_df["release_year"].isna() | (movies_df["release_year"] <= MODEL_KNOWLEDGE_CUTOFF_YEAR)
    ]["movieId"]
    filtered_ratings = ratings_df[ratings_df["movieId"].isin(eligible_movies)].copy()
    filtered_movies = movies_df[movies_df["movieId"].isin(eligible_movies)].copy()
    removed = ratings_df["movieId"].nunique() - filtered_movies["movieId"].nunique()
    print(
        f"Applied knowledge cutoff <= {MODEL_KNOWLEDGE_CUTOFF_YEAR}: "
        f"removed {max(0, removed)} movies outside the cutoff."
    )
    return filtered_ratings, filtered_movies


def filter_data(ratings_df):
    """Filter to power users and popular items."""
    user_counts = ratings_df["userId"].value_counts()
    power_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    item_counts = ratings_df["movieId"].value_counts()
    popular_items = item_counts[item_counts >= MIN_ITEM_RATINGS].index

    filtered = ratings_df[
        ratings_df["userId"].isin(power_users) &
        ratings_df["movieId"].isin(popular_items)
    ].copy()
    print(
        f"  After filtering: {len(filtered):,} ratings, "
        f"{filtered['userId'].nunique():,} users, {filtered['movieId'].nunique():,} movies"
    )
    return filtered


def engineer_features(ratings_df, movies_df):
    """Build per-user behavioral feature vectors."""
    print("Engineering user features...")

    all_genres = sorted({genre for genres in movies_df["genres"].dropna() for genre in genres.split("|")})
    print(f"  Found {len(all_genres)} genres")

    user_stats = ratings_df.groupby("userId")["rating"].agg(
        mean_rating="mean",
        std_rating="std",
        n_ratings="count",
    ).reset_index()
    user_stats["std_rating"] = user_stats["std_rating"].fillna(0.0)

    ratings_with_genres = ratings_df[["userId", "movieId"]].merge(
        movies_df[["movieId", "genres"]], on="movieId", how="left"
    )
    genre_exploded = ratings_with_genres.assign(
        genre=ratings_with_genres["genres"].fillna("(no genres listed)").str.split("|")
    ).explode("genre")

    genre_counts = genre_exploded.groupby(["userId", "genre"]).size().unstack(fill_value=0)
    for genre in all_genres:
        if genre not in genre_counts.columns:
            genre_counts[genre] = 0
    genre_counts = genre_counts[all_genres]

    row_sums = genre_counts.sum(axis=1).replace(0, np.nan)
    genre_props = genre_counts.div(row_sums, axis=0).fillna(0)
    inverse_simpson = 1.0 / (genre_props.pow(2).sum(axis=1).replace(0, np.nan))
    inverse_simpson = inverse_simpson.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    genre_props.columns = [f"genre_{genre}" for genre in all_genres]
    genre_props["inverse_simpson"] = inverse_simpson.values
    genre_props = genre_props.reset_index()

    features = user_stats.merge(genre_props, on="userId", how="inner")
    print(f"  Built features for {len(features):,} users")
    return features


def choose_cluster_count(x_scaled):
    """Pick k using silhouette score diagnostics, optionally fixed to N_ARCHETYPES."""
    silhouette_scores = {}
    for k in CLUSTER_K_CANDIDATES:
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = model.fit_predict(x_scaled)
        silhouette_scores[k] = silhouette_score(x_scaled, labels)
        print(f"  k={k}: silhouette={silhouette_scores[k]:.4f}")

    if AUTO_SELECT_K:
        best_k = max(silhouette_scores, key=silhouette_scores.get)
    else:
        best_k = N_ARCHETYPES
    print(f"  Using k={best_k}")
    return best_k, silhouette_scores


def cluster_archetypes(features_df):
    """K-Means clustering to discover behavioral archetypes."""
    print("Clustering user archetypes...")
    feature_cols = [column for column in features_df.columns if column != "userId"]
    x = features_df[feature_cols].values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    best_k, silhouette_scores = choose_cluster_count(x_scaled)
    model = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    features_df = features_df.copy()
    features_df["cluster"] = model.fit_predict(x_scaled)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Cluster-count diagnostics")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "silhouette_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    x_2d = tsne.fit_transform(x_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=features_df["cluster"], cmap="Set2", alpha=0.5, s=8)
    ax.set_title("t-SNE visualization of user archetypes")
    plt.colorbar(scatter, label="Cluster")
    fig.savefig(FIGURES_DIR / "tsne_archetypes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return features_df, model, scaler, silhouette_scores


def label_clusters(features_df):
    """Assign stable archetype names by sorting clusters on interpretable statistics."""
    cluster_stats = []
    genre_cols = [column for column in features_df.columns if column.startswith("genre_")]
    for cluster_id, group in features_df.groupby("cluster"):
        cluster_stats.append({
            "cluster_id": int(cluster_id),
            "mean_rating": float(group["mean_rating"].mean()),
            "std_rating": float(group["std_rating"].mean()),
            "inverse_simpson": float(group["inverse_simpson"].mean()),
            "specialization": float(group[genre_cols].mean().max()),
            "size": int(len(group)),
        })

    if len(cluster_stats) == 5:
        remaining = {row["cluster_id"]: row for row in cluster_stats}
        label_map = {}

        casual = max(remaining.values(), key=lambda row: (row["mean_rating"], -row["std_rating"]))
        label_map[casual["cluster_id"]] = "Casual Positive Rater"
        remaining.pop(casual["cluster_id"])

        critical = min(remaining.values(), key=lambda row: (row["mean_rating"], -row["std_rating"]))
        label_map[critical["cluster_id"]] = "Critical Analyst"
        remaining.pop(critical["cluster_id"])

        blockbuster = min(remaining.values(), key=lambda row: (row["inverse_simpson"], -row["mean_rating"]))
        label_map[blockbuster["cluster_id"]] = "Blockbuster Follower"
        remaining.pop(blockbuster["cluster_id"])

        specialist = max(remaining.values(), key=lambda row: (row["specialization"], -row["inverse_simpson"]))
        label_map[specialist["cluster_id"]] = "Genre Specialist"
        remaining.pop(specialist["cluster_id"])

        for cluster_id in remaining:
            label_map[cluster_id] = "Niche Explorer"
    else:
        label_map = {
            row["cluster_id"]: ARCHETYPE_LABELS[idx] if idx < len(ARCHETYPE_LABELS) else f"Archetype_{idx}"
            for idx, row in enumerate(sorted(cluster_stats, key=lambda item: item["cluster_id"]))
        }

    return label_map


def interpret_archetypes(features_df, label_map, silhouette_scores):
    cluster_profiles = []
    genre_cols = [column for column in features_df.columns if column.startswith("genre_")]
    for cluster_id in sorted(features_df["cluster"].unique()):
        cluster_data = features_df[features_df["cluster"] == cluster_id]
        top_genres = cluster_data[genre_cols].mean().nlargest(3)
        profile = {
            "cluster_id": int(cluster_id),
            "label": label_map.get(cluster_id, f"Archetype_{cluster_id}"),
            "size": int(len(cluster_data)),
            "mean_rating_avg": round(float(cluster_data["mean_rating"].mean()), 3),
            "std_rating_avg": round(float(cluster_data["std_rating"].mean()), 3),
            "inverse_simpson_avg": round(float(cluster_data["inverse_simpson"].mean()), 3),
            "n_ratings_avg": round(float(cluster_data["n_ratings"].mean()), 1),
            "top_genres": {genre.replace("genre_", ""): round(float(score), 3) for genre, score in top_genres.items()},
        }
        cluster_profiles.append(profile)
        print(
            f"  Cluster {cluster_id} ({profile['label']}): "
            f"n={profile['size']}, mu={profile['mean_rating_avg']}, "
            f"sigma={profile['std_rating_avg']}, ISI={profile['inverse_simpson_avg']}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "archetype_profiles.json", "w") as handle:
        json.dump(cluster_profiles, handle, indent=2)

    diagnostics = {
        "selected_k": int(features_df["cluster"].nunique()),
        "silhouette_scores": {int(k): round(float(v), 4) for k, v in silhouette_scores.items()},
    }
    with open(RESULTS_DIR / "clustering_diagnostics.json", "w") as handle:
        json.dump(diagnostics, handle, indent=2)

    return cluster_profiles


def select_seed_users(features_df, model, scaler, label_map):
    """Select representative seed users nearest to cluster centroids."""
    print("Selecting seed users...")
    feature_cols = [column for column in features_df.columns if column not in {"userId", "cluster"}]
    x = features_df[feature_cols].values
    x_scaled = scaler.transform(x)
    centroids = model.cluster_centers_

    seed_users = {}
    for cluster_id in sorted(features_df["cluster"].unique()):
        mask = features_df["cluster"] == cluster_id
        cluster_indices = features_df.index[mask].to_numpy()
        distances = np.linalg.norm(x_scaled[mask] - centroids[int(cluster_id)], axis=1)
        n_select = min(SEED_USERS_PER_ARCHETYPE, len(cluster_indices))
        selected_indices = cluster_indices[np.argsort(distances)[:n_select]]
        selected = features_df.loc[selected_indices, "userId"].astype(int).tolist()
        label = label_map.get(cluster_id, f"Archetype_{cluster_id}")
        seed_users[label] = selected
        print(f"  {label}: selected {len(selected)} seed users")

    with open(RESULTS_DIR / "seed_users.json", "w") as handle:
        json.dump(seed_users, handle, indent=2)

    return seed_users


def main():
    download_movielens()
    ratings = load_ratings()
    movies = load_movies()
    _tags = load_tags()

    ratings, movies = apply_movie_cutoff(ratings, movies)
    filtered_ratings = filter_data(ratings)
    features = engineer_features(filtered_ratings, movies)
    features, model, scaler, silhouette_scores = cluster_archetypes(features)
    label_map = label_clusters(features)
    interpret_archetypes(features, label_map, silhouette_scores)
    select_seed_users(features, model, scaler, label_map)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(DATA_DIR / "user_features.csv", index=False)
    filtered_ratings.to_csv(DATA_DIR / "filtered_ratings.csv", index=False)
    movies.to_csv(DATA_DIR / "movies.csv", index=False)
    print(f"\nPreprocessing complete. Seed users saved to {RESULTS_DIR / 'seed_users.json'}")


if __name__ == "__main__":
    main()

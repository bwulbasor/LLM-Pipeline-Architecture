"""Step 2: Generate explanation tasks with grounded user history and lightweight hybrid recommendation scoring."""
import json
import random
from collections import Counter
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RANDOM_SEED, RESULTS_DIR, SEED_USERS_PER_ARCHETYPE

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NEIGHBOR_POOL_SIZE = 30
MIN_NEIGHBOR_SUPPORT = 3


def load_data():
    ratings = pd.read_csv(DATA_DIR / "filtered_ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    if "release_year" not in movies.columns:
        movies["release_year"] = movies["title"].apply(
            lambda title: int(match.group(1)) if (match := re.search(r"\((\d{4})\)\s*$", str(title))) else np.nan
        )
    features = pd.read_csv(DATA_DIR / "user_features.csv")
    with open(RESULTS_DIR / "seed_users.json") as handle:
        seed_users = json.load(handle)
    return ratings, movies, features, seed_users


def build_feature_index(features):
    feature_cols = [column for column in features.columns if column not in {"userId", "cluster"}]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features[feature_cols].values)
    model = NearestNeighbors(n_neighbors=NEIGHBOR_POOL_SIZE + 1, metric="cosine")
    model.fit(x_scaled)
    user_to_row = {int(uid): idx for idx, uid in enumerate(features["userId"].astype(int).tolist())}
    return feature_cols, scaler, x_scaled, model, user_to_row


def build_neighbor_lookup(features, x_scaled, nn_model, user_to_row):
    neighbor_lookup = {}
    distances_all, indices_all = nn_model.kneighbors(x_scaled, return_distance=True)
    for user_id, row_idx in user_to_row.items():
        neighbors = []
        for dist, idx in zip(distances_all[row_idx], indices_all[row_idx]):
            neighbor_id = int(features.iloc[idx]["userId"])
            if neighbor_id == user_id:
                continue
            neighbors.append({"userId": neighbor_id, "distance": float(dist)})
        neighbor_lookup[user_id] = neighbors[:NEIGHBOR_POOL_SIZE]
    return neighbor_lookup


def merge_ratings_movies(ratings, movies):
    merged = ratings.merge(movies[["movieId", "title", "genres", "release_year"]], on="movieId", how="left")
    merged["genres"] = merged["genres"].fillna("")
    return merged


def top_genres_from_history(user_rows, min_rating=4):
    liked = user_rows[user_rows["rating"] >= min_rating]
    counter = Counter()
    for genres in liked["genres"]:
        for genre in str(genres).split("|"):
            if genre:
                counter[genre] += 1
    return [genre for genre, _count in counter.most_common(3)]


def summarize_history(user_rows):
    if user_rows.empty:
        return "Very limited viewing history.", "Very limited viewing history."

    top_movies = user_rows.nlargest(5, "rating")["title"].tolist()
    top_genres = top_genres_from_history(user_rows)
    low_rated = user_rows.nsmallest(3, "rating")["title"].tolist()
    avg = user_rows["rating"].mean()
    std = user_rows["rating"].std(ddof=0) if len(user_rows) > 1 else 0.0

    short_parts = [
        f"You have rated {len(user_rows)} movies with an average rating of {avg:.2f}/5 and a rating spread of {std:.2f}.",
    ]
    long_parts = list(short_parts)
    if top_genres:
        short_parts.append(f"Your strongest positive genre signals are {', '.join(top_genres)}.")
        long_parts.append(f"Your strongest positive genre signals are {', '.join(top_genres)}.")
    if top_movies:
        short_parts.append(f"Some of your highest-rated movies are {', '.join(top_movies[:2])}.")
        long_parts.append(f"Some of your highest-rated movies are {', '.join(top_movies[:3])}.")
    if low_rated:
        long_parts.append(f"You were more critical of titles such as {', '.join(low_rated[:2])}.")
    return " ".join(short_parts), " ".join(long_parts)


def precompute_user_profiles(ratings, movies, features, seed_user_ids):
    """Precompute history, genres, and neighbor evidence for all seed users."""
    all_seed_ids = sorted({int(uid) for ids in seed_user_ids.values() for uid in ids})
    ratings_with_movies = merge_ratings_movies(ratings, movies)

    _, _, x_scaled, nn_model, user_to_row = build_feature_index(features)
    neighbor_lookup = build_neighbor_lookup(features, x_scaled, nn_model, user_to_row)

    relevant_user_ids = set(all_seed_ids)
    for user_id in all_seed_ids:
        relevant_user_ids.update(item["userId"] for item in neighbor_lookup.get(int(user_id), []))

    relevant_history = ratings_with_movies[ratings_with_movies["userId"].isin(relevant_user_ids)].copy()
    ratings_by_user = {
        int(user_id): user_rows[["movieId", "rating", "title", "genres", "release_year"]].copy()
        for user_id, user_rows in relevant_history.groupby("userId")
    }
    seed_history = relevant_history[relevant_history["userId"].isin(all_seed_ids)].copy()

    profiles = {}
    for user_id, user_rows in seed_history.groupby("userId"):
        top_movies = user_rows.nlargest(5, "rating")[["movieId", "title", "genres", "rating"]].to_dict("records")
        liked_genres = top_genres_from_history(user_rows)
        mean_rating = float(user_rows["rating"].mean())
        genre_affinity = Counter()
        for row in user_rows.itertuples(index=False):
            weight = float(row.rating) - mean_rating
            genres = [genre for genre in str(row.genres).split("|") if genre]
            for genre in genres:
                genre_affinity[genre] += weight
        history_summary_short, history_summary_long = summarize_history(user_rows)
        profiles[int(user_id)] = {
            "top_movies": top_movies,
            "liked_genres": liked_genres,
            "genre_affinity": dict(genre_affinity),
            "history_summary_short": history_summary_short,
            "history_summary_long": history_summary_long,
            "n_ratings": int(len(user_rows)),
            "mean_rating": round(mean_rating, 3),
            "std_rating": round(float(user_rows["rating"].std(ddof=0) if len(user_rows) > 1 else 0.0), 3),
            "neighbor_ids": [item["userId"] for item in neighbor_lookup.get(int(user_id), [])],
        }

    return profiles, seed_history, ratings_by_user


def build_movie_catalog(movies):
    catalog = movies.copy()
    catalog["genre_set"] = catalog["genres"].fillna("").apply(
        lambda genres: set(str(genres).split("|")) if genres else set()
    )
    return catalog.to_dict("records")


def jaccard_similarity(left, right):
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def anchor_contributions(profile, candidate):
    candidate_genres = set(str(candidate.get("genres", "")).split("|")) if candidate.get("genres") else set()
    contributions = []
    for anchor in profile["top_movies"]:
        anchor_genres = set(str(anchor.get("genres", "")).split("|")) if anchor.get("genres") else set()
        sim = jaccard_similarity(candidate_genres, anchor_genres)
        rating_strength = max(float(anchor.get("rating", 0.0)) - 3.0, 0.0)
        contributions.append({
            "movieId": int(anchor["movieId"]),
            "title": anchor["title"],
            "genres": anchor.get("genres", ""),
            "rating": float(anchor.get("rating", 0.0)),
            "genre_overlap": sorted(candidate_genres & anchor_genres),
            "similarity": sim,
            "contribution": sim * (1.0 + rating_strength),
        })
    contributions.sort(key=lambda item: (item["contribution"], item["similarity"], item["rating"]), reverse=True)
    return contributions


def score_candidate_movies(uid, profile, movie_catalog, ratings_by_user):
    seen_movies = set(ratings_by_user.get(uid, pd.DataFrame())["movieId"].tolist())
    liked_genres = set(profile["liked_genres"])
    affinity = profile.get("genre_affinity", {})

    support_counter = Counter()
    rating_sum = Counter()
    rating_count = Counter()
    for neighbor_id in profile["neighbor_ids"]:
        neighbor_rows = ratings_by_user.get(neighbor_id)
        if neighbor_rows is None:
            continue
        for row in neighbor_rows.itertuples(index=False):
            movie_id = int(row.movieId)
            if movie_id in seen_movies:
                continue
            rating_count[movie_id] += 1
            rating_sum[movie_id] += float(row.rating)
            if float(row.rating) >= 4.0:
                support_counter[movie_id] += 1

    candidates = []
    for movie in movie_catalog:
        movie_id = int(movie["movieId"])
        if movie_id in seen_movies:
            continue
        genre_match = len(movie["genre_set"] & liked_genres) if liked_genres else 0
        genre_affinity_score = sum(float(affinity.get(genre, 0.0)) for genre in movie["genre_set"])
        support = support_counter.get(movie_id, 0)
        count = rating_count.get(movie_id, 0)
        avg = (rating_sum[movie_id] / count) if count else 0.0
        anchor_scores = anchor_contributions(profile, movie)
        top_anchor = anchor_scores[0] if anchor_scores else None
        content_similarity = top_anchor["similarity"] if top_anchor else 0.0
        candidates.append({
            "movieId": movie_id,
            "title": movie["title"],
            "genres": movie.get("genres", ""),
            "release_year": movie.get("release_year"),
            "genre_match": genre_match,
            "genre_affinity_score": genre_affinity_score,
            "neighbor_support": support,
            "neighbor_avg_rating": avg,
            "neighbor_rating_count": count,
            "anchor_scores": anchor_scores,
            "content_similarity": content_similarity,
        })

    return candidates


def pick_recommendation(uid, profile, movie_catalog, ratings_by_user):
    candidates = score_candidate_movies(uid, profile, movie_catalog, ratings_by_user)
    if not candidates:
        return None

    for candidate in candidates:
        popularity_bonus = np.log1p(candidate["neighbor_rating_count"])
        content_component = candidate["content_similarity"] * 4.0
        affinity_component = candidate["genre_affinity_score"] * 0.35
        neighbor_component = candidate["neighbor_support"] * 1.25 + candidate["neighbor_avg_rating"] * 0.45
        candidate["score"] = (
            content_component +
            affinity_component +
            candidate["genre_match"] * 1.5 +
            neighbor_component +
            popularity_bonus * 0.3
        )
        candidate["score_breakdown"] = {
            "content_component": round(content_component, 3),
            "affinity_component": round(affinity_component, 3),
            "neighbor_component": round(neighbor_component, 3),
            "popularity_bonus": round(float(popularity_bonus), 3),
        }

    filtered = [
        candidate for candidate in candidates
        if candidate["content_similarity"] > 0 or candidate["genre_match"] > 0 or candidate["neighbor_support"] >= MIN_NEIGHBOR_SUPPORT
    ]
    filtered = filtered or candidates
    filtered.sort(
        key=lambda candidate: (
            candidate["score"],
            candidate["content_similarity"],
            candidate["neighbor_support"],
            candidate["genre_match"],
            candidate["neighbor_avg_rating"],
        ),
        reverse=True,
    )
    top_pool = filtered[:5]
    return top_pool[0]


def generate_feature_explanation(profile, rec_movie):
    anchor_scores = rec_movie.get("anchor_scores", [])
    if not anchor_scores:
        return f"We recommend '{rec_movie.get('title', 'this movie')}' based on your viewing history."

    anchor_a = anchor_scores[0]
    anchor_b = anchor_scores[1] if len(anchor_scores) > 1 else anchor_a
    shared = anchor_a.get("genre_overlap", [])[:2]
    shared_text = " and ".join(shared) if shared else "similar thematic and stylistic"

    return (
        f"Based on your high ratings for '{anchor_a['title']}' and "
        f"'{anchor_b['title']}', "
        f"we believe you'll enjoy '{rec_movie['title']}' because it shares {shared_text} elements you tend to reward."
    )


def generate_neighbor_explanation(rec_movie):
    support = int(rec_movie.get("neighbor_support", 0))
    avg_rating = float(rec_movie.get("neighbor_avg_rating", 0.0))
    rating_count = int(rec_movie.get("neighbor_rating_count", 0))

    if support <= 0:
        return (
            f"Users with behavior closest to yours provide limited direct evidence for '{rec_movie['title']}', "
            "so the recommendation relies more on broader similarity patterns than strong crowd support."
        )

    return (
        f"Users with behavior most similar to yours also rated '{rec_movie['title']}' highly. "
        f"{support} similar users gave it at least 4/5, and the broader neighbor set produced an average rating "
        f"of {avg_rating:.1f} over {rating_count} ratings."
    )


def generate_counterfactual_explanation(profile, rec_movie):
    anchor_scores = rec_movie.get("anchor_scores", [])
    if not anchor_scores:
        return "If your highest-rated items were different, this recommendation would likely change."

    trigger = anchor_scores[0]
    shared = trigger.get("genre_overlap", [])
    shared_text = ", ".join(shared[:2]) if shared else "their overlapping themes"

    return (
        f"We recommend '{rec_movie['title']}' primarily because you rated '{trigger['title']}' highly "
        f"({trigger['rating']}/5). If you had rated that title below 3, this movie would likely fall out of your "
        f"recommendation set because the model relies on {shared_text} to connect them."
    )


def generate_adversarial_explanations(movies_df):
    """Generate 40 deliberately flawed explanations for the disconfirmation probe."""
    sample_movies = movies_df.sample(40, random_state=RANDOM_SEED)
    adversarial = []
    categories = ["logical_contradiction", "irrelevant_reasoning", "factual_fabrication", "empty_circular"]
    templates = [
        lambda title: f"We recommend '{title}' because you have shown a strong preference for lighthearted family comedies and dislike complex narratives.",
        lambda title: f"We recommend '{title}' because it has a runtime of 148 minutes and was released on a Tuesday.",
        lambda title: f"Because you rated '{title}', an action comedy starring Will Smith and directed by Christopher Nolan, five stars last week.",
        lambda title: f"Based on advanced AI analysis of your unique profile and sophisticated preference modeling, '{title}' is a great fit for you.",
    ]

    for index, (_, movie) in enumerate(sample_movies.iterrows()):
        category_idx = index % 4
        adversarial.append({
            "id": f"ADV-{index:03d}",
            "category": categories[category_idx],
            "movie": movie["title"],
            "explanation": templates[category_idx](movie["title"]),
        })

    return adversarial


def build_evaluation_tasks(ratings, movies, features, seed_users):
    """Build 1500 grounded evaluation tasks."""
    print("Precomputing user profiles and neighbor evidence...")
    profiles, seed_history, ratings_by_user = precompute_user_profiles(ratings, movies, features, seed_users)
    movie_catalog = build_movie_catalog(movies)

    tasks = []
    user_histories = {}
    task_id = 0
    for archetype, user_ids in seed_users.items():
        print(f"  Processing {archetype} ({len(user_ids)} seed users)...")
        for uid in user_ids[:SEED_USERS_PER_ARCHETYPE]:
            uid = int(uid)
            profile = profiles.get(uid)
            if not profile:
                continue

            rec_movie = pick_recommendation(uid, profile, movie_catalog, ratings_by_user)
            if rec_movie is None:
                continue

            user_histories[str(uid)] = {
                "short": profile["history_summary_short"],
                "long": profile["history_summary_long"],
            }

            explanations = {
                "feature": generate_feature_explanation(profile, rec_movie),
                "neighbor": generate_neighbor_explanation(rec_movie),
                "counterfactual": generate_counterfactual_explanation(profile, rec_movie),
            }
            for explanation_type, explanation_text in explanations.items():
                tasks.append({
                    "task_id": f"TASK-{task_id:04d}",
                    "archetype": archetype,
                    "user_id": uid,
                    "recommended_movie": rec_movie.get("title", "Unknown"),
                    "recommended_movie_id": int(rec_movie.get("movieId")),
                    "recommended_movie_genres": rec_movie.get("genres", ""),
                    "recommended_movie_year": None if pd.isna(rec_movie.get("release_year")) else int(rec_movie.get("release_year")),
                    "explanation_type": explanation_type,
                    "explanation_text": explanation_text,
                    "user_top_movies": [movie["title"] for movie in profile["top_movies"][:3]],
                    "user_top_genres": profile["liked_genres"],
                    "user_history_summary_short": profile["history_summary_short"],
                    "user_history_summary_long": profile["history_summary_long"],
                    "user_history_summary": profile["history_summary_long"],
                    "neighbor_support": int(rec_movie.get("neighbor_support", 0)),
                    "neighbor_avg_rating": round(float(rec_movie.get("neighbor_avg_rating", 0.0)), 3),
                    "neighbor_rating_count": int(rec_movie.get("neighbor_rating_count", 0)),
                    "content_similarity": round(float(rec_movie.get("content_similarity", 0.0)), 3),
                    "score_breakdown": rec_movie.get("score_breakdown", {}),
                    "anchor_evidence": [
                        {
                            "title": item["title"],
                            "rating": item["rating"],
                            "similarity": round(float(item["similarity"]), 3),
                            "contribution": round(float(item["contribution"]), 3),
                            "genre_overlap": item["genre_overlap"],
                        }
                        for item in rec_movie.get("anchor_scores", [])[:3]
                    ],
                })
                task_id += 1

    print(f"Generated {len(tasks)} evaluation tasks")
    return tasks, user_histories


def main():
    ratings, movies, features, seed_users = load_data()
    tasks, user_histories = build_evaluation_tasks(ratings, movies, features, seed_users)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "evaluation_tasks.json", "w") as handle:
        json.dump(tasks, handle, indent=2)

    with open(RESULTS_DIR / "seed_user_histories.json", "w") as handle:
        json.dump(user_histories, handle, indent=2)

    adversarial = generate_adversarial_explanations(movies)
    with open(RESULTS_DIR / "adversarial_explanations.json", "w") as handle:
        json.dump(adversarial, handle, indent=2)

    print(f"Saved {len(tasks)} tasks, {len(user_histories)} history summaries, and {len(adversarial)} adversarial probes.")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------- CONTENT BASED -------------------

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.movies_df = None

    def fit(self, movies_df):
        self.movies_df = movies_df.copy()
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies_df["clean_title"]
        )

    def recommend(self, movie_title, n=5):
        if movie_title not in self.movies_df["clean_title"].values:
            return pd.DataFrame([{"error": "Movie not found"}])

        idx = self.movies_df[self.movies_df["clean_title"] == movie_title].index[0]
        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        top_indices = scores.argsort()[::-1][1:n+1]
        recs = self.movies_df.iloc[top_indices][["clean_title"]].copy()
        recs["similarity_score"] = scores[top_indices]

        return recs.reset_index(drop=True)


# ------------------- USER-BASED COLLAB FILTERING -------------------

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.global_mean = None

    def train(self, ratings_df):
        self.user_item_matrix = ratings_df.pivot_table(
            values="rating", index="user_id", columns="movie_id"
        ).fillna(0)

        self.global_mean = ratings_df["rating"].mean()

        sim = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def predict_rating(self, user_id, movie_id):
        # Handle unseen users or movies
        if user_id not in self.user_item_matrix.index:
            return self.global_mean

        if movie_id not in self.user_item_matrix.columns:
            return self.global_mean

        ratings = self.user_item_matrix[movie_id]
        sim = self.user_similarity[user_id]

        mask = ratings > 0
        if mask.sum() == 0:
            return self.global_mean

        ratings = ratings[mask]
        sim = sim[mask]

        denom = np.abs(sim).sum()
        if denom == 0:
            return self.global_mean

        return float((sim * ratings).sum() / denom)

    def recommend_for_user(self, user_id, movies_df, ratings_df, n=10):
        rated = ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist()
        candidates = [m for m in movies_df["movie_id"] if m not in rated]

        preds = [(m, self.predict_rating(user_id, m)) for m in candidates]
        preds.sort(key=lambda x: x[1], reverse=True)

        top_ids = [m for m, _ in preds[:n]]
        top_scores = [s for _, s in preds[:n]]

        recs = movies_df[movies_df["movie_id"].isin(top_ids)].copy()
        recs["predicted_rating"] = top_scores

        return recs[["clean_title", "predicted_rating"]]


# ------------------- HYBRID -------------------

class HybridRecommender:
    def __init__(self, w_content=0.4, w_cf=0.6):
        self.w_content = w_content
        self.w_cf = w_cf
        self.content_model = ContentBasedRecommender()
        self.cf_model = CollaborativeFilteringRecommender()
        self.movies_df = None
        self.ratings_df = None

    def fit(self, ratings_df, movies_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.content_model.fit(movies_df)
        self.cf_model.train(ratings_df)

    def recommend(self, movie_title, user_id, n=10):
    
        # If user has no ratings yet -> fallback to content only
        if user_id not in self.ratings_df["user_id"].unique():
            return self.content_model.recommend(movie_title, n)
        
        content = self.content_model.recommend(movie_title, n)
        cf = self.cf_model.recommend_for_user(user_id, self.movies_df, self.ratings_df, n)

        if "similarity_score" in content.columns:
            content["score"] = (content["similarity_score"] / content["similarity_score"].max()) * self.w_content

        cf["score"] = (cf["predicted_rating"] / cf["predicted_rating"].max()) * self.w_cf

        combined = pd.concat([
            content[["clean_title", "score"]],
            cf[["clean_title", "score"]]
        ])

        return combined.groupby("clean_title").sum().sort_values("score", ascending=False).head(n).reset_index()

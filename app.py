import streamlit as st
import pandas as pd
from src.recommender import HybridRecommender, ContentBasedRecommender, CollaborativeFilteringRecommender
from src.data_loader import load_movielens_data

# st.write("# 🎬 Movie Recommendation System")
# ------- Load Data --------
@st.cache_data
@st.cache_data
def load_data():
    ratings, movies = load_movielens_data()

    # Load custom movies if file exists
    try:
        custom_movies = pd.read_csv("data/custom_movies.csv")
        movies = pd.concat([movies, custom_movies], ignore_index=True)
        movies.drop_duplicates(subset=["clean_title"], inplace=True)

    except FileNotFoundError:
        pass

    return ratings, movies


# ------- Train Model --------
@st.cache_resource
def train_models(ratings, movies):
    hybrid = HybridRecommender()
    hybrid.fit(ratings, movies)

    return {
        "hybrid": hybrid,
        "content": hybrid.content_model,
        "collab": hybrid.cf_model
    }


# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered" )

st.title("🎥 Movie Recommendation System")
st.write("Find movies based on **content similarity**, **similar users**, or a **hybrid model**.")

# Load data
ratings, movies = load_data()

# Train models
models = train_models(ratings, movies)

# Sidebar Inputs
st.sidebar.header("🔧 Recommendation Settings")

user_id = st.sidebar.selectbox("👤 Select User ID", sorted(ratings["user_id"].unique()))

movie_title = st.sidebar.selectbox(
    "🎞 Choose a Movie You Like",
    sorted(movies["clean_title"].unique())
)

n = st.sidebar.slider("🔢 Number of Recommendations", 3, 15, 5)

method = st.sidebar.radio(
    "🤖 Recommendation Method",
    ["Content-Based", "Collaborative Filtering", "Hybrid"]
)

st.sidebar.subheader("➕ Add New Movie")

new_movie = st.sidebar.text_input("Movie Title")

if st.sidebar.button("Add Movie"):
    if new_movie.strip():
        df = pd.DataFrame([{
            "movie_id": str(len(movies) + 1),
            "title": new_movie,
            "clean_title": new_movie.strip()
        }])

        df.to_csv("data/custom_movies.csv", mode="a", header=not pd.io.common.file_exists("data/custom_movies.csv"), index=False)
        st.success(f"🎉 '{new_movie}' added successfully! Restart the app or retrain to include it.")
    else:
        st.warning("Please enter a valid movie name.")
st.write("---")

# Run Recommendation
if st.button("✨ Get Recommendations"):
    st.subheader(f"📌 Top {n} Recommendations for '{movie_title}'")

    if method == "Content-Based":
        recs = models["content"].recommend(movie_title, n)
    elif method == "Collaborative Filtering":
        recs = models["collab"].recommend_for_user(user_id, movies, ratings, n)
    else:
        recs = models["hybrid"].recommend(movie_title, user_id, n)

    st.table(recs)

st.write("---")
if st.button("🔄 Retrain Model with New Data"):
    models = train_models(ratings, movies)
    st.success("🚀 Model retrained successfully!")


st.info("💡 Tip: Try different movies and users to compare recommendation behaviors!")

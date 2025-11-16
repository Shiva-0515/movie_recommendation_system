# import pandas as pd
# import numpy as np
# import requests
# import zipfile
# import os
# def download_movielens_data():
# # """Download and extract MovieLens 100K dataset"""
#     url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
#     # Create data directory
#     os.makedirs("data/raw", exist_ok=True)
#     # Download the zip file
#     response = requests.get(url)
#     with open("data/raw/ml-100k.zip", "wb") as f:
#     f.write(response.content)
#     # Extract the zip file
#     with zipfile.ZipFile("data/raw/ml-100k.zip", 'r') as zip_ref:
#     zip_ref.extractall("data/raw/")
#     print("MovieLens data downloaded successfully!")
#     # Run this function to download data
#     download_movielens_data()

# def load_movielens_data():
# """Load MovieLens datasets"""
# #Load ratings data
# ratings = pd.read_csv('data/raw/ml-100k/u.data',
# sep='\t',
# names=['user_id', 'movie_id', 'rating', 'timestamp'])
# #Load movie information
# movies = pd.read_csv('data/raw/ml-100k/u.item',
# sep='',
# encoding='latin-1',
# names=['movie_id', 'title', 'release_date', 'video_release_dat
# e',
# 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'])
# #Load user information
# users =
# pd.read_csv('data/raw/ml-100k/u.user',
# sep='',
# names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
# return ratings, movies, users
# # Load data
# ratings, movies, users = load_movielens_data()
# # Basic exploration
# print("Ratings shape:", ratings.shape)
# print("Movies shape:", movies.shape)
# print("Users shape:", users.shape)
# print("\nRatings sample:")
# print(ratings.head())


import pandas as pd
import numpy as np
import requests
import zipfile
import os


def download_movielens_data():
    """Download and extract MovieLens 100K dataset"""
    
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

    # Create directory
    os.makedirs("data/raw", exist_ok=True)

    # Download file
    response = requests.get(url)
    with open("data/raw/ml-100k.zip", "wb") as f:
        f.write(response.content)

    # Extract file
    with zipfile.ZipFile("data/raw/ml-100k.zip", 'r') as zip_ref:
        zip_ref.extractall("data/raw/")

    print("MovieLens data downloaded successfully!")


import os
import pandas as pd

def load_movielens_data():
    # --- Load Base Movielens Dataset ---
    ratings = pd.read_csv(
        "data/raw/ml-100k/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/raw/ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        usecols=[0, 1],
        names=["movie_id", "title"]
    )

    # Convert IDs to strings
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["movie_id"] = ratings["movie_id"].astype(str)

    # Clean titles
    movies["clean_title"] = movies["title"].str.replace(
        r"\(\d{4}\)", "", regex=True
    ).str.strip()

    # --- Load Custom Movies if exist ---
    custom_movies_path = "data/custom_movies.csv"
    if os.path.exists(custom_movies_path):
        custom_movies = pd.read_csv(custom_movies_path)
        movies = pd.concat([movies, custom_movies], ignore_index=True)

    # --- Load Custom Ratings if exist ---
    custom_ratings_path = "data/custom_ratings.csv"
    if os.path.exists(custom_ratings_path):
        custom_ratings = pd.read_csv(custom_ratings_path)
        ratings = pd.concat([ratings, custom_ratings], ignore_index=True)

    return ratings, movies



# ---- Run and test ----

# Only run download if not exists
if not os.path.exists("data/raw/ml-100k"):
    download_movielens_data()

# ratings, movies, users = load_movielens_data()

# print("Ratings shape:", ratings.shape)
# print("Movies shape:", movies.shape)
# print("Users shape:", users.shape)

# print("\nRatings sample:")
# print(ratings.head())

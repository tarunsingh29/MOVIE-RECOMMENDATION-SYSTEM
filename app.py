import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# creating a dataframe and loading the data
# df = pd.read_csv(r"D:\Tarun's code\mini Project(MRS)\MOVIE-RECOMMENDATION-SYSTEM\LargeSet\movies.csv")

#preprocessing
movies = pd.read_csv(r"smallset\movies.csv")
ratings = pd.read_csv(r"smallset\ratings.csv")
tags = pd.read_csv(r"smallset\tags.csv")

movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
movie_stats = movie_stats.join(movies.set_index('movieId'), on='movieId')

# group all tags for each movie into one string
tag_text = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.astype(str)))

# join into movies dataframe
movies = movies.join(tag_text, on="movieId")

# replace NaN tags with empty string
movies["tag"] = movies["tag"].fillna("")

# replacing '|' with space in genres column

movies["text"] = movies["genres"] + " " + movies["tag"]

tfidf = TfidfVectorizer(stop_words="english") #removing english stop words like is , am , the etc
# tfidf = tfidf.fit_transform(movies["text"]) #fitting and transforming the genres column
tfidf_matrix = tfidf.fit_transform(movies["text"])
#calculating cosine similarity matrix





movies["title_lower"] = movies["title"].str.lower()
indices = pd.Series(movies.index, index=movies["title_lower"]).drop_duplicates()


def recommend(title, num=10, min_count=50):
    title = str(title).lower().strip()

    # 1) Exact match
    if title in indices:
        idx = indices[title]
    else:
        # 2) Partial match (safe)
        matches = movies[movies["title_lower"].str.contains(title, na=False)]
        if matches.empty:
            return None
        idx = matches.index[0]

    # 3) Compute similarity ONLY for this movie
    sim_vector = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Sort movie indices by similarity
    sorted_indices = sim_vector.argsort()[::-1]

    filtered = []

    for i in sorted_indices:
        if i == idx:  # skip the same movie
            continue

        score = sim_vector[i]
        movie_id = movies.loc[i, "movieId"]

        # Check if stats exist
        if movie_id not in movie_stats.index:
            continue

        avg_rating = movie_stats.loc[movie_id, "mean"]
        rating_count = movie_stats.loc[movie_id, "count"]

        # Apply minimum rating threshold
        if rating_count >= min_count:
            filtered.append((i, score, avg_rating, rating_count))

        if len(filtered) >= num:
            break

    # Convert to DataFrame
    results = pd.DataFrame([
        {
            "title": movies.loc[i, "title"],
            "genres": movies.loc[i, "genres"],
            "rating_count": int(rating_count),
        }
        for (i, score, avg_rating, rating_count) in filtered
    ])

    return results



# Streamlit UI
# -------------------------------
st.title("🎬 Movie Recommendation System")
st.write("Type a movie name and get similar movie recommendations based on genres.")

title = st.text_input("Enter a movie title:", "")


if st.button("Recommend"):
    if title.strip() == "":
        st.warning("Please enter a movie name!")
    else:
        results = recommend(title)
        
        if results is None:
            st.error("Movie not found. Check spelling or try another movie.")
        else:
            st.success(f"Top Recommendations for: {title.title()}")
            st.table(results)
            


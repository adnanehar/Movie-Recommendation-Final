import streamlit as st
import pickle
import numpy as np
import pandas as pd
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.plotting import Plot

# Load the model from the pkl file
with open('deep_model_data.pkl', 'rb') as f:
    df_filtered, smd, id_map, cosine_similarity, user_embeddings, movie_embeddings, user_id_mapping, movie_id_mapping = pickle.load(f)

# Function to predict similar movies
def predict_similar_movies(userId, title, movie_count):
    movieId = id_map.loc[title]['movieId'].astype('int')
    # Map the userId and movieId to the corresponding integer values
    user_input = np.array([user_id_mapping[userId]])
    movie_input = np.array([movie_id_mapping[movieId]])

    # Get the learned embeddings for the user and input movie
    user_emb = user_embeddings[user_input][0]
    movie_emb = movie_embeddings[movie_input][0]

    # Compute cosine similarity between the input movie and all other movies
    sim_scores = cosine_similarity([movie_emb], movie_embeddings)[0]

    # Get the indices of the most similar movies
    movie_indices = sim_scores.argsort()[-(movie_count+1):-1][::-1]

    similar_movies = pd.DataFrame(columns=['title'])
    for idx in movie_indices:
        movie_id = smd.iloc[idx]['id']
        if movie_id in smd['id'].values:
            title = smd.loc[smd['id'] == movie_id, 'title'].values[0]
            similar_movies.loc[len(similar_movies)] = [title]
        else:
            # If the movie ID is not in smd, skip it
            continue

    st.write(f"Here are {movie_count} movie recommendations for user {userId} based on the movie '{title}':")
    st.table(similar_movies.head(movie_count))

# Create the Streamlit app
st.title('Deep Movie Recommender')

# Get user input
userId = st.number_input('Enter your user ID', min_value=1, max_value=610, value=1)
title = st.text_input('Enter a movie title')
movie_count = st.number_input('Enter the number of recommendations you want', min_value=1, max_value=25, value=10)

# Make recommendations based on user input
if st.button('Get Recommendations'):
    predict_similar_movies(userId, title, movie_count)

    # Load the movies metadata from CSV
    movies_df = pd.read_csv('movies_metadata.csv')

    # Select relevant columns for Aequitas analysis
    selected_columns = ['budget', 'imdb_id']
    movies_selected = movies_df[selected_columns]

    
    movies_selected['score'] = 94
    movies_selected['label_value'] = 20

    # Preprocess data for Aequitas
    aequitas_df_processed, _ = preprocess_input_df(movies_selected)
    g = Group()
    xtab, _ = g.get_crosstabs(aequitas_df_processed)
    
    # Check for zero division error
    if xtab.empty:
        st.write("No data available for fairness evaluation.")
    else:
        absolute_metrics = g.list_absolute_metrics(xtab)
        st.write("Absolute Metrics")
        st.write(absolute_metrics)
        st.write(xtab)

        p = Plot()
        st.write("Fairness Metrics Plots")
        
        # Specify a fairness metric for plotting (e.g., 'ppr', 'pprev', 'fdr', 'for', etc.)
        fairness_metric = 'ppr'
        st.write(p.plot_group_metric_all(xtab, metrics=[fairness_metric]))

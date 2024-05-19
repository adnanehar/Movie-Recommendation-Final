import streamlit as st
import pickle
import pandas as pd
from surprise import Reader, Dataset, SVD
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.plotting import Plot

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pandas.core.indexes.numeric":
            return pd.Index
        return super().find_class(module, name)

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return CustomUnpickler(f).load()

# Load the model from the pkl file using the defined function
file_path = 'C:\\Users\\dell\\Documents\\AJAX-Movie-Recommendation\\Movie-Recommender-App\\models\\The-Movie-Model\\model.pkl'
smd, id_map, indices_map, cosine_sim, svd = load_pickle_file(file_path)

# My Hybrid Recommender
def recommend_movie(userId, title, movie_count):
    try:
        indices = pd.Series(smd.index, index=smd['title'])
        idx = indices[title]

        tmdbId = id_map.loc[title]['id']
        movie_id = id_map.loc[title]['movieId']
    except KeyError:
        st.error('Movie not found in the database. Please enter a valid movie title.')
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)

    st.write(f"Here are {movie_count} movie recommendations for user {userId} based on the movie '{title}':")
    st.table(movies.head(movie_count))

    return movies

# Fairness evaluation with Aequitas
def evaluate_fairness(movies):
    # Example data with sensitive attributes for fairness evaluation
    example_data = {
        'user_id': [1, 2, 3, 4, 5],
        'score': [0.8, 0.6, 0.7, 0.9, 0.5],  # Example model predictions (range from 0 to 1)
        'label_value': [1, 0, 1, 1, 0],  # Example ground truth labels
        'gender': ['male', 'female', 'male', 'female', 'male'],  # Example sensitive attribute (gender)
    }
    example_df = pd.DataFrame(example_data)

    # Binarize scores based on threshold
    threshold = 0.7
    example_df['score'] = example_df['score'].apply(lambda x: 1 if x >= threshold else 0)

    # Preprocess data for Aequitas
    aequitas_df_processed, _ = preprocess_input_df(example_df)
    g = Group()
    xtab, _ = g.get_crosstabs(aequitas_df_processed)
    absolute_metrics = g.list_absolute_metrics(xtab)
    
    st.write("Absolute Metrics")
    st.write(absolute_metrics)
    st.write(xtab)

    # Plot fairness metrics
    p = Plot()

    # List of all possible fairness metrics
    all_fairness_metrics = ['ppr', 'pprev', 'fdr', 'for', 'fpr', 'fnr', 'tpr', 'tnr', 'npv']

    # Check which metrics are available in xtab
    available_metrics = [metric for metric in all_fairness_metrics if metric in xtab.columns]

    if available_metrics:
        st.write("Fairness Metrics Plots")
        st.write(p.plot_group_metric_all(xtab, metrics=available_metrics))
    else:
        st.warning("No available fairness metrics to plot.")

# Create the Streamlit app
st.title('Movie Recommender')

# Get user input
userId = st.number_input('Enter your user ID', min_value=1, max_value=610, value=1)
title = st.text_input('Enter a movie title')
movie_count = st.number_input('Enter the number of recommendations you want', min_value=1, max_value=25, value=10)

# Make recommendations based on user input
if st.button('Get Recommendations'):
    recommended_movies = recommend_movie(userId, title, movie_count)
    if not recommended_movies.empty:
        evaluate_fairness(recommended_movies)

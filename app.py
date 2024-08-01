import streamlit as st
import pickle
import numpy as np
import pandas as pd
import difflib
from sklearn.preprocessing import MultiLabelBinarizer

# Load the trained model and preprocessing objects
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('mlb_genres.pkl', 'rb') as mlb_file:
    mlb_genres = pickle.load(mlb_file)

# Load feature names
with open('feature_names.pkl', 'rb') as feature_file:
    feature_names = pickle.load(feature_file)

# Load movie data
df = pd.read_pickle('all_movies.pkl')

# Ensure genres are in list format
if not isinstance(df['genres'].iloc[0], list):
    df['genres'] = df['genres'].apply(eval)

# Preprocess the data
# Use the pre-fitted MultiLabelBinarizer from the saved file
genres_encoded = mlb_genres.transform(df['genres'])

def find_closest_movie(movie_title, movie_list):
    closest_matches = difflib.get_close_matches(movie_title, movie_list, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

def predict_earnings(genres, budget, runtime):
    # Encode genres using the same MultiLabelBinarizer
    genres_encoded = mlb_genres.transform([genres])
    
    # Ensure input features match the training feature set
    input_features = np.hstack([genres_encoded, [[budget, runtime]]])
    
    # Check if the input features have the correct shape
    if input_features.shape[1] != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} features, but got {input_features.shape[1]}")
    
    # Predict earnings
    predicted_revenue = model.predict(input_features)
    return predicted_revenue[0]

# Streamlit app
st.title('Movie Earnings Prediction')

# Input fields
movie_name = st.text_input('Movie Name')
actor_name = st.text_input('Actor Name')

# Get list of all movies for validation
all_movies = df['title'].tolist()

# Validate movie title
if movie_name:
    closest_movie = find_closest_movie(movie_name, all_movies)
    if closest_movie and closest_movie.lower() != movie_name.lower():
        st.write(f"Did you mean: {closest_movie}?")
        movie_name = closest_movie

genres = st.multiselect('Genres', mlb_genres.classes_)
budget = st.number_input('Budget', min_value=0)
runtime = st.number_input('Runtime', min_value=0)

# Predict button
if st.button('Predict Earnings'):
    if genres and budget > 0 and runtime > 0:
        try:
            earnings = predict_earnings(genres, budget, runtime)
            st.write(f'Predicted Earnings for {movie_name} starring {actor_name}: ${earnings:,.2f}')
        except ValueError as e:
            st.write(f"Error: {e}")
    else:
        st.write('Please fill in all the fields correctly')


import Levenshtein
import pandas as pd
from ease import EASE


class Recommender:
    def __init__(self, ratings_path, movie_names_path):
        self.ease = EASE()
        self.movie_names = self.read_movies_names_to_df(movie_names_path)
        self.ratings_path = ratings_path        # Ratings will be needed only if the model isn't already trained, so only store the path

    def read_ratings_to_df(self, file_path):
        """ Reads the ratings from the given file path and returns a DataFrame with columns user_id, item_id, and rating. """
        df = pd.read_json(file_path, lines=True, orient='records')
        return df[['user_id', 'item_id', 'rating']]

    def read_movies_names_to_df(self, file_path):
        """ Reads the movie names from the given file path and returns a DataFrame with columns item_id, title, popularity, and imdbId. """
        df = pd.read_json(file_path, lines=True, orient='records', dtype={'imdbId': str})
        return df[['item_id', 'title', 'popularity', 'imdbId']]
    
    def distance(self, movie_name, database_name):
        """ Calculates the Levenshtein distance between two strings. Some movies in the database have multiple titles, separated by " (a.k.a. ". In this case, the distance is calculated for each title and the minimum distance is returned.
        Args:
            movie_name (str): The name of the first (user) movie
            database_name (str): The name of the second (database) movie
        Returns:
            int: The Levenshtein distance between the two strings
        """

        movie_name_low = movie_name.lower()
        database_name_low = database_name.lower()
        aka_expression = " (a.k.a. "
        
        if aka_expression not in database_name_low:
            return Levenshtein.distance(movie_name_low, database_name_low)

        parts = database_name_low.split(aka_expression)
        distances = [Levenshtein.distance(movie_name_low, part) for part in parts]
        return min(distances)


    def get_movie_id(self, movies, movie_name, max_distance_threshold = 2):
        """ Gets the movie_id from the database based on the movie name. If the movie name is not found, it will return None.
        Args:
            movies (pd.DataFrame): DataFrame with columns item_id and title
            movie_name (str): The name of the movie to search for
            max_distance_threshold (int, optional): The maximum Levenshtein distance allowed for a match. Defaults to 2.
        Returns:
            int: The item_id of the movie if found, otherwise None
        """
        best_distance = float('inf')
        best_match = None

        remove_year_from_title = lambda title: title[:-7]   # Release year is always the last 7 characters

        for i, row in movies.iterrows():
            database_name = remove_year_from_title(row['title'])
            # distance = Levenshtein.distance(movie_name.lower(), database_name.lower())
            distance = self.distance(movie_name, database_name)

            if distance == 0:
                best_distance = distance
                best_match = row['item_id']
                break

            if distance < best_distance:
                best_distance = distance
                best_match = row['item_id']

        if best_distance > max_distance_threshold:
            print(f'Could not find a match for "{movie_name}"')
            return None

        found_movie_name = movies[movies['item_id'] == best_match]['title'].values[0]
        found_movie_id = movies[movies['item_id'] == best_match]['item_id'].values[0]
        print(f'Looking for "{movie_name}". Found "{found_movie_name}"')
        return found_movie_id
    
    def convert_movie_names_to_ids(self, user_ratings):
        """ Converts a list of (movie_name, rating) tuples into a list of (movie_id, rating) tuples. If a movie name is not found, it will be ignored. """
        converted_ratings = []
        for movie, rating in user_ratings:
            movie_id = self.get_movie_id(self.movie_names, movie)
            if movie_id is not None:
                converted_ratings.append((movie_id, rating))
                
        return converted_ratings
        
        

    def fit(self, model_path=None, lambda_=0.5, sparsity_coefficient=98, implicit=False):
        """ Fits the EASE model to the ratings data."""
        self.ease.fit(self, lambda_=lambda_, sparsity_coefficient=sparsity_coefficient, implicit=implicit, model_path=model_path)

    def merge_with_movie_names(self, predictions):
        """ Merges the predictions with the movie names and returns the DataFrame sorted by score in descending order."""
        return predictions.merge(self.movie_names, on='item_id').sort_values('score', ascending=False)

    def predict_most_popular(self, k=10):
        """ Predicts the k most popular movies."""
        
        # self.movie_names is already sorted by popularity
        # returns k random movies from 10*k most popular movies
        recommendations = self.movie_names[:10*k].sample(k)[['item_id', 'title', 'popularity', 'imdbId']].sort_values('popularity', ascending=False)
        
        # Score is a normalized popularity between 0 and 1
        recommendations['score'] = recommendations['popularity'] / recommendations['popularity'].max()
        recommendations = recommendations[['item_id', 'score', 'title', 'popularity', 'imdbId']]    # Reorder columns

        return recommendations


    def predict(self, user_ratings, k=10):
        """ Predicts k movies for the user based on the user's ratings.
        Args:
            user_ratings (list): List of tuples with item_id and rating [(item_id, rating), ...]
            k (int, optional): Number of recommendations to return. Defaults to 10.
        Returns:
            pandas.DataFrame: DataFrame with columns item_id, score, title, popularity, imdbId
        """
        
        no_ratings_provided = len(user_ratings) == 0
        if no_ratings_provided:
            return self.predict_most_popular(k)

        predictions =  self.ease.predict(user_ratings, k)
        return self.merge_with_movie_names(predictions)
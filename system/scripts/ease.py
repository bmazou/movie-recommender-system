# https://github.com/Darel13712/ease_rec

import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user_id'])
        items = self.item_enc.fit_transform(df.loc[:, 'item_id'])
        return users, items

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.user_enc, self.item_enc, self.B), f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.user_enc, self.item_enc, self.B = pickle.load(f)

    def fit(self, recommender, lambda_: float = 0.5, sparsity_coefficient=98, implicit=False, model_path=None):
        """
        df: pandas.DataFrame with columns user_id, item_id, and (rating)
        lambda_: l2-regularization term
        sparsity_coefficient: percentage of the smallest values to zero out in the B matrix before converting it to sparse
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        model_path: Path to save or load the model. If provided, will attempt to load the model; if not found, trains a new model.
        """


        if model_path is not None and os.path.exists(model_path):
            print("Loading the model")
            self.load_model(model_path)
            print("Model loaded successfully.")
            return
        
        ratings = recommender.read_ratings_to_df(recommender.ratings_path)

        print(f"Starting the fit process with lambda={lambda_}...")
        users, items = self._get_users_and_items(ratings)
        print("Preparing values...")
        values = (
            np.ones(ratings.shape[0])
            if implicit
            else ratings['rating'].to_numpy() / ratings['rating'].max()
        )

        print("Creating matrix X...")
        print(f'Users: {users.shape}. Items: {items.shape}. Values: {values.shape}.')
        X = csr_matrix((values, (users, items)), shape=(len(set(users)), len(set(items))))
        print(f'X shape: {X.shape}.')

        print("Computing matrix G...")
        G = X.T.dot(X).toarray()
        print("Adding lambda")
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        print("Inverting matrix G...")
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        print("Converting the matrix to sparse...")
        # Converts B to sparse by zeroing out the smallest absolute values
        threshold = np.percentile(np.abs(B), sparsity_coefficient)
        B[np.abs(B) < threshold] = 0
        self.B = csr_matrix(B)

        if model_path is not None:
            self.save_model(model_path)
            print(f"Model saved successfully at {model_path}.")

    def predict(self, new_user_ratings, k=10):
        """ Predicts the top k items for a new user based on their ratings.
        Args:
            new_user_ratings (list): List of tuples (item_id, rating) for the new user
            k (int, optional): Number of items to recommend. Defaults to 10.
        Returns: 
            pd.DataFrame: DataFrame with columns item_id and score, sorted by score in descending order   
        """
        movie_ids = [x[0] for x in new_user_ratings]
        ratings = [x[1] for x in new_user_ratings]

        try:
            transformed_movie_ids = self.item_enc.transform(movie_ids)
        except ValueError:
            # Handles unknown movies by ignoring them
            valid_indices = [i for i, movie_id in enumerate(movie_ids) if movie_id in self.item_enc.classes_]
            transformed_movie_ids = self.item_enc.transform([movie_ids[i] for i in valid_indices])
            ratings = [ratings[i] for i in valid_indices]


        # Create a user vector with ratings for the movies they've rated
        user_vector = np.zeros(self.B.shape[1])
        user_vector[transformed_movie_ids] = ratings

        scores = user_vector.dot(self.B.toarray())          # Compute the score for each item

        scores[transformed_movie_ids] = -np.inf   # Remove items user has already rated

        # Get the top k items
        recommended_item_indices = np.argpartition(scores, -k)[-k:]
        recommended_scores = scores[recommended_item_indices]

        # Transform item indices back to original movie IDs
        recommended_movie_ids = self.item_enc.inverse_transform(recommended_item_indices)

        recommendations = pd.DataFrame({
            'item_id': recommended_movie_ids,
            'score': recommended_scores
        }).sort_values(by='score', ascending=False).reset_index(drop=True)

        return recommendations
import argparse
import json
import os

import numpy as np
import pandas as pd
from openai import OpenAI
from recommendation_logic import RecommendationLogic


class DatasetGenerator:
    """ Class for generating data for experiments. Saves the data in the experiments/data directory. Each user has 4 files saved:
    - <user_id>_sample.json - randomly selected ratings, format [{item_id, user_id, rating}]
    - <user_id>_target.json - remaining ratings, format [{item_id, user_id, rating}]
    - <user_id>_sample_textual.txt - textual representation of sample ratings, format <movie_title>: <rating>
    - <user_id>_full_prompt.txt - GPT-4o prompt, natural language representation of sample ratings.
    """
    
    def __init__ (self, config_path):
        self.ratings = pd.read_json("../data/ratings_test.json", orient="records", lines=True)
        self.movies = pd.read_json("../data/movies.json", orient="records", lines=True)
        self.movie_plots = pd.read_csv("../data/movie_plots.csv")
        self.client = self.load_client(config_path)
        
    def load_client(self, config_path):
        """ Load OpenAI client from config file.
        Args:
            config_path (str): Path to config file. 
        Returns:
            client (OpenAI): OpenAI client.
        """

        with open(config_path, "r") as f:
            config = json.load(f)
            
        api_key = config["openai_api_key"]
        return OpenAI(api_key=api_key)
   
    def save_file(self, data, path):
        with open(path, "w") as f:
            f.write(data)
    
    def gpt_convert_ratings_to_prompt(self, text, include_general_prefs):
        """ Generate a prompt from GPT-4o model given a set of movie ratings.
        Args:
            text (str): Textual representation of movie ratings.
            include_general_prefs (bool): Whether GPT should find general prefernces in the ratings and include them in the prompt. True if doing "recall" task, False if doing "movie-search" task.
        Returns:
            prompt (str): Resulting GPT-4o prompt.
        """
        
        example_prompt1 = """Inception: 5.0
Interstellar: 4.0
Kill Bill: Vol. 2: 3
Home Alone: 0.5
Django Unchained: 4.0
"""
        example_answer1 = f"""I absolutely loved "Inception" and I enjoyed both "Interstellar" and "Django Unchained." "Kill Bill: Vol. 2" is okay for me, but I really hated "Home Alone." """
        

        example_prompt2 = """The Dark Knight: 4.0
Zootopia: 2.5
Signs: 1.5
No Country for Old Men: 5.0
There Will Be Blood: 4.0
"""
        example_answer2 = f"""I really love "No Country for Old Men" and some other movies I like are "The Dark Knight" and "There Will Be Blood". "Zootopia" wasn't too great for me and I really dislike like "Signs." """
        
        
        messages = [
            {"role": "system", "content": f"Your task is to translate movie ratings into textual format, where you use words such as \"I like/love/don't like/hate etc.\" to express the movie ratings. You are forbidden from mentioning the scores themselves."},
            {"role": "user", "content": example_prompt1},
            {"role": "assistant", "content": example_answer1},
            {"role": "user", "content": example_prompt2},
            {"role": "assistant", "content": example_answer2},
            {"role": "user", "content": text}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def gpt_convert_movie_title_to_prompt(self, movie_title):
        """ Asks GPT-4o to generally describe a given movie.

        Args:
            movie_title (str): Title of the movie.
        """
        
        example_prompt1 = "Iron Man (2008)"
        example_answer1 = "a superhero action film focusing on technology."
        
        example_prompt2 = "Blade Runner 2049 (2017)"
        example_answer2 = "a dystopian science fiction film with themes of human identity."

        messages = [
            {"role": "system", "content": "You will be given a movie title and your task is to describe the movie in a few words. You are forbidden from mentioning the movie's name, plot details, or any specific characters."},
            {"role": "user", "content": example_prompt1},
            {"role": "assistant", "content": example_answer1},
            {"role": "user", "content": example_prompt2},
            {"role": "assistant", "content": example_answer2},
            {"role": "user", "content": movie_title}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def convert_ratings_to_text(self, ratings):
        """ Convert movie ratings to text.
        Args:
            ratings (pd.DataFrame): Movie ratings.
        Returns:
            text (str): Textual representation of ratings.
        """
        
        remove_year_from_title = lambda title: title[:-7]
        def transform_title(title):
            title = remove_year_from_title(title)
            if " (a.k.a. " in title:
                return title.split(" (a.k.a. ")[0]
            else:
                return title
        
        text = ""
        for i, row in ratings.iterrows():
            movie = self.movies[self.movies["item_id"] == row["item_id"]].iloc[0]
            text += f'{transform_title(movie["title"])}: {row["rating"]}\n'
        return text
        
    def split_user_ratings(self, user_id, sample_size=5):
        """ Split user ratings into sample and target sets. Done for "recall" task.
        Args:
            user_id (int): User ID.
            sample_size (int): Number of ratings to sample.
        Returns:
            sample (pd.DataFrame): Sample ratings.
            target (pd.DataFrame): Target ratings. 
        """
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        
        maximum_k = 20
        if len(user_ratings) < sample_size + maximum_k:
            print(f'Not enough ratings for user {user_id}, only {len(user_ratings)} ratings available.')
            return None, None

        sample = user_ratings.sample(sample_size)
        target = user_ratings.drop(sample.index)
        return sample, target
    

        
    def generate_data_for_user(self, user_id, sample_size, directory_path="experiments"):
        """ Chooses a random movie that the user rated highly as a target, the movie has to be in movie_plots dataset. Then, samples other movies that the user rated. Converts the rating into a textual desciption, describes the target movie, and creates a full prompt for GPT-4o by concatenating them. Saves all the files."""
        
                
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        viable_movies = user_ratings[user_ratings["rating"] >= 4] 
        viable_movies = viable_movies[viable_movies["item_id"].isin(self.movie_plots["item_id"])]
        
        if len(viable_movies) == 0:
            print(f'No viable movie with plot present for user {user_id}.')
            return None
        
        target = viable_movies.sample(1)
        target_movie_title = self.movies[self.movies["item_id"] == target["item_id"].values[0]]["title"].values[0]
        target_movie_description = self.gpt_convert_movie_title_to_prompt(target_movie_title)
        
        sample = user_ratings.drop(target.index).sample(sample_size)
        
        target_not_in_sample = target["item_id"].values[0] not in sample["item_id"].values
        assert target_not_in_sample, f'Target movie {target["item_id"].values[0]} is in the sample.'
        
        sample_textual = self.convert_ratings_to_text(sample)
        sample_prompt = self.gpt_convert_ratings_to_prompt(sample_textual, include_general_prefs=False)

        full_prompt = f"""{sample_prompt} Currently, I'm in a mood for {target_movie_description}"""

        sample.to_json(f"{directory_path}/{user_id}_sample.json", orient="records", lines=True)
        target.to_json(f"{directory_path}/{user_id}_target.json", orient="records", lines=True)
        self.save_file(sample_textual, f"{directory_path}/{user_id}_sample_textual.txt")
        self.save_file(sample_prompt, f"{directory_path}/{user_id}_sample_prompt.txt")
        self.save_file(target_movie_title, f"{directory_path}/{user_id}_target_title.txt")
        self.save_file(target_movie_description, f"{directory_path}/{user_id}_target_description.txt")
        self.save_file(full_prompt, f"{directory_path}/{user_id}_full_prompt.txt")
   
    def generate_data(self, sample_size, n_users, file_path):                  
        """ Generate data for n_users. For each user, sample sample_size ratings and save the data in the directory_path."""

        # if file at file_path exists, load user_ids from file
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                user_ids = json.load(f)
        else:
            user_ids = self.choose_users(n_users, file_path)


        for user_id in user_ids:
            self.generate_data_for_user(user_id, sample_size)


    def choose_users(self, n_users, file_path):
        """ Returns n random user IDs to test model on. The users must have at least min_ratings ratings.

            n_users (int): Number of users to choose.
            min_ratings (int): Minimum number of ratings for a user.
            directory_path (str): Path to save the chosen users.
        """
        
        users = self.ratings["user_id"].unique()
        selected_users = np.random.choice(users, n_users, replace=False)
        with open(file_path, "w") as f:
            json.dump(selected_users.tolist(), f)
            
        return selected_users



class ExperimentManager:
    """ Class responsible for running experiments. The class loads user IDs from a file, and for each user, it generates recommendations and calculates metrics.
    """
    def __init__ (self, llm_name, directory_path):
        """ Initialize the ExperimentManager.
        Args:
            llm_name (str): Name of the language model. Either "gpt", "mistral" or "mixtral".
            directory_path (str): Path to the directory with the user IDs file.
        """
        self.directory_path = directory_path
        self.llm_name = llm_name
        self.user_ids = self.load_user_ids()
        self.recommendation_logic = RecommendationLogic(llm_name=llm_name,
                                                        ease_model_path="../models/model_10000_sparse.pkl",
                                                        ratings_path="../data/ratings.json",
                                                        movie_names_path="../data/movies.json",
                                                        movie_plots_path="../data/movie_plots.csv")
        
        
    def load_user_ids(self):
        """ Loads all user IDs that have data generated in the directory. The ids are saved in files named <user_id>_full_prompt.txt.
        """
        prompt_file_name = "full_prompt.txt"
        
        user_ids = []
        for file in os.listdir(self.directory_path):
            if file.endswith(prompt_file_name):
                user_ids.append(int(file.split("_")[0]))
        
        return sorted(user_ids)
        
        
    def predict_user_movies(self, user_id, n_movies):
        """ Recommend movies for a user and save the recommendations to a file.
        Args:
            user_id (int): User ID.
            n_movies (int): Number of movies to recommend.
        """
        
        
        print(f"Recommendations for user {user_id}.")
        user_prompt_path = f"{self.directory_path}/{user_id}_full_prompt.txt"
        with open(user_prompt_path, "r") as f:
            user_prompt = f.read()
            
        recommendations = self.recommendation_logic.get_llm_recommendations(user_prompt=user_prompt, user_ratings=[], rs_importance=50, k=n_movies)["recommendations"]
        
        
        with open(f"{self.directory_path}/{self.llm_name}/{user_id}_predictions.json", "w") as f:
            json.dump(recommendations, f)
            
            
    def predict_movies(self, n_movies, starting_user_id=None):
        """ Predict movies for all users in the directory and save the predictions to files.
        Args:
            n_movies (int): Number of movies candidate movies to recommend before they are filtered down to top k
            starting_user_id (int, optional): User ID to start from. If None, starts from the first user. Defaults to None.
        """

        start_idx = 0 if starting_user_id is None else self.user_ids.index(starting_user_id)
        
        if not os.path.exists(f"{self.directory_path}/{self.llm_name}"):
            os.makedirs(f"{self.directory_path}/{self.llm_name}")
            
        for user_id in self.user_ids[start_idx:]:
            self.predict_user_movies(user_id, n_movies)
    
    
    def calculate_accuracy(self, top_k):
        """ Calculate accuracy for the recommendations. Accuracy is calculated as the percentage of users for which the target movies was in the top k recommendations."""
        hits_rs = 0
        hits_both = 0
        # for user_id in self.user_ids:

        # user_ids are all files in the directory, ending with _predictions.json. 
        
        user_ids = []
        predictions_dir = f'{self.directory_path}/{self.llm_name}'
        for file in os.listdir(predictions_dir):
            if not file.endswith("_predictions.json"):
                continue
            user_id = int(file.split("_")[0])
            user_ids.append(user_id)

            target = pd.read_json(f"{self.directory_path}/{user_id}_target.json", orient="records", lines=True)

            # predictions = pd.read_json(f"{path}_predictions.json")
            predictions = pd.read_json(f"{predictions_dir}/{user_id}_predictions.json")
            prediction_rs = predictions.sort_values(by="rs_score", ascending=False)[:top_k]
            prediction_both = predictions.sort_values(by="total_score", ascending=False)[:top_k]

            if prediction_rs["item_id"].isin(target["item_id"]).any():
                hits_rs += 1
            if prediction_both["item_id"].isin(target["item_id"]).any():
                hits_both += 1
                

        acc_rs = round(hits_rs / len(user_ids),3)
        acc_both = round(hits_both / len(user_ids),3)
        print(f'Accuracy for rs: {acc_rs}')
        print(f'Accuracy for both: {acc_both}')
        with open(f"{self.directory_path}/{self.llm_name}/results_{top_k}.json", "w") as f:
            json.dump({"users": user_ids, "hits_rs": hits_rs, "hits_both": hits_both, "accuracy_rs": acc_rs, "accuracy_both": acc_both}, f)
            
        return acc_rs, acc_both
      
 
 
    def calculate_conversion_loss(self, llm_name):
        """ Calculate the conversion loss for the recommendations. Conversion loss is calculated as the sum of differences between the true ratings and the converted ratings. """
        def calculate_diff_between_ratings(true_ratings, converted_ratings):
            true_ratings_dict = {item_id: rating for item_id, rating in true_ratings}
            converted_ratings_dict = {item_id: rating for item_id, rating in converted_ratings}
            score_diff = 0
            item_diff = 0
            for item_id, rating in true_ratings_dict.items():
                if item_id in converted_ratings_dict:
                    score_diff += abs(rating - converted_ratings_dict[item_id])
                else:
                    item_diff += 1
                    
            item_diff += len(converted_ratings_dict) - len(true_ratings_dict) 

            return score_diff, item_diff
        
        
        
        total_score_diff = 0
        total_item_diff = 0
        for user_id in self.user_ids[:]:
            sample = pd.read_json(f"{self.directory_path}/{user_id}_sample.json", orient="records", lines=True)
            true_ratings = sample[["item_id", "rating"]].values.tolist()
            # true_ratings = sorted(true_ratings, key=lambda x: x[0])
            
            # read full_prompt.txt
            
            with open(f"{self.directory_path}/{user_id}_full_prompt.txt", "r") as f:
                prompt = f.read()
            
            converted_ratings = self.recommendation_logic.get_llm_recommendations(user_prompt=prompt, user_ratings=[], rs_importance=100, k=20)
            # converted_ratings = sorted(converted_ratings, key=lambda x: x[0])
            print(true_ratings)
            print(converted_ratings)
            
            score_diff, item_diff = calculate_diff_between_ratings(true_ratings, converted_ratings)
            total_score_diff += score_diff
            total_item_diff += item_diff
        
        print(f'Number of different items for {llm_name}: {total_item_diff}')
        print(f'Average number of different items for {llm_name}: {total_item_diff / len(self.user_ids)}')
        print(f'Total score difference for {llm_name}: {total_score_diff}')
        print(f'Average score difference for {llm_name}: {total_score_diff / len(self.user_ids)}')

            
        
      
def calculate_convertion_loss(llm_name):
    """ Create an ExperimentManager and calculate the conversion loss for the recommendations. """
    experiment_manager = ExperimentManager(llm_name=llm_name, directory_path="experiments")
    experiment_manager.calculate_conversion_loss(llm_name)
        
def generate_data(sample_size, n_users):    
    """ Create a DatasetGenerator and generate data for n_users. """
    generator = DatasetGenerator("config.json")
    generator.generate_data(sample_size, n_users, file_path=f"experiments/{n_users}_selected_users.json")

def make_predictions(llm_name, n_movies, starting_user_id=None):
    """ Create an ExperimentManager and make predictions for all users in the directory. """
    experiment_manager = ExperimentManager(llm_name=llm_name, directory_path="experiments")
    print(experiment_manager.user_ids)
    experiment_manager.predict_movies(n_movies=n_movies, starting_user_id=starting_user_id)
    
def calculate_accuracy(llm_name):
    """ Create an ExperimentManager and calculate accuracy for the recommendations. """
    print("-"*50)
    print("Accuracy for:", llm_name)
    experiment_manager = ExperimentManager(llm_name=llm_name, directory_path="experiments")
    results = {}
    for top_k in [1, 5, 10, 20]:#, 30, 40, 50]:
        acc_rs, acc_both = experiment_manager.calculate_accuracy(top_k)
        results[top_k] = {"rs": acc_rs, "both": acc_both}

    print(results)
    with open(f"experiments/{llm_name}/results.json", "w") as f:
        json.dump(results, f)         
         

def main(n_users=250, sample_size=8, n_movies=50, llm_name="gpt"):
    """ Main function for running experiments. The first experiment's three steps don't have to be run right after each other (the intermediate results are saved to files). """
    print(f"Running experiments with {args.llm_name} LLM.") 
    # generate_data(sample_size, n_users)                       #* Step 1 - Uncomment to generate data
    # make_predictions(llm_name=llm_name, n_movies=n_movies)    #* Step 2 - Uncomment to make predictions using the LLM
    # calculate_accuracy(llm_name)                              #* Step 3 - Uncomment to calculate accuracy
    
    
    # calculate_convertion_loss(llm_name)                       #* Second experiment time - Uncomment to calculate conversion loss
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments with a specified LLM.')
    parser.add_argument('--llm_name', type=str, default='gpt', choices=['mistral', 'mixtral', 'gpt'], help='Name of the language learning model')
    args = parser.parse_args()

    
    main(llm_name=args.llm_name)
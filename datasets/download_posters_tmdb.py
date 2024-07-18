import csv
import json
import logging

import pandas as pd
import requests

logging.basicConfig(filename='download_posters_tmdb.log', level=logging.ERROR)


def download_movie_poster(imdb_id, tmdb_id, api_key, output_dir):
    # Define the URL with the movie_id
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/images"

    # Set up headers including the Authorization with the provided API key
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Make the API request
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # This will raise an exception for HTTP errors

    # Process the JSON data to find the most voted English poster
    data = response.json()
    posters = data.get('posters', [])
    english_posters = [p for p in posters if p['iso_639_1'] == 'en']
    if not english_posters:
        best_poster = sorted(posters, key=lambda x: (x['vote_count'], x['vote_average']), reverse=True)[0]
    else:
        # Sort the English posters by vote_count (and by vote_average as a secondary sort key)
        best_poster = sorted(english_posters, key=lambda x: (x['vote_count'], x['vote_average']), reverse=True)[0]

    # Construct the full URL for the best poster image
    base_image_url = "https://image.tmdb.org/t/p/w500"
    poster_url = f"{base_image_url}{best_poster['file_path']}"

    # Download the poster
    poster_response = requests.get(poster_url)
    poster_response.raise_for_status()

    file_path = f"{output_dir}{imdb_id}.jpg"
    with open(file_path, 'wb') as file:
        file.write(poster_response.content)


def load_mapping(csv_file):
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        imdb_to_tmdb = {row['imdbId']: row['tmdbId'] for row in reader}
    return imdb_to_tmdb


def main():
    with open("config.json", "r") as f:
        config = json.load(f)
        
    api_key = config["TMD_API_KEY"]
    imdb_to_tmdb = load_mapping("original-data/links.csv")
    movies_df = pd.read_json("data/movies.json", lines=True, dtype={'imdbId': str})     # imdbId may have leading zeros, so it can't be read as an int
    
    for index, row in movies_df.iterrows():
        try:
            imdb_id = row['imdbId']
            tmdb_id = imdb_to_tmdb.get(imdb_id)
            if tmdb_id is None:
                logging.error(f"{imdb_id} - {row['title']} not found in mapping")
                continue
            download_movie_poster(imdb_id, tmdb_id, api_key, "data/posters/")
        except Exception as e:
            logging.error(f"{imdb_id} - {row['title']}: {e}")
            continue


if __name__ == "__main__":
    main()
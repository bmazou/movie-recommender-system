from flask import Flask, jsonify, request
from models import Movie, Tag


def title_relevance(title, search_term):
    """ Calculates the relevance of a movie title to a search term.
        The relevance is calculated based on the position of the search term in the title.
        If the search term is at the beginning of a word in the title, the relevance is higher. The sooner the search term appears in the title, the higher the relevance.
        The search term has relevance even it's not at the beginning of a word, but the relevance is much lower.
    """
    words = title.split()
    for i, word in enumerate(words):
        if word.lower().startswith(search_term):
            return 1 / (i + 1)
        elif search_term in word.lower():
            return 0.1 / (i + 1)
    return 0

def get_movies(k=15):
    """ Returns k best movies based on the search term and the overall popularity of the movies.
    Args:
        k (int, optional): Number of movies to return. Defaults to 15.
    Returns:
        list: List of dictionaries with keys 'title' and 'item_id'
    """
    def top_movies(n):
        """ Return first n movies in the database. The database is ordered by popularity (number of ratings)."""
        top_movies = Movie.query.limit(n).all()
        movies = [{'title': movie.title, 'item_id': movie.item_id} for movie in top_movies]
        return movies
    
    search_term = request.json['search']
    if len(search_term.strip()) == 0:
        return jsonify(top_movies(k))
    
    movies_with_scores = []
    search_term_lower = search_term.lower()
    search_result = Movie.query.filter(Movie.title.ilike(f'%{search_term}%')).all()
    for movie in search_result:
        title = movie.title.lower() 
        relevance = title_relevance(title, search_term_lower)
        combined_score = movie.popularity * relevance
        
        movies_with_scores.append({
            'title': movie.title,
            'item_id': movie.item_id,
            'combined_score': combined_score
        })
    
    movies_with_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    limited_movies = movies_with_scores[:k]

    movies = [{'title': movie['title'], 'item_id': movie['item_id']} for movie in limited_movies]
    
    return jsonify(movies)

def get_tags():
    """ Returns all tags or filters tags based on the search term.
    Returns:
        list: List of dictionaries with keys 'title' and 'item_id'
    """
    tags = Tag.query.all()
    search_term = request.json['search']

    if len(search_term.strip()) == 0:
        return jsonify([{'title': tag.tag, 'item_id': tag.id} for tag in tags])
        
    search_term_lower = search_term.lower()

    # Filters tags by checking if any word in the tag starts with the search term
    filtered_tags = [tag for tag in tags if any(word.lower().startswith(search_term_lower) for word in tag.tag.split())]

    return jsonify([{'title': tag.tag, 'item_id': tag.id} for tag in filtered_tags])
from flask import render_template, request
from models import Movie, Tag
from recommendation_logic import RecommendationLogic


def get_validated_ratings(form):
    """Extracts and validates the ratings from the form data."""
    movie_ids = form.getlist('movieIds[]')
    ratings = form.getlist('movieRatings[]')
    
    validated_ratings = []
    for movie_id, rating in zip(movie_ids, ratings):
        if not rating.isdigit() or not movie_id.isdigit():
            continue
            
        movie_id = int(movie_id)
        movie = Movie.query.filter_by(item_id=movie_id).first()
        if not movie:
            continue
        rating = int(rating)
        if 1 <= rating <= 5:
            validated_ratings.append((movie_id, rating))
    
    validated_ratings = dict(validated_ratings)     # Removes duplicates
    return list(validated_ratings.items())

def get_validated_tags(form):
    """Extracts and validates the tags from the form data."""
    tag_ids = form.getlist('tagIds[]')
    validated_tag_ids = []
    for tag_id in tag_ids:
        if not tag_id.isdigit():
            continue
        tag_id = int(tag_id)
        tag = Tag.query.filter_by(id=tag_id).first()
        if tag:
            validated_tag_ids.append(tag_id)
    
    validated_tag_ids = list(set(validated_tag_ids))   # Removes duplicates
    return validated_tag_ids

def get_validated_rs_importance(form):
    """Extracts and validates the rs_importance from the form data."""
    try:
        rs_importance = int(form.get('importanceSlider'))
        if rs_importance < 0 or rs_importance > 100:
            rs_importance = 50
    except:
        rs_importance = 50
        
    return rs_importance

def validate_and_transform_form(form, get_user_ratings):
    """Validates the form data and transforms it into the format required by the recommendation logic."""
    tag_ids = get_validated_tags(form)
    rs_importance = get_validated_rs_importance(form)
    user_ratings = get_validated_ratings(form)

    use_past_ratings = form.get('usePastRatings') == 'on'
    past_ratings = get_user_ratings() if use_past_ratings else []
    user_ratings.extend(past_ratings)
    
    return user_ratings, tag_ids, rs_importance

def home_lite(get_user_ratings):
    """ Home route for the lite version of the application.
    Args:
        get_user_ratings (function): Function that returns the user's past ratings, if user is logged in.
    Returns:
        render_template: Renders the template with the results
    """
    if request.method == 'POST':
        user_ratings, tag_ids, rs_importance = validate_and_transform_form(request.form, get_user_ratings)
        
        recommendation_logic = RecommendationLogic(llm_name='gpt-3.5',
                                            ease_model_path="../models/model_10000_sparse.pkl",
                                            ratings_path="../data/ratings.json",
                                            movie_names_path="../data/movies.json",
                                            movie_plots_path="../data/movie_plots.csv")
        
        results = recommendation_logic.get_tag_recommendations(user_ratings, tag_ids, rs_importance, k=60)
        
        ratings_titles = [(Movie.query.filter_by(item_id=movie_id).first().title, rating) for movie_id, rating in user_ratings]
        tag_names = [Tag.query.filter_by(id=tag_id).first().tag for tag_id in tag_ids]
        
        return render_template('results_lite.html', recommendations=results.to_dict('records'), rs_importance=rs_importance, user_ratings=ratings_titles, tag_names=tag_names)
# 

    return render_template('index_lite.html')
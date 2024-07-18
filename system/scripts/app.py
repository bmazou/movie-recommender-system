import json
import os

from download_posters_tmdb import download_movie_poster, load_mapping
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from flask_login import (LoginManager, current_user, login_required,
                         login_user, logout_user)
from lite import home_lite
from models import Movie, Rating, User, db, populate_database
from recommendation_logic import RecommendationLogic
from retrieval_functions import get_movies, get_tags
from werkzeug.security import check_password_hash, generate_password_hash

with open('config.json') as config_file:
    config = json.load(config_file)

app = Flask(__name__)
app.config['SECRET_KEY'] = config["APP_SECRET_KEY"]
app.config['SQLALCHEMY_DATABASE_URI'] = config["APP_DATABASE_URI"]  # SQLite is in our application for its simplicity
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

tmdb_api_key = config["TMDB_API_KEY"]
imdb_to_tmdb_map = load_mapping("../data/links.csv")


    
@login_manager.user_loader
def load_user(user_id):
    """Loads the user with the given ID."""
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registers a new user."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Logs in a user."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logs out the user."""
    logout_user()
    return redirect(url_for('home'))


@app.route('/rate_movie', methods=['POST'])
@login_required
def rate_movie():
    """ Creates or updates a rating for a movie. """
    imdb_id = request.form['imdb_id']
    rating = int(request.form['rating'])
    existing_rating = Rating.query.filter_by(user_id=current_user.id, imdb_id=imdb_id).first()
    if existing_rating:
        existing_rating.rating = rating
    else:
        new_rating = Rating(user_id=current_user.id, imdb_id=imdb_id, rating=rating)
        db.session.add(new_rating)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/delete_rating', methods=['POST'])
@login_required
def delete_rating():
    """ Deletes a rating for a movie. """
    data = request.get_json()
    imdb_id = data['imdb_id']
    rating = Rating.query.filter_by(user_id=current_user.id, imdb_id=imdb_id).first()
    if rating:
        db.session.delete(rating)
        db.session.commit()
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Rating not found'}), 404

@app.route('/profile')
@login_required
def profile():
    """ Displays the user's profile page with their ratings. """
    user_ratings = db.session.query(Rating, Movie) \
        .join(Movie, Rating.imdb_id == Movie.imdb_id) \
        .filter(Rating.user_id == current_user.id) \
        .all()
    rated_movies = [{'imdb_id': rating.imdb_id, 'title': movie.title, 'year': movie.year, 'rating': rating.rating} for rating, movie in user_ratings]
    return render_template('profile.html', rated_movies=rated_movies)


@app.route('/posters/<imdb_id>')
def poster(imdb_id):
    """ Downloads and serves the poster for the given IMDb ID. 
    Args:
        imdb_id (str): The IMDb ID of the movie, starting with 'tt'.
    Returns:
        send_from_directory: The poster image file.
    """
    
    if not os.path.exists('static/posters'):
        os.makedirs('static/posters')
    
    poster_path = f'static/posters/{imdb_id}.jpg'
    imdb_id_num = imdb_id[2:]   # Remove the 'tt' prefix
    if not os.path.isfile(poster_path):
        tmdb_id = imdb_to_tmdb_map.get(imdb_id_num)  
        if tmdb_id:
            download_movie_poster(imdb_id, tmdb_id, tmdb_api_key, 'static/posters/')
        else:
            return "Poster not found", 404
    return send_from_directory('static/posters', f'{imdb_id}.jpg')

def get_user_ratings():
    """ Returns the current user's ratings as a list of tuples (imdb_id, rating). """
    if current_user.is_authenticated:
        user_ratings = db.session.query(Rating, Movie) \
            .join(Movie, Rating.imdb_id == Movie.imdb_id) \
            .filter(Rating.user_id == current_user.id) \
            .all()
        return [(movie.item_id, rating.rating) for rating, movie in user_ratings]
    return []


@app.route('/', methods=['GET', 'POST'])
def home():
    """ Displays the home page with the recommendation form or takes a form submission and returns the resulting recommendations. """
    use_past_ratings = False
    if request.method == 'POST':
        prompt = request.form['prompt']
        rs_importance = int(request.form['importanceSlider'])
        llm_importance = 100 - rs_importance
        llm_name = request.form.get('llm_name', 'gpt')
        use_past_ratings = request.form.get('use_past_ratings') == 'on'
        k_movies = int(request.form.get('k_movies'))
        print(f'Rs importance: {rs_importance}, LLM importance: {llm_importance}')
        
        user_ratings = get_user_ratings() if use_past_ratings else []
        
        print("\n", f'Running with LLM: {llm_name}',"\n")
        
        recommendation_logic = RecommendationLogic(llm_name=llm_name,
                                            ease_model_path="../models/model_10000_sparse.pkl",
                                            ratings_path="../data/ratings.json",
                                            movie_names_path="../data/movies.json",
                                            movie_plots_path="../data/movie_plots.csv") 
        
        result = recommendation_logic.get_llm_recommendations(prompt, user_ratings, rs_importance, k=k_movies)
        return render_template('results.html', recommendations=result['recommendations'], rs_importance=rs_importance)
    return render_template('index.html', use_past_ratings=use_past_ratings)

@app.route('/lite', methods=['GET', 'POST'])
def home_lite_route():
    """ Home route for the lite version of the application. """
    return home_lite(get_user_ratings)
    
@app.route('/get_movies', methods=['POST'])
def get_movies_route():
    """ Route for getting the movie data. """
    return get_movies()

@app.route('/get_tags', methods=['POST'])
def get_tags_route():
    """ Route for getting the tag data. """
    return get_tags()


if __name__ == '__main__':
    with app.app_context():
        populate_database()
        
    app.run(debug=True)
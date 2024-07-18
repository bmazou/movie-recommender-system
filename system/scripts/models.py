import pandas as pd
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Movie(db.Model):
    imdb_id = db.Column(db.String(15), primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    popularity = db.Column(db.Integer, nullable=False)
    directed_by = db.Column(db.String(100), nullable=False)
    starring = db.Column(db.String(250), nullable=False)
    avg_rating = db.Column(db.Float, nullable=False)
    item_id = db.Column(db.Integer, nullable=True)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    imdb_id = db.Column(db.String(15), db.ForeignKey('movie.imdb_id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    
class Tag(db.Model):
    auto_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id = db.Column(db.Integer, nullable=False)
    tag = db.Column(db.String(100), nullable=False)
    
    def __init__ (self, id, tag):
        self.id = id
        self.tag = tag

class TagCount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('movie.item_id'), nullable=False)
    tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'), nullable=False)
    num = db.Column(db.Integer, nullable=False)
    
def populate_tag_counts():
    """Populates the tag_count table with data from tag_count.json"""
    tag_counts = pd.read_json('../data/tag_count.json', orient='records', lines=True)

    for _, tag_count in tag_counts.iterrows():
        # tag_count.json seems to be weirdly formatted, so we need to convert to int
        new_tag_count = TagCount(
            item_id=int(tag_count['item_id']),
            tag_id=int(tag_count['tag_id']),
            num=int(tag_count['num'])
        )
        db.session.add(new_tag_count)
    db.session.commit()

    for tag_count in TagCount.query.limit(5).all():
        print(f'Item id: {tag_count.item_id}, Tag id: {tag_count.tag_id}, Num: {tag_count.num}')

def populate_movies():
    """Populates the movie table with data from movies.json"""
    movies = pd.read_json('../data/movies.json', orient='records', lines=True)

    for _, movie in movies.iterrows():
        new_movie = Movie(
            imdb_id=movie['imdbId'],
            title=movie['title'],
            year=movie['year'],
            popularity=movie['popularity'],
            directed_by=movie['directedBy'],
            starring=movie['starring'],
            avg_rating=movie['avgRating'],
            item_id=movie['item_id']
        )
        db.session.add(new_movie)
    db.session.commit()

    for movie in Movie.query.limit(5).all():
        print(f'Imdb id: {movie.imdb_id}, Item id: {movie.item_id}, Title: {movie.title}, Popularity: {movie.popularity}')
        
def populate_database():
    """Populates the database with data from the JSON files if the tables are empty"""
    db.create_all()
    if Movie.query.first() is None:
        populate_movies()
    if Tag.query.first() is None:
        populate_tags()
    if TagCount.query.first() is None:
        populate_tag_counts()        


def sort_tags():
    """ Sorts tags by the number of times they are used in tag_count.json
    """
    tags = pd.read_json('../data/tags.json', orient='records', lines=True)
    tag_count = pd.read_json('../data/tag_count.json', orient='records', lines=True)
    tag_count = tag_count.groupby('tag_id').sum().reset_index()
    tags = tags.merge(tag_count, left_on='id', right_on='tag_id')
    tags = tags.sort_values('num', ascending=False)
    tags = tags.drop(columns=['tag_id', 'item_id', 'num'])
    tags.to_json('../data/tags_sorted.json', orient='records', lines=True)
    

def populate_tags():
    """Populates the tag table with sorted data from tags.json"""
    import os
    if not os.path.exists('../data/tags_sorted.json'):
        sort_tags()
    
    tags = pd.read_json('../data/tags_sorted.json', orient='records', lines=True)

    for _, tag in tags.iterrows():
        new_tag = Tag(
            id=tag['id'],
            tag=tag['tag']
        )
        db.session.add(new_tag)
    db.session.commit()
    
    for tag in Tag.query.limit(5).all():
        print(f'Id: {tag.id}, Tag: {tag.tag}')
        

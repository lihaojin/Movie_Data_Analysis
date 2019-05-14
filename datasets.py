import pandas as pd
import numpy as np
import matplotlib as plt
import ast
from ast import literal_eval





#Function to generate column with only 'name' key
def getLang (langs):
    x = ast.literal_eval(langs)
    if(len(x) != 0):
        new_x = []
        for i in range(len(x)):
            name = list(x[i].get("name"))
            new_x.append(x[i].get("name"))

        return(new_x)

    return langs


def getMetadata(clean=False):
    # Flag to reclean dataset or return cleaned one
    if(not clean):
        df = pd.read_csv('./clean_datasets/clean_metadata.csv')
        return df
    #Read csv file
    df = pd.read_csv('./data/movies_metadata.csv')
    #Drop columns
    cols_to_drop = ['homepage', 'tagline', 'poster_path', 'adult', 'original_title', 'imdb_id']
    df = df[df.columns.drop(cols_to_drop)]
    #Replaced values with 0 to NaN.
    df['revenue'] = df['revenue'].replace(0, np.nan)
    df['budget'] = df['budget'].replace(0, np.nan)
    #Drop rows with EOF error
    df = df.drop([19729, 19730, 29502, 29503, 35586, 35587])
    #Make languages column
    languages = list(df.spoken_languages)
    lang_df = []
    for i in range(0, len(languages), 1):
        try:
            #lang = []
            lang_df.append(getLang(languages[i]))
        #lang_df.append(lang)
        except:
            print(i)
    #Add languages column to dataframe
    df = df.assign(languages = lang_df)
    #Drop original languages column
    df = df.drop(columns = ['spoken_languages'])
    #Make genres column
    genres = list(df.genres)
    genres_df = []
    for i in range(0, len(genres), 1):
        try:
            genres_df.append(getLang(genres[i]))
        except:
            print(i)
    #Add genres column to dataframe
    df = df.assign(movie_genres = genres_df)
    #Drop original genres column
    df = df.drop(columns = ['genres'])
    #Make countries column
    countries = list(df.production_countries)
    countries_df = []
    for i in range(0, len(countries), 1):
        try:
            countries_df.append(getLang(countries[i]))
        except:
            print(i)
    #Add countries column to dataframe
    df = df.assign(countries = countries_df)
    #Drop original countries column
    df = df.drop(columns = ['production_countries'])
    #Make companies column
    companies = list(df.production_companies)
    companies_df = []
    for i in range(0, len(companies), 1):
        try:
            companies_df.append(getLang(companies[i]))
        except:
            print(i)
    #Add companies column
    df = df.assign(companies = companies_df)
    #Drop original companies column
    df = df.drop(columns = ['production_companies'])
    #Convert release_date to Year, Month, Day columns
    df['release_date'] = df['release_date'].astype(str)
    df['release_date'] = [d.split('-') for d in df.release_date]
    df[['year', 'month', 'day']] = pd.DataFrame(df.release_date.values.tolist(), index= df.index)
    df = df.drop(columns = ['release_date'])
    df['budget'] = df['budget'].astype(str).astype(int)
    df['revenue'] = df['revenue'].astype(str).astype(float)
    df['id'] = df['id'].astype(str).astype(int)
    # Convert objects to an int
    df.rename(columns ={'id': 'movieId'}, inplace=True)
    # Strip belongs_to_collection object to just names
    def cleanName(obj):
        if(not pd.isnull(obj)):
            return ast.literal_eval(obj)['name']
    df['belongs_to_collection'] = df['belongs_to_collection'].apply(cleanName)
    return df

def getRatings(clean=False):
    if(not clean):
        print("Getting cleaned ratings")
        ratings_df = pd.read_csv("./clean_datasets/clean_ratings.csv")
        return ratings_df
    ratings_df = pd.read_csv("data/ratings.csv")
    # Calculate the average rating for each movie, drop userId and timestamp
    ratings_df = ratings_df.drop(columns = ['userId', 'timestamp'])
    ratings_df = ratings_df.groupby('movieId', as_index=False).mean()
    ratings_df.to_csv("./clean_datasets/clean_ratings.csv", index=False)
    return ratings_df

def get_cast(clean=False):
    if(not clean):
        credits_df = pd.read_csv("clean_datasets/clean_cast.csv")
        return credits_df
    # returns a dataset with the name of the actor/actress, movies they've been in, and characters they've played as
    credits_df = pd.read_csv("data/credits.csv")
    credits_dict = {}
    """ create the credits dictionary """
    for row in range(len(credits_df)):
        cast = literal_eval(credits_df.cast[row])
        for i in range(len(cast)):
            if cast[i]['cast_id'] in credits_dict:
                credits_dict[cast[i]['cast_id']]['movies'].append(credits_df.movieId[row])
                credits_dict[cast[i]['cast_id']]['character'].append(cast[i]['character'])
            else:
                credits_dict[cast[i]['cast_id']] = {}
                credits_dict[cast[i]['cast_id']]['name'] = []
                credits_dict[cast[i]['cast_id']]['movies'] = []
                credits_dict[cast[i]['cast_id']]['character'] = []
                credits_dict[cast[i]['cast_id']]['gender'] = []
                credits_dict[cast[i]['cast_id']]['name'].append(cast[i]['name'])
                credits_dict[cast[i]['cast_id']]['movies'].append(credits_df.movieId[row])
                credits_dict[cast[i]['cast_id']]['character'].append(cast[i]['character'])
                credits_dict[cast[i]['cast_id']]['gender'].append(cast[i]['gender'])

    cast, val = zip(*credits_dict.items())

    """ create a dataframe using the dictionary"""
    cast_df = pd.DataFrame({'castId': cast, 'data': val})
    cast_df['name'] = ''
    cast_df['in_movies'] = ''
    cast_df['played_as'] = ''
    cast_df['gender'] = ''

    for i in range(len(cast_df)):
        cast_df.name[i] = cast_df.data[i]['name'][0]
        cast_df.in_movies[i] = cast_df.data[i]['movies']
        cast_df.played_as[i] = cast_df.data[i]['character']
        cast_df.gender[i] = int(cast_df.data[i]['gender'][0])

    """ drop original column """
    cast_df = cast_df.drop('data', axis = 1)

    """ create a csv file for cast_df """
    cast_df.to_csv("./clean_datasets/clean_cast.csv", index=False)

    return cast_df

def get_actors(clean=False):
    if(not clean):
        credits_df = pd.read_csv('./clean_datasets/clean_actor_director.csv')
        return credits_df
    credits_df = pd.read_csv("data/credits.csv")
    credits_dict = {}
    """ create dictionary to hold the actors of each movie"""
    for index, row in credits_df.iterrows():
        cast = literal_eval(credits_df.cast[index])
        for i in range(len(cast)):
            if credits_df['id'][index] in credits_dict:
                credits_dict[credits_df['id'][index]].append(cast[i]['name'])
            else:
                credits_dict[credits_df['id'][index]] = []
                credits_dict[credits_df['id'][index]].append(cast[i]['name'])
    # separate the keys from the values
    movieId, actor_actress = zip(*credits_dict.items())
    actor_df = pd.DataFrame({'movieId': movieId, 'actor_actress': actor_actress})
    actor_df = pd.DataFrame(actor_df.actor_actress.tolist(), index= actor_df.movieId)
    actor_df.reset_index(level=0, inplace=True)
    director = {}
    """ create dictionary to hold the directors of each movie"""
    for index, row in credits_df.iterrows():
        cast = literal_eval(credits_df.crew[index])
        for i in range(len(cast)):
            if cast[i]['job'] == 'Director':
                director[credits_df['id'][index]] = []
                director[credits_df['id'][index]].append(cast[i]['name'])
    # separate the keys from the values
    movie, directors = zip(*director.items())
    directors_df = pd.DataFrame({'movieId': movie, 'director': directors})
    directors_df.director = directors_df.director.map(lambda x: x[0])
    df = pd.merge(directors_df, actor_df, on='movieId')  
    # only need 5 actors
    df.drop(df.columns[7:], axis=1, inplace=True)
    col = ['actor1', 'actor2', 'actor3', 'actor4', 'actor5']
    df.rename(columns=dict(zip(df.columns[2:], col)),inplace=True)
    # create a csv file for the df
    df.to_csv("./clean_datasets/clean_actor_director.csv", index=False)
    return df

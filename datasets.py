import pandas as pd
import numpy as np
import matplotlib as plt
import ast
from ast import literal_eval


def getCredits():
    credits_df = pd.read_csv("data/credits.csv")
    # Changing the column names of credits_df for merging
    credits_df.columns = ['cast', 'crew', 'movieId']
    credits_df.sort_values('movieId')
    droprows = []
    for i, row in credits_df.iterrows():
        if(len(credits_df.iloc[i, 0]) == 2): # Some rows have an empty string array that we can drop
            droprows.append(i)

    credits_df = credits_df.drop(droprows)
    credits_df = credits_df.drop_duplicates()
    credits_df = credits_df.reset_index(drop = True)
    credits_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    return credits_df

def getMetadata():
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

    return df

def getRatings():
    ratings_df = pd.read_csv("data/ratings.csv")
    # Calculate the average rating for each movie, drop userId and timestamp
    ratings_df = ratings_df.drop(columns = ['userId', 'timestamp'])
    ratings_df = ratings_df.groupby('movieId', as_index=False).mean()
    return ratings_df

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

def getRatings():
    ratings_df = pd.read_csv("data/ratings.csv")
    # Calculate the average rating for each movie, drop userId and timestamp
    ratings_df = ratings_df.drop(columns = ['userId', 'timestamp'])
    ratings_df = ratings_df.groupby('movieId', as_index=False).mean()
    return ratings_df
